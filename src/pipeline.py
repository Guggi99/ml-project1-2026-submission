"""The chain from raw data to F1 score or submission CSV, driven by one config.

One experiment = one config dict. The same config runs in two modes:

- ``develop(config, x, y)``: Phase A, k-fold CV on the training rows. Answers
  "is this recipe better?" with F1 mean, std and per-fold values.
- ``produce(config, x_train, y_train, x_test, test_ids, out)``: Phase B, fit
  once on all training rows, predict the test rows, check and write the CSV.

A score only counts if it comes from ``run_experiment``: the same as
``develop``, plus a record in ``results/runs/`` (see ``src/runs.py``, #21).

Each part of the config belongs to one person, so the parts can change
independently and every score comes out of the same harness::

    config = {
        "preprocessing": {...},  # Ahmed: keyword arguments of fit_preprocessor
        "model": {...},          # David: method name and its hyperparameters
        "threshold": "...",      # Peter (#11): how scores become -1 / 1
    }

Run the baseline from the repo root with ``python -m src.pipeline``.
"""

import time
import warnings
from pathlib import Path

import numpy as np

import implementations
from src.helpers import create_csv_submission, load_csv_data
from src.preprocessing import fit_preprocessor, transform
from src.runs import RUNS_DIR, log_run
from src.validation import cross_validate, f1_score

# The recipe of the 25 Sep chain run (CV F1 0.3946 on all rows).
BASELINE = {
    "preprocessing": {},
    "model": {"name": "mean_squared_error_gd", "max_iters": 200, "gamma": 0.05},
    "threshold": "positive_rate",
}

# CHANGE Peter, 2026-10-10, #11 step 3: rule for configs that name none
DEFAULT_THRESHOLD = "train_f1"

# ---------------------------------------------------------------------------
# MODELS: name -> which arguments it takes after (y, tx), in the course's order.
# initial_w is always zeros, sized by the preprocessed matrix, so a config
# never has to know how many columns preprocessing produces.
# ---------------------------------------------------------------------------
MODEL_ARGS = {
    "mean_squared_error_gd": ["initial_w", "max_iters", "gamma"],
    "mean_squared_error_sgd": ["initial_w", "max_iters", "gamma"],
    "least_squares": [],
    "ridge_regression": ["lambda_"],
    "logistic_regression": ["initial_w", "max_iters", "gamma"],
    "reg_logistic_regression": ["lambda_", "initial_w", "max_iters", "gamma"],
}

# The data has labels in {-1, 1}; the logistic methods expect {0, 1}.
LOGISTIC = {"logistic_regression", "reg_logistic_regression"}

# Only the team's implementations.py, so a run can never use another module by
# mistake (David's review of #28). A config may still name it explicitly.
MODULES = {"implementations": implementations}

CACHE_NAME = "cache_full.npz"
CACHE_KEYS = ["x_train", "x_test", "y_train", "train_ids", "test_ids"]

# produce() refuses a fit whose training F1 is this far below the run's CV F1.
F1_TOLERANCE = 0.02


def load_data(data_dir="data"):
    """Load the five arrays of load_csv_data, cached after the first call.

    Parsing the CSVs takes ~40 s; the cache (``data/cache_full.npz``, ~1.1 GB,
    git-ignored with ``data/``) loads in under a second. Delete it to reparse.
    """
    cache = Path(data_dir) / CACHE_NAME
    if cache.exists():
        cached = np.load(cache)
        return tuple(cached[k] for k in CACHE_KEYS)
    arrays = load_csv_data(str(data_dir))
    np.savez(cache, **dict(zip(CACHE_KEYS, arrays)))
    return arrays


def fit_model(model_config, y, tx):
    """Call the configured method on (y, tx) and return its weights ``w``.

    Converts the labels to {0, 1} for the logistic methods and warns when the
    step size of MSE gradient descent is too large to converge.
    """
    settings = dict(model_config)
    name = settings.pop("name")
    module_name = settings.pop("module", "implementations")
    if module_name not in MODULES:
        raise ValueError(f"unknown module {module_name!r}; known: {sorted(MODULES)}")
    module = MODULES[module_name]
    if name not in MODEL_ARGS:
        raise ValueError(f"unknown model {name!r}; known: {sorted(MODEL_ARGS)}")

    if name in LOGISTIC:
        y = (y == 1).astype(float)
    if name.startswith("mean_squared_error"):
        _warn_if_step_too_large(tx, settings["gamma"])

    settings["initial_w"] = np.zeros(tx.shape[1])
    args = [settings[arg] for arg in MODEL_ARGS[name]]
    w, _ = getattr(module, name)(y, tx, *args)
    return w


def _warn_if_step_too_large(tx, gamma):
    # GD on MSE converges only for gamma < 2 / L, L = largest eigenvalue of
    # tx.T @ tx / N. Beyond that the loss silently becomes inf/NaN.
    largest = np.linalg.eigvalsh(tx.T @ tx / tx.shape[0]).max()
    if gamma >= 2 / largest:
        warnings.warn(f"gamma {gamma} >= 2/L = {2 / largest:.4f}: GD will diverge")


def fit_threshold(rule, scores, y):
    """Choose the score above which a row is predicted 1, from training rows only."""
    if rule == "zero":
        return 0.0
    if rule == "positive_rate":
        # Flag as many rows as are positive in training (~8.8 %).
        return float(np.quantile(scores, 1.0 - np.mean(y == 1)))
    # CHANGE Peter, 2026-10-10, #11 step 2: third rule "train_f1",
    # the cut-off with the highest F1 on these training rows (best_threshold)
    if rule == "train_f1":
        return float(best_threshold(scores, y))
    raise ValueError(f"unknown threshold rule {rule!r}")


# CHANGE Peter, 2026-10-10, #11 step 1: best_threshold moved here unchanged from
# notebook experiments/260924_pg_walking_skeleton.ipynb (cell Run 7B)
def best_threshold(scores, y):
    """Return the training-score cutoff that maximises training F1.

    Scores are sorted once, so every possible prediction boundary is checked
    in O(N log N). Candidate boundaries end complete groups of tied scores,
    because the later `score > threshold` rule cannot split equal scores.
    This function must receive training rows only.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y)

    if scores.ndim != 1 or y.ndim != 1 or scores.shape != y.shape:
        raise ValueError("scores and y must be one-dimensional and equally sized")
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be non-empty and finite")

    order = np.argsort(-scores, kind="stable")  # highest score first
    sorted_scores = scores[order]
    is_positive = (y[order] == 1).astype(float)
    positive_count = is_positive.sum()
    if positive_count == 0:
        return np.inf  # with no positives, predict no positives

    predicted_positive = np.arange(1, scores.size + 1)
    true_positives = np.cumsum(is_positive)

    # F1 = 2*TP / (predicted positives + actual positives).
    f1 = 2.0 * true_positives / (predicted_positive + positive_count)

    # A strict threshold can separate rows only where the next score is lower.
    group_end = np.r_[sorted_scores[:-1] > sorted_scores[1:], True]
    candidates = np.flatnonzero(group_end)
    best_index = candidates[int(np.argmax(f1[candidates]))]

    # Move one floating-point step below the boundary so `>` includes it.
    return np.nextafter(sorted_scores[best_index], -np.inf)


def make_pipeline(config):
    """Build ``(fit_fn, predict_fn)`` for cross_validate from one config.

    The fitted model is ``(params, w, threshold)``: preprocessing parameters,
    weights and cut-off, all learned from the rows passed to ``fit_fn``.
    """

    def fit_fn(x, y):
        params = fit_preprocessor(x, **config["preprocessing"])
        tx = transform(x, params)
        w = fit_model(config["model"], y, tx)
        # CHANGE Peter, 2026-10-10, #11 step 3: the config's rule, else the default
        threshold = fit_threshold(config.get("threshold", DEFAULT_THRESHOLD), tx @ w, y)
        return params, w, threshold

    def predict_fn(model, x):
        params, w, threshold = model
        return np.where(transform(x, params) @ w > threshold, 1, -1)

    return fit_fn, predict_fn


def develop(config, x, y):
    """Phase A: k-fold CV of the recipe; returns cross_validate's dict + runtime."""
    start = time.time()
    result = cross_validate(x, y, *make_pipeline(config))
    result["seconds"] = round(time.time() - start, 1)
    return result


def run_experiment(
    config, x, y, *, name, hypothesis, author, baseline=None, runs_dir=RUNS_DIR
):
    """develop() plus a run record; the only way a score counts (#21).

    ``name``: short, lowercase with '-', e.g. "one-hot". ``hypothesis``: one
    sentence on what should change and why. ``author``: initials (pg, aa, dg).
    ``baseline``: id of the run this one is compared with, e.g.
    "261004_pg_logistic". Returns the record (its ``id`` is the file name).
    """
    # CHANGE Peter, 2026-10-10, #11 step 4: write the rule into the run record,
    # also when the config relies on DEFAULT_THRESHOLD
    config = {"threshold": DEFAULT_THRESHOLD, **config}
    result = develop(config, x, y)
    record = log_run(
        result,
        config,
        len(y),
        name=name,
        hypothesis=hypothesis,
        author=author,
        baseline=baseline,
        runs_dir=runs_dir,
    )
    print(
        f"{record['id']}: F1 {record['f1_mean']:.4f} +- {record['f1_std']:.4f}"
        + ("  [dirty: uncommitted code, exploratory only]" if record["dirty"] else "")
    )
    return record


def produce(config, x_train, y_train, x_test, test_ids, out, *, cv_f1=None):
    """Phase B: fit on all training rows, predict x_test, check, write ``out``.

    Stops before writing when the model or its predictions look broken:

    - the step size makes gradient descent diverge (a warning in Phase A, an
      error here). The rate check below cannot see this: with the
      "positive_rate" threshold any model, diverged or not, flags ~8.8 %.
    - ``cv_f1`` given (the run's CV F1 mean): F1 on the training rows is more
      than ``F1_TOLERANCE`` below it, so this fit does not behave like the run
      it claims to reproduce, e.g. because the code changed since.
    - the share of predicted positives is not within a factor 2 of the
      training rate.
    """
    fit_fn, predict_fn = make_pipeline(config)
    with warnings.catch_warnings():
        # Turn only the divergence warning of fit_model into an exception
        warnings.filterwarnings("error", message="gamma .* will diverge")
        try:
            model = fit_fn(x_train, y_train)
        except UserWarning as err:
            raise RuntimeError(f"not writing {out}: {err}") from None

    if cv_f1 is not None:
        # A full-data fit scores within ~0.001 of its CV F1 on the training rows
        # (measured 4 Oct on four configs); far below means a different model
        f1_train = f1_score(y_train, predict_fn(model, x_train))
        if f1_train < cv_f1 - F1_TOLERANCE:
            raise RuntimeError(
                f"training F1 {f1_train:.4f} vs CV F1 {cv_f1:.4f} of the run: "
                "not the model that was scored"
            )

    y_pred = predict_fn(model, x_test)

    assert len(y_pred) == len(test_ids)
    assert set(np.unique(y_pred)) <= {-1, 1}
    rate_pred = float(np.mean(y_pred == 1))
    rate_train = float(np.mean(y_train == 1))
    if not 0.5 * rate_train <= rate_pred <= 2 * rate_train:
        raise RuntimeError(
            f"{rate_pred:.1%} predicted positive vs {rate_train:.1%} in training"
        )

    create_csv_submission(test_ids, y_pred, str(out))
    return rate_pred


if __name__ == "__main__":
    x_train, x_test, y_train, train_ids, test_ids = load_data()
    r = develop(BASELINE, x_train, y_train)
    print(f"F1 {r['f1_mean']:.4f} +- {r['f1_std']:.4f}  ({r['seconds']} s)")
    print("per fold:", [round(f, 4) for f in r["f1_folds"]])
