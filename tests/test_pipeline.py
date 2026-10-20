"""Checks on src/pipeline.py with a small synthetic dataset (no real data needed)."""

import numpy as np
import pytest

# CHANGE Peter, 2026-10-10, #11 step 5: imports for the threshold tests
from src.pipeline import (
    BASELINE,
    DEFAULT_THRESHOLD,
    best_threshold,
    develop,
    fit_model,
    fit_threshold,
    produce,
    run_experiment,
)
from src.validation import f1_score
from src.runs import load_runs, load_submissions, log_submission, summary_markdown

from run import submission_config

RNG = np.random.default_rng(0)
N, D = 400, 5
X = RNG.normal(size=(N, D))
# Positive class ~15 %, driven by the first two columns, so a model can learn it.
Y = np.where(X[:, 0] + X[:, 1] + 0.5 * RNG.normal(size=N) > 1.5, 1, -1)


def config(**model):
    return {"preprocessing": {}, "model": model, "threshold": "positive_rate"}


def test_develop_baseline_learns_something():
    result = develop(BASELINE, X, Y)
    assert len(result["f1_folds"]) == result["k"]
    assert result["f1_mean"] > 0.5


def test_logistic_gets_labels_in_0_1():
    # implementations.logistic_regression expects y in {0, 1}; the data has -1 / 1.
    cfg = config(name="logistic_regression", max_iters=50, gamma=0.5)
    assert develop(cfg, X, Y)["f1_mean"] > 0.5


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        fit_model({"name": "random_forest"}, Y, X)


def test_only_the_team_module_is_used():
    # The reference implementations_pg.py is not reachable from a config (#28).
    with pytest.raises(ValueError, match="unknown module"):
        fit_model({"name": "least_squares", "module": "implementations_pg"}, Y, X)


def test_too_large_step_warns():
    with pytest.warns(UserWarning, match="diverge"):
        fit_model({"name": "mean_squared_error_gd", "max_iters": 1, "gamma": 100}, Y, X)


# ---------------------------------------------------------------------------
# THRESHOLD (#11)
# CHANGE Peter, 2026-10-10, #11 step 5: tests for best_threshold and train_f1
# ---------------------------------------------------------------------------
def test_best_threshold_beats_every_other_cut():
    # Brute force: F1 of "scores > c" for c at every score and below the lowest
    scores = X[:, 0] + X[:, 1]
    best = f1_score(Y, np.where(scores > best_threshold(scores, Y), 1, -1))
    cuts = np.r_[scores, scores.min() - 1]
    assert all(best >= f1_score(Y, np.where(scores > c, 1, -1)) for c in cuts)


def test_best_threshold_keeps_tied_scores_together():
    # Rows 2-4 tie at 3.0. Flagging rows 0-2 would give F1 0.86, but no cut-off
    # can split the tie; the best possible is rows 0-1 (F1 0.80).
    scores = np.array([5.0, 4.0, 3.0, 3.0, 3.0, 1.0])
    y = np.array([1, 1, 1, -1, -1, -1])
    flagged = scores > best_threshold(scores, y)
    assert flagged.tolist() == [True, True, False, False, False, False]


def test_best_threshold_without_positives_flags_nobody():
    scores = np.array([0.3, 0.1, 0.2])
    assert best_threshold(scores, np.array([-1, -1, -1])) == np.inf


def test_best_threshold_rejects_bad_input():
    with pytest.raises(ValueError, match="equally sized"):
        best_threshold(np.zeros(3), np.ones(4))
    with pytest.raises(ValueError, match="finite"):
        best_threshold(np.array([0.0, np.nan]), np.array([1, -1]))


def test_unknown_threshold_rule_is_rejected():
    with pytest.raises(ValueError, match="unknown threshold rule"):
        fit_threshold("median", X[:, 0], Y)


def test_default_threshold_is_used_and_recorded(tmp_path):
    no_rule = {"preprocessing": {}, "model": BASELINE["model"]}
    explicit = dict(no_rule, threshold=DEFAULT_THRESHOLD)
    # Same folds, same numbers: a config without a rule runs with the default
    assert develop(no_rule, X, Y)["f1_folds"] == develop(explicit, X, Y)["f1_folds"]
    # ... and its run record names the rule (step 4)
    record = run_experiment(
        no_rule, X, Y, name="default", hypothesis="h", author="pg", runs_dir=tmp_path
    )
    assert record["config"]["threshold"] == "train_f1"


def test_produce_writes_submission(tmp_path):
    out = tmp_path / "submission.csv"
    ids = np.arange(1000, 1000 + N)
    rate = produce(BASELINE, X, Y, X, ids, out)
    lines = out.read_text().splitlines()
    assert lines[0] == "Id,Prediction"
    assert len(lines) == N + 1
    assert rate == pytest.approx(np.mean(Y == 1), abs=0.05)


# ---------------------------------------------------------------------------
# RUN TRACKING (#21): records are written to a temporary folder, never to
# results/runs/, so running the tests leaves no files behind.
# ---------------------------------------------------------------------------
def test_run_experiment_writes_one_record_per_run(tmp_path):
    first = run_experiment(
        BASELINE, X, Y, name="base", hypothesis="h", author="pg", runs_dir=tmp_path
    )
    second = run_experiment(
        BASELINE, X, Y, name="base", hypothesis="h", author="pg", runs_dir=tmp_path
    )
    assert second["id"] == first["id"] + "-2"  # never overwritten
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert first["protocol"] == {
        "rows": N,
        "folds": 5,
        "seed": 42,
    }
    assert first["decision"] == ""


def test_bad_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="lowercase"):
        run_experiment(
            BASELINE,
            X,
            Y,
            name="My Run",
            hypothesis="h",
            author="pg",
            runs_dir=tmp_path,
        )


def test_summary_compares_with_baseline(tmp_path):
    base = run_experiment(
        BASELINE, X, Y, name="base", hypothesis="h", author="pg", runs_dir=tmp_path
    )
    # A threshold change is an experiment, measured the same way: compared fold
    # by fold.
    zero = dict(BASELINE, threshold="zero")
    run_experiment(
        zero,
        X,
        Y,
        name="zero",
        hypothesis="cut at 0",
        author="aa",
        baseline=base["id"],
        runs_dir=tmp_path,
    )
    # Fewer rows is a different protocol: no fold-by-fold comparison.
    run_experiment(
        BASELINE,
        X[:200],
        Y[:200],
        name="half",
        hypothesis="half the rows",
        author="aa",
        baseline=base["id"],
        runs_dir=tmp_path,
    )
    rows = summary_markdown(load_runs(tmp_path)).splitlines()
    zero_row = next(r for r in rows if "_aa_zero" in r)
    half_row = next(r for r in rows if "_aa_half" in r)
    assert "/5" in zero_row and "protocol differs" not in zero_row
    assert "protocol differs" in half_row and "/5" not in half_row


# ---------------------------------------------------------------------------
# SUBMISSIONS (#9)
# ---------------------------------------------------------------------------
def test_submission_record_and_summary(tmp_path):
    run = run_experiment(
        BASELINE, X, Y, name="base", hypothesis="h", author="pg", runs_dir=tmp_path
    )
    csv = tmp_path / "sub.csv"
    rate = produce(BASELINE, X, Y, X, np.arange(N), csv)
    first = log_submission(run, csv, rate, submissions_dir=tmp_path / "subs")
    second = log_submission(run, csv, rate, submissions_dir=tmp_path / "subs")
    assert second["id"].endswith("_2")  # never overwritten
    assert first["csv_sha256"] == second["csv_sha256"]  # same file, same checksum
    table = summary_markdown([run], load_submissions(tmp_path / "subs"))
    assert "## Submissions" in table and first["id"] in table


def test_submission_refuses_the_reference_module():
    record = {
        "id": "x",
        "dirty": False,
        "config": dict(
            BASELINE, model={"name": "least_squares", "module": "implementations_pg"}
        ),
    }
    with pytest.raises(ValueError, match="implementations.py"):
        submission_config(record)
    clean = {"id": "y", "dirty": False, "config": BASELINE}
    assert submission_config(clean) == BASELINE


@pytest.mark.parametrize("dirty", [True, None])
def test_submission_refuses_a_run_without_clean_commit(dirty):
    with pytest.raises(ValueError, match="clean commit"):
        submission_config({"id": "x", "dirty": dirty, "config": BASELINE})


def test_produce_refuses_a_diverging_fit(tmp_path):
    # With the positive-rate threshold the diverged model still flags ~15 %, so
    # only the divergence check can stop it.
    out = tmp_path / "sub.csv"
    diverging = config(name="mean_squared_error_gd", max_iters=50, gamma=100)
    with pytest.raises(RuntimeError, match="diverge"):
        produce(diverging, X, Y, X, np.arange(N), out)
    assert not out.exists()


def test_produce_refuses_a_fit_unlike_its_run(tmp_path):
    out = tmp_path / "sub.csv"
    with pytest.raises(RuntimeError, match="not the model that was scored"):
        produce(BASELINE, X, Y, X, np.arange(N), out, cv_f1=0.99)
    assert not out.exists()
    # Its own CV F1 passes
    cv_f1 = develop(BASELINE, X, Y)["f1_mean"]
    produce(BASELINE, X, Y, X, np.arange(N), out, cv_f1=cv_f1)
    assert out.exists()
