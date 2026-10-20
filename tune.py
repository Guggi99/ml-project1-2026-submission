"""Tune issue #12, confirm the winner, and create a submission.

Run from the repository root after committing the implementation::

    python tune.py --data-dir data --submission submission_tuned.csv

The command performs three leakage-safe stages:

1. Inspect gamma and iteration checkpoints while reusing each CV fold.
2. Run a lambda grid through the standard ``develop`` runner.
3. Confirm the selected config with ``run_experiment``, train on all labelled
   rows, and write the test submission.

It writes a JSON result and report plot under ``results/tuning/``.  The final
run is also written under ``results/runs/`` so ``run.py --run <printed id>``
can reproduce it later.
"""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from src.pipeline import load_data, produce, run_experiment
from src.preprocessing import fit_preprocessor, transform
from src.runs import (
    RUNS_DIR,
    git_state,
    load_run,
    load_runs,
    load_submissions,
    log_submission,
    summary_markdown,
)
from src.tuning import (
    checkpoint_cross_validate,
    fit_logistic_checkpoints,
    model_config,
    newton_logistic_reference,
    regularization_grid,
    regularized_logistic_objective,
    select_near_best,
)

BASE_RUN = "261011_pg_v2-train-f1"
GAMMAS = (0.05, 0.08, 0.10, 0.12)
CHECKPOINTS = (500, 1000, 2000, 3000)
LAMBDAS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
SELECTION_TOLERANCE = 0.0005
SUMMARY_PATH = Path("results") / "SUMMARY.md"


def comma_separated(text, cast):
    values = tuple(cast(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("provide at least one comma-separated value")
    return values


def next_artifact_stem(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = f"{datetime.date.today():%y%m%d}_dg_model-tuning"
    stem = base
    suffix = 1
    while (output_dir / f"{stem}.json").exists():
        suffix += 1
        stem = f"{base}-{suffix}"
    return stem


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_result(path, result):
    Path(path).write_text(json.dumps(json_ready(result), indent=2) + "\n")


def plot_search(convergence, regularization, baseline_f1, selected, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    left, middle, right = axes
    gammas = sorted({row["gamma"] for row in convergence})
    for gamma in gammas:
        rows = sorted(
            (row for row in convergence if row["gamma"] == gamma),
            key=lambda row: row["max_iters"],
        )
        left.errorbar(
            [row["max_iters"] for row in rows],
            [row["f1_mean"] for row in rows],
            yerr=[row["f1_std"] for row in rows],
            marker="o",
            capsize=3,
            label=f"gamma={gamma:g}",
        )
        middle.plot(
            [row["max_iters"] for row in rows],
            [row["objective_mean"] for row in rows],
            marker="o",
            label=f"gamma={gamma:g}",
        )
    left.axhline(baseline_f1, color="grey", linestyle="--", label="current best")
    left.set_xlabel("gradient iterations")
    left.set_ylabel("5-fold CV F1")
    left.set_title("Convergence and step size")
    left.legend(fontsize=8)

    middle.set_xlabel("gradient iterations")
    middle.set_ylabel("mean fold-training objective")
    middle.set_title("Training loss convergence")

    labels = [
        "0" if row["lambda_"] == 0 else f"{row['lambda_']:.0e}"
        for row in regularization
    ]
    positions = np.arange(len(regularization))
    right.errorbar(
        positions,
        [row["f1_mean"] for row in regularization],
        yerr=[row["f1_std"] for row in regularization],
        marker="o",
        capsize=3,
    )
    right.axhline(baseline_f1, color="grey", linestyle="--")
    right.set_xticks(positions)
    right.set_xticklabels(labels)
    right.set_xlabel("lambda")
    right.set_ylabel("5-fold CV F1")
    right.set_title(
        f"Regularization at gamma={selected['gamma']:g}, "
        f"{selected['max_iters']} iterations"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_newton_reference(x, y, preprocessing, selected, rows):
    count = min(int(rows), len(y))
    indices = np.random.default_rng(0).choice(len(y), size=count, replace=False)
    sample_x, sample_y = x[indices], y[indices]
    params = fit_preprocessor(sample_x, **preprocessing)
    tx = transform(sample_x, params)
    lambda_ = selected["lambda_"]
    snapshots = fit_logistic_checkpoints(
        sample_y,
        tx,
        selected["gamma"],
        [selected["max_iters"]],
        lambda_=lambda_,
    )
    gd = snapshots[selected["max_iters"]]
    newton = newton_logistic_reference(sample_y, tx, lambda_=lambda_)
    # Do not store two large weight vectors in the JSON artifact.
    return {
        "rows": count,
        "columns": tx.shape[1],
        "lambda_": lambda_,
        "gd_objective": gd["objective"],
        "newton_objective": newton["objective"],
        "gd_gap_to_newton": gd["objective"] - newton["objective"],
        "newton_iterations": newton["iterations"],
        "newton_converged": newton["converged"],
        "newton_objective_recomputed": regularized_logistic_objective(
            (sample_y == 1).astype(float), tx, newton["w"], lambda_
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=BASE_RUN, help="current best run id")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--submission", type=Path, default=Path("submission_tuned.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "tuning")
    parser.add_argument(
        "--gammas",
        type=lambda text: comma_separated(text, float),
        default=GAMMAS,
        help="comma-separated gamma candidates",
    )
    parser.add_argument(
        "--checkpoints",
        type=lambda text: comma_separated(text, int),
        default=CHECKPOINTS,
        help="comma-separated iteration checkpoints",
    )
    parser.add_argument(
        "--lambdas",
        type=lambda text: comma_separated(text, float),
        default=LAMBDAS,
        help="comma-separated regularization candidates",
    )
    parser.add_argument(
        "--selection-tolerance",
        type=float,
        default=SELECTION_TOLERANCE,
        help="prefer cheaper settings within this F1 distance of the maximum",
    )
    parser.add_argument(
        "--newton-reference-rows",
        type=int,
        default=0,
        help="optional subsample size for a costly Newton optimum check; 0 disables it",
    )
    parser.add_argument(
        "--log-submission",
        action="store_true",
        help="also create a results/submissions record for later AIcrowd entry",
    )
    args = parser.parse_args()

    commit, dirty = git_state()
    if commit is None or dirty is not False:
        parser.error(
            "commit or stash tracked changes before tuning; the selected run must "
            "be reproducible from a clean commit"
        )
    if args.selection_tolerance < 0:
        parser.error("--selection-tolerance must be non-negative")
    if args.newton_reference_rows < 0:
        parser.error("--newton-reference-rows must be non-negative")
    if any(value < 0 for value in args.lambdas):
        parser.error("--lambdas must contain only non-negative values")

    base_run = load_run(args.run, RUNS_DIR)
    base_config = base_run["config"]
    if base_config["model"]["name"] not in {
        "logistic_regression",
        "reg_logistic_regression",
    }:
        parser.error("the base run must use logistic regression")

    x_train, x_test, y_train, _, test_ids = load_data(args.data_dir)
    print(
        f"stage 1/3: gamma and convergence search: {args.gammas}, "
        f"checkpoints {args.checkpoints}"
    )
    convergence = checkpoint_cross_validate(
        x_train,
        y_train,
        base_config["preprocessing"],
        args.gammas,
        args.checkpoints,
        threshold=base_config.get("threshold", "train_f1"),
    )
    for row in convergence["candidates"]:
        print(
            f"  gamma={row['gamma']:g}, iterations={row['max_iters']:4d}: "
            f"F1 {row['f1_mean']:.4f} +- {row['f1_std']:.4f}, "
            f"loss {row['objective_mean']:.5f}"
        )
    if convergence["invalid_gammas"]:
        print("  rejected gammas:", convergence["invalid_gammas"])
    print(
        "  estimated conservative gamma limit across folds: "
        f"{convergence['estimated_gamma_limit_min']:.3f}"
    )
    optimizer = select_near_best(
        convergence["candidates"],
        args.selection_tolerance,
        preference=("max_iters", "gamma"),
    )

    print(
        f"stage 2/3: lambda grid at gamma={optimizer['gamma']:g}, "
        f"iterations={optimizer['max_iters']}"
    )

    def show_lambda(row):
        print(
            f"  lambda={row['lambda_']:g}: F1 {row['f1_mean']:.4f} "
            f"+- {row['f1_std']:.4f} ({row['seconds']:.1f} s)"
        )

    regularization = regularization_grid(
        base_config,
        x_train,
        y_train,
        gamma=optimizer["gamma"],
        max_iters=optimizer["max_iters"],
        lambdas=args.lambdas,
        on_result=show_lambda,
    )
    selected_lambda = select_near_best(
        regularization,
        args.selection_tolerance,
        preference=("lambda_",),
    )
    selected_config = {
        **base_config,
        "model": model_config(
            optimizer["gamma"],
            optimizer["max_iters"],
            selected_lambda["lambda_"],
        ),
    }

    stem = next_artifact_stem(args.output_dir)
    json_path = args.output_dir / f"{stem}.json"
    plot_path = args.output_dir / f"{stem}.png"
    result = {
        "base_run": args.run,
        "commit": commit,
        "rows": len(y_train),
        "search": {
            "gammas": args.gammas,
            "checkpoints": args.checkpoints,
            "lambdas": args.lambdas,
            "selection_tolerance": args.selection_tolerance,
        },
        "convergence": convergence,
        "selected_optimizer": optimizer,
        "regularization": regularization,
        "selected_config": selected_config,
    }
    if args.newton_reference_rows:
        print(
            f"optional diagnostic: Newton reference on "
            f"{args.newton_reference_rows:,} rows"
        )
        result["newton_reference"] = run_newton_reference(
            x_train,
            y_train,
            base_config["preprocessing"],
            selected_lambda,
            args.newton_reference_rows,
        )

    plot_search(
        convergence["candidates"],
        regularization,
        base_run["f1_mean"],
        optimizer,
        plot_path,
    )
    write_result(json_path, result)

    print("stage 3/3: official CV confirmation and full-data submission fit")
    official = run_experiment(
        selected_config,
        x_train,
        y_train,
        name="tuned-logistic",
        hypothesis="converged gamma, iteration count and lambda improve the v2 model",
        author="dg",
        baseline=args.run,
    )
    result["official_run"] = official
    rate = produce(
        selected_config,
        x_train,
        y_train,
        x_test,
        test_ids,
        args.submission,
        cv_f1=official["f1_mean"],
    )
    result["submission"] = {
        "path": str(args.submission),
        "predicted_positive": rate,
    }
    if args.log_submission:
        result["submission_record"] = log_submission(official, args.submission, rate)
    SUMMARY_PATH.write_text(summary_markdown(load_runs(), load_submissions()))
    write_result(json_path, result)

    print(f"selected config: {json.dumps(selected_config, sort_keys=True)}")
    print(
        f"official run {official['id']}: F1 {official['f1_mean']:.4f} "
        f"+- {official['f1_std']:.4f}"
    )
    print(f"wrote {args.submission} ({rate:.2%} predicted positive)")
    print(f"search details: {json_path}")
    print(f"report plot: {plot_path}")
    print(f"updated {SUMMARY_PATH}")
    print(
        "reproduce this selected model with: "
        f"python run.py --run {official['id']} --data-dir {args.data_dir} "
        f"--submission {args.submission}"
    )


if __name__ == "__main__":
    main()
