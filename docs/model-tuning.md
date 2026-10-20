# Model fitting and hyperparameters (#12)

Issue #12 tunes the optimizer around the current best recorded recipe instead
of copying that recipe into another notebook. The command reads the complete
preprocessing and threshold configuration from
`results/runs/261011_pg_v2-train-f1.json`.

## What the command does

1. It evaluates logistic gradient descent at 500, 1,000, 2,000 and 3,000
   iterations for gamma 0.05, 0.08, 0.10 and 0.12. Each fold is preprocessed
   once and each gamma is trained once to its largest checkpoint.
2. It selects the least expensive setting within 0.0005 F1 of the maximum.
3. At that gamma and iteration count, it runs the standard five-fold pipeline
   for lambda 0, 1e-6, 1e-5, 1e-4, 1e-3 and 1e-2.
4. It confirms the selected configuration through `run_experiment`, trains it
   on every labelled row, and writes a submission.

The threshold is fitted on each fold's training rows. Validation rows are used
only to measure F1. The final confirmation therefore follows the same protocol
as every record in `results/runs/`.

The six functions in `implementations.py` are unchanged. Intermediate weights
are exposed only by `src/tuning.py`, whose update matches those functions and
is covered by equivalence tests.

## Selected result and limitation

The selected run is `261012_dg_tuned-logistic`: gamma 0.1, 1,000 iterations,
lambda 0, and the training-F1 threshold. It achieved 0.4394 +- 0.0036 in
five-fold CV and 0.450 F1 / 0.883 accuracy on AIcrowd. The preceding recorded
AIcrowd submission scored 0.418 F1, so the test-set improvement is +0.032.

This run inherits the v2 preprocessing option `flags=True`, which includes the
`HAREHAB1` "was asked" flag. `HAREHAB1` is only asked after a reported heart
attack and is therefore a survey-routing target proxy. It remains enabled so
the frozen run reproduces the verified AIcrowd CSV; measuring and deciding on
a no-leak replacement is tracked separately in issue #23.

## Why gamma 0.05 is not the upper limit

For mean logistic loss, if `L` is the largest eigenvalue of `X.T @ X / N`, the
curvature is bounded by `L / 4`. The conservative unregularized condition is
therefore `gamma < 8 / L`, rather than the MSE condition `gamma < 2 / L`.
The tuning command estimates this limit inside every fold with power iteration
and reports the smallest estimate.

## Run it

The command deliberately refuses a dirty working tree: commit the experiment
implementation first so the resulting run can be reproduced by `run.py`.

```bash
source .venv/bin/activate
python tune.py --data-dir data --submission submission_tuned.csv
```

The search JSON and a three-panel report plot are written under
`results/tuning/`. The terminal prints the selected parameters, official run
ID, CV F1, and the exact `run.py` command that reproduces the submission. The
panels show CV F1 by optimizer setting, training-loss convergence, and CV F1 by
lambda. `results/SUMMARY.md` is regenerated after the official run.

Newton's method is available only as an optional diagnostic on a subsample.
It is not the default because constructing a Hessian on the full wide matrix is
substantially more expensive than gradient descent:

```bash
python tune.py --data-dir data --submission submission_tuned.csv \
  --newton-reference-rows 30000
```

Use `--log-submission` only for a CSV that will actually be uploaded. The
created submission record leaves the AIcrowd score blank for manual entry.
