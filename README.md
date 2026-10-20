# CS-433 Project 1 — predicting MICHD from BRFSS data

EPFL Machine Learning (CS-433), Fall 2026. Binary classification of coronary
heart disease / heart attack (MICHD) from lifestyle and health survey features.
Deadline: **Thu 29 Oct 2026, 16:00** at mlcourse.epfl.ch.

> This is the repository we hand in: the code, run records and report at the
> submitted commit.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put `x_train.csv`, `y_train.csv`, `x_test.csv` into `data/` (gitignored — never
commit data or submissions). They are in
[`projects/project1/data/dataset_to_release.zip`](https://github.com/epfml/ML_course/tree/main/projects/project1/data)
in the course repo, or on
[AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1).

## Reproduce our best submission

```bash
python run.py --data-dir data --submission submission.csv
```

`run.py` is frozen to `261014_aa_tuned-logistic-v2-aspirin`. A clean checkout reproduces
the submitted CSV with SHA-256 prefix `0a97469b`, which scored **0.455 F1** and
**0.882 accuracy** on AIcrowd (CV F1 0.4437).


The frozen preprocessing has `flags=True`, including the `HAREHAB1` "question
was asked" flag. Because that question is asked only after a reported heart
attack, it is a survey-routing target proxy rather than a prospective medical
feature. The frozen run also has `aspirin_interactions=True`, and daily aspirin is
often a consequence of heart disease, so those columns partly reveal the label
too. We preserve both here to reproduce the scored CSV; the decision and the
no-flags ablation are discussed in the report (Limitations) and listed in
`results/SUMMARY.md`.

## Tune logistic fitting

The optimizer search checks convergence, gamma, iteration count and logistic
regularization around the current best recipe, confirms the selected settings
with the official five-fold runner, then writes a submission:

```bash
python tune.py --data-dir data --submission submission_tuned.csv
```

See [the model-tuning guide](docs/model-tuning.md) for the search space,
selection rule, output files and optional Newton diagnostic.

## Layout

| Path | Contents |
|---|---|
| `implementations.py` | The six graded methods (NumPy only) |
| `run.py` | Reproduces the best AIcrowd CSV exactly |
| `tune.py` | Tunes logistic convergence and regularization, then creates a candidate CSV |
| `src/helpers.py` | Course-provided CSV loader and submission writer |
| `src/preprocessing.py` | Fit/transform preprocessing and feature engineering |
| `src/validation.py` | Stratified k-fold CV, F1, accuracy — the only way we score models |
| `results/runs/`, `results/submissions/` | One JSON record per cross-validated run and per AIcrowd upload; `results/SUMMARY.md` is generated from them |
| `results/log.csv` | Early hand-kept results, before the run records |
| `report/` | Report PDF and LaTeX source |
| `docs/` | Feature registry (`feature_registry.csv`, read by the preprocessing), [pipeline notes](docs/pipeline.md), [model-tuning guide](docs/model-tuning.md) |
| `tests/` | Our own tests; the course grading tests run in CI as well |

## Rules of the project

- Python standard library and NumPy only. matplotlib/seaborn for plots only.
  No pandas, scikit-learn, PyTorch, external code or external data.
- Grading runs on Python 3.9 / NumPy 1.23.

## Team

Agreed at the kick-off on 28 Sep 2026.

| Role | Member | Owns |
|---|---|---|
| Pipeline, tests, validation | Peter Guggenbach (@Guggi99) | `src/pipeline.py`, `src/validation.py`, `src/runs.py`, `tests/`, CI, feature registry, `run.py` |
| Data preprocessing | Ahmed Azaiez (@Le-Azouz) | EDA, `src/preprocessing.py` |
| ML methods and models | David Gorgiev (@dgorgiev) | `implementations.py`, threshold, hyperparameters, class imbalance, report editing |

Everyone wrote the report section for the part they own.

## Use of AI tools

AI assistants were used in this project in compliance with EPFL rules. All design
decisions, code and text were reviewed and are the responsibility of the authors.
