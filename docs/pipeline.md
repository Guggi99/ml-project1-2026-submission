# The pipeline — how a score and a submission are made

Everything runs through one file, [`src/pipeline.py`](../src/pipeline.py), driven by
one config dict. Each of us owns one part of it, so we can work in parallel and
every score still comes out of the same harness and is comparable.

| Who | Owns | File | Config key |
|---|---|---|---|
| Ahmed | preprocessing: missing values, codes, encodings, features | `src/preprocessing.py` | `"preprocessing"` |
| David | the six methods; the threshold rule (#11) | `implementations.py` | `"model"`, `"threshold"` |
| Peter | the chain itself, CV harness, tests, `run.py` | `src/pipeline.py`, `src/validation.py`, `tests/` | — |

## The whole process: a complete run in 11 steps

Gray steps are done by hand. Most of the time you only loop through steps 2–6:
change one thing, score it, compare. Phase B runs a few times in the whole
project, only for the run the team picked. The notebook
[`experiments/261004_pg_full_run.ipynb`](../experiments/261004_pg_full_run.ipynb, not included here)
has one cell per step from 2 to 9.

The same diagram as an image: [flow_11_steps.svg](flow_11_steps.svg). With the
file and the functions behind every step, including the inside of `fit_fn` and
`predict_fn`: [flow_complete_run_functions.svg](flow_complete_run_functions.svg).

```mermaid
flowchart TD
    s1["1. Commit your code<br/>by hand, git<br/>else the run is dirty and never counts"]

    subgraph A ["Phase A: every experiment - is this recipe better?"]
        direction TB
        s2["2. Write the config<br/>experiments/ notebook<br/>CONFIG, NAME, HYPOTHESIS, BASELINE"]
        s3["3. Run 5-fold CV<br/>src/pipeline.py, src/validation.py<br/>develop, cross_validate, f1_score"]
        s4["4. Log the run<br/>src/pipeline.py, src/runs.py<br/>run_experiment, log_run, git_state<br/>-> results/runs/ID.json"]
        s5["5. Compare with the baseline<br/>src/runs.py<br/>load_run, fold-by-fold F1 delta"]
        s6["6. Rebuild SUMMARY.md<br/>tools/make_summary.py, src/runs.py<br/>load_runs, summary_markdown"]
        s2 --> s3 --> s4 --> s5 --> s6
        s6 -. "next experiment" .-> s2
    end

    subgraph B ["Phase B: chosen run only - the AIcrowd file"]
        direction TB
        s7["7. Thursday decision<br/>by hand, the team reads SUMMARY.md<br/>fill in decision, pick the run id"]
        s8["8. Make the CSV<br/>run.py --run ID, src/pipeline.py<br/>load_run, submission_config, produce"]
        s9["9. Log the submission<br/>run.py --log, src/runs.py<br/>log_submission, sha256_of"]
        s10["10. Upload to AIcrowd<br/>by hand<br/>type in aicrowd_f1, then redo step 6"]
        s7 --> s8 --> s9 --> s10
    end

    s11["11. Clear outputs, commit<br/>by hand, git<br/>run JSON, SUMMARY.md, submission JSON"]

    s1 --> s2
    s6 --> s7
    s6 --> s11
    s10 --> s11

    classDef hand fill:#f1efe8,stroke:#5f5e5a,color:#000
    classDef phaseA fill:#eeedfe,stroke:#534ab7,color:#000
    classDef phaseB fill:#e1f5ee,stroke:#0f6e56,color:#000
    class s1,s7,s10,s11 hand
    class s2,s3,s4,s5,s6 phaseA
    class s8,s9 phaseB
```

Step 8 stops before writing anything when the run is dirty or used
`implementations_pg`, when the fit diverges, or when the full-data fit scores
far below the run's CV F1.

## Inside `fit_fn` and `predict_fn`

The same four steps run in Phase A (once per fold) and Phase B (once on all
rows). Everything is **learned from the training rows only** and then applied
unchanged to the rows being predicted — that is what keeps the CV score honest.

```mermaid
flowchart TB
    x["x<br/>raw rows"]
    subgraph fit ["fit_fn(x, y) - learns"]
        direction LR
        p1["fit_preprocessor(x)<br/>AHMED<br/>medians, scales, codes"]
        t1["transform(x)<br/>-> tx, no NaN,<br/>+ intercept column"]
        m1["fit_model(y, tx)<br/>DAVID<br/>-> weights w"]
        th["fit_threshold(tx @ w, y)<br/>DAVID #11<br/>-> cut-off"]
        p1 --> t1 --> m1 --> th
    end
    subgraph pred ["predict_fn(model, x_new) - applies"]
        direction LR
        t2["transform(x_new)<br/>same params"]
        s2["scores = tx_new @ w"]
        y2["score > cut-off ? 1 : -1"]
        t2 --> s2 --> y2
    end
    x --> p1
    th -- "model = (params, w, cut-off)" --> t2

    classDef ahmed fill:#e3f2e1,stroke:#2e7d32,color:#000
    classDef david fill:#e3ecfa,stroke:#1f5fbf,color:#000
    class p1 ahmed
    class m1,th david
```

## Trying your change

From the repo root, with the data in `data/`:

```python
from src.pipeline import BASELINE, load_data, develop

x_train, x_test, y_train, train_ids, test_ids = load_data()

config = {
    "preprocessing": {"one_hot": True},          # Ahmed: options of fit_preprocessor
    "model": {"name": "logistic_regression",     # David: any of the six methods
              "max_iters": 500, "gamma": 0.05},
    "threshold": "positive_rate",                # or "zero"; #11 adds better rules
}
result = develop(config, x_train, y_train)       # ~1-2 min on all rows
print(result["f1_mean"], result["f1_std"], result["f1_folds"])
```

`python -m src.pipeline` runs the baseline: **F1 0.3946 +- 0.0039** (MSE gradient
descent, 200 steps, gamma 0.05, positive-rate threshold). Compare your result
with that, on the same folds.

Good to know:

- **Labels:** the data has y in {-1, 1}. The pipeline converts to {0, 1} for the
  two logistic methods by itself — your functions receive what they expect.
- **`initial_w`** is always zeros, sized to the preprocessed matrix; a config
  never needs to know how many columns preprocessing produces.
- **Step size:** MSE gradient descent warns when gamma is too large to converge
  (gamma must stay below 2/L).

## What happens when you run an experiment

The three diagrams follow one call of `run_experiment` from a notebook cell, with
the function that does each step and where it lives. Example: the run
`261005_pg_longer-training` (logistic regression, 2000 steps). Line numbers are
those of `main` at commit `34dfb3a`; they shift when the files change, the
function names do not.

### 1. The whole cell

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 520}}}%%
flowchart TD
    a["<b>Notebook cell starts</b><br/>experiments/261005_pg_longer-training.ipynb<br/>calls load_data(), then run_experiment(config, x, y, ...)"]
    b["<b>Load the data</b><br/>load_data() — src/pipeline.py:69<br/>np.load(data/cache_full.npz) at :77, else load_csv_data() — src/helpers.py:8"]
    c["<b>Cross-validation, 5 folds</b> (diagram 2, about 7 min)<br/>run_experiment() — src/pipeline.py:157 → develop() — :149<br/>cross_validate() — src/validation.py:189, called at pipeline.py:152"]
    d["<b>Write the run record</b> (diagram 3)<br/>log_run() — src/runs.py:58, called at pipeline.py:168<br/>→ results/runs/261005_pg_longer-training.json"]
    e["<b>Printed result</b><br/>print() in run_experiment() — src/pipeline.py:178<br/>261005_pg_longer-training: F1 0.4065 +- 0.0039"]
    a --> b --> c --> d --> e

    classDef code fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    classDef out fill:#E1F5EE,stroke:#0F6E56,color:#085041
    class b,c,d code
    class e out
```

### 2. Inside one fold (runs 5 times)

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 520}}}%%
flowchart TD
    a["<b>Split the rows</b>: 262,508 train, 65,627 held back<br/>stratified_k_fold_indices() — src/validation.py:67<br/>loop over the 5 folds at validation.py:211"]
    b["<b>Learn and apply preprocessing</b> (321 → 249 columns)<br/>fit_fn() — pipeline.py:135 → fit_preprocessor() — preprocessing.py:82<br/>→ transform() — src/preprocessing.py:473, called at pipeline.py:137"]
    c["<b>Train</b>: 2000 gradient steps, gamma 0.05<br/>fit_model() — pipeline.py:84 → logistic_regression() — implementations.py:184<br/>(looked up by name with getattr at pipeline.py:106)<br/>each step: compute_logistic_gradient() :179 → sigmoid() :153"]
    d["<b>Choose the cut-off</b>: top 8.8 % of training scores<br/>fit_threshold('positive_rate') — src/pipeline.py:118, called at :139"]
    e["<b>Predict the held-back rows and score them</b><br/>predict_fn() — pipeline.py:142 → transform() — preprocessing.py:473<br/>f1_score() — validation.py:131, accuracy() — :151 (called at :214, :215)"]
    a --> b --> c --> d --> e

    classDef code fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    classDef out fill:#E1F5EE,stroke:#0F6E56,color:#085041
    class b,c,d code
    class e out
```

Everything in boxes 2–4 is learned from the 262,508 training rows only; the
held-back rows are just transformed and scored. After the 5th fold,
`cross_validate` returns the five F1 values and `develop` adds the runtime.

### 3. Logging the run

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 520}}}%%
flowchart TD
    a["<b>Check name and author, choose the id</b><br/>log_run() — src/runs.py:58 (NAME_PATTERN :30)<br/>261005_pg_longer-training, never overwritten (-2, -3 ...)"]
    b["<b>Which code version?</b> commit 34dfb3a, dirty: false<br/>git_state() — src/runs.py:33, called at runs.py:89<br/>runs git rev-parse and git status --porcelain"]
    c["<b>Write the JSON record</b><br/>record built in log_run(), written at src/runs.py:112<br/>→ results/runs/261005_pg_longer-training.json"]
    d["<b>Later, by hand: python tools/make_summary.py</b><br/>load_runs() — src/runs.py:116 → summary_markdown() — :126<br/>→ results/SUMMARY.md (+0.0043, 4/5 folds)"]
    a --> b --> c -.-> d

    classDef code fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    classDef out fill:#E1F5EE,stroke:#0F6E56,color:#085041
    class a,b code
    class c out
```

The last step is not part of the cell: run it yourself after one or more runs,
then commit the new JSON files together with `results/SUMMARY.md`.
