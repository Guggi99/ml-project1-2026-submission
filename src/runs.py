"""Run tracking: one JSON record per experiment, and the summary table (#21).

Every run that should count goes through ``src.pipeline.run_experiment``, which
calls :func:`log_run`. Each run gets its own file in ``results/runs/``, so two
people logging at the same time never touch the same file (no merge
conflicts). ``results/SUMMARY.md`` is generated from these files by
``python tools/make_summary.py`` and never edited by hand.

A record answers four questions:

- who ran what, and why: ``author``, ``name``, ``hypothesis``, ``baseline``
- can it be reproduced: ``commit`` and ``dirty`` (uncommitted code -> the run
  is exploratory and never goes into the report)
- was it measured the same way as its baseline: ``protocol`` = rows, folds and
  seed. The threshold rule is part of ``config``, not of the protocol: it is
  something we test (#11, #14), not how we measure.
- what came out: ``f1_folds``, ``f1_mean``, ``f1_std``, ``seconds``

``decision`` (keep / drop / unclear) starts empty and is filled in by hand at
the Thursday meeting.

Submissions (#9): ``run.py --log`` writes one record per AIcrowd candidate to
``results/submissions/``: which run it comes from, the commit, and a SHA-256
checksum of the CSV, so a later ``run.py`` can prove it reproduces the file
byte for byte (#15). ``aicrowd_f1`` is the one number typed in by hand, from
the AIcrowd page after the upload.
"""

import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

RUNS_DIR = Path("results") / "runs"
SUBMISSIONS_DIR = Path("results") / "submissions"
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def git_state():
    """Return ``(commit, dirty)`` of the working copy, ``(None, None)`` outside git.

    ``dirty`` is True when tracked files have uncommitted changes. Untracked
    files (a new notebook, say) do not count: they cannot change the code a
    committed run used.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(changes)


def log_run(
    result,
    config,
    n_rows,
    *,
    name,
    hypothesis,
    author,
    baseline=None,
    runs_dir=RUNS_DIR,
    today=None,
):
    """Write one run record as ``<runs_dir>/<yymmdd>_<author>_<name>.json``.

    ``result`` is the dict returned by ``develop``. An existing file is never
    overwritten: a second run with the same name the same day gets ``-2``,
    ``-3``, ... appended. Returns the record, including its ``id``.
    """
    for label, value in (("name", name), ("author", author)):
        if not NAME_PATTERN.match(value):
            raise ValueError(f"{label} {value!r}: use lowercase letters, digits, '-'")

    today = today or datetime.date.today()
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{today:%y%m%d}_{author}_{name}"
    suffix = 1
    while (runs_dir / f"{run_id}.json").exists():
        suffix += 1
        run_id = f"{today:%y%m%d}_{author}_{name}-{suffix}"

    commit, dirty = git_state()
    record = {
        "id": run_id,
        "date": today.isoformat(),
        "author": author,
        "name": name,
        "hypothesis": hypothesis,
        "baseline": baseline,
        "commit": commit,
        "dirty": dirty,
        "protocol": {
            "rows": int(n_rows),
            "folds": result["k"],
            "seed": result["seed"],
        },
        "config": config,
        "f1_folds": [round(f, 6) for f in result["f1_folds"]],
        "f1_mean": round(result["f1_mean"], 6),
        "f1_std": round(result["f1_std"], 6),
        "acc_mean": round(result["acc_mean"], 6),
        "seconds": result.get("seconds"),
        "decision": "",
    }
    (runs_dir / f"{run_id}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def load_run(run_id, runs_dir=RUNS_DIR):
    """The record of one run, by id."""
    path = Path(runs_dir) / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no run record {path}")
    return json.loads(path.read_text())


def sha256_of(path):
    """Hex SHA-256 of a file: equal checksums mean byte-identical files."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log_submission(
    run, csv_path, predicted_positive, *, submissions_dir=SUBMISSIONS_DIR, today=None
):
    """Write one submission record as ``<dir>/<yymmdd>_<n>.json`` (n = 1, 2, ...).

    ``run`` is the run record whose config produced ``csv_path``.
    """
    today = today or datetime.date.today()
    submissions_dir = Path(submissions_dir)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while (submissions_dir / f"{today:%y%m%d}_{n}.json").exists():
        n += 1
    commit, dirty = git_state()
    record = {
        "id": f"{today:%y%m%d}_{n}",
        "date": today.isoformat(),
        "run": run["id"],
        "cv_f1_mean": run["f1_mean"],
        "commit": commit,
        "dirty": dirty,
        "csv_sha256": sha256_of(csv_path),
        "predicted_positive": round(predicted_positive, 6),
        "aicrowd_f1": "",
        "aicrowd_accuracy": "",
    }
    path = submissions_dir / f"{record['id']}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def load_submissions(submissions_dir=SUBMISSIONS_DIR):
    """All submission records, sorted by id (date, then number)."""
    paths = sorted(Path(submissions_dir).glob("*.json"))
    return [json.loads(p.read_text()) for p in paths]


def load_runs(runs_dir=RUNS_DIR):
    """All run records, sorted by id: by date, then author, then name.

    Runs of the same day are therefore in alphabetical, not chronological
    order; the baseline column, not the row order, says what compares to what.
    """
    paths = sorted(Path(runs_dir).glob("*.json"))
    return [json.loads(p.read_text()) for p in paths]


def summary_markdown(runs, submissions=()):
    """The SUMMARY.md table: one row per run, compared with its baseline.

    "folds better" counts the folds in which the run beat its baseline; with
    the same seed both runs use the same folds, so this is a fair comparison
    that a single lucky fold cannot fake. Flags: ``dirty`` (not reproducible,
    exploratory only), ``protocol differs`` (rows, folds or seed differ from
    the baseline, so the delta means little), ``baseline
    missing`` (no record with that id).
    """
    by_id = {r["id"]: r for r in runs}
    lines = [
        "# Run summary",
        "",
        "Generated by `python tools/make_summary.py` from `results/runs/` —"
        " do not edit by hand.",
        "",
        "| run | author | hypothesis | F1 ± std | Δ vs baseline | folds better"
        " | flags | decision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        delta, better, flags = "", "", []
        base = by_id.get(r["baseline"]) if r["baseline"] else None
        if r["baseline"] and base is None:
            flags.append("baseline missing")
        if base is not None:
            delta = f"{r['f1_mean'] - base['f1_mean']:+.4f}"
            if r["protocol"] == base["protocol"]:
                wins = sum(a > b for a, b in zip(r["f1_folds"], base["f1_folds"]))
                better = f"{wins}/{len(r['f1_folds'])}"
            else:
                flags.append("protocol differs")
        if r["dirty"]:
            flags.append("dirty")
        lines.append(
            f"| `{r['id']}` | {r['author']} | {r['hypothesis']} "
            f"| {r['f1_mean']:.4f} ± {r['f1_std']:.4f} | {delta} | {better} "
            f"| {', '.join(flags)} | {r['decision']} |"
        )
    if submissions:
        lines += [
            "",
            "## Submissions",
            "",
            "CV F1 is the estimate, AIcrowd F1 the test score. A large gap means"
            " the CV protocol does not represent the test set.",
            "",
            "| submission | run | CV F1 | AIcrowd F1 | gap | predicted positive"
            " | CSV sha256 |",
            "|---|---|---|---|---|---|---|",
        ]
        for s in submissions:
            gap = ""
            if s["aicrowd_f1"] != "":
                gap = f"{float(s['aicrowd_f1']) - s['cv_f1_mean']:+.4f}"
            lines.append(
                f"| `{s['id']}` | `{s['run']}` | {s['cv_f1_mean']:.4f} "
                f"| {s['aicrowd_f1']} | {gap} | {s['predicted_positive']:.1%} "
                f"| `{s['csv_sha256'][:12]}` |"
            )
    return "\n".join(lines) + "\n"
