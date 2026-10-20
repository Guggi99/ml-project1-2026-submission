"""Build docs/feature_registry.csv: one row per column of x_train, for #6.

The script fills the profile columns from the data (missing share, distinct
values, most frequent values, codes the preprocessing detects today). The
columns we fill by hand from the CDC 2015 codebook (type, special_codes,
decision, notes, checked_by) are kept when the script runs again, so it can be
rerun at any time without losing anyone's work.

Usage, from the repo root:

    python tools/make_feature_registry.py [--codebook codebook15.txt]

``--codebook`` takes the text version of the CDC codebook
(``pdftotext -layout codebook15_llcp.pdf codebook15.txt``) and adds each
column's question title. The codebook itself stays out of the repo.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import load_data  # noqa: E402
from src.preprocessing import fit_preprocessor  # noqa: E402

# ---------------------------------------------------------------------------
# SETTINGS: where the table goes, which columns are done first, and which
# fields the script fills (PROFILE) versus which we fill by hand (MANUAL).
# ---------------------------------------------------------------------------
OUT = Path("docs") / "feature_registry.csv"

# The 15 risk factors of the interaction experiment (25 Sep): the first batch.
FIRST_BATCH = [
    "_AGE80",
    "SEX",
    "GENHLTH",
    "_BMI5",
    "PHYSHLTH",
    "DIABETE3",
    "CVDSTRK3",
    "BPHIGH4",
    "TOLDHI2",
    "SMOKE100",
    "CHCCOPD1",
    "CHCKIDNY",
    "DIFFWALK",
    "HAVARTH3",
    "EMPLOY1",
]

PROFILE = [
    "column",
    "index",
    "label",
    "first_batch",
    "missing_pct",
    "n_distinct",
    "top_values",
    "detected_codes",
    "kept_now",
]
MANUAL = [
    "type",
    "special_codes",
    "decision",
    "notes",
    "checked_by",
    # machine-readable fields read by src/preprocessing.py (codes=True, ...)
    "nan_codes",
    "zero_codes",
    "not_asked_fill",
    "family",
    "priority",
]


# ---------------------------------------------------------------------------
# COLUMN NAMES, from the first line of x_train.csv.
# load_csv_data returns only numbers, so the names are read separately. The
# first header entry is "Id", which load_csv_data also removes from x_train;
# dropping it here keeps names and data columns aligned (name j = column j).
# ---------------------------------------------------------------------------
def column_names(data_dir="data"):
    """Feature names from the header of x_train.csv, without the Id column."""
    with open(Path(data_dir) / "x_train.csv") as f:
        return f.readline().strip().split(",")[1:]


# ---------------------------------------------------------------------------
# QUESTION TITLES, from the text version of the CDC 2015 codebook.
# Optional: without --codebook the label field stays as it was.
# ---------------------------------------------------------------------------
def codebook_titles(path):
    """Map variable name -> question title, read from the codebook text.

    Each entry looks like: title line, "Section: ... Type: Num" line, then a
    line ending in "SAS Variable Name: NAME". The title is the last non-empty
    line before the "Type:" line (calculated variables split the word
    "Section", so "Type:" is the reliable anchor).
    """
    lines = Path(path).read_text().splitlines()
    titles = {}
    for i, line in enumerate(lines):
        match = re.search(r"SAS Variable Name:\s*(\S+)", line)
        if not match:
            continue
        j = i - 1
        while j > 0 and "Type:" not in lines[j]:
            j -= 1
        j -= 1
        while j > 0 and not lines[j].strip():
            j -= 1
        titles.setdefault(match.group(1), " ".join(lines[j].split()))
    return titles


# ---------------------------------------------------------------------------
# MOST FREQUENT VALUES of one column, as "value: share; value: share; ...".
# Empty cells (NaN) are counted too, so a skipped follow-up question shows up
# as "NaN: 86.6%" instead of disappearing from the list.
# ---------------------------------------------------------------------------
def top_values(col, n=5):
    """The n most frequent values with their share, NaN included."""
    values, counts = np.unique(col[~np.isnan(col)], return_counts=True)
    shares = [(f"{v:g}", c / len(col)) for v, c in zip(values, counts)]
    missing = np.isnan(col).mean()
    if missing:
        shares.append(("NaN", missing))
    shares.sort(key=lambda s: -s[1])
    return "; ".join(f"{v}: {s:.1%}" for v, s in shares[:n])


# ---------------------------------------------------------------------------
# PROFILE: one dict per column with every field the script can measure.
# fit_preprocessor runs once on all training rows, only to record what
# today's preprocessing decides: which codes it turns into NaN
# (detected_codes) and whether it uses the column at all (kept_now).
# ---------------------------------------------------------------------------
def profile_rows(x, names, titles):
    """Measurable fields for every column of ``x``, in column order."""
    params = fit_preprocessor(x)
    kept = set(params["keep_cols"].tolist())
    rows = []
    for j, name in enumerate(names):
        col = x[:, j]
        rows.append(
            {
                "column": name,
                "index": j,
                "label": titles.get(name, ""),
                "first_batch": "yes" if name in FIRST_BATCH else "",
                "missing_pct": f"{np.isnan(col).mean() * 100:.1f}",
                "n_distinct": len(np.unique(col[~np.isnan(col)])),
                "top_values": top_values(col),
                "detected_codes": " ".join(
                    f"{c:g}" for c in params["sentinels"].get(j, ())
                ),
                "kept_now": "yes" if j in kept else "no",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# MAIN: build the table in five steps and write it.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook", type=Path)
    args = parser.parse_args()

    # 1. Names and data: column j of x_train belongs to names[j].
    names = column_names()
    x_train = load_data()[0]

    # 2. Question titles, only when the codebook text is given.
    titles = codebook_titles(args.codebook) if args.codebook else {}

    # 3. Read the existing table, if any, to keep what was filled by hand.
    previous = {}
    if OUT.exists():
        with open(OUT, newline="") as f:
            previous = {row["column"]: row for row in csv.DictReader(f)}

    # 4. Measure every column, then copy the hand-filled fields back in
    #    (and an earlier title when this run had no codebook).
    rows = profile_rows(x_train, names, titles)
    for row in rows:
        old = previous.get(row["column"], {})
        for key in MANUAL:
            row[key] = old.get(key, "")
        if not row["label"]:
            row["label"] = old.get("label", "")

    # 5. Write the table: script fields first, hand-filled fields last.
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE + MANUAL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT}: {len(rows)} columns, {len(FIRST_BATCH)} in the first batch")


if __name__ == "__main__":
    main()
