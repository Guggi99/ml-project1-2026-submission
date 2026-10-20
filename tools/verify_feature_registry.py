"""Check docs/feature_registry.csv against the CDC codebook AND the real data.

For every column it answers three questions:

1. Does every code in ``nan_codes`` / ``zero_codes`` really occur in x_train?
   (A code that never occurs is usually written at the wrong scale.)
2. Does every "don't know / refused" code of the codebook that occurs in the
   data appear in ``nan_codes``?  Codes are looked up at four scales, because
   the CDC stores some computed columns with implied decimals and our data
   applies them for some columns but not for others (e.g. PAFREQ1_ 99000 in
   the codebook is 99 in x_train, _DRNKWEK 99900 stays 99900).
3. Are there values in the data that the codebook does not describe at all?

x_test is scanned too: a code absent from x_train may still occur there.

Usage, from the repo root (the codebook text is not in the repo):

    pdftotext -layout codebook15_llcp.pdf data/codebook15.txt
    python3 tools/verify_feature_registry.py data/codebook15.txt

Writes docs/feature_registry_check.csv (one row per column) and prints the
columns that need attention. Only the standard library and NumPy are used.
"""

import csv
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing import VALUE_MAPS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "docs" / "feature_registry.csv"
OUT = ROOT / "docs" / "feature_registry_check.csv"
SCALES = (1.0, 0.1, 0.01, 0.001)
DK = re.compile(r"^(Don.?t\s*know|Refused|Not\s*sure)", re.I)
NONE = re.compile(r"^(None|Never|No drinks)", re.I)


# ---------------------------------------------------------------------------
# CODEBOOK: one entry per variable with its value lines.
# ---------------------------------------------------------------------------
def parse_codebook(path):
    lines = Path(path).read_text().splitlines()
    heads = [k for k, line in enumerate(lines) if "SAS Variable Name:" in line]
    book = {}
    for a, k in enumerate(heads):
        name = re.search(r"SAS Variable Name:\s*(\S+)", lines[k]).group(1)
        end = heads[a + 1] if a + 1 < len(heads) else len(lines)
        if a + 1 < len(heads):
            while end > k and "Type:" not in lines[end]:
                end -= 1
        values, in_values = [], False
        for line in lines[k + 1 : end]:
            s = " ".join(line.split())
            if s.startswith("Value") and "Value Label" in s:
                in_values = True
                continue
            if not in_values or not s or s.startswith(("Notes:", "-")):
                continue
            m = re.match(r"^(\d+)(?:\s*-\s*(\d+))?\s+(.*)$", s)
            if m:
                lo = float(m.group(1))
                hi = float(m.group(2)) if m.group(2) else None
                label = re.sub(r"[\d,.\s]+$", "", m.group(3)).strip()
                values.append((lo, hi, label))
        book.setdefault(name, []).extend(values)
    return book


# ---------------------------------------------------------------------------
# DATA: value counts of every column of x_train, read in chunks.
# ---------------------------------------------------------------------------
def value_counts(path, chunk=40000):
    counts, nan = None, None
    with open(path) as f:
        header = next(f).strip().split(",")[1:]
        counts = [Counter() for _ in header]
        nan = np.zeros(len(header), dtype=int)
        rows = 0
        while True:
            block = [line for _, line in zip(range(chunk), f)]
            if not block:
                break
            x = np.genfromtxt(block, delimiter=",")[:, 1:]
            rows += len(x)
            nan += np.isnan(x).sum(axis=0)
            for j in range(x.shape[1]):
                col = x[~np.isnan(x[:, j]), j]
                v, c = np.unique(np.round(col, 6), return_counts=True)
                counts[j].update(dict(zip(v.tolist(), c.tolist())))
    return header, counts, nan, rows


def occurs(counter, value):
    return counter.get(round(value, 6), 0)


def best_scale(counter, entries):
    """Scale at which the data values fit the codebook entries best."""
    total = sum(counter.values()) or 1
    best, best_fit = 1.0, -1.0
    for s in SCALES:
        fit = 0
        for v, c in counter.items():
            for lo, hi, _ in entries:
                if (hi is None and abs(v - lo * s) < 1e-6) or (
                    hi is not None and lo * s - 1e-9 <= v <= hi * s + 1e-9
                ):
                    fit += c
                    break
        if fit / total > best_fit + 1e-9:
            best, best_fit = s, fit / total
    return best, best_fit


def main(codebook_path):
    book = parse_codebook(codebook_path)
    cache = ROOT / "data" / "value_counts.pkl"  # data/ is git-ignored
    if cache.exists():
        names, counts, nan, rows = pickle.load(open(cache, "rb"))
    else:
        names, counts, nan, rows = value_counts(ROOT / "data" / "x_train.csv")
        pickle.dump((names, counts, nan, rows), open(cache, "wb"))
    tcache = ROOT / "data" / "value_counts_test.pkl"
    if tcache.exists():
        test_counts = pickle.load(open(tcache, "rb"))
    else:
        test_counts = value_counts(ROOT / "data" / "x_test.csv")[1]
        pickle.dump(test_counts, open(tcache, "wb"))
    registry = {r["column"]: r for r in csv.DictReader(open(REGISTRY))}
    report = []
    for j, name in enumerate(names):
        r, cnt, entries = registry[name], counts[j], book.get(name, [])
        tcnt = test_counts[j]
        reg_nan = [float(c) for c in r["nan_codes"].split()]
        reg_zero = [float(c) for c in r["zero_codes"].split()]
        problems = []

        # 1. every registry code must occur in the data
        raw_codes = {lo for lo, hi, _ in entries if hi is None}
        for c in reg_nan + reg_zero:
            if occurs(cnt, c) == 0 and occurs(tcnt, c) == 0 and c not in raw_codes:
                problems.append(
                    f"registry code {c:g} occurs neither in the data nor in the codebook"
                )

        # 2. every codebook special code that occurs must be in the registry
        expected_nan, expected_zero = [], []
        for lo, hi, label in entries:
            if hi is not None:
                continue
            hits = [(s, occurs(cnt, lo * s) + occurs(tcnt, lo * s)) for s in SCALES]
            found = [(s, n) for s, n in hits if n > 0]
            if not found:
                continue
            s, n = found[0]
            code = lo * s
            if DK.match(label):
                expected_nan.append(code)
                if not any(abs(code - c) < 1e-6 for c in reg_nan):
                    problems.append(
                        f"codebook {lo:g} '{label[:30]}' occurs as {code:g} ({n}x) but is not in nan_codes"
                    )
            elif NONE.match(label) and r["type"] == "count":
                expected_zero.append(code)
                if r["decision"] != "drop" and not any(
                    abs(code - c) < 1e-6 for c in reg_zero + reg_nan
                ):
                    problems.append(
                        f"codebook {lo:g} '{label[:30]}' occurs as {code:g} ({n}x): not 0 and not NaN"
                    )

        # 3. values the codebook does not describe
        scale, fit = best_scale(cnt, entries) if entries else (1.0, 1.0)
        unexplained = []
        for v, c in sorted(cnt.items(), key=lambda kv: -kv[1]):
            ok = any(
                (hi is None and abs(v - lo * scale) < 1e-6)
                or (hi is not None and lo * scale - 1e-9 <= v <= hi * scale + 1e-9)
                for lo, hi, _ in entries
            )
            if not ok:
                unexplained.append(f"{v:g} ({c}x)")
        if entries and unexplained and r["decision"] != "drop":
            problems.append("values not in the codebook: " + ", ".join(unexplained[:5]))

        report.append(
            {
                "column": name,
                "decision": r["decision"],
                "missing_pct": f"{100 * nan[j] / rows:.1f}",
                "data_scale": f"{scale:g}",
                "fit_pct": f"{100 * fit:.1f}",
                "registry_nan": r["nan_codes"],
                "expected_nan": " ".join(f"{c:g}" for c in expected_nan),
                "registry_zero": r["zero_codes"],
                "none_never_codes": " ".join(f"{c:g}" for c in expected_zero),
                "top_values": "; ".join(f"{v:g}: {c}" for v, c in cnt.most_common(6)),
                "problems": " | ".join(problems),
            }
        )

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)
    flagged = [r for r in report if r["problems"]]
    print(f"{rows} rows, {len(report)} columns, {len(flagged)} with problems -> {OUT}")
    for r in flagged:
        print(f"\n{r['column']} [{r['decision']}] scale {r['data_scale']}")
        for p in r["problems"].split(" | "):
            print("   -", p)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "codebook15.txt")
