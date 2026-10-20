"""Write results/SUMMARY.md from the run records in results/runs/ (#21).

Usage, from the repo root:

    python tools/make_summary.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runs import load_runs, load_submissions, summary_markdown  # noqa: E402

OUT = Path("results") / "SUMMARY.md"

if __name__ == "__main__":
    runs, submissions = load_runs(), load_submissions()
    OUT.write_text(summary_markdown(runs, submissions))
    print(f"wrote {OUT}: {len(runs)} runs, {len(submissions)} submissions")
