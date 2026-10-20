"""Reproduce the team's best AIcrowd submission.

Must produce exactly the CSV that scored best on AIcrowd (fixed seeds, no
manual steps). The recipe is not written here but read from the record of the
run it was chosen from, so the CSV and its CV score always belong together.
Usage:

    python run.py                          # the frozen run (FROZEN_RUN)
    python run.py --run 261004_pg_logistic # any logged run
    python run.py --run ... --log          # also record it as a submission (#9)

The submission must use the team's implementations.py: runs made with the
reference implementations_pg.py are refused, and so are runs made on
uncommitted code (dirty). produce() then refuses a fit that diverges or whose
training F1 falls far below the run's CV F1.
"""

import argparse
from pathlib import Path

from src.helpers import load_csv_data
from src.pipeline import produce
from src.runs import RUNS_DIR, load_run, log_submission

# Id of the run in results/runs/ whose CSV scored best on AIcrowd. Set when the
# pipeline is frozen (#15); until then, pass --run.
FROZEN_RUN = "261014_aa_tuned-logistic-v2-aspirin"


def submission_config(run):
    """The config of a run record, checked that it can be reproduced and that it
    uses the team's methods."""
    # dirty = uncommitted code, None = no git state recorded: either way the
    # code that made the score is unknown
    if run["dirty"] is not False:
        raise ValueError(
            f"run {run['id']} has no clean commit (dirty={run['dirty']}); "
            "commit the code and repeat the run"
        )
    config = run["config"]
    module = config["model"].get("module", "implementations")
    if module != "implementations":
        raise ValueError(
            f"run {run['id']} used {module}; a submission must use "
            "implementations.py - repeat the run without 'module'"
        )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=FROZEN_RUN, help="run id in results/runs/")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--submission", type=Path, default=Path("submission.csv"))
    parser.add_argument(
        "--log", action="store_true", help="record in results/submissions/"
    )
    args = parser.parse_args()
    if args.run is None:
        parser.error("no frozen run yet: pass --run <id from results/runs/>")

    run = load_run(args.run, RUNS_DIR)
    config = submission_config(run)
    x_train, x_test, y_train, _, test_ids = load_csv_data(str(args.data_dir))
    rate = produce(
        config,
        x_train,
        y_train,
        x_test,
        test_ids,
        args.submission,
        cv_f1=run["f1_mean"],
    )
    print(f"wrote {args.submission}: {rate:.2%} predicted positive (run {run['id']})")

    if args.log:
        record = log_submission(run, args.submission, rate)
        print(f"logged submission {record['id']}: upload the CSV, then fill in")
        print(f"aicrowd_f1 in results/submissions/{record['id']}.json")


if __name__ == "__main__":
    main()
