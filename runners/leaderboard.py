# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow", "jiwer>=4,<5"]
# ///
"""Combine every runner's per-page scorecard into one validated, ordered leaderboard.

Each runner (run_openai, score_dataset) writes a per-page scorecard parquet carrying a
`model` column. This globs them, concatenates the rows, and runs the single report across all models —
so hosted VLMs, self-served/uv-scripts specialists, and the tesseract baseline land on one board with
consistent micro-averaging, volume-bootstrap CIs, per-volume metrics and eligibility.

  uv run runners/leaderboard.py 'data/scorecards/*.parquet'
"""
import argparse
import glob
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scoring"))
import report as RPT  # noqa: E402

def _parse_args(argv=None):
    default = str(pathlib.Path(__file__).resolve().parent.parent / "data" / "scorecards" / "*.parquet")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="*", default=[default],
                        help="scorecard parquet glob(s); defaults to data/scorecards/*.parquet")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="include deliberately partial scorecards as conditional/ineligible diagnostics")
    parser.add_argument("--policy-diagnostics", action="store_true",
                        help="include nested full-text diplomatic/reading CER diagnostics")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    patterns = args.patterns
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"ERROR: no scorecards matched {patterns}")
    print(f"combining {len(paths)} scorecards:")
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        required = {"model", "page_id"}
        if missing := required - set(df.columns):
            raise ValueError(f"Scorecard {p} is missing required columns: {sorted(missing)}")
        print(f"  {pathlib.Path(p).name}: {len(df)} rows, {df['model'].nunique()} model(s)")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    rows = df.to_dict("records")
    result = RPT.report(rows, require_complete=not args.allow_incomplete,
                        include_policy_diagnostics=args.policy_diagnostics)
    print("\n" + RPT.format_report(result))
    out = pathlib.Path(__file__).resolve().parent.parent / "data" / "leaderboard.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nleaderboard -> {out}")


if __name__ == "__main__":
    main()
