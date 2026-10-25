# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "huggingface-hub", "pandas", "pyarrow", "jiwer>=4,<5"]
# ///
"""Diagnostic: the board's reading-lane CER with diacritics folded on both sides.

NOT a board metric. The reading lane deliberately keeps accents (scoring/normalizers.py), and
the published numbers stay as they are. This answers one question for RESULTS.md: how much of a
row's CER is accent loss (é -> e, æ -> e) rather than misread or missing text?

It runs the real scorer (runners/score_dataset.py) and the real aggregation (scoring/report.py)
over every row's cached normalized output, with one change: after the reading lane, both GT and
OCR text are NFD-decomposed, combining marks dropped, and æ/œ spelled ae/oe. Nothing else about
scoring moves. `--control` skips the change and must reproduce data/leaderboard.json exactly,
which is how the harness is checked.

  uv run scripts/diag_accent_folded.py            # -> data/full-2026-08/diag-accent-folded/folded.json
  uv run scripts/diag_accent_folded.py --control  # -> .../control.json, compared to the board
"""
import argparse
import json
import pathlib
import sys
import unicodedata

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scoring"))
sys.path.insert(0, str(ROOT / "runners"))
import normalizers as N  # noqa: E402
import report as RPT  # noqa: E402
import score_dataset  # noqa: E402

DATA = ROOT / "data" / "full-2026-08"
GT = "davanstrien/bhl-eval-impact-full-2165-v1"
GT_REVISION = "3ee88b4bfd553622561b8b2c002ac4e293a5e06e"


def fold_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    kept = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return kept.replace("æ", "ae").replace("œ", "oe")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", action="store_true",
                    help="score without folding; must match data/leaderboard.json")
    args = ap.parse_args()

    if not args.control:
        reading = N.LANES["reading"]
        N.LANES["reading"] = lambda t: fold_diacritics(reading(t))

    name = "control" if args.control else "folded"
    out_dir = DATA / "diag-accent-folded" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    for normalized in sorted((DATA / "normalized").glob("*.parquet")):
        slug = normalized.stem
        sys.argv = ["score_dataset.py", "--ocr", str(normalized), "--ocr-col", "markdown",
                    "--key-col", "PageID", "--model-col", "model",
                    "--gt", GT, "--gt-revision", GT_REVISION,
                    "--run-provenance-file", str(DATA / "provenance" / f"{slug}.json"),
                    "--out", str(out_dir / f"scorecard-{slug}.parquet")]
        score_dataset.main()

    frames = [pd.read_parquet(p) for p in sorted(out_dir.glob("scorecard-*.parquet"))]
    rows = pd.concat(frames, ignore_index=True).to_dict("records")
    result = RPT.report(rows, require_complete=True)
    out = DATA / "diag-accent-folded" / f"{name}.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"{name} -> {out}")


if __name__ == "__main__":
    main()
