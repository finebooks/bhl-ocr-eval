# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub", "pandas", "pyarrow"]
# ///
"""DIAG: split unlimited-ocr2-multipage-diag.py window rows into one row per page.

The output has the consolidated shape (`PageID`, `raw`, `model`, `finish_reason`, ...), so it
goes through normalize_outputs.py and score_dataset.py unchanged. A window whose output has the
wrong number of `<PAGE>` markers, or that hit the token budget, cannot be split reliably: its
pages get an error sentinel (`__ERR__Producer:` / `__ERR__FinishReason:length`), as
consolidate_run.py writes for producer errors and truncations.

  uv run scripts/diag_uocr2_split_windows.py --run hf://buckets/.../multi16/ --out x.parquet
"""

import argparse
import re

import pandas as pd

PAGE_RE = re.compile(r"<PAGE>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="saturate output dir (hf:// or local)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    root = args.run.removeprefix("hf://").rstrip("/")
    windows = pd.concat([pd.read_parquet(fs.open(f)) for f in fs.glob(f"{root}/part-*.parquet")])
    # Keep the last record per window id (a retried window can have an earlier error row).
    windows = windows.drop_duplicates("id", keep="last")

    rows, counts = [], {"ok": 0, "truncated": 0, "page_mismatch": 0, "error": 0}
    for w in windows.itertuples():
        ids = list(w.page_ids) if w.page_ids is not None else []
        if isinstance(w.error, str) and w.error:
            counts["error"] += 1
            texts, status = None, f"__ERR__Producer:{w.error[:200]}"
        elif w.finish_reason == "length":
            counts["truncated"] += 1
            texts, status = None, "__ERR__FinishReason:length"
        else:
            segments = PAGE_RE.split(w.raw)[1:]
            if len(segments) != len(ids):
                counts["page_mismatch"] += 1
                texts, status = None, f"__ERR__Producer:page_mismatch {len(segments)}/{len(ids)}"
            else:
                counts["ok"] += 1
                texts, status = segments, None
        for k, pid in enumerate(ids):
            rows.append({
                "PageID": str(pid),
                "raw": texts[k] if texts else status,
                "model": "baidu/Unlimited-OCR-2",
                "finish_reason": "stop" if texts else "error",
                "window": w.id,
            })
    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    print(f"{len(windows)} windows {counts}; {len(out)} page rows, "
          f"{int(out.raw.str.startswith('__ERR__').sum())} sentinel rows -> {args.out}")


if __name__ == "__main__":
    main()
