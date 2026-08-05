# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface-hub", "pandas", "pyarrow"]
# ///
"""Turn a saturate run's part files into one table the scorer can read.

This is the missing step between `drivers/*.py` and `runners/normalize_outputs.py`. The drivers
write via saturate, whose output shape does not match what scoring expects in three ways, each of
which fails LATE — after the GPU time is spent:

1. **The id column is called `id`.** saturate keys every row by `id` (a string), while
   `score_dataset.py` joins ground truth on one `--key-col` used for both sides, and the benchmark
   calls it `PageID`. Launch commands pass `--id-column PageID` so the VALUES are page ids, but the
   COLUMN NAME is still `id`. Scoring fails closed on this (`missing required columns: ['PageID']`),
   so nothing is corrupted — it just fails after the run.

2. **Error rows carry no `raw` key at all.** `saturate/sink.py` writes failures as `{id, error}`,
   which lands as a parquet null. `normalize_outputs._as_text` is called outside its try block, so a
   SINGLE null raw raises and kills the entire normalize pass for that model. Two models in the
   2026-08 run (glm-ocr, qianfan-ocr) have exactly 2 such rows each.

3. **Truncation is recorded but never acted on.** DESIGN.md states a response whose finish reason is
   not `stop` "is recorded as an error page — excluded from aggregates and retryable — never as a
   transcription, because a truncated read scored as a short read silently corrupts the comparison."
   That rule lives only in `run_openai.py`; the saturate drivers store `finish_reason` and move on.

So this runner renames the key, fills error rows with a durable sentinel, and applies the truncation
rule — then reports the counts, because those counts are a board column, not a diagnostic.

**Truncated is not the same as errored.** Both are excluded from the scored text, but they mean
different things: an errored page is a run defect (retryable), while a truncated page is the MODEL
failing on that page — on this corpus, essentially always a repetition loop hitting the token cap.
They are written with distinguishable sentinels (`__ERR__FinishReason:*` vs `__ERR__Producer:*`) so
the scorer can keep that distinction rather than collapsing both into ineligibility.

  uv run runners/consolidate_run.py \
      --run hf://buckets/finebooks/bhl-ocr-runs/full-2026-08/glm-ocr/ \
      --model zai-org/GLM-OCR --out data/full-2026-08/consolidated/glm-ocr.parquet

  # then the normal path:
  uv run runners/normalize_outputs.py --raw data/full-2026-08/consolidated/glm-ocr.parquet \
      --raw-col raw --out-col markdown --model-col model
"""

import argparse
import json
import pathlib
import sys

ERROR_PREFIX = "__ERR__"  # matches runners/normalize_outputs.py and score_dataset._is_error


def _read_parts(run: str):
    """All `part-*.parquet` under a run prefix -> one DataFrame. Local dir or hf:// URI."""
    import pandas as pd

    if run.startswith(("hf://", "s3://", "gs://")):
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        paths = sorted(fs.glob(f"{run.rstrip('/')}/part-*.parquet"))
        if not paths:
            raise SystemExit(f"no part-*.parquet under {run!r}")
        frames = []
        for p in paths:
            with fs.open(p, "rb") as fh:
                frames.append(pd.read_parquet(fh))
        return pd.concat(frames, ignore_index=True), len(paths)

    directory = pathlib.Path(run)
    paths = sorted(directory.glob("part-*.parquet")) if directory.is_dir() else [directory]
    if not paths:
        raise SystemExit(f"no part-*.parquet under {run!r}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True), len(paths)


def consolidate(frame, *, id_col, key_col, raw_col, error_col, finish_col, expect_rows=None):
    """Rename the key, seal error and truncated rows, and return (frame, counts).

    Never drops a row: a page that vanishes here would fail the scorer's completeness gate as a
    missing page, which reads as "the model was not run on it" rather than "it failed".
    """
    import pandas as pd

    if id_col not in frame.columns:
        raise SystemExit(f"input has no {id_col!r} column; got {sorted(frame.columns)}")
    if key_col in frame.columns and key_col != id_col:
        raise SystemExit(f"input already has a {key_col!r} column as well as {id_col!r}; "
                         "refusing to guess which one scoring should join on")
    frame = frame.rename(columns={id_col: key_col})

    superseded = 0
    if frame[key_col].duplicated().any():
        # A RETRY (`--retry-errors`) appends a fresh row for each failed id and leaves the
        # original error row in place, so a retried prefix legitimately holds two rows per
        # retried page. Resolve that ONE way only: a successful row supersedes an error row
        # for the same id. Two successful rows for one id is not a retry — it is two runs
        # blended into one prefix, which is the exact corruption the new-prefix rule exists
        # to prevent, so that still refuses.
        is_err_row = frame[raw_col].isna() if raw_col in frame.columns else pd.Series(False, index=frame.index)
        if error_col in frame.columns:
            is_err_row = is_err_row | frame[error_col].notna()
        dup_ids = frame.loc[frame[key_col].duplicated(keep=False), key_col].unique()
        ambiguous = [i for i in dup_ids
                     if (~is_err_row[frame[key_col] == i]).sum() > 1]
        if ambiguous:
            raise SystemExit(
                f"{len(ambiguous)} id(s) have MORE THAN ONE successful row, e.g. {ambiguous[:5]} — "
                "that is two runs written to one prefix, not a retry. Refusing to consolidate."
            )
        # Keep the successful row per id: a stable sort puts non-error rows first, so taking
        # the first occurrence after sorting selects the success over the error it superseded.
        order = frame.assign(_err=is_err_row).sort_values(["_err"], kind="stable")
        keep = order.drop_duplicates(subset=[key_col], keep="first").index
        superseded = len(frame) - len(keep)
        frame = frame.loc[sorted(keep)].reset_index(drop=True)

    raw = frame[raw_col] if raw_col in frame.columns else pd.Series([None] * len(frame))
    errors = frame[error_col] if error_col in frame.columns else pd.Series([None] * len(frame))
    finish = frame[finish_col] if finish_col in frame.columns else pd.Series([None] * len(frame))

    # 1. producer errors: no `raw` key at all -> parquet null. Seal with the recorded reason.
    is_err = raw.isna() | errors.notna()
    # 2. truncation: DESIGN.md's rule, applied to rows that DID return text.
    is_trunc = (~is_err) & finish.notna() & (finish != "stop")

    sealed = raw.copy()
    sealed[is_err] = [f"{ERROR_PREFIX}Producer:{e if pd.notna(e) else 'missing raw'}"
                      for e in errors[is_err]]
    sealed[is_trunc] = [f"{ERROR_PREFIX}FinishReason:{f}" for f in finish[is_trunc]]
    # Keep the pre-sealing completion. Sealing is a SCORING decision, and this repo's
    # standing promise is that raw model output is cached once and re-scored forever —
    # overwriting it in place would make a truncated page permanently uninspectable from
    # the consolidated artifact, and those pages are exactly the ones worth reading (the
    # loop usually starts AFTER a correctly transcribed prefix). Costs a column.
    frame[f"{raw_col}_verbatim"] = raw
    frame[raw_col] = sealed

    counts = {
        "rows": int(len(frame)),
        "ok": int((~is_err & ~is_trunc).sum()),
        "producer_errors": int(is_err.sum()),
        "truncated": int(is_trunc.sum()),
        "finish_reasons": {str(k): int(v) for k, v in finish.value_counts().items()},
    }
    counts["truncation_rate"] = round(counts["truncated"] / max(counts["rows"], 1), 4)
    counts["superseded_by_retry"] = superseded
    if expect_rows is not None and counts["rows"] != expect_rows:
        raise SystemExit(f"expected {expect_rows} rows, got {counts['rows']} — "
                         "incomplete run; scoring would fail closed on the page set anyway")
    return frame, counts


def main():
    ap = argparse.ArgumentParser(
        description="Turn a saturate run's part files into one table the scorer can read.")
    ap.add_argument("--run", required=True, help="run prefix (hf://buckets/... or a local dir)")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--model", default=None, help="override/set the model column")
    ap.add_argument("--id-col", default="id", help="saturate's id column (default: id)")
    ap.add_argument("--key-col", default="PageID", help="column scoring joins on (default: PageID)")
    ap.add_argument("--raw-col", default="raw")
    ap.add_argument("--error-col", default="error")
    ap.add_argument("--finish-col", default="finish_reason")
    ap.add_argument("--expect-rows", type=int, default=None,
                    help="fail unless the run has exactly this many rows")
    args = ap.parse_args()

    frame, n_parts = _read_parts(args.run)
    frame, counts = consolidate(frame, id_col=args.id_col, key_col=args.key_col,
                                raw_col=args.raw_col, error_col=args.error_col,
                                finish_col=args.finish_col, expect_rows=args.expect_rows)
    if args.model:
        frame["model"] = args.model

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)

    counts["run"] = args.run
    counts["parts"] = n_parts
    counts["out"] = str(out)
    print(json.dumps(counts, indent=2), file=sys.stderr)
    print(f"{counts['ok']} ok · {counts['truncated']} truncated "
          f"({counts['truncation_rate']:.1%}) · {counts['producer_errors']} errors -> {out}")


if __name__ == "__main__":
    main()
