# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "huggingface-hub", "pandas", "pyarrow", "jiwer>=4,<5"]
# ///
"""Score OCR output that some OTHER tool already produced — the source-agnostic entrypoint.

Consumes any table with a per-page OCR column and a join key (an HF dataset, a local parquet, a vLLM
batch dump, a colleague's run), joins it to the GT sample (`text` / `body_text` / `furniture_text` /
`regions_json`), scores each (page, model) row with the frozen scorer, and prints the ordered report.
Point `--ocr-col` / `--key-col` / `--model-col` at whatever the producer named them — the harness makes
no assumptions about who generated the text. (To run the models yourself through an OpenAI-compatible
endpoint, use `run_openai.py` instead.)

  # a table with one row per (page, model):
  uv run runners/score_dataset.py --ocr ./some_run.parquet --ocr-col markdown --key-col PageID \
    --model-col model --run-provenance-file producer-run.json
  # OCR-specialist/native OCR mode (producer-run.json may contain {"prompt": null,
  # "note": "model-native OCR mode", "producer": "dots-ocr batch"}):
  uv run runners/score_dataset.py --ocr davanstrien/some-ocr-run --ocr-col markdown \
    --model dots-ocr --run-provenance-file producer-run.json
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scoring"))
import gt_score as GS  # noqa: E402
import report as RPT  # noqa: E402

# Producers signal a failed page with a sentinel STRING in the OCR column (a runner limitation, see
# DESIGN.md): our own runners use "__ERR__…", uv-scripts emit bracketed markers like "[OCR ERROR]" /
# "[SURYA GENERATE ERROR]". Treat all of these as errors so a half-failed run is EXCLUDED, not scored
# as if the model read nothing (which would silently inflate its CER).
_ERR_SENTINEL = re.compile(r"^\s*\[[^\]]*error[^\]]*\]\s*$", re.I)


def _is_error(t):
    return isinstance(t, str) and (t.startswith("__ERR__") or bool(_ERR_SENTINEL.match(t)))


# A truncation is an error row, but a DIFFERENT KIND of one: the model hit its token cap,
# which on this corpus is essentially always a repetition loop (see consolidate_run.py).
# That is the model failing on a page, not the run failing, so it excludes the page from
# the aggregates without making the model ineligible — otherwise a handful of looped pages
# out of 2165 would disqualify every neural model and leave only the classical engines
# rankable. Written by consolidate_run.py; run_openai.py emits the same shape.
def _is_truncated(t):
    return GS.is_truncated(t)   # shared with run_openai.py — see gt_score.TRUNCATION_SENTINEL_PREFIX


def _as_text(value, *, row_index, column):
    """Accept exact OCR strings (including empty); reject missing and non-string cells."""
    if isinstance(value, str):
        return value
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    kind = "missing" if missing else f"non-string {type(value).__name__}"
    raise ValueError(f"OCR row {row_index}, column {column!r} contains a {kind} value; "
                     "OCR cells must be strings (an empty string is valid).")


def _join_key(value):
    """Safely canonicalize only integer-like values needed for parquet dtype round-trips."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return GS.canonical_page_id(value)


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path_or_repo, split="train"):
    """Load OCR rows and return a deterministic source-content identity."""
    from datasets import load_dataset

    p = pathlib.Path(path_or_repo)
    if p.exists() and p.suffix == ".parquet":
        return pd.read_parquet(p), {"local_file_sha256": _file_sha256(p)}
    ds = load_dataset(str(p) if p.exists() else path_or_repo, split=split)
    fingerprint = getattr(ds, "_fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Loaded OCR dataset has no content fingerprint for run provenance.")
    return ds.to_pandas(), {"loaded_dataset_fingerprint": fingerprint}


def _read_run_provenance(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read --run-provenance-file {path} as JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--run-provenance-file must contain a JSON object.")
    if not value:
        raise ValueError("--run-provenance-file must contain a non-empty producer JSON object.")
    return value


def main():
    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr", required=True, help="OCR output: HF dataset id or local parquet/dir")
    ap.add_argument("--gt", default="davanstrien/bhl-eval-impact-sample")
    ap.add_argument("--gt-revision", "--dataset-revision", dest="gt_revision", default=None,
                    help="optional requested HF GT revision pinned in benchmark provenance")
    ap.add_argument("--ocr-col", default="markdown")
    ap.add_argument("--key-col", default="PageID", help="join key present in BOTH ocr + gt")
    ap.add_argument("--model-col", default=None, help="per-row model column (else use --model)")
    ap.add_argument("--model", default="model")
    ap.add_argument("--run-provenance-file", required=True, type=pathlib.Path,
                    help="JSON object describing the external producer run (prompt may be null)")
    ap.add_argument("--postproc-col", default="postproc_version",
                    help="column carrying the normalize_outputs registry version; when the input "
                         "has no such column the rows are scored as-is and stamped '0'")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write conditional diagnostics for incomplete models (they remain ineligible)")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent.parent / "data" / "scorecard.parquet"))
    args = ap.parse_args()

    resolved_revision = GS.resolve_dataset_revision(args.gt, args.gt_revision)
    gtds = load_dataset(args.gt, revision=resolved_revision, split="train")
    benchmark = GS.benchmark_provenance(
        args.gt, args.gt_revision, gtds, resolved_revision=resolved_revision,
        page_key=args.key_col)
    if "image" in gtds.column_names:  # the scorer needs text, not pixels — skip image decode
        gtds = gtds.remove_columns(["image"])
    if args.key_col not in gtds.column_names:
        raise ValueError(f"GT dataset has no join column {args.key_col!r}.")
    gt = {}
    for gt_row in gtds:
        key = _join_key(gt_row.get(args.key_col))
        if key is None:
            raise ValueError(f"GT dataset contains a missing {args.key_col!r} value.")
        if key in gt:
            raise ValueError(f"GT dataset contains duplicate {args.key_col}={key!r}.")
        gt[key] = gt_row
    ocr, ocr_identity = _load(args.ocr)
    required = {args.key_col, args.ocr_col}
    if args.model_col:
        required.add(args.model_col)
    if missing_columns := required - set(ocr.columns):
        raise ValueError(f"OCR input is missing required columns: {sorted(missing_columns)}")
    if ocr.empty:
        raise ValueError("OCR input has no rows.")
    postproc_col = args.postproc_col if args.postproc_col in ocr.columns else None
    postproc_note = postproc_col or "nowhere (scored as-is, stamped '0')"
    print(f"{len(ocr)} OCR rows vs {len(gt)} GT pages; join on {args.key_col}; "
          f"postproc from {postproc_note}")

    rows, identities, pages_by_model = [], set(), {}
    producer_provenance = _read_run_provenance(args.run_provenance_file)
    run_provenance = {
        **producer_provenance,
        "harness": {
            "runner": "score_dataset", "ocr_source": args.ocr, "gt_source": args.gt,
            "gt_requested_revision": args.gt_revision,
            "gt_resolved_revision": benchmark["resolved_revision"], "ocr_column": args.ocr_col,
            "key_column": args.key_col, "model_column": args.model_col,
            **ocr_identity,
        },
    }
    truncated_by_model: dict[str, int] = {}
    for index, record in enumerate(ocr.to_dict("records")):
        key = _join_key(record.get(args.key_col))
        if key is None:
            raise ValueError(f"OCR row {index} has a missing {args.key_col!r} value.")
        if key not in gt:
            raise ValueError(f"OCR row {index} has {args.key_col}={key!r}, which is not in the GT page set.")
        model_value = record.get(args.model_col) if args.model_col else args.model
        if model_value is None or (not isinstance(model_value, str) and pd.isna(model_value)) \
                or not str(model_value).strip():
            raise ValueError(f"OCR row {index} has a missing model ID.")
        model = str(model_value)
        identity = (model, key)
        if identity in identities:
            raise ValueError(f"Duplicate OCR row for model={model!r}, {args.key_col}={key!r}.")
        identities.add(identity)
        pages_by_model.setdefault(model, set()).add(key)
        text = _as_text(record.get(args.ocr_col), row_index=index, column=args.ocr_col)
        if _is_truncated(text):
            truncated_by_model[model] = truncated_by_model.get(model, 0) + 1
        row = GS.score_gt_row(gt[key], text, err=_is_error(text), truncated=_is_truncated(text),
                              page_id=gt[key][args.key_col], model=model)
        row["run_provenance"] = run_provenance
        row["benchmark_provenance"] = benchmark
        if postproc_col:
            postproc = record.get(postproc_col)
            if not isinstance(postproc, str) or not postproc.strip():
                raise ValueError(f"OCR row {index} has a missing or non-string {postproc_col!r} value; "
                                 "normalized input must carry its registry version on every row.")
            row["postproc_version"] = postproc
        else:
            row["postproc_version"] = "0"
        rows.append(row)

    incomplete = {model: sorted(set(gt) - pages) for model, pages in pages_by_model.items()
                  if pages != set(gt)}
    if incomplete and not args.allow_incomplete:
        details = "; ".join(f"{model}: {len(missing)} missing (examples: {missing[:3]})"
                            for model, missing in incomplete.items())
        raise ValueError("Every model must submit exactly one row for every GT page. " + details +
                         ". Use --allow-incomplete only for conditional diagnostics.")

    result = RPT.report(rows, expected_page_ids=gt, require_complete=not args.allow_incomplete,
                        truncated_by_model=truncated_by_model)
    print("\n" + RPT.format_report(result))
    flat = pd.DataFrame(GS.flat_scorecard_rows(rows))
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    flat.to_parquet(args.out, index=False)
    print(f"\nper-page scorecard -> {args.out}")


if __name__ == "__main__":
    main()
