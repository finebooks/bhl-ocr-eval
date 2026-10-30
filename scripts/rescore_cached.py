# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "pandas", "pyarrow", "jiwer==4.0.0", "rapidfuzz==3.14.5"]
# ///
"""Re-score existing scorecards under the CURRENT normalizer/scorer, from the cached OCR text.

A norm bump has to move every row at once, and the board refuses mixed norm versions. This rebuilds
each scorecard row from the exact text that produced it: the OCR file is found by the
`local_file_sha256` recorded in the row's own run provenance (at its recorded path, or under a
`--search` directory when that path has moved), so a re-score can never quietly read a different
run. The producer and benchmark provenance and the postproc version are carried over unchanged;
only the scores, `norm_version` and `score_provenance` are new.

  uv run scripts/rescore_cached.py data/full-2026-08/scorecard*.parquet \
      --search data/full-2026-08 data/full-2026-08/normalized --out-dir data/full-2026-08/rescored

Per model it prints how many pages changed. `--expect-unchanged-without-tables` (on by default)
also asserts that every page whose text holds no table the normalizer regroups scores
identically to before — the regression guard for a table-only change.
"""
import argparse
import hashlib
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scoring"))
sys.path.insert(0, str(ROOT / "runners"))
import gt_score as GS  # noqa: E402
import normalizers as N  # noqa: E402
import score_dataset as SD  # noqa: E402

SCORE_PREFIXES = ("cer_", "recall", "over_", "furniture_", "region_", "script_match")
CARRIED = ("run_provenance", "benchmark_provenance", "postproc_version")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_source(harness, search, cache, override=None):
    """The OCR parquet whose content hash matches the run provenance — never a same-named stand-in.
    `override` (from --source) is used as given; the no-table guard then checks its content."""
    if override is not None:
        return override
    want = harness["local_file_sha256"]
    candidates = [pathlib.Path(harness["ocr_source"]), ROOT / harness["ocr_source"]]
    candidates += [p for d in search for p in sorted(pathlib.Path(d).glob("*.parquet"))]
    for path in candidates:
        if path.is_file():
            if path not in cache:
                cache[path] = _sha256(path)
            if cache[path] == want:
                return path
    raise SystemExit(f"no OCR file with sha256 {want} (recorded source {harness['ocr_source']})")


def _same_scores(old, new):
    for key, value in new.items():
        if not key.startswith(SCORE_PREFIXES):
            continue
        before = old.get(key)
        if isinstance(value, dict):
            if json.loads(before) != value:
                return False
        elif (before is None or pd.isna(before)) != (value is None):
            return False
        elif value is not None and abs(float(before) - float(value)) > 1e-12:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scorecards", nargs="+", type=pathlib.Path)
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--search", nargs="*", default=[], help="directories (not recursive) to look in for moved OCR files")
    ap.add_argument("--source", action="append", default=[], metavar="MODEL=PATH",
                    help="OCR file for a model whose recorded file was rewritten (same text, new bytes). "
                         "Needs the no-table guard: every table-free page must then re-score identically, "
                         "which checks the substitute's content page by page")
    ap.add_argument("--no-expect-unchanged-without-tables", dest="guard", action="store_false")
    args = ap.parse_args()
    overrides = {m: pathlib.Path(p) for m, p in (item.split("=", 1) for item in args.source)}
    if overrides and not args.guard:
        raise SystemExit("--source needs the no-table guard")
    from datasets import load_dataset

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt_cache, sha_cache = {}, {}
    for card in args.scorecards:
        old = pd.read_parquet(card)
        rows = []
        for model, group in old.groupby("model", sort=False):
            run = json.loads(group["run_provenance"].iloc[0])
            harness = run["harness"]
            bench = json.loads(group["benchmark_provenance"].iloc[0])
            gt_key = (bench["dataset_id"], bench["resolved_revision"])
            if gt_key not in gt_cache:
                ds = load_dataset(gt_key[0], revision=gt_key[1], split="train").remove_columns(["image"])
                gt_cache[gt_key] = {GS.canonical_page_id(r[harness["key_column"]]): r for r in ds}
            gt = gt_cache[gt_key]
            source = _find_source(harness, args.search, sha_cache, overrides.get(model))
            ocr = pd.read_parquet(source)
            if harness.get("model_column"):
                ocr = ocr[ocr[harness["model_column"]] == model]
            text_by_key = {GS.canonical_page_id(k): t for k, t in zip(ocr[harness["key_column"]],
                                                                       ocr[harness["ocr_column"]], strict=True)}
            changed = 0
            for record in group.to_dict("records"):
                key = GS.canonical_page_id(record["page_id"])
                text = text_by_key[key]
                err, truncated = SD._is_error(text), SD._is_truncated(text)
                if (bool(record["error"]), bool(record["truncated"])) != (err, truncated):
                    raise SystemExit(f"{model} page {key}: error/truncated flags differ from the stored row")
                row = GS.score_gt_row(gt[key], text, err=err, truncated=truncated,
                                      page_id=gt[key][harness["key_column"]], model=model)
                for field in CARRIED:
                    row[field] = record[field]
                same = _same_scores(record, row)
                changed += not same
                if args.guard and not same and not err and N.regroup_tables(text) == text:
                    raise SystemExit(f"{model} page {key}: score moved but no table was regrouped")
                rows.append(row)
            print(f"{card.name}: {model}: {len(group)} pages, {changed} re-scored differently "
                  f"(source {source})")
        frame = pd.DataFrame(GS.flat_scorecard_rows(rows))
        frame.to_parquet(args.out_dir / card.name, index=False)
        print(f"  -> {args.out_dir / card.name}")


if __name__ == "__main__":
    main()
