# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "kraken @ git+https://github.com/mittagessen/kraken@3e6893a0d5eb06273494bfcdb9d3cac6159e0611",
#     "datasets>=3.1.0",
#     "huggingface-hub",
#     "pandas",
#     "pyarrow",
#     "pillow",
# ]
# ///
"""kraken PP-OCRv6 — the line-based ATR engine, as a self-contained inference driver.

Run:
    hf jobs uv run --detach --flavor a10g-small -s HF_TOKEN --timeout 8h \\
        drivers/kraken-ppocrv6-port.py -- \\
        --input-dataset <benchmark> --revision <sha> --id-column PageID \\
        --output hf://buckets/<owner>/<bucket>/<prefix>/

Inference only. It writes `part-*.parquet` files of `{id, raw, error, ...}` and stops, which is
the shape `runners/consolidate_run.py` already reads; scoring is the same path every other row
takes. Nothing here cleans text — `runners/normalize_outputs.py` owns every transform, so a
judgement call about the output costs a re-score rather than a re-run.

WHAT MAKES THIS ROW DIFFERENT FROM EVERY OTHER ONE ON THE BOARD

Every other model is one checkpoint doing page-in/text-out. kraken is a two-stage pipeline: a
baseline SEGMENTER finds the text lines and puts them in reading order, then a CTC line
RECOGNISER transcribes each one. Two consequences the board has to state rather than hide:

  1. A score here is attributable to the PAIR, not to the recogniser alone. If segmentation
     misses lines, the recogniser wears the recall loss. `n_lines` and `n_records` are recorded
     per page to show how much of a page reached the recogniser at all.

     They do NOT detect every dropped line, and it is worth being precise about why. When
     polygonisation fails, `calculate_polygonal_environment` returns None and the caller simply
     does not build a line (kraken/lib/vgsl/spred.py:141-147) — so the line never enters
     `Segmentation.lines`, `n_lines` already excludes it, and the two counts agree. Those drops
     are silent recall loss and the only record of them is a "Polygonizer failed" WARNING in the
     job log. Count those warnings; the count, not the line number, is the information (kraken
     polygonises one baseline at a time, so the message always says "line 0"). A future run
     wanting this as a column should either count those log records or pass
     `raise_on_error=True` on the segmentation config, which converts the drops into exceptions.
  2. The segmenter is not a choice we made. `SegmentationTaskModel.load_model()` with no path
     loads `blla.mlmodel` from inside the kraken package, so the segmenter's identity is pinned
     by the kraken commit and is exactly what a user gets out of the box. Same policy as the
     Falcon row's vendor pipeline: score the deployable product, not a tuned variant of it.

There is also a structural claim worth measuring rather than repeating. The model card says the
family "should offer similar accuracy and generalization to VLM-based recognizers WITHOUT
HALLUCINATIONS and with vastly higher throughput". The board's sparse/blank stratum is the
instrument for the first half of that: a CTC recogniser cannot invent a paragraph where the
segmenter found no line, whereas board VLMs reach sparse CER 13-27. A blank page here should
produce an empty string, and an empty string is a VALID transcription, not an error — so this
driver must never register `require_non_empty` downstream, and an empty read is written as such.

WHY THE KRAKEN PIN IS A GIT SHA AND NOT THE 7.1 WHEEL

The 7.1 wheel (2026-08-04) mis-places tensors during PP-OCRv6 inference: `_lengths_and_mask`
builds the attention mask from `torch.arange(..., device=feat.device)` against sequence lengths
that are still on the host, so a GPU run dies rather than falling back. Upstream fixed it in
PR #799 (`0a3218f0`, merged 2026-08-15) and there has been no 7.1.1 since, so the pin is the
main commit that carries the fix. `SERVING["kraken_release"]` records this rather than letting
the row claim a plain "kraken 7.1" a reader could not reproduce.

The gap between the 7.1 tag (`eff0571`) and the pinned commit was read rather than assumed: it
is three files — that crash fix, one docs example, and an `htrmopo>=0.5` -> `>=0.6` requirement
bump. Nothing in it changes what the models compute. So this row is 7.1's numerics with the fix
that makes GPU inference possible, not a moving target checked out of main.

THE CHECKPOINTS ARE ON ZENODO, NOT THE HUB

There is no Hub repo to pin a revision against, so the pin is the DOI plus a SHA-256 of the
weights, verified at load time. That is a stronger identity claim than a Hub revision, not a
weaker one — it names the bytes. Zenodo's own record IDs are immutable per version.

Card-vs-artifact note, recorded because the board's habit is to measure rather than transcribe:
the Zenodo card says the architecture raises "line height to 128px", while the checkpoint's own
`kraken_meta` declares `height: 96`. kraken reads the artifact, so 96 is what runs.
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

START = time.time()

# Per-value provenance, in the shape `scripts/emit_run_provenance.py` reads by AST:
#
# - model / recognition_*: the published checkpoint. `recognition_sha256` is verified against the
#   downloaded bytes before the model is loaded, so a silently re-uploaded file aborts the run.
# - segmentation_model: kraken's bundled default, loaded by passing NO path. A house choice only
#   in the sense that we declined to substitute something else; see the module docstring.
# - kraken: the pin. See "WHY THE KRAKEN PIN IS A GIT SHA" above.
# - batch_size 32 / precision 32-true: house choices. The PP-OCRv6 datamodule pads every batch to
#   the widest line in it, so batching is only worth what the width spread allows; a page's own
#   lines are close in width, which is why batching is PER PAGE here rather than across pages.
#   Precision stays fp32 because kraken itself warns that fixed 16-bit is "likely to cause
#   unstable recognition", and a benchmark row is the wrong place to take that trade.
# - reading_order / padding / bidi_reordering / text_direction: kraken defaults, stated so the
#   record shows they were defaults rather than unexamined.
# - line_join: the CLI's own text serialisation is `'\n'.join(record.prediction)` (kraken.py),
#   so the page text this driver caches is byte-identical to `kraken ... segment -bl ocr` output.
SERVING = {
    "model": "kraken/PP-OCRv6-medium",
    "recognition_doi": "10.5281/zenodo.21788410",
    "recognition_weights": "https://zenodo.org/records/21788410/files/medium.safetensors",
    "recognition_sha256": "15313b51ace64cbfa81f8f6ef25ad64f04e5a6fb7f7823e67b107527bc081ac9",
    "recognition_variant": "medium",
    "recognition_params": 15_920_372,
    "segmentation_model": "kraken bundled blla.mlmodel (SegmentationTaskModel.load_model(), no path)",
    "kraken": "git+https://github.com/mittagessen/kraken@3e6893a0d5eb06273494bfcdb9d3cac6159e0611",
    "kraken_release": "7.1 plus upstream commits to 2026-08-21, incl. PR #799 (GPU tensor placement)",
    "batch_size": 32,
    "precision": "32-true",
    "padding": 16,
    "num_line_workers": 0,
    "text_direction": "horizontal-lr",
    "reading_order": "polygonal_reading_order (kraken default)",
    "bidi_reordering": True,
    "line_join": "\n",
}

# The other two published sizes, so a size sweep is a re-run rather than a rewrite. Each hash was
# taken from the file Zenodo served on 2026-09-03; params are summed from the safetensors header,
# the same way the board measures every other row.
VARIANTS = {
    "tiny": {
        "doi": "10.5281/zenodo.21788403",
        "url": "https://zenodo.org/records/21788403/files/tiny.safetensors",
        "sha256": "972700b1c72de14bdde028f9b06d77f13c88ec60da5e942bb9e57aca421fa361",
        "params": 691_729,
    },
    "small": {
        "doi": "10.5281/zenodo.21788405",
        "url": "https://zenodo.org/records/21788405/files/small.safetensors",
        "sha256": "3108259779f5abfed3908a1c9eafba64d17e4a2a169368c539ecfeff04c018f3",
        "params": 3_242_809,
    },
    "medium": {
        "doi": SERVING["recognition_doi"],
        "url": SERVING["recognition_weights"],
        "sha256": SERVING["recognition_sha256"],
        "params": SERVING["recognition_params"],
    },
}

FLUSH_EVERY = 100  # pages per part file — see write_part()


def log(msg):
    print(f"[{time.time() - START:8.1f}s] {msg}", flush=True)


def to_pil(value):
    """One dataset image cell -> a PIL image, whichever shape `datasets` hands over."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value)))
    raise ValueError(f"unsupported image value: {type(value)}")


def fetch_weights(variant: str, dest: Path) -> Path:
    """Download the Zenodo checkpoint and verify its SHA-256 before anything loads it.

    A mismatch aborts. The board's identity claim for this row IS the hash — there is no Hub
    revision to fall back on — so a file that does not match is not a warning, it is a different
    model, and scoring it would attribute someone else's bytes to this row.
    """
    spec = VARIANTS[variant]
    path = dest / f"ppocrv6_{variant}.safetensors"
    if not path.exists():
        log(f"downloading {variant} weights from {spec['url']}")
        dest.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{spec['url']}?download=1", path)

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != spec["sha256"]:
        sys.exit(
            f"checkpoint hash mismatch for {variant}: expected {spec['sha256']}, got {actual}. "
            "The published bytes changed, or the download is truncated — refusing to run, "
            "because this row's model identity is the hash."
        )
    log(f"{variant} weights verified: sha256 {actual[:16]}… ({path.stat().st_size} bytes)")
    return path


def load_stack(weights: Path, accelerator: str, batch_size: int):
    """Load the segmenter and recogniser plus their inference configs.

    The segmenter is loaded with NO path on purpose: that is the code path a user hits, and it
    resolves to `blla.mlmodel` inside the installed kraken, which the git pin fixes.
    """
    from kraken.configs import RecognitionInferenceConfig, SegmentationInferenceConfig
    from kraken.tasks import RecognitionTaskModel, SegmentationTaskModel

    seg_model = SegmentationTaskModel.load_model()
    rec_model = RecognitionTaskModel.load_model(str(weights))

    seg_config = SegmentationInferenceConfig(
        text_direction=SERVING["text_direction"],
        accelerator=accelerator,
        precision=SERVING["precision"],
    )
    rec_config = RecognitionInferenceConfig(
        batch_size=batch_size,
        num_line_workers=SERVING["num_line_workers"],
        padding=SERVING["padding"],
        bidi_reordering=SERVING["bidi_reordering"],
        accelerator=accelerator,
        precision=SERVING["precision"],
    )
    log(f"loaded segmenter + recogniser on accelerator={accelerator}, batch_size={batch_size}")
    return seg_model, rec_model, seg_config, rec_config


def transcribe(image, seg_model, rec_model, seg_config, rec_config):
    """One page image -> (text, diagnostics).

    Both stages are timed and both line counts are kept. `n_lines` is what the segmenter
    RETURNED, `n_records` is what the recogniser produced from it.

    A gap between the two does NOT report polygonisation drops — see the module docstring. Those
    lines are discarded before `Segmentation.lines` exists, so they are already missing from
    `n_lines`, and the two counts agree while the page is short a line. The pair still earns its
    place: it says how much of a page reached the recogniser at all, which separates "the
    segmenter found nothing here" from "the recogniser read it badly" when a page scores poorly.
    """
    t0 = time.time()
    segmentation = seg_model.predict(im=image, config=seg_config)
    seg_secs = time.time() - t0

    n_lines = len(segmentation.lines or [])
    if n_lines == 0:
        # Not an error. A blank or plate-only page legitimately has no text lines, and an empty
        # transcription is the correct answer there — see the hallucination note in the docstring.
        return "", {"n_lines": 0, "n_records": 0, "seg_secs": seg_secs, "rec_secs": 0.0}

    t1 = time.time()
    records = list(rec_model.predict(image, segmentation, rec_config))
    rec_secs = time.time() - t1

    text = SERVING["line_join"].join(record.prediction for record in records)
    return text, {
        "n_lines": n_lines,
        "n_records": len(records),
        "seg_secs": seg_secs,
        "rec_secs": rec_secs,
    }


def part_paths(output: str):
    """Existing part files under an output prefix — local dir or hf:// URI."""
    if output.startswith(("hf://", "s3://", "gs://")):
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        return sorted(fs.glob(f"{output.rstrip('/')}/part-*.parquet")), fs
    directory = Path(output)
    return (sorted(str(p) for p in directory.glob("part-*.parquet")) if directory.is_dir() else []), None


def done_ids(output: str) -> set:
    """Ids already written under the output prefix, so a re-run resumes instead of repeating.

    The board's operational note for the 2026-08 run says the same thing about saturate: a job
    that dies at hour five is restarted, not lost. This is the same property for a driver that
    holds its model in-process and so cannot use saturate's pump.
    """
    import pandas as pd

    paths, fs = part_paths(output)
    if not paths:
        return set()
    ids = set()
    for path in paths:
        if fs is not None:
            with fs.open(path, "rb") as fh:
                frame = pd.read_parquet(fh, columns=["id"])
        else:
            frame = pd.read_parquet(path, columns=["id"])
        ids.update(frame["id"].tolist())
    log(f"resume: {len(ids)} ids already present in {len(paths)} part file(s)")
    return ids


def write_part(rows, output: str, rank: int, index: int) -> str:
    """Flush a batch of rows as one part file.

    Flushing periodically rather than once at the end is the difference between losing an hour
    and losing a run: the classical baseline before this one wrote a single parquet after ninety
    minutes, and a timeout at minute eighty-nine cost all of it.

    The rank is in the FILENAME, not just the directory, because concurrent shards write to one
    prefix. Numbering parts from the count of existing files would have two shards both claim
    `part-00000` and one silently overwrite the other's pages — which scoring would then report
    as a missing page rather than as the clobber it was.
    """
    import pandas as pd

    path = f"{output.rstrip('/')}/part-r{rank:02d}-{index:05d}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    log(f"wrote {len(rows)} rows -> {path}")
    return path


def next_part_index(existing_parts, rank: int) -> int:
    """One past the highest part index this rank has already written.

    Never a count: a hole in the sequence (a part deleted, or a partial upload cleaned up by
    hand) would make a count point at a filename that already exists, and writing it would
    destroy pages that `done_ids()` has already marked done — silent loss rather than a loud
    failure. Max-plus-one only ever moves forward, so a resume can add parts but never replace
    one. Filenames that do not parse are ignored rather than guessed at.
    """
    pattern = re.compile(rf"part-r{rank:02d}-(\d+)\.parquet$")
    indices = [int(m.group(1)) for p in existing_parts if (m := pattern.search(str(p)))]
    return max(indices) + 1 if indices else 0


def parse_shard(spec: str) -> tuple[int, int]:
    """`RANK/WORLD` -> (rank, world), matching drivers/falcon-ocr-port.py."""
    rank, _, world = spec.partition("/")
    rank, world = int(rank), int(world or 1)
    if world < 1 or not 0 <= rank < world:
        raise argparse.ArgumentTypeError(
            f"--shard must be rank/world with 0 <= rank < world, got {spec!r}")
    return rank, world


def main():
    ap = argparse.ArgumentParser(description="kraken PP-OCRv6 — inference only")
    ap.add_argument("--input-dataset", required=True)
    # REQUIRED, no default — a forgotten --output must not resume into another run's prefix.
    ap.add_argument("--output", required=True,
                    help="output prefix, e.g. hf://buckets/<owner>/<bucket>/<run>/<model>/")
    ap.add_argument("--revision", default=None, help="pin the benchmark revision")
    ap.add_argument("--split", default="train")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--id-column", default="PageID")
    ap.add_argument("--variant", default="medium", choices=sorted(VARIANTS),
                    help="published checkpoint size (default: medium, the board row)")
    ap.add_argument("--batch-size", type=int, default=SERVING["batch_size"],
                    help="lines per recognition batch, within one page")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test the path against N pages. NOT a publishable run — the page "
                         "set will not match the benchmark fingerprint.")
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="RANK/WORLD",
                    help="strided fan-out across jobs, e.g. 2/4 (default: 0/1). Every shard of a "
                         "world writes to ONE output prefix and together they cover the whole "
                         "page set, so a sharded run is publishable; a SINGLE shard is not, for "
                         "the same page-set reason as --limit. Doubles as the canary sampler: "
                         "0/36 is a 61-page probe spread across all six volumes.")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run without a GPU. Off by default: CPU recognition measured ~2.8 s per "
                         "line against ~0.03 s on GPU, so a silent CPU run does not fail, it just "
                         "never finishes.")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset

    if args.variant != SERVING["recognition_variant"]:
        # The SERVING dict is the provenance record and it names ONE checkpoint. Running a
        # different size through it would file the wrong identity, so say so loudly.
        log(f"NOTE: running variant {args.variant!r}; SERVING names the "
            f"{SERVING['recognition_variant']!r} checkpoint, so this run's provenance must "
            f"record model kraken/PP-OCRv6-{args.variant}")

    if torch.cuda.is_available():
        accelerator = "cuda"
        log(f"CUDA: {torch.cuda.get_device_name(0)}")
    elif args.allow_cpu:
        accelerator = "cpu"
        log("WARNING: no CUDA device; running on CPU because --allow-cpu was passed")
    else:
        sys.exit("no CUDA device available. Recognition on CPU is roughly 90x slower per line, "
                 "which turns this run into days rather than hours — pass --allow-cpu if that is "
                 "genuinely what you want.")

    weights = fetch_weights(args.variant, Path(os.environ.get("TMPDIR", "/tmp")) / "kraken-ppocrv6")
    seg_model, rec_model, seg_config, rec_config = load_stack(weights, accelerator, args.batch_size)

    rank, world = args.shard
    dataset = load_dataset(args.input_dataset, revision=args.revision, split=args.split)
    if world > 1:
        # Strided, not contiguous: the benchmark is ordered by volume, so a contiguous slice
        # would give one shard the Latin volume and another the French, and a shard that died
        # would take a whole book's worth of pages with it.
        dataset = dataset.select(range(rank, len(dataset), world))
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    already = done_ids(args.output)
    # Resume numbering from THIS rank's own parts, by the HIGHEST index present rather than by
    # how many there are. Counting is wrong the moment the sequence has a hole: with
    # part-r00-00000 and part-r00-00002 on disk, a count says "2" and the next flush overwrites
    # part-r00-00002 — whose ids done_ids() has already loaded, so those pages are skipped as
    # done and vanish from the prefix. Counting every part in the prefix regardless of rank
    # would be worse still, colliding with a sibling shard's filenames.
    existing_parts, _ = part_paths(args.output)
    part_index = next_part_index(existing_parts, rank)

    rows, written, errors, empties = [], 0, 0, 0
    total = len(dataset)
    page_clock = time.time()
    for i, row in enumerate(dataset):
        page_id = row[args.id_column]
        if page_id in already:
            continue
        try:
            image = to_pil(row[args.image_column]).convert("RGB")
            text, diagnostics = transcribe(image, seg_model, rec_model, seg_config, rec_config)
            record = {"id": page_id, "raw": text, "error": None, **diagnostics}
            if not text:
                empties += 1
        except Exception as exc:  # a durable error row, never a sentinel string in the text column
            record = {"id": page_id, "raw": None, "error": f"{type(exc).__name__}: {exc}",
                      "n_lines": None, "n_records": None, "seg_secs": None, "rec_secs": None}
            errors += 1
        record["model"] = f"kraken/PP-OCRv6-{args.variant}"
        record["variant"] = args.variant
        rows.append(record)

        if len(rows) >= FLUSH_EVERY:
            write_part(rows, args.output, rank, part_index)
            part_index += 1
            written += len(rows)
            rows = []
        if (i + 1) % 50 == 0:
            done = written + len(rows)
            rate = (time.time() - page_clock) / max(done, 1)
            log(f"  {i + 1}/{total} pages · {rate:.2f}s/page · {errors} errors · {empties} empty")

    if rows:
        write_part(rows, args.output, rank, part_index)
        written += len(rows)

    elapsed = time.time() - START
    summary = {
        "pages_written": written,
        "pages_skipped_resume": len(already),
        "errors": errors,
        "empty_transcriptions": empties,
        "elapsed_s": round(elapsed, 1),
        "s_per_page": round((time.time() - page_clock) / max(written, 1), 3),
        "variant": args.variant,
        "accelerator": accelerator,
        "shard": f"{rank}/{world}",
    }
    print("PORT kraken-ppocrv6 " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
