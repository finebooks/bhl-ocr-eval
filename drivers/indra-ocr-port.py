# /// script
# requires-python = ">=3.10"
# dependencies = ["indra-ocr==1.1.2", "datasets", "huggingface-hub", "pandas", "pyarrow", "pillow"]
# ///
"""datalab-to/indra-ocr — CPU-only OCR pipeline, as a self-contained inference driver.

Run:
    hf jobs uv run --detach --flavor cpu-upgrade -s HF_TOKEN --timeout 2h \\
        --image ghcr.io/astral-sh/uv:python3.12-bookworm \\
        drivers/indra-ocr-port.py -- \\
        --input-dataset <benchmark> --revision <sha> --id-column PageID \\
        --output hf://buckets/<owner>/<bucket>/<prefix>/

Indra OCR is a pipeline of small ONNX models (detector, orientation, script classifier, four
line recognisers, layout and reading order; ~7M parameters, ~1.25M in the main Latin
recogniser) run by ONNX Runtime on CPU — no vLLM, no GPU, so the board-wide vLLM pin does not
apply. Like the tesseract and kraken rows it runs in-process.

Each page goes through the package's OWN image path: the image bytes are written to a temp
file and read with `Converter.convert(path)` — the call the `indra-ocr` CLI makes — so decoding
to grayscale, orientation and deskew are the vendor's. Settings are the package defaults
(`indra_ocr.load()`: mode "auto", which for an image is OCR; furniture=False, i.e. page headers
and footers dropped from the markdown; math=False), with one single-threaded worker process
per core, as `convert_many` and the CLI do.

`raw` is `page.markdown` verbatim. A page the package reports as failed (`source == "error"`)
becomes an error row with its note, never an empty transcription. An empty markdown on a blank
leaf is a correct answer, so the normalizer registers no `require_non_empty`, as for kraken.

Model pin: `indra-ocr` 1.1.2 pins its model set to the `v1.1.0` tag of datalab-to/indra-ocr
(`indra_ocr/models.py: HF_REVISION`). The driver downloads that tag explicitly and records the
commit it resolves to.

Licence: code Apache-2.0; weights a modified OpenRAIL-M (free for research, personal use and
startups under $20M; not for use competing with Datalab's API). Recorded in RESULTS.md.
"""

import argparse
import io
import os
import tempfile

SERVING = {
    "model": "datalab-to/indra-ocr",
    "package": "indra-ocr==1.1.2",
    # vendor: the model-set tag indra-ocr 1.1.2 is built and tested against.
    "model_tag": "v1.1.0",
    # vendor defaults of indra_ocr.load(); written out so the provenance record names them.
    "mode": "auto",
    "furniture": False,
    "math": False,
    "dpi": 200.0,
    # vendor: one single-threaded worker per core (convert_many / the CLI).
    "threads_per_worker": 1,
}

_CONV = None


def _init(model_path):
    global _CONV
    import indra_ocr

    _CONV = indra_ocr.load(
        path=model_path, threads=SERVING["threads_per_worker"], dpi=SERVING["dpi"],
        mode=SERVING["mode"], furniture=SERVING["furniture"], math=SERVING["math"],
    )


def _ocr_one(item):
    page_id, data, suffix = item
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        f.write(data)
        f.flush()
        pages = list(_CONV.convert(f.name))
    if len(pages) != 1:
        return page_id, None, f"expected 1 page, got {len(pages)}", None
    page = pages[0][1]
    if page.source == "error":
        return page_id, None, f"indra page error: {page.note}", page.source
    return page_id, page.markdown, None, page.source


def image_bytes(value) -> tuple[bytes, str]:
    """One undecoded dataset image cell -> (bytes, file suffix)."""
    if isinstance(value, dict) and value.get("bytes"):
        data = value["bytes"]
    elif isinstance(value, (bytes, bytearray)):
        data = bytes(value)
    else:
        raise ValueError(f"unsupported image value: {type(value)}")
    from PIL import Image

    fmt = (Image.open(io.BytesIO(data)).format or "PNG").lower()
    return data, "." + {"jpeg": "jpg", "jpeg2000": "jp2"}.get(fmt, fmt)


def main():
    ap = argparse.ArgumentParser(description="Indra OCR — inference only")
    ap.add_argument("--input-dataset", required=True)
    ap.add_argument("--output", required=True,
                    help="hf://buckets/<owner>/<bucket>/<prefix>/ or a local .parquet path")
    ap.add_argument("--revision", default=None, help="pin the benchmark revision")
    ap.add_argument("--split", default="train")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--id-column", default="PageID")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes (default: CPU_CORES on Jobs, else os.cpu_count())")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test the path against N pages. NOT a publishable run.")
    args = ap.parse_args()

    from multiprocessing import get_context

    import pandas as pd
    from datasets import Image as ImageFeature
    from datasets import load_dataset
    from huggingface_hub import HfApi

    import indra_ocr

    commit = HfApi().model_info(SERVING["model"], revision=SERVING["model_tag"]).sha
    model_path = str(indra_ocr.model_dir(revision=commit))
    SERVING["model_revision"] = commit
    print(f"indra-ocr {indra_ocr.__version__ if hasattr(indra_ocr, '__version__') else '1.1.2'} | "
          f"models {SERVING['model_tag']} = {commit}", flush=True)

    ds = load_dataset(args.input_dataset, revision=args.revision, split=args.split)
    ds = ds.cast_column(args.image_column, ImageFeature(decode=False))
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    # Jobs sets CPU_CORES to the container's quota; os.cpu_count() reports the host's cores.
    workers = args.workers or int(float(os.environ.get("CPU_CORES") or 0)) or os.cpu_count() or 1

    def items():
        for r in ds:
            data, suffix = image_bytes(r[args.image_column])
            yield r[args.id_column], data, suffix

    rows = []
    with get_context("spawn").Pool(workers, initializer=_init, initargs=(model_path,)) as pool:
        for i, (page_id, raw, err, source) in enumerate(pool.imap(_ocr_one, items(), chunksize=4)):
            rows.append({args.id_column: page_id, "raw": raw, "error": err, "source": source,
                         "model": SERVING["model"], "model_revision": commit})
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(ds)}", flush=True)

    df = pd.DataFrame(rows)
    out = args.output.rstrip("/")
    out = out if out.endswith(".parquet") else f"{out}/indra-ocr.parquet"
    df.to_parquet(out, index=False)
    print(f"{len(df)} pages -> {out} ({int(df['error'].notna().sum())} error rows, "
          f"{workers} workers)", flush=True)


if __name__ == "__main__":
    main()
