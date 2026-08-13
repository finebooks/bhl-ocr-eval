# /// script
# requires-python = ">=3.11"
# dependencies = ["saturate[hf]>=0.1.1", "pillow"]
# ///
"""tiiuae/Falcon-OCR — saturate port driving the vendor pipeline endpoint (port-matrix row).

Run (inside ghcr.io/tiiuae/falcon-ocr, which ships both services; this driver boots them):
    hf jobs run --image ghcr.io/tiiuae/falcon-ocr@sha256:... --flavor a10g-small \
      --secrets HF_TOKEN -v hf://buckets/davanstrien/jobs-artifacts:/artifacts:ro -- \
      bash -lc 'pip install -q "saturate[hf]>=0.1.1" pillow && python3 /artifacts/falcon-ocr-port.py \
        --input-dataset <benchmark> --id-column PageID --limit 20 --output hf://buckets/...'

Two things differ from every other port in this matrix:

- Falcon-OCR is a CROP model by design: the card's flow is layout detection -> crops ->
  per-crop OCR with category prompts. Whole-page OCR is the vendor's own Pipeline service
  (layout + OCR + markdown assembly), so that is what this benchmark scores — the
  deployable full-page product, not the off-label direct-VLM path. `--skip-layout` exists
  to measure that off-label path deliberately.
- The serving stack is the vendor image's own entrypoint (`/app/entrypoint_single.sh`,
  single-GPU: vLLM on :8000 + pipeline on :5002), not saturate's Engine. The driver boots
  it as a subprocess and gates on BOTH health endpoints before pumping.

The pipeline response carries no finish_reason; `total_output_tokens` is recorded per page
and the loop question is answered downstream from output shape, not a cap flag.
"""
import argparse
import base64
import io
import json
import os
import subprocess
import time
import urllib.request

SERVING = {
    # Per-value provenance:
    # - model: the only Falcon-OCR checkpoint; the image bundles/loads it itself — the
    #   served weights are whatever ghcr.io/tiiuae/falcon-ocr@<digest> ships, so the
    #   IMAGE DIGEST is the revision pin for this row (record it in run provenance).
    # - entrypoint_single.sh: the image's own CMD (single-GPU: VLLM_GPU == PIPELINE_GPU).
    # - max_pixels_longest_edge 2048: house choice, payload-shrink only, matching the
    #   other ports; the pipeline's layout detector re-sizes internally as it needs.
    # - skip_layout False: the vendor-intended full-page path (see module docstring).
    "model": "tiiuae/Falcon-OCR",
    "image": "ghcr.io/tiiuae/falcon-ocr",
    "entrypoint": "/app/entrypoint_single.sh",
    "pipeline_port": 5002,
    "vllm_port": 8000,
    "max_pixels_longest_edge": 2048,
    "skip_layout": False,
}

BOOT_TIMEOUT_S = 1800  # model load + layout model load; generous, fails loud


def to_pil(value):
    """One dataset image cell -> a PIL image (decoded PIL, {"bytes"} dict, or raw bytes)."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value)))
    raise ValueError(f"unsupported image value: {type(value)}")


def encode_jpeg(value, longest_edge: int) -> str:
    """RGB-convert, downscale to longest_edge, return base64 JPEG q95."""
    from PIL import Image

    img = to_pil(value).convert("RGB")
    w, h = img.size
    if max(w, h) > longest_edge:
        scale = longest_edge / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def wait_healthy(url: str, deadline: float) -> None:
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception as e:  # noqa: BLE001 - boot polling, every failure kind is "not yet"
            last = e
        time.sleep(5)
    raise RuntimeError(f"service at {url} not healthy after {BOOT_TIMEOUT_S}s: {last}")


def boot_services() -> subprocess.Popen:
    """Start the vendor image's own single-GPU entrypoint; gate on both health endpoints."""
    env = dict(os.environ)
    # The driver's own deps arrive via pip --target + PYTHONPATH (the image's python has
    # no ensurepip, so no venv). The services must NOT inherit that path: the driver's
    # dep tree carries transformers 5.x while the image's vendored vLLM pins <5.
    env.pop("PYTHONPATH", None)
    # entrypoint_single.sh serves both processes from one GPU; make that explicit even
    # where the image defaults agree, so the record shows the intent.
    env.setdefault("EXPOSED_GPU_IDS", "0")
    env.setdefault("VLLM_GPU", "0")
    env.setdefault("PIPELINE_GPU", "0")
    proc = subprocess.Popen(["/bin/bash", SERVING["entrypoint"]], env=env)
    deadline = time.time() + BOOT_TIMEOUT_S
    wait_healthy(f"http://127.0.0.1:{SERVING['vllm_port']}/health", deadline)
    wait_healthy(f"http://127.0.0.1:{SERVING['pipeline_port']}/health", deadline)
    return proc


def parse_shard(spec: str) -> tuple[int, int]:
    rank, _, world = spec.partition("/")
    rank, world = int(rank), int(world or 1)
    if world < 1 or not 0 <= rank < world:
        raise argparse.ArgumentTypeError(f"--shard must be rank/world with 0 <= rank < world, got {spec!r}")
    return rank, world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dataset", required=True,
                    help="Input dataset repo id (rows with an image column)")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--config", default=None, help="Dataset config name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--revision", default=None,
                    help="Pin the input revision (index ids are only stable per revision)")
    ap.add_argument("--id-column", default=None,
                    help="Column to use as row id (default: split-index ids)")
    # REQUIRED, no default — a forgotten --output must not resume into another run's output.
    ap.add_argument("--output", required=True,
                    help="output prefix, e.g. hf://buckets/<owner>/<bucket>/<run>/<model>/")
    ap.add_argument("--skip-layout", action="store_true",
                    help="off-label direct-VLM path (no layout detection); default is the "
                         "vendor pipeline, which is what the board scores")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="RANK/WORLD",
                    help="Strided fan-out across jobs, e.g. 2/8 (default: 0/1)")
    ap.add_argument("--retry-errors", action="store_true")
    args = ap.parse_args()

    from saturate import Auto, dataset_rows, pump, shard_select

    rank, world = args.shard
    skip_layout = args.skip_layout or SERVING["skip_layout"]

    rows = dataset_rows(
        args.input_dataset, config=args.config, split=args.split,
        columns=[args.image_column], ids=args.id_column or "index",
        revision=args.revision, limit=args.limit,
    )
    if world > 1:
        rows = shard_select(rows, rank=rank, world=world)

    def to_request(row):
        b64 = encode_jpeg(row[args.image_column], SERVING["max_pixels_longest_edge"])
        return {"images": [f"data:image/jpeg;base64,{b64}"], "skip_layout": skip_layout}

    def parse(row, body):
        # /falconocr/parse: markdown_result is the assembled page; json_result carries the
        # layout regions (label/bbox/score/content) — kept verbatim so the raw cache can be
        # re-assembled under a different reading-order or region policy without GPUs.
        return {"raw": body.get("markdown_result"),
                "regions_json": json.dumps(body.get("json_result")),
                "model": SERVING["model"],
                "skip_layout": skip_layout,
                "completion_tokens": body.get("total_output_tokens"),
                "processing_ms": body.get("processing_time_ms")}

    proc = boot_services()
    try:
        stats = pump(rows, to_request, parse, f"http://127.0.0.1:{SERVING['pipeline_port']}",
                     route="/falconocr/parse", output=args.output,
                     window=Auto(initial=2, target_waiting=4, max_limit=8, step=1),
                     shard=(rank, world),
                     retry_errors=args.retry_errors)
    finally:
        proc.terminate()
    print("PORT falcon-ocr " + stats.to_json(), flush=True)
    # Interpreter finalization segfaults in this image (pyarrow/torch atexit clash,
    # observed exit 139 AFTER a clean pump + stats line), which would mark every
    # successful job ERROR. Parts + stats are already flushed; skip finalizers.
    os._exit(0)


if __name__ == "__main__":
    main()
