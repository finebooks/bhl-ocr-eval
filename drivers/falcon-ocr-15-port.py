# /// script
# requires-python = ">=3.11"
# dependencies = ["saturate[hf]==0.2.0", "pillow"]
# ///
"""tiiuae/Falcon-OCR v1.5 — the vendor pipeline endpoint, end-to-end route (board row).

Run (inside ghcr.io/tiiuae/falcon-ocr, which ships both services; this driver boots them):
    hf jobs run --image ghcr.io/tiiuae/falcon-ocr@sha256:d0b4120b... --flavor a10g-small \
      --secrets HF_TOKEN -v hf://buckets/davanstrien/jobs-artifacts:/artifacts:ro -- \
      /bin/bash -c "pip install -q --target /tmp/drvlib 'saturate[hf]==0.2.0' pillow && \
        PYTHONPATH=/tmp/drvlib python3 /artifacts/falcon-ocr-15/falcon-ocr-15-port.py \
        --input-dataset <benchmark> --revision <sha> --id-column PageID --shard 0/2 \
        --no-stage --skip-layout --output hf://buckets/..."

What the row is:

- WEIGHTS. v1.5 replaced the weights in the same Hub repo (commit d259a7fb74, 2026-09-11). Vendor
  images from digest c61192cd on bake v1.5; `--no-stage` keeps the image exactly as shipped and
  only checks that the baked model.safetensors has the v1.5 SHA-256, so the row cannot silently
  score v1. Without `--no-stage` the pinned Hub snapshot is staged over the baked model dir first
  (how the first v1.5 smoke runs worked, on the v1 image); that also replaces the image's
  config.json, so the board row does not use it.
- ROUTE. `--skip-layout` is the pipeline's direct-VLM route, one request per page, which the
  v1.5 card recommends. The v1 row (drivers/falcon-ocr-port.py) scores the layout route.
- SERVING. The vendor image's own entrypoint (`/app/entrypoint_single.sh`, single GPU: vLLM on
  :8000 + pipeline on :5002), not saturate's Engine. The driver boots it as a subprocess and
  gates on both health endpoints before pumping.
- saturate is pinned to 0.2.0: in that version a raise inside `parse` becomes a durable error
  row that `--retry-errors` can supersede. 0.3 changed that and the shard assignment.

The pipeline response carries no finish_reason. `total_output_tokens` is recorded per page, and a
page at the route's cap is sealed as a truncation downstream (see E2E_MAX_TOKENS).
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
    "model_revision": "d259a7fb74",  # Hub commit "Release Falcon OCR v1.5", 2026-09-11
    # LFS sha256 of model.safetensors at that commit (from the Hub file metadata); v1 = 6e7f73a5...
    "weights_sha256": "3df91e403dc48794bf1c48511e75c3508b1cc52df599dcc15f1080d46101ab16",
    "model_dir": "/models/Falcon-OCR",
    "image": "ghcr.io/tiiuae/falcon-ocr",
    "entrypoint": "/app/entrypoint_single.sh",
    "pipeline_port": 5002,
    "vllm_port": 8000,
    "max_pixels_longest_edge": 2048,
    "skip_layout": False,
}

# The pipeline's direct-VLM route (skip_layout) sends one request per page and caps it at this
# many output tokens. The response carries no finish_reason, but it does report the token count,
# so a page at the cap is a truncation: pass this to `consolidate_run.py --token-cap`. Observed on
# image d0b4120b (19 of 2,165 pages at exactly 8192); it is the image's value, not ours.
E2E_MAX_TOKENS = 8192

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


def sha256_of(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_weights(no_stage: bool = False) -> str:
    """Overwrite the image's baked checkpoint with the pinned Hub revision; fail loud on mismatch."""
    from huggingface_hub import HfApi, snapshot_download

    model_dir = SERVING["model_dir"]
    entrypoint = open(SERVING["entrypoint"]).read()
    print("=== /app/entrypoint_single.sh ===\n" + entrypoint + "\n=== end ===", flush=True)
    if model_dir not in entrypoint:
        raise RuntimeError(f"entrypoint does not reference {model_dir}; refusing to guess where weights load from")
    weights = os.path.join(model_dir, "model.safetensors")
    baked = sha256_of(weights)
    print(f"baked weights sha256 (v1 6e7f73a5... / v1.5 3df91e40...): {baked}", flush=True)
    if no_stage:
        if baked != SERVING["weights_sha256"]:
            raise RuntimeError(f"--no-stage but baked weights {baked[:12]} != v1.5 {SERVING['weights_sha256'][:12]}")
        print("--no-stage: keeping the image's own config/code files", flush=True)
        return "baked:" + baked[:12]
    resolved = HfApi().model_info(SERVING["model"], revision=SERVING["model_revision"]).sha
    snapshot_download(SERVING["model"], revision=resolved, local_dir=model_dir,
                      allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt"])
    got = sha256_of(weights)
    if got != SERVING["weights_sha256"]:
        raise RuntimeError(f"staged weights sha256 {got} != expected {SERVING['weights_sha256']}")
    print(f"staged {SERVING['model']}@{resolved} into {model_dir}; sha256 verified {got[:12]}", flush=True)
    return resolved


def boot_services(no_stage: bool = False) -> subprocess.Popen:
    """Start the vendor image's own single-GPU entrypoint; gate on both health endpoints."""
    resolved = stage_weights(no_stage)
    os.environ["FALCON_MODEL_REVISION"] = resolved
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


def pipeline_error(json_result, markdown) -> str | None:
    """The error the pipeline swallowed, or None.

    When the pipeline's own request to vLLM fails it still answers HTTP 200, with empty markdown
    and the failure inside json_result (`[{"content": "", "error": "..."}]`). Left alone, that
    row scores as a model that read nothing. Returning the message makes it an error row, which
    a retry can supersede.

    Only a page with NO text counts. On the layout route one region can fail while the rest of
    the page is assembled; that partial page is what the vendor pipeline delivers, so it is
    scored as delivered, the same way the v1 row was.
    """
    if (markdown or "").strip() or not isinstance(json_result, list):
        return None
    errors = [r["error"] for r in json_result if isinstance(r, dict) and r.get("error")]
    return errors[0] if errors else None


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
    ap.add_argument("--max-window", type=int, default=8,
                    help="saturate adaptive window cap (board default 8); raise to let vLLM batch")
    ap.add_argument("--no-stage", action="store_true",
                    help="do not overwrite the baked checkpoint; only verify its sha256 == v1.5 (use on images that already bake v1.5, keeps the image's own config.json)")
    ap.add_argument("--ids", default=None,
                    help="comma-separated page ids to run (retry of pipeline-internal errors)")
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
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        rows = ((rid, r) for rid, r in rows if str(rid) in wanted)  # saturate yields (id, row)

    def to_request(row):
        b64 = encode_jpeg(row[args.image_column], SERVING["max_pixels_longest_edge"])
        return {"images": [f"data:image/jpeg;base64,{b64}"], "skip_layout": skip_layout}

    def parse(row, body):
        # /falconocr/parse: markdown_result is the assembled page; json_result carries the
        # layout regions (label/bbox/score/content) — kept verbatim so the raw cache can be
        # re-assembled under a different reading-order or region policy without GPUs.
        failure = pipeline_error(body.get("json_result"), body.get("markdown_result"))
        if failure:
            raise RuntimeError(f"pipeline swallowed a vLLM error: {failure}")
        return {"raw": body.get("markdown_result"),
                "regions_json": json.dumps(body.get("json_result")),
                "model": SERVING["model"],
                "model_revision": os.environ.get("FALCON_MODEL_REVISION"),
                "skip_layout": skip_layout,
                "completion_tokens": body.get("total_output_tokens"),
                "processing_ms": body.get("processing_time_ms")}

    proc = boot_services(args.no_stage)
    try:
        stats = pump(rows, to_request, parse, f"http://127.0.0.1:{SERVING['pipeline_port']}",
                     route="/falconocr/parse", output=args.output,
                     window=Auto(initial=4, target_waiting=max(4, args.max_window // 2), max_limit=args.max_window, step=2),
                     shard=(rank, world),
                     retry_errors=args.retry_errors)
    finally:
        proc.terminate()
    print("PORT falcon-ocr-1.5 " + stats.to_json(), flush=True)
    # Interpreter finalization segfaults in this image (pyarrow/torch atexit clash,
    # observed exit 139 AFTER a clean pump + stats line), which would mark every
    # successful job ERROR. Parts + stats are already flushed; skip finalizers.
    os._exit(0)


if __name__ == "__main__":
    main()
