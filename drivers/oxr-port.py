# /// script
# requires-python = ">=3.11"
# dependencies = ["saturate[hf]>=0.1.1", "pillow", "huggingface_hub"]
# ///
"""SoMarkAI/OXR-1.0 — saturate port driving the vendor all-in-one parse service (port-matrix row).

Run (inside the image built from the vendor's docker/Dockerfile.allinone, see SERVING["image"]):
    hf jobs run --flavor a100-large --secrets HF_TOKEN <SERVING["image"]> -- bash -c \
      "hf buckets cp hf://buckets/<b>/oxr-port.py /tmp/oxr-port.py && \
       uv run /tmp/oxr-port.py --model-revision <sha> \
         --input-dataset <benchmark> --id-column PageID --output <prefix> --limit 20"

Like Falcon-OCR, OXR is a pipeline, not a single VLM: a D-FINE layout model finds regions, then
the OXR VLM reads each region and orders them, and the vendor's `oxr` API assembles the page. The
vendor's whole-page product is the `/v1/parse/sync` endpoint, so that is what this benchmark
scores. The all-in-one entrypoint boots all three services (vLLM :8080, layout :8081, API :8000);
the driver starts it as a subprocess and gates on the API's /docs page, which the entrypoint only
serves after both model services report ready.

Two things differ from the other ports:

- The weights are NOT mounted with `-v hf://SoMarkAI/OXR-1.0:...`. On 2026-10-26 the Jobs model
  mount served short prefixes of the small files under plugins/vllm/ (pyproject.toml 276 of 435
  bytes) with a matching stat size, and the entrypoint's plugin install failed on the truncated
  TOML. The driver downloads the pinned revision with snapshot_download before boot instead.
- The image is not the board-wide vllm/vllm-openai:v0.26.0. The vendor plugin requires vLLM
  0.25.1 + Transformers 5.13.1 and the entrypoint refuses any other runtime, so the row runs on
  the vendor's own pinned base. Recorded exception, as with LightOn-3 on v0.30.0.

The response carries no finish_reason or token counts; the pipeline's per-region generation caps
are internal (oxr_model.max_tokens 8192 in the vendor config). `raw` is the API's markdown output
verbatim; the per-block JSON (type/bbox/content/format) is kept in `blocks_json` so the page can
be re-assembled under a different policy without GPUs.
"""
import argparse
import io
import json
import os
import subprocess
import time
import urllib.request

SERVING = {
    # Per-value provenance:
    # - model: the only OXR checkpoint. Weights are pinned by --model-revision and downloaded
    #   to MODEL_PATH before boot (see the module docstring for why not a mount).
    # - image: cr.hf.co/davanstrien/oxr, built with kaniko from SoMarkAI/OXR @ source_commit,
    #   docker/Dockerfile.allinone, unmodified. That Dockerfile's base is the vendor's pinned
    #   vllm/vllm-openai:v0.25.1@sha256:e4f88a83...
    # - entrypoint: the image's own ENTRYPOINT. HF Jobs runs the job command instead of the
    #   ENTRYPOINT, so the driver calls it explicitly.
    # - output_formats markdown + json: the vendor README's example request.
    # - feature_config / element_formats: not sent, so vendor defaults apply. That means
    #   keep_header_footer=False (Page-header/Page-footer blocks dropped, the same policy as the
    #   chandra-ocr-2 row) and the default per-element formats.
    # - image encoding: house choice. The bench stores webp; every page is sent as lossless PNG
    #   at its stored resolution. The pipeline's layout model resizes internally.
    "model": "SoMarkAI/OXR-1.0",
    "image": "cr.hf.co/davanstrien/oxr@sha256:aa6a4fb538bd58e19d0d032d6ed7e6ddb53bc524b64361ecb554b72ffdab233d",
    "source_repo": "https://github.com/SoMarkAI/OXR",
    "source_commit": "c1a1dcb312ee9f691404f127785510453c0d9350",
    "entrypoint": ["/opt/oxr-venv/bin/python", "/app/docker/allinone_entrypoint.py"],
    "model_path": "/models/oxr-model",
    "api_port": 8000,
    "output_formats": ["markdown", "json"],
}

BOOT_TIMEOUT_S = 1800  # the entrypoint's own startup timeout is 900 s; fail loud after that


def to_png_bytes(value) -> bytes:
    """One dataset image cell -> lossless PNG bytes (decoded PIL, {"bytes"} dict, or raw bytes)."""
    from PIL import Image

    if isinstance(value, Image.Image):
        img = value
    elif isinstance(value, dict) and value.get("bytes"):
        img = Image.open(io.BytesIO(value["bytes"]))
    elif isinstance(value, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(value)))
    else:
        raise ValueError(f"unsupported image value: {type(value)}")
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def wait_healthy(url: str, deadline: float, proc: subprocess.Popen) -> None:
    last = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"OXR entrypoint exited with code {proc.returncode} before {url} was up")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except Exception as e:  # noqa: BLE001 - boot polling, every failure kind is "not yet"
            last = e
        time.sleep(5)
    raise RuntimeError(f"service at {url} not healthy after {BOOT_TIMEOUT_S}s: {last}")


def download_weights(revision: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(SERVING["model"], revision=revision, local_dir=SERVING["model_path"])


def boot_services() -> subprocess.Popen:
    """Start the vendor all-in-one entrypoint; gate on the API, which comes up last."""
    env = dict(os.environ)
    # The driver runs in uv's environment. The services must not inherit it: each vendor
    # service has its own venv with its own pins.
    for key in ("PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    env["MODEL_PATH"] = SERVING["model_path"]
    proc = subprocess.Popen(SERVING["entrypoint"], env=env)
    wait_healthy(f"http://127.0.0.1:{SERVING['api_port']}/docs", time.time() + BOOT_TIMEOUT_S, proc)
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
    # REQUIRED here, unlike the vLLM ports: the weights are downloaded by this script, so an
    # unpinned run would silently score whatever main is on the day.
    ap.add_argument("--model-revision", required=True,
                    help="OXR-1.0 commit sha to download and serve")
    ap.add_argument("--id-column", default=None,
                    help="Column to use as row id (default: split-index ids)")
    # REQUIRED, no default — a forgotten --output must not resume into another run's output.
    ap.add_argument("--output", required=True,
                    help="output prefix, e.g. hf://buckets/<owner>/<bucket>/<run>/<model>/")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="RANK/WORLD",
                    help="Strided fan-out across jobs, e.g. 2/8 (default: 0/1)")
    ap.add_argument("--retry-errors", action="store_true")
    args = ap.parse_args()

    from saturate import Auto, dataset_rows, make_multipart_request, pump, shard_select

    rank, world = args.shard
    SERVING["model_revision"] = args.model_revision

    rows = dataset_rows(
        args.input_dataset, config=args.config, split=args.split,
        columns=[args.image_column], ids=args.id_column or "index",
        revision=args.revision, limit=args.limit,
    )
    if world > 1:
        rows = shard_select(rows, rank=rank, world=world)

    def to_request(row):
        png = to_png_bytes(row[args.image_column])
        return make_multipart_request(
            "/v1/parse/sync",
            data={"api_key": "dummy", "output_formats": SERVING["output_formats"]},
            files={"file": ("page.png", png, "image/png")},
        )

    def parse(row, body):
        # The API answers HTTP 200 with a non-zero `code` when the pipeline fails; that must
        # become an error row, not a blank page scored as unread.
        if body.get("code") != 0 or (body.get("data") or {}).get("error"):
            raise RuntimeError(f"OXR parse failed: code={body.get('code')} "
                               f"message={body.get('message')!r} error={(body.get('data') or {}).get('error')!r}")
        data = body["data"]
        outputs = data["result"]["outputs"]
        return {"raw": outputs["markdown"],
                "blocks_json": json.dumps(outputs.get("json"), ensure_ascii=False),
                "model": SERVING["model"],
                "warnings": json.dumps(body.get("warnings") or []),
                "processing_ms": (data.get("metadata") or {}).get("processing_time_ms")}

    download_weights(args.model_revision)
    proc = boot_services()
    try:
        stats = pump(rows, to_request, parse, f"http://127.0.0.1:{SERVING['api_port']}",
                     output=args.output,
                     window=Auto(initial=4, target_waiting=4, max_limit=16, step=2),
                     shard=(rank, world),
                     retry_errors=args.retry_errors,
                     prepare_workers=4)
    finally:
        proc.terminate()
    print("PORT oxr " + stats.to_json(), flush=True)
    print("SERVING " + json.dumps(SERVING), flush=True)


if __name__ == "__main__":
    main()
