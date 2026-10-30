# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "surya-ocr==0.23.0",
#     "datasets>=3.1.0",
#     "huggingface-hub>=1.5.0,<2",
#     "pandas",
#     "pyarrow",
#     "pillow",
# ]
#
# [tool.uv]
# exclude-newer = "2026-10-30T00:00:00Z"
# ///
"""datalab-to/surya-ocr-2.5 — the surya-ocr package's full-page OCR path, as an inference driver.

Run:
    hf jobs uv run --detach --flavor a100-large -s HF_TOKEN --timeout 3h \\
        --image vllm/vllm-openai@sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089 \\
        drivers/surya-ocr25-port.py -- \\
        --input-dataset <benchmark> --revision <sha> --id-column PageID \\
        --output hf://buckets/<owner>/<bucket>/<prefix>/

Inference only. It writes `part-*.parquet` files of `{id, raw, error, finish_reason, ...}` and
stops, which is the shape `runners/consolidate_run.py` reads. Nothing here cleans text:
`runners/normalize_outputs.py` turns `raw` into the scored column.

WHAT RUNS

Surya 2.5 is one 689M VLM (Qwen3.5 architecture) behind the `surya-ocr` package. The package is
the product: `RecognitionPredictor(manager)(images, full_page=True)` is the call `surya_ocr` (the
CLI) makes. Per page it sends one full-page request, parses the model's block stream into
labelled HTML blocks, drops text blocks over blank regions, and, when the full-page output is
empty on a non-blank page, fails to parse, loops or hits the token cap, re-reads that page in
block mode (a layout request, then one request per block). This driver calls exactly that and
stores the result. It does not reimplement any of it.

Two things differ from a plain `surya_ocr` run, and neither changes what the model sees:

  1. The vLLM server. The package spawns `vllm/vllm-openai:v0.25.1` with `docker run`, which a
     Job cannot do. So the Job runs ON that image, and this driver starts `vllm serve` itself
     with the flags the package's spawn function would pass (surya/inference/backends/vllm.py,
     `spawn_fn`), then points the package at it with `SURYA_INFERENCE_URL`, the package's own
     documented attach path. The flags are written out in SERVING and checked at start-up
     against the values the installed package computes, so they cannot drift silently.
     The package and vLLM stay in separate environments, as the model card requires
     (surya-ocr pins opencv 4.11, vLLM needs >= 4.13): vLLM is the image's, surya-ocr is this
     script's uv environment.
  2. The model revision. The package serves the Hub head of `datalab-to/surya-ocr-2.5`. Here the
     checkpoint is downloaded at a pinned commit and vLLM serves that directory under the
     package's expected model name, so the package attaches unchanged and the row names its
     bytes.

Every other setting is the package default (surya-ocr 0.23.0, surya/settings.py): image size
capped by `scale_to_fit` at 3072x2048 px, 12288 full-page tokens, temperature 0 with the
client's own repeat-retry, progressive-temperature regeneration OFF, guided layout JSON ON. The
board pages are images, not PDFs, so the package's DPI settings never apply.

WHAT IS STORED

`raw` is the package's page result as JSON: the blocks in reading order, each with its label,
the model's raw label, its HTML ("" for figures, which the package does not read) and its error
flag. That is the output a user of the package gets. Next to it, from a wrapper around the
package's `generate`:

  fullpage_raw / fullpage_finish_reason / fullpage_tokens
      the model's verbatim full-page completion, before the package parses it
  fallback          the package re-read the page in block mode
  n_capped_blocks   block-mode requests cut by their token cap
  n_failed_blocks   block-mode requests that failed after the client's retries

`finish_reason` follows the board's truncation rule. The package never accepts a capped
full-page output, but in block mode it keeps a capped block's partial HTML and flags it. A page
whose final result holds a capped block is written with `finish_reason = "length"`, so
consolidate_run seals it as truncated, as for every vLLM row. A failed block request is a run
defect, not a model failure: the page becomes an error row and `--retry-errors` re-reads it.
"""

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

START = time.time()

# Per-value provenance, in the shape `scripts/emit_run_provenance.py` reads by AST:
#
# - model / model_revision: the Hub checkpoint, pinned at the commit current on 2026-10-30
#   (released 2026-10-29). Downloaded at that commit and served from disk; see the docstring.
# - package: the surya-ocr release that shipped with the checkpoint. Its transitive deps are
#   fixed by the `exclude-newer` date in the script header.
# - vllm_image: the package's own VLLM_DOCKER_IMAGE default, pinned by manifest digest.
# - vllm_gpu_type: the package's VLLM_GPU_TYPE knob for the flavor this row runs on
#   (a100-large = one A100 80 GB). The package derives max_num_seqs and max_batched_tokens from
#   it; both are written out below and asserted against the package's own arithmetic.
# - vllm_args: verbatim from the package's spawn_fn, with that GPU type's numbers. No MTP flag:
#   the checkpoint config declares no MTP head (mtp_num_hidden_layers 0), so the package's
#   auto rule leaves speculation off.
# - client_parallel: the package's auto value, min(max_num_seqs, MAX_AUTO_PARALLEL=96). Pages
#   are submitted one per call from this many threads; each call is the package's own path.
# - mode / fullpage_regen / max_tokens_full_page / guided_layout: package defaults, stated.
SERVING = {
    "model": "datalab-to/surya-ocr-2.5",
    "model_revision": "00c843e9cae45611dca0a6dc2c9bbbf2dadaaed3",
    "package": "surya-ocr==0.23.0",
    "vllm_image": "vllm/vllm-openai:v0.25.1@sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089",
    "vllm_gpu_type": "a100-80",
    "vllm_args": [
        "--no-enforce-eager",
        "--max-num-seqs", "104",
        "--dtype", "bfloat16",
        "--max-model-len", "24576",
        "--max-num-batched-tokens", "16384",
        "--gpu-memory-utilization", "0.85",
        "--enable-prefix-caching",
        "--mm-processor-kwargs", "{\"min_pixels\": 3136, \"max_pixels\": 6291456}",
    ],
    "client_parallel": 96,
    "mode": "full_page (RecognitionPredictor(..., full_page=True), as the surya_ocr CLI)",
    "fullpage_regen": False,
    "max_tokens_full_page": 12288,
    "guided_layout": True,
    "image_preprocessing": "convert('RGB'), as surya.input.load.load_image; scale_to_fit in the client",
}

PORT = 18500
FLUSH_EVERY = 100  # pages per part file


def log(msg):
    print(f"[{time.time() - START:8.1f}s] {msg}", flush=True)


# --- vLLM server ---------------------------------------------------------------------------


def image_vllm() -> str | None:
    """The image's own `vllm` binary, looking past this script's venv."""
    own_bin = os.path.realpath(os.path.join(sys.prefix, "bin"))
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                           if os.path.realpath(p) != own_bin)
    return shutil.which("vllm", path=path)


def check_package_flags():
    """Fail if SERVING's server flags differ from what the installed package would pass."""
    from surya.inference.backends.vllm import VllmBackend, _gpu_settings
    from surya.settings import settings

    max_batched_tokens, max_num_seqs = _gpu_settings(SERVING["vllm_gpu_type"])
    expected = [
        "--no-enforce-eager",
        "--max-num-seqs", str(max_num_seqs),
        "--dtype", settings.VLLM_DTYPE,
        "--max-model-len", str(settings.VLLM_MAX_MODEL_LEN),
        "--max-num-batched-tokens", str(max_batched_tokens),
        "--gpu-memory-utilization", str(settings.VLLM_GPU_MEMORY_UTILIZATION),
        "--enable-prefix-caching",
        "--mm-processor-kwargs", json.dumps({"min_pixels": 3136, "max_pixels": 6291456}),
    ]
    checks = {
        "vllm_args": (SERVING["vllm_args"], expected),
        "client_parallel": (SERVING["client_parallel"],
                            min(max_num_seqs, VllmBackend.MAX_AUTO_PARALLEL)),
        "fullpage_regen": (SERVING["fullpage_regen"], settings.SURYA_FULLPAGE_REGEN),
        "max_tokens_full_page": (SERVING["max_tokens_full_page"],
                                 settings.SURYA_MAX_TOKENS_FULL_PAGE),
        "guided_layout": (SERVING["guided_layout"], settings.SURYA_GUIDED_LAYOUT),
        "model": (SERVING["model"], settings.SURYA_MODEL_CHECKPOINT),
        "vllm_image": (SERVING["vllm_image"].split("@")[0], settings.VLLM_DOCKER_IMAGE),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        sys.exit(f"SERVING disagrees with the installed surya-ocr defaults: {bad}")
    log("SERVING matches the installed surya-ocr defaults")


@contextlib.contextmanager
def vllm_server():
    """Serve the pinned checkpoint for the duration of the block, under the package's name."""
    from huggingface_hub import snapshot_download

    vllm = image_vllm()
    if not vllm:
        sys.exit("no `vllm` on PATH outside this venv: run on the vllm/vllm-openai image "
                 f"({SERVING['vllm_image']})")
    model_dir = snapshot_download(SERVING["model"], revision=SERVING["model_revision"])
    cmd = [vllm, "serve", model_dir, "--host", "127.0.0.1", "--port", str(PORT),
           *SERVING["vllm_args"], "--served-model-name", SERVING["model"]]
    # This script's VIRTUAL_ENV must not leak into the image's vLLM.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    log("starting vLLM: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        deadline = time.time() + 1200
        while True:
            if proc.poll() is not None:
                sys.exit(f"vLLM exited during startup (code {proc.returncode})")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
                    if SERVING["model"] in r.read().decode():
                        break
            except OSError:
                pass
            if time.time() > deadline:
                sys.exit("vLLM did not become ready within 20 minutes")
            time.sleep(5)
        log("vLLM ready")
        yield
    finally:
        stop_process_group(proc)


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def stop_process_group(proc: subprocess.Popen):
    """SIGTERM the server's process group, then SIGKILL; always reap the parent."""
    pgid = proc.pid  # start_new_session=True: the server leads its own group
    for sig, wait_s in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
        if not group_alive(pgid):
            break
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, sig)
        deadline = time.time() + wait_s
        while group_alive(pgid) and time.time() < deadline:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=1)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5)


# --- the package, plus a trace of its requests ----------------------------------------------

_TRACE = threading.local()


def build_predictor():
    """The package's manager and predictor, attached to our server, with `generate` traced.

    The wrapper only records what each request returned. It hands the package's outputs back
    unchanged, so the page result is the package's own.
    """
    from surya.inference import SuryaInferenceManager
    from surya.recognition import RecognitionPredictor

    manager = SuryaInferenceManager(method="vllm")
    manager.start()
    backend = manager.backend
    inner = backend.generate

    def traced(batch):
        outputs = inner(batch)
        calls = getattr(_TRACE, "calls", None)
        if calls is not None:
            for item, out in zip(batch, outputs, strict=True):
                calls.append({"prompt_type": str(item.prompt_type), "raw": out.raw,
                              "finish_reason": out.finish_reason, "tokens": out.token_count,
                              "error": bool(out.error)})
        return outputs

    backend.generate = traced
    log(f"attached to {backend.handle.base_url} as {backend.handle.model_name}; "
        f"client parallel {backend._client_parallel()}")
    return RecognitionPredictor(manager)


def read_page(predictor, image) -> dict:
    """One page through the package's full-page path -> one output row (minus id/model)."""
    from surya.inference.prompts import PROMPT_TYPE_HIGH_ACCURACY_BBOX

    _TRACE.calls = []
    t0 = time.time()
    try:
        page = predictor([image], full_page=True)[0]
    finally:
        calls, _TRACE.calls = _TRACE.calls, None
    secs = time.time() - t0

    full = [c for c in calls if c["prompt_type"] == str(PROMPT_TYPE_HIGH_ACCURACY_BBOX)]
    other = [c for c in calls if c["prompt_type"] != str(PROMPT_TYPE_HIGH_ACCURACY_BBOX)]
    first = full[0] if full else {}
    blocks = [{"label": b.label, "raw_label": b.raw_label, "reading_order": b.reading_order,
               "html": b.html, "skipped": b.skipped, "error": b.error, "confidence": b.confidence,
               "bbox": [round(v, 1) for v in b.bbox]} for b in page.blocks]
    n_capped = sum(1 for c in other if c["finish_reason"] == "length")
    n_failed = sum(1 for c in other if c["error"])
    # Only block-mode outputs reach the page with a cap: the package never keeps a capped
    # full-page output. Layout calls are in `other` too; a capped layout yields no block.
    capped_in_page = any(b["error"] and b["html"] for b in blocks)
    return {
        "raw": json.dumps(blocks, ensure_ascii=False),
        "error": f"{n_failed} block request(s) failed" if n_failed else None,
        "finish_reason": "length" if capped_in_page else "stop",
        "fullpage_raw": first.get("raw"),
        "fullpage_finish_reason": first.get("finish_reason"),
        "fullpage_tokens": first.get("tokens"),
        "fallback": bool(other),
        "n_requests": len(calls),
        "n_blocks": len(blocks),
        "n_capped_blocks": n_capped,
        "n_failed_blocks": n_failed,
        "secs": round(secs, 2),
    }


# --- output parts and resume (as drivers/kraken-ppocrv6-port.py) ------------------------------


def part_paths(output: str):
    if output.startswith(("hf://", "s3://", "gs://")):
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        return sorted(fs.glob(f"{output.rstrip('/')}/part-*.parquet")), fs
    directory = Path(output)
    return (sorted(str(p) for p in directory.glob("part-*.parquet")) if directory.is_dir() else []), None


def done_ids(output: str, retry_errors: bool) -> set:
    """Ids already written under the output prefix. With retry_errors, an id whose rows are all
    error rows is not done, so it is read again; consolidate_run keeps the success."""
    import pandas as pd

    paths, fs = part_paths(output)
    if not paths:
        return set()
    frames = []
    for path in paths:
        if fs is not None:
            with fs.open(path, "rb") as fh:
                frames.append(pd.read_parquet(fh, columns=["id", "error"]))
        else:
            frames.append(pd.read_parquet(path, columns=["id", "error"]))
    frame = pd.concat(frames, ignore_index=True)
    if retry_errors:
        frame = frame[frame["error"].isna()]
    ids = set(frame["id"].tolist())
    log(f"resume: {len(ids)} ids done in {len(paths)} part file(s)"
        + (" (error rows not counted)" if retry_errors else ""))
    return ids


def write_part(rows, output: str, rank: int, index: int) -> str:
    """Flush rows as one part file. The rank is in the filename because shards share a prefix."""
    import pandas as pd

    path = f"{output.rstrip('/')}/part-r{rank:02d}-{index:05d}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    log(f"wrote {len(rows)} rows -> {path}")
    return path


def next_part_index(existing_parts, rank: int) -> int:
    """One past the highest part index this rank has written (never a count; see kraken)."""
    pattern = re.compile(rf"part-r{rank:02d}-(\d+)\.parquet$")
    indices = [int(m.group(1)) for p in existing_parts if (m := pattern.search(str(p)))]
    return max(indices) + 1 if indices else 0


def parse_shard(spec: str) -> tuple[int, int]:
    rank, _, world = spec.partition("/")
    rank, world = int(rank), int(world or 1)
    if world < 1 or not 0 <= rank < world:
        raise argparse.ArgumentTypeError(
            f"--shard must be rank/world with 0 <= rank < world, got {spec!r}")
    return rank, world


def main():
    ap = argparse.ArgumentParser(description="Surya OCR 2.5 (surya-ocr package) — inference only")
    ap.add_argument("--input-dataset", required=True)
    # REQUIRED, no default — a forgotten --output must not resume into another run's prefix.
    ap.add_argument("--output", required=True,
                    help="output prefix, e.g. hf://buckets/<owner>/<bucket>/<run>/<model>/")
    ap.add_argument("--revision", default=None, help="pin the benchmark revision")
    ap.add_argument("--split", default="train")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--id-column", default="PageID")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test against N pages. NOT a publishable run.")
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="RANK/WORLD",
                    help="strided fan-out across jobs, e.g. 2/4 (default: 0/1)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-read pages whose only rows are error rows")
    args = ap.parse_args()

    # The package reads its settings at import, so set the attach path first.
    os.environ["SURYA_INFERENCE_URL"] = f"http://127.0.0.1:{PORT}/v1"
    os.environ["SURYA_INFERENCE_BACKEND"] = "vllm"
    os.environ["VLLM_GPU_TYPE"] = SERVING["vllm_gpu_type"]
    os.environ.setdefault("DISABLE_TQDM", "true")

    from datasets import load_dataset

    check_package_flags()
    rank, world = args.shard
    dataset = load_dataset(args.input_dataset, revision=args.revision, split=args.split)
    if world > 1:
        dataset = dataset.select(range(rank, len(dataset), world))
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    already = done_ids(args.output, args.retry_errors)
    existing_parts, _ = part_paths(args.output)
    part_index = next_part_index(existing_parts, rank)

    rows, written = [], 0
    counts = {"errors": 0, "truncated": 0, "fallback": 0, "empty": 0}
    total = len(dataset)
    with vllm_server():
        predictor = build_predictor()
        parallel = SERVING["client_parallel"]
        page_clock = time.time()

        def work(page_id, image):
            try:
                return {"id": page_id, **read_page(predictor, image)}
            except Exception as exc:  # a durable error row, never a sentinel in the text column
                return {"id": page_id, "raw": None, "error": f"{type(exc).__name__}: {exc}",
                        "finish_reason": None}

        def collect(future):
            nonlocal rows, written, part_index
            record = future.result()
            record["model"] = SERVING["model"]
            if record.get("error"):
                counts["errors"] += 1
            if record.get("finish_reason") == "length":
                counts["truncated"] += 1
            if record.get("fallback"):
                counts["fallback"] += 1
            if record.get("raw") == "[]":
                counts["empty"] += 1
            rows.append(record)
            if len(rows) >= FLUSH_EVERY:
                write_part(rows, args.output, rank, part_index)
                part_index += 1
                written += len(rows)
                rows = []
                rate = (time.time() - page_clock) / max(written, 1)
                log(f"  {written}/{total - len(already)} pages · {rate:.2f}s/page · {counts}")

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            in_flight = set()
            for row in dataset:
                page_id = row[args.id_column]
                if page_id in already:
                    continue
                image = row[args.image_column].convert("RGB")
                in_flight.add(pool.submit(work, page_id, image))
                if len(in_flight) >= 2 * parallel:
                    finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in finished:
                        collect(future)
            for future in in_flight:
                future.result()
            for future in list(in_flight):
                collect(future)

        if rows:
            write_part(rows, args.output, rank, part_index)
            written += len(rows)

    summary = {
        "pages_written": written,
        "pages_skipped_resume": len(already),
        **counts,
        "elapsed_s": round(time.time() - START, 1),
        "s_per_page": round((time.time() - page_clock) / max(written, 1), 3),
        "shard": f"{rank}/{world}",
    }
    print("PORT surya-ocr25 " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
