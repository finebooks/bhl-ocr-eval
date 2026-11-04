# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mineru-vl-utils==2.0.7",
#     "datasets>=3.1.0",
#     "huggingface-hub>=1.5.0,<2",
#     "pandas",
#     "pyarrow",
#     "pillow",
# ]
#
# [tool.uv]
# exclude-newer = "2026-11-01T00:00:00Z"
# ///
"""opendatalab/MinerU2.5-Pro-2605-1.2B — MinerU's two-step page extraction, as an inference driver.

Run:
    hf jobs uv run --detach --flavor a10g-large -s HF_TOKEN --timeout 4h \\
        --image vllm/vllm-openai@sha256:a230095847e93bd4df9888b33dab956fa9504537b828a23657d2b26fed57b5c9 \\
        drivers/mineru25-pro-port.py -- \\
        --input-dataset <benchmark> --revision <sha> --id-column PageID \\
        --output hf://buckets/<owner>/<bucket>/<prefix>/

Inference only. It writes `part-*.parquet` files of `{id, raw, error, finish_reason, ...}` and
stops, which is the shape `runners/consolidate_run.py` reads. `runners/normalize_outputs.py`
turns `raw` into the scored column.

WHAT RUNS

MinerU2.5-Pro is a 1.2B VLM (Qwen2-VL architecture) behind MinerU's own client,
`mineru-vl-utils`. The client is the product: `MinerUClient.two_step_extract(image)` runs a
layout request on a 1036 x 1036 copy of the page, crops each detected block from the original
page, sends one recognition request per block, and post-processes the blocks; `json2md` joins
them into the page's markdown. This driver calls exactly that, as Datalab's omni_parse_bench
harness does (harness/providers/serve/mineru, commit 74629e1): same vLLM image digest
(v0.21.0), same mineru-vl-utils release (2.0.7), same checkpoint commit, same server flags
(MinerU's logits processor, GPU memory 0.85) and the client's default settings. The page is
handed over as an RGB image at its stored pixels, as that harness opens an image file.

vLLM loads MinerU's logits processor (`--logits-processors mineru_vl_utils:MinerULogitsProcessor`,
the no-repeat-ngram processor the client's sampling settings drive), so mineru-vl-utils must be
importable by the image's own Python. The driver installs the pinned release into it at start-up
(`uv pip install --system`), as the harness's Dockerfile does, then starts `vllm serve` from the
image. The client side runs from this script's uv environment.

WHAT IS STORED

`raw` is the page's markdown as MinerU's `json2md` renders the client's blocks: the output a user
of the client gets. That rendering is the vendor's, so it is stored as the response and the
normalizer only strips outer whitespace (as for tiiuae/Falcon-OCR and datalab-to/indra-ocr,
whose packages also return assembled markdown). `blocks` keeps the client's blocks as JSON
({type, bbox, angle, content}, boxes from 0 to 1) for inspection.

The client does not report finish reasons, so `finish_reason` is "stop" for every page that
returned; a request error after the client's own retries becomes an error row, which
`--retry-errors` re-reads. A blank page answered with no blocks is stored as "" (the right
answer on a blank leaf).
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
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

START = time.time()

# Per-value provenance, in the shape `scripts/emit_run_provenance.py` reads by AST:
#
# - model / model_revision: the Hub checkpoint at the commit omni_parse_bench's mineru provider
#   pins (its start.sh). Downloaded at that commit and served from disk under the model's name.
# - package: the mineru-vl-utils release that harness pins (its Dockerfile). Client-side deps are
#   fixed by the `exclude-newer` date in the script header.
# - vllm_image: that harness's base image, pinned by manifest digest.
# - vllm_args: that harness's start.sh flags, verbatim (MinerU's logits processor, memory 0.85).
#   No max-model-len: neither the harness nor MinerU's README sets one.
# - client: MinerUClient's defaults for the http-client backend (sampling, prompts, layout image
#   size, retries), as the harness constructs it, except use_tqdm off.
# - page_parallel: house. Pages are submitted from this many threads to one shared client, as the
#   harness's threaded server shares one client; each page is the client's own two-step path.
SERVING = {
    "model": "opendatalab/MinerU2.5-Pro-2605-1.2B",
    "model_revision": "08aaea840498d49ce16247b6263196cf02814885",
    "package": "mineru-vl-utils==2.0.7",
    "vllm_image": "vllm/vllm-openai:v0.21.0@sha256:a230095847e93bd4df9888b33dab956fa9504537b828a23657d2b26fed57b5c9",
    "vllm_args": [
        "--logits-processors", "mineru_vl_utils:MinerULogitsProcessor",
        "--gpu-memory-utilization", "0.85",
    ],
    "client": "MinerUClient(backend='http-client', use_tqdm=False), package defaults otherwise",
    "page_parallel": 32,
    "mode": "two_step_extract (layout, then one request per block), json2md",
    "image_preprocessing": "convert('RGB'), the stored pixels, as omni_parse_bench opens an image",
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


def image_python() -> str | None:
    """The image's own `python3`, looking past this script's venv."""
    own_bin = os.path.realpath(os.path.join(sys.prefix, "bin"))
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                           if os.path.realpath(p) != own_bin)
    return shutil.which("python3", path=path)


def install_logits_processor():
    """Install the pinned mineru-vl-utils into the image's Python, so vLLM can load its processor."""
    python = image_python()
    uv = shutil.which("uv")
    if not python or not uv:
        sys.exit(f"need the image's python3 and uv on PATH (python3={python}, uv={uv})")
    cmd = [uv, "pip", "install", "--python", python, "--system", "--break-system-packages",
           "--no-cache", SERVING["package"]]
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    log("installing the logits processor into the image: " + " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)
    check = subprocess.run([python, "-c", "import mineru_vl_utils as m; print(m.__version__)"],
                           env=env, capture_output=True, text=True, check=True)
    log(f"image python has mineru-vl-utils {check.stdout.strip()}")


@contextlib.contextmanager
def vllm_server():
    """Serve the pinned checkpoint for the duration of the block, under the model's name."""
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


# --- the client ------------------------------------------------------------------------------


def build_client():
    """MinerU's client, attached to our server, with the package defaults (as the harness)."""
    from mineru_vl_utils import MinerUClient

    client = MinerUClient(backend="http-client", server_url=f"http://127.0.0.1:{PORT}",
                          use_tqdm=False)
    log(f"MinerUClient attached to http://127.0.0.1:{PORT}")
    return client


def read_page(client, image) -> dict:
    """One page through the client's two-step path -> one output row (minus id/model)."""
    from mineru_vl_utils.post_process import json2md

    t0 = time.time()
    blocks = [dict(b) for b in client.two_step_extract(image)]
    secs = time.time() - t0
    for b in blocks:  # drop non-JSON extras the client may attach (scores are off by default)
        b.pop("scored", None)
    return {
        "raw": json2md(blocks),
        "blocks": json.dumps(blocks, ensure_ascii=False, default=str),
        "error": None,
        "finish_reason": "stop",
        "n_blocks": len(blocks),
        "block_types": json.dumps(sorted({b.get("type", "") for b in blocks})),
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
    ap = argparse.ArgumentParser(description="MinerU2.5-Pro (mineru-vl-utils) — inference only")
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

    from datasets import load_dataset

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
    counts = {"errors": 0, "empty": 0}
    total = len(dataset)
    install_logits_processor()
    with vllm_server():
        client = build_client()
        parallel = SERVING["page_parallel"]
        page_clock = time.time()

        def work(page_id, image):
            try:
                return {"id": page_id, **read_page(client, image)}
            except Exception as exc:  # a durable error row, never a sentinel in the text column
                return {"id": page_id, "raw": None, "error": f"{type(exc).__name__}: {exc}"[:2000],
                        "finish_reason": None}

        def collect(future):
            nonlocal rows, written, part_index
            record = future.result()
            record["model"] = SERVING["model"]
            if record.get("error"):
                counts["errors"] += 1
            elif record.get("raw") == "":
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
    print("PORT mineru25-pro " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
