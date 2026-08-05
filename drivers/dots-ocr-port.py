# /// script
# requires-python = ">=3.10"
# dependencies = ["saturate[hf]>=0.1.1", "pillow"]
# ///
"""rednote-hilab/dots.ocr — saturate port of uv-scripts/ocr/dots-ocr.py (port-matrix row).

Run:
    hf jobs uv run --image vllm/vllm-openai:v0.26.0 --flavor a10g-small \
      --secrets HF_TOKEN dots-ocr-port.py -- \
      --input-dataset <benchmark> --id-column PageID --limit 20

Input is a Hub dataset with an image column; `--id-column PageID` keys the output rows
by the benchmark's page id, which is what scoring joins ground truth on. Fan out over
several jobs with `--shard i/n` (strided assignment, one job per rank; parquet parts
and completion markers are per-rank, so they land in one output repo).

Default --prompt-mode ocr. Note vs the recipe: its "[OCR ERROR]" fallback string is
deleted — failures become durable saturate error rows instead.

`parse` stores the completion VERBATIM in `raw` and transforms nothing: the per-model
post-processing that turns raw into the scored `markdown` column lives in
bhl-ocr-eval/runners/normalize_outputs.py, so a rule can be corrected without paying
for GPUs again.
"""
import argparse
import base64
import io

SERVING = {
    # provenance: dots-ocr.py LLM(...) kwargs + 07-16 sweep house flags
    "model": "rednote-hilab/dots.ocr",
    "image": "vllm/vllm-openai:v0.26.0",
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.8,
    "extra_args": [
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
        "--limit-mm-per-prompt", '{"image": 1}',
    ],
    "max_tokens": 8192,
    "temperature": 0.0,
}
assert SERVING["max_tokens"] < SERVING["max_model_len"], "context math: max_tokens must leave input room"

PROMPT = "Extract the text content from this image."

# Formats whose bytes go to the server untouched when the image is already RGB.
PASSTHROUGH_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg"}


def encode_image(value) -> tuple[str, str]:
    """Return `(mime, base64)` for one dataset image cell, at full resolution.

    A `datasets` image column hands over a decoded `PIL.Image`, an undecoded
    `{"bytes", "path"}` dict, or raw bytes depending on how the dataset stores it, so
    all three shapes land here rather than one being assumed.

    Already-RGB PNG/JPEG bytes are passed through byte-for-byte — this driver applies
    no client-side resize (the processor does its own clamping), so re-encoding buys
    nothing. Everything else (grayscale, bitonal, palette, CMYK, RGBA — the common
    shapes for library scans, and what a decoded image column yields) is converted to
    RGB and re-encoded as lossless PNG, which is the format this driver has always
    sent; no resize and no lossy step is introduced.
    """
    from PIL import Image

    raw = None
    if isinstance(value, dict) and value.get("bytes"):
        raw = value["bytes"]
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)

    if raw is not None:
        img = Image.open(io.BytesIO(raw))
        mime = PASSTHROUGH_FORMATS.get(img.format)
        if mime and img.mode == "RGB":
            return mime, base64.b64encode(raw).decode()
    elif isinstance(value, Image.Image):
        img = value
    else:
        raise ValueError(f"unsupported image value: {type(value)}")

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "image/png", base64.b64encode(buf.getvalue()).decode()


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
    ap.add_argument("--model-revision", default=None,
                    help="Pin the SERVED model checkpoint sha (vllm serve --revision)")
    ap.add_argument("--id-column", default=None,
                    help="Column to use as row id (default: split-index ids)")
    # REQUIRED, with no default. A forgotten --output must not resume into some other
    # run's output: saturate anti-joins on id, so a stray default would blend two
    # configurations into one complete, fingerprint-passing table. (The old default
    # also pointed at a private scratch repo, which is meaningless outside this laptop.)
    ap.add_argument("--output", required=True,
                    help="output prefix, e.g. hf://buckets/<owner>/<bucket>/<run>/<model>/")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="RANK/WORLD",
                    help="Strided fan-out across jobs, e.g. 2/8 (default: 0/1)")
    ap.add_argument("--max-tokens", type=int, default=SERVING["max_tokens"])
    ap.add_argument("--retry-errors", action="store_true")
    # Admission cap. Default 12 is the board-wide setting and changes nothing; lower it
    # when peak prefill activation OOMs the engine. Concurrent large-image prefills are
    # what blows the allocator, and this workload is prefill-bound (see AXES.md), so
    # cutting the cap costs little wall-clock. A CHANGED VALUE IS A CONFIG CHANGE:
    # write to a new output prefix, never resume an existing one.
    ap.add_argument("--max-window", type=int, default=12,
                    help="Auto max_limit (default 12, the board-wide value)")
    # vLLM reserves this fraction of the card as a KV pool up front. Measured KV usage on
    # this corpus is 2.5-6%, i.e. the pool is >10x oversized, and the reservation leaves no
    # headroom for vision-encoder activations at these image sizes — which is what actually
    # OOM'd the 2026-08-03 runs (num_running_reqs=4, kv_cache_usage=0.025, allocator short
    # by 266 MiB). Lowering it is a DOCUMENTED DEVIATION where the value is card-verbatim.
    # A CHANGED VALUE IS A CONFIG CHANGE: write to a new output prefix.
    ap.add_argument("--gpu-memory-utilization", type=float,
                    default=SERVING["gpu_memory_utilization"],
                    help="override the card/recipe value (see comment)")
    args = ap.parse_args()

    from saturate import Auto, Engine, dataset_rows, pump, shard_select

    rank, world = args.shard

    rows = dataset_rows(
        args.input_dataset, config=args.config, split=args.split,
        columns=[args.image_column], ids=args.id_column or "index",
        revision=args.revision, limit=args.limit,
    )
    if world > 1:
        rows = shard_select(rows, rank=rank, world=world)

    def to_request(row):
        mime, b64 = encode_image(row[args.image_column])
        return {
            "model": SERVING["model"],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ]}],
            "temperature": SERVING["temperature"],
            "max_tokens": args.max_tokens,
        }

    def parse(row, body):
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        return {"raw": choice["message"]["content"],
                "model": SERVING["model"],
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}

    extra = ["--max-model-len", str(SERVING["max_model_len"]),
             "--gpu-memory-utilization", str(args.gpu_memory_utilization),
             *SERVING["extra_args"]]
    # The served checkpoint is pinned at boot, not read back from the Hub afterwards:
    # a post-hoc head read is not guaranteed to be what vLLM actually loaded. Placed
    # before any --logits_processors append, which must stay last.
    if args.model_revision:
        SERVING["model_revision"] = args.model_revision
        extra += ["--revision", args.model_revision]
    with Engine(SERVING["model"], engine="vllm", extra_args=extra) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, args.output,
                     window=Auto(initial=min(4, args.max_window), target_waiting=4,
                                 max_limit=args.max_window, step=2),
                     shard=(rank, world),
                     retry_errors=args.retry_errors)
    print("PORT dots-ocr " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
