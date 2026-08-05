# /// script
# requires-python = ">=3.10"
# dependencies = ["saturate[hf]>=0.1.1", "pillow"]
# ///
"""PaddlePaddle/PaddleOCR-VL-1.6 — saturate port of uv-scripts/ocr/paddleocr-vl-1.6.py (port-matrix row).

Run:
    hf jobs uv run --image vllm/vllm-openai:v0.26.0 --flavor a10g-small \
      --secrets HF_TOKEN paddleocr-vl-16-port.py -- \
      --input-dataset <benchmark> --id-column PageID \
      --output hf://datasets/<org>/<repo>/data --limit 20

Input is a Hub dataset with an image column, not a bucket glob. `--id-column PageID`
is what makes the output joinable: scoring joins OCR rows to ground truth on PageID,
so the row id must BE the PageID (the default `index` ids are only stable per
revision — pin `--revision` if you use them). Fan out over several jobs with
`--shard i/n`; parquet parts and completion markers are per-rank, so the ranks can
share one output repo.

Default --task-mode ocr: image-then-"OCR:" chat message. The recipe's client-side
smart_resize (28-multiple dims into [101920, 1003520] px) is reproduced here —
the bytes sent must be pre-resized this way to reproduce results. Recipe's
offline LLM kwargs (enforce_eager, max_num_batched_tokens) become serve flags.

`parse` stores the completion VERBATIM in `raw` and transforms nothing: the per-model
post-processing that turns raw into the scored `markdown` column lives in
bhl-ocr-eval/runners/normalize_outputs.py, so a rule can be corrected without paying
for GPUs again.
"""
import argparse
import base64
import io
import math

SERVING = {
    # provenance: paddleocr-vl-1.6.py LLM(...) kwargs + 07-16 sweep house flags
    "model": "PaddlePaddle/PaddleOCR-VL-1.6",
    "image": "vllm/vllm-openai:v0.26.0",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8,
    "extra_args": [
        "--trust-remote-code",
        "--enforce-eager",
        "--max-num-batched-tokens", "16384",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
        "--limit-mm-per-prompt", '{"image": 1}',
    ],
    "max_tokens": 4096,
    "temperature": 0.0,
}
assert SERVING["max_tokens"] < SERVING["max_model_len"], "context math: max_tokens must leave input room"

PROMPT = "OCR:"

FACTOR = 28
MIN_PIXELS = 28 * 28 * 130
MAX_PIXELS = 28 * 28 * 1280


def smart_resize(h: int, w: int):
    h_bar = round(h / FACTOR) * FACTOR
    w_bar = round(w / FACTOR) * FACTOR
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt(h * w / MAX_PIXELS)
        h_bar = math.floor(h / beta / FACTOR) * FACTOR
        w_bar = math.floor(w / beta / FACTOR) * FACTOR
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (h * w))
        h_bar = math.ceil(h * beta / FACTOR) * FACTOR
        w_bar = math.ceil(w * beta / FACTOR) * FACTOR
    return h_bar, w_bar


def open_image(value):
    """One dataset image cell -> PIL image.

    A dataset image column yields a PIL image, a {"bytes": ...} dict, or raw bytes
    depending on how the dataset stores it; all three land here. There is no
    pass-the-original-bytes-through path like some drivers have: smart_resize below
    is part of reproducing the recipe, so every image is re-encoded anyway.
    """
    from PIL import Image

    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value)))
    if isinstance(value, Image.Image):
        return value
    raise ValueError(f"unsupported image value: {type(value)}")


def parse_shard(spec: str) -> tuple[int, int]:
    rank, _, world = spec.partition("/")
    rank, world = int(rank), int(world or 1)
    if world < 1 or not 0 <= rank < world:
        raise argparse.ArgumentTypeError(f"--shard must be rank/world with 0 <= rank < world, got {spec!r}")
    return rank, world


def encode_png(value) -> str:
    from PIL import Image

    img = open_image(value).convert("RGB")
    w, h = img.size
    h_bar, w_bar = smart_resize(h, w)
    if (w_bar, h_bar) != (w, h):
        img = img.resize((w_bar, h_bar), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dataset", required=True,
                    help="Input dataset repo id (rows with an image column)")
    # REQUIRED, with no default. A forgotten --output must not resume into some other
    # run's output: saturate anti-joins on id, so a stray default would blend two
    # configurations into one complete, fingerprint-passing table. (The old default
    # also pointed at a private scratch repo, which is meaningless outside this laptop.)
    ap.add_argument("--output", required=True,
                    help="output prefix, e.g. hf://buckets/<owner>/<bucket>/<run>/<model>/")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--config", default=None, help="Dataset config name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--revision", default=None,
                    help="Pin the input revision (index ids are only stable per revision)")
    ap.add_argument("--model-revision", default=None,
                    help="Pin the SERVED model checkpoint sha (vllm serve --revision)")
    ap.add_argument("--id-column", default=None,
                    help="Column to use as row id (default: split-index ids)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="RANK/WORLD",
                    help="Strided fan-out across jobs, e.g. 2/8 (default: 0/1)")
    ap.add_argument("--max-tokens", type=int, default=SERVING["max_tokens"])
    ap.add_argument("--retry-errors", action="store_true")
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
        b64 = encode_png(row[args.image_column])
        return {
            "model": SERVING["model"],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
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
             "--gpu-memory-utilization", str(SERVING["gpu_memory_utilization"]),
             *SERVING["extra_args"]]
    # The served checkpoint is pinned at boot, not read back from the Hub afterwards:
    # a post-hoc head read is not guaranteed to be what vLLM actually loaded. Placed
    # before any --logits_processors append, which must stay last.
    if args.model_revision:
        SERVING["model_revision"] = args.model_revision
        extra += ["--revision", args.model_revision]
    with Engine(SERVING["model"], engine="vllm", extra_args=extra) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, args.output,
                     window=Auto(initial=4, target_waiting=4, max_limit=12, step=2),
                     shard=(rank, world),
                     retry_errors=args.retry_errors)
    print("PORT paddleocr-vl-16 " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
