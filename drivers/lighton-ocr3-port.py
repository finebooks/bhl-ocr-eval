# /// script
# requires-python = ">=3.11"
# dependencies = ["saturate[hf]>=0.1.1", "pillow"]
# ///
"""lightonai/LightOnOCR-3-4B and -0.8B — saturate driver for the BHL OCR benchmark.

Run (one model per job; --model picks the checkpoint):
    hf jobs uv run --image vllm/vllm-openai:v0.30.0 --flavor a10g-small \
      --secrets HF_TOKEN lighton-ocr3-port.py -- \
      --model lightonai/LightOnOCR-3-4B --model-revision <sha> \
      --input-dataset <benchmark> --id-column PageID --output <prefix> --limit 20

ONE DRIVER, TWO ROWS. The 4B and 0.8B checkpoints share everything the card specifies —
architecture (stock Qwen3_5ForConditionalGeneration), processor, request shape, sampling
and rendering advice — so two files would differ only in the model id. Each row is still
its own run with its own model revision, job ids and provenance record. The 1B variant is
out of scope: it is a Pixtral model with different pins, not a size of this one.

Request shape is the card's verbatim transcription call: the message is the image ONLY.
The card's example sends `PROMPT = "grounding"  # or "" for transcription-only` and
appends a text part only when PROMPT is non-empty, so transcription is image-only — the
same trained format as the LightOnOCR-2 row. Grounding (layout boxes) is a different
output and is not run here.

Images: the card renders "at 400 DPI ... to a target longest dimension of 2048px" and its
example calls `pil_image.thumbnail((2048, 2048))` before PNG-encoding. This driver does
exactly that: thumbnail() only ever DOWNSCALES and keeps the aspect ratio. Unlike the
LightOnOCR-2 driver (which forces 1540 px in both directions, for historical reasons),
nothing is upscaled; every BHL scan is larger than 2048 px on its long side anyway.

`parse` stores the completion VERBATIM in `raw` and transforms nothing. Post-processing
lives in runners/normalize_outputs.py, so a rule can be corrected without a GPU run.
"""
import argparse
import base64
import io

MODELS = ["lightonai/LightOnOCR-3-4B", "lightonai/LightOnOCR-3-0.8B"]

SERVING = {
    # Per-value provenance. "card" = the model card's own vllm serve / request example
    # (identical on the 4B and 0.8B cards); "upstream" = LightOn's GitHub repo;
    # "house" = our choice.
    "model": MODELS[0],  # replaced by --model
    # upstream, and a deliberate EXCEPTION to the board-wide v0.26.0 pin. LightOn's
    # pyproject.toml (github.com/lightonai/LightOnOCR) says "vLLM 0.27.1 makes the models
    # loop; 0.30.0 is tested". A pin that upstream says makes the model loop would benchmark
    # the engine, not the model. Same kind of recorded exception as kraken, Falcon-OCR and
    # surya-ocr-2.
    "image": "vllm/vllm-openai:v0.30.0",
    # house. The card gives no max_model_len (native is 262144, which profiling cannot boot
    # on an a10g). A 2048 px page is at most 64 x 64 = 4096 image tokens (patch 16, merge 2,
    # processor_config.json), so 16384 leaves room for the card's 4096 output tokens with
    # margin; asserted below. Same value as the Qwen3.5-9B row.
    "max_model_len": 16384,
    # house: the specialist peers' value (lighton-ocr2, ovis-ocr2). The card sets none.
    "gpu_memory_utilization": 0.8,
    "extra_args": [
        # card, verbatim: thinking is disabled server-side. It is also sent per request below,
        # so the row does not depend on the server default alone.
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        # house OCR defaults, same as every peer: one image per request and no video (the
        # checkpoint has a video processor); OCR never reuses an image, so the caches only
        # cost memory.
        "--limit-mm-per-prompt", '{"image": 1, "video": 0}',
        "--mm-processor-cache-gb", "0",
        "--no-enable-prefix-caching",
        # --trust-remote-code (card) DROPPED: config.json has no auto_map and the
        # architecture is registered in vLLM itself, so the flag buys nothing — the same
        # reason every other driver here drops it.
    ],
    # card, verbatim: the sampling block in the card's example request (as the LightOnOCR-2
    # row, which also runs the card's temperature 0.2 / top_p 0.9). No seed, as on the
    # whole board.
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.9,
}

# Image-token budget from processor_config.json (both checkpoints): Qwen2VLImageProcessor,
# patch_size 16, merge_size 2 -> one visual token per 32 x 32 px.
PATCH_SIZE = 16
MERGE_SIZE = 2
TARGET_LONGEST = 2048


def image_tokens(width: int, height: int) -> int:
    step = PATCH_SIZE * MERGE_SIZE
    return -(-width // step) * -(-height // step)


# Worst case is a SQUARE page at the 2048 px clamp.
IMAGE_TOKENS_WORST_CASE = image_tokens(TARGET_LONGEST, TARGET_LONGEST)
assert IMAGE_TOKENS_WORST_CASE + SERVING["max_tokens"] + 256 <= SERVING["max_model_len"], (
    f"context math: {IMAGE_TOKENS_WORST_CASE} image tokens + {SERVING['max_tokens']} output "
    f"tokens + template overhead exceeds max_model_len {SERVING['max_model_len']}"
)


def to_pil(value):
    """One dataset image cell -> a PIL image (decoded image, {"bytes","path"} dict or bytes)."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value)))
    raise ValueError(f"unsupported image value: {type(value)}")


def encode_png(value) -> str:
    """RGB-convert, thumbnail to 2048 px (card, verbatim), return base64 PNG."""
    img = to_pil(value).convert("RGB")
    img.thumbnail((TARGET_LONGEST, TARGET_LONGEST))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def parse_shard(spec: str) -> tuple[int, int]:
    rank, _, world = spec.partition("/")
    rank, world = int(rank), int(world or 1)
    if world < 1 or not 0 <= rank < world:
        raise argparse.ArgumentTypeError(f"--shard must be rank/world with 0 <= rank < world, got {spec!r}")
    return rank, world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS,
                    help="Checkpoint to serve; each one is its own board row")
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
    # REQUIRED, with no default: a forgotten --output must not resume into another run's
    # output (saturate anti-joins on id).
    ap.add_argument("--output", required=True,
                    help="output prefix, e.g. hf://buckets/<owner>/<bucket>/<run>/<model>/")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="RANK/WORLD",
                    help="Strided fan-out across jobs, e.g. 2/8 (default: 0/1)")
    ap.add_argument("--max-tokens", type=int, default=SERVING["max_tokens"])
    ap.add_argument("--retry-errors", action="store_true")
    args = ap.parse_args()
    SERVING["model"] = args.model

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
            # Image ONLY — the card's transcription call (PROMPT = "").
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            "temperature": SERVING["temperature"],
            "top_p": SERVING["top_p"],
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def parse(row, body):
        choices = body.get("choices") or []
        if not choices:
            raise ValueError(f"no choices in response: {body!r}")
        usage = body.get("usage") or {}
        # `or ""` is envelope handling, not text cleaning: a null `content` is malformed.
        return {"raw": choices[0]["message"].get("content") or "",
                "model": SERVING["model"],
                "finish_reason": choices[0].get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}

    extra = ["--max-model-len", str(SERVING["max_model_len"]),
             "--gpu-memory-utilization", str(SERVING["gpu_memory_utilization"]),
             *SERVING["extra_args"]]
    # The served checkpoint is pinned at boot, not read back from the Hub afterwards.
    if args.model_revision:
        SERVING["model_revision"] = args.model_revision
        extra += ["--revision", args.model_revision]
    with Engine(SERVING["model"], engine="vllm", extra_args=extra) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, args.output,
                     window=Auto(initial=4, target_waiting=4, max_limit=12, step=2),
                     shard=(rank, world),
                     retry_errors=args.retry_errors)
    tag = "lighton-ocr3-" + args.model.rsplit("-", 1)[-1].lower()
    print(f"PORT {tag} " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
