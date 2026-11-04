# /// script
# requires-python = ">=3.11"
# dependencies = ["saturate[hf]==0.2.0", "pillow"]
# ///
"""Logics-MLLM/Logics-Parsing-v2 — saturate driver for the BHL OCR benchmark.

Run (4.44B params, bf16):
    hf jobs uv run --detach --image vllm/vllm-openai:v0.26.0 --flavor a100-large \
      -s HF_TOKEN --timeout 5h logics-parsing-v2-port.py -- \
      --model-revision <sha> --input-dataset <benchmark> --revision <sha> \
      --id-column PageID --output <prefix> --limit 20

A Qwen3-VL fine-tune (Qwen3VLForConditionalGeneration, stock in vLLM) that writes a page as
HTML blocks with `data-bbox` coordinates. Added to compare the board with the seven models of
Datalab's benchmark-trust study (datalab.to/blog, 2026-11-04), which ran Logics-Parsing-v2.

Everything model-side is the vendor's own inference script, `inference_v2.py` in
github.com/alibaba/Logics-Parsing at 516925b (its last commit before the repo moved to V3):
- Prompt: "QwenVL HTML" (the script's default), as a text part AFTER the image.
- Sampling: max_new_tokens 16384, temperature 0.1, top_p 0.5, repetition_penalty 1.05.
  Not greedy; that is the vendor's setting. No seed, as on the whole board.
- Image: the script sets the processor's max_pixels to 7200 * 32 * 32 and min_pixels to 3136.
  This driver does that resize client-side with the script's own `smart_resize` (factor 32,
  bicubic as processor_config resample=3), so the processor receives an already-aligned image
  inside its default bounds (16,777,216 / 65,536 px) and does not resize it again. That fixes
  the visual budget at 7,200 tokens per page whatever the processor's kwargs handling.

`parse` stores the completion VERBATIM in `raw`. Turning the HTML into scored text is a
decision about the text, so it lives in runners/normalize_outputs.py (the vendor's own
`qwenvl_cast_html_tag`, ported there).
"""
import argparse
import base64
import io
import math

MODEL = "Logics-MLLM/Logics-Parsing-v2"

SERVING = {
    # Per-value provenance. "vendor" = inference_v2.py at 516925b; "house" = our choice.
    "model": MODEL,
    # house: the board-wide pin. Qwen3-VL is stock in this vLLM.
    "image": "vllm/vllm-openai:v0.26.0",
    # house: 7,200 image tokens + 16,384 output tokens + template overhead, asserted below.
    "max_model_len": 32768,
    # house: the specialist peers' value.
    "gpu_memory_utilization": 0.85,
    "extra_args": [
        # house OCR defaults, same as every peer: one image per request and no video; OCR
        # never reuses an image, so the caches only cost memory.
        "--limit-mm-per-prompt", '{"image": 1, "video": 0}',
        "--mm-processor-cache-gb", "0",
        "--no-enable-prefix-caching",
    ],
    # vendor, verbatim.
    "prompt": "QwenVL HTML",
    "max_tokens": 16384,
    "temperature": 0.1,
    "top_p": 0.5,
    "repetition_penalty": 1.05,
    "image_budget": "vendor smart_resize: factor 32, min_pixels 3136, max_pixels 7200*32*32, bicubic",
}

FACTOR = 32  # patch_size 16 x merge_size 2 (preprocessor_config.json)
MIN_PIXELS = 3136  # vendor: processor.image_processor.min_pixels
MAX_PIXELS = 7200 * 32 * 32  # vendor: processor.image_processor.max_pixels
TEMPLATE_OVERHEAD = 256

IMAGE_TOKENS_WORST_CASE = MAX_PIXELS // (FACTOR * FACTOR)
assert IMAGE_TOKENS_WORST_CASE + SERVING["max_tokens"] + TEMPLATE_OVERHEAD <= SERVING["max_model_len"], (
    f"context math: {IMAGE_TOKENS_WORST_CASE} image tokens + {SERVING['max_tokens']} output tokens "
    f"+ template overhead exceeds max_model_len {SERVING['max_model_len']}"
)


def smart_resize(height: int, width: int, factor: int = FACTOR, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """inference_v2.py's smart_resize, verbatim apart from the defaults being named constants."""
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


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


def encode_png(value) -> tuple[str, tuple[int, int]]:
    """RGB-convert, one bicubic resize to the vendor's budget, return (base64 PNG, (w, h))."""
    from PIL import Image

    img = to_pil(value).convert("RGB")
    height, width = smart_resize(img.height, img.width)
    if (height, width) != (img.height, img.width):
        img = img.resize((width, height), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), (width, height)


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
    ap.add_argument("--model-revision", required=True,
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
    SERVING["model_revision"] = args.model_revision

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
        b64, _ = encode_png(row[args.image_column])
        return {
            "model": MODEL,
            # vendor: the image, then the prompt text.
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": SERVING["prompt"]},
            ]}],
            "temperature": SERVING["temperature"],
            "top_p": SERVING["top_p"],
            "repetition_penalty": SERVING["repetition_penalty"],
            "max_tokens": args.max_tokens,
        }

    def parse(row, body):
        choices = body.get("choices") or []
        if not choices:
            raise ValueError(f"no choices in response: {body!r}")
        usage = body.get("usage") or {}
        # `or ""` is envelope handling, not text cleaning: a null `content` is malformed.
        return {"raw": choices[0]["message"].get("content") or "",
                "model": MODEL,
                "finish_reason": choices[0].get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}

    extra = ["--max-model-len", str(SERVING["max_model_len"]),
             "--gpu-memory-utilization", str(SERVING["gpu_memory_utilization"]),
             *SERVING["extra_args"],
             # The served checkpoint is pinned at boot, not read back from the Hub afterwards.
             "--revision", args.model_revision]
    with Engine(MODEL, engine="vllm", extra_args=extra) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, args.output,
                     window=Auto(initial=8, target_waiting=4, max_limit=48, step=4),
                     shard=(rank, world),
                     retry_errors=args.retry_errors)
    print("PORT logics-parsing-v2 " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
