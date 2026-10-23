# /// script
# requires-python = ">=3.11"
# dependencies = ["saturate[hf]>=0.1.1", "pillow"]
# ///
"""datalab-to/chandra-ocr-2 — saturate driver for the BHL OCR benchmark.

Run (5.3B params, bf16 ~10.6 GB):
    hf jobs uv run --image vllm/vllm-openai:v0.26.0 --flavor a10g-small \
      --secrets HF_TOKEN chandra-ocr2-port.py -- --model-revision <sha> \
      --input-dataset <benchmark> --id-column PageID --output <prefix> --limit 20

A port of the vendor's own vLLM client, `chandra-ocr` v0.2.0 (github.com/datalab-to/chandra,
tag v0.2.0 = 497dfd38d7625259307c81c979d44093e9f75f44): `chandra/model/vllm.py`'s
`generate_vllm`, with the `ocr_layout` prompt the CLI uses (`chandra/scripts/cli.py`) and the
image scaling it applies (`chandra/model/util.py: scale_to_fit`). The prompt and the scaling
function below are copied verbatim from that tag; the prompt is hash-checked at import.

The model replies with HTML layout blocks (`<div data-bbox=.. data-label=..>`). `raw` stores
that reply verbatim. The vendor's HTML -> markdown conversion (`chandra/output.py:
parse_markdown`) is a decision about the text, so it lives in runners/normalize_outputs.py.

ONE DEVIATION from the vendor client, deliberately: `generate_vllm` RETRIES a page up to six
times when it detects a repetition loop, at a raised temperature (0.2 per attempt, up to 0.8,
top_p 0.95). This driver makes one greedy attempt per page, like every other row on the board,
so a loop shows up in the loop-rate column instead of being resampled away. That makes this a
model-plus-default-decoding score, not a vendor-pipeline score.

Licence: code Apache-2.0; weights a modified OpenRAIL-M (free for research, personal use and
startups under $2M; not for use competing with Datalab's API). Recorded in RESULTS.md.
"""
import argparse
import base64
import hashlib
import io
from typing import Tuple

from PIL import Image

SERVING = {
    # Per-value provenance. "vendor" = chandra-ocr v0.2.0 (settings.py, scripts/vllm.py,
    # model/vllm.py); "house" = our choice.
    "model": "datalab-to/chandra-ocr-2",
    # house: the board-wide pin. The vendor's launcher pins vllm/vllm-openai:v0.17.0; the
    # architecture (Qwen3_5ForConditionalGeneration) is served on v0.26.0 by the Qwen3.5-9B
    # and OvisOCR2 rows, so one engine across rows keeps a difference the model's.
    "image": "vllm/vllm-openai:v0.26.0",
    # house, and a CORRECTION of the vendor's 18000. scale_to_fit allows 3072 x 2048 =
    # 6,291,456 px, which is 6144 visual tokens at 32 px per token (patch 16, merge 2,
    # preprocessor_config.json). Every BHL scan is larger than that, so every page lands at
    # the cap: 6144 + the vendor's 12384 output tokens + the ~600-token prompt does not fit in
    # 18000, and the server would reject the request. 20480 fits; asserted below.
    "max_model_len": 20480,
    # house: the specialist peers' value. The vendor uses 0.85.
    "gpu_memory_utilization": 0.8,
    "extra_args": [
        # vendor: the processor pixel bounds from scripts/vllm.py.
        "--mm-processor-kwargs", '{"min_pixels": 3136, "max_pixels": 6291456}',
        # house OCR defaults, same as every peer. (The vendor enables prefix caching; OCR never
        # reuses an image, so the caches only cost memory.)
        "--limit-mm-per-prompt", '{"image": 1, "video": 0}',
        "--mm-processor-cache-gb", "0",
        "--no-enable-prefix-caching",
        # vendor: bfloat16.
        "--dtype", "bfloat16",
    ],
    # vendor: settings.MAX_OUTPUT_TOKENS and generate_vllm's defaults.
    "max_tokens": 12384,
    "temperature": 0.0,
    "top_p": 0.1,
}

# The chat template always opens an empty <think></think> block on the generation prompt, so
# thinking is off without chat_template_kwargs.

# ---- verbatim from chandra-ocr v0.2.0 chandra/prompts.py ----
ALLOWED_TAGS = [
    "math",
    "br",
    "i",
    "b",
    "u",
    "del",
    "sup",
    "sub",
    "table",
    "tr",
    "td",
    "p",
    "th",
    "div",
    "pre",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "ul",
    "ol",
    "li",
    "input",
    "a",
    "span",
    "img",
    "hr",
    "tbody",
    "small",
    "caption",
    "strong",
    "thead",
    "big",
    "code",
    "chem",
]
ALLOWED_ATTRIBUTES = [
    "class",
    "colspan",
    "rowspan",
    "display",
    "checked",
    "type",
    "border",
    "value",
    "style",
    "href",
    "alt",
    "align",
    "data-bbox",
    "data-label",
]

PROMPT_ENDING = f"""
Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()

OCR_LAYOUT_PROMPT = f"""
OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure
- Chemical-Block
- Diagram
- Bibliography
- Blank-Page

{PROMPT_ENDING}
""".strip()
# ---- end verbatim ----
assert hashlib.sha256(OCR_LAYOUT_PROMPT.encode()).hexdigest()[:16] == "025935f3e1de1acd", (
    "OCR_LAYOUT_PROMPT differs from chandra-ocr v0.2.0"
)

# ---- verbatim from chandra-ocr v0.2.0 chandra/model/util.py ----
def scale_to_fit(
    img: Image.Image,
    max_size: Tuple[int, int] = (3072, 2048),
    min_size: Tuple[int, int] = (1792, 28),
    grid_size: int = 28,
):
    resample_method = Image.Resampling.LANCZOS

    width, height = img.size

    # Check for empty or invalid image
    if width <= 0 or height <= 0:
        return img

    original_ar = width / height
    current_pixels = width * height
    max_pixels = max_size[0] * max_size[1]
    min_pixels = min_size[0] * min_size[1]

    # 1. Determine ideal float scale based on pixel bounds
    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5

    # 2. Convert dimensions to integer "grid blocks"
    w_blocks = max(1, round((width * scale) / grid_size))
    h_blocks = max(1, round((height * scale) / grid_size))

    # 3. Refinement Loop: Ensure we are under the max limit
    while (w_blocks * h_blocks * grid_size * grid_size) > max_pixels:
        if w_blocks == 1 and h_blocks == 1:
            break

        if w_blocks == 1:
            h_blocks -= 1
            continue
        if h_blocks == 1:
            w_blocks -= 1
            continue

        # Compare distortion: Which move preserves Aspect Ratio better?
        ar_w_loss = abs(((w_blocks - 1) / h_blocks) - original_ar)
        ar_h_loss = abs((w_blocks / (h_blocks - 1)) - original_ar)

        if ar_w_loss < ar_h_loss:
            w_blocks -= 1
        else:
            h_blocks -= 1

    # 4. Calculate final pixel dimensions
    new_width = w_blocks * grid_size
    new_height = h_blocks * grid_size

    # Return original if no changes were needed
    if (new_width, new_height) == (width, height):
        return img

    return img.resize((new_width, new_height), resample=resample_method)
# ---- end verbatim ----

PIXELS_PER_TOKEN = 32 * 32  # patch_size 16 x merge_size 2
MAX_PIXELS = 3072 * 2048
PROMPT_TOKENS_BOUND = 1024  # the ~2.2k-character prompt plus the chat template, generously
assert MAX_PIXELS // PIXELS_PER_TOKEN + SERVING["max_tokens"] + PROMPT_TOKENS_BOUND <= SERVING["max_model_len"], (
    "context math: visual tokens at the scale_to_fit cap + max_tokens + prompt exceed max_model_len"
)


def to_pil(value):
    """One dataset image cell -> a PIL image (decoded image, {"bytes","path"} dict or bytes)."""
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value)))
    raise ValueError(f"unsupported image value: {type(value)}")


def encode_png(value) -> str:
    """RGB-convert, scale_to_fit (vendor), base64 PNG (vendor: image_to_base64)."""
    img = scale_to_fit(to_pil(value).convert("RGB"))
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
    # REQUIRED, with no default: a forgotten --output must not resume into another run's output.
    ap.add_argument("--output", required=True,
                    help="output prefix, e.g. hf://buckets/<owner>/<bucket>/<run>/<model>/")
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
            # vendor: image part first, then the prompt text.
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": OCR_LAYOUT_PROMPT},
            ]}],
            "temperature": SERVING["temperature"],
            "top_p": SERVING["top_p"],
            "max_tokens": args.max_tokens,
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
    print("PORT chandra-ocr2 " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
