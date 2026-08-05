# /// script
# requires-python = ">=3.10"
# dependencies = ["saturate[hf]>=0.1.1", "pillow"]
# ///
"""ATH-MaaS/OvisOCR2 — saturate port of uv-scripts/ocr/ovis-ocr2-server.py (port-matrix row).

Run:
    hf jobs uv run --image vllm/vllm-openai:v0.26.0 --flavor a10g-small \
      --secrets HF_TOKEN ovis-ocr2-port.py -- \
      --input-dataset <benchmark> --id-column PageID \
      --output hf://datasets/<org>/<repo>/data --limit 20

Input is a Hub dataset with an image column, not a bucket glob. `--id-column PageID`
is what makes the output joinable: scoring joins OCR rows to ground truth on PageID,
so the row id must BE the PageID (the default `index` ids are only stable per
revision — pin `--revision` if you use them). Fan out over several jobs with
`--shard i/n`; parquet parts and completion markers are per-rank, so the ranks can
share one output repo.

Request shape is the server sibling's verbatim: image downscaled client-side to
max_pixels (the same clamp the processor would apply), JPEG q95, image-then-text,
enable_thinking=False via chat_template_kwargs. min/max pixel bounds move to the
engine boot flag. Deviation noted: the card's clean_truncated_repeats trimmer is
omitted (no-op under 8000 chars; these pages are shorter).

`parse` stores the completion VERBATIM in `raw` — bbox image tags and all — and
transforms nothing: the per-model post-processing that turns raw into the scored
`markdown` column, here the drop of the card prompt's `<img src="images/bbox_…">`
placeholder blocks, lives in bhl-ocr-eval/runners/normalize_outputs.py, so a rule can
be corrected without paying for GPUs again.
"""
import argparse
import base64
import io
import math

SERVING = {
    # provenance: ovis-ocr2-server.py (SERVE_ARGS + request body) + model card
    "model": "ATH-MaaS/OvisOCR2",
    "image": "vllm/vllm-openai:v0.26.0",
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.8,
    "extra_args": [
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
        "--limit-mm-per-prompt", '{"image": 1}',
        "--mm-processor-kwargs", '{"images_kwargs": {"min_pixels": 200704, "max_pixels": 8294400}}',
    ],
    "max_tokens": 16384,
    "temperature": 0.0,
}
assert SERVING["max_tokens"] < SERVING["max_model_len"], "context math: max_tokens must leave input room"

MAX_PIXELS = 8294400

OCR_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    'represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, '
    "where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. Preserve the original text "
    "without translation or paraphrasing."
)


def open_image(value):
    """One dataset image cell -> PIL image.

    A dataset image column yields a PIL image, a {"bytes": ...} dict, or raw bytes
    depending on how the dataset stores it; all three land here. There is no
    pass-the-original-bytes-through path like some drivers have: the client-side
    max_pixels clamp and the JPEG q95 re-encode are part of reproducing the server
    sibling's request, so every image is re-encoded anyway.
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


def encode_jpeg(value) -> str:
    from PIL import Image

    img = open_image(value).convert("RGB")
    w, h = img.size
    if w * h > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (w * h))
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
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
        b64 = encode_jpeg(row[args.image_column])
        return {
            "model": SERVING["model"],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": OCR_PROMPT},
            ]}],
            "temperature": SERVING["temperature"],
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
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
    print("PORT ovis-ocr2 " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
