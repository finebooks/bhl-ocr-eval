# /// script
# requires-python = ">=3.11"
# dependencies = ["saturate[hf]>=0.1.1", "pillow"]
# ///
"""ds4sd/SmolDocling-256M-preview — saturate port of uv-scripts/ocr/smoldocling-ocr.py (port-matrix row).

Run:
    hf jobs uv run --image vllm/vllm-openai:v0.26.0 --flavor a10g-small \
      --secrets HF_TOKEN smoldocling-port.py -- \
      --input-dataset <benchmark> --id-column PageID --limit 20

Input is a Hub dataset with an image column; `--id-column PageID` keys the output rows
by the benchmark's page id, which is what scoring joins ground truth on. Fan out over
several jobs with `--shard i/n` (strided assignment, one job per rank; parquet parts
and completion markers are per-rank, so they land in one output repo).

The benchmark's low-end anchor: 256M params, ungated, native ctx only 8192 (so the
context assert below is a real constraint, not a formality).

Two things differ from every other port in this matrix:

- **Output is DocTags, not markdown.** The model emits a structured tag format
  (`<doctag><text><loc_..>..`), which scores near zero as-is on a markdown/plain-text
  column. `parse` still stores it VERBATIM: the DocTags -> markdown conversion (the
  card's docling-core path, plus the guard that turns "non-empty DocTags in, empty
  markdown out" into a durable error row rather than a silent blank) lives in
  bhl-ocr-eval/runners/normalize_outputs.py, so the export options can be changed —
  and re-exported to html/json — without re-inference. This driver therefore needs no
  docling-core dependency.
- **Message order is load-bearing.** The repo chat template branches on
  `content[0]['type'] == 'image'` (image first -> "User:", text first -> "User: "), so
  the image part must come first to reproduce the card's prompt string exactly.

The completion is written to both `raw` (the column every driver on this board writes)
and `doctags` (this model's own name for the same bytes, kept for readers that expect
it).
"""
import argparse
import base64
import io

SERVING = {
    # Per-value provenance:
    # - model / temperature 0.0: card-verbatim (the card's vLLM snippet is
    #   SamplingParams(temperature=0.0, max_tokens=8192)).
    # - max_model_len 8192: recipe-inherited AND the hard native ceiling
    #   (config.json max_position_embeddings = 8192). Cannot be raised without
    #   RoPE scaling; this is the whole reason max_tokens had to move (below).
    # - max_tokens 6144: HOUSE CHOICE, deliberately NOT the card's 8192. The card
    #   and the recipe both set max_tokens == max_model_len == 8192, which leaves
    #   zero room for the image and 400s every request the moment an image is
    #   attached. 6144 leaves ~2k for the image + prompt (budget below).
    # - gpu_memory_utilization 0.8: recipe-inherited (smoldocling-ocr.py LLM kwarg).
    # - extra_args: house OCR flags (OCR never reuses an image, so prefix/processor
    #   caches only cost memory). NOTE: the recipe passes trust_remote_code=True but
    #   this repo ships no auto_map/custom code — the arch is Idefics3ForConditional-
    #   Generation, registered natively in vLLM — so --trust-remote-code is dropped.
    # - max_pixels_longest_edge 2048: house choice, payload-shrink only. It is the
    #   processor's own size.longest_edge (preprocessor_config.json), so the model
    #   sees the same pixels either way; doing it client-side just stops full-res
    #   scans crossing the wire.
    "model": "ds4sd/SmolDocling-256M-preview",
    "image": "vllm/vllm-openai:v0.26.0",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8,
    "extra_args": [
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
        "--limit-mm-per-prompt", '{"image": 1}',
    ],
    "max_tokens": 6144,
    "temperature": 0.0,
    "max_pixels_longest_edge": 2048,
}

# Idefics3 image budget: do_image_splitting with size.longest_edge 2048 and
# max_image_size.longest_edge 512 gives at most 4x4 tiles + 1 global = 17 crops;
# scale_factor 4 => 64 tokens per crop => 1088, plus the <row_i_col_j>/<global-img>
# separators and the instruction. Round up to 1600.
IMAGE_TOKEN_BUDGET = 1600
assert SERVING["max_tokens"] + IMAGE_TOKEN_BUDGET <= SERVING["max_model_len"], (
    "context math: image tokens + max_tokens must fit 8192 (native ctx, not raisable) "
    "or every request 400s — this is why max_tokens is 6144, not the card's 8192"
)

# Card-verbatim, from the transformers/chat example — which is the path a chat-completions
# request takes. The card's hand-rolled vLLM snippet writes it "Convert page to Docling."
# instead; --prompt flips to that (it is also the recipe's default).
PROMPT = "Convert this page to docling."


def to_pil(value):
    """One dataset image cell -> a PIL image.

    A `datasets` image column hands over a decoded `PIL.Image`, an undecoded
    `{"bytes", "path"}` dict, or raw bytes depending on how the dataset stores it, so
    all three shapes land here rather than one being assumed.
    """
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value)))
    raise ValueError(f"unsupported image value: {type(value)}")


def encode_jpeg(value, longest_edge: int) -> str:
    """RGB-convert, downscale to the processor's longest_edge, return base64 JPEG q95."""
    from PIL import Image

    img = to_pil(value).convert("RGB")
    w, h = img.size
    if max(w, h) > longest_edge:
        scale = longest_edge / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
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
    # REQUIRED, with no default. A forgotten --output must not resume into some other
    # run's output: saturate anti-joins on id, so a stray default would blend two
    # configurations into one complete, fingerprint-passing table. (The old default
    # also pointed at a private scratch repo, which is meaningless outside this laptop.)
    ap.add_argument("--output", required=True,
                    help="output prefix, e.g. hf://buckets/<owner>/<bucket>/<run>/<model>/")
    ap.add_argument("--prompt", default=PROMPT)
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
        b64 = encode_jpeg(row[args.image_column], SERVING["max_pixels_longest_edge"])
        return {
            "model": SERVING["model"],
            # image FIRST: the chat template branches on content[0]['type'] == 'image'
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": args.prompt},
            ]}],
            "temperature": SERVING["temperature"],
            "max_tokens": args.max_tokens,
            # DocTags (<doctag>, <loc_*>) are added special tokens; the card decodes with
            # skip_special_tokens=False. vLLM's chat API defaults to True, which strips them
            # and hands docling-core streams it silently converts to empty/partial markdown.
            "skip_special_tokens": False,
        }

    def parse(row, body):
        choice = body["choices"][0]
        content = choice["message"]["content"]
        usage = body.get("usage") or {}
        # `doctags` is the same verbatim completion under this model's own name for it;
        # it is kept so the column survives for readers that join on it.
        return {"raw": content,
                "doctags": content,
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
    print("PORT smoldocling " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
