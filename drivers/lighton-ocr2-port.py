# /// script
# requires-python = ">=3.11"
# dependencies = ["saturate[hf]>=0.1.1", "pillow"]
# ///
"""lightonai/LightOnOCR-2-1B — saturate port of uv-scripts/ocr/lighton-ocr2.py (port-matrix row).

Run:
    hf jobs uv run --image vllm/vllm-openai:v0.26.0 --flavor a10g-small \
      --secrets HF_TOKEN lighton-ocr2-port.py -- \
      --input-dataset <benchmark> --id-column PageID --limit 20

Input is a Hub dataset with an image column, not a bucket glob. `--id-column PageID` is
what makes the output joinable: scoring joins OCR rows to ground truth on PageID, so the
row id must BE the PageID (the default `index` ids are only stable per revision — pin
`--revision` if you use them). Fan out over several jobs with `--shard i/n`; parquet parts
and completion markers are per-rank, so the ranks can share one output repo.

Request shape is the card's verbatim: the message is the image ONLY, with no text prompt —
LightOnOCR-2's trained format, and the one thing that changed from v1 (which used an empty
text prefix). Images are re-encoded client-side as PNG with the longest side forced to
exactly 1540 px, the resolution the card asks for ("Render PDFs at 200 DPI to images using
a target longest dimension of 1540px") and the same clamp `processor_config.json` declares
(`image_processor.size.longest_edge = 1540`).

`--trust-remote-code` is NOT passed. The repo ships no `auto_map` and no custom modelling
code — `LightOnOCRForConditionalGeneration` is registered in vLLM itself — so the flag would
buy nothing. The offline recipe (`lighton-ocr2.py`) does pass `trust_remote_code=True` to
`LLM(...)`; that is an inherited flag, not a known configuration, and it is dropped here for
the same reason the other drivers drop it.

`parse` stores the completion VERBATIM in `raw` and transforms nothing — not even the
`.strip()` the upstream driver applied. The per-model post-processing that turns raw into
the scored `markdown` column lives in bhl-ocr-eval/runners/normalize_outputs.py, so a rule
can be corrected without paying for GPUs again. For this model that chain is a strip and
nothing else; the large CER-reading (0.1063) vs CER-diplomatic (0.3014) split this model
shows on the board is a FINDING about its formatting and orthography, not something for a
driver or a normalizer to paper over.
"""
import argparse
import base64
import io

SERVING = {
    # Per-value provenance. "card" = the model card's own vllm serve / request example;
    # "recipe" = inherited from uv-scripts/ocr/lighton-ocr2.py; "house" = our choice.
    "model": "lightonai/LightOnOCR-2-1B",
    # BOARD-WIDE PIN, overriding the recipe. The port matrix recorded v0.22.1 for this row,
    # but that was simply the tag current when the recipe was written, not a hard requirement:
    # LightOnOCRForConditionalGeneration is registered in v0.26.0, which every other model on
    # this board runs. One pinned vLLM across all rows means a difference between rows is the
    # model rather than the engine. If the engine ever fails at model-registration rather than
    # at load, this pin is the first thing to raise, not the request shape.
    "image": "vllm/vllm-openai:v0.26.0",
    # house, and a deliberate CHANGE from the upstream saturate driver's 8192. The card gives
    # no max_model_len at all; 16384 is the model's own native context
    # (config.json text_config.max_position_embeddings), so this is the ceiling, not an
    # extension. 8192 does not survive the context assert below under the pessimistic
    # image-token bound, and the recipe records the same failure empirically ("with
    # --max-tokens 4096 output that overflows the old 8192 default at admission and vLLM
    # rejects the request") — which is why the recipe itself now defaults to 16384. At 1B
    # params the extra KV allocation is cheap; a 400-storm is not.
    "max_model_len": 16384,
    # recipe: lighton-ocr2.py's --gpu-memory-utilization default, and the house value the
    # other drivers in this directory use. The card's serve command sets no value (vLLM's own
    # default is 0.9).
    "gpu_memory_utilization": 0.8,
    # card, verbatim: the three flags in the card's `vllm serve` command. OCR never reuses an
    # image, so the two caches only cost memory.
    "extra_args": [
        "--limit-mm-per-prompt", '{"image": 1}',
        "--mm-processor-cache-gb", "0",
        "--no-enable-prefix-caching",
    ],
    # card, verbatim: the sampling block in the card's example request. Note this is the one
    # driver here that is NOT near-greedy — the card asks for temperature 0.2 / top_p 0.9, and
    # a house override to 0.0 would be benchmarking our settings, not the model's.
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.9,
}

# Image-token budget, derived from the model's own processor_config.json rather than guessed:
# patch_size 14, spatial_merge_size 2, and an image_processor clamp of longest_edge 1540.
PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
TARGET_LONGEST = 1540


def image_tokens(width: int, height: int, *, merge: int = SPATIAL_MERGE_SIZE) -> int:
    """Pixtral/Mistral3 image-token count for one page: merged patches plus a row-break token.

    `config.json` is `model_type: mistral3` with a Pixtral vision tower, whose token count is
    one token per (merged) patch plus one `[IMG_BREAK]`/`[IMG_END]` token per patch row.
    """
    cols = width // (PATCH_SIZE * merge)
    rows = height // (PATCH_SIZE * merge)
    return (cols + 1) * rows


# Worst case is a SQUARE page at the clamp — the largest area 1540 px on the longest side can
# cover. `merge=1` is the pessimistic bound: it assumes the spatial merge is NOT applied, which
# is the only reading that explains the recipe's empirically observed overflow of an 8192
# budget (the merged count for the same page is ~3k, which would have fit). Sizing the context
# for the bound that was actually observed is the point of the assert.
IMAGE_TOKENS_WORST_CASE = image_tokens(TARGET_LONGEST, TARGET_LONGEST, merge=1)
assert IMAGE_TOKENS_WORST_CASE + SERVING["max_tokens"] <= SERVING["max_model_len"], (
    f"context math: {IMAGE_TOKENS_WORST_CASE} image tokens (a {TARGET_LONGEST}px square page) "
    f"+ {SERVING['max_tokens']} output tokens exceeds max_model_len {SERVING['max_model_len']}; "
    "vLLM rejects such a request at admission, so every page would 400"
)


def to_pil(value):
    """One dataset image cell -> a PIL image.

    A `datasets` image column hands over a decoded `PIL.Image`, an undecoded
    `{"bytes", "path"}` dict, or raw bytes depending on how the dataset stores it, so all
    three shapes land here rather than one being assumed.
    """
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value)))
    raise ValueError(f"unsupported image value: {type(value)}")


def encode_png(value) -> str:
    """RGB-convert, force the longest side to exactly TARGET_LONGEST, return base64 PNG.

    DIVERGENCE from the offline recipe, preserved from the upstream saturate driver this is a
    port of: `lighton-ocr2.py`'s `resize_image_to_target` only ever DOWNSCALES (it returns
    early when the page is already smaller than 1540), while this forces the longest side to
    1540 in both directions. It matters only for pages under 1540 px, and the board row for
    this model was produced by the forcing version — changing it here would move a published
    score without a run to justify it.
    """
    from PIL import Image

    img = to_pil(value).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest != TARGET_LONGEST:
        scale = TARGET_LONGEST / longest
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
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
            # Image ONLY — no text part. LightOnOCR-2's trained format per the card.
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
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
        # `or ""` is envelope handling, not text cleaning: a null `content` field is a
        # malformed response, and the raw column holds strings (an empty one is valid).
        return {"raw": choices[0]["message"].get("content") or "",
                "model": SERVING["model"],
                "finish_reason": choices[0].get("finish_reason"),
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
    print("PORT lighton-ocr2 " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
