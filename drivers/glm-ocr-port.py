# /// script
# requires-python = ">=3.11"
# dependencies = ["saturate[hf]>=0.1.1", "pillow"]
# ///
"""zai-org/GLM-OCR — saturate port of uv-scripts/ocr/glm-ocr.py (port-matrix row).

Run:
    hf jobs uv run --image vllm/vllm-openai:v0.26.0 --flavor a10g-small \
      --secrets HF_TOKEN glm-ocr-port.py -- \
      --input-dataset <benchmark> --id-column PageID --limit 20

Input is a Hub dataset with an image column; `--id-column PageID` keys the output rows
by the benchmark's page id, which is what scoring joins ground truth on. Fan out over
several jobs with `--shard i/n` (strided assignment, one job per rank; parquet parts
and completion markers are per-rank, so they land in one output repo).

0.9B CogViT + GLM-0.5B decoder, MIT, 94.62 OmniDocBench V1.5. Card's document-parsing
prompt "Text Recognition:" verbatim, image BEFORE text (the card's own message order).

Two deviations from the recipe, both deliberate. (1) The recipe passes
trust_remote_code=True; the repo ships no auto_map and GlmOcrForConditionalGeneration
is in the vLLM registry, so --trust-remote-code is dropped (it would only widen the
attack surface for nothing). (2) The recipe's "[OCR ERROR]" fallback string is gone —
failures become durable saturate error rows instead.

`parse` stores the completion VERBATIM in `raw` and transforms nothing: the per-model
post-processing that turns raw into the scored `markdown` column — including the raise
on empty output, so a silently-blank page is an error row rather than an empty markdown
cell — lives in bhl-ocr-eval/runners/normalize_outputs.py, so a rule can be corrected
without paying for GPUs again. The no-`choices` guard below stays here: a malformed
response envelope is not text post-processing, and the normalizer never sees it.

Currency receipt: the card pins vLLM NIGHTLY + transformers from git, because
GlmOcrForConditionalGeneration was not in a release when the card was written. It is in
v0.26.0 (released 2026-07-27), which is what this board pins everywhere — see the image
note in SERVING. Following the card would have made this the only row whose image cannot
be reproduced, since `:nightly` moves daily. transformers comes from the image, not the
PEP 723 deps, so a stale image is still the likely failure mode (GLM-OCR needs
transformers>=5.1) — which is an argument for pinning a known tag, not against it.

Context math: native ctx is 131072 — booting uncapped on 24GB dies on the KV profile,
so max_model_len is capped explicitly and the assert below checks that a worst-case
image (12288 tokens at the processor's own pixel ceiling) plus max_tokens still fits.
"""
import argparse
import base64
import io

SERVING = {
    # Per-value provenance:
    # - model / prompt / image-before-text: card-verbatim (README "Prompt Limited"
    #   section + the Transformers example's message order).
    # - image "vllm/vllm-openai:v0.26.0": BOARD-WIDE PIN, a deliberate deviation from the
    #   card, which pins nightly wheels + `pip install git+.../transformers.git` in its
    #   vLLM section. The card was written when GlmOcrForConditionalGeneration was not yet
    #   in a release; it is registered in v0.26.0 (released 2026-07-27), confirmed by
    #   preflight against that tag. Following the card here would make this the one row on
    #   the board whose image cannot be reproduced — `:nightly` moves daily — on a board
    #   whose claim is a pinned image, a pinned model revision and a pinned script commit.
    #   Every model here runs the same pinned vLLM so that differences between rows are the
    #   models, not thirteen cards' advice from thirteen different months.
    # - max_model_len 32768: HOUSE choice. The card sets none and native ctx is
    #   131072 (config.json max_position_embeddings) — never boot uncapped on 24GB.
    #   32768 matches the previous full benchmark run of this model; a prior spike
    #   receipted 16384 at 4,776 pages/hour on a10g, but 16384 cannot hold a
    #   worst-case image (12288 tok) + 8192 output, so the larger cap is kept.
    # - gpu_memory_utilization 0.8 + the three extra_args: HOUSE OCR defaults
    #   (inherited from the recipe's LLM kwargs / the 07-16 sweep). OCR never reuses
    #   an image, so prefix + processor caches only cost memory.
    #   The card's serve line also carries `--allowed-local-media-path /`; omitted
    #   because we send base64 data URIs, never local file paths.
    # - max_tokens 8192: card-verbatim (the Transformers example's max_new_tokens=8192).
    #   Also what the previous benchmark run used. The GLM-OCR SDK uses 16384 — noted,
    #   not followed, since the card's own example is the narrower documented value.
    # - temperature 0.01 / top_p 1e-05 / repetition_penalty 1.1: INHERITED from the
    #   uv-scripts recipe, which sourced them from the authors' SDK
    #   (github.com/zai-org/GLM-OCR, glmocr/config.py PageLoaderConfig), not from the
    #   card. The card documents no sampling params at all; the shipped
    #   generation_config.json sets do_sample=false, i.e. greedy. 0.01/1e-05 is
    #   numerically greedy too, so there is no real contradiction — these are kept
    #   over a bare temperature 0.0 to stay comparable with the recorded prior run.
    # - max_pixels 9633792: author-shipped, but from preprocessor_config.json
    #   ("size.longest_edge"), NOT from card prose. It is the processor's own ceiling,
    #   so clamping client-side is the same clamp the server would apply, moved
    #   forward to shrink the payload. The recipe defaults to no cap.
    "model": "zai-org/GLM-OCR",
    "image": "vllm/vllm-openai:v0.26.0",
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.8,
    "extra_args": [
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
        "--limit-mm-per-prompt", '{"image": 1}',
    ],
    "max_tokens": 8192,
    "temperature": 0.01,
    "top_p": 0.00001,
    "repetition_penalty": 1.1,
    "max_pixels": 9633792,
}

# patch_size 14 * merge_size 2 => one visual token per 28x28 px (preprocessor_config.json).
PIXELS_PER_TOKEN = 28 * 28
MAX_IMAGE_TOKENS = SERVING["max_pixels"] // PIXELS_PER_TOKEN  # 12288 at the processor ceiling

assert MAX_IMAGE_TOKENS + SERVING["max_tokens"] < SERVING["max_model_len"], (
    "context math: a worst-case image plus max_tokens must fit inside max_model_len, "
    "or every full-page request 400s"
)

PROMPT = "Text Recognition:"


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


def encode_png(value, max_pixels: int) -> str:
    """RGB-convert, downscale to the processor's pixel ceiling if needed, base64 PNG.

    PNG (not JPEG) to match the recipe — no lossy artefacts on top of already-degraded
    scans. Never upscales.
    """
    from PIL import Image

    img = to_pil(value).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
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
        b64 = encode_png(row[args.image_column], SERVING["max_pixels"])
        return {
            "model": SERVING["model"],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
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
    print("PORT glm-ocr " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
