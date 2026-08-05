# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]>=0.1.1",
#     "pillow>=10",
# ]
# ///
"""
Convert document images to markdown using Baidu Unlimited-OCR via saturate.

Production driver for the BHL OCR benchmark, promoted from `unlimited-ocr-probe.py`
(10-page currency check) to a full-corpus run. `baidu/Unlimited-OCR` (3.3B MoE, MIT,
a DeepSeek-OCR descendant) has never been scored on this benchmark and is currently
the #3 trending image-text-to-text model on the Hub (~2.6M downloads), so this row
gets read.

Run on HF Jobs (the script starts `vllm serve` itself; --image provides the binary):

  hf jobs uv run --detach --flavor a10g-small -s HF_TOKEN --timeout 8h \\
      --image vllm/vllm-openai:v0.26.0 \\
      unlimited-ocr-port.py -- --input-dataset <benchmark> --id-column PageID \\
        --output hf://buckets/<owner>/<bucket>/<prefix>/

Fan out over several jobs with `--shard i/n` (strided assignment, one job per rank;
parquet parts and completion markers are per-rank, so they land in one output repo).

Output layout: the output repo holds `data/part-*.parquet` with rows
`{id, raw, grounding, model, prompt_tokens, completion_tokens, finish_reason,
error}` keyed by the input row id (`--id-column`, or `<split>-<index>` by default).
Read it with `datasets.load_dataset(<output>, data_dir="data")` or
`saturate.read_output`; join back to the input on id. Run metadata lands in
`data/completions/`.

Image pin (the probe's finding, banked here): the uv-scripts recipe and the model
card pin Baidu's dedicated `vllm/vllm-openai:unlimited-ocr` image. That pin is
OBSOLETE — `UnlimitedOCRForCausalLM` is in the vLLM registry at v0.26.0/main, and
the probe (job `6a6a35b1`, 10/10 pages, 0 errors) confirmed the model serves on
plain `vllm/vllm-openai:latest` with its grounding markup intact. This driver uses
the plain image.

Grounding markup (the decision, now downstream): the model emits layout-grounded
output — each block prefixed with `<|det|>type [bbox]<|/det|>`, DeepSeek-style
`<|ref|>…<|/ref|>` spans around text. The benchmark scores TRANSCRIBED TEXT, and the
markup is not text: leaving it in inflates every string metric and turns coordinate
digits into fake characters. So the scored `markdown` column is always stripped — but
the stripping happens in bhl-ocr-eval/runners/normalize_outputs.py, not here. `parse`
stores the grounded completion VERBATIM in `raw` (edge specials, markup and all),
which is the whole point: the strip is a chain of judgement calls — the card's own
OmniDocBench `remove_det`, a sweep for the double-bracket and inline `<|ref|>` shapes
its line-anchored regex cannot see, and a raise on anything `<|…|>`-shaped that
survives — and every one of them can now be corrected and re-run without paying for
the GPUs again. `grounding` is kept as a second copy of the same bytes for readers
that expect it; `--no-grounding-column` drops it.

Family-c deviations from the offline recipe, unchanged from the probe: the raw
`"<image>document parsing."` prompt becomes an image_url + text pair and the chat
path inserts the image token itself (the model ships no chat template; vLLM's
fallback is what the probe validated), and `skip_special_tokens=False` rides in the
request body so the grounding tags survive detokenization.

The SERVING dict below is the per-model tuning prior (serve flags + client sampling
+ context math). Agents can `ast.literal_eval` it without running the script; the
script itself consumes it, so it cannot drift from reality.
"""

import argparse
import base64
import io
import sys

# Serving starting values for baidu/Unlimited-OCR. Per-value provenance:
# - image `vllm/vllm-openai:latest`: probe-proven (see docstring). NOT the card's
#   `vllm/vllm-openai:unlimited-ocr` pin, which predates registry support.
# - max_model_len 32768: recipe-inherited AND card-verbatim (`max_length=32768`);
#   it is also the model's native `max_position_embeddings`, so no cap is needed.
# - gpu_memory_utilization 0.8 + the cache/mm flags: recipe-inherited house OCR
#   defaults (OCR never reuses an image, so prefix/processor caches only cost memory).
# - --trust-remote-code: required, the repo ships `auto_map` custom code.
# - logits_processor + ngram_size 35 / window_size 128: card-verbatim
#   (`no_repeat_ngram_size=35, ngram_window=128` for SINGLE images; the card's 1024
#   window is the multi-page setting and does not apply here). vLLM's own
#   unlimited_ocr module documents the same pair via `SamplingParams.extra_args`.
#   Server-mode wiring = boot flag + per-request `vllm_xargs` (vLLM custom-logitsproc
#   docs). NOTE: this is the one value NOT yet exercised on Jobs — the probe ran
#   without it. It fails loudly at boot if wrong, so smoke it with `--limit 10`
#   before a corpus run; `--no-anti-repeat` falls back to the exact probe config.
#   Kept ON by default because coordinate-token loops are a documented failure mode
#   of this family on dense historical scans, and each loop burns a full max_tokens.
# - max_tokens 8192: recipe-inherited (the card bounds total length, not output).
# - temperature 0.0: card-verbatim.
# - no client-side resize (house choice, deliberate): single-image requests always
#   take the gundam crop path (vLLM `UnlimitedOCRProcessingInfo`, base_size=1024 /
#   image_size=640 / max 32 crops), and the crop grid is chosen FROM the input
#   pixel size — downscaling would buy payload and cost tiles, i.e. exactly the
#   resolution that makes tiny historical type legible. Images are only normalised
#   to RGB when they are not already (see `encode_image`).
SERVING = {
    "model": "baidu/Unlimited-OCR",
    "image": "vllm/vllm-openai:v0.26.0",
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.8,
    "serve_args": [
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
        "--limit-mm-per-prompt", '{"image": 1}',
    ],
    "logits_processor": "vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor",
    "ngram_size": 35,
    "window_size": 128,
    "max_tokens": 8192,
    "temperature": 0.0,
}

# Worst-case image tokens for a single (always-cropped) request, computed from
# vLLM's UnlimitedOCRProcessingInfo.get_num_image_tokens at the widest grid it can
# select (4x8 = 32 crops): global 16*(16+1)=272, local (8*10)*(4*10+1)=3280, +1.
MAX_IMAGE_TOKENS = 3553

assert SERVING["max_tokens"] + MAX_IMAGE_TOKENS < SERVING["max_model_len"], (
    "context math: image tokens + max_tokens must fit max_model_len "
    "(input + output <= max_model_len, or every request 400s)"
)

# Card's `<image>document parsing.` minus the image token: over /chat/completions
# the image rides as its own content part and the template inserts the token.
PROMPT = "document parsing."

# Formats whose bytes go to the server untouched when the image is already RGB.
PASSTHROUGH_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg"}


def encode_image(value) -> tuple[str, str]:
    """Return `(mime, base64)` for one dataset image cell, at full resolution.

    Already-RGB PNG/JPEG bytes are passed through byte-for-byte — the crop grid is
    chosen from the input pixels, so re-encoding buys nothing and risks losing them.
    Everything else (grayscale, bitonal, palette, CMYK, RGBA — the common shapes for
    library scans) is converted to RGB and re-encoded as lossless PNG: an
    unconverted non-RGB scan is what made the June serving path 500 on "loading
    multimodal data" while the offline recipe, which converts, handled it fine.
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
    ap = argparse.ArgumentParser(description="Unlimited-OCR batch OCR via saturate")
    # Flags, not positionals, and --output takes a full URI. This driver was the exemplar the
    # other thirteen were converted against, and it ended up the only one still on positional
    # args writing to a hardcoded hf://datasets/ path — so it could not write to a bucket,
    # which is where this benchmark's run output goes.
    ap.add_argument("--input-dataset", required=True,
                    help="Input dataset repo id (rows with an image column)")
    ap.add_argument("--output", required=True,
                    help="Output URI, e.g. hf://buckets/<owner>/<bucket>/<prefix>/ or "
                         "hf://datasets/<repo>/data")
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
    ap.add_argument("--no-anti-repeat", action="store_true",
                    help="Drop the card's n-gram loop suppressor (falls back to the probe config)")
    ap.add_argument("--no-grounding-column", action="store_true",
                    help="Do not keep the second copy of the grounded output alongside `raw`")
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-admit rows whose only record is an error row")
    args = ap.parse_args()

    from saturate import Auto, Engine, dataset_rows, pump, shard_select

    anti_repeat = not args.no_anti_repeat
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
        body = {
            "model": SERVING["model"],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ]}],
            "temperature": SERVING["temperature"],
            "max_tokens": args.max_tokens,
            "skip_special_tokens": False,
        }
        if anti_repeat:
            body["vllm_xargs"] = {"ngram_size": SERVING["ngram_size"],
                                  "window_size": SERVING["window_size"]}
        return body

    def parse(row, body):
        # Envelope check, not text cleaning: a response with no choices or a null
        # `content` field never held a completion to store.
        choice = (body.get("choices") or [None])[0]
        if not choice or choice.get("message", {}).get("content") is None:
            raise ValueError(f"no content in response: {str(body)[:200]}")
        grounded = choice["message"]["content"]
        usage = body.get("usage") or {}
        out = {
            "raw": grounded,
            "model": SERVING["model"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            # "length" is this family's loop signature — kept so a truncated page is
            # auditable on the board instead of silently short.
            "finish_reason": choice.get("finish_reason"),
        }
        if not args.no_grounding_column:
            # Same bytes as `raw` now that nothing is stripped on the way in; kept
            # under its own name for readers that expect the grounded column.
            out["grounding"] = grounded
        return out

    extra = ["--max-model-len", str(SERVING["max_model_len"]),
             "--gpu-memory-utilization", str(SERVING["gpu_memory_utilization"]),
             *SERVING["serve_args"]]
    # The served checkpoint is pinned at boot, not read back from the Hub afterwards:
    # a post-hoc head read is not guaranteed to be what vLLM actually loaded. Placed
    # before any --logits_processors append, which must stay last.
    if args.model_revision:
        SERVING["model_revision"] = args.model_revision
        extra += ["--revision", args.model_revision]
    if anti_repeat:
        extra += ["--logits_processors", SERVING["logits_processor"]]

    with Engine(SERVING["model"], engine="vllm", extra_args=extra) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, args.output,
                     window=Auto(initial=4, target_waiting=4, max_limit=12, step=2),
                     shard=(rank, world),
                     retry_errors=args.retry_errors)

    print(f"{args.output} "
          f"({stats.rows_processed} ok, {stats.rows_failed} error rows)", file=sys.stderr)
    print("PORT unlimited-ocr " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
