# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]>=0.1.1",
#     "pyarrow>=15",
#     "pillow>=10",
# ]
# ///
"""
Convert document images to markdown using baidu/Qianfan-OCR via saturate.

Production driver, promoted from `qianfan-ocr-probe.py` (10 pages, 0 errors).
The probe existed to answer one question — does this model still need the
`--hf-overrides '{"architectures": ["InternVLChatModel"]}'` remap the card
documents? It does not. This file is the full-corpus driver built on that
answer, with the serving configuration re-derived from the card and
config.json rather than inherited from the recipe.

Run on HF Jobs (the script starts `vllm serve` itself; --image supplies the
`vllm` binary):

  hf jobs uv run --detach --flavor a10g-small -s HF_TOKEN --timeout 8h \\
      --image vllm/vllm-openai:v0.26.0 qianfan-ocr-port.py \\
      --input-dataset <benchmark> --id-column PageID

Input is a Hub dataset with an image column; `--id-column PageID` keys the output rows
by the benchmark's page id, which is what scoring joins ground truth on. Fan out over
several jobs with `--shard i/n` (strided assignment, one job per rank; parquet parts
and completion markers are per-rank, so they land in one output repo).

Output layout: `{output}/part-*.parquet` with rows
`{id, raw, model, finish_reason, prompt_tokens, completion_tokens, error}`,
id = the `--id-column` value (or `<split>-<index>` by default, which is only stable
for a pinned `--revision`). `raw` is the model's
completion VERBATIM — `parse` transforms nothing. The per-model post-processing
that turns raw into the scored `markdown` column (the Layout-as-Thought strip,
which raises on an unterminated block, then the raise on empty output) lives in
bhl-ocr-eval/runners/normalize_outputs.py, so a rule can be corrected without
paying for GPUs again. The schema is
DECLARED (see OUTPUT_SCHEMA), not inferred, so every part is byte-compatible
whether it holds successes, error rows, or both. Re-running the same command
resumes: ids already durable are anti-joined out before any request is sent.

SERVING FINDINGS (the point of the probe — both flags on the card's serve line
are wrong for this checkpoint today):

- `--hf-overrides '{"architectures": ["InternVLChatModel"]}'` — OBSOLETE.
  The card's serve command predates vLLM registering the model natively; the
  remap forced it to load through the InternVL path. `QianfanOCRForConditionalGeneration`
  is now registered at vLLM main, and the probe served and decoded 10 pages
  with no override. Carrying it forward would silently benchmark the InternVL
  code path instead of the model's own.
- `--trust-remote-code` — NOT NEEDED. The repo ships no Python at all (no
  modeling_*.py, no configuration_*.py, no image_processing_*.py) and
  config.json has no `auto_map`. The processor stack is stock transformers:
  `processor_class: InternVLProcessor`, `image_processor_type: GotOcr2ImageProcessor`.
  There is no remote code for the flag to trust, so it is inert — the recipe
  passes it only because the card's serve line does. Dropped. This is a small
  thing, but the benchmark's claim is that its serving configuration is known
  rather than inherited, and an inert flag is an unexamined one.

Model: baidu/Qianfan-OCR (4.7B, Apache-2.0, Qianfan-ViT + Qwen3-4B)
- The card's exact document-parsing prompt, image BEFORE text (card ordering).
- Layout-as-Thought explicitly DISABLED. The card: "Enable thinking for
  heterogeneous pages with mixed element types (exam papers, technical reports,
  newspapers). Disable for homogeneous documents (single-column text, simple
  forms) for better results and lower latency." Scanned book pages are the
  homogeneous case. The chat template already defaults thinking off
  (`enable_thinking is defined and enable_thinking`); passing it explicitly
  pins the behaviour against a future template change rather than relying on
  a default. NOTE: the recipe's `--think` appends a literal "<think>" to the
  prompt text, which lands in the same place the template would put it — the
  mechanism differs, the tokens do not.
- No client-side resize. Unlike the Ovis/olmOCR ports, the card documents no
  pixel bounds: Qianfan-ViT is AnyResolution and preprocessor_config.json
  clamps internally at max_patches=12 (+ thumbnail) of 448x448. Downscaling
  here would be an undocumented intervention, so raw bytes go up as-is and
  the server does the tiling and the RGB conversion (`do_convert_rgb: true`).
  Pillow is a dependency only because a `datasets` image column may hand over
  an ALREADY-DECODED PIL image, which has to be re-encoded to get bytes on the
  wire at all; that path saves lossless PNG at the source resolution and in the
  source mode, so it still resizes nothing and converts nothing the server
  would not. Encoded cells (bytes / `{"bytes"}` dicts) never touch pillow —
  their bytes go up untouched, as before.

The SERVING dict below is the per-model tuning prior (serve flags + client
sampling + context math). Agents can `ast.literal_eval` it without running the
script; the script itself consumes it, so it cannot drift from reality.
"""

import argparse
import base64
import io
import sys

# Serving starting values for baidu/Qianfan-OCR. Per-value provenance:
# - model / image: the ungated checkpoint on the standard vLLM job image.
# - max_model_len 16384: recipe-inherited (qianfan-ocr.py). Native context is
#   32768 (text_config.max_position_embeddings) — the cap is deliberate, not a
#   limitation: the context math below shows the full 32768 buys nothing here,
#   and halving the KV profile is what keeps boot honest on a 24GB card.
# - gpu_memory_utilization 0.85: recipe-inherited. Card documents none.
# - serve_args: house OCR defaults. One image per prompt, and both caches off —
#   OCR never reuses an image or a prefix, so they only cost memory.
#   NOT PRESENT, deliberately: --trust-remote-code (inert, no remote code in
#   the repo) and --hf-overrides (the architecture is registered natively now).
#   Both are on the card's serve line; see the docstring for the evidence.
# - max_tokens 8192: recipe-inherited, and the only defensible reading of the
#   card, which shows 512 (a demo value, short of a dense page) and 16384 (the
#   thinking-mode budget, and we are not thinking). 8192 clears any single
#   scanned page while leaving real context headroom.
# - temperature 0.0: card-verbatim semantics (`do_sample=False`, greedy).
# - top_p 1.0: recipe-inherited. A no-op under greedy decoding; kept so the
#   request is identical to the recipe's SamplingParams rather than merely
#   equivalent.
# - max_image_tokens 4096: card-verbatim ("max 4,096 tokens per image"), used
#   as the conservative bound for the context assert. The shipped preprocessor
#   is tighter still — max_patches 12 + thumbnail = 13 tiles x 256 tok = 3,328
#   — so 4096 is the architecture ceiling, not this checkpoint's.
# Probe receipt (a10g-small, 10 pages): 0 errors, no override, no remote code.
SERVING = {
    "model": "baidu/Qianfan-OCR",
    "image": "vllm/vllm-openai:v0.26.0",
    "max_model_len": 16384,
    "gpu_memory_utilization": 0.85,
    "serve_args": [
        "--limit-mm-per-prompt", '{"image": 1}',
        "--mm-processor-cache-gb", "0",
        "--no-enable-prefix-caching",
    ],
    "max_tokens": 8192,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_image_tokens": 4096,
}
assert SERVING["max_image_tokens"] + SERVING["max_tokens"] < SERVING["max_model_len"], (
    "context math: worst-case image tokens + max_tokens must fit under "
    "max_model_len (input + output <= max_model_len, or every request 400s). "
    f"{SERVING['max_image_tokens']} + {SERVING['max_tokens']} vs {SERVING['max_model_len']}"
)

# Card-verbatim, from every Quick Start example.
PROMPT = "Parse this document to Markdown."

# Magic bytes -> media type. The probe hard-coded image/png because its source
# was a *.png glob; over a whole corpus a mislabelled data URI is a decode failure
# at the server with no clue as to why.
MAGIC = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


def sniff_mime(raw: bytes) -> str:
    for prefix, mime in MAGIC:
        if raw.startswith(prefix):
            return mime
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError(f"unrecognised image magic bytes: {raw[:8]!r}")


# Modes PIL's PNG writer can hold. Anything outside it (CMYK, YCbCr, F) has to be
# converted; everything else — including the mode "1" bitonal that library scans
# come in as — is written as-is, so no page gains bytes or loses tone on the way out.
PNG_SAFE_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA"})


def data_uri(value) -> str:
    """base64 data URI for one dataset image cell.

    A `datasets` image column hands over an undecoded `{"bytes", "path"}` dict, raw
    bytes, or an already-decoded `PIL.Image` depending on how the dataset stores it,
    so all three shapes land here rather than one being assumed. Encoded bytes go up
    exactly as stored, with the media type sniffed from their magic bytes — the
    no-client-side-intervention path this driver is built around. A decoded PIL cell
    has no stored bytes left to send, so it is re-encoded as lossless PNG at the
    source resolution and mode: same pixels, no resize.
    """
    raw = None
    if isinstance(value, dict) and value.get("bytes"):
        raw = value["bytes"]
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    if raw is not None:
        return f"data:{sniff_mime(raw)};base64," + base64.b64encode(raw).decode()

    from PIL import Image

    if not isinstance(value, Image.Image):
        raise ValueError(f"unsupported image value: {type(value)}")
    img = value if value.mode in PNG_SAFE_MODES else value.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_shard(spec: str) -> tuple[int, int]:
    rank, _, world = spec.partition("/")
    rank, world = int(rank), int(world or 1)
    if world < 1 or not 0 <= rank < world:
        raise argparse.ArgumentTypeError(f"--shard must be rank/world with 0 <= rank < world, got {spec!r}")
    return rank, world


def main():
    ap = argparse.ArgumentParser(description="Qianfan-OCR full-corpus OCR via saturate")
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
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap the run (default: the whole dataset). Counts rows READ "
                         "from the input, including ones a resume then skips.")
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="RANK/WORLD",
                    help="Strided fan-out across jobs, e.g. 2/8 (default: 0/1)")
    ap.add_argument("--max-tokens", type=int, default=SERVING["max_tokens"])
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-admit rows whose only record is an error row")
    args = ap.parse_args()

    import pyarrow as pa
    from saturate import Auto, Engine, dataset_rows, existing_ids, pump, shard_select

    rank, world = args.shard

    # Declared, immutable output schema. Inferred schemas are set by whatever
    # the first flushed part happens to contain, which over a long run is a
    # coin toss between a success row and an error row; pyarrow.dataset then
    # reads the whole output through that first fragment's schema.
    schema = pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("raw", pa.string()),
        pa.field("model", pa.string()),
        pa.field("finish_reason", pa.string()),
        pa.field("prompt_tokens", pa.int64()),
        pa.field("completion_tokens", pa.int64()),
        pa.field("error", pa.string()),
    ])

    # Resume receipt, read BEFORE the engine boots: `pump` anti-joins on the same
    # set anyway, but knowing the run is already complete is worth more before a
    # GPU is paid for than in the closing stats. retry_errors must ride along or the
    # count would contradict the rows the run then re-admits. (The bucket path also
    # used this set to skip fetches; `dataset_rows` has no such hook — the reader is
    # a streaming pass over the dataset, so pump's anti-join is the only filter.)
    done = existing_ids(args.output, retry_errors=args.retry_errors)
    if done:
        print(f"[resume] {len(done)} ids already durable", file=sys.stderr)

    window = Auto(initial=4, target_waiting=4, max_limit=12, step=2)
    rows = dataset_rows(
        args.input_dataset, config=args.config, split=args.split,
        columns=[args.image_column], ids=args.id_column or "index",
        revision=args.revision, limit=args.limit,
    )
    if world > 1:
        rows = shard_select(rows, rank=rank, world=world)

    def to_request(row):
        return {
            "model": SERVING["model"],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri(row[args.image_column])}},
                {"type": "text", "text": PROMPT},
            ]}],
            "temperature": SERVING["temperature"],
            "top_p": SERVING["top_p"],
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def parse(row, body):
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        return {
            # `or ""` is envelope handling, not text cleaning: a null `content` field is
            # a malformed response, and raw holds strings (an empty one is valid).
            "raw": choice["message"].get("content") or "",
            "model": SERVING["model"],
            # Recorded, not raised: finish_reason == "length" means the page hit
            # max_tokens. That is a real property of this serving config and
            # belongs in the results, but the partial markdown is still the best
            # transcription we have — throwing it away would be worse.
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

    extra = ["--max-model-len", str(SERVING["max_model_len"]),
             "--gpu-memory-utilization", str(SERVING["gpu_memory_utilization"]),
             *SERVING["serve_args"]]
    # The served checkpoint is pinned at boot, not read back from the Hub afterwards:
    # a post-hoc head read is not guaranteed to be what vLLM actually loaded. Placed
    # before any --logits_processors append, which must stay last.
    if args.model_revision:
        SERVING["model_revision"] = args.model_revision
        extra += ["--revision", args.model_revision]
    with Engine(SERVING["model"], engine="vllm", extra_args=extra) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, args.output,
                     window=window, shard=(rank, world),
                     retry_errors=args.retry_errors, schema=schema)

    print(f"{args.output} ({stats.rows_processed} ok, {stats.rows_failed} error rows)",
          file=sys.stderr)
    print("PORT qianfan-ocr " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
