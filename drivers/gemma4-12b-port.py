# /// script
# requires-python = ">=3.11"
# dependencies = ["saturate[hf]>=0.1.1", "pillow"]
# ///
"""google/gemma-4-12B-it — saturate driver for the BHL OCR benchmark (port-matrix row).

Run (12.0B params ≈ 24GB of bf16 weights, so a 48GB card is the right size):
    hf jobs uv run --image vllm/vllm-openai:v0.26.0 --flavor l40sx1 \
      --secrets HF_TOKEN gemma4-12b-port.py -- \
      --input-dataset <benchmark> --id-column PageID --limit 20

Input is a Hub dataset with an image column; `--id-column PageID` keys the output rows
by the benchmark's page id, which is what scoring joins ground truth on. Fan out over
several jobs with `--shard i/n` (strided assignment, one job per rank; parquet parts
and completion markers are per-rank, so they land in one output repo).

One of the two odd ones out on this board. Every other row is an OCR specialist run with
its own card's recommended OCR prompt, verbatim. gemma-4 is a general-purpose VLM: its
card lists OCR among the model's capabilities but documents no OCR prompt for it. So the
prompt here is a HOUSE CHOICE with no card basis — see PROMPTS below — and the board row
must carry the variant name with the score. `--prompt` selects the variant; `neutral` is
the default and the one that appears on the board.

This driver and qwen35-9b-port.py are a MATCHED PAIR: they answer the board's
generalist-vs-specialist question, so the PROMPTS dict below is byte-identical between the
two files and both run the same client-side image budget order and the same `max_tokens`.
Any gap between their two rows has to be the model, not the wording. Do not tune the
prompt in one file without making the identical edit in the other.

UNIFIED (`Gemma4UnifiedForConditionalGeneration`, model_type `gemma4_unified`): the 12B
has no dedicated encoders — raw image patches and audio waveforms are projected straight
into the LLM embedding space. Two consequences the driver acts on. (1) Output is still
text-only, so the OpenAI chat shape and the `raw` string column need no change. (2) Unlike
the MoE sibling this replaces, this checkpoint really does ship an audio tower
(`audio_config` is populated, `Gemma4UnifiedAudioFeatureExtractor` in processor_config), so
zeroing audio in `--limit-mm-per-prompt` is load-bearing, not cosmetic.

No recipe exists in uv-scripts for this model; this driver is written from the model card
+ the checkpoint's own config/processor JSON and the vLLM Gemma4 recipe rather than ported
from an existing script.

`parse` stores the completion VERBATIM in `raw` and transforms nothing: the per-model
post-processing that turns raw into the scored `markdown` column — for this model the
whole-reply code-fence unwrap a generalist keeps producing, then the raise on an empty
completion — lives in bhl-ocr-eval/runners/normalize_outputs.py, so a rule can be
corrected without paying for GPUs again. The prompt variant and its hash still ride on
every row, because the score cannot be read without them.
"""

import argparse
import base64
import hashlib
import io
import math

# Serving starting values for google/gemma-4-12B-it. Per-value provenance:
# - temperature 1.0 / top_p 0.95 / top_k 64: CARD-VERBATIM. The card gives one
#   standardized block — "Use the following standardized sampling configuration
#   across all use cases: temperature=1.0, top_p=0.95, top_k=64" — with no
#   task-specific override.
#   NOT USED BY DEFAULT — see below. Recorded because it is the card's answer, and
#   `--card-sampling` sends it verbatim, but the board runs greedy.
# - GREEDY IS THE DEFAULT, and it is a deliberate, documented deviation from the card.
#   The card's block is a standardized recommendation for general use; this is
#   transcription. Every OCR specialist on this board runs at or near temperature 0.0
#   (ovis 0.0, olmOCR 0.1) on their own cards' authority, and running the generalists
#   stochastic while the specialists run greedy would confound the one comparison these
#   two models exist to make.
#   NO SEED, here or anywhere on this board: a seed is not a card-documented value, and on
#   a continuous-batching server it fixes the sampling RNG but not the forward-pass
#   numerics, so it would imply a reproducibility it cannot deliver. Run-to-run variance
#   is a stated caveat in RESULTS.md instead.
# - max_soft_tokens 1120: CARD-DERIVED. The card documents a "configurable visual token
#   budget" of {70, 140, 280, 560, 1120} and tells you to use higher budgets for
#   fine-grained detail; 1120 is the documented maximum, so it is the card's own answer
#   for reading small print. Verified against the checkpoint: the image processor's
#   `max_soft_tokens` default is 280 (processor_config.json) and the vision tower's
#   `mm_posemb_size` is 1120, i.e. 1120 is the hard ceiling and not an extrapolation.
#   Wire-up (the `max_soft_tokens` processor kwarg) is from the vLLM Gemma4 recipe.
# - chat_template_kwargs enable_thinking=False: pinned. The card describes thinking as
#   the `<|think|>` token at the start of the system prompt; the shipped
#   chat_template.jinja exposes that as `enable_thinking`, defaulting to false and
#   injecting the token into the first system turn when true. Sending it explicitly means
#   a template change cannot silently prepend a reasoning channel to the transcription.
# - max_model_len 16384: HOUSE CHOICE. Native ctx is 262144 (text_config
#   max_position_embeddings; the card says 256K) — booting uncapped profiles a full-context
#   KV cache and the boot dies. vLLM recipe suggests 16384-32768; 16384 is ample here (one
#   image at 1120 soft tokens + one short prompt).
# - max_num_batched_tokens 16384: HOUSE CHOICE, working around a known failure. The
#   default 2048 is smaller than this model family's per-image encoder budget and boot
#   fails with "Chunked MM input disabled but max_tokens_per_mm_item (2496) is larger than
#   max_num_batched_tokens (2048)" (vllm-project/recipes#441) — and 2496 is measured
#   at the *default* 280-token budget, so at 1120 the requirement is higher again.
#   16384 clears it with room to spare.
# - gpu_memory_utilization 0.90: vLLM Gemma4 recipe ("0.90-0.95 to maximize KV
#   cache"). Higher than the 0.8 peers use, because the weights are the tight
#   constraint here, not the KV cache.
# - limit-mm-per-prompt image=1/audio=0/video=0: HOUSE OCR default. This checkpoint is a
#   unified any-to-any model with a real audio tower (`audio_config` populated, unlike the
#   MoE sibling this row replaces, where it was null); the recipe notes that image-only
#   workloads should zero the audio limit so the audio path is never allocated.
# - cache flags: HOUSE OCR defaults, same as every peer (OCR never reuses an image,
#   so prefix/processor caches only cost memory).
# - max_tokens 8192: HOUSE CHOICE (card sets none). A dense BHL page fits well
#   inside this; the context assert below keeps it under max_model_len. Matched to
#   qwen35-9b-port.py so the two generalist rows share an output ceiling.
SERVING = {
    "model": "google/gemma-4-12B-it",
    "image": "vllm/vllm-openai:v0.26.0",
    "max_model_len": 16384,
    "gpu_memory_utilization": 0.90,
    "extra_args": [
        "--max-num-batched-tokens", "16384",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
        "--limit-mm-per-prompt", '{"image": 1, "audio": 0, "video": 0}',
        "--mm-processor-kwargs", '{"max_soft_tokens": 1120}',
    ],
    "max_tokens": 8192,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
}
assert SERVING["max_tokens"] < SERVING["max_model_len"], "context math: max_tokens must leave input room"

# ---------------------------------------------------------------------------
# PROMPTS — every one of these is a HOUSE CHOICE. There is no card-recommended OCR
# prompt for gemma-4 (the card lists OCR as a capability but gives no wording for it),
# unlike every OCR specialist on this board, so the score cannot be detached from the
# wording that produced it: the variant name and a hash of its text are written into
# every output row.
#
# BYTE-IDENTICAL to the PROMPTS dict in qwen35-9b-port.py. The two generalists are a
# matched pair; a difference in wording would make their rows incomparable.
#
#   neutral  (default, the board row) — a plain transcription instruction with no
#            task coaxing, no format schema, no hint that the material is historical.
#            The honest "a straightforward instruction we wrote" baseline; the
#            headline claim about generalist-vs-specialist rides on this one, so it
#            deliberately gives the model no advantage the specialists' own prompts
#            would not also have.
#   format   — tuned: names the output contract (markdown, tables, formulas,
#            reading order, no commentary). Tests how much of a generalist's gap is
#            output-shape mismatch rather than reading ability.
#   archival — tuned: names the material (scanned historical book pages, period
#            spelling/typography, damage) and forbids modernising. Tests how much is
#            the model "correcting" the source it was never told was historical.
#   archival_format — tuned: both levers at once; the model's best realistic shot,
#            and the ceiling against which `neutral` should be read.
# ---------------------------------------------------------------------------
PROMPTS = {
    "neutral": "Transcribe all of the text in this image.",
    "format": (
        "Transcribe all of the text in this image as Markdown.\n"
        "Follow the natural reading order of the page. Render tables as Markdown "
        "tables and mathematical expressions as LaTeX. Reproduce the text only — do "
        "not summarise, translate, or add any commentary, headings, or notes of your "
        "own. Output the transcription and nothing else."
    ),
    "archival": (
        "This is a scanned page from a historical printed book.\n"
        "Transcribe all of the text on the page exactly as it appears. Keep the "
        "original spelling, capitalisation, punctuation, hyphenation, and typography, "
        "including archaic or inconsistent forms and the long s. Do not modernise, "
        "correct, or normalise anything. Where the scan is damaged or a character is "
        "genuinely illegible, transcribe what you can read and do not invent text to "
        "fill the gap."
    ),
    "archival_format": (
        "This is a scanned page from a historical printed book.\n"
        "Transcribe all of the text on the page as Markdown, following the natural "
        "reading order. Keep the original spelling, capitalisation, punctuation, "
        "hyphenation, and typography, including archaic or inconsistent forms and the "
        "long s. Do not modernise, correct, or normalise anything. Where the scan is "
        "damaged or a character is genuinely illegible, transcribe what you can read "
        "and do not invent text to fill the gap. Render tables as Markdown tables and "
        "mathematical expressions as LaTeX. Output the transcription and nothing else."
    ),
}

# Payload guard only, not a quality knob: the Gemma4 unified image processor does its own
# resize down to the max_soft_tokens budget, so this bound is well above anything the
# encoder can represent (1120 soft tokens x 3x3 pooling x 16px patches ≈ 2.6MP) and
# only exists to stop a 600dpi plate scan becoming a 40MB base64 body. JPEG q95
# matches the peer convention on this board.
MAX_PIXELS = 8294400


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


def encode_jpeg(value) -> str:
    from PIL import Image

    img = to_pil(value).convert("RGB")
    w, h = img.size
    if w * h > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (w * h))
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
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
    ap.add_argument("--prompt", default="neutral", choices=sorted(PROMPTS),
                    help="prompt variant; all are house choices (see PROMPTS)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="RANK/WORLD",
                    help="Strided fan-out across jobs, e.g. 2/8 (default: 0/1)")
    ap.add_argument("--max-tokens", type=int, default=SERVING["max_tokens"])
    # Greedy is the DEFAULT for this board, and it is a deliberate deviation from the card.
    # The card's sampling block is its recommendation for general chat, not for transcription:
    # every OCR specialist on this board runs at temperature 0.0, and running the generalists
    # stochastic while the specialists run greedy would confound the one comparison this model
    # exists to make. --card-sampling restores the card's block verbatim.
    ap.add_argument("--card-sampling", dest="greedy", action="store_false", default=True,
                    help="use the card's sampling block (temperature 1.0, top_p 0.95, top_k 64) "
                         "instead of this board's greedy default")
    ap.add_argument("--retry-errors", action="store_true")
    args = ap.parse_args()

    from saturate import Auto, Engine, dataset_rows, pump, shard_select

    prompt = PROMPTS[args.prompt]
    prompt_sha8 = hashlib.sha256(prompt.encode()).hexdigest()[:8]
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
        req = {
            "model": SERVING["model"],
            # card: "place Image content before the text in your prompt"
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": args.max_tokens,
            # off by default in the chat template; pinned so a template change cannot
            # silently prepend a reasoning channel to the transcription
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if args.greedy:
            req["temperature"] = 0.0
        else:
            req |= {"temperature": SERVING["temperature"],
                    "top_p": SERVING["top_p"], "top_k": SERVING["top_k"]}
        return req

    def parse(row, body):
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        # `or ""` is envelope handling, not text cleaning: a null `content` field is a
        # malformed response, and the raw column holds strings (an empty one is valid).
        return {"raw": choice["message"]["content"] or "",
                "model": SERVING["model"],
                "finish_reason": choice.get("finish_reason"),
                "prompt_variant": args.prompt,
                "prompt_sha8": prompt_sha8,
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
    print(f"PORT gemma4-12b prompt={args.prompt}/{prompt_sha8} " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
