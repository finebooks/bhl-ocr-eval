# /// script
# requires-python = ">=3.11"
# dependencies = ["saturate[hf]>=0.1.1", "pillow"]
# ///
"""Qwen/Qwen3.5-9B — saturate driver for the BHL OCR benchmark (port-matrix row).

Run (9.7B params ≈ 19GB of bf16 weights, so a 48GB card is comfortable):
    hf jobs uv run --image vllm/vllm-openai:v0.26.0 --flavor l40sx1 \
      --secrets HF_TOKEN qwen35-9b-port.py -- \
      --input-dataset <benchmark> --id-column PageID --limit 20

Input is a Hub dataset with an image column; `--id-column PageID` keys the output rows
by the benchmark's page id, which is what scoring joins ground truth on. Fan out over
several jobs with `--shard i/n` (strided assignment, one job per rank; parquet parts
and completion markers are per-rank, so they land in one output repo).

One of the two odd ones out on this board. Every other row is an OCR specialist run with
its own card's recommended OCR prompt, verbatim. Qwen3.5-9B is a general-purpose VLM: its
card documents no OCR prompt, because transcription is not a documented task for it. So
the prompt here is a HOUSE CHOICE with no card basis — see PROMPTS below — and the board
row must carry the variant name with the score. `--prompt` selects the variant; `neutral`
is the default and the one that appears on the board.

This driver and gemma4-12b-port.py are a MATCHED PAIR: they answer the board's
generalist-vs-specialist question, so the PROMPTS dict below is byte-identical between the
two files and both run the same client-side image budget order and the same `max_tokens`.
Any gap between their two rows has to be the model, not the wording. Do not tune the
prompt in one file without making the identical edit in the other.

No recipe exists in uv-scripts for this model; this driver is written from the model card
+ the checkpoint's own config/processor JSON rather than ported from an existing script.

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

# Serving starting values for Qwen/Qwen3.5-9B. Per-value provenance:
# - temperature 0.7 / top_p 0.8 / top_k 20 / min_p 0.0 / presence_penalty 1.5 /
#   repetition_penalty 1.0: CARD-VERBATIM. The card's Best Practices block gives four
#   sampling profiles; transcription is a general task run with thinking OFF, so the row
#   used here is "Instruct (or non-thinking) mode for general tasks: temperature=0.7,
#   top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0".
#   FLAG, deliberately not overridden: presence_penalty=1.5 is a large flat penalty on
#   any token already emitted, and a page of prose reuses its function words constantly,
#   so this value is a real risk on a long transcription — the card's own caveat is that
#   "using a higher value may occasionally result in language mixing and a slight
#   decrease in model performance".
#   NOT USED BY DEFAULT — see below. This block is recorded because it is the card's
#   answer, and `--card-sampling` sends it verbatim, but the board runs greedy.
# - GREEDY IS THE DEFAULT, and it is a deliberate, documented deviation from the card.
#   The card's block is its recommendation for general chat; this is transcription. Every
#   OCR specialist on this board runs at or near temperature 0.0 (ovis 0.0, glm 0.01,
#   olmOCR 0.1, dots.mocr 0.1, LightOnOCR 0.2), and presence_penalty 1.5 penalises exactly
#   the function-word reuse that ordinary prose is made of. Running the generalists
#   stochastic while the specialists run near-greedy would confound the one comparison
#   these two models exist to make.
#   NO SEED, here or anywhere on this board: a seed is not a card-documented value, and on
#   a continuous-batching server it fixes the sampling RNG but not the forward-pass
#   numerics, so it would imply a reproducibility it cannot deliver. Run-to-run variance
#   is a stated caveat in RESULTS.md instead.
# - chat_template_kwargs enable_thinking=False: CARD-VERBATIM wire-up. The card says to
#   pass `"chat_template_kwargs": {"enable_thinking": False}` on OpenAI-compatible
#   endpoints. This is NOT belt-and-braces: the shipped chat_template.jinja opens a
#   `<think>` block on the generation prompt unless `enable_thinking is false`, i.e.
#   thinking is ON by default and an unset flag would put a reasoning channel in front of
#   every transcription.
# - --reasoning-parser qwen3 DROPPED: the card's serve line carries it. Omitted on
#   purpose. A reasoning parser splits the completion across `content` and
#   `reasoning_content`, and this driver's contract is that `raw` holds the completion
#   verbatim in one field. With thinking disabled there is no reasoning channel to parse,
#   and a parser that finds no closing tag can route the whole reply into
#   `reasoning_content` and leave `content` empty — a silent board-wide zero.
# - max_model_len 16384: HOUSE CHOICE. Native ctx is 262144 (text_config
#   max_position_embeddings; the card's serve line boots at that) — profiling a full
#   262144 KV cache kills the boot. 16384 is ample here: one image at the visual-token
#   budget below (~2048) + a short prompt + max_tokens.
# - max_num_batched_tokens 16384: HOUSE CHOICE, margin not a fix. One page image is
#   ~2048 visual tokens; vLLM refuses to boot when a single multimodal item is larger
#   than max_num_batched_tokens and chunked MM input is off. 16384 clears it with room.
# - gpu_memory_utilization 0.90: HOUSE CHOICE. Higher than the 0.8 the specialist peers
#   use; matched to gemma4-12b-port.py so the two generalist rows differ by model, not by
#   serving slack. Cheap here — this is a hybrid-attention model (only 8 of 32 layers are
#   full attention), so the KV cache is small and the weights are the real constraint.
# - limit-mm-per-prompt image=1/video=0: HOUSE OCR default. The checkpoint has a
#   video_token_id and no audio config, so video is the only extra modality to zero.
# - cache flags: HOUSE OCR defaults, same as every peer (OCR never reuses an image, so
#   prefix/processor caches only cost memory).
# - max_tokens 8192: HOUSE CHOICE, a documented deviation. The card recommends "an output
#   length of 32,768 tokens for most queries", but that is a generic ceiling for
#   reasoning-heavy use, not a transcription constraint: a dense BHL page fits well inside
#   8192, and 32768 mostly buys room for a repetition loop to burn GPU time in. Truncation
#   is not hidden — `finish_reason` is stored on every row, so a page that hits the cap is
#   visible in the data.
SERVING = {
    "model": "Qwen/Qwen3.5-9B",
    "image": "vllm/vllm-openai:v0.26.0",
    "max_model_len": 16384,
    "gpu_memory_utilization": 0.90,
    "extra_args": [
        "--max-num-batched-tokens", "16384",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
        "--limit-mm-per-prompt", '{"image": 1, "video": 0}',
    ],
    "max_tokens": 8192,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}
assert SERVING["max_tokens"] < SERVING["max_model_len"], "context math: max_tokens must leave input room"

# ---------------------------------------------------------------------------
# PROMPTS — every one of these is a HOUSE CHOICE. There is no card-recommended OCR
# prompt for Qwen3.5-9B (the card does not mention OCR or document transcription at
# all), unlike every OCR specialist on this board, so the score cannot be detached from
# the wording that produced it: the variant name and a hash of its text are written into
# every output row.
#
# BYTE-IDENTICAL to the PROMPTS dict in gemma4-12b-port.py. The two generalists are a
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

# Unlike gemma-4, whose processor resizes down to a fixed soft-token budget on its own,
# Qwen's ViT is native-resolution: visual tokens scale with the pixels we send, at
# (patch_size 16 x spatial_merge_size 2)^2 = 1024 px per token (vision_config). The
# checkpoint's own cap is 16777216 px — 16384 visual tokens, more than max_model_len — so
# the image budget has to be set HERE or a big plate scan silently 400s the request.
# 2048 tokens is a HOUSE CHOICE: it lands the effective resolution next to the
# ~2.6MP that gemma-4's maximum 1120-soft-token budget represents, keeping the matched
# pair matched on detail as well as on wording. JPEG q95 matches the peer convention.
VISUAL_TOKEN_BUDGET = 2048
PIXELS_PER_VISUAL_TOKEN = 1024
MAX_PIXELS = VISUAL_TOKEN_BUDGET * PIXELS_PER_VISUAL_TOKEN


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
    # every OCR specialist on this board runs at temperature 0.0, and presence_penalty 1.5
    # penalises exactly the function-word reuse that ordinary prose is made of. Running the
    # generalists stochastic while the specialists run greedy would confound the one comparison
    # this model exists to make. --card-sampling restores the card's block verbatim.
    ap.add_argument("--card-sampling", dest="greedy", action="store_false", default=True,
                    help="use the card's sampling block (temperature 0.7, top_p/top_k/min_p and "
                         "the presence/repetition penalties) instead of this board's greedy default")
    ap.add_argument("--card-penalties", action="store_true",
                    help="keep greedy temperature but restore the card's presence/repetition "
                         "penalties — the anti-repetition guard every specialist keeps")
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
            # card: the image_url content part comes first, then the text query
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": args.max_tokens,
            # ON by default in the shipped chat template — the card's documented way to
            # keep a reasoning channel out of the transcription
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if args.greedy:
            req["temperature"] = 0.0
            # The greedy default was chosen so the generalist-vs-specialist comparison is
            # not confounded by sampling temperature. But the card's block ALSO carries
            # presence_penalty 1.5, which the card recommends specifically against endless
            # repetition — so plain greedy silently stripped this model's anti-repetition
            # guard while every specialist kept its own (NGram processors on the DeepSeek
            # pair, repetition_penalty on GLM-OCR and olmOCR-2). That asymmetry biases the
            # one comparison these two rows exist to make. --card-penalties restores just
            # the penalty, at greedy temperature, so the guard matches the specialists'
            # treatment without reintroducing the sampling confound.
            if args.card_penalties:
                req |= {"presence_penalty": SERVING["presence_penalty"],
                        "repetition_penalty": SERVING["repetition_penalty"]}
        else:
            req |= {"temperature": SERVING["temperature"],
                    "top_p": SERVING["top_p"], "top_k": SERVING["top_k"],
                    "min_p": SERVING["min_p"],
                    "presence_penalty": SERVING["presence_penalty"],
                    "repetition_penalty": SERVING["repetition_penalty"]}
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
    print(f"PORT qwen35-9b prompt={args.prompt}/{prompt_sha8} " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
