# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]>=0.1.1",
#     "pillow>=10",
# ]
# ///
"""deepseek-ai/DeepSeek-OCR — saturate port of uv-scripts/ocr/deepseek-ocr-vllm.py (port-matrix row).

Run:
    hf jobs uv run --image vllm/vllm-openai:v0.26.0 --flavor a10g-small \
      --secrets HF_TOKEN deepseek-ocr-port.py -- \
      --input-dataset <benchmark> --id-column PageID \
      --output hf://datasets/<org>/<repo>/data --limit 20

Input is a Hub dataset with an image column, not a bucket glob. `--id-column PageID`
is what makes the output joinable: scoring joins OCR rows to ground truth on PageID,
so the row id must BE the PageID (the default `index` ids are only stable per
revision — pin `--revision` if you use them). Fan out over several jobs with
`--shard i/n`; parquet parts and completion markers are per-rank, so the ranks can
share one output repo.

Deliberate twin of deepseek-ocr2-port.py: same request shape, same serve flags, same
sampling, same output schema, so the benchmark's v1-vs-v2 delta is the model and only
the model. Differences from the v2 driver are listed at the bottom of this docstring.

Currency receipt: unlike the v2 recipe (nightly wheels), deepseek-ocr-vllm.py pins
stable `vllm>=0.15.1` and DeepseekOCRForCausalLM has been in the vLLM registry since
well before v0.26.0 — vllm/vllm-openai:latest is the boring, expected path here.

Known deviations from the offline recipe (family-c request shape): the raw
"<image>\\n<|grounding|>..." prompt becomes image_url + text and the chat template
inserts the image token. skip_special_tokens=False is passed in-body so grounding tags
survive.

Anti-repeat is ON — a recovered capability, not a deviation. This driver used to state
that the per-request NGramPerReqLogitsProcessor args were not expressible over
/chat/completions. That was an error: `vllm_xargs` IS `SamplingParams.extra_args`, and
the official vLLM DeepSeek-OCR recipe documents exactly this server path (register the
class at boot, send the params per request). v1 is the model the processor was written
for, so it matters most here. `--no-anti-repeat` drops both halves and reproduces the
previous configuration in one flag. `finish_reason` stays in the output schema and is
the instrument for measuring whether the processor helped: a "length" finish is the
repetition-loop signature, and without it a degenerate page is indistinguishable from
a long one.

`parse` stores the completion VERBATIM in `raw` and transforms nothing: the per-model
post-processing that turns raw into the scored `markdown` column lives in
bhl-ocr-eval/runners/normalize_outputs.py, so a rule can be corrected without paying
for GPUs again.

Context math: native ctx is 8192 (max_position_embeddings), which is also the ceiling —
it cannot be raised. Input is bounded and small: no mm_processor_kwargs are passed, so
vLLM uses its Gundam default (BASE_SIZE 1024 / IMAGE_SIZE 640 / CROP_MODE True,
MAX_CROPS 6), i.e. n*100 tile tokens + 256 global-view tokens = 856 at n=6, 1156 at the
model's absolute n=9. INPUT_BUDGET below rounds that to 1280 to cover prompt and chat
template. max_tokens therefore drops 8192 -> 4096: the recipe's 8192/8192 pair is
offline-only and 400s on every request in server mode, and 4096 both clears the budget
with ~2.9k spare and matches the v2 driver exactly, so neither version gets more room
to finish a page than the other.

Deliberate differences vs deepseek-ocr2-port.py (all documentation or driver hygiene —
none change what the model sees):
  - per-value SERVING provenance comment (this port's brief); v2 has the one-liner
  - the empty-output raise this driver used to carry is now a registry entry
    (`require_non_empty`, registered for v1 and not v2) — a real difference, still
    worth backporting, but it belongs to normalize_outputs.py now, not here.
    (`finish_reason` is NOT a difference: the v2 driver records it too)
  - requires-python >=3.11 (v2: >=3.10); no 3.10 consumer exists for these drivers
  - INPUT_BUDGET assert instead of the bare `max_tokens < max_model_len` assert
"""
import argparse
import base64
import io

# Serving starting values for deepseek-ai/DeepSeek-OCR. Per-value provenance:
# - model: card.
# - max_model_len 8192: recipe-inherited (deepseek-ocr-vllm.py default) AND the hard
#   ceiling — it equals the model's native max_position_embeddings.
# - gpu_memory_utilization 0.8: recipe-inherited; neither the card nor the vLLM recipe
#   page states a value.
# - --no-enable-prefix-caching, --mm-processor-cache-gb 0: verbatim from the official
#   `vllm serve` line on the vLLM DeepSeek-OCR recipe page.
# - --trust-remote-code: recipe-inherited (LLM(trust_remote_code=True)); the repo ships
#   auto_map custom code. The recipe page's serve line omits it; serving without it
#   fails to load, so the recipe is right and the page is incomplete.
# - --limit-mm-per-prompt '{"image": 1}': house OCR flag. The v1 recipe's LLM() does not
#   set it (the v2 recipe's does); one image per page, so it costs nothing and keeps the
#   two drivers' serve lines identical.
# - logits_processor + ngram_size 30 / window_size 90 / whitelist_token_ids
#   [128821, 128822]: card-verbatim AND recipe-verbatim, not house choices. The model
#   card's vLLM example passes `ngram_size=30, window_size=90,
#   whitelist_token_ids={128821, 128822}` (whitelist: <td>, </td>), and the official
#   vLLM DeepSeek-OCR recipe (vllm-project/recipes DeepSeek/DeepSeek-OCR.md) is the
#   source for the SERVER-side path: register the class at boot with
#   `--logits_processors <FQCN>`, then send the same three values per request as
#   `vllm_xargs`, which is `SamplingParams.extra_args` under the hood. The list
#   spelling is deliberate — `vllm_xargs` is JSON, so the card's Python set becomes a
#   list. This driver previously left the processor off on the (wrong) grounds that
#   /chat/completions could not carry the args; off is the configuration that diverged
#   from the offline recipe. NOT yet exercised on Jobs: it fails loudly (bad FQCN at
#   boot, bad params per request via validate_params), so smoke it with `--limit 10`
#   before a corpus run. `--no-anti-repeat` drops both halves. Same call as v2.
# - max_tokens 4096: HOUSE CHOICE, not card-recommended. Card and vLLM recipe both say
#   8192, which is only valid offline where input tokens are not charged against the
#   same window. See the context-math paragraph above.
# - temperature 0.0: card-verbatim (card and vLLM recipe agree).
# - image resolution: nothing passed, so vLLM's default Gundam preset applies — which is
#   also the mode the card recommends for documents. Recipe-inherited and
#   card-recommended coincide, so there is nothing to override.
SERVING = {
    "model": "deepseek-ai/DeepSeek-OCR",
    "image": "vllm/vllm-openai:v0.26.0",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.8,
    "extra_args": [
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
        "--limit-mm-per-prompt", '{"image": 1}',
    ],
    "logits_processor": "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor",
    "ngram_size": 30,
    "window_size": 90,
    "whitelist_token_ids": [128821, 128822],
    "max_tokens": 4096,
    "temperature": 0.0,
}
# Gundam worst case 9*100 + 256 = 1156 vision tokens, rounded up for prompt + template.
INPUT_BUDGET = 1280
assert SERVING["max_tokens"] + INPUT_BUDGET <= SERVING["max_model_len"], (
    "context math: max_tokens + image/prompt tokens must fit in 8192 or every request 400s"
)

PROMPT = "<|grounding|>Convert the document to markdown."

# Formats whose bytes go to the server untouched when the image is already RGB.
PASSTHROUGH_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg"}


def encode_image(value) -> tuple[str, str]:
    """Return `(mime, base64)` for one dataset image cell, at full resolution.

    A dataset image column yields a PIL image, a {"bytes": ...} dict, or raw bytes
    depending on how the dataset stores it; all three land here. Already-RGB
    PNG/JPEG bytes are passed through byte-for-byte, which is what the bucket path
    used to do; everything else (grayscale, bitonal, palette, CMYK, RGBA — the
    common shapes for library scans) is converted to RGB and re-encoded as lossless
    PNG, because an unconverted non-RGB scan 500s the serving path on "loading
    multimodal data". No resize either way: the Gundam crop grid is chosen from the
    input pixels (see the context-math note above).
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
    ap.add_argument("--no-anti-repeat", action="store_true",
                    help="Drop the card's n-gram loop suppressor (boot flag + per-request args)")
    ap.add_argument("--retry-errors", action="store_true")
    args = ap.parse_args()

    if args.max_tokens + INPUT_BUDGET > SERVING["max_model_len"]:
        ap.error(f"--max-tokens must be <= {SERVING['max_model_len'] - INPUT_BUDGET} (8192 ctx minus image/prompt budget)")

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
                                  "window_size": SERVING["window_size"],
                                  "whitelist_token_ids": SERVING["whitelist_token_ids"]}
        return body

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
    # Last, deliberately: --logits_processors is list-valued, so any following token
    # that does not start with "-" would be swallowed as a second processor.
    if anti_repeat:
        extra += ["--logits_processors", SERVING["logits_processor"]]
    with Engine(SERVING["model"], engine="vllm", extra_args=extra) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, args.output,
                     window=Auto(initial=4, target_waiting=4, max_limit=12, step=2),
                     shard=(rank, world),
                     retry_errors=args.retry_errors)
    print("PORT deepseek-ocr " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
