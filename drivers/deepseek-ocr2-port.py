# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "saturate[hf]>=0.1.1",
#     "pillow>=10",
# ]
# ///
"""deepseek-ai/DeepSeek-OCR-2 — saturate port of uv-scripts/ocr/deepseek-ocr2-vllm.py (port-matrix row).

Run:
    hf jobs uv run --image vllm/vllm-openai:v0.26.0 --flavor a10g-small \
      --secrets HF_TOKEN deepseek-ocr2-port.py -- \
      --input-dataset <benchmark> --id-column PageID \
      --output hf://datasets/<org>/<repo>/data --limit 20

Input is a Hub dataset with an image column, not a bucket glob. `--id-column PageID`
is what makes the output joinable: scoring joins OCR rows to ground truth on PageID,
so the row id must BE the PageID (the default `index` ids are only stable per
revision — pin `--revision` if you use them). Fan out over several jobs with
`--shard i/n`; parquet parts and completion markers are per-rank, so the ranks can
share one output repo.

Currency receipt: the recipe pins vLLM NIGHTLY wheels; DeepseekOCR2ForCausalLM is in
the v0.26.0 registry, so this runs on vllm/vllm-openai:latest — either outcome is a
finding. Known deviations from the offline recipe (family-c request shape): the raw
"<image>\\n<|grounding|>..." prompt becomes image_url + text and the chat template
inserts the image token. skip_special_tokens=False is passed in-body so grounding
tags survive.
Context math: native ctx is 8192, so max_tokens drops 8192 -> 4096 to leave input
room (the recipe's 8192/8192 pair would 400 on every request in server mode).

Anti-repeat is ON — a recovered capability, not a deviation. This driver used to state
that the per-request NGramPerReqLogitsProcessor args were not expressible over
/chat/completions. That was an error: `vllm_xargs` IS `SamplingParams.extra_args`, and
the official vLLM DeepSeek-OCR-2 recipe documents exactly this server path (registering
v1's class, since deepseek_ocr2.py ships no processor of its own). `--no-anti-repeat`
drops both halves and reproduces the previous configuration in one flag.
`finish_reason` is recorded per row and is the instrument for measuring whether the
processor helped: a "length" finish is the repetition-loop signature, and without it a
degenerate page is indistinguishable from a long one.

`parse` stores the completion VERBATIM in `raw` and transforms nothing: the
per-model post-processing that turns raw into the scored `markdown` column lives in
bhl-ocr-eval/runners/normalize_outputs.py, so a rule can be corrected without paying
for GPUs again.
"""
import argparse
import base64
import io

SERVING = {
    # provenance: deepseek-ocr2-vllm.py LLM(...) kwargs; max_tokens halved for server-mode context math.
    # logits_processor / ngram_size 30 / window_size 90 / whitelist_token_ids: card- and
    # recipe-verbatim, not house choices. The official vLLM DeepSeek-OCR-2 recipe
    # (vllm-project/recipes DeepSeek/DeepSeek-OCR-2.md) is the source for the SERVER-side
    # path — register the class at boot with `--logits_processors <FQCN>`, send the params
    # per request as `vllm_xargs` (= SamplingParams.extra_args) — and it registers *v1's*
    # class, hence the deepseek_ocr FQCN below: deepseek_ocr2.py ships no processor of its
    # own. ngram_size/window_size are verbatim from that recipe. The whitelist ids are the
    # v1 card's <td>/</td> pair, carried over by the v2 recipe: INFERRED, not verified —
    # the v2 tokenizer was never diffed against v1's, so "still <td>/</td> for v2" is an
    # assumption, cheap to check if v2 adoption goes ahead. A JSON list, not the card's
    # Python set, because that is what vllm_xargs accepts. NOT yet exercised on Jobs: it
    # fails loudly (bad FQCN at boot, bad params per request via validate_params), so smoke
    # it with `--limit 10` before a corpus run; `--no-anti-repeat` drops both halves.
    "model": "deepseek-ai/DeepSeek-OCR-2",
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
assert SERVING["max_tokens"] < SERVING["max_model_len"], "context math: max_tokens must leave input room"

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
    multimodal data". No resize either way: nothing is passed about image
    resolution, so vLLM's default preset picks the crop grid from the input pixels.
    Identical to the v1 driver's, so the v1-vs-v2 delta stays the model.
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
                    help="Drop the recipe's n-gram loop suppressor (boot flag + per-request args)")
    ap.add_argument("--retry-errors", action="store_true")
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
    print("PORT deepseek-ocr2 " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
