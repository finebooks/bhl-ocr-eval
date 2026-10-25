# /// script
# requires-python = ">=3.11"
# dependencies = ["saturate[hf]>=0.1.1", "pillow"]
# ///
"""nvidia/Nemotron-3.5-Parse — saturate driver for the BHL OCR benchmark.

Run (0.63B params, bf16 ~2.5 GB; Hopper GPU, see FA3 below). The base image needs the vendor
plugin first, so this is `hf jobs run` with docker/nemotron-parse-35/install-plugin.sh, not
`hf jobs uv run` (or build docker/nemotron-parse-35/Dockerfile and run the driver in that):
    hf jobs run --flavor h200 --secrets HF_TOKEN -v hf://buckets/<b>:/artifacts:ro \
      <SERVING["image"]> -- bash -c "bash /artifacts/install-plugin.sh && \
      uv run /artifacts/nemotron-parse-35-port.py --model-revision <sha> \
      --input-dataset <benchmark> --id-column PageID --output <prefix> --limit 20"

The architecture (RadioMBartPrefixForCausalLM: C-RADIOv4-SO400M encoder, 10-layer mBART
decoder over a causal visual prefix, one MTP head) is not in vLLM 0.26.0's registry. NVIDIA
serves it through a vLLM general plugin, `nemotron-parse` 3.5.0.post7, shipped inside the
model repo. The image in SERVING is the board-wide vllm/vllm-openai:v0.26.0 with that plugin
installed into the image's own Python under NVIDIA's published inference lock
(reference/parse_3_5/inference.constraints.txt); vLLM 0.26.0 is NVIDIA's reference serving
contract too (docs/serving.md), so no engine change from the other rows. The weights are
not baked in: they are served from the Hub, pinned with --model-revision.

Request shape, sampling and server flags are the card's (docs/serving.md "Generation
contract" and the H100 `vllm serve` block): text part first, then the image; temperature 0,
top_k 1, repetition_penalty 1.1, max_tokens 8744, skip_special_tokens False. The page is
sent at its original pixels as PNG, like the vendor client (vllm_example.py) sends the file
as-is: the server-side processor already fits it to 2048 x 1664, so a client-side resize
would resample twice.

The reply interleaves coordinate tokens, `<class_*>` tokens and markdown text. `raw` stores
it verbatim (that is what skip_special_tokens False is for). Turning it into scored text is
a decision about the text, so it lives in runners/normalize_outputs.py.

Two vendor defaults kept, on purpose:
- One-token MTP speculative decoding, which the plugin turns on unless told otherwise. Under
  greedy decoding the verifier keeps the target model's own argmax, so it changes speed, not
  the text, up to batch-level floating-point noise. `NP_VLLM_MTP=0` in the job env is the
  matched normal-decoding path if that ever needs checking.
- FlashAttention 3 (`--attention-config {"flash_attn_version": 3}`), from the card's serve
  command. FA3 is a Hopper kernel, hence the h200 flavour; there is no H100 flavour on Jobs.

One vendor file NOT used: logitsprocs/nemotron_parse_vllm_logitprocs.py, an optional
repetition-blocking processor that the documented serve command does not load. Every row on
the board decodes without anti-repeat help unless its card's own serving path enables it, so
a loop shows up in the loop-rate column.

Licence: OpenMDW-1.1 (weights and plugin); tokenizer MIT. Recorded in RESULTS.md.
"""
import argparse
import base64
import io

from PIL import Image

SERVING = {
    # Per-value provenance. "vendor" = nvidia/Nemotron-3.5-Parse docs/serving.md and README
    # at the pinned revision; "house" = our choice.
    "model": "nvidia/Nemotron-3.5-Parse",
    # house: the board-wide pin, by its amd64 digest, plus the vendor plugin from the model repo
    # at 62cb1e4ba153d113ff64aaf2be6e068c377cf53c installed by
    # docker/nemotron-parse-35/install-plugin.sh under NVIDIA's full inference lock.
    "image": "vllm/vllm-openai:v0.26.0@sha256:770fe65b2c73ee74a5c42165cf3433de4048cc2cd9c57a937ca4e35aba5aa87b",
    # vendor: --max-model-len 16384, which is also the model's max_position_embeddings.
    "max_model_len": 16384,
    # house: the specialist peers' value. The vendor leaves vLLM's default (0.9).
    "gpu_memory_utilization": 0.8,
    "extra_args": [
        # vendor: the plugin and the processor are remote code in the repo.
        "--trust-remote-code",
        # vendor: the rest of the H100 serve block, verbatim. --max-num-seqs 192 is the
        # vendor's throughput profile; our window (12 in flight) never reaches it.
        "--dtype", "bfloat16",
        "--distributed-executor-backend", "uni",
        "--max-num-seqs", "192",
        "--max-num-batched-tokens", "16384",
        "--no-enable-prefix-caching",
        "--limit-mm-per-prompt", '{"image": 1}',
        "--attention-config", '{"flash_attn_version": 3}',
    ],
    # vendor: the generation contract.
    "prompt": "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>",
    "max_tokens": 8744,
    "temperature": 0.0,
    "top_k": 1,
    "repetition_penalty": 1.1,
}

# The repo's chat_template.jinja concatenates message content and Transformers 5 loads it
# with the tokenizer, so the card's `--chat-template chat_template.jinja` is the default
# here and is not passed (it would need a local path inside the job).

# Context maths. The plugin counts image tokens as (H // 16) * ((W // 16) // 4) + 1, and the
# processor caps every page at 2048 x 1664: 128 * 26 + 1 = 3329 at most.
IMAGE_TOKENS_WORST_CASE = (2048 // 16) * ((1664 // 16) // 4) + 1
PROMPT_TOKENS_BOUND = 64  # six control tokens plus template; generously
assert IMAGE_TOKENS_WORST_CASE + SERVING["max_tokens"] + PROMPT_TOKENS_BOUND <= SERVING["max_model_len"], (
    f"context math: {IMAGE_TOKENS_WORST_CASE} image tokens + {SERVING['max_tokens']} output "
    f"+ {PROMPT_TOKENS_BOUND} prompt exceed max_model_len {SERVING['max_model_len']}"
)


def to_pil(value):
    """One dataset image cell -> a PIL image (decoded image, {"bytes","path"} dict or bytes)."""
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value)))
    raise ValueError(f"unsupported image value: {type(value)}")


def encode_png(value) -> str:
    """RGB-convert at original size, base64 PNG (vendor: one RGB page, file sent as-is)."""
    img = to_pil(value).convert("RGB")
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
    # REQUIRED, with no default: a forgotten --output must not resume into another run's output.
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
            # vendor: text part first, then the image (vllm_example.py, README client).
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": SERVING["prompt"]},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            "temperature": SERVING["temperature"],
            "max_tokens": args.max_tokens,
            # vendor: sent as extra_body in the card's client, i.e. top-level body fields.
            "top_k": SERVING["top_k"],
            "repetition_penalty": SERVING["repetition_penalty"],
            "skip_special_tokens": False,
        }

    def parse(row, body):
        choices = body.get("choices") or []
        if not choices:
            raise ValueError(f"no choices in response: {body!r}")
        usage = body.get("usage") or {}
        # `or ""` is envelope handling, not text cleaning: a null `content` is malformed.
        return {"raw": choices[0]["message"].get("content") or "",
                "model": SERVING["model"],
                "finish_reason": choices[0].get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}

    extra = ["--max-model-len", str(SERVING["max_model_len"]),
             "--gpu-memory-utilization", str(SERVING["gpu_memory_utilization"]),
             *SERVING["extra_args"]]
    # The served checkpoint is pinned at boot, not read back from the Hub afterwards.
    if args.model_revision:
        SERVING["model_revision"] = args.model_revision
        extra += ["--revision", args.model_revision]
    with Engine(SERVING["model"], engine="vllm", extra_args=extra) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, args.output,
                     window=Auto(initial=4, target_waiting=4, max_limit=12, step=2),
                     shard=(rank, world),
                     retry_errors=args.retry_errors)
    print("PORT nemotron-parse-35 " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
