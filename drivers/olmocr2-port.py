# /// script
# requires-python = ">=3.10"
# dependencies = ["saturate[hf]>=0.1.1", "pillow", "pyyaml"]
# ///
"""allenai/olmOCR-2-7B-1025-FP8 — saturate port of uv-scripts/ocr/olmocr2-vllm.py (port-matrix row).

Run:
    hf jobs uv run --image vllm/vllm-openai:v0.26.0 --flavor a10g-small \
      --secrets HF_TOKEN olmocr2-port.py -- \
      --input-dataset <benchmark> --id-column PageID --limit 20

Input is a Hub dataset with an image column; `--id-column PageID` keys the output rows
by the benchmark's page id, which is what scoring joins ground truth on. Fan out over
several jobs with `--shard i/n` (strided assignment, one job per rank; parquet parts
and completion markers are per-rank, so they land in one output repo).

FP8 checkpoint on a10g (Ampere) exercises the Marlin-kernel dequant path — boot
outcome is itself a SERVING finding. No-anchoring v4 YAML prompt (no page metadata
needed), text BEFORE image (the one recipe with that order). Client-side resize is
mandatory: longest side forced to exactly 1288 px (up- or downscale). Output = YAML
front matter + markdown body.

`parse` stores that completion VERBATIM in `raw`, front matter and all, and transforms
nothing: the split that turns raw into the scored `markdown` column lives in
bhl-ocr-eval/runners/normalize_outputs.py, so a rule can be corrected without paying
for GPUs again. The front matter is ALSO parsed here into a `markdown_metadata` JSON
column — that is page metadata being read, not page text being cleaned.
"""
import argparse
import base64
import io
import json
import re

SERVING = {
    # provenance: olmocr2-vllm.py LLM(...) kwargs + olmOCR pipeline sampling
    "model": "allenai/olmOCR-2-7B-1025-FP8",
    "image": "vllm/vllm-openai:v0.26.0",
    "max_model_len": 16384,
    "gpu_memory_utilization": 0.8,
    "extra_args": [
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb", "0",
        "--limit-mm-per-prompt", '{"image": 1}',
    ],
    "max_tokens": 8192,
    "temperature": 0.1,
    # repetition_penalty / stop: recipe-inherited from uv-scripts/ocr/olmocr2-vllm.py
    # (:395 `"repetition_penalty": 1.05,  # Discourage repetitive output`, and :396 for the
    # stop pair). Both were previously hardcoded inside to_request(), which put two real
    # request settings outside the one place this board promises to record where a setting
    # came from. The values are unchanged; only their visibility is.
    "repetition_penalty": 1.05,
    "stop": ["<|im_end|>", "<|endoftext|>"],
}
assert SERVING["max_tokens"] < SERVING["max_model_len"], "context math: max_tokens must leave input room"

PROMPT = (
    "Attached is one page of a document that you must process. Just return the plain "
    "text representation of this document as if you were reading it naturally. Convert "
    "equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown syntax "
    "![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top specifying values "
    "for the primary_language, is_rotation_valid, rotation_correction, is_table, and "
    "is_diagram parameters."
)

TARGET_LONGEST = 1288
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


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


def encode_png(value) -> str:
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

    import yaml
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
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            "temperature": SERVING["temperature"],
            "repetition_penalty": SERVING["repetition_penalty"],
            "stop": SERVING["stop"],
            "max_tokens": args.max_tokens,
        }

    def parse(row, body):
        choice = body["choices"][0]
        content = choice["message"]["content"]
        # The front matter is page METADATA, not page text, so it is parsed into its own
        # column here. The matching SPLIT — dropping it off the top of the scored text —
        # is post-processing and lives in normalize_outputs.py (`drop_front_matter`).
        m = FRONT_MATTER_RE.match(content.strip())
        metadata = (yaml.safe_load(m.group(1)) or {}) if m else {}
        usage = body.get("usage") or {}
        return {"raw": content,
                "markdown_metadata": json.dumps(metadata, ensure_ascii=False),
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
    print("PORT olmocr2 " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
