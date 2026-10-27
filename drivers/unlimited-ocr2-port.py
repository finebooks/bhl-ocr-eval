# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "saturate[hf]==0.2.0",
#     "pillow>=10",
# ]
# ///
"""
Convert document images to grounded text using Baidu Unlimited-OCR-2 via saturate.

Benchmark driver for the BHL OCR board, ported from `uv-scripts/ocr/unlimited-ocr2.py`
(the 16-pages-per-request document recipe) and reduced to ONE page per request: the
benchmark scores pages, and a page read alone takes the model card's single-page path.
`baidu/Unlimited-OCR-2` (3.3B MoE, ~0.5B active, MIT, released 2026-10-27) is the
successor to `baidu/Unlimited-OCR`, which is already on the board.

Run on HF Jobs (the script installs Baidu's SGLang wheel and starts the server itself;
--image provides the SGLang runtime the wheel was built against):

  hf jobs uv run --detach --flavor rtx-pro-6000 -s HF_TOKEN --timeout 4h \\
      --image lmsysorg/sglang:v0.5.20-cu130 \\
      unlimited-ocr2-port.py -- --input-dataset <benchmark> --id-column PageID \\
        --model-revision <sha> --output hf://buckets/<owner>/<bucket>/<prefix>/

Output layout: `part-*.parquet` with rows `{id, raw, model, prompt_tokens,
completion_tokens, finish_reason, error}`, keyed by the input row id. `raw` is the
completion VERBATIM: `<BLOCK>type[x1, y1, x2, y2]<FORK>text` blocks with 0-1000 boxes,
`<LEVELn>` on titles and `<MERGE>` between joined blocks. Turning that into scored text
is runners/normalize_outputs.py's job (Baidu's own converter, vendored there), so the
rule can change without paying for the GPUs again.

No vLLM support exists for v2 (the v1 driver's vLLM path does not load it). Baidu ships
a patched SGLang wheel in the model repo; its pins (torch 2.13.0, transformers 5.12.1,
flashinfer 0.6.18 cu13) match `lmsysorg/sglang:v0.5.20-cu130` exactly, so the wheel is
installed `--no-deps` into that image's own Python and the server runs from there.

The SERVING dict below is the per-model tuning prior. Agents can `ast.literal_eval` it
without running the script; the script itself consumes it, so it cannot drift.
"""

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys

# Serving values for baidu/Unlimited-OCR-2. Per-value provenance:
# - image `lmsysorg/sglang:v0.5.20-cu130` + wheel: recipe-inherited (uv-scripts
#   unlimited-ocr2.py, validated on Jobs 2026-10-27). The wheel is read from the model
#   repo at the PINNED revision (`--model-revision`), so wheel and weights move together.
# - env + serve_args: card-verbatim launch command (tuned by Baidu for an 80 GB A800;
#   validated on rtx-pro-6000 and a100-large). The doc-window env vars only act on
#   multi-page requests and are inert here; kept so the server is the card's server.
# - crop_mode True (gundam: dynamic tiles + a global view): card-verbatim single-page
#   default ("A single page is encoded with dynamic tiling (gundam mode) by default").
#   The recipe's base mode is the card's MULTI-page default and does not apply.
# - prompt "<PAGE><image>Multi page parsing.": card-verbatim. The card builds the same
#   prompt for one page as for many (`"<PAGE><image>" * n + "Multi page parsing."`).
# - ngram guard 35 / window 128 / page guard 20000 / eos 1 / page token 128827:
#   card-verbatim client example (window 1024 is the multi-page setting; 128 is the
#   card's value for one image).
# - max_tokens 8000, temperature 0: card-verbatim (`tokens_per_page=8000`).
# - no client-side resize (house choice, as for the v1 driver): gundam picks its tile
#   grid FROM the input pixels, so downscaling would cost tiles, i.e. exactly the
#   resolution that makes small historical type legible. Images are only converted to
#   RGB when they are not already (see `encode_image`).
SERVING = {
    "model": "baidu/Unlimited-OCR-2",
    "served_name": "Unlimited-OCR-2",
    "image": "lmsysorg/sglang:v0.5.20-cu130",
    "wheel": "sglang-0.5.20.dev10%2Bg8d62f5fb9c-py3-none-any.whl",
    "extra_pip": ["fastokens==0.3.2"],
    "env": {
        "SGLANG_UOCR_EANABLE_DOC_LIMIT": "1",  # (sic) upstream spelling
        "SGLANG_UOCR_DOC_GUNDAM_PAGE_WINDOW": "10",
        "SGLANG_UOCR_DOC_BASE_PAGE_WINDOW": "10",
        "SGLANG_UOCR_MAX_PAGE_TOKENS": "20000",
        "SGLANG_OPT_UOCR_SAM_REL_POS_BIAS_MB": "64",
        "SGLANG_UOCR_SHM_WRAP_MODEL_DATA": "1",
        "SGLANG_UOCR_SHM_WRAP_MIN_MB": "1",
        "SGLANG_VLM_CACHE_SIZE_MB": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    "serve_args": [
        "--chat-template", "unlimited-ocr",
        "--tokenizer-backend", "fastokens", "--tokenizer-worker-num", "8",
        "--mem-fraction-static", "0.7",
        "--max-running-requests", "64",
        "--max-prefill-tokens", "2048",
        "--enable-custom-logit-processor", "--enable-streaming-session",
        "--allow-auto-truncate", "--disable-radix-cache",
        "--mm-preprocess-cache-size-mb", "0",
        "--cuda-graph-backend-decode", "full", "--cuda-graph-max-bs-decode", "64",
        "--cuda-graph-backend-prefill", "full",
        "--cuda-graph-max-bs-prefill", "16384",
        "--cuda-graph-bs-prefill", "1024", "2048",
        "--cuda-graph-config", '{"prefill": {"backend": "full", "full_prefill_max_req": 64}}',
    ],
    "prompt": "<PAGE><image>Multi page parsing.",
    "crop_mode": True,
    "max_tokens": 8000,
    "temperature": 0.0,
    "ngram_size": 35,
    "window_size": 128,
    "page_token_id": 128827,  # <PAGE>, = sparse_attention_config.page_token_id
    "eos_token_id": 1,
    "max_page_tokens": 20000,
}

# The card's no-repeat n-gram guard: DeepseekOCRNoRepeatNGramLogitProcessor.to_str(),
# i.e. json({"callable": dill.dumps(cls).hex()}). dill pickles an importable class by
# reference (module + name), so this is a fixed string and needs no sglang here.
NGRAM_PROCESSOR = json.dumps({"callable": (
    "80059559000000000000008c2a73676c616e672e7372742e73616d706c696e672e637573746f6d5f6c6f6769"
    "745f70726f636573736f72948c26446565707365656b4f43524e6f5265706561744e4772616d4c6f67697450"
    "726f636573736f729493942e")})

# Formats whose bytes go to the server untouched when the image is already RGB.
PASSTHROUGH_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg"}


def encode_image(value) -> tuple[str, str]:
    """Return `(mime, base64)` for one dataset image cell, at full resolution.

    Same rule as the v1 driver: already-RGB PNG/JPEG bytes pass through byte-for-byte;
    anything else (grayscale, bitonal, palette, CMYK, RGBA) is converted to RGB and
    re-encoded as lossless PNG.
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


def start_server_cmd(model_revision) -> list[str]:
    """Install Baidu's wheel into the image's sglang Python and return the server command."""
    # The SGLang image keeps sglang in its own venv (/opt/sglang/bin), first on PATH;
    # `uv run` puts this script's venv in front, so look past it.
    own_bin = os.path.realpath(os.path.join(sys.prefix, "bin"))
    path = os.pathsep.join(p for p in os.environ.get("PATH", "").split(os.pathsep)
                           if os.path.realpath(p) != own_bin)
    py = shutil.which("python3", path=path)
    if not py or subprocess.run([py, "-c", "import sglang"], check=False,
                                capture_output=True).returncode != 0:
        sys.exit(f"No Python with sglang found: run this on --image {SERVING['image']}")
    wheel = (f"https://huggingface.co/{SERVING['model']}/resolve/"
             f"{model_revision or 'main'}/wheel/{SERVING['wheel']}")
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    pip = ["uv", "pip", "install", "-q", "--python", py]
    subprocess.run([*pip, "--no-deps", wheel], check=True, env=env)
    subprocess.run([*pip, *SERVING["extra_pip"]], check=True, env=env)
    versions = subprocess.run(
        [py, "-c", "import sglang, torch, transformers; "
                   "print(sglang.__version__, torch.__version__, transformers.__version__)"],
        check=True, capture_output=True, text=True, env=env).stdout.split()
    print(f"versions sglang={versions[0]} torch={versions[1]} transformers={versions[2]}",
          file=sys.stderr, flush=True)
    os.environ.update(SERVING["env"])  # inherited by the server process
    cmd = [py, "-m", "sglang.launch_server", "--model-path", SERVING["model"],
           "--served-model-name", SERVING["served_name"], *SERVING["serve_args"],
           "--enable-metrics", "--host", "127.0.0.1", "--port", "8000"]
    # The served checkpoint is pinned at boot, not read back from the Hub afterwards.
    if model_revision:
        cmd += ["--revision", model_revision]
    return cmd


def main():
    ap = argparse.ArgumentParser(description="Unlimited-OCR-2 batch OCR via saturate")
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
                    help="Pin the SERVED model checkpoint sha (and the wheel read from it)")
    ap.add_argument("--id-column", default=None,
                    help="Column to use as row id (default: split-index ids)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=parse_shard, default=(0, 1), metavar="RANK/WORLD",
                    help="Strided fan-out across jobs, e.g. 2/8 (default: 0/1)")
    ap.add_argument("--max-tokens", type=int, default=SERVING["max_tokens"])
    ap.add_argument("--retry-errors", action="store_true",
                    help="Re-admit rows whose only record is an error row")
    args = ap.parse_args()

    import saturate.transport
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
        mime, b64 = encode_image(row[args.image_column])
        return {
            "model": SERVING["served_name"],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": SERVING["prompt"]},
            ]}],
            "max_tokens": args.max_tokens,
            "temperature": SERVING["temperature"],
            "skip_special_tokens": False,  # <BLOCK>/<FORK> markers are the structure
            "images_config": {"crop_mode": SERVING["crop_mode"]},
            "custom_logit_processor": NGRAM_PROCESSOR,
            "custom_params": {
                "ngram_size": SERVING["ngram_size"],
                "window_size": SERVING["window_size"],
                "page_token_id": SERVING["page_token_id"],
                "eos_token_id": SERVING["eos_token_id"],
                "max_page_tokens": SERVING["max_page_tokens"],
            },
        }

    def parse(row, body):
        # Envelope check, not text cleaning: a response with no choices or a null
        # `content` field never held a completion to store.
        choice = (body.get("choices") or [None])[0]
        if not choice or choice.get("message", {}).get("content") is None:
            raise ValueError(f"no content in response: {str(body)[:200]}")
        usage = body.get("usage") or {}
        return {
            "raw": choice["message"]["content"],
            "model": SERVING["model"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            # "length" is the loop signature — kept so a truncated page is auditable.
            "finish_reason": choice.get("finish_reason"),
        }

    if args.model_revision:
        SERVING["model_revision"] = args.model_revision
    # A looping page decodes the full 8000 tokens behind up to 63 others; saturate
    # 0.2.0 caps a request's retries at 300 s in total. Match the read timeout.
    saturate.transport.RETRY_BUDGET_S = 3600.0
    with Engine(SERVING["model"], engine="sglang", cmd=start_server_cmd(args.model_revision)) as endpoint:
        stats = pump(rows, to_request, parse, endpoint, args.output,
                     window=Auto(initial=32, max_limit=64),
                     shard=(rank, world), read_timeout=3600,
                     retry_errors=args.retry_errors)

    print(f"{args.output} "
          f"({stats.rows_processed} ok, {stats.rows_failed} error rows)", file=sys.stderr)
    print("PORT unlimited-ocr2 " + stats.to_json(), flush=True)


if __name__ == "__main__":
    main()
