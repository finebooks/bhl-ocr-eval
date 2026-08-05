# drivers/ — the inference side

The rest of this repo is a client and a scorer: it takes OCR output someone else produced and
scores it, and it never manages a server. These files are the exception, and they are here for one
reason — **a score has to be traceable to the exact code that produced it.**

Each driver runs one model over the benchmark as a single HF Job: it starts `vllm serve` in-process,
pumps every page through it with [saturate](https://github.com/davanstrien/saturate), and writes
resumable parquet. One model, one job, one command:

```bash
hf jobs uv run --detach --flavor a10g-small -s HF_TOKEN --timeout 4h \
    --image vllm/vllm-openai:latest \
    drivers/olmocr2-port.py -- --source <benchmark> --output <destination>
```

## What a driver is responsible for — and what it is not

A driver stores the model's completion **verbatim**, in a `raw` column. It does no text cleaning at
all. Every model-specific transform that turns `raw` into the scored `markdown` column — dropping
bbox placeholder blocks, stripping grounding markup, converting DocTags — lives in
[`../runners/normalize_outputs.py`](../runners/normalize_outputs.py), which is versioned
(`POSTPROC_VERSION`) and re-runnable over cached output.

That split is deliberate. Post-processing rules are judgement calls, and they used to be applied at
inference time, which meant correcting one cost a GPU run. Now a disputed rule costs a re-score.

The line between the two is *"is this a fact about the response, or a decision about the text?"*
Parsing YAML front matter into its own column is a fact. Deciding to drop it from the scored text is
a decision. The first stays here; the second lives in the normalizer.

## The SERVING dict

Every driver opens with a `SERVING` dict whose comment block gives **per-value provenance**: for each
setting, whether it is verbatim from the model card, inherited from the `uv-scripts/ocr` recipe the
driver was ported from, or a house choice — and why. Read it before changing anything.

This exists so the benchmark can say which settings came from a model's authors and which came from
us. Two cases worth knowing: `google/gemma-4-12B-it` and `Qwen/Qwen3.5-9B` are general-purpose VLMs
with **no** card-recommended OCR prompt, so their prompt is entirely ours and their drivers carry
several declared variants — and because those two rows exist to be compared with each other, they
share one byte-identical `PROMPTS` dict, which must be edited in both files or neither; and several
drivers deliberately *drop* flags their source recipes passed
(`--trust-remote-code` where the repo ships no custom code, an obsolete `--hf-overrides` remap)
because an inherited flag is not a known configuration.

Each driver also asserts its own context maths at import time — image tokens plus `max_tokens` must
fit inside `max_model_len`, or the server 400s every request. Four of these drivers were found to
have inherited an invalid budget from their source recipe, so the assert is not decorative.

## Relationship to uv-scripts/ocr

These began as ports of recipes in
[`uv-scripts/ocr`](https://huggingface.co/datasets/uv-scripts/ocr), which remains the home for
general-purpose, reusable OCR recipes. The copies here are **benchmark-tuned** — fixed output
columns, fixed error semantics, raw-passthrough — and this directory is their source of truth. They
are not kept in sync with uv-scripts and should not be edited to match it.
