# bhl-ocr-eval

A harness for evaluating OCR models — especially VLM-based OCR — on historical printed books. It
builds on the IMPACT-BHL ground truth
([`finebooks/bhl-eval-impact`](https://huggingface.co/datasets/finebooks/bhl-eval-impact)): 2,165
pages from six 18th–19th-century natural-history volumes, hand-transcribed by experts for the EU's
IMPACT digitisation programme with BHL-Europe.

The scores go to the
[BHL OCR Leaderboard](https://huggingface.co/spaces/finebooks/bhl-ocr-leaderboard). Point a runner
at a model's transcriptions and you get a scorecard broken out by axis, not one collapsed grade.

## Quickstart

Get OCR scored — the harness is a *client + scorer*, it never manages a server:

```bash
# 1. Serve the model yourself, then point the runner at it. Any OpenAI-compatible endpoint works;
#    routed endpoints are refused (see "What counts as a score" in RESULTS.md).
hf jobs run --detach --expose 8000 --flavor a10g-small -s HF_TOKEN \
  vllm/vllm-openai vllm serve ATH-MaaS/OvisOCR2 --max-model-len 32768
uv run runners/run_openai.py --models ATH-MaaS/OvisOCR2 \
  --base-url https://<job_id>--8000.hf.jobs/v1 --prompt-file ./transcription-prompt.txt

# 2. Score OCR some OTHER tool already produced — include its producer provenance. For an OCR
#    specialist, producer-run.json may be {"prompt":null,"note":"model-native OCR mode"}:
uv run runners/score_dataset.py --ocr ./some_run.parquet --ocr-col markdown --key-col PageID \
  --run-provenance-file ./producer-run.json

# Combine every runner's complete scorecard into one validated board:
uv run runners/leaderboard.py 'data/scorecards/*.parquet'
# Deliberately partial scorecards can be shown only as conditional/ineligible diagnostics:
uv run runners/leaderboard.py --allow-incomplete 'data/scorecards/*.parquet'
```

The scoring core is a small installed package (`pyproject.toml`); `uv sync` sets up the dev env and
`uv run pytest` runs the full suite (pytest + Hypothesis; deps + a pinned `uv.lock` come from the
`dev` group, so no `--with` flags). The `runners/` stay self-contained PEP-723 `uv run` scripts.
Run the frozen-core self-tests standalone with `uv run scoring/scorer.py` / `uv run scoring/normalizers.py`.

## Reading the board

The board covers every model on all 2,165 pages, including the sparse/blank stratum. Models are
ordered by micro-averaged character error rate, with volume-bootstrap confidence intervals
alongside; where the intervals overlap, the models are tied. Recall, over-extraction, furniture
policy, loop rate and the per-stratum and per-language breakdowns sit beside CER as columns,
because each carries signal the others hide: a model can read more words yet align worse.

We ran the inference for every row ourselves, under a pinned image, model revision, script commit
and job id, recorded per model in
[`data/full-2026-08/provenance/`](data/full-2026-08/provenance/). Scores from hosted inference
routers are not admissible; see [`RESULTS.md`](RESULTS.md), *What counts as a score*.

`scripts/bake_leaderboard.py` writes the current scores into
[`docs/leaderboard/index.html`](docs/leaderboard/index.html) and uploads the page to the Space.

## Design in brief

See [`DESIGN.md`](DESIGN.md) for the full methodology and trade-offs. The scoring core is versioned
and frozen: a published number never changes silently — the version is bumped and cached outputs
are re-scored instead. The other load-bearing decisions:

- **Don't collapse** — a model gets a *profile* across axes, never one grade.
- **CER orders; recall is the co-equal robustness check.** CER (micro-averaged, in an NFC *diplomatic*
  and an NFKC *reading* lane, following ISRI/UNLV and OCR-D `dinglehopper`) is the field-comparable
  headline used for ordering — but it is normalization- and alignment-*hostage*. Token **recall** is
  format-immune, order-free and needs no markup handling, so it is reported co-equally as the "did it
  read the words" check; the two can disagree, and reading both is the point.
- **Fairness via the GT schema, not a growing regex** — furniture (page numbers, running heads,
  catch-words, signatures) is identified from the typed GT and excluded from the headline, reported as
  directionless `furniture_global_token_recall` policy evidence, not accuracy.
- **Run once, re-score forever** — raw outputs are cached and checkpointed, so scoring and
  reporting re-run for free as the scorer evolves. Prompts and request settings are
  identity/provenance inputs; a response that stops for any reason other than natural completion is
  cached as an error page, never as a transcription; existing scorecards are never silently
  overwritten. (Operational detail — checkpointing, resume, spend confirmation, request settings —
  lives in `runners/run_openai.py`'s docstring.)
- **Completeness before comparison** — every row pins the GT revision to an immutable commit SHA and
  carries the benchmark's page-set fingerprint. Partial or errored rows can show conditional
  diagnostics but never enter the ranking. Mixed scorer/normalizer/post-processing/benchmark
  provenance, missing per-model run provenance, duplicate identities, and unequal page sets are all
  rejected.

Token coverage is count-aware: repeated GT/OCR words contribute according to their multiplicities,
and report-level recall/over-extraction/furniture values micro-pool their integer supports. Region and
furniture diagnostics are explicitly **global token-evidence proxies**, not spatial attribution;
per-label GT tokens are pooled on each page and overlapping labels may reuse the same OCR evidence.

The per-page scorecard parquet / `leaderboard.json` is the canonical output; the board, and
anything else, is a view over it.

## Docs

- [`RESULTS.md`](RESULTS.md) — the board's numbers and how to read them
- [`GLOSSARY.md`](GLOSSARY.md) — every number on the board, in plain language
- [`DESIGN.md`](DESIGN.md) — why the numbers are built this way
- [`AXES.md`](AXES.md) — the per-axis catalogue, generated from the registry
- [`MIGRATION.md`](MIGRATION.md) — the breaking-change contract for old scorecards

## Layout

```
scoring/         # the importable library (the only place scoring logic lives):
  normalizers.py #   frozen, versioned symmetric transforms + edit-op accounting (the only place norm lives)
  scorer.py      #   frozen: PageContext (built once) + declarative Axis registry; score_page -> per-page dict
  report.py      #   thick: validation, micro-avg CER, bootstrap CIs, ordering, eligibility, provenance
  gt_docling.py  #   pure body/furniture/region extraction from the docling column
  gt_score.py    #   one place a GT row + an OCR string become a scored scorecard row (shared by the runners)
runners/         # the REPEATABLE pipeline: prep (prep_sample · add_language) -> OCR
                 # -> consolidate_run (saturate parts -> one scoreable table; id->PageID,
                 #    seals producer errors and truncations) -> normalize_outputs (versioned
                 #    per-model post-processing) -> score_dataset -> board (leaderboard);
                 # plus run_openai (OpenAI-compatible endpoints) and gen_unit_tests
tests/           # pytest + Hypothesis invariants for the frozen core
scripts/         # maintenance + run plumbing: gen_axis_catalogue.py -> AXES.md ·
                 # crosscheck_mapping.py · bake_leaderboard.py -> docs/leaderboard ·
                 # emit_launch_commands.py (jobs from the plan, model shas pinned) ·
                 # emit_run_provenance.py (per-model provenance READ from each driver's
                 # SERVING dict, never hand-authored)
```

Plots live *outside* the harness — the product is the raw `leaderboard.json` / scorecard parquets.

## Breaking scorecard migration

Reporting fails closed on scorecard schema `2.1`, scorer `3.0`, the current normalizer and
post-processing registry version, and complete provenance: **old scorecard parquets are rejected,
never shimmed** — regenerate them by re-scoring retained raw OCR outputs (no model inference
needed). The full migration contract (cache identity, stratum classification, shard consolidation,
string-cell rules) is in [`MIGRATION.md`](MIGRATION.md).

## Ground truth & license

Ground truth: IMPACT Centre of Competence / BHL-Europe, **CC-BY 3.0** (via
[`impactcentre/groundtruth-bhl`](https://github.com/impactcentre/groundtruth-bhl)). Page images:
Biodiversity Heritage Library. This repo's **code** is MIT (see `LICENSE`); it contains no
ground-truth data.

Built and maintained by [Daniel van Strien](https://huggingface.co/davanstrien).
