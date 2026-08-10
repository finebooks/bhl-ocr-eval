<!-- DRAFT 2026-08-04 — for Daniel's voice pass before the repo goes public. Numbers regenerate
deterministically from cached raw output (frozen norm 2026-07-20a, scorer 3.0, postproc 3). -->

# Results — what the board shows

The findings walkthrough for the current leaderboard. [GLOSSARY.md](GLOSSARY.md) explains how to
read each number; [DESIGN.md](DESIGN.md) explains why the numbers are built this way.

## One board

**2,165 pages, 14 models.** Every page of the IMPACT-BHL ground truth
(via [`finebooks/bhl-impact-gt`](https://huggingface.co/datasets/finebooks/bhl-impact-gt)),
including 428 sparse/blank pages (plates, blanks, near-empty pages), scored for every model on the
board. We ran 16 models; two are held back — see *What counts as a score*, below.

**A page can end three ways.** Scored; *looped* — the model hit its token cap mid repetition loop,
which excludes the page from that row's aggregates and is counted in the Loop % column; or
*errored* — the producer failed, and a single producer error makes the row ineligible for ranking
(fail-closed). Every row below has zero producer errors.

## Headline results (full 2,165 pages · scored 2026-08-04)

| Rank | Model | CER reading | 95% CI | Content CER | Sparse/blank CER | Recall | Loop % |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | rednote-hilab/dots.ocr | 0.0235 | [0.0156, 0.0318] | 0.0220 | 1.79 | 0.9704 | 3.88 |
| 2 | rednote-hilab/dots.mocr | 0.0237 | [0.0143, 0.0334] | 0.0216 | 2.23 | 0.9836 | 0.88 |
| 3 | ATH-MaaS/OvisOCR2 | 0.0305 | [0.0232, 0.0357] | 0.0280 | 2.79 | 0.9614 | 2.54 |
| 4 | PaddlePaddle/PaddleOCR-VL-1.6 | 0.0392 | [0.0259, 0.0499] | 0.0367 | 2.69 | 0.9415 | 6.47 |
| 5 | allenai/olmOCR-2-7B-1025-FP8 | 0.0432 | [0.0199, 0.0694] | 0.0302 | 13.64 | 0.9573 | 0.55 |
| 6 | lightonai/LightOnOCR-2-1B | 0.0489 | [0.0269, 0.0706] | 0.0351 | 15.79 | 0.9688 | 2.08 |
| 7 | zai-org/GLM-OCR | 0.0491 | [0.0256, 0.0720] | 0.0236 | 26.93 | 0.9674 | 0.51 |
| 8 | Qwen/Qwen3.5-9B | 0.0507 | [0.0292, 0.0771] | 0.0314 | 19.98 | 0.9587 | 1.52 |
| 9 | baidu/Qianfan-OCR | 0.0575 | [0.0312, 0.0876] | 0.0404 | 17.89 | 0.9540 | 1.06 |
| 10 | baidu/Unlimited-OCR | 0.0604 | [0.0351, 0.0820] | 0.0451 | 16.19 | 0.9537 | 0.46 |
| 11 | deepseek-ai/DeepSeek-OCR | 0.0617 | [0.0467, 0.0754] | 0.0590 | 2.96 | 0.9206 | 0.55 |
| 12 | deepseek-ai/DeepSeek-OCR-2 | 0.0620 | [0.0453, 0.0833] | 0.0600 | 2.09 | 0.9262 | 0.32 |
| 13 | tesseract-5 | 0.0642 | [0.0421, 0.0794] | 0.0591 | 5.49 | 0.9210 | 0.00 |
| 14 | ds4sd/SmolDocling-256M-preview | 0.1965 | [0.1353, 0.2511] | 0.1946 | 2.18 | 0.8519 | 5.31 |

Where 95% CIs overlap, read the models as tied — the top two are statistically identical, and most
mid-table neighbours overlap. Loop % carries a selection bias: looped pages are excluded from that
row's other numbers, so a high loop rate means the row is scored with its hardest pages removed
(the #1 row is scored on 96.1% of the corpus, PaddleOCR-VL-1.6 on 93.5%, DeepSeek-OCR-2 on 99.7%).
Read it alongside CER, never on its own.

## Reading this board

- **CIs first, ranks second.** Where the 95% CIs overlap — the top two do, and most mid-table
  neighbours do — read the models as tied.
- **Loop %** is the repetition-loop rate, the characteristic neural failure on this material
  (Tesseract's is 0.00). Looped pages are excluded from that row's other numbers, so a high rate
  also means the row is scored with its hardest pages removed — read it alongside CER, never alone.
- **Sparse CER** is where models diverge most: strong content OCR can coexist with runaway
  hallucination on near-empty pages (compare GLM-OCR's 0.0236 content vs 26.93 sparse). If your
  collection has plates and blanks — most real collections do — weight this column.
- **The two CER lanes** separate formatting from misreading: *reading* folds long-s, case and
  ligatures; *diplomatic* keeps them. A wide gap (LightOnOCR-2: 0.0864 dip / 0.0489 read) means
  normalization choices, not reading errors.
- **Reads is a policy, not a quality**: whether a model transcribes page furniture (running heads,
  page numbers). The body-only CER neither rewards nor penalises it — but if you need furniture
  preserved, this column matters as much as CER.
- **Params are measured**, from each repo's safetensors index, not model names: GLM-OCR "0.9B" is
  1.33B; olmOCR-2-"7B" is 8.29B.

What the run *found* — specialists vs the generalist, what triggers repetition loops, how serving
configuration moved one model from the bottom of the board to the top — is written up in the
accompanying blog post rather than here.

## The Qwen decoding ablation

Qwen3.5-9B is the only board model whose card recommends sampling (T=0.7, top_p 0.8,
presence_penalty 1.5). Every OCR specialist runs greedy. Three complete 2,165-page variants:

| Variant | Config | CER reading | Loop % |
|---|---|---:|---:|
| greedy, no penalties | T=0.0 | 0.0440 | 2.31 |
| **greedy + card penalties** *(board row)* | T=0.0, presence 1.5 | 0.0507 | 1.52 |
| full card sampling | T=0.7, top_p 0.8, presence 1.5 | 0.0533 | 0.79 |

The board row keeps greedy decoding (comparable with every other row) while restoring the card's
presence penalty — plain greedy would silently strip an anti-repetition guard the card specifies.
Note how the three variants order: the *lowest* CER belongs to the variant that loops most, because
looped pages are excluded and they are disproportionately hard pages — the Loop % selection bias,
visible inside a single model. On any of the three variants, Qwen3.5-9B trails the leading
specialists on this corpus.

## Caveats

- **Wide CIs by design.** The ground truth spans six volumes (books), and the bootstrap resamples
  volumes, not pages — so CIs are honest about book-to-book variation and correspondingly wide.
  Where CIs overlap, treat models as tied.
- **Loop % is also a selection effect.** See the note under the table: excluding looped pages
  flatters heavy loopers' other numbers. The column exists so that this is visible.
- **Numbers from earlier scorer versions are not comparable.** The scorer moved to count-aware
  semantics (3.0), the normalizer was frozen at `2026-07-20a`, and post-processing is versioned
  (registry `3`, stamped on every scorecard row and refused board-wide if mixed). Every number
  here was re-scored from cached raw model output under that frozen trio. If you saw different
  values in earlier working notes — including a dots.mocr at 0.2665 and DeepSeek rows scored on
  raw grounding markup — this is why.
- **Ground-truth caveats** (known transcription conventions and their handling) are documented in
  [DESIGN.md](DESIGN.md).
- **Scoring fails closed.** A producer error or missing page makes a model ineligible for ranking;
  its metrics may still be shown as conditional diagnostics. No model on the current board is in
  that state — two models that could not meet the bar are held off the board instead (below).

## What counts as a score

**We ran the inference ourselves, for every model on the board** — a pinned container image, a
pinned model revision, a pinned script commit, and the id of the job that produced each score,
with the per-model records generated from the runs themselves in
[`data/full-2026-08/provenance/`](data/full-2026-08/provenance/). That is what lets a score be
attributed to the named model: a hosted inference router does not record which provider served a
request, at what quantization, or under what serving configuration — and character error rate is
sensitive to all three.

surya-ocr-2 and PP-OCRv6 join the board in v1.1, once their runs meet the same bar.

## Where the numbers come from

Every scorecard traces to cached raw model output (the run bucket, `hf://buckets/finebooks/bhl-ocr-runs/full-2026-08/`)
plus a pinned ground-truth revision, so any score can be reproduced without re-running inference: consolidation
(`runners/consolidate_run.py`) → versioned post-processing (`runners/normalize_outputs.py`) →
one scoring pass (`runners/score_dataset.py`) → one report (`runners/leaderboard.py`), written
into the board page by `scripts/bake_leaderboard.py`. The ground truth is the IMPACT-BHL expert
transcription set, CC-BY, credited to the IMPACT Centre of Competence / BHL-Europe — see the
source dataset [`finebooks/bhl-impact-gt`](https://huggingface.co/datasets/finebooks/bhl-impact-gt).
