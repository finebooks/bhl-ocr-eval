<!-- DRAFT 2026-08-04 — for Daniel's voice pass before the repo goes public. Numbers regenerate
deterministically from cached raw output (frozen norm 2026-07-20a, scorer 3.0, postproc 3). -->

# Results — what the board shows

The findings walkthrough for the current leaderboard. [GLOSSARY.md](GLOSSARY.md) explains how to
read each number; [DESIGN.md](DESIGN.md) explains why the numbers are built this way.

## One board

**2,165 pages, 24 models.** Every page of the IMPACT-BHL ground truth
(via [`finebooks/bhl-impact-gt`](https://huggingface.co/datasets/finebooks/bhl-impact-gt)),
including 428 sparse/blank pages (plates, blanks, near-empty pages), scored for every model on the
board. We ran 26 models; two are held back — see *What counts as a score*, below.

**A page can end three ways.** Scored; *looped* — the model hit its token cap mid repetition loop,
which excludes the page from that row's aggregates and is counted in the Loop % column; or
*errored* — the producer failed, and a single producer error makes the row ineligible for ranking
(fail-closed). Every row below has zero producer errors.

## Headline results (full 2,165 pages · scored 2026-08-04; four rows added 2026-10-23, one 2026-10-25, two 2026-10-26, one 2026-10-27)

| Rank | Model | CER reading | 95% CI | Content CER | Sparse/blank CER | Recall | Loop % |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | rednote-hilab/dots.ocr | 0.0235 | [0.0156, 0.0318] | 0.0220 | 1.79 | 0.9704 | 3.88 |
| 2 | rednote-hilab/dots.mocr | 0.0237 | [0.0143, 0.0334] | 0.0216 | 2.23 | 0.9836 | 0.88 |
| 3 | **datalab-to/chandra-ocr-2** (new) | 0.0285 | [0.0198, 0.0380] | 0.0256 | 3.14 | 0.9810 | 0.09 |
| 4 | ATH-MaaS/OvisOCR2 | 0.0305 | [0.0232, 0.0357] | 0.0280 | 2.79 | 0.9614 | 2.54 |
| 5 | **lightonai/LightOnOCR-3-4B** (new) | 0.0312 | [0.0238, 0.0404] | 0.0290 | 2.28 | 0.9711 | 0.42 |
| 6 | **baidu/Unlimited-OCR-2** (new) | 0.0329 | [0.0258, 0.0417] | 0.0311 | 1.90 | 0.9675 | 0.14 |
| 7 | kraken/PP-OCRv6-medium | 0.0338 | [0.0237, 0.0442] | 0.0316 | 2.34 | 0.9760 | 0.00 |
| 8 | **ATH-MaaS/OvisOCR3** (new) | 0.0382 | [0.0283, 0.0481] | 0.0284 | 10.38 | 0.9650 | 0.46 |
| 9 | PaddlePaddle/PaddleOCR-VL-1.6 | 0.0392 | [0.0259, 0.0499] | 0.0367 | 2.69 | 0.9415 | 6.47 |
| 10 | allenai/olmOCR-2-7B-1025-FP8 | 0.0432 | [0.0199, 0.0694] | 0.0302 | 13.64 | 0.9573 | 0.55 |
| 11 | **lightonai/LightOnOCR-3-0.8B** (new) | 0.0455 | [0.0308, 0.0571] | 0.0429 | 2.80 | 0.9659 | 0.88 |
| 12 | lightonai/LightOnOCR-2-1B | 0.0489 | [0.0269, 0.0706] | 0.0351 | 15.79 | 0.9688 | 2.08 |
| 13 | zai-org/GLM-OCR | 0.0491 | [0.0256, 0.0720] | 0.0236 | 26.93 | 0.9674 | 0.51 |
| 14 | Qwen/Qwen3.5-9B | 0.0507 | [0.0292, 0.0771] | 0.0314 | 19.98 | 0.9587 | 1.52 |
| 15 | baidu/Qianfan-OCR | 0.0575 | [0.0312, 0.0876] | 0.0404 | 17.89 | 0.9540 | 1.06 |
| 16 | baidu/Unlimited-OCR | 0.0604 | [0.0351, 0.0820] | 0.0451 | 16.19 | 0.9537 | 0.46 |
| 17 | **nvidia/Nemotron-3.5-Parse** (new) | 0.0611 | [0.0485, 0.0761] | 0.0480 | 13.88 | 0.9202 | 6.56 |
| 18 | deepseek-ai/DeepSeek-OCR | 0.0617 | [0.0467, 0.0754] | 0.0590 | 2.96 | 0.9206 | 0.55 |
| 19 | deepseek-ai/DeepSeek-OCR-2 | 0.0620 | [0.0453, 0.0833] | 0.0600 | 2.09 | 0.9262 | 0.32 |
| 20 | **datalab-to/indra-ocr** (new) | 0.0642 | [0.0257, 0.1010] | 0.0621 | 2.30 | 0.9493 | 0.00 |
| 21 | tesseract-5 | 0.0642 | [0.0421, 0.0794] | 0.0591 | 5.49 | 0.9210 | 0.00 |
| 22 | ds4sd/SmolDocling-256M-preview | 0.0660 | [0.0342, 0.1007] | 0.0649 | 1.21 | 0.8892 | 5.22 |
| 23 | **SoMarkAI/OXR-1.0** (new) | 0.1117 | [0.0334, 0.1790] | 0.1098 | 2.11 | 0.9551 | 0.00 |
| 24 | tiiuae/Falcon-OCR | 0.1425 | [0.0892, 0.1927] | 0.1164 | 27.66 | 0.9564 | 0.00 |

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

## The line recogniser

`kraken/PP-OCRv6-medium` is the first row that is not a single end-to-end model: kraken's bundled
`blla` segmenter finds the text lines and orders them, then a 15.9M-parameter CTC recogniser
transcribes each one. It ranks 7th on the headline (4th when it was added in September), with a
confidence interval overlapping most of the top ten. Three things about it are worth more than the rank.

**It wins the diplomatic lane outright.** CER diplomatic 0.0405, against 0.0503 for the next best
(dots.mocr) and 0.0560 for the model that leads the reading lane. The two lanes differ in whether
long-s, ligatures and case are folded before scoring, so diplomatic divided by reading measures
how much of a model's headline score depends on that folding. kraken's is 1.20x (0.0405/0.0338);
dots.ocr's is 2.38x (0.0560/0.0235) — more than half of its reading-lane score is the fold. The
generative models partly win the reading lane by modernising orthography as they read, and on a
board about historical print the diplomatic column is the one that says who transcribed the page
in front of them.

**It is scored on every page.** n = 2,165, loop rate 0.00 — a CTC recogniser has no token cap to
run into. The three rows above it are scored on 96.1%, 99.1% and 97.5% of the corpus respectively,
with their looped pages (disproportionately the hard ones) excluded from their own aggregates. The
comparison is not like-for-like in kraken's favour.

**Its weakness is segmentation, not recognition.** Two failures show up, and neither is visible in
the error or loop columns:

- *Dropped decorated initials.* 90 lines across 77 pages (3.6% of pages) begin exactly one capital
  short - "HIS is so rare" for THIS, "EEBOHM observe" for SEEBOHM, "ESPECTING the" for RESPECTING.
  The drop capital is its own region, and the line that reaches the recogniser starts after it.
- *Captions.* Global token evidence 0.53, against 0.84 for dots.ocr - the one region type where
  kraken is clearly beaten. It leads on body text, footnotes, section headers and both page
  furniture types (page headers 0.91 vs dots.ocr's 0.29).

Separately, 36 lines across the whole corpus were lost to polygonisation failures - kraken drops a
line it cannot turn into a polygon rather than failing the page. That is 0.05% of lines, but it is
silent: it appears only as a warning in the job log, never in the output, so it is recorded here
rather than inferred from the scores. See the driver docstring for how to measure it.

A note on throughput, which this board deliberately does not score. The model card claims the
family offers VLM-level accuracy "with vastly higher throughput". The recogniser is indeed fast -
0.039 s per line on an A10G. The pipeline is not: 12.3 s per page against 3-5 s for most
specialists on identical hardware, because ~90% of the time is CPU-side segmentation (a five-scale
ridge filter over an upsampled heatmap, which runs even on blank pages) and the run was serial per
page while the VLM rows had a concurrency window. The recogniser's claim survives; the pipeline's
does not, as deployed by default.

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

## Four rows added in October 2026

Run 2026-10-23 on the same pinned benchmark revision, norm, scorer and post-processing version as
the rest of the board, so the numbers are directly comparable. Combining all twenty scorecards
reproduces every existing row's numbers exactly. Provenance for each row is in
`data/full-2026-08/provenance/` (`chandra-ocr2`, `lighton-ocr3-4b`, `lighton-ocr3-0.8b`, `indra-ocr`).

**datalab-to/chandra-ocr-2 (3rd, 0.0285).** The lowest measured loop rate of any generative model
on the board (0.09%, 2 pages; Falcon-OCR's 0.00 is by construction, its pipeline reports no finish
reason), and the fourth-best content CER. Its CI overlaps both dots rows. The driver
ports Datalab's own vLLM client (`chandra-ocr` v0.2.0: `ocr_layout` prompt, image scaling) with one
deliberate difference: the vendor client resamples a looping page up to six times at a raised
temperature, and this driver makes one greedy attempt, like every other row. The model answers in
HTML layout blocks; the normalizer applies the vendor's own HTML-to-markdown conversion, minus
figure blocks (the vendor writes a hashed image link and an alt-text description into them).
Its diplomatic CER (0.093) is much worse than its reading CER. Part of that is markup, which the
diplomatic lane does not strip: the converted text carries about 22,000 HTML tags (tables stay
HTML, `<sup>`/`<sub>`) and 3,500 markdown escapes. Weights: modified OpenRAIL-M (free for research,
personal use and startups under $2M; not for use competing with Datalab's API).

**lightonai/LightOnOCR-3-4B (5th, 0.0312) and -0.8B (9th when added, 11th now; 0.0455).** Both improve on
LightOnOCR-2-1B (0.0489), and both cut its sparse/blank hallucination by a factor of five or more
(2.28 and 2.80 against 15.79). The 4B is better than LightOnOCR-2 on every volume; the Latin
volume is the hardest for both sizes (0.091 and 0.115). Both run the card's request (image only,
2048 px, temperature 0.2 / top_p 0.9) on `vllm/vllm-openai:v0.30.0`, an exception to the
board-wide pin, because LightOn's repo says 0.27.1 makes these models loop. Apache-2.0.

**datalab-to/indra-ocr (17th when added, 20th now; 0.0642).** A CPU pipeline of small ONNX models: the whole 2,165-page
run took one 8-vCPU job. It ties Tesseract on the headline, reads French best of any row
(`histoirenaturell10cuvi` 0.015), and is weakest on the Conchologia plate volume (0.120). It
recovers only 8% of caption tokens (the 4B recovers 70%). The package's markdown drops only page
headers and footers by label, so the caption loss happens in its detection, not in
post-processing. Weights: modified OpenRAIL-M (free for research, personal use and startups under
$20M; not for use competing with Datalab's API).

**nvidia/Nemotron-3.5-Parse (15th when added, 17th now; 0.0611; added 2026-10-25).** Run on the same pinned benchmark,
norm, scorer and post-processing version; combining all 21 scorecards leaves every other row's
numbers unchanged. Three things hold it back, in order of size.

*Blank pages.* On blank and near-blank pages it often writes fluent modern English that is not on
the page ("…provided by the National Natural Science Foundation, which is based on the National
Natural Science Foundation…") and loops until the token cap. 142 pages (6.6%) loop, 141 of them
sparse/blank (a third of that stratum; 99 are Conchologia plates), and the sparse/blank pages that
do finish score 13.88 against 1.8–3.1 for the other recent specialists. Looped pages are excluded
from the aggregates, so the headline is computed on an easier page set than most rows.

*Accents.* It drops diacritics the page has: é→e alone accounts for about 9,800 substitutions on
content pages, â→a 8,200, è→e 4,600, with ä, ü, ö, ê, à, æ and œ following. The raw completions
already lack them, so this is the model, not the renderer. A diagnostic re-score with diacritics
folded on both sides (`scripts/diag_accent_folded.py`; not a board metric, the reading lane keeps
accents on purpose) lowers its CER by 0.0094, against 0.0003–0.0044 for every other row, and
would have placed it 13th when added (15th now); on content pages it would score 0.0386, level with LightOnOCR-3-0.8B
(0.0398). The same harness with folding switched off reproduces the board exactly. The card's repetition penalty is
not the cause: on 75 accent-heavy content pages (25 each from the French, German and Conchologia
volumes) it keeps 30.5% of the GT's accents at 1.1 and 31.1% at 1.0. The loss depends on the
language: German umlauts survive (73–87%), French accents mostly do not (13–23%), and Conchologia's
Latin â/æ almost never do (5%). With accents folded those pages score about 0.008, so the letters
themselves are read well.

*Italics.* It marks italics with underscores (`_Corvus corax_`, 5,881 spans), which the frozen
scorer does not strip. The normalizer unwraps them (`strip_underscore_emphasis`), as it removes
other models' markup; the GT holds one underscore across all content pages. That moved the row
from 0.0648 to 0.0611.

On content pages it scores 0.0480. It reads French best (0.045) and Latin worst (0.089), and
recovers 54% of caption tokens. Every decoding and serving setting is the card's: the six-token
default prompt, temperature 0, top-k 1, repetition penalty 1.1, 8,744 output tokens, special
tokens kept, and the card's H100 `vllm serve` flags on vLLM 0.26.0, NVIDIA's stated reference
runtime. The architecture needs NVIDIA's vLLM plugin, installed into the board-pinned image under
NVIDIA's own inference lock (`docker/nemotron-parse-35/`). Plugin defaults are kept: one-token MTP
speculative decoding (lossless under greedy decoding) and FlashAttention 3, hence an H200. This is
the card's single-pass mode; the double and triple passes behind its headline ParseBench number
re-read table and chart crops, which this corpus barely has. The normalizer applies NVIDIA's own
renderer (`render_markdown`, default options) at the served revision; a plain strip of the
coordinate and class tokens scores slightly worse. Weights and plugin: OpenMDW-1.1; tokenizer MIT.

## Two rows added 2026-10-26

Both rows use the same pinned benchmark, norm, scorer and post-processing version as the rest of
the board. Combining all 23 scorecards leaves every earlier row's numbers unchanged.

**ATH-MaaS/OvisOCR3 (7th when added, 8th now; 0.0382).** The 0.8B successor to OvisOCR2, run with the card's
`document_markdown` prompt, which is OvisOCR2's prompt word for word. On content pages the two are
level (0.0284 against 0.0280). The whole gap is on sparse/blank pages (10.38 against 2.79): on 111
of the 428, mostly Conchologia plates (70) and Birds plates (39), it writes a note about the page
instead of nothing — "Based on the image provided, there is no readable text, charts, or formulas
present. The image appears to be a blank page with a pale yellow background… Therefore, the
extracted content is empty." Its CI overlaps OvisOCR2's and most of the rows from 3rd to 9th.
10 pages (0.46%) loop. Board-wide `vllm/vllm-openai:v0.26.0`. Apache-2.0.

**SoMarkAI/OXR-1.0 (22nd when added, 23rd now; 0.1117).** A pipeline, like Falcon-OCR: a D-FINE layout model finds
regions, a 1.5B VLM reads each one and sets the reading order, and the vendor API assembles the
page. The board scores the vendor's `/v1/parse/sync` endpoint with its default settings. The CI is
the widest on the board ([0.0334, 0.1790]) because a minority of pages carry most of the error:
the worst 50 pages hold 60% of all edits. Two things put it there, and both are extra text rather
than misreading. Recall (0.955) is level with olmOCR-2, Qwen3.5-9B and the Baidu rows (0.954–0.959), and on sparse/blank pages it
mostly writes nothing (2.11).

*Text read twice.* On 273 of the 1,737 content pages the layout model returns a region together
with smaller regions inside it, and the pipeline reads both, so the same passage appears twice.
Those pages carry 64% of the row's edits. Conchologia is hit hardest: 140 of its 419 content pages,
43 of them coming out at more than 1.5 times the reference length. The pipeline has an opt-in
cleanup for this (`full_overlap_dedupe`), which is off by default and runs a different chain of
clean-ups when on. The board keeps the shipped default, as for Falcon-OCR; the vendor's benchmark
notes do not say which setting their published numbers used.

*Runaway leader dots.* On three Trudy contents and index pages one region comes back with thousands of
`. . .` — 16,421 characters for a three-line entry whose dot leader has about 20 dots — and the
page number never appears. That is a repetition loop up to the per-region generation cap, but the
API reports no finish reason, so, as with Falcon-OCR, it cannot be counted in Loop % and is scored
as text. These three pages carry 18% of the row's edits.

The vendor's plugin and entrypoint require vLLM 0.25.1, so the row runs on an image built
unmodified from the vendor's `docker/Dockerfile.allinone`, an exception to the board-wide pin.
Weights are downloaded at the pinned revision when the job starts rather than mounted: on
2026-10-26 the Jobs model mount served some small repo files with the wrong size, and the vendor's
plugin install failed on a truncated `pyproject.toml`. Licence: Apache-2.0 with additional
commercial-use thresholds.

## One row added 2026-10-27

Same pinned benchmark, norm, scorer and post-processing version as the rest of the board.
Combining all 24 scorecards leaves every earlier row's numbers unchanged. Provenance:
`data/full-2026-08/provenance/unlimited-ocr2.json`.

**baidu/Unlimited-OCR-2 (6th, 0.0329).** The successor to Unlimited-OCR (15th when this row was
added, 16th now; 0.0604), released the same day, and the largest version-to-version gain on the
board: the headline error falls by almost half and every volume improves. Conchologia drops from
0.091 to 0.028 and the Latin volume from 0.185 to 0.118, still the hardest volume for this row, as
for most. The other change is on sparse/blank pages: 1.90 against v1's 16.19, the third-lowest on
the board after SmolDocling (1.21) and dots.ocr (1.79). On most plates it returns one whole-page
`image` block and no text. Three pages (0.14%) loop, two of them sparse Latin pages. Its CI
overlaps every row from 1st to 16th, v1 included: with six volumes to resample, a halving on the
point estimate is still inside the volume-to-volume spread, so read the gain from the per-volume
numbers, which all move the same way.

Its median page CER (0.0173) sits with OvisOCR2 and OvisOCR3 (0.0186, 0.0175), well above dots.ocr,
chandra-ocr-2 and olmOCR-2 (0.003-0.006): few pages are bad, but two in three content pages carry
some error. On a sample of median pages the edits are small and spread out: a modern spelling for a historical one
("Brunsvic" read as "Brunswick", "Massachusets" as "Massachusetts"), a misread letter in a
small-caps heading ("SYLVIINÆ" as "SYLVILNÆ"), and a drop capital split from its word ("AN" as
"A N"). Caption tokens recovered: 54%, about the same as Qianfan-OCR and Nemotron-3.5-Parse, and up
from v1's 45%.

The model reads many pages per request; this row sends one page per request, as the benchmark scores
pages, and uses the card's single-page settings: tiled ("gundam") encoding, the card's no-repeat
n-gram guard with window 128, 8,000 output tokens, temperature 0. Its raw output is
`<BLOCK>type[box]<FORK>text` blocks; the normalizer runs Baidu's own Markdown converter
(`utils/process_unlimitedOCR2.py`, vendored verbatim) and drops the figure placeholder links it
writes, as for OXR-1.0. Running heads and page numbers are kept, as for Unlimited-OCR. There is no
vLLM support for v2: the row runs Baidu's SGLang wheel from the model repo on
`lmsysorg/sglang:v0.5.20-cu130`, an exception to the board-wide pin. 2,165 pages took 42 minutes
on one RTX PRO 6000. MIT.

*Why one page per request.* A diagnostic run of the model's multi-page mode (16 consecutive pages
per request, base mode at 1280 px, as in the uv-scripts recipe) scored 0.247. Most of that is page
attribution around plates, not reading: in Conchologia, 108 of 297 pages with text carry the next
page's text, though every request returned the right number of pages. On the three volumes without
plates it is level with or better than one page per request in the same mode (Trudy 0.050 against
0.060), but still behind this row's tiled single pages, which are worth more than the context: one
page per request in base mode scores 0.0503. Not a board row; the driver and results are on the
`diag/uocr2-multipage` branch.

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

surya-ocr-2 and PaddlePaddle's PP-OCRv6 join the board in v1.1, once their runs meet the same
bar. Note that PaddlePaddle's PP-OCRv6 is a different model from the `kraken/PP-OCRv6-medium` row
above, which shares the architecture but not the publisher or the training corpus.

## Where the numbers come from

Every scorecard traces to cached raw model output (the run bucket, `hf://buckets/finebooks/bhl-ocr-runs/full-2026-08/`)
plus a pinned ground-truth revision, so any score can be reproduced without re-running inference: consolidation
(`runners/consolidate_run.py`) → versioned post-processing (`runners/normalize_outputs.py`) →
one scoring pass (`runners/score_dataset.py`) → one report (`runners/leaderboard.py`), written
into the board page by `scripts/bake_leaderboard.py`. The ground truth is the IMPACT-BHL expert
transcription set, CC-BY, credited to the IMPACT Centre of Competence / BHL-Europe — see the
source dataset [`finebooks/bhl-impact-gt`](https://huggingface.co/datasets/finebooks/bhl-impact-gt).
