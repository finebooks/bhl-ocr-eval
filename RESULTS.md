<!-- DRAFT 2026-08-04 — for Daniel's voice pass before the repo goes public. Numbers regenerate
deterministically from cached raw output (frozen norm 2026-10-30a, scorer 3.0, postproc 3). -->

# Results — what the board shows

The findings walkthrough for the current leaderboard. [GLOSSARY.md](GLOSSARY.md) explains how to
read each number; [DESIGN.md](DESIGN.md) explains why the numbers are built this way.

## One board

**2,165 pages, 26 models.** Every page of the IMPACT-BHL ground truth
(via [`finebooks/bhl-impact-gt`](https://huggingface.co/datasets/finebooks/bhl-impact-gt)),
including 428 sparse/blank pages (plates, blanks, near-empty pages), scored for every model on the
board. We ran 28 models; two are held back — see *What counts as a score*, below.

**A page can end three ways.** Scored; *looped* — the model hit its token cap mid repetition loop,
which excludes the page from that row's aggregates and is counted in the Loop % column; or
*errored* — the producer failed, and a single producer error makes the row ineligible for ranking
(fail-closed). Every row below has zero producer errors.

## Headline results (full 2,165 pages · scored 2026-08-04; four rows added 2026-10-23, one 2026-10-25, two 2026-10-26, one 2026-10-27, one 2026-10-28, one 2026-10-30; all rows re-scored 2026-10-30 under norm 2026-10-30a)

| Rank | Model | CER reading | 95% CI | Content CER | Sparse/blank CER | Recall | Loop % |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | rednote-hilab/dots.mocr | 0.0211 | [0.0137, 0.0281] | 0.0190 | 2.23 | 0.9836 | 0.88 |
| 2 | **datalab-to/chandra-ocr-2** (new) | 0.0214 | [0.0151, 0.0279] | 0.0184 | 3.14 | 0.9810 | 0.09 |
| 3 | rednote-hilab/dots.ocr | 0.0224 | [0.0156, 0.0294] | 0.0208 | 1.79 | 0.9704 | 3.88 |
| 4 | **lightonai/LightOnOCR-3-4B** (new) | 0.0247 | [0.0184, 0.0303] | 0.0225 | 2.28 | 0.9711 | 0.42 |
| 5 | allenai/olmOCR-2-7B-1025-FP8 | 0.0279 | [0.0121, 0.0439] | 0.0260 | 1.96 | 0.9573 | 0.55 |
| 6 | ATH-MaaS/OvisOCR2 | 0.0299 | [0.0228, 0.0353] | 0.0274 | 2.79 | 0.9614 | 2.54 |
| 7 | **baidu/Unlimited-OCR-2** (new) | 0.0329 | [0.0258, 0.0417] | 0.0311 | 1.90 | 0.9675 | 0.14 |
| 8 | **openbmb/MiniCPM-OCR-1.0** (new) | 0.0334 | [0.0151, 0.0468] | 0.0312 | 2.29 | 0.9634 | 0.00 |
| 9 | kraken/PP-OCRv6-medium | 0.0338 | [0.0237, 0.0442] | 0.0316 | 2.34 | 0.9760 | 0.00 |
| 10 | **datalab-to/surya-ocr-2.5** (new) | 0.0351 | [0.0299, 0.0413] | 0.0326 | 2.62 | 0.9672 | 0.00 |
| 11 | **lightonai/LightOnOCR-3-0.8B** (new) | 0.0363 | [0.0259, 0.0463] | 0.0337 | 2.80 | 0.9659 | 0.88 |
| 12 | **ATH-MaaS/OvisOCR3** (new) | 0.0373 | [0.0282, 0.0466] | 0.0275 | 10.38 | 0.9650 | 0.46 |
| 13 | PaddlePaddle/PaddleOCR-VL-1.6 | 0.0392 | [0.0259, 0.0499] | 0.0367 | 2.69 | 0.9415 | 6.47 |
| 14 | lightonai/LightOnOCR-2-1B | 0.0408 | [0.0248, 0.0569] | 0.0271 | 15.79 | 0.9688 | 2.08 |
| 15 | zai-org/GLM-OCR | 0.0491 | [0.0256, 0.0720] | 0.0236 | 26.93 | 0.9674 | 0.51 |
| 16 | Qwen/Qwen3.5-9B | 0.0498 | [0.0286, 0.0759] | 0.0305 | 19.98 | 0.9587 | 1.52 |
| 17 | baidu/Qianfan-OCR | 0.0575 | [0.0312, 0.0876] | 0.0404 | 17.89 | 0.9540 | 1.06 |
| 18 | baidu/Unlimited-OCR | 0.0575 | [0.0327, 0.0772] | 0.0422 | 16.19 | 0.9537 | 0.46 |
| 19 | **nvidia/Nemotron-3.5-Parse** (new) | 0.0575 | [0.0478, 0.0676] | 0.0444 | 13.88 | 0.9202 | 6.56 |
| 20 | deepseek-ai/DeepSeek-OCR-2 | 0.0614 | [0.0448, 0.0833] | 0.0594 | 2.09 | 0.9262 | 0.32 |
| 21 | deepseek-ai/DeepSeek-OCR | 0.0616 | [0.0467, 0.0753] | 0.0589 | 2.96 | 0.9206 | 0.55 |
| 22 | **datalab-to/indra-ocr** (new) | 0.0621 | [0.0257, 0.0989] | 0.0600 | 2.30 | 0.9493 | 0.00 |
| 23 | tesseract-5 | 0.0642 | [0.0421, 0.0794] | 0.0591 | 5.49 | 0.9210 | 0.00 |
| 24 | ds4sd/SmolDocling-256M-preview | 0.0660 | [0.0342, 0.1007] | 0.0649 | 1.21 | 0.8892 | 5.22 |
| 25 | **SoMarkAI/OXR-1.0** (new) | 0.1104 | [0.0330, 0.1762] | 0.1085 | 2.11 | 0.9551 | 0.00 |
| 26 | tiiuae/Falcon-OCR | 0.1384 | [0.0892, 0.1835] | 0.1124 | 27.66 | 0.9564 | 0.00 |

Where 95% CIs overlap, read the models as tied — the top three (dots.mocr, chandra-ocr-2, dots.ocr)
are a statistical tie, as the top two were before the 2026-10-30 re-score, and most mid-table
neighbours overlap. The order among them moved when index tables started to be read by column
group (see *Index tables are read one column group at a time*); it is not a measured difference. Loop % carries a selection bias: looped pages are excluded from that
row's other numbers, so a high loop rate means the row is scored with its hardest pages removed
(dots.ocr is scored on 96.1% of the corpus, PaddleOCR-VL-1.6 on 93.5%, DeepSeek-OCR-2 on 99.7%).
Read it alongside CER, never on its own.

## Reading this board

- **CIs first, ranks second.** Where the 95% CIs overlap — the top three do, and most mid-table
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
transcribes each one. It ranks 9th on the headline (4th when it was added in September), with a
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

**datalab-to/chandra-ocr-2 (3rd when added, 2nd now; 0.0285).** The lowest measured loop rate of any generative model
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

**lightonai/LightOnOCR-3-4B (5th when added, 4th now; 0.0312) and -0.8B (9th when added, 11th now; 0.0455).** Both improve on
LightOnOCR-2-1B (0.0489), and both cut its sparse/blank hallucination by a factor of five or more
(2.28 and 2.80 against 15.79). The 4B is better than LightOnOCR-2 on every volume; the Latin
volume is the hardest for both sizes (0.091 and 0.115). Both run the card's request (image only,
2048 px, temperature 0.2 / top_p 0.9) on `vllm/vllm-openai:v0.30.0`, not the default
vLLM image (see *Serving images*), because LightOn's repo says 0.27.1 makes these models loop. Apache-2.0.

**datalab-to/indra-ocr (17th when added, 22nd now; 0.0642).** A CPU pipeline of small ONNX models: the whole 2,165-page
run took one 8-vCPU job. It ties Tesseract on the headline, reads French best of any row
(`histoirenaturell10cuvi` 0.015), and is weakest on the Conchologia plate volume (0.120). It
recovers only 8% of caption tokens (the 4B recovers 70%). The package's markdown drops only page
headers and footers by label, so the caption loss happens in its detection, not in
post-processing. Weights: modified OpenRAIL-M (free for research, personal use and startups under
$20M; not for use competing with Datalab's API).

**nvidia/Nemotron-3.5-Parse (15th when added, 19th now; 0.0611; added 2026-10-25).** Run on the same pinned benchmark,
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
would have placed it 13th when added (17th now); on content pages it would score 0.0386, level with LightOnOCR-3-0.8B
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
runtime. The architecture needs NVIDIA's vLLM plugin, installed into the default vLLM image under
NVIDIA's own inference lock (`docker/nemotron-parse-35/`). Plugin defaults are kept: one-token MTP
speculative decoding (lossless under greedy decoding) and FlashAttention 3, hence an H200. This is
the card's single-pass mode; the double and triple passes behind its headline ParseBench number
re-read table and chart crops, which this corpus barely has. The normalizer applies NVIDIA's own
renderer (`render_markdown`, default options) at the served revision; a plain strip of the
coordinate and class tokens scores slightly worse. Weights and plugin: OpenMDW-1.1; tokenizer MIT.

## Two rows added 2026-10-26

Both rows use the same pinned benchmark, norm, scorer and post-processing version as the rest of
the board. Combining all 23 scorecards leaves every earlier row's numbers unchanged.

**ATH-MaaS/OvisOCR3 (7th when added, 12th now; 0.0382).** The 0.8B successor to OvisOCR2, run with the card's
`document_markdown` prompt, which is OvisOCR2's prompt word for word. On content pages the two are
level (0.0284 against 0.0280). The whole gap is on sparse/blank pages (10.38 against 2.79): on 111
of the 428, mostly Conchologia plates (70) and Birds plates (39), it writes a note about the page
instead of nothing — "Based on the image provided, there is no readable text, charts, or formulas
present. The image appears to be a blank page with a pale yellow background… Therefore, the
extracted content is empty." Its CI overlaps OvisOCR2's and most of the rows from 3rd to 9th.
10 pages (0.46%) loop. Default `vllm/vllm-openai:v0.26.0`. Apache-2.0.

**SoMarkAI/OXR-1.0 (22nd when added, 25th now; 0.1117).** A pipeline, like Falcon-OCR: a D-FINE layout model finds
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
unmodified from the vendor's `docker/Dockerfile.allinone`, not the default vLLM image (see *Serving images*).
Weights are downloaded at the pinned revision when the job starts rather than mounted: on
2026-10-26 the Jobs model mount served some small repo files with the wrong size, and the vendor's
plugin install failed on a truncated `pyproject.toml`. Licence: Apache-2.0 with additional
commercial-use thresholds.

## One row added 2026-10-27

Same pinned benchmark, norm, scorer and post-processing version as the rest of the board.
Combining all 24 scorecards leaves every earlier row's numbers unchanged. Provenance:
`data/full-2026-08/provenance/unlimited-ocr2.json`.

**baidu/Unlimited-OCR-2 (6th when added, 7th now; 0.0329).** The successor to Unlimited-OCR (15th when this row was
added, 18th now; 0.0604), released the same day, and the largest version-to-version gain on the
board: the headline error falls by almost half and every volume improves. Conchologia drops from
0.091 to 0.028 and the Latin volume from 0.185 to 0.118, still the hardest volume for this row, as
for most. The other change is on sparse/blank pages: 1.90 against v1's 16.19, the third-lowest on
the board after SmolDocling (1.21) and dots.ocr (1.79). On most plates it returns one whole-page
`image` block and no text. Three pages (0.14%) loop, two of them sparse Latin pages. Its CI
overlaps every row from 1st to 16th when added, v1 included: with six volumes to resample, a halving on the
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
`lmsysorg/sglang:v0.5.20-cu130` (see *Serving images*). 2,165 pages took 42 minutes
on one RTX PRO 6000. MIT.

*Why one page per request.* A diagnostic run of the model's multi-page mode (16 consecutive pages
per request, base mode at 1280 px, as in the uv-scripts recipe) scored 0.247. Most of that is page
attribution around plates, not reading: in Conchologia, 108 of 297 pages with text carry the next
page's text, though every request returned the right number of pages. On the three volumes without
plates it is level with or better than one page per request in the same mode (Trudy 0.050 against
0.060), but still behind this row's tiled single pages, which are worth more than the context: one
page per request in base mode scores 0.0503. Not a board row; the driver and results are on the
`diag/uocr2-multipage` branch.

## One row added 2026-10-28

Same pinned benchmark, norm, scorer and post-processing version as the rest of the board.
Combining all 25 scorecards leaves every earlier row's numbers unchanged. Provenance:
`data/full-2026-08/provenance/minicpm-ocr.json`.

**openbmb/MiniCPM-OCR-1.0 (8th when added, 8th now; 0.0368).** A pipeline, like OXR-1.0 and kraken: OpenBMB's
`minicpm-ocr` SDK finds regions with PP-DocLayoutV3, a 1.3B model reads each region with a
prompt for its type (text, table, formula), and the SDK joins the results in reading order.
Its median page CER (0.0059) is the fifth-lowest on the board, close to chandra-ocr-2 and
dots.ocr. On the Birds volume it has the board's lowest error (0.0065, against 0.0141 for
dots.mocr). The Latin volume is its hardest (0.128, 14th): long s comes out as "f"
("Infpeftore"), as for most rows. Its CI overlaps every row but Falcon-OCR.

The row keeps footnotes and side notes. The SDK reads page numbers, running heads, footers,
footnotes and side notes, then by default leaves all five out of the Markdown. The benchmark
scores footnotes as body text and every other row outputs them, so the row runs with
`--keep-labels footnote aside_text`. With the SDK defaults the same model scores 0.0509 (15th);
most of the difference is deletions (109,000 characters against 61,000). With footnotes kept it
recovers 98% of footnote tokens. Running heads and page numbers stay out, as the SDK ships.

Most of what remains is plates. Of the 150 content pages where it recovers less than half the
text, 123 are in Conchologia, and the median output there is 6 characters: the layout model
marks the plate as one figure, the SDK writes an image placeholder (dropped, as for OXR-1.0),
and the figure numbers and printer's line the ground truth carries are lost: 16% of caption tokens
recovered, among the lowest on the board. These pages are short, so the cost on the headline is
small. On the index pages of the same volume the model returns the two-column index as an HTML
table read across, which half the board also gets wrong (about 0.69 on those pages, under the
previous norm; see *Index tables are read one column group at a time*).

The model has no vLLM release support yet, so the row runs on a vLLM nightly (5d35ce3) with
OpenBMB's transformers pin and the SDK in its own environment, built into a public image,
`cr.hf.co/davanstrien/minicpm-ocr` (see *Serving images*). Decoding uses the
matching DSpark draft model, which changes speed, not output. The producer is the uv-scripts
recipe `ocr/minicpm-ocr.py` at 210c5cf, copied byte-for-byte to
`drivers/minicpm-ocr-port.py`. 2,165 pages took 49 minutes on one RTX PRO 6000. Apache-2.0.

## One row added 2026-10-30

Same pinned benchmark, norm, scorer and post-processing version as the rest of the board.
Combining all 26 scorecards leaves every earlier row's numbers unchanged. Provenance:
`data/full-2026-08/provenance/surya-ocr25.json`.

**datalab-to/surya-ocr-2.5 (12th, 0.0432).** A 689M VLM that does layout and OCR in one pass,
run through Datalab's `surya-ocr` package (0.23.0) exactly as its `surya_ocr` command runs it:
one full-page request per page, the package's parser and blank-region filter, and its block-mode
fallback when a full-page answer fails. It tied olmOCR-2 on the headline when added (0.0432; olmOCR-2 is 0.0321 since the
front-matter fix below), with a much narrower CI, which overlaps every row except dots.ocr, dots.mocr and Falcon-OCR. Median page CER 0.0181. No loops and
no errors: the package never keeps a capped full-page answer, and only one page fell back to
block mode. On sparse/blank pages it scores 2.62, with the recent specialists; 347 of the 428
come back empty. One blank Cuvier leaf comes back as a paragraph of invented pseudo-German.

Index pages carry almost half its error. The model writes the multi-column indexes of
Conchologia and Birds as HTML tables, and the frozen reading lane reads a table row by row,
which interleaves the index columns. The 53 content pages with a table hold 45% of its
content-page error (CER 0.44 on those pages, 0.023 on the rest). Half the board meets the same
problem on these pages (see MiniCPM-OCR-1.0 above). Norm 2026-10-30a reads these tables one column
group at a time, which takes the row to 0.0351 (10th); see *Index tables are read one column group
at a time*. The
Latin volume is its hardest (0.116), as for most rows.

Page headers and footers are kept. The package returns them as ordinary text blocks and drops
nothing; the normalizer renders the blocks with chandra-ocr-2's converter (same vendor, same
labels) without chandra's header/footer drop. Figures are not read by the package and add no
text, so plate-number labels inside figures are lost, as for MiniCPM-OCR-1.0.

Every setting is the package default. Datalab's own benchmark harness (`omni_parse_bench`)
runs this checkpoint the same way, with one difference: it turns on `SURYA_FULLPAGE_REGEN`,
which re-asks a looping or unparseable full-page answer at rising temperature. A diagnostic run
with it on scored 0.0443: it fired on no page, so the 0.0011 gap is run-to-run variation (993 of
2,165 first completions differ between the two runs; 169 pages' scored text differs). The
package normally starts vLLM in Docker; on Jobs the driver starts the same image
(`vllm/vllm-openai:v0.25.1`, the package's pin, by digest) with the flags the package's launcher
sets, serves the checkpoint at a pinned commit, and lets the package attach to it. 2,165 pages
took 27 minutes on one A100 80 GB. Weights: modified OpenRAIL-M (free for research, personal use
and startups under $5M; outputs carry the same licence).

## Fix: olmOCR-2 blank pages (2026-10-30)

olmOCR-2 is prompted to start every answer with a YAML front matter (language, rotation, table and
diagram flags). On a blank page it answers with the front matter alone. The post-processing rule
that drops the front matter required a line after the closing `---`, so on those 342 pages, all in
the sparse/blank stratum, the YAML itself was scored as the page's transcription. Those pages now
score as empty output. olmOCR-2 moved from 0.0432 (11th) to 0.0321 (6th) under norm 2026-07-20a; its sparse/blank CER
from 13.64 to 1.96. Content pages, recall and every other row are unchanged. The CI
[0.0151, 0.0510] stays wide and overlaps every row except Falcon-OCR. The fix changes one rule that
only this row uses, so it is made without a post-processing version bump, as for Falcon-OCR's
token strip; only olmOCR-2 was re-normalized and re-scored.

## Index tables are read one column group at a time (norm 2026-10-30a)

Re-scored 2026-10-30 from the cached outputs of all 26 rows; no model was re-run. The rule is
board-wide and the same for every row; nothing in it names a model.

**The problem.** Conchologia, Birds and the Russian entomological volume have list pages printed
in two or three columns: "planidens, Mich. …… XXXI. 190" down the left half, the next letters down
the right. The ground truth reads one printed column top to bottom, then the next. Many models
write such a page as one HTML or markdown table, with each printed column spanning several table
columns (name | plate | species), and the old reading lane stripped the tags row by row — so it
read across the page and interleaved the two printed columns. A model that had read every entry
correctly scored around 0.6 CER on those pages, as a penalty for output style rather than for
misreading. Models that wrote one table per printed column, or plain lines, were not affected.

**The rule.** If a table's first row is a header whose non-empty labels repeat — `[ , Plate.,
Species., , Plate., Species.]`, `[Vol. page] × 3` — the labels are compared ignoring case and
punctuation, and the number of repeats r is taken from them. The data columns are then split into
r equal-width groups (width = data columns / r, so a header that lost a blank cell, or a
four-cell header over six data columns, still splits 3 + 3), and each group is read top to bottom
before the next. Every other table is read as before: no repeated header, a data width that r does
not divide, a headerless grid, a caption or stray text inside the table, or a nested table. Cell
text is passed through unchanged, so only the order of cells moves. The ground truth contains no
tables, so its scoring is unchanged. Real data tables keep the row-major reading the ground truth
uses for them; an errata grid (chandra-ocr-2 on page 12342054) is a test fixture.

**What moved.** 193 page-rows on 35 pages in three volumes changed: 181 better, 12 worse. Every page
with no regrouped table scores exactly as before (`scripts/rescore_cached.py` asserts this for
every row). Token recall is order-free and does not move at four decimals. Of the 12 worse
page-rows, seven are tables whose cells the model had itself misaligned or padded with extra
columns (DeepSeek-OCR-2, Unlimited-OCR, OXR-1.0, Indra OCR), so grouping by column moves wrong cells
together. Three are Indra OCR pages that move by under 0.003. On two Falcon-OCR pages the model wrote
the same table twice; read in the right order, the duplicate now costs more insertions than the
interleaved reading did.

"Before" is the board after the olmOCR-2 front-matter fix above.

| Rank (before → after) | Model | CER reading before | CER reading after | Pages better / worse |
|---|---|---:|---:|---:|
| 2 → 1 | rednote-hilab/dots.mocr | 0.0237 | 0.0211 | 8 / 0 |
| 3 → 2 | datalab-to/chandra-ocr-2 | 0.0285 | 0.0214 | 17 / 0 |
| 1 → 3 | rednote-hilab/dots.ocr | 0.0235 | 0.0224 | 4 / 0 |
| 5 → 4 | lightonai/LightOnOCR-3-4B | 0.0312 | 0.0247 | 16 / 0 |
| 6 → 5 | allenai/olmOCR-2-7B-1025-FP8 | 0.0321 | 0.0279 | 11 / 0 |
| 4 → 6 | ATH-MaaS/OvisOCR2 | 0.0305 | 0.0299 | 3 / 0 |
| 7 → 7 | baidu/Unlimited-OCR-2 | 0.0329 | 0.0329 | 0 / 0 |
| 9 → 8 | openbmb/MiniCPM-OCR-1.0 | 0.0368 | 0.0334 | 8 / 0 |
| 8 → 9 | kraken/PP-OCRv6-medium | 0.0338 | 0.0338 | 0 / 0 |
| 12 → 10 | datalab-to/surya-ocr-2.5 | 0.0432 | 0.0351 | 20 / 0 |
| 13 → 11 | lightonai/LightOnOCR-3-0.8B | 0.0455 | 0.0363 | 23 / 0 |
| 10 → 12 | ATH-MaaS/OvisOCR3 | 0.0382 | 0.0373 | 3 / 0 |
| 11 → 13 | PaddlePaddle/PaddleOCR-VL-1.6 | 0.0392 | 0.0392 | 0 / 0 |
| 14 → 14 | lightonai/LightOnOCR-2-1B | 0.0489 | 0.0408 | 20 / 0 |
| 15 → 15 | zai-org/GLM-OCR | 0.0491 | 0.0491 | 0 / 0 |
| 16 → 16 | Qwen/Qwen3.5-9B | 0.0507 | 0.0498 | 3 / 0 |
| 17 → 17 | baidu/Qianfan-OCR | 0.0575 | 0.0575 | 0 / 0 |
| 18 → 18 | baidu/Unlimited-OCR | 0.0604 | 0.0575 | 9 / 1 |
| 19 → 19 | nvidia/Nemotron-3.5-Parse | 0.0611 | 0.0575 | 10 / 0 |
| 21 → 20 | deepseek-ai/DeepSeek-OCR-2 | 0.0620 | 0.0614 | 2 / 1 |
| 20 → 21 | deepseek-ai/DeepSeek-OCR | 0.0617 | 0.0616 | 1 / 0 |
| 22 → 22 | datalab-to/indra-ocr | 0.0642 | 0.0621 | 6 / 5 |
| 23 → 23 | tesseract-5 | 0.0642 | 0.0642 | 0 / 0 |
| 24 → 24 | ds4sd/SmolDocling-256M-preview | 0.0660 | 0.0660 | 0 / 0 |
| 25 → 25 | SoMarkAI/OXR-1.0 | 0.1117 | 0.1104 | 7 / 3 |
| 26 → 26 | tiiuae/Falcon-OCR | 0.1425 | 0.1384 | 10 / 2 |

**The top three are tied, before and after.** dots.mocr 0.0211 [0.0137, 0.0281], chandra-ocr-2
0.0214 [0.0151, 0.0279] and dots.ocr 0.0224 [0.0156, 0.0294] have almost the same intervals. Before
the re-score dots.ocr (0.0235) and dots.mocr (0.0237) were already tied. The new order among the
three is not a measured difference; read it as a tie. The whole effect comes from about 35 index
pages, so a row's move says how it writes index tables, not how well it reads.

**Known limits.** A headerless grid (the first row is already data) is left row-major, and so is a
header with a misspelt label (`Plafe.`); both still interleave. olmOCR-2 writes several Conchologia
indexes that way, and dots.ocr, LightOnOCR-3-0.8B, OvisOCR3 and DeepSeek-OCR-2 one or two each.
Catching them as well would move olmOCR-2 by about 0.0005 and dots.ocr by about 0.0004, without
changing the top-three tie. A model that prints the two columns as interleaved plain lines gets no
help: plain text carries no column structure to recover, and that is a reading-order error.

The sections above describe each row as it was added, so the numbers they quote for that day are
under norm `2026-07-20a`; this table and the headline are current.

## Caveats

- **Wide CIs by design.** The ground truth spans six volumes (books), and the bootstrap resamples
  volumes, not pages — so CIs are honest about book-to-book variation and correspondingly wide.
  Where CIs overlap, treat models as tied.
- **Loop % is also a selection effect.** See the note under the table: excluding looped pages
  flatters heavy loopers' other numbers. The column exists so that this is visible.
- **Numbers from earlier scorer versions are not comparable.** The scorer moved to count-aware
  semantics (3.0), the normalizer is frozen at `2026-10-30a` (index tables by column group; `2026-07-20a`
before that), and post-processing is versioned
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

surya-ocr-2 was held back from v1 because its library run could not meet that bar; its
successor, Surya OCR 2.5, is on the board through a driver that does. PaddlePaddle's PP-OCRv6
joins once its run meets the same bar. Note that PaddlePaddle's PP-OCRv6 is a different model from the `kraken/PP-OCRv6-medium` row
above, which shares the architecture but not the publisher or the training corpus.

## Serving images

Every row's exact image is pinned in its provenance file (`job.docker_image` in
[`data/full-2026-08/provenance/`](data/full-2026-08/provenance/)). Rows served by vLLM use
`vllm/vllm-openai:v0.26.0` unless the model needs otherwise; these rows do not:

| Row | Image | Why |
|---|---|---|
| LightOnOCR-3-4B, LightOnOCR-3-0.8B | `vllm/vllm-openai:v0.30.0` | LightOn's repo says 0.27.1 makes these models loop |
| Nemotron-3.5-Parse | `vllm/vllm-openai:v0.26.0` + NVIDIA's vLLM plugin | the architecture needs the plugin, installed under NVIDIA's inference lock |
| OXR-1.0 | `cr.hf.co/davanstrien/oxr` (vendor `Dockerfile.allinone`, vLLM 0.25.1) | the vendor's plugin and entrypoint require vLLM 0.25.1 |
| Unlimited-OCR-2 | `lmsysorg/sglang:v0.5.20-cu130` + Baidu's SGLang wheel | no vLLM support for v2 |
| MiniCPM-OCR-1.0 | `cr.hf.co/davanstrien/minicpm-ocr` (vLLM nightly 5d35ce3 + SDK) | the architecture is not in a vLLM release yet |
| Falcon-OCR | `ghcr.io/tiiuae/falcon-ocr` (vendor image) | vendor pipeline port |
| Surya OCR 2.5 | `vllm/vllm-openai:v0.25.1` (by digest) + `surya-ocr` 0.23.0 in its own environment | the package's own vLLM pin; the package and vLLM need separate environments |
| tesseract-5, kraken PP-OCRv6, Indra OCR | `ghcr.io/astral-sh/uv:python3.12-bookworm` | CPU pipelines, no vLLM |

## Where the numbers come from

Every scorecard traces to cached raw model output plus a pinned ground-truth revision. The raw
output, the scored text and the per-page scorecard of every row are public in
[`hf://buckets/finebooks/bhl-ocr-outputs`](https://huggingface.co/buckets/finebooks/bhl-ocr-outputs),
one folder per row, with a `manifest.json` giving each row's model revision, job ids, driver
commit, versions and licence (OXR-1.0's outputs are not included; the Datalab rows carry Datalab's
licence in their folders). From that cache any score can be reproduced without re-running
inference: consolidation (`runners/consolidate_run.py`) → versioned post-processing (`runners/normalize_outputs.py`) →
one scoring pass (`runners/score_dataset.py`) → one report (`runners/leaderboard.py`), written
into the board page by `scripts/bake_leaderboard.py`. The ground truth is the IMPACT-BHL expert
transcription set, CC-BY, credited to the IMPACT Centre of Competence / BHL-Europe — see the
source dataset [`finebooks/bhl-impact-gt`](https://huggingface.co/datasets/finebooks/bhl-impact-gt).
