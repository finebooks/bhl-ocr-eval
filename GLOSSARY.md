# Reading this leaderboard — a plain-language guide

This benchmark measures how well OCR and vision-language models transcribe **historical book pages**
(the IMPACT-BHL ground truth). It reports several numbers per model instead of one score, because a
model can be good at one thing and bad at another. This page explains what the numbers mean for
someone choosing a model for a real collection — no machine-learning background assumed.

`DESIGN.md` is the technical companion; this is the human one.

## First: there is no single "best" — it depends what you need

Ask which of these you care about, then read the matching column:

- **"I want a faithful transcription that keeps the original spelling"** (long-s `ſ`, ligatures,
  old orthography) → look at **CER (strict)**. This is the *diplomatic transcription* number.
- **"I just want good, searchable text; I don't care about old spelling or layout"** →
  look at **recall** and **CER (reading)**.
- **"Will it invent text that isn't on the page?"** → look at **over-extraction** and be wary of
  models that score well on average but hallucinate on blank or sparse pages.
- **"Which model for *my* material?"** → check the **per-language** and **per-region** breakdowns, not
  the headline. A model can be excellent on English but fail on German, or transcribe the body text
  perfectly while dropping every footnote.

The table is ordered by the CER point estimate, but small gaps should be read alongside the reported
confidence intervals rather than turned into claims of equivalence or meaningful separation.

## Glossary

**OCR** — Optical Character Recognition: turning a page image into machine-readable text. **VLM**
(vision-language model) and **OCR specialist** are two kinds of model that do this; the leaderboard
compares both.

**Ground truth** — the correct, human-checked transcription each model is scored against. Here it is
expert transcription of the IMPACT-BHL pages.

**CER — Character Error Rate** — of every 100 characters, how many are wrong (a wrong, missing, or
extra character). **Lower is better.** CER `0.05` ≈ 5 characters wrong per 100, i.e. about 95%
character-accurate. It's the standard OCR-quality number libraries have used for decades.

**CER (strict) vs CER (reading)** — we report CER two ways, because "wrong" depends on how strict you
are about historical spelling:
- **strict** (the *diplomatic* lane) keeps the original forms — a long-s `ſ` read as a modern `s`
  counts as an *error*. Use this if fidelity to the original matters (scholarly editions, linguistics).
- **reading** (the *modernised* lane) forgives long-s, capitalisation, and formatting. Use this if you
  just want usable, searchable text.
Read the two separate values together: a large difference often means formatting or transcription
convention rather than a reading failure. The report does not turn that difference into a delta axis.

**Recall (micro)** — of the BODY words that *should* be on the page, how many did the model actually
produce. **Higher is better.** Duplicate words count with their multiplicity, and model-level recall
pools matched/reference token counts across pages. It is order- and formatting-immune, but still
normalization-dependent. Good for judging “will full-text search work on this?”

**Over-extraction** — how much of the model's output is text that is *not* on the page: garbled
characters, repeated boilerplate, or invented ("hallucinated") passages. **Lower is better.** An OCR
word receives credit for an illegible-marked GT word only when the whole normalized word fits its
visible characters and each U+FFFD stands for exactly one character, one-to-one. This prevents an
unrelated or arbitrarily long word being forgiven; it still cannot prove which hidden glyph was meant.
A high number is a warning sign even if other scores look fine.

**Hallucination / fabrication** — when a model invents plausible-looking text that isn't actually on
the page (common on blank plates or hard-to-read pages). The damaging kind is a fluent invented
sentence, which can fool automatic scoring — which is why we report several numbers, not one.

**Page furniture / furniture global token recall** — running heads, page numbers, signatures, and
catch-words around the main text. The diagnostic asks whether those tokens appear anywhere in OCR.
Models disagree on whether to transcribe them; that is a directionless *policy* choice, so higher is
not labelled better.

**Region global token recall** — token evidence broken down by typed GT label (paragraph, caption,
footnote, header, page number). GT counters are pooled by label on each page and compared with the
same global OCR counter. It can show likely drops, but it is **not spatial attribution**: overlapping
labels can reuse OCR evidence, and the value says nothing about coordinates or reading order.

**Script fidelity** — does the model keep the correct alphabet? Some models silently transliterate or
translate (e.g. Cyrillic Russian rendered in the Latin alphabet). Near-1 is good; it only matters for
non-Latin material — and the current sample contains none (every page is Latin script), so here this
axis is a guard rather than a discriminator.

**Micro-average** — combine raw supports first and divide once, rather than averaging page rates. CER
pools character edits/reference characters; recall pools matched/reference tokens; over-extraction
pools extra/output tokens. A blank/all-wildcard page has no standalone CER when its normalized
reference length is zero, but any emitted characters remain insertions in a benchmark pool whose total
reference denominator is nonzero. A long page therefore contributes more evidence than a short one. Explicit
`*_micro` names distinguish these values from page macros, which are not retained as aliases.

**Content / sparse_blank stratum** — pages are classified by GT text length (default threshold 80),
not filtered out. The benchmark pins one threshold and expected counts; explicit row labels are checked,
and the report shows submitted, successfully scored n, errors, CER, recall, and over-extraction for
both expected strata even when one stratum failed entirely.

**Confidence interval** — the benchmark is 2,165 pages, and every score still has some wobble. The
interval shows uncertainty under resampling whole volumes. The table remains ordered by point score;
overlapping intervals alone do not establish equivalence.

**Eligibility** — a model is eligible only when its actual page set matches the authoritative full-GT
count and fingerprint and it has **no producer errors**: transport failures, absent response bodies,
or output our post-processing could not model. The comparison order contains eligible models only.
Conditional metrics may be shown in a separate diagnostic section, but remain ineligible when
scorecard parquet files are combined later.

A **repetition loop is not a producer error** and does not disqualify — see *Loop rate*.

**Loop rate (loop%)** — how often a model ran away with itself: it kept generating until it hit its
token limit instead of stopping. On this corpus that is almost always a repetition loop, usually
triggered by repeating typography — a table of contents with rows of dot leaders, a ruled index —
and the model has often transcribed the page correctly before it gets stuck.

Those pages are excluded from the accuracy scores, because a page that emits thousands of characters
against a few hundred of real text would swamp the average. But they do **not** make a model
ineligible: looping is the model failing on a page, not the run failing, and on this corpus every
model that generates text loops somewhere (0.32%–6.47%), so disqualifying them would leave only the
non-neural engines ranked.

**Read loop% alongside CER, never on its own.** Because looped pages are removed, a model that loops
a lot is scored on an easier remainder — loops cluster on the hardest layouts. A model at 6.5% is
judged on 93.5% of the corpus with many of the hard pages taken out; one at 0.3% is judged on 99.7%.

**Post-processing version (POSTPROC)** — the per-model cleanup applied to raw model output before
scoring (dropping bounding-box markup, converting DocTags to text, and so on). It is versioned and
stamped on every row, so a score can always be traced to the rules that produced it. Because raw
output is cached, correcting a rule costs a re-score rather than re-running models on GPUs.

**Benchmark provenance** — the identical GT identity attached to every scorecard row: dataset ID,
requested revision (if any), immutable resolved Hub commit, stable dataset identity, expected page
count, full-page-set fingerprint, and the stratum threshold/counts. Local/in-memory datasets use a
loaded content fingerprint fallback. When sample records contain sampler metadata it also includes
the sampler version, seed, requested size, and immutable source repository revision. It prevents a
moving branch or uniformly partial or relabelled submission from masquerading as a fixed, complete
benchmark.

**Run provenance** — a nonempty producer-run description attached to every row. It must be constant
within one model so rows from different runs cannot be spliced together; different models may describe
different runs.

**Parameters (model size)** — roughly how big a model is, in billions of parameters (B). Bigger models
are usually slower and more expensive to run. A key finding here is that sub-1B specialists match or
beat models roughly 10× their size — so bigger is not automatically better.

**Pareto frontier (the size-vs-accuracy plot)** — a way to see "best quality for the size." A model is
*on the frontier* if no smaller model is as good. Models below the frontier are dominated — you could
get the same or better accuracy from something smaller (cheaper, faster). It's how you pick an
efficient model, not just the most accurate one.

## What the boards currently show

The findings walkthrough lives in [RESULTS.md](RESULTS.md) — headline results on the 2,165-page
benchmark, what drives the differences (sparse/blank-page hallucination above all), and the caveats. Two durable lessons worth repeating here:

- **Size doesn't decide.** Sub-1B OCR specialists sit level with 70B+ vision-language models on this
  material. Read the CIs, not the ranks.
- **Configuration can matter more than model choice.** Tesseract with the wrong language pack scored
  three times worse than Tesseract with the right one. For classical engines, setup is part of the
  result.
