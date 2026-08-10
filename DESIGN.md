# Design notes — BHL OCR evaluation harness

Why the harness is built the way it is, and the trade-offs behind the numbers. The dataset card
(`finebooks/bhl-impact-gt`) documents the ground truth; this documents the **scorer**.

## What this is

A standalone, runner-agnostic OCR-vs-ground-truth scorer. It consumes any runner's `(page, ocr_text)`
plus the IMPACT PAGE-XML ground truth and emits a **broken-out scorecard, not a single grade**. The
guiding principle from the field's own 2021 survey and the 2026 "Character Error Vector" work: a page
has many independent failure modes, so report a decomposed vector and resist collapsing it.

Four files, one concern each — the first three live in `scoring/` (the importable library):

- `normalizers.py` — **frozen-ish, versioned.** Named symmetric transforms + the edit-op accounting.
- `scorer.py` — **frozen.** Per-page primitives only: a `PageContext` (built once) + a declarative
  `Axis` registry. No aggregation.
- `report.py` — **thick, changeable.** Validation, micro-averaging, CIs, ordering, eligibility, provenance.
- `runners/` — produce `(page, ocr)`, cache raw outputs, write the per-page scorecard.

At a glance — the core is a pure library (no I/O); runners are the thin shell around it, and every
scoring runner emits the same artifact (a per-page scorecard parquet), which is what makes the board
model-source-agnostic:

```
data prep (run rarely)        get OCR + score                    aggregate
  prep_sample.py                run_openai.py    (any OpenAI-
  add_language.py                                 compatible endpoint)
        |                       drivers/ (inference only)              leaderboard.py
        v                       score_dataset.py (someone else's OCR)      |
  sample dataset  ── GT ──>       |                                        v
  (HF, private)                   +──> score_page() ──> scorecard.parquet ──> leaderboard.json
                                        (frozen core:                        (report.py: micro-avg,
                                         normalizers/scorer/gt_docling)       CIs, order, strata)
```

The durable interface between layers is the per-page scorecard, stamped with normalization, scorer,
and scorecard-schema versions plus scoring, run, and authoritative benchmark provenance. Benchmark
provenance identifies the requested GT revision, its immutable resolved Hub commit, stable dataset
identity, expected count, full page-set fingerprint, sampling-stratum threshold/counts, and sampler
provenance when available on every row (local/in-memory datasets fall back to their loaded content
fingerprint). The report has its own schema version. **Run models once (expensive);
re-score and re-report forever (free)** as the
normalizer/scorer/report evolve. That property is why the core is frozen and versioned rather than
edited in place.

## Getting OCR to score — client + scorer, serving abstracted

The harness never manages a model server. It is a **client** to any **OpenAI-compatible endpoint** plus
a scorer, and it stays deliberately ignorant of *how* a model is served:

- `run_openai.py` drives inference through `--base-url` — HF's Inference-Providers router (hosted VLMs,
  no GPU), or a **vLLM / SGLang** server you run locally or on an HF Job for an OCR specialist. The
  OpenAI chat protocol is the one interface nearly every model already speaks, so a single runner covers
  hosted and self-served models alike. Serving (GPU, launching vLLM, model-specific quirks) is out of
  scope — you bring a running endpoint. The runner has no benchmark-owned prompt: exactly one of
  `--prompt` or `--prompt-file` is required (an explicitly empty prompt is valid), and the exact text
  is sent unchanged, included in the page-set-aware cache key, and recorded with request settings.
  `--request-settings-file` accepts one validated JSON object per invocation, merged over the runner's
  `max_tokens=16384` / `temperature=0` defaults. A response whose finish reason is not `stop` (length
  truncation, content filtering) is recorded as a TRUNCATED page — excluded from aggregates, never
  scored as a transcription, because a runaway page emits thousands of characters against a few
  hundred of ground truth and scoring it whole turns micro-CER into a loop counter. **Truncated is
  a third state, not an error**: a repetition loop is the model failing on that page, not the run
  failing, so it does NOT make a model ineligible (see *Eligibility* below). An absent response
  body IS a producer error and does disqualify. Its effective contents—not its local path—are sent to
  the endpoint and included in cache/run identity; provider-specific controls belong under
  `extra_body`, where collisions with runner-owned or first-class request fields are rejected, and
  models requiring different settings run separately. A prompt-file UTF-8 BOM is
  preserved but emits a warning. Submission is bounded; interruption cancels queued requests and
  atomically saves incorporated completions. `--no-retry-errors` can retain cached
  error rows without paid retries; like checkpoint frequency, it is operational rather than identity.
- `score_dataset.py` scores OCR text produced by *some other tool entirely* — any table with an OCR
  column and a join key. It requires a producer-owned JSON provenance object (which may explicitly say
  `{"prompt": null, "note": "model-native OCR mode"}` for an OCR specialist) and augments it with
  harness-owned source/column metadata plus a local-file SHA256 or loaded-dataset fingerprint. OCR
  cells are accepted only as strings (including deliberate `""`); missing and non-string scalar cells
  fail closed with row/column context.

This is why we do **not** design around any single runner (e.g. ocrscout): coupling to one niche tool
would trade the harness's independence for convenience. The scorer is the IP; the OCR source is a
swappable input.

## The scorer abstraction (why not just functions)

Three layers, so a human or agent can see *why* each axis exists, not just how it is computed:

1. **`PageContext`** does all normalization / alignment / tokenization once. Axes never re-normalize.
2. **`Axis` registry** — each axis is a record carrying its `compute` fn together with `what` / `why`
   / `caveats` / `field_ref`. `REGISTRY` is the single source of truth for both scoring and the
   generated `AXES.md`, so the rationale can't drift from the code and `report.py` reads the same
   metadata to caption columns.
3. **compute fns** are thin — the mechanics live down in `PageContext`, the meaning lives up in the
   registry records. Adding an axis is one record, never a loop edit.

This is the lighteval "metric = data + two functions" split, sized down; not a plugin framework
(over-engineered for one repo) and not bare functions with drifting comments (rationale detaches).

## Two normalization lanes

Every axis runs in one of two named, **symmetric** lanes (applied identically to GT and OCR):

- **diplomatic** — NFC, case-sensitive, keeps long-s (ſ), ligatures, diacritics. The
  field-comparable transcription-fidelity lane (dinglehopper / OCR-D level-1). A model that reads ſ
  as `s` is scored *wrong* here.
- **reading** — NFKC (folds ſ→s, ﬁ→fi), case-fold, unify quote/hyphen variants, de-hyphenate, drop
  invisible formatting codepoints (soft hyphen, zero-width), strip a tiny fixed markup set. The
  reading-ability lane (OCR-D level-2/3).

The markup strip removes only structure a model may *add*, so it matches a **fixed vocabulary of
HTML/XML tag names** rather than any `<…>`-shaped run. Shape alone cannot tell a tag from
angle-bracketed prose, and the GT contains real instances (`<Gelenkknöchelchen>`): deleting those
would drop a legitimate recall target *and* charge a model that transcribed them faithfully with
over-extraction. Unrecognized angle brackets are treated as transcription — new model junk should
surface in `over_extraction`, not earn a regex patch.

Reporting both lanes makes CER's normalization-sensitivity **visible and versioned** instead of a
silent config choice — the survey's single biggest credibility recommendation. There is deliberately
no delta axis: the two lanes remain separate profile values rather than a derived quality signal.

**The lanes are NOT nested.** It is tempting to assume reading CER ≤ diplomatic CER (reading folds
more, so it should only match more). Property-based testing (Hypothesis) refuted this immediately:
`casefold('ß') → 'ss'` and NFKC ligature expansion *lengthen* the string, so on adversarial inputs
reading CER can *exceed* diplomatic (`ref='A'`, `hyp='ß'` → reading 2.0 vs diplomatic 1.0). So the two
lanes are genuinely different measurements, not a strict/lenient nesting — do not report one as a
bound on the other.

Edit ops are counted over Unicode **codepoints**, not grapheme clusters (jiwer's model). For NFC
Latin the gap is negligible; it is stamped in the provenance (`grapheme_mode`) rather than hidden.

## Aggregation: micro-average is the headline

CER **pools** every page's `(s, d, i, h)` counts and divides once (length-weighted). Token metrics do
the same with integer supports: recall pools matched/reference BODY-token counts; over-extraction
pools extra/output OCR-token counts; furniture evidence pools matched/reference furniture-token
counts. Repeated words are multiset members, not presence flags. An empty normalized CER reference emits
integer counts `(0, 0, len(hypothesis), 0)` and has an undefined per-page rate; those insertions are
still pooled with every other valid page, and the aggregate is undefined only when the aggregate
reference denominator is zero. The scorer emits all raw supports,
validates their bounds and per-page rate identities, and `report.py` exposes token metrics only as
explicit `*_micro` model values plus pooled numerators and denominators—never misleading token-macro
aliases. `script_match` remains a separately named per-page macro diagnostic. The per-page **median** is kept as a robustness diagnostic; the per-page mean is deliberately
NOT reported (a 40-char title page can yield CER > 1 and the mean is hostage to such sparse-GT
outliers — reporting a number this document calls misleading invited misuse). Rows the runner
flagged as errors are excluded from every aggregate and surfaced as a separate error count (an
*unflagged* empty output is a real model failure and IS scored). These metrics are explicitly
conditional on successful pages. **Producer errors** — transport failures, absent response bodies,
post-processing that could not model the output — make a model ineligible. **Truncations** do not:
they are excluded from the aggregates exactly like errors, but reported separately as a
repetition-loop rate, because every generating model on this corpus truncates somewhere (measured
0.32%–6.47%) and disqualifying all of them would leave only the non-neural engines rankable.

That exclusion has a cost the board must state rather than hide: **removing truncated pages flatters
the models that truncate most**, because loops cluster on dot-leader indexes and dense tabular pages
that are hard for everyone. A model at 6.5% is scored on 93.5% of the corpus with the hardest pages
disproportionately removed; one at 0.3% is scored on 99.7%. The loop rate is therefore a board
column, not a diagnostic, and the two numbers have to be read together. Models are ordered
by their CER point estimates; volume-bootstrap confidence intervals are reported as uncertainty and
are not converted into statistical-tie claims.

Completeness is a separate gate: every model must contain exactly one row for the authoritative full GT
page set recorded before any subset. `score_dataset.py --allow-incomplete` can write separately labelled
conditional diagnostics, but those rows remain ineligible after parquet aggregation; the comparison
order contains eligible models only. The combined leaderboard checks every actual model page set against
identical benchmark provenance, not against the first submitted model. Missing IDs, duplicate
`(model, page_id)` rows, unequal/full-set-mismatched page sets, missing or mixed per-model run provenance,
and mixed normalization, scorer, scorecard-schema, scoring, or benchmark provenance fail closed with
actionable errors. Schema 2.0 also requires every body/full character-count group to be four finite
nonnegative integers and verifies support-derived fraction bounds plus each per-page body CER/count
identity, including `None` at a zero reference denominator.

The bootstrap resamples **volumes, not pages**: pages within a book share scan quality, typeface,
and language, so page-level resampling would treat ~370 correlated pages as independent and
manufacture over-narrow CIs (per-book CER varies by >20× on this sample; the effective sample size
is closer to the number of books). Volume-level CIs are wider and honester — with only six books
they are the dominant caveat on any claimed separation between models.

## The furniture trade-off (the one to be explicit about)

Historical-book GT labels "furniture": running heads, page numbers, signatures, catch-words. In this
dataset the docling FURNITURE layer is exactly `{page_header, page_footer}`. Models disagree on
whether to emit furniture — clean-reading models drop it, verbatim models keep it. **This is a policy
difference, not a quality difference.** So what should the headline reward?

- **Full-text headline** — literal ("we scored `text`"), no dependency on the docling labels, but it
  *conflates policy with accuracy*: it silently penalizes a clean-reading model (or rewards a verbatim
  one) for a stylistic choice. This is the confound the red-team flagged.
- **Body-only headline** — measures reading of the actual content, fair across policies; matches
  OmniDocBench's abandon-set and olmOCR's `text_absent`. Cost: it leans on the docling BODY/FURNITURE
  labelling being right, and a model that drops *body* footnotes but keeps furniture could look good
  unless furniture is *also* reported.

**Decision: body-only is the headline, but the fork is instrumented, not hidden.** Concretely:

- `recall` and `cer_*` score the **body** layer (furniture excluded).
- `over_extraction` scores against the **full** text (body + furniture), so a model that faithfully
  reads furniture is **not** counted as emitting junk.
- Furniture is reported as `furniture_global_token_recall`: evidence that furniture tokens occur
  somewhere in output. It is directionless (`higher_is_better=None`) because verbatim versus clean is
  policy, not accuracy.
- Full-text diplomatic/reading edit counts remain in each scorecard only so an explicitly requested
  nested `policy_diagnostics.full_text_cer` block can micro-pool them. They are absent by default and
  are not registry axes; there is no full/body delta.

One consequence to state plainly: **CER-body still penalizes a verbatim model's furniture as
insertions** (furniture text the body target lacks), whereas **`recall` is furniture-immune** — it
asks "are the body tokens present" and ignores the extra. That gap is exactly why recall is reported
**co-equally with CER**, not folded into it: they measure different things and can disagree. All of
this is why the harness reports a vector, not a grade.

## Global token-evidence diagnostics, not spatial attribution

Furniture and region diagnostics compare typed GT token multisets with the **global** OCR token
multiset. Region GT counters are first pooled by label on each page and reported with structured
`rate`, `matched`, and `reference` supports. These values answer “is token evidence for this label
present somewhere?” They do not assign OCR text to coordinates or prove layout understanding.
Overlapping labels can reuse the same OCR token evidence. Spatial allocation and region IoU remain
out of scope.

## Sparse/blank sampling stratum

Sparse and blank pages are no longer removed below `--min-text`. That threshold classifies every
source page as `content` (length at least the threshold) or `sparse_blank` (below it). Sampling takes
exactly `n` rows, or all source rows when fewer exist; proportional stratum quotas use largest
remainders and include both nonempty strata when `n >= 2`. Within each stratum rows are spread across
volumes, with undersized-volume quota redistributed deterministically. Dataset rows carry
`sample_stratum` and `sample_stratum_threshold`; legacy GT rows derive them from text at threshold 80
when re-scored. Explicit labels are checked against deterministic text-length classification. Benchmark
provenance records the one threshold and expected count for each stratum; reports reject row-threshold
mismatches, cross-model label conflicts for the same canonical page, and complete-model count drift.
Reports show per-stratum submitted, clean n, errors, micro CER, recall, and over-extraction, retaining
an expected stratum even if all of its submitted pages errored. On a sparse page with an empty BODY but
correct furniture in FULL GT, body CER treats emitted furniture as zero-reference insertions: verbatim
output is penalized while empty output is not. This is the existing furniture-policy confound, now
concentrated in the `sparse_blank` stratum; read its CER alongside furniture evidence and the opt-in
full-text policy diagnostic rather than attributing a shift solely to OCR accuracy.

Sample construction resolves `prep_sample.py --source-revision` to one immutable Hugging Face commit
before downloading metadata or images, then stamps every output record with uniform sampler version,
seed, requested size, source repository/resolved commit, and threshold fields. Benchmark provenance
carries this sampler provenance when those fields are present, making reconstruction auditable.

This changes the authoritative page set. Retained OCR for old IDs can be re-scored under scorer 3.0,
but sparse/blank IDs absent from an older inference run require new inference. Scorecard schema 2.0
is fail-closed and has no compatibility shim.

## U+FFFD illegible wildcard

The GT marks an illegible glyph with U+FFFD (194 pages, 1366 occurrences). It is a **scoring
wildcard**: in character alignment, any edit op whose *reference* side is U+FFFD is treated as a hit
and excluded from both error count and reference length; GT tokens containing U+FFFD are dropped from
recall targets; and in `over_extraction` an illegible-bearing GT token can forgive one compatible otherwise-unmatched
OCR token. Compatibility is whole-token and one-to-one: visible normalized characters must match in
place and every U+FFFD realizes exactly one OCR codepoint; OCR and wildcard occurrences are consumed
once. Unrelated or arbitrarily long output therefore remains extra. The remaining caveat is that this
is token-local and position-blind: it cannot establish which hidden glyph was actually intended, so
multiple same-shaped resolutions remain indistinguishable. U+FFFD is never stripped and never a
character error. Property test: injecting U+FFFD into the GT can only lower the error count, never
raise it.

## CER ordering; recall is the co-equal robustness check

The board **orders point estimates by CER** because it is the field-comparable number (ISRI/UNLV,
OCR-D `dinglehopper`) — what the literature reports, so the comparison is familiar. But CER
is *hostage to the normalizer* (hence the two lanes + the provenance stamp) and to alignment. Token
**recall** has neither weakness: it is alignment-free, format-immune, and symmetric-normalized, so it
is reported **co-equally** as the robustness check on the CER ranking. Empirically the two are only
loosely correlated (recall ρ≈0.5 vs CER across the current board) — genuinely independent signal — so
a model can appear lower in CER order yet read more of the page (the reads-more-but-mangles-order case). Read
them together; neither is collapsed into a grade.

## Known GT caveats & corrections

Two properties of the ground truth shape what the numbers can claim:

- **Language labels are per-page (GlotLID), not per-volume.** The original volume→language map called
  the "russ" entomology journal Russian; it is mostly German and French with ~no Cyrillic (confirmed by
  the GT's author, and independently flagged by the `script_match` axis, which reads ~100% Latin across
  every model). `add_language.py` identifies each page's language from its own GT text and the wrong
  guess is kept only as `language_volume` for audit. Consequence: **the sample contains no non-Latin
  script**, so per-language results are per-GlotLID-page claims, and nothing here tests Cyrillic OCR.
  (The earlier tesseract "ru" CER of 0.86 was a config artifact — the `rus` pack applied to German/French
  text — not a reading result.)
- **GT reading order is heuristic.** The GT's linear text order was assembled by trial and error (per
  its author), so alignment-based CER partly charges models for the GT's own ordering choices, not just
  misreads. One more reason **order-free recall is read co-equally** with CER, which is
  normalization- and order-sensitive by construction.

## Validating the metric (three legs)

The scorer is the contribution, so it needs validating, not just running:

1. **Within-scorer axis redundancy** — Spearman across per-page axes (prior run flagged WER ≈ CER,
   ρ≈0.88; acted on in scorer 2.1 — the WER pair was dropped from the registry, along with
   `furniture_delta`/full-CER axes and the per-page mean, after an external design review agreed they
   bought neither robustness nor fairness). Raw full-text character counts now support only the
   opt-in policy diagnostic described above.
2. **Cross-benchmark transfer** — does an external benchmark (olmOCR-bench) predict our ranking? On
   18th-century scans it did not (ρ≈0), which is *why* a GT-anchored, corpus-specific scorer earns
   its keep.
3. **Leaderboard face-validity vs human preference** *(future phase)* — a sanity check, **not** a
   proposal to rank by human taste. If a model tops the score but readers consistently judge its
   output clearly worse, that **divergence is the finding**: it flags a blind spot in the metric, the
   GT, or the normalization. Method: blind pairwise preference on model outputs, fit a Bradley-Terry
   ranking, compare to the score ranking, and do error analysis on the residuals. Caveat, stated
   up front: humans are not gold here (they lack the GT and carry taste biases) — divergence is a flag
   to investigate, not a verdict that the metric is wrong.

## Cross-check against the retired GT

The old content-matched set (`davanstrien/bhl-impact-groundtruth`) is superseded but useful for one
check: old `gt_page_id` is the IMPACT `pcGtsId`, which is the new `xml_path` basename. Joining on it,
1168/1170 of the old content-verified rows find a new partner, with median text agreement 0.986. The
~68 low-agreement pages are **old-side truncation** (spot-checked: the old `full_text` is a partial
subset; the new `text` is the complete reading-order transcription incl. furniture). So the rebase is
both consistent and an improvement, and it removes the old set's selection bias (it kept only pages
2008 IA OCR could already read).

## Surfacing the board

The canonical output is the per-page scorecard **parquet** plus the aggregated `leaderboard.json`;
everything else is a *view* over it. The intended first surface is a **simple leaderboard view of our
own** (not yet built).

Beyond that, publishing to [Hugging Face eval-results](https://huggingface.co/docs/hub/eval-results) is
**one option**, not a commitment: the benchmark dataset would carry an `eval.yaml` whose **`tasks[]`**
are per-axis leaderboards (CER, recall, per-language, …) — matching the core rule that each axis is its
own leaderboard — with `evaluation_framework` naming this harness (`bhl-ocr-eval`, a custom identifier,
the route `math-arena` took), making the repo the citable evaluation source.

Either way the harness deliberately does **not** depend on `inspect-ai`: a thin Inspect `@scorer`
wrapper calling `score_page` is a possible later addition for the verified-scores path, but the
cross-model ordering/CI layer and the scoring of externally-produced OCR tables both live outside Inspect's
per-run, model-in-the-loop model, and the per-page scorecard parquet stays canonical.

## References

dinglehopper (OCR-D) · ISRI/UNLV ocreval · OmniDocBench · olmOCR-bench · jiwer · lighteval ·
Neudecker et al. 2021 (OCR eval survey, runs on IMPACT) · Stringalign / "The Character Error Vector"
(2026).
