# Breaking scorecard migration

Operational reference for re-scoring cached outputs across scorer/normalizer version bumps.
The short version lives in [README.md](README.md); this is the full fail-closed contract.

Current reporting fails closed on scorecard schema `2.0`, scorer `3.0`, the current normalizer, and
complete scoring/run/benchmark provenance. **Old scorecard parquets are rejected without a
compatibility shim:** regenerate them by re-scoring retained raw OCR outputs. Existing raw OCR can be
re-scored for the changed count-aware metrics and markup normalization without model inference, but
only when compatible outputs still exist under the current immutable-revision-aware cache key. The old
Tesseract runner did not cache raw outputs, and OpenAI caches created under legacy or pre-revision-
resolution identities are not discovered automatically; migrate or rename them only after independently
verifying every request and dataset identity field.

The new sample design also includes sparse/blank page IDs that older samples filtered out; those newly
added IDs require inference. Existing GT rows without stratum metadata are deterministically classified
from `text` at threshold 80 while re-scoring; an explicit label that conflicts with its text and
threshold is rejected. Every schema-2.0 body/full edit-count group must contain four finite nonnegative
integers, including `(0, 0, insertions, 0)` for an empty normalized reference; old all-missing groups
are invalid. The generated `leaderboard.json` retains ordered `eligible_order` and separately labelled
`ineligible_order`, with no tiers or model-level macro aliases.

Full-text diplomatic/reading CER is not a leaderboard axis. Raw full-text edit counts are retained for
a directionally neutral policy audit and appear only under the nested `policy_diagnostics` report block
when `leaderboard.py --policy-diagnostics` is requested; the text report renders that nested block too.

`prep_sample.py --min-text` no longer filters pages: it classifies them into `content` and
`sparse_blank`, samples an exact requested size proportionally across both strata, and spreads each
stratum over volumes. Use `--source-revision` to select a source branch/tag/commit; preparation resolves
it once to an immutable Hugging Face commit and uses that commit for metadata and every image.
Every record persists uniform sampler version/seed/requested-size/source/threshold provenance, which
is carried into benchmark provenance. Scorecards and reports preserve the stratum, including a
submitted/clean/error diagnostic when every page in one stratum fails.

External OCR must be consolidated before scoring: merge all producer shards into one table containing
exactly one row per `(model, page ID)` across the authoritative full GT page set, then pass that table
to `score_dataset.py`. Do not score shards independently and concatenate their scorecards, because
run provenance and full-page completeness are validated per model. Use `--allow-incomplete` only when
you deliberately want conditional diagnostics that can never enter the eligible order. External OCR
cells must be strings: `""` is a valid deliberate empty output, while null/NaN and numeric or boolean
scalars fail closed with row/column context instead of being coerced.
