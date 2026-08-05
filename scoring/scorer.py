# /// script
# requires-python = ">=3.10"
# dependencies = ["jiwer>=4,<5", "rapidfuzz>=3,<4"]
# ///
"""Versioned per-page OCR scorer.

The scorer emits a small profile of body CER, count-aware token coverage, over-extraction, script
fidelity, and policy/layout diagnostics. Every pooled metric has raw integer supports on each page;
aggregation belongs to :mod:`report`.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

import normalizers as N

SCORER_VERSION = "3.0"


def _legible_counter(tokens) -> Counter:
    return Counter(t for t in tokens if N.REPLACEMENT not in t)


def _recall_support(reference: Counter, output: Counter) -> tuple[int, int]:
    denominator = sum(reference.values())
    matched = sum(min(count, output.get(token, 0)) for token, count in reference.items())
    return matched, denominator


def _support_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


@dataclass(frozen=True)
class PageContext:
    """Immutable normalized/aligned/tokenized substrate shared by every axis."""

    full: str
    body: str
    furniture: str
    ocr: str
    language: str | None
    regions: tuple
    ec: dict
    tok_ocr: Counter
    tok_body: Counter
    tok_full_legible: Counter
    tok_full_wildcard: Counter
    tok_furniture: Counter
    gt_script: str | None


def build_context(
    text: str | None,
    ocr: str | None,
    *,
    body_text: str | None = None,
    furniture_text: str | None = None,
    regions: list | None = None,
    language: str | None = None,
) -> PageContext:
    """Build a page context once. Without a docling split, body is the full text."""
    full = text or ""
    body = full if body_text is None else (body_text or "")
    furniture = furniture_text or ""
    ocr = ocr or ""

    def ec(gt, lane):
        return N.edit_counts(N.norm(gt, lane), N.norm(ocr, lane))

    full_tokens = N.tokens(full, "reading")
    return PageContext(
        full=full,
        body=body,
        furniture=furniture,
        ocr=ocr,
        language=language,
        regions=tuple(regions or []),
        ec={
            "cer_body_dip": ec(body, "diplomatic"),
            "cer_body_read": ec(body, "reading"),
            # Full-text counts are retained only for opt-in policy diagnostics in report.py.
            "cer_full_dip": ec(full, "diplomatic"),
            "cer_full_read": ec(full, "reading"),
        },
        tok_ocr=Counter(N.tokens(ocr, "reading")),
        tok_body=_legible_counter(N.tokens(body, "reading")),
        tok_full_legible=_legible_counter(full_tokens),
        tok_full_wildcard=Counter(token for token in full_tokens if N.REPLACEMENT in token),
        tok_furniture=_legible_counter(N.tokens(furniture, "reading")),
        gt_script=N.dominant_script(full),
    )


@dataclass(frozen=True)
class Axis:
    name: str
    kind: str
    lane: str
    higher_is_better: bool | None
    compute: Callable[[PageContext], object]
    what: str
    why: str
    caveats: str
    field_ref: str


def _rate(ctx: PageContext, key: str) -> float | None:
    rate = ctx.ec[key].rate()
    return round(rate, 4) if rate is not None else None


def _recall(ctx: PageContext) -> float | None:
    return _support_rate(*_recall_support(ctx.tok_body, ctx.tok_ocr))


def _wildcard_compatible(pattern: str, token: str) -> bool:
    """Whether ``token`` realizes a whole GT token with one codepoint per U+FFFD."""
    return len(pattern) == len(token) and all(
        expected == N.REPLACEMENT or expected == actual
        for expected, actual in zip(pattern, token, strict=True)
    )


def _wildcard_match_count(patterns: Counter, outputs: Counter) -> int:
    """Maximum one-to-one matching between wildcard GT and still-unmatched OCR occurrences."""
    gt_occurrences = [token for token, count in sorted(patterns.items()) for _ in range(count)]
    ocr_occurrences = [token for token, count in sorted(outputs.items()) for _ in range(count)]
    matched_output: dict[int, int] = {}

    def augment(gt_index: int, seen: set[int]) -> bool:
        for output_index, token in enumerate(ocr_occurrences):
            if output_index in seen or not _wildcard_compatible(gt_occurrences[gt_index], token):
                continue
            seen.add(output_index)
            previous = matched_output.get(output_index)
            if previous is None or augment(previous, seen):
                matched_output[output_index] = gt_index
                return True
        return False

    return sum(augment(gt_index, set()) for gt_index in range(len(gt_occurrences)))


def _over_extraction_support(ctx: PageContext) -> tuple[int, int]:
    output = sum(ctx.tok_ocr.values())
    exact_matches = sum(
        min(output_count, ctx.tok_full_legible.get(token, 0))
        for token, output_count in ctx.tok_ocr.items()
    )
    unmatched = ctx.tok_ocr.copy()
    for token, reference_count in ctx.tok_full_legible.items():
        unmatched[token] -= min(reference_count, unmatched.get(token, 0))
        if not unmatched[token]:
            del unmatched[token]
    wildcard_matches = _wildcard_match_count(ctx.tok_full_wildcard, unmatched)
    return output - exact_matches - wildcard_matches, output


def _over_extraction(ctx: PageContext) -> float:
    extra, output = _over_extraction_support(ctx)
    return round(extra / output, 4) if output else 0.0


def _furniture_global_token_recall(ctx: PageContext) -> float | None:
    return _support_rate(*_recall_support(ctx.tok_furniture, ctx.tok_ocr))


def _region_global_token_recall(ctx: PageContext) -> dict:
    """Pool GT tokens by label, then compare each label with the same global OCR counter.

    This is deliberately not spatial attribution: overlapping labels may reuse OCR evidence.
    """
    pooled = defaultdict(Counter)
    for region in ctx.regions:
        text = region.get("text")
        if text:
            pooled[region.get("label") or "?"].update(_legible_counter(N.tokens(text, "reading")))
    result = {}
    for label, reference in sorted(pooled.items()):
        matched, denominator = _recall_support(reference, ctx.tok_ocr)
        if denominator:
            result[label] = {
                "rate": _support_rate(matched, denominator),
                "matched": matched,
                "reference": denominator,
            }
    return result


REGISTRY = [
    Axis(
        "recall", "coverage", "format_immune", True, _recall,
        what="Count-aware fraction of legible BODY GT tokens supported by the global OCR token multiset.",
        why="The format-immune backbone catches dropped content without penalizing formatting or order.",
        caveats="Body only; order-free, but duplicate token multiplicities are respected.",
        field_ref="bWER / Nougat set-recall / OCR-D bag-of-words.",
    ),
    Axis(
        "over_extraction", "over_extraction", "format_immune", False, _over_extraction,
        what="Count-aware fraction of OCR tokens exceeding the legible FULL-GT token multiset.",
        why="Flags hallucination, boilerplate, and repeated text while allowing faithful furniture.",
        caveats="Output-normalized and position-blind. U+FFFD credit is one-to-one and token-local: visible characters and length must match, with each marker realizing exactly one codepoint; this cannot verify which hidden glyph was intended.",
        field_ref="ISRI 'generated / spurious' characters.",
    ),
    Axis(
        "cer_diplomatic", "faithfulness", "diplomatic", False,
        lambda c: _rate(c, "cer_body_dip"),
        what="Character error rate, NFC + case-sensitive, body only.",
        why="Field-comparable transcription fidelity: long-s, ligatures, and case remain distinctions.",
        caveats="Normalization- and alignment-sensitive; aggregate by pooling raw counts. A zero-length normalized reference has no page rate, but its insertions remain in benchmark pooling.",
        field_ref="dinglehopper / OCR-D / ISRI CER.",
    ),
    Axis(
        "cer_reading", "faithfulness", "reading", False,
        lambda c: _rate(c, "cer_body_read"),
        what="Character error rate, NFKC + case-fold + narrow markup stripping, body only.",
        why="Separates reading ability from transcription convention.",
        caveats="Hides long-s, case, and ligature errors by design; not field-comparable. A zero-length normalized reference has no page rate, but its insertions remain in benchmark pooling.",
        field_ref="OCR-D level-2/3 normalized lane.",
    ),
    Axis(
        "script_match", "fidelity", "na", True,
        lambda c: N.script_match(c.ocr, c.full),
        what="Fraction of OCR letters in the GT's dominant Unicode script.",
        why="Catches script switching or transliteration that CER cannot isolate.",
        caveats="Undefined without GT-script or OCR-letter evidence; current sample is nearly all Latin.",
        field_ref="No field equivalent — our differentiator.",
    ),
    Axis(
        "furniture_global_token_recall", "furniture", "format_immune", None,
        _furniture_global_token_recall,
        what="Count-aware FURNITURE-token evidence found anywhere in the OCR output.",
        why="Shows whether the model tends to emit page furniture; it is a policy diagnostic, not quality.",
        caveats="Global token-evidence proxy, not spatial attribution; directionless by design.",
        field_ref="OmniDocBench abandon-set (reported separately).",
    ),
    Axis(
        "region_global_token_recall", "layout", "format_immune", True,
        _region_global_token_recall,
        what="Per-label count-aware GT-token evidence found anywhere in the OCR output.",
        why="Shows which kinds of text may be dropped without claiming spatial assignment.",
        caveats="Global token-evidence proxy, not spatial attribution: labels are pooled per page and overlapping labels may reuse the same OCR evidence.",
        field_ref="Flexible-Character-Accuracy-spirit proxy.",
    ),
]


def score_page(
    text: str | None,
    ocr: str | None,
    *,
    body_text: str | None = None,
    furniture_text: str | None = None,
    regions: list | None = None,
    language: str | None = None,
    page_id: int | str | None = None,
) -> dict:
    """Score one page and emit axes, integer supports, edit counts, and scoring versions."""
    ctx = build_context(
        text, ocr, body_text=body_text, furniture_text=furniture_text,
        regions=regions, language=language,
    )
    row = {axis.name: axis.compute(ctx) for axis in REGISTRY}
    row["recall_matched"], row["recall_reference"] = _recall_support(ctx.tok_body, ctx.tok_ocr)
    row["over_extraction_extra"], row["over_extraction_output"] = _over_extraction_support(ctx)
    row["furniture_global_token_recall_matched"], row["furniture_global_token_recall_reference"] = (
        _recall_support(ctx.tok_furniture, ctx.tok_ocr)
    )
    for key, counts in ctx.ec.items():
        for part in ("s", "d", "i", "h"):
            row[f"{key}_{part}"] = getattr(counts, part)
    row.update(
        language=language,
        gt_script=ctx.gt_script,
        page_id=page_id,
        norm_version=N.NORM_VERSION,
        scorer_version=SCORER_VERSION,
    )
    return row


AXIS_NAMES = [axis.name for axis in REGISTRY]
COUNT_KEYS = ("cer_body_dip", "cer_body_read", "cer_full_dip", "cer_full_read")
COUNT_COLUMNS = [f"{key}_{part}" for key in COUNT_KEYS for part in ("s", "d", "i", "h")]
SUPPORT_GROUPS = {
    "recall": ("recall_matched", "recall_reference"),
    "over_extraction": ("over_extraction_extra", "over_extraction_output"),
    "furniture_global_token_recall": (
        "furniture_global_token_recall_matched", "furniture_global_token_recall_reference",
    ),
}
SUPPORT_COLUMNS = [column for columns in SUPPORT_GROUPS.values() for column in columns]


if __name__ == "__main__":
    body = "The tree sparrow is found throughout eastern Asia."
    furniture = "BIRDS OF GREAT BRITAIN 94"
    full = f"{furniture}\n{body}"
    regions = [{"label": "text", "text": body}, {"label": "page-header", "text": furniture}]
    score = score_page(full, body, body_text=body, furniture_text=furniture, regions=regions)
    assert score["recall"] == score["recall_matched"] / score["recall_reference"] == 1.0
    assert score["over_extraction"] == 0.0
    assert score["furniture_global_token_recall"] == 0.0
    assert score["region_global_token_recall"]["text"]["rate"] == 1.0
    print(f"scorer {SCORER_VERSION} / norm {N.NORM_VERSION}; all scorer self-tests passed.")
