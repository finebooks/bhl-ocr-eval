"""Scorer invariants: property-based bounds/monotonicity + example-based behaviour & structure.

The example tests pin the two decisions that would silently flip and corrupt a leaderboard: the
furniture ignore-set (recall/CER on body only; over-extraction against full) and the diplomatic vs
reading lane split. The parametrized registry test guards the self-documenting contract.
"""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import normalizers as N
import scorer as S

GLYPHS = "abcde ABCDE ſﬁßäö абвгд"
txt = st.text(alphabet=GLYPHS, max_size=40)
_FRAC_AXES = ("recall", "over_extraction", "furniture_global_token_recall", "script_match")


@settings(max_examples=150, deadline=None)
@given(text=txt, ocr=txt)
def test_fraction_axes_in_bounds(text, ocr):
    row = S.score_page(text, ocr)
    for k in _FRAC_AXES:
        v = row[k]
        if v is not None:
            assert 0.0 <= v <= 1.0


@settings(max_examples=150, deadline=None)
@given(text=st.text(alphabet="abcde ", min_size=3, max_size=30),
       ocr=st.text(alphabet="abcde ", max_size=30))
def test_recall_monotonic(text, ocr):
    """Adding a real GT token to the OCR can never lower recall (presence-based coverage)."""
    toks = N.tokens(text, "reading")
    if not toks:
        return
    r0 = S.score_page(text, ocr)["recall"]
    r1 = S.score_page(text, ocr + " " + toks[0])["recall"]
    if r0 is not None and r1 is not None:
        assert r1 >= r0 - 1e-9


@settings(max_examples=100, deadline=None)
@given(text=txt, ocr=txt)
def test_emitted_counts_match_cer(text, ocr):
    """The cer_reading axis equals the rate recomputed from the emitted raw counts — so report.py's
    pooled micro-average is scoring the same thing the per-page axis reports."""
    row = S.score_page(text, ocr)  # body == full here
    s, d, i, h = (row["cer_body_read_s"], row["cer_body_read_d"],
                  row["cer_body_read_i"], row["cer_body_read_h"])
    if row["cer_reading"] is not None and (s + d + h) > 0:
        assert abs(row["cer_reading"] - (s + d + i) / (s + d + h)) < 1e-4


def test_perfect_match_anchor():
    gt = "the quick brown fox jumps"
    row = S.score_page(gt, gt)
    assert row["cer_diplomatic"] == 0.0 and row["cer_reading"] == 0.0
    assert row["recall"] == 1.0 and row["over_extraction"] == 0.0


def test_lane_divergence_golden():
    row = S.score_page("the ſong", "the song")   # long-s vs modern s
    assert row["cer_diplomatic"] > 0        # diplomatic: ſ ≠ s
    assert row["cer_reading"] == 0.0        # reading: folds ſ → s
    assert "cer_delta" not in row            # no redundant delta axis


def test_script_match_golden():
    """script_match = fraction of OCR letters in the GT's dominant Unicode script — the harness's
    billed differentiator (catches Cyrillic→Latin transliteration that CER can't isolate from misreads).
    GT-derived and label-free; a silent change to the script bucketing would slip past the bounds test."""
    assert S.score_page("абвгд", "абвгд")["script_match"] == 1.0    # same script → no switching
    assert S.score_page("абвгд", "abvgd")["script_match"] == 0.0    # transliterated to Latin
    assert S.score_page("абвгд", "абв ef")["script_match"] == 0.6   # 3 Cyrillic of 5 letters (space ignored)


def test_furniture_fairness():
    body = "alpha beta gamma delta epsilon zeta"
    furniture = "RUNNING HEADER 42"
    full = furniture + " " + body
    # (a) dropping furniture leaves the body headline untouched
    clean = S.score_page(full, body, body_text=body, furniture_text=furniture)
    assert clean["recall"] == 1.0 and clean["cer_reading"] == 0.0
    assert clean["furniture_global_token_recall"] == 0.0
    # (b) emitting furniture that IS in full text is not punished as over-extraction
    verbatim = S.score_page(full, full, body_text=body, furniture_text=furniture)
    assert verbatim["over_extraction"] == 0.0
    assert verbatim["furniture_global_token_recall"] == 1.0


def test_count_aware_recall_and_axis_support_identity():
    row = S.score_page("echo echo echo other", "echo other")
    assert (row["recall_matched"], row["recall_reference"]) == (2, 4)
    assert row["recall"] == row["recall_matched"] / row["recall_reference"] == 0.5


def test_repeated_token_over_extraction_and_support_identity():
    row = S.score_page("echo", "echo echo echo")
    assert (row["over_extraction_extra"], row["over_extraction_output"]) == (2, 3)
    assert row["over_extraction"] == round(2 / 3, 4)


def test_region_rows_are_pooled_by_label_and_reuse_global_evidence():
    regions = [
        {"label": "caption", "text": "echo echo"},
        {"label": "caption", "text": "echo other"},
        {"label": "overlap", "text": "echo"},
    ]
    result = S.score_page("echo echo echo other", "echo other", regions=regions)[
        "region_global_token_recall"
    ]
    assert result["caption"] == {"rate": 0.5, "matched": 2, "reference": 4}
    assert result["overlap"] == {"rate": 1.0, "matched": 1, "reference": 1}


def test_full_text_cer_counts_are_raw_only_not_axes():
    row = S.score_page("HEAD body", "body", body_text="body", furniture_text="HEAD")
    assert row["cer_body_read_s"] == 0 and row["cer_body_read_d"] == 0
    assert row["cer_full_read_d"] > 0
    assert "cer_full_reading" not in row and "cer_delta" not in row
    assert not any(axis.name.startswith("cer_full") for axis in S.REGISTRY)


def test_over_extraction_forgives_resolved_illegible_glyph():
    """U+FFFD GT tokens are wildcards: correctly reading a marred glyph must NOT be charged as
    over-extraction (it can't match the illegible GT token) — the recall lane already treats it so.
    Guards the frozen-core fix; a regression here would penalize the models that read illegible text."""
    fffd = N.REPLACEMENT
    gt = f"recon{fffd}aissance ends here"          # one illegible glyph inside a GT word
    resolved = S.score_page(gt, "reconnaissance ends here")
    assert resolved["recall"] == 1.0
    assert resolved["over_extraction"] == 0.0      # resolved illegible word forgiven, not junk
    # a genuine hallucination ALONGSIDE the illegible glyph is still counted (discount is bounded)
    plus_junk = S.score_page(gt, "reconnaissance ends here FOOBAR")
    assert plus_junk["over_extraction"] == round(1 / 4, 4)  # FOOBAR charged; the resolution forgiven
    # Unrelated or arbitrarily long tokens do not receive unconditional wildcard credit.
    unrelated = S.score_page(f"a{fffd} b{fffd} c{fffd}", "resolved")
    assert unrelated["over_extraction_extra"] == 1
    too_long = S.score_page(f"recon{fffd}aissance", "reconnaissanceeeee")
    assert too_long["over_extraction_extra"] == 1


def test_over_extraction_wildcard_credit_is_one_to_one_with_multiplicity():
    fffd = N.REPLACEMENT
    row = S.score_page(f"a{fffd} a{fffd}", "ab ab ab")
    assert (row["over_extraction_extra"], row["over_extraction_output"]) == (1, 3)


def test_over_extraction_all_wildcard_token_requires_exact_codepoint_count():
    fffd = N.REPLACEMENT
    compatible = S.score_page(fffd * 2, "xy")
    too_long = S.score_page(fffd * 2, "xyz")
    assert compatible["over_extraction_extra"] == 0
    assert too_long["over_extraction_extra"] == 1


def test_over_extraction_uses_maximum_one_to_one_wildcard_matching():
    fffd = N.REPLACEMENT
    # The broad wildcard token must not consume the only output that can satisfy the constrained one.
    row = S.score_page(f"{fffd}{fffd} a{fffd}", "ab cd")
    assert row["over_extraction_extra"] == 0


@pytest.mark.parametrize("ax", S.REGISTRY, ids=[a.name for a in S.REGISTRY])
def test_axis_metadata_present(ax):
    """Every axis carries its rationale (drives AXES.md + report captions) and a real compute fn."""
    assert ax.what.strip() and ax.why.strip() and ax.caveats.strip() and ax.field_ref.strip()
    assert callable(ax.compute)
    assert ax.kind and ax.lane


def test_score_page_returns_all_axes_and_provenance():
    row = S.score_page("hello world", "hello world")
    for ax in S.REGISTRY:
        assert ax.name in row
    for col in (*S.COUNT_COLUMNS, *S.SUPPORT_COLUMNS):
        assert col in row
    assert row["scorer_version"] == S.SCORER_VERSION
    assert row["norm_version"] == N.NORM_VERSION
