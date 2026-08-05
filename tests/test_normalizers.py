"""Property-based (Hypothesis) invariants for the frozen normalization + edit-op core.

These pin the guarantees the whole "run once, re-score forever" design leans on. A targeted
alphabet mixes ASCII with the historical-print glyphs that actually matter (long-s, ligatures, ß,
Latin diacritics, Cyrillic) so the fuzzing hits the interesting cases, not random ASCII.
"""
import jiwer
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

import normalizers as N

GLYPHS = "abcde ABCDE ſﬁﬂßäöéçñ абвгд"
FFFD = N.REPLACEMENT
LANES = ("diplomatic", "reading")

txt = st.text(alphabet=GLYPHS, max_size=40)
txt_fffd = st.text(alphabet=GLYPHS + FFFD, max_size=40)
# For the lane-ordering property, exclude the folds that CHANGE LENGTH (ß→ss, ﬁ→fi): Hypothesis
# showed those make reading CER *exceed* diplomatic (ref='A', hyp='ß' → reading 2.0 > diplomatic 1.0),
# i.e. the two lanes are NOT nested. Documented in DESIGN.md; here we test the common no-expansion case.
txt_noexpand = st.text(alphabet="abcde ABCDE äöéçñ абвгд", max_size=40)


@settings(max_examples=200, deadline=None)
@given(s=txt, lane=st.sampled_from(LANES))
def test_normalizer_idempotent(s, lane):
    """norm is idempotent — a second pass is a no-op (catches non-stable transforms). The single
    `norm(_, lane)` entrypoint also *is* the symmetry guarantee: GT and OCR can only be normalized
    the same way."""
    once = N.norm(s, lane)
    assert N.norm(once, lane) == once


@settings(max_examples=300, deadline=None)
@given(ref=txt_fffd, hyp=txt)
@example(ref="ab" + FFFD + "d", hyp="abZd")
def test_fffd_never_adds_errors(ref, hyp):
    """U+FFFD is a wildcard: our discounted counts are always ≤ the naive jiwer counts, never more;
    insertions are untouched; and with no U+FFFD present the two agree exactly."""
    r, h = N.norm(ref, "diplomatic"), N.norm(hyp, "diplomatic")
    ec = N.edit_counts(r, h)
    if ec is None:
        return
    naive = jiwer.process_characters(r, h, reference_transform=N._CHAR_T, hypothesis_transform=N._CHAR_T)
    assert ec.s + ec.d <= naive.substitutions + naive.deletions
    assert ec.i == naive.insertions
    if FFFD not in r:
        assert (ec.s, ec.d, ec.i) == (naive.substitutions, naive.deletions, naive.insertions)


@settings(max_examples=200, deadline=None)
@given(ref=txt, hyp=txt)
def test_rate_identity(ref, hyp):
    """(s+d+i)/n is consistent with EditCounts.rate() — the load-bearing invariant for pooling
    counts into a micro-average downstream."""
    ec = N.edit_counts(N.norm(ref, "reading"), N.norm(hyp, "reading"))
    rate = ec.rate()
    if rate is not None:
        assert abs(rate - (ec.s + ec.d + ec.i) / ec.n) < 1e-9


@settings(max_examples=300, deadline=None)
@given(ref=txt_noexpand, hyp=txt_noexpand)
def test_reading_le_diplomatic_no_expansion(ref, hyp):
    """With only length-preserving folds (case, diacritics, Cyrillic), the reading lane removes
    distinctions without changing length, so reading-lane CER ≤ diplomatic-lane CER. NB this does NOT
    hold once ß/ligature expansion is in play (see the alphabet note above) — the lanes aren't nested."""
    dip = N.edit_counts(N.norm(ref, "diplomatic"), N.norm(hyp, "diplomatic"))
    read = N.edit_counts(N.norm(ref, "reading"), N.norm(hyp, "reading"))
    dip_rate, read_rate = dip.rate(), read.rate()
    if dip_rate is not None and read_rate is not None:
        assert read_rate <= dip_rate + 1e-9


def test_lanes_diverge_on_long_s():
    """Golden: diplomatic keeps ſ / ﬁ; reading folds them."""
    assert "ſ" in N.norm("ſong", "diplomatic")
    assert "ſ" not in N.norm("ſong", "reading")
    assert N.norm("ﬁsh", "reading") == "fish"


def test_reading_markup_removes_delimiters_but_keeps_content():
    assert N.norm("<|ref|>Title<|/ref|>", "reading") == "title"
    assert N.norm("<b>Title</b>", "reading") == "title"
    assert N.norm("$x+y$", "reading") == "x+y"
    assert N.strip_markup("<|ref|>Title<|/ref|>").strip() == "Title"


def test_fffd_survives_tokenization_without_fragmenting():
    """A word holding an illegible glyph must stay ONE U+FFFD-bearing token — not fragment into
    bogus required tokens around the stripped glyph (norm 2026-07-01b regression guard)."""
    assert N.tokens("recon" + FFFD + "aissance ends") == ["recon" + FFFD + "aissance", "ends"]


def test_empty_reference_emits_integer_insertion_counts_with_undefined_rate():
    empty = N.edit_counts("", "abc")
    both_empty = N.edit_counts("", "")
    assert (empty.s, empty.d, empty.i, empty.h, empty.n, empty.rate()) == (0, 0, 3, 0, 0, None)
    assert (both_empty.s, both_empty.d, both_empty.i, both_empty.h, both_empty.rate()) == (
        0, 0, 0, 0, None,
    )


def test_all_wildcard_reference_keeps_integer_counts_and_zero_denominator():
    ec = N.edit_counts(FFFD * 2, "abc")
    assert (ec.s, ec.d, ec.i, ec.h, ec.n, ec.rate()) == (0, 0, 1, 0, 0, None)


def test_fffd_equal_char_excluded_from_reference_length():
    """Even when OCR literally emits U+FFFD, the GT wildcard is excluded from hits and length."""
    ec = N.edit_counts("a" + FFFD + "b", "a" + FFFD + "bXYZ")
    assert (ec.s, ec.d, ec.i, ec.h, ec.n) == (0, 0, 3, 2, 2)
    assert ec.rate() == 1.5


def test_fffd_word_dropped_from_recall_target():
    """End to end through the scorer: OCR that correctly reads the illegible-marked word must get
    recall 1.0 (the wildcard word is not required) and zero over-extraction charge on other words."""
    import scorer as S
    row = S.score_page("recon" + FFFD + "aissance ends here", "reconnaissance ends here")
    assert row["recall"] == 1.0


# --- norm 2026-07-20a: tag vocabulary, invisible codepoints, deterministic script tie-break ---
def test_angle_bracketed_gt_prose_is_transcription_not_markup():
    """Shape alone cannot separate an HTML tag from angle-bracketed prose, and the full GT contains
    a real instance (`<Gelenkknöchelchen>`, PageID 12341888). Deleting it would drop a recall target
    AND charge a model that read it correctly with over-extraction — wrong on both token axes."""
    import scorer as S
    assert N.tokens("the <Gelenkknöchelchen> was seen") == ["the", "gelenkknöchelchen", "was", "seen"]
    assert N.tokens("the <Hand> was seen") == ["the", "hand", "was", "seen"]
    assert N.tokens("5 < 3 and a > b") == ["5", "3", "and", "a", "b"]

    faithful = S.score_page("the <Gelenkknöchelchen> was seen", "the <Gelenkknöchelchen> was seen")
    assert faithful["recall"] == 1.0
    assert faithful["over_extraction"] == 0.0


@pytest.mark.parametrize(("markup", "expected"), [
    ("<b>Title</b>", ["title"]),
    ("<i>a</i> <em>b</em> <strong>c</strong>", ["a", "b", "c"]),
    ('<td colspan="2">cell</td>', ["cell"]),
    ("<BR/>x", ["x"]),                      # case-insensitive, self-closing
    ('<img src="a.png">y', ["y"]),
    ("<h2>Head</h2>", ["head"]),
    ("<blockquote>quoted</blockquote>", ["quoted"]),
])
def test_known_html_tags_are_still_stripped(markup, expected):
    assert N.tokens(markup) == expected


@pytest.mark.parametrize("invisible", ["­", "​", "‌", "‍", "⁠", "﻿"])
def test_invisible_codepoints_do_not_fragment_a_word(invisible):
    """A soft hyphen or zero-width joiner inside a GT word previously split it into two recall
    targets no model reading the joined form could match. Symmetric: both sides are normalized."""
    import scorer as S
    assert N.tokens(f"re{invisible}con") == ["recon"]
    assert S.score_page(f"re{invisible}con ends here", "recon ends here")["recall"] == 1.0


def test_diplomatic_lane_still_keeps_invisible_codepoints_and_brackets():
    """The diplomatic lane is deliberately minimal — NFC + whitespace only — so it must NOT inherit
    the reading lane's invisible-codepoint deletion."""
    assert N.norm("re­con <b>x</b>", "diplomatic") == "re­con <b>x</b>"


def test_dominant_script_tie_breaks_deterministically_not_by_insertion_order():
    """Equal script evidence must resolve identically regardless of which script appears first."""
    assert N.dominant_script("abаб") == N.dominant_script("абab") == "cyrillic"
    assert N.dominant_script("abcаб") == "latin"
    assert N.dominant_script("") is None


def test_provenance_records_the_alignment_engine_that_produced_the_supports():
    """The CER rate is invariant to which minimal alignment jiwer/rapidfuzz returns, but the emitted
    s/d/i/h split is not — so the engine version must be auditable from the scorecard alone."""
    p = N.provenance()
    assert p["norm_version"] == N.NORM_VERSION
    assert p["rapidfuzz_version"] and isinstance(p["rapidfuzz_version"], str)
