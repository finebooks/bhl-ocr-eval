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


# --- norm 2026-10-30a: repeated-header index tables are read one column group at a time ---
import pathlib  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _html(*rows, cell="td"):
    body = "".join("<tr>" + "".join(f"<{cell}>{c}</{cell}>" for c in row) + "</tr>" for row in rows)
    return f"<table>{body}</table>"


TWO_COLUMN_INDEX = [
    ["", "Plate.", "Species.", "", "Plate.", "Species."],
    ["abbreviatus, Jan.", "LIX.", "399", "auratus, Pfr.", "LI.", "335"],
    ["Abyssinicus, Ruppell", "XLVI.", "296", "aureus, Swainson", "XXIX.", "187"],
]
GROUP_MAJOR = ("plate. species. plate. species. abbreviatus, jan. lix. 399 abyssinicus, ruppell xlvi. 296 "
               "auratus, pfr. li. 335 aureus, swainson xxix. 187")


def test_repeated_header_index_is_read_one_printed_column_at_a_time():
    """The GT reads a two-column index down the left column, then the right; row-major tag
    stripping interleaved the two. Each 3-cell group is read top to bottom, then the next."""
    assert N.norm(_html(*TWO_COLUMN_INDEX), "reading") == GROUP_MAJOR


def test_th_header_cells_count_as_the_header_row():
    table = _html(TWO_COLUMN_INDEX[0], cell="th")[:-len("</table>")] + _html(*TWO_COLUMN_INDEX[1:])[len("<table>"):]
    assert N.norm(table, "reading") == GROUP_MAJOR


def test_header_missing_a_blank_cell_still_regroups_by_data_width():
    """Models often drop the second blank header cell: 5 header cells over 6 data columns. The repeat
    count comes from the non-empty labels (2), the group width from the data (6 / 2 = 3)."""
    header = ["", "Plate.", "Species.", "Plate.", "Species."]
    assert N.norm(_html(header, *TWO_COLUMN_INDEX[1:]), "reading") == GROUP_MAJOR


def test_four_cell_header_over_six_data_columns_keeps_each_entry_with_its_numbers():
    """[Plate, Species, Plate, Species] over 6 data columns must split 3+3, not 2+2+2 — splitting by
    header width would separate a name from its plate and species numbers."""
    header = ["Plate.", "Species.", "Plate.", "Species."]
    assert N.norm(_html(header, *TWO_COLUMN_INDEX[1:]), "reading") == GROUP_MAJOR


def test_a_single_label_repeated_per_column_reads_column_major():
    """Period 1: ['Vol. page'] x 3 over 3 columns, each cell a whole entry — three printed columns."""
    table = _html(["Vol. page"] * 3, ["a .. ii 1", "c .. ii 3", "e .. ii 5"], ["b .. ii 2", "d .. ii 4", "f .. ii 6"])
    assert N.norm(table, "reading") == "vol. page vol. page vol. page a .. ii 1 b .. ii 2 c .. ii 3 d .. ii 4 e .. ii 5 f .. ii 6"


def test_header_labels_match_ignoring_case_and_punctuation():
    header = ["", "Plate.", "Species.", "", "plate,", "SPECIES"]
    assert N.norm(_html(header, *TWO_COLUMN_INDEX[1:]), "reading").endswith("aureus, swainson xxix. 187")
    assert N.regroup_tables(_html(header, *TWO_COLUMN_INDEX[1:])) != _html(header, *TWO_COLUMN_INDEX[1:])


@pytest.mark.parametrize("header", [
    ["", "Plate.", "Species.", "", "Plafe.", "Species."],   # an OCR misspelling breaks the repeat: known limit
    ["", "Plate.", "Species."],                              # one printed column per table: nothing to regroup
    ["1", "1", "1", "1", "1", "1"],                          # a repeated value with no letters is not a header
])
def test_tables_without_a_repeated_word_header_are_unchanged(header):
    table = _html(header, *TWO_COLUMN_INDEX[1:])
    assert N.regroup_tables(table) == table


def test_headerless_grid_is_unchanged():
    """A grid whose first row is data has no header to repeat. Left row-major — a known limit."""
    table = _html(*TWO_COLUMN_INDEX[1:])
    assert N.regroup_tables(table) == table


def test_data_width_not_divisible_by_the_repeat_count_is_unchanged():
    table = _html(["Plate.", "Species.", "Plate.", "Species."], ["a", "I.", "1", "b", "II."])
    assert N.regroup_tables(table) == table


def test_real_data_table_is_unchanged():
    """An errata grid (chandra-ocr-2, page 12342054) is row-major in the GT; its header does not
    repeat, so it must keep the row-major reading."""
    table = (FIXTURES / "chandra-ocr-2_12342054_errata_table.html").read_text(encoding="utf-8")
    assert N.regroup_tables(table) == table
    assert N.regroup_tables(f"Druckfehler.\n{table}\nEnde") == f"Druckfehler.\n{table}\nEnde"


def test_markdown_pipe_tables_regroup_the_same_way():
    lines = ["| " + " | ".join(row) + " |" for row in TWO_COLUMN_INDEX]
    lines.insert(1, "|---|---|---|---|---|---|")
    assert N.norm("\n".join(lines), "reading") == GROUP_MAJOR
    plain = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert N.regroup_tables(plain) == plain


def test_colspan_and_rowspan_keep_columns_aligned():
    """A rowspan in the left group must not shift the right group's cells into it."""
    table = ("<table><tr><th></th><th>Pl.</th><th></th><th>Pl.</th></tr>"
             "<tr><td rowspan=\"2\">a</td><td>1</td><td>c</td><td>3</td></tr>"
             "<tr><td>2</td><td>d</td><td>4</td></tr></table>")
    assert N.norm(table, "reading") == "pl. pl. a 1 2 c 3 d 4"
    titled = _html(["Index"], *TWO_COLUMN_INDEX).replace("<td>Index</td>", "<td colspan=\"6\">Index</td>")
    assert N.regroup_tables(titled) == titled   # a title row is the first row: no repeat, unchanged


def test_caption_and_nested_tables_are_left_alone():
    captioned = _html(*TWO_COLUMN_INDEX).replace("<table>", "<table><caption>Bulimus</caption>")
    assert N.regroup_tables(captioned) == captioned
    nested = _html(*TWO_COLUMN_INDEX).replace("<td>LIX.</td>", "<td><table><tr><td>LIX.</td></tr></table></td>")
    assert N.norm(nested, "reading") == N.norm(N.strip_markup(nested), "reading")


def test_regrouping_only_reorders_cells_never_rewrites_them():
    """Entities, inner tags and unknown angle-bracket text pass through raw, so the fixed markup strip
    treats each cell exactly as before. Token recall is order-free, so it is unchanged."""
    import scorer as S
    rows = [row[:] for row in TWO_COLUMN_INDEX]
    rows[1][0], rows[2][3] = "abbreviatus, <i>D&#x27;Orb.</i>", "aureus, <Hand>"
    table = _html(*rows)
    regrouped = N.regroup_tables(table)
    assert "D&#x27;Orb." in regrouped and "<Hand>" in regrouped and "<i>" in regrouped
    assert sorted(N.tokens(regrouped)) == sorted(N.tokens(N.strip_markup(table)))
    gt = "Plate. Species. abbreviatus LIX. 399 Abyssinicus XLVI. 296 auratus LI. 335 aureus XXIX. 187"
    assert S.score_page(gt, table)["recall"] == S.score_page(gt, N.strip_markup(table))["recall"]


@settings(max_examples=200, deadline=None)
@given(s=st.text(alphabet=GLYPHS + "<>/|-\n", max_size=60))
def test_text_without_a_regroupable_table_is_untouched(s):
    """The regression guard: text with no repeated-header table reaches the markup strip unchanged,
    so every such page scores exactly as under the previous norm."""
    assert N.regroup_tables(s) == s
