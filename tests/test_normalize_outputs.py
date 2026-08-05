"""Per-model output normalization: each transform ported from a driver, and its raising behaviour.

The transforms here decide what text a model is scored on, so these tests are behaviour pins, not
coverage: if one of them starts failing, a benchmark score has moved.
"""
import pandas as pd
import pytest

import normalize_outputs as NO


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------


def test_identity_returns_the_text_untouched():
    assert NO.identity("  spacing  kept \n") == "  spacing  kept \n"


def test_strip_outer_whitespace_trims_the_template_newline():
    assert NO.strip_outer_whitespace("\nTHE BIRDS OF GREAT BRITAIN\n\n") == "THE BIRDS OF GREAT BRITAIN"


def test_require_non_empty_passes_text_through():
    assert NO.require_non_empty("Chapter I.") == "Chapter I."


def test_require_non_empty_passes_empty_through_to_the_scorer():
    # POSTPROC 2. Measured over the 2026-08 run, 654 of 823 empty completions are on pages
    # whose GT body text is zero-length — correct answers. The frozen scorer already scores
    # empty-vs-empty as a match and empty-vs-text as a total miss, so raising discarded the
    # distinction and disqualified models for accuracy (Qianfan-OCR: 98 of 98 empties blank).
    assert NO.require_non_empty("") == ""
    assert NO.require_non_empty("real text") == "real text"


def test_drop_bbox_blocks_removes_ovis_image_placeholders_only():
    raw = ('# Plate XII\n\n'
           '<img src="images/bbox_120_340_800_910.jpg" />\n\n'
           'Fig. 3. — The lesser spotted woodpecker.\n\n'
           '<img src="images/bbox_10_20_30_40.jpg" />')
    assert NO.drop_bbox_blocks(raw) == "# Plate XII\n\nFig. 3. — The lesser spotted woodpecker."


def test_drop_bbox_blocks_keeps_ordinary_image_tags():
    raw = '<img src="plate.jpg" />\n\nText.'
    assert NO.drop_bbox_blocks(raw) == raw


def test_clean_repeated_substrings_collapses_a_runaway_tail_to_one_copy():
    page = "x" * 2000
    out = NO.clean_repeated_substrings(page + "abcde" * 12)
    assert out == page + "abcde"


def test_clean_repeated_substrings_is_a_noop_below_the_2000_char_floor():
    short = "y" * 100 + "abcde" * 12
    assert NO.clean_repeated_substrings(short) == short  # HunyuanOCR's own floor, kept verbatim


def test_clean_repeated_substrings_leaves_a_long_clean_page_alone():
    page = "".join(f"{i}. Anas boschas, the common wild duck of the northern marshes.\n"
                   for i in range(40))  # >2000 chars, every line distinct
    assert NO.clean_repeated_substrings(page) == page


def test_clean_repeated_substrings_also_collapses_genuine_repetition():
    # A known property of the card's rule, pinned rather than "fixed": it cannot tell a repetition
    # loop from a page that really ends in ten identical lines (a ruled table column, say). The
    # driver shipped this behaviour, so the port keeps it.
    page = "x" * 2000 + "| — |\n" * 12
    assert NO.clean_repeated_substrings(page) == "x" * 2000 + "| — |\n"


def test_unwrap_fence_removes_a_language_tagged_wrapper():
    assert NO.unwrap_fence("```markdown\n# Title\n\nBody text.\n```") == "# Title\n\nBody text."


def test_unwrap_fence_removes_a_bare_wrapper():
    assert NO.unwrap_fence("```\n# Title\n```") == "# Title"


def test_unwrap_fence_leaves_a_page_with_several_fences_alone():
    raw = "```\ncode\n```\n\ntext\n\n```\nmore\n```"
    assert NO.unwrap_fence(raw) == raw


def test_unwrap_fence_keeps_a_first_line_that_is_content_not_a_language_tag():
    assert NO.unwrap_fence("```# Title 1\nBody\n```") == "# Title 1\nBody"


def test_nuextract3_strip_thinking_keeps_the_answer_after_a_wellformed_block():
    assert NO.nuextract3_strip_thinking("<think>weighing it up</think>\n# Page\n") == "# Page"


def test_nuextract3_strip_thinking_tolerates_a_lone_open_tag():
    # DISCREPANCY with qianfan_strip_thinking, deliberately preserved: NuExtract3 needs BOTH tags
    # and never raises.
    assert NO.nuextract3_strip_thinking("  <think> truncated ") == "<think> truncated"


def test_qianfan_strip_thinking_keeps_the_answer_after_a_wellformed_block():
    assert NO.qianfan_strip_thinking("<think>layout</think>\n Parsed page. ") == "Parsed page."


def test_qianfan_strip_thinking_splits_on_a_closing_tag_alone():
    assert NO.qianfan_strip_thinking("reasoning</think>\nPage text.") == "Page text."


def test_qianfan_strip_thinking_raises_on_an_unterminated_block():
    with pytest.raises(ValueError, match="unterminated <think> block"):
        NO.qianfan_strip_thinking("<think>truncated reasoning trace")


def test_qianfan_strip_thinking_passes_ordinary_output_through_unstripped():
    assert NO.qianfan_strip_thinking("Parsed page.\n") == "Parsed page.\n"


def test_drop_front_matter_returns_the_body_below_olmocr_metadata():
    raw = ("---\nprimary_language: de\nis_rotation_valid: true\nis_table: false\n---\n"
           "Die Vögel Mitteleuropas.\n")
    # No trailing strip: the driver returned the regex group as-is.
    assert NO.drop_front_matter(raw) == "Die Vögel Mitteleuropas.\n"


def test_drop_front_matter_leaves_a_page_without_front_matter_alone():
    raw = "Die Vögel Mitteleuropas.\n\n--- a rule, not front matter ---\n"
    assert NO.drop_front_matter(raw) == raw


def test_strip_edge_specials_removes_the_sentence_markers():
    raw = "<｜begin▁of▁sentence｜>\nTHE TEXT\n<｜end▁of▁sentence｜>"
    assert NO.strip_edge_specials(raw) == "THE TEXT"


def test_strip_edge_specials_is_a_noop_when_they_are_absent():
    assert NO.strip_edge_specials("THE TEXT") == "THE TEXT"


DET_PAGE = (
    "<|det|>title [10,20,300,40]<|/det|>THE BIRDS OF GREAT BRITAIN\n"
    "<|det|>text [10,50,300,90]<|/det|>Chapter one begins here.\n"
    "and runs onto a second line\n"
    "<|det|>image [0,0,500,500]<|/det|>\n"
)


def test_remove_det_groups_blocks_and_drops_image_regions():
    assert NO.remove_det(DET_PAGE) == (
        "THE BIRDS OF GREAT BRITAIN\n\nChapter one begins here.\nand runs onto a second line")


def test_unlimited_ocr_to_markdown_also_sweeps_the_inline_ref_span_shape():
    raw = DET_PAGE + "<|ref|>A late caption<|/ref|><|det|>[[1,2,3,4]]<|/det|>\n"
    assert NO.unlimited_ocr_to_markdown(raw) == (
        "THE BIRDS OF GREAT BRITAIN\n\nChapter one begins here.\nand runs onto a second line\n"
        "A late caption")


def test_unlimited_ocr_to_markdown_raises_when_unmodelled_markup_survives():
    with pytest.raises(ValueError, match="unhandled grounding markup survived strip"):
        NO.unlimited_ocr_to_markdown("<|det|>text [1,2,3,4]<|/det|>Body <|unknown_tag|> tail")


def test_unlimited_ocr_to_markdown_returns_empty_when_only_image_regions_remain():
    # POSTPROC 2: empty out, not an error row. 239 of Unlimited-OCR's 250 such pages are blank.
    assert NO.unlimited_ocr_to_markdown("<|det|>image [0,0,500,500]<|/det|>\n") == ""


def test_unlimited_ocr_to_markdown_still_raises_on_leaked_markup():
    # The OTHER raise is kept: surviving markup means the parser failed, which is a real
    # modelling gap, not a blank page. POSTPROC 3 handles stray det/ref delimiters (they carry
    # no text), so this needs a token the transform genuinely does not model.
    with pytest.raises(ValueError, match="unhandled grounding markup survived strip"):
        NO.unlimited_ocr_to_markdown("text <|unknown_tag|> more")


def test_unlimited_ocr_to_markdown_removes_an_orphaned_header(): 
    # POSTPROC 3: the completion begins mid-block, so the opening delimiter is missing and the
    # paired regexes cannot see it. Its label and bbox digits must not reach the scored column.
    out = NO.unlimited_ocr_to_markdown("text [264, 188, 638, 205]<|/det|>1902")
    assert out == "1902"


def test_unlimited_ocr_to_markdown_removes_a_header_that_lost_both_delimiters():
    assert NO.unlimited_ocr_to_markdown("title [333, 63, 540, 93]TABLE") == "TABLE"


def test_unlimited_ocr_to_markdown_keeps_bracketed_numbers_inside_real_prose():
    # The bare-header rule is start-anchored precisely so this survives.
    text = "See plate [1, 2, 3, 4] for the figure."
    assert NO.unlimited_ocr_to_markdown(text) == text


def test_smoldocling_textless_doctags_yield_empty_not_an_error(monkeypatch):
    # POSTPROC 2: 317 of SmolDocling's 475 such pages are sparse_blank with zero-length GT.
    monkeypatch.setattr(NO, "doctags_to_markdown", lambda _doctags, _image=None: "")
    assert NO.smoldocling_doctags_to_markdown("<doctag/>") == ""


def test_smoldocling_guard_allows_empty_in_empty_out(monkeypatch):
    monkeypatch.setattr(NO, "doctags_to_markdown", lambda _doctags, _image=None: "")
    assert NO.smoldocling_doctags_to_markdown("") == ""


def test_smoldocling_guard_returns_the_converted_markdown(monkeypatch):
    monkeypatch.setattr(NO, "doctags_to_markdown", lambda _doctags, _image=None: "# Title")
    assert NO.smoldocling_doctags_to_markdown("<doctag>…</doctag>") == "# Title"


def test_doctags_to_markdown_keeps_furniture_and_drops_picture_placeholders():
    pytest.importorskip("docling_core")  # not a test-env dependency; runner declares it in PEP 723
    doctags = ("<doctag><page_header><loc_10><loc_10><loc_400><loc_30>RUNNING HEAD</page_header>"
               "<text><loc_20><loc_60><loc_480><loc_120>Hello world of birds.</text>"
               "<page_footer><loc_10><loc_480><loc_100><loc_495>42</page_footer></doctag>")
    out = NO.doctags_to_markdown(doctags)
    assert "RUNNING HEAD" in out and "42" in out  # FURNITURE layer kept
    assert "<!-- image -->" not in out


# ---------------------------------------------------------------------------
# Registry + resolution
# ---------------------------------------------------------------------------


def test_every_registered_model_has_named_callable_transforms():
    for model, transforms in NO.REGISTRY.items():
        assert isinstance(transforms, tuple), model
        for fn in transforms:
            assert callable(fn) and fn.__doc__, f"{model}: {fn}"


def test_registry_covers_the_markdown_producing_drivers():
    assert set(NO.REGISTRY) == {
        "deepseek-ai/DeepSeek-OCR", "deepseek-ai/DeepSeek-OCR-2", "rednote-hilab/dots.mocr",
        "rednote-hilab/dots.ocr", "google/gemma-4-12B-it", "zai-org/GLM-OCR",
        "tencent/HunyuanOCR", "lightonai/LightOnOCR-2-1B", "numind/NuExtract3",
        "allenai/olmOCR-2-7B-1025-FP8",
        "ATH-MaaS/OvisOCR2", "PaddlePaddle/PaddleOCR-VL-1.6", "baidu/Qianfan-OCR",
        "Qwen/Qwen3.5-9B", "ds4sd/SmolDocling-256M-preview", "baidu/Unlimited-OCR",
    }


def test_sibling_drivers_keep_their_differing_empty_output_rules():
    # deepseek-ocr v1 raises on empty, v2 does not; same for dots.mocr vs dots.ocr. Porting these
    # as one rule would change which pages become error rows.
    assert NO.require_non_empty in NO.REGISTRY["deepseek-ai/DeepSeek-OCR"]
    assert NO.require_non_empty not in NO.REGISTRY["deepseek-ai/DeepSeek-OCR-2"]
    assert NO.require_non_empty in NO.REGISTRY["rednote-hilab/dots.mocr"]
    assert NO.require_non_empty not in NO.REGISTRY["rednote-hilab/dots.ocr"]


@pytest.mark.parametrize("model", ["google/gemma-4-12B-it", "Qwen/Qwen3.5-9B"])
def test_generalists_unwrap_the_fence_before_the_empty_check(model):
    chain = NO.REGISTRY[model]
    assert chain.index(NO.unwrap_fence) < chain.index(NO.require_non_empty)


def test_the_two_generalists_normalize_identically():
    # gemma4-12b-port.py and qwen35-9b-port.py are a matched pair answering one question
    # (generalist vs specialist). A difference in post-processing would put a normalizer
    # decision into a comparison that is supposed to isolate the model.
    assert NO.REGISTRY["google/gemma-4-12B-it"] == NO.REGISTRY["Qwen/Qwen3.5-9B"]


def test_lightonocr2_is_strip_only_because_both_of_its_sources_are():
    # lighton-ocr2-port.py's upstream stored `content.strip()` and the offline recipe stored
    # `output.outputs[0].text.strip()`; the card gives no post-processing at all. The model's
    # wide CER-reading/CER-diplomatic split is a finding about the model, not a missing rule, so
    # nothing beyond the universal strip may be added here without a run to justify it.
    assert NO.REGISTRY["lightonai/LightOnOCR-2-1B"] == (NO.strip_outer_whitespace,)


def test_lightonocr2_leaves_a_blank_read_as_a_scored_empty_page():
    # No require_non_empty: neither source raised on an empty completion, so a blank page stays a
    # board row rather than becoming a durable error row.
    raw = pd.DataFrame({"PageID": [1], "model": ["lightonai/LightOnOCR-2-1B"], "raw_text": ["   "]})
    out = NO.normalize_frame(raw, raw_col="raw_text", out_col="markdown", model_col="model")
    assert out["markdown"].iloc[0] == ""
    assert out["normalize_error"].iloc[0] is None


def test_resolve_transforms_returns_the_registered_chain():
    assert NO.resolve_transforms("ATH-MaaS/OvisOCR2") == (
        NO.strip_outer_whitespace, NO.drop_bbox_blocks)


def test_resolve_transforms_fails_closed_on_an_unregistered_model():
    with pytest.raises(ValueError, match="no registered post-processing"):
        NO.resolve_transforms("some-org/BrandNewOCR")


def test_resolve_transforms_can_be_told_to_pass_through_explicitly():
    assert NO.resolve_transforms("some-org/BrandNewOCR", identity_for_unregistered=True) == (
        NO.identity,)


def test_resolve_transforms_names_the_reason_for_an_extraction_only_model():
    with pytest.raises(ValueError, match="no markdown column"):
        NO.resolve_transforms("LiquidAI/LFM2.5-VL-1.6B-Extract")


def test_registered_empty_chain_resolves_to_an_explicit_identity(monkeypatch):
    monkeypatch.setitem(NO.REGISTRY, "org/NoOpOCR", ())
    assert NO.resolve_transforms("org/NoOpOCR") == (NO.identity,)


# ---------------------------------------------------------------------------
# apply_transforms + normalize_frame
# ---------------------------------------------------------------------------


def test_apply_transforms_records_the_chain_it_ran():
    out, applied, error = NO.apply_transforms(
        "\n# Page\n", (NO.strip_outer_whitespace, NO.require_non_empty))
    assert (out, applied, error) == ("# Page", ["strip_outer_whitespace", "require_non_empty"], None)


def test_apply_transforms_reports_the_failing_transform_and_what_ran_before_it():
    # Uses a transform that still raises under POSTPROC 2 (leaked grounding markup means the
    # parser failed). require_non_empty no longer raises, so it can no longer stand in here.
    out, applied, error = NO.apply_transforms(
        "  text <|unknown_tag|> more  ", (NO.strip_outer_whitespace, NO.unlimited_ocr_to_markdown))
    assert out is None
    assert applied == ["strip_outer_whitespace"]
    assert error[0] == "unlimited_ocr_to_markdown"
    assert "unhandled grounding markup survived strip" in error[1]


def test_apply_transforms_passes_the_image_only_to_image_aware_transforms(monkeypatch):
    seen = []
    monkeypatch.setattr(NO, "doctags_to_markdown",
                        lambda _doctags, image=None: seen.append(image) or "# Converted")
    out, applied, error = NO.apply_transforms(
        "<doctag/>", (NO.smoldocling_doctags_to_markdown,), image="PAGE-IMAGE")
    assert (out, applied, error) == ("# Converted", ["smoldocling_doctags_to_markdown"], None)
    assert seen == ["PAGE-IMAGE"]


def _raw_frame():
    return pd.DataFrame({
        "PageID": [1, 2],
        "model": ["ATH-MaaS/OvisOCR2", "tencent/HunyuanOCR"],
        "raw_text": ['\n# Plate I\n\n<img src="images/bbox_1_2_3_4.jpg" />', "  Seite 12.  "],
    })


def test_normalize_frame_keeps_the_raw_text_and_stamps_provenance():
    out = NO.normalize_frame(_raw_frame(), raw_col="raw_text", out_col="markdown",
                             model_col="model")
    assert list(out["raw_text"]) == list(_raw_frame()["raw_text"])  # raw preserved byte-for-byte
    assert list(out["markdown"]) == ["# Plate I", "Seite 12."]
    assert list(out["postproc_version"]) == [NO.POSTPROC_VERSION] * 2
    assert out["transforms_applied"].iloc[0] == ["strip_outer_whitespace", "drop_bbox_blocks"]
    assert out["transforms_applied"].iloc[1] == ["strip_outer_whitespace",
                                                 "clean_repeated_substrings"]
    assert list(out["normalize_error"]) == [None, None]


def test_normalize_frame_accepts_a_single_model_table_without_a_model_column():
    raw = pd.DataFrame({"PageID": [1], "raw_text": ["  Text.  "]})
    out = NO.normalize_frame(raw, raw_col="raw_text", out_col="markdown",
                             model="rednote-hilab/dots.ocr")
    assert list(out["markdown"]) == ["Text."]


def test_normalize_frame_writes_a_durable_error_row_when_a_transform_raises():
    raw = pd.DataFrame({"PageID": [1], "model": ["baidu/Unlimited-OCR"],
                        "raw_text": ["  text <|unknown_tag|> more  "]})
    out = NO.normalize_frame(raw, raw_col="raw_text", out_col="markdown", model_col="model")
    assert out["markdown"].iloc[0].startswith(NO.ERROR_PREFIX)
    assert "unlimited_ocr_to_markdown" in out["markdown"].iloc[0]
    assert "unhandled grounding markup" in out["normalize_error"].iloc[0]
    assert out["raw_text"].iloc[0] == "  text <|unknown_tag|> more  "  # raw survives the failure


def test_normalize_frame_scores_an_empty_completion_instead_of_erroring_it():
    # POSTPROC 2, the counterpart: a blank read is now the scorer's problem, not an error row.
    raw = pd.DataFrame({"PageID": [1], "model": ["zai-org/GLM-OCR"], "raw_text": ["   "]})
    out = NO.normalize_frame(raw, raw_col="raw_text", out_col="markdown", model_col="model")
    assert out["markdown"].iloc[0] == ""
    assert out["normalize_error"].iloc[0] is None


def test_normalize_frame_fail_fast_aborts_instead_of_writing_an_error_row():
    raw = pd.DataFrame({"PageID": [1], "model": ["baidu/Unlimited-OCR"],
                        "raw_text": ["  text <|unknown_tag|> more  "]})
    with pytest.raises(ValueError, match=r"raw row 0 .*failed in unlimited_ocr_to_markdown"):
        NO.normalize_frame(raw, raw_col="raw_text", out_col="markdown", model_col="model",
                           fail_fast=True)


def test_normalize_frame_refuses_to_overwrite_the_raw_column():
    raw = pd.DataFrame({"model": ["rednote-hilab/dots.ocr"], "markdown": ["Text."]})
    with pytest.raises(ValueError, match="--raw-out-col"):
        NO.normalize_frame(raw, raw_col="markdown", out_col="markdown", model_col="model")


def test_normalize_frame_can_keep_a_legacy_markdown_column_under_a_second_name():
    raw = pd.DataFrame({"model": ["ATH-MaaS/OvisOCR2"],
                        "markdown": ['A\n\n<img src="images/bbox_1_2_3_4.jpg" />']})
    out = NO.normalize_frame(raw, raw_col="markdown", out_col="markdown", model_col="model",
                             raw_out_col="raw_text")
    assert out["raw_text"].iloc[0] == 'A\n\n<img src="images/bbox_1_2_3_4.jpg" />'
    assert out["markdown"].iloc[0] == "A"


def test_normalize_frame_rejects_a_missing_raw_column():
    raw = pd.DataFrame({"model": ["rednote-hilab/dots.ocr"], "text": ["Text."]})
    with pytest.raises(ValueError, match=r"missing required columns: \['raw_text'\]"):
        NO.normalize_frame(raw, raw_col="raw_text", out_col="markdown", model_col="model")


def test_normalize_frame_rejects_an_empty_input():
    with pytest.raises(ValueError, match="no rows"):
        NO.normalize_frame(pd.DataFrame({"raw_text": [], "model": []}), raw_col="raw_text",
                           out_col="markdown", model_col="model")


def test_normalize_frame_rejects_a_missing_model_id():
    raw = pd.DataFrame({"model": [None], "raw_text": ["Text."]})
    with pytest.raises(ValueError, match="missing model ID"):
        NO.normalize_frame(raw, raw_col="raw_text", out_col="markdown", model_col="model")


@pytest.mark.parametrize("value", [None, float("nan"), 123, True])
def test_normalize_frame_fails_closed_on_a_non_string_raw_cell(value):
    raw = pd.DataFrame({"model": ["rednote-hilab/dots.ocr"], "raw_text": [value]})
    with pytest.raises(ValueError, match=r"raw row 0, column 'raw_text'.*must be strings"):
        NO.normalize_frame(raw, raw_col="raw_text", out_col="markdown", model_col="model")


def test_normalize_frame_rejects_a_missing_image_column():
    raw = _raw_frame()
    with pytest.raises(ValueError, match="no image column 'image'"):
        NO.normalize_frame(raw, raw_col="raw_text", out_col="markdown", model_col="model",
                           image_col="image")


def test_postproc_version_is_a_string_stamp():
    assert isinstance(NO.POSTPROC_VERSION, str) and NO.POSTPROC_VERSION


# ---------------------------------------------------------------------------
# POSTPROC 3: DeepSeek grounding markup
# ---------------------------------------------------------------------------


def test_deepseek_strip_grounding_removes_labels_and_boxes_but_keeps_the_text():
    raw = ('<|ref|>title<|/ref|><|det|>[[370, 75, 568, 125]]<|/det|>\n# BIRDS  \n\n'
           '<|ref|>text<|/ref|><|det|>[[456, 168, 485, 184]]<|/det|>\nOF  ')
    assert NO.deepseek_strip_grounding(raw) == "# BIRDS  \n\nOF"


def test_deepseek_strip_grounding_drops_the_region_label_not_just_the_delimiters():
    # The bug this transform was written to avoid: REF_RE alone leaves "title" as scored text.
    out = NO.deepseek_strip_grounding("<|ref|>title<|/ref|><|det|>[[1, 2, 3, 4]]<|/det|>\nReal text")
    assert "title" not in out
    assert out == "Real text"


def test_deepseek_strip_grounding_yields_empty_for_a_bare_image_region():
    # A whole-page image region with no transcription. POSTPROC 2 says score it, not error it.
    assert NO.deepseek_strip_grounding("<|ref|>image<|/ref|><|det|>[[0, 0, 999, 1005]]<|/det|>") == ""


def test_deepseek_strip_grounding_raises_on_markup_it_did_not_model():
    with pytest.raises(ValueError, match="unhandled grounding markup survived strip"):
        NO.deepseek_strip_grounding("text <|unknown_tag|> more")


def test_both_deepseek_models_are_registered_with_the_strip():
    for model in ("deepseek-ai/DeepSeek-OCR", "deepseek-ai/DeepSeek-OCR-2"):
        assert NO.deepseek_strip_grounding in NO.REGISTRY[model], model
