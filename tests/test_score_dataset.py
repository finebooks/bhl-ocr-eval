"""Pure guards for external OCR provenance and page-key normalization."""
import json

import pandas as pd
import pytest

import score_dataset as SD


def test_as_text_preserves_exact_strings_and_deliberate_empty_output():
    assert SD._as_text("hello\n", row_index=4, column="ocr") == "hello\n"
    assert SD._as_text("", row_index=4, column="ocr") == ""


@pytest.mark.parametrize("value", [None, float("nan"), pd.NA, 123, 1.5, True])
def test_as_text_fails_closed_with_row_and_column_context(value):
    with pytest.raises(ValueError, match=r"OCR row 7, column 'markdown'.*must be strings"):
        SD._as_text(value, row_index=7, column="markdown")


def test_join_key_collapses_numeric_roundtrips():
    keys = {SD._join_key(value) for value in (123, "123", 123.0, "123.0", " 123 ")}
    assert keys == {"123"}


def test_join_key_handles_missing_and_nonnumeric_values():
    assert SD._join_key(None) is None
    assert SD._join_key(float("nan")) is None
    assert SD._join_key(pd.NA) is None
    assert SD._join_key(" abc ") == "abc"
    assert SD._join_key("12.5") == "12.5"


def test_join_key_preserves_huge_leading_zero_and_scientific_string_ids():
    huge = "99999999999999999999999999999999999999999999999999"
    assert SD._join_key(huge) == huge
    assert SD._join_key("000123") == "000123"
    assert SD._join_key("000123.0") == "000123.0"
    assert SD._join_key("1e3") == "1e3"
    assert SD._join_key("9.999999999999999999e40") == "9.999999999999999999e40"


def test_read_run_provenance_requires_json_object_and_preserves_null_prompt(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"prompt": None, "note": "model-native OCR mode"}))
    assert SD._read_run_provenance(path)["prompt"] is None
    path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        SD._read_run_provenance(path)


def test_read_run_provenance_rejects_empty_producer_before_harness_metadata(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="non-empty producer JSON object"):
        SD._read_run_provenance(path)
