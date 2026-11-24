"""The Falcon-OCR 1.5 driver's check for errors the vendor pipeline swallows."""
import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parent.parent / "drivers" / "falcon-ocr-15-port.py"
_spec = importlib.util.spec_from_file_location("falcon_ocr_15_port", _PATH)
assert _spec is not None and _spec.loader is not None, f"cannot load driver at {_PATH}"
DRV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DRV)

SWALLOWED = [{"content": "", "error": "[Errno None] Can not write request body for http://127.0.0.1:8000/v1/chat/completions"}]


def test_a_swallowed_vllm_error_is_reported():
    assert "Can not write request body" in DRV.pipeline_error(SWALLOWED, "")


def test_an_empty_answer_with_no_error_is_a_real_blank_page():
    assert DRV.pipeline_error([{"content": ""}], "") is None
    assert DRV.pipeline_error([], None) is None


def test_text_wins_over_a_region_level_error():
    # The layout route can fail one crop and still assemble the rest of the page.
    regions = [{"content": "Plate IV."}, {"content": "", "error": "timeout"}]
    assert DRV.pipeline_error(regions, "Plate IV.") is None
