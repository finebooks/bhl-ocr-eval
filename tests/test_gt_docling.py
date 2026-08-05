"""Golden tests for the docling GT extraction — load-bearing: if this drifts, every board number is
wrong. Mirrors the real DoclingDocument shape (flat `texts` with content_layer + label)."""
import json

import gt_docling as G

DOC = {
    "texts": [
        {"label": "section_header", "content_layer": "body", "text": "THE WAXWING"},
        {"label": "text", "content_layer": "body", "text": "An abundant winter visitor."},
        {"label": "caption", "content_layer": "body", "text": ""},          # empty → dropped
        {"label": "page_header", "content_layer": "furniture", "text": "18"},
        {"label": "page_footer", "content_layer": "furniture", "text": "BIRDS"},
    ]
}


def test_body_text_is_body_layer_only():
    assert G.body_text(DOC) == "THE WAXWING\nAn abundant winter visitor."


def test_furniture_text_is_furniture_layer_only():
    assert G.furniture_text(DOC) == "18\nBIRDS"


def test_regions_drop_empty_and_keep_labels():
    r = G.regions(DOC)
    assert [x["label"] for x in r] == ["section_header", "text", "page_header", "page_footer"]
    assert all(x["text"] for x in r)  # the empty caption is gone


def test_accepts_json_string_or_dict():
    assert G.as_dict(json.dumps(DOC)) == DOC
    assert G.as_dict(DOC) == DOC


def test_empty_docling_is_safe():
    for empty in (None, "", {}, {"texts": []}):
        assert G.body_text(empty) == "" and G.furniture_text(empty) == "" and G.regions(empty) == []
