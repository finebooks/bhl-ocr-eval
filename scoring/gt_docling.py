"""Pure helpers to read the finebooks/bhl-eval-impact `docling` column.

The dataset ships NO separate regions column — the `DoclingDocument` is the region source. Its flat
`texts` list is in reading order (the card: "three views, one pass … region order identical by
construction"), each item carrying a `content_layer` ("body" | "furniture") and a `label`
(text / caption / section_header / footnote / page_header / page_footer). So:

  - BODY (headline)      = the {text, caption, section_header, footnote} items
  - FURNITURE (ignore-set) = the {page_header, page_footer} items
  - regions              = (label, text) pairs for stratified recall

`docling` decodes to a dict via `datasets`, or is a JSON string in the raw parquet — both handled.
"""
import json

BODY_LAYER = "body"
FURNITURE_LAYER = "furniture"


def as_dict(docling):
    if isinstance(docling, dict):
        return docling
    if isinstance(docling, str) and docling:
        return json.loads(docling)
    return {}


def text_items(docling):
    """The flat `texts` list of a DoclingDocument, in reading order. Each item carries the raw
    `content_layer` / `label` / `text` / `prov` (bbox) fields — the geometry `body_text`/`regions`
    don't expose, which the unit-test generator needs."""
    return as_dict(docling).get("texts", []) or []


def layer_text(docling, layer):
    """Reading-order plain text of one content layer (blank string if none)."""
    return "\n".join(t.get("text", "") for t in text_items(docling)
                     if t.get("content_layer") == layer and t.get("text"))


def body_text(docling):
    return layer_text(docling, BODY_LAYER)


def furniture_text(docling):
    return layer_text(docling, FURNITURE_LAYER)


def regions(docling):
    """[{label, text}] for every text item (drives region_global_token_recall)."""
    return [{"label": t.get("label"), "text": t.get("text", "")}
            for t in text_items(docling) if t.get("text")]
