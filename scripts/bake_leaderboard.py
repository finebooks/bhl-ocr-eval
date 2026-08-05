# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Bake leaderboard JSON snapshots into the static board at leaderboard/index.html.

Reads the full-benchmark report output, joins model metadata
(params/type/display name — METADATA, not a scoring concern, so it lives here and not in the
frozen scorer), and regenerates the block between the __BAKE_START__/__BAKE_END__ markers.
The HTML is the product for v0; data-driven serving is tracked separately (issue #21).

  uv run scripts/bake_leaderboard.py \
      --full data/full-2026-08/leaderboard.json
  # then deploy the baked page to the Space that serves it:
  hf upload finebooks/bhl-ocr-leaderboard docs/leaderboard/index.html index.html --repo-type space

The 372-page sample board was DROPPED on 2026-08-04. Its raw completions were cached
neither locally nor in any bucket — only its scorecards survive — so it could not be
re-scored under POSTPROC 3, and nine of its ten models are in the full run anyway. Left
in place it would have shown DeepSeek-OCR at 0.1017 beside the corrected 0.0617, a gap
a reader would attribute to sample size rather than to the grounding-markup fix.
"""

import argparse
import datetime
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
HTML = ROOT / "docs" / "leaderboard" / "index.html"

# Parameter counts (billions), model family, short display name.
#
# Counts are MEASURED, not transcribed from the model name or card — every weight in the
# repo's safetensors index, summed across dtypes, so vision towers and projectors are
# included. Regenerate with:
#
#   from huggingface_hub import get_safetensors_metadata
#   sum(get_safetensors_metadata(model_id).parameter_count.values()) / 1e9
#
# Measured 2026-08-03. Two names mislead badly enough to be worth stating: GLM-OCR is
# 1.33B (widely listed as 0.9B, including in uv-scripts/ocr/models.json), and
# olmOCR-2-7B is 8.29B once its vision tower is counted.
META = {
    "ATH-MaaS/OvisOCR2": (0.853, "specialist", "OvisOCR2"),
    "datalab-to/surya-ocr-2": (0.686, "specialist", "surya-ocr-2"),
    "zai-org/GLM-OCR": (1.325, "specialist", "GLM-OCR"),
    "rednote-hilab/dots.mocr": (3.039, "specialist", "dots.mocr"),
    "rednote-hilab/dots.ocr": (3.039, "specialist", "dots.ocr"),
    "deepseek-ai/DeepSeek-OCR": (3.336, "specialist", "DeepSeek-OCR"),
    "lightonai/LightOnOCR-2-1B": (1.006, "specialist", "LightOnOCR-2"),
    "allenai/olmOCR-2-7B-1025-FP8": (8.294, "specialist", "olmOCR-2"),
    "ds4sd/SmolDocling-256M-preview": (0.256, "specialist", "SmolDocling"),
    "PaddlePaddle/PaddleOCR-VL-1.6": (0.959, "specialist", "PaddleOCR-VL-1.6"),
    "PaddlePaddle/PP-OCRv6_medium": (None, "classical", "PP-OCRv6"),
    "tesseract-5": (None, "classical", "Tesseract 5"),
    # Added 2026-08-04 for the full-2026-08 run, measured the same way. Note
    # Unlimited-OCR is 3.336B — byte-identical to DeepSeek-OCR's count, and it ships
    # DeepSeek's NGram anti-repetition processor pattern too, so treat the two as
    # architecturally related rather than independent data points.
    "deepseek-ai/DeepSeek-OCR-2": (3.389, "specialist", "DeepSeek-OCR-2"),
    "baidu/Unlimited-OCR": (3.336, "specialist", "Unlimited-OCR"),
    "baidu/Qianfan-OCR": (4.741, "specialist", "Qianfan-OCR"),
    # The only VLM on the board, and the reason the generalist-vs-specialist comparison
    # exists again after the router-served entries were withdrawn. Type is "VLM", NOT
    # "generalist": the board's CATCOL/filter vocabulary is {VLM, specialist, classical},
    # and an unknown type renders a colourless dot with no error.
    "Qwen/Qwen3.5-9B": (9.653, "VLM", "Qwen3.5-9B"),
    # No router-served entries. Six models scored via the Inference Providers router were
    # held out on 2026-08-03 — provider, quantization and serving configuration were never
    # recorded, so their scores are not attributable to the named model. Rationale and the
    # held-out scorecards: data/sample/excluded-inference-providers/README.md. A model
    # returns here only once it has been re-run under a pinned serve.
}


# Entries whose board identifier is NOT a resolvable Hub repo. Two of the models here are
# pipelines rather than single checkpoints, which is why they cannot be named by one id:
# PP-OCRv6 is a detection model plus a recognition model, and Tesseract is a CPU binary with
# per-language trained data and no Hub presence at all. The key stays as-is because it is the
# scorecard's `model` value and the board's join key; `repo` is what may be linked, and is
# None here precisely so nothing links to an id that 404s.
PIPELINES = {
    "PaddlePaddle/PP-OCRv6_medium": {
        "det": "PaddlePaddle/PP-OCRv6_medium_det",
        "rec": "PaddlePaddle/PP-OCRv6_medium_rec",
    },
    "tesseract-5": {"engine": "tesseract 5 — CPU binary, no Hub repo"},
}


def _round(value, digits=4):
    return None if value is None else round(value, digits)


def _row(model_id, card, *, with_strata):
    params, mtype, display = META[model_id]
    org = model_id.split("/")[0] if "/" in model_id else "—"
    row = {
        "name": display,
        "org": org,
        "model_id": model_id,
        "repo": None if model_id in PIPELINES else model_id,
        "pipeline": PIPELINES.get(model_id),
        "type": mtype,
        "params": params,
        "cer_read": _round(card["cer_reading_micro"]),
        "cer_dip": _round(card["cer_diplomatic_micro"]),
        "ci": [_round(card["cer_ci"][0]), _round(card["cer_ci"][1])]
        if card.get("cer_ci")
        else None,
        "recall": _round(card["recall_micro"]),
        "overx": _round(card["over_extraction_micro"]),
        "furn": _round(card["furniture_global_token_recall_micro"]),
        "lang": {k: _round(v) for k, v in (card.get("by_language") or {}).items()},
        "n": card["n"],  # already the successful-page count (submitted - errors)
        "err": card.get("producer_errors", card["errors"]),
        # Repetition-loop rate: a board column, not a diagnostic. Truncated pages are
        # excluded from the aggregates, so a HIGHER rate means the row was scored on an
        # easier effective page set — read it alongside CER, never on its own.
        "loop": _round(card.get("truncation_rate"), 5),
        "eligible": card.get("eligible", True),
    }
    if with_strata:
        strata = card.get("by_sample_stratum") or {}
        row["content_cer"] = _round(
            (strata.get("content") or {}).get("cer_reading_micro")
        )
        row["sparse_cer"] = _round(
            (strata.get("sparse_blank") or {}).get("cer_reading_micro")
        )
    return row


def load_board(path, *, with_strata):
    board = json.loads(pathlib.Path(path).read_text())
    scored, excluded = [], []
    # order = eligible models; ineligible_order = errored models, which still carry
    # conditional diagnostics unless every page errored (then no score exists at all).
    for model_id in board["order"] + board.get("ineligible_order", []):
        card = board["scorecards"][model_id]
        if model_id not in META:
            raise SystemExit(
                f"Model {model_id!r} has no META entry — add params/type/display first."
            )
        if card["cer_reading_micro"] is None:  # all pages errored → no score exists
            _, _, display = META[model_id]
            excluded.append(
                {
                    "name": display,
                    "model_id": model_id,
                    "err": card["errors"],
                    "n": card["n"],
                }
            )
            continue
        scored.append(_row(model_id, card, with_strata=with_strata))
    scored.sort(key=lambda r: r["cer_read"])
    return board, scored, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=str(ROOT / "data" / "full" / "leaderboard.json"))
    ap.add_argument(
        "--baked-date",
        default=None,
        help="date stamped on the board (defaults to today, ISO)",
    )
    args = ap.parse_args()

    full_board, full_rows, full_excluded = load_board(args.full, with_strata=True)
    fp = full_board["provenance"]

    prov = {
        "norm": fp["norm_version"],
        "scorer": fp["scorer_version"],
        "postproc": fp["postproc_version"],
        "grapheme": fp["score"]["grapheme_mode"],
        "volumes": 6,
        "full_pages": fp["page_count"],
        "baked": args.baked_date
        or datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat(),
    }

    block = (
        "/* __BAKE_START__ — generated by scripts/bake_leaderboard.py; do not edit by hand */\n"
        f"const PROV={json.dumps(prov)};\n"
        f"const DATA={json.dumps(full_rows)};\n"
        f"const EXCLUDED={json.dumps(full_excluded)};\n"
        "/* __BAKE_END__ */"
    )
    html = HTML.read_text()
    pattern = re.compile(r"/\* __BAKE_START__.*?__BAKE_END__ \*/", re.DOTALL)
    if not pattern.search(html):
        raise SystemExit("Bake markers not found in leaderboard/index.html.")
    HTML.write_text(pattern.sub(lambda _: block, html))
    print(
        f"Baked {len(full_rows)} models on the full 2,165-page benchmark "
        f"({len(full_excluded)} excluded) into {HTML.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
