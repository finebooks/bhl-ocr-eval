# Axis catalogue

> Generated from `scorer.REGISTRY` by `scripts/gen_axis_catalogue.py` — **do not hand-edit.**

Scorer `3.0` · norm `2026-07-20a` · grapheme mode `codepoint` · illegible marker = U+FFFD wildcard.

The headline scores the docling **BODY** layer (furniture is an ignore-set); `over_extraction` scores against the **full** text so faithful furniture reading is not punished; furniture is reported as a directionless policy diagnostic. Token diagnostics are global evidence proxies, not spatial attribution. See `DESIGN.md`.

## coverage

### `recall`  (↑ higher better, lane: format_immune)
- **What:** Count-aware fraction of legible BODY GT tokens supported by the global OCR token multiset.
- **Why:** The format-immune backbone catches dropped content without penalizing formatting or order.
- **Caveats:** Body only; order-free, but duplicate token multiplicities are respected.
- **Field precedent:** bWER / Nougat set-recall / OCR-D bag-of-words.

## faithfulness

### `cer_diplomatic`  (↓ lower better, lane: diplomatic)
- **What:** Character error rate, NFC + case-sensitive, body only.
- **Why:** Field-comparable transcription fidelity: long-s, ligatures, and case remain distinctions.
- **Caveats:** Normalization- and alignment-sensitive; aggregate by pooling raw counts. A zero-length normalized reference has no page rate, but its insertions remain in benchmark pooling.
- **Field precedent:** dinglehopper / OCR-D / ISRI CER.

### `cer_reading`  (↓ lower better, lane: reading)
- **What:** Character error rate, NFKC + case-fold + narrow markup stripping, body only.
- **Why:** Separates reading ability from transcription convention.
- **Caveats:** Hides long-s, case, and ligature errors by design; not field-comparable. A zero-length normalized reference has no page rate, but its insertions remain in benchmark pooling.
- **Field precedent:** OCR-D level-2/3 normalized lane.

## over_extraction

### `over_extraction`  (↓ lower better, lane: format_immune)
- **What:** Count-aware fraction of OCR tokens exceeding the legible FULL-GT token multiset.
- **Why:** Flags hallucination, boilerplate, and repeated text while allowing faithful furniture.
- **Caveats:** Output-normalized and position-blind. U+FFFD credit is one-to-one and token-local: visible characters and length must match, with each marker realizing exactly one codepoint; this cannot verify which hidden glyph was intended.
- **Field precedent:** ISRI 'generated / spurious' characters.

## fidelity

### `script_match`  (↑ higher better, lane: na)
- **What:** Fraction of OCR letters in the GT's dominant Unicode script.
- **Why:** Catches script switching or transliteration that CER cannot isolate.
- **Caveats:** Undefined without GT-script or OCR-letter evidence; current sample is nearly all Latin.
- **Field precedent:** No field equivalent — our differentiator.

## furniture

### `furniture_global_token_recall`  (↔ policy diagnostic (directionless), lane: format_immune)
- **What:** Count-aware FURNITURE-token evidence found anywhere in the OCR output.
- **Why:** Shows whether the model tends to emit page furniture; it is a policy diagnostic, not quality.
- **Caveats:** Global token-evidence proxy, not spatial attribution; directionless by design.
- **Field precedent:** OmniDocBench abandon-set (reported separately).

## layout

### `region_global_token_recall`  (↑ higher better, lane: format_immune)
- **What:** Per-label count-aware GT-token evidence found anywhere in the OCR output.
- **Why:** Shows which kinds of text may be dropped without claiming spatial assignment.
- **Caveats:** Global token-evidence proxy, not spatial attribution: labels are pooled per page and overlapping labels may reuse the same OCR evidence.
- **Field precedent:** Flexible-Character-Accuracy-spirit proxy.
