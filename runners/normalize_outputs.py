# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "huggingface-hub", "pandas", "pyarrow", "docling-core>=2.89", "pillow", "beautifulsoup4", "markdownify==1.1.0", "six"]
# ///
"""Turn cached RAW model output into the scored `markdown` column — versioned and re-runnable.

Model-specific post-processing (dropping bbox placeholders, stripping grounding markup, converting
DocTags to markdown, trimming repetition loops) used to happen inside each inference driver's
`parse()`, which stored only the cleaned text. Every one of those rules is a judgement call, and
doing it at inference time destroys the pre-decision output: getting one wrong meant re-running the
model on GPUs. That breaks the benchmark's core promise — cache raw model output once, re-score
forever.

This runner is the missing step. It reads a table of raw completions, applies a per-model,
per-transform registry to produce the scored column, and stamps `POSTPROC_VERSION` on every row, so
a score can always be traced back to the post-processing that produced it (alongside the scorer's
own `norm_version` / `scorer_version`). Change a rule, bump the version, re-run this — no GPUs.

  uv run runners/normalize_outputs.py --raw ./run_raw.parquet --raw-col raw_text
  uv run runners/normalize_outputs.py --raw davanstrien/some-ocr-run --raw-col raw_text \
      --model ATH-MaaS/OvisOCR2 --out data/ovis_normalized.parquet
  # then score it through the normal path:
  uv run runners/score_dataset.py --ocr data/normalized.parquet --ocr-col markdown \
      --key-col PageID --model-col model --run-provenance-file ./producer-run.json

Run note: `docling-core` (SmolDocling only, imported lazily) resolves below this repo's
`[tool.uv] exclude-newer` pin, so `uv run` from inside the repo cannot install it. Normalizing
SmolDocling rows therefore needs the script run from outside the project directory (`cd /tmp &&
uv run --no-project <abs path>/normalize_outputs.py …`), which is how the drivers run it anyway.
Every other model normalizes fine in-repo — the import only happens on a SmolDocling row.

Every transform below is ported VERBATIM from the driver that owned it (the `*-port.py` files in
saturate-ocr-ports). This is a refactor, not a redesign: differing rules between drivers are kept
as they are and the discrepancy is noted in a comment, because unifying them would silently move
benchmark scores. Transforms that raise keep raising — a page whose output we failed to model
becomes a durable error row, never quietly corrupted text.
"""
import argparse
import pathlib
import re

import pandas as pd

# Bump this whenever ANY transform's behaviour changes, or when a model's transform list changes.
# It is recorded per row so a stored score is always attributable to the rules that produced it.
POSTPROC_VERSION = "3"

# Error sentinel for a row whose transform chain raised. score_dataset.py treats a leading
# "__ERR__" as a failed page and excludes it from scoring instead of scoring it as a blank read.
ERROR_PREFIX = "__ERR__"


# ---------------------------------------------------------------------------
# Transforms. Each one is small, pure, and individually testable. The docstring says what it does,
# WHY, and where the rule came from (driver + model card section).
# ---------------------------------------------------------------------------


def identity(text: str) -> str:
    """Return the text unchanged.

    For models registered with no transforms. Registering `()` and applying this EXPLICITLY (and
    recording the name in `transforms_applied`) is the point: a pass-through must be a decision on
    the record, not the absence of one.
    """
    return text


def strip_outer_whitespace(text: str) -> str:
    """`text.strip()` — the universal first step.

    Source: every markdown-producing driver calls `.strip()` on the completion content before
    anything else (e.g. dots-ocr-port.py `parse`, paddleocr-vl-16-port.py `parse`). Chat completions
    routinely carry a leading newline from the template.
    """
    return text.strip()


def require_non_empty(text: str) -> str:
    """POSTPROC 2: pass an empty transcription through to the scorer instead of raising.

    Until v1 this raised, on the reasoning that "a durable error row beats a silent empty
    transcription on the board". Measured at full scale, that rule was backwards. Of the 823
    empty completions across the 2026-08 run, **654 are on pages whose ground-truth body text
    is zero-length** — genuinely blank leaves. Every one of Qianfan-OCR's 98 empties is such a
    page: the model was right 98 times and the raise disqualified it for accuracy.

    The frozen scorer already distinguishes the two cases correctly and needs no help: empty
    output against empty GT scores as a match, empty output against real text scores as a total
    miss. Raising discarded that information and collapsed "correctly said nothing" into
    "failed". The 169 genuine content-page misses (158 SmolDocling, 11 Unlimited-OCR) are still
    penalised — as misses, which is what they are.

    Kept as a registered no-op rather than deleted from the chains: which models once asserted
    non-emptiness is part of the provenance record, and `transforms_applied` should keep showing
    that the decision was made deliberately. The sibling asymmetry this docstring used to
    preserve (DeepSeek-OCR raising where DeepSeek-OCR-2 does not; dots.mocr where dots.ocr does
    not) is now moot — no driver's chain can turn an empty page into an error row.
    """
    return text


def drop_bbox_blocks(text: str) -> str:
    """Drop `<img src="images/bbox_…">` placeholder blocks (OvisOCR2).

    Ported verbatim from ovis-ocr2-port.py. The card's OCR prompt asks the model to represent
    charts and figures as an HTML image tag with bbox coordinates in the filename; those tags carry
    no transcribed text, and their coordinate digits would score as invented characters.

    NOT ported, deliberately: the OvisOCR2 card's `clean_truncated_repeats` trimmer. The driver
    documents omitting it ("no-op under 8000 chars; these pages are shorter"), so adding it here
    would be a behaviour CHANGE, not a port. See `clean_repeated_substrings` for the one
    repeat-trimmer that a driver does apply (HunyuanOCR's).
    """
    blocks = text.split("\n\n")
    return "\n\n".join(b for b in blocks if not b.strip().startswith('<img src="images/bbox_'))


def clean_repeated_substrings(text: str, min_repeats: int = 10) -> str:
    """Trim a degenerate repeated tail (HunyuanOCR's official `clean_repeated_substrings`).

    Ported verbatim from hunyuan-ocr-15-port.py, which took it from the model card's official
    post-processing. Finds the shortest suffix that repeats at least `min_repeats` times at the end
    of the output and collapses the repeats to one. No-op under 2000 chars.

    This is repetition-loop handling and it is consequential: micro-averaged CER is dominated by
    individual runaway pages (RESULTS.md records a single repetition-loop page moving a model's
    aggregate from 0.030 to 0.044), so both the trimming and its 2000-char floor are load-bearing
    and must not be "improved".
    """
    n = len(text)
    if n < 2000:
        return text
    for length in range(2, n // 10):
        suffix = text[n - length:]
        count = 1
        while n - (count + 1) * length >= 0 and \
                text[n - (count + 1) * length: n - count * length] == suffix:
            count += 1
        if count >= min_repeats:
            return text[: n - length * (count - 1)]
    return text


def unwrap_fence(text: str) -> str:
    """Strip a single outer ```/```markdown fence if the whole reply is wrapped in one.

    Ported verbatim from gemma4-26b-a4b-port.py, and inherited unchanged by the two generalist
    drivers that replaced it (gemma4-12b-port.py, qwen35-9b-port.py). Format-only: a generalist often returns the
    transcription as one fenced block. Nothing inside the fence is touched, and a reply containing
    several fences (a page that genuinely has code/verse blocks) is left alone.
    """
    if not text.startswith("```") or not text.endswith("```") or text.count("```") != 2:
        return text
    body = text[3:-3]
    head, _, rest = body.partition("\n")
    return (rest if head.strip().isalpha() or not head.strip() else body).strip()


def nuextract3_strip_thinking(text: str) -> str:
    """Drop a `<think>…</think>` block (NuExtract3's tolerant variant).

    Ported verbatim from nuextract3-port.py. Thinking is disabled via `chat_template_kwargs`, so
    the tags should not appear; if a well-formed pair does, only the answer after `</think>` is
    kept.

    DISCREPANCY, preserved: qianfan-ocr-port.py implements the same idea differently — see
    `qianfan_strip_thinking`. NuExtract3 requires BOTH tags before splitting and never raises;
    Qianfan splits on `</think>` alone and RAISES on a lone `<think>`. Two drivers, two rules; both
    are kept because collapsing them would change which pages become error rows.
    """
    if "<think>" in text and "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def qianfan_strip_thinking(text: str) -> str:
    """Drop a Layout-as-Thought block if one appears (Qianfan-OCR's strict variant).

    Ported verbatim from qianfan-ocr-port.py. Thinking is off (the card: disable it for
    homogeneous documents, which scanned book pages are), so neither tag should ever show up. A
    well-formed block is stripped rather than trusted; an unterminated one means the output is a
    truncated reasoning trace, not markdown, and is raised so the row is a durable error instead of
    layout coordinates written into the markdown column.

    See `nuextract3_strip_thinking` for the deliberately different sibling rule.
    """
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if "<think>" in text:
        raise ValueError(
            "unterminated <think> block: thinking is disabled, so this output "
            "is a truncated reasoning trace, not document markdown"
        )
    return text


# olmOCR-2 is prompted to emit a YAML front-matter header (primary_language, is_rotation_valid,
# rotation_correction, is_table, is_diagram) above the transcription.
# The body is optional: on a blank page the model answers with the front matter alone.
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n(.*))?$", re.DOTALL)


def drop_front_matter(text: str) -> str:
    """Return the body below olmOCR-2's YAML front matter (or the text unchanged if absent).

    Ported verbatim from olmocr2-port.py, whose prompt asks for "a front matter section on top
    specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table,
    and is_diagram parameters" — page metadata, not page text, so scoring it would be scoring the
    model's answer to a different question.

    Note the body is returned EXACTLY as `FRONT_MATTER_RE` captured it, with no trailing strip:
    that is what the driver did, and a stray trailing newline is not worth a behaviour change. The
    parsed YAML itself is not this runner's business — the driver keeps it in its own
    `markdown_metadata` column.

    A page whose output is the front matter ALONE (no line after the closing `---`, which is how
    olmOCR-2 answers a blank page) has an empty body. The driver's pattern required a newline
    after the closing fence, so those pages fell through unchanged and their YAML was scored as
    transcription — 342 sparse/blank pages on the full run. Fixed without a POSTPROC_VERSION bump,
    following tiiuae/Falcon-OCR: this transform belongs to olmOCR-2 alone, so only that row's
    normalized text changes, and the row is re-normalized and re-scored with the fix.
    """
    m = FRONT_MATTER_RE.match(text)
    return (m.group(2) or "") if m else text


# skip_special_tokens=False (needed so Unlimited-OCR's grounding tags survive detokenization) also
# leaves the DeepSeek tokenizer's fullwidth-bar sentence markers on the edges of the completion.
EDGE_SPECIALS = ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>")


def strip_edge_specials(text: str) -> str:
    """Remove Unlimited-OCR's begin/end-of-sentence specials from the edges of the completion.

    Ported verbatim from unlimited-ocr-port.py `parse`. Idempotent: `removeprefix`/`removesuffix`
    are no-ops when the tokens are absent, so this is safe to re-run over a raw cache that was
    written either before or after the driver applied it.
    """
    for token in EDGE_SPECIALS:
        text = text.removeprefix(token).removesuffix(token)
    return text.strip()


# Card-verbatim OmniDocBench post-processing (block-anchored: one `<|det|>` header per block,
# content follows on the same or later lines).
DET_RE = re.compile(r"<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)", re.DOTALL)

# Recipe-inherited sweep for the inline DeepSeek-OCR shape (`<|ref|>text<|/ref|>` followed by an
# inline `<|det|>[[x,y,…]]<|/det|>` box) that the line-anchored card regex cannot match.
DET_SPAN_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
REF_RE = re.compile(r"<\|/?ref\|>")
# DeepSeek emits `<|ref|>LABEL<|/ref|>` where LABEL is a region type ("text", "title",
# "image"). REF_RE strips only the delimiters and would leave the LABEL behind as scored
# text, injecting a fake word per block. The whole span has to go.
REF_SPAN_RE = re.compile(r"<\|ref\|>.*?<\|/ref\|>", re.DOTALL)

# POSTPROC 3. Unlimited-OCR sometimes emits a grounding header that has LOST one or both of
# its delimiters — the completion can begin mid-block, e.g. "text [264, 188, 638, 205]<|/det|>".
# The paired-span regexes cannot see those, so the orphan survives and its label and bbox
# digits land in the scored column as invented characters. Each rule is deliberately narrow:
# a header is `label [bbox]`, and the bare-header rule is anchored to the START of the string
# only, because "word [1, 2, 3, 4]" is a shape real prose can produce mid-text.
ORPHAN_CLOSE_RE = re.compile(r"(?:^|\n)\s*[A-Za-z_-]+\s*\[[0-9,\s]+\]\s*<\|/det\|>")
ORPHAN_OPEN_RE = re.compile(r"<\|det\|>\s*[A-Za-z_-]+\s*\[[0-9,\s]+\]\s*(?!<\|/det\|>)")
LEADING_HEADER_RE = re.compile(r"^[A-Za-z_-]+\s*\[[0-9,\s]+\]")
BARE_DET_RE = re.compile(r"<\|/?det\|>")

# Anything still `<|…|>`-shaped after both passes is markup we did not model — the scored column
# must not carry it. Covers the ASCII bar and the fullwidth bar the DeepSeek tokenizer uses.
RESIDUAL_MARKUP_RE = re.compile(r"<\s*[|｜][^|｜]{0,40}[|｜]\s*>")


def remove_det(raw: str) -> str:
    """Strip `<|det|>type [bbox]<|/det|>` markers, group lines belonging to the
    same block with \\n, and separate different blocks with \\n\\n.

    Ported verbatim (via unlimited-ocr-port.py) from the Unlimited-OCR model card's OmniDocBench
    post-processing — including the `image` skip, which drops figure regions that carry a box but
    no text.
    """
    blocks = []
    cur = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = DET_RE.match(line)
        if m:
            category, content = m.group(1).strip(), m.group(2).strip()
            if category == "image":
                continue
            if cur is not None:
                blocks.append(cur)
            cur = [content] if content else []
            continue
        if cur is None:
            cur = []
        cur.append(line)
    if cur is not None:
        blocks.append(cur)
    return "\n\n".join("\n".join(b) for b in blocks).strip()


def unlimited_ocr_to_markdown(raw: str) -> str:
    """Grounded Unlimited-OCR output -> the plain text the benchmark scores.

    Ported verbatim from unlimited-ocr-port.py `to_markdown`. Two passes: the card's own
    `remove_det` (line-anchored, single-bracket `[bbox]` headers only), then a sweep for the shapes
    the card regex cannot see — double-bracket headers and inline `<|ref|>…<|/ref|>` spans.

    Raises if markup survives, or if nothing textual is left: both are error rows, not board rows.
    The benchmark scores TRANSCRIBED TEXT, and the markup is not text — leaving it in inflates
    every string metric and turns coordinate digits into fake characters, so a page whose markup we
    failed to model must fail loudly rather than land quietly corrupted on the board.
    """
    text = ORPHAN_CLOSE_RE.sub("\n", raw)   # POSTPROC 3: unpaired header BEFORE the paired passes
    text = remove_det(text)
    text = DET_SPAN_RE.sub("", text)
    text = ORPHAN_OPEN_RE.sub("", text)      # POSTPROC 3
    text = BARE_DET_RE.sub("", text)         # POSTPROC 3: delimiters with no header left
    text = REF_RE.sub("", text)
    text = LEADING_HEADER_RE.sub("", text.strip())  # POSTPROC 3: header that lost BOTH delimiters
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    residue = RESIDUAL_MARKUP_RE.findall(text)
    if residue:
        raise ValueError(f"unhandled grounding markup survived strip: {residue[:5]}")
    # POSTPROC 2: an empty result is returned, not raised. See require_non_empty.
    return text


def deepseek_strip_grounding(raw: str) -> str:
    """Grounded DeepSeek-OCR output -> the plain text the benchmark scores.

    ADDED POSTPROC 3, and it is a correction rather than a port: neither DeepSeek driver had
    a transform for this, so until now BOTH models were scored on their raw grounding markup.
    Every one of the 2,153 / 2,158 scored pages in the 2026-08 run still contained
    `<|ref|>…<|/ref|><|det|>[[bbox]]<|/det|>`, which is why their diplomatic CER (0.3418 /
    0.3377) sat 4x above their reading CER (0.0848 / 0.0898) and their furniture recall
    bottomed out at 0.187 — the coordinate digits were being scored as invented characters.
    Their sibling Unlimited-OCR emits the same markup family and always had a strip.

    Deliberately a POST-PROCESSING rule, not a driver change: the raw completions are cached,
    so correcting this costs a re-score rather than GPU time, and if the rule turns out to be
    wrong it can be revised and re-run again. That is the whole point of this layer.

    Order matters. The ref SPAN goes first (see REF_SPAN_RE — stripping only the delimiters
    leaves the region label as text), then the det span, then any stray delimiter. Verified
    over all 4,311 scored pages of both models: zero residual markup afterwards.
    """
    text = REF_SPAN_RE.sub("", raw)
    text = DET_SPAN_RE.sub("", text)
    text = REF_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if residue := RESIDUAL_MARKUP_RE.findall(text):
        raise ValueError(f"unhandled grounding markup survived strip: {residue[:5]}")
    return text


def doctags_to_markdown(doctags: str, image=None) -> str:
    """Convert DocTags to markdown via docling-core (verified against docling-core 2.89).

    Ported verbatim from smoldocling-port.py. Card path:
    DocTagsDocument.from_doctags_and_image_pairs -> DoclingDocument.load_from_doctags ->
    export_to_markdown. Two export deviations from the defaults, both deliberate:

    - included_content_layers adds FURNITURE: by default export_to_markdown emits BODY only,
      silently dropping <page_header>/<page_footer> — i.e. running heads and page numbers,
      which ARE page text and DO appear in OCR ground truth.
    - image_placeholder="": the default emits a literal "<!-- image -->" per <picture>, which
      is pure noise in a scored text column.

    `image` is OPTIONAL here, unlike in the driver, which always had the page pixels to hand.
    docling-core accepts `images=None` and the markdown export is image-independent (the image only
    sets page geometry; text comes from the tokens) — verified by converting the same DocTags with
    and without an image and diffing the markdown. Pass one anyway (`--image-col`) if the raw table
    carries it, so the port is exact rather than merely equivalent.
    """
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.common.content_layer import ContentLayer
    from docling_core.types.doc.document import DocTagsDocument

    dt_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], None if image is None else [image])
    doc = DoclingDocument.load_from_doctags(dt_doc, document_name="Document")
    return doc.export_to_markdown(
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE},
        image_placeholder="",
    ).strip()


def smoldocling_doctags_to_markdown(text: str, image=None) -> str:
    """DocTags -> markdown, raising when non-empty DocTags convert to nothing.

    Ported verbatim from smoldocling-port.py `parse`. SmolDocling emits a structured tag format
    (`<doctag><text><loc_..>..`), not markdown; storing it raw would score the model near zero on a
    markdown/plain-text column. docling-core NEVER raises on garbage — it returns empty markdown —
    so the guard is what turns "non-empty DocTags in, empty markdown out" into a durable error row
    instead of a silent blank that scores as an unread page.
    """
    # POSTPROC 2: DocTags that carry no text yield an empty transcription rather than an
    # error row. 317 of SmolDocling's 475 such pages are sparse_blank with zero-length GT,
    # i.e. correct answers. See require_non_empty.
    return doctags_to_markdown(text, image)


def chandra_html_to_markdown(text: str, *, include_headers_footers: bool = False) -> str:
    """Chandra's HTML layout blocks -> markdown (datalab-to/chandra-ocr-2).

    chandra-ocr-2 answers the `ocr_layout` prompt with HTML `<div data-bbox data-label>` blocks.
    This is the vendor's own conversion, `chandra/output.py: parse_markdown` from chandra-ocr
    v0.2.0 (tag 497dfd38d7625259307c81c979d44093e9f75f44); `parse_html`, `Markdownify` and the
    converter options are copied verbatim below (`get_image_name` is not needed, see below).

    Two decisions, both about the scored text rather than the model:
    - include_images=False. The vendor default (True) keeps `Image`/`Figure` blocks and writes
      a placeholder `<img src="<md5>_<n>_img.webp">` into them, which markdownify turns into a
      `![...](<hash>.webp)` link: invented characters on every plate page, plus the model's
      alt-text DESCRIPTION of the figure, which is not page text either.
    - include_headers_footers=False is the vendor default and is kept for chandra: Page-Header/
      Page-Footer blocks are dropped (the board's furniture column records this policy). The
      keyword exists for surya_blocks_to_markdown, which keeps them (see there).
    The vendor wraps the conversion in a try/except that returns "" on failure; here a failure
    RAISES, so it becomes an error row instead of a silent blank scored as an unread page.
    """
    import re as _re

    import six
    from bs4 import BeautifulSoup
    from markdownify import MarkdownConverter, re_whitespace

    re = _re  # the verbatim code below refers to `re`

    def get_image_name(*_args):  # only reached with include_images=True
        raise AssertionError("include_images is False")

    # ---- verbatim from chandra-ocr v0.2.0 chandra/output.py ----
    def parse_html(
        html: str, include_headers_footers: bool = False, include_images: bool = True
    ):
        soup = BeautifulSoup(html, "html.parser")
        top_level_divs = soup.find_all("div", recursive=False)
        out_html = ""
        image_idx = 0
        div_idx = 0
        for div in top_level_divs:
            div_idx += 1
            label = div.get("data-label")

            if label == "Blank-Page":
                continue

            # Skip headers and footers if not included
            if label and not include_headers_footers:
                if label in ["Page-Header", "Page-Footer"]:
                    continue
            if label and not include_images:
                if label in ["Image", "Figure"]:
                    continue

            if label in ["Image", "Figure"]:
                img = div.find("img")
                img_src = get_image_name(html, div_idx)

                # If no tag, add one in
                if img:
                    img["src"] = img_src
                    image_idx += 1
                else:
                    img = BeautifulSoup(f"<img src='{img_src}'/>", "html.parser")
                    div.append(img)

            # Strip img tags without src in non-image blocks (model hallucinations)
            if label not in ["Image", "Figure"]:
                for img_tag in div.find_all("img"):
                    if not img_tag.get("src"):
                        img_tag.decompose()

            # Wrap text content in <p> tags if no inner HTML tags exist
            if label in ["Text"] and not re.search(
                "<.+>", str(div.decode_contents()).strip()
            ):
                # Add inner p tags if missing for text blocks
                text_content = str(div.decode_contents()).strip()
                text_content = f"<p>{text_content}</p>"
                div.clear()
                div.append(BeautifulSoup(text_content, "html.parser"))

            content = str(div.decode_contents())
            out_html += content
        return out_html

    class Markdownify(MarkdownConverter):
        def __init__(
            self,
            inline_math_delimiters,
            block_math_delimiters,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.inline_math_delimiters = inline_math_delimiters
            self.block_math_delimiters = block_math_delimiters

        def convert_math(self, el, text, parent_tags):  # noqa: ARG002 (verbatim vendor signature)
            block = el.has_attr("display") and el["display"] == "block"
            if block:
                return (
                    "\n"
                    + self.block_math_delimiters[0]
                    + text.strip()
                    + self.block_math_delimiters[1]
                    + "\n"
                )
            else:
                return (
                    " "
                    + self.inline_math_delimiters[0]
                    + text.strip()
                    + self.inline_math_delimiters[1]
                    + " "
                )

        def convert_table(self, el, text, parent_tags):  # noqa: ARG002 (verbatim vendor signature)
            return "\n\n" + str(el) + "\n\n"

        def convert_a(self, el, text, parent_tags):
            text = self.escape(text)
            # Escape brackets and parentheses in text
            text = re.sub(r"([\[\]()])", r"\\\1", text)
            return super().convert_a(el, text, parent_tags)

        def escape(self, text, parent_tags=None):
            text = super().escape(text, parent_tags)
            if self.options["escape_dollars"]:
                text = text.replace("$", r"\$")
            return text

        def process_text(self, el, parent_tags=None):  # noqa: ARG002 (verbatim vendor signature)
            text = six.text_type(el) or ""

            # normalize whitespace if we're not inside a preformatted element
            if not el.find_parent("pre"):
                text = re_whitespace.sub(" ", text)

            # escape special characters if we're not inside a preformatted or code element
            if not el.find_parent(["pre", "code", "kbd", "samp", "math"]):
                text = self.escape(text)

            # remove trailing whitespaces if any of the following condition is true:
            # - current text node is the last node in li
            # - current text node is followed by an embedded list
            if el.parent.name == "li" and (
                not el.next_sibling or el.next_sibling.name in ["ul", "ol"]
            ):
                text = text.rstrip()

            return text
    # ---- end verbatim ----

    # parse_markdown(html, include_headers_footers=False, include_images=False), verbatim
    # converter options, minus the try/except (see above).
    html = parse_html(text, include_headers_footers, False)
    md_cls = Markdownify(
        heading_style="ATX",
        bullets="-",
        escape_misc=False,
        escape_underscores=True,
        escape_asterisks=True,
        escape_dollars=True,
        sub_symbol="<sub>",
        sup_symbol="<sup>",
        inline_math_delimiters=("$", "$"),
        block_math_delimiters=("$$", "$$"),
    )
    return md_cls.convert(html).strip()


def surya_blocks_to_markdown(raw: str) -> str:
    """The surya-ocr package's page blocks -> markdown (datalab-to/surya-ocr-2.5).

    `raw` is the package's page result as JSON (drivers/surya-ocr25-port.py): blocks in reading
    order, each with the model's raw label (`Text`, `Page-Header`, `Image`, ...) and its HTML.
    The package has no markdown renderer of its own, so each block is put back in the
    `<div data-label=...>` shape Datalab's chandra-ocr-2 writes, and the chandra converter does
    the HTML -> markdown step (same vendor, same label set, same HTML conventions).

    One difference from the chandra row, and it is a decision about the scored text: page
    headers and footers are KEPT. The surya package returns them as ordinary blocks with text;
    nothing in the package drops them. Chandra's converter drops them by default, and that
    default belongs to chandra's own `parse_markdown`, not to surya. Figures are dropped either
    way: the package does not read Image/Figure/Diagram blocks (their HTML is "").

    Bare-text blocks (no tags) are wrapped in <p>, as the chandra converter does for Text blocks,
    so neighbouring headers stay separate words.

    A block the package flagged as an error keeps its HTML: a capped block's partial text is
    what the package returns, and consolidate_run has already sealed such pages as truncated.
    Malformed JSON raises, so a broken row is an error row, not a blank page.
    """
    import html as _html
    import json

    blocks = json.loads(raw)
    divs = []
    for block in sorted(blocks, key=lambda b: b["reading_order"]):
        if block["skipped"] or not block["html"]:
            continue
        label = _html.escape(block["raw_label"] or block["label"], quote=True)
        body = block["html"]
        if not re.search("<.+>", body.strip()):
            # Page headers and footers often come back as bare text ("204", "APPENDIX"). The
            # chandra converter wraps bare text in <p> for Text blocks only, so two bare
            # headers would run together as "204APPENDIX". Wrap every bare block the same way.
            body = f"<p>{body.strip()}</p>"
        divs.append(f'<div data-label="{label}">{body}</div>')
    if not divs:
        return ""
    return chandra_html_to_markdown("".join(divs), include_headers_footers=True)


def falcon_strip_unused_tokens(text: str) -> str:
    """Drop leaked `>>UNUSED_N<<` vocab tokens (tiiuae/Falcon-OCR).

    The vendor pipeline occasionally emits raw unused-vocabulary tokens instead of text —
    on the full-2165 run, 76 pages carried `>>UNUSED_261<<` (71 of them genuinely blank
    pages whose only "text" was the token). That is markup leakage, not transcription, so
    it is stripped like OvisOCR2's bbox placeholders. Effect on the aggregate is tiny
    (0.1428 -> 0.1425): the model emits OTHER junk on most blank pages too, and that
    stays — it is real model output. Reported upstream to TII.

    Registered without a POSTPROC_VERSION bump, deliberately: no existing transform
    changed, and the leaderboard refuses mixed stamps — a bump would force a no-op
    re-normalize of every cached run. A NEW model's first rules are part of the
    version it is born under.
    """
    return re.sub(r">>UNUSED_\d+<<", "", text)


# The vendor renderer is ~4.8k lines across nemotron_parse/engine/, too large to port verbatim
# here, so it is loaded from the model repo at the revision the board's run served.
NEMOTRON_PARSE_REVISION = "62cb1e4ba153d113ff64aaf2be6e068c377cf53c"
_nemotron_render_markdown = None


def _load_nemotron_renderer():
    global _nemotron_render_markdown
    if _nemotron_render_markdown is None:
        import sys

        from huggingface_hub import snapshot_download

        root = snapshot_download("nvidia/Nemotron-3.5-Parse", revision=NEMOTRON_PARSE_REVISION,
                                 allow_patterns=["nemotron_parse/*", "nemotron_parse/**"])
        sys.path.insert(0, root)
        from nemotron_parse.postprocessing import render_markdown

        _nemotron_render_markdown = render_markdown
    return _nemotron_render_markdown


def nemotron_parse_to_markdown(text: str) -> str:
    """Render Nemotron 3.5 Parse's raw block stream as markdown (nvidia/Nemotron-3.5-Parse).

    The raw reply is a sequence of `<x_..><y_..>text<x_..><y_..><class_..>` blocks. The card's
    Postprocessing section turns it into a page with `postprocess_document(raw)["markdown"]`,
    which is `nemotron_parse.postprocessing.render_markdown` under default options: it repairs
    empty serialization prefixes, removes literal duplicate elements, drops the coordinate and
    class tokens, renders tables as HTML, and normalizes list bullets and soft hyphens. This
    calls exactly that function at the served revision (imported lazily, like docling-core for
    SmolDocling), so the scored text is the vendor's own rendering, as for chandra-ocr-2.
    """
    return _load_nemotron_renderer()(text).strip()


_UNDERSCORE_EMPHASIS = re.compile(r"(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])")


def strip_underscore_emphasis(text: str) -> str:
    """Unwrap markdown `_italic_` spans, keeping the text (nvidia/Nemotron-3.5-Parse).

    Nemotron 3.5 Parse marks italics with underscores (`_Corvus corax_`); species names make
    that frequent on this corpus — 5,881 spans on the full run, against 0-10 for every other row
    but SmolDocling (472). The frozen scorer strips `*` emphasis but not `_`, so without this the
    model is charged one inserted character per delimiter for formatting other rows express with
    `*` or not at all. Delimiters only: the wrapped text is kept, and an underscore not shaped
    like an emphasis delimiter (`column_1`, `kpi_strip`, runs of `___`) is left alone. The GT
    contains a single underscore across all 1,737 content pages, so this cannot remove a target.
    """
    return _UNDERSCORE_EMPHASIS.sub(r"\1", text)


# Transforms that take the page image as a second argument. Everything else is `(str) -> str`.
_MARKDOWN_IMAGE_LINK = re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)")


def oxr_drop_image_links(text: str) -> str:
    """Drop markdown image links for cropped figures (SoMarkAI/OXR-1.0).

    The vendor API writes every Picture block as `![](outputs/<task>/imgs/0_3.png)`, a link to a
    crop it saved on the server. That is a placeholder, not transcription: left in, the reading
    lane keeps `!`, the brackets and the file path, so every plate page is charged for invented
    characters (5x rough CER on the Conchologia plates in the 2026-10-26 smoke). Same decision
    as chandra_html_to_markdown's include_images=False and OvisOCR2's drop_bbox_blocks. The
    captions OXR writes next to the link (`*30. a*`) are page text and stay.

    Registered without a POSTPROC_VERSION bump, following tiiuae/Falcon-OCR: a new model's
    first rules are part of the version it is born under.
    """
    return _MARKDOWN_IMAGE_LINK.sub("", text)


# Unlimited-OCR-2's block markup. Anything matching after Baidu's converter ran is markup it did
# not model, and the scored column must not carry it (same fail-loud rule as Unlimited-OCR).
UOCR2_RESIDUAL_RE = re.compile(r"<\s*/?\s*(?:BLOCK|FORK|PAGE|MERGE|LEVEL)\b[^>]*>|<\s*[|｜][^|｜]{0,40}[|｜]\s*>")


def unlimited_ocr2_to_markdown(raw: str) -> str:
    """Grounded Unlimited-OCR-2 output -> the plain text the benchmark scores (baidu/Unlimited-OCR-2).

    The raw reply is `<BLOCK>type[x1, y1, x2, y2]<FORK>text` blocks, `<LEVELn>` on titles and
    `<MERGE>` between joined blocks. This runs Baidu's own Markdown converter (`convert` from
    utils/process_unlimitedOCR2.py, vendored verbatim in runners/unlimited_ocr2_postprocess.py),
    as chandra-ocr-2 and Nemotron 3.5 Parse are scored on their vendors' rendering. Its defaults
    are kept: running heads, footers and page numbers stay (as for Unlimited-OCR), titles become
    `#` headings. One decision on top: the converter writes every figure block as a link to a
    crop it never saves (`![](page1_x1_y1_x2_y2.jpg)`); that is a placeholder, not transcription,
    and is dropped, as for OXR-1.0 (see oxr_drop_image_links). A caption block next to a figure is
    page text and stays.

    Raises if block markup survives. An empty result is returned, not raised (POSTPROC 2).
    Registered without a POSTPROC_VERSION bump: a new model's first rules are part of the version
    it is born under.
    """
    import unlimited_ocr2_postprocess

    text = unlimited_ocr2_postprocess.convert(raw)
    text = _MARKDOWN_IMAGE_LINK.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    residue = UOCR2_RESIDUAL_RE.findall(text)
    if residue:
        raise ValueError(f"unhandled Unlimited-OCR-2 markup survived convert: {residue[:5]}")
    return text


IMAGE_AWARE = {"smoldocling_doctags_to_markdown"}


# ---------------------------------------------------------------------------
# Registry: model id -> the ordered transforms that turn its raw output into the scored column.
# Keys are the exact `SERVING["model"]` strings the drivers write into their `model` column.
# ---------------------------------------------------------------------------
REGISTRY = {
    # deepseek-ocr-port.py — plain markdown out; raises on an empty completion.
    "deepseek-ai/DeepSeek-OCR": (strip_outer_whitespace, deepseek_strip_grounding,
                                require_non_empty),
    # deepseek-ocr2-port.py — the twin driver, minus the empty-output raise (see require_non_empty).
    "deepseek-ai/DeepSeek-OCR-2": (strip_outer_whitespace, deepseek_strip_grounding),
    # falcon-ocr-port.py — vendor pipeline endpoint returns assembled markdown; strips leaked
    # unused-vocab tokens (see falcon_strip_unused_tokens). No empty-output raise: the pipeline
    # answers every page, and a blank page answered with junk must SCORE as junk.
    "tiiuae/Falcon-OCR": (strip_outer_whitespace, falcon_strip_unused_tokens),
    # dots-mocr-port.py — raises on empty; the recipe's "[OCR ERROR]" sentinel strings are gone.
    "rednote-hilab/dots.mocr": (strip_outer_whitespace, require_non_empty),
    # dots-ocr-port.py — the 1.7B sibling; strip only, no empty-output raise.
    "rednote-hilab/dots.ocr": (strip_outer_whitespace,),
    # gemma4-12b-port.py — generalist VLM; unwraps a whole-reply code fence, then raises on
    # empty. Order matters: the fence is unwrapped BEFORE the emptiness check, so a reply that is
    # nothing but an empty fence is an error row.
    "google/gemma-4-12B-it": (strip_outer_whitespace, unwrap_fence, require_non_empty),
    # glm-ocr-port.py — raises on empty. Its other guard (no `choices` in the response) is a
    # response-envelope check, not text post-processing, and stays in the driver.
    "zai-org/GLM-OCR": (strip_outer_whitespace, require_non_empty),
    # hunyuan-ocr-15-port.py — the card's official repetition-loop trimmer.
    "tencent/HunyuanOCR": (strip_outer_whitespace, clean_repeated_substrings),
    # kraken-ppocrv6-port.py — strip only, and deliberately nothing else. A CTC line recogniser
    # emits no markup, no fence and no grounding syntax; the driver has already joined the
    # recognised lines with newlines, exactly as kraken's own text serialisation does.
    #
    # NO require_non_empty, and that omission is load-bearing. This row's reason for existing is
    # the model card's claim that the family works "without hallucinations", and the board's
    # sparse/blank stratum is how that gets tested: on a blank leaf the segmenter finds no line
    # and the correct transcription is the empty string. Raising there would convert this
    # model's best-behaved pages into error rows and, under fail-closed scoring, make the whole
    # row ineligible. Compare zai-org/GLM-OCR, where an empty completion IS a failure because a
    # generative model returning nothing has not read the page. Same empty string, opposite
    # meaning; see require_non_empty's docstring for the general rule.
    #
    # Registered without a POSTPROC_VERSION bump, following tiiuae/Falcon-OCR: no existing
    # transform changed, and the leaderboard refuses mixed stamps, so a bump would force a
    # no-op re-normalize of every cached run.
    "kraken/PP-OCRv6-medium": (strip_outer_whitespace,),
    # lighton-ocr2-port.py — strip only. Both sources agree and neither does anything else: the
    # upstream saturate driver's `parse` stored `content.strip()`, the offline recipe stored
    # `output.outputs[0].text.strip()`, and the model card gives no post-processing guidance at
    # all. No empty-output raise either, matching both (see require_non_empty for the drivers
    # that do raise). The model's large CER-reading vs CER-diplomatic split on the board is a
    # formatting/orthography finding about the model, NOT a missing transform — inventing a
    # cleanup here would hide the finding behind a normalizer decision.
    "lightonai/LightOnOCR-2-1B": (strip_outer_whitespace,),
    # lighton-ocr3-port.py (both sizes) — strip only, as for LightOnOCR-2. The card gives no
    # post-processing. LightOn's client also repairs escaped `$` math delimiters
    # (fix_escaped_dollars); that is a decision about math markup, which this corpus does not
    # contain, so it is not applied. Thinking is disabled at the server and per request; the
    # canary had no `<think>` tags, so no thinking strip is registered.
    # indra-ocr-port.py — strip only. `raw` is the package's own markdown. No
    # require_non_empty: an empty page is the right answer on a blank leaf (as for kraken).
    "datalab-to/indra-ocr": (strip_outer_whitespace,),
    # chandra-ocr2-port.py — the vendor's HTML -> markdown conversion (see the function).
    "datalab-to/chandra-ocr-2": (strip_outer_whitespace, chandra_html_to_markdown),
    # surya-ocr25-port.py — `raw` is the surya-ocr package's page blocks as JSON; rendered with the
    # chandra converter, page headers and footers kept (see surya_blocks_to_markdown). No
    # require_non_empty: a blank leaf answered with no blocks is the right answer (as for kraken).
    # No POSTPROC_VERSION bump (new model).
    "datalab-to/surya-ocr-2.5": (strip_outer_whitespace, surya_blocks_to_markdown),
    "lightonai/LightOnOCR-3-4B": (strip_outer_whitespace,),
    "lightonai/LightOnOCR-3-0.8B": (strip_outer_whitespace,),
    # nemotron-parse-35-port.py — the vendor's own block-stream -> markdown rendering, then its
    # `_italic_` delimiters unwrapped (see both functions). Registered without a POSTPROC_VERSION
    # bump, following tiiuae/Falcon-OCR: no existing model's chain changed.
    "nvidia/Nemotron-3.5-Parse": (strip_outer_whitespace, nemotron_parse_to_markdown,
                                  strip_underscore_emphasis),
    # nuextract3-port.py, markdown mode only. NOTE the driver applies strip_thinking to the RAW
    # content without a prior `.strip()`; nuextract3_strip_thinking strips internally on both
    # branches, so the result is identical and the chain stays uniform.
    "numind/NuExtract3": (nuextract3_strip_thinking,),
    # olmocr2-port.py — YAML front matter split off the top; the body is NOT re-stripped.
    "allenai/olmOCR-2-7B-1025-FP8": (strip_outer_whitespace, drop_front_matter),
    # ovis-ocr2-port.py — bbox placeholder blocks dropped, applied AFTER the strip.
    "ATH-MaaS/OvisOCR2": (strip_outer_whitespace, drop_bbox_blocks),
    # ovis-ocr3-port.py — the same prompt as OvisOCR2 (prompts/document_markdown.txt), so the
    # same bbox placeholder blocks, dropped the same way. The card's repeated-tail cleanup is
    # not ported, as for OvisOCR2 (see drop_bbox_blocks). No POSTPROC_VERSION bump (new model).
    "ATH-MaaS/OvisOCR3": (strip_outer_whitespace, drop_bbox_blocks),
    # oxr-port.py — `raw` is the vendor API's assembled markdown; figure crops are written as
    # image links and dropped (see oxr_drop_image_links). LaTeX-wrapped labels
    # (`$\mathrm{Msst}$`) need no rule: the frozen reading-lane markup strip removes LaTeX
    # macros and delimiters. A footnote the vendor pipeline writes twice (its
    # full_overlap_dedupe is off by default) is real output and stays. No require_non_empty:
    # the pipeline answers blank leaves with "", which is the right answer (as for kraken).
    "SoMarkAI/OXR-1.0": (strip_outer_whitespace, oxr_drop_image_links),
    # minicpm-ocr-port.py — `raw` is the minicpm-ocr SDK's assembled page Markdown (layout regions
    # read one by one, joined in reading order; tables as HTML, formulas as LaTeX). Each figure is
    # written as `![image]()`, a placeholder, and dropped as for OXR-1.0. The row keeps footnotes
    # and side notes (`--keep-labels footnote aside_text`); the SDK leaves them out by default.
    # No require_non_empty: a page with no text regions assembles to "".
    "openbmb/MiniCPM-OCR-1.0": (strip_outer_whitespace, oxr_drop_image_links),
    # paddleocr-vl-16-port.py — strip only.
    "PaddlePaddle/PaddleOCR-VL-1.6": (strip_outer_whitespace,),
    # qianfan-ocr-port.py — strict thinking strip (raises on an unterminated block), then raises on
    # empty. finish_reason == "length" is recorded by the driver, deliberately not raised: the
    # partial markdown is still the best transcription available.
    "baidu/Qianfan-OCR": (strip_outer_whitespace, qianfan_strip_thinking, require_non_empty),
    # qwen35-9b-port.py — the other generalist VLM, same chain as gemma-4-12B for the same reason:
    # a general-purpose model volunteers a whole-reply code fence, and the fence is unwrapped
    # BEFORE the emptiness check so a reply that is nothing but an empty fence is an error row.
    # The two generalists are a matched pair and must normalize identically.
    "Qwen/Qwen3.5-9B": (strip_outer_whitespace, unwrap_fence, require_non_empty),
    # smoldocling-port.py — DocTags -> markdown with the raise-on-empty guard folded in.
    "ds4sd/SmolDocling-256M-preview": (strip_outer_whitespace, smoldocling_doctags_to_markdown),
    # unlimited-ocr-port.py — edge specials off, then the two-pass grounding strip, which raises
    # both on surviving `<|…|>` markup and on an empty result.
    "baidu/Unlimited-OCR": (strip_outer_whitespace, strip_edge_specials, unlimited_ocr_to_markdown),
    # unlimited-ocr2-port.py — Baidu's own converter, figure placeholders dropped (see
    # unlimited_ocr2_to_markdown). No require_non_empty: an empty page is an empty transcription.
    # No POSTPROC_VERSION bump (new model).
    "baidu/Unlimited-OCR-2": (strip_outer_whitespace, unlimited_ocr2_to_markdown),
}

# Drivers that produce an `extraction` column and no markdown at all: schema-guided image->JSON.
# They are named here rather than omitted so that pointing this runner at one fails with the
# reason rather than with "unregistered model".
EXTRACTION_ONLY = {
    # lfm2-vl-extract-port.py — schema fields as the system prompt, JSON out (`extraction`).
    "LiquidAI/LFM2.5-VL-1.6B-Extract":
        "lfm2-vl-extract-port.py stores an `extraction` JSON column, not markdown",
    # nuextract3-port.py with --template switches to the same shape; its markdown mode IS
    # registered above, so only the templated run is unnormalizable here.
}


def resolve_transforms(model: str, *, identity_for_unregistered: bool = False):
    """Look up a model's transform chain, failing closed on anything unrecognised.

    A registered model with an EMPTY transform tuple resolves to `(identity,)`, so a pass-through
    is applied and recorded by name rather than happening silently.
    """
    if model in EXTRACTION_ONLY:
        raise ValueError(f"model {model!r} has no markdown column: {EXTRACTION_ONLY[model]}")
    if model in REGISTRY:
        return REGISTRY[model] or (identity,)
    if identity_for_unregistered:
        return (identity,)
    raise ValueError(
        f"model {model!r} has no registered post-processing. Add it to REGISTRY (and bump "
        f"POSTPROC_VERSION), or pass --identity-for-unregistered to normalize it as a pass-through. "
        f"Registered: {sorted(REGISTRY)}"
    )


def apply_transforms(text: str, transforms, *, image=None) -> tuple:
    """Run a chain over one raw completion.

    Returns `(markdown, applied_names, error)`. `error` is None on success; on a raising transform
    it is `(transform_name, message)` and `markdown` is None — the caller writes the error row.
    A raise is never swallowed into text: that is the whole point of the raising transforms.
    """
    applied = []
    for fn in transforms:
        try:
            text = fn(text, image) if fn.__name__ in IMAGE_AWARE else fn(text)
        except Exception as exc:
            return None, applied, (fn.__name__, f"{type(exc).__name__}: {exc}")
        applied.append(fn.__name__)
    return text, applied, None


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _as_text(value, *, row_index, column):
    """Accept exact raw strings (including empty); reject missing and non-string cells.

    Mirrors score_dataset._as_text: a raw cache cell that is not a string is a producer bug, and
    coercing it here would hide it behind a plausible-looking score.
    """
    if isinstance(value, str):
        return value
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    kind = "missing" if missing else f"non-string {type(value).__name__}"
    raise ValueError(f"raw row {row_index}, column {column!r} contains a {kind} value; "
                     "raw cells must be strings (an empty string is valid).")


def _as_image(value):
    """Best-effort decode of one image cell (datasets dict / raw bytes / PIL), or None."""
    if value is None:
        return None
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    raw = value.get("bytes") if isinstance(value, dict) else value
    if isinstance(raw, (bytes, bytearray)):
        import io

        return Image.open(io.BytesIO(bytes(raw)))
    raise ValueError(f"unsupported image value: {type(value)}")


def _load(path_or_repo, split="train"):
    """Load raw rows from a local parquet/dir or an HF dataset id."""
    p = pathlib.Path(path_or_repo)
    if p.exists() and p.suffix == ".parquet":
        return pd.read_parquet(p)
    from datasets import load_dataset

    return load_dataset(str(p) if p.exists() else path_or_repo, split=split).to_pandas()


def normalize_frame(raw, *, raw_col, out_col, model_col=None, model=None, image_col=None,
                    raw_out_col=None, identity_for_unregistered=False, fail_fast=False):
    """Normalize a raw DataFrame, returning a new frame with the scored column added.

    Input columns are preserved untouched — including the raw text, which is the artifact this
    whole runner exists to keep re-usable.
    """
    if raw.empty:
        raise ValueError("raw input has no rows.")
    required = {raw_col} | ({model_col} if model_col else set())
    if missing_columns := required - set(raw.columns):
        raise ValueError(f"raw input is missing required columns: {sorted(missing_columns)}")
    if raw_col == out_col and not raw_out_col:
        raise ValueError(
            f"--raw-col and --out-col are both {raw_col!r}; the normalized column would overwrite "
            "the raw text. Pass --raw-out-col to keep the raw text under a second name."
        )
    if image_col and image_col not in raw.columns:
        raise ValueError(f"raw input has no image column {image_col!r}.")

    markdowns, applied_names, errors, versions = [], [], [], []
    for index, record in enumerate(raw.to_dict("records")):
        model_value = record.get(model_col) if model_col else model
        if model_value is None or (not isinstance(model_value, str) and pd.isna(model_value)) \
                or not str(model_value).strip():
            raise ValueError(f"raw row {index} has a missing model ID.")
        transforms = resolve_transforms(str(model_value),
                                        identity_for_unregistered=identity_for_unregistered)
        text = _as_text(record.get(raw_col), row_index=index, column=raw_col)
        # Already sealed upstream (consolidate_run.py: a producer error or a non-`stop` finish
        # reason). There is no transcription to normalize, and running the chain over a sentinel
        # risks a transform rewriting it into something that no longer reads as an error.
        if text.startswith(ERROR_PREFIX):
            markdowns.append(text)
            applied_names.append([])
            versions.append(POSTPROC_VERSION)
            errors.append(None)
            continue
        image = _as_image(record.get(image_col)) if image_col else None
        out, applied, error = apply_transforms(text, transforms, image=image)
        if error:
            name, message = error
            if fail_fast:
                raise ValueError(f"raw row {index} (model {model_value!r}) failed in {name}: {message}")
            out = f"{ERROR_PREFIX}{name}: {message}"
            errors.append(f"{name}: {message}")
        else:
            errors.append(None)
        markdowns.append(out)
        applied_names.append(applied)  # on failure: the transforms that ran BEFORE the raise
        versions.append(POSTPROC_VERSION)

    out_frame = raw.copy()
    if raw_out_col:
        out_frame[raw_out_col] = out_frame[raw_col]
    out_frame[out_col] = markdowns
    out_frame["transforms_applied"] = applied_names
    out_frame["postproc_version"] = versions
    out_frame["normalize_error"] = errors
    return out_frame


def main():
    ap = argparse.ArgumentParser(
        description="Apply versioned per-model post-processing to cached raw OCR output.")
    ap.add_argument("--raw", required=True, help="raw model output: HF dataset id or local parquet/dir")
    ap.add_argument("--split", default="train", help="split to load when --raw is an HF dataset")
    ap.add_argument("--raw-col", default="raw_text", help="column holding the RAW model completion")
    ap.add_argument("--out-col", default="markdown", help="column to write the normalized text to")
    ap.add_argument("--raw-out-col", default=None,
                    help="also copy the raw text here (required when --raw-col == --out-col)")
    ap.add_argument("--model-col", default="model", help="per-row model column (else use --model)")
    ap.add_argument("--model", default=None, help="model id for a single-model raw table")
    ap.add_argument("--image-col", default=None,
                    help="optional page-image column, passed to image-aware transforms")
    ap.add_argument("--identity-for-unregistered", action="store_true",
                    help="normalize unregistered models as an explicit pass-through instead of failing")
    ap.add_argument("--fail-fast", action="store_true",
                    help="abort on the first raising transform instead of writing an error row")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent.parent / "data" / "normalized.parquet"))
    args = ap.parse_args()

    raw = _load(args.raw, args.split)
    # --model wins for a single-model table; otherwise the per-row column is used.
    model_col = None if args.model else args.model_col
    if not model_col and not args.model:
        raise ValueError("pass --model for a single-model table, or --model-col for a mixed one.")
    print(f"{len(raw)} raw rows from {args.raw}; postproc_version={POSTPROC_VERSION}")

    out_frame = normalize_frame(
        raw, raw_col=args.raw_col, out_col=args.out_col, model_col=model_col, model=args.model,
        image_col=args.image_col, raw_out_col=args.raw_out_col,
        identity_for_unregistered=args.identity_for_unregistered, fail_fast=args.fail_fast)

    failed = int(out_frame["normalize_error"].notna().sum())
    models = out_frame[model_col] if model_col else pd.Series([args.model] * len(out_frame),
                                                              index=out_frame.index)
    for model_value, group in out_frame.groupby(models):
        names = " -> ".join(group["transforms_applied"].iloc[0])
        print(f"  {model_value}: {len(group)} rows [{names}]")
    print(f"{len(out_frame) - failed} normalized, {failed} error rows")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_parquet(out, index=False)
    print(f"\nnormalized output -> {out}")


if __name__ == "__main__":
    main()
