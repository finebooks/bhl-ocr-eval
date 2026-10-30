# /// script
# requires-python = ">=3.10"
# dependencies = ["jiwer>=4,<5", "rapidfuzz>=3,<4"]
# ///
"""Named, versioned, SYMMETRIC text normalization for the BHL OCR scorer.

Two pipelines, applied *identically* to ground truth and OCR output so "same treatment on both
sides" is enforced by construction, not by comment:

  - **diplomatic** — NFC, case-SENSITIVE, keeps long-s (ſ), ligatures (ﬁ), diacritics. Minimal:
    NFC + whitespace collapse only. The field-comparable lane (dinglehopper / OCR-D "level 1").
    A model that reads ſ as `s` is scored *wrong* here — that's the point.
  - **reading** — NFKC (folds ſ→s, ﬁ→fi), case-fold, unify quote/hyphen variants, de-hyphenate
    line-end breaks, strip a tiny FIXED markup set. Reading-ability (OCR-D "level 2/3").

Also owns the character edit-op accounting (`EditCounts`) with the **U+FFFD-illegible WILDCARD**
convention: the GT uses U+FFFD to mark an illegible glyph (194 pages, 1366×); it matches ANY OCR
character and is never counted as an error and never stripped.

Design note (see DESIGN.md): edit ops are counted over Unicode **codepoints**, not grapheme
clusters (jiwer's model). For NFC Latin the gap is negligible; it is stamped in the provenance
(`GRAPHEME_MODE`) so the number is honest. Bump `NORM_VERSION` to re-score cached outputs — never
silently change a past number.
"""
import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser

import jiwer

# Bump these to re-score from cached raw outputs (the "run once, re-score forever" property).
# 2026-10-30a: the reading lane reads a table whose header labels repeat (an index printed in two or
# three columns, emitted as one table: [ , Plate, Species, , Plate, Species]) one column group at a
# time, top to bottom, instead of row by row. Every other table, and all non-table text, is unchanged.
# This intentionally changes reading-lane CER on those pages. Token recall is order-free; it moves only
# where the old row-by-row strip had joined a hyphen-ended cell onto the next row's first cell.
# 2026-07-20a: the HTML/XML strip matches real tag shapes from a fixed tag vocabulary instead of any
# short `<...>` pair, so angle-bracketed GT prose (e.g. the historical German `<Gelenkknöchelchen>`)
# is transcribed content again rather than deleted from the reading lane; and the reading lane drops
# soft hyphens and zero-width formatting characters so an invisible codepoint cannot fragment one
# word into two recall targets. Both intentionally change reading-lane and token scores.
# 2026-07-07a: paired special-token and LaTeX math delimiters are removed while their enclosed
# transcription survives. This intentionally changes reading-lane scores.
# Earlier U+FFFD fixes keep illegible-bearing words intact and exclude wildcard positions from both
# errors and the reference denominator.
NORM_VERSION = "2026-10-30a"
GRAPHEME_MODE = "codepoint"  # not grapheme-cluster; Latin-NFC gap is tiny — see DESIGN.md

REPLACEMENT = "�"  # GT illegible marker → scoring wildcard (matches any OCR char, never an error)

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s�]", re.UNICODE)  # keep U+FFFD: it marks a WORD as illegible-bearing
_QUOTES = {ord(c): "'" for c in "'’‘ʼ`´"}
_QUOTES.update({ord(c): '"' for c in "“”"})
_HYPHENS = {ord(c): "-" for c in "‐‑‒–—―"}
_DEHYPH = re.compile(r"-\s*\n\s*")  # join line-end hyphenation: "recon-\naissant" -> "reconnaissant"
# Invisible formatting codepoints carry no transcription. Deleted (not spaced) in the reading lane so
# a soft-hyphenated or zero-width-joined GT word stays ONE token instead of fragmenting into two
# recall targets no model could match. The diplomatic lane stays deliberately minimal and keeps them.
_INVISIBLE = {ord(c): None for c in "­​‌‍⁠﻿"}

# FROZEN markup strip (reading lane only). Removes ONLY structure a model may add that plain-text GT
# never contains: special/grounding tokens, bbox arrays, HTML/XML tags, markdown + LaTeX scaffolding.
# Do NOT grow this per model — new junk should hit `over_extraction`, not earn a regex patch here.

# A FIXED vocabulary of HTML/XML tag names, not "any short <...> pair". Shape alone cannot separate a
# tag from angle-bracketed prose (`<Gelenkknöchelchen>`, `<Hand>`), and deleting GT prose would both
# drop real recall targets and charge a model that read them correctly with over-extraction. Anything
# outside this vocabulary is treated as transcription.
_HTML_TAGS = (
    "a|b|big|blockquote|br|caption|center|code|col|colgroup|div|em|figcaption|figure|font|h[1-6]|hr|"
    "i|img|li|mark|ol|p|pre|s|small|span|strike|strong|sub|sup|table|tbody|td|tfoot|th|thead|tr|tt|u|ul"
)
_MARKUP = [
    # Delimiters are scaffolding, not the transcription they wrap: remove only the fixed delimiter.
    re.compile(r"<\|/?[A-Za-z0-9_]+\|>"),            # <|ref|>Title<|/ref|> -> Title
    re.compile(r"\[\[\s*\d[\d,\s]*\]\]"),           # [[290, 20, 490, 45]] bbox arrays
    re.compile(rf"</?(?:{_HTML_TAGS})(?:\s[^>\n]{{0,80}})?\s*/?>", re.I),  # <b>T</b>, <td colspan=2>
    re.compile(r"^[ \t]*#{1,6}[ \t]+", re.M),       # markdown headings
    re.compile(r"^[ \t]*[-*+][ \t]+", re.M),        # markdown list bullets
    re.compile(r"^[ \t]*[-=_*]{3,}[ \t]*$", re.M),  # --- === horizontal rules / table separators
    re.compile(r"\\[a-zA-Z]+\*?"),                  # \gamma \mathcal LaTeX macros
    re.compile(r"[`*|~{}$\\]"),                     # $x+y$ -> x+y; emphasis/table/residual LaTeX marks
]


def strip_markup(t: str | None) -> str:
    t = t or ""
    for rx in _MARKUP:
        t = rx.sub(" ", t)
    return t


# --- repeated-header index tables (reading lane only) ---
# A multi-column printed index ("name …… Plate. Species." twice across the page) is often emitted
# as ONE table whose header repeats: [ , Plate, Species, , Plate, Species]. Stripping its tags reads
# it row by row, which interleaves two printed columns; the GT reads one printed column top to
# bottom, then the next. When the header's non-empty labels repeat r times, the table is read as r
# equal-width column groups, each group top to bottom. Every other table is left exactly as it was
# (row-major after the tag strip): no header repeat, a width that does not divide by r, a headerless
# grid, text outside the cells, or a nested table. Only the ORDER of cells changes — cell text is
# passed through raw (tags, entities and all) so the strip below treats it as it always did.
_HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.S | re.I)
_PIPE_SEPARATOR = re.compile(r":?-{2,}:?")
_LABEL_NOISE = re.compile(r"[\W_]+")


class _TableGrid(HTMLParser):
    """Rows of raw cell text for one HTML table; colspan/rowspan expand to empty cells so column
    positions line up. `ok` turns False on anything the regrouping should not touch."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.rows: list[list[str]] = []
        self.ok = True
        self._cell: list[str] | None = None   # raw pieces of the open cell
        self._at = 0                           # its column in the current row
        self._spanned: set[tuple[int, int]] = set()
        self._depth = 0

    @staticmethod
    def _span(attrs, name):
        digits = re.search(r"\d+", dict(attrs).get(name) or "")
        return max(1, min(int(digits.group(0)), 50)) if digits else 1

    def _commit(self):
        """Close the open cell. Also called on an implied close (`<td>a<td>b`, a missing `</td>`)."""
        if self._cell is not None:
            self.rows[-1][self._at] = "".join(self._cell)
            self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._depth += 1
            self.ok = self.ok and self._depth == 1   # nested table: leave the whole thing alone
        elif tag == "tr":
            self._commit()
            self.rows.append([])
        elif tag in ("td", "th"):
            self._commit()
            if not self.rows:
                self.rows.append([])
            row = self.rows[-1]
            while (len(self.rows) - 1, len(row)) in self._spanned:
                row.append("")
            r, c = len(self.rows) - 1, len(row)
            colspan, rowspan = self._span(attrs, "colspan"), self._span(attrs, "rowspan")
            self._at = c
            row.extend([""] * colspan)
            self._spanned.update((r + dr, c + dc) for dr in range(1, rowspan) for dc in range(colspan))
            self._cell = []
        elif self._cell is not None:
            self._cell.append(self.get_starttag_text() or "")

    def handle_endtag(self, tag):
        if tag in ("td", "th", "tr"):
            self._commit()
        elif tag == "table":
            self._commit()
            self._depth -= 1
        elif self._cell is not None and tag not in ("tbody", "thead", "tfoot"):
            self._cell.append(f"</{tag}>")

    def _text(self, text):
        if self._cell is not None:
            self._cell.append(text)
        elif text.strip():
            self.ok = False   # caption or stray text: regrouping would move or drop it

    def handle_data(self, data):
        self._text(data)

    def handle_entityref(self, name):
        self._text(f"&{name};")

    def handle_charref(self, name):
        self._text(f"&#{name};")

    def handle_comment(self, data):
        self._text(f"<!--{data}-->")


def _html_grid(html: str) -> list[list[str]] | None:
    parser = _TableGrid()
    parser.feed(html)
    parser.close()
    parser._commit()
    return parser.rows if parser.ok and parser._depth == 0 else None


def _header_repeats(header: list[str]) -> int | None:
    """How many times the header's non-empty labels repeat (case/punctuation-insensitive), or None.
    `["", "Plate.", "Species.", "Plate", "species"]` -> 2; `["Vol. page"] * 3` -> 3."""
    labels = [_LABEL_NOISE.sub(" ", strip_markup(h)).strip().casefold() for h in header]
    labels = [label for label in labels if label]
    n = len(labels)
    for period in range(1, n // 2 + 1):
        if n % period == 0 and all(labels[i] == labels[i % period] for i in range(n)) \
                and any(ch.isalpha() for label in labels[:period] for ch in label):
            return n // period
    return None


def _group_major(grid: list[list[str]]) -> str | None:
    """The regrouped reading of a table, or None to leave it as it is."""
    if len(grid) < 2:
        return None
    repeats = _header_repeats(grid[0])
    width = max(len(row) for row in grid[1:])
    if not repeats or width == 0 or width % repeats:
        return None
    step = width // repeats
    parts = [" ".join(c for c in grid[0] if c.strip())]
    for start in range(0, width, step):
        for row in grid[1:]:
            parts.extend(c for c in row[start:start + step] if c.strip())
    # Space-joined, never newline-joined: the regrouping must not invent a line-end de-hyphenation.
    return " " + " ".join(parts) + " "


def _pipe_tables(text: str) -> str:
    """Regroup markdown pipe tables (2+ consecutive lines starting with `|`)."""
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        j = i
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            j += 1
        if j - i < 2:
            out.append(lines[i])
            i += 1
            continue
        grid = []
        for line in lines[i:j]:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(_PIPE_SEPARATOR.fullmatch(c or "--") for c in cells):
                grid.append(cells)
        regrouped = _group_major(grid)
        out.extend([regrouped] if regrouped is not None else lines[i:j])
        i = j
    return "\n".join(out)


def regroup_tables(t: str) -> str:
    """Read repeated-header index tables group by group; leave every other table unchanged."""
    if "<table" not in t.lower() and "|" not in t:
        return t

    def html(match):
        grid = _html_grid(match.group(0))
        regrouped = _group_major(grid) if grid is not None else None
        return regrouped if regrouped is not None else match.group(0)

    return _pipe_tables(_HTML_TABLE.sub(html, t))


def _diplomatic(t):
    # NFC, case-sensitive, keep everything historical (ſ, ﬁ, diacritics). Whitespace only.
    return _WS.sub(" ", unicodedata.normalize("NFC", t or "")).strip()


def _reading(t):
    # NFKC folds ligatures/long-s/width variants; case-fold; unify quotes+hyphens; de-hyphenate.
    # Invisible formatting codepoints are dropped BEFORE de-hyphenation so a soft hyphen cannot
    # survive as a word-splitting artifact (NFKC does not remove them).
    t = strip_markup(regroup_tables(t or ""))
    t = unicodedata.normalize("NFKC", t).translate(_INVISIBLE).translate(_QUOTES).translate(_HYPHENS)
    t = _DEHYPH.sub("", t).casefold()
    return _WS.sub(" ", t).strip()


LANES = {"diplomatic": _diplomatic, "reading": _reading}


def norm(t: str | None, lane: str) -> str:
    """Normalize `t` through a named lane. Symmetric by construction: the SAME function is the only
    way GT or OCR text is normalized, so both sides always get identical treatment."""
    return LANES[lane](t)


def tokens(t: str | None, lane: str = "reading") -> list[str]:
    """Whitespace word tokens for the coverage axes (recall / over-extraction). Punctuation dropped;
    defaults to the reading lane so recall is case-/format-immune ('The' == 'the'). U+FFFD survives
    tokenization so a word containing an illegible glyph stays ONE token (which the recall target
    then drops as not-required) instead of fragmenting into bogus required tokens."""
    return _PUNCT.sub(" ", norm(t, lane)).split()


# --- edit-op accounting with the U+FFFD wildcard convention ---
@dataclass(frozen=True)
class EditCounts:
    """Per-page substitutions/deletions/insertions/hits over one lane. `n` is the counted reference
    length (hits+subs+deletes; U+FFFD positions are excluded, not counted as errors OR length).
    report.py micro-averages by POOLING these across pages, then dividing once."""
    s: int
    d: int
    i: int
    h: int

    @property
    def n(self) -> int:
        return self.s + self.d + self.h

    def rate(self) -> float | None:
        return (self.s + self.d + self.i) / self.n if self.n else None


def _counts_from_alignment(ref_seq, chunks):
    s = d = i = h = 0
    for c in chunks:
        if c.type == "equal":
            for k in range(c.ref_start_idx, c.ref_end_idx):
                if REPLACEMENT in ref_seq[k]:  # illegible GT matches, but never counts toward length
                    continue
                h += 1
        elif c.type == "substitute":  # equal-length ref/hyp spans in jiwer
            for k in range(c.ref_end_idx - c.ref_start_idx):
                unit = ref_seq[c.ref_start_idx + k]
                if REPLACEMENT in unit:  # illegible GT → wildcard matches the aligned OCR unit; skip both
                    continue
                s += 1
        elif c.type == "delete":
            for k in range(c.ref_start_idx, c.ref_end_idx):
                if REPLACEMENT in ref_seq[k]:  # illegible + unread → skip (never a deletion error)
                    continue
                d += 1
        elif c.type == "insert":
            i += c.hyp_end_idx - c.hyp_start_idx
    return EditCounts(s, d, i, h)


# Explicit, no-normalization char tokenizer so jiwer never re-lowercases the diplomatic lane behind
# our back.
_CHAR_T = jiwer.Compose([jiwer.ReduceToListOfListOfChars()])


def edit_counts(ref: str | None, hyp: str | None) -> EditCounts:
    """Character edit-op counts for normalized ``ref``/``hyp``, U+FFFD-discounted.

    Empty references still have valid integer supports: every hypothesis codepoint is an insertion.
    Their per-page rate is undefined because the reference denominator is zero, but retaining the
    insertions lets a report pool them with nonempty pages without silently discarding model output.
    """
    ref, hyp = ref or "", hyp or ""
    if not ref:
        return EditCounts(0, 0, len(hyp), 0)
    out = jiwer.process_characters(ref, hyp, reference_transform=_CHAR_T,
                                   hypothesis_transform=_CHAR_T)
    return _counts_from_alignment(out.references[0], out.alignments[0])


# --- script / language fidelity (GT-derived, label-free) ---
# Orthogonal to CER (how-wrong): catches script SWITCHING — e.g. Cyrillic transliterated to Latin,
# the failure behind the GLM-vs-DeepSeek language-fidelity finding. Expected script comes from the GT
# TEXT ITSELF, never a volume→language guess (that guess was wrong: the "russ" journal is German/Latin).
def _script(ch):
    if not ch.isalpha():
        return None
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "other"
    for s in ("CYRILLIC", "LATIN", "GREEK", "ARABIC", "HAN", "HEBREW"):
        if s in name:
            return s.lower()
    return "other"


def _script_counts(text):
    from collections import Counter
    return Counter(s for s in (_script(c) for c in (text or "")) if s and s != "other")


def dominant_script(text: str | None) -> str | None:
    # Ties break on script name, not Counter insertion order: a page with equal Latin and Cyrillic
    # evidence must resolve the same way no matter which script the GT happened to mention first.
    c = _script_counts(text)
    return min(c.items(), key=lambda item: (-item[1], item[0]))[0] if c else None


def script_match(ocr_text: str | None, gt_text: str | None) -> float | None:
    """Fraction of OCR alphabetic chars in the GT's dominant script (1.0 = no script switching).
    GT-derived → label-free. None if GT has no script or OCR has no alphabetic chars."""
    from collections import Counter
    want = dominant_script(gt_text)
    if not want:
        return None
    counts = Counter(s for s in (_script(c) for c in (ocr_text or "")) if s)
    total = sum(counts.values())
    return round(counts.get(want, 0) / total, 4) if total else None


def provenance() -> dict:
    """The credibility stamp attached to every scorecard (the survey's #1 recommendation)."""
    import importlib.metadata
    return {
        "norm_version": NORM_VERSION,
        "jiwer_version": importlib.metadata.version("jiwer"),  # the alignment engine is part of the number
        # jiwer delegates character alignment to rapidfuzz. The CER *rate* is invariant to which
        # minimal alignment is returned, but the emitted s/d/i/h split is not — so the engine that
        # produced a scorecard's raw supports is recorded, not just the wrapper's version.
        "rapidfuzz_version": importlib.metadata.version("rapidfuzz"),
        "lanes": {"diplomatic": "NFC, case-sensitive, keep long-s/ligatures/diacritics",
                  "reading": "NFKC, case-fold, unify quotes+hyphens, de-hyphenate, strip fixed markup, "
                              "read repeated-header tables group by group"},
        "grapheme_mode": GRAPHEME_MODE,
        "word_boundary": "whitespace split, punctuation dropped for token axes",
        "illegible_marker": "U+FFFD = wildcard (matches any OCR char, excluded from error + length)",
    }


if __name__ == "__main__":
    # self-test: the two lanes must diverge exactly on long-s / ligature / case, and U+FFFD must be free.
    long_s = "the ſong of the fiſh"          # diplomatic keeps ſ; reading folds ſ->s
    print("diplomatic:", repr(norm(long_s, "diplomatic")))
    print("reading:   ", repr(norm(long_s, "reading")))
    assert "ſ" in norm(long_s, "diplomatic") and "ſ" not in norm(long_s, "reading")

    gt = "abcdef"
    # OCR mangles positions 3-4; if those GT chars were illegible (U+FFFD) it should cost nothing.
    clean = edit_counts(norm(gt, "diplomatic"), norm("abXYef", "diplomatic"))
    wild = edit_counts(norm("ab��ef", "diplomatic"), norm("abXYef", "diplomatic"))
    print(f"\nclean  S/D/I/H/N = {clean.s}/{clean.d}/{clean.i}/{clean.h}/{clean.n} rate={clean.rate():.3f}")
    print(f"wild   S/D/I/H/N = {wild.s}/{wild.d}/{wild.i}/{wild.h}/{wild.n} rate={wild.rate():.3f}")
    assert clean.s == 2 and wild.s == 0, "U+FFFD must absorb the substitutions"
    assert wild.n == 4, "U+FFFD positions excluded from reference length"

    # rate identity (the micro-average contract)
    ec = edit_counts(norm("the cat sat on the mat", "reading"), norm("the dog sat on a mat", "reading"))
    rate = ec.rate()
    assert rate is not None and abs(rate - (ec.s + ec.d + ec.i) / ec.n) < 1e-9
    print("\nrate identity OK; provenance:", provenance()["illegible_marker"])
    print("all normalizer self-tests passed.")

