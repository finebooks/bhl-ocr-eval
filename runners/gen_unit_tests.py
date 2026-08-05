# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets", "huggingface-hub", "pandas", "pyarrow", "pillow",
#   "rapidfuzz", "fuzzysearch", "jiwer>=4,<5",
# ]
# ///
"""Generate olmOCR-bench-style pass/fail unit tests as DATA from the BHL GT (issue: unit-test axis).

This is the generator + a versioned test-set artifact ONLY — it does NOT wire tests into the scorer.
Each test is a deterministic, page-anchored assertion a future axis can run against any model's OCR:

  - **presence** — a distinctive GT snippet that is UNIQUE on its page (after reading-lane norm) must
    appear in the OCR output. The CER-robust "did it read this exact thing" check.
  - **order** — snippet A must appear before snippet B in the OCR. Emitted ONLY where the GT order is
    unambiguous, never where it rests on the GT's heuristic global reading order:
      · `intra_region`   — A and B are two snippets from ONE region's own linear text (a paragraph
                            reads top-to-bottom; that order is real, not assembled).
      · `vertical_blocks`— A and B are in two different body regions on a SINGLE-COLUMN page (no two
                            body regions overlap vertically), A's block entirely above B's. Top-to-
                            bottom is then geometric fact, not the heuristic list order.
    Pages with side-by-side columns (any vertical overlap between body regions) emit no cross-region
    order tests — exactly the ambiguity the DESIGN.md "GT reading order is heuristic" caveat warns of.

Mechanics mirror olmOCR-bench `tests.py`: a snippet carries `max_diffs` (allowed Levenshtein edits,
= round(len * edit_frac)); the runner finds it with `fuzzysearch.find_near_matches(..., max_l_dist=
max_diffs)`, equivalently a `rapidfuzz.partial_ratio` clearing `threshold = 1 - max_diffs/len`. The
generator uses BOTH to select and validate: a snippet is emitted only if it has exactly one distinct
near-match on its page (unique) and clears the partial_ratio threshold (findable).

Uniqueness is judged after the SAME reading lane the scorer uses (`normalizers.norm(..., "reading")`),
so the artifact is stamped with `norm_version` — bump it or `edit_frac` and regenerate.

  uv run runners/gen_unit_tests.py                       # -> data/unit_tests.parquet + _report.md
  uv run runners/gen_unit_tests.py --edit-frac 0.05 --n 200
"""
import argparse
import pathlib
import random
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scoring"))
import gt_docling as G  # noqa: E402
import normalizers as N  # noqa: E402

TEST_SET_VERSION = "2026-07-06a"

# Snippet-selection knobs (defaults chosen for ~6-10 word distinctive phrases on this corpus).
WIN_WORDS = (7, 11)     # window sizes tried, in words
STRIDE = 4              # word stride between candidate windows
MIN_CHARS = 28         # a normalized snippet shorter than this isn't distinctive enough
MIN_WORDS = 5
MAX_CHARS = 140        # cap so a "snippet" stays a snippet, not a paragraph
MIN_ALPHA_FRAC = 0.5   # reject mostly-digit/punctuation windows (table gutters, plate numbers)
PRESENCE_PER_PAGE = 3
INTRA_ORDER_PER_PAGE = 2
INTRA_MIN_WORD_GAP = 3  # A and B must be this many words apart within a region
VOVERLAP_TOL = 0.30     # two regions "share a row" if they overlap vertically by >tol*min(height)


# ---------------------------------------------------------------------------
# pure helpers (unit-tested without network / heavy deps)
# ---------------------------------------------------------------------------
def read_norm(t):
    """The one reading lane the scorer uses — uniqueness must be judged in the same space."""
    return N.norm(t, "reading")


def max_diffs(norm_len, edit_frac):
    """Allowed Levenshtein edits for a snippet of this normalized length (olmOCR-bench convention)."""
    return max(1, round(norm_len * edit_frac))


def pr_threshold(norm_len, md):
    """rapidfuzz partial_ratio threshold equivalent to a `max_diffs` budget: 1 - md/len."""
    return round(1 - md / norm_len, 4) if norm_len else 0.0


def is_distinctive(norm_snip):
    """A snippet worth testing: long enough, enough words, and not mostly digits/punctuation."""
    if not (MIN_CHARS <= len(norm_snip) <= MAX_CHARS):
        return False
    if len(norm_snip.split()) < MIN_WORDS:
        return False
    alpha = sum(c.isalpha() for c in norm_snip)
    return alpha >= MIN_ALPHA_FRAC * len(norm_snip)


def bbox(item):
    """(l, t, r, b) from a docling text item's prov, or None. TOPLEFT origin: t < b (y grows down)."""
    prov = (item.get("prov") or [{}])[0]
    bb = prov.get("bbox")
    if not bb:
        return None
    return (bb.get("l"), bb.get("t"), bb.get("r"), bb.get("b"))


def v_overlap(a, b):
    """Vertical overlap (px) of two (l,t,r,b) boxes; 0 if they don't share any y-range."""
    return max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def is_single_column(boxes, tol=VOVERLAP_TOL):
    """True if no two boxes share a row — i.e. the page stacks top-to-bottom with no side-by-side
    columns, so cross-region top→bottom order is a geometric fact rather than the heuristic list."""
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            h = min(boxes[i][3] - boxes[i][1], boxes[j][3] - boxes[j][1])
            if h > 0 and v_overlap(boxes[i], boxes[j]) > tol * h:
                return False
    return True


def distinct_starts(matches, target_len):
    """Collapse fuzzysearch's overlapping near-matches into one start per real occurrence."""
    starts = sorted(m.start for m in matches)
    groups = []
    for s in starts:
        if not groups or s - groups[-1] > target_len * 0.5:
            groups.append(s)
    return groups


def unique_match(target_norm, page_norm, md):
    """(is_unique, start) — the snippet occurs exactly once on the page within the edit budget."""
    from fuzzysearch import find_near_matches
    matches = find_near_matches(target_norm, page_norm, max_l_dist=md)
    starts = distinct_starts(matches, len(target_norm))
    return (len(starts) == 1, starts[0] if starts else None)


def findable(target_norm, page_norm, threshold):
    """rapidfuzz partial_ratio confirms the snippet is present at the stated threshold (0-1 scale)."""
    from rapidfuzz import fuzz
    return fuzz.partial_ratio(target_norm, page_norm) >= threshold * 100


def candidates(region_text, region_idx, region_label, edit_frac):
    """Distinctive word-windows of one region: [{word_start, raw, norm, md, threshold, context}]."""
    words = region_text.split()
    out, seen = [], set()
    for start in range(0, len(words), STRIDE):
        for w in WIN_WORDS:
            if start + w > len(words):
                continue
            raw = " ".join(words[start:start + w])
            norm = read_norm(raw)
            if norm in seen or not is_distinctive(norm):
                continue
            seen.add(norm)
            md = max_diffs(len(norm), edit_frac)
            ctx = " ".join(words[max(0, start - 5):start + w + 5])
            out.append({"region_idx": region_idx, "region_label": region_label, "word_start": start,
                        "raw": raw, "norm": norm, "md": md, "threshold": pr_threshold(len(norm), md),
                        "context": ctx})
    return out


# ---------------------------------------------------------------------------
# building tests for one page
# ---------------------------------------------------------------------------
def body_regions(docling):
    """Body regions in docling order with geometry: [{idx, label, text, bbox}]."""
    out = []
    for i, t in enumerate(G.text_items(docling)):
        if t.get("content_layer") == G.BODY_LAYER and t.get("text"):
            out.append({"idx": i, "label": t.get("label"), "text": t["text"], "bbox": bbox(t)})
    return out


def _valid(cand, page_norm):
    """A candidate survives iff it is unique on the page and findable at its threshold."""
    ok, start = unique_match(cand["norm"], page_norm, cand["md"])
    if not (ok and findable(cand["norm"], page_norm, cand["threshold"])):
        return None
    return start


def _spread(items, k):
    """Up to k items, evenly spaced across the list (keeps tests spread over the page)."""
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def page_tests(page_id, volume, docling, page_norm, edit_frac):
    regions = body_regions(docling)
    valid_by_region = {}  # region_idx -> [(cand, start)] in word order
    for r in regions:
        cands = candidates(r["text"], r["idx"], r["label"], edit_frac)
        good = [(c, s) for c in cands if (s := _valid(c, page_norm)) is not None]
        if good:
            valid_by_region[r["idx"]] = good

    tests = []
    # presence: spread across all valid candidates on the page
    all_valid = [c for good in valid_by_region.values() for (c, _s) in good]
    for c in _spread(sorted(all_valid, key=lambda c: (c["region_idx"], c["word_start"])),
                     PRESENCE_PER_PAGE):
        tests.append(_presence_row(page_id, volume, c))

    # intra-region order: earliest A + latest B in one region, well separated, A precedes B on page
    intra = 0
    for r in regions:
        if intra >= INTRA_ORDER_PER_PAGE:
            break
        good = valid_by_region.get(r["idx"], [])
        if len(good) < 2:
            continue
        (ca, sa), (cb, sb) = good[0], good[-1]
        gap = cb["word_start"] - (ca["word_start"] + max(WIN_WORDS))
        if gap >= INTRA_MIN_WORD_GAP and sa < sb:
            tests.append(_order_row(page_id, volume, "intra_region", ca, sa, cb, sb))
            intra += 1

    # cross-region order: only on single-column pages, one representative snippet per region, top→bottom
    geo = [r for r in regions if r["bbox"] and r["idx"] in valid_by_region]
    if len(geo) >= 2 and is_single_column([r["bbox"] for r in geo]):
        geo.sort(key=lambda r: r["bbox"][1])  # by top-y
        top, bot = geo[0], geo[-1]
        ca, sa = valid_by_region[top["idx"]][0]
        cb, sb = valid_by_region[bot["idx"]][0]
        if top["bbox"][3] <= bot["bbox"][1] and sa < sb and ca["norm"] != cb["norm"]:
            tests.append(_order_row(page_id, volume, "vertical_blocks", ca, sa, cb, sb))  # one pair/page
    return tests


def _presence_row(page_id, volume, c):
    return {
        "page_id": page_id, "volume": volume, "test_type": "presence", "basis": "presence",
        "region_label": c["region_label"],
        "target": c["raw"], "target_norm": c["norm"],
        "target_max_diffs": c["md"], "target_threshold": c["threshold"],
        "_context": c["context"],
    }


def _order_row(page_id, volume, basis, ca, sa, cb, sb):
    return {
        "page_id": page_id, "volume": volume, "test_type": "order", "basis": basis,
        "before": ca["raw"], "before_norm": ca["norm"],
        "before_max_diffs": ca["md"], "before_threshold": ca["threshold"],
        "before_region_label": ca["region_label"], "before_pos": sa,
        "after": cb["raw"], "after_norm": cb["norm"],
        "after_max_diffs": cb["md"], "after_threshold": cb["threshold"],
        "after_region_label": cb["region_label"], "after_pos": sb,
        "_before_ctx": ca["context"], "_after_ctx": cb["context"],
    }


# ---------------------------------------------------------------------------
# report + main
# ---------------------------------------------------------------------------
def render_report(rows, edit_frac, sample_n, seed=0):
    pres = [r for r in rows if r["test_type"] == "presence"]
    order = [r for r in rows if r["test_type"] == "order"]
    rng = random.Random(seed)
    take_p = rng.sample(pres, min(sample_n * 2 // 3, len(pres)))
    take_o = rng.sample(order, min(sample_n - len(take_p), len(order)))
    out = [f"# Unit-test eyeball report ({TEST_SET_VERSION})", "",
           f"{len(rows)} tests over {len({r['page_id'] for r in rows})} pages "
           f"({len(pres)} presence, {len(order)} order) · edit_frac={edit_frac} · "
           f"norm `{N.NORM_VERSION}`. Sample below (`《》` marks the tested snippet in GT context).", ""]
    if take_p:
        out += ["## Presence (sampled)", ""]
        for r in take_p:
            ctx = r["_context"].replace(r["target"], f"《{r['target']}》", 1)
            out += [f"- **p{r['page_id']}** · {r['volume']} · region=`{r['region_label']}` · "
                    f"max_diffs={r['target_max_diffs']} thr={r['target_threshold']}",
                    f"  - target: `{r['target']}`", f"  - context: …{ctx}…", ""]
    if take_o:
        out += ["## Order (sampled)", ""]
        for r in take_o:
            bc = r["_before_ctx"].replace(r["before"], f"《{r['before']}》", 1)
            ac = r["_after_ctx"].replace(r["after"], f"《{r['after']}》", 1)
            out += [f"- **p{r['page_id']}** · {r['volume']} · basis=`{r['basis']}` "
                    f"({r['before_region_label']} → {r['after_region_label']})",
                    f"  - before: `{r['before']}`  …{bc}…",
                    f"  - after:  `{r['after']}`  …{ac}…", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="davanstrien/bhl-eval-impact-sample")
    ap.add_argument("--out", default=str(ROOT / "data" / "unit_tests.parquet"))
    ap.add_argument("--report", default=str(ROOT / "data" / "unit_tests_report.md"))
    ap.add_argument("--edit-frac", type=float, default=0.04)
    ap.add_argument("--n", type=int, default=None, help="limit to first N pages (debug)")
    ap.add_argument("--sample", type=int, default=30, help="tests to show in the eyeball report")
    args = ap.parse_args()
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="train").remove_columns(["image"])
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))

    rows = []
    for row in ds:
        page_norm = read_norm(row["text"] or "")
        if not page_norm:
            continue
        rows += page_tests(int(row["PageID"]), row["volume"], row["docling"], page_norm, args.edit_frac)
    for i, r in enumerate(rows):
        r["test_id"] = f"{r['page_id']}:{r['test_type']}:{i}"
        r["test_set_version"] = TEST_SET_VERSION
        r["norm_version"] = N.NORM_VERSION
        r["edit_frac"] = args.edit_frac

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    report = render_report(rows, args.edit_frac, args.sample)
    pathlib.Path(args.report).write_text(report)

    n_pages = df["page_id"].nunique() if len(df) else 0
    by_basis = dict(df["basis"].value_counts()) if len(df) else {}
    print(f"wrote {out} ({len(df)} tests over {n_pages} pages) · {by_basis}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
