# /// script
# requires-python = ">=3.10"
# dependencies = ["jiwer>=4,<5"]
# ///
"""Generate AXES.md from scorer.REGISTRY — the registry is the single source of truth, so the docs
can't drift from the code. Run after adding/editing an axis:  uv run scripts/gen_axis_catalogue.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scoring"))
import normalizers as N  # noqa: E402
import scorer as S  # noqa: E402

ARROW = {True: "↑ higher better", False: "↓ lower better", None: "↔ policy diagnostic (directionless)"}


def render():
    prov = N.provenance()
    out = [
        "# Axis catalogue",
        "",
        "> Generated from `scorer.REGISTRY` by `scripts/gen_axis_catalogue.py` — **do not hand-edit.**",
        "",
        f"Scorer `{S.SCORER_VERSION}` · norm `{prov['norm_version']}` · grapheme mode "
        f"`{prov['grapheme_mode']}` · illegible marker = U+FFFD wildcard.",
        "",
        "The headline scores the docling **BODY** layer (furniture is an ignore-set); "
        "`over_extraction` scores against the **full** text so faithful furniture reading is not "
        "punished; furniture is reported as a directionless policy diagnostic. Token diagnostics are "
        "global evidence proxies, not spatial attribution. See `DESIGN.md`.",
        "",
    ]
    by_kind = {}
    for ax in S.REGISTRY:
        by_kind.setdefault(ax.kind, []).append(ax)
    for kind in ("coverage", "faithfulness", "over_extraction", "fidelity", "furniture", "layout"):
        axes = by_kind.get(kind, [])
        if not axes:
            continue
        out.append(f"## {kind}")
        out.append("")
        for ax in axes:
            out.append(f"### `{ax.name}`  ({ARROW[ax.higher_is_better]}, lane: {ax.lane})")
            out.append(f"- **What:** {ax.what}")
            out.append(f"- **Why:** {ax.why}")
            out.append(f"- **Caveats:** {ax.caveats}")
            out.append(f"- **Field precedent:** {ax.field_ref}")
            out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    path = ROOT / "AXES.md"
    path.write_text(render())
    print(f"wrote {path} ({len(S.REGISTRY)} axes)")
