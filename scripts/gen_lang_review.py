# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "huggingface-hub", "pandas", "pyarrow", "pillow"]
# ///
"""Build a self-contained HTML review sheet for the low-confidence GlotLID language labels (issue #2).

The language labels are best-effort (GlotLID on historic GT text drifts to oddball Latin-script
languages); pages below the confidence bar take a volume-majority fallback that nobody has eyeballed.
This joins each low-conf page from the `data/languages.parquet` sidecar back to its GT text + page
image in the sample dataset and renders one card per page: image, GT excerpt, the raw/final/volume
labels + confidence, and a keyboard-driven accept/correct control that exports a corrections JSON.

The output is a single file that opens from `file://` with no server — hand-check is meant to be ~15
min: read the card, press a key (a=accept, e/d/f/l/r=lang, m=mixed, u=und), auto-advance, then Export.

  uv run scripts/gen_lang_review.py                 # -> data/lang_review.html (conf < 0.5)
  uv run scripts/gen_lang_review.py --max-conf 0.6  # widen the review set
"""
import argparse
import base64
import html
import io
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ISO639-3 -> readable name, for the oddball raw GlotLID guesses seen on historic text. The point of
# showing the readable name is so a "war_Latn" guess on a plainly-English page reads as the confusion
# it is (Waray, a Philippine language) rather than an opaque code.
ISO_NAME = {
    "eng": "English", "deu": "German", "fra": "French", "lat": "Latin", "rus": "Russian",
    "ita": "Italian", "spa": "Spanish", "nld": "Dutch", "por": "Portuguese", "ron": "Romanian",
    "zsm": "Malay (Standard)", "msa": "Malay", "ltz": "Luxembourgish", "sme": "Northern Sami",
    "gsw": "Swiss German", "war": "Waray", "dag": "Dagbani", "frp": "Franco-Provencal",
    "bar": "Bavarian", "sco": "Scots", "wes": "Cameroon Pidgin", "mlt": "Maltese", "tur": "Turkish",
    "cat": "Catalan", "oci": "Occitan", "afr": "Afrikaans", "nno": "Norwegian Nynorsk",
    "dan": "Danish", "swe": "Swedish", "fin": "Finnish", "pol": "Polish", "ces": "Czech",
    "hun": "Hungarian", "vol": "Volapuk", "epo": "Esperanto", "ina": "Interlingua",
}

# Quick-pick language buttons (code, label, keyboard key). Covers the sample's real languages plus the
# conventions the issue asks about (mixed / und).
PICKS = [
    ("en", "English", "e"), ("de", "German", "d"), ("fr", "French", "f"),
    ("la", "Latin", "l"), ("ru", "Russian", "r"),
    ("mixed", "mixed", "m"), ("und", "und", "u"),
]


def iso_name(glotlid_label):
    """'deu_Latn' -> 'German (deu_Latn)'; unknown code -> the raw label unchanged."""
    iso = glotlid_label.split("_")[0]
    name = ISO_NAME.get(iso)
    return f"{name} ({glotlid_label})" if name else glotlid_label


def img_data_uri(image, max_w=1100, quality=80):
    """PIL image -> a downscaled JPEG data URI, small enough to embed 40 of them in one file."""
    im = image.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def volume_order(volumes):
    """trudy first (the volume the whole correction was about), then the rest alphabetically."""
    rest = sorted(v for v in volumes if not v.startswith("trudy"))
    trudy = sorted(v for v in volumes if v.startswith("trudy"))
    return trudy + rest


def build_records(sidecar, dataset, max_conf):
    """Join low-conf sidecar rows to the sample dataset's text + image. Returns list of dicts."""
    from datasets import load_dataset

    side = pd.read_parquet(sidecar)
    low = side[side["language_conf"] < max_conf].copy()
    want = set(low["PageID"].astype(int))
    print(f"{len(low)} pages below conf {max_conf} (of {len(side)})")

    ds = load_dataset(dataset, split="train")
    keep = [i for i, pid in enumerate(ds["PageID"]) if int(pid) in want]
    ds = ds.select(keep)

    by_pid = {int(r.PageID): r for r in low.itertuples()}
    recs = []
    for row in ds:
        pid = int(row["PageID"])
        s = by_pid[pid]
        recs.append({
            "page_id": pid,
            "volume": row["volume"],
            "text": row["text"] or "",
            "final": s.language,
            "glotlid": s.language_glotlid,
            "volume_prior": s.language_volume,
            "conf": float(s.language_conf),
            "img": img_data_uri(row["image"]),
        })
    # trudy first, then by volume, then least-confident first within a volume.
    order = {v: i for i, v in enumerate(volume_order({r["volume"] for r in recs}))}
    recs.sort(key=lambda r: (order[r["volume"]], r["conf"]))
    return recs


def render_card(r):
    e = html.escape
    picks = "".join(
        f'<button class="pick" data-code="{c}" title="key: {k}">{lbl}</button>'
        for c, lbl, k in PICKS
    )
    return f"""
<article class="card" data-pageid="{r['page_id']}" data-volume="{e(r['volume'])}"
         data-final="{e(r['final'])}" data-glotlid="{e(r['glotlid'])}"
         data-prior="{e(r['volume_prior'])}" data-conf="{r['conf']:.4f}">
  <div class="img"><img loading="lazy" src="{r['img']}" alt="page {r['page_id']}"></div>
  <div class="panel">
    <table class="meta">
      <tr><th>page</th><td class="mono">{r['page_id']}</td>
          <th>volume</th><td class="mono">{e(r['volume'])}</td></tr>
      <tr><th>final label</th><td><span class="lab final">{e(r['final'])}</span></td>
          <th>volume prior</th><td class="mono">{e(r['volume_prior'])}</td></tr>
      <tr><th>GlotLID raw</th><td>{e(iso_name(r['glotlid']))}</td>
          <th>confidence</th><td class="mono">{r['conf']:.3f}</td></tr>
    </table>
    <div class="gt" title="GT text GlotLID scored">{e(r['text'])}</div>
    <div class="controls">
      <button class="accept" title="key: a">&#10003; accept <b>{e(r['final'])}</b></button>
      <span class="sep">correct &rarr;</span>
      {picks}
      <button class="other" title="free-text label">other&hellip;</button>
      <input class="note" placeholder="note (e.g. de+fr quotations)">
      <span class="status pending">pending</span>
    </div>
  </div>
</article>"""


def render(recs, dataset, max_conf):
    by_vol = {}
    for r in recs:
        by_vol.setdefault(r["volume"], []).append(r)
    sections = []
    for vol in volume_order(by_vol):
        rows = by_vol[vol]
        cards = "".join(render_card(r) for r in rows)
        sections.append(f'<h2 class="volhdr">{html.escape(vol)} '
                         f'<span class="voln">{len(rows)} page(s)</span></h2>{cards}')
    body = "\n".join(sections)
    return _PAGE.replace("__N__", str(len(recs))).replace("__DATASET__", html.escape(dataset)) \
                .replace("__MAXCONF__", f"{max_conf}").replace("__BODY__", body)


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GlotLID low-confidence language review</title>
<style>
  :root { --ink:#1a1a1a; --mut:#6b6b6b; --line:#d9d5cc; --bg:#f7f5f0; --card:#fff;
          --ok:#3f7d4e; --okbg:#eaf3ec; --amber:#9a6a00; --amberbg:#fbf1dd; --active:#2b5c8a; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
         color:var(--ink); background:var(--bg); }
  .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; }
  header { position:sticky; top:0; z-index:10; background:var(--card); border-bottom:1px solid var(--line);
           padding:10px 18px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header .sub { color:var(--mut); font-size:13px; }
  header .prog { margin-left:auto; font-variant-numeric:tabular-nums; }
  header button { font-size:13px; padding:5px 11px; border:1px solid var(--line); background:#fff;
                  border-radius:5px; cursor:pointer; }
  header button:hover { border-color:var(--active); }
  header button.primary { background:var(--active); color:#fff; border-color:var(--active); }
  .help { padding:8px 18px; color:var(--mut); font-size:13px; background:var(--card);
          border-bottom:1px solid var(--line); }
  .help kbd { font-family:ui-monospace,monospace; background:#eee; border:1px solid var(--line);
              border-radius:3px; padding:0 5px; font-size:12px; }
  main { padding:16px 18px 120px; max-width:1180px; margin:0 auto; }
  .volhdr { font-size:14px; text-transform:uppercase; letter-spacing:.05em; color:var(--mut);
            border-bottom:1px solid var(--line); padding-bottom:5px; margin:26px 0 12px; }
  .voln { text-transform:none; letter-spacing:0; font-weight:400; }
  .card { display:grid; grid-template-columns:340px 1fr; gap:16px; background:var(--card);
          border:1px solid var(--line); border-left:4px solid var(--line); border-radius:7px;
          padding:12px; margin:12px 0; scroll-margin-top:120px; }
  .card.active { border-left-color:var(--active); box-shadow:0 0 0 1px var(--active); }
  .card.done-accept { border-left-color:var(--ok); background:var(--okbg); }
  .card.done-correct { border-left-color:var(--amber); background:var(--amberbg); }
  .card .img img { width:100%; border:1px solid var(--line); border-radius:4px; cursor:zoom-in;
                   background:#fff; }
  .card .img img.big { cursor:zoom-out; }
  table.meta { border-collapse:collapse; font-size:13px; margin-bottom:8px; width:100%; }
  table.meta th { text-align:left; color:var(--mut); font-weight:500; padding:2px 8px 2px 0;
                  white-space:nowrap; vertical-align:top; }
  table.meta td { padding:2px 16px 2px 0; }
  .lab { font-family:ui-monospace,monospace; padding:1px 7px; border-radius:4px; background:#eef; }
  .lab.final { background:#e7edf5; font-weight:600; }
  .gt { white-space:pre-wrap; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px;
        line-height:1.45; background:#fbfaf7; border:1px solid var(--line); border-radius:5px;
        padding:8px 10px; max-height:200px; overflow:auto; margin-bottom:10px; }
  .controls { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
  .controls button { font-size:13px; padding:4px 9px; border:1px solid var(--line); background:#fff;
                     border-radius:5px; cursor:pointer; }
  .controls button:hover { border-color:var(--active); }
  .controls .accept { border-color:var(--ok); color:var(--ok); }
  .controls .pick.sel, .controls .other.sel { background:var(--active); color:#fff; border-color:var(--active); }
  .controls .accept.sel { background:var(--ok); color:#fff; border-color:var(--ok); }
  .controls .sep { color:var(--mut); font-size:12px; }
  .controls .note { flex:1; min-width:150px; font-size:13px; padding:4px 8px; border:1px solid var(--line);
                    border-radius:5px; }
  .status { font-size:12px; padding:2px 8px; border-radius:10px; background:#eee; color:var(--mut); }
  .status.accepted { background:var(--okbg); color:var(--ok); }
  .status.corrected { background:var(--amberbg); color:var(--amber); }
</style></head>
<body>
<header>
  <h1>GlotLID low-confidence review</h1>
  <span class="sub">__N__ pages &middot; conf &lt; __MAXCONF__ &middot; __DATASET__</span>
  <span class="prog" id="prog">0 / __N__ reviewed</span>
  <button class="primary" id="export">Export corrections JSON</button>
  <button id="copy">Copy JSON</button>
  <button id="reset">Reset</button>
</header>
<div class="help">
  Keys on the highlighted card: <kbd>a</kbd> accept &middot; <kbd>e</kbd>/<kbd>d</kbd>/<kbd>f</kbd>/<kbd>l</kbd>/<kbd>r</kbd> en/de/fr/la/ru
  &middot; <kbd>m</kbd> mixed &middot; <kbd>u</kbd> und &middot; <kbd>j</kbd>/<kbd>k</kbd> next/prev.
  Click a page image to zoom. Progress autosaves; corrections export as one JSON.
</div>
<main>__BODY__</main>
<script>
const KEY = "bhl_lang_review_v1";
const cards = [...document.querySelectorAll(".card")];
let active = 0;
const state = JSON.parse(localStorage.getItem(KEY) || "{}");

function save(){ localStorage.setItem(KEY, JSON.stringify(state)); refresh(); }

function setDecision(card, decision, code){
  const pid = card.dataset.pageid;
  state[pid] = {decision, code: code || null, note: card.querySelector(".note").value || ""};
  save();
}

function paint(card){
  const pid = card.dataset.pageid;
  const s = state[pid];
  card.classList.remove("done-accept","done-correct");
  card.querySelectorAll(".pick,.other,.accept").forEach(b=>b.classList.remove("sel"));
  const status = card.querySelector(".status");
  status.className = "status pending"; status.textContent = "pending";
  if(!s) return;
  if(s.decision === "accept"){
    card.classList.add("done-accept");
    card.querySelector(".accept").classList.add("sel");
    status.className = "status accepted"; status.textContent = "accepted: " + card.dataset.final;
  } else {
    card.classList.add("done-correct");
    const btn = [...card.querySelectorAll(".pick")].find(b=>b.dataset.code===s.code);
    if(btn) btn.classList.add("sel"); else card.querySelector(".other").classList.add("sel");
    status.className = "status corrected"; status.textContent = "corrected: " + (s.code||"?");
  }
  if(s.note) card.querySelector(".note").value = s.note;
}

function refresh(){
  cards.forEach(paint);
  const done = Object.keys(state).length;
  document.getElementById("prog").textContent = done + " / " + cards.length + " reviewed";
}

function setActive(i){
  if(i<0||i>=cards.length) return;
  cards[active].classList.remove("active");
  active = i; cards[active].classList.add("active");
  cards[active].scrollIntoView({behavior:"smooth", block:"start"});
}

cards.forEach((card, i) => {
  card.addEventListener("click", () => { cards[active].classList.remove("active"); active=i; card.classList.add("active"); });
  card.querySelector(".accept").addEventListener("click", () => { setDecision(card,"accept"); });
  card.querySelectorAll(".pick").forEach(b =>
    b.addEventListener("click", () => setDecision(card,"correct",b.dataset.code)));
  card.querySelector(".other").addEventListener("click", () => {
    const v = prompt("Corrected language code / label:", card.dataset.final);
    if(v) setDecision(card,"correct",v.trim());
  });
  card.querySelector(".note").addEventListener("change", () => {
    if(state[card.dataset.pageid]) setDecision(card, state[card.dataset.pageid].decision, state[card.dataset.pageid].code);
  });
  card.querySelector("img").addEventListener("click", e => { e.stopPropagation(); e.target.classList.toggle("big");
    e.target.style.width = e.target.classList.contains("big") ? "auto" : "100%"; });
});

document.addEventListener("keydown", e => {
  if(e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey) return;
  const card = cards[active];
  const k = e.key.toLowerCase();
  const map = {e:"en", d:"de", f:"fr", l:"la", r:"ru", m:"mixed", u:"und"};
  if(k === "j"){ setActive(active+1); e.preventDefault(); }
  else if(k === "k"){ setActive(active-1); e.preventDefault(); }
  else if(k === "a"){ setDecision(card,"accept"); setActive(active+1); e.preventDefault(); }
  else if(map[k]){ setDecision(card,"correct",map[k]); setActive(active+1); e.preventDefault(); }
});

function corrections(){
  return cards.map(card => {
    const pid = card.dataset.pageid, s = state[pid];
    const accepted = s && s.decision === "accept";
    return {
      page_id: +pid, volume: card.dataset.volume,
      original_label: card.dataset.final, glotlid_raw: card.dataset.glotlid,
      volume_prior: card.dataset.prior, confidence: +card.dataset.conf,
      reviewed: !!s,
      corrected_label: !s ? null : (accepted ? card.dataset.final : s.code),
      changed: !!s && !accepted && s.code !== card.dataset.final,
      note: (s && s.note) || ""
    };
  });
}

document.getElementById("export").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(corrections(), null, 2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "lang_corrections.json"; a.click();
});
document.getElementById("copy").addEventListener("click", async () => {
  await navigator.clipboard.writeText(JSON.stringify(corrections(), null, 2));
  const b = document.getElementById("copy"); b.textContent = "Copied!"; setTimeout(()=>b.textContent="Copy JSON", 1200);
});
document.getElementById("reset").addEventListener("click", () => {
  if(confirm("Clear all review decisions?")){ for(const k in state) delete state[k]; save(); }
});

if(cards.length) cards[0].classList.add("active");
refresh();
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="davanstrien/bhl-eval-impact-sample")
    ap.add_argument("--sidecar", default=str(ROOT / "data" / "languages.parquet"))
    ap.add_argument("--out", default=str(ROOT / "data" / "lang_review.html"))
    ap.add_argument("--max-conf", type=float, default=0.5)
    args = ap.parse_args()

    recs = build_records(args.sidecar, args.dataset, args.max_conf)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(recs, args.dataset, args.max_conf))
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({len(recs)} cards, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
