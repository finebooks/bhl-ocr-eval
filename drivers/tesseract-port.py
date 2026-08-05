# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets", "huggingface-hub", "pandas", "pyarrow", "pillow", "pytesseract"]
# ///
"""tesseract-5 — the classical baseline, as a self-contained inference driver.

Run:
    hf jobs uv run --detach --flavor cpu-upgrade -s HF_TOKEN --timeout 4h \\
        drivers/tesseract-port.py -- \\
        --input-dataset <benchmark> --revision <sha> --id-column PageID \\
        --output hf://buckets/<owner>/<bucket>/<prefix>/

Inference only. It writes `{PageID, markdown, lang, error}` and stops; scoring is
`runners/score_dataset.py --ocr <output> --ocr-col markdown --key-col PageID`, the same
path every other model's cached output goes through. The previous runner scored inline,
which coupled it to `scoring/` and meant it could not run on Jobs at all — only that one
file is uploaded, so `import gt_score` died at startup. Splitting it makes the classical
baseline structurally identical to the other sixteen rows: inference produces raw text,
scoring happens separately over cached output, and a scorer change costs a re-score
rather than a re-run.

Why it belongs on Jobs rather than a laptop: every other row records a pinned container
image and a job id. Tesseract was the one whose environment was "whatever brew installed
on one machine" — an unpinned binary and unpinned language data on a board whose claim is
pinned provenance. Here the image pins both.

THE LANGUAGE MAP IS THE RESULT. Tesseract scored CER 0.19 on this corpus with the wrong
language pack and 0.067 with the right per-volume ones, which is the difference between
last place and third. So a missing pack is not a warning to be skimmed — `--strict-langs`
(default on) aborts rather than silently falling back to English and producing a number
that looks plausible and is wrong.
"""

import argparse
import io
import shutil
import subprocess
import sys

# Per-book language packs. Provenance: derived from the ground truth itself, not from
# catalogue metadata — an earlier pass trusted the catalogue and ran Cyrillic on a
# German/French volume. `trudy` is genuinely mixed, hence the '+' pair.
TESS = {
    "birdsofgreatbrit02butl": "eng",
    "conchologiaiconi05reev": "eng",
    "daschitinskelett00prel": "deu",
    "histoirenaturell10cuvi": "fra",
    "pisciumquerelaee00sche": "lat",
    "trudyrusskagoent161881russ": "deu+fra",
}


def ensure_tesseract(packs: set[str]) -> None:
    """Install the tesseract binary and language data if missing (a no-op locally).

    The base install is fatal on failure — there is nothing to OCR with. Pack installs are
    best-effort here, but the caller decides what a missing pack means; see --strict-langs.
    """
    if shutil.which("tesseract") is None:
        print("tesseract not found — installing via apt (Jobs containers run as root)", flush=True)
        try:
            subprocess.run(["apt-get", "update", "-qq"], check=True)
            subprocess.run(["apt-get", "install", "-y", "-qq", "tesseract-ocr"], check=True)
        except Exception as e:
            sys.exit(f"could not apt-install tesseract-ocr: {e} — use an --image with it preinstalled")
        if shutil.which("tesseract") is None:
            sys.exit("tesseract still not on PATH after install")

    import pytesseract

    try:
        have = set(pytesseract.get_languages(config=""))
    except Exception:
        have = set()
    missing = sorted(packs - have - {"osd"})
    if missing:
        print(f"installing language packs: {missing}", flush=True)
        try:
            subprocess.run(["apt-get", "update", "-qq"], check=True)
            subprocess.run(["apt-get", "install", "-y", "-qq",
                            *[f"tesseract-ocr-{c}" for c in missing]], check=True)
        except Exception as e:
            print(f"WARNING: pack install failed: {e}", flush=True)


def to_pil(value):
    """One dataset image cell -> a PIL image, whichever shape `datasets` hands over."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict) and value.get("bytes"):
        return Image.open(io.BytesIO(value["bytes"]))
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(value)))
    raise ValueError(f"unsupported image value: {type(value)}")


def main():
    ap = argparse.ArgumentParser(description="Tesseract 5 baseline — inference only")
    ap.add_argument("--input-dataset", required=True)
    ap.add_argument("--output", required=True,
                    help="hf://buckets/<owner>/<bucket>/<prefix>/ or a local .parquet path")
    ap.add_argument("--revision", default=None, help="pin the benchmark revision")
    ap.add_argument("--split", default="train")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--id-column", default="PageID")
    ap.add_argument("--volume-column", default="volume")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test the path against N pages. NOT a publishable run — the page "
                         "set will not match the benchmark fingerprint.")
    ap.add_argument("--loose-langs", dest="strict", action="store_false", default=True,
                    help="fall back to 'eng' when a language pack is missing instead of aborting. "
                         "Off by default: the fallback silently costs ~3x the character error rate.")
    args = ap.parse_args()

    import pandas as pd
    import pytesseract
    from datasets import load_dataset

    ds = load_dataset(args.input_dataset, revision=args.revision, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    ensure_tesseract({p for v in TESS.values() for p in v.split("+")})
    have = set(pytesseract.get_languages(config=""))
    version = str(pytesseract.get_tesseract_version())

    lang_by_volume = {}
    for volume in sorted(set(ds[args.volume_column])):
        want = TESS.get(volume, "eng")
        if all(p in have for p in want.split("+")):
            lang_by_volume[volume] = want
        elif args.strict:
            sys.exit(f"language pack {want!r} missing for volume {volume!r} (have: {sorted(have)}). "
                     f"Falling back to 'eng' would roughly triple this model's error rate, so this "
                     f"aborts rather than producing a plausible wrong number. Pass --loose-langs to "
                     f"override deliberately.")
        else:
            print(f"WARNING: pack {want!r} missing for {volume!r} — falling back to 'eng'", flush=True)
            lang_by_volume[volume] = "eng"

    print(f"tesseract {version} | languages: {lang_by_volume}", flush=True)

    rows = []
    for i, r in enumerate(ds):
        lang = lang_by_volume[r[args.volume_column]]
        try:
            text = pytesseract.image_to_string(to_pil(r[args.image_column]).convert("RGB"), lang=lang)
            err = None
        except Exception as e:  # a durable error row, never a sentinel string in the text column
            text, err = "", f"{type(e).__name__}: {e}"
        rows.append({args.id_column: r[args.id_column], "markdown": text, "lang": lang,
                     "model": "tesseract-5", "tesseract_version": version, "error": err})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(ds)}", flush=True)

    df = pd.DataFrame(rows)
    out = args.output.rstrip("/")
    out = out if out.endswith(".parquet") else f"{out}/tesseract-5.parquet"
    df.to_parquet(out, index=False)
    n_err = int(df["error"].notna().sum())
    print(f"{len(df)} pages -> {out} ({n_err} error rows)", flush=True)


if __name__ == "__main__":
    main()
