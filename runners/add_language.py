# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "huggingface-hub", "pandas", "pyarrow", "fasttext-predict"]
# ///
"""Per-page language labels from GlotLID over the GT text — the required post-step after prep_sample.

The old volume→language map was wrong (the "russ" journal is mostly German/French, ~no Cyrillic), so
`language` is identified from each page's OWN GT text, never guessed from the volume. Pages too short
or too uncertain to identify inherit the majority label of their volume's confidently-identified pages.

CAVEAT — these labels are a best-effort ATTEMPT, not ground truth. GlotLID is trained on modern text;
historic orthography, Latin species binomials, and heavy abbreviation pull its raw guesses toward
unrelated Latin-script languages (observed on this sample: Waray/Maltese/Luxembourgish/Swiss-German/
Malay on pages that are plainly English or German — ~10% of pages fall below the confidence bar and
take the volume-majority fallback). That is why the raw label + confidence + the old volume prior are
all kept as audit columns: treat `language` as a stratification convenience to be spot-checked, not a
per-page fact. Low-confidence pages should be hand-checked before the labels are proposed upstream.

Writes a local sidecar (always), optionally replaces the sample dataset's complete single `train`
split with the corrected columns (old column preserved as `language_volume`, plus
`language_glotlid`/`language_conf` for audit), and can restamp existing per-page scorecard parquets in
place — `language` is stratification metadata no axis reads, so restamping cached scorecards is
identical to re-scoring them from cached OCR. A Hub push refreshes the card's generated schema metadata,
then reloads the immutable resulting commit to fail closed on card/parquet drift.

  HF_TOKEN=... uv run runners/add_language.py --push                     # correct the Hub dataset
  uv run runners/add_language.py --restamp 'data/scorecards/*.parquet'   # relabel cached scorecards
"""
import argparse
import glob
import pathlib
from collections import Counter

import pandas as pd

# glotlid ISO639-3 -> the short codes the board stratifies by; anything else -> "und".
SHORT = {"eng": "en", "deu": "de", "fra": "fr", "lat": "la", "rus": "ru"}


def short_code(label):
    """'__label__deu_Latn' / 'deu_Latf' -> 'de'; unmapped languages -> 'und'."""
    iso = label.removeprefix("__label__").split("_")[0]
    return SHORT.get(iso, "und")


def predict(texts):
    """GlotLID top-1 per text -> [(raw_label, confidence)]. fasttext needs single-line input."""
    import fasttext  # lazy: tests exercise the pure functions without the model/dep
    from huggingface_hub import hf_hub_download

    model = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))
    out = []
    for t in texts:
        labels, confs = model.predict((t or "").replace("\n", " "))
        out.append((labels[0], float(confs[0])))
    return out


def resolve(pages, min_conf=0.5, min_chars=100):
    """Final per-page label. `pages` = [{page_id, volume, text, glotlid, conf}].

    A page stands on its own iff conf >= min_conf AND its code maps AND the text is long enough;
    the rest inherit the majority label among their volume's confident pages ("und" if none) —
    a self-consistent volume prior, not a hand-map."""
    for p in pages:
        p["code"] = short_code(p["glotlid"])
        p["confident"] = (p["conf"] >= min_conf and p["code"] != "und"
                          and len(p["text"] or "") >= min_chars)
    majority = {}
    for vol in {p["volume"] for p in pages}:
        codes = [p["code"] for p in pages if p["volume"] == vol and p["confident"]]
        majority[vol] = Counter(codes).most_common(1)[0][0] if codes else "und"
    return {p["page_id"]: (p["code"] if p["confident"] else majority[p["volume"]]) for p in pages}


def restamp_df(df, mapping):
    """Rewrite ONLY the `language` column by page_id; everything else untouched. Pure."""
    out = df.copy()
    out["language"] = out["page_id"].map(mapping).fillna(out["language"])
    return out


def push_and_verify(dataset, repo_id, *, dataset_dict_cls, load_dataset_fn):
    """Replace a single-train Hub dataset and attest its generated schema at the new commit."""
    split = str(dataset.split)
    known_splits = set(dataset.info.splits or {})
    if split != "train" or known_splits != {"train"}:
        raise ValueError(
            "--push requires a dataset whose complete default configuration is one train split; "
            f"loaded split={split!r}, known splits={sorted(known_splits)!r}."
        )

    commit = dataset_dict_cls({"train": dataset}).push_to_hub(repo_id, private=True)
    revision = getattr(commit, "oid", None)
    if not revision:
        raise RuntimeError("Hub push did not return an immutable commit SHA; refusing to continue.")

    verified = load_dataset_fn(repo_id, split="train", revision=revision)
    mismatches = []
    if len(verified) != len(dataset):
        mismatches.append(f"rows {len(verified)} != {len(dataset)}")
    if verified.column_names != dataset.column_names:
        mismatches.append(
            f"columns {verified.column_names!r} != {dataset.column_names!r}"
        )
    if verified.features != dataset.features:
        mismatches.append("features differ")
    if mismatches:
        raise RuntimeError(
            f"Pinned verification failed for {repo_id}@{revision}: " + "; ".join(mismatches)
        )
    return revision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="davanstrien/bhl-eval-impact-sample")
    ap.add_argument("--sidecar", default=str(pathlib.Path(__file__).resolve().parent.parent / "data" / "languages.parquet"))
    ap.add_argument(
        "--push", action="store_true",
        help="replace the complete single train split, refresh its Hub schema, and verify the new commit",
    )
    ap.add_argument("--restamp", default=None, metavar="GLOB", help="rewrite `language` in matching scorecard parquets")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--min-chars", type=int, default=100)
    args = ap.parse_args()
    from datasets import DatasetDict, load_dataset  # lazy: tests exercise pure functions without it

    sidecar = pathlib.Path(args.sidecar)
    if sidecar.exists() and not (args.push):
        side = pd.read_parquet(sidecar)
        print(f"reusing sidecar {sidecar} ({len(side)} pages)")
    else:
        ds = load_dataset(args.dataset, split="train")
        cols = ds.remove_columns([c for c in ds.column_names if c not in
                                  ("PageID", "volume", "text", "language", "language_volume")])
        raw = predict(cols["text"])
        pages = [{"page_id": int(pid), "volume": vol, "text": txt, "glotlid": lab, "conf": conf}
                 for pid, vol, txt, (lab, conf) in zip(cols["PageID"], cols["volume"], cols["text"], raw, strict=True)]
        final = resolve(pages, args.min_conf, args.min_chars)
        old = cols["language_volume"] if "language_volume" in cols.column_names else cols["language"]
        side = pd.DataFrame({
            "PageID": [p["page_id"] for p in pages],
            "language": [final[p["page_id"]] for p in pages],
            # glotlid is a str label; ty widens the mixed-value dict literal so can't prove it
            "language_glotlid": [p["glotlid"].removeprefix("__label__") for p in pages],  # ty: ignore[unresolved-attribute]
            "language_conf": [round(p["conf"], 4) for p in pages],
            "language_volume": old,
        })
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        side.to_parquet(sidecar, index=False)
        print(f"sidecar -> {sidecar}")
        for vol in sorted({p["volume"] for p in pages}):
            sub = side[[p["volume"] == vol for p in pages]]
            print(f"  {vol}: {dict(sub.language.value_counts())} (was {sub.language_volume.iloc[0]!r})")

        if args.push:
            if "language_volume" not in ds.column_names:
                ds = ds.rename_column("language", "language_volume")
            for c in ("language", "language_glotlid", "language_conf"):
                if c in ds.column_names:
                    ds = ds.remove_columns([c])
                ds = ds.add_column(c, list(side[c]))
            revision = push_and_verify(
                ds, args.dataset, dataset_dict_cls=DatasetDict, load_dataset_fn=load_dataset,
            )
            print(f"pushed corrected language columns -> {args.dataset}@{revision}")

    if args.restamp:
        mapping = dict(zip(side["PageID"], side["language"], strict=True))
        for p in sorted(glob.glob(args.restamp)):
            df = pd.read_parquet(p)
            new = restamp_df(df, mapping)
            changed = int((new["language"] != df["language"]).sum())
            new.to_parquet(p, index=False)
            print(f"  {pathlib.Path(p).name}: {changed} of {len(df)} rows relabelled")


if __name__ == "__main__":
    main()
