# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "huggingface-hub", "pandas", "pyarrow", "jiwer>=4,<5"]
# ///
"""Cross-check: does the NEW finebooks positional GT↔page mapping agree with the OLD content-matched
pairing where they overlap? Both datasets derive their transcription from the same IMPACT PAGE-XML
keyed by `pcGtsId` — new `xml_path` basename (pc-00667793) == old `gt_page_id` — so a high text
agreement on the overlap confirms finebooks didn't mis-thread reading order or mis-assign a page.
(The old set's content-verified IA-scan rows are the trustworthy overlap; the new set is authoritative.)

  uv run scripts/crosscheck_mapping.py
"""
import argparse
import os
import pathlib
import statistics
import sys

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scoring"))
import normalizers as N  # noqa: E402

NEW = "finebooks/bhl-eval-impact"
OLD = "davanstrien/bhl-impact-groundtruth"


def agreement(a, b):
    """1 - reading-lane CER between two transcriptions of the same page (1.0 = identical)."""
    ec = N.edit_counts(N.norm(a, "reading"), N.norm(b, "reading"))
    rate = ec.rate()
    if rate is None:
        return None
    return round(1 - min(rate, 1.0), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=OLD)
    ap.add_argument("--new", default=NEW)
    args = ap.parse_args()

    new = pd.read_parquet(hf_hub_download(args.new, "metadata.parquet", repo_type="dataset"))
    new["pcid"] = new["xml_path"].map(lambda p: os.path.basename(p).replace(".xml", ""))
    new_by_pcid = dict(zip(new["pcid"], new["text"], strict=True))
    print(f"new: {len(new)} pages")

    old = load_dataset(args.old, split="train").to_pandas()
    ia = old[old["image_source"] == "IA-scan"].copy()
    print(f"old: {len(old)} rows, {len(ia)} content-verified (IA-scan)\n")

    matched, unmatched, scores, low = 0, 0, [], []
    for r in ia.itertuples():
        new_text = new_by_pcid.get(r.gt_page_id)
        if new_text is None:
            unmatched += 1
            continue
        matched += 1
        a = agreement(r.full_text, new_text)
        if a is not None:
            scores.append(a)
            if a < 0.85:
                low.append((r.gt_page_id, r.volume, a))

    print(f"pcGtsId join: {matched} matched / {unmatched} old IA-scan rows had no new partner")
    if scores:
        scores.sort()
        print(f"text agreement (1 - reading CER): median={statistics.median(scores):.3f} "
              f"mean={statistics.mean(scores):.3f} p10={scores[len(scores)//10]:.3f}")
        print(f"  ≥0.95: {sum(s>=0.95 for s in scores)}/{len(scores)}  "
              f"≥0.85: {sum(s>=0.85 for s in scores)}/{len(scores)}")
    if low:
        print(f"\n{len(low)} low-agreement (<0.85) pages — inspect these:")
        for pcid, vol, a in sorted(low, key=lambda x: x[2])[:15]:
            print(f"   {pcid}  {vol:28} agree={a}")
    else:
        print("\nno low-agreement pages — mapping is consistent on the overlap.")


if __name__ == "__main__":
    main()
