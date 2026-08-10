# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "huggingface-hub", "pandas", "pyarrow", "pillow"]
# ///
"""Build a deterministic content/sparse-blank evaluation sample.

Sampling helpers in this module use only the standard library, so their boundary and determinism
contracts can be tested without importing datasets, pandas, Pillow, or network clients. Heavy imports
are confined to :func:`main`. After building a sample, run ``runners/add_language.py --push`` for its
per-page best-effort language labels before producing language-stratified benchmark reports.
"""
import argparse
import json
import pathlib
import random
import sys
from collections import defaultdict

SRC = "finebooks/bhl-impact-gt"
STRATA = ("content", "sparse_blank")
SAMPLER_VERSION = "1.0"


def resolve_source_revision(source_repo, requested_revision=None, *, api=None):
    """Resolve a branch/tag once to the immutable dataset commit used by every download."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    info = api.dataset_info(repo_id=source_repo, revision=requested_revision)
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(f"Could not resolve an immutable source revision for {source_repo!r}.")
    return revision


def sampler_provenance(*, seed, requested_n, source_repo, source_revision, threshold):
    """Uniform primitive fields persisted on every sampled dataset record."""
    return {
        "sampler_version": SAMPLER_VERSION,
        "sampler_seed": seed,
        "sampler_requested_n": requested_n,
        "sampler_source_repo": source_repo,
        "sampler_source_revision": source_revision,
        "sample_stratum_threshold": threshold,
    }


def classify_sample_stratum(text, threshold=80):
    """Classify by raw GT character length; the boundary belongs to ``content``."""
    if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 0:
        raise ValueError("sample stratum threshold must be a nonnegative integer")
    text = text if isinstance(text, str) else ""
    return "content" if len(text) >= threshold else "sparse_blank"


def proportional_quotas(capacities, n, *, guarantee_nonempty=False):
    """Allocate exactly ``min(n, total)`` with largest remainders and capacity bounds."""
    nonempty = {key: int(size) for key, size in capacities.items() if size > 0}
    if n < 0:
        raise ValueError("sample size must be nonnegative")
    target = min(n, sum(nonempty.values()))
    if not target:
        return {key: 0 for key in capacities}
    raw = {key: target * size / sum(nonempty.values()) for key, size in nonempty.items()}
    quotas = {key: min(nonempty[key], int(raw[key])) for key in nonempty}
    remaining = target - sum(quotas.values())
    order = sorted(nonempty, key=lambda key: (-(raw[key] - int(raw[key])), str(key)))
    while remaining:
        progressed = False
        for key in order:
            if quotas[key] < nonempty[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:  # defensive: target is capacity-bounded, so this should be unreachable
            raise RuntimeError("could not allocate requested sample quota")
    if guarantee_nonempty and target >= len(nonempty):
        empty = [key for key in sorted(nonempty, key=str) if quotas[key] == 0]
        for key in empty:
            donors = sorted(
                (candidate for candidate in nonempty if quotas[candidate] > 1),
                key=lambda candidate: (-quotas[candidate], str(candidate)),
            )
            if not donors:
                break
            quotas[donors[0]] -= 1
            quotas[key] = 1
    return {key: quotas.get(key, 0) for key in capacities}


def _volume_quotas(capacities, n, rng):
    """Spread a stratum quota evenly across volumes, redistributing when a volume is undersized."""
    quotas = {volume: 0 for volume in capacities}
    available = [volume for volume, size in capacities.items() if size]
    rng.shuffle(available)
    target = min(n, sum(capacities.values()))
    # Round-robin is an equal-volume allocation. Removing full volumes naturally redistributes their
    # unfillable share while preserving broad coverage of every nonempty volume.
    while sum(quotas.values()) < target:
        progressed = False
        for volume in available:
            if quotas[volume] < capacities[volume]:
                quotas[volume] += 1
                progressed = True
                if sum(quotas.values()) == target:
                    break
        if not progressed:
            raise RuntimeError("could not redistribute volume quota")
    return quotas


def sample_indices(rows, n, seed=0, *, threshold=80, text_key="text", volume_key="BarCode"):
    """Return deterministic source indices for an exact stratified sample.

    Stratum quotas are proportional (largest remainder), both nonempty strata are guaranteed when
    ``n >= 2``, and each stratum's quota is spread across volumes before rows are sampled.
    """
    records = list(rows)
    if n < 0:
        raise ValueError("sample size must be nonnegative")
    target = min(n, len(records))
    if target == len(records):
        return list(range(len(records)))
    grouped = {stratum: defaultdict(list) for stratum in STRATA}
    for index, row in enumerate(records):
        stratum = classify_sample_stratum(row.get(text_key), threshold)
        volume = row.get(volume_key)
        grouped[stratum][volume].append(index)
    stratum_capacities = {
        stratum: sum(len(indices) for indices in volumes.values())
        for stratum, volumes in grouped.items()
    }
    quotas = proportional_quotas(stratum_capacities, target, guarantee_nonempty=target >= 2)
    rng = random.Random(seed)
    selected = []
    for stratum in STRATA:
        volumes = grouped[stratum]
        volume_quotas = _volume_quotas(
            {volume: len(indices) for volume, indices in volumes.items()}, quotas[stratum], rng,
        )
        for volume in sorted(volumes, key=str):
            candidates = volumes[volume]
            count = volume_quotas[volume]
            selected.extend(rng.sample(candidates, count))
    if len(selected) != target or len(set(selected)) != target:
        raise RuntimeError("sampling failed its exact-size/uniqueness contract")
    return sorted(selected)


def stratified(rows, n, seed, threshold=80):
    """Convenience wrapper supporting either records or a pandas-like DataFrame."""
    if hasattr(rows, "to_dict") and hasattr(rows, "iloc"):
        records = rows.to_dict("records")
        return rows.iloc[sample_indices(records, n, seed, threshold=threshold)].reset_index(drop=True)
    records = list(rows)
    return [records[index] for index in sample_indices(records, n, seed, threshold=threshold)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60, help="exact pages, or all rows when source is smaller")
    parser.add_argument("--repo", default="davanstrien/bhl-eval-impact-sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source-revision", default=None,
        help="source branch/tag/commit; resolved once to an immutable commit before downloading",
    )
    parser.add_argument(
        "--min-text", type=int, default=80,
        help="classify pages below this GT length as sparse_blank (pages are no longer filtered)",
    )
    parser.add_argument("--public", action="store_true", help="push public (default: private)")
    parser.add_argument("--no-push", action="store_true", help="build + save locally, skip the push")
    args = parser.parse_args()

    import pandas as pd
    from datasets import Dataset, Image, Value
    from huggingface_hub import hf_hub_download

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scoring"))
    import gt_docling as G

    resolved_revision = resolve_source_revision(SRC, args.source_revision)
    metadata = hf_hub_download(
        SRC, "metadata.parquet", repo_type="dataset", revision=resolved_revision,
    )
    frame = pd.read_parquet(metadata)
    sample = stratified(frame, args.n, args.seed, threshold=args.min_text)
    sample["sample_stratum"] = sample["text"].map(
        lambda text: classify_sample_stratum(text, args.min_text)
    )
    sample["sample_stratum_threshold"] = args.min_text
    print(
        f"sampled {len(sample)} pages: volumes={dict(sample.BarCode.value_counts())}; "
        f"strata={dict(sample.sample_stratum.value_counts())}"
    )

    records, image_paths = [], []
    uniform_provenance = sampler_provenance(
        seed=args.seed,
        requested_n=args.n,
        source_repo=SRC,
        source_revision=resolved_revision,
        threshold=args.min_text,
    )
    for _, row in sample.iterrows():
        image_paths.append(hf_hub_download(
            SRC, row["file_name"], repo_type="dataset", revision=resolved_revision,
        ))
        records.append({
            "PageID": int(row["PageID"]),
            "BarCode": row["BarCode"],
            "volume": row["BarCode"],
            "language": None,
            "text": row["text"],
            "body_text": G.body_text(row["docling"]),
            "furniture_text": G.furniture_text(row["docling"]),
            "regions_json": json.dumps(G.regions(row["docling"]), ensure_ascii=False),
            "sample_stratum": row["sample_stratum"],
            **uniform_provenance,
            "docling": row["docling"],
            "xml_path": row["xml_path"],
        })
    dataset = Dataset.from_list(records).add_column("image", image_paths).cast_column("image", Image())
    dataset = dataset.cast_column("docling", Value("large_string"))
    print(dataset)

    if args.no_push:
        output = pathlib.Path(__file__).resolve().parent.parent / "data" / "sample"
        dataset.save_to_disk(str(output))
        print(f"saved locally -> {output}")
        return
    dataset.push_to_hub(args.repo, private=not args.public)
    print(f"pushed {len(dataset)} pages -> {args.repo} ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
