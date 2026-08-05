"""One place the runners turn a GT sample row + an OCR string into a scored scorecard row.

`run_openai` and `score_dataset` both join the same GT sample
(`text` / `body_text` / `furniture_text` / `regions_json` / `language` / `volume` / `PageID`) to a
model's OCR and call `scorer.score_page` with an identical argument shape. Keeping that shape in one
helper stops the three runners drifting apart (e.g. a renamed column, or one runner coercing
`page_id` while another doesn't — which would defeat the leaderboard's (model, page_id) dedup).
"""
import hashlib
import json
import math
import numbers
import pathlib
import re
from decimal import Decimal

import normalizers as N
import scorer as S

SCORECARD_SCHEMA_VERSION = "2.1"

# Producers signal a failed page with a sentinel STRING in the OCR column. A truncation is a
# different kind of failure from a producer error — the model hit its token cap rather than the
# run breaking — and only the latter disqualifies a model (report.model_scorecard). The prefix
# lives HERE, in the module both runners already share, because duplicating it is exactly what
# went wrong: score_dataset.py classified truncations while run_openai.py did not, so a board
# built from run_openai output would have marked every truncating model ineligible.
TRUNCATION_SENTINEL_PREFIX = "__ERR__FinishReason"


def is_truncated(text) -> bool:
    """True when a producer sealed this page as a token-cap truncation."""
    return isinstance(text, str) and text.startswith(TRUNCATION_SENTINEL_PREFIX)
DEFAULT_SAMPLE_STRATUM_THRESHOLD = 80
SAMPLE_STRATA = ("content", "sparse_blank")
SAMPLER_PROVENANCE_FIELDS = (
    "sampler_version", "sampler_seed", "sampler_requested_n", "sampler_source_repo",
    "sampler_source_revision",
)


def sample_stratum(text, threshold=DEFAULT_SAMPLE_STRATUM_THRESHOLD):
    """Deterministically classify a GT page by raw transcription length."""
    if not isinstance(threshold, numbers.Integral) or isinstance(threshold, bool) or threshold < 0:
        raise ValueError("sample stratum threshold must be a nonnegative integer")
    text = text if isinstance(text, str) else ""
    return "content" if len(text) >= threshold else "sparse_blank"
_CANONICAL_DECIMAL_INTEGER = re.compile(r"-?(?:0|[1-9]\d*)\.0+")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def canonical_page_id(value):
    """Canonical page identity without interpreting arbitrary string identifiers as numbers.

    Numeric integral values and canonical decimal spellings such as ``123.0`` collapse to ``123``
    for int/string/parquet round-trips. Leading-zero and scientific-notation strings remain exact.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if _CANONICAL_DECIMAL_INTEGER.fullmatch(value):
            return str(int(Decimal(value)))
        return value
    try:
        if math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, Decimal) and value == value.to_integral_value():
        return str(int(value))
    return str(value)


def page_set_fingerprint(page_ids):
    """Return a stable, order-independent SHA256 over canonical page identities.

    Canonical JSON is length-delimited, unlike a newline join: arbitrary IDs containing newlines
    therefore cannot make two distinct page sets serialize to the same byte stream.
    """
    ids = [canonical_page_id(page_id) for page_id in page_ids]
    if None in ids:
        raise ValueError("Dataset page IDs must be non-empty.")
    if len(set(ids)) != len(ids):
        raise ValueError("Dataset page IDs must be unique after safe canonicalization.")
    payload = json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_dataset_revision(dataset_id, requested_revision=None):
    """Resolve a remote HF dataset revision to its immutable commit SHA before loading it.

    Existing local paths have no Hub revision, so callers load them normally and
    :func:`benchmark_provenance` derives a stable identity from their loaded content fingerprint.
    The Hub dependency stays lazy so the scoring core remains lightweight.
    """
    if pathlib.Path(dataset_id).exists():
        return None
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id=str(dataset_id), revision=requested_revision)
    resolved = getattr(info, "sha", None)
    if not isinstance(resolved, str) or not _COMMIT_SHA.fullmatch(resolved):
        raise ValueError(f"Hugging Face did not return a 40-character commit SHA for GT dataset {dataset_id!r}.")
    return resolved


def _dataset_column(dataset, key):
    try:
        return list(dataset[key])
    except (KeyError, TypeError, ValueError):
        return None


def _uniform_column(dataset, field, *, expected_length):
    values = _dataset_column(dataset, field)
    if values is None:
        return None
    if len(values) != expected_length:
        raise ValueError(f"GT dataset column {field!r} has an inconsistent length.")
    unique = {json.dumps(value, sort_keys=True, default=str) for value in values}
    if len(unique) != 1:
        raise ValueError(f"GT dataset column {field!r} must be uniform across the benchmark.")
    return values[0] if values else None


def benchmark_provenance(dataset_id, requested_revision, dataset, *, resolved_revision=None,
                         page_key="PageID"):
    """Describe the authoritative full GT dataset before any runner-side selection."""
    fingerprint = getattr(dataset, "_fingerprint", None)
    if resolved_revision is not None:
        if not isinstance(resolved_revision, str) or not _COMMIT_SHA.fullmatch(resolved_revision):
            raise ValueError("Resolved GT dataset revision must be a 40-character commit SHA.")
        dataset_identity = resolved_revision
        stable_revision = resolved_revision
    else:
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("Loaded local/in-memory GT dataset has no non-empty content fingerprint; "
                             "cannot pin benchmark provenance.")
        dataset_identity = fingerprint
        stable_revision = f"local:{fingerprint}"

    page_ids = _dataset_column(dataset, page_key)
    if page_ids is None:
        raise ValueError(f"GT dataset has no page identity column {page_key!r}.")
    texts = _dataset_column(dataset, "text")
    if texts is None or len(texts) != len(page_ids):
        raise ValueError("GT dataset needs a text column aligned with its page identities.")

    declared_threshold = _uniform_column(
        dataset, "sample_stratum_threshold", expected_length=len(page_ids),
    )
    threshold = DEFAULT_SAMPLE_STRATUM_THRESHOLD if declared_threshold is None else declared_threshold
    if not isinstance(threshold, numbers.Integral) or isinstance(threshold, bool) or threshold < 0:
        raise ValueError("GT dataset sample_stratum_threshold must be one uniform nonnegative integer.")
    threshold = int(threshold)
    derived_strata = [sample_stratum(text, threshold) for text in texts]
    declared_strata = _dataset_column(dataset, "sample_stratum")
    if declared_strata is not None:
        if len(declared_strata) != len(page_ids):
            raise ValueError("GT dataset sample_stratum column has an inconsistent length.")
        conflicts = [
            index for index, (declared, derived) in enumerate(zip(declared_strata, derived_strata, strict=True))
            if declared is not None and declared != derived
        ]
        if conflicts:
            raise ValueError(
                "GT dataset sample_stratum conflicts with deterministic text classification "
                f"at threshold {threshold} (first conflicting row: {conflicts[0]})."
            )

    provenance = {
        "dataset_id": str(dataset_id),
        "requested_revision": requested_revision,
        "resolved_revision": stable_revision,
        "dataset_fingerprint": dataset_identity,
        "expected_page_count": len(page_ids),
        "full_page_set_fingerprint": page_set_fingerprint(page_ids),
        "sample_stratum_threshold": threshold,
        "expected_stratum_counts": {
            stratum: derived_strata.count(stratum) for stratum in SAMPLE_STRATA
        },
    }
    sampler_presence = {
        field: _dataset_column(dataset, field) is not None for field in SAMPLER_PROVENANCE_FIELDS
    }
    if any(sampler_presence.values()):
        if not all(sampler_presence.values()):
            missing = [field for field, present in sampler_presence.items() if not present]
            raise ValueError("GT dataset has incomplete sampler provenance: " + ", ".join(missing))
        sampler = {
            field: _uniform_column(dataset, field, expected_length=len(page_ids))
            for field in SAMPLER_PROVENANCE_FIELDS
        }
        for field in ("sampler_version", "sampler_source_repo", "sampler_source_revision"):
            if not isinstance(sampler[field], str) or not sampler[field].strip():
                raise ValueError(f"GT dataset {field} must be a uniform non-empty string.")
        for field in ("sampler_seed", "sampler_requested_n"):
            if not isinstance(sampler[field], numbers.Integral) or isinstance(sampler[field], bool):
                raise ValueError(f"GT dataset {field} must be a uniform integer.")
            sampler[field] = int(sampler[field])
        if sampler["sampler_requested_n"] < 0:
            raise ValueError("GT dataset sampler_requested_n must be nonnegative.")
        sampler["sample_stratum_threshold"] = threshold
        provenance["sampler_provenance"] = sampler
    return provenance


def flat_scorecard_rows(rows):
    """Scorecard rows with dict fields serialized as canonical JSON, ready for a parquet frame.

    Every runner must stringify provenance identically: the board compares these values as strings
    when it rejects mixed scorer/normalizer/schema/benchmark provenance, so a serialization that
    drifted in one runner would make that runner's scorecards unmergeable with the others'.
    Frame construction stays with the callers, keeping pandas out of the scoring core.
    """
    return [
        {key: (json.dumps(value) if isinstance(value, dict) else value) for key, value in row.items()}
        for row in rows
    ]


def score_gt_row(gt, ocr, *, model, err, page_id=None, truncated=False):
    """Score one GT sample row `gt` against `ocr` → the flat scorecard row, tagged model/volume/error.

    `err=True` (the runner already saw an API error / error sentinel) scores an empty OCR so the page
    counts as a real miss, and the row is flagged for exclusion from aggregates. `regions_json` is
    parsed when present and otherwise passed through as `None` (the scorer accepts region-less pages).
    `page_id` defaults to the row's `PageID`; pass it explicitly to control the dedup key's type.

    `truncated=True` marks an error row whose cause was the model hitting its token cap rather than
    the run failing. It is a SUBSET of `err`: the page is excluded from aggregates either way, but
    only a non-truncated error makes a model ineligible (see report.model_scorecard). Schema 2.1
    added this field because the distinction has to travel WITH the row — it is derived from the
    OCR text's sentinel, which no downstream layer can see, and without it `leaderboard.py` would
    rebuild a board on which every generating model is ineligible.
    """
    regions = json.loads(gt["regions_json"]) if gt.get("regions_json") else None
    s = S.score_page(gt["text"], "" if err else ocr,
                     body_text=gt.get("body_text"), furniture_text=gt.get("furniture_text"),
                     regions=regions, language=gt.get("language"),
                     page_id=gt.get("PageID") if page_id is None else page_id)
    threshold = gt.get("sample_stratum_threshold")
    if threshold is None:
        threshold = DEFAULT_SAMPLE_STRATUM_THRESHOLD
    elif not isinstance(threshold, numbers.Integral) or isinstance(threshold, bool) or threshold < 0:
        raise ValueError("GT sample_stratum_threshold must be a nonnegative integer when declared.")
    threshold = int(threshold)
    derived_stratum = sample_stratum(gt.get("text"), threshold)
    stratum = gt.get("sample_stratum")
    if stratum is None:
        stratum = derived_stratum
    elif stratum not in SAMPLE_STRATA or stratum != derived_stratum:
        raise ValueError(
            "GT sample_stratum conflicts with deterministic text classification "
            f"at threshold {threshold}: declared {stratum!r}, expected {derived_stratum!r}."
        )
    s.update(model=model, volume=gt.get("volume"), error=err, truncated=bool(truncated),
             sample_stratum=stratum, sample_stratum_threshold=threshold,
             scorecard_schema_version=SCORECARD_SCHEMA_VERSION,
             score_provenance=N.provenance())
    return s
