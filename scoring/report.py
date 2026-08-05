# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow", "jiwer>=4,<5"]
# ///
"""Validate and aggregate versioned per-page scorecards into model profiles."""
import json
import math
import numbers
import random
import statistics
from collections import defaultdict

import gt_score as GS
import normalizers as N
import scorer as S

ERROR_FLAG = "error"
REPORT_SCHEMA_VERSION = "2.1"
# postproc_version has no entry in current_global_provenance(): the registry lives in
# runners/normalize_outputs.py, which the frozen scoring core must not import. The board
# still refuses to mix rows normalized under different registry versions ("mixed" check
# below); "0" is the explicit value for text scored with no post-processing at all.
_REQUIRED_GLOBAL_PROVENANCE = (
    "norm_version", "scorer_version", "scorecard_schema_version", "postproc_version",
    "score_provenance", "benchmark_provenance",
)
_VERSION_FIELDS = ("norm_version", "scorer_version", "scorecard_schema_version", "postproc_version")
_STRUCTURED_FIELDS = ("score_provenance", "benchmark_provenance", "run_provenance")
_REQUIRED_BENCHMARK_FIELDS = (
    "dataset_id", "requested_revision", "resolved_revision", "dataset_fingerprint",
    "expected_page_count", "full_page_set_fingerprint", "sample_stratum_threshold",
    "expected_stratum_counts",
)
_REQUIRED_SCORE_PROVENANCE_FIELDS = (
    "norm_version", "jiwer_version", "lanes", "grapheme_mode", "word_boundary",
    "illegible_marker",
)
_REQUIRED_LANES = ("diplomatic", "reading")
_REQUIRED_PAYLOAD_FIELDS = (
    ERROR_FLAG, "volume", "sample_stratum", "sample_stratum_threshold",
    *S.COUNT_COLUMNS, *S.SUPPORT_COLUMNS, *S.AXIS_NAMES,
)


def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _is_error_flag(value):
    if value is None or _is_missing(value):
        return False
    try:
        return bool(value)
    except Exception:
        return False


def _vals(rows, axis):
    return [row[axis] for row in rows if row.get(axis) is not None and row[axis] == row[axis]]


def _count_pool(rows, key):
    s = d = i = h = 0
    for row in rows:
        cs = row.get(f"{key}_s")
        if _is_missing(cs):
            continue
        rd, ri, rh = (row.get(f"{key}_d") or 0), (row.get(f"{key}_i") or 0), (row.get(f"{key}_h") or 0)
        s, d, i, h = s + cs, d + rd, i + ri, h + rh
    return s, d, i, h


def micro_rate(rows, key):
    """Pool character edit supports and divide once."""
    s, d, i, h = _count_pool(rows, key)
    denominator = s + d + h
    return round((s + d + i) / denominator, 4) if denominator else None


def _pool_support(rows, numerator_field, denominator_field):
    numerator = sum(row[numerator_field] for row in rows)
    denominator = sum(row[denominator_field] for row in rows)
    return numerator, denominator


def _support_rate(numerator, denominator, *, empty=None):
    return round(numerator / denominator, 4) if denominator else empty


def median(rows, axis):
    values = _vals(rows, axis)
    return round(statistics.median(values), 4) if values else None


def mean(rows, axis):
    values = _vals(rows, axis)
    return round(statistics.mean(values), 4) if values else None


def _pct(values, probability):
    values = sorted(values)
    k = (len(values) - 1) * probability
    floor = int(k)
    ceiling = min(floor + 1, len(values) - 1)
    return values[floor] + (values[ceiling] - values[floor]) * (k - floor)


def bootstrap_ci(rows, stat_fn, n=1000, alpha=0.05, seed=0, cluster="volume"):
    """Deterministic percentile bootstrap, resampling whole volumes where possible."""
    if not rows:
        return None, None
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(cluster)].append(row)
    units = list(groups.values()) if cluster and None not in groups and len(groups) > 1 else [[row] for row in rows]
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [row for _ in range(len(units)) for row in units[rng.randrange(len(units))]]
        value = stat_fn(sample)
        if value is not None:
            stats.append(value)
    if not stats:
        return None, None
    return round(_pct(stats, alpha / 2), 4), round(_pct(stats, 1 - alpha / 2), 4)


def stratify_micro(rows, key, by):
    groups = defaultdict(list)
    for row in rows:
        if row.get(by) is not None:
            groups[row[by]].append(row)
    return {group: micro_rate(group_rows, key) for group, group_rows in sorted(groups.items())}


def stratify_mean(rows, axis, by):
    groups = defaultdict(list)
    for row in rows:
        if row.get(by) is not None and row.get(axis) is not None and row[axis] == row[axis]:
            groups[row[by]].append(row[axis])
    return {group: round(statistics.mean(values), 4) for group, values in sorted(groups.items()) if values}


def _as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def agg_region_global_token_recall(rows):
    supports = defaultdict(lambda: [0, 0])
    for row in rows:
        for label, entry in _as_dict(row.get("region_global_token_recall")).items():
            supports[label][0] += entry["matched"]
            supports[label][1] += entry["reference"]
    return {
        label: {"rate": _support_rate(matched, reference), "matched": matched, "reference": reference}
        for label, (matched, reference) in sorted(supports.items())
    }


def _stratum_scorecards(rows, expected_strata=()):
    groups = defaultdict(list)
    for row in rows:
        groups[row["sample_stratum"]].append(row)
    result = {}
    for stratum in sorted(set(groups) | set(expected_strata)):
        submitted_rows = groups[stratum]
        group_rows = [row for row in submitted_rows if not _is_error_flag(row.get(ERROR_FLAG))]
        recall_matched, recall_reference = _pool_support(group_rows, "recall_matched", "recall_reference")
        extra, output = _pool_support(group_rows, "over_extraction_extra", "over_extraction_output")
        result[stratum] = {
            "submitted": len(submitted_rows),
            "n": len(group_rows),
            "errors": len(submitted_rows) - len(group_rows),
            "cer_reading_micro": micro_rate(group_rows, "cer_body_read"),
            "recall_micro": _support_rate(recall_matched, recall_reference),
            "recall_matched": recall_matched,
            "recall_reference": recall_reference,
            "over_extraction_micro": _support_rate(extra, output, empty=0.0),
            "over_extraction_extra": extra,
            "over_extraction_output": output,
        }
    return result


def model_scorecard(
    rows,
    ci_n=1000,
    *,
    expected_n=None,
    missing_n=None,
    extra_n=0,
    page_set_match=True,
    include_policy_diagnostics=False,
    expected_strata=(),
    truncated_n=0,
):
    """Aggregate one model by micro-pooling every token and CER support.

    `truncated_n` is the subset of this model's error rows whose completion hit the token
    cap (`finish_reason != "stop"`), counted upstream where the sentinel is still visible.
    Truncated pages are excluded from every aggregate exactly like other error rows — a
    runaway page emits thousands of characters against a few hundred of ground truth, so
    scoring it whole would turn micro-CER into a loop counter. But they do NOT make a model
    ineligible: a repetition loop is the MODEL failing on that page, not the run failing,
    and disqualifying every neural model over a handful of pages out of 2165 would leave
    only the classical engines rankable. See `truncation_rate` — it is a board column, not
    a diagnostic, and it must be read alongside CER because a higher rate means a model was
    scored on an easier effective page set.
    """
    clean = [row for row in rows if not _is_error_flag(row.get(ERROR_FLAG))]
    errors = len(rows) - len(clean)
    # Schema 2.1 carries `truncated` per row, so the count travels WITH the data and any caller —
    # leaderboard.py rebuilding from stored scorecards, not just score_dataset.py in one pass —
    # gets the right eligibility. `truncated_n` remains an override for pre-2.1 rows.
    from_rows = sum(1 for row in rows if _is_error_flag(row.get("truncated")))
    truncated = min(from_rows or truncated_n, errors)  # never claim more truncations than errors
    producer_errors = errors - truncated
    missing = missing_n if missing_n is not None else (
        max(0, expected_n - len(rows)) if expected_n is not None else 0
    )
    lo, hi = bootstrap_ci(clean, lambda sample: micro_rate(sample, "cer_body_read"), n=ci_n)
    recall_matched, recall_reference = _pool_support(clean, "recall_matched", "recall_reference")
    extra, output = _pool_support(clean, "over_extraction_extra", "over_extraction_output")
    furniture_matched, furniture_reference = _pool_support(
        clean,
        "furniture_global_token_recall_matched",
        "furniture_global_token_recall_reference",
    )
    card = {
        "n": len(clean),
        "submitted": len(rows),
        "errors": errors,
        "truncated": truncated,
        "producer_errors": producer_errors,
        "truncation_rate": (truncated / len(rows)) if rows else None,
        "missing": missing,
        "extra": extra_n,
        "benchmark_page_set_match": page_set_match,
        "eligible": (
            producer_errors == 0 and missing == 0 and extra_n == 0 and page_set_match
            and micro_rate(clean, "cer_body_dip") is not None
            and micro_rate(clean, "cer_body_read") is not None
        ),
        "cer_diplomatic_micro": micro_rate(clean, "cer_body_dip"),
        "cer_reading_micro": micro_rate(clean, "cer_body_read"),
        "cer_ci": (lo, hi),
        "cer_reading_median": median(clean, "cer_reading"),
        "recall_micro": _support_rate(recall_matched, recall_reference),
        "recall_matched": recall_matched,
        "recall_reference": recall_reference,
        "over_extraction_micro": _support_rate(extra, output, empty=0.0),
        "over_extraction_extra": extra,
        "over_extraction_output": output,
        "script_match": mean(clean, "script_match"),
        "furniture_global_token_recall_micro": _support_rate(furniture_matched, furniture_reference),
        "furniture_global_token_recall_matched": furniture_matched,
        "furniture_global_token_recall_reference": furniture_reference,
        "by_language": stratify_micro(clean, "cer_body_read", "language"),
        "by_volume": stratify_micro(clean, "cer_body_read", "volume"),
        "by_script": stratify_mean(clean, "script_match", "gt_script"),
        "by_sample_stratum": _stratum_scorecards(rows, expected_strata),
        "region_global_token_recall": agg_region_global_token_recall(clean),
    }
    if include_policy_diagnostics:
        card["policy_diagnostics"] = {
            "full_text_cer": {
                "cer_diplomatic_micro": micro_rate(clean, "cer_full_dip"),
                "cer_reading_micro": micro_rate(clean, "cer_full_read"),
            }
        }
    return card


def _canonical_id(value):
    return GS.canonical_page_id(value)


def _structured_value(value, field, *, row_index=None):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            where = f" on scorecard row {row_index}" if row_index is not None else ""
            raise ValueError(f"Provenance field {field!r}{where} is not valid JSON.") from exc
    if not isinstance(value, dict):
        where = f" on scorecard row {row_index}" if row_index is not None else ""
        raise ValueError(f"Provenance field {field!r}{where} must be a JSON object.")
    return value


def _stable(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def stable_provenance(value):
    """Canonical comparison form for a provenance value, whether stored as a dict or a JSON string.

    Runners serialize dict provenance to JSON before parquet, so callers that need to compare an
    on-disk scorecard's provenance with an in-memory one must normalize both through this.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return _stable(value)
    return _stable(value)


def current_global_provenance():
    """Global provenance a scorecard must carry to be comparable under this report version.

    Exposed so callers can check scorecards for comparability *before* producing more of them,
    without duplicating the version contract that :func:`_validated_rows` enforces.
    """
    return {
        "norm_version": N.NORM_VERSION,
        "scorer_version": S.SCORER_VERSION,
        "scorecard_schema_version": GS.SCORECARD_SCHEMA_VERSION,
    }


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _validate_score_provenance(value, norm_version, *, row_index):
    missing = [field for field in _REQUIRED_SCORE_PROVENANCE_FIELDS if field not in value]
    if missing:
        raise ValueError(
            f"score_provenance is missing required fields on scorecard row {row_index}: "
            + ", ".join(missing)
        )
    for field in ("norm_version", "jiwer_version", "grapheme_mode", "word_boundary", "illegible_marker"):
        if not _nonempty_string(value[field]):
            raise ValueError(f"score_provenance.{field} on scorecard row {row_index} must be a non-empty string.")
    if value["norm_version"] != norm_version:
        raise ValueError(
            f"score_provenance.norm_version on scorecard row {row_index} does not match the row's norm_version."
        )
    lanes = value["lanes"]
    if not isinstance(lanes, dict):
        raise ValueError(f"score_provenance.lanes on scorecard row {row_index} must be a JSON object.")
    missing_lanes = [lane for lane in _REQUIRED_LANES if not _nonempty_string(lanes.get(lane))]
    if missing_lanes:
        raise ValueError(
            f"score_provenance.lanes is missing non-empty lane definitions on scorecard row {row_index}: "
            + ", ".join(missing_lanes)
        )


def _finite_number(value):
    return (
        isinstance(value, numbers.Real) and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonnegative_integer(value):
    return _finite_number(value) and float(value).is_integer() and value >= 0


def _validate_region_global_token_recall(value, *, row_index):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"region_global_token_recall on scorecard row {row_index} is not valid JSON."
            ) from exc
    if not isinstance(value, dict):
        raise ValueError(
            f"region_global_token_recall on scorecard row {row_index} must be a JSON object or object string."
        )
    for label, entry in value.items():
        if not isinstance(label, str) or not label or not isinstance(entry, dict):
            raise ValueError(f"region_global_token_recall on scorecard row {row_index} has a malformed entry.")
        if set(entry) != {"rate", "matched", "reference"}:
            raise ValueError(f"region_global_token_recall on scorecard row {row_index} needs rate/matched/reference.")
        matched, reference, rate = entry["matched"], entry["reference"], entry["rate"]
        if not _nonnegative_integer(matched) or not _nonnegative_integer(reference) or matched > reference:
            raise ValueError(f"region_global_token_recall on scorecard row {row_index} has malformed supports.")
        if not reference or not _finite_number(rate) or rate != round(matched / reference, 4):
            raise ValueError(f"region_global_token_recall on scorecard row {row_index} rate does not equal supports.")


def _validate_payload(row, *, row_index):
    missing = [field for field in _REQUIRED_PAYLOAD_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Scorecard row {row_index} is missing required payload fields: " + ", ".join(missing))
    if not isinstance(row[ERROR_FLAG], bool):
        raise ValueError(f"Scorecard row {row_index} field {ERROR_FLAG!r} must be boolean.")
    if not _nonempty_string(row["volume"]):
        raise ValueError(f"Scorecard row {row_index} field 'volume' must be a non-empty string.")
    if row["sample_stratum"] not in {"content", "sparse_blank"}:
        raise ValueError(f"Scorecard row {row_index} has an invalid sample_stratum.")
    if not _nonnegative_integer(row["sample_stratum_threshold"]):
        raise ValueError(f"Scorecard row {row_index} sample_stratum_threshold must be a nonnegative integer.")
    for key in S.COUNT_KEYS:
        columns = [f"{key}_{part}" for part in ("s", "d", "i", "h")]
        values = [row[column] for column in columns]
        if any(not _nonnegative_integer(value) for value in values):
            raise ValueError(
                f"Scorecard row {row_index} count group {key!r} must contain four finite nonnegative integers."
            )
    for name, (numerator_field, denominator_field) in S.SUPPORT_GROUPS.items():
        numerator, denominator = row[numerator_field], row[denominator_field]
        if not _nonnegative_integer(numerator) or not _nonnegative_integer(denominator) or numerator > denominator:
            raise ValueError(
                f"Scorecard row {row_index} support group {name!r} must contain nonnegative integers with numerator <= denominator."
            )
        expected = _support_rate(numerator, denominator, empty=0.0 if name == "over_extraction" else None)
        value = row[name]
        if (_is_missing(value) and expected is not None) or (not _is_missing(value) and value != expected):
            raise ValueError(f"Scorecard row {row_index} axis {name!r} does not equal its supports.")
        if not _is_missing(value) and not 0 <= value <= 1:
            raise ValueError(f"Scorecard row {row_index} fraction axis {name!r} must be in [0, 1].")
    for axis, key in (("cer_diplomatic", "cer_body_dip"), ("cer_reading", "cer_body_read")):
        s, d, i, h = (row[f"{key}_{part}"] for part in ("s", "d", "i", "h"))
        denominator = s + d + h
        expected = round((s + d + i) / denominator, 4) if denominator else None
        value = row[axis]
        if (_is_missing(value) and expected is not None) or (not _is_missing(value) and value != expected):
            raise ValueError(
                f"Scorecard row {row_index} axis {axis!r} does not equal its body edit-count rate."
            )
    for axis in S.AXIS_NAMES:
        value = row[axis]
        if axis == "region_global_token_recall":
            _validate_region_global_token_recall(value, row_index=row_index)
        elif not _is_missing(value) and not _finite_number(value):
            raise ValueError(f"Scorecard row {row_index} axis {axis!r} must be missing or a finite number.")
        elif axis == "script_match" and not _is_missing(value) and not 0 <= value <= 1:
            raise ValueError(f"Scorecard row {row_index} fraction axis {axis!r} must be in [0, 1].")


def page_set_fingerprint(page_ids):
    return GS.page_set_fingerprint(page_ids)


def _validated_rows(rows, expected_page_ids=None, *, require_complete=True):
    if not rows:
        raise ValueError("No scorecard rows supplied.")
    by_model = defaultdict(list)
    identities = set()
    provenance_values = {key: {} for key in _REQUIRED_GLOBAL_PROVENANCE}
    run_values = defaultdict(dict)
    expected_versions = current_global_provenance()
    for index, row in enumerate(rows):
        _validate_payload(row, row_index=index)
        model = row.get("model")
        page_id = _canonical_id(row.get("page_id"))
        if model is None or _is_missing(model) or not str(model).strip():
            raise ValueError(f"Scorecard row {index} has no model ID; every row needs a non-empty 'model'.")
        model = str(model)
        if page_id is None:
            raise ValueError(f"Scorecard row {index} for model {model!r} has no page ID.")
        identity = model, page_id
        if identity in identities:
            raise ValueError(f"Duplicate scorecard row for model={model!r}, page_id={page_id!r}.")
        identities.add(identity)
        for key in _REQUIRED_GLOBAL_PROVENANCE:
            if key not in row or row[key] is None or _is_missing(row[key]):
                raise ValueError(f"Scorecard row {index} is missing required provenance field {key!r}.")
            value = _structured_value(row[key], key, row_index=index) if key in _STRUCTURED_FIELDS else row[key]
            if key in _VERSION_FIELDS and not _nonempty_string(value):
                raise ValueError(f"Provenance version field {key!r} on scorecard row {index} must be a non-empty string.")
            if key in expected_versions and value != expected_versions[key]:
                raise ValueError(
                    f"Unsupported {key} {value!r} on scorecard row {index}; expected {expected_versions[key]!r}. Re-score raw OCR with the current scorer."
                )
            if key == "score_provenance":
                _validate_score_provenance(value, row.get("norm_version"), row_index=index)
            provenance_values[key][_stable(value)] = value
        if "run_provenance" not in row or row["run_provenance"] is None or _is_missing(row["run_provenance"]):
            raise ValueError(f"Scorecard row {index} is missing required provenance field 'run_provenance'.")
        run_value = _structured_value(row["run_provenance"], "run_provenance", row_index=index)
        if not run_value:
            raise ValueError(f"Scorecard row {index} has empty 'run_provenance'; a producer run identity is required.")
        run_values[model][_stable(run_value)] = run_value
        by_model[model].append(row)

    mixed = [key for key, values in provenance_values.items() if len(values) != 1]
    if mixed:
        raise ValueError("Mixed scorecard provenance is not comparable; conflicting fields: " + ", ".join(mixed))
    mixed_runs = [model for model, values in run_values.items() if len(values) != 1]
    if mixed_runs:
        raise ValueError("A model cannot combine multiple run provenance objects (Frankenstein run): " + ", ".join(sorted(mixed_runs)))

    benchmark = next(iter(provenance_values["benchmark_provenance"].values()))
    missing_benchmark = [field for field in _REQUIRED_BENCHMARK_FIELDS if field not in benchmark]
    if missing_benchmark:
        raise ValueError("benchmark_provenance is missing required fields: " + ", ".join(missing_benchmark))
    expected_n = benchmark["expected_page_count"]
    if not isinstance(expected_n, int) or isinstance(expected_n, bool) or expected_n <= 0:
        raise ValueError("benchmark_provenance.expected_page_count must be a positive integer.")
    full_fingerprint = benchmark["full_page_set_fingerprint"]
    if not isinstance(full_fingerprint, str) or not full_fingerprint:
        raise ValueError("benchmark_provenance.full_page_set_fingerprint must be a non-empty string.")
    for field in ("dataset_id", "resolved_revision", "dataset_fingerprint"):
        if not isinstance(benchmark[field], str) or not benchmark[field]:
            raise ValueError(f"benchmark_provenance.{field} must be a non-empty string.")
    revision = benchmark["requested_revision"]
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError("benchmark_provenance.requested_revision must be null or a non-empty string.")
    stratum_threshold = benchmark["sample_stratum_threshold"]
    if not _nonnegative_integer(stratum_threshold):
        raise ValueError("benchmark_provenance.sample_stratum_threshold must be a nonnegative integer.")
    expected_strata = benchmark["expected_stratum_counts"]
    if not isinstance(expected_strata, dict) or set(expected_strata) != set(GS.SAMPLE_STRATA):
        raise ValueError(
            "benchmark_provenance.expected_stratum_counts must contain exactly content and sparse_blank."
        )
    if any(not _nonnegative_integer(count) for count in expected_strata.values()) \
            or sum(expected_strata.values()) != expected_n:
        raise ValueError(
            "benchmark_provenance.expected_stratum_counts must be nonnegative integers summing to expected_page_count."
        )
    sampler = benchmark.get("sampler_provenance")
    if sampler is not None:
        required_sampler_fields = {*GS.SAMPLER_PROVENANCE_FIELDS, "sample_stratum_threshold"}
        if not isinstance(sampler, dict) or set(sampler) != required_sampler_fields:
            raise ValueError(
                "benchmark_provenance.sampler_provenance has incomplete or unknown fields."
            )
        for field in ("sampler_version", "sampler_source_repo", "sampler_source_revision"):
            if not _nonempty_string(sampler[field]):
                raise ValueError(
                    f"benchmark_provenance.sampler_provenance.{field} must be a non-empty string."
                )
        if not _nonnegative_integer(sampler["sampler_requested_n"]):
            raise ValueError(
                "benchmark_provenance.sampler_provenance.sampler_requested_n must be a nonnegative integer."
            )
        if not isinstance(sampler["sampler_seed"], numbers.Integral) \
                or isinstance(sampler["sampler_seed"], bool):
            raise ValueError(
                "benchmark_provenance.sampler_provenance.sampler_seed must be an integer."
            )
        if sampler["sample_stratum_threshold"] != stratum_threshold:
            raise ValueError(
                "benchmark_provenance sampler and benchmark stratum thresholds must match."
            )

    page_strata = {}
    for model, model_rows in by_model.items():
        observed_counts = defaultdict(int)
        for row in model_rows:
            if row["sample_stratum_threshold"] != stratum_threshold:
                raise ValueError(
                    f"Scorecard row threshold for model {model!r} does not match benchmark_provenance."
                )
            page_id = _canonical_id(row["page_id"])
            stratum = row["sample_stratum"]
            previous = page_strata.setdefault(page_id, stratum)
            if previous != stratum:
                raise ValueError(
                    f"Conflicting sample_stratum for canonical page {page_id!r} across models."
                )
            observed_counts[stratum] += 1
        if any(observed_counts[stratum] > expected_strata[stratum] for stratum in GS.SAMPLE_STRATA):
            raise ValueError(
                f"Model {model!r} sample-stratum counts exceed benchmark_provenance expectations."
            )
        if len(model_rows) == expected_n and any(
            observed_counts[stratum] != expected_strata[stratum] for stratum in GS.SAMPLE_STRATA
        ):
            raise ValueError(
                f"Model {model!r} sample-stratum counts do not match benchmark_provenance."
            )

    page_sets = {model: {_canonical_id(row["page_id"]) for row in model_rows} for model, model_rows in by_model.items()}
    explicit_expected = None
    if expected_page_ids is not None:
        expected_list = list(expected_page_ids)
        explicit_expected = {_canonical_id(page_id) for page_id in expected_list}
        if None in explicit_expected:
            raise ValueError("Expected page set contains a missing page ID.")
        if len(explicit_expected) != len(expected_list):
            raise ValueError("Expected page set contains duplicate page IDs after canonicalization.")
        if len(explicit_expected) != expected_n or page_set_fingerprint(explicit_expected) != full_fingerprint:
            raise ValueError("Supplied expected page IDs do not match authoritative benchmark_provenance (count or full-page-set fingerprint differs).")

    mismatched = {}
    for model, pages in page_sets.items():
        if len(pages) != expected_n or page_set_fingerprint(pages) != full_fingerprint:
            if explicit_expected is None:
                missing, extra = max(0, expected_n - len(pages)), max(0, len(pages) - expected_n)
            else:
                missing, extra = len(explicit_expected - pages), len(pages - explicit_expected)
            mismatched[model] = missing, extra
    unequal = len({frozenset(pages) for pages in page_sets.values()}) > 1
    if require_complete and (mismatched or unequal):
        details = "; ".join(
            f"{model}: missing {counts[0]}, extra {counts[1]}, page-set fingerprint mismatch"
            for model, counts in sorted(mismatched.items())
        )
        raise ValueError("Models must match the authoritative complete benchmark page set. " + details)
    return by_model, page_sets, explicit_expected, provenance_values, run_values, benchmark


def report(rows, expected_page_ids=None, *, require_complete=True, include_policy_diagnostics=False,
           truncated_by_model=None):
    """Validate rows and return eligible ordering plus conditional diagnostics.

    `truncated_by_model` maps model id -> count of rows whose completion hit the token cap.
    From scorecard schema 2.1 it is redundant — the flag travels on each row, so ANY caller
    rebuilding a board from stored scorecards (leaderboard.py, not just score_dataset.py in a
    single pass) gets the same eligibility. Kept only for callers holding the tally out of
    band; pre-2.1 scorecards cannot reach here at all, because _validated_rows rejects a
    mismatched scorecard_schema_version outright.
    """
    by_model, page_sets, expected, values, run_values, benchmark = _validated_rows(
        rows, expected_page_ids=expected_page_ids, require_complete=require_complete,
    )
    expected_n = benchmark["expected_page_count"]
    full_fingerprint = benchmark["full_page_set_fingerprint"]
    cards = {}
    for model, model_rows in by_model.items():
        pages = page_sets[model]
        page_set_match = len(pages) == expected_n and page_set_fingerprint(pages) == full_fingerprint
        if expected is None:
            missing_n, extra_n = max(0, expected_n - len(pages)), max(0, len(pages) - expected_n)
        else:
            missing_n, extra_n = len(expected - pages), len(pages - expected)
        cards[model] = model_scorecard(
            model_rows,
            truncated_n=(truncated_by_model or {}).get(model, 0),
            expected_n=expected_n,
            missing_n=missing_n,
            extra_n=extra_n,
            page_set_match=page_set_match,
            include_policy_diagnostics=include_policy_diagnostics,
            expected_strata=benchmark["expected_stratum_counts"],
        )

    def sort_key(model):
        value = cards[model]["cer_reading_micro"]
        return value is None, value if value is not None else 0, model

    eligible_order = sorted((model for model, card in cards.items() if card["eligible"]), key=sort_key)
    ineligible_order = sorted((model for model, card in cards.items() if not card["eligible"]), key=sort_key)

    def one(key):
        return next(iter(values[key].values()))

    provenance = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "scorecard_schema_version": one("scorecard_schema_version"),
        "norm_version": one("norm_version"),
        "scorer_version": one("scorer_version"),
        "postproc_version": one("postproc_version"),
        "score": one("score_provenance"),
        "benchmark": benchmark,
        "page_set_fingerprint": full_fingerprint,
        "page_count": expected_n,
        "run_by_model": {model: next(iter(run_values[model].values())) for model in by_model},
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "provenance": provenance,
        "scorecards": cards,
        "order": eligible_order,
        "eligible_order": eligible_order,
        "ineligible_order": ineligible_order,
    }


def format_report(result):
    prov = result["provenance"]
    cards = result["scorecards"]
    lines = [
        "=" * 116,
        "BHL OCR SCORECARD — ordered point estimates; headline = micro-averaged CER (reading lane), lower better",
        f"provenance: norm={prov['norm_version']} · scorer={prov['scorer_version']} · "
        f"{prov['score']['grapheme_mode']} · pages={prov['page_count']} · page-set={prov['page_set_fingerprint'][:12]}",
        "=" * 116,
        f"{'model':30} {'CERdip':>7} {'CERread':>7} {'CER95%CI':>16} {'recallµ':>8} "
        f"{'overµ':>7} {'furnµ':>7} {'n':>5} {'err':>4} {'loop%':>6} {'miss':>4} {'extra':>5} "
        f"{'eligible':>8}",
    ]

    def append_card(model):
        card = cards[model]
        ci = card["cer_ci"]
        ci_string = f"[{ci[0]}, {ci[1]}]" if ci[0] is not None else "—"
        rate = card.get("truncation_rate")
        loop_pct = f"{100 * rate:.2f}" if rate is not None else "—"
        lines.append(
            f"{model[:30]:30} {card['cer_diplomatic_micro']!s:>7} {card['cer_reading_micro']!s:>7} "
            f"{ci_string:>16} {card['recall_micro']!s:>8} {card['over_extraction_micro']!s:>7} "
            f"{card['furniture_global_token_recall_micro']!s:>7} {card['n']:>5} "
            f"{card['producer_errors']:>4} {loop_pct:>6} "
            f"{card['missing']:>4} {card['extra']:>5} {str(card['eligible']):>8}"
        )

    for model in result["eligible_order"]:
        append_card(model)
    if result["ineligible_order"]:
        lines.append("-- CONDITIONAL DIAGNOSTICS (INELIGIBLE; NOT PART OF COMPARISON ORDER) --")
        for model in result["ineligible_order"]:
            append_card(model)
    displayed_models = result["eligible_order"] + result["ineligible_order"]
    for model in displayed_models:
        lines.append(f"\nper-volume micro CER ({model}): {cards[model]['by_volume']}")
        lines.append(f"per-stratum profile ({model}): {cards[model]['by_sample_stratum']}")
        policy = cards[model].get("policy_diagnostics")
        if policy is not None:
            full = policy["full_text_cer"]
            lines.append(
                f"full-text policy CER ({model}): diplomatic={full['cer_diplomatic_micro']} · "
                f"reading={full['cer_reading_micro']}"
            )
    if result["eligible_order"]:
        first = result["eligible_order"][0]
        lines.append(f"per-language micro CER ({first}): {cards[first]['by_language']}")
        lines.append(f"region global token evidence ({first}): {cards[first]['region_global_token_recall']}")
    lines.append("Flagged-error metrics are conditional on successful pages. A PRODUCER error or a missing page\n"
        "makes a model ineligible; a truncation (loop%) excludes the page but not the model — and a\n"
        "higher loop% means the row was scored on an easier effective page set.")
    return "\n".join(lines)


if __name__ == "__main__":
    pages = [("HEADER alpha beta", "alpha beta", "HEADER", "content", "a"),
             ("42", "", "42", "sparse_blank", "b")]
    rows = []
    for page_id, (full, body, furniture, stratum, volume) in enumerate(pages):
        row = S.score_page(full, body, body_text=body, furniture_text=furniture, page_id=page_id)
        row.update(
            model="reader", volume=volume, error=False, sample_stratum=stratum,
            sample_stratum_threshold=80, scorecard_schema_version=GS.SCORECARD_SCHEMA_VERSION,
            score_provenance=N.provenance(), run_provenance={"runner": "report-self-test"},
        )
        rows.append(row)
    benchmark = {
        "dataset_id": "synthetic", "requested_revision": None,
        "resolved_revision": "local:embedded-synthetic-v1", "dataset_fingerprint": "v1",
        "expected_page_count": len(rows),
        "full_page_set_fingerprint": page_set_fingerprint([row["page_id"] for row in rows]),
        "sample_stratum_threshold": 80,
        "expected_stratum_counts": {"content": 1, "sparse_blank": 1},
    }
    for row in rows:
        row["benchmark_provenance"] = benchmark
    print(format_report(report(rows, include_policy_diagnostics=True)))
