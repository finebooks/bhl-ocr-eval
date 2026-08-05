"""Deterministic guards for the load-bearing report mechanics (not the stochastic CI/bootstrap)."""
import json

import pandas as pd
import pytest

import gt_score as GS
import normalizers as N
import report as R
import scorer as S


def _stamp(rows, benchmark_page_ids=None):
    """Add the mandatory scorecard identity/provenance envelope to synthetic rows."""
    counters = {}
    for row in rows:
        model = row.setdefault("model", "model")
        counters[model] = counters.get(model, 0) + 1
        row.setdefault("page_id", counters[model])
        row.setdefault("error", False)
        row.setdefault("volume", "test-volume")
        row.setdefault("sample_stratum", "content")
        row.setdefault("sample_stratum_threshold", 80)
        for column in R.S.COUNT_COLUMNS:
            row.setdefault(column, 1 if column.endswith("_h") else 0)
        recall = row.get("recall", 0.0)
        row.setdefault("recall_matched", int(round(recall * 100)))
        row.setdefault("recall_reference", 100)
        row["recall"] = round(row["recall_matched"] / row["recall_reference"], 4)
        row.setdefault("over_extraction_extra", 0)
        row.setdefault("over_extraction_output", 1)
        row.setdefault("over_extraction", 0.0)
        row.setdefault("furniture_global_token_recall_matched", 0)
        row.setdefault("furniture_global_token_recall_reference", 0)
        row.setdefault("furniture_global_token_recall", None)
        for axis in R.S.AXIS_NAMES:
            row.setdefault(axis, {} if axis == "region_global_token_recall" else 0.0)
        for axis, key in (("cer_diplomatic", "cer_body_dip"), ("cer_reading", "cer_body_read")):
            counts = [row[f"{key}_{part}"] for part in ("s", "d", "i", "h")]
            if all(isinstance(value, int) for value in counts):
                s, d, i, h = counts
                row[axis] = round((s + d + i) / (s + d + h), 4) if s + d + h else None
        row.setdefault("norm_version", N.NORM_VERSION)
        row.setdefault("scorer_version", S.SCORER_VERSION)
        row.setdefault("scorecard_schema_version", GS.SCORECARD_SCHEMA_VERSION)
        row.setdefault("postproc_version", "0")
        row.setdefault("score_provenance", N.provenance())
        row.setdefault("run_provenance", {"producer": row["model"]})

    page_ids = benchmark_page_ids or sorted({row["page_id"] for row in rows})
    stratum_by_page = {}
    for row in rows:
        stratum_by_page.setdefault(GS.canonical_page_id(row["page_id"]), row["sample_stratum"])
    counts = {stratum: 0 for stratum in GS.SAMPLE_STRATA}
    for page_id in page_ids:
        canonical_id = GS.canonical_page_id(page_id)
        assert canonical_id is not None
        stratum = stratum_by_page.get(canonical_id, "content")
        assert stratum in counts
        counts[stratum] += 1
    benchmark = {
        "dataset_id": "test/gt", "requested_revision": None,
        "resolved_revision": "local:content-1", "dataset_fingerprint": "content-1",
        "expected_page_count": len(page_ids),
        "full_page_set_fingerprint": R.page_set_fingerprint(page_ids),
        "sample_stratum_threshold": 80,
        "expected_stratum_counts": counts,
    }
    for row in rows:
        row.setdefault("benchmark_provenance", benchmark)
    return rows


def test_micro_is_length_weighted_not_macro():
    # page A: 1 err / 100 chars (rate .01); page B: 5 err / 10 chars (rate .5)
    rows = [
        {"cer_body_read_s": 1, "cer_body_read_d": 0, "cer_body_read_i": 0, "cer_body_read_h": 99},
        {"cer_body_read_s": 5, "cer_body_read_d": 0, "cer_body_read_i": 0, "cer_body_read_h": 5},
    ]
    micro = R.micro_rate(rows, "cer_body_read")
    assert abs(micro - 6 / 110) < 1e-3            # pooled, length-weighted (micro_rate rounds to 4dp)
    assert abs(micro - (0.01 + 0.5) / 2) > 0.1    # and genuinely different from the macro mean


def test_error_rows_excluded_from_aggregates():
    ok = S.score_page("word " * 10, "word " * 10)
    ok.update(model="m", error=False, volume="v", sample_stratum="content")
    bad = S.score_page("word " * 10, "")
    bad.update(model="m", error=True, volume="v", sample_stratum="content")
    card = R.model_scorecard([ok, bad])
    assert card["n"] == 1 and card["errors"] == 1
    assert card["recall_micro"] == 1.0             # error row did not pollute pooled supports


def test_nan_error_flag_is_not_treated_as_error():
    ok = S.score_page("word " * 10, "word " * 10)
    ok.update(model="m", error=False, volume="v", sample_stratum="content")
    missing = S.score_page("word " * 10, "word " * 9)
    missing.update(model="m", error=float("nan"), volume="v", sample_stratum="content")
    card = R.model_scorecard([ok, missing])
    assert card["n"] == 2 and card["errors"] == 0
    assert card["recall_micro"] == 0.95


def test_bootstrap_resamples_volumes_not_pages():
    # two volumes with very different rates: cluster resampling draws whole volumes, so the CI must
    # reach the all-a (0.0) and all-b (0.5) extremes; a page-level bootstrap on 200 pages would hug
    # the pooled mean and never get near either.
    rows = ([{"volume": "a", "cer_body_read_s": 0, "cer_body_read_d": 0, "cer_body_read_i": 0,
              "cer_body_read_h": 100}] * 100
            + [{"volume": "b", "cer_body_read_s": 50, "cer_body_read_d": 0, "cer_body_read_i": 0,
                "cer_body_read_h": 50}] * 100)
    lo, hi = R.bootstrap_ci(rows, lambda rs: R.micro_rate(rs, "cer_body_read"))
    assert lo == 0.0 and hi == 0.5


def test_micro_rate_skips_nan_counts():
    rows = [{"cer_body_read_s": 1, "cer_body_read_d": 0, "cer_body_read_i": 0, "cer_body_read_h": 99},
            {"cer_body_read_s": float("nan"), "cer_body_read_d": float("nan"),
             "cer_body_read_i": float("nan"), "cer_body_read_h": float("nan")}]
    assert R.micro_rate(rows, "cer_body_read") == 0.01  # NaN row skipped, not poisoning the pool


def test_micro_rate_pools_insertions_from_all_wildcard_page():
    # The page rate is undefined at zero reference support, but its valid insertion support remains
    # part of a benchmark-level pool that has a nonzero aggregate denominator.
    rows = [{"cer_body_read_s": 1, "cer_body_read_d": 0, "cer_body_read_i": 0, "cer_body_read_h": 99},
            {"cer_body_read_s": 0, "cer_body_read_d": 0, "cer_body_read_i": 5, "cer_body_read_h": 0}]
    assert R.micro_rate(rows, "cer_body_read") == 0.06
    assert R.micro_rate(rows[1:], "cer_body_read") is None


def test_mean_median_exclude_nan():
    # parquet turns per-page None into NaN; it must not poison the aggregate (regression guard)
    rows = [{"x": 0.5}, {"x": float("nan")}, {"x": 0.7}, {"x": None}]
    assert R.mean(rows, "x") == round((0.5 + 0.7) / 2, 4)
    assert R.median(rows, "x") == round((0.5 + 0.7) / 2, 4)


def test_model_scorecard_micro_pools_token_supports_not_page_rates():
    first = S.score_page("a " * 100, "a " * 90, furniture_text="head")
    second = S.score_page("b " * 10, "b " * 5, furniture_text="")
    for row, language in ((first, "en"), (second, "la")):
        row.update(error=False, language=language, volume=language, sample_stratum="content")
    card = R.model_scorecard([first, second])
    assert card["recall_micro"] == round(95 / 110, 4)
    assert card["recall_matched"] == 95 and card["recall_reference"] == 110
    assert card["recall_micro"] != round((0.9 + 0.5) / 2, 4)
    assert "recall" not in card and "over_extraction" not in card
    assert set(card["by_language"]) == {"en", "la"}


def test_report_orders_point_estimates_without_tiers():
    def rows_for(model, recall):
        return [{"model": model, "volume": v, "recall": recall, "error": False,
                 "cer_body_read_s": 1, "cer_body_read_d": 0, "cer_body_read_i": 0, "cer_body_read_h": 99}
                for v in ("a", "b", "c")]
    rows = rows_for("hi_recall", 0.99) + rows_for("lo_recall", 0.10)
    result = R.report(_stamp(rows))
    assert result["scorecards"]["hi_recall"]["cer_reading_micro"] == \
        result["scorecards"]["lo_recall"]["cer_reading_micro"]
    assert set(result["order"]) == {"hi_recall", "lo_recall"}
    assert "tiers" not in result


def test_format_report_headline_is_cer_lower_better():
    rows = [
        {"model": "worse", "error": False,
         "cer_body_read_s": 5, "cer_body_read_d": 0, "cer_body_read_i": 0, "cer_body_read_h": 5},
        {"model": "better", "error": False,
         "cer_body_read_s": 0, "cer_body_read_d": 0, "cer_body_read_i": 0, "cer_body_read_h": 10},
    ]
    rendered = R.format_report(R.report(_stamp(rows)))
    assert "lower better" in rendered
    assert rendered.index("better") < rendered.index("worse")  # sorted by CER ascending


def test_report_rejects_duplicate_model_page_rows():
    rows = _stamp([{"model": "m"}, {"model": "m"}])
    rows[1]["page_id"] = rows[0]["page_id"]
    with pytest.raises(ValueError, match="Duplicate scorecard row"):
        R.report(rows)


def test_report_rejects_unequal_page_sets():
    rows = _stamp([{"model": "a"}, {"model": "a"}, {"model": "b"}])
    with pytest.raises(ValueError, match="authoritative complete benchmark page set"):
        R.report(rows)


def test_report_rejects_old_or_unknown_versions_without_compatibility_shim():
    rows = _stamp([{"model": "a"}])
    rows[0]["scorecard_schema_version"] = "1.0"
    with pytest.raises(ValueError, match="Unsupported scorecard_schema_version"):
        R.report(rows)


def test_report_rejects_mixed_postproc_versions():
    # One model normalized under registry v2, another under v3 is not one board: the
    # post-processing rules differ, so the scores are not comparable. This is the exact
    # state a partially re-normalized run directory produces.
    rows = _stamp([{"model": "a"}, {"model": "b"}])
    rows[0]["postproc_version"] = "2"
    rows[1]["postproc_version"] = "3"
    with pytest.raises(ValueError, match="Mixed scorecard provenance.*postproc_version"):
        R.report(rows)


def test_report_surfaces_postproc_version_in_provenance():
    result = R.report(_stamp([{"model": "a"}]))
    assert result["provenance"]["postproc_version"] == "0"


def test_error_model_has_conditional_metrics_but_is_ineligible():
    rows = _stamp([{"model": "m", "error": False, "cer_body_read_s": 0,
                    "cer_body_read_d": 0, "cer_body_read_i": 0, "cer_body_read_h": 10},
                   {"model": "m", "error": True, "cer_body_read_s": 10,
                    "cer_body_read_d": 0, "cer_body_read_i": 0, "cer_body_read_h": 0}])
    card = R.report(rows)["scorecards"]["m"]
    assert card["cer_reading_micro"] == 0.0
    assert card["errors"] == 1
    assert card["eligible"] is False


def test_report_rejects_missing_and_mixed_run_provenance():
    missing = _stamp([{"model": "m"}])
    del missing[0]["run_provenance"]
    with pytest.raises(ValueError, match="missing required provenance field 'run_provenance'"):
        R.report(missing)

    mixed = _stamp([{"model": "m"}, {"model": "m"}])
    mixed[1]["run_provenance"] = {"producer": "another-run"}
    with pytest.raises(ValueError, match="Frankenstein run"):
        R.report(mixed)


def test_report_preserves_current_string_versions_after_json_field_roundtrip():
    rows = _stamp([{"model": "m"}])
    rows[0]["score_provenance"] = json.dumps(rows[0]["score_provenance"])
    rows[0]["benchmark_provenance"] = json.dumps(rows[0]["benchmark_provenance"])
    rows[0]["run_provenance"] = json.dumps(rows[0]["run_provenance"])
    provenance = R.report(rows)["provenance"]
    assert provenance["norm_version"] == N.NORM_VERSION
    assert provenance["scorer_version"] == S.SCORER_VERSION
    assert isinstance(provenance["norm_version"], str)
    assert provenance["report_schema_version"] == R.REPORT_SCHEMA_VERSION
    assert "report_version" not in provenance


def test_authoritative_benchmark_provenance_prevents_uniform_partial_laundering():
    rows = _stamp([{"model": "a", "page_id": 1}, {"model": "b", "page_id": 1}],
                  benchmark_page_ids=[1, 2])
    with pytest.raises(ValueError, match="authoritative complete benchmark page set"):
        R.report(rows)
    diagnostic = R.report(rows, require_complete=False)
    assert diagnostic["order"] == []
    assert diagnostic["ineligible_order"] == ["a", "b"]


def test_report_requires_resolved_benchmark_revision():
    rows = _stamp([{"model": "m", "page_id": 1}])
    del rows[0]["benchmark_provenance"]["resolved_revision"]
    with pytest.raises(ValueError, match="resolved_revision"):
        R.report(rows)


def test_report_rejects_claimed_full_set_fingerprint_mismatch():
    rows = _stamp([{"model": "m", "page_id": 1}])
    rows[0]["benchmark_provenance"] = {**rows[0]["benchmark_provenance"],
                                       "full_page_set_fingerprint": "not-the-actual-set"}
    with pytest.raises(ValueError, match="page-set fingerprint mismatch"):
        R.report(rows)


def test_incomplete_conditional_report_marks_missing_and_ineligible():
    rows = _stamp([{"model": "m", "page_id": 1, "cer_body_read_s": 0,
                    "cer_body_read_d": 0, "cer_body_read_i": 0, "cer_body_read_h": 10}],
                  benchmark_page_ids=[1, 2])
    result = R.report(rows, expected_page_ids=[1, 2], require_complete=False)
    assert result["scorecards"]["m"]["missing"] == 1
    assert result["scorecards"]["m"]["eligible"] is False
    assert result["order"] == []
    assert result["ineligible_order"] == ["m"]


@pytest.mark.parametrize("field", [
    "error", "volume", "sample_stratum", "sample_stratum_threshold",
    *R.S.COUNT_COLUMNS, *R.S.SUPPORT_COLUMNS, *R.S.AXIS_NAMES,
])
def test_report_rejects_truncated_payload(field):
    rows = _stamp([{"model": "m"}])
    del rows[0][field]
    with pytest.raises(ValueError, match="missing required payload fields"):
        R.report(rows)


@pytest.mark.parametrize("value", [None, 0, 1, "false", float("nan")])
def test_report_requires_boolean_error_after_parquet_roundtrip(value):
    rows = _stamp([{"model": "m"}])
    rows[0]["error"] = value
    with pytest.raises(ValueError, match="'error' must be boolean"):
        R.report(rows)


@pytest.mark.parametrize("value", [None, "", "   ", 123, float("nan")])
def test_report_requires_nonempty_string_volume(value):
    rows = _stamp([{"model": "m"}])
    rows[0]["volume"] = value
    with pytest.raises(ValueError, match="'volume' must be a non-empty string"):
        R.report(rows)


@pytest.mark.parametrize("values", [
    (0, None, 0, 1), (-1, 0, 0, 1), (0.5, 0, 0, 1), (0, 0, float("inf"), 1),
])
def test_report_rejects_malformed_count_groups(values):
    rows = _stamp([{"model": "m"}])
    for suffix, value in zip(("s", "d", "i", "h"), values, strict=True):
        rows[0][f"cer_body_read_{suffix}"] = value
    with pytest.raises(ValueError, match="finite nonnegative integers"):
        R.report(rows)


def test_report_rejects_all_missing_count_groups_under_schema_2():
    rows = _stamp([{"model": "m"}])
    for column in R.S.COUNT_COLUMNS:
        rows[0][column] = float("nan")
    with pytest.raises(ValueError, match="four finite nonnegative integers"):
        R.report(rows)


def test_report_rejects_missing_full_text_count_group_too():
    rows = _stamp([{"model": "m"}])
    for part in ("s", "d", "i", "h"):
        rows[0][f"cer_full_read_{part}"] = None
    with pytest.raises(ValueError, match="cer_full_read.*four finite nonnegative integers"):
        R.report(rows)


@pytest.mark.parametrize("value", ["not-json", "[]", {"caption": "bad"}, {"": {}}, None])
def test_report_rejects_invalid_region_global_token_recall(value):
    rows = _stamp([{"model": "m"}])
    rows[0]["region_global_token_recall"] = value
    with pytest.raises(ValueError, match="region_global_token_recall"):
        R.report(rows)


@pytest.mark.parametrize(("field", "value"), [
    ("recall_matched", -1),
    ("recall_matched", 0.5),
    ("recall_matched", 101),
    ("over_extraction_extra", 2),
])
def test_report_rejects_malformed_token_supports(field, value):
    rows = _stamp([{"model": "m"}])
    rows[0][field] = value
    with pytest.raises(ValueError, match="support group"):
        R.report(rows)


def test_report_emits_per_stratum_micro_profile():
    rows = _stamp([
        {"model": "m", "sample_stratum": "content", "recall_matched": 9, "recall_reference": 10},
        {"model": "m", "sample_stratum": "sparse_blank", "recall_matched": 1, "recall_reference": 2,
         "over_extraction_extra": 1, "over_extraction_output": 2, "over_extraction": 0.5},
    ])
    strata = R.report(rows)["scorecards"]["m"]["by_sample_stratum"]
    assert strata["content"]["submitted"] == strata["content"]["n"] == 1
    assert strata["content"]["errors"] == 0 and strata["content"]["recall_micro"] == 0.9
    assert strata["sparse_blank"]["n"] == 1
    assert strata["sparse_blank"]["recall_micro"] == 0.5
    assert strata["sparse_blank"]["over_extraction_micro"] == 0.5
    assert "cer_reading_micro" in strata["sparse_blank"]


def test_policy_full_text_cer_is_opt_in_and_nested():
    rows = _stamp([{"model": "m"}])
    default = R.report(rows)["scorecards"]["m"]
    assert "policy_diagnostics" not in default
    opted_in = R.report(rows, include_policy_diagnostics=True)["scorecards"]["m"]
    assert opted_in["policy_diagnostics"]["full_text_cer"] == {
        "cer_diplomatic_micro": 0.0,
        "cer_reading_micro": 0.0,
    }
    rendered = R.format_report(R.report(rows, include_policy_diagnostics=True))
    assert "full-text policy CER (m): diplomatic=0.0 · reading=0.0" in rendered


def test_json_parquet_roundtrip_retains_structured_supports(tmp_path):
    rows = _stamp([{
        "model": "m",
        "region_global_token_recall": {
            "caption": {"rate": 0.5, "matched": 1, "reference": 2},
        },
    }])
    flat = [{key: json.dumps(value) if isinstance(value, dict) else value for key, value in row.items()}
            for row in rows]
    path = tmp_path / "score.parquet"
    pd.DataFrame(flat).to_parquet(path, index=False)
    result = R.report(pd.read_parquet(path).to_dict("records"))
    assert result["scorecards"]["m"]["region_global_token_recall"]["caption"] == {
        "rate": 0.5, "matched": 1, "reference": 2,
    }


@pytest.mark.parametrize("score_provenance", [
    {},
    {"grapheme_mode": "codepoint"},
    {"norm_version": "test-norm", "jiwer_version": "4.0", "lanes": {},
     "grapheme_mode": "codepoint", "word_boundary": "whitespace", "illegible_marker": "U+FFFD"},
])
def test_report_rejects_empty_or_incomplete_score_provenance(score_provenance):
    rows = _stamp([{"model": "m"}])
    rows[0]["score_provenance"] = score_provenance
    with pytest.raises(ValueError, match="score_provenance"):
        R.report(rows)


@pytest.mark.parametrize(("field", "value"), [
    ("script_match", -0.1),
    ("script_match", 1.1),
    ("recall", 1.1),
])
def test_report_rejects_fraction_axes_outside_unit_interval(field, value):
    rows = _stamp([{"model": "m"}])
    rows[0][field] = value
    with pytest.raises(ValueError, match="fraction axis|does not equal its supports"):
        R.report(rows)


@pytest.mark.parametrize(("axis", "value"), [
    ("cer_diplomatic", 0.5),
    ("cer_reading", None),
])
def test_report_rejects_per_page_cer_inconsistent_with_body_counts(axis, value):
    rows = _stamp([{"model": "m"}])
    rows[0][axis] = value
    with pytest.raises(ValueError, match="does not equal its body edit-count rate"):
        R.report(rows)


def test_report_requires_none_per_page_cer_at_zero_denominator():
    rows = _stamp([{"model": "m"}])
    for key in ("cer_body_dip", "cer_body_read"):
        for part in ("s", "d", "h"):
            rows[0][f"{key}_{part}"] = 0
        rows[0][f"{key}_i"] = 3
    rows[0]["cer_diplomatic"] = 0.0
    rows[0]["cer_reading"] = 0.0
    with pytest.raises(ValueError, match="body edit-count rate"):
        R.report(rows)


def test_report_rejects_row_threshold_that_differs_from_benchmark():
    rows = _stamp([{"model": "m"}])
    rows[0]["sample_stratum_threshold"] = 79
    with pytest.raises(ValueError, match="does not match benchmark_provenance"):
        R.report(rows)


def test_report_requires_sampler_seed_to_be_an_integer_type():
    rows = _stamp([{"model": "m"}])
    rows[0]["benchmark_provenance"] = {
        **rows[0]["benchmark_provenance"],
        "sampler_provenance": {
            "sampler_version": "1.0",
            "sampler_seed": 7.0,
            "sampler_requested_n": 1,
            "sampler_source_repo": "source/repo",
            "sampler_source_revision": "a" * 40,
            "sample_stratum_threshold": 80,
        },
    }
    with pytest.raises(ValueError, match="sampler_seed must be an integer"):
        R.report(rows)


def test_report_rejects_complete_model_stratum_counts_that_differ_from_benchmark():
    rows = _stamp([
        {"model": "m", "page_id": 1, "sample_stratum": "content"},
        {"model": "m", "page_id": 2, "sample_stratum": "content"},
    ])
    benchmark = {
        **rows[0]["benchmark_provenance"],
        "expected_stratum_counts": {"content": 1, "sparse_blank": 1},
    }
    for row in rows:
        row["benchmark_provenance"] = benchmark
    with pytest.raises(ValueError, match="counts (exceed|do not match) benchmark_provenance"):
        R.report(rows)


def test_report_rejects_conflicting_page_stratum_across_models():
    rows = _stamp([
        {"model": "a", "page_id": 1, "sample_stratum": "content"},
        {"model": "b", "page_id": 1, "sample_stratum": "sparse_blank"},
    ])
    with pytest.raises(ValueError, match="Conflicting sample_stratum"):
        R.report(rows)


def test_per_stratum_diagnostics_preserve_all_error_stratum():
    rows = _stamp([
        {"model": "m", "page_id": 1, "sample_stratum": "content", "error": False},
        {"model": "m", "page_id": 2, "sample_stratum": "sparse_blank", "error": True},
    ])
    card = R.report(rows)["scorecards"]["m"]
    sparse = card["by_sample_stratum"]["sparse_blank"]
    assert sparse["submitted"] == 1 and sparse["n"] == 0 and sparse["errors"] == 1
    assert sparse["cer_reading_micro"] is None and sparse["recall_micro"] is None
    assert sparse["over_extraction_micro"] == 0.0
