"""Truncation is a third state: excluded from aggregates, but not disqualifying.

These pin a policy decision, not an implementation detail. Under the previous rule
(`errors == 0`) every neural model on the 2026-08 board would have been ineligible —
all eleven truncate, 0.32% to 6.47% — leaving only the classical engines rankable.
"""
import report as RPT


def _rows(n_clean=8, n_err=0, model="m"):
    """Minimal scorecard rows: only the fields model_scorecard actually reads."""
    def row(err):
        return {
            "error": err, "volume": "v1", "sample_stratum": "content",
            "cer_body_read_s": 0 if err else 100, "cer_body_read_d": 0,
            "cer_body_read_i": 0, "cer_body_read_h": 0,
            "cer_body_dip_s": 0 if err else 100, "cer_body_dip_d": 0,
            "cer_body_dip_i": 0, "cer_body_dip_h": 0,
            "recall_matched": 0 if err else 10, "recall_reference": 0 if err else 10,
            "over_extraction_extra": 0, "over_extraction_output": 0 if err else 10,
            "furniture_global_token_recall_matched": 0,
            "furniture_global_token_recall_reference": 0,
            "model": model,
        }
    return [row(False) for _ in range(n_clean)] + [row(True) for _ in range(n_err)]


def test_a_model_that_only_truncates_stays_eligible():
    card = RPT.model_scorecard(_rows(n_clean=97, n_err=3), ci_n=5,
                               expected_n=100, missing_n=0, truncated_n=3)
    assert card["truncated"] == 3
    assert card["producer_errors"] == 0
    assert card["eligible"] is True, "a repetition loop is model behaviour, not a broken run"


def test_a_producer_error_still_disqualifies():
    card = RPT.model_scorecard(_rows(n_clean=97, n_err=3), ci_n=5,
                               expected_n=100, missing_n=0, truncated_n=2)
    assert card["producer_errors"] == 1
    assert card["eligible"] is False, "a transport failure is still a broken run"


def test_truncated_pages_are_excluded_from_the_aggregates():
    # Same clean pages, different truncation counts -> identical CER. The metric must not
    # move, or a runaway page would be scored as a transcription.
    a = RPT.model_scorecard(_rows(n_clean=50, n_err=0), ci_n=5, expected_n=50, missing_n=0)
    b = RPT.model_scorecard(_rows(n_clean=50, n_err=10), ci_n=5, expected_n=60,
                            missing_n=0, truncated_n=10)
    assert a["cer_reading_micro"] == b["cer_reading_micro"]
    assert a["n"] == b["n"] == 50


def test_truncation_rate_is_reported_over_submitted_rows():
    card = RPT.model_scorecard(_rows(n_clean=90, n_err=10), ci_n=5,
                               expected_n=100, missing_n=0, truncated_n=10)
    assert card["truncation_rate"] == 0.1
    assert card["submitted"] == 100


def test_omitting_the_count_preserves_the_old_strict_behaviour():
    card = RPT.model_scorecard(_rows(n_clean=97, n_err=3), ci_n=5,
                               expected_n=100, missing_n=0)
    assert card["truncated"] == 0
    assert card["producer_errors"] == 3
    assert card["eligible"] is False


def test_truncated_count_cannot_exceed_error_rows():
    # Defensive: a miscounted upstream tally must not manufacture eligibility.
    card = RPT.model_scorecard(_rows(n_clean=99, n_err=1), ci_n=5,
                               expected_n=100, missing_n=0, truncated_n=99)
    assert card["truncated"] == 1
    assert card["producer_errors"] == 0


def test_missing_pages_still_disqualify_regardless_of_truncation():
    card = RPT.model_scorecard(_rows(n_clean=90, n_err=5), ci_n=5,
                               expected_n=100, missing_n=5, truncated_n=5)
    assert card["producer_errors"] == 0
    assert card["eligible"] is False, "an unsubmitted page is not the same as a looped one"
