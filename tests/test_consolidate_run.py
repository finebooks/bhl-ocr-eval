"""Consolidating a saturate run into a scoreable table.

Every assertion here guards a failure that costs GPU time to discover: the run finishes, and the
problem only surfaces at scoring. They are behaviour pins, not coverage.
"""
import pandas as pd
import pytest

import consolidate_run as CR


def _frame(**overrides):
    base = {
        "id": ["1", "2", "3"],
        "raw": ["page one", "page two", "page three"],
        "error": [None, None, None],
        "finish_reason": ["stop", "stop", "stop"],
        "model": ["m", "m", "m"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _consolidate(frame, **kw):
    kw.setdefault("id_col", "id")
    kw.setdefault("key_col", "PageID")
    kw.setdefault("raw_col", "raw")
    kw.setdefault("error_col", "error")
    kw.setdefault("finish_col", "finish_reason")
    return CR.consolidate(frame, **kw)


def test_renames_saturates_id_to_the_scoring_key():
    out, counts = _consolidate(_frame())
    assert "PageID" in out.columns and "id" not in out.columns
    assert list(out["PageID"]) == ["1", "2", "3"]
    assert counts["ok"] == 3 and counts["truncated"] == 0 and counts["producer_errors"] == 0


def test_null_raw_is_sealed_rather_than_left_to_kill_normalize():
    # saturate writes failures as {id, error} with no `raw` key at all -> parquet null.
    out, counts = _consolidate(_frame(raw=["ok", None, "ok"], error=[None, "boom", None]))
    assert out["raw"][1].startswith(CR.ERROR_PREFIX)
    assert "boom" in out["raw"][1]
    assert counts["producer_errors"] == 1 and counts["ok"] == 2


def test_null_raw_with_no_recorded_error_still_seals():
    out, counts = _consolidate(_frame(raw=["ok", None, "ok"]))
    assert out["raw"][1].startswith(CR.ERROR_PREFIX)
    assert counts["producer_errors"] == 1


def test_truncated_rows_are_sealed_and_counted_separately_from_errors():
    out, counts = _consolidate(_frame(finish_reason=["stop", "length", "stop"]))
    assert out["raw"][1] == f"{CR.ERROR_PREFIX}FinishReason:length"
    assert counts["truncated"] == 1
    assert counts["producer_errors"] == 0  # a model failure, NOT a run defect
    assert counts["truncation_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_truncated_and_errored_use_distinguishable_sentinels():
    # The board needs to tell "the run broke" from "the model looped into its token cap".
    out, _ = _consolidate(_frame(raw=["ok", None, "ok"],
                                 error=[None, "transport", None],
                                 finish_reason=["length", "stop", "stop"]))
    assert out["raw"][0].startswith(f"{CR.ERROR_PREFIX}FinishReason:")
    assert out["raw"][1].startswith(f"{CR.ERROR_PREFIX}Producer:")
    assert out["raw"][0] != out["raw"][1]


def test_a_producer_error_is_not_also_counted_as_truncated():
    # An error row has no meaningful finish_reason; counting it twice would inflate the board column.
    out, counts = _consolidate(_frame(raw=["ok", None, "ok"],
                                      error=[None, "boom", None],
                                      finish_reason=["stop", "length", "stop"]))
    assert counts["producer_errors"] == 1 and counts["truncated"] == 0
    assert out["raw"][1].startswith(f"{CR.ERROR_PREFIX}Producer:")


def test_the_pre_sealing_completion_is_kept():
    # Sealing is a scoring decision; the raw completion must stay inspectable, or a
    # truncated page can never be read back from the consolidated artifact.
    out, _ = _consolidate(_frame(raw=["ok", "looped text", None],
                                 error=[None, None, "boom"],
                                 finish_reason=["stop", "length", "stop"]))
    assert list(out["raw_verbatim"])[:2] == ["ok", "looped text"]
    assert out["raw"][1].startswith(CR.ERROR_PREFIX)      # scored column is sealed
    assert out["raw_verbatim"][1] == "looped text"        # original survives alongside
    assert pd.isna(out["raw_verbatim"][2])                # a genuinely absent raw stays absent


def test_rows_are_never_dropped():
    # A vanished page reads as "never run" at the completeness gate, not as "failed".
    out, counts = _consolidate(_frame(raw=[None, None, None], error=["a", "b", "c"]))
    assert len(out) == 3 and counts["rows"] == 3


def test_duplicate_ids_refuse_to_consolidate():
    # Duplicated SUCCESSFUL rows are never legitimate — see the retry tests below for the
    # one duplicate shape that is (an error row superseded by its retry).
    with pytest.raises(SystemExit, match="MORE THAN ONE successful row"):
        _consolidate(_frame(id=["1", "1", "2"]))


def test_missing_id_column_is_a_hard_error():
    with pytest.raises(SystemExit, match="no 'id' column"):
        _consolidate(_frame().rename(columns={"id": "row_key"}))


def test_refuses_to_guess_when_both_key_columns_exist():
    frame = _frame()
    frame["PageID"] = ["9", "9", "9"]
    with pytest.raises(SystemExit, match="refusing to guess"):
        _consolidate(frame)


def test_expect_rows_catches_an_incomplete_run():
    with pytest.raises(SystemExit, match="expected 2165"):
        _consolidate(_frame(), expect_rows=2165)


def test_finish_reason_histogram_is_reported():
    _, counts = _consolidate(_frame(finish_reason=["stop", "length", "length"]))
    assert counts["finish_reasons"] == {"stop": 1, "length": 2}


def test_sealed_rows_pass_through_normalize_untouched():
    # The other half of the contract: normalize must not run transforms over a sentinel.
    import normalize_outputs as NO

    raw = pd.DataFrame({
        "raw": [f"{CR.ERROR_PREFIX}FinishReason:length", "real text"],
        "model": ["zai-org/GLM-OCR", "zai-org/GLM-OCR"],
    })
    out = NO.normalize_frame(raw, raw_col="raw", out_col="markdown", model_col="model")
    assert out["markdown"][0] == f"{CR.ERROR_PREFIX}FinishReason:length"
    assert out["transforms_applied"][0] == []
    assert out["normalize_error"][0] is None


# ---------------------------------------------------------------------------
# Retries: a re-run appends rows and leaves the originals in place
# ---------------------------------------------------------------------------


def test_a_retried_success_supersedes_the_error_row_it_replaces():
    # `--retry-errors` writes a fresh row for the failed id; the original error row stays in
    # the part files. Both are legitimately present, and the successful one wins.
    frame = pd.DataFrame({
        "id": ["1", "2", "2"],
        "raw": ["page one", None, "recovered text"],
        "error": [None, "transport: TimeoutError", None],
        "finish_reason": ["stop", None, "stop"],
        "model": ["m", "m", "m"],
    })
    out, counts = _consolidate(frame)
    assert len(out) == 2
    assert counts["superseded_by_retry"] == 1
    assert counts["producer_errors"] == 0
    assert out.loc[out.PageID == "2", "raw"].iloc[0] == "recovered text"


def test_two_successful_rows_for_one_id_still_refuse():
    # Not a retry — two runs written into one prefix, which is the corruption the
    # new-prefix-on-config-change rule exists to prevent.
    frame = pd.DataFrame({
        "id": ["1", "1"],
        "raw": ["from run A", "from run B"],
        "error": [None, None],
        "finish_reason": ["stop", "stop"],
        "model": ["m", "m"],
    })
    with pytest.raises(SystemExit, match="MORE THAN ONE successful row"):
        _consolidate(frame)


def test_a_retry_that_also_failed_keeps_one_error_row():
    frame = pd.DataFrame({
        "id": ["1", "1"],
        "raw": [None, None],
        "error": ["first failure", "retry failed too"],
        "finish_reason": [None, None],
        "model": ["m", "m"],
    })
    out, counts = _consolidate(frame)
    assert len(out) == 1 and counts["producer_errors"] == 1
