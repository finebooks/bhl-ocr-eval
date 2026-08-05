"""Integration-light guards for leaderboard completeness policy wiring."""
import pandas as pd
import pytest

import leaderboard as LB


@pytest.mark.parametrize(("argv", "expected"), [(["score.parquet"], True),
                                                  (["--allow-incomplete", "score.parquet"], False)])
def test_leaderboard_wires_allow_incomplete_to_report(monkeypatch, argv, expected):
    seen = {}

    class StopAfterReport(Exception):
        pass

    monkeypatch.setattr(LB.glob, "glob", lambda _pattern: ["score.parquet"])
    monkeypatch.setattr(LB.pd, "read_parquet", lambda _path: pd.DataFrame(
        {"model": ["m"], "page_id": [1]}))

    def report(_rows, *, require_complete, include_policy_diagnostics):
        seen["require_complete"] = require_complete
        seen["include_policy_diagnostics"] = include_policy_diagnostics
        raise StopAfterReport

    monkeypatch.setattr(LB.RPT, "report", report)
    with pytest.raises(StopAfterReport):
        LB.main(argv)
    assert seen == {"require_complete": expected, "include_policy_diagnostics": False}


def test_leaderboard_wires_policy_diagnostics(monkeypatch):
    seen = {}

    class StopAfterReport(Exception):
        pass

    monkeypatch.setattr(LB.glob, "glob", lambda _pattern: ["score.parquet"])
    monkeypatch.setattr(LB.pd, "read_parquet", lambda _path: pd.DataFrame(
        {"model": ["m"], "page_id": [1]}))

    def report(_rows, **kwargs):
        seen.update(kwargs)
        raise StopAfterReport

    monkeypatch.setattr(LB.RPT, "report", report)
    with pytest.raises(StopAfterReport):
        LB.main(["--policy-diagnostics", "score.parquet"])
    assert seen["include_policy_diagnostics"] is True
