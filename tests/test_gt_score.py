"""The shared runner helper must produce exactly the row the runners used to build inline."""
import json
from types import SimpleNamespace

import huggingface_hub

import pytest

import gt_score as GS
import normalizers as N
import scorer as S

GT = {
    "text": "BIRDS OF GREAT BRITAIN\nThe tree sparrow is common.\n94",
    "body_text": "The tree sparrow is common.",
    "furniture_text": "BIRDS OF GREAT BRITAIN 94",
    "regions_json": json.dumps([{"label": "text", "text": "The tree sparrow is common."}]),
    "language": "en",
    "volume": "birdsofgreatbrit02butl",
    "PageID": 7,
}


def _expected(gt, ocr, *, model, err, page_id, truncated=False):
    s = S.score_page(gt["text"], "" if err else ocr, body_text=gt["body_text"],
                     furniture_text=gt["furniture_text"], regions=json.loads(gt["regions_json"]),
                     language=gt["language"], page_id=page_id)
    s.update(model=model, volume=gt["volume"], error=err, truncated=truncated,
             sample_stratum=GS.sample_stratum(gt["text"]), sample_stratum_threshold=80,
             scorecard_schema_version=GS.SCORECARD_SCHEMA_VERSION,
             score_provenance=N.provenance())
    return s


def test_matches_inline_score_page_clean():
    got = GS.score_gt_row(GT, "The tree sparrow is common.", model="m", err=False, page_id=7)
    assert got == _expected(GT, "The tree sparrow is common.", model="m", err=False, page_id=7)


def test_error_row_scores_empty_and_flags():
    got = GS.score_gt_row(GT, "__ERR__Timeout", model="m", err=True, page_id=7)
    # err=True scores an empty OCR (a real miss) and flags the row for exclusion
    assert got["error"] is True and got["recall"] == 0.0
    assert got == _expected(GT, "ignored-because-err", model="m", err=True, page_id=7)


def test_page_id_defaults_to_row_pageid():
    got = GS.score_gt_row(GT, "The tree sparrow is common.", model="m", err=False)
    assert got["page_id"] == 7  # falls back to gt["PageID"]


def test_benchmark_provenance_pins_authoritative_content_revision_and_full_page_set():
    class Dataset:
        _fingerprint = "dataset-content-fingerprint"
        columns = {
            "PageID": ["0007", "8", "9"],
            "text": ["x" * 80, "short", "x" * 100],
        }

        def __getitem__(self, key):
            if key not in self.columns:
                raise KeyError(key)
            return self.columns[key]

    provenance = GS.benchmark_provenance("owner/gt", "revision-1", Dataset())
    assert provenance == {
        "dataset_id": "owner/gt",
        "requested_revision": "revision-1",
        "resolved_revision": "local:dataset-content-fingerprint",
        "dataset_fingerprint": "dataset-content-fingerprint",
        "expected_page_count": 3,
        "full_page_set_fingerprint": GS.page_set_fingerprint(["0007", "8", "9"]),
        "sample_stratum_threshold": 80,
        "expected_stratum_counts": {"content": 2, "sparse_blank": 1},
    }


def test_remote_revision_is_resolved_without_loading_or_live_network(monkeypatch):
    calls = []

    class FakeApi:
        def dataset_info(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(sha="a" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert GS.resolve_dataset_revision("owner/gt", "release") == "a" * 40
    assert calls == [{"repo_id": "owner/gt", "revision": "release"}]


def test_local_revision_uses_loaded_content_fallback_without_hub(tmp_path, monkeypatch):
    local = tmp_path / "dataset"
    local.mkdir()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: (_ for _ in ()).throw(AssertionError))
    assert GS.resolve_dataset_revision(local, None) is None


def test_remote_benchmark_identity_is_resolved_sha_not_private_fingerprint():
    class Dataset:
        _fingerprint = "private-transform-fingerprint"

        def __getitem__(self, key):
            if key == "PageID":
                return [1]
            if key == "text":
                return ["short"]
            raise KeyError(key)

    sha = "b" * 40
    provenance = GS.benchmark_provenance("owner/gt", None, Dataset(), resolved_revision=sha)
    assert provenance["resolved_revision"] == sha
    assert provenance["dataset_fingerprint"] == sha


def test_page_set_fingerprint_is_injective_for_newline_bearing_ids():
    assert GS.page_set_fingerprint(["a", "b\nc"]) != GS.page_set_fingerprint(["a\nb", "c"])


def test_missing_regions_json_degrades_to_no_regions():
    gt = {k: v for k, v in GT.items() if k != "regions_json"}
    got = GS.score_gt_row(gt, "The tree sparrow is common.", model="m", err=False, page_id=7)
    assert got["region_global_token_recall"] == {}  # region-less page scores, does not raise
    assert got["recall"] == 1.0


def test_sample_stratum_propagates_and_legacy_rows_derive_at_80():
    explicit = {**GT, "sample_stratum": "content", "sample_stratum_threshold": 7}
    got = GS.score_gt_row(explicit, "", model="m", err=False)
    assert (got["sample_stratum"], got["sample_stratum_threshold"]) == ("content", 7)

    legacy = {**GT, "text": "x" * 79}
    got = GS.score_gt_row(legacy, "", model="m", err=False)
    assert (got["sample_stratum"], got["sample_stratum_threshold"]) == ("sparse_blank", 80)
    legacy["text"] += "x"
    assert GS.score_gt_row(legacy, "", model="m", err=False)["sample_stratum"] == "content"


def test_score_gt_row_rejects_explicit_stratum_conflicting_with_text():
    gt = {**GT, "text": "short", "sample_stratum": "content", "sample_stratum_threshold": 80}
    with pytest.raises(ValueError, match="conflicts with deterministic"):
        GS.score_gt_row(gt, "", model="m", err=False)


def test_benchmark_provenance_rejects_declared_stratum_conflicting_with_text():
    class Dataset:
        _fingerprint = "bad-strata"
        columns = {
            "PageID": [1],
            "text": ["short"],
            "sample_stratum": ["content"],
            "sample_stratum_threshold": [80],
        }

        def __getitem__(self, key):
            if key not in self.columns:
                raise KeyError(key)
            return self.columns[key]

    with pytest.raises(ValueError, match="conflicts with deterministic"):
        GS.benchmark_provenance("sample/repo", None, Dataset())


def test_benchmark_provenance_includes_uniform_sampler_provenance():
    class Dataset:
        _fingerprint = "sample-fingerprint"
        columns = {
            "PageID": [1, 2],
            "text": ["x" * 80, "short"],
            "sample_stratum": ["content", "sparse_blank"],
            "sample_stratum_threshold": [80, 80],
            "sampler_version": ["1.0", "1.0"],
            "sampler_seed": [7, 7],
            "sampler_requested_n": [2, 2],
            "sampler_source_repo": ["source/repo", "source/repo"],
            "sampler_source_revision": ["abc123", "abc123"],
        }

        def __getitem__(self, key):
            if key not in self.columns:
                raise KeyError(key)
            return self.columns[key]

    provenance = GS.benchmark_provenance("sample/repo", "sample-rev", Dataset())
    assert provenance["sampler_provenance"] == {
        "sampler_version": "1.0",
        "sampler_seed": 7,
        "sampler_requested_n": 2,
        "sampler_source_repo": "source/repo",
        "sampler_source_revision": "abc123",
        "sample_stratum_threshold": 80,
    }


def test_truncated_is_a_flagged_subset_of_error():
    # Schema 2.1. A truncated page is excluded from aggregates like any error row, but the
    # flag has to travel WITH the row: it is derived from the OCR sentinel, which no
    # downstream layer can see, and report() uses it to decide eligibility.
    got = GS.score_gt_row(GT, "__ERR__FinishReason:length", model="m", err=True,
                          truncated=True, page_id=7)
    assert got["error"] is True and got["truncated"] is True
    assert got == _expected(GT, "ignored", model="m", err=True, page_id=7, truncated=True)


def test_truncated_defaults_false_so_existing_callers_are_unchanged():
    got = GS.score_gt_row(GT, "some text", model="m", err=False, page_id=7)
    assert got["truncated"] is False


def test_the_truncation_predicate_is_shared_so_runners_cannot_diverge():
    # run_openai.py once classified truncations differently from score_dataset.py, which would
    # have marked every truncating model ineligible on any board built from its output.
    assert GS.is_truncated("__ERR__FinishReason:length") is True
    assert GS.is_truncated("__ERR__Producer:transport: TimeoutError") is False
    assert GS.is_truncated("real transcribed text") is False
    assert GS.is_truncated(None) is False
