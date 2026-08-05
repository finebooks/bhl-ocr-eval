"""add_language pure functions: labels, scorecard restamp, and fail-closed Hub push."""
from types import SimpleNamespace

import pandas as pd
import pytest

from add_language import push_and_verify, resolve, restamp_df, short_code

LONG = "x" * 200


def test_short_code_mapping():
    assert short_code("__label__deu_Latn") == "de"
    assert short_code("fra_Latn") == "fr"
    assert short_code("lat_Latn") == "la"
    assert short_code("eng_Latn") == "en"
    assert short_code("deu_Latf") == "de"  # Fraktur script variant still maps
    assert short_code("nds_Latn") == "und"  # unmapped language -> und, not a guess


def _page(pid, vol, glotlid, conf, text=LONG):
    return {"page_id": pid, "volume": vol, "glotlid": glotlid, "conf": conf, "text": text}


def test_resolve_confident_pages_stand_alone():
    out = resolve([_page(1, "v", "deu_Latn", 0.9), _page(2, "v", "fra_Latn", 0.9)])
    assert out == {1: "de", 2: "fr"}


def test_resolve_low_confidence_inherits_volume_majority():
    out = resolve([_page(1, "v", "deu_Latn", 0.9), _page(2, "v", "deu_Latn", 0.9),
                   _page(3, "v", "rus_Cyrl", 0.2)])
    assert out[3] == "de"


def test_resolve_short_text_not_confident_even_at_high_conf():
    out = resolve([_page(1, "v", "deu_Latn", 0.9), _page(2, "v", "fra_Latn", 0.99, text="Le chat")])
    assert out[2] == "de"  # short page falls back to the volume majority


def test_resolve_unmapped_code_not_confident():
    out = resolve([_page(1, "v", "deu_Latn", 0.9), _page(2, "v", "nds_Latn", 0.99)])
    assert out[2] == "de"  # 'und' pages inherit too


def test_resolve_volume_with_no_confident_pages_is_und():
    out = resolve([_page(1, "v", "nds_Latn", 0.2, text="hi")])
    assert out == {1: "und"}


def test_restamp_only_language_changes():
    df = pd.DataFrame({"page_id": [1, 2, 3], "language": ["ru", "ru", "en"],
                       "cer_body_dip": [0.1, 0.2, 0.3], "model": ["m"] * 3})
    out = restamp_df(df, {1: "de", 2: "fr"})
    assert list(out["language"]) == ["de", "fr", "en"]  # unmapped page 3 keeps its label
    pd.testing.assert_frame_equal(out.drop(columns="language"), df.drop(columns="language"))
    assert list(df["language"]) == ["ru", "ru", "en"]  # input not mutated


class _Dataset:
    split = "train"
    info = SimpleNamespace(splits={"train": object()})
    column_names = ["PageID", "language"]
    features = {"PageID": "int64", "language": "string"}

    def __len__(self):
        return 2


class _DatasetDict(dict):
    calls = []

    def push_to_hub(self, repo_id, *, private):
        self.calls.append((dict(self), repo_id, private))
        return SimpleNamespace(oid="abc123")


def test_push_and_verify_replaces_complete_dataset_and_reloads_pinned_commit():
    _DatasetDict.calls = []
    seen = []

    def load(repo_id, *, split, revision):
        seen.append((repo_id, split, revision))
        return _Dataset()

    revision = push_and_verify(
        _Dataset(), "org/sample", dataset_dict_cls=_DatasetDict, load_dataset_fn=load,
    )

    assert revision == "abc123"
    assert len(_DatasetDict.calls) == 1
    payload, repo_id, private = _DatasetDict.calls[0]
    assert set(payload) == {"train"}
    assert isinstance(payload["train"], _Dataset)
    assert (repo_id, private) == ("org/sample", True)
    assert seen == [("org/sample", "train", "abc123")]


def test_push_and_verify_rejects_multi_split_dataset_before_push():
    dataset = _Dataset()
    dataset.info = SimpleNamespace(splits={"train": object(), "test": object()})
    _DatasetDict.calls = []

    with pytest.raises(ValueError, match="complete default configuration"):
        push_and_verify(
            dataset, "org/sample", dataset_dict_cls=_DatasetDict,
            load_dataset_fn=lambda *_args, **_kwargs: None,
        )

    assert _DatasetDict.calls == []


def test_push_and_verify_requires_commit_sha():
    class MissingOidDict(dict):
        def push_to_hub(self, _repo_id, *, private):
            assert private is True
            return SimpleNamespace(oid=None)

    with pytest.raises(RuntimeError, match="immutable commit SHA"):
        push_and_verify(
            _Dataset(), "org/sample", dataset_dict_cls=MissingOidDict,
            load_dataset_fn=lambda *_args, **_kwargs: None,
        )


def test_push_and_verify_rejects_pinned_schema_mismatch():
    verified = _Dataset()
    verified.column_names = ["PageID", "language_volume"]

    with pytest.raises(RuntimeError, match="Pinned verification failed.*columns"):
        push_and_verify(
            _Dataset(), "org/sample", dataset_dict_cls=_DatasetDict,
            load_dataset_fn=lambda *_args, **_kwargs: verified,
        )
