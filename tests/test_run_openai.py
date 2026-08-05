"""Pure tests for the OpenAI runner's prompt transport and raw-output cache key."""
import json
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

import gt_score as GS
import report as RPT
import run_openai as R


def test_cache_path_uses_full_model_identifier(tmp_path):
    """Models with the same basename must not share OCR cache files."""
    a = R.cache_path("org-a/OCR-Model", "dataset", "https://endpoint.test/v1", prompt="p", cache_dir=tmp_path)
    b = R.cache_path("org-b/OCR-Model", "dataset", "https://endpoint.test/v1", prompt="p", cache_dir=tmp_path)
    assert a != b
    assert a.parent == tmp_path and b.parent == tmp_path
    assert "/" not in a.name and a.suffix == ".parquet"


def test_cache_path_changes_with_dataset_or_endpoint(tmp_path):
    base = R.cache_path("org/model", "dataset-a", "https://endpoint.test/v1", prompt="p", cache_dir=tmp_path)
    other_dataset = R.cache_path("org/model", "dataset-b", "https://endpoint.test/v1", prompt="p", cache_dir=tmp_path)
    other_endpoint = R.cache_path("org/model", "dataset-a", "https://other.test/v1", prompt="p", cache_dir=tmp_path)
    assert len({base, other_dataset, other_endpoint}) == 3


def test_cache_path_normalizes_trailing_endpoint_slash(tmp_path):
    a = R.cache_path("org/model", "dataset", "https://endpoint.test/v1", prompt="p", cache_dir=tmp_path)
    b = R.cache_path("org/model", "dataset", "https://endpoint.test/v1/", prompt="p", cache_dir=tmp_path)
    assert a == b


def test_prompt_text_preserves_empty_inline_and_file_line_endings(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_bytes(b"a\r\n\r\n")
    assert R._prompt_text("", prompt_file) == ""
    assert R._prompt_text(None, prompt_file) == "a\r\n\r\n"


def test_prompt_file_warns_about_bom_without_stripping_it(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_bytes(b"\xef\xbb\xbfexact\r\n")
    with pytest.warns(UserWarning, match="UTF-8 BOM.*preserved"):
        text = R._prompt_text(None, prompt_file)
    assert text == "\ufeffexact\r\n"


def test_cache_path_changes_with_exact_prompt_dataset_content_revision_and_page_set(tmp_path):
    kwargs = {"prompt": "", "cache_dir": tmp_path, "dataset_fingerprint": "content-a",
              "page_set_fingerprint": "pages-a", "dataset_revision": "rev-a",
              "resolved_revision": "sha-a"}
    base = R.cache_path("org/model", "dataset", "https://endpoint.test/v1", **kwargs)
    whitespace = R.cache_path("org/model", "dataset", "https://endpoint.test/v1",
                              **{**kwargs, "prompt": "\n"})
    content = R.cache_path("org/model", "dataset", "https://endpoint.test/v1",
                           **{**kwargs, "dataset_fingerprint": "content-b"})
    pages = R.cache_path("org/model", "dataset", "https://endpoint.test/v1",
                         **{**kwargs, "page_set_fingerprint": "pages-b"})
    revision = R.cache_path("org/model", "dataset", "https://endpoint.test/v1",
                            **{**kwargs, "dataset_revision": "rev-b"})
    resolved = R.cache_path("org/model", "dataset", "https://endpoint.test/v1",
                            **{**kwargs, "resolved_revision": "sha-b"})
    assert len({base, whitespace, content, pages, revision, resolved}) == 6


class _FakeDS:
    def __init__(self, ids):
        self.ids = ids

    def __getitem__(self, key):
        assert key == "PageID"
        return self.ids


def test_page_set_fingerprint_is_order_and_dtype_stable():
    assert R.page_set_fingerprint(_FakeDS([1, 2, 3])) == R.page_set_fingerprint(_FakeDS(["3", "1", "2"]))


def test_cached_ocr_rejects_null_or_non_string_cells(tmp_path, monkeypatch):
    cache = tmp_path / "cache.parquet"
    cache.touch()
    ds = _FakeDS([1, 2])
    monkeypatch.setattr(R.pd, "read_parquet", lambda _path: pd.DataFrame(
        {"PageID": [1, 2], "ocr": ["ok", None]}))
    with pytest.raises(ValueError, match="null or non-string OCR cells"):
        R.run_model(None, "model", ds, 1, cache, "prompt")


def test_run_model_resumes_partial_cache_and_retries_only_missing_or_error_rows(tmp_path, monkeypatch):
    class Dataset:
        records = [
            {"PageID": 1, "volume": "v", "image": "image-1"},
            {"PageID": 2, "volume": "v", "image": "image-2"},
            {"PageID": 3, "volume": "w", "image": "image-3"},
        ]

        def __getitem__(self, key):
            return [row[key] for row in self.records]

        def __iter__(self):
            return iter(self.records)

    cache = tmp_path / "resume.parquet"
    cache.touch()
    stored = pd.DataFrame({"PageID": [1, 2], "volume": ["v", "v"],
                           "ocr": ["keep-success", "__ERR__Old"]})
    calls = []

    def ocr_one(_client, _model, image, _prompt):
        calls.append(image)
        return "retry-success" if image == "image-2" else "__ERR__StillFailing"

    writes = []

    def tracked_write(frame, _path):
        nonlocal stored
        stored = frame.copy()
        writes.append(stored)

    monkeypatch.setattr(R.pd, "read_parquet", lambda _path: stored.copy())
    monkeypatch.setattr(R, "ocr_one", ocr_one)
    monkeypatch.setattr(R, "_atomic_write_parquet", tracked_write)
    out = R.run_model(None, "model", Dataset(), 1, cache, "prompt", checkpoint_every=1)

    assert calls == ["image-2", "image-3"]
    assert out["PageID"].tolist() == [1, 2, 3]
    assert out["ocr"].tolist() == ["keep-success", "retry-success", "__ERR__StillFailing"]
    assert len(writes) >= 2  # partial checkpoints plus the exact final output
    assert stored.equals(out)


def test_no_retry_errors_retains_cached_failures_and_only_runs_missing_pages(tmp_path, monkeypatch):
    class Dataset:
        records = [
            {"PageID": 1, "volume": "v", "image": "image-1"},
            {"PageID": 2, "volume": "v", "image": "image-2"},
            {"PageID": 3, "volume": "v", "image": "image-3"},
        ]

        def __getitem__(self, key):
            return [row[key] for row in self.records]

        def __iter__(self):
            return iter(self.records)

    cache = tmp_path / "retain-errors.parquet"
    cache.touch()
    stored = pd.DataFrame({"PageID": [1, 2], "ocr": ["ok", "__ERR__Permanent"]})
    calls = []
    monkeypatch.setattr(R.pd, "read_parquet", lambda _path: stored.copy())
    monkeypatch.setattr(R, "ocr_one", lambda _client, _model, image, _prompt: calls.append(image) or "new")

    out = R.run_model(None, "model", Dataset(), 1, cache, "prompt", retry_errors=False)

    assert calls == ["image-3"]
    assert out["ocr"].tolist() == ["ok", "__ERR__Permanent", "new"]


def test_interruption_cancels_bounded_queue_and_exception_checkpoints_incorporated_results(
        tmp_path, monkeypatch):
    class Dataset:
        records = [{"PageID": i, "volume": "v", "image": f"image-{i}"} for i in range(1, 21)]

        def __init__(self):
            self.pulled = []

        def __getitem__(self, key):
            return [row[key] for row in self.records]

        def __iter__(self):
            for row in self.records:
                self.pulled.append(row["PageID"])
                yield row

    dataset = Dataset()
    first_finished = threading.Event()
    interrupt_started = threading.Event()
    release_interrupt = threading.Event()
    calls = []
    writes = []
    real_wait = R.wait
    wait_calls = 0

    def interrupted_ocr(_client, _model, image, _prompt):
        calls.append(image)
        if image == "image-1":
            first_finished.set()
            return "first"
        if image == "image-2":
            interrupt_started.set()
            assert release_interrupt.wait(timeout=2)
            raise KeyboardInterrupt
        return "started-before-cancellation"

    def ordered_wait(futures, *, return_when):
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            assert first_finished.wait(timeout=2)
            done, pending = real_wait(futures, return_when=return_when)
            assert len(done) == 1
            return done, pending
        assert interrupt_started.wait(timeout=2)
        assert writes == []  # checkpoint_every has not been reached
        interrupt_future = next(future for future in futures if future.running())
        release_interrupt.set()
        done, _ = real_wait({interrupt_future}, return_when=return_when)
        return done, set(futures) - done

    def tracked_write(frame, _path):
        writes.append(frame.copy())

    monkeypatch.setattr(R, "ocr_one", interrupted_ocr)
    monkeypatch.setattr(R, "wait", ordered_wait)
    monkeypatch.setattr(R, "_atomic_write_parquet", tracked_write)

    with pytest.raises(KeyboardInterrupt):
        R.run_model(None, "model", dataset, 1, tmp_path / "interrupt.parquet", "prompt",
                    checkpoint_every=10)

    assert len(writes) == 1
    assert writes[0]["PageID"].tolist() == [1]
    assert writes[0]["ocr"].tolist() == ["first"]
    assert len(calls) <= 3  # two-worker-multiple bound: queued work is cancelled, not the corpus
    assert not any(image in calls for image in ("image-4", "image-20"))
    # Pending rows are drawn lazily: only in-flight rows were ever pulled (and would have decoded
    # their images), not the whole corpus up front.
    assert len(dataset.pulled) <= 6 < len(Dataset.records)


def test_atomic_write_parquet_round_trips_real_file(tmp_path):
    path = tmp_path / "nested" / "checkpoint.parquet"
    expected = pd.DataFrame({"PageID": [2, 1], "volume": ["b", "a"], "ocr": ["", "exact\n"]})

    R._atomic_write_parquet(expected, path)

    assert path.exists()
    assert pd.read_parquet(path).equals(expected)
    assert not list(path.parent.glob(f".{path.stem}-*.parquet"))


def test_atomic_write_failure_preserves_existing_parquet_and_cleans_temporary_file(
        tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.parquet"
    existing = pd.DataFrame({"PageID": [1], "volume": ["a"], "ocr": ["existing"]})
    existing.to_parquet(path, index=False)
    original_bytes = path.read_bytes()
    replacement = pd.DataFrame({"PageID": [2], "volume": ["b"], "ocr": ["replacement"]})

    def fail_after_writing_junk(_frame, temporary, *, index):
        assert index is False
        R.pathlib.Path(temporary).write_bytes(b"incomplete parquet")
        raise OSError("simulated parquet write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_after_writing_junk)

    with pytest.raises(OSError, match="simulated parquet write failure"):
        R._atomic_write_parquet(replacement, path)

    assert path.read_bytes() == original_bytes
    assert pd.read_parquet(path).equals(existing)
    assert not list(path.parent.glob(f".{path.stem}-*.parquet"))


BENCHMARK = {"dataset_id": "ds", "expected_page_count": 1}


def _plan(tmp_path, model="org/model", pages=372):
    return [(model, pages, tmp_path / "scorecards" / "org-model-abc.parquet")]


def _scorecard(model="other/model", **overrides):
    """A row shaped like a scorecard the board would happily read."""
    row = {"model": model, "page_id": "1", "benchmark_provenance": json.dumps(BENCHMARK),
           **RPT.current_global_provenance()}
    row.update(overrides)
    return pd.DataFrame([row])


def test_preflight_gates_interactive_paid_requests(tmp_path):
    empty = tmp_path / "scorecards"
    empty.mkdir()
    plan = _plan(tmp_path)
    call = dict(benchmark=BENCHMARK, scorecards_dir=empty)
    with pytest.raises(SystemExit, match="aborted before any paid request"):
        R._preflight(plan, 16384, 8, assume_yes=False, interactive=True,
                     ask=lambda _prompt: "n", **call)
    R._preflight(plan, 16384, 8, assume_yes=False, interactive=True,
                 ask=lambda _prompt: " Y ", **call)
    R._preflight(plan, 16384, 8, assume_yes=True, interactive=True,
                 ask=lambda _prompt: pytest.fail("--yes must not prompt"), **call)
    R._preflight(plan, 16384, 8, assume_yes=False, interactive=False,
                 ask=lambda _prompt: pytest.fail("non-interactive runs must not prompt"), **call)


def test_preflight_treats_eof_as_declining_to_spend(tmp_path):
    empty = tmp_path / "scorecards"
    empty.mkdir()

    def eof(_prompt):
        raise EOFError

    with pytest.raises(SystemExit, match="aborted before any paid request"):
        R._preflight(_plan(tmp_path), 16384, 8, assume_yes=False, interactive=True, ask=eof,
                     benchmark=BENCHMARK, scorecards_dir=empty)


def test_preflight_does_not_ask_when_nothing_would_be_requested(tmp_path):
    empty = tmp_path / "scorecards"
    empty.mkdir()
    R._preflight(_plan(tmp_path, pages=0), 16384, 8, assume_yes=False, interactive=True,
                 ask=lambda _prompt: pytest.fail("a fully cached run needs no consent"),
                 benchmark=BENCHMARK, scorecards_dir=empty)


@pytest.mark.parametrize("write", [
    pytest.param(lambda path: _scorecard("org/model").to_parquet(path, index=False),
                 id="same-model-under-another-run-identity"),
    pytest.param(lambda path: path.write_bytes(b"not a parquet file"),
                 id="unreadable-file-the-glob-still-reads"),
    pytest.param(lambda path: _scorecard().drop(columns=["page_id"]).to_parquet(path, index=False),
                 id="missing-board-required-page_id"),
    pytest.param(lambda path: _scorecard(scorer_version="0.0-stale").to_parquet(path, index=False),
                 id="stale-scorer-version"),
    pytest.param(lambda path: _scorecard().drop(columns=["norm_version"]).to_parquet(path, index=False),
                 id="missing-provenance-column"),
    pytest.param(lambda path: _scorecard(
        benchmark_provenance=json.dumps({"dataset_id": "other"})).to_parquet(path, index=False),
        id="different-benchmark"),
])
def test_preflight_refuses_before_spending_on_anything_that_breaks_the_board(tmp_path, write):
    """leaderboard.py globs the whole directory, so the preflight's promise — pay, then get a board —
    only holds if every file the glob picks up would pass. Each of these fails the board *after* a
    campaign is paid for unless it is caught here."""
    scorecards = tmp_path / "scorecards"
    scorecards.mkdir()
    write(scorecards / "earlier-run.parquet")

    with pytest.raises(SystemExit, match="aborted before any paid request"):
        R._preflight(_plan(tmp_path), 16384, 8, assume_yes=True, interactive=False,
                     benchmark=BENCHMARK, scorecards_dir=scorecards)


def test_board_blockers_passes_a_directory_the_board_can_build(tmp_path):
    scorecards = tmp_path / "scorecards"
    scorecards.mkdir()
    own = scorecards / "org-model-abc.parquet"
    _scorecard("org/model").to_parquet(own, index=False)  # this run rewrites its own output
    _scorecard("other/model").to_parquet(scorecards / "compatible.parquet", index=False)

    assert R.board_blockers(["org/model"], [own], BENCHMARK, scorecards) == {}


def test_archiving_blocking_scorecards_is_opt_in_and_moves_them_out_of_the_glob(tmp_path):
    """Superseding a scorecard must be a choice, and a recoverable one: the files move into a
    timestamped subdirectory the board's non-recursive glob no longer sees, rather than vanishing."""
    scorecards = tmp_path / "scorecards"
    scorecards.mkdir()
    stale = scorecards / "earlier-run.parquet"
    _scorecard("org/model").to_parquet(stale, index=False)
    keep = scorecards / "compatible.parquet"
    _scorecard("other/model").to_parquet(keep, index=False)

    R._preflight(_plan(tmp_path), 16384, 8, assume_yes=True, interactive=False,
                 benchmark=BENCHMARK, scorecards_dir=scorecards, archive_blocking=True)

    assert not stale.exists()
    assert keep.exists()  # a compatible scorecard is left alone
    archived = list(scorecards.glob("archived-*/earlier-run.parquet"))
    assert len(archived) == 1, "the superseded scorecard must be recoverable, not deleted"
    assert list(scorecards.glob("*.parquet")) == [keep], "the board's glob no longer sees it"


def test_combined_out_inside_the_globbed_directory_is_rejected(tmp_path):
    scorecards = tmp_path / "scorecards"
    scorecards.mkdir()
    R.check_combined_out(None, scorecards)
    R.check_combined_out(tmp_path / "combined.parquet", scorecards)
    for inside in (scorecards / "combined.parquet", scorecards / "nested" / "combined.parquet"):
        with pytest.raises(SystemExit, match="resolves inside"):
            R.check_combined_out(inside, scorecards)


def test_pending_page_count_reports_requests_not_total_pages(tmp_path):
    cache = tmp_path / "cache.parquet"
    assert R.pending_page_count(cache, ["1", "2", "3"]) == 3  # no cache: everything is requested

    pd.DataFrame({"PageID": [1, 2], "ocr": ["done", "__ERR__Timeout"]}).to_parquet(cache, index=False)
    assert R.pending_page_count(cache, ["1", "2", "3"], retry_errors=True) == 2  # page 3 + the error
    assert R.pending_page_count(cache, ["1", "2", "3"], retry_errors=False) == 1  # error retained

    cache.write_bytes(b"corrupt")
    assert R.pending_page_count(cache, ["1", "2", "3"]) == 3  # unreadable: never understate spend


def test_flat_scorecard_rows_serializes_dict_fields_as_json():
    rows = GS.flat_scorecard_rows([
        {"PageID": 1, "cer": 0.5, "run_provenance": {"b": 1, "a": 2}},
        {"PageID": 2, "cer": 0.1, "run_provenance": {"b": 1, "a": 2}},
    ])
    assert [row["PageID"] for row in rows] == [1, 2]
    assert json.loads(rows[0]["run_provenance"]) == {"a": 2, "b": 1}


def test_run_model_rejects_cached_rows_outside_dataset(tmp_path, monkeypatch):
    cache = tmp_path / "cache.parquet"
    cache.touch()
    stored = pd.DataFrame({"PageID": [1, 99], "ocr": ["ok", "extra"]})
    monkeypatch.setattr(R.pd, "read_parquet", lambda _path: stored)
    with pytest.raises(ValueError, match="outside the selected dataset"):
        R.run_model(None, "model", _FakeDS([1, 2]), 1, cache, "prompt")


def _client_returning(response, calls=None):
    def create(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return response

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_ocr_one_sends_exact_prompt(monkeypatch):
    calls = []
    client = _client_returning(SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="ocr"), finish_reason="stop")]), calls)
    monkeypatch.setattr(R, "data_url", lambda _image: "data:image/jpeg;base64,x")
    prompt = " exact prompt\n\n"
    assert R.ocr_one(client, "model", object(), prompt) == "ocr"
    assert calls[-1]["messages"][0]["content"][0]["text"] == prompt
    assert calls[-1]["max_tokens"] == 16384
    assert calls[-1]["temperature"] == 0


def test_ocr_one_flags_truncated_response_as_error_without_retrying(monkeypatch):
    """A finish_reason other than "stop" (length truncation, content filter) must become an
    __ERR__ row — excluded and retryable — never a cached transcription; and re-requesting a
    deterministic truncation in-run would only re-pay for the same result."""
    calls = []
    client = _client_returning(SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="partial transcription..."), finish_reason="length")]), calls)
    monkeypatch.setattr(R, "data_url", lambda _image: "data:image/jpeg;base64,x")
    assert R.ocr_one(client, "model", object(), "prompt") == "__ERR__FinishReason:length"
    assert len(calls) == 1


def test_ocr_one_flags_missing_content_as_error_but_keeps_legitimate_empty_string(monkeypatch):
    monkeypatch.setattr(R, "data_url", lambda _image: "data:image/jpeg;base64,x")
    none_content = _client_returning(SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=None), finish_reason="stop")]))
    assert R.ocr_one(none_content, "model", object(), "prompt") == "__ERR__EmptyContent"
    empty_string = _client_returning(SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=""), finish_reason="stop")]))
    assert R.ocr_one(empty_string, "model", object(), "prompt") == ""


def test_request_settings_file_loads_qwen_non_thinking_configuration(tmp_path):
    path = tmp_path / "qwen.json"
    path.write_text("""{
      "max_tokens": 32768,
      "temperature": 0.7,
      "top_p": 0.8,
      "presence_penalty": 1.5,
      "extra_body": {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": false}
      }
    }""")

    assert R._load_request_settings(path) == {
        "max_tokens": 32768,
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


@pytest.mark.parametrize(("body", "message"), [
    ('[1, 2]', "one JSON object"),
    ('{"model": "override"}', "Unsupported request setting"),
    ('{"temperature": NaN}', "Non-finite JSON number"),
    ('{"top_p": 2}', "top_p must be between"),
    ('{"extra_body": []}', "extra_body must be a JSON object"),
    ('{"extra_body": {"model": "override"}}', "extra_body cannot override"),
    ('{"extra_body": {"messages": []}}', "extra_body cannot override"),
    ('{"extra_body": {"temperature": 1}}', "extra_body cannot override"),
    ('{"extra_body": {"stream": true}}', "extra_body cannot override"),
    ('{"extra_body": {"max_completion_tokens": 10}}', "extra_body cannot override"),
    ('{"extra_body": {"metadata": {}}}', "extra_body cannot override"),
    ('{"extra_body": {"prompt_cache_retention": "24h"}}', "extra_body cannot override"),
    ('{"extra_body": {"scale": 1e400}}', "Non-finite number"),
    ('{"max_tokens": 1, "max_tokens": 2}', "Duplicate request-settings key"),
])
def test_request_settings_file_rejects_unsafe_or_invalid_values(tmp_path, body, message):
    path = tmp_path / "bad.json"
    path.write_text(body)
    with pytest.raises(ValueError, match=message):
        R._load_request_settings(path)


def test_request_settings_are_sent_and_change_cache_identity(tmp_path, monkeypatch):
    calls = []
    client = _client_returning(SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=""), finish_reason="stop")]), calls)
    monkeypatch.setattr(R, "data_url", lambda _image: "data:image/jpeg;base64,x")
    settings = {
        "max_tokens": 32768,
        "temperature": 0.7,
        "top_p": 0.8,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }

    assert R.ocr_one(client, "model", object(), "prompt", settings) == ""
    assert {key: calls[-1][key] for key in settings} == settings
    default = R.cache_path(
        "org/model", "dataset", "https://endpoint.test/v1", prompt="p", cache_dir=tmp_path,
    )
    configured = R.cache_path(
        "org/model", "dataset", "https://endpoint.test/v1", prompt="p", cache_dir=tmp_path,
        request_settings=settings,
    )
    reordered = R.cache_path(
        "org/model", "dataset", "https://endpoint.test/v1", prompt="p", cache_dir=tmp_path,
        request_settings={
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "top_p": 0.8, "temperature": 0.7, "max_tokens": 32768,
        },
    )
    assert default != configured == reordered
    payload = R._cache_payload(
        "org/model", "dataset", "https://endpoint.test/v1", "p",
        request_settings=settings,
    )
    assert payload["max_tokens"] == 32768
    assert payload["temperature"] == 0.7
    assert payload["request_settings"] == {
        "top_p": 0.8,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def test_explicit_default_request_settings_keep_existing_cache_identity(tmp_path):
    implicit = R.cache_path(
        "org/model", "dataset", "https://endpoint.test/v1", prompt="p", cache_dir=tmp_path,
    )
    explicit = R.cache_path(
        "org/model", "dataset", "https://endpoint.test/v1", prompt="p", cache_dir=tmp_path,
        request_settings={"max_tokens": 16384, "temperature": 0},
    )
    assert implicit == explicit


def test_run_model_forwards_request_settings(tmp_path, monkeypatch):
    class Dataset:
        records = [{"PageID": 1, "volume": "v", "image": "image"}]

        def __getitem__(self, key):
            return [row[key] for row in self.records]

        def __iter__(self):
            return iter(self.records)

    seen = []

    def ocr_one(_client, _model, _image, _prompt, *, request_settings):
        seen.append(request_settings)
        return "ocr"

    monkeypatch.setattr(R, "ocr_one", ocr_one)
    monkeypatch.setattr(R, "_atomic_write_parquet", lambda _frame, _path: None)
    settings = {"max_tokens": 32768, "temperature": 0.7, "top_p": 0.8}

    out = R.run_model(
        None, "model", Dataset(), 1, tmp_path / "cache.parquet", "prompt",
        request_settings=settings,
    )

    assert seen == [settings]
    assert out["ocr"].tolist() == ["ocr"]
