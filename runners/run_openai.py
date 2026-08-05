# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets", "huggingface-hub", "openai", "pillow", "pandas", "pyarrow", "jiwer>=4,<5"]
# ///
"""Run OCR models over the eval sample through ANY OpenAI-compatible chat/completions endpoint, then
score with the frozen scorer + thick report. The OpenAI protocol is the universal interface — vLLM,
SGLang and most hosted APIs all speak it — so one runner covers everything you can serve.

`--base-url` is REQUIRED and must point at an endpoint whose serving stack you control. Scores from
hosted inference routers are not admissible on this benchmark: a routed request does not record which
provider served it, at what quantization, or under what serving configuration, and CER is sensitive
to all three (see RESULTS.md, "What counts as a score").

  - a model you serve yourself on an HF Job:
      hf jobs run --detach --expose 8000 --flavor a10g-small -s HF_TOKEN \
        vllm/vllm-openai vllm serve <model> --max-model-len 32768
      uv run runners/run_openai.py --prompt-file prompt.txt --models <model> \
        --base-url https://<job_id>--8000.hf.jobs/v1
  - a vLLM / SGLang server you run locally:
      uv run runners/run_openai.py --base-url http://localhost:8000/v1 --api-key-env NONE \
        --prompt "Transcribe this page." --models dots-ocr

Use ``--request-settings-file settings.json`` for model-recommended generation controls. The validated
JSON is merged over the runner defaults, sent to the endpoint, and recorded by value in cache/run
identity; run models separately when their settings differ.

Reads `image` + GT (`text` / `body_text` / `furniture_text` / `regions_json`) from the sample, caches
each model's RAW outputs (so re-scoring is free as the scorer evolves), writes each model's per-page
scorecard into `data/scorecards/` as soon as that model finishes (so a crash on model N keeps models
1..N-1's scorecards), and prints the ordered comparison. The caller must supply the transcription
prompt explicitly; the benchmark does not own a default prompt. To score outputs a
DIFFERENT tool already produced, use `score_dataset.py` instead.
"""
import argparse
import base64
import datetime
import hashlib
import io
import json
import math
import os
import pathlib
import re
import sys
import tempfile
import urllib.parse
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scoring"))
import gt_score as GS  # noqa: E402
import report as RPT  # noqa: E402

CACHE = pathlib.Path(__file__).resolve().parent.parent / "data" / "vlm_cache"
SCORECARDS = pathlib.Path(__file__).resolve().parent.parent / "data" / "scorecards"
CACHE_KEY_VERSION = "2026-07-20a"
IMAGE_MAX_EDGE = 1536
IMAGE_JPEG_QUALITY = 90
MAX_TOKENS = 16384
TEMPERATURE = 0
DEFAULT_REQUEST_SETTINGS = {"max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}
ALLOWED_REQUEST_SETTINGS = {
    "max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty",
    "stop", "seed", "reasoning_effort", "response_format", "logprobs", "top_logprobs",
    "extra_body",
}
RESERVED_EXTRA_BODY_FIELDS = ALLOWED_REQUEST_SETTINGS | {
    "audio", "function_call", "functions", "logit_bias", "max_completion_tokens", "messages",
    "metadata", "modalities", "model", "moderation", "n", "parallel_tool_calls", "prediction",
    "prompt_cache_key", "prompt_cache_retention", "safety_identifier", "service_tier", "store", "stream",
    "stream_options", "tool_choice", "tools", "user", "verbosity", "web_search_options",
}


def data_url(img):
    im = img.convert("RGB")
    im.thumbnail((IMAGE_MAX_EDGE, IMAGE_MAX_EDGE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=IMAGE_JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _slug(s, max_len=80):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-._")
    return (slug or "model")[:max_len].rstrip("-._") or "model"


# Hosts that route a request to an unrecorded third-party provider. A score obtained this way is
# not attributable to the named model — see RESULTS.md, "What counts as a score".
ROUTED_HOSTS = ("router.huggingface.co",)


def _reject_routed_endpoint(base_url):
    host = urllib.parse.urlsplit(base_url or "").hostname or ""
    if any(host == h or host.endswith("." + h) for h in ROUTED_HOSTS):
        raise SystemExit(
            f"--base-url points at {host}, which routes to an unrecorded provider. Scores from "
            "routed endpoints are not admissible on this benchmark: the provider, quantization and "
            "serving configuration are not recorded and may change between requests, and CER is "
            "sensitive to all three. Serve the model yourself and point --base-url at that "
            "endpoint (see the module docstring)."
        )


def _prompt_text(inline, prompt_file):
    """Return caller-owned prompt text without trimming or universal-newline conversion."""
    if inline is not None:
        return inline
    text = prompt_file.read_bytes().decode("utf-8")
    if text.startswith("\ufeff"):
        warnings.warn(
            f"Prompt file {prompt_file} starts with a UTF-8 BOM; it is preserved and will be sent "
            "to the model as part of the prompt.", UserWarning, stacklevel=2)
    return text


def _load_request_settings(path):
    """Load caller-owned generation settings while keeping model/messages runner-owned."""
    if path is None:
        return None

    def object_without_duplicates(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"Duplicate request-settings key {key!r}.")
            out[key] = value
        return out

    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number {value!r} is not allowed in request settings.")

    try:
        settings = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=object_without_duplicates,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read --request-settings-file {path} as JSON: {exc}") from exc
    if not isinstance(settings, dict):
        raise ValueError("--request-settings-file must contain one JSON object.")
    unknown = set(settings) - ALLOWED_REQUEST_SETTINGS
    if unknown:
        raise ValueError(
            f"Unsupported request setting(s): {sorted(unknown)}. Put provider-specific fields "
            "inside extra_body; model and messages are owned by the runner."
        )

    def check_finite(value, location="request settings"):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite number at {location} is not allowed.")
        if isinstance(value, dict):
            for key, child in value.items():
                check_finite(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                check_finite(child, f"{location}[{index}]")

    check_finite(settings)
    effective = {**DEFAULT_REQUEST_SETTINGS, **settings}
    if isinstance(effective["max_tokens"], bool) or not isinstance(effective["max_tokens"], int) \
            or effective["max_tokens"] <= 0:
        raise ValueError("max_tokens must be a positive integer.")
    for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
        if key in effective and (isinstance(effective[key], bool)
                                 or not isinstance(effective[key], (int, float))):
            raise ValueError(f"{key} must be a number.")
    if not 0 <= effective["temperature"] <= 2:
        raise ValueError("temperature must be between 0 and 2.")
    if "top_p" in effective and not 0 <= effective["top_p"] <= 1:
        raise ValueError("top_p must be between 0 and 1.")
    for key in ("frequency_penalty", "presence_penalty"):
        if key in effective and not -2 <= effective[key] <= 2:
            raise ValueError(f"{key} must be between -2 and 2.")
    if "seed" in effective and (isinstance(effective["seed"], bool)
                                or not isinstance(effective["seed"], int)):
        raise ValueError("seed must be an integer.")
    if "stop" in effective:
        stop = effective["stop"]
        if not isinstance(stop, str) and not (
                isinstance(stop, list) and all(isinstance(item, str) for item in stop)):
            raise ValueError("stop must be a string or a list of strings.")
    if "response_format" in effective and not isinstance(effective["response_format"], dict):
        raise ValueError("response_format must be a JSON object.")
    extra_body = effective.get("extra_body")
    if extra_body is not None:
        if not isinstance(extra_body, dict):
            raise ValueError("extra_body must be a JSON object.")
        collisions = set(extra_body) & RESERVED_EXTRA_BODY_FIELDS
        if collisions:
            raise ValueError(
                f"extra_body cannot override runner-owned or first-class fields: {sorted(collisions)}."
            )
    if "logprobs" in effective and not isinstance(effective["logprobs"], bool):
        raise ValueError("logprobs must be a boolean.")
    if "top_logprobs" in effective and (
            isinstance(effective["top_logprobs"], bool)
            or not isinstance(effective["top_logprobs"], int)
            or not 0 <= effective["top_logprobs"] <= 20):
        raise ValueError("top_logprobs must be an integer between 0 and 20.")
    if "reasoning_effort" in effective and not isinstance(effective["reasoning_effort"], str):
        raise ValueError("reasoning_effort must be a string.")
    return effective


def _cache_payload(model, dataset, base_url, prompt, dataset_fingerprint=None,
                   page_set_fingerprint=None, dataset_revision=None, resolved_revision=None,
                   request_settings=None):
    effective = {**DEFAULT_REQUEST_SETTINGS, **(request_settings or {})}
    payload = {
        "version": CACHE_KEY_VERSION,
        "model": model,
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        "resolved_revision": resolved_revision,
        "dataset_fingerprint": dataset_fingerprint,
        "page_set_fingerprint": page_set_fingerprint,
        "split": "train",
        "base_url": (base_url or "").rstrip("/"),
        "prompt": prompt,
        "image_max_edge": IMAGE_MAX_EDGE,
        "image_jpeg_quality": IMAGE_JPEG_QUALITY,
        "max_tokens": effective["max_tokens"],
        "temperature": effective["temperature"],
    }
    additional = {key: value for key, value in effective.items()
                  if key not in {"max_tokens", "temperature"}}
    if additional:
        payload["request_settings"] = additional
    return payload


def page_set_fingerprint(ds):
    """Stable full page-set identity used in both cache and benchmark provenance."""
    return GS.page_set_fingerprint(ds["PageID"])


def cache_path(model, dataset, base_url, *, prompt, cache_dir=CACHE, dataset_fingerprint=None,
               page_set_fingerprint=None, dataset_revision=None, resolved_revision=None,
               request_settings=None):
    """Stable raw-output cache path keyed by dataset content, page set, and OCR request inputs."""
    payload = json.dumps(_cache_payload(
        model, dataset, base_url, prompt, dataset_fingerprint, page_set_fingerprint,
        dataset_revision, resolved_revision, request_settings), sort_keys=True)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return pathlib.Path(cache_dir) / f"{_slug(model)}-{digest}.parquet"


def ocr_one(client, model, img, prompt, request_settings=None):
    """One OCR request; returns the transcription, or an ``__ERR__`` marker for exceptions,
    non-``stop`` finish reasons (length truncation, content filtering, ...), and content-less
    responses — a truncated or filtered response is never scored as a transcription. It is excluded from
# aggregates like an error, but flagged `truncated` so it does not disqualify the model — see
# DESIGN.md, Eligibility."""
    effective = {**DEFAULT_REQUEST_SETTINGS, **(request_settings or {})}
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model, **effective,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url(img)}}]}])
            choice = r.choices[0]
            if choice.finish_reason != "stop":
                return f"__ERR__FinishReason:{choice.finish_reason}"
            if choice.message.content is None:
                return "__ERR__EmptyContent"
            return choice.message.content
        except Exception as e:
            if attempt == 2:
                return f"__ERR__{type(e).__name__}"


def _atomic_write_parquet(frame, path):
    """Replace a cache checkpoint atomically so interruption cannot leave a truncated parquet."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        try:
            pathlib.Path(temporary).unlink()
        except FileNotFoundError:
            pass


def _validate_cached_rows(out, cache, expected_ids):
    required = {"PageID", "ocr"}
    if missing_columns := required - set(out.columns):
        raise ValueError(f"Cached OCR {cache} is missing columns: {sorted(missing_columns)}")
    cached_ids = [GS.canonical_page_id(page_id) for page_id in out["PageID"]]
    if out["PageID"].isna().any() or None in cached_ids or len(set(cached_ids)) != len(cached_ids):
        raise ValueError(f"Cached OCR {cache} has missing or duplicate PageID values; delete it and rerun.")
    invalid_ocr = [index for index, value in out["ocr"].items() if not isinstance(value, str)]
    if invalid_ocr:
        raise ValueError(f"Cached OCR {cache} has null or non-string OCR cells at rows "
                         f"{invalid_ocr[:5]}; delete it and rerun.")
    extra = set(cached_ids) - set(expected_ids)
    if extra:
        raise ValueError(f"Cached OCR {cache} contains {len(extra)} pages outside the selected dataset; "
                         "delete it and rerun.")
    return cached_ids


def run_model(client, model, ds, workers, cache, prompt, checkpoint_every=25, *, retry_errors=True,
              request_settings=None):
    """Resume a raw OCR cache and atomically checkpoint results incorporated by this process.

    Pending rows are drawn lazily, so decoded page images exist only for in-flight requests —
    at full-benchmark scale the dataset's images must never be resident all at once."""
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be a positive integer.")
    if workers <= 0:
        raise ValueError("workers must be a positive integer.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    page_ids = list(ds["PageID"])
    expected_ids = [GS.canonical_page_id(page_id) for page_id in page_ids]
    if None in expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Dataset PageID values must be present and unique after safe canonicalization.")

    cached = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["PageID", "ocr"])
    cached_ids = _validate_cached_rows(cached, cache, expected_ids)
    known = {page_id: ocr for page_id, ocr in zip(cached_ids, cached["ocr"], strict=True)}
    successful = {page_id: ocr for page_id, ocr in known.items() if not ocr.startswith("__ERR__")}
    retained = successful if retry_errors else known
    if cache.exists():
        error_action = "to retry" if retry_errors else "retained without retry"
        print(f"  [{model}] {len(successful)} successful cached; "
              f"{len(known) - len(successful)} errors {error_action}")

    rows = (row for row in ds
            if GS.canonical_page_id(row["PageID"]) not in retained)
    current = dict(known)
    volumes = dict(zip(expected_ids, ds["volume"], strict=True))

    def frame():
        return pd.DataFrame([
            {"PageID": page_id, "volume": volumes[canonical], "ocr": current[canonical]}
            for page_id, canonical in zip(page_ids, expected_ids, strict=True) if canonical in current
        ])

    completed = 0
    executor = ThreadPoolExecutor(max_workers=workers)
    in_flight = {}
    exhausted = False
    max_in_flight = workers * 2

    def fill_queue():
        nonlocal exhausted
        while not exhausted and len(in_flight) < max_in_flight:
            try:
                row = next(rows)
            except StopIteration:
                exhausted = True
                break
            if request_settings is None:
                future = executor.submit(ocr_one, client, model, row["image"], prompt)
            else:
                future = executor.submit(
                    ocr_one, client, model, row["image"], prompt, request_settings=request_settings,
                )
            in_flight[future] = row

    try:
        fill_queue()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                row = in_flight.pop(future)
                text = future.result()
                if not isinstance(text, str):
                    raise ValueError(f"OCR inference returned a non-string value for PageID={row['PageID']!r}.")
                current[GS.canonical_page_id(row["PageID"])] = text
                completed += 1
                if completed % checkpoint_every == 0:
                    _atomic_write_parquet(frame(), cache)
            fill_queue()
    except BaseException:
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        if completed:
            _atomic_write_parquet(frame(), cache)
        raise
    else:
        executor.shutdown(wait=True)
        if completed:
            _atomic_write_parquet(frame(), cache)

    missing = set(expected_ids) - set(current)
    if missing:
        raise ValueError(f"Raw OCR inference did not produce {len(missing)} selected dataset pages.")
    out = frame()
    final_ids = _validate_cached_rows(out, cache, expected_ids)
    if final_ids != expected_ids:
        raise ValueError("Final raw OCR output is not in the selected dataset's exact page order.")
    _atomic_write_parquet(out, cache)
    print(f"  [{model}] ran {completed}, retained {len(out) - completed} -> {cache}")
    return out


def pending_page_count(cache, expected_ids, *, retry_errors=True):
    """How many pages this invocation would actually request for one model.

    Best-effort by design: an absent or unreadable cache counts as nothing retained, so the number
    is an upper bound on spend and never understates what a confirmation is authorizing. Strict
    validation of a malformed cache stays in :func:`run_model`, which fails before any request.
    """
    if not cache.exists():
        return len(expected_ids)
    try:
        cached = pd.read_parquet(cache)
        retained = {
            GS.canonical_page_id(page_id)
            for page_id, ocr in zip(cached["PageID"], cached["ocr"], strict=True)
            if isinstance(ocr, str) and not (retry_errors and ocr.startswith("__ERR__"))
        }
    except (OSError, ValueError, KeyError):
        return len(expected_ids)
    return len(set(expected_ids) - retained)


def check_combined_out(out, scorecards_dir=SCORECARDS):
    """Reject a combined --out that would land inside the directory the board globs.

    The per-model scorecards already cover every row, so a combined copy in the same directory is a
    second copy of every (model, page_id) and the default board fails its duplicate check.
    """
    if out is None:
        return
    destination = pathlib.Path(out).resolve()
    directory = pathlib.Path(scorecards_dir).resolve()
    if destination.parent == directory or directory in destination.parents:
        raise SystemExit(
            f"--out {out} resolves inside {scorecards_dir}, where this runner already writes one "
            "scorecard per model. A combined copy there duplicates every (model, page_id) row and "
            "the default leaderboard would refuse to build. Choose a path outside that directory.")


def board_blockers(models, targets, benchmark, scorecards_dir=SCORECARDS):
    """Every reason ``leaderboard.py`` would refuse to build a board over this directory afterwards.

    The preflight's promise is that paying for a run yields a board, and the default board globs the
    whole directory — so it is not enough to look for the requested model names in files that happen
    to parse. Any file the glob picks up must be readable, carry the columns the board requires,
    score no model this run rewrites, and share this run's global provenance; otherwise the campaign
    is paid for and the board still fails. Returns ``{path: [reasons]}``, empty when the board can build.
    """
    target_paths = {pathlib.Path(target).resolve() for target in targets}
    wanted = set(models)
    expected_versions = RPT.current_global_provenance()
    expected_benchmark = RPT.stable_provenance(benchmark)
    blockers = {}
    for path in sorted(pathlib.Path(scorecards_dir).glob("*.parquet")):
        if path.resolve() in target_paths:
            continue  # this run's own output is rewritten in place, not a pre-existing blocker
        reasons = []
        try:
            existing = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - any unreadable file breaks the board's glob
            blockers[path] = [f"cannot be read ({type(exc).__name__}), and the board reads every "
                              "parquet in this directory"]
            continue
        if missing := {"model", "page_id"} - set(existing.columns):
            reasons.append(f"is missing board-required column(s): {', '.join(sorted(missing))}")
        if "model" in existing.columns and (overlap := wanted & set(existing["model"].dropna().unique())):
            reasons.append("already scores " + ", ".join(sorted(overlap))
                           + " (duplicate rows once this run writes them again)")
        for field, expected in expected_versions.items():
            if field not in existing.columns:
                reasons.append(f"is missing provenance column {field!r}")
            elif found := set(existing[field].dropna().unique()) - {expected}:
                reasons.append(f"carries {field} {', '.join(sorted(map(str, found)))} "
                               f"but this run stamps {expected!r}")
        if "benchmark_provenance" not in existing.columns:
            reasons.append("is missing provenance column 'benchmark_provenance'")
        elif any(RPT.stable_provenance(value) != expected_benchmark
                 for value in existing["benchmark_provenance"].dropna().unique()):
            reasons.append("was scored against a different benchmark (mixed benchmark provenance)")
        if reasons:
            blockers[path] = reasons
    return blockers


def archive_scorecards(paths, scorecards_dir=SCORECARDS, *, now=None):
    """Move blocking scorecards out of the globbed directory, keeping them recoverable.

    Superseded scorecards are moved rather than deleted: they are derived artifacts, but the raw
    OCR they came from may no longer be reachable (a cache-key bump retires old caches), so the
    numbers can be the only surviving record of a run.
    """
    stamp = (now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    destination = pathlib.Path(scorecards_dir) / f"archived-{stamp}"
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        os.replace(path, destination / pathlib.Path(path).name)
    return destination


def _preflight(plan, max_tokens, workers, benchmark, *, assume_yes, interactive, ask=input,
               scorecards_dir=SCORECARDS, archive_blocking=False):
    """Refuse anything that would break the board, show the real page-level spend, and take consent —
    all before the first paid request. Non-interactive runs (jobs, CI) proceed after the preview."""
    blockers = board_blockers([model for model, _, _ in plan], [target for _, _, target in plan],
                              benchmark, scorecards_dir)
    if blockers:
        listing = "\n".join(f"  {path}\n    - " + "\n    - ".join(reasons)
                            for path, reasons in blockers.items())
        if not archive_blocking:
            raise SystemExit(
                f"aborted before any paid request: {scorecards_dir} holds scorecards that would stop "
                f"the default leaderboard building after this run:\n{listing}\n"
                "Re-score them with the current scorer, move them aside, or rerun with "
                "--archive-blocking-scorecards to set them aside automatically.")
        archived = archive_scorecards(blockers, scorecards_dir)
        print(f"superseded scorecards moved to {archived}:\n{listing}")
    total = sum(pages for _, pages, _ in plan)
    for model, pages, _ in plan:
        print(f"  {model}: {pages} page(s) to request")
    print(f"planned spend: {total} request(s) across {len(plan)} model(s); "
          f"max_tokens={max_tokens}, workers={workers}")
    if assume_yes or not interactive:
        return
    if total == 0:
        return  # nothing to pay for; resuming a complete run needs no consent
    try:
        answer = ask("proceed with paid requests? [y/N] ")
    except EOFError:
        raise SystemExit("aborted before any paid request") from None
    if answer.strip().lower() not in {"y", "yes"}:
        raise SystemExit("aborted before any paid request")


def score_rows(model, ocr_df, gt_by_page, run_provenance, benchmark_provenance):
    rows = []
    for r in ocr_df.itertuples():
        text = r.ocr
        gt = gt_by_page[GS.canonical_page_id(r.PageID)]
        row = GS.score_gt_row(gt, text, model=model, err=text.startswith("__ERR__"),
                              truncated=GS.is_truncated(text), page_id=gt["PageID"])
        row["run_provenance"] = run_provenance
        row["benchmark_provenance"] = benchmark_provenance
        # Live completions are scored as returned — no normalize_outputs pass — and "0"
        # says so explicitly. The board refuses to mix these with post-processed rows;
        # to combine, cache the raw and re-score through consolidate/normalize.
        row["postproc_version"] = "0"
        rows.append(row)
    return rows


def main():
    from datasets import load_dataset
    from openai import OpenAI

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="davanstrien/bhl-eval-impact-sample")
    ap.add_argument("--dataset-revision", default=None,
                    help="optional requested HF dataset revision pinned in benchmark/cache provenance")
    ap.add_argument("--models", required=True)
    ap.add_argument("--base-url", required=True,
                    help="OpenAI-compatible endpoint you serve yourself (an exposed HF Job, a local "
                         "vLLM/SGLang server, ...). Router endpoints are not admissible — see the "
                         "module docstring.")
    ap.add_argument("--api-key-env", default="HF_TOKEN",
                    help="env var holding the API key; use a name that is unset for a keyless local server")
    prompts = ap.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", help="exact prompt text (an explicit empty string is allowed)")
    prompts.add_argument("--prompt-file", type=pathlib.Path,
                         help="UTF-8 file read exactly, without trimming or newline changes")
    ap.add_argument(
        "--request-settings-file", type=pathlib.Path,
        help="JSON generation settings merged over max_tokens=16384 and temperature=0; contents are run identity",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="atomically save partial raw OCR after this many completed pages (not cache identity)")
    ap.add_argument("--no-retry-errors", action="store_true",
                    help="retain cached error rows without another request (operational; not cache identity)")
    ap.add_argument("--out", default=None,
                    help="optional combined scorecard parquet; per-model scorecards always land in "
                         "data/scorecards/ under their cache identity (stale identities for the same "
                         "model must be deleted before combining, or the board rejects the duplicate)")
    ap.add_argument("--yes", action="store_true",
                    help="skip the interactive spend confirmation")
    ap.add_argument("--archive-blocking-scorecards", action="store_true",
                    help="move scorecards that would break the board into a timestamped archive/ "
                         "subdirectory instead of aborting (they are moved, never deleted)")
    args = ap.parse_args()

    _reject_routed_endpoint(args.base_url)
    prompt = _prompt_text(args.prompt, args.prompt_file)
    request_settings = _load_request_settings(args.request_settings_file)
    check_combined_out(args.out)
    resolved_revision = GS.resolve_dataset_revision(args.dataset, args.dataset_revision)
    ds = load_dataset(args.dataset, revision=resolved_revision, split="train")
    page_ids = ds["PageID"]
    canonical_page_ids = [GS.canonical_page_id(page_id) for page_id in page_ids]
    if any(pd.isna(page_id) for page_id in page_ids) or None in canonical_page_ids \
            or len(set(canonical_page_ids)) != len(ds):
        raise ValueError("Dataset PageID values must be present and unique after safe canonicalization.")
    gt_rows = ds.remove_columns("image") if "image" in (getattr(ds, "column_names", None) or []) else ds
    gt_by_page = {GS.canonical_page_id(r["PageID"]): r for r in gt_rows}
    benchmark = GS.benchmark_provenance(
        args.dataset, args.dataset_revision, ds, resolved_revision=resolved_revision)
    dataset_fingerprint = benchmark["dataset_fingerprint"]
    pages_fingerprint = benchmark["full_page_set_fingerprint"]
    print(f"{len(ds)} pages across {len(set(ds['volume']))} books via {args.base_url}\n")

    client = OpenAI(base_url=args.base_url, api_key=os.environ.get(args.api_key_env, "EMPTY"), timeout=120)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    effective_settings = request_settings or DEFAULT_REQUEST_SETTINGS
    retry_errors = not args.no_retry_errors

    runs = []
    for model in models:
        run_provenance = _cache_payload(
            model, args.dataset, args.base_url, prompt, dataset_fingerprint, pages_fingerprint,
            args.dataset_revision, benchmark["resolved_revision"], request_settings)
        run_provenance.update(runner="run_openai")
        cache = cache_path(
            model, args.dataset, args.base_url, prompt=prompt,
            dataset_fingerprint=dataset_fingerprint, page_set_fingerprint=pages_fingerprint,
            dataset_revision=args.dataset_revision, resolved_revision=benchmark["resolved_revision"],
            request_settings=request_settings)
        runs.append((model, run_provenance, cache, SCORECARDS / f"{cache.stem}.parquet"))

    _preflight(
        [(model, pending_page_count(cache, canonical_page_ids, retry_errors=retry_errors), scorecard)
         for model, _, cache, scorecard in runs],
        effective_settings["max_tokens"], args.workers, benchmark,
        assume_yes=args.yes, interactive=sys.stdin.isatty(),
        archive_blocking=args.archive_blocking_scorecards)

    all_rows = []
    for model, run_provenance, cache, scorecard in runs:
        print(f"running {model} ...")
        output = run_model(
            client, model, ds, args.workers, cache, prompt,
            checkpoint_every=args.checkpoint_every, retry_errors=retry_errors,
            request_settings=request_settings,
        )
        rows = score_rows(model, output, gt_by_page, run_provenance, benchmark)
        all_rows += rows
        _atomic_write_parquet(pd.DataFrame(GS.flat_scorecard_rows(rows)), scorecard)
        print(f"  [{model}] scorecard -> {scorecard}")

    result = RPT.report(all_rows, expected_page_ids=page_ids)
    print("\n" + RPT.format_report(result))
    if args.out is not None:
        flat = pd.DataFrame(GS.flat_scorecard_rows(all_rows))
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        flat.to_parquet(args.out, index=False)
        print(f"\ncombined per-page scorecard -> {args.out}")


if __name__ == "__main__":
    main()
