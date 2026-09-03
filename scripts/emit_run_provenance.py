# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface-hub", "pandas", "pyarrow"]
# ///
"""Generate one run-provenance JSON per model from the driver's own SERVING dict.

RESULTS.md's central claim is that every score on this board came from a run we
controlled: a pinned container image, a pinned model checkpoint, a pinned script commit,
and the id of the job that produced it. Until now those files were HAND-AUTHORED, which
is the one place a provenance claim can quietly drift from what actually ran — a
transcription slip in a JSON file is invisible, and it is exactly the failure the
withdrawn router scores were withdrawn for.

So nothing here is typed. Every value is READ from a source of record:

  SERVING dict + PROMPT   the driver file, parsed as a literal (never imported — the
                          drivers pull vLLM-scale dependencies we do not need here)
  model checkpoint sha    data/full-2026-08/model-revisions.json (sticky, resolved at
                          emit time by scripts/emit_launch_commands.py)
  job id / image / flavor / the HF Jobs record itself, fetched live by job id
  command / durations
  output row counts       the consolidated parquet the scorer actually read

If a driver's SERVING changes, or a job is re-run, regenerating picks it up. The file
cannot disagree with the run unless the run itself was not recorded.

  uv run scripts/emit_run_provenance.py --job-map data/full-2026-08/job-map.tsv \
      --out data/full-2026-08/provenance
"""

import argparse
import ast
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Rows whose producer is not a saturate driver: the model runs in-process rather than behind an
# HTTP endpoint a pump can drive. The value is the driver filename, for the rows where it is not
# simply `<slug>-port.py`.
#
# Membership also decides how the `producer` line reads. That line used to say "via saturate" for
# every row including tesseract, which was untrue and exactly the kind of quiet drift between the
# record and the run that this script exists to prevent — the whole file is generated so that no
# claim in it can be a typo. A row here is described as in-process instead.
NON_SATURATE = {"tesseract": "tesseract-port.py", "kraken-ppocrv6": "kraken-ppocrv6-port.py"}


def literal_from_module(path: pathlib.Path, name: str):
    """Read a module-level literal by AST, without importing the module.

    Importing a driver would pull `saturate`, `pillow` and friends into this script for no
    benefit, and would run module-level asserts that assume a GPU context.
    """
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return None
    return None


def job_record(job_id: str):
    from huggingface_hub import HfApi

    api = HfApi()
    job = api.inspect_job(job_id=job_id)
    raw = job.__dict__ if hasattr(job, "__dict__") else dict(job)
    return json.loads(json.dumps(raw, default=str))


def output_identity(slug: str):
    """Row count and content hash of the consolidated table the scorer actually read."""
    import pandas as pd

    path = ROOT / "data" / "full-2026-08" / "consolidated" / f"{slug}.parquet"
    if not path.exists():
        return {"consolidated": None, "note": "no consolidated table found for this slug"}
    frame = pd.read_parquet(path, columns=["PageID"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "consolidated": str(path.relative_to(ROOT)),
        "rows": int(len(frame)),
        "unique_page_ids": int(frame.PageID.nunique()),
        "sha256": digest,
    }


def build(slug, model, job_id, model_revision, plan, revisions):
    driver_name = NON_SATURATE.get(slug, f"{slug}-port.py")
    driver = ROOT / "drivers" / driver_name
    serving = literal_from_module(driver, "SERVING") if driver.exists() else None
    prompt = literal_from_module(driver, "PROMPT") if driver.exists() else None
    prompts = literal_from_module(driver, "PROMPTS") if driver.exists() else None

    mechanism = "in-process (no saturate pump)" if slug in NON_SATURATE else "via saturate"
    record = {
        "producer": f"bhl-ocr-eval drivers/{driver_name} {mechanism} on HF Jobs",
        "run_id": plan["run_id"],
        "script": f"drivers/{driver_name}",
        "script_commit": plan["driver_revision"],
        "benchmark_dataset": plan["benchmark"]["repo"],
        "benchmark_resolved_revision": plan["benchmark"]["revision"],
        "benchmark_page_set_fingerprint": plan["benchmark"]["page_set_fingerprint"],
        "model": model,
        "model_revision": model_revision or revisions.get(model),
        "prompt": prompt if prompt is not None else prompts,
        "settings": serving,
        # NOT from the launch plan: that is a launch-TIME document and goes stale the moment
        # a post-processing rule changes (it still said "1" after POSTPROC 3 shipped). The
        # normalizer module is the source of record for its own version.
        "postproc_version": literal_from_module(
            ROOT / "runners" / "normalize_outputs.py", "POSTPROC_VERSION"),
        "norm_version": plan["scoring"]["norm_version"],
        "scorer_version": plan["scoring"]["scorer_version"],
    }
    if serving is None:
        record["settings_note"] = (
            "No SERVING dict: this row is not a saturate/vLLM driver. Serving parameters "
            "are whatever the job command and image pin — see `job` below."
        )
    try:
        record["job"] = job_record(job_id)
    except Exception as exc:  # a missing job record must be visible, never silently absent
        record["job"] = {"id": job_id, "error": f"{type(exc).__name__}: {exc}"}
    record["outputs"] = output_identity(slug)
    return record


def main():
    ap = argparse.ArgumentParser(description="Generate per-model run-provenance from SERVING.")
    ap.add_argument("--job-map", default=str(ROOT / "data" / "full-2026-08" / "job-map.tsv"),
                    help="TSV: slug<TAB>job_id<TAB>model_revision")
    ap.add_argument("--plan", default=str(ROOT / "data" / "full-2026-08" / "launch-plan.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "full-2026-08" / "provenance"))
    args = ap.parse_args()

    plan = json.loads(pathlib.Path(args.plan).read_text())
    rev_path = pathlib.Path(args.plan).parent / "model-revisions.json"
    revisions = json.loads(rev_path.read_text()) if rev_path.exists() else {}
    slug_to_model = {j["driver"].replace("-port.py", "").replace(".py", ""): j["model"]
                     for j in plan["jobs"]}

    # Resolved, because the summary lines below print paths relative to ROOT and `relative_to`
    # raises on a relative argument — passing `--out data/…` used to crash AFTER writing every
    # file, which looks like a failed run and is not one.
    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for line in pathlib.Path(args.job_map).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        slug, job_id = parts[0], parts[1]
        model_revision = parts[2] if len(parts) > 2 else None
        model = slug_to_model.get(slug)
        if model is None:
            print(f"  SKIP {slug}: not in the launch plan")
            continue
        record = build(slug, model, job_id, model_revision, plan, revisions)
        path = out_dir / f"{slug}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n")
        pinned = "pinned" if record["model_revision"] else "NO MODEL REVISION"
        served = "SERVING" if record["settings"] else "no SERVING"
        print(f"  {slug:20s} {pinned:18s} {served:12s} -> {path.relative_to(ROOT)}")
        written.append(slug)
    print(f"\n{len(written)} provenance files -> {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
