"""Part-file numbering for the kraken driver's resume path.

One assertion, guarding one way to lose pages silently. When a run resumes, `done_ids()` marks
every id in every existing part as done, so a new part written over an existing filename destroys
pages that will never be recomputed. Numbering therefore has to move forward always, never
land on a name that exists.

The driver is loaded by path because its filename is hyphenated and so not importable by name;
it has no third-party imports at module level, so this costs nothing.
"""
import importlib.util
import pathlib

_DRIVER = pathlib.Path(__file__).resolve().parent.parent / "drivers" / "kraken-ppocrv6-port.py"
_spec = importlib.util.spec_from_file_location("kraken_ppocrv6_port", _DRIVER)
# Both are Optional in the stdlib signatures, and a None here would mean the driver was renamed
# or moved — in which case these tests should say so plainly rather than fail later with an
# AttributeError on None.
assert _spec is not None and _spec.loader is not None, f"cannot load driver at {_DRIVER}"
KP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(KP)


def test_first_run_starts_at_zero():
    assert KP.next_part_index([], rank=0) == 0


def test_resume_continues_after_the_highest_index():
    parts = ["run/part-r00-00000.parquet", "run/part-r00-00001.parquet"]
    assert KP.next_part_index(parts, rank=0) == 2


def test_a_hole_in_the_sequence_does_not_reuse_a_live_filename():
    # THE regression this file exists for. Counting gives 2 here, which names an existing file;
    # overwriting it drops pages that done_ids() has already recorded as complete.
    parts = ["run/part-r00-00000.parquet", "run/part-r00-00002.parquet"]
    assert KP.next_part_index(parts, rank=0) == 3


def test_other_ranks_do_not_move_this_rank_forward():
    # Shards share one output prefix; each numbers only its own parts, or concurrent shards
    # would race for the same names.
    parts = [
        "run/part-r00-00000.parquet",
        "run/part-r01-00000.parquet",
        "run/part-r01-00001.parquet",
    ]
    assert KP.next_part_index(parts, rank=0) == 1
    assert KP.next_part_index(parts, rank=1) == 2
    assert KP.next_part_index(parts, rank=2) == 0


def test_unparseable_names_are_ignored_not_guessed_at():
    parts = ["run/part-r00-00000.parquet", "run/part-r00-notanumber.parquet", "run/notes.txt"]
    assert KP.next_part_index(parts, rank=0) == 1


def test_hf_uri_listings_parse_the_same_as_local_paths():
    # In production `part_paths()` returns HfFileSystem glob results, not local paths. The `$`
    # anchor means only the basename can match, so a bucket prefix containing "part-r00-" — or a
    # run directory named after a shard — cannot be mistaken for a part file.
    parts = [
        "buckets/davanstrien/bhl-ocr-runs/kraken-ppocrv6-2026-09/medium/part-r00-00000.parquet",
        "buckets/davanstrien/bhl-ocr-runs/part-r00-99999/medium/part-r00-00001.parquet",
    ]
    assert KP.next_part_index(parts, rank=0) == 2


def test_rank_beyond_two_digits_still_matches_its_own_parts():
    # The filename uses {rank:02d}, which does not truncate: rank 100 writes "part-r100-". The
    # parser builds its pattern the same way, so the two stay in step above 99 shards.
    parts = ["run/part-r100-00000.parquet", "run/part-r10-00007.parquet"]
    assert KP.next_part_index(parts, rank=100) == 1
    assert KP.next_part_index(parts, rank=10) == 8


def test_main_actually_uses_the_helper():
    # The helper being correct is worth nothing if main() computes the index itself — which is
    # exactly the bug this file exists for. Checked at the source level because main() cannot be
    # called without a GPU, a benchmark dataset and a writable bucket.
    import ast

    tree = ast.parse(_DRIVER.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = {n.func.id for n in ast.walk(main)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "next_part_index" in called
