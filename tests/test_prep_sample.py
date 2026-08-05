"""Pure sampling contracts; importing prep_sample must not require datasets/network dependencies."""
import gt_score as GS
import prep_sample as P


def _rows():
    rows = []
    for volume, count in (("large", 12), ("small", 2), ("tiny", 1)):
        rows.extend({"id": f"c-{volume}-{i}", "BarCode": volume, "text": "x" * 80}
                    for i in range(count))
        rows.extend({"id": f"s-{volume}-{i}", "BarCode": volume, "text": "x" * 10}
                    for i in range(count))
    return rows


def test_stratum_boundary():
    assert P.classify_sample_stratum("x" * 79, 80) == "sparse_blank"
    assert P.classify_sample_stratum("x" * 80, 80) == "content"


def test_prep_and_score_stratum_classifiers_cannot_drift():
    for threshold in range(0, 6):
        for length in range(0, 9):
            text = "x" * length
            assert P.classify_sample_stratum(text, threshold) == GS.sample_stratum(text, threshold)
    for text in (None, 123, "  ", "x" * 80):
        assert P.classify_sample_stratum(text, 80) == GS.sample_stratum(text, 80)


def test_sampling_is_deterministic_exact_and_includes_both_strata():
    rows = _rows()
    first = P.sample_indices(rows, 11, seed=7)
    assert first == P.sample_indices(rows, 11, seed=7)
    assert len(first) == len(set(first)) == 11
    strata = {P.classify_sample_stratum(rows[index]["text"]) for index in first}
    assert strata == {"content", "sparse_blank"}


def test_sampling_spreads_each_stratum_across_volumes_and_redistributes():
    rows = _rows()
    selected = [rows[index] for index in P.sample_indices(rows, 24, seed=2)]
    for stratum in P.STRATA:
        volumes = {
            row["BarCode"] for row in selected
            if P.classify_sample_stratum(row["text"]) == stratum
        }
        assert volumes == {"large", "small", "tiny"}
    assert len(selected) == 24


def test_sampling_returns_all_source_rows_when_n_is_larger():
    rows = _rows()
    assert P.sample_indices(rows, len(rows) + 100, seed=99) == list(range(len(rows)))


def test_largest_remainder_and_nonempty_guarantee():
    assert P.proportional_quotas({"content": 99, "sparse_blank": 1}, 2,
                                 guarantee_nonempty=True) == {
        "content": 1, "sparse_blank": 1,
    }


def test_source_revision_is_resolved_to_immutable_commit_without_network():
    calls = []

    class Info:
        sha = "deadbeef"

    class Api:
        def dataset_info(self, **kwargs):
            calls.append(kwargs)
            return Info()

    assert P.resolve_source_revision("source/repo", "release-tag", api=Api()) == "deadbeef"
    assert calls == [{"repo_id": "source/repo", "revision": "release-tag"}]


def test_sampler_provenance_fields_are_uniform_record_primitives():
    provenance = P.sampler_provenance(
        seed=7, requested_n=60, source_repo="source/repo", source_revision="deadbeef",
        threshold=80,
    )
    assert provenance == {
        "sampler_version": P.SAMPLER_VERSION,
        "sampler_seed": 7,
        "sampler_requested_n": 60,
        "sampler_source_repo": "source/repo",
        "sampler_source_revision": "deadbeef",
        "sample_stratum_threshold": 80,
    }
