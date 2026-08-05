"""Pure, dependency-free helpers of the unit-test generator (the selection/geometry logic).

The end-to-end generator needs the private HF sample + heavy deps (fuzzysearch/rapidfuzz); these
tests pin only the pure functions, which carry the load-bearing decisions: the edit budget, the
distinctiveness filter, and the single-column geometry check that gates cross-region order tests.
"""
import types

import gen_unit_tests as U


def test_max_diffs_is_edit_frac_of_length_min_one():
    assert U.max_diffs(100, 0.04) == 4
    assert U.max_diffs(10, 0.04) == 1     # round(0.4)=0 → floored to the minimum budget of 1
    assert U.max_diffs(0, 0.04) == 1


def test_pr_threshold_is_one_minus_md_over_len():
    assert U.pr_threshold(100, 4) == 0.96
    assert U.pr_threshold(0, 4) == 0.0    # guarded: no division by zero on an empty snippet


def test_is_distinctive_filters_short_digit_and_few_word_snippets():
    assert U.is_distinctive("the tree sparrow is common in eastern asia")   # long, wordy, alpha
    assert not U.is_distinctive("short")                                     # < MIN_CHARS
    assert not U.is_distinctive("a" * 200)                                   # > MAX_CHARS
    assert not U.is_distinctive("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15")       # mostly digits
    assert not U.is_distinctive("one two three four")                        # < MIN_WORDS words


def test_v_overlap():
    assert U.v_overlap((0, 0, 10, 10), (0, 5, 10, 15)) == 5.0   # rows 5-10 shared
    assert U.v_overlap((0, 0, 10, 10), (0, 20, 10, 30)) == 0.0  # disjoint in y


def test_is_single_column_stacked_vs_side_by_side():
    stacked = [(0, 0, 100, 40), (0, 50, 100, 90)]           # one above the other
    assert U.is_single_column(stacked)
    two_col = [(0, 0, 40, 100), (60, 0, 100, 100)]          # side by side, full vertical overlap
    assert not U.is_single_column(two_col)
    slight = [(0, 0, 100, 40), (0, 38, 100, 78)]            # 2px touch, < tol*height → still single
    assert U.is_single_column(slight)


def test_distinct_starts_collapses_overlapping_near_matches():
    # fuzzysearch yields objects with a `.start`; three real occurrences at ~0, ~50, ~100 with
    # jitter within 0.5*len collapsing to one start each
    matches = [types.SimpleNamespace(start=s) for s in (0, 2, 50, 51, 100)]
    assert U.distinct_starts(matches, target_len=20) == [0, 50, 100]
    assert U.distinct_starts([], target_len=20) == []
