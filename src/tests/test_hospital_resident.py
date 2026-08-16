from io import StringIO

import numpy as np
import pytest

from simplematching.hospital_resident import HospitalResident as HR
from matching.games import HospitalResident as HR2

# The seeds of the instances that the randomized tests below run on.
# They are fixed so that a failure can be reproduced and located; widen the
# range to test more instances.
SEEDS = list(range(100))


def test_can_generate():
    hr = HR.generate(12, 4, 12345678)

    assert hr.num_residents == 12
    assert hr.num_hospitals == 4

    prefs_expected = np.loadtxt(StringIO("""
2 3 1 0
0 1 3 2
3 1 2 0
1 2 3 0
2 0 1 3
1 2 3 0
0 3 2 1
2 3 0 1
2 1 0 3
2 0 1 3
1 2 0 3
3 0 1 2
"""))
    #assert np.array_equal(np.array(hr.resident_prefs), prefs_expected)
    assert np.array(hr.resident_prefs).tolist() == prefs_expected.tolist()

    prefs_expected = np.loadtxt(StringIO("""
2 6 0 5 10 9 7 8 11 4 3 1
2 11 10 9 7 1 0 5 3 4 8 6
3 10 9 2 6 0 4 11 1 7 5 8
1 0 5 9 7 2 11 8 6 4 10 3
"""))
    #assert np.array_equal(np.array(hr.hospital_prefs), prefs_expected)
    assert np.array(hr.hospital_prefs).tolist() == prefs_expected.tolist()

    assert hr.capacities == [3, 3, 3, 3]


def test_can_solve():
    hr = HR.generate(12, 4, 12345678)
    m_residents, m_hospitals = hr.solve()

    assert m_residents == [2, 0, 3, 1, 2, 1, 0, 3, 0, 2, 1, 3]
    assert m_hospitals == [[6, 8, 1], [10, 5, 3], [9, 0, 4], [7, 2, 11]]


@pytest.mark.parametrize('seed', SEEDS)
def test_same_result_1(seed):
    hr = HR.generate(60, 4, seed)

    m1, _ = hr.solve()

    game = HR2.create_from_dictionaries(*hr.to_dicts())
    game.solve()

    m2 = [r.matching.name for r in game.residents]

    assert m1 == m2


def test_set_capacities_uniform():
    hr = HR.generate(12, 4, 12345678)

    hr.set_capacities(5)
    assert hr.capacities == [5, 5, 5, 5]

    hr.set_capacities()
    assert hr.capacities == [3, 3, 3, 3]


def test_set_capacities_non_uniform():
    hr = HR.generate(12, 4, 12345678)

    hr.set_capacities([5, 2, 8, 1])
    assert hr.capacities == [5, 2, 8, 1]

    m_residents, m_hospitals = hr.solve()
    for h in range(hr.num_hospitals):
        assert len(m_hospitals[h]) <= hr.capacities[h]


def test_set_capacities_non_uniform_wrong_length():
    hr = HR.generate(12, 4, 12345678)

    with pytest.raises(AssertionError, match='Expected 4 capacities, got 3'):
        hr.set_capacities([1, 2, 3])


def test_cut_prefs_without_tie_last():
    hr = HR.generate(12, 4, 12345678)
    hr.cut_resident_prefences(1)

    assert hr.tie_last is False
    assert hr.resident_prefs[0] == [2]
    assert hr.resident_prefs_completed() == hr.resident_prefs

    # Every resident applies to a single hospital, hence the residents
    # rejected by it are left unmatched
    m_residents, _ = hr.solve()
    assert m_residents.count(-1) > 0
    assert hr.count_unlisted_matches(m_residents) == 0


def test_cut_prefs_with_tie_last_keep():
    hr = HR.generate(12, 4, 12345678)
    prefs_full = [list(prefs) for prefs in hr.resident_prefs]
    m_expected, _ = hr.solve()

    hr.cut_resident_prefences(1, tie_last=True, tie_break='keep')

    assert hr.tie_last is True
    assert hr.resident_prefs[0] == [2]
    assert hr.resident_prefs_rest[0] == [3, 1, 0]

    # Breaking the tie by the original order leaves the preference lists
    # (and hence the matching) unchanged
    assert hr.resident_prefs_completed() == prefs_full
    m_residents, _ = hr.solve()
    assert m_residents == m_expected


def test_cut_prefs_with_tie_last_random():
    hr = HR.generate(12, 4, 12345678)
    hr.cut_resident_prefences(1, tie_last=True, seed=98765)

    for r in range(hr.num_residents):
        # The hospitals out of the list follow the listed one, each of them
        # exactly once
        completed = hr.resident_prefs_completed()[r]
        assert completed[:1] == hr.resident_prefs[r]
        assert sorted(completed) == list(range(hr.num_hospitals))

    # Every hospital is acceptable, hence every resident is matched
    # (the total capacity is large enough)
    m_residents, _ = hr.solve()
    assert m_residents.count(-1) == 0
    assert hr.count_unlisted_matches(m_residents) > 0


def test_cut_prefs_with_tie_last_is_reproducible():
    m1 = []
    for _ in range(2):
        hr = HR.generate(60, 4, 12345678)
        hr.cut_resident_prefences(2, tie_last=True, seed=42)
        m1.append(hr.solve()[0])
    assert m1[0] == m1[1]


def test_set_tie_last_on_incomplete_instance():
    hr = HR.generate(12, 4, 12345678)
    hr.cut_resident_prefences(2)
    hr.set_tie_last(tie_break='keep')

    assert hr.resident_prefs[0] == [2, 3]
    assert hr.resident_prefs_rest[0] == [0, 1]

    hr.set_tie_last(False)
    assert hr.tie_last is False
    assert hr.resident_prefs_completed() == hr.resident_prefs


@pytest.mark.parametrize('capacity', range(1, 4))
@pytest.mark.parametrize('seed', SEEDS)
def test_tie_last_leaves_residents_unmatched_when_capacity_is_short(seed, capacity):
    # A resident is unmatched only if every hospital has rejected them, and a
    # hospital that has rejected someone is full and stays full.  Hence the
    # number of unmatched residents is exactly the number of residents that
    # the hospitals cannot hold, whatever the preference lists are.
    hr = HR.generate(12, 4, seed)
    hr.set_capacities(capacity)
    hr.cut_resident_prefences(1, tie_last=True)

    m_residents, m_hospitals = hr.solve()
    assert m_residents.count(-1) == 12 - 4 * capacity
    for h in range(hr.num_hospitals):
        assert len(m_hospitals[h]) == capacity


@pytest.mark.parametrize('seed', SEEDS)
def test_same_result_tie_last(seed):
    hr = HR.generate(60, 4, seed)
    hr.cut_resident_prefences(2, tie_last=True)

    m1, _ = hr.solve()

    game = HR2.create_from_dictionaries(*hr.to_dicts())
    game.solve()

    m2 = [r.matching.name if r.matching else -1 for r in game.residents]

    assert m1 == m2


@pytest.mark.parametrize('seed', SEEDS)
def test_same_result_2(seed):
    hr = HR.generate(60, 4, seed)
    hr.set_capacities(10)

    m1, _ = hr.solve()

    game = HR2.create_from_dictionaries(*hr.to_dicts())
    game.solve()

    m2 = [r.matching.name if r.matching else -1 for r in game.residents]

    assert m1 == m2
