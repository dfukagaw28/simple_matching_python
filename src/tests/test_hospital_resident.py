import copy
import random
from io import StringIO

import numpy as np
import pytest

from simplematching.hospital_resident import HospitalResident as HR
from simplematching.utils import normalize
from matching.games import HospitalResident as HR2

# The seeds of the instances that the randomized tests below run on.  They are
# fixed so that a failure can be reproduced and located, and grouped into
# batches so that a run reports one case per batch instead of one per seed.
# Widen NUM_SEEDS to test more instances.
NUM_SEEDS = 1000
BATCH_SIZE = 100

SEED_BATCHES = [
    range(start, min(start + BATCH_SIZE, NUM_SEEDS))
    for start in range(0, NUM_SEEDS, BATCH_SIZE)
]


def _batch_id(seeds: range) -> str:
    """Name a batch of seeds after the seeds it holds, e.g. 'seeds0-99'."""
    return f'seeds{seeds.start}-{seeds.stop - 1}'


seed_batches = pytest.mark.parametrize('seeds', SEED_BATCHES, ids=_batch_id)


def _inverse(order: list[int]) -> list[int]:
    """Map each old number to the new one it is renumbered to."""
    inverse = [0] * len(order)
    for new, old in enumerate(order):
        inverse[old] = new
    return inverse


def _relabel(hr: HR, rng: random.Random) -> HR:
    """Renumber the residents and the hospitals at random.

    The result is an instance isomorphic to the given one, which normalize()
    has to turn back into the very same instance.
    """
    hr = copy.deepcopy(hr)

    resident_order = rng.sample(range(hr.num_residents), hr.num_residents)
    hospital_order = rng.sample(range(hr.num_hospitals), hr.num_hospitals)
    new_resident = _inverse(resident_order)
    new_hospital = _inverse(hospital_order)

    hr.resident_prefs = [
        [new_hospital[h] for h in hr.resident_prefs[r]] for r in resident_order
    ]
    if hr.tie_last:
        hr.resident_prefs_rest = [
            [new_hospital[h] for h in hr.resident_prefs_rest[r]]
            for r in resident_order
        ]
    hr.hospital_prefs = [
        [new_resident[r] for r in hr.hospital_prefs[h]] for h in hospital_order
    ]
    hr.capacities = [hr.capacities[h] for h in hospital_order]

    return hr


def _popularity_keys(prefs: list[list[int]], num_items: int) -> list[tuple]:
    """(number of firsts, number of seconds, ...) for each item."""
    counts = [[0] * num_items for _ in range(num_items)]
    for pref in prefs:
        for position, item in enumerate(pref):
            counts[item][position] += 1
    return [tuple(counts[item]) for item in range(num_items)]


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


@seed_batches
def test_same_result_1(seeds):
    for seed in seeds:
        hr = HR.generate(60, 4, seed)

        m1, _ = hr.solve()

        game = HR2.create_from_dictionaries(*hr.to_dicts())
        game.solve()

        m2 = [r.matching.name for r in game.residents]

        assert m1 == m2, f'seed={seed}'


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


def test_set_capacities_tight():
    hr = HR.generate(300, 24, 12345678)

    hr.set_capacities(tight=True)

    # The hospitals hold exactly the 300 residents, 12 of them holding one
    # more than the 300 // 24 the others hold
    assert len(hr.capacities) == 24
    assert sum(hr.capacities) == 300
    assert sorted(set(hr.capacities)) == [12, 13]
    assert hr.capacities.count(13) == 300 % 24

    # Everyone is matched, the lists being complete and the seats enough
    m_residents, m_hospitals = hr.solve()
    assert m_residents.count(-1) == 0
    for h in range(hr.num_hospitals):
        assert len(m_hospitals[h]) == hr.capacities[h]


def test_set_capacities_tight_when_it_divides_evenly():
    hr = HR.generate(300, 25, 12345678)

    hr.set_capacities(tight=True)

    assert hr.capacities == [12] * 25


def test_set_capacities_tight_with_fewer_residents_than_hospitals():
    hr = HR.generate(10, 24, 12345678)

    hr.set_capacities(tight=True)

    assert sum(hr.capacities) == 10
    assert sorted(set(hr.capacities)) == [0, 1]

    m_residents, m_hospitals = hr.solve()
    assert m_residents.count(-1) == 0
    for h in range(hr.num_hospitals):
        assert len(m_hospitals[h]) <= hr.capacities[h]


def test_set_capacities_tight_gives_the_spare_seats_to_the_popular_hospitals():
    hr = HR.generate(300, 24, 12345678)

    hr.set_capacities(tight=True)

    # Every hospital that got a spare seat is at least as popular as every
    # hospital that did not
    keys = _popularity_keys(hr.resident_prefs, hr.num_hospitals)
    spared = [keys[h] for h in range(24) if hr.capacities[h] == 13]
    others = [keys[h] for h in range(24) if hr.capacities[h] == 12]
    assert min(spared) >= max(others)


def test_set_capacities_tight_rejects_a_capacity_of_its_own():
    hr = HR.generate(12, 4, 12345678)

    for capacities in (5, [5, 2, 8, 1]):
        with pytest.raises(AssertionError, match="Cannot specify both"):
            hr.set_capacities(capacities, tight=True)


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
@seed_batches
def test_tie_last_leaves_residents_unmatched_when_capacity_is_short(seeds, capacity):
    # A resident is unmatched only if every hospital has rejected them, and a
    # hospital that has rejected someone is full and stays full.  Hence the
    # number of unmatched residents is exactly the number of residents that
    # the hospitals cannot hold, whatever the preference lists are.
    for seed in seeds:
        hr = HR.generate(12, 4, seed)
        hr.set_capacities(capacity)
        hr.cut_resident_prefences(1, tie_last=True)

        m_residents, m_hospitals = hr.solve()
        assert m_residents.count(-1) == 12 - 4 * capacity, f'seed={seed}'
        for h in range(hr.num_hospitals):
            assert len(m_hospitals[h]) == capacity, f'seed={seed}'


def test_normalize_orders_by_popularity():
    hr = HR.generate(12, 4, 12345678)
    resident_order, hospital_order = normalize(hr)

    assert hospital_order == [2, 1, 0, 3]
    assert resident_order == [2, 1, 3, 10, 0, 6, 11, 9, 5, 7, 4, 8]

    # The hospital ranked first by the most residents comes first, ties broken
    # by the number of the second choices, then of the third ones, and so on
    keys = _popularity_keys(hr.resident_prefs, hr.num_hospitals)
    assert keys == sorted(keys, reverse=True)
    assert [key[0] for key in keys] == [5, 3, 2, 2]

    # ... and the same holds for the residents in the hospitals' lists
    keys = _popularity_keys(hr.hospital_prefs, hr.num_residents)
    assert keys == sorted(keys, reverse=True)
    assert [key[0] for key in keys] == [2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_normalize_is_idempotent():
    hr = HR.generate(60, 8, 12345678)
    normalize(hr)

    prefs = copy.deepcopy((hr.resident_prefs, hr.hospital_prefs, hr.capacities))
    resident_order, hospital_order = normalize(hr)

    assert resident_order == list(range(hr.num_residents))
    assert hospital_order == list(range(hr.num_hospitals))
    assert (hr.resident_prefs, hr.hospital_prefs, hr.capacities) == prefs


@seed_batches
def test_normalize_keeps_the_matching(seeds):
    for seed in seeds:
        hr = HR.generate(40, 6, seed)
        hr.set_capacities([5, 6, 7, 8, 9, 10])
        m_residents, _ = hr.solve()

        normalized = copy.deepcopy(hr)
        resident_order, hospital_order = normalize(normalized)
        new_resident = _inverse(resident_order)
        new_hospital = _inverse(hospital_order)

        # The instance is renumbered, not altered: the matching of the
        # normalized instance is the matching of the original one, renumbered
        m_normalized, _ = normalized.solve()
        assert m_normalized == [
            new_hospital[h] if h >= 0 else -1
            for h in (m_residents[r] for r in resident_order)
        ], f'seed={seed}'

        # The capacity follows its hospital
        assert normalized.capacities == [hr.capacities[h] for h in hospital_order], \
            f'seed={seed}'


@seed_batches
def test_normalize_gives_isomorphic_instances_the_same_form(seeds):
    for seed in seeds:
        hr = HR.generate(40, 6, seed)
        normalize(hr)

        relabelled = _relabel(hr, random.Random(seed))
        normalize(relabelled)

        assert relabelled.resident_prefs == hr.resident_prefs, f'seed={seed}'
        assert relabelled.hospital_prefs == hr.hospital_prefs, f'seed={seed}'


@seed_batches
def test_normalize_with_tie_last(seeds):
    for seed in seeds:
        hr = HR.generate(40, 6, seed)
        hr.cut_resident_prefences(2, tie_last=True, seed=seed)
        m_residents, _ = hr.solve()

        normalized = copy.deepcopy(hr)
        resident_order, hospital_order = normalize(normalized)
        new_hospital = _inverse(hospital_order)

        # The hospitals tied at the last rank are renumbered as well, and the
        # residents keep being matched to the very same hospitals
        m_normalized, _ = normalized.solve()
        assert m_normalized == [
            new_hospital[h] if h >= 0 else -1
            for h in (m_residents[r] for r in resident_order)
        ], f'seed={seed}'
        assert all(
            sorted(prefs + rest) == list(range(hr.num_hospitals))
            for prefs, rest
            in zip(normalized.resident_prefs, normalized.resident_prefs_rest)
        ), f'seed={seed}'


@seed_batches
def test_same_result_tie_last(seeds):
    for seed in seeds:
        hr = HR.generate(60, 4, seed)
        hr.cut_resident_prefences(2, tie_last=True)

        m1, _ = hr.solve()

        game = HR2.create_from_dictionaries(*hr.to_dicts())
        game.solve()

        m2 = [r.matching.name if r.matching else -1 for r in game.residents]

        assert m1 == m2, f'seed={seed}'


@seed_batches
def test_same_result_2(seeds):
    for seed in seeds:
        hr = HR.generate(60, 4, seed)
        hr.set_capacities(10)

        m1, _ = hr.solve()

        game = HR2.create_from_dictionaries(*hr.to_dicts())
        game.solve()

        m2 = [r.matching.name if r.matching else -1 for r in game.residents]

        assert m1 == m2, f'seed={seed}'
