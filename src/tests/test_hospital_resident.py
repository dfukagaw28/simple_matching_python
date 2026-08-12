from io import StringIO

import numpy as np

from simplematching.hospital_resident import HospitalResident as HR
from matching.games import HospitalResident as HR2


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


def test_same_result_1():
    for _ in range(1000):
        hr = HR.generate(60, 4)

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

    try:
        hr.set_capacities([1, 2, 3])
        assert False, 'AssertionError expected'
    except AssertionError:
        pass


def test_same_result_2():
    for _ in range(1000):
        hr = HR.generate(60, 4)
        hr.capacities = [10] * 4

        m1, _ = hr.solve()

        game = HR2.create_from_dictionaries(*hr.to_dicts())
        game.solve()

        m2 = [r.matching.name if r.matching else -1 for r in game.residents]

        assert m1 == m2
