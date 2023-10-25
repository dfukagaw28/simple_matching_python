from io import StringIO

import numpy as np

from simplematching.hospital_resident import HospitalResident as HR
from matching.games import HospitalResident as HR2


def test_can_generate():
    hr = HR.generate(12, 4, 12345678)

    assert hr.num_residents == 12
    assert hr.num_hospitals == 4

    prefs_expected = np.loadtxt(StringIO("""
3 2 1 0
2 3 1 0
3 1 2 0
3 2 1 0
1 0 2 3
0 2 3 1
3 1 0 2
3 2 0 1
0 2 3 1
3 0 1 2
3 1 0 2
1 2 0 3
"""))
    #assert np.array_equal(np.array(hr.resident_prefs), prefs_expected)
    assert np.array(hr.resident_prefs).tolist() == prefs_expected.tolist()

    prefs_expected = np.loadtxt(StringIO("""
11 4 1 8 2 9 5 0 7 3 10 6
9 2 7 4 1 0 8 5 10 6 3 11
10 8 1 5 7 4 0 11 6 3 2 9
2 0 3 11 7 1 5 6 8 10 4 9
"""))
    #assert np.array_equal(np.array(hr.hospital_prefs), prefs_expected)
    assert np.array(hr.hospital_prefs).tolist() == prefs_expected.tolist()

    assert hr.capacities == [3, 3, 3, 3]


def test_can_solve():
    hr = HR.generate(12, 4, 12345678)
    m_residents, m_hospitals = hr.solve()

    assert m_residents == [3, 2, 3, 3, 1, 0, 1, 2, 0, 0, 1, 2]
    assert m_hospitals == [[8, 9, 5], [4, 10, 6], [1, 7, 11], [2, 0, 3]]


def test_same_result_1():
    for _ in range(1000):
        hr = HR.generate(60, 4)

        m1, _ = hr.solve()

        game = HR2.create_from_dictionaries(*hr.to_dicts())
        game.solve()

        m2 = [r.matching.name for r in game.residents]

        assert m1 == m2


def test_same_result_2():
    for _ in range(1000):
        hr = HR.generate(60, 4)
        hr.capacities = [10] * 4

        m1, _ = hr.solve()

        game = HR2.create_from_dictionaries(*hr.to_dicts())
        game.solve()

        m2 = [r.matching.name if r.matching else -1 for r in game.residents]

        assert m1 == m2
