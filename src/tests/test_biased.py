import numpy as np
import pytest

from simplematching.biased import (
    generate_biased_HR,
    generate_correlated_preferences,
)


def test_preferences_are_permutations():
    prefs = generate_correlated_preferences(
        20, 6, alpha=0.5, rng=np.random.default_rng(0))

    assert len(prefs) == 20
    for pref in prefs:
        assert sorted(pref) == list(range(6))


def test_preferences_are_cut_to_the_list_length():
    prefs = generate_correlated_preferences(
        20, 6, alpha=0.5, list_length=3, rng=np.random.default_rng(0))

    for pref in prefs:
        assert len(pref) == 3
        assert len(set(pref)) == 3
        assert all(0 <= h < 6 for h in pref)


@pytest.mark.parametrize('alpha, agree', [(0.0, False), (1.0, True)])
def test_alpha_makes_the_choosers_agree(alpha, agree):
    prefs = generate_correlated_preferences(
        20, 6, alpha=alpha, rng=np.random.default_rng(0))

    # Sharing the whole opinion makes every chooser rank the items alike,
    # sharing none of it makes them disagree
    assert all(pref == prefs[0] for pref in prefs) == agree


@pytest.mark.parametrize('alpha', [0.0, 0.25, 0.5, 0.75, 0.9])
def test_alpha_is_the_correlation_of_the_scores(alpha):
    num_choosers, num_choices = 40, 400
    prefs = generate_correlated_preferences(
        num_choosers, num_choices, alpha, rng=np.random.default_rng(0))

    # Rank of every item in every list
    ranks = np.empty((num_choosers, num_choices))
    for i, pref in enumerate(prefs):
        ranks[i, pref] = np.arange(num_choices)

    correlations = np.corrcoef(ranks)[np.triu_indices(num_choosers, 1)]

    # The scores are standard normal and correlate by alpha, hence the ranks
    # of two choosers correlate by (6 / pi) * arcsin(alpha / 2)
    assert correlations.mean() == pytest.approx(
        6 / np.pi * np.arcsin(alpha / 2), abs=0.02)


def test_alpha_out_of_range_is_rejected():
    with pytest.raises(AssertionError, match=r'alpha in \[0, 1\]'):
        generate_correlated_preferences(4, 4, alpha=1.5)


def test_the_more_alpha_the_more_the_residents_agree():
    shares = []
    for alpha in (0.0, 0.5, 1.0):
        # Share of the residents that rank the most wanted hospital of their
        # own instance first, on average
        instance_shares = []
        for seed in range(20):
            hr = generate_biased_HR(60, 3, alpha_resident=alpha, seed=seed)
            counts = np.bincount(
                [prefs[0] for prefs in hr.resident_prefs], minlength=3)
            instance_shares.append(counts.max() / counts.sum())
        shares.append(np.mean(instance_shares))

    assert shares[0] < shares[1] < shares[2]
    assert shares[2] == 1.0


def test_the_instance_is_solvable():
    hr = generate_biased_HR(60, 8, list_length=3, seed=12345)

    assert hr.num_residents == 60
    assert hr.num_hospitals == 8
    assert hr.capacities == [8] * 8
    assert all(len(prefs) == 3 for prefs in hr.resident_prefs)
    assert all(sorted(prefs) == list(range(60)) for prefs in hr.hospital_prefs)

    m_residents, m_hospitals = hr.solve()
    for r, h in enumerate(m_residents):
        assert h == -1 or h in hr.resident_prefs[r]
    for h in range(hr.num_hospitals):
        assert len(m_hospitals[h]) <= hr.capacities[h]


def test_the_same_seed_gives_the_same_instance():
    instances = [generate_biased_HR(60, 8, seed=12345) for _ in range(2)]

    assert instances[0].resident_prefs == instances[1].resident_prefs
    assert instances[0].hospital_prefs == instances[1].hospital_prefs
    assert instances[0].seed == 12345

    other = generate_biased_HR(60, 8, seed=12346)
    assert other.resident_prefs != instances[0].resident_prefs


def test_a_generator_can_be_passed_instead_of_a_seed():
    instances = [
        generate_biased_HR(60, 8, rng=np.random.default_rng(12345))
        for _ in range(2)
    ]

    assert instances[0].resident_prefs == instances[1].resident_prefs
    assert instances[0].seed is None


def test_the_instance_can_be_saved(tmp_path):
    hr = generate_biased_HR(12, 4, list_length=2, seed=12345)
    path = tmp_path / 'biased.txt'

    hr.save(path)
    loaded = type(hr).load(path)

    assert loaded.resident_prefs == hr.resident_prefs
    assert loaded.hospital_prefs == hr.hospital_prefs
    assert f'# seed: {hr.seed}' in path.read_text()


def test_the_unlisted_hospitals_are_unacceptable_by_default():
    hr = generate_biased_HR(60, 8, list_length=3, seed=12345)

    assert hr.tie_last is False
    assert hr.resident_prefs_rest is None
    assert hr.solve()[0].count(-1) > 0


def test_the_unlisted_hospitals_can_be_tied_last_on_generation():
    hr = generate_biased_HR(60, 8, list_length=3, tie_last=True, seed=12345)
    expected = generate_biased_HR(60, 8, list_length=3, seed=12345)
    expected.set_tie_last()

    assert hr.tie_last is True
    assert hr.resident_prefs_rest == expected.resident_prefs_rest
    assert hr.solve()[0] == expected.solve()[0]


def test_the_unlisted_hospitals_can_be_tied_last():
    hr = generate_biased_HR(60, 8, list_length=3, seed=12345)

    # The instance holds a generator of the kind HospitalResident uses, so
    # that the hospitals out of a list can be shuffled without a seed
    hr.set_tie_last()

    assert all(
        sorted(prefs + rest) == list(range(hr.num_hospitals))
        for prefs, rest in zip(hr.resident_prefs, hr.resident_prefs_rest)
    )

    m_residents, _ = hr.solve()
    assert m_residents.count(-1) == 0
