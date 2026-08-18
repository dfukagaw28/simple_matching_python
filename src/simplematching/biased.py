"""Generation of HR instances whose preferences are biased.

The residents (hospitals) do not rank the hospitals (residents) independently
of each other: they share a common opinion on how good each of them is, the
weight of which is given by `alpha`.
"""

import numpy as np

from simplematching.hospital_resident import HospitalResident
from simplematching.pcg32 import Pcg32Rng


def generate_correlated_preferences(
        num_choosers: int,
        num_choices: int,
        alpha: float = 0.5,
        list_length: int | None = None,
        rng: np.random.Generator | None = None,
) -> list[list[int]]:
    """Generate `num_choosers` preference lists over `num_choices` items.

    Every chooser scores an item as `sqrt(alpha)` times an opinion that all
    the choosers share plus `sqrt(1 - alpha)` times an opinion of their own,
    both drawn from the standard normal distribution, and ranks the items by
    that score.  The square roots make `alpha` the correlation of the scores
    of any two choosers (and leave the scores standard normal themselves), so
    `alpha = 0` makes the lists independent of each other, `alpha = 1` makes
    them all the same, and `alpha = 0.5` puts them halfway.

    Only the best `list_length` items are kept in a list, all of them if it is
    None.  Beware that a *hospital* has to rank every resident: a resident it
    leaves out is not treated as unacceptable by `HospitalResident.solve`.
    """
    assert 0 <= alpha <= 1, f'Expected an alpha in [0, 1], got {alpha}'

    if rng is None:
        rng = np.random.default_rng()

    if list_length is None:
        list_length = num_choices

    # The opinion all the choosers share, and the one each of them holds alone
    common = rng.normal(0, 1, size=num_choices)
    idiosyncratic = rng.normal(0, 1, size=(num_choosers, num_choices))

    scores = np.sqrt(alpha) * common + np.sqrt(1 - alpha) * idiosyncratic

    return np.argsort(-scores, axis=1)[:, :list_length].tolist()


def generate_biased_HR(
        num_residents: int,
        num_hospitals: int,
        alpha_resident: float = 0.5,
        alpha_hospital: float = 0.5,
        list_length: int | None = 8,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
) -> HospitalResident:
    """Generate a Hospital-Resident instance with biased preferences.

    `alpha_resident` is how much the residents agree with each other on the
    hospitals, and `alpha_hospital` how much the hospitals agree with each
    other on the residents, 0 being no agreement beyond chance and 1 being a
    complete one (see `generate_correlated_preferences`).  The
    residents rank their best `list_length` hospitals only (all of them if it
    is None), while the hospitals always rank every resident.

    The instance is generated from `seed`, a random one if none is given,
    unless a generator is passed as `rng`; the instance is then generated from
    it and holds no seed of its own.
    """
    if rng is None:
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 1 << 30))
        rng = np.random.default_rng(seed)

    # Create a instance
    hr = HospitalResident()
    hr.num_residents = num_residents
    hr.num_hospitals = num_hospitals
    hr.seed = seed

    # Generate resident preferences
    hr.resident_prefs = generate_correlated_preferences(
        num_residents, num_hospitals, alpha_resident, list_length, rng)

    # Generate hospital preferences, which have to rank every resident
    hr.hospital_prefs = generate_correlated_preferences(
        num_hospitals, num_residents, alpha_hospital, None, rng)

    # Give the instance a generator of the kind it uses itself, so that the
    # hospitals out of a resident's list can be shuffled (see set_tie_last)
    hr.rng = Pcg32Rng(int(rng.integers(0, 1 << 30)))

    # Set capacities for each hospital
    hr.set_capacities()

    return hr
