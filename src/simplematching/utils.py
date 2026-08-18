"""Utilities on top of the HR instances, kept out of the instance itself."""

import numpy as np

from simplematching.hospital_resident import HospitalResident


def _popularity_keys(prefs: list[list[int]], num_items: int) -> list[tuple]:
    """Key that orders the items by how popular they are in the lists.

    An item is the more popular the more often it is ranked first, then the
    more often it is ranked second, and so on.  The keys are negated counts,
    so that sorting the items by their key puts the most popular one first.
    """
    depth = max((len(pref) for pref in prefs), default=0)

    counts = [[0] * depth for _ in range(num_items)]
    for pref in prefs:
        for position, item in enumerate(pref):
            counts[item][position] += 1

    return [tuple(-count for count in counts[item]) for item in range(num_items)]


def _classes(keys: list) -> list[int]:
    """Number the items so that the items with an equal key share a number.

    The numbers follow the order of the keys, i.e. the items holding the
    smallest key are the class 0.
    """
    classes = {key: c for c, key in enumerate(sorted(set(keys)))}
    return [classes[key] for key in keys]


def _canonical_orders(
        resident_prefs: list[list[int]],
        hospital_prefs: list[list[int]],
        num_residents: int,
        num_hospitals: int,
) -> tuple[list[int], list[int]]:
    """Order the residents and the hospitals, the most popular ones first.

    Popularity (see `_popularity_keys`) comes first.  The items that are
    equally popular are told apart by the lists they hold themselves: an item
    comes first if the items on its list are the more popular ones.  That
    refinement is repeated as long as it tells more items apart, so that the
    order depends on the instance only and not on the numbers the residents
    and the hospitals happen to have.  Items that stay tied (which needs them
    to be interchangeable) keep the order of their current numbers.
    """
    resident_keys = _popularity_keys(hospital_prefs, num_residents)
    hospital_keys = _popularity_keys(resident_prefs, num_hospitals)

    while True:
        resident_classes = _classes(resident_keys)
        hospital_classes = _classes(hospital_keys)

        refined_residents = [
            (resident_keys[r], tuple(hospital_classes[h] for h in resident_prefs[r]))
            for r in range(num_residents)
        ]
        refined_hospitals = [
            (hospital_keys[h], tuple(resident_classes[r] for r in hospital_prefs[h]))
            for h in range(num_hospitals)
        ]

        if (_classes(refined_residents) == resident_classes
                and _classes(refined_hospitals) == hospital_classes):
            break

        resident_keys = refined_residents
        hospital_keys = refined_hospitals

    return (
        sorted(range(num_residents), key=lambda r: (resident_keys[r], r)),
        sorted(range(num_hospitals), key=lambda h: (hospital_keys[h], h)),
    )


def normalize(instance: HospitalResident) -> tuple[list[int], list[int]]:
    """Renumber the residents and the hospitals of an instance by popularity.

    The hospital that the residents rank first most often becomes the
    hospital 0, and the resident that the hospitals rank first most often
    becomes the resident 0; ties are broken by the number of the second
    choices, then of the third ones, and so on.  The residents (hospitals)
    that are still tied are told apart by the lists they hold themselves,
    the one whose hospitals (residents) are the more popular coming first
    (see `_canonical_orders`).  The preference lists that the popularity
    is counted in are the completed ones, i.e. the hospitals tied at the
    last rank are counted as well (see `resident_prefs_completed`).

    The instance is renumbered in place, into one that is isomorphic to
    it: solving it gives the same matching, up to the renumbering.  The
    two orders are returned, resident first, each holding the old numbers
    in their new order (i.e. `resident_order[i]` is the resident that
    becomes the resident `i`).
    """

    resident_order, hospital_order = _canonical_orders(
        instance.resident_prefs_completed(),
        instance.hospital_prefs,
        instance.num_residents,
        instance.num_hospitals,
    )

    # Number that each resident and each hospital is renumbered to
    new_resident = np.argsort(resident_order).tolist()
    new_hospital = np.argsort(hospital_order).tolist()

    instance.resident_prefs = [
        [new_hospital[h] for h in instance.resident_prefs[r]]
        for r in resident_order
    ]
    if instance.tie_last:
        instance.resident_prefs_rest = [
            [new_hospital[h] for h in instance.resident_prefs_rest[r]]
            for r in resident_order
        ]
    instance.hospital_prefs = [
        [new_resident[r] for r in instance.hospital_prefs[h]]
        for h in hospital_order
    ]
    instance.capacities = [instance.capacities[h] for h in hospital_order]

    return resident_order, hospital_order
