from collections import defaultdict, deque
from datetime import datetime
import io
from pathlib import Path

import numpy as np

from simplematching.pcg32 import Pcg32Rng, permutations


def _insert(
        items: list[int],
        ranks: list[int],
        new_item: int,
) -> None:
    """Insert a item new_item into the list items,
    which is sorted according to ranks.
    """
    i = len(items)
    while i > 0:
        if ranks[items[i-1]] < ranks[new_item]:
            break
        i -= 1
    items.insert(i, new_item)


def _read_noncomment_line(file: io.IOBase) -> None:
    """Read a line skipping the followings:
      comments (which start with '#')
      empty line (the line of)
    """
    for line in file:
        line = line.strip()
        if len(line) > 0 and line[0] != '#':
            yield line

class HospitalResident:
    def __init__(self):
        """Constructor"""

        # Whether the hospitals out of a resident's preference list are
        # acceptable (and tied at the last rank) or not.
        self.tie_last = False

        # Tie-broken order of the hospitals out of the preference list,
        # for each resident (used only when tie_last is True).
        self.resident_prefs_rest = None

    def init_random(self,
                    num_residents: int,
                    num_hospitals: int,
                    seed: int | None = None,
    ) -> None:
        """Randomly initialize the instance."""

        # Parameters
        self.num_residents = num_residents
        self.num_hospitals = num_hospitals

        # Initialize random number generator
        if seed is None:
            seed = np.random.default_rng().integers(0, 1 << 30)
        self.seed = seed
        rng = Pcg32Rng(seed)

        # Generate preference list (resident -> hospital)
        self.resident_prefs = permutations(rng, num_residents, num_hospitals)

        # Generate preference list (hospital -> resident)
        self.hospital_prefs = permutations(rng, num_hospitals, num_residents)

        # Keep the generator so that the random numbers consumed later
        # (e.g. by cut_resident_prefences) do not overlap with the ones
        # already used for the preference lists
        self.rng = rng

        # Set capacities for each hospital
        self.set_capacities()

    def set_capacities(
            self,
            capacities: int | list[int] = -1,
            tight: bool = False,
    ):
        """Set the capacities for each hospital.

        Pass a single int to give every hospital the same capacity
        (a negative value falls back to the default, evenly-divided
        capacity), or a list of length `num_hospitals` to give each
        hospital its own capacity.

        The default capacity rounds up, so that the hospitals hold a few more
        residents than there are.  Pass `tight` instead (and no capacity of
        your own) to have them hold exactly `num_residents` residents: every
        hospital gets `num_residents // num_hospitals` seats, and the seats
        left over go, one each, to the hospitals that the most residents rank
        first (then second, and so on; see `_popularity_keys`), ties broken at
        random.  The lists the popularity is counted in are the submitted
        ones, so a hospital no resident lists gets a spare seat by luck only.
        """

        if tight:
            from simplematching.utils import _popularity_keys

            assert capacities == -1, "Cannot specify both 'capacities' and 'tight'"

            capacities = self.num_residents // self.num_hospitals
            if self.num_residents != self.num_hospitals * capacities:
                hospital_keys = _popularity_keys(self.resident_prefs, self.num_hospitals)
                random_keys = self._get_rng().next_uint32_bulk(self.num_hospitals)
                sorted_hospitals = sorted(range(self.num_hospitals), key=lambda h: (hospital_keys[h], random_keys[h]))

                capacities = [capacities] * self.num_hospitals
                for i in range(self.num_residents % self.num_hospitals):
                    capacities[sorted_hospitals[i]] += 1
                assert self.num_residents == sum(capacities)
            else:
                capacities = [capacities] * self.num_hospitals
            self.capacities = capacities
            return

        if isinstance(capacities, list):
            assert len(capacities) == self.num_hospitals, (
                f'Expected {self.num_hospitals} capacities, got {len(capacities)}'
            )
            self.capacities = list(capacities)
            return

        if capacities < 0:
            capacities = 1 + (self.num_residents - 1) // self.num_hospitals
        self.capacities = [capacities] * self.num_hospitals

    def cut_resident_prefences(
            self,
            k: int,
            tie_last: bool = False,
            tie_break: str = 'random',
            seed: int | None = None,
    ):
        """Extract the top k preferences for each resident.

        By default the hospitals removed from a list are unacceptable:
        a resident rejected by all the k hospitals is left unmatched.

        With `tie_last=True` the removed hospitals are, instead, acceptable
        and tied at the last rank.  A tie has to be broken to run the
        (resident-oriented) Gale-Shapley algorithm; `tie_break` chooses how:
          'random': shuffle the removed hospitals at random (default)
          'keep':   keep the order they have in the current list, i.e. the
                    resident-oriented Gale-Shapley algorithm behaves exactly
                    as if the preference lists were not cut at all
        `seed` seeds the shuffle; when it is None the generator of the
        instance is used (or a random one, for a loaded instance).
        """

        rests = [prefs[k:] for prefs in self.resident_prefs]

        for r in range(self.num_residents):
            self.resident_prefs[r] = self.resident_prefs[r][:k]

        if tie_last:
            self._set_prefs_rest(rests, tie_break, seed)
        else:
            self.tie_last = False
            self.resident_prefs_rest = None

    def set_tie_last(
            self,
            tie_last: bool = True,
            tie_break: str = 'random',
            seed: int | None = None,
    ):
        """Treat every hospital out of a resident's preference list as
        acceptable and tied at the last rank.

        This is the counterpart of `cut_resident_prefences(..., tie_last=True)`
        for an instance whose preference lists are already incomplete
        (e.g. loaded from a file).  See `cut_resident_prefences` for
        `tie_break` and `seed`; here 'keep' means the ascending order of the
        hospital numbers.
        """

        if not tie_last:
            self.tie_last = False
            self.resident_prefs_rest = None
            return

        rests = []
        for prefs in self.resident_prefs:
            listed = set(prefs)
            rests.append([h for h in range(self.num_hospitals) if h not in listed])

        self._set_prefs_rest(rests, tie_break, seed)

    def _set_prefs_rest(self, rests: list[list[int]], tie_break: str, seed: int | None):
        """Store the tie-broken order of the hospitals out of the lists."""

        assert tie_break in ('random', 'keep'), f'Unknown tie_break: {tie_break}'

        if tie_break == 'random':
            rests = self._shuffle_lists(rests, seed)

        self.resident_prefs_rest = rests
        self.tie_last = True

    def _shuffle_lists(self, lists: list[list[int]], seed: int | None) -> list[list[int]]:
        """Shuffle each list at random (lists of the same length at once)."""

        rng = self._get_rng(seed)

        # Group the lists by their lengths, since `permutations` generates
        # permutations of a single length at a time
        groups = defaultdict(list)
        for i, items in enumerate(lists):
            groups[len(items)].append(i)

        shuffled = [None] * len(lists)
        for n, indices in sorted(groups.items()):
            for i, perm in zip(indices, permutations(rng, len(indices), n)):
                shuffled[i] = [lists[i][j] for j in perm]

        return shuffled

    def _get_rng(self, seed: int | None = None) -> Pcg32Rng:
        """Get a random number generator.

        A seed, if given, makes a new generator.  Otherwise the generator of
        the instance is reused (and made, at random, if it does not exist).
        """

        if seed is not None:
            return Pcg32Rng(seed)

        rng = getattr(self, 'rng', None)
        if rng is None:
            rng = Pcg32Rng(int(np.random.default_rng().integers(0, 1 << 30)))
            self.rng = rng
        return rng

    def resident_prefs_completed(self) -> list[list[int]]:
        """Preference list (resident -> hospital) that the algorithm follows.

        It is the preference list itself unless the hospitals out of the list
        are tied at the last rank; in that case they follow the listed ones.
        """

        if not self.tie_last:
            return self.resident_prefs

        return [
            prefs + rest
            for prefs, rest in zip(self.resident_prefs, self.resident_prefs_rest)
        ]

    def count_unlisted_matches(self, m_residents: list[int]) -> int:
        """Count the residents matched to a hospital out of their list."""

        return sum(
            1
            for r, h in enumerate(m_residents)
            if h >= 0 and h not in self.resident_prefs[r]
        )

    @staticmethod
    def generate(num_residents: int, num_hospitals: int, seed: int | None = None):
        """Generate a random instance."""

        instance = HospitalResident()
        instance.init_random(num_residents, num_hospitals, seed)
        return instance

    def show(self) -> None:
        """Print the instance."""

        print('Residents:', self.num_residents)
        print('Hospitals:', self.num_hospitals)

        for r in range(self.num_residents):
            if self.tie_last:
                print(f'{r}:', self.resident_prefs[r],
                      '(', *self.resident_prefs_rest[r], ')')
            else:
                print(f'{r}:', self.resident_prefs[r])

        for h in range(self.num_hospitals):
            print(f'{h}:', self.hospital_prefs[h])

    def save(self, path: Path) -> None:
        """Save the instance to a file.

        The preference lists are saved as they are, i.e. an incomplete list
        is saved incomplete.  Note that neither the tie_last flag nor the
        tie-broken order of the hospitals out of the lists is saved;
        call `set_tie_last()` again after loading.
        """

        # Make sure that the parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Do not overwrite
        assert not path.exists()

        # Print
        with open(path, 'w') as file:
            # Header
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            print('#', timestamp, file=file)
            print('# seed:', self.seed, file=file)
            print(f'HR {self.num_residents} {self.num_hospitals}', file=file)

            # Preference list (resident -> hospital)
            for prefs in self.resident_prefs:
                print(*prefs, file=file)

            # Preference list (hospital -> resident)
            for prefs in self.hospital_prefs:
                print(*prefs, file=file)

    @staticmethod
    def load(path: Path) -> None:
        """Load an instance from a file.

        A preference list of a resident may be incomplete, in which case the
        hospitals out of the list are unacceptable; call `set_tie_last()` to
        make them acceptable and tied at the last rank instead.
        """

        # New instance (which is empty)
        instance = HospitalResident()

        with open(path, 'r') as file:
            # Generator
            lines = _read_noncomment_line(file)

            # Header
            line = next(lines)
            assert line[:3] == 'HR ', f'Header excepted, found: {line}'
            num_residents, num_hospitals = map(int, line[3:].split())
            instance.num_residents = num_residents
            instance.num_hospitals = num_hospitals

            # Preference list (resident -> hospital)
            instance.resident_prefs = [
                list(map(int, next(lines).split()))
                for _ in range(num_residents)
            ]

            # Preference list (hospital -> resident)
            instance.hospital_prefs = [
                list(map(int, next(lines).split()))
                for _ in range(num_hospitals)
            ]

            # Set capacities for each hospital
            capacities = 1 + (num_residents - 1) // num_hospitals
            instance.capacities = [capacities] * num_hospitals

        return instance

    def solve(self) -> tuple[list[int], list[list[int]]]:
        """Solve the HR instance."""

        # Parametes
        num_residents = self.num_residents
        num_hospitals = self.num_hospitals
        resident_prefs = self.resident_prefs_completed()
        hospital_prefs = self.hospital_prefs

        # Ranks of residents, for each hospital
        hospital_ranks = [[-1] * num_residents for h in range(num_hospitals)]
        for h in range(num_hospitals):
            for k, r in enumerate(hospital_prefs[h]):
                hospital_ranks[h][r] = k

        # Head indices for each resident
        resident_head = [0] * num_residents

        # Solution (matching)
        m_residents = [-1 for _ in range(num_residents)]
        m_hospitals = [[] for _ in range(num_hospitals)]

        free_residents = deque(range(num_residents))
        while free_residents:
            # Pick a free resident r
            r = free_residents.popleft()

            # Pick a hospital that is the best available for the resident r
            if resident_head[r] >= len(resident_prefs[r]):
                continue
            h = resident_prefs[r][resident_head[r]]
            resident_head[r] += 1

            # Match the resident r to the hospital h
            m_residents[r] = h

            # Temporarily add the resident to the list of th hospital
            _insert(m_hospitals[h], hospital_ranks[h], r)

            # If the applicants exceeds the capacity,
            # remove one that is the worst for the hospital
            if len(m_hospitals[h]) > self.capacities[h]:
                r_worst = m_hospitals[h].pop()
                m_residents[r_worst] = -1
                free_residents.append(r_worst)

        return m_residents, m_hospitals

    def to_dicts(self) -> list[dict]:
        """Transform the instance to that of https://pypi.org/project/matching/ """

        return [
            dict(enumerate(self.resident_prefs_completed())),
            dict(enumerate(self.hospital_prefs)),
            dict(enumerate(self.capacities)),
        ]
