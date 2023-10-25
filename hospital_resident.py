from collections import deque
from datetime import datetime
import io
from pathlib import Path

import numpy as np


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
        pass

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
            seed = np.random.default_rng().integers(0, 1<<63)
        self.seed = seed
        rng = np.random.default_rng(seed)

        # Generate preference list (resident -> hospital)
        self.resident_prefs = [
            np.argsort(rng.random(size=num_hospitals))
            for _ in range(num_residents)
        ]

        # Generate preference list (hospital -> resident)
        self.hospital_prefs = [
            np.argsort(rng.random(size=num_residents))
            for _ in range(num_hospitals)
        ]

        # Set capacities for each hospital
        self.set_capacities()

    def set_capacities(self, capacities: int = -1):
        """Set the capacities for each hospital"""

        if capacities < 0:
            capacities = 1 + (self.num_residents - 1) // self.num_hospitals
        self.capacities = [capacities] * self.num_hospitals

    def cut_resident_prefences(self, k: int):
        """Extract the top k preferences for each resident"""

        for r in range(self.num_residents):
            self.resident_prefs[r] = self.resident_prefs[r][:k]

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
            print(f'{r}:', self.resident_prefs[r])

        for h in range(self.num_hospitals):
            print(f'{h}:', self.hospital_prefs[h])

    def save(self, path: Path) -> None:
        """Save the instance to a file."""

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
        """Load an instance from a file."""

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
        resident_prefs = self.resident_prefs
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
            dict(enumerate(self.resident_prefs)),
            dict(enumerate(self.hospital_prefs)),
            dict(enumerate(self.capacities)),
        ]
