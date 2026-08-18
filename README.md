# simple_matching_python

A simple implementation of Gale-Shapley algorithm (GA algorithm, a.k.a. deferred-acceptance algorithm; DA algorithm) for stable matching (hospital-resident) problem.

Tens times faster than [matching](https://github.com/daffidwilde/matching) package (see [`running_time.ipynb`](./docs/running_time.ipynb) for details).

## Requirements

We recommend using [`uv`](https://docs.astral.sh/uv/) for dependency management and project setup, but it is not required.

## Install

```
$ git clone https://github.com/dfukagaw28/simple_matching_python.git
$ cd simple_matching_python
$ uv sync
```

## Run

Generate a random instance

```
$ uv run src/hospital_resident_cli.py generate 300 20
```

Solve

```
$ uv run src/hospital_resident_cli.py solve instances/HR_r300_h020_s05685763110016471008.txt
```

Simulation

```
$ uv run src/hospital_resident_cli.py simulate 100 10 20 -v -r 1000
```

Usage: `python src/hospital_resident_cli.py simulate [OPTIONS] NUM_RESIDENTS NUM_HOSPITALS CAPACITY_MAX`

- `NUM_RESIDENTS`: The number of residents
- `NUM_HOSPITALS`: The number of hospitals
- `CAPACITY_MAX`: The capacity of each hospital will vary from `CAPACITY_MIN` to `CAPACITY_MAX`
  - `CAPACITY_MIN` is the minimum integer no less than `NUM_RESIDENTS` / `NUM_HOSPITALS`
- `-r INTEGER`: The number of repetitions for each case (parameter setting)
- `-v`: Verbose level
- `--tie-last`: Treat the hospitals out of a preference list as acceptable and tied at the last rank (see below)
- `--help`: Help message

The value recorded for each case is the mean number of residents that are not matched to a hospital of their (cut) preference list.
By default the hospitals out of the list are unacceptable, so that such a resident is left unmatched.
With `--tie-last` they are acceptable and tied at the last rank, so that such a resident may be matched to a hospital out of their list.

## Biased preferences

`HospitalResident.generate()` draws every preference list independently, so that no hospital is more wanted than another.
`generate_biased_HR()` instead lets the residents agree with each other on how good the hospitals are, and the hospitals on how good the residents are.

```python
from simplematching.biased import generate_biased_HR

hr = generate_biased_HR(300, 24, alpha_resident=0.5, alpha_hospital=0.5,
                        list_length=8, seed=12345)
m_residents, m_hospitals = hr.solve()
```

Every resident scores a hospital as an opinion that all the residents share plus an opinion of their own, and ranks the hospitals by that score.
`alpha_resident` is how much of the score the shared opinion makes: it is the correlation of the scores of any two residents, from 0 (independent lists, as in `generate()`) to 1 (identical lists).
`alpha_hospital` does the same for the lists of the hospitals.
The residents rank their best `list_length` hospitals only (all of them if it is None), while the hospitals always rank every resident.
The hospitals a resident leaves out are unacceptable to them, unless `tie_last=True` makes them acceptable and tied at the last rank (see below).

The more the residents agree, the more of them miss out: with 300 residents, 24 hospitals of capacity 13 and lists of 8, the number of unmatched residents grows from 0.1 at `alpha_resident=0` to 14 at 0.25, 37 at 0.5 and 78 at 0.75, on average over the instances (a single instance varies widely around it).

## Incomplete preference lists

A preference list of a resident may be incomplete, either because it is cut by `cut_resident_prefences(k)` or because the instance file gives a short list.
The hospitals out of the list are unacceptable by default; a resident rejected by all the listed hospitals is left unmatched.

```python
hr = HospitalResident.generate(300, 20, seed=12345)
hr.cut_resident_prefences(5)
m_residents, m_hospitals = hr.solve()
```

Passing `tie_last=True` makes them acceptable and tied at the last rank instead: a resident rejected by all the listed hospitals keeps applying to the remaining ones.
A tie has to be broken to run the (resident-oriented) Gale-Shapley algorithm, which is done by `tie_break`:

- `'random'` (default): the hospitals out of the list are shuffled at random (seeded by `seed`, if given)
- `'keep'`: they keep the order they have in the current preference list, i.e. the algorithm behaves exactly as if the list were not cut at all

```python
hr = HospitalResident.generate(300, 20, seed=12345)
hr.cut_resident_prefences(5, tie_last=True)
# hr = HospitalResident.load(path)                  # already incomplete instance
# hr.set_tie_last(tie_break='random')

m_residents, m_hospitals = hr.solve()
n = hr.count_unlisted_matches(m_residents)          # matched out of the list
```

With `tie_last=True` the number of unmatched residents is no longer random: it is exactly `max(0, num_residents - sum(capacities))`.
A resident ends up unmatched only when every hospital has rejected them, and a hospital that has rejected someone is full and stays full, so every hospital is full whenever a resident is left unmatched.
The quantity of interest is therefore the number of residents matched out of their list, `count_unlisted_matches()`, rather than the number of unmatched residents.

Note that the resident-oriented Gale-Shapley algorithm only requires *weak* stability here, so any tie-breaking gives a stable matching.
For a randomly generated instance the two tie-breaking rules even give the same distribution of matchings, since the tail of a random permutation is itself a random order.

## Normalizing an instance

`normalize()` renumbers the residents and the hospitals of an instance by popularity, in place, which makes an instance easier to read.

```python
from simplematching.hospital_resident import HospitalResident
from simplematching.utils import normalize

hr = HospitalResident.generate(300, 24, seed=12345)
resident_order, hospital_order = normalize(hr)
```

The hospital that the most residents rank first becomes the hospital 0, and the resident that the most hospitals rank first becomes the resident 0; ties are broken by the number of the second choices, then of the third ones, and so on (see the docstrings for the details).
The instance is only renumbered, never altered: solving it gives the same matching, up to the renumbering.
The returned orders hold the old numbers in their new order, so `resident_order[i]` is the resident that becomes the resident `i`.

## Test

```
$ uv run pytest src/tests/test_hospital_resident.py -vv
```

or, more simply,

```
$ uv run pytest -vv
```
