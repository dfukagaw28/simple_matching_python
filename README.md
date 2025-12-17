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
- `--help`: Help message

## Test

```
$ uv run pytest src/tests/test_hospital_resident.py -vv
```

or, more simply,

```
$ uv run pytest -vv
```
