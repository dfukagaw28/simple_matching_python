# simple_matching_python

A simple implementation of Gale-Shapley algorithm (GA algorithm, a.k.a. deferred-acceptance algorithm; DA algorithm) for stable matching (hospital-resident) problem.

Tens times faster than [matching](https://github.com/daffidwilde/matching) package (see [`running_time.ipynb`](./running_time.ipynb) for details).

## Requirements

```
$ pip install click numpy pandas
```

Tested in the following environment:

```
click==8.1.7
numpy==1.26.1
pandas==2.1.1
```

## Install

```
$ git clone https://github.com/dfukagaw28/simple_matching_python.git
$ cd matching_python
```

## Run

Generate a random instance

```
$ python hospital_resident_cli.py generate 300 20
```

Solve

```
$ python hospital_resident_cli.py solve instances/HR_r300_h020_s05685763110016471008.txt
```

Simulation

```
$ python hospital_resident_cli.py simulate 100 10 20 -v -r 1000
```

Usage: `python hospital_resident_cli.py simulate [OPTIONS] NUM_RESIDENTS NUM_HOSPITALS CAPACITY_MAX`

- `NUM_RESIDENTS`: The number of residents
- `NUM_HOSPITALS`: The number of hospitals
- `CAPACITY_MAX`: The capacity of each hospital will vary from `CAPACITY_MIN` to `CAPACITY_MAX`
  - `CAPACITY_MIN` is the minimum integer no less than `NUM_RESIDENTS` / `NUM_HOSPITALS`
- `-r INTEGER`: The number of repetitions for each case (parameter setting)
- `-v`: Verbose level
- `--help`: Help message

## Test

```
$ pip install matching
```

```
$ pytest tests/test_hospital_resident.py -vv
```
