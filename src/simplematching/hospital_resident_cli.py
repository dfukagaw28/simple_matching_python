from pathlib import Path

import click
import datetime
import numpy as np
import pandas as pd

from hospital_resident import HospitalResident as HR


@click.group()
def main():
    pass


@main.command()
@click.argument('num_residents', type=int)
@click.argument('num_hospitals', type=int)
@click.option('-s', '--seed', default=-1, type=int)
@click.option('--outdir', default='./instances', type=click.Path())
def generate(num_residents, num_hospitals, seed, outdir):
    """Generate an instance and save it."""
    # Parameters
    if seed < 0:
        seed = np.random.default_rng().integers(0, 1<<63)

    # Generate an instance
    hr = HR.generate(num_residents, num_hospitals, seed)

    # Save the instance into a file
    filename = f'HR_r{num_residents:03}_h{num_hospitals:03}_s{seed:020}.txt'
    path = Path(outdir) / filename
    hr.save(path)
    click.echo(f'Saved to {path}')


@main.command()
@click.argument('path', type=click.Path())
def solve(path: Path):
    """Load an instance and solve it."""
    hr = HR.load(path)
    solution = hr.solve()
    m_residents, m_hospitals = solution
    click.echo(m_residents)


@main.command()
@click.argument('num_residents', type=int)
@click.argument('num_hospitals', type=int)
@click.argument('capacity_max', type=int)
@click.option('-r', '--repeat', default=10, type=int)
@click.option('-s', '--seed', default=-1, type=int)
@click.option('-v', '--verbose', count=True)
@click.option('--outdir', default='./sim_results', type=click.Path())
def simulate(num_residents, num_hospitals, capacity_max, repeat, seed, verbose, outdir):
    """Run a simulation."""

    # Seed generator
    if seed < 0:
        seed = np.random.default_rng().integers(0, 1<<63)
    master_seed = seed
    rng = np.random.default_rng(master_seed)

    capacity_min = 1 + (num_residents - 1) // num_hospitals

    df_result = pd.DataFrame({
        resident_pref_max: np.full(capacity_max - capacity_min + 1, None)
        for resident_pref_max in range(1, num_hospitals + 1)
    }, dtype=float)
    df_result['capacity'] = list(range(capacity_min, capacity_max + 1))
    df_result = df_result.set_index('capacity')

    capacity_is_enough = 0
    for capacity in range(capacity_min, capacity_max + 1):
        for resident_pref_max in range(1, num_hospitals + 1):
            # Generate random seeds
            seeds = [rng.integers(0, 1<<63) for _ in range(repeat)]

            # Result array
            n_unmatches = []

            # Run
            for seed in seeds:
                # Generate an instance
                hr = HR.generate(num_residents, num_hospitals, seed)

                # Set caacities
                hr.set_capacities(capacity)

                # Extract the top `resident_pref_max` preferences
                hr.cut_resident_prefences(resident_pref_max)

                # Solve
                m_residents, m_hospitals = hr.solve()

                # Count the number of unmatched residents
                n_unmatch = m_residents.count(-1)

                if verbose >= 2:
                    click.echo(f'R:{num_residents} H:{num_hospitals} C:{capacity} M:{resident_pref_max} S:{seed}  ->  U:{n_unmatch}')

                n_unmatches.append(n_unmatch)

            mean = np.mean(n_unmatch)

            if verbose >= 1:
                click.echo(f'R:{num_residents} H:{num_hospitals} C:{capacity} M:{resident_pref_max}  ->  U:{mean:.4f}')

            df_result.at[capacity, resident_pref_max] = mean

            if resident_pref_max >= 3:
                recent_results = list(df_result.loc[capacity].dropna())[-3:]
                if np.max(recent_results) == 0:
                    if verbose >= 1:
                        click.echo('In recent 3 cases, all residents has matched.  The process will be skipped.')
                    if resident_pref_max == 3:
                        capacity_is_enough += 1
                    else:
                        capacity_is_enough = 0
                    break
        if capacity_is_enough >= 3:
            break

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    df_result.to_csv(f'result_R{num_residents}_H{num_hospitals}_{timestamp}.csv')


if __name__ == '__main__':
    main()
