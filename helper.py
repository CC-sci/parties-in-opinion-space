"""
A general purpose helper module written to support other programmes with common
functionality, especially simulations.

Dependencies: Numpy

Copyright (c) 2024 Christopher Campbell
"""

import warnings
try:
    import numpy as np
except ModuleNotFoundError:
    warnings.warn('Numpy is not available, some functions will fail')


def printTrajectories(filename: str, labels: list[str], positions: list, to3D=False):
    """
    Overwrites or creates a file in the current directory saving trajectories.
    The last line is repeated as often as there are particles.

    The .xyz format is
    $Number of particles$
    Time = $timestep$
    label    x  y  z
    :param filename: name of the file to save to
    :param labels: list of names of each particle (size n)
    :param positions: 3D array of positions (e.g. size (n, time, dimensions))
    :param to3D: convert 2D data for 3D plotting routines
    """
    if len(labels) != len(positions) or np.ndim(positions) != 3:
        warnings.warn('Argument sizes may not be compatible')

    n = len(labels)
    p = np.array(positions)
    block = ""

    if to3D:
        if np.size(positions, axis=2) == 2:
            blockEnd = " 0.0\n"
        else:
            warnings.warn('Incompatible option, data must be 2D')
            blockEnd = "\n"
    else:
        blockEnd = "\n"

    # If positions has dimensions (n, time, ndim), this iterates along time and
    # then particles
    with (open(filename, 'w') as f):
        for i in range(np.size(p, 1)):
            block += f"{n}\nTime = {int(i)}\n"

            for j in range(n):
                block += f"{labels[j]}"
                for dim in range(np.size(p, 2)):
                    block += f" {p[j, i, dim]}"
                block += blockEnd

        f.write(block)


def printTabSeparated(positions: list):
    """
    Prints a 3D list (e.g. size (n, time, dimensions)) to n different files
    track each particle's progress, dimensions seperated by tabs.

    Compare printTrajectories, this splits into separate files. This is useful
    for exporting to matlab.
    :param positions: 3D array of positions (e.g. size (n, time, dimensions))
    """
    for i, party in enumerate(positions):
        f = open(f'party{i}.txt', mode='w')
        for step in party:
            for value in step:
                f.write(f'{value}\t')
            f.write('\n')
        f.close()


# See SolarHelper for reading files and finding periods