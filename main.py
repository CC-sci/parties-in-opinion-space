"""
Opinion space party dynamics.

The starting point to model the motion of political parties under
"gravitational attraction" towards voters. Parties move on a grid towards
equilibrium to maximise their voters, using Metropolis Monte-Carlo.
It remains to be seen what form this force takes and also the dimensionality
of opinion space.

:Author: Christopher Campbell
"""
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from scipy.stats import multivariate_normal
from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt
import argparse
from random import shuffle

import helper
from opinion_grid import OpinionGrid
from party import Party, LineParty
from helper import printTrajectories


def relativeBoltzmannFactor(e1, e2, temperature, muN=0.0):
    return np.exp((e2 - e1 - muN) / temperature)


# ToDo: What should the distribution be, should I use scipy.stats.qmc
# ToDo: Currently they can pass each other, maybe alright
# Can use stepDis = Distribution([0,0], covariance)
#         step = stepDis.random(1)
# or      step = multivariate_normal.rvs([0,0], covariance)
# Could directly edit ._mean, maybe faster but no checking
def metropolisStep(grid: OpinionGrid, distribution, covariance, volatility):
    """
    Implements the Metropolis-Hastings Monte Carlo algorithm.

    The parties (one after another) take a step on the grid, its favourability
    is evaluated (Boltzmann distribution) and the step is accepted or rejected.

    Energetically favourable steps are always accepted. Unfavourable ones have
    a chance of being accepted or rejected.

    The programme returns -1 if a party attempted to go out of bounds (this is
    prevented), -2 if a party moved to an energetically favourable position, -3
    if a party moved to an energetically unfavourable position but was allowed
    and -4 if the party would have moved to an energetically unfavourable
    position but was not allowed. These are returned as a list, one integer for
    each party. Many -4's indicate only very unfavourable steps are possible,
    that is they indicate equilibrium.
    :param grid: The grid on which to act
    :param distribution: The step of a party is drawn from ``distribution.rvs()``
    :param covariance: For step distribution
    :param volatility: Thermal volatility
    :return: exit status
    :rtype: list of int
    """
    # Randomise the turn order (modifies original rather than copying)
    shuffle(grid.parties)
    exit = []

    for i, party in enumerate(grid.parties):
        # The copy() is necessary because otherwise point to same memory
        # The step is centred on the party, so step difference centred at zero
        initialVotes = party.votes
        initialPos = party.position.copy()
        step = distribution.rvs([0, 0], covariance)
        party.move(step[0], step[1])

        newVotes = grid.energy()[i]

        # If energetically favourable, accept. If unfavourable, accept with the
        # relative probability of both voters, currently Boltzmann distribution
        if abs(party.position[0]) > 1 or abs(party.position[1]) > 1:
            party.position = initialPos
            exit.append(-1)
        elif initialVotes < newVotes:
            exit.append(-2)
        elif uniform.rvs() <= relativeBoltzmannFactor(initialVotes,
                                                      newVotes, volatility):
            exit.append(-3)
        else:
            # This means Boltzmann is very small so new position much worse
            party.position = initialPos
            exit.append(-4)
    return exit

        # Grand canonical simulation, varying the number of parties, is not used
    #     mu = 300.0
    #     grandCanonical = True
    #     # removing parties
    #     if grandCanonical:
    #         if newVotes < mu and uniform.rvs() <= relativeBoltzmannFactor(-newVotes,
    #                                         0.0, -mu*len(grid.parties), volatility):
    #             del grid.parties[i]
    #
    # # Maybe don't call on every step
    # # Maybe take this step with probability [0, 1] < 1/(1+N) where N is the
    # # number of steps since the last grandCanonical test
    # if grandCanonical:
    #     # Test if a new party should form (then see if one should die)
    #     # Create the test party
    #     testPosition = [uniform.rvs(-1, 2), uniform.rvs(-1, 2)]
    #     testParty = Party('New-Party', testPosition)
    #     grid.parties.append(testParty)
    #     testVotes = grid.energy()[-1]
    #     if uniform.rvs() <= relativeBoltzmannFactor(-testVotes, 0.0, volatility, mu*len(grid.parties)):
    #         pass  # accept
    #     else:
    #         del grid.parties[-1]


def main():
    # Set up simulation parameters from the command line
    # First positional argument, then flags (some are boolean, some take args)
    # ArgumentDefaultsHelpFormatter will print default values with --help
    # ToDo: Simulation parameters
    parser = argparse.ArgumentParser(description='Optional Simulation Parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('steps', nargs='?', type=int, default=1000,
                        help='the number of steps the simulation runs for')
    parser.add_argument('-o', '--output', type=str, default='output.dat',
                        help='the name of the output file to be overwritten/created')
    # parser.add_argument('-P', '--print', action='store_true',
    #                     help='output observables to the console/standard output')
    parser.add_argument('-H', '--nohistogram', action='store_true',
                        help='supress histograms of the parties\' positions')
    parser.add_argument('-S', '--scatter', action='store_true',
                        help='show scatterplot of parties\'s past and final positions')
    parser.add_argument('-D', '--distribution', action='store_true',
                        help='display plots of the voter distribution')
    parser.add_argument('-m', '--mat', action='store_true',
                        help='output tab seperated files for each party, matlab compatible')
    parser.add_argument('-1d', '--line', action='store_true',
                        help='run a one-dimensional Hotelling simulation')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print status during execution')

    grid = OpinionGrid()
    args = parser.parse_args()

    # Check whether to simulate 1d Hotelling
    if args.line:
        # Sets voters to be uniform, or zero off the line
        grid.weight = np.zeros_like(grid.weight)
        grid.weight[int(np.size(grid.xGrid, axis=0)/2), :] = 1
        grid.addParties(LineParty.test())
    else:
        grid.weight[:, :] = 1; print('Uniform voter distribution')  # temp
        # grid.weight = (grid.xGrid ** 2 + grid.yGrid ** 2) ** 1.5; print('Extreme voters')
        grid.addParties(Party.test())

    # Cov matrix, assume identity matrix and scale
    # FWQM about 1.6651 sqrt(var)
    covFactor = 0.03
    positions = np.zeros((len(grid.parties), args.steps+1, np.ndim(grid.weight)))
    positions[:,0] = [p.position for p in grid.parties]
    # positions = positions.tolist()  for grand canonical
    stepNum = 0
    j = 0  # 4-measure

    # Plotting
    if args.scatter:
        figSt, axSt = plt.subplots()

    if args.verbose:
        print('Setup complete.')

    # As 'converges', usually returns -4 and less often -3 then -2
    # Being in a different position becomes much less favourable, similarly
    # could use the magnitude of the Boltzmann factor as a measure
    # The order of parties is randomised during the step to avoid bias through
    # random turn orders. Then the list is sorted again to get the data in the
    # right order. This could be slow, but fine for relatively few parties.
    while stepNum < args.steps:
        exit = metropolisStep(grid, multivariate_normal,
                              covFactor*np.array([[1, 0], [0, 1]]), 0.1)

        if exit[0] == -4 and exit[1] == -4:
            j += 1
        else:
            j = 0
        stepNum += 1
        grid.parties.sort(key=lambda p: p.name)

        for i, party in enumerate(grid.parties):
            positions[i, stepNum] = party.position

        if args.verbose and stepNum % 50 == 0:
            print(f"Step {stepNum}", end=' ... ')

    if args.verbose:
        print("Printing...")

    printTrajectories(args.output, list([p.name for p in grid.parties]),
                      positions, to3D=True)
    if args.mat:
        helper.printTabSeparated(positions)

    if args.verbose:
        print('\nFinal positions | votes')
        [print(f'\t{p.name}: {p.position} | {p.votes} votes') for p in grid.parties]
        print('Plotting...')

    # Generate the scatter plot
    if args.scatter:
        axSt.scatter(positions[:, :, 0], positions[:, :, 1], s=10, color='gray')
        for party in grid.parties:
            axSt.scatter(party.position[0], party.position[1], marker='o', s=150)
        figSt.show()

    # Histograms
    if not args.nohistogram:
        for i in range(0, len(grid.parties)):
            figH, axH = plt.subplots()
            axH.hist2d(positions[i, :, 0], positions[i, :, 1], bins=25)
            axH.set_xlim(-1,1)
            axH.set_ylim(-1, 1)
            axH.set_facecolor('#440154')
            figH.suptitle(f'Party {i+1}')
            figH.show()

    print(f'Simulated {stepNum} steps.')

    # Party step distribution
    # gaussian = MultivariateNormalQMC([0,0], covFactor*np.array([[1,0], [0,1]]))
    # sample1 = gaussian.random(512)
    # sample2 = multivariate_normal.rvs([0,0], covFactor*np.array([[1,0], [0,1]]), size=500)
    # plt.scatter(sample1[:,0], sample1[:,1])
    # plt.axis('square')
    # plt.ylim(-1,1)
    # plt.xlim((-1,1))
    # plt.show()
    # plt.scatter(sample2[:,0], sample2[:,1])
    # plt.axis('square')
    # plt.ylim(-1, 1)
    # plt.xlim((-1, 1))
    # plt.show()

    # Voter distribution
    if args.distribution:
        grid.plotDistribution()


if __name__ == '__main__':
    main()