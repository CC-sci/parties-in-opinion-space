"""
Opinion space dynamics.

The starting point to model the motion of political parties under
"gravitational attraction" towards voters. Parties move on a grid towards
equilibrium to maximise their voters, using Metropolis Monte-Carlo.
It remains to be seen what form this force takes and also the dimensionality
of opinion space.
"""
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from scipy.stats.qmc import MultivariateNormalQMC
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

def relativeBoltzmannFactor(e1, e2, temperature):
    return np.exp((e2-e1)/temperature)


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
    :param distribution: The step of a party is drawn from distribution.rvs()
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


def main():
    # Set up simulation parameters from the command line
    # First positional argument, then flags (some are boolean, some take args)
    # ArgumentDefaultsHelpFormatter will print default values with --help
    # ToDo: Verbose, simulation parameters, number of steps, whether to output plots
    parser = argparse.ArgumentParser(description='Optional Simulation Parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('steps', nargs='?', type=int, default=500,
                        help='the number of steps the simulation runs for')
    parser.add_argument('-o', '--output', type=str, default='output.dat',
                        help='the name of the output file to be overwritten/created')
    parser.add_argument('-O', '--print', action='store_true',
                        help='output observables to the console/standard output')
    parser.add_argument('-H', '--histogram', action='store_true',
                        help='display histograms of the parties\' positions')
    parser.add_argument('-D', '--distribution', action='store_true',
                        help='display plots of the voter distribution')
    parser.add_argument('-m', '--mat', action='store_true',
                        help='output tab seperated files for each party, matlab compatible')
    parser.add_argument('-1d', '--line', action='store_true',
                        help='run a one-dimensional Hotelling simulation')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='output plots')
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
        grid.weight[:, :] = 1
        grid.addParties(Party.test())

    # Cov matrix, assume identity matrix and scale
    # FWQM about 1.6651 sqrt(var)
    covFactor = 0.03
    positions = np.zeros((len(grid.parties), args.steps+1, np.ndim(grid.weight)))
    positions[:,0] = [p.position for p in grid.parties]
    stepNum = 0
    j = 0  # 4-measure

    # Plotting
    figSt, axSt = plt.subplots()
    # ax.scatter(grid.xGrid, grid.yGrid, color='gray', s=1, marker='.')

    if args.verbose:
        print('Setup complete.')

    # ToDo: Some convergence condition
    # ToDo: The simulation gets ever-slower, why? Should something be cleared?
    # As converges, usually returns -4 and less often -3 then -2
    # Being in a different position becomes much less favourable, similarly
    # could use the magnitude of the Boltzmann factor as a measure
    # The 4-measure works well (already <20 decent), or limit with steps stepNum
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
            axSt.scatter(party.position[0], party.position[1], s=10, color='black')
            positions[i,stepNum] = party.position

        if args.verbose and stepNum % 50 == 0:
            print(f"Step {stepNum}", end=' ... ')

    for party in grid.parties:
        axSt.scatter(party.position[0], party.position[1], marker='o', s=150)

    printTrajectories(args.output, list([p.name for p in grid.parties]),
                      positions, to3D=True)
    if args.mat:
        helper.printTabSeparated(positions)

    # Histograms
    print('Plotting...')
    figSt.show() #Todo: Remove
    print('Shown One')

    if args.histogram:
        figH0, axH0 = plt.subplots()
        axH0.hist2d(positions[0, :, 0], positions[0, :, 1], bins=25)
        axH0.set_xlim(-1,1)
        axH0.set_ylim(-1, 1)
        axH0.set_facecolor('#440154')
        plt.title('Party 1')
        figH0.show()
        figH1, axH1 = plt.subplots()
        axH1.hist2d(positions[1, :, 0], positions[1, :, 1], bins=25)  # or hexbin with gridsize=
        axH1.set_xlim(-1, 1)
        axH1.set_ylim(-1, 1)
        axH1.set_facecolor('#440154')
        plt.title('Party 2')
        figH1.show()
        plt.title('Party 2')
        figH2, axH2 = plt.subplots()
        axH2.hist2d(positions[2, :, 0], positions[2, :, 1], bins=25)
        axH2.set_xlim(-1, 1)
        axH2.set_ylim(-1, 1)
        axH2.set_facecolor('#440154')
        plt.title('Party 3')
        figH2.show()

    print(f'Simulated {stepNum} steps.')

    # Party step distribution
    # gaussian = MultivariateNormalQMC([0,0], covFactor*np.array([[1,0], [0,1]]))
    # sample1 = gaussian.random(512)
    # sample2 = multivariate_normal.rvs([0,0], covFactor*np.array([[1,0], [0,1]]), size=500)
    # plt.scatter(sample1[:,0], sample1[:,1])
    # plt.show()
    # plt.scatter(sample2[:,0], sample2[:,1])
    # plt.show()

    # Voter distribution
    if args.distribution:
        grid.plotDistribution()


if __name__ == '__main__':
    main()