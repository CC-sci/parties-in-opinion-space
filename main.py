"""
Opinion space party dynamics.

The starting point to model the motion of political parties under an attractive
force towards voters. Analogous to minimising energy under gravity/EM/etc.,
parties move on a grid towards equilibrium to maximise their voters, using
Metropolis Monte-Carlo steps.

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
import warnings

import helper
from opinion_grid import OpinionGrid
from party import Party, LineParty


def relativeBoltzmannFactor(e1, e2, temperature, muN=0.0):
    return np.exp((e2 - e1 - muN) / temperature)


def metropolisStep(grid: OpinionGrid, distribution, covariance, volatility,
                   ordering: bool, turnoutParameter, activistParameter):
    """
    Implements the Metropolis-Hastings Monte Carlo algorithm.

    The parties (one after another) take a Gaussian step on the grid, its
    favourability is evaluated (Boltzmann distribution) and the step is accepted
    or rejected (parties can pass each other). The aim of every party is to
    maximise its votes.

    Energetically favourable steps are always accepted. Unfavourable ones have
    a chance of being accepted or rejected.

    Every grid point, i.e. voter, votes for the party nearest to it. Turnout is
    represented by the fact that votes decrease further away from the party, so
    that there is relatively little contribution from voters whose closest party
    is very far away. Activists represent the tendency of extreme party members
    to wield lobbying influence or more often be candidates. A positive number
    will add a contribution making it more favourable for the party to be more
    extreme than its voters. A negative number will produce centrist activists,
    so it is more favourable to be more moderate than a party's average voter.

    The function returns -1 if a party attempted to go out of bounds (this is
    prevented), -2 if a party moved to an energetically favourable position, -3
    if a party moved to an energetically unfavourable position but was allowed
    and -4 if the party would have moved to an energetically unfavourable
    position but was not allowed. These are returned as a list, one integer for
    each party. Many -4's indicate only very unfavourable steps are possible,
    that is they might indicate equilibrium.
    :param grid: The grid on which to act
    :param distribution: The step of a party is drawn from ``distribution.rvs()``
    :param covariance: For step distribution
    :param float volatility: Thermal volatility i.e. 'temperature'
    :param bool ordering: Preserve ordering on the line
    :param float turnoutParameter: The higher this number the quicker the voter falloff
    :param float activistParameter: How effective activists are
    :return: exit status
    :rtype: list of int
    """
    # Randomise the turn order (modifies original rather than copying)
    shuffle(grid.parties)
    exit = []
    if ordering: initialOrdering = sorted(grid.parties, key=lambda p: p.position[0])

    for i, party in enumerate(grid.parties):
        # The copy() is necessary because otherwise point to same memory
        # The step is centred on the party, so step difference centred at zero
        initialVotes = party.votes
        initialPos = party.position.copy()
        step = distribution.rvs([0, 0], covariance)

        party.move(step[0], step[1])
        newVotes = grid.energy(turnoutParameter, activistParameter)[i]

        # If energetically favourable, accept. If unfavourable, accept with the
        # relative probability of both voters, currently Boltzmann distribution
        if abs(party.position[0]) > 1 or abs(party.position[1]) > 1:
            party.position = initialPos
            exit.append(-1)
        elif ordering and (initialOrdering != sorted(grid.parties, key=lambda p: p.position[0])):
            party.position = initialPos
        elif initialVotes < newVotes:
            exit.append(-2)
        elif uniform.rvs() <= relativeBoltzmannFactor(initialVotes, newVotes, volatility):
            exit.append(-3)
        else:
            # This means Boltzmann is very small so new position much worse
            party.position = initialPos
            exit.append(-4)
    return exit

    # Grand canonical simulation, varying the number of parties, is not used
    # Result is simply that one chooses a value of mu and that number of parties becomes optimal
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
    # History of winners would be a good addition if party-specific or
    #  asymmetric situations were introduced
    parser = argparse.ArgumentParser(description='Optional Simulation Parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('steps', nargs='?', type=int, default=500,
                        help='the number of steps the simulation runs for')
    parser.add_argument('-1d', '--line', action='store_true',
                        help='run a one-dimensional Hotelling simulation')
    parser.add_argument('-o', '--output', type=str, default='output.dat',
                        help='the name of the output file to be overwritten/created')
    parser.add_argument('-H', '--nohistogram', action='store_true',
                        help='supress histograms of the parties\' positions')
    parser.add_argument('-S', '--scatter', action='store_true',
                        help='show scatterplot of parties\'s past and final positions')
    parser.add_argument('-D', '--distribution', action='store_true',
                        help='display plots of the voter distribution')
    parser.add_argument('-t', '--tab', action='store_true',
                        help='output tab seperated files for each party')
    parser.add_argument('-M', '--matpol', action='store_true',
                        help='run MATLAB routine to calculate polarisation')
    parser.add_argument('-n', '--number', type=int,
                        help='set the number of parties, otherwise 2 if -1d or 4')
    parser.add_argument('-r', '--ranking', action='store_true',
                        help='preserve the left-right ordering of the parties, requires --line')
    parser.add_argument('-p1', '--turnoutparameter', type=float, default=1.0,
                        help='scales effects of imperfect turnout, farther voters contribute less')
    parser.add_argument('-p2', '--activistparameter', type=float, default=0.5,
                        help='scales how effectively activists influence their parties, positive '
                             'value gives a tendency towards extremes, negative a central tendency')
    parser.add_argument('-d', '--distributionchoice', choices=['s', 'm', 'u'], default='s',
                        help='choose the voter distribution, either with a single central '
                             'peak (s), a more mildly peaked distribution (m) or the '
                             'uniform distribution (u)')
    parser.add_argument('-s', '--stats', action='store_true',
                        help='print detailed results after the simulation')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print status during execution (includes -s)')

    grid = OpinionGrid()
    args = parser.parse_args()

    if args.ranking and not args.line:
        args.line = True
        print('Incompatible options (--ranking). Set --line to True.')

    # Default party number if not set
    if args.number is None:
        if args.line:
            args.number = 2
        else:
            args.number = 4

    # Check whether to simulate 1d Hotelling and set the distribution
    if args.line:
        grid.weight = np.zeros_like(grid.weight)
        grid.addParties(LineParty.test(args.number))
        match args.distributionchoice:
            case 's':
                grid.weight[:, int(np.size(grid.xGrid, axis=1)/2)] = np.cos(grid.xGrid[:, 0] * np.pi/2)**2
            case 'm':
                grid.weight[:, int(np.size(grid.xGrid, axis=1)/2)] = np.cos(grid.xGrid[:, 0] * np.pi/2)**0.5
            case 'u':
                grid.weight[:, int(np.size(grid.xGrid, axis=1)/2)] = 1
    else:
        grid.addParties(Party.test(args.number))
        match args.distributionchoice:
            case 's':
                pass
            case 'm':
                grid.weight = grid.weight ** 0.25
            case 'u':
                grid.weight[:, :] = 1

    # Cov matrix, assume identity matrix and scale
    # FWQM about 1.6651 sqrt(var)
    covFactor = 0.02
    covMatrix = covFactor*np.array([[1, 0], [0, 1]])
    positions = np.zeros((len(grid.parties), args.steps+1, np.ndim(grid.weight)))
    positions[:, 0] = [p.position for p in grid.parties]
    turnouts = np.zeros((2, args.steps+1))
    # positions = positions.tolist()  for grand canonical
    stepNum = 0

    # Plotting
    if args.scatter:
        figSt, axSt = plt.subplots()

    if args.verbose:
        print(args.turnoutparameter, args.activistparameter)
        print('Setup complete.')

    # Core simulation loop
    # --------------------
    # The order of parties is randomised during the step to avoid bias through
    # random turn orders. Then the list is sorted again to get the data in the
    # right order. This could be slow, but fine for relatively few parties.
    # The system is not particularly temperature-sensitive, values above 1 blur
    while stepNum < args.steps:
        metropolisStep(grid, multivariate_normal, covMatrix, 1e-30,
                       args.ranking, args.turnoutparameter, args.activistparameter)

        stepNum += 1
        turnouts[:, stepNum] = grid.turnout
        grid.parties.sort(key=lambda p: p.name)
        for i, party in enumerate(grid.parties):
            positions[i, stepNum] = party.position

        if args.verbose and stepNum % 50 == 0:
            print(f"Step {stepNum}", end=' ... ')

    if args.verbose:
        print("Printing...")

    helper.printTrajectories(args.output, list([p.name for p in grid.parties]),
                             positions, to3D=True)
    if args.tab:
        helper.printTabSeparated(positions)

    if args.verbose or args.stats:
        print('\nFinal positions | votes | non-activist votes')
        [print(f'\t{p.name}: {p.position} | {p.votes:.1f} votes | {p.realVotes:.1f} non-activist votes') for p in grid.parties]
        print(f'\033[4m{max(grid.parties, key=lambda p: p.votes).name}\033[0m won, but '
              f'\033[4m{max(grid.parties, key=lambda p: p.realVotes).name}\033[0m '
              f'won by votes excluding activist influence.')
        print(f'\nTurnout without activists was {grid.turnout[0]:.1f}% after the'
              f' end of the simulation and on average {np.mean(turnouts[0, 1:]):.1f}%.')
        print(f'Turnout with activists was {grid.turnout[1]:.1f}% after the'
              f' end of the simulation and on average {np.mean(turnouts[1, 1:]):.1f}%.')

    if args.verbose:
        print('Plotting...')

    # Generate the scatter plot
    if args.scatter:
        axSt.scatter(positions[:, :, 0], positions[:, :, 1], s=10, color='gray')
        for party in grid.parties:
            axSt.scatter(party.position[0], party.position[1], marker='o', s=150)
            axSt.scatter(party.centreOfBase[0], party.centreOfBase[1], marker='x', s=100)
            axSt.set_xlim(-1, 1)
            axSt.set_ylim(-1, 1)
            figSt.suptitle(f'Turnout {args.turnoutparameter}, activists {args.activistparameter}')
        figSt.show()

    # Histograms
    if not args.nohistogram:
        for i in range(0, len(grid.parties)):
            figH, axH = plt.subplots()
            axH.hist2d(positions[i, :, 0], positions[i, :, 1], bins=25)
            axH.set_xlim(-1, 1)
            axH.set_ylim(-1, 1)
            axH.set_xlabel('x')
            axH.set_ylabel('y')
            axH.set_facecolor('#440154')
            figH.suptitle(f'Party {i+1}')
            figH.show()

    plt.show()
    print(f'Simulated {stepNum} steps.')

    if args.matpol:
        if args.verbose:
            print("Calculating polarisation with MATLAB.")
        try:
            import matlab.engine
            eng = matlab.engine.start_matlab()
            eng.addpath('./Matlab')
            print("Polarisation measure:", eng.get_mean_polarisation(args.output))
        except:
            warnings.warn("Failed to access MATLAB, so no polarisation measure.")

    # Voter distribution
    if args.distribution:
        grid.plotDistribution()


if __name__ == '__main__':
    main()