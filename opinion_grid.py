import numpy as np
import matplotlib.pyplot as plt
import math
from party import Party


class OpinionGrid:

    # ToDo: Customise these in an initialiser, currently they are static
    # Create two lines and take the Cartesian product to generate the grid
    # Make the number odd so there is a centre
    xGrid, yGrid = np.meshgrid(np.linspace(-1, 1, num=101),
                               np.linspace(-1, 1, num=101))
    weight = np.cos((xGrid**2 + yGrid**2)**0.5 * math.pi/2)**2


    def __init__(self):
        self.parties = []


    def plotDistribution(self):
        distribution = plt.figure().add_subplot(projection='3d')
        distribution.plot_surface(self.xGrid, self.yGrid, self.weight)
        plt.show()

        plt.scatter(self.xGrid, self.yGrid, self.weight)
        plt.show()

        plt.contourf(self.xGrid, self.yGrid, self.weight)
        plt.colorbar()
        plt.show()


    # Todo: Support for adding one party
    def addParties(self, parties: list):
        for party in parties:
            if isinstance(party, Party):
                self.parties.append(party)
            else:
                raise TypeError('Is not an instance of Party')


    def findPartyVoters(self, x, y):
        """
        Calculates the points closest to each party and adds their weight to the
        parties' vote count.

        The parties and voter distribution are properties of the grid.
        The parameters can be lists (such as the whole grid) in which case this
        method acts element-wise. This method modifies Party objects.

        ToDo: Currently does not support (x,y) being a single coordinate.

        :param x: x-coordinates
        :type: [float] array (or float)
        :param y: y-coordinates
        :type: [float] array
        :return: (distance, closestParties) tuple where can be a tuple of arrays
        :rtype: tuple of (distance, closestParties)
        """
        distances = []

        # The distance of every point to every party
        for party in self.parties:
            # There is no need for expensive sqrt here since ranking unchanged
            thisDistance = (party.position[0]-x)**2 + (party.position[1]-y)**2
            distances.append(thisDistance)
            party.votes = 0

        # Distances is e.g. (3,5,5) — (parties, x, y)
        # So this minimises w.r.t. party
        distances = np.array(distances)
        closest = np.argmin(distances, axis=0)
        closestParties = np.array(self.parties)[closest]
        # ToDo: Check explicitly, this is an inverse square law
        turnoutVotes = self.weight / (1 + distances[closest[0], closest[1]])

        # ToDo: Chokepoint, how can I make this faster
        for index, party in np.ndenumerate(closestParties):
            party.votes += turnoutVotes[index]

        # Returns distance and closest party of every point
        # But I don't currently use this
        return np.sqrt(distances), closestParties


    def energy(self):
        distance, party = self.findPartyVoters(self.xGrid, self.yGrid)
        energies = []

        # Plots
        # distribution = plt.figure().add_subplot(projection='3d')
        # # distribution.plot_surface(self.xGrid, self.yGrid, distance[0])
        # distribution.plot_surface(self.xGrid, self.yGrid, distance[1])
        # # #distribution.plot_surface(self.xGrid, self.yGrid, distance[2])
        # plt.show()

        # plt.contourf(self.xGrid, self.yGrid, distance[0])
        # plt.colorbar()
        # plt.show()

        # ToDo: Party 1 is favoured because ties are currently given to the lowest index
        # Might now be fixed because party list is shuffled every turn
        for party in self.parties:
            energies.append(party.votes)
            # print(party.name, party.votes, party.position)

        return energies