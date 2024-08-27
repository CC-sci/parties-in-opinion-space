import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from party import Party


class OpinionGrid:
    """
    A two-dimensional grid on which parties move.

    The opinion grid is where parties and voters are located. Its most important
    job is to calculate how many votes every party gets at its current position.

    The grid encodes the position of voters, specifically how many voters/how
    much weight each grid point/opinion has. For example, it can be a uniform
    distribution where each point has one vote, or a distribution with a maximum
    in the centre, etc. It is an abstraction of political views, like a
    political spectrum if it is limited to one dimension, or a political compass.
    The grid then assigns voters to a party based on which party is closets, and
    may also account for factors such as turnout or activists.

    Attributes:
        parties: List of Party objects on the grid.
        xGrid: Together with yGrid forms the grid by listing all coordinates in
               arrays, see ``numpy.meshgrid``.
        yGrid: See ``xGrid``.
        weight: Voter distribution, 2D array.
    """

    # Create two lines and take the Cartesian product to generate the grid
    # Make the number odd so there is a centre
    # This meshgrid is actually quite unwieldy, it would be easier if it were
    # scaled so that all coordinates were integers
    xGrid, yGrid = np.meshgrid(np.linspace(-1, 1, num=101),
                               np.linspace(-1, 1, num=101), indexing='ij')
    weight = np.cos((xGrid**2 + yGrid**2)**0.5 * np.pi/2)**2
    # weight = np.sin((xGrid**2 + yGrid**2)**0.5 * np.pi/1)**2


    def __init__(self):
        self.parties = []


    def plotDistribution(self):
        """Plots the voter distribution."""
        distribution = plt.figure().add_subplot(projection='3d')
        distribution.plot_surface(self.xGrid, self.yGrid, self.weight)
        plt.show()

        plt.scatter(self.xGrid, self.yGrid, self.weight)
        plt.show()

        plt.contourf(self.xGrid, self.yGrid, self.weight)
        plt.colorbar()
        plt.show()


    def addParties(self, parties: list[Party]):
        """Adds a list of Party objects to the grid."""
        for party in parties:
            if isinstance(party, Party):
                self.parties.append(party)
            else:
                raise TypeError('Is not an instance of Party')


    def findPartyVoters(self, x, y, turnoutParam, activistParam):
        """
        Calculates the points closest to each party and adds their weight to the
        parties' vote count.

        The parties and voter distribution are properties of the grid.
        The parameters can be lists (such as the whole grid) in which case this
        method acts element-wise. `Usually`, x and y will be outputs of
        ``numpy.meshgrid``. This method modifies Party objects.

        Turnout is accounted for by reducing the given vote according to a
        polynomial (default inverse square) with distance from the party.

        Activists are accounted for by weighting the vote by distance from the
        origin. A positive number favours extremes, a negative number the centre.

        This method also sets the centre of the party's voters. This is the
        centre of each one's Voronoi polyhedral, weighted by votes, and is a
        property of the party.

        This method can cause runtime warnings if a party has no votes.

        ToDo: Also calculate real vote to be able to determine winners.

        :param x: x-coordinates
        :param y: y-coordinates
        :param float turnoutParam: larger the quicker turnout drops off with distance
        :param float activistParam: larger the more extreme tendency
        :return: (distance, closestParties) tuple which can be a tuple of arrays.
                 `Note:` The first term gives the distance squared of every point
                  on the grid to every party. Size: (number of parties, x grid
                  size, y grid size).
                 The second term gives the nearest ``Party`` instance.
        :rtype: (ndarray[float], ndarray[float])
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
        # The index of the closest party to every point is given by closest. The
        # corresponding object is in closestParties.
        distances = np.array(distances)
        closest = np.argmin(distances, axis=0)
        closestParties = np.array(self.parties)[closest]

        # Finds the centre of each party's voters by setting others to zero and
        # finding the centre of mass in terms of indices
        for i in range(len(self.parties)):
            theseVotes = np.where(closest == i, self.weight, 0.0)
            # This can prompt a divide by zero warning if a party has no votes
            self.parties[i].centreOfBase = (np.array(center_of_mass(theseVotes))
                                            / 50 - 1)

        # This is an inverse square law (for turnoutParameter=1)
        # 30-fold speed increase compared to for loop
        # This method maps the coordinates to their indices so that a coordinate's
        # value can be used as its index
        # Adds one to translate into the first quadrant, then scales to integer
        xAsInt = np.rint((self.xGrid + 1) * 50).astype(int)
        yAsInt = np.rint((self.yGrid + 1) * 50).astype(int)
        turnoutVotes = self.weight / ((1 + distances[closest[xAsInt, yAsInt],
                                                    xAsInt, yAsInt]) ** turnoutParam)
        # This is the effect of extreme activists on a party
        activistVotes = turnoutVotes * ((x**2 + y**2) ** activistParam)

        # If I used map it would double the speed, ndindex for just index
        for index, party in np.ndenumerate(closestParties):
            party.votes += activistVotes[index]

        # Returns distance and closest party of every point
        # But I don't currently use this
        return distances, closestParties


    def energy(self, turnoutParameter, activistParameter):
        """
        Public-facing method for calculating all parties' energies/votes.

        This method calls ``findPartyVoters`` on the whole grid.
        :param float turnoutParameter: Degree of vote falloff with distance.
        :param float activistParameter: Degree of vote increase with polarisation.
        :return: List of votes of each party in ``parties``.
        """
        distanceSquared, closestParty = self.findPartyVoters(self.xGrid, self.yGrid,
                                                             turnoutParameter, activistParameter)
        energies = []

        for party in self.parties:
            energies.append(party.votes)
            # print(party.name, party.votes, party.position)

        return energies