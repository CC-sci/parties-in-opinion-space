import numpy as np
from scipy.stats import uniform


class Party:
    """
    This class represents a political party or more generally a particle moving in a space.

    It does not do calculations itself, but rather serves to store properties.

    Attributes:
        name: A label.
        position: The current position in 2D.
        votes: The current votes/energy.
        realVotes: The current votes not accounting for activists.
        centreOfBase: The location of the centre of the party's voters, i.e. its weighted Voronoi polyhedral.

    Public Methods:
        move(x, y): Move the party's position by the vector (x, y).

    Class Methods:
        test(): Create four parties at random positions.
    """

    def __init__(self, name: str, position: list[float]):
        self.name = name
        self.position = np.array(position)
        self.votes = 0
        self.realVotes = 0
        self.centreOfBase = self.position


    def move(self, x: float, y: float):
        """
        Translates the party by (x,y).
        :param float x: x-shift
        :param float y: y-shift
        """
        self.position += np.array([x, y])


    @classmethod
    def test(cls, n):
        """
        Create and return `n` parties at random positions.
        :param int n: Number of parties.
        :return: A list of party objects.
        """
        parties = []
        for i in range(n):
            parties.append(Party.oneTest(f'Party-{i+1}', [uniform.rvs(-1, 2), uniform.rvs(-1, 2)]))
        return parties


    @classmethod
    def oneTest(cls, name='Party-1', position=[-0.2, -0.2]):
        return cls(name, position)


class LineParty(Party):
    """
    This class represents a political party/particle moving on a line.

    It is analogous to its parent class, ``Party``, in one dimension. See the parent class for documentation.
    """

    def move(self, x: float, y=0.0):
        """
        Translates the party by x, regardless of y.
        :param x: x-shift
        :param y: inherited vestige
        """
        Party.move(self, x, 0)


    @classmethod
    def test(cls, n):
        """
        Instantiates and returns `n` parties at random positions on the line.
        """
        parties = []
        for i in range(n):
            parties.append(LineParty.oneTest(f'Party-{i+1}', [-uniform.rvs(), 0]))
        return parties