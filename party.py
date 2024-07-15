import numpy as np
from scipy.stats import uniform


class Party:

    def __init__(self, name, position):
        self.name = name
        self.position = np.array(position)
        self.votes = 0
        self.centreOfBase = None


    def move(self, x: float, y: float):
        """
        Translates the party by (x,y).
        :param x: x-shift
        :param y: y-shift
        """
        self.position += np.array([x, y])


    @classmethod
    def test(cls):
        party1 = Party.oneTest(position=[uniform.rvs(-1, 2), uniform.rvs(-1, 2)])
        party2 = Party.oneTest('Party-2', [uniform.rvs(-1, 2), uniform.rvs(-1, 2)])
        party3 = Party.oneTest('Party-3', [uniform.rvs(-1, 2), uniform.rvs(-1, 2)])
        party4 = Party.oneTest('Party-4', [uniform.rvs(-1, 2), uniform.rvs(-1, 2)])


        return [party1, party2, party3, party4]


    @classmethod
    def oneTest(cls, name='Party-1', position=[-0.2, -0.2]):
        return cls(name, position)


class LineParty(Party):

    def move(self, x: float, y=0.0):
        """
        Translates the party by x, regardless of y.
        :param x: x-shift
        :param y: inherited vestige
        """
        Party.move(self, x, 0)


    @classmethod
    def test(cls):
        """
        Instantiates and returns two parties at random positions on the line.
        """
        party1 = LineParty.oneTest('Party-1', [-uniform.rvs(), 0])
        party2 = LineParty.oneTest('Party-2', [uniform.rvs(), 0])

        return [party1, party2]