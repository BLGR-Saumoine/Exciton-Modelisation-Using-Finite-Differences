import numpy as np
import scipy.special
import math
import utils

class BaseBessel() :
    def __init__(self, radius, n_base, n_point, m):

        self.radius = radius
        self.mass = m

        self.r = np.linspace(0, radius, num=n_point)
        self.dr = radius/(n_point-1)

        self.zeroes = scipy.special.jn_zeros(0, n_base)

        self.alpha = self.zeroes / radius

        self.norm = 1/(math.sqrt(math.pi) * radius * abs(scipy.special.jv(1,self.zeroes)))

        self.E = utils.hbar**2 * self.zeroes ** 2 / (2*m*radius**2)

        self.phi = 

