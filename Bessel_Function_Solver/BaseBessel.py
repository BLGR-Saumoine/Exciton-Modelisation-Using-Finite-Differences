import numpy as np
import scipy.special
import math
import utils

class BaseBessel() :
    def __init__(self, radius, n_base, n_point, m, acc = 1e-4):

        self.radius = radius
        self.mass = m

        self.r = np.linspace(0, radius, num=n_point)
        self.dr = radius/(n_point-1)

        self.zeroes = scipy.special.jn_zeros(0, n_base)

        self.alpha = self.zeroes / radius

        self.norm = 1/(math.sqrt(math.pi) * radius * abs(scipy.special.jv(1,self.zeroes)))

        self.E = utils.hbar**2 * self.zeroes ** 2 / (2*m*radius**2)

        self.phi = self.norm[:,None] * scipy.special.jv(0,self.r[None,:] * self.alpha[:,None])

        maxVal = self.verif_ortho(n_base)
        if maxVal >= acc :
            print(f"Matrice de Base Bessel Bizarre (Code BBB) | max = {maxVal}")
        else :
            print(f"Matrice de Base Bessel pas Bizarre | max = {maxVal}")

    def verif_ortho(self, n_base)  :

        ortho = (self.phi * self.r[None,:]) @ self.phi.T * 2 * np.pi * self.dr
        identity = np.eye(n_base)

        max = np.max(np.abs(ortho - identity))

        return max



