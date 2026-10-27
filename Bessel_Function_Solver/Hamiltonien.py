from Potentiel import Potentiel
from BaseBessel import BaseBessel
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

class Hamiltonien():

    def __init__(self, baseBessel : BaseBessel, potentiel : Potentiel):
        self.base = baseBessel
        self.potentiel = potentiel
        self.V_r = self.potentiel.makePotentiel(baseBessel.r)
        self.Vnm = self.build_V()
        self.energies = None
        self.eigenvectors = None
        self.H = np.diag(self.base.E) + self.Vnm


    def build_V(self) :
        V_nm = (self.base.phi * (self.base.r[None,:] * self.V_r[None,:])) @ self.base.phi.T * 2 * np.pi * self.base.dr
        return V_nm

    def solve(self) :
        energies, eigenvectors = eigh(self.H)
        self.energies = energies
        self.eigenvectors = eigenvectors
        return energies, eigenvectors



    def plot(self, lvl=3, n_theta = 360, rotate=True, density = False) :

        psi_all = self.eigenvectors[:, :lvl].T @ self.base.phi
        if density :
            psi_all = np.abs(psi_all)**2

        amplitude_max = np.max(np.abs(psi_all))
        if lvl > 1:
            ecart = np.min(np.diff(self.energies[:lvl]))
        else:
            ecart = np.abs(np.min(self.V_r))

        facteur = 0.3 * ecart / amplitude_max

        if rotate :
            angle = np.linspace(0, 2*np.pi, num=n_theta)
            R_grid, Theta_grid = np.meshgrid(self.base.r, angle)
            X = R_grid * np.cos(Theta_grid)
            Y = R_grid * np.sin(Theta_grid)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        else :
            x_coupe = np.concatenate((-self.base.r[::-1], self.base.r[1:]))
            V_coupe = np.concatenate((self.V_r[::-1], self.V_r[1:]))
            fig, ax = plt.subplots()
            ax.plot(x_coupe, V_coupe, color='black', label='V(x)')

        for i in range(lvl) :
            psi = psi_all[i]
            if rotate :
                psi_2D = np.broadcast_to(psi, X.shape)
                ax.plot_surface(X, Y, self.energies[i] + facteur * psi_2D, alpha=0.7)
            else :
                psi_coupe = np.concatenate((psi[::-1], psi[1:]))
                ax.axhline(self.energies[i], linestyle='--', linewidth=0.5)
                ax.plot(x_coupe, self.energies[i] + facteur * psi_coupe, label=f"E{i} = {self.energies[i]:.1f} meV")

        ax.set_xlabel("x (nm)")
        temp = "Densités des " if density else ""
        if rotate :
            fig.suptitle(f"{temp}États pairs (uniquement) modélisés par rotation")
            ax.set_ylabel("y (nm)")
            ax.set_zlabel("E (meV)")
        else :
            fig.suptitle(f"{temp}États pairs vue de coupe")
            ax.set_ylabel("E (meV)")
            ax.legend()
        plt.show()