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


    def plot(self, lvl=3, n_theta=360, rotate=True, density=False, dessus=True):

        if self.eigenvectors is None:
            self.solve()

        m = self.base.m

        psi_all = self.eigenvectors[:, :lvl].T @ self.base.phi

        if density:
            amplitude_max = np.max(np.abs(psi_all)**2)
        else:
            amplitude_max = np.max(np.abs(psi_all))

        if lvl > 1:
            ecart = np.min(np.diff(self.energies[:lvl]))
        else:
            ecart = np.abs(np.min(self.V_r))

        facteur = 0.3 * ecart / amplitude_max

        # Grille polaire, utilisée par la 3D et par la vue de dessus
        angle = np.linspace(0, 2*np.pi, num=n_theta)
        R_grid, Theta_grid = np.meshgrid(self.base.r, angle)
        X = R_grid * np.cos(Theta_grid)
        Y = R_grid * np.sin(Theta_grid)

        if rotate:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        else:
            x_coupe = np.concatenate((-self.base.r[::-1], self.base.r[1:]))
            V_coupe = np.concatenate((self.V_r[::-1], self.V_r[1:]))
            fig, ax = plt.subplots()
            ax.plot(x_coupe, V_coupe, color='black', label='V(x)')

        if dessus:
            fig_d, axes_d = plt.subplots(1, lvl, figsize=(4*lvl, 4), squeeze=False)
            axes_d = axes_d[0]

        for i in range(lvl):
            psi = psi_all[i]

            psi_2D = psi[None, :] * np.cos(m * Theta_grid)
            if density:
                psi_2D = np.abs(psi_2D)**2

            if rotate:
                ax.plot_surface(X, Y, self.energies[i] + facteur * psi_2D, alpha=0.7)
            else:
                psi_coupe = np.concatenate(((-1)**m * psi[::-1], psi[1:]))
                if density:
                    psi_coupe = np.abs(psi_coupe)**2
                ax.axhline(self.energies[i], linestyle='--', linewidth=0.5)
                ax.plot(x_coupe, self.energies[i] + facteur * psi_coupe,
                        label=f"E{i} = {self.energies[i]:.1f} meV")

            if dessus:
                a = axes_d[i]
                if density:
                    im = a.pcolormesh(X, Y, psi_2D, cmap='inferno', shading='gouraud')
                else:
                    vmax = np.max(np.abs(psi_2D))
                    im = a.pcolormesh(X, Y, psi_2D, cmap='RdBu_r',
                                    vmin=-vmax, vmax=vmax, shading='gouraud')
                a.set_aspect('equal')
                a.set_title(f"E{i} = {self.energies[i]:.1f} meV")
                a.set_xlabel("x (nm)")
                fig_d.colorbar(im, ax=a, shrink=0.8)

        ax.set_xlabel("x (nm)")
        temp = "Densités des " if density else ""
        if rotate:
            fig.suptitle(f"{temp}états m = {m}, vue en rotation")
            ax.set_ylabel("y (nm)")
            ax.set_zlabel("E (meV)")
        else:
            fig.suptitle(f"{temp}états m = {m}, vue de coupe")
            ax.set_ylabel("E (meV)")
            ax.legend()

        if dessus:
            axes_d[0].set_ylabel("y (nm)")
            fig_d.suptitle(f"{temp}états m = {m}, vue de dessus")
            fig_d.tight_layout()

        plt.show()