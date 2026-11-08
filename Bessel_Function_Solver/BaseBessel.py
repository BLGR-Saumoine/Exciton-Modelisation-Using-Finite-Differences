import numpy as np
import scipy.special
import math
import utils

class BaseBessel() :
    def __init__(self, radius, n_point, mass, E_cut, acc = 1e-4):
        """
        E_cut : Combien de fonctions on prends pour la base
        E_max : On ne montre pas les fonctions avec une énergie supérieurs à E_max il sera dans Hamiltonien mais je le précise ici
        """

        self.radius = radius
        self.mass = mass

        self.r = np.linspace(0, radius, num=n_point)
        self.dr = radius/(n_point-1)
        self.poids = np.full(n_point, self.dr)
        self.poids[0] = self.poids[-1] = self.dr / 2 # poids dr/2 aux deux extrémités (trapèzes)
        self.zeroes = []
        self.m = []
        self.n = []
        self.types = []
        self.angulaire = None

        j_cut = self.radius * np.sqrt(2*self.mass*E_cut) / utils.hbar

        m_courant = 0
        n = 1
        while True :
            a = scipy.special.jn_zeros(m_courant, n)[-1]

            if a <= j_cut and m_courant == 0:
                self.zeroes.append(a)
                self.m.append(m_courant)
                self.n.append(n)
                self.types.append(np.cos)
                
                n += 1

            elif a <= j_cut and m_courant > 0:
                self.zeroes.append(a)
                self.zeroes.append(a)
                self.m.append(m_courant)
                self.m.append(m_courant)
                self.n.append(n)
                self.n.append(n)
                self.types.append(np.cos)
                self.types.append(np.sin)
                
                n += 1

            elif a > j_cut and n > 1:
                m_courant += 1
                n = 1

            elif a > j_cut and n == 1 :
                break
            else :
                print(f"Cas pas prévu : jn_zero : {scipy.special.jn_zeros(1, n)}, n : {n}, m : {m_courant}")

        self.zeroes = np.array(self.zeroes)
        self.m = np.array(self.m)
        self.n = np.array(self.n)

        self.alpha = self.zeroes / radius

        area = np.where(self.m == 0, 1, np.sqrt(2))
        self.norm = area/(math.sqrt(math.pi) * radius * abs(scipy.special.jv(self.m + 1, self.zeroes))) # Pour m >= 1 on a pi au lieu de 2pi dans la normalisation

        self.E = utils.hbar**2 * self.zeroes ** 2 / (2*mass*radius**2)
        self.phi = self.norm[:,None] * scipy.special.jv(self.m[:,None],self.r[None,:] * self.alpha[:,None])
        self.phi_prime = self.norm[:,None] * self.alpha[:,None] / 2 * (scipy.special.jv(self.m[:,None]-1,self.r[None,:] * self.alpha[:,None]) - scipy.special.jv(self.m[:,None]+1,self.r[None,:] * self.alpha[:,None]))
        self.getAngulaire()

        erreur_max, erreur_mse = self.verif_ortho()

        if erreur_max >= acc :
            print(f"Matrice de Base Bessel Bizarre (Code BBB) | max = {erreur_max} | MSE = {erreur_mse}")
        else :
            print(f"Matrice de Base Bessel pas Bizarre | max = {erreur_max} | MSE = {erreur_mse}")

    def verif_ortho(self):
        """
        Vérifie que la base est orthonormée : renvoie l'erreur maximale
        et l'erreur quadratique moyenne entre la matrice de recouvrement et l'identité.
        """
        N = len(self.zeroes)

        # 1. Partie radiale : ∫ Φ_a Φ_b r dr, pour tous les couples (a, b)
        radial = (self.phi * (self.r * self.poids)[None, :]) @ self.phi.T

        ortho = radial * self.angulaire

        identite = np.eye(N)
        erreur_max = np.max(np.abs(ortho - identite))
        erreur_mse = np.mean((ortho - identite)**2)
        return erreur_max, erreur_mse

    def getAngulaire(self) :
        # 2. Partie angulaire : ∫ A_a A_b dθ, pour tous les couples     (a, b)
        self.est_sin = np.array([t is np.sin for t in self.types])       # True si la fonction est en sin
        meme_m = self.m[:, None] == self.m[None, :]                   # True si m_a == m_b
        meme_type = self.est_sin[:, None] == self.est_sin[None, :]              # True si même type (cos/cos ou sin/sin)

        valeur = np.where(self.m[:, None] == 0, 2*np.pi, np.pi)       # 2π si m = 0, π sinon
        self.angulaire = np.where(meme_m & meme_type, valeur, 0.0)         # 0 si m différents ou types différents
        


