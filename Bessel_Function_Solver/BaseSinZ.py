import numpy as np
import math
import utils
 
class BaseSinZ() :
    def __init__(self, hauteur, n_point, m_ref, E_cut, acc = 1e-4):
        """
        hauteur : hauteur de la boîte en z, les fonctions s'annulent en z = ±hauteur/2
        E_cut : Combien de fonctions on prends pour la base (même E_cut que BaseBessel)
        """
 
        self.hauteur = hauteur
        self.mass = m_ref
 
        self.z = np.linspace(-hauteur/2, hauteur/2, num=n_point)
        self.dz = hauteur/(n_point-1)
        self.poids = np.full(n_point, self.dz)
        self.poids[0] = self.poids[-1] = self.dz / 2 # poids dz/2 aux deux extrémités (trapèzes)
 
        q_cut = hauteur * np.sqrt(2*self.mass*E_cut) / utils.hbar

        k_max = int(q_cut / np.pi) # plus grand k tel que k*pi <= j_cut
        self.k = np.arange(1, k_max + 1)
 
        self.q = self.k * np.pi / hauteur
 
        self.norm = math.sqrt(2 / hauteur) * np.ones(len(self.k))
        self.est_pair = (self.k % 2 == 1) # k impair -> fonction paire en z, k pair -> fonction impaire
 
        self.E = utils.hbar**2 * self.q ** 2 / (2*self.mass)
        self.chi = self.norm[:,None] * np.sin(self.q[:,None] * (self.z[None,:] + hauteur/2))
        self.chi_prime = self.norm[:,None] * self.q[:,None] * np.cos(self.q[:,None] * (self.z[None,:] + hauteur/2))
 
        erreur_max, erreur_mse = self.verif_ortho()
 
        if erreur_max >= acc :
            print(f"Matrice de Base Sin Z Bizarre (Code BSZB) | max = {erreur_max} | MSE = {erreur_mse}")
        else :
            print(f"Matrice de Base Sin Z pas Bizarre | max = {erreur_max} | MSE = {erreur_mse}")
 
    def verif_ortho(self):
        """
        Vérifie que la base est orthonormée : renvoie l'erreur maximale
        et l'erreur quadratique moyenne entre la matrice de recouvrement et l'identité.
        """
        N = len(self.k)
 
        # 1. Recouvrement : ∫ χ_k χ_k' dz, pour tous les couples (k, k')
        ortho = (self.chi * self.poids[None, :]) @ self.chi.T
 
        identite = np.eye(N)
        erreur_max = np.max(np.abs(ortho - identite))
        erreur_mse = np.mean((ortho - identite)**2)
        return erreur_max, erreur_mse
 