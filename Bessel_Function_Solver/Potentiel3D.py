import numpy as np
from utils import m_puit_e, m_ext_e
 
class Potentiel3D() :
    def __init__(self, type2boite, constantes, m_puit=m_puit_e, m_ext=m_ext_e):
        self.type2boite = type2boite
        self.constantes = constantes
        self.m_puit = m_puit
        self.m_ext = m_ext
 
    def makeTermes(self, r, z) :
        # Je retourne V et 1/m sous forme de termes (c, g(r), h(z)) : V = somme des c * g(r) * h(z)
 
        if self.type2boite == "cylindre":
            # constante = [rayon, hauteur, profondeur]
            R, h, V0 = self.constantes
            in_r = (r < R).astype(float)
            in_z = (np.abs(z) < h/2).astype(float)
 
            termes_V = [(V0, in_r, in_z)]
            termes_inv_m = [(1/self.m_ext, np.ones_like(r), np.ones_like(z)), (1/self.m_puit - 1/self.m_ext, in_r, in_z)]
            return termes_V, termes_inv_m
 
        elif self.type2boite == "anneau":
            # constante = [petit rayon, grand rayon, hauteur, profondeur]
            L1, L2, h, V0 = self.constantes
            in_r = ((r > L1) & (r < L2)).astype(float)
            in_z = (np.abs(z) < h/2).astype(float)
 
            termes_V = [(V0, in_r, in_z)]
            termes_inv_m = [(1/self.m_ext, np.ones_like(r), np.ones_like(z)), (1/self.m_puit - 1/self.m_ext, in_r, in_z)]
            return termes_V, termes_inv_m
 
        elif self.type2boite == "harmonique":
            # constante = [k_r, k_z] (raideurs), V = 1/2 k_r r^2 + 1/2 k_z z^2 : une somme donc deux termes
            k_r, k_z = self.constantes
 
            termes_V = [(0.5 * k_r, r**2, np.ones_like(z)), (0.5 * k_z, np.ones_like(r), z**2)]
            termes_inv_m = [(1/self.m_puit, np.ones_like(r), np.ones_like(z))]
            return termes_V, termes_inv_m
        else :
            raise ValueError(f"Type de potentiel inconnu : {self.type2boite!r}")
 
    def makeCarte(self, r, z) :
        # Je retourne V(r,z) et m(r,z) en 2D (forme (len(r), len(z))), pour vérifier les termes et tracer
 
        if self.type2boite == "cylindre":
            # constante = [rayon, hauteur, profondeur]
            R, h, V0 = self.constantes
            in_the_well = (r[:,None] < R) & (np.abs(z[None,:]) < h/2)
 
            return np.where(in_the_well, V0, 0), np.where(in_the_well, self.m_puit, self.m_ext)
 
        elif self.type2boite == "anneau":
            # constante = [petit rayon, grand rayon, hauteur, profondeur]
            L1, L2, h, V0 = self.constantes
            in_the_well = ((r[:,None] > L1) & (r[:,None] < L2)) & (np.abs(z[None,:]) < h/2)
 
            return np.where(in_the_well, V0, 0), np.where(in_the_well, self.m_puit, self.m_ext)
 
        elif self.type2boite == "harmonique":
            # constante = [k_r, k_z] (raideurs)
            k_r, k_z = self.constantes
            V = 0.5 * k_r * r[:,None]**2 + 0.5 * k_z * z[None,:]**2
            return V, np.full_like(V, self.m_puit)
        else :
            raise ValueError(f"Type de potentiel inconnu : {self.type2boite!r}")
 