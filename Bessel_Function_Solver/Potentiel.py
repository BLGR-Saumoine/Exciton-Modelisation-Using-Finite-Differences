import numpy as np
from utils import m_puit_e, m_ext_e

class Potentiel() :
    def __init__(self, type2boite, constantes, m_puit=m_puit_e, m_ext=m_ext_e):
        self.type2boite = type2boite
        self.constantes = constantes
        self.m_puit = m_puit
        self.m_ext = m_ext

    def makePotentiel(self, r) :
        # Je retourne aussi la masse 

        if self.type2boite == "disque":
            # constante = [rayon, profondeur]
            L, V0 = self.constantes
            in_the_well = (np.abs(r) < L)

            return np.where(in_the_well, V0, 0), np.where(in_the_well, self.m_puit, self.m_ext)

        elif self.type2boite == "anneau":
            # constante = [petit rayon, grand rayon, profondeur]
            L1, L2, V0 = self.constantes
            in_the_well = ((np.abs(r) > L1) & (np.abs(r) < L2))

            return np.where(in_the_well, V0, 0), np.where(in_the_well, self.m_puit, self.m_ext)

        elif self.type2boite == "harmonique":
            # constante = k (raideur)
            k = self.constantes
            return 0.5 * k * r**2, np.full_like(r, self.m_puit)
        else :
            return np.zeros_like(r), np.full_like(r, self.m_puit) 