import numpy as np
class Potentiel() :
    def __init__(self, type2boite, constantes):
        self.type2boite = type2boite
        self.constantes = constantes
        

    def makePotentiel(self, r) :

        if self.type2boite == "disque":
            # constante = [rayon, profondeur]
            L, V0 = self.constantes
            in_the_well = (np.abs(r) < L)

            return np.where(in_the_well, V0, 0)
        
        elif self.type2boite == "anneau":
            # constante = [petit rayon, grand rayon, profondeur]
            L1, L2, V0 = self.constantes
            in_the_well = ((np.abs(r) > L1) & (np.abs(r) < L2))

            return np.where(in_the_well, V0, 0)
        
        elif self.type2boite == "harmonique":
            # constante = k (raideur)
            k = self.constantes
            return 0.5 * k * r**2
        else :
            return np.zeros_like(r)