"""
Constantes physiques du problème.

Système d'unités :
    énergies  : meV
    longueurs : nm
    masses    : en unités de m0 (masse de l'électron libre)
"""

# ======= Constante de Planck réduite =======
# hbar^2 / (2 m0) = 38.1 meV.nm^2  =>  hbar^2 = 76.2  =>  hbar = 8.729
hbar = 8.729

# ======= Masses effectives (en m0) =======
# Électron
m_puit_e = 0.067   # GaAs
m_ext_e  = 0.15    # barrière

# Trou
m_puit_h = 0.45
m_ext_h  = 0.60

# ======= Interaction de Coulomb =======
epsilon   = 12.5             # permittivité relative
e2        = 1440.0           # e^2 / (4 pi eps0) en meV.nm
e2_eps    = e2 / epsilon     # meV.nm
a_coulomb = 2.0              # paramètre de lissage (nm)

# ======= Gaps (meV) =======
gap_in  = 1519.0   # GaAs
gap_out = 1919.0   # barrière