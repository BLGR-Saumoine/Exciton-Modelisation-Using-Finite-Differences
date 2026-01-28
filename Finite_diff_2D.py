# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 17:57:40 2026

@author: Maël
Notes perso pour le github :

Sauvegarder le fichier

git status

git add .

git commit -m "Description du commit"

git push

"""

import numpy as np
from numpy.linalg import inv
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from scipy.linalg import eigh
from numba import njit
import matplotlib.pyplot as plt
from matplotlib import cm
import time

hbar = 8.729
massExciton = 1
L_x = 40.0 #Dix unités
L_y = 40.0
N_x = 500 #500 points pour le moment
N_y = 500
pas_x = L_x/(N_x-1)
pas_y = L_y/(N_y-1)
numLVL = 6 #Nombre de niveaux d'énergies

m_puit_e = 0.067
m_ext_e = 0.15

m_puit_h = 0.45
m_ext_h = 0.60

epsilon = 12.5
e2_eps = 1440.0 / epsilon  
a_coulomb = 2.0    # Paramètre de lissage (nm)

F_val = 1.17
geometrie_puits = [
    (-6, -6, 6, 6, -200), # Puits gauche : centre x=-6, longueur = 6,largeur 6, prof -200
    (6, 6, 8, 8, -200),   # Puits droit : centre x=6, longueur = 8, largeur 8, prof -200
    (-14, 14, 10, 4, -200)
]

toler = 1e-4
#---------------Faiseur de Potentiel-----------------

def makePotentiel2D(type2Boite, x, y, constante):
    """
    Génère un vecteur potentiel V en fonction du type demandé.
    x : vecteur de position
    y : second vecteur de position
    constante : dictionnaire ou valeur unique selon le besoin
    """
    #print("Potentiel de la forme : " + type2Boite)
    #print("\n")
    X, Y = np.meshgrid(x, y)
    if type2Boite == "coulomb":
        # constante = [a, e_epsilon]
        a, e_eps = constante
        return -e_eps / np.sqrt(X**2 + Y**2 + a**2)
    
    elif type2Boite == "carre":
        # constante = [largeur, longueur, profondeur]
        
        W, L, V0 = constante
        in_the_well = (np.abs(X) < W/2) & (np.abs(Y) < L/2)
        
        return np.where(in_the_well, V0, 0)
    
    elif type2Boite == "harmonique":
        # constante = k (raideur)
        k = constante
        return 0.5 * k * X**2 * Y**2
    
    elif type2Boite == "harmonique2D" :
        # constante = k (raideur)
        k = constante
        return 0.5 * k * (X**2 + Y**2)
    
    elif type2Boite == "stark":
        # constante = F (force du champ)
        F = constante
        return F * X * Y
    
    elif type2Boite == "stark2D":
        # constante = Fx (force du champ en x) et Fy (force du champs en y)
        Fx, Fy = constante
        return Fx * X + Fy * Y
    
    elif type2Boite == "multi-carre":
        """
        Penser à implémenter une vérification pour que les puits ne se chevauchent pas si leur centre est trop proche et que leur largeur est trop grande
        """
        # constante = [(centre1_x, centre1_y, largeur1, longueur1, profondeur1),(centre2_x, centre2_y, largeur2, longueur2, profondeur2),...]
        V = np.zeros_like(X)
        for centre_x, centre_y, W, L, V0 in constante:
            in_the_well = (X > (centre_x - W/2)) & (X < (centre_x + W/2)) & (Y < (centre_y + L/2)) & (Y > (centre_y - L/2)) 
            V[in_the_well] = V0
        return V
    
    else:
        return np.zeros_like(X)

#================ Hamiltonien numba + sparse matrix 2D====================
@njit 
def makeSparseHamiltonien2D(V, masses, N_x, N_y, pasX, pasY, bdd = False) :
    
    N_tot = N_x * N_y
    middle = V.flatten().astype(np.float64)
    bottom = np.zeros((N_tot - N_x), dtype=np.float64)
    top = np.zeros((N_tot - N_x), dtype=np.float64)
    
    left = np.zeros((N_tot - 1), dtype=np.float64)
    right = np.zeros((N_tot - 1), dtype=np.float64)
    """
    if bdd :
        for i in range(N-1) :
            
            m_avg = (masses[i] + masses[i+1]) / 2.0
            val_t = - (hbar**2 / (2 * pas**2)) * (1.0 / m_avg)

            middle[i] -= val_t
            middle[i+1] -= val_t
            top[i] = val_t
            bottom[i] = val_t
        middle[N-1] -= val_t
        """
    if False :
        print("temp")
        
    else :
        
        tx = -hbar**2 / (2*massExciton*pasX**2)
        ty = -hbar**2 / (2*massExciton*pasY**2)
        
        for i in range(N_tot - N_x) :
            
            if (i+1) % N_x == 0 :
                left[i] = 0.0
                right[i] = 0.0
            else : 
                left[i] = tx
                right[i] = tx
            
            middle[i] += -2*tx -2*ty
            
            top[i] = ty
            bottom[i] = ty
        for j in range(N_tot - N_x, N_x*N_y - 1) :
            
            if (j+1) % N_x == 0 :
                left[j] = 0.0
                right[j] = 0.0
            else : 
                left[j] = tx
                right[j] = tx
                
            middle[j] += -2*tx -2*ty

        middle[N_x*N_y - 1] += -2*tx -2*ty
        
    return bottom, left, middle, right, top


#----------------------Plot main---------------------

x = np.linspace(-L_x/2, L_x/2, num=N_x)
y = np.linspace(-L_y/2, L_y/2, num=N_y)
X, Y = np.meshgrid(x, y)


V_2D = makePotentiel2D("multi-carre", x, y, geometrie_puits)


plt.figure(figsize=(10, 8))

c = plt.pcolormesh(X, Y, V_2D, cmap='viridis', shading='auto')

plt.title("Visualisation du Puits 2D")
plt.xlabel("Position x (nm)")
plt.ylabel("Position y (nm)")
plt.axis('equal')
plt.show()

bottom, left, middle, right, top = makeSparseHamiltonien2D(V_2D, 0, N_x, N_y, pas_x, pas_y)

diag = [bottom, left, middle, right, top]
offsets = [-N_x, -1, 0, 1, N_x]

H_2D = diags(diag, offsets, format='csr')

listEnergies, wavefunc = eigsh(H_2D, k=numLVL, which='SA', tol = toler)

fig, axes = plt.subplots(1, numLVL, figsize=(4 * numLVL, 4))
if numLVL == 1: axes = [axes] 

x_plot = np.linspace(-L_x/2, L_x/2, N_x)
y_plot = np.linspace(-L_y/2, L_y/2, N_y)
X_plot, Y_plot = np.meshgrid(x_plot, y_plot) 

for i in range(numLVL):

    psi_1D = wavefunc[:, i]

    psi_2D = psi_1D.reshape((N_y, N_x))
    
    density = np.abs(psi_2D)**2
    
    ax = axes[i]
    im = ax.pcolormesh(X_plot, Y_plot, density, cmap='inferno', shading='auto')
    
    ax.set_title(f"État #{i}\nE = {listEnergies[i]:.2f} meV")
    ax.set_xlabel("x (nm)")
    if i == 0:
        ax.set_ylabel("y (nm)")
    else:
        ax.set_yticks([]) 
    
    ax.set_aspect('equal')


lvl_3D = 2

fig.colorbar(im, ax=axes.ravel().tolist(), label="Densité de probabilité |Psi|^2")

plt.suptitle(f"Fonctions d'onde dans le potentiel {geometrie_puits}", y=1.05)
plt.show()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

psi_1D = wavefunc[:, lvl_3D]

psi_2D = np.abs(psi_1D.reshape((N_y, N_x)))**2

surf = ax.plot_surface(X, Y, psi_2D, linewidth=0, cmap = cm.viridis, antialiased=False)



# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()



