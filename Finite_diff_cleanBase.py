# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:03:30 2026

@author: Maël
"""

import numpy as np
from numpy.linalg import inv
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from scipy.linalg import eigh
from numba import njit
import matplotlib.pyplot as plt
import time

#=======Constantes physiques ou de géométrie du problème=============

hbar = 8.729
massExciton = 1
L = 40 #Distance dans la boite
N = 500 #500 points pour le moment
pas = L/(N-1)
numLVL = 5 #Nombre de niveaux d'énergies

m_puit_e = 0.067
m_ext_e = 0.15

m_puit_h = 0.45
m_ext_h = 0.60

epsilon = 12.5
e2_eps = 144.0 / epsilon  
#ICI était l'erreur
a_coulomb = 2.0    # Paramètre de lissage (nm)

# Gap typique GaAs (1519 meV) et Barrière adaptée à tes potentiels (1919 meV)
gap_in = 1519.0
gap_out = 1919.0

F_val = 1.17
geometrie_puits = [
    (-6, 6, -200), # Puits gauche : centre x=-6, largeur 6, prof -200
    (6, 8, -200),   # Puits droit   : centre x=6, largeur 8, prof -200
    #(-14, 4, -200)
]

#-----------Creation du potentiel---------------
def makePotentiel(type2Boite, x, constante):
    """
    Génère un vecteur potentiel V en fonction du type demandé, la taille du potentiel suit le linspace x
    x : vecteur de position
    constante : dictionnaire ou valeur unique selon le besoin
    """

    if type2Boite == "coulomb":
        # constante = [a, e_epsilon]
        a, e_eps = constante
        return -e_eps / np.sqrt(x**2 + a**2)
    
    elif type2Boite == "carre":
        # constante = [largeur, profondeur]
        #On considère le puits comme centré en 0
        W, V0 = constante
        in_the_well = np.abs(x) < W/2
        
        return np.where(in_the_well, V0, 0)
    
    elif type2Boite == "harmonique":
        # constante = k (raideur)
        k = constante
        return 0.5 * k * x**2
    
    elif type2Boite == "stark":
        # constante = F (force du champ)
        F = constante
        return F * x
    elif type2Boite == "multi-carre":
        """
        Penser à implémenter une vérification pour que les puits ne se chevauchent pas si leur centre est trop proche et que leur largeur est trop grande
        """
        # constante = [(centre1,largeur1, profondeur1),(centre2,largeur2, profondeur2),...]
        V = np.zeros_like(x)
        for centre, W, V0 in constante:
            in_the_well = (x > (centre - W/2)) & (x < (centre + W/2))
            V[in_the_well] = V0
        return V
    
    else:
        return np.zeros_like(x)
    
#-----------------Génération de la masse
def makeMasse(type2Boite, x, constante):
    """
    Génère un vecteur de masses effectives qui suit le linspace x
    type2Boite : chaine de charactère qui indique le type de boite et donc la forme que suivra la masse
    x : vecteur de position
    constante : paramètres de masse
    """
    if type2Boite == "carre":
        # constante = [épaisseur, masse à l'intérieur du puits, masse à l'exterieur]
         #On considère le puits comme centré en 0
        W, m_puit, m_ext = constante
        in_the_well = np.abs(x) < W/2
        return np.where(in_the_well, m_puit, m_ext)
    
    elif type2Boite == "multi-carre" :
        # constante = [geometrie, masse à l'intérieur du puits, masse à l'exterieur]
        #geometrie est également une liste qui contien [centre de la boite, épaisseur de la boite, profondeur]
        liste_puits, m_puit, m_ext = constante
        
        est_dans_un_puit = np.zeros_like(x, dtype=bool)
        
        for centre, W, V0 in liste_puits:
            zone_puit = (x > (centre - W/2)) & (x < (centre + W/2))
            est_dans_un_puit = est_dans_un_puit | zone_puit #Le | est un ou logique bit à bit qui permet de comparer entre les éléments du meme indice
            #La variable 'est_dans_un_puit' est maintenant l'intersection de tout les puits (False quand il n'y a pas de puits et True avec un puits)
            
        return np.where(est_dans_un_puit, m_puit, m_ext)
        
    else:
        # Pour Coulomb, Harmonique ou Stark, le matériau est souvent uniforme et j'utilise jamais cette forme
        # constante = m_fixe
        m_fixe = constante
        return np.ones_like(x) * m_fixe


#================Numba version + Sparse Matrix================
@njit 
def makeSparseHamiltonien(V, masses, N, pas, bdd = False) :
    """
    Renvoie des listes qui représentes les diagonales de la matrice du hamiltonien
    V : Le potentiel de base issue du type de boite
    masses : Utilisée pour le hamiltonien de BenDanielDuke, contient un array numpy de la taille de x des masses à chaque points
    pas : Écart entre les points dans le linspace x, explicité ici pour ne pas devoir redémarré le kernel en cas de changement de pas
    bdd : booleen pour préciser le type de hamiltonien
    """
    
    middle = V.copy().astype(np.float64) # La diagonale centrale, égale à V pour le moment
    #Les diagonales inférieures et supérieures 
    bottom = np.zeros((N - 1), dtype=np.float64) 
    top = np.zeros((N -1), dtype=np.float64)
    
    if bdd :
        #Hamiltonien de BenDanielDuke
        for i in range(N-1) :
            
            m_avg = (masses[i] + masses[i+1]) / 2.0
            val_t = - (hbar**2 / (2 * pas**2)) * (1.0 / m_avg)

            middle[i] -= val_t
            middle[i+1] -= val_t
            top[i] = val_t
            bottom[i] = val_t
        middle[N-1] -= val_t
        
    else :
        t = hbar**2 / (2*massExciton*pas**2)
        
        for i in range(N - 1) :
            top[i] = -t
            bottom[i] = -t
            middle[i] = V[i] + 2*t
        middle[N-1] = V[N-1] + 2*t
        
    return top, middle, bottom 

#==========================Calcul du Gap, peut être pour la version Probabiliste=============================
def makeGap(x, constante, gapConstant = True, Varshni = False) :
    """
    Renvoie une valeur de Gap, cette valeur est un array meme quand le gap est constant afin d'éviter les erreures potentielles
    x : vecteur de position pour le gap non constant
    gapConstant : J'ai vu que le Gap pouvait changer en focntion de la température, je vais essayer de simuler ce Gap
    constante : en fonction du type de Gap, elle changeront
    Pour le moment, je considère qu'on estime le Gap uniquement pour des boites carrés ou multi carrés
    Varshni : Booleen qui indique si on estime le Gap à partir de la loi de Varshni
    """
       
    if gapConstant and not Varshni :
        #Si on prend un Gap constant et qu'on ne le calcul pas avec la loi de Varshni, on ressort juste le gap 
        gap = constante
        return np.ones_like(x) * gap

    elif gapConstant and Varshni :
        #Si on veut toujours un gap constant mais cette fois-ci il est calculé avec la loi de Varshni
        eg0, alpha, beta, T = constante
        return np.ones_like(x) * (eg0 - (alpha * T**2 /(T + beta)))
    
    elif not gapConstant and not Varshni :
        #Si on a un Gap non constant et qu'on ne veux pas le calculer avec la loi de Varshni
        liste_puits, gap_in, gap_out = constante
        
        est_dans_un_puit = np.zeros_like(x, dtype=bool)
        
        for centre, W, V0 in liste_puits:
            zone_puit = (x > (centre - W/2)) & (x < (centre + W/2))
            est_dans_un_puit = est_dans_un_puit | zone_puit #Le | est un ou logique bit à bit qui permet de comparer entre les éléments du meme indice
            #La variable 'est_dans_un_puit' est maintenant l'intersection de tout les puits (False quand il n'y a pas de puits et True avec un puits)

        return np.where(est_dans_un_puit, gap_in, gap_out)
        
    else :
        # Quand on ne veux pas un gap constant tout en utilisant la loi de Varshni (on estime que T est constante dans tout le système)
        liste_puits, eg0_in, eg0_out, alpha_in, beta_in, alpha_out, beta_out, T = constante
        
        est_dans_un_puit = np.zeros_like(x, dtype=bool)
        
        for centre, W, V0 in liste_puits:
            zone_puit = (x > (centre - W/2)) & (x < (centre + W/2))
            est_dans_un_puit = est_dans_un_puit | zone_puit #Le | est un ou logique bit à bit qui permet de comparer entre les éléments du meme indice
            #La variable 'est_dans_un_puit' est maintenant l'intersection de tout les puits (False quand il n'y a pas de puits et True avec un puits)
            
        return np.where(est_dans_un_puit, eg0_in - (alpha_in * T**2 /(T + beta_in)), eg0_out - (alpha_out * T**2 /(T + beta_out)))
       
#====================== Méthode de Hartree ================================

def hartree_sparse(
        x, Ve, Vh, m_e, m_h, N, pas, V_coul, lvl_e = 0, lvl_h = 0, 
        maxi = 5000, toler = 1e-4, jacobi = True, phi_e_guess=None,
        phi_h_guess=None, alpha = 1, recouvrement_hart = False):
    """
    Applique la méthode de hartree à un trou et un électron 
    x : linspace sur lequel on discrétise les positions
    Ve : Potentiel créé par la géométrie de la boite de l'éléctron (potentiel de base, on ne considère pas encore les interactions coulombiennes)
    Vh : Potentiel créé par la géométrie de la boite du trou
    N : Nombre de points dans le linspace x, explicité ici pour ne pas devoir redémarré le kernel en cas de changement de pas
    pas : Écart entre les points dans le linspace x, explicité ici pour ne pas devoir redémarré le kernel en cas de changement de pas
    lvl_e : indique le niveau d'énergie de l'electron sur lequel on va travailler 
    lvl_h : indique le niveau d'énergie du trou sur lequel on va travailler 
    maxi : nombre d'itérations maximum qu'on autorise si on a toujours pas convergé
    toler : valeur minimum de différence entre les énergies calculées à chaque itérations, si la différences est inférieure à la tolérance, on considère qu'on a convergé
    jacobi : booleen qui indique l'ordre suivi dans la méthode de hartree
    phi_e_guess : fonction d'onde de l'éléctron, utilisé quand on applique un champs à l'exciton, afin de ne pas repartir de 0
    phi_h_guess : même chose que phi_e_guess mais appliqué au trou
    alpha : constante entre 0 et 1 qui permet de mélanger le potentiel calculé entre l'itération actuelle et l'itération précédente
    recouvrement_hart : booleen qui indique si je veux calculer le recouvrement de l'exciton
    """
    
    step = 0 # Nombre d'itérations qu'on fera
    E_prec = 50000000.0 #J'initialise à cette valeur afin de rentrer dans la condition while 
    E_actuelle = 0.0
    stockage_e = np.zeros((maxi, N))
    stockage_h = np.zeros((maxi, N))
    phi_e = np.zeros(N)
    phi_h = np.zeros(N)
    prev_vh = np.zeros(N) 
    prev_ve = np.zeros(N)
    #On initialise les diagonale de la matrice hamiltonienne, le potentiel est nul comme ça on a uniquement les valeurs pour
    #les termes cinétiques
    top_e, middle_e, bottom_e = makeSparseHamiltonien(prev_ve, masses=m_e, N=N, pas=pas, bdd=True)
    top_h, middle_h, bottom_h = makeSparseHamiltonien(prev_ve, masses=m_h, N=N, pas=pas, bdd=True)

    demarrage_a_froid = True
    
    if phi_e_guess is not None and phi_h_guess is not None:
        #Si on fournit des fonctions d'ondes précédentes, on les mets dans phi_e et phi_h
        phi_e[:] = phi_e_guess
        phi_h[:] = phi_h_guess
        demarrage_a_froid = False

    if demarrage_a_froid:
        #Si les fonctions d'ondes ne sont pas fournies, on les calcule, top_e et bottom_e sont les termes cinétiques donc 
        # ils ne changent pas avec le potentiel, et middle_e reçoit le potentiel appliqué sur l'électron (boite + stark)
        
        He = diags([top_e, middle_e + Ve, bottom_e], [1, 0, -1], format='csr')
        listEnergies_e, wavefunc_e = eigsh(He, k=lvl_e+1, which='SA', tol = toler)
        
        Hh = diags([top_h, middle_h + Vh, bottom_h], [1, 0, -1], format='csr')
        listEnergies_h, wavefunc_h = eigsh(Hh, k=lvl_h+1, which='SA', tol = toler)
        
        norm_fact_e = np.sqrt(np.sum(wavefunc_e[:,lvl_e]**2) * pas)
        phi_e = wavefunc_e[:,lvl_e] / norm_fact_e
        #Quand on prend l'élément [:,lvl_e] cela nous permet de récupérer la fonction d'onde associée à l'énergie 
        #correspondante au niveau 'lvl_e'
        
        norm_fact_h = np.sqrt(np.sum(wavefunc_h[:,lvl_h]**2) * pas)
        phi_h = wavefunc_h[:,lvl_h] / norm_fact_h
    
    #On ajoute les fonctions d'onde au stockage
    stockage_e[0,:] = phi_e
    stockage_h[0,:] = phi_h
    
    if jacobi :
        #Première méthode (on calcul les attractions de coulomb pour le trou et l'électron puis on applique au deux et on recommence)
        while step < maxi-1 and np.abs(E_actuelle - E_prec) > toler :
            #Tant que le nombre d'itération maximum n'est pas atteint ou tant que l'on a pas convergé
            step += 1
            
            E_prec = E_actuelle
            
            if step == 1 :
                #Je vais détailler pourquoi ça marche dans un pdf mais ça marche
                Vh_hartree = V_coul @ phi_e**2 * pas
                Ve_hartree = V_coul @ phi_h**2 * pas 
                #On utilise la fonction d'onde de l'électron pour calculer l'attraction de Coulomb que subit le trou
                #On utilise la fonction d'onde du trou pour calculer l'attraction de Coulomb que subit l'électron
            else :
                #Même chose sauf qu'on dilue le potentiel avec celui calculé précédement, l'objectif et que le saut soit 
                #moins haut, (pour le debuggage je ne l'utilise plus en mettant alpha = 1)
                Vh_hartree = (V_coul @ phi_e**2 * alpha) * pas + prev_vh * (1 - alpha)
                Ve_hartree = (V_coul @ phi_h**2 * alpha) * pas + prev_ve * (1 - alpha)
            
            Ve_temp = Ve + Ve_hartree 
            #Le potentiel appliqué à l'électron est maintenant le potentiel de l'environement (Boite + Stark = Ve) auquel
            # on ajoute l'attraction de Coulomb (Ve_hartree)
        
            He = diags([top_e, middle_e + Ve_temp, bottom_e], [1, 0, -1], format='csr')
            #On crée notre matrice hamiltonienne avec les termes cinétiques auquel on ajoute pour middle_e le potentiel
            
            listEnergies_e, wavefunc_e = eigsh(He, k=lvl_e+3, which='SA', v0 = phi_e, tol = toler)
            #On diagonalise la matrice creuse avec eigsh en donnant comme vecteur de départ la fonction d'onde de la dernière 
            #itération
            
            Vh_temp = Vh + Vh_hartree
            #Même chose pour le trou, on définit son potentiel total
            
            Hh = diags([top_h, middle_h + Vh_temp, bottom_h], [1, 0, -1], format='csr')
            #On crée sa matrice creuse
            
            listEnergies_h, wavefunc_h = eigsh(Hh, k=lvl_h+3, which='SA', v0 = phi_h, tol = toler)
            #On diagonalise pout récupérer les énergies et fonction d'ondes
            
            E_actuelle = listEnergies_e[lvl_e] + listEnergies_h[lvl_h]
            #On somme les deux énergies calculées 
            
            norm_fact_e = np.sqrt(np.sum(wavefunc_e[:,lvl_e]**2) * pas)
            phi_e = wavefunc_e[:,lvl_e] / norm_fact_e
            #On normalise les fonctions d'ondes
            
            norm_fact_h = np.sqrt(np.sum(wavefunc_h[:,lvl_h]**2) * pas)
            phi_h = wavefunc_h[:,lvl_h] / norm_fact_h
            
            stockage_e[step,:] = phi_e
            stockage_h[step,:] = phi_h
            #On ajoute les fonctions d'onde à la liste (utiles pour voir comment les fonctions d'ondes ont évoluées pendant l'algorithme)
            prev_vh = Vh_hartree
            prev_ve = Ve_hartree
            

    else :
        #Seconde méthode, on calcul pour une particule, on applique l'attraction de coulomb puis on calcul pour la suivante
        while step < maxi-1 and np.abs(E_actuelle - E_prec) > toler :
            step += 1
            
            #On commence pour le trou en calculant l'attraction de Coulomb qu'il subit
            E_prec = E_actuelle
            if step == 1 :    
                Vh_hartree = V_coul @ phi_e**2 * pas
            else :
                Vh_hartree = (V_coul @ phi_e**2 * alpha) * pas + prev_vh * (1 - alpha)
            
            Vh_temp = Vh + Vh_hartree
    
            Hh = diags([top_h, middle_h + Vh_temp, bottom_h], [1, 0, -1], format='csr')
            
            listEnergies_h, wavefunc_h = eigsh(Hh, k=lvl_h+1, which='SA', v0 = phi_h, tol = toler)
            
            norm_fact_h = np.sqrt(np.sum(wavefunc_h[:,lvl_h]**2) * pas)
            phi_h = wavefunc_h[:,lvl_h] / norm_fact_h
            
            stockage_h[step,:] = phi_h
            
            if step == 1 :
                Ve_hartree = V_coul @ phi_h**2 * pas
            else :
                Ve_hartree = (V_coul @ phi_h**2 * alpha) * pas + prev_ve * (1 - alpha)
                
            Ve_temp = Ve + Ve_hartree
    
            He = diags([top_e, middle_e + Ve_temp, bottom_e], [1, 0, -1], format='csr')
            
            listEnergies_e, wavefunc_e = eigsh(He, k=lvl_e+1, which='SA', v0 = phi_e, tol = toler)
            
            norm_fact_e = np.sqrt(np.sum(wavefunc_e[:,lvl_e]**2) * pas)
            phi_e = wavefunc_e[:,lvl_e] / norm_fact_e
            
            stockage_e[step,:] = phi_e
            
            E_actuelle = listEnergies_e[lvl_e] + listEnergies_h[lvl_h]
            prev_vh = Vh_hartree
            prev_ve = Ve_hartree

            if step % 500 == 0 :
                print(f"{step//500 * 10} %")
                
    E_exciton = E_actuelle - Ve_hartree @ phi_e**2 * pas
    #On retire à l'exciton 
    
    print(f"Convergence en {step} étape(s) pour le {lvl_e}em niveau d'énergie de e et le {lvl_h}em niveau d'énergie de h")
    
    if recouvrement_hart :
        #Pour calculer le recouvrement mais pas utile pour régler le problème
        recouvrement = np.sum(np.multiply(phi_e, phi_h) * pas) ** 2
        return E_exciton, Ve_temp, Vh_temp, phi_e, phi_h, stockage_e, stockage_h, recouvrement

    return E_exciton, Ve_temp, Vh_temp, phi_e, phi_h, stockage_e, stockage_h, 0


def configInteraction(
        x, Ve, Vh, m_e, m_h, N, pas, V_coul, lvl_e = 0, lvl_h = 0, 
        maxi = 5000, toler = 1e-4, jacobi = True, phi_e_guess=None,
        phi_h_guess=None, alpha = 1, recouvrement_hart = False):
    """
    fuck
    """

    He = diags(makeSparseHamiltonien(Ve, masses=m_e, N=N, pas=pas, bdd=True), [1, 0, -1], format='csr')
    listEnergies_e, wavefunc_e = eigsh(He, k=lvl_e + 1, which='SA', tol = toler)
    
    Hh = diags(makeSparseHamiltonien(Vh, masses=m_h, N=N, pas=pas, bdd=True), [1, 0, -1], format='csr')
    listEnergies_h, wavefunc_h = eigsh(Hh, k=lvl_h + 1, which='SA', tol = toler)
    
    # Normalisation
    for i in range(lvl_e) :
        wavefunc_e[:, i] /= np.sqrt(np.sum(wavefunc_e[:, i]**2) * pas)
    for i in range(lvl_h) :
        wavefunc_h[:, i] /= np.sqrt(np.sum(wavefunc_h[:, i]**2) * pas)
    
    # 3. Calculer les éléments de la matrice Hamiltonienne 4×4
    H_CI = np.zeros((4, 4))
    
    configs = [(0,0), (0,1), (1,0), (1,1)]
    
    for i in range(4):
        n_e_i, n_h_i = configs[i]
        for j in range(4):
            n_e_j, n_h_j = configs[j]
            
            if i == j:
                H_CI[i, j] = listEnergies_e[n_e_i] + listEnergies_h[n_h_i]
            
            # Calculer l'élément d'interaction V_ij
            V_element = 0.0
            
            # Double somme sur les positions discrètes
            for k in range(N):      # position électron
                for l in range(N):  # position trou
                    psi_e_i = wavefunc_e[k, n_e_i]
                    psi_h_i = wavefunc_h[l, n_h_i]
                    psi_e_j = wavefunc_e[k, n_e_j]
                    psi_h_j = wavefunc_h[l, n_h_j]
                    
                    V_element += psi_e_i * psi_h_i * V_coul[k, l] * psi_e_j * psi_h_j * pas * pas
            
            H_CI[i, j] += V_element
            
            print(f"H[{i},{j}] = {H_CI[i, j]:.2f} meV", end=" | ")
        print()
    
    energies_CI, eigenvectors_CI = np.linalg.eigh(H_CI)
    
    print(f"\nÉnergies CI (avec interaction): {energies_CI}")
    print(f"État fondamental CI: {eigenvectors_CI[:, 0]}")
    
    # Calculer l'énergie de liaison excitonique
    E_non_inter = listEnergies_e[0] + listEnergies_h[0]
    E_binding = energies_CI[0] - E_non_inter
    print(f"\nÉnergie de liaison: {E_binding:.2f} meV")
    
    return energies_CI, eigenvectors_CI, wavefunc_e, wavefunc_h
    

#-----------------Main-------------------------------
x = np.linspace(-L/2, L/2, num=N)
m = makeMasse("multi-carre", x, [geometrie_puits, m_puit_e, m_ext_e])

gap = makeGap(x, [gap_in])

dist = np.abs(np.subtract.outer(x, x)) 
V_coul = -e2_eps / np.sqrt(dist**2 + a_coulomb**2)   

V_boite = makePotentiel("multi-carre", x, geometrie_puits)
Vstark  = makePotentiel("stark", x, F_val)

m_e = makeMasse("multi-carre", x, [geometrie_puits, m_puit_e, m_ext_e])

m_h = makeMasse("multi-carre", x, [geometrie_puits, m_puit_h, m_ext_h])

F_vals = np.linspace(-6, 6, 800)

num_niveaux = 2
energies_stockage = np.zeros((len(F_vals), num_niveaux))
gaps = np.zeros(len(F_vals))

energies_E0 = []
energies_E1 = []
energies_E2 = []

energies_E0_e = []
energies_E1_e = []
energies_E2_e = []

energies_E0_h = []
energies_E1_h = []
energies_E2_h = []

overlap = []
phi_e_prec = None
phi_h_prec = None

guess_e_1 = np.zeros(N) 
guess_h_1 = np.zeros(N)

guess_e_2 = np.zeros(N) 
guess_h_2 = np.zeros(N)

premier_tour = True


plt.figure(figsize=(10, 8))
plt.title("Points de resonance de l'exciton")

V_boite = makePotentiel("multi-carre", x, geometrie_puits)

configInteraction(x, V_boite, V_boite, m_e, m_h, N, pas, V_coul, lvl_e=2, lvl_h=2)


t0_n = time.time()
for i, F in enumerate(F_vals):

    V_boite = makePotentiel("multi-carre", x, geometrie_puits)
    Vstark_e  = makePotentiel("stark", x, F)
    Ve_init = V_boite + Vstark_e + gap
    Vstark_h  = makePotentiel("stark", x, -1*F)
    Vh_init = V_boite + Vstark_h
    
        
    top_h, middle_h, bottom_h = makeSparseHamiltonien(Vh_init, masses=m_h, N=N, pas=pas, bdd=True)
    top_e, middle_e, bottom_e = makeSparseHamiltonien(Ve_init, masses=m_e, N=N, pas=pas, bdd=True)
    
    He = diags([top_e, middle_e, bottom_e], [1, 0, -1], format='csr')
    listEnergies_e, wavefunc_e = eigsh(He, k=3, which='SA', tol = 1e-4)
    
    Hh = diags([top_h, middle_h, bottom_h], [1, 0, -1], format='csr')
    listEnergies_h, wavefunc_h = eigsh(Hh, k=3, which='SA', tol = 1e-4)
    
    energies_E0_e.append(listEnergies_e[0])
    energies_E1_e.append(listEnergies_e[1])
    energies_E2_e.append(listEnergies_e[2])
    
    energies_E0_h.append(listEnergies_h[0])
    energies_E1_h.append(listEnergies_h[1])   
    energies_E2_h.append(listEnergies_h[2])

    E_exciton_0, _, _, phi_e_out_1, phi_h_out_1, _, _, recouvrement = hartree_sparse(
            x, Ve_init, Vh_init, m_e, m_h, N, pas, V_coul, 
            lvl_e=0, lvl_h=0, jacobi=True, maxi = 500, recouvrement_hart=True)
    energies_E0.append(E_exciton_0)  
    overlap.append(recouvrement)
    
    E_exciton_1, _, _, phi_e_out_2, phi_h_out_2, _, _, _ = hartree_sparse(
        x, Ve_init, Vh_init, m_e, m_h, N, pas, V_coul,
        lvl_e=0, lvl_h=1, jacobi=True, maxi = 500)
    
    energies_E1.append(E_exciton_1)
    
    E_exciton_2, _, _, phi_e_out_3, phi_h_out_3, _, _, _ = hartree_sparse(
        x, Ve_init, Vh_init, m_e, m_h, N, pas, V_coul,
        lvl_e=1, lvl_h=0, jacobi=True, maxi = 500)
    
    energies_E2.append(E_exciton_2)
    
    guess_e_1 = phi_e_out_1
    guess_h_1 = phi_h_out_1
    
    guess_e_2 = phi_e_out_2
    guess_h_2 = phi_h_out_2
    
    premier_tour = False
t1_n = time.time()
plt.plot(F_vals, energies_E0, label="Fondamental (e0-h0)", color='blue', marker='.')
plt.plot(F_vals, energies_E1, label="1er Excité (e0-h1)", color='red', marker='.')
plt.plot(F_vals, energies_E2, label="2nd Excité (e1-h0)", color='green', marker='.')

plt.xlabel("Champ électrique F")
plt.ylabel("Énergie totale de l'exciton (meV)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 8))
plt.title("Recouvrement de l'Exciton")
plt.plot(F_vals, overlap, label="Recouvrement", color='blue', marker='.')
plt.xlabel("Champ électrique F")
plt.ylabel("Recouvrement de l'exciton")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


print(t1_n - t0_n)

plt.figure(figsize=(10, 8))
plt.title("Points de resonance du trou (hors attraction de Coulomb)")
plt.plot(F_vals, energies_E0_h, label="Fondamental (h0)", color='blue', marker='.')
plt.plot(F_vals, energies_E1_h, label="1er Excité (h1)", color='red', marker='.')
plt.plot(F_vals, energies_E2_h, label="2nd Excité (h2)", color='green', marker='.')
plt.xlabel("Champ électrique F")
plt.ylabel("Énergie totale du trou (meV)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


plt.figure(figsize=(10, 8))
plt.title("Points de resonance de l'électron (hors attraction de Coulomb)")
plt.plot(F_vals, energies_E0_e, label="Fondamental (e0)", color='blue', marker='.')
plt.plot(F_vals, energies_E1_e, label="1er Excité (e1)", color='red', marker='.')
plt.plot(F_vals, energies_E2_e, label="2nd Excité (e2)", color='green', marker='.')
plt.xlabel("Champ électrique F")
plt.ylabel("Énergie totale de l'électron (meV)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()











