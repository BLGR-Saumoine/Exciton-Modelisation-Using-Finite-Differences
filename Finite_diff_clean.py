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
N = 1000 #500 points pour le moment
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

    Parameters
    ----------
    type2Boite : string
        Le type de potentiel à générer ("coulomb", "carre", "harmonique", "stark", "multi-carre").
        
    x : numpy.ndarray
        vecteur de position.
        
    constante : list or float or tuple
        dictionnaire ou valeur unique selon le besoin (paramètres du potentiel).

    Returns
    -------
    V : numpy.ndarray
        Le vecteur potentiel calculé sur la grille x.
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

    Parameters
    ----------
    type2Boite : str
        chaine de charactère qui indique le type de boite et donc la forme que suivra la masse.
        
    x : numpy.ndarray
        vecteur de position.
        
    constante : list or float
        paramètres de masse.

    Returns
    -------
    masse : numpy.ndarray
        Le vecteur des masses effectives calculé sur la grille x.
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
    Renvoie des listes qui représentes les diagonales de la matrice du hamiltonien, utilisation de numba pour accélérer le calcul

    Parameters
    ----------
    V : numpy.ndarray
        Le potentiel de base issue du type de boite.
        
    masses : numpy.ndarray
        Utilisée pour le hamiltonien de BenDanielDuke, contient un array numpy de la taille de x des masses à chaque points.
        
    N : int
        Nombre de points dans le linspace x.
        
    pas : float
        Écart entre les points dans le linspace x, explicité ici pour ne pas devoir redémarré le kernel en cas de changement de pas.
        
    bdd : bool, optional
        booleen pour préciser le type de hamiltonien. The default is False.

    Returns
    -------
    top : numpy.ndarray
        Diagonale supérieure de la matrice hamiltonienne.
    middle : numpy.ndarray
        Diagonale principale de la matrice hamiltonienne.
    bottom : numpy.ndarray
        Diagonale inférieure de la matrice hamiltonienne.
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

    Parameters
    ----------
    x : numpy.ndarray
        vecteur de position pour le gap non constant.
        
    constante : list or float
        en fonction du type de Gap, elle changeront. Pour le moment, je considère qu'on estime le Gap uniquement pour des boites carrés ou multi carrés.
    
    gapConstant : bool, optional
        J'ai vu que le Gap pouvait changer en focntion de la température, je vais essayer de simuler ce Gap. The default is True.
    
    Varshni : bool, optional
        Booleen qui indique si on estime le Gap à partir de la loi de Varshni. The default is False.

    Returns
    -------
    gap : numpy.ndarray
        Le vecteur représentant la valeur du Gap sur la grille x.
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

    Parameters
    ----------
    x : numpy.ndarray
        linspace sur lequel on discrétise les positions.
    Ve : numpy.ndarray
        Potentiel créé par la géométrie de la boite de l'éléctron (potentiel de base, on ne considère pas encore les interactions coulombiennes).
    Vh : numpy.ndarray
        Potentiel créé par la géométrie de la boite du trou.
    m_e : numpy.ndarray
        Masse effective de l'électron sur la grille.
    m_h : numpy.ndarray
        Masse effective du trou sur la grille.
    N : int
        Nombre de points dans le linspace x, explicité ici pour ne pas devoir redémarré le kernel en cas de changement de pas.
    pas : float
        Écart entre les points dans le linspace x, explicité ici pour ne pas devoir redémarré le kernel en cas de changement de pas.
    V_coul : numpy.ndarray
        Matrice d'interaction coulombienne.
    lvl_e : int, optional
        indique le niveau d'énergie de l'electron sur lequel on va travailler. The default is 0.
    lvl_h : int, optional
        indique le niveau d'énergie du trou sur lequel on va travailler. The default is 0.
    maxi : int, optional
        nombre d'itérations maximum qu'on autorise si on a toujours pas convergé. The default is 5000.
    toler : float, optional
        valeur minimum de différence entre les énergies calculées à chaque itérations, si la différences est inférieure à la tolérance, on considère qu'on a convergé. The default is 1e-4.
    jacobi : bool, optional
        booleen qui indique l'ordre suivi dans la méthode de hartree. The default is True.
    phi_e_guess : numpy.ndarray, optional
        fonction d'onde de l'éléctron, utilisé quand on applique un champs à l'exciton, afin de ne pas repartir de 0. The default is None.
    phi_h_guess : numpy.ndarray, optional
        même chose que phi_e_guess mais appliqué au trou. The default is None.
    alpha : float, optional
        constante entre 0 et 1 qui permet de mélanger le potentiel calculé entre l'itération actuelle et l'itération précédente. The default is 1.
    recouvrement_hart : bool, optional
        booleen qui indique si je veux calculer le recouvrement de l'exciton. The default is False.

    Returns
    -------
    E_exciton : float
        Énergie totale de l'exciton après convergence.
    Ve_temp : numpy.ndarray
        Potentiel final ressenti par l'électron.
    Vh_temp : numpy.ndarray
        Potentiel final ressenti par le trou.
    phi_e : numpy.ndarray
        Fonction d'onde finale de l'électron.
    phi_h : numpy.ndarray
        Fonction d'onde finale du trou.
    stockage_e : numpy.ndarray
        Historique des fonctions d'onde de l'électron au cours des itérations.
    stockage_h : numpy.ndarray
        Historique des fonctions d'onde du trou au cours des itérations.
    recouvrement : float
        Valeur du recouvrement si demandé, sinon 0.
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
        x, Ve, Vh, m_e, m_h, N, Ne, Nh, pas, V_coul, lvl_e = 0, lvl_h = 0, 
        toler = 1e-4, lvl_exciton= 0, matriciel = True):

    He = diags(makeSparseHamiltonien(Ve, masses=m_e, N=N, pas=pas, bdd=True), [1, 0, -1], format='csr')
    listEnergies_e, wavefunc_e = eigsh(He, k=lvl_e + 1, which='SA', tol = toler)
    
    Hh = diags(makeSparseHamiltonien(Vh, masses=m_h, N=N, pas=pas, bdd=True), [1, 0, -1], format='csr')
    listEnergies_h, wavefunc_h = eigsh(Hh, k=lvl_h + 1, which='SA', tol = toler)
    
    # Normalisation
    for i in range(lvl_e) :
        wavefunc_e[:, i] /= np.sqrt(np.sum(wavefunc_e[:, i]**2) * pas)
    for i in range(lvl_h) :
        wavefunc_h[:, i] /= np.sqrt(np.sum(wavefunc_h[:, i]**2) * pas)
    
    
    H_CI = np.zeros((lvl_e * lvl_h, lvl_e * lvl_h))
    
    configs = []
    
    for i in range(lvl_e):
        for j in range(lvl_h):
            configs.append((i,j))
    # [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
    if matriciel :
        for i in range(len(configs)):
            n_e_i, n_h_i = configs[i]
            for j in range(len(configs)):
                n_e_j, n_h_j = configs[j]
                
                if i == j:
                    H_CI[i, j] = listEnergies_e[n_e_i] + listEnergies_h[n_h_i]
                
                V_element = ((wavefunc_e[:, n_e_i] * wavefunc_e[:, n_e_j]) @ V_coul) @ (wavefunc_h[:, n_h_j] * wavefunc_h[:, n_h_i]) * pas**2
                
                H_CI[i, j] += V_element
                
    else :
        for i in range(len(configs)):
            n_e_i, n_h_i = configs[i]
            for j in range(len(configs)):
                n_e_j, n_h_j = configs[j]
                
                if i == j:
                    H_CI[i, j] = listEnergies_e[n_e_i] + listEnergies_h[n_h_i]
        
                V_element = 0.0
                
                # Double somme sur les positions discrètes (je vais faire le matriciel sous peu)
                for k in range(N):      # position électron
                    for l in range(N):  # position trou
                        psi_e_i = wavefunc_e[k, n_e_i]
                        psi_h_i = wavefunc_h[l, n_h_i]
                        psi_e_j = wavefunc_e[k, n_e_j]
                        psi_h_j = wavefunc_h[l, n_h_j]
                        
                        V_element += psi_e_i * psi_h_i * V_coul[k, l] * psi_e_j * psi_h_j * pas * pas
                
                H_CI[i, j] += V_element
                
    
    energies_CI, eigenvectors_CI = eigh(H_CI)
    
    C_coeffs = eigenvectors_CI[:, lvl_exciton] 
    
    Psi_exciton_2D = np.zeros((N, N))
    
    for k in range(len(configs)):
        n_e, n_h = configs[k]
        coef = C_coeffs[k]

        Psi_exciton_2D += coef * np.outer(wavefunc_e[:, n_e], wavefunc_h[:, n_h])
    
    return energies_CI, eigenvectors_CI, wavefunc_e, wavefunc_h, Psi_exciton_2D


def configInteractionTrionPlus(
        x, Ve, Vh, m_e, m_h, N, pas, V_coul, lvl_e1 = 4, lvl_e2 = 4, lvl_h = 4, 
        toler = 1e-4, lvl_exciton= 0, wavefunc = True):


    He1 = diags(makeSparseHamiltonien(Ve, masses=m_e, N=N, pas=pas, bdd=True), [1, 0, -1], format='csr')
    listEnergies_e1, wavefunc_e1 = eigsh(He1, k=lvl_e1 + 1, which='SA', tol = toler)
    
    He2 = diags(makeSparseHamiltonien(Ve, masses=m_e, N=N, pas=pas, bdd=True), [1, 0, -1], format='csr')
    listEnergies_e2, wavefunc_e2 = eigsh(He2, k=lvl_e2 + 1, which='SA', tol = toler)
    
    Hh = diags(makeSparseHamiltonien(Vh, masses=m_h, N=N, pas=pas, bdd=True), [1, 0, -1], format='csr')
    listEnergies_h, wavefunc_h = eigsh(Hh, k=lvl_h + 1, which='SA', tol = toler)
    
    # Normalisation
    for i in range(lvl_e1) :
        wavefunc_e1[:, i] /= np.sqrt(np.sum(wavefunc_e1[:, i]**2) * pas)
    for i in range(lvl_e2) :
        wavefunc_e2[:, i] /= np.sqrt(np.sum(wavefunc_e2[:, i]**2) * pas)
    for i in range(lvl_h) :
        wavefunc_h[:, i] /= np.sqrt(np.sum(wavefunc_h[:, i]**2) * pas)
    
    
    H_CI = np.zeros((lvl_e1 * lvl_e2 * lvl_h, lvl_e1 * lvl_e2 * lvl_h))
    
    configs = []
    
    for i in range(lvl_e1):
        for j in range(lvl_e2):
            for k in range(lvl_h):
                configs.append((i,j,k))
            
    # Ici vu qu'on a 3 particules ça ressemble plus à avec e1 e2 h
    #[(0,0,0), (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,1,2), (0,2,0), (0,2,1), (0,2,2), (1,0,0) etc...]
    

    
    for m in range(len(configs)):
        i, j, k = configs[m]
        for n in range(len(configs)):
            ip, jp, kp = configs[n]
            
            V_element = 0
            
            if i == ip and j == jp and k ==kp :
                H_CI[m, n] = listEnergies_e1[i] + listEnergies_e2[j] + listEnergies_h[k]
            
            if i == ip :
                V_element += ((wavefunc_e2[:, j] * wavefunc_e2[:, jp]) @ V_coul) @ (wavefunc_h[:, k] * wavefunc_h[:, kp]) * pas**2
                
            if j == jp :
                V_element += ((wavefunc_e1[:, i] * wavefunc_e1[:, ip]) @ V_coul) @ (wavefunc_h[:, k] * wavefunc_h[:, kp]) * pas**2
                
            if k == kp :
                V_element += ((wavefunc_e1[:, i] * wavefunc_e1[:, ip]) @ (-1 * V_coul)) @ (wavefunc_e2[:, j] * wavefunc_e2[:, jp]) * pas**2
            
            
            H_CI[m, n] += V_element
                
    
    energies_CI, eigenvectors_CI = eigh(H_CI)
    
    C_coeffs = eigenvectors_CI[:, lvl_exciton] 
    
    Psi_trion_3D = np.zeros((N, N, N))
    if wavefunc :
        for m in range(len(configs)):
            i,j,k = configs[m]
            coef = C_coeffs[m]
            
            Psi_trion_3D += coef * np.multiply.outer(np.multiply.outer(wavefunc_e1[:, i], wavefunc_e2[:, j]), wavefunc_h[:, k])

    return energies_CI, eigenvectors_CI, wavefunc_e1, wavefunc_e2, wavefunc_h, Psi_trion_3D

    
#----------------------------------------------------------------------------------
#Je sais pas comment l'appeler mais c'est ce qui va me serir à calculer l'expression de mes résusltats sur la base des états propres
#avant l'attraction de Coulomb

def diagonalisationBase(
        x, Ve, Vh, m_e, m_h, N, PsiE, PsiH, lvlParticle = 0,
        toler = 1e-4):
    """
    Parameters
    ----------
    x : numpy.ndarray
        linspace sur lequel on discrétise les positions.
    Ve : numpy.ndarray
        Potentiel créé par la géométrie de la boite de l'éléctron (potentiel de base, on ne considère pas encore les interactions coulombiennes).
    Vh : numpy.ndarray
        Potentiel créé par la géométrie de la boite du trou (même chose que pour Ve).
    m_e : numpy.ndarray
        Masse effective de l'électron.
    m_h : numpy.ndarray
        Masse effective du trou.
    N : int
        Nombre de points dans le linspace x.
    PsiE : numpy.ndarray
        Fonction d'onde de l'electron calculé avant d'ont on va calculer le produit scalaire avec la fonction d'onde "brute".
    PsiH : numpy.ndarray
        Fonction d'onde du trou calculé avant d'ont on va calculer le produit scalaire avec la fonction d'onde "brute".
    lvlParticle : int, optional
        DESCRIPTION. The default is 0.
    toler : float, optional
        DESCRIPTION. The default is 1e-4.

    Returns
    -------
    overlap_e : float
        valeur du recouvrement de PsiE et la fonction d'onde de l'electron hors attraction de coulomb.
    overlap_h : float
        valeur du recouvrement de PsiH et la fonction d'onde du trou hors attraction de coulomb.
    """
    
    He = diags(makeSparseHamiltonien(Ve, masses=m_e, N=N, pas=pas, bdd=True), [1, 0, -1], format='csr')
    listEnergies_e, wavefunc_e = eigsh(He, k=lvlParticle + 1, which='SA', tol = toler)
    
    Hh = diags(makeSparseHamiltonien(Vh, masses=m_h, N=N, pas=pas, bdd=True), [1, 0, -1], format='csr')
    listEnergies_h, wavefunc_h = eigsh(Hh, k=lvlParticle + 1, which='SA', tol = toler)
    
    overlap_e = wavefunc_e[:,lvlParticle] @ PsiE
    overlap_h = wavefunc_h[:,lvlParticle] @ PsiH
    
    return overlap_e, overlap_h

#------------------------------------------------------------------------


def applyField(x, m, lowPotential, highPotential, discretePotential, Ve, Vh, m_e, m_h, energyLVL) :
    """
    Simule l'effet d'un champ électrique variable (Effet Stark) sur l'exciton en comparant les méthodes Hartree et CI.

    Parameters
    ----------
    x : numpy.ndarray
        Linspace des positions sur lequel le système est défini.
    m : numpy.ndarray
        (Argument non utilisé actuellement dans la logique, hérité de la structure précédente).
    lowPotential : float
        Valeur minimale du champ électrique à appliquer (en kV/cm ou unité du système).
    highPotential : float
        Valeur maximale du champ électrique à appliquer.
    discretePotential : int
        Nombre de points de simulation pour le champ électrique (résolution du balayage).
    Ve : numpy.ndarray
        Potentiel de structure de l'électron (Puits quantiques sans champ).
    Vh : numpy.ndarray
        Potentiel de structure du trou.
    m_e : numpy.ndarray
        Carte des masses effectives de l'électron sur la grille x.
    m_h : numpy.ndarray
        Carte des masses effectives du trou sur la grille x.
    energyLVL : int
        Nombre de niveaux d'énergie excitoniques à calculer, suivre et stocker (ex: 3 pour fondamental + 2 excités).

    Returns
    -------
    F_vals : numpy.ndarray
        Le vecteur des valeurs de champ électrique calculées (axe X des graphes).
    energiesHart : numpy.ndarray
        Matrice (energyLVL, discretePotential) contenant les énergies totales triées calculées par Hartree.
    energiesCI : numpy.ndarray
        Matrice (energyLVL, discretePotential) contenant les énergies totales calculées par Interaction de Configuration.
    overlapHart : numpy.ndarray
        Matrice des recouvrements spatiaux (|integrale(psi_e * psi_h)|^2) associés aux états Hartree triés.
    overlapCI : numpy.ndarray
        Matrice des projections (overlap) entre l'état CI n°j et le produit des états libres n°j.
    """
    
    energiesHart = np.zeros((energyLVL, discretePotential))
    energiesCI = np.zeros((energyLVL, discretePotential))

    dist = np.abs(np.subtract.outer(x, x)) 
    V_coul = -e2_eps / np.sqrt(dist**2 + a_coulomb**2)   
    #On initialise le potentiel de Coulomb en chaque point, on n'a pas encore appliqué la fonction d'onde 
    
    overlapCI = np.zeros((energyLVL, discretePotential))
    overlapHart = np.zeros((energyLVL, discretePotential))
    
    F_vals = np.linspace(lowPotential, highPotential, discretePotential)
    gap = makeGap(x, [gap_in])
    
    for i, F in enumerate(F_vals):
        Vstark_e  = makePotentiel("stark", x, F)
        Vstark_h  = makePotentiel("stark", x, -1*F)
        
        Ve_total = Ve + Vstark_e + gap
        Vh_total = Vh + Vstark_h
        #On recrée les potentiels du trou et de l'électron avec le nouveau champs électrique
        
        E_CI_raw, Vec_CI, base_e, base_h, _ = configInteraction(
            x, Ve_total, Vh_total, m_e, m_h, N, pas, V_coul, 
            lvl_e=energyLVL + 3, lvl_h=energyLVL + 3, lvl_exciton=0
        )
        
        energiesCI[:, i] = E_CI_raw[:energyLVL]
        
        for j in range(energyLVL):

            linear_index = j * (energyLVL + 3) + j 
            
            coeff = Vec_CI[linear_index, j]
            overlapCI[j, i] = coeff**2
            
        target_configs = [(0, 0), (0, 1), (1, 0)]
        last_phi_e = [None] * energyLVL
        last_phi_h = [None] * energyLVL
        for j, (le, lh) in enumerate(target_configs):
            if j >= energyLVL: break

            # Si c'est le premier point (i==0), on utilise la base libre
            # Sinon, on utilise la fonction d'onde du point précédent (F-1)
            guess_e = last_phi_e[j] if i > 0 else base_e[:, le]
            guess_h = last_phi_h[j] if i > 0 else base_h[:, lh]

            E_hart, _, _, phi_e_hart, phi_h_hart, _, _, _ = hartree_sparse(
                x, Ve_total, Vh_total, m_e, m_h, N, pas, V_coul,
                lvl_e=le, lvl_h=lh, 
                phi_e_guess=guess_e, # Suivi d'état activé ici
                phi_h_guess=guess_h,
                toler=1e-4
            )

            # Mise à jour de la mémoire pour le prochain champ F
            energiesHart[j, i] = E_hart
            last_phi_e[j] = phi_e_hart
            last_phi_h[j] = phi_h_hart
            
            # Recouvrement (projection sur la base libre pour voir l'évolution)
            ov_e = np.sum(phi_e_hart * base_e[:, le]) * pas
            ov_h = np.sum(phi_h_hart * base_h[:, lh]) * pas
            overlapHart[j, i] = (ov_e * ov_h)**2
            
    return F_vals, energiesHart, energiesCI, overlapHart, overlapCI


def generalApplyField(x, lowPotential, highPotential, discretePotential, Ve, Vh, m_e, m_h, energyLVL, ) :
    
    energiesCI = np.zeros((energyLVL, discretePotential))

    dist = np.abs(np.subtract.outer(x, x)) 
    V_coul = -e2_eps / np.sqrt(dist**2 + a_coulomb**2)   
    #On initialise le potentiel de Coulomb en chaque point, on n'a pas encore appliqué la fonction d'onde 
    
    
    F_vals = np.linspace(lowPotential, highPotential, discretePotential)
    gap = makeGap(x, [gap_in])
    
    for i, F in enumerate(F_vals):
        Vstark_e  = makePotentiel("stark", x, F)
        Vstark_h  = makePotentiel("stark", x, -1*F)
        
        Ve_total = Ve + Vstark_e + gap
        Vh_total = Vh + Vstark_h
        #On recrée les potentiels du trou et de l'électron avec le nouveau champs électrique
        
        E_CI, eigenvectors_CI, wavefunc_e1, wavefunc_e2, wavefunc_h, Psi_exciton_2D = configInteractionTrionPlus(
            x, Ve_total, Vh_total, m_e, m_h, N, pas, V_coul, wavefunc = False)

        energiesCI[:, i] = E_CI[:energyLVL]

    
    return energiesCI, F_vals



#-----------------Main-------------------------------

x = np.linspace(-L/2, L/2, num=N)
m = makeMasse("multi-carre", x, [geometrie_puits, m_puit_e, m_ext_e])


V_boite = makePotentiel("multi-carre", x, geometrie_puits)

m_e = makeMasse("multi-carre", x, [geometrie_puits, m_puit_e, m_ext_e])

m_h = makeMasse("multi-carre", x, [geometrie_puits, m_puit_h, m_ext_h])

dist = np.abs(np.subtract.outer(x, x)) 
V_coul = -e2_eps / np.sqrt(dist**2 + a_coulomb**2)   


"""
t0_n = time.time()

F_vals, energiesHart, energiesCI, overlapHart, overlapCI = applyField(
    x=x, m=None, 
    lowPotential=-6, highPotential=6, discretePotential=800, 
    Ve=V_boite, Vh=V_boite, m_e=m_e, m_h=m_h, energyLVL=3)

t1_n = time.time()
print(f"Temps écoulé {t1_n - t0_n}")


plt.figure(figsize=(10, 8))
plt.title("Resonnance avec CI")
for i in range(3):
    plt.plot(F_vals, energiesCI[i,:] , marker='.')

plt.xlabel("Champ électrique F")
plt.ylabel("Énergie totale de l'exciton (meV)")
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(10, 8))
plt.title("Resonnance avec Hartree")
for i in range(3):
    plt.plot(F_vals, energiesHart[i,:] , marker='.')

plt.xlabel("Champ électrique F")
plt.ylabel("Énergie totale de l'exciton (meV)")
plt.grid(True, alpha=0.3)
plt.show()


plt.figure(figsize=(10, 8))
plt.title("Overlap pour les Deux méthodes")
for i in range(3):
    plt.plot(F_vals, overlapHart[i, :], '-', label=f'Recouv e-h Hartree {i}')
    plt.plot(F_vals, overlapCI[i, :], '--', label=f'Projection CI {i}')

plt.xlabel("Champ électrique F")
plt.ylabel("Recouvrement de l'exciton")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""

Vstark_e  = makePotentiel("stark", x, 0)
Vstark_h  = makePotentiel("stark", x, 0)
gap = makeGap(x, [gap_in])

Ve = V_boite + Vstark_e + gap
Vh = V_boite + Vstark_h


t0_n = time.time()

energiesCI, F_vals = generalApplyField(x, 0, 6, 1200, Ve, Vh, m_e, m_h, energyLVL = 7)

t1_n = time.time()

print(t1_n - t0_n)

plt.figure(figsize=(10, 8))
plt.title("Resonnance avec CI")
for i in range(5):
    plt.plot(F_vals, energiesCI[i,:] , marker='.')

plt.xlabel("Champ électrique F")
plt.ylabel("Énergie totale de l'exciton (meV)")
plt.grid(True, alpha=0.3)
plt.show()

"""
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

"""







