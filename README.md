# Quantum Exciton Simulation: Finite Difference Methods

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) 
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-green) 
![SciPy](https://img.shields.io/badge/SciPy-1.10%2B-red) 
![Numba](https://img.shields.io/badge/Numba-JIT-orange)

Researsh project by **Gendronneau Maël**.

## Project Overview

Ce projet porte sur la résolution numérique de l'équation de Schrödinger pour l'étude des **excitons** (paires électron-trou) dans des nanostructures semi-conductrices. L'objectif est de modéliser le comportement de ces particules sous l'influence de potentiels complexes et de champs électriques externes via des méthodes de **différences finies**.

## Repository Structure

* **`Finite_diff_2D.py`** : Solveur 2D utilisant des matrices creuses et Numba pour calculer les fonctions d'onde et les énergies dans des potentiels à géométries variables (coulombien, harmonique, multi-carre).
* **`Finite_diff_clean.py`** : Script optimisé implémentant la méthode de Hartree (champ auto-cohérent) et l'Interaction de Configuration (CI) pour une analyse 1D approfondie de l'interaction électron-trou.
* **`Finite_diff_cleanBase.py`** : Version de référence pour les calculs de résonance, l'effet Stark et la comparaison des niveaux d'énergie fondamentaux et excités.

## Key Components Implemented

### Quantum Solvers
* **1D & 2D Finite Difference Solvers**: Discrétisation de l'Hamiltonien pour extraire les états propres via l'algorithme d'Arnoldi (`eigsh`).
* **BenDaniel-Duke Hamiltonian**: Prise en compte de la masse effective dépendante de la position aux interfaces des puits.
* **Sparse Matrix Optimization**: Utilisation de formats CSR (`scipy.sparse`) pour une gestion efficace de la mémoire et de la vitesse de calcul.

### Interaction Models
* **Méthode de Hartree**: Calcul itératif du potentiel de Coulomb pour obtenir une solution auto-cohérente des fonctions d'onde.
* **Configuration Interaction (CI)** : Diagonalisation de l'Hamiltonien sur une base d'états libres pour capturer les corrélations exactes.
* **Potentiels**: Support pour les potentiels de Coulomb lissés, puits carrés multiples, potentiels harmoniques et effet Stark.

## Methodology

* **Système Physique**: Paramètres basés sur le GaAs/AlGaAs (masses effectives, gap de ~1519 meV, constante diélectrique).
* **Optimisation**: Utilisation de `njit` de Numba pour accélérer la construction des matrices Hamiltoniennes.
* **Étude de l'Effet Stark**: Balayage du champ électrique (`applyField`) pour observer le décalage d'énergie et la modification du recouvrement spatial des porteurs.

## Results & Insights

* Visualisation de la **densité de probabilité** $|\Psi|^2$ en 2D et 3D.
* Analyse de la **convergence** de l'énergie de liaison excitonique.
* Observation des **points de résonance** et du recouvrement électron-trou sous champ électrique externe.

## Author
* **Gendronneau Maël**
