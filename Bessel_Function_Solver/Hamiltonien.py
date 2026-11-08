from Potentiel import Potentiel
from BaseBessel import BaseBessel
from utils import hbar
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class Hamiltonien():

    def __init__(self, baseBessel : BaseBessel, potentiel : Potentiel, E_max = 0):
        self.base = baseBessel
        self.potentiel = potentiel
        self.E_max = E_max
        self.V_r, self.m_r = self.potentiel.makePotentiel(baseBessel.r)
        self.Vnm = self.build_V()
        self.Tnm = self.build_T()
        self.energies = None
        self.eigenvectors = None
        self.H = self.Tnm + self.Vnm

    def build_T(self):
        #Maintenant on prend en compte la masse éffective
        Aij = (self.base.phi_prime * (self.base.r / self.m_r * self.base.poids)[None,:]) @ self.base.phi_prime.T
        r = self.base.r
        r_sur = np.where(r > 0, r, 1)
        w = np.where(r > 0, 1 / (self.m_r * r_sur), 0)
        Bij = (self.base.phi * (w * self.base.poids)[None, :]) @ self.base.phi.T
        return hbar**2 / 2 * (Aij + self.base.m[:, None]**2 * Bij) * self.base.angulaire
    
    def build_V(self) :
        V_nm = (self.base.phi * (self.base.r * self.V_r * self.base.poids)[None,:]) @ self.base.phi.T * self.base.angulaire
        return V_nm

    def solve(self) :
        #Je solve par famille (m,type)
        list_energies = []
        list_eigenvectors = []
        tag_m = []
        tag_type = []
        for m in np.unique(self.base.m) :
            for typeSin in np.unique(self.base.est_sin) :
                mask = (self.base.m == m) & (self.base.est_sin == typeSin)
                idx = np.where(mask)[0]
                k = len(idx)
                if k == 0 :
                    continue
                tag_m.append(np.full(k, m))
                tag_type.append(np.full(k, typeSin))

                H_reduced = self.H[np.ix_(idx, idx)]
                energies, eigenvectors = eigh(H_reduced)

                matrix = np.zeros((len(self.base.m),k))
                matrix[idx, :] = eigenvectors

                list_energies.append(energies)
                list_eigenvectors.append(matrix)
                
        energies = np.concatenate(list_energies)
        m_etats = np.concatenate(tag_m)
        sin_etats = np.concatenate(tag_type)
        vecteurs = np.hstack(list_eigenvectors)

        # Ordre de tri par énergie croissante
        ordre = np.argsort(energies)

        # Stocker les résultats triés
        self.energies = energies[ordre]
        self.tag_m = m_etats[ordre]
        self.tag_type = sin_etats[ordre]
        self.eigenvectors = vecteurs[:, ordre]

        return self.energies, self.eigenvectors


    def plot(self, E_max=0, rotate=True, density=False, dessus=True, seulement_cos=True,
            n_r_plot=200, n_theta=None, n_cart=201, echelle=0.4):
        """
        E_max         : on trace les états d'énergie inférieure à E_max (meV)
        rotate        : True -> vue 3D, False -> deux vues de coupe (axes x et y)
        density       : True -> |psi|^2, False -> psi
        dessus        : ajoute une figure avec la vue de dessus de chaque état
        seulement_cos : True -> un seul état (le cos) par niveau dégénéré
        n_r_plot      : nombre de points radiaux utilisés pour le dessin
        n_theta       : nombre d'angles pour la 3D (None -> choisi selon le m maximal)
        n_cart        : résolution (points par côté) de la vue de dessus
        echelle       : hauteur des courbes, en fraction de l'écart typique entre niveaux
        """

        if self.eigenvectors is None:
            self.solve()

        # 1. Sélection des états à tracer
        selection = self.energies < E_max
        if seulement_cos:
            selection = selection & (~self.tag_type)          # un seul état par niveau dégénéré
        etats = np.where(selection)[0]
        if len(etats) == 0:
            print(f"Aucun état sous E_max = {E_max} meV")
            return

        # 2. Parties radiales des états sélectionnés, sur la grille fine
        psi_r_all = self.eigenvectors[:, etats].T @ self.base.phi

        # 3. Sous-échantillonnage pour le dessin (les calculs restent faits sur la grille fine)
        pas = max(1, len(self.base.r) // n_r_plot)
        r_plot = self.base.r[::pas]
        V_plot = self.V_r[::pas]
        psi_r_all = psi_r_all[:, ::pas]

        # 4. Nombre d'angles : au moins ~12 points par lobe (2m lobes sur un tour)
        m_max = int(np.max(self.tag_m[etats]))
        if n_theta is None:
            n_theta = max(120, 24 * m_max)

        # 5. Hauteur visuelle des courbes : fraction de l'écart typique entre niveaux
        niveaux = np.unique(np.round(self.energies[etats], 6))
        ecarts = np.diff(niveaux)
        ecarts = ecarts[ecarts > 1e-3 * (niveaux.max() - niveaux.min() + 1e-12)]   # ignore les quasi-dégénérescences
        if len(ecarts) > 0:
            ecart_typique = np.median(ecarts)
        elif np.min(self.V_r) != 0:
            ecart_typique = np.abs(np.min(self.V_r))
        else:
            ecart_typique = np.abs(niveaux[0])
        hauteur = echelle * ecart_typique

        temp = "Densités des " if density else ""
        couleurs = px.colors.qualitative.Dark24

        # 6. Informations sur chaque état sélectionné
        infos = []
        for j, i in enumerate(etats):
            m = self.tag_m[i]
            est_sin = self.tag_type[i]
            infos.append({
                "psi_r": psi_r_all[j],
                "m": m,
                "angulaire": np.sin if est_sin else np.cos,
                "nom": f"E = {self.energies[i]:.1f} meV (m = {m}, {'sin' if est_sin else 'cos'})",
                "E": self.energies[i],
                "couleur": couleurs[j % len(couleurs)],
            })

        # 7. Vue 3D, ou vues de coupe
        if rotate:
            angle = np.linspace(0, 2*np.pi, num=n_theta)
            R_grid, Theta_grid = np.meshgrid(r_plot, angle)
            X = R_grid * np.cos(Theta_grid)
            Y = R_grid * np.sin(Theta_grid)

            fig = go.Figure()
            for info in infos:
                psi_2D = info["psi_r"][None, :] * info["angulaire"](info["m"] * Theta_grid)
                if density:
                    psi_2D = np.abs(psi_2D)**2
                courbe = psi_2D / np.max(np.abs(psi_2D)) * hauteur     # même hauteur pour chaque état
                fig.add_trace(go.Surface(
                    x=X, y=Y, z=info["E"] + courbe,
                    surfacecolor=psi_2D,
                    colorscale="Inferno" if density else "RdBu_r",
                    cmid=None if density else 0,
                    showscale=False, opacity=0.9,
                    name=info["nom"], showlegend=True,
                ))
            fig.update_layout(
                title=f"{temp}états sous {E_max} meV, vue en rotation",
                scene=dict(xaxis_title="x (nm)", yaxis_title="y (nm)", zaxis_title="E (meV)",
                        aspectmode="manual", aspectratio=dict(x=1, y=1, z=1.2)),
            )
            fig.show()

        else:
            x_coupe = np.concatenate((-r_plot[::-1], r_plot[1:]))
            V_coupe = np.concatenate((V_plot[::-1], V_plot[1:]))
            # Deux coupes : (angle côté positif, angle côté négatif)
            coupes = [(0, np.pi), (np.pi/2, 3*np.pi/2)]

            fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                                subplot_titles=["Coupe le long de x", "Coupe le long de y"])
            for col in (1, 2):
                fig.add_trace(go.Scatter(x=x_coupe, y=V_coupe, mode="lines",
                                        line=dict(color="black"), name="V",
                                        legendgroup="V", showlegend=(col == 1)),
                            row=1, col=col)

            for info in infos:
                deja_en_legende = False
                for col, (theta_pos, theta_neg) in enumerate(coupes, start=1):
                    A_pos = info["angulaire"](info["m"] * theta_pos)
                    A_neg = info["angulaire"](info["m"] * theta_neg)
                    if abs(A_pos) < 1e-12 and abs(A_neg) < 1e-12:
                        continue                           # l'état est nul le long de cet axe
                    psi_coupe = np.concatenate((A_neg * info["psi_r"][::-1], A_pos * info["psi_r"][1:]))
                    if density:
                        psi_coupe = np.abs(psi_coupe)**2
                    courbe = psi_coupe / np.max(np.abs(psi_coupe)) * hauteur
                    fig.add_hline(y=info["E"], line_dash="dash", line_width=0.5,
                                line_color="grey", row=1, col=col)
                    fig.add_trace(go.Scatter(x=x_coupe, y=info["E"] + courbe,
                                            mode="lines", line=dict(color=info["couleur"]),
                                            name=info["nom"], legendgroup=info["nom"],
                                            showlegend=not deja_en_legende),
                                row=1, col=col)
                    deja_en_legende = True

            fig.update_xaxes(title_text="x (nm)", row=1, col=1)
            fig.update_xaxes(title_text="y (nm)", row=1, col=2)
            fig.update_yaxes(title_text="E (meV)", row=1, col=1)
            fig.update_layout(title=f"{temp}états sous {E_max} meV, vues de coupe")
            fig.show()

        # 8. Vue de dessus : tous les états, sur une grille de cartes
        if dessus:
            x = np.linspace(-self.base.radius, self.base.radius, n_cart)
            Xc, Yc = np.meshgrid(x, x)
            rho = np.sqrt(Xc**2 + Yc**2)
            theta = np.arctan2(Yc, Xc)

            n_col = min(len(infos), 4)                     # au plus 4 cartes par ligne
            n_lig = int(np.ceil(len(infos) / n_col))
            fig_d = make_subplots(rows=n_lig, cols=n_col,
                                subplot_titles=[info["nom"] for info in infos])

            for k, info in enumerate(infos):
                psi_c = np.interp(rho, r_plot, info["psi_r"]) * info["angulaire"](info["m"] * theta)
                if density:
                    psi_c = np.abs(psi_c)**2
                psi_c = np.where(rho > self.base.radius, np.nan, psi_c)   # hors du disque
                lig, col = k // n_col + 1, k % n_col + 1
                fig_d.add_trace(go.Heatmap(
                    x=x, y=x, z=psi_c,
                    colorscale="Inferno" if density else "RdBu_r",
                    zmid=None if density else 0,
                    showscale=False, name=info["nom"],
                ), row=lig, col=col)
                fig_d.update_yaxes(scaleanchor=f"x{k+1}" if k > 0 else "x", row=lig, col=col)

            fig_d.update_layout(title=f"{temp}états sous {E_max} meV, vue de dessus",
                                height=320 * n_lig + 80, width=300 * n_col)
            fig_d.update_annotations(font_size=10)
            fig_d.show()