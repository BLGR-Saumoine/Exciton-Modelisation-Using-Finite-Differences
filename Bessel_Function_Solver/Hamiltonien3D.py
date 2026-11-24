from Potentiel3D import Potentiel3D
from BaseBessel import BaseBessel
from BaseSinZ import BaseSinZ
from utils import hbar
import numpy as np
from scipy.linalg import eigh
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Hamiltonien3D():

    def __init__(self, baseBessel : BaseBessel, baseSinZ : BaseSinZ, potentiel : Potentiel3D, E_max = 0):
        self.base = baseBessel
        self.baseZ = baseSinZ
        self.potentiel = potentiel
        self.E_max = E_max
        self.termes_V, self.termes_inv_m = self.potentiel.makeTermes(baseBessel.r, baseSinZ.z)
        self.blocs = self.build_blocs()
        self.energies = None
        self.eigenvectors = None

    def build_blocs(self):
        # Un bloc par famille (m,type), comme dans Hamiltonien.solve
        blocs = []
        for m in np.unique(self.base.m) :
            for typeSin in np.unique(self.base.est_sin) :
                mask = (self.base.m == m) & (self.base.est_sin == typeSin)
                idx = np.where(mask)[0]
                if len(idx) == 0 :
                    continue
                blocs.append({"m": m, "est_sin": typeSin, "idx": idx})
        return blocs

    def build_T(self, idx, m):
        # On prend en compte la masse éffective : 1/m = somme des c * g(r) * h(z)
        phi, phi_prime = self.base.phi[idx], self.base.phi_prime[idx]
        chi, chi_prime = self.baseZ.chi, self.baseZ.chi_prime
        r = self.base.r
        r_sur = np.where(r > 0, r, 1)
        w = np.where(r > 0, 1 / r_sur, 0)
        angulaire = self.base.angulaire[idx[0], idx[0]] # π ou 2π, le même pour tout le bloc

        T = 0
        for c, g, h in self.termes_inv_m :
            # Partie radiale (comme Hamiltonien.build_T, avec g à la place de 1/m_r)
            Aij = (phi_prime * (r * g * self.base.poids)[None,:]) @ phi_prime.T * angulaire
            Bij = (phi * (w * g * self.base.poids)[None,:]) @ phi.T * angulaire
            Sij = (phi * (r * g * self.base.poids)[None,:]) @ phi.T * angulaire
            # Partie en z
            Skl = (chi * (h * self.baseZ.poids)[None,:]) @ chi.T
            Dkl = (chi_prime * (h * self.baseZ.poids)[None,:]) @ chi_prime.T

            T = T + c * (np.kron(Aij + m**2 * Bij, Skl) + np.kron(Sij, Dkl))
        return hbar**2 / 2 * T

    def build_V(self, idx) :
        # V = somme des c * g(r) * h(z)
        phi = self.base.phi[idx]
        chi = self.baseZ.chi
        angulaire = self.base.angulaire[idx[0], idx[0]]

        V = 0
        for c, g, h in self.termes_V :
            Sij = (phi * (self.base.r * g * self.base.poids)[None,:]) @ phi.T * angulaire
            Skl = (chi * (h * self.baseZ.poids)[None,:]) @ chi.T

            V = V + c * np.kron(Sij, Skl)
        return V

    def solve(self) :
        #Je solve par famille (m,type), sans jamais construire le H complet
        list_energies = []
        list_eigenvectors = []
        tag_m = []
        tag_type = []
        tag_bloc = []
        tag_colonne = []
        for b, bloc in enumerate(self.blocs) :
            idx = bloc["idx"]
            H_reduced = self.build_T(idx, bloc["m"]) + self.build_V(idx)
            energies, eigenvectors = eigh(H_reduced)
            k = len(energies)

            tag_m.append(np.full(k, bloc["m"]))
            tag_type.append(np.full(k, bloc["est_sin"]))
            tag_bloc.append(np.full(k, b))
            tag_colonne.append(np.arange(k))

            list_energies.append(energies)
            list_eigenvectors.append(eigenvectors) # on garde les vecteurs dans leur bloc

        energies = np.concatenate(list_energies)

        # Ordre de tri par énergie croissante
        ordre = np.argsort(energies)

        # Stocker les résultats triés
        self.energies = energies[ordre]
        self.tag_m = np.concatenate(tag_m)[ordre]
        self.tag_type = np.concatenate(tag_type)[ordre]
        self.tag_bloc = np.concatenate(tag_bloc)[ordre]
        self.tag_colonne = np.concatenate(tag_colonne)[ordre]
        self.eigenvectors = list_eigenvectors

        return self.energies, self.eigenvectors

    def getPsi(self, i) :
        # Partie (r,z) de l'état i (trié par énergie) : Ψ(r,z) = Σ c_nk φ_n(r) χ_k(z), forme (len(r), len(z))
        if self.eigenvectors is None:
            self.solve()

        b = self.tag_bloc[i]
        idx = self.blocs[b]["idx"]
        c = self.eigenvectors[b][:, self.tag_colonne[i]]
        C = c.reshape(len(idx), len(self.baseZ.k)) # indice a = n*N_z + k
        return self.base.phi[idx].T @ C @ self.baseZ.chi

    def plot(self, E_max=0, density=False, coupe=True, dessus=True, iso=False, seulement_cos=True,
            n_r_plot=200, n_z_plot=150, n_cart=201, n_iso=40, niveau_iso=0.3, n_iso_max=8):
        """
        E_max         : on trace les états d'énergie inférieure à E_max (meV)
        density       : True -> |psi|^2, False -> psi
        coupe         : vue de coupe dans le plan (x,z), à y = 0
        dessus        : vue de dessus (x,y), au z où l'état est le plus présent
        iso           : isosurfaces en 3D (plus lourd, limité à n_iso_max états)
        seulement_cos : True -> un seul état (le cos) par niveau dégénéré
        n_r_plot      : nombre de points radiaux utilisés pour le dessin
        n_z_plot      : nombre de points en z utilisés pour le dessin
        n_cart        : résolution (points par côté) de la vue de dessus
        n_iso         : résolution (points par côté) de la grille 3D des isosurfaces
        niveau_iso    : niveau des isosurfaces, en fraction du maximum
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

        # 2. Sous-échantillonnage pour le dessin (les calculs restent faits sur la grille fine)
        pas_r = max(1, len(self.base.r) // n_r_plot)
        pas_z = max(1, len(self.baseZ.z) // n_z_plot)
        r_plot = self.base.r[::pas_r]
        z_plot = self.baseZ.z[::pas_z]

        temp = "Densités des " if density else ""
        echelle = "Inferno" if density else "RdBu_r"

        # 3. Informations sur chaque état sélectionné
        infos = []
        for i in etats:
            m = self.tag_m[i]
            est_sin = self.tag_type[i]
            psi_rz = self.getPsi(i)[::pas_r, ::pas_z]
            infos.append({
                "psi_rz": psi_rz,
                "m": m,
                "angulaire": np.sin if est_sin else np.cos,
                "nom": f"E = {self.energies[i]:.1f} meV (m = {m}, {'sin' if est_sin else 'cos'})",
                "E": self.energies[i],
                "i_z": np.argmax(np.sum(psi_rz**2, axis=0)),   # z où l'état est le plus présent
            })

        n_col = min(len(infos), 4)                     # au plus 4 cartes par ligne
        n_lig = int(np.ceil(len(infos) / n_col))

        # 4. Vue de coupe dans le plan (x,z) : x > 0 -> θ = 0, x < 0 -> θ = π
        if coupe:
            x_coupe = np.concatenate((-r_plot[::-1], r_plot[1:]))
            contour = self.contourDot()

            fig_c = make_subplots(rows=n_lig, cols=n_col,
                                subplot_titles=[info["nom"] for info in infos])
            for k, info in enumerate(infos):
                A_pos = info["angulaire"](info["m"] * 0)
                A_neg = info["angulaire"](info["m"] * np.pi)
                psi_c = np.concatenate((A_neg * info["psi_rz"][::-1], A_pos * info["psi_rz"][1:]))
                if density:
                    psi_c = np.abs(psi_c)**2
                lig, col = k // n_col + 1, k % n_col + 1
                fig_c.add_trace(go.Heatmap(
                    x=x_coupe, y=z_plot, z=psi_c.T,
                    colorscale=echelle,
                    zmid=None if density else 0,
                    showscale=False, name=info["nom"],
                ), row=lig, col=col)
                if contour is not None:
                    for xc, zc in contour:
                        fig_c.add_trace(go.Scatter(x=xc, y=zc, mode="lines",
                                                line=dict(color="grey", width=1, dash="dash"),
                                                showlegend=False, hoverinfo="skip"),
                                        row=lig, col=col)
                fig_c.update_xaxes(title_text="x (nm)", row=lig, col=col)
                fig_c.update_yaxes(title_text="z (nm)", scaleanchor=f"x{k+1}" if k > 0 else "x", row=lig, col=col)

            fig_c.update_layout(title=f"{temp}états sous {E_max} meV, coupe dans le plan (x,z)",
                                height=320 * n_lig + 80, width=300 * n_col)
            fig_c.update_annotations(font_size=10)
            fig_c.show()

        # 5. Vue de dessus : coupe (x,y) au z où l'état est le plus présent
        if dessus:
            x = np.linspace(-self.base.radius, self.base.radius, n_cart)
            Xc, Yc = np.meshgrid(x, x)
            rho = np.sqrt(Xc**2 + Yc**2)
            theta = np.arctan2(Yc, Xc)

            titres = [f"{info['nom']}, z = {z_plot[info['i_z']]:.1f} nm" for info in infos]
            fig_d = make_subplots(rows=n_lig, cols=n_col, subplot_titles=titres)

            for k, info in enumerate(infos):
                psi_r = info["psi_rz"][:, info["i_z"]]
                psi_c = np.interp(rho, r_plot, psi_r) * info["angulaire"](info["m"] * theta)
                if density:
                    psi_c = np.abs(psi_c)**2
                psi_c = np.where(rho > self.base.radius, np.nan, psi_c)   # hors du disque
                lig, col = k // n_col + 1, k % n_col + 1
                fig_d.add_trace(go.Heatmap(
                    x=x, y=x, z=psi_c,
                    colorscale=echelle,
                    zmid=None if density else 0,
                    showscale=False, name=info["nom"],
                ), row=lig, col=col)
                fig_d.update_yaxes(scaleanchor=f"x{k+1}" if k > 0 else "x", row=lig, col=col)

            fig_d.update_layout(title=f"{temp}états sous {E_max} meV, vue de dessus",
                                height=320 * n_lig + 80, width=300 * n_col)
            fig_d.update_annotations(font_size=10)
            fig_d.show()

        # 6. Isosurfaces en 3D
        if iso:
            infos_iso = infos[:n_iso_max]
            n_col_i = min(len(infos_iso), 4)
            n_lig_i = int(np.ceil(len(infos_iso) / n_col_i))
            fig_i = make_subplots(rows=n_lig_i, cols=n_col_i,
                                specs=[[{"type": "scene"}] * n_col_i] * n_lig_i,
                                subplot_titles=[info["nom"] for info in infos_iso])

            x = np.linspace(-self.base.radius, self.base.radius, n_iso)
            z = np.linspace(z_plot[0], z_plot[-1], n_iso)
            X, Y, Z = np.meshgrid(x, x, z, indexing="ij")
            rho = np.sqrt(X**2 + Y**2)
            theta = np.arctan2(Y, X)
            dans_boite = rho <= self.base.radius

            for k, info in enumerate(infos_iso):
                interpolateur = RegularGridInterpolator((r_plot, z_plot), info["psi_rz"],
                                                        bounds_error=False, fill_value=0)
                psi_3D = interpolateur(np.stack((np.minimum(rho, r_plot[-1]), Z), axis=-1))
                psi_3D = np.where(dans_boite, psi_3D * info["angulaire"](info["m"] * theta), 0)
                if density:
                    psi_3D = np.abs(psi_3D)**2
                psi_max = np.max(np.abs(psi_3D))
                lig, col = k // n_col_i + 1, k % n_col_i + 1

                if density:
                    isomin, isomax, nb = niveau_iso * psi_max, niveau_iso * psi_max, 1
                else:
                    isomin, isomax, nb = -niveau_iso * psi_max, niveau_iso * psi_max, 2   # lobe négatif et positif
                fig_i.add_trace(go.Isosurface(
                    x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=psi_3D.ravel(),
                    isomin=isomin, isomax=isomax, surface_count=nb,
                    colorscale=echelle, cmid=None if density else 0,
                    caps=dict(x_show=False, y_show=False, z_show=False),
                    showscale=False, name=info["nom"],
                ), row=lig, col=col)

            fig_i.update_scenes(xaxis_title="x (nm)", yaxis_title="y (nm)", zaxis_title="z (nm)",
                                aspectmode="data")
            fig_i.update_layout(title=f"{temp}états sous {E_max} meV, isosurfaces à {niveau_iso:.0%} du maximum",
                                height=400 * n_lig_i + 80, width=400 * n_col_i)
            fig_i.update_annotations(font_size=10)
            fig_i.show()

    def contourDot(self):
        # Contour du dot dans le plan (x,z), pour le dessiner sur les coupes (cylindre et anneau seulement)
        if self.potentiel.type2boite == "cylindre":
            R, h, V0 = self.potentiel.constantes
            rectangles = [(-R, R)]
        elif self.potentiel.type2boite == "anneau":
            L1, L2, h, V0 = self.potentiel.constantes
            rectangles = [(-L2, -L1), (L1, L2)]
        else:
            return None
        return [(np.array([a, b, b, a, a]), np.array([-h/2, -h/2, h/2, h/2, -h/2])) for a, b in rectangles]