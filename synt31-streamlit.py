# -*- coding: utf-8 -*-
# =============================================
# app_streamlit.py
# Conversion Streamlit de ThinFilmOptimizerApp
# GÉNÉRÉ PAR IA - NÉCESSITE VÉRIFICATION, DÉBOGAGE ET TESTS APPROFONDIS
# =============================================

import streamlit as st
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.lax import scan, cond
import numpy as np
jax.config.update("jax_enable_x64", True) # Utiliser float64 pour la précision
import matplotlib.pyplot as plt
import pandas as pd
import functools
from typing import Union, Tuple, Dict, List, Any, Callable, Optional
from scipy.optimize import minimize, OptimizeResult # Attention aux dépendances SciPy
import time
import datetime
import traceback
from collections import deque # Utilisé pour l'historique "Undo"

# --- Constantes ---
MIN_THICKNESS_PHYS_NM = 0.01
BASE_NEEDLE_THICKNESS_NM = 0.1
DEFAULT_NEEDLE_SCAN_STEP_NM = 2.0
AUTO_NEEDLES_PER_CYCLE = 5
AUTO_MAX_CYCLES = 5
MSE_IMPROVEMENT_TOLERANCE = 1e-9
EXCEL_FILE_PATH = "indices.xlsx" # ** FICHIER INDISPENSABLE **

# =============================================
# SECTION 1 : FONCTIONS DE CALCUL & LOGIQUE MÉTIER
# (Issues du code original, potentiellement adaptées pour ne plus dépendre de 'self')
# =============================================

# --- Gestion des Données Matériaux ---

@st.cache_data # Mise en cache Streamlit
def load_material_data_from_xlsx_sheet(file_path: str, sheet_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Charge n, k depuis une feuille Excel. Retourne (l, n, k, logs)."""
    logs = []
    try:
        # Vérifier l'existence du fichier de manière robuste
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        except FileNotFoundError:
            # Utiliser st.error pour afficher un message clair dans l'app
            st.error(f"Fichier Excel '{file_path}' introuvable. Vérifiez sa présence.")
            logs.append(f"Erreur critique: Fichier Excel introuvable: {file_path}")
            # Retourner None pour indiquer l'échec, sera géré par l'appelant
            return None, None, None, logs
        except Exception as e:
            st.error(f"Erreur lors de la lecture Excel ('{file_path}', feuille '{sheet_name}'): {e}")
            logs.append(f"Erreur Excel inattendue ({type(e).__name__}): {e}")
            return None, None, None, logs

        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        numeric_df = numeric_df.dropna(how='all') # Enlève lignes entièrement vides

        # S'assurer qu'on a au moins 3 colonnes avant de dropper sur leur base
        if numeric_df.shape[1] >= 3:
            cols_to_check = numeric_df.columns[:3]
            numeric_df = numeric_df.dropna(subset=cols_to_check) # Enlève lignes avec NaN dans les 3 premières cols
        else:
            logs.append(f"Avertissement: Feuille '{sheet_name}' ne contient pas 3 colonnes numériques.")
            # Retourner des tableaux vides pour indiquer qu'aucune donnée exploitable n'a été trouvée
            return np.array([]), np.array([]), np.array([]), logs

        # Vérifier si le dataframe est vide après dropna
        if numeric_df.empty:
             logs.append(f"Avertissement: Aucune donnée numérique valide trouvée dans '{sheet_name}' après nettoyage.")
             return np.array([]), np.array([]), np.array([]), logs

        # Trier par longueur d'onde (colonne 0)
        try:
             numeric_df = numeric_df.sort_values(by=numeric_df.columns[0])
        except IndexError:
             logs.append(f"Erreur: Impossible de trier les données pour la feuille '{sheet_name}'. Colonne d'index 0 manquante?")
             return np.array([]), np.array([]), np.array([]), logs


        l_nm = numeric_df.iloc[:, 0].values.astype(np.float64)
        n = numeric_df.iloc[:, 1].values.astype(np.float64)
        k = numeric_df.iloc[:, 2].values.astype(np.float64)

        # Vérifier si k est négatif (non physique)
        if np.any(k < -1e-9): # Tolérance numérique
             invalid_k_indices = np.where(k < -1e-9)[0]
             logs.append(f"AVERTISSEMENT: Valeurs k négatives détectées et mises à 0 pour '{sheet_name}' aux indices: {invalid_k_indices.tolist()}")
             k = np.maximum(k, 0.0) # Forcer k >= 0

        if len(l_nm) == 0:
            logs.append(f"Avertissement: Aucune ligne de données valide après conversion dans '{sheet_name}'.")
            return np.array([]), np.array([]), np.array([]), logs

        logs.append(f"Données chargées '{sheet_name}': {len(l_nm)} pts [{l_nm.min():.1f}-{l_nm.max():.1f} nm]")
        return l_nm, n, k, logs

    except ValueError as ve:
        logs.append(f"Erreur de valeur Excel ('{sheet_name}'): {ve}")
        return None, None, None, logs # Indiquer l'échec
    except Exception as e:
        logs.append(f"Erreur inattendue lecture Excel ('{sheet_name}'): {type(e).__name__} - {e}")
        # traceback.print_exc() # Pour débogage serveur si besoin
        return None, None, None, logs

def get_available_materials_from_excel(excel_path: str) -> Tuple[List[str], List[str]]:
    """Lit les noms de feuilles (matériaux) depuis le fichier Excel."""
    logs = []
    try:
        xl = pd.ExcelFile(excel_path)
        # Exclure les noms de feuilles par défaut comme "Sheet1", "Sheet2"...
        sheet_names = [name for name in xl.sheet_names if not name.startswith("Sheet")]
        logs.append(f"Matériaux trouvés dans {excel_path}: {sheet_names}")
        return sheet_names, logs
    except FileNotFoundError:
        # Gérer l'erreur de fichier non trouvé plus spécifiquement
        st.error(f"Fichier Excel '{excel_path}' introuvable pour lister les matériaux.")
        logs.append(f"Erreur critique FNF: Fichier Excel {excel_path} non trouvé pour lister matériaux.")
        # Renvoyer une liste vide en cas d'erreur critique
        return [], logs
    except Exception as e:
        st.error(f"Erreur lors de la lecture des noms de feuilles depuis '{excel_path}': {e}")
        logs.append(f"Erreur lecture noms feuilles depuis {excel_path}: {type(e).__name__} - {e}")
        return [], logs


# Fonctions d'indice pour matériaux prédéfinis
@jax.jit
def get_n_fused_silica(l_nm: jnp.ndarray) -> jnp.ndarray:
    l_um_sq = (l_nm / 1000.0)**2
    # Utiliser jnp.maximum pour éviter les valeurs négatives sous la racine carrée dues à des erreurs numériques
    term1_den = jnp.maximum(l_um_sq - 0.0684043**2, 1e-12)
    term2_den = jnp.maximum(l_um_sq - 0.1162414**2, 1e-12)
    term3_den = jnp.maximum(l_um_sq - 9.896161**2, 1e-12)
    n_sq_raw = 1.0 + (0.6961663 * l_um_sq) / term1_den + \
                 (0.4079426 * l_um_sq) / term2_den + \
                 (0.8974794 * l_um_sq) / term3_den
    n_sq = jnp.maximum(n_sq_raw, 1.0) # n^2 doit être >= 1
    n = jnp.sqrt(n_sq)
    k = jnp.zeros_like(n) # Pas d'absorption définie ici
    return n + 1j * k

@jax.jit
def get_n_bk7(l_nm: jnp.ndarray) -> jnp.ndarray:
    l_um_sq = (l_nm / 1000.0)**2
    term1_den = jnp.maximum(l_um_sq - 0.00600069867, 1e-12)
    term2_den = jnp.maximum(l_um_sq - 0.0200179144, 1e-12)
    term3_den = jnp.maximum(l_um_sq - 103.560653, 1e-12)
    n_sq_raw = 1.0 + (1.03961212 * l_um_sq) / term1_den + \
                 (0.231792344 * l_um_sq) / term2_den + \
                 (1.01046945 * l_um_sq) / term3_den
    n_sq = jnp.maximum(n_sq_raw, 1.0) # n^2 doit être >= 1
    n = jnp.sqrt(n_sq)
    k = jnp.zeros_like(n)
    return n + 1j * k

@jax.jit
def get_n_d263(l_nm: jnp.ndarray) -> jnp.ndarray:
    n = jnp.full_like(l_nm, 1.523, dtype=jnp.float64)
    k = jnp.zeros_like(n)
    return n + 1j * k

# Interpolation JIT-compilable
@jax.jit
def interp_nk_cached(l_target: jnp.ndarray, l_data: jnp.ndarray, n_data: jnp.ndarray, k_data: jnp.ndarray) -> jnp.ndarray:
    # Assurer que k reste >= 0 après interpolation
    n_interp = jnp.interp(l_target, l_data, n_data)
    k_interp_raw = jnp.interp(l_target, l_data, k_data)
    k_interp = jnp.maximum(k_interp_raw, 0.0) # Force k >= 0
    return n_interp + 1j * k_interp

MaterialInputType = Union[complex, float, int, str, Tuple[np.ndarray, np.ndarray, np.ndarray]]

# Fonction principale pour obtenir les indices pour un vecteur lambda
def _get_nk_array_for_lambda_vec(material_definition: MaterialInputType,
                                 l_vec_target_jnp: jnp.ndarray,
                                 excel_file_path: str) -> Tuple[Optional[jnp.ndarray], List[str]]:
    """Prépare l'array n+ik pour un vecteur lambda. Retourne (array, logs) ou (None, logs) en cas d'échec."""
    logs = []
    try:
        if isinstance(material_definition, (complex, float, int)):
            # Cas simple : indice constant
            nk_complex = jnp.asarray(material_definition, dtype=jnp.complex128)
            # Vérifier la validité physique
            if nk_complex.real <= 0:
                logs.append(f"AVERTISSEMENT: Indice constant n'={nk_complex.real} <= 0 pour '{material_definition}'. Utilisation de n'=1.0.")
                nk_complex = complex(1.0, nk_complex.imag)
            if nk_complex.imag < 0:
                 logs.append(f"AVERTISSEMENT: Indice constant k={nk_complex.imag} < 0 pour '{material_definition}'. Utilisation de k=0.0.")
                 nk_complex = complex(nk_complex.real, 0.0)
            result = jnp.full(l_vec_target_jnp.shape, nk_complex)

        elif isinstance(material_definition, str):
            # Cas d'un nom de matériau (prédéfinie ou Excel)
            mat_upper = material_definition.upper()
            if mat_upper == "FUSED SILICA":
                result = get_n_fused_silica(l_vec_target_jnp)
            elif mat_upper == "BK7":
                result = get_n_bk7(l_vec_target_jnp)
            elif mat_upper == "D263":
                 result = get_n_d263(l_vec_target_jnp)
            else:
                # C'est une feuille Excel
                sheet_name = material_definition
                # Utiliser la fonction cachée pour charger les données
                l_data, n_data, k_data, load_logs = load_material_data_from_xlsx_sheet(excel_file_path, sheet_name)
                logs.extend(load_logs)
                # Vérifier si le chargement a réussi (retourne None ou des tableaux vides si échec/vide)
                if l_data is None or len(l_data) == 0:
                    st.error(f"Impossible de charger ou données vides pour le matériau '{sheet_name}' depuis {excel_file_path}.")
                    logs.append(f"Erreur critique: Échec chargement données pour '{sheet_name}'.")
                    return None, logs # Échec critique

                l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))

                # Vérifier limites d'interpolation
                l_target_min = jnp.min(l_vec_target_jnp)
                l_target_max = jnp.max(l_vec_target_jnp)
                l_data_min = jnp.min(l_data_jnp)
                l_data_max = jnp.max(l_data_jnp)

                if l_target_min < l_data_min - 1e-6 or l_target_max > l_data_max + 1e-6:
                     logs.append(f"AVERTISSEMENT: Interpolation pour '{sheet_name}' hors limites [{l_data_min:.1f}, {l_data_max:.1f}] nm (cible: [{l_target_min:.1f}, {l_target_max:.1f}] nm). Extrapolation utilisée.")

                result = interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)

        elif isinstance(material_definition, tuple) and len(material_definition) == 3:
             # Cas données brutes passées (moins probable en Streamlit)
             l_data, n_data, k_data = material_definition
             l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
             if not len(l_data_jnp): raise ValueError("Données matériau brutes vides.")
             # Assurer le tri
             sort_indices = jnp.argsort(l_data_jnp)
             l_data_jnp = l_data_jnp[sort_indices]
             n_data_jnp = n_data_jnp[sort_indices]
             k_data_jnp = k_data_jnp[sort_indices]
             # Vérifier k>=0
             if np.any(k_data_jnp < -1e-9):
                 logs.append("AVERTISSEMENT: k<0 dans les données matériau brutes. Mise à 0.")
                 k_data_jnp = jnp.maximum(k_data_jnp, 0.0)
             result = interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)

        else:
            raise TypeError(f"Type de définition matériau non supporté: {type(material_definition)}")

        # Vérification finale post-traitement
        if jnp.any(jnp.isnan(result.real)) or jnp.any(result.real <= 0):
            logs.append(f"AVERTISSEMENT: n'<=0 ou NaN détecté pour '{material_definition}'. Remplacé par n'=1.")
            result = jnp.where(jnp.isnan(result.real) | (result.real <= 0), 1.0 + 1j*result.imag, result)
        if jnp.any(jnp.isnan(result.imag)) or jnp.any(result.imag < 0):
             logs.append(f"AVERTISSEMENT: k<0 ou NaN détecté pour '{material_definition}'. Remplacé par k=0.")
             result = jnp.where(jnp.isnan(result.imag) | (result.imag < 0), result.real + 0.0j, result)

        return result, logs

    except Exception as e:
        logs.append(f"Erreur préparation données matériau pour '{material_definition}': {e}")
        # traceback.print_exc() # Utile pour débogage
        st.error(f"Erreur critique lors de la préparation du matériau '{material_definition}': {e}")
        return None, logs # Échec critique


# Fonction pour obtenir n+ik à UNE SEULE longueur d'onde (pour QWOT etc.)
# Pas besoin de cache Streamlit ici car appelée avec des singletons, rapide.
def _get_nk_at_lambda(material_definition: MaterialInputType, l_nm_target: float, excel_file_path: str) -> Tuple[Optional[complex], List[str]]:
    """Obtient n+ik pour une seule longueur d'onde."""
    logs = []
    if l_nm_target <= 0:
        logs.append(f"Erreur: Longueur d'onde cible {l_nm_target}nm invalide pour obtenir n+ik.")
        return None, logs

    l_vec_jnp = jnp.array([l_nm_target], dtype=jnp.float64)
    nk_array, prep_logs = _get_nk_array_for_lambda_vec(material_definition, l_vec_jnp, excel_file_path)
    logs.extend(prep_logs)

    if nk_array is None:
        # L'erreur a déjà été loguée et affichée par _get_nk_array_for_lambda_vec
        return None, logs
    else:
        # Retourner la valeur complexe unique
        nk_complex = complex(nk_array[0])
        return nk_complex, logs


# --- Calcul Transmittance (Adapté pour retourner logs) ---

@jax.jit
def _compute_layer_matrix_scan_step_jit(carry_matrix: jnp.ndarray, layer_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, None]:
    """Fonction interne pour JAX scan (identique à celle dans _compute_layer_matrix_scan_step)."""
    thickness, Ni, l_val = layer_data
    eta = Ni
    safe_l_val = jnp.maximum(l_val, 1e-9)
    phi = (2 * jnp.pi / safe_l_val) * (Ni * thickness)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    def compute_M_layer(thickness_: jnp.ndarray) -> jnp.ndarray:
        safe_eta = jnp.where(jnp.abs(eta) < 1e-12, 1e-12 + 0j, eta)
        m01 = (1j / safe_eta) * sin_phi
        m10 = 1j * eta * sin_phi
        M_layer = jnp.array([[cos_phi, m01], [m10, cos_phi]], dtype=jnp.complex128)
        return M_layer @ carry_matrix

    def compute_identity(thickness_: jnp.ndarray) -> jnp.ndarray:
        return carry_matrix

    new_matrix = cond(thickness > 1e-12, compute_M_layer, compute_identity, thickness)
    return new_matrix, None

@jax.jit
def compute_stack_matrix_core_jax(ep_vector: jnp.ndarray, layer_indices: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    """Calcule la matrice caractéristique pour une seule lambda (version générique)."""
    num_layers = len(ep_vector)
    layers_scan_data = (ep_vector, layer_indices, jnp.full(num_layers, l_val))
    M_initial = jnp.eye(2, dtype=jnp.complex128)
    M_final, _ = scan(_compute_layer_matrix_scan_step_jit, M_initial, layers_scan_data)
    return M_final

@jax.jit
def calculate_single_wavelength_T_core(l_val: jnp.ndarray, ep_vector_contig: jnp.ndarray,
                                     layer_indices_at_lval: jnp.ndarray, nSub_at_lval: jnp.ndarray) -> jnp.ndarray:
    """Calcule T pour une seule lambda (version générique)."""
    etainc = 1.0 + 0j # Indice du milieu incident (air)
    etasub = nSub_at_lval

    # Fonction interne pour gérer l > 0
    def calculate_for_valid_l(l_: jnp.ndarray) -> jnp.ndarray:
        # Assurer que les indices de couche sont 1D pour cette lambda
        current_layer_indices = layer_indices_at_lval # Doit être déjà 1D
        M = compute_stack_matrix_core_jax(ep_vector_contig, current_layer_indices, l_)

        m00, m01 = M[0, 0], M[0, 1]
        m10, m11 = M[1, 0], M[1, 1]

        # Calcul de la transmittance (formule standard)
        rs_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        rs_denominator_abs = jnp.abs(rs_denominator)

        # Éviter division par zéro
        safe_denominator = jnp.where(rs_denominator_abs < 1e-12, 1e-12 + 0j, rs_denominator)
        ts = (2.0 * etainc) / safe_denominator

        # Calcul de T = (n_sub / n_inc) * |ts|^2
        real_etasub = jnp.real(etasub)
        real_etainc = jnp.real(etainc) # = 1.0

        # Gérer le cas où n_inc serait 0 (ne devrait pas arriver ici)
        safe_real_etainc = jnp.maximum(real_etainc, 1e-9)
        Ts_complex = (real_etasub / safe_real_etainc) * (ts * jnp.conj(ts))

        # T doit être réel et entre 0 et 1 (généralement)
        Ts = jnp.real(Ts_complex)

        # Retourner NaN si le dénominateur était trop petit, sinon T
        # Remplacer NaN par 0.0 ensuite avec nan_to_num
        return jnp.where(rs_denominator_abs < 1e-12, jnp.nan, Ts)

    # Fonction interne pour gérer l <= 0
    def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray:
        return jnp.nan # Longueur d'onde invalide

    # Utiliser cond pour choisir la branche
    Ts_result = cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)
    return Ts_result


# --- Fonction principale pour T(lambda) pour séquence H/L ---
def calculate_T_from_ep_jax(ep_vector: Union[np.ndarray, List[float]],
                            nH_material: MaterialInputType,
                            nL_material: MaterialInputType,
                            nSub_material: MaterialInputType,
                            l_vec: Union[np.ndarray, List[float]],
                            excel_file_path: str) -> Tuple[Optional[Dict[str, np.ndarray]], List[str]]:
    """Calcule T(lambda) pour une séquence H/L standard."""
    logs = []
    l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64)
    ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)

    if not l_vec_jnp.size:
        logs.append("Vecteur lambda vide, aucun calcul de T effectué.")
        return {'l': np.array([]), 'Ts': np.array([])}, logs

    if not ep_vector_jnp.size:
         logs.append("Structure vide (0 couches). Calcul du substrat nu.")
         # Calculer juste la réflectance/transmittance du substrat nu si nécessaire
         # Pour simplifier, on retourne T=1 (ou presque) pour le substrat nu si non absorbant
         nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
         logs.extend(logs_sub)
         if nSub_arr is None:
             return None, logs # Echec chargement substrat
         # Formule de Fresnel pour l'interface Air/Substrat (approximée ici)
         n_sub = jnp.real(nSub_arr)
         k_sub = jnp.imag(nSub_arr)
         # Approximation simple : si k est petit, T ~ 1. Sinon, ???
         # Une meilleure approche serait de calculer T = 4n / ( (n+1)^2 + k^2 ) mais JAX nécessaire
         # Ici on retourne 1 pour simplifier pour le cas 0 couche
         Ts = jnp.ones_like(l_vec_jnp)
         return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts)}, logs


    logs.append(f"Préparation des indices pour {len(l_vec_jnp)} lambdas...")
    start_time = time.time()
    nH_arr, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_jnp, excel_file_path)
    logs.extend(logs_h)
    nL_arr, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_jnp, excel_file_path)
    logs.extend(logs_l)
    nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
    logs.extend(logs_sub)

    # Vérifier si un chargement a échoué
    if nH_arr is None or nL_arr is None or nSub_arr is None:
        logs.append("Erreur critique: Échec du chargement d'un des indices matériau.")
        return None, logs # Retourner None pour indiquer l'échec

    logs.append(f"Préparation indices terminée en {time.time() - start_time:.3f}s.")

    # Préparer la fonction JITée pour vmap
    calculate_single_wavelength_T_hl_jit = jax.jit(calculate_single_wavelength_T_core)

    num_layers = len(ep_vector_jnp)
    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH_arr, nL_arr) # Shape (n_layers, n_lambda)
    # Transposer pour vmap: on veut itérer sur lambda, donc shape (n_lambda, n_layers)
    indices_alternating_T = indices_alternating.T

    # vmap sur la dimension lambda (axe 0)
    # La fonction attend (lambda, ep_vector, indices_pour_cette_lambda, nSub_pour_cette_lambda)
    Ts_arr_raw = vmap(calculate_single_wavelength_T_hl_jit, in_axes=(0, None, 0, 0))(
        l_vec_jnp, ep_vector_jnp, indices_alternating_T, nSub_arr
    )

    # Remplacer les NaN potentiels (dus à l<=0 ou dénominateur nul) par 0.0
    Ts_arr = jnp.nan_to_num(Ts_arr_raw, nan=0.0)
    # Assurer que T est dans [0, 1] physiquement (écrêtage numérique)
    Ts_arr_clipped = jnp.clip(Ts_arr, 0.0, 1.0)

    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts_arr_clipped)}, logs

# --- Fonction principale pour T(lambda) pour séquence Arbitraire ---
def calculate_T_from_ep_arbitrary_jax(ep_vector: Union[np.ndarray, List[float]],
                                     material_sequence: List[str], # Liste des noms de matériaux
                                     nSub_material: MaterialInputType,
                                     l_vec: Union[np.ndarray, List[float]],
                                     excel_file_path: str) -> Tuple[Optional[Dict[str, np.ndarray]], List[str]]:
    """Calcule T(lambda) pour une séquence de matériaux arbitraire."""
    logs = []
    l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64)
    ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)
    num_layers = len(ep_vector_jnp)

    if num_layers != len(material_sequence):
         logs.append("Erreur: La taille de ep_vector et material_sequence doit correspondre.")
         return None, logs

    if not l_vec_jnp.size:
        logs.append("Vecteur lambda vide.")
        return {'l': np.array([]), 'Ts': np.array([])}, logs

    if not ep_vector_jnp.size:
         # Cas 0 couche (identique à H/L)
         logs.append("Structure vide (0 couches). Calcul du substrat nu.")
         nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
         logs.extend(logs_sub)
         if nSub_arr is None: return None, logs
         Ts = jnp.ones_like(l_vec_jnp) # Approximation T=1
         return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts)}, logs

    logs.append(f"Préparation des indices pour séquence arbitraire ({num_layers} couches, {len(l_vec_jnp)} lambdas)...")
    start_time = time.time()
    layer_indices_list = []
    materials_ok = True
    for i, mat_name in enumerate(material_sequence):
        nk_arr, logs_layer = _get_nk_array_for_lambda_vec(mat_name, l_vec_jnp, excel_file_path)
        logs.extend(logs_layer)
        if nk_arr is None:
            logs.append(f"Erreur critique: Échec chargement matériau '{mat_name}' (couche {i+1}).")
            materials_ok = False
            break # Arrêter si un matériau échoue
        layer_indices_list.append(nk_arr)

    nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
    logs.extend(logs_sub)
    if nSub_arr is None:
        logs.append("Erreur critique: Échec chargement matériau substrat.")
        materials_ok = False

    if not materials_ok:
        return None, logs # Retourner échec

    # Empiler les indices des couches : shape (n_layers, n_lambda)
    if layer_indices_list:
        layer_indices_arr = jnp.stack(layer_indices_list, axis=0)
    else: # Devrait être couvert par le cas num_layers == 0 mais sécurité
         layer_indices_arr = jnp.empty((0, len(l_vec_jnp)), dtype=jnp.complex128)

    logs.append(f"Préparation indices terminée en {time.time() - start_time:.3f}s.")

    # Préparer la fonction JITée pour vmap
    calculate_single_wavelength_T_arb_jit = jax.jit(calculate_single_wavelength_T_core)

    # Transposer les indices pour vmap: itérer sur lambda, shape (n_lambda, n_layers)
    layer_indices_arr_T = layer_indices_arr.T

    # vmap sur la dimension lambda (axe 0)
    Ts_arr_raw = vmap(calculate_single_wavelength_T_arb_jit, in_axes=(0, None, 0, 0))(
        l_vec_jnp, ep_vector_jnp, layer_indices_arr_T, nSub_arr
    )

    Ts_arr = jnp.nan_to_num(Ts_arr_raw, nan=0.0)
    Ts_arr_clipped = jnp.clip(Ts_arr, 0.0, 1.0)

    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts_arr_clipped)}, logs


# --- Helpers QWOT / Épaisseur Initiale (Adapté pour retourner logs) ---

def calculate_initial_ep(emp: Union[List[float], Tuple[float,...]], l0: float,
                         nH0_material: MaterialInputType, nL0_material: MaterialInputType,
                         excel_file_path: str) -> Tuple[Optional[np.ndarray], List[str]]:
    """Calcule les épaisseurs initiales (ep) à partir des multiplicateurs QWOT (emp)."""
    logs = []
    num_layers = len(emp)
    ep_initial = np.zeros(num_layers, dtype=np.float64)

    if l0 <= 0:
        logs.append(f"Avertissement: l0={l0} <= 0 dans calculate_initial_ep. Épaisseurs initiales mises à 0.")
        return ep_initial, logs

    nH_complex_at_l0, logs_h = _get_nk_at_lambda(nH0_material, l0, excel_file_path)
    logs.extend(logs_h)
    nL_complex_at_l0, logs_l = _get_nk_at_lambda(nL0_material, l0, excel_file_path)
    logs.extend(logs_l)

    if nH_complex_at_l0 is None or nL_complex_at_l0 is None:
        logs.append(f"Erreur: Impossible d'obtenir les indices H ou L à l0={l0}nm. Épaisseurs initiales mises à 0.")
        st.error(f"Erreur critique lors de l'obtention des indices à l0={l0}nm pour le calcul des épaisseurs initiales.")
        return None, logs # Échec critique

    nH_real_at_l0 = nH_complex_at_l0.real
    nL_real_at_l0 = nL_complex_at_l0.real

    if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
         logs.append(f"AVERTISSEMENT: n'H({nH_real_at_l0:.3f}) ou n'L({nL_real_at_l0:.3f}) à l0={l0}nm est <= 0. Calcul QWOT peut être incorrect.")
         # Continuer mais avec prudence

    for i in range(num_layers):
        multiplier = emp[i]
        is_H_layer = (i % 2 == 0)
        n_real_layer_at_l0 = nH_real_at_l0 if is_H_layer else nL_real_at_l0

        if n_real_layer_at_l0 <= 1e-9:
            ep_initial[i] = 0.0 # Épaisseur nulle si indice invalide
        else:
            ep_initial[i] = multiplier * l0 / (4.0 * n_real_layer_at_l0)

    # Vérifier si des épaisseurs sont trop fines et les clamper à MIN_THICKNESS_PHYS_NM ou 0
    ep_initial_phys = np.where(ep_initial < MIN_THICKNESS_PHYS_NM, 0.0, ep_initial)
    num_clamped_zero = np.sum((ep_initial > 1e-12) & (ep_initial < MIN_THICKNESS_PHYS_NM))
    if num_clamped_zero > 0:
        logs.append(f"Avertissement: {num_clamped_zero} épaisseurs initiales < {MIN_THICKNESS_PHYS_NM}nm ont été mises à 0.")
        ep_initial = np.where(ep_initial < MIN_THICKNESS_PHYS_NM, 0.0, ep_initial)


    # Nouvelle vérification : s'assurer qu'aucune épaisseur n'est exactement 0 si le multiplicateur > 0
    # Ceci peut arriver si n_real_l0 est invalide (<=0)
    valid_indices = True
    for i in range(num_layers):
         if emp[i] > 1e-9 and ep_initial[i] < 1e-12:
             layer_type = "H" if i % 2 == 0 else "L"
             n_val = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
             logs.append(f"Erreur: Couche {i+1} ({layer_type}) a QWOT={emp[i]} mais épaisseur=0 (probablement n'({layer_type},l0)={n_val:.3f} <= 0).")
             valid_indices = False

    if not valid_indices:
         st.error("Erreur lors du calcul des épaisseurs initiales due à des indices invalides à l0.")
         return None, logs # Échec critique

    return ep_initial, logs

def calculate_qwot_from_ep(ep_vector: np.ndarray, l0: float,
                           nH0_material: MaterialInputType, nL0_material: MaterialInputType,
                           excel_file_path: str) -> Tuple[Optional[np.ndarray], List[str]]:
    """Calcule les multiplicateurs QWOT à partir des épaisseurs physiques (ep)."""
    logs = []
    num_layers = len(ep_vector)
    qwot_multipliers = np.full(num_layers, np.nan, dtype=np.float64) # Initier avec NaN

    if l0 <= 0:
        logs.append(f"Avertissement: l0={l0} <= 0 dans calculate_qwot_from_ep. QWOT mis à NaN.")
        return qwot_multipliers, logs # Retourne NaN

    nH_complex_at_l0, logs_h = _get_nk_at_lambda(nH0_material, l0, excel_file_path)
    logs.extend(logs_h)
    nL_complex_at_l0, logs_l = _get_nk_at_lambda(nL0_material, l0, excel_file_path)
    logs.extend(logs_l)

    if nH_complex_at_l0 is None or nL_complex_at_l0 is None:
        logs.append(f"Erreur: Impossible d'obtenir n'H ou n'L à l0={l0}nm pour calculer QWOT. Retourne NaN.")
        st.error(f"Erreur calcul QWOT : indices H/L non trouvés à l0={l0}nm.")
        return None, logs # Échec critique

    nH_real_at_l0 = nH_complex_at_l0.real
    nL_real_at_l0 = nL_complex_at_l0.real

    if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
         logs.append(f"AVERTISSEMENT: n'H({nH_real_at_l0:.3f}) ou n'L({nL_real_at_l0:.3f}) à l0={l0}nm est <= 0. Calcul QWOT peut être incorrect/NaN.")

    indices_ok = True
    for i in range(num_layers):
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real_layer_at_l0 <= 1e-9:
            # Si l'indice est invalide, le QWOT n'est pas défini (reste NaN)
            if ep_vector[i] > 1e-9: # Si l'épaisseur n'est pas nulle
                 layer_type = "H" if i % 2 == 0 else "L"
                 logs.append(f"Avertissement: Impossible de calculer QWOT pour couche {i+1} ({layer_type}) car n'({l0}nm) <= 0.")
                 indices_ok = False
            else:
                 qwot_multipliers[i] = 0.0 # Si épaisseur nulle, QWOT=0
        else:
            qwot_multipliers[i] = ep_vector[i] * (4.0 * n_real_layer_at_l0) / l0

    if not indices_ok:
        st.warning("Certains QWOT n'ont pas pu être calculés (indices invalides à l0). Ils apparaissent comme NaN.")
        # On retourne quand même le tableau avec des NaN potentiels
        return qwot_multipliers, logs
    else:
        return qwot_multipliers, logs


# --- Fonctions de Calcul MSE et Optimisation ---

def calculate_final_mse(res: Dict[str, np.ndarray], active_targets: List[Dict]) -> Tuple[Optional[float], int]:
    """Calcule le MSE final basé sur les résultats T(lambda) et les cibles actives."""
    total_squared_error = 0.0
    total_points_in_targets = 0
    mse = None # Défaut

    # Vérifier si les données de résultat sont valides
    if not active_targets or 'Ts' not in res or res['Ts'] is None or 'l' not in res or res['l'] is None:
        return mse, total_points_in_targets # Retourne None, 0

    res_l_np = np.asarray(res['l'])
    res_ts_np = np.asarray(res['Ts'])

    if res_l_np.size == 0 or res_ts_np.size == 0 or res_l_np.size != res_ts_np.size:
         return mse, total_points_in_targets # Données vides ou incohérentes

    for target in active_targets:
        # Assurer que les clés existent et sont valides
        try:
            l_min = float(target['min'])
            l_max = float(target['max'])
            t_min = float(target['target_min'])
            t_max = float(target['target_max'])
            if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0): continue # Ignorer cible invalide
            if l_max < l_min: continue # Ignorer cible invalide
        except (KeyError, ValueError, TypeError):
            continue # Ignorer cible mal formée

        # Trouver les indices dans la plage de la cible
        indices = np.where((res_l_np >= l_min) & (res_l_np <= l_max))[0]

        if indices.size > 0:
            # Prendre les T calculés dans cette zone
            calculated_Ts_in_zone = res_ts_np[indices]
            # Prendre les lambdas correspondants
            target_lambdas_in_zone = res_l_np[indices]

            # Exclure les NaN potentiels (même si nan_to_num a été fait avant, sécurité)
            finite_mask = np.isfinite(calculated_Ts_in_zone)
            calculated_Ts_in_zone = calculated_Ts_in_zone[finite_mask]
            target_lambdas_in_zone = target_lambdas_in_zone[finite_mask]

            if calculated_Ts_in_zone.size == 0: continue # Si aucun point valide dans la zone

            # Calculer la transmittance cible pour chaque point lambda dans la zone
            if abs(l_max - l_min) < 1e-9: # Cible constante
                interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
            else: # Cible en rampe
                slope = (t_max - t_min) / (l_max - l_min)
                interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)

            # Calculer les erreurs carrées
            squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
            total_squared_error += np.sum(squared_errors)
            total_points_in_targets += len(calculated_Ts_in_zone)

    # Calculer le MSE final
    if total_points_in_targets > 0:
        mse = total_squared_error / total_points_in_targets
    # Si aucune cible n'a contribué (zones vides, etc.), mse reste None

    return mse, total_points_in_targets


# Fonction coût pour l'optimisation (H/L) avec pénalité pour couches fines
@jax.jit
def calculate_mse_for_optimization_penalized_jax(ep_vector: jnp.ndarray,
                                                 nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                                 l_vec_optim: jnp.ndarray,
                                                 active_targets_tuple: Tuple[Tuple[float, float, float, float], ...],
                                                 min_thickness_phys_nm: float) -> jnp.ndarray:
    """Fonction coût pour l'optimisation (MSE + pénalité épaisseur)."""

    # 1. Calcul de la pénalité pour les couches trop fines (mais > 0)
    # On pénalise si ep < min_thickness mais ep > epsilon (pour ne pas pénaliser les couches à 0)
    below_min_mask = (ep_vector < min_thickness_phys_nm) & (ep_vector > 1e-12)
    # Pénalité quadratique qui augmente quand on s'approche de 0 depuis min_thickness
    penalty_thin = jnp.sum(jnp.where(below_min_mask, (min_thickness_phys_nm - ep_vector)**2, 0.0))
    # Facteur de poids pour la pénalité (peut nécessiter ajustement)
    penalty_weight = 1e5
    penalty_cost = penalty_thin * penalty_weight

    # 2. Calculer T avec les épaisseurs "clampées" pour l'évaluation du MSE
    # Les épaisseurs utilisées pour le calcul physique sont >= min_thickness
    ep_vector_calc = jnp.maximum(ep_vector, min_thickness_phys_nm)

    # 3. Calculer T(lambda) pour la structure clampée (séquence H/L)
    num_layers = len(ep_vector_calc)
    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH_arr, nL_arr)
    indices_alternating_T = indices_alternating.T # Transposer pour vmap sur lambda

    # vmap de la fonction de calcul T pour une seule lambda
    calculate_T_single_jit = jax.jit(calculate_single_wavelength_T_core)
    Ts_raw = vmap(calculate_T_single_jit, in_axes=(0, None, 0, 0))(
        l_vec_optim, ep_vector_calc, indices_alternating_T, nSub_arr
    )
    Ts = jnp.nan_to_num(Ts_raw, nan=0.0) # Remplacer NaN par 0

    # 4. Calculer le MSE par rapport aux cibles
    total_squared_error = 0.0
    total_points_in_targets = 0
    for i in range(len(active_targets_tuple)):
        l_min, l_max, t_min, t_max = active_targets_tuple[i]
        target_mask = (l_vec_optim >= l_min) & (l_vec_optim <= l_max)

        # Calculer la pente de la cible (0 si constante)
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        # Calculer la valeur cible pour chaque lambda du vecteur d'optimisation
        interpolated_target_t_full = t_min + slope * (l_vec_optim - l_min)

        # Erreur carrée pour tous les lambdas
        squared_errors_full = (Ts - interpolated_target_t_full)**2
        # Appliquer le masque pour ne sommer que dans la zone cible
        masked_sq_error = jnp.where(target_mask, squared_errors_full, 0.0)

        total_squared_error += jnp.sum(masked_sq_error)
        total_points_in_targets += jnp.sum(target_mask)

    # Calculer le MSE, éviter division par zéro si aucun point dans les cibles
    mse = jnp.where(total_points_in_targets > 0,
                    total_squared_error / total_points_in_targets,
                    jnp.inf) # MSE infini si aucune cible n'est atteinte

    # 5. Coût final = MSE + Pénalité
    final_cost = mse + penalty_cost

    # Retourner une valeur finie (remplacer Inf/NaN par une très grande valeur si besoin)
    return jnp.nan_to_num(final_cost, nan=jnp.inf, posinf=jnp.inf)

# Fonction coût pour séquence arbitraire (PAS de pénalité épaisseur ici, gérée par l'optimiseur Scipy via bounds)
@jax.jit
def calculate_mse_arbitrary_sequence_jax(ep_vector: jnp.ndarray,
                                         layer_indices_arr: jnp.ndarray, # Shape (n_layers, n_lambda)
                                         nSub_arr: jnp.ndarray,
                                         l_vec_eval: jnp.ndarray,
                                         active_targets_tuple: Tuple[Tuple[float, float, float, float], ...]) -> jnp.ndarray:
    """Calcule le MSE pour une séquence arbitraire (utilisé pour évaluer, pas pour optim direct avec pénalité)."""

    # Calculer T(lambda)
    layer_indices_arr_T = layer_indices_arr.T # Transpose to (n_lambda, n_layers) for vmap
    calculate_T_single_jit = jax.jit(calculate_single_wavelength_T_core)
    Ts_raw = vmap(calculate_T_single_jit, in_axes=(0, None, 0, 0))(
        l_vec_eval, ep_vector, layer_indices_arr_T, nSub_arr
    )
    Ts = jnp.nan_to_num(Ts_raw, nan=0.0)

    # Calculer le MSE (identique à la partie MSE de la fonction pénalisée)
    total_squared_error = 0.0
    total_points_in_targets = 0
    for i in range(len(active_targets_tuple)):
        l_min, l_max, t_min, t_max = active_targets_tuple[i]
        target_mask = (l_vec_eval >= l_min) & (l_vec_eval <= l_max)
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t_full = t_min + slope * (l_vec_eval - l_min)
        squared_errors_full = (Ts - interpolated_target_t_full)**2
        masked_sq_error = jnp.where(target_mask, squared_errors_full, 0.0)
        total_squared_error += jnp.sum(masked_sq_error)
        total_points_in_targets += jnp.sum(target_mask)

    mse = jnp.where(total_points_in_targets > 0,
                    total_squared_error / total_points_in_targets,
                    jnp.inf)

    return jnp.nan_to_num(mse, nan=jnp.inf, posinf=jnp.inf)

# --- Algorithmes d'Optimisation (Wrappers et Logique Principale) ---

def _run_core_optimization(ep_start_optim: np.ndarray,
                           validated_inputs: Dict, active_targets: List[Dict],
                           min_thickness_phys: float, log_prefix: str = ""
                           ) -> Tuple[Optional[np.ndarray], bool, float, List[str], str, int, int]:
    """Exécute l'optimisation L-BFGS-B avec gradient JAX."""
    logs = []
    num_layers_start = len(ep_start_optim)
    optim_success = False
    final_cost = np.inf
    result_message_str = "Optimisation non lancée ou échouée précocement."
    nit_total = 0
    nfev_total = 0
    final_ep = None # Initialiser à None

    if num_layers_start == 0:
        logs.append(f"{log_prefix}Impossible d'optimiser une structure vide.")
        return None, False, np.inf, logs, "Structure vide", 0, 0

    try:
        # Récupérer paramètres nécessaires
        l_min_optim = validated_inputs['l_range_deb']
        l_max_optim = validated_inputs['l_range_fin']
        l_step_optim = validated_inputs['l_step']
        nH_material = validated_inputs['nH_material']
        nL_material = validated_inputs['nL_material']
        nSub_material = validated_inputs['nSub_material']
        maxiter = validated_inputs['maxiter']
        maxfun = validated_inputs['maxfun']

        # Préparer le vecteur lambda pour l'optimisation
        # Utiliser geomspace comme dans Tkinter pour la cohérence
        num_pts_optim = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
        l_vec_optim_np = np.geomspace(l_min_optim, l_max_optim, num_pts_optim)
        l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
        if not l_vec_optim_np.size:
            raise ValueError("Échec de génération du vecteur lambda pour l'optimisation.")
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        logs.append(f"{log_prefix}Préparation indices dispersifs pour {len(l_vec_optim_jax)} lambdas...")

        # Obtenir les arrays d'indices
        prep_start_time = time.time()
        nH_arr_optim, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_h)
        nL_arr_optim, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_l)
        nSub_arr_optim, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_sub)

        if nH_arr_optim is None or nL_arr_optim is None or nSub_arr_optim is None:
             raise RuntimeError("Échec chargement indices pour l'optimisation.") # Erreur critique

        logs.append(f"{log_prefix} Préparation indices finie en {time.time() - prep_start_time:.3f}s.")

        # Préparer les arguments statiques pour JAX
        active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
        static_args_for_jax = (
            nH_arr_optim, nL_arr_optim, nSub_arr_optim,
            l_vec_optim_jax, active_targets_tuple,
            min_thickness_phys # Passer la constante
        )

        # Fonction coût et gradient JITée
        value_and_grad_fn = jax.jit(jax.value_and_grad(calculate_mse_for_optimization_penalized_jax))

        # Wrapper pour Scipy : doit retourner float et ndarray(float64)
        def scipy_obj_grad_wrapper(ep_vector_np_in, *args):
            try:
                # Assurer le type et la forme corrects
                ep_vector_jax = jnp.asarray(ep_vector_np_in, dtype=jnp.float64)
                value_jax, grad_jax = value_and_grad_fn(ep_vector_jax, *args)

                # Vérifier si le résultat est valide avant conversion
                if not jnp.isfinite(value_jax):
                     # logs.append(f"{log_prefix} AVERTISSEMENT: Coût non fini ({value_jax}) pour ep={ep_vector_np_in[:3]}...") # Trop verbeux pour log
                     value_np = np.inf # Retourner Inf si le coût JAX n'est pas fini
                     grad_np = np.zeros_like(ep_vector_np_in, dtype=np.float64) # Gradient nul ou aléatoire ? Zéro est plus sûr.
                else:
                     value_np = float(np.array(value_jax))
                     # Assurer que le gradient est aussi valide et du bon type
                     grad_np_raw = np.array(grad_jax, dtype=np.float64)
                     grad_np = np.nan_to_num(grad_np_raw, nan=0.0, posinf=1e6, neginf=-1e6) # Remplacer NaN/Inf dans le gradient

                return value_np, grad_np
            except Exception as e_wrap:
                 # En cas d'erreur dans le wrapper (très rare), retourner Inf
                 print(f"Erreur dans scipy_obj_grad_wrapper: {e_wrap}") # Log console pour débogage
                 return np.inf, np.zeros_like(ep_vector_np_in, dtype=np.float64)


        # Définir les bornes pour L-BFGS-B : épaisseur >= min_thickness_phys
        lbfgsb_bounds = [(min_thickness_phys, None)] * num_layers_start

        options = {'maxiter': maxiter, 'maxfun': maxfun,
                   'disp': False, # Mettre à True pour plus de détails Scipy dans la console
                   'ftol': 1e-12, 'gtol': 1e-8} # Tolérances (ajuster si besoin)

        logs.append(f"{log_prefix}Lancement L-BFGS-B avec JAX gradient...")
        opt_start_time = time.time()

        # Appel à Scipy minimize
        # Utiliser ep_start_optim qui est déjà un ndarray numpy
        result = minimize(scipy_obj_grad_wrapper,
                          ep_start_optim, # Doit être un ndarray numpy float64
                          args=static_args_for_jax,
                          method='L-BFGS-B',
                          jac=True, # Indique qu'on fournit valeur ET gradient
                          bounds=lbfgsb_bounds,
                          options=options)

        logs.append(f"{log_prefix}L-BFGS-B (JAX grad) fini en {time.time() - opt_start_time:.3f}s.")

        # Analyser le résultat
        final_cost = result.fun if np.isfinite(result.fun) else np.inf
        # Gérer message en bytes ou str
        result_message_str = result.message.decode('utf-8') if isinstance(result.message, bytes) else str(result.message)
        nit_total = result.nit if hasattr(result, 'nit') else 0
        nfev_total = result.nfev if hasattr(result, 'nfev') else 0

        # Considérer succès si Scipy dit succès OU si la limite d'itérations/évaluations est atteinte (status=1) ET que le coût est fini
        is_success_or_limit = (result.success or result.status == 1) and np.isfinite(final_cost)

        if is_success_or_limit:
            # Appliquer les bornes au résultat final (L-BFGS-B peut légèrement les dépasser)
            final_ep_raw = result.x
            final_ep = np.maximum(final_ep_raw, min_thickness_phys)
            optim_success = True
            log_status = "succès" if result.success else "limite atteinte"
            logs.append(f"{log_prefix}Optimisation terminée ({log_status}). Coût final: {final_cost:.3e}, Itérations: {nit_total}, Evals: {nfev_total}, Msg: {result_message_str}")
        else:
            # Échec de l'optimisation
            optim_success = False
            final_ep = np.maximum(ep_start_optim, min_thickness_phys) # Revenir au point de départ (clampé)
            logs.append(f"{log_prefix}Optimisation ÉCHOUÉE. Status: {result.status}, Msg: {result_message_str}, Coût: {final_cost:.3e}")
            # Essayer de recalculer le coût du point de départ pour info
            try:
                reverted_cost, _ = scipy_obj_grad_wrapper(final_ep, *static_args_for_jax)
                logs.append(f"{log_prefix}Retour à la structure initiale (clampée). Coût recalculé: {reverted_cost:.3e}")
                final_cost = reverted_cost if np.isfinite(reverted_cost) else np.inf
            except Exception as cost_e:
                logs.append(f"{log_prefix}Retour à la structure initiale (clampée). ERREUR recalcul coût: {cost_e}")
                final_cost = np.inf

    except Exception as e_optim:
        logs.append(f"{log_prefix}ERREUR majeure durant l'optimisation JAX/Scipy: {e_optim}\n{traceback.format_exc(limit=2)}")
        st.error(f"Erreur critique pendant l'optimisation: {e_optim}")
        # Revenir à l'état initial si possible, sinon None
        final_ep = np.maximum(ep_start_optim, min_thickness_phys) if ep_start_optim is not None else None
        optim_success = False
        final_cost = np.inf
        result_message_str = f"Exception: {e_optim}"
        nit_total = 0
        nfev_total = 0

    # Retourner le vecteur ep final (ou None si échec total), succès, coût, logs, message, it/fev
    return final_ep, optim_success, final_cost, logs, result_message_str, nit_total, nfev_total

# -*- coding: utf-8 -*-
# =============================================
# app_streamlit.py (Suite)
# (Assurez-vous d'avoir inclus les imports, constantes, fonctions de calcul,
# et l'initialisation de st.session_state DÉFINIS PRÉCÉDEMMENT)
# =============================================

# --- Algorithmes d'Optimisation (Suite) ---

def _perform_layer_merge_or_removal_only(ep_vector_in: np.ndarray, min_thickness_phys: float,
                                         log_prefix: str = "", target_layer_index: Optional[int] = None,
                                         threshold_for_removal: Optional[float] = None) -> Tuple[Optional[np.ndarray], bool, List[str]]:
    """Tente de supprimer/fusionner la couche la plus fine (ou une cible) respectant les critères."""
    current_ep = ep_vector_in.copy()
    logs = []
    num_layers = len(current_ep)
    structure_changed = False
    ep_after_merge = None # Initialiser

    # Conditions initiales pour pouvoir fusionner/supprimer
    if num_layers <= 2 and target_layer_index is None:
        logs.append(f"{log_prefix}Structure <= 2 couches. Suppression/fusion non possible sans cible.")
        return current_ep, False, logs
    elif num_layers < 1:
        logs.append(f"{log_prefix}Structure vide.")
        return current_ep, False, logs

    try:
        thin_layer_index = -1
        min_thickness_found = np.inf

        # --- Étape 1: Identifier la couche candidate ---
        if target_layer_index is not None:
            # Logique si une couche est explicitement ciblée (moins courant dans le flux auto)
            if 0 <= target_layer_index < num_layers and current_ep[target_layer_index] >= min_thickness_phys:
                thin_layer_index = target_layer_index
                min_thickness_found = current_ep[target_layer_index]
                logs.append(f"{log_prefix}Ciblage manuel couche {thin_layer_index + 1} ({min_thickness_found:.3f} nm).")
            else:
                logs.append(f"{log_prefix}Cible manuelle {target_layer_index+1} invalide/trop fine. Recherche auto.")
                target_layer_index = None # Forcer recherche auto

        if target_layer_index is None:
            # Recherche automatique de la couche la plus fine éligible
            candidate_indices = np.where(current_ep >= min_thickness_phys)[0]

            if candidate_indices.size == 0:
                logs.append(f"{log_prefix}Aucune couche >= {min_thickness_phys:.3f} nm trouvée.")
                return current_ep, False, logs # Pas de candidat

            candidate_thicknesses = current_ep[candidate_indices]

            # Filtrer par seuil si fourni (pour le mode auto)
            indices_to_consider = candidate_indices
            thicknesses_to_consider = candidate_thicknesses
            if threshold_for_removal is not None:
                mask_below_threshold = thicknesses_to_consider < threshold_for_removal
                if np.any(mask_below_threshold):
                    indices_to_consider = indices_to_consider[mask_below_threshold]
                    thicknesses_to_consider = thicknesses_to_consider[mask_below_threshold]
                    logs.append(f"{log_prefix}Recherche parmi couches < {threshold_for_removal:.3f} nm.")
                else:
                    logs.append(f"{log_prefix}Aucune couche éligible (< {threshold_for_removal:.3f} nm) trouvée.")
                    return current_ep, False, logs # Pas de candidat sous le seuil

            # Trouver la plus fine parmi les candidats restants
            if indices_to_consider.size > 0:
                min_idx_local = np.argmin(thicknesses_to_consider)
                thin_layer_index = indices_to_consider[min_idx_local]
                min_thickness_found = thicknesses_to_consider[min_idx_local]
            else:
                # Devrait seulement arriver si seuil appliqué et aucune couche trouvée dessous
                logs.append(f"{log_prefix}Aucune couche candidate finale trouvée.")
                return current_ep, False, logs

        # Si on n'a toujours pas trouvé de couche (devrait être impossible si on est arrivé ici)
        if thin_layer_index == -1:
            logs.append(f"{log_prefix}Échec identification couche (cas inattendu).")
            return current_ep, False, logs

        # --- Étape 2: Effectuer la fusion/suppression ---
        thin_layer_thickness = current_ep[thin_layer_index]
        logs.append(f"{log_prefix}Couche identifiée pour action: Index {thin_layer_index} (Couche {thin_layer_index + 1}), épaisseur {thin_layer_thickness:.3f} nm.")

        # Logique de fusion/suppression basée sur la position de la couche fine
        if num_layers <= 2: # Sécurité, devrait être impossible ici
             logs.append(f"{log_prefix}Erreur logique: tentative de fusion sur <= 2 couches.")
             return current_ep, False, logs

        # Cas 1: Couche fine est la première (index 0)
        elif thin_layer_index == 0:
            # Supprimer les 2 premières couches (la fine et sa voisine)
            ep_after_merge = current_ep[2:]
            merged_info = f"Suppression des 2 premières couches."
            structure_changed = True

        # Cas 2: Couche fine est la dernière (index num_layers - 1)
        elif thin_layer_index == num_layers - 1:
             # Supprimer les 2 dernières couches (la fine et sa voisine)
             # ATTENTION: Logique originale semblait différente. Vérifier l'intention.
             # Si l'intention est de supprimer juste la dernière: ep_after_merge = current_ep[:-1]
             # Si l'intention est de fusionner avec l'avant-dernière (si même matériau, ce qui n'est pas vérifié ici)
             # ou supprimer les deux dernières (si matériaux différents)?
             # On part sur la suppression des 2 dernières pour être cohérent avec le cas index 0.
             if num_layers >= 2: # Nécessaire pour qu'il y ait une avant-dernière
                 ep_after_merge = current_ep[:-2]
                 merged_info = f"Suppression des 2 dernières couches."
                 structure_changed = True
             else: # Ne devrait pas arriver si num_layers > 2
                 logs.append(f"{log_prefix}Cas spécial: impossible supprimer 2 dernières couches (num_layers={num_layers}).")
                 return current_ep, False, logs

        # Cas 3: Couche fine est interne (index > 0 et < num_layers - 1)
        else:
            # Fusionner les épaisseurs des couches N-1 et N+1, supprimer N-1, N, N+1
            merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
            # Construire le nouveau vecteur ep
            ep_before = current_ep[:thin_layer_index - 1]
            ep_after = current_ep[thin_layer_index + 2:]
            ep_after_merge = np.concatenate((ep_before, [merged_thickness], ep_after))
            merged_info = f"Fusion des couches {thin_layer_index} et {thin_layer_index + 2} autour de la couche {thin_layer_index + 1} supprimée -> nouvelle épaisseur {merged_thickness:.3f} nm."
            structure_changed = True

        # --- Étape 3: Finalisation ---
        if structure_changed and ep_after_merge is not None:
            logs.append(f"{log_prefix}{merged_info} Nouvelle structure: {len(ep_after_merge)} couches.")
            # Assurer que les épaisseurs fusionnées respectent le minimum physique
            ep_after_merge = np.maximum(ep_after_merge, min_thickness_phys)
            return ep_after_merge, True, logs
        elif structure_changed and ep_after_merge is None:
             logs.append(f"{log_prefix}Erreur logique: structure_changed=True mais ep_after_merge=None.")
             return current_ep, False, logs # Retourner l'original en cas d'erreur
        else:
             # Cas où aucune action n'a été prise (normalement géré plus tôt)
             logs.append(f"{log_prefix}Aucune modification de structure effectuée.")
             return current_ep, False, logs

    except Exception as e_merge:
        logs.append(f"{log_prefix}ERREUR durant la logique de fusion/suppression: {e_merge}\n{traceback.format_exc(limit=1)}")
        st.error(f"Erreur interne lors de la suppression/fusion de couche: {e_merge}")
        return current_ep, False, logs # Retourner l'original

def _perform_needle_insertion_scan(ep_vector_in: np.ndarray,
                                   nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
                                   l_vec_optim_np: np.ndarray, active_targets: List[Dict],
                                   cost_function_jax: Callable, # Fonction coût JITée (e.g., calculate_mse_for_optimization_penalized_jax)
                                   min_thickness_phys: float, base_needle_thickness_nm: float,
                                   scan_step: float, l0_repr: float, # l0 pour déterminer le type de needle (H ou L)
                                   excel_file_path: str, log_prefix: str = ""
                                   ) -> Tuple[Optional[np.ndarray], float, List[str], int]:
    """Scanne les positions d'insertion pour une aiguille et retourne la meilleure structure trouvée."""
    logs = []
    num_layers_in = len(ep_vector_in)

    if num_layers_in == 0:
        logs.append(f"{log_prefix}Scan aiguille impossible sur structure vide.")
        return None, np.inf, logs, -1 # Pas de meilleur EP, coût infini, index -1

    logs.append(f"{log_prefix}Démarrage scan aiguille ({num_layers_in} couches). Pas: {scan_step} nm, ép. aiguille: {base_needle_thickness_nm:.3f} nm.")

    try:
        # Préparer les données JAX nécessaires (indices, cibles) une seule fois
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        nH_arr_optim, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, excel_file_path)
        logs.extend(logs_h)
        nL_arr_optim, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, excel_file_path)
        logs.extend(logs_l)
        nSub_arr_optim, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, excel_file_path)
        logs.extend(logs_sub)

        if nH_arr_optim is None or nL_arr_optim is None or nSub_arr_optim is None:
             raise RuntimeError("Échec chargement indices pour scan aiguille.")

        active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)

        static_args_cost_fn = (
            nH_arr_optim, nL_arr_optim, nSub_arr_optim,
            l_vec_optim_jax, active_targets_tuple,
            min_thickness_phys
        )

        # Calculer le coût initial
        initial_cost_jax = cost_function_jax(jnp.asarray(ep_vector_in), *static_args_cost_fn)
        initial_cost = float(np.array(initial_cost_jax))

        if not np.isfinite(initial_cost):
            logs.append(f"{log_prefix} ERREUR: Coût initial non fini ({initial_cost}). Scan annulé.")
            st.error("Erreur Scan Aiguille: Le coût de la structure de départ n'est pas fini.")
            return None, np.inf, logs, -1

        logs.append(f"{log_prefix} Coût initial: {initial_cost:.6e}")

    except Exception as e_prep:
        logs.append(f"{log_prefix} ERREUR préparation scan aiguille: {e_prep}")
        st.error(f"Erreur préparation scan aiguille: {e_prep}")
        return None, np.inf, logs, -1

    # Initialisation des meilleurs résultats trouvés
    best_ep_found = None
    min_cost_found = initial_cost
    best_insertion_idx = -1 # Index de la couche *dans laquelle* on insère

    tested_insertions = 0
    ep_cumsum = np.cumsum(ep_vector_in)
    total_thickness = ep_cumsum[-1] if num_layers_in > 0 else 0.0

    # Itérer sur les positions d'insertion potentielles (z)
    for z in np.arange(scan_step, total_thickness, scan_step):
        # Trouver dans quelle couche (index i) se trouve la position z
        current_layer_idx = -1
        layer_start_z = 0.0
        for i in range(num_layers_in):
            layer_end_z = ep_cumsum[i]
            if z > layer_start_z and z <= layer_end_z:
                # Vérifier si la découpe laisse des épaisseurs suffisantes
                t_part1 = z - layer_start_z
                t_part2 = layer_end_z - z
                if t_part1 >= min_thickness_phys and t_part2 >= min_thickness_phys:
                    current_layer_idx = i
                else:
                    # Pas assez d'épaisseur pour couper ici
                    current_layer_idx = -2 # Marqueur pour skipper
                break # Sortir de la boucle interne une fois la couche trouvée
            layer_start_z = layer_end_z

        # Si la couche n'a pas été trouvée ou n'était pas assez épaisse
        if current_layer_idx < 0:
            continue # Passer à la position z suivante

        tested_insertions += 1

        # Construire la structure temporaire avec l'aiguille insérée
        t_layer_split_1 = z - (ep_cumsum[current_layer_idx-1] if current_layer_idx > 0 else 0.0)
        t_layer_split_2 = ep_cumsum[current_layer_idx] - z

        # Créer le nouveau vecteur d'épaisseurs
        ep_temp_np = np.concatenate((
            ep_vector_in[:current_layer_idx],             # Couches avant
            [t_layer_split_1, base_needle_thickness_nm, t_layer_split_2], # Couche coupée + aiguille
            ep_vector_in[current_layer_idx+1:]            # Couches après
        ))
        # S'assurer que toutes les épaisseurs sont >= min_thickness_phys
        # Note: On ne clampe pas à 0 ici, car t_part1/t_part2 sont déjà vérifiés >= min_thickness
        ep_temp_np_clamped = np.maximum(ep_temp_np, min_thickness_phys)


        # Calculer le coût de cette structure temporaire
        try:
            current_cost_jax = cost_function_jax(jnp.asarray(ep_temp_np_clamped), *static_args_cost_fn)
            current_cost = float(np.array(current_cost_jax))

            # Si le coût est meilleur, mettre à jour le meilleur résultat
            if np.isfinite(current_cost) and current_cost < min_cost_found:
                min_cost_found = current_cost
                best_ep_found = ep_temp_np_clamped.copy() # Sauvegarder la structure clampée
                best_insertion_idx = current_layer_idx
                # logs.append(f"{log_prefix}   Nouveau meilleur coût {min_cost_found:.6e} trouvé à z={z:.2f} (dans couche {best_insertion_idx+1})") # Optionnel: log très verbeux

        except Exception as e_cost:
            logs.append(f"{log_prefix} AVERTISSEMENT: Échec calcul coût pour z={z:.2f}. {e_cost}")
            continue # Ignorer ce point et passer au suivant

    # Fin de la boucle de scan
    if best_ep_found is not None:
        improvement = initial_cost - min_cost_found
        logs.append(f"{log_prefix} Scan terminé. {tested_insertions} points testés.")
        logs.append(f"{log_prefix} Meilleure amélioration trouvée: {improvement:.6e} (MSE {min_cost_found:.6e})")
        logs.append(f"{log_prefix} Insertion optimale dans couche originale {best_insertion_idx + 1}.")
        return best_ep_found, min_cost_found, logs, best_insertion_idx
    else:
        logs.append(f"{log_prefix} Scan terminé. {tested_insertions} points testés. Aucune amélioration trouvée.")
        return None, initial_cost, logs, -1


def _run_needle_iterations(ep_start: np.ndarray, num_needles: int,
                           validated_inputs: Dict, active_targets: List[Dict],
                           min_thickness_phys: float, l_vec_optim_np: np.ndarray,
                           scan_step_nm: float, base_needle_thickness_nm: float,
                           excel_file_path: str, log_prefix: str = ""
                           ) -> Tuple[np.ndarray, float, List[str], int, int, int]:
    """Exécute plusieurs itérations d'insertion d'aiguille + ré-optimisation."""
    logs = []
    best_ep_overall = np.asarray(ep_start).copy()
    best_mse_overall = np.inf
    total_nit_needles = 0
    total_nfev_needles = 0
    successful_reopts_count = 0

    # Obtenir matériaux & l0
    nH_material = validated_inputs['nH_material']
    nL_material = validated_inputs['nL_material']
    nSub_material = validated_inputs['nSub_material']
    l0_repr = validated_inputs.get('l0', 500.0) # Pour déterminer le type d'aiguille

    # Fonction coût JITée (pour scan et évaluation initiale)
    cost_fn_penalized_jit = jax.jit(calculate_mse_for_optimization_penalized_jax)

    # Calculer MSE initial
    try:
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        nH_arr, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, excel_file_path)
        logs.extend(logs_h)
        nL_arr, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, excel_file_path)
        logs.extend(logs_l)
        nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, excel_file_path)
        logs.extend(logs_sub)

        if nH_arr is None or nL_arr is None or nSub_arr is None:
            raise RuntimeError("Échec chargement indices pour itérations aiguilles.")

        active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
        static_args_cost_fn = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, min_thickness_phys)

        initial_cost_jax = cost_fn_penalized_jit(jnp.asarray(best_ep_overall), *static_args_cost_fn)
        best_mse_overall = float(np.array(initial_cost_jax))

        if not np.isfinite(best_mse_overall):
             raise ValueError("MSE initial pour itérations aiguilles non fini.")
        logs.append(f"{log_prefix} Démarrage itérations aiguilles ({num_needles} max). MSE initial: {best_mse_overall:.6e}")

    except Exception as e_init:
        logs.append(f"{log_prefix} ERREUR calcul MSE initial pour itérations aiguilles: {e_init}")
        st.error(f"Erreur initialisation itérations aiguilles: {e_init}")
        # Retourner l'état de départ en cas d'échec initialisation
        return ep_start, np.inf, logs, 0, 0, 0

    # Boucle principale des itérations aiguilles
    for i in range(num_needles):
        logs.append(f"{log_prefix} --- Itération Aiguille {i + 1}/{num_needles} ---")
        current_ep_iter = best_ep_overall.copy()
        num_layers_current = len(current_ep_iter)

        if num_layers_current == 0:
            logs.append(f"{log_prefix} Structure vide, arrêt itérations aiguilles."); break

        # Étape 1: Scan pour trouver la meilleure position d'insertion
        st.write(f"{log_prefix} Scan aiguille {i+1}...") # Feedback UI
        ep_after_scan, cost_after_scan, scan_logs, inserted_idx = _perform_needle_insertion_scan(
            current_ep_iter,
            nH_material, nL_material, nSub_material,
            l_vec_optim_np, active_targets,
            cost_fn_penalized_jit, # Passer la fonction coût JITée
            min_thickness_phys, base_needle_thickness_nm, scan_step_nm, l0_repr,
            excel_file_path, log_prefix=f"{log_prefix}  [Scan {i+1}] "
        )
        logs.extend(scan_logs)

        # Si le scan n'a rien trouvé de mieux, on arrête
        if ep_after_scan is None:
            logs.append(f"{log_prefix} Scan aiguille {i + 1} n'a pas trouvé d'amélioration. Arrêt des itérations aiguilles."); break

        # Étape 2: Ré-optimisation de la structure après insertion
        logs.append(f"{log_prefix} Scan {i + 1} a trouvé amélioration potentielle. Ré-optimisation...")
        st.write(f"{log_prefix} Ré-optimisation après aiguille {i+1}...") # Feedback UI

        # Lancer l'optimisation principale sur la nouvelle structure
        ep_after_reopt, optim_success, final_cost_reopt, optim_logs, optim_status_msg, nit_reopt, nfev_reopt = \
            _run_core_optimization(ep_after_scan, validated_inputs, active_targets,
                                   min_thickness_phys, log_prefix=f"{log_prefix}  [Re-Opt {i+1}] ")
        logs.extend(optim_logs)

        if not optim_success:
            logs.append(f"{log_prefix} Ré-optimisation après scan {i + 1} ÉCHOUÉE. Arrêt des itérations aiguilles."); break

        # Ré-optimisation réussie
        logs.append(f"{log_prefix} Ré-optimisation {i + 1} réussie. Nouveau MSE: {final_cost_reopt:.6e}. (Iter/Eval: {nit_reopt}/{nfev_reopt})")
        total_nit_needles += nit_reopt
        total_nfev_needles += nfev_reopt
        successful_reopts_count += 1

        # Comparer avec le meilleur MSE global trouvé jusqu'ici
        if final_cost_reopt < best_mse_overall - MSE_IMPROVEMENT_TOLERANCE:
            logs.append(f"{log_prefix}  MSE amélioré par rapport au meilleur précédent ({best_mse_overall:.6e}). Mise à jour.")
            best_ep_overall = ep_after_reopt.copy()
            best_mse_overall = final_cost_reopt
        else:
            logs.append(f"{log_prefix}  Nouveau MSE ({final_cost_reopt:.6e}) pas significativement meilleur que le précédent ({best_mse_overall:.6e}). Arrêt des itérations aiguilles.")
            # On garde quand même le résultat de cette dernière optimisation réussie
            best_ep_overall = ep_after_reopt.copy()
            best_mse_overall = final_cost_reopt
            break # Arrêter car pas d'amélioration significative

    # Fin de la boucle d'itérations
    logs.append(f"{log_prefix} Fin itérations aiguilles. Meilleur MSE final: {best_mse_overall:.6e}")
    logs.append(f"{log_prefix} Total Iter/Eval durant {successful_reopts_count} ré-optimisations réussies: {total_nit_needles}/{total_nfev_needles}")

    return best_ep_overall, best_mse_overall, logs, total_nit_needles, total_nfev_needles, successful_reopts_count

# --- Mode Auto ---
def run_auto_mode(initial_ep: Optional[np.ndarray], # Peut être None si on part du nominal
                  validated_inputs: Dict, active_targets: List[Dict],
                  excel_file_path: str, log_callback: Callable):
    """Exécute le mode automatique: Needle -> Thin Removal -> Optimize cycles."""
    logs = []
    start_time_auto = time.time()
    log_callback("#"*10 + f" Démarrage Mode Auto (Max {AUTO_MAX_CYCLES} Cycles) " + "#"*10)

    best_ep_so_far = None
    best_mse_so_far = np.inf
    num_cycles_done = 0
    termination_reason = f"Max {AUTO_MAX_CYCLES} cycles atteints"
    threshold_for_thin_removal = validated_inputs.get('auto_thin_threshold', 1.0)
    log_callback(f"  Seuil suppression auto: {threshold_for_thin_removal:.3f} nm")

    # Statistiques globales du mode auto
    total_iters_auto = 0
    total_evals_auto = 0
    optim_runs_auto = 0 # Compte chaque appel réussi à _run_core_optimization

    try:
        # --- Étape 0: Obtenir la structure de départ et le MSE initial ---
        current_ep = None
        if initial_ep is not None:
             log_callback("  Mode Auto: Utilisation de la structure optimisée précédente.")
             current_ep = initial_ep.copy()
             # Calculer le MSE de départ
             l_min_optim, l_max_optim = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
             l_step_optim = validated_inputs['l_step']
             num_pts = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
             l_vec_optim_np = np.geomspace(l_min_optim, l_max_optim, num_pts)
             l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
             if not l_vec_optim_np.size: raise ValueError("Échec génération lambda pour calcul MSE initial auto.")

             l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
             nH_arr, log_h = _get_nk_array_for_lambda_vec(validated_inputs['nH_material'], l_vec_optim_jax, excel_file_path)
             nL_arr, log_l = _get_nk_array_for_lambda_vec(validated_inputs['nL_material'], l_vec_optim_jax, excel_file_path)
             nSub_arr, log_sub = _get_nk_array_for_lambda_vec(validated_inputs['nSub_material'], l_vec_optim_jax, excel_file_path)
             log_callback(log_h); log_callback(log_l); log_callback(log_sub)
             if nH_arr is None or nL_arr is None or nSub_arr is None: raise RuntimeError("Échec chargement indices pour MSE initial auto.")

             active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
             static_args = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, MIN_THICKNESS_PHYS_NM)
             cost_fn_jit = jax.jit(calculate_mse_for_optimization_penalized_jax)
             initial_mse_jax = cost_fn_jit(jnp.asarray(current_ep), *static_args)
             initial_mse = float(np.array(initial_mse_jax))
             if not np.isfinite(initial_mse): raise ValueError("MSE initial (depuis état optimisé) non fini.")
             best_mse_so_far = initial_mse
             best_ep_so_far = current_ep.copy()
             log_callback(f"  MSE initial (depuis état optimisé): {best_mse_so_far:.6e}")

        else:
            # Partir de la structure nominale + optimisation initiale
            log_callback("  Mode Auto: Utilisation de la structure nominale (QWOT).")
            emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
            if not emp_list: raise ValueError("QWOT nominal vide.")

            ep_nominal, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'],
                                                             validated_inputs['nH_material'], validated_inputs['nL_material'],
                                                             excel_file_path)
            log_callback(logs_ep_init)
            if ep_nominal is None: raise RuntimeError("Échec calcul épaisseurs nominales initiales.")

            log_callback(f"  Structure nominale: {len(ep_nominal)} couches. Lancement optimisation initiale...")
            st.info("Mode Auto : Optimisation initiale de la structure nominale...") # Feedback UI

            ep_after_initial_opt, initial_opt_success, initial_mse, initial_opt_logs, initial_opt_msg, initial_nit, initial_nfev = \
                _run_core_optimization(ep_nominal, validated_inputs, active_targets,
                                       MIN_THICKNESS_PHYS_NM, log_prefix="  [Auto Init Opt] ")
            log_callback(initial_opt_logs)

            if not initial_opt_success:
                 log_callback(f"ERREUR: Échec optimisation initiale en Mode Auto ({initial_opt_msg}). Annulation.")
                 st.error(f"Échec de l'optimisation initiale du Mode Auto: {initial_opt_msg}")
                 return None, np.inf, logs, 0, 0 # Échec

            log_callback(f"  Optimisation initiale terminée. MSE: {initial_mse:.6e} (Iter/Eval: {initial_nit}/{initial_nfev})")
            best_ep_so_far = ep_after_initial_opt.copy()
            best_mse_so_far = initial_mse
            total_iters_auto += initial_nit; total_evals_auto += initial_nfev; optim_runs_auto += 1


        # --- Étape 1: Boucle des Cycles Auto ---
        if best_ep_so_far is None or not np.isfinite(best_mse_so_far):
             raise RuntimeError("État de départ invalide pour les cycles Auto.")

        log_callback(f"--- Démarrage des Cycles Auto (MSE départ: {best_mse_so_far:.6e}, {len(best_ep_so_far)} couches) ---")

        for cycle_num in range(AUTO_MAX_CYCLES):
            log_callback(f"\n--- Cycle Auto {cycle_num + 1} / {AUTO_MAX_CYCLES} ---")
            st.info(f"Cycle Auto {cycle_num + 1}/{AUTO_MAX_CYCLES} | MSE actuel: {best_mse_so_far:.3e}") # Feedback UI

            mse_at_cycle_start = best_mse_so_far
            ep_at_cycle_start = best_ep_so_far.copy()
            cycle_improved_globally = False # Flag pour savoir si ce cycle a amélioré le MSE global

            # --- 1a. Phase Aiguille (Needle) ---
            log_callback(f"  [Cycle {cycle_num+1}] Phase Aiguille ({AUTO_NEEDLES_PER_CYCLE} itérations max)...")
            st.write(f"Cycle {cycle_num + 1}: Phase Aiguille...") # Feedback plus fin

            # Préparer le vecteur lambda pour les itérations aiguilles (peut être le même que l'optim)
            l_min_optim, l_max_optim = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
            l_step_optim = validated_inputs['l_step']
            num_pts = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
            l_vec_optim_np_needle = np.geomspace(l_min_optim, l_max_optim, num_pts)
            l_vec_optim_np_needle = l_vec_optim_np_needle[(l_vec_optim_np_needle > 0) & np.isfinite(l_vec_optim_np_needle)]
            if not l_vec_optim_np_needle.size:
                 log_callback("  ERREUR: impossible de générer lambda pour phase aiguille. Cycle annulé.")
                 break

            ep_after_needles, mse_after_needles, needle_logs, nit_needles, nfev_needles, reopts_in_needles = \
                _run_needle_iterations(best_ep_so_far, AUTO_NEEDLES_PER_CYCLE, validated_inputs, active_targets,
                                       MIN_THICKNESS_PHYS_NM, l_vec_optim_np_needle,
                                       DEFAULT_NEEDLE_SCAN_STEP_NM, BASE_NEEDLE_THICKNESS_NM,
                                       excel_file_path, log_prefix=f"    [Needle {cycle_num+1}] ")
            log_callback(needle_logs)
            log_callback(f"  [Cycle {cycle_num+1}] Fin Phase Aiguille. MSE: {mse_after_needles:.6e} (Iter/Eval: {nit_needles}/{nfev_needles})")
            total_iters_auto += nit_needles; total_evals_auto += nfev_needles; optim_runs_auto += reopts_in_needles

            # Mettre à jour le meilleur état si la phase aiguille a amélioré
            if mse_after_needles < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                log_callback(f"    Amélioration globale par phase aiguille (vs {best_mse_so_far:.6e}).")
                best_ep_so_far = ep_after_needles.copy()
                best_mse_so_far = mse_after_needles
                cycle_improved_globally = True
            else:
                 # Même si pas d'amélioration globale, on continue le cycle avec ce résultat
                 log_callback(f"    Pas d'amélioration globale par phase aiguille (vs {best_mse_so_far:.6e}).")
                 best_ep_so_far = ep_after_needles.copy()
                 best_mse_so_far = mse_after_needles # Mettre à jour le MSE courant pour la phase suivante

            # --- 1b. Phase Suppression Couches Fines (Thin Removal) ---
            log_callback(f"  [Cycle {cycle_num+1}] Phase Suppression (< {threshold_for_thin_removal:.3f} nm) + Re-Opt...")
            st.write(f"Cycle {cycle_num + 1}: Phase Suppression...") # Feedback UI

            layers_removed_this_cycle = 0
            # Boucle de suppression: on essaie de supprimer tant qu'on trouve des couches fines
            # Limiter le nombre d'itérations pour éviter boucle infinie si qqch se passe mal
            max_thinning_attempts = len(best_ep_so_far) + 2
            for attempt in range(max_thinning_attempts):
                current_num_layers_thin = len(best_ep_so_far)
                if current_num_layers_thin <= 2:
                    log_callback("    Structure trop petite (< 3 couches), arrêt suppression.")
                    break # Sortir de la boucle de suppression

                # Essayer de supprimer UNE couche fine (celle < threshold OU la plus fine si threshold=None)
                ep_after_single_removal, structure_changed, removal_logs = \
                    _perform_layer_merge_or_removal_only(best_ep_so_far, MIN_THICKNESS_PHYS_NM,
                                                        log_prefix=f"    [Thin {cycle_num+1}.{attempt+1}] ",
                                                        threshold_for_removal=threshold_for_thin_removal)
                log_callback(removal_logs)

                if structure_changed and ep_after_single_removal is not None:
                    layers_removed_this_cycle += 1
                    log_callback(f"    Couche supprimée/fusionnée ({layers_removed_this_cycle} dans ce cycle). Ré-optimisation ({len(ep_after_single_removal)} couches)...")
                    st.write(f"Cycle {cycle_num + 1}: Ré-opt après suppression {layers_removed_this_cycle}...") # Feedback

                    # Ré-optimiser la structure réduite
                    ep_after_thin_reopt, thin_reopt_success, mse_after_thin_reopt, thin_reopt_logs, thin_reopt_msg, nit_thin_reopt, nfev_thin_reopt = \
                        _run_core_optimization(ep_after_single_removal, validated_inputs, active_targets,
                                               MIN_THICKNESS_PHYS_NM, log_prefix=f"      [ReOptThin {cycle_num+1}.{attempt+1}] ")
                    log_callback(thin_reopt_logs)
                    total_iters_auto += nit_thin_reopt; total_evals_auto += nfev_thin_reopt

                    if thin_reopt_success:
                        optim_runs_auto += 1 # Compter l'optim réussie
                        log_callback(f"      Ré-optimisation réussie. MSE: {mse_after_thin_reopt:.6e} (Iter/Eval: {nit_thin_reopt}/{nfev_thin_reopt})")
                        # Mettre à jour le meilleur état si amélioration globale
                        if mse_after_thin_reopt < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                            log_callback(f"      Amélioration globale par suppression+reopt (vs {best_mse_so_far:.6e}).")
                            best_ep_so_far = ep_after_thin_reopt.copy()
                            best_mse_so_far = mse_after_thin_reopt
                            cycle_improved_globally = True
                        else:
                            log_callback(f"      Pas d'amélioration globale (vs {best_mse_so_far:.6e}). On continue avec ce résultat.")
                            best_ep_so_far = ep_after_thin_reopt.copy()
                            best_mse_so_far = mse_after_thin_reopt
                        # Continuer la boucle de suppression pour voir si d'autres couches sont fines

                    else:
                        # La ré-optimisation a échoué, on arrête la phase de suppression pour ce cycle
                        log_callback(f"    AVERTISSEMENT: Ré-optimisation après suppression ÉCHOUÉE ({thin_reopt_msg}). Arrêt suppression pour ce cycle.")
                        # Revenir à l'état *avant* cette tentative de suppression échouée
                        best_ep_so_far = ep_after_single_removal.copy() # Garder la structure réduite mais non optimisée
                        # Recalculer le MSE de cet état non optimisé
                        try:
                             current_mse_jax = cost_fn_penalized_jit(jnp.asarray(best_ep_so_far), *static_args_cost_fn)
                             best_mse_so_far = float(np.array(current_mse_jax))
                             if not np.isfinite(best_mse_so_far): best_mse_so_far = np.inf
                             log_callback(f"      MSE après échec re-opt (structure réduite non opt): {best_mse_so_far:.6e}")
                        except Exception as e_cost_fail:
                             log_callback(f"      ERREUR recalcul MSE après échec re-opt: {e_cost_fail}")
                             best_mse_so_far = np.inf
                        break # Sortir de la boucle de suppression

                else:
                    # Aucune couche n'a été supprimée (soit pas assez fine, soit erreur)
                    log_callback("    Aucune autre couche à supprimer/fusionner dans cette phase.")
                    break # Sortir de la boucle de suppression

            log_callback(f"  [Cycle {cycle_num+1}] Fin Phase Suppression. {layers_removed_this_cycle} couche(s) supprimée(s).")

            # --- Fin du Cycle ---
            num_cycles_done += 1
            log_callback(f"--- Fin Cycle Auto {cycle_num + 1} --- Meilleur MSE actuel: {best_mse_so_far:.6e} ({len(best_ep_so_far)} couches) ---")

            # Condition d'arrêt prématuré: si le cycle n'a pas amélioré le MSE par rapport au début DU CYCLE
            if not cycle_improved_globally and best_mse_so_far >= mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE:
                 log_callback(f"Aucune amélioration significative dans Cycle {cycle_num + 1} (Début: {mse_at_cycle_start:.6e}, Fin: {best_mse_so_far:.6e}). Arrêt Mode Auto.")
                 termination_reason = f"Pas d'amélioration (Cycle {cycle_num + 1})"
                 # Si le MSE a augmenté, revenir à l'état du début du cycle? C'est plus sûr.
                 if best_mse_so_far > mse_at_cycle_start + MSE_IMPROVEMENT_TOLERANCE:
                      log_callback("  MSE a augmenté, retour à l'état précédent le cycle.")
                      best_ep_so_far = ep_at_cycle_start.copy()
                      best_mse_so_far = mse_at_cycle_start
                 break # Sortir de la boucle des cycles

        # --- Fin de Tous les Cycles ---
        log_callback(f"\n--- Mode Auto Terminé après {num_cycles_done} cycles ---")
        log_callback(f"Raison: {termination_reason}")
        log_callback(f"Meilleur MSE final: {best_mse_so_far:.6e} avec {len(best_ep_so_far)} couches.")

        # Afficher les stats globales
        avg_nit_str = f"{total_iters_auto / optim_runs_auto:.1f}" if optim_runs_auto > 0 else "N/A"
        avg_nfev_str = f"{total_evals_auto / optim_runs_auto:.1f}" if optim_runs_auto > 0 else "N/A"
        log_callback(f"Stats Globales Auto: {optim_runs_auto} optimisations réussies, Sum Iter/Eval: {total_iters_auto}/{total_evals_auto}, Moy Iter/Eval: {avg_nit_str}/{avg_nfev_str}")

        # Retourner le meilleur résultat trouvé
        return best_ep_so_far, best_mse_so_far, logs, total_iters_auto, total_evals_auto

    except (ValueError, RuntimeError, TypeError) as e:
        log_callback(f"ERREUR fatale durant le Mode Auto (Setup/Workflow): {e}")
        st.error(f"Erreur Mode Auto: {e}")
        # traceback.print_exc()
        return None, np.inf, logs, total_iters_auto, total_evals_auto # Échec
    except Exception as e_fatal:
         log_callback(f"ERREUR inattendue fatale durant le Mode Auto: {type(e_fatal).__name__}: {e_fatal}")
         st.error(f"Erreur inattendue Mode Auto: {e_fatal}")
         traceback.print_exc() # Log complet dans la console
         return None, np.inf, logs, total_iters_auto, total_evals_auto # Échec

# --- QWOT Scan + Opt ---
# Fonctions spécifiques au scan QWOT (split stack)

@jax.jit
def calculate_M_for_thickness(thickness: jnp.ndarray, n_complex_layer: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    """Calcule la matrice pour une couche unique à une lambda."""
    eta = n_complex_layer
    safe_l_val = jnp.maximum(l_val, 1e-9)
    safe_eta = jnp.where(jnp.abs(eta) < 1e-12, 1e-12 + 0j, eta)
    phi = (2 * jnp.pi / safe_l_val) * (n_complex_layer * thickness)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    # M_layer = [[cos_phi, 1j * sin_phi / eta], [1j * eta * sin_phi, cos_phi]]
    m00 = jnp.where(thickness > 1e-12, cos_phi, 1.0)
    m01 = jnp.where(thickness > 1e-12, (1j / safe_eta) * sin_phi, 0.0)
    m10 = jnp.where(thickness > 1e-12, 1j * eta * sin_phi, 0.0)
    m11 = jnp.where(thickness > 1e-12, cos_phi, 1.0)

    M_layer = jnp.array([[m00, m01], [m10, m11]], dtype=jnp.complex128)
    return M_layer

# vmap sur lambda pour obtenir les matrices pour toutes les lambdas pour UNE épaisseur/indice
calculate_M_batch_for_thickness = vmap(calculate_M_for_thickness, in_axes=(None, None, 0))

@jax.jit
def get_layer_matrices_qwot(layer_idx: int, initial_layer_number: int, l0: float,
                            nH_c_l0: jnp.ndarray, nL_c_l0: jnp.ndarray,
                            l_vec: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Prépare les matrices pour 1 et 2 QWOT pour une couche donnée et un vecteur lambda."""
    predicate_is_H = (layer_idx % 2 == 0)
    n_real_l0 = jax.lax.select(predicate_is_H, nH_c_l0.real, nL_c_l0.real)
    n_complex_for_matrix = jax.lax.select(predicate_is_H, nH_c_l0, nL_c_l0) # Utiliser l'indice à l0 pour le calcul de matrice (approximation QWOT)

    # Calculer épaisseurs 1 et 2 QWOT
    denom = 4.0 * jnp.maximum(n_real_l0, 1e-9) # Eviter division par zero pour n_real
    safe_l0 = jnp.maximum(l0, 1e-9)
    ep1_calc = 1.0 * safe_l0 / denom
    ep2_calc = 2.0 * safe_l0 / denom

    # Épaisseur = 0 si indice invalide
    ep1 = jnp.where(n_real_l0 > 1e-9, ep1_calc, 0.0)
    ep2 = jnp.where(n_real_l0 > 1e-9, ep2_calc, 0.0)

    # Optionnel: Clamper à MIN_THICKNESS si > 0? Non, garder la valeur QWOT exacte pour le scan.
    # ep1 = jnp.maximum(ep1, MIN_THICKNESS_PHYS_NM * (ep1 > 1e-12))
    # ep2 = jnp.maximum(ep2, MIN_THICKNESS_PHYS_NM * (ep2 > 1e-12))

    # Calculer les matrices pour toutes les lambdas pour ep1 et ep2
    M_1qwot_batch = calculate_M_batch_for_thickness(ep1, n_complex_for_matrix, l_vec)
    M_2qwot_batch = calculate_M_batch_for_thickness(ep2, n_complex_for_matrix, l_vec)

    return M_1qwot_batch, M_2qwot_batch

@jax.jit
def compute_half_product(multiplier_indices: jnp.ndarray, # Indices (0 ou 1) pour une combinaison
                         layer_matrices_half: jnp.ndarray  # Matrices précalculées pour cette moitié (N_half, 2, n_lambda, 2, 2)
                         ) -> jnp.ndarray: # Produit pour cette combinaison (n_lambda, 2, 2)
    """Calcule le produit matriciel pour une moitié de la structure pour UNE combinaison."""
    N_half = layer_matrices_half.shape[0]
    L = layer_matrices_half.shape[2] # Nombre de lambdas
    # Initialiser le produit à l'identité pour chaque lambda
    init_prod = jnp.tile(jnp.eye(2, dtype=jnp.complex128), (L, 1, 1)) # Shape (n_lambda, 2, 2)

    # Fonction pour l'étape du scan: multiplier par la matrice de la couche k
    def multiply_step(carry_prod: jnp.ndarray, layer_idx: int) -> Tuple[jnp.ndarray, None]:
        multiplier_idx = multiplier_indices[layer_idx] # 0 pour 1 QWOT, 1 pour 2 QWOT
        # Sélectionner M_k pour 1 ou 2 QWOT: shape (n_lambda, 2, 2)
        M_k = layer_matrices_half[layer_idx, multiplier_idx, :, :, :]
        # Multiplier à gauche: M_k @ M_{k-1...1} pour chaque lambda
        new_prod = vmap(jnp.matmul)(M_k, carry_prod) # vmap sur la dimension lambda
        return new_prod, None

    # Scan sur les couches de cette moitié
    final_prod, _ = jax.lax.scan(multiply_step, init_prod, jnp.arange(N_half))
    return final_prod

@jax.jit
def get_T_from_batch_matrix(M_batch: jnp.ndarray, # Matrices totales (n_lambda ou n_comb, n_lambda, 2, 2)
                            nSub_arr: jnp.ndarray   # Indices substrat (n_lambda,)
                            ) -> jnp.ndarray: # Transmittances (n_lambda,) ou (n_comb, n_lambda)
    """Calcule T à partir d'un batch de matrices caractéristiques."""
    etainc = 1.0 + 0j
    etasub_batch = nSub_arr # Doit être broadcastable avec M_batch sur la dim lambda

    # Extraire les éléments de matrice (potentiellement sur dim combinaison et lambda)
    m00 = M_batch[..., 0, 0]; m01 = M_batch[..., 0, 1]
    m10 = M_batch[..., 1, 0]; m11 = M_batch[..., 1, 1]

    # Calculer T (idem que dans calculate_single_wavelength_T_core)
    rs_den = (etainc * m00 + etasub_batch * m11 + etainc * etasub_batch * m01 + m10)
    rs_den_abs = jnp.abs(rs_den)
    safe_den = jnp.where(rs_den_abs < 1e-12, 1e-12 + 0j, rs_den)
    ts = (2.0 * etainc) / safe_den
    real_etasub_batch = jnp.real(etasub_batch)
    safe_real_etainc = 1.0 # n_inc = 1
    Ts_complex = (real_etasub_batch / safe_real_etainc) * (ts * jnp.conj(ts))
    Ts = jnp.real(Ts_complex)

    # Gérer les cas limites
    return jnp.where(rs_den_abs < 1e-12, 0.0, jnp.nan_to_num(Ts, nan=0.0))

@jax.jit
def calculate_mse_basic_jax(Ts: jnp.ndarray, # Transmittances (n_lambda,)
                            l_vec: jnp.ndarray,  # Lambdas (n_lambda,)
                            targets_tuple: Tuple[Tuple[float, float, float, float], ...] # Cibles
                            ) -> jnp.ndarray: # MSE scalaire
    """Calcule le MSE simple (utilisé dans le scan QWOT)."""
    total_squared_error = 0.0
    total_points_in_targets = 0
    for i in range(len(targets_tuple)):
        l_min, l_max, t_min, t_max = targets_tuple[i]
        target_mask = (l_vec >= l_min) & (l_vec <= l_max)
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t = t_min + slope * (l_vec - l_min)
        squared_errors = (Ts - interpolated_target_t)**2
        masked_sq_error = jnp.where(target_mask, squared_errors, 0.0)
        total_squared_error += jnp.sum(masked_sq_error)
        total_points_in_targets += jnp.sum(target_mask)

    # Gérer le cas où aucun point n'est dans les cibles
    mse = jnp.where(total_points_in_targets > 0,
                    total_squared_error / total_points_in_targets,
                    jnp.inf) # MSE infini si pas de points

    return jnp.nan_to_num(mse, nan=jnp.inf, posinf=jnp.inf) # Renvoyer Inf si NaN/Inf

@jax.jit
def combine_and_calc_mse(prod1: jnp.ndarray, prod2: jnp.ndarray, # Produits partiels (n_lambda, 2, 2)
                         nSub_arr_in: jnp.ndarray,           # Indice substrat (n_lambda,)
                         l_vec_in: jnp.ndarray,              # Lambdas (n_lambda,)
                         targets_tuple_in: Tuple             # Cibles
                         ) -> jnp.ndarray: # MSE scalaire pour cette combinaison
    """Combine les produits partiels, calcule T, puis le MSE."""
    # Produit total M = M_half2 @ M_half1 pour chaque lambda
    M_total = vmap(jnp.matmul)(prod2, prod1) # vmap sur la dimension lambda
    # Calculer T pour toutes les lambdas
    Ts = get_T_from_batch_matrix(M_total, nSub_arr_in)
    # Calculer le MSE
    mse = calculate_mse_basic_jax(Ts, l_vec_in, targets_tuple_in)
    return mse

def _execute_split_stack_scan(current_l0: float, initial_layer_number: int,
                              nH_c_l0: complex, nL_c_l0: complex, # Indices complexes à l0
                              nSub_arr_scan: jnp.ndarray,       # Indice substrat sur le grid de scan
                              l_vec_eval_sparse_jax: jnp.ndarray, # Lambdas du grid de scan
                              active_targets_tuple: Tuple,
                              log_callback: Callable) -> Tuple[float, Optional[np.ndarray], List[str]]:
    """Exécute le scan QWOT (1.0/2.0) par la méthode 'split stack'."""
    logs = []
    L_sparse = len(l_vec_eval_sparse_jax)
    num_combinations = 2**initial_layer_number
    log_callback(f"  [Scan l0={current_l0:.2f}] Test {num_combinations:,} comb. QWOT (1.0/2.0)...")

    # --- 1. Précalcul des matrices pour 1 et 2 QWOT pour chaque couche ---
    precompute_start_time = time.time()
    st.write(f"Scan l0={current_l0:.1f}: Précalcul matrices...") # Feedback UI
    layer_matrices_list = []
    try:
        # Compiler la fonction de calcul des matrices QWOT
        get_layer_matrices_qwot_jit = jax.jit(get_layer_matrices_qwot)
        for i in range(initial_layer_number):
            # Obtenir les matrices (batch sur lambda) pour 1 QWOT et 2 QWOT
            m1, m2 = get_layer_matrices_qwot_jit(i, initial_layer_number, current_l0,
                                                 jnp.asarray(nH_c_l0), jnp.asarray(nL_c_l0),
                                                 l_vec_eval_sparse_jax)
            # Empiler [M_1qwot, M_2qwot] pour cette couche i
            layer_matrices_list.append(jnp.stack([m1, m2], axis=0)) # Shape (2, n_lambda, 2, 2)
        # Empiler sur toutes les couches
        all_layer_matrices = jnp.stack(layer_matrices_list, axis=0) # Shape (N, 2, n_lambda, 2, 2)
        # Forcer l'exécution JAX
        all_layer_matrices.block_until_ready()
        log_callback(f"    Précalcul matrices (l0={current_l0:.2f}) terminé en {time.time() - precompute_start_time:.3f}s.")
    except Exception as e_mat:
        logs.append(f"  ERREUR Précalcul Matrices pour l0={current_l0:.2f}: {e_mat}")
        st.error(f"Erreur précalcul matrices QWOT scan: {e_mat}")
        return np.inf, None, logs # Échec critique

    # --- 2. Calcul des produits partiels (Méthode Split Stack) ---
    N = initial_layer_number
    N1 = N // 2
    N2 = N - N1
    num_comb1 = 2**N1
    num_comb2 = 2**N2

    log_callback(f"    Calcul produits partiels 1 ({num_comb1:,} comb)...")
    st.write(f"Scan l0={current_l0:.1f}: Produits partiels 1...") # Feedback UI
    half1_start_time = time.time()
    # Indices binaires pour les multiplicateurs (0 ou 1) pour la 1ère moitié
    indices1 = jnp.arange(num_comb1)
    powers1 = 2**jnp.arange(N1)
    multiplier_indices1 = jnp.not_equal(indices1[:, None] & powers1, 0).astype(jnp.int32) # Shape (num_comb1, N1)
    matrices_half1 = all_layer_matrices[:N1] # Shape (N1, 2, n_lambda, 2, 2)

    # vmap sur les combinaisons de la 1ère moitié
    compute_half_product_jit = jax.jit(compute_half_product)
    partial_products1 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices1, matrices_half1) # Shape (num_comb1, n_lambda, 2, 2)
    partial_products1.block_until_ready()
    log_callback(f"    Produits partiels 1 terminés en {time.time() - half1_start_time:.3f}s.")

    log_callback(f"    Calcul produits partiels 2 ({num_comb2:,} comb)...")
    st.write(f"Scan l0={current_l0:.1f}: Produits partiels 2...") # Feedback UI
    half2_start_time = time.time()
    # Indices binaires pour la 2ème moitié
    indices2 = jnp.arange(num_comb2)
    powers2 = 2**jnp.arange(N2)
    multiplier_indices2 = jnp.not_equal(indices2[:, None] & powers2, 0).astype(jnp.int32) # Shape (num_comb2, N2)
    matrices_half2 = all_layer_matrices[N1:] # Shape (N2, 2, n_lambda, 2, 2)

    # vmap sur les combinaisons de la 2ème moitié
    partial_products2 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices2, matrices_half2) # Shape (num_comb2, n_lambda, 2, 2)
    partial_products2.block_until_ready()
    log_callback(f"    Produits partiels 2 terminés en {time.time() - half2_start_time:.3f}s.")

    # --- 3. Combinaison et Calcul MSE ---
    log_callback(f"    Combinaison et calcul MSE ({num_comb1 * num_comb2:,} total)...")
    st.write(f"Scan l0={current_l0:.1f}: Combinaison & MSE...") # Feedback UI
    combine_start_time = time.time()

    # Compiler la fonction qui combine et calcule le MSE
    combine_and_calc_mse_jit = jax.jit(combine_and_calc_mse)

    # Utiliser vmap imbriqué pour calculer le MSE pour chaque paire (prod1, prod2)
    # vmap interne sur prod2 pour un prod1 donné
    vmap_inner = vmap(combine_and_calc_mse_jit, in_axes=(None, 0, None, None, None))
    # vmap externe sur prod1
    vmap_outer = vmap(vmap_inner, in_axes=(0, None, None, None, None))

    # Calculer tous les MSEs
    all_mses_nested = vmap_outer(partial_products1, partial_products2, nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple) # Shape (num_comb1, num_comb2)
    all_mses_nested.block_until_ready() # Attendre la fin du calcul JAX
    log_callback(f"    Combinaison et MSE terminés en {time.time() - combine_start_time:.3f}s.")

    # --- 4. Trouver le Meilleur Résultat ---
    # Aplatir le tableau des MSEs pour trouver le minimum global
    all_mses_flat = all_mses_nested.reshape(-1)
    # Trouver l'index du MSE minimum
    best_idx_flat = jnp.argmin(all_mses_flat)
    current_best_mse = float(all_mses_flat[best_idx_flat]) # Convertir en float Python

    if not np.isfinite(current_best_mse):
        logs.append(f"    Avertissement: Aucun résultat valide (MSE fini) trouvé pour l0={current_l0:.2f}.")
        return np.inf, None, logs # Pas de résultat valide

    # Retrouver les indices des meilleures moitiés correspondant à l'index plat
    best_idx_half1, best_idx_half2 = jnp.unravel_index(best_idx_flat, (num_comb1, num_comb2))

    # Reconstruire la séquence de multiplicateurs (1.0 ou 2.0) correspondante
    best_indices_h1 = multiplier_indices1[best_idx_half1] # (N1,) d'indices 0 ou 1
    best_indices_h2 = multiplier_indices2[best_idx_half2] # (N2,) d'indices 0 ou 1
    # Convertir les indices (0/1) en multiplicateurs (1.0/2.0)
    best_multipliers_h1 = 1.0 + best_indices_h1.astype(jnp.float64)
    best_multipliers_h2 = 1.0 + best_indices_h2.astype(jnp.float64)
    # Concaténer pour obtenir la séquence complète
    current_best_multipliers = jnp.concatenate([best_multipliers_h1, best_multipliers_h2])

    logs.append(f"    Meilleur MSE pour scan l0={current_l0:.2f}: {current_best_mse:.6e}")
    # Convertir en array numpy pour le retour
    return current_best_mse, np.array(current_best_multipliers), logs

# =============================================
# SECTION 2 : FONCTIONS HELPER STREAMLIT & WRAPPERS
# =============================================

# -*- coding: utf-8 -*-
# =============================================
# app_streamlit.py (Suite et Fin - SECTION 2 et suivantes)
# ATTENTION : Ce code DOIT être précédé par la SECTION 1 (imports, fonctions, etc.)
# GÉNÉRÉ PAR IA - NÉCESSITE INTÉGRATION, VÉRIFICATION, DÉBOGAGE ET TESTS
# =============================================

# =============================================
# SECTION 2 : FONCTIONS HELPER STREAMLIT & WRAPPERS
# =============================================

def add_log(message: Union[str, List[str]]):
    """Ajoute un message (ou liste) au log dans st.session_state."""
    # Assurer que la liste existe dans l'état de session
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    # Limiter la taille du log pour éviter de saturer la mémoire/affichage
    MAX_LOG_LINES = 500
    
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    new_logs = []
    if isinstance(message, list):
        for msg in message:
            new_logs.append(f"[{timestamp}] {str(msg)}")
    else:
        new_logs.append(f"[{timestamp}] {str(message)}")
    
    # Ajouter les nouveaux logs au début (plus récent en haut)
    st.session_state.log_messages = new_logs + st.session_state.log_messages
    # Tronquer si trop long
    if len(st.session_state.log_messages) > MAX_LOG_LINES:
        st.session_state.log_messages = st.session_state.log_messages[:MAX_LOG_LINES]

def get_material_input(role: str) -> Tuple[Optional[MaterialInputType], str]:
    """Récupère la définition du matériau depuis st.session_state (H, L, ou Sub)."""
    # Détermine les clés dans st.session_state basées sur le rôle
    sel_key = f"selected_{role}" # e.g., selected_H
    # Gérer les clés spécifiques pour n'/k constant
    if role == 'H':
        const_r_key, const_i_key = "nH_r", "nH_i"
    elif role == 'L':
        const_r_key, const_i_key = "nL_r", "nL_i"
    elif role == 'Sub':
        const_r_key, const_i_key = "nSub_r", None # Pas de k pour le substrat dans l'UI originale
    else:
        st.error(f"Rôle matériau inconnu : {role}")
        return None, "Erreur Rôle"

    selection = st.session_state.get(sel_key)

    if selection == "Constant":
        # Récupérer n' et k depuis l'état, avec valeurs par défaut
        n_real = st.session_state.get(const_r_key, 1.0 if role != 'Sub' else 1.5) # Défaut 1.0 sauf pour substrat
        n_imag = 0.0
        if const_i_key and role in ['H', 'L']: # k seulement pour H et L
           n_imag = st.session_state.get(const_i_key, 0.0)

        # --- Validation Physique ---
        valid_n = True
        valid_k = True
        if n_real <= 0:
             add_log(f"AVERTISSEMENT: n' constant pour {role} <= 0 ({n_real:.3f}), utilisation de 1.0.")
             n_real = 1.0
             valid_n = False
        if n_imag < 0:
             add_log(f"AVERTISSEMENT: k constant pour {role} < 0 ({n_imag:.3f}), utilisation de 0.0.")
             n_imag = 0.0
             valid_k = False

        mat_repr = f"Constant ({n_real:.3f}{'+' if n_imag>=0 else ''}{n_imag:.3f}j)"
        if not valid_n or not valid_k:
             mat_repr += " (Ajusté)"

        return complex(n_real, n_imag), mat_repr

    elif isinstance(selection, str) and selection:
        # C'est un nom de matériau (prédéfini ou Excel)
        return selection, selection
    else:
        # Cas d'erreur : sélection invalide ou non trouvée
        st.error(f"Sélection matériau pour '{role}' invalide ou manquante dans session_state.")
        add_log(f"Erreur critique: Sélection matériau '{role}' invalide: {selection}")
        return None, "Erreur Sélection"

def validate_targets() -> Optional[List[Dict]]:
    """Valide les cibles définies dans st.session_state.targets. Retourne la liste des cibles actives et valides, ou None si erreur."""
    active_targets = []
    logs = []
    is_valid = True
    if 'targets' not in st.session_state or not isinstance(st.session_state.targets, list):
         st.error("Erreur interne : Liste de cibles manquante ou invalide dans session_state.")
         return None

    for i, target_state in enumerate(st.session_state.targets):
        if target_state.get('enabled', False):
            try:
                # Essayer de convertir en float, gère les erreurs de type/format
                l_min = float(target_state['min'])
                l_max = float(target_state['max'])
                t_min = float(target_state['target_min'])
                t_max = float(target_state['target_max'])

                # Vérifications de cohérence
                if l_max < l_min:
                     logs.append(f"Erreur Cible {i+1}: λ max ({l_max:.1f}) < λ min ({l_min:.1f}).")
                     is_valid = False; continue # Passer à la suivante si invalide
                if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0):
                     logs.append(f"Erreur Cible {i+1}: Transmittance hors de [0, 1] (Tmin={t_min:.2f}, Tmax={t_max:.2f}).")
                     is_valid = False; continue

                # Ajouter la cible validée à la liste
                active_targets.append({
                    'min': l_min, 'max': l_max,
                    'target_min': t_min, 'target_max': t_max
                })
            except (KeyError, ValueError, TypeError) as e:
                 # Erreur si une clé manque, ou si la valeur n'est pas convertible en float
                 logs.append(f"Erreur Cible {i+1}: Données manquantes ou invalides ({e}).")
                 is_valid = False; continue

    # Afficher les erreurs dans les logs Streamlit si nécessaire
    if not is_valid:
         add_log(["Erreurs détectées dans la définition des cibles actives:"] + logs)
         st.warning("Des erreurs existent dans la définition des cibles spectrales actives. Veuillez corriger.")
         # On ne retourne None que si une erreur bloquante est détectée,
         # sinon on retourne les cibles valides trouvées. Ici on considère les erreurs comme bloquantes.
         return None
    elif not active_targets:
         add_log("Aucune cible spectrale n'est activée.")
         # Ce n'est pas une erreur bloquante en soi, mais certaines actions échoueront.
         # Retourner une liste vide.
         return []
    else:
        add_log(f"{len(active_targets)} cible(s) active(s) et valide(s) trouvée(s).")
        return active_targets

def get_lambda_range_from_targets(validated_targets: Optional[List[Dict]]) -> Tuple[Optional[float], Optional[float]]:
    """Détermine la plage lambda globale à partir des cibles validées."""
    overall_min, overall_max = None, None
    if validated_targets: # Assurer que la liste n'est pas None ou vide
        all_mins = [t['min'] for t in validated_targets]
        all_maxs = [t['max'] for t in validated_targets]
        if all_mins: overall_min = min(all_mins)
        if all_maxs: overall_max = max(all_maxs)
    return overall_min, overall_max

def clear_optimized_state():
    """Réinitialise l'état lié à une optimisation précédente."""
    add_log("Nettoyage de l'état optimisé et de l'historique.")
    st.session_state.optimized_ep = None
    st.session_state.is_optimized_state = False
    st.session_state.ep_history = deque(maxlen=5) # Réinitialiser l'historique aussi
    st.session_state.optimized_qwot_str = ""
    st.session_state.last_mse = None
    # Mettre à jour l'affichage ? Normalement géré par le rerun de Streamlit.

def set_optimized_as_nominal_wrapper():
    """Met à jour le QWOT nominal avec celui de l'état optimisé actuel."""
    add_log("Tentative de définition de l'Optimisé comme Nominal...")
    if not st.session_state.get('is_optimized_state') or st.session_state.get('optimized_ep') is None:
        st.error("Aucune structure optimisée valide à définir comme nominale.")
        add_log("Erreur: Pas de structure optimisée à définir comme nominale.")
        return

    try:
        l0 = st.session_state.l0
        nH_mat, _ = get_material_input('H')
        nL_mat, _ = get_material_input('L')
        if nH_mat is None or nL_mat is None:
             st.error("Impossible de récupérer les matériaux H/L pour recalculer le QWOT.")
             add_log("Erreur: Matériaux H/L non valides pour recalcul QWOT.")
             return

        optimized_qwots, logs_qwot = calculate_qwot_from_ep(st.session_state.optimized_ep, l0, nH_mat, nL_mat, EXCEL_FILE_PATH)
        add_log(logs_qwot)

        if optimized_qwots is None:
             st.error("Erreur lors du recalcul du QWOT à partir de la structure optimisée.")
             add_log("Erreur: recalcul QWOT a échoué (retourné None).")
             return

        if np.any(np.isnan(optimized_qwots)):
            st.warning("Le QWOT recalculé contient des NaN (probablement indice invalide à l0). QWOT nominal non mis à jour.")
            add_log("Avertissement: QWOT recalculé contient NaN. Nominal non mis à jour.")
            # Optionnel: on pourrait quand même clearer l'état optimisé ? Non, plus sûr de ne rien faire.
        else:
            # Formatage du QWOT avec plus de décimales pour la précision
            new_qwot_str = ",".join([f"{q:.6f}" for q in optimized_qwots])
            st.session_state.current_qwot = new_qwot_str
            add_log(f"QWOT Nominal mis à jour : {new_qwot_str}")
            st.success("Structure optimisée définie comme nouveau Nominal (QWOT mis à jour).")
            # Nettoyer l'état optimisé après succès
            clear_optimized_state()

    except Exception as e:
        st.error(f"Erreur inattendue lors de la définition de l'optimisé comme nominal: {e}")
        add_log(f"Erreur inattendue (set_optimized_as_nominal): {e}\n{traceback.format_exc(limit=1)}")

def undo_remove_wrapper():
    """Restaure l'état précédent depuis l'historique."""
    add_log("Tentative d'annulation de la dernière suppression...")
    if not st.session_state.get('ep_history'):
        st.info("Historique d'annulation vide.")
        add_log("Historique vide, annulation impossible.")
        return

    try:
        # Restaurer l'état précédent
        last_ep = st.session_state.ep_history.pop() # pop() enlève le dernier élément
        st.session_state.optimized_ep = last_ep.copy()
        st.session_state.is_optimized_state = True # On revient à un état optimisé

        add_log(f"État restauré ({len(last_ep)} couches). {len(st.session_state.ep_history)} états restants dans l'historique.")

        # Recalculer le QWOT optimisé pour affichage
        l0 = st.session_state.l0
        nH_mat, _ = get_material_input('H')
        nL_mat, _ = get_material_input('L')
        if nH_mat is not None and nL_mat is not None:
             qwots_recalc, logs_qwot = calculate_qwot_from_ep(last_ep, l0, nH_mat, nL_mat, EXCEL_FILE_PATH)
             add_log(logs_qwot)
             if qwots_recalc is not None and not np.any(np.isnan(qwots_recalc)):
                 st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_recalc])
             else:
                  st.session_state.optimized_qwot_str = "QWOT N/A (après undo)"
        else:
             st.session_state.optimized_qwot_str = "QWOT Erreur Matériau (après undo)"

        # Déclencher un recalcul complet pour mettre à jour les plots et le MSE
        st.info("État restauré. Recalcul en cours...")
        # Indiquer qu'il faut recalculer lors du prochain rerun
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {
             'is_optimized_run': True,
             'method_name': "Optimized (Undo)",
             'force_ep': st.session_state.optimized_ep # Forcer l'utilisation de l'EP restauré
             }
        # Le rerun se fera automatiquement à la fin du script

    except IndexError:
         st.warning("Historique d'annulation vide (erreur interne?).")
         add_log("Erreur: Tentative de pop sur historique vide.")
    except Exception as e:
        st.error(f"Erreur inattendue lors de l'annulation: {e}")
        add_log(f"Erreur inattendue (undo_remove): {e}\n{traceback.format_exc(limit=1)}")
        # Essayer de nettoyer l'état en cas d'erreur ?
        clear_optimized_state()


# --- Wrappers pour les Actions Principales ---

def run_calculation_wrapper(is_optimized_run: bool, method_name: str = "", force_ep: Optional[np.ndarray] = None):
    """Wrapper pour lancer un calcul (nominal ou optimisé) et afficher les résultats."""
    calc_type = 'Optimisé' if is_optimized_run else 'Nominal'
    add_log(f"\n{'='*10} Démarrage Calcul {calc_type} {'('+method_name+')' if method_name else ''} {'='*10}")
    st.session_state.last_calc_results = {} # Pour stocker les résultats intermédiaires si besoin
    st.session_state.last_mse = None # Réinitialiser le MSE affiché

    with st.spinner(f"Calcul {calc_type} en cours..."):
        try:
            # 1. Valider les cibles (nécessaires pour la plage lambda)
            active_targets = validate_targets()
            if active_targets is None: # Erreur de validation
                 st.error("Définition des cibles invalide. Vérifiez les logs et corrigez.")
                 add_log("Calcul annulé: Cibles invalides.")
                 return # Arrêter
            if not active_targets:
                 st.warning("Aucune cible active. Plage lambda par défaut utilisée (400-700nm). Le calcul MSE sera N/A.")
                 # Définir une plage par défaut si aucune cible n'est active
                 l_min_plot, l_max_plot = 400.0, 700.0
                 # On continue, mais le MSE ne sera pas calculable
            else:
                # Obtenir la plage lambda depuis les cibles validées
                l_min_plot, l_max_plot = get_lambda_range_from_targets(active_targets)
                if l_min_plot is None or l_max_plot is None or l_max_plot < l_min_plot:
                    st.error("Impossible de déterminer une plage lambda valide depuis les cibles.")
                    add_log("Calcul annulé: Plage lambda invalide depuis cibles.")
                    return

            # 2. Récupérer les paramètres et matériaux
            # Note: 'l_range_deb' et 'l_range_fin' ne sont pas dans l'UI, on utilise l_min/max_plot
            validated_inputs = {
                'l0': st.session_state.l0,
                'l_step': st.session_state.l_step, # Utilisé pour la grille d'optim/MSE display
                'emp_str': st.session_state.current_qwot,
                # Ajouter les params d'optim même si non utilisés directement ici, par cohérence
                'maxiter': st.session_state.maxiter,
                'maxfun': st.session_state.maxfun,
                'auto_thin_threshold': st.session_state.auto_thin_threshold,
                # Plage pour le plot fin et le calcul MSE
                'l_range_deb': l_min_plot,
                'l_range_fin': l_max_plot,
            }
            nH_mat, nH_repr = get_material_input('H')
            nL_mat, nL_repr = get_material_input('L')
            nSub_mat, nSub_repr = get_material_input('Sub')

            if nH_mat is None or nL_mat is None or nSub_mat is None:
                 st.error("Erreur de définition de matériau. Vérifiez les sélections et/ou les fichiers Excel.")
                 add_log("Calcul annulé: Erreur matériau.")
                 return

            validated_inputs['nH_material'] = nH_mat
            validated_inputs['nL_material'] = nL_mat
            validated_inputs['nSub_material'] = nSub_mat
            add_log(f"Matériaux utilisés: H={nH_repr}, L={nL_repr}, Sub={nSub_repr}")

            # 3. Déterminer le vecteur d'épaisseurs (ep) à utiliser
            ep_to_calculate = None
            if force_ep is not None:
                 ep_to_calculate = force_ep.copy()
                 add_log("Utilisation d'un vecteur ep forcé.")
            elif is_optimized_run and st.session_state.get('optimized_ep') is not None:
                ep_to_calculate = st.session_state.optimized_ep.copy()
                add_log("Utilisation de la structure optimisée actuelle.")
            else:
                # Calcul nominal ou état optimisé inexistant -> recalcul depuis QWOT
                add_log("Utilisation de la structure nominale (QWOT).")
                emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
                if not emp_list and calc_type == 'Nominal':
                     add_log("QWOT Nominal vide, calcul pour substrat nu.")
                     ep_to_calculate = np.array([], dtype=np.float64) # Vecteur vide pour cas 0 couche
                elif not emp_list and calc_type == 'Optimisé':
                     st.error("Impossible de lancer un calcul optimisé si le QWOT nominal est vide et qu'il n'y a pas d'état optimisé précédent.")
                     add_log("Erreur: Calcul optimisé demandé mais état initial vide.")
                     return
                else:
                     ep_calc, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                     add_log(logs_ep_init)
                     if ep_calc is None:
                          st.error("Échec du calcul des épaisseurs initiales depuis le QWOT.")
                          add_log("Calcul annulé: échec calcul ep initial.")
                          return
                     ep_to_calculate = ep_calc.copy()

            # Stocker l'ep utilisé pour ce calcul
            st.session_state.current_ep = ep_to_calculate.copy() if ep_to_calculate is not None else None

            # 4. Définir le vecteur lambda pour le plot fin
            # Utiliser linspace pour une meilleure résolution visuelle
            num_plot_points = max(501, int(np.round((l_max_plot - l_min_plot) / validated_inputs['l_step'])) * 3 + 1) # Plus de points
            l_vec_plot_fine_np = np.linspace(l_min_plot, l_max_plot, num_plot_points)
            l_vec_plot_fine_np = l_vec_plot_fine_np[(l_vec_plot_fine_np > 0) & np.isfinite(l_vec_plot_fine_np)]
            if not l_vec_plot_fine_np.size:
                 st.error("Impossible de générer un vecteur lambda valide pour le tracé.")
                 add_log("Calcul annulé: vecteur lambda pour plot invalide.")
                 return
            add_log(f"Calcul T(lambda) sur {len(l_vec_plot_fine_np)} points pour le tracé [{l_min_plot:.1f}-{l_max_plot:.1f} nm].")

            # 5. Lancer le calcul JAX (H/L standard, séquence arbitraire non gérée ici pour simplifier)
            # TODO: Ajouter la gestion de material_sequence si nécessaire
            start_calc_time = time.time()
            results_fine, calc_logs = calculate_T_from_ep_jax(
                ep_to_calculate, nH_mat, nL_mat, nSub_mat, l_vec_plot_fine_np, EXCEL_FILE_PATH
            )
            add_log(calc_logs)
            if results_fine is None:
                 st.error("Le calcul principal de la transmittance a échoué.")
                 add_log("Erreur critique: calculate_T_from_ep_jax a retourné None.")
                 return # Arrêter
            add_log(f"Calcul T(lambda) terminé en {time.time() - start_calc_time:.3f}s.")

            # Stocker les résultats pour l'affichage
            st.session_state.last_calc_results = {
                'res_fine': results_fine, # Résultats sur grille fine pour plot
                'method_name': method_name,
                # Ajouter d'autres infos si besoin pour le plot (ep, l0, matériaux...)
                'ep_used': ep_to_calculate.copy() if ep_to_calculate is not None else None,
                'l0_used': validated_inputs['l0'],
                'nH_used': nH_mat, 'nL_used': nL_mat, 'nSub_used': nSub_mat,
            }

            # 6. Calculer le MSE pour affichage (basé sur la grille d'optimisation)
            if active_targets:
                num_pts_optim = max(2, int(np.round((l_max_plot - l_min_plot) / validated_inputs['l_step'])) + 1)
                l_vec_optim_np = np.geomspace(l_min_plot, l_max_plot, num_pts_optim)
                l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]

                if l_vec_optim_np.size > 0:
                     add_log(f"Calcul T(lambda) sur {len(l_vec_optim_np)} points pour affichage MSE...")
                     results_optim_grid, logs_mse_calc = calculate_T_from_ep_jax(
                         ep_to_calculate, nH_mat, nL_mat, nSub_mat, l_vec_optim_np, EXCEL_FILE_PATH
                     )
                     add_log(logs_mse_calc)
                     if results_optim_grid is not None:
                          mse_display, num_pts_mse = calculate_final_mse(results_optim_grid, active_targets)
                          st.session_state.last_mse = mse_display
                          add_log(f"MSE calculé pour affichage: {mse_display:.4e} (sur {num_pts_mse} points)" if mse_display is not None else "MSE N/A (pas de points valides dans cibles)")
                          st.session_state.last_calc_results['res_optim_grid'] = results_optim_grid # Pour les markers sur le plot
                     else:
                          add_log("Échec calcul T sur grille optim pour MSE.")
                          st.session_state.last_mse = None
                else:
                     add_log("Grille d'optimisation vide, MSE non calculé.")
                     st.session_state.last_mse = None
            else:
                 add_log("Pas de cibles actives, MSE non calculé.")
                 st.session_state.last_mse = None

            # 7. Mettre à jour l'état global
            st.session_state.is_optimized_state = is_optimized_run
            if not is_optimized_run: # Si c'était un calcul nominal, effacer l'ancien état optimisé
                 clear_optimized_state() # Efface optimized_ep, history, etc.
                 st.session_state.current_ep = ep_to_calculate.copy() if ep_to_calculate is not None else None # Assurer que current_ep est bien le nominal


            st.success(f"Calcul {calc_type} terminé.")
            add_log(f"--- Fin Calcul {calc_type} ---")

        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Erreur durant le calcul {calc_type}: {e}")
            add_log(f"ERREUR (Calcul {calc_type}): {e}\n{traceback.format_exc(limit=1)}")
        except Exception as e_fatal:
             st.error(f"Erreur inattendue durant le calcul {calc_type}: {e_fatal}")
             add_log(f"ERREUR FATALE (Calcul {calc_type}): {e_fatal}\n{traceback.format_exc()}")

# Ajouter les wrappers pour les autres boutons (Local Opt, Scan Opt, Auto, Remove Thin)
# sur le même modèle que run_calculation_wrapper :
# 1. Lire st.session_state
# 2. Valider entrées / état
# 3. Appeler la fonction logique correspondante (_run_core_optimization, run_auto_mode, etc.)
# 4. Mettre à jour st.session_state avec les résultats
# 5. Logger et afficher status/erreurs

def run_local_optimization_wrapper():
    """Wrapper pour l'optimisation locale."""
    add_log(f"\n{'='*10} Démarrage Optimisation Locale {'='*10}")
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None
    clear_optimized_state() # Effacer l'ancien état optimisé et l'historique

    with st.spinner("Optimisation locale en cours..."):
        try:
            # Validation et récupération paramètres (similaire à run_calculation_wrapper)
            active_targets = validate_targets()
            if active_targets is None or not active_targets:
                 st.error("Optimisation locale nécessite des cibles actives et valides.")
                 add_log("Optimisation locale annulée: cibles invalides ou manquantes.")
                 return

            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None:
                 st.error("Impossible de déterminer la plage lambda pour l'optimisation.")
                 add_log("Optimisation locale annulée: plage lambda invalide.")
                 return

            validated_inputs = { # Récupérer tous les params nécessaires
                'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                'maxiter': st.session_state.maxiter, 'maxfun': st.session_state.maxfun,
                'emp_str': st.session_state.current_qwot,
                'auto_thin_threshold': st.session_state.auto_thin_threshold, # Non utilisé ici mais garder cohérence
                'l_range_deb': l_min_opt, 'l_range_fin': l_max_opt,
             }
            nH_mat, nH_repr = get_material_input('H')
            nL_mat, nL_repr = get_material_input('L')
            nSub_mat, nSub_repr = get_material_input('Sub')
            if nH_mat is None or nL_mat is None or nSub_mat is None:
                 st.error("Erreur définition matériau pour optimisation.")
                 add_log("Optimisation locale annulée: erreur matériau.")
                 return
            validated_inputs['nH_material'] = nH_mat
            validated_inputs['nL_material'] = nL_mat
            validated_inputs['nSub_material'] = nSub_mat
            add_log(f"Matériaux Opt: H={nH_repr}, L={nL_repr}, Sub={nSub_repr}")

            # Obtenir la structure de départ (nominale)
            emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
            if not emp_list:
                 st.error("QWOT nominal vide, impossible de démarrer l'optimisation locale.")
                 add_log("Optimisation locale annulée: QWOT nominal vide.")
                 return
            ep_start, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
            add_log(logs_ep_init)
            if ep_start is None:
                 st.error("Échec calcul épaisseurs initiales pour optimisation locale.")
                 add_log("Optimisation locale annulée: échec calcul ep initial.")
                 return
            add_log(f"Démarrage optimisation locale depuis {len(ep_start)} couches nominales.")

            # Lancer l'optimisation principale
            final_ep, success, final_cost, optim_logs, msg, nit, nfev = \
                _run_core_optimization(ep_start, validated_inputs, active_targets,
                                       MIN_THICKNESS_PHYS_NM, log_prefix="  [Opt Local] ")
            add_log(optim_logs)

            # Mettre à jour l'état de session avec le résultat
            if success and final_ep is not None:
                 st.session_state.optimized_ep = final_ep.copy()
                 st.session_state.current_ep = final_ep.copy() # Afficher le résultat optimisé
                 st.session_state.is_optimized_state = True
                 st.session_state.last_mse = final_cost

                 # Recalculer QWOT optimisé pour affichage
                 qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                 add_log(logs_qwot)
                 if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                      st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                 else:
                      st.session_state.optimized_qwot_str = "QWOT N/A"

                 st.success(f"Optimisation locale terminée ({msg}). MSE: {final_cost:.4e}")
                 add_log(f"--- Fin Optimisation Locale (Succès) ---")
                 # Déclencher un recalcul/affichage final
                 st.session_state.needs_rerun_calc = True
                 st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Opt Local ({nit}/{nfev})"}

            else:
                 st.error(f"L'optimisation locale a échoué: {msg}")
                 add_log(f"--- Fin Optimisation Locale (Échec) ---")
                 # Garder l'état nominal affiché
                 st.session_state.is_optimized_state = False
                 st.session_state.optimized_ep = None
                 st.session_state.current_ep = ep_start.copy() # Revenir à l'EP nominal
                 st.session_state.last_mse = None

        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Erreur durant l'optimisation locale: {e}")
            add_log(f"ERREUR (Opt Locale): {e}\n{traceback.format_exc(limit=1)}")
            clear_optimized_state()
        except Exception as e_fatal:
             st.error(f"Erreur inattendue durant l'optimisation locale: {e_fatal}")
             add_log(f"ERREUR FATALE (Opt Locale): {e_fatal}\n{traceback.format_exc()}")
             clear_optimized_state()


# Ajouter ici les wrappers pour run_scan_optimization_wrapper, run_auto_mode_wrapper, run_remove_thin_wrapper
# ... (sur le même modèle que run_local_optimization_wrapper) ...
# Ces wrappers deviendront complexes car ils enchaînent plusieurs appels aux fonctions logiques.


# =============================================
# SECTION 3 : INTERFACE UTILISATEUR STREAMLIT (UI)
# =============================================

# --- Configuration de la Page ---
st.set_page_config(layout="wide", page_title="Optimiseur Film Mince (Streamlit)")
st.title("🔬 Optimiseur de Films Minces (Streamlit + JAX)")
st.markdown("""
*Conversion Streamlit de l'outil Tkinter. Se concentre sur les calculs H/L.*
**Note:** Ce code est généré par IA et nécessite validation et débogage.
""")

# --- Initialisation de l'État de Session (si pas déjà fait au début) ---
# Essentiel pour que Streamlit se souvienne des valeurs entre les interactions
if 'init_done' not in st.session_state:
    st.session_state.log_messages = ["[Initialisation] Bienvenue dans l'optimiseur Streamlit."]
    st.session_state.current_ep = None # Épaisseurs actuellement affichées/utilisées
    st.session_state.current_qwot = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1" # QWOT nominal par défaut (20 couches)
    st.session_state.optimized_ep = None # Stockage du dernier résultat optimisé
    st.session_state.is_optimized_state = False # Flag état actuel (Nominal ou Optimisé)
    st.session_state.optimized_qwot_str = "" # String QWOT optimisé pour affichage
    st.session_state.material_sequence = None # Non utilisé dans cette version simplifiée
    st.session_state.ep_history = deque(maxlen=5) # Historique pour "Undo" (limité à 5 états)
    st.session_state.last_mse = None # Dernier MSE calculé pour affichage
    st.session_state.needs_rerun_calc = False # Flag pour déclencher recalcul après certaines actions
    st.session_state.rerun_calc_params = {} # Paramètres pour le recalcul automatique

    # Charger les matériaux disponibles une seule fois
    try:
        mats, logs = get_available_materials_from_excel(EXCEL_FILE_PATH)
        add_log(logs) # Ajouter les logs du chargement initial
        st.session_state.available_materials = sorted(list(set(["Constant"] + mats)))
        base_subs = ["Constant", "Fused Silica", "BK7", "D263"]
        st.session_state.available_substrates = sorted(list(set(base_subs + st.session_state.available_materials)))
        add_log(f"Matériaux H/L chargés: {st.session_state.available_materials}")
        add_log(f"Substrats chargés: {st.session_state.available_substrates}")
    except Exception as e:
         st.error(f"Erreur initiale chargement matériaux depuis {EXCEL_FILE_PATH}: {e}")
         add_log(f"ERREUR CRITIQUE: Chargement initial matériaux échoué: {e}")
         st.session_state.available_materials = ["Constant"]
         st.session_state.available_substrates = ["Constant"]

    # Définir les valeurs par défaut des paramètres
    st.session_state.l0 = 500.0
    st.session_state.l_step = 10.0
    st.session_state.maxiter = 1000
    st.session_state.maxfun = 1000
    st.session_state.auto_thin_threshold = 1.0
    # Sélections matériaux par défaut
    st.session_state.selected_H = next((m for m in ["Nb2O5-Helios", "Constant"] if m in st.session_state.available_materials), "Constant")
    st.session_state.selected_L = next((m for m in ["SiO2-Helios", "Constant"] if m in st.session_state.available_materials), "Constant")
    st.session_state.selected_Sub = next((m for m in ["Fused Silica", "Constant"] if m in st.session_state.available_substrates), "Constant")
    # Cibles par défaut
    st.session_state.targets = [
        {'enabled': True, 'min': 400.0, 'max': 500.0, 'target_min': 1.0, 'target_max': 1.0},
        {'enabled': True, 'min': 500.0, 'max': 600.0, 'target_min': 1.0, 'target_max': 0.2},
        {'enabled': True, 'min': 600.0, 'max': 700.0, 'target_min': 0.2, 'target_max': 0.2},
        {'enabled': False, 'min': 700.0, 'max': 800.0, 'target_min': 0.0, 'target_max': 0.0},
        {'enabled': False, 'min': 800.0, 'max': 900.0, 'target_min': 0.0, 'target_max': 0.0},
    ]
    # Initialiser les clés pour les indices constants
    st.session_state.nH_r = 2.35
    st.session_state.nH_i = 0.0
    st.session_state.nL_r = 1.46
    st.session_state.nL_i = 0.0
    st.session_state.nSub_r = 1.52

    st.session_state.init_done = True # Marquer l'initialisation comme faite
    add_log("État de session initialisé.")
    st.session_state.needs_rerun_calc = True # Forcer un calcul initial au premier lancement
    st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Initial Load"}

# --- Interface Sidebar (Contrôles) ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- Section Matériaux ---
    st.subheader("Matériaux")
    st.session_state.selected_H = st.selectbox(
        "Matériau H", options=st.session_state.available_materials,
        index=st.session_state.available_materials.index(st.session_state.selected_H),
        key="sb_H"
    )
    if st.session_state.selected_H == "Constant":
        hc1, hc2 = st.columns(2)
        st.session_state.nH_r = hc1.number_input("n' H", value=st.session_state.nH_r, format="%.4f", key="nH_r_const")
        st.session_state.nH_i = hc2.number_input("k H", value=st.session_state.nH_i, min_value=0.0, format="%.4f", key="nH_i_const")

    st.session_state.selected_L = st.selectbox(
        "Matériau L", options=st.session_state.available_materials,
        index=st.session_state.available_materials.index(st.session_state.selected_L),
        key="sb_L"
    )
    if st.session_state.selected_L == "Constant":
         lc1, lc2 = st.columns(2)
         st.session_state.nL_r = lc1.number_input("n' L", value=st.session_state.nL_r, format="%.4f", key="nL_r_const")
         st.session_state.nL_i = lc2.number_input("k L", value=st.session_state.nL_i, min_value=0.0, format="%.4f", key="nL_i_const")

    st.session_state.selected_Sub = st.selectbox(
        "Substrat", options=st.session_state.available_substrates,
        index=st.session_state.available_substrates.index(st.session_state.selected_Sub),
        key="sb_Sub"
    )
    if st.session_state.selected_Sub == "Constant":
         st.session_state.nSub_r = st.number_input("n' Substrat", value=st.session_state.nSub_r, format="%.4f", key="nSub_const")
         # st.session_state.nSub_i = 0.0 # Pas d'input pour k substrat

    if st.button("🔄 Recharger Matériaux Excel", key="reload_mats"):
        st.cache_data.clear() # Effacer cache Streamlit
        # Note: Pas de cache_clear() pour functools ici, géré par @st.cache_data
        try:
            mats, logs = get_available_materials_from_excel(EXCEL_FILE_PATH)
            add_log(logs)
            st.session_state.available_materials = sorted(list(set(["Constant"] + mats)))
            base_subs = ["Constant", "Fused Silica", "BK7", "D263"]
            st.session_state.available_substrates = sorted(list(set(base_subs + st.session_state.available_materials)))
            add_log("Liste matériaux rechargée.")
            # Ajuster sélections si elles deviennent invalides
            if st.session_state.selected_H not in st.session_state.available_materials: st.session_state.selected_H = "Constant"
            if st.session_state.selected_L not in st.session_state.available_materials: st.session_state.selected_L = "Constant"
            if st.session_state.selected_Sub not in st.session_state.available_substrates: st.session_state.selected_Sub = "Constant"
            st.rerun() # Rafraîchir l'UI
        except Exception as e:
            st.error(f"Erreur rechargement matériaux: {e}")
            add_log(f"Erreur rechargement matériaux: {e}")

    st.divider()

    # --- Section Structure ---
    st.subheader("Structure Nominale")
    st.session_state.current_qwot = st.text_area(
        "QWOT Nominal (multiplicateurs séparés par ',')",
        value=st.session_state.current_qwot,
        key="qwot_input",
        on_change=clear_optimized_state # Si QWOT nominal change, reset l'état optimisé
    )
    num_layers_from_qwot = len([q for q in st.session_state.current_qwot.split(',') if q.strip()])
    st.caption(f"Couches d'après QWOT: {num_layers_from_qwot}")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.session_state.l0 = st.number_input("λ₀ centrage (nm)", value=st.session_state.l0, min_value=1.0, format="%.2f", key="l0_input")
    with c2:
         init_layers_num = st.number_input("N couches:", min_value=0, value=num_layers_from_qwot, step=1, key="init_layers_gen_num", label_visibility="collapsed")
         if st.button("Générer 1s", key="gen_qwot_btn", use_container_width=True):
            if init_layers_num > 0:
                new_qwot = ",".join(['1'] * init_layers_num)
                if new_qwot != st.session_state.current_qwot:
                     st.session_state.current_qwot = new_qwot # Met à jour QWOT
                     clear_optimized_state() # Reset état optimisé
                     # >>> AJOUTER CES LIGNES <<<
                     st.session_state.needs_rerun_calc = True # Déclencher calcul
                     st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (Généré 1s)"}
                     # >>> FIN AJOUT <<<
                     st.rerun() # Rerun pour prendre en compte QWOT et lancer calcul
            elif st.session_state.current_qwot != "":
                 st.session_state.current_qwot = ""
                 clear_optimized_state()
                 # >>> AJOUTER CES LIGNES (optionnel, lancer calcul pour substrat nu?) <<<
                 st.session_state.needs_rerun_calc = True # Déclencher calcul
                 st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (QWOT Effacé)"}
                 # >>> FIN AJOUT <<<
                 st.rerun()

    st.divider()

    # --- Section Paramètres ---
    st.subheader("Paramètres Calcul & Optimisation")
    st.session_state.l_step = st.number_input("Pas λ (nm) (optimisation)", value=st.session_state.l_step, min_value=0.1, format="%.2f", key="l_step_input")
    st.session_state.maxiter = st.number_input("Max Iter (Opt)", value=st.session_state.maxiter, min_value=1, step=50, key="maxiter_input")
    st.session_state.maxfun = st.number_input("Max Eval (Opt)", value=st.session_state.maxfun, min_value=1, step=50, key="maxfun_input")
    st.session_state.auto_thin_threshold = st.number_input("Seuil suppression auto (nm)", value=st.session_state.auto_thin_threshold, min_value=MIN_THICKNESS_PHYS_NM, format="%.3f", key="auto_thin_input")

    st.divider()

    # --- Section Cibles ---
    st.subheader("Cibles Spectrales (Transmission T)")
    header_cols = st.columns([0.5, 1.5, 1.5, 1.5, 1.5])
    headers = ["Actif", "λ min", "λ max", "T@λmin", "T@λmax"]
    for col, header in zip(header_cols, headers): col.caption(header)

    for i in range(len(st.session_state.targets)):
        target = st.session_state.targets[i]
        cols = st.columns([0.5, 1.5, 1.5, 1.5, 1.5])
        # Utiliser la valeur de l'état pour le checkbox
        current_enabled = target.get('enabled', False)
        new_enabled = cols[0].checkbox("", value=current_enabled, key=f"target_enable_{i}", label_visibility="collapsed")
        st.session_state.targets[i]['enabled'] = new_enabled
        # Utiliser la valeur de l'état pour les number_input
        st.session_state.targets[i]['min'] = cols[1].number_input("λmin", value=target.get('min', 0.0), format="%.1f", key=f"target_min_{i}", label_visibility="collapsed")
        st.session_state.targets[i]['max'] = cols[2].number_input("λmax", value=target.get('max', 0.0), format="%.1f", key=f"target_max_{i}", label_visibility="collapsed")
        st.session_state.targets[i]['target_min'] = cols[3].number_input("Tmin", value=target.get('target_min', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmin_{i}", label_visibility="collapsed")
        st.session_state.targets[i]['target_max'] = cols[4].number_input("Tmax", value=target.get('target_max', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmax_{i}", label_visibility="collapsed")

    # Afficher une prévisualisation des cibles actives
    with st.expander("👁️ Prévisualiser Cibles Actives", expanded=False):
        active_targets_preview = validate_targets() # Récupérer les cibles valides
        if active_targets_preview is not None: # Vérifier si la validation a réussi
             # --- Définition de create_target_preview_fig ---
             def create_target_preview_fig(active_targets_list):
                 fig_prev, ax_prev = plt.subplots(figsize=(5, 3))
                 if not active_targets_list:
                     ax_prev.text(0.5, 0.5, "Pas de cibles actives/valides", ha='center', va='center', transform=ax_prev.transAxes)
                     ax_prev.set_ylim(-0.05, 1.05)
                 else:
                     all_l_min = min(t['min'] for t in active_targets_list)
                     all_l_max = max(t['max'] for t in active_targets_list)
                     padding = (all_l_max - all_l_min) * 0.05 + 1 # Ajouter 1nm padding min
                     ax_prev.set_xlim(all_l_min - padding, all_l_max + padding)
                     ax_prev.set_ylim(-0.05, 1.05)
                     plotted_legend = False
                     for i, t in enumerate(active_targets_list):
                         label = f"Cible {i+1}" if not plotted_legend else "_nolegend_"
                         ax_prev.plot([t['min'], t['max']], [t['target_min'], t['target_max']],
                                      'r-', linewidth=1.5, marker='x', markersize=5, label=label)
                         plotted_legend = True
                     if plotted_legend: ax_prev.legend(fontsize='small')
                 ax_prev.set_title("Prévisualisation Cibles", fontsize=10)
                 ax_prev.set_xlabel("λ (nm)", fontsize=9)
                 ax_prev.set_ylabel("T Cible", fontsize=9)
                 ax_prev.grid(True, linestyle=':', linewidth=0.5)
                 ax_prev.tick_params(axis='both', which='major', labelsize=8)
                 fig_prev.tight_layout()
                 return fig_prev
             # --- Fin Définition ---

             fig_preview = create_target_preview_fig(active_targets_preview)
             st.pyplot(fig_preview)
             plt.close(fig_preview) # Fermer pour libérer mémoire
        else:
             st.warning("Impossible d'afficher la prévisualisation (erreurs dans les cibles).")

    st.divider()

    # --- Section Actions ---
    st.subheader("▶️ Actions")
    if st.button("📊 Évaluer Structure Nominale", key="eval_nom", use_container_width=True):
        # Déclencher le recalcul via le flag
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (Évalué)"}
        st.rerun() # Force le rerun pour que le calcul soit pris en compte en haut du script

    if st.button("✨ Optimisation Locale", key="optim_local", use_container_width=True):
        # Appeler directement le wrapper (le rerun se fera à la fin)
        run_local_optimization_wrapper()
        # Pas besoin de rerun ici, le wrapper met à jour l'état et le rerun naturel fera l'affichage

    # if st.button("🚀 Scan Initial + Optimisation", key="optim_scan", use_container_width=True):
        # run_scan_optimization_wrapper() # Implémenter ce wrapper
        # st.warning("Fonction Scan + Opt pas encore implémentée dans ce wrapper.")

    # if st.button("🤖 Mode Auto (Needle > Thin > Opt)", key="optim_auto", use_container_width=True):
        # run_auto_mode_wrapper() # Implémenter ce wrapper
        # st.warning("Fonction Mode Auto pas encore implémentée dans ce wrapper.")

    st.divider()
    st.subheader("🛠️ Actions sur Optimisé")
    # Boutons conditionnels basés sur l'état optimisé
    can_optimize = st.session_state.get('is_optimized_state', False) and st.session_state.get('optimized_ep') is not None
    can_remove = can_optimize and len(st.session_state.optimized_ep) > 2
    can_undo = bool(st.session_state.get('ep_history'))

    if st.button("🗑️ Suppr. Couche Fine + Ré-opt", key="remove_thin", use_container_width=True, disabled=not can_remove):
        # run_remove_thin_wrapper() # Implémenter ce wrapper
        st.warning("Fonction Suppr. Couche Fine pas encore implémentée.")

    if st.button("💾 Optimisé -> Nominal", key="set_optim_as_nom", use_container_width=True, disabled=not can_optimize):
         set_optimized_as_nominal_wrapper()
         st.rerun() # Forcer rerun pour màj UI après changement d'état nominal

    if st.button(f"↩️ Annuler Suppr. ({len(st.session_state.get('ep_history', deque()))})", key="undo_remove", use_container_width=True, disabled=not can_undo):
         undo_remove_wrapper()
         # Le wrapper doit mettre needs_rerun_calc=True pour réafficher l'état restauré
         st.rerun() # Forcer rerun pour màj UI après undo


# --- Zone Principale (Affichage Résultats) ---
col_main, col_log = st.columns([2, 1]) # Colonne pour résultats/plots, colonne pour logs

with col_main:
    st.header("📈 Résultats")

    # --- Affichage de l'état actuel ---
    state_desc = "Optimisé" if st.session_state.is_optimized_state else "Nominal"
    ep_display = st.session_state.optimized_ep if st.session_state.is_optimized_state else st.session_state.current_ep # Afficher le bon EP
    num_layers_display = len(ep_display) if ep_display is not None else 0
    st.subheader(f"État Actuel : {state_desc} ({num_layers_display} couches)")

    res_cols = st.columns(3)
    with res_cols[0]:
        if st.session_state.last_mse is not None and np.isfinite(st.session_state.last_mse):
            st.metric("MSE", f"{st.session_state.last_mse:.4e}")
        else:
             st.metric("MSE", "N/A")
    with res_cols[1]:
         min_thick_str = "N/A"
         if ep_display is not None and ep_display.size > 0:
              valid_thick = ep_display[ep_display >= MIN_THICKNESS_PHYS_NM - 1e-9] # Tolérance
              if valid_thick.size > 0:
                   min_thick_str = f"{np.min(valid_thick):.3f} nm"
         st.metric("Ép. Min", min_thick_str)
         # Optionnel: Barre de progression
         # if 'min_thick' in locals() and np.isfinite(min_thick):
         #     st.progress(min(min_thick / 10.0, 1.0))

    with res_cols[2]:
         # Afficher le QWOT optimisé si pertinent
         if st.session_state.is_optimized_state and st.session_state.get('optimized_qwot_str'):
              st.text_input("QWOT Opt.", value=st.session_state.optimized_qwot_str, disabled=True, key="opt_qwot_display_main")


    # --- Section Graphiques ---
    st.subheader("Graphiques")
    # Vérifier s'il y a des résultats à plotter
    if 'last_calc_results' in st.session_state and st.session_state.last_calc_results:
        results_data = st.session_state.last_calc_results
        res_fine_plot = results_data.get('res_fine')
        ep_plot = results_data.get('ep_used')
        l0_plot = results_data.get('l0_used')
        nH_plot = results_data.get('nH_used')
        nL_plot = results_data.get('nL_used')
        nSub_plot = results_data.get('nSub_used')
        active_targets_plot = validate_targets() # Revalider pour être sûr
        mse_plot = st.session_state.last_mse
        is_optimized_plot = st.session_state.is_optimized_state
        method_name_plot = results_data.get('method_name', '')
        res_optim_grid_plot = results_data.get('res_optim_grid') # Pour les markers

        # S'assurer que les données nécessaires sont présentes
        if res_fine_plot and ep_plot is not None and l0_plot is not None and \
           nH_plot is not None and nL_plot is not None and nSub_plot is not None and active_targets_plot is not None:

            # --- Définition draw_plots_st ---
            # (Doit être définie ici ou importée, contenant la logique de plot Matplotlib adaptée)
            def draw_plots_st(res: Dict, current_ep: np.ndarray, l0_repr: float,
                              nH_material_in: MaterialInputType, nL_material_in: MaterialInputType, nSub_material_in: MaterialInputType,
                              active_targets_for_plot: List[Dict], mse: Optional[float],
                              is_optimized: bool = False, method_name: str = "",
                              res_optim_grid: Optional[Dict] = None,
                              material_sequence: Optional[List[str]] = None):
                """Génère la figure Matplotlib avec les 3 plots. Retourne fig."""
                # plt.style.use('seaborn-v0_8-whitegrid') # Style optionnel
                fig, axes = plt.subplots(1, 3, figsize=(18, 5)) # Ajuster taille si besoin

                opt_method_str = f" ({method_name})" if method_name else ""
                window_title = f'Résultats {"Optimisé" if is_optimized else "Nominal"}{opt_method_str}'
                fig.suptitle(window_title, fontsize=14, weight='bold')

                num_layers = len(current_ep) if current_ep is not None else 0

                # --- Plot 1: Spectre T(lambda) ---
                ax_spec = axes[0]
                line_ts = None
                try:
                    if res and 'l' in res and 'Ts' in res and res['l'] is not None and len(res['l']) > 0:
                        res_l_plot = np.asarray(res['l'])
                        res_ts_plot = np.asarray(res['Ts'])
                        line_ts, = ax_spec.plot(res_l_plot, res_ts_plot, label='Transmittance', linestyle='-', color='blue', linewidth=1.5)

                        plotted_target_label = False
                        if active_targets_for_plot:
                            for i, target in enumerate(active_targets_for_plot):
                                l_min, l_max = target['min'], target['max']
                                t_min, t_max_corr = target['target_min'], target['target_max']
                                x_coords, y_coords = [l_min, l_max], [t_min, t_max_corr]
                                # label = f'Cible {i+1}' if not plotted_target_label else "_nolegend_"
                                label = 'Cible(s)' if not plotted_target_label else "_nolegend_"
                                line_target, = ax_spec.plot(x_coords, y_coords, 'r--', linewidth=1.0, alpha=0.7, label=label, zorder=5)
                                marker_target = ax_spec.plot(x_coords, y_coords, marker='x', color='red', markersize=6, linestyle='none', label='_nolegend_', zorder=6)
                                plotted_target_label = True

                                # Ajouter markers de la grille d'optim si dispo
                                if res_optim_grid and 'l' in res_optim_grid and res_optim_grid['l'].size > 0:
                                    res_l_optim = np.asarray(res_optim_grid['l'])
                                    indices_optim = np.where((res_l_optim >= l_min) & (res_l_optim <= l_max))[0]
                                    if indices_optim.size > 0:
                                        optim_lambdas = res_l_optim[indices_optim]
                                        if abs(l_max - l_min) < 1e-9: optim_target_t = np.full_like(optim_lambdas, t_min)
                                        else: slope = (t_max_corr - t_min) / (l_max - l_min); optim_target_t = t_min + slope * (optim_lambdas - l_min)
                                        ax_spec.plot(optim_lambdas, optim_target_t, marker='.', color='darkred', linestyle='none', markersize=4, alpha=0.5, label='_nolegend_', zorder=6)


                        ax_spec.set_xlabel("Longueur d'onde (nm)")
                        ax_spec.set_ylabel('Transmittance')
                        ax_spec.set_title(f"Spectre{opt_method_str}")
                        ax_spec.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
                        ax_spec.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')
                        ax_spec.minorticks_on()
                        if len(res_l_plot) > 0: ax_spec.set_xlim(res_l_plot[0], res_l_plot[-1])
                        ax_spec.set_ylim(-0.05, 1.05)
                        if plotted_target_label or (line_ts is not None): ax_spec.legend(fontsize=8)

                        # Afficher MSE
                        if mse is not None and np.isfinite(mse): mse_text = f"MSE = {mse:.3e}"
                        else: mse_text = "MSE: N/A"
                        ax_spec.text(0.98, 0.98, mse_text, transform=ax_spec.transAxes, ha='right', va='top', fontsize=9,
                                     bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
                    else:
                        ax_spec.text(0.5, 0.5, "Pas de données spectrales", ha='center', va='center', transform=ax_spec.transAxes)
                except Exception as e_spec:
                    ax_spec.text(0.5, 0.5, f"Erreur plot spectre:\n{e_spec}", ha='center', va='center', transform=ax_spec.transAxes, color='red')
                    add_log(f"Erreur plot spectre: {e_spec}")


                # --- Plot 2: Profil d'indice ---
                ax_idx = axes[1]
                try:
                    # Obtenir les indices à l0
                    nH_c_repr, logs_h = _get_nk_at_lambda(nH_material_in, l0_repr, EXCEL_FILE_PATH)
                    nL_c_repr, logs_l = _get_nk_at_lambda(nL_material_in, l0_repr, EXCEL_FILE_PATH)
                    nSub_c_repr, logs_s = _get_nk_at_lambda(nSub_material_in, l0_repr, EXCEL_FILE_PATH)
                    add_log(logs_h); add_log(logs_l); add_log(logs_s)

                    if nH_c_repr is None or nL_c_repr is None or nSub_c_repr is None:
                         raise ValueError("Indices à l0 non trouvés pour plot profil.")

                    nH_r_repr, nL_r_repr, nSub_r_repr = nH_c_repr.real, nL_c_repr.real, nSub_c_repr.real

                    # TODO: Gérer material_sequence si besoin
                    if material_sequence:
                         n_real_layers_repr = [] # Calculer les n' pour chaque matériau de la séquence à l0
                         # ... (logique à ajouter) ...
                         add_log("AVERTISSEMENT: Plot profil pour séquence arbitraire non implémenté.")
                    else: # Cas H/L standard
                         n_real_layers_repr = [nH_r_repr if i % 2 == 0 else nL_r_repr for i in range(num_layers)]

                    ep_cumulative = np.cumsum(current_ep) if num_layers > 0 else np.array([0])
                    total_thickness = ep_cumulative[-1] if num_layers > 0 else 0
                    margin = max(50, 0.1 * total_thickness) if total_thickness > 0 else 50

                    # Construire les coordonnées pour steps-post
                    x_coords_plot = [-margin] # Start in substrate
                    y_coords_plot = [nSub_r_repr]
                    if num_layers > 0:
                        x_coords_plot.append(0) # Interface Substrat/L1
                        y_coords_plot.append(nSub_r_repr)
                        for i in range(num_layers):
                             layer_start = ep_cumulative[i-1] if i > 0 else 0
                             layer_end = ep_cumulative[i]
                             layer_n_real = n_real_layers_repr[i]
                             x_coords_plot.extend([layer_start, layer_end])
                             y_coords_plot.extend([layer_n_real, layer_n_real])
                        # Dernière interface vers Air
                        last_layer_end = ep_cumulative[-1]
                        x_coords_plot.extend([last_layer_end, last_layer_end + margin]) # Point final dans l'air
                        y_coords_plot.extend([1.0, 1.0]) # Indice air = 1.0
                    else: # Cas 0 couche
                        x_coords_plot.extend([0, 0, margin])
                        y_coords_plot.extend([nSub_r_repr, 1.0, 1.0])


                    ax_idx.plot(x_coords_plot, y_coords_plot, drawstyle='steps-post', label=f'n\'(λ={l0_repr:.0f}nm)', color='purple', linewidth=1.5)
                    ax_idx.set_xlabel('Profondeur (depuis substrat) (nm)')
                    ax_idx.set_ylabel("Partie Réelle Indice (n')")
                    ax_idx.set_title(f"Profil Indice (à λ={l0_repr:.0f}nm)")
                    ax_idx.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
                    ax_idx.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')
                    ax_idx.minorticks_on()
                    ax_idx.set_xlim(x_coords_plot[0], x_coords_plot[-1])

                    # Ajuster limites Y
                    valid_n = [n for n in [1.0, nSub_r_repr] + n_real_layers_repr if np.isfinite(n)]
                    min_n = min(valid_n) if valid_n else 0.9
                    max_n = max(valid_n) if valid_n else 2.5
                    y_padding = (max_n - min_n) * 0.1 + 0.05
                    ax_idx.set_ylim(bottom=min_n - y_padding, top=max_n + y_padding)

                    # Ajouter labels Air/Substrat
                    # ... (ajouter la logique de texte de l'original si besoin) ...

                    if ax_idx.get_legend_handles_labels()[1]: ax_idx.legend(fontsize=8)

                except Exception as e_idx:
                     ax_idx.text(0.5, 0.5, f"Erreur plot indice:\n{e_idx}", ha='center', va='center', transform=ax_idx.transAxes, color='red')
                     add_log(f"Erreur plot indice: {e_idx}")


                # --- Plot 3: Structure Empilement ---
                ax_stack = axes[2]
                try:
                    if num_layers > 0:
                         # Utiliser les mêmes indices réels qu'au dessus
                         # nH_c_repr, nL_c_repr définis pour plot 2

                         indices_complex_repr = []
                         if material_sequence:
                              # ... (logique pour obtenir n+ik à l0 pour chaque mat de la séquence) ...
                              add_log("AVERTISSEMENT: Plot structure pour séquence arbitraire non implémenté.")
                              # Fallback à H/L pour la couleur/label
                              indices_complex_repr = [nH_c_repr if i % 2 == 0 else nL_c_repr for i in range(num_layers)]
                              layer_types = [f"Mat{i+1}" for i in range(num_layers)] # Label générique
                         else: # Cas H/L standard
                              indices_complex_repr = [nH_c_repr if i % 2 == 0 else nL_c_repr for i in range(num_layers)]
                              layer_types = ["H" if i % 2 == 0 else "L" for i in range(num_layers)]

                         colors = ['lightblue' if i % 2 == 0 else 'lightcoral' for i in range(num_layers)] # Simple H/L color
                         bar_pos = np.arange(num_layers)
                         bars = ax_stack.barh(bar_pos, current_ep, align='center', color=colors, edgecolor='grey', height=0.8)

                         yticks_labels = []
                         for i in range(num_layers):
                             n_comp_repr = indices_complex_repr[i]
                             layer_type = layer_types[i]
                             n_str = f"{n_comp_repr.real:.3f}" if np.isfinite(n_comp_repr.real) else "N/A"
                             k_val = n_comp_repr.imag
                             if np.isfinite(k_val) and abs(k_val) > 1e-6: n_str += f"{k_val:+.3f}j"
                             yticks_labels.append(f"L{i + 1} ({layer_type}) n≈{n_str}")

                         ax_stack.set_yticks(bar_pos)
                         ax_stack.set_yticklabels(yticks_labels, fontsize=7) # Taille réduite pour plus de couches
                         ax_stack.invert_yaxis() # Couche 1 en haut

                         # Ajouter labels épaisseurs sur les barres
                         max_ep_plot = max(current_ep) if current_ep.size > 0 else 1.0
                         fontsize_bar = max(6, 9 - num_layers // 15)
                         for i, bar in enumerate(bars):
                              width = bar.get_width()
                              ha_pos = 'left' if width < max_ep_plot * 0.3 else 'right'
                              x_text_pos = width * 1.02 if ha_pos == 'left' else width * 0.98
                              text_color = 'black' if ha_pos == 'left' else 'white'
                              ax_stack.text(x_text_pos, bar.get_y() + bar.get_height()/2., f"{width:.2f}",
                                            va='center', ha=ha_pos, color=text_color, fontsize=fontsize_bar, weight='bold')

                    else: # Pas de couches
                        ax_stack.text(0.5, 0.5, "Structure Vide", ha='center', va='center', fontsize=10, color='grey', transform=ax_stack.transAxes)
                        ax_stack.set_yticks([]); ax_stack.set_xticks([])

                    ax_stack.set_xlabel('Épaisseur (nm)')
                    stack_title_prefix = f'Structure {"Optimisée" if is_optimized else "Nominale"}'
                    ax_stack.set_title(f"{stack_title_prefix} ({num_layers} couches)")
                    # Ajuster limites X
                    max_ep_plot = max(current_ep) if num_layers > 0 else 10
                    ax_stack.set_xlim(right=max_ep_plot * 1.1)

                except Exception as e_stack:
                    ax_stack.text(0.5, 0.5, f"Erreur plot structure:\n{e_stack}", ha='center', va='center', transform=ax_stack.transAxes, color='red')
                    add_log(f"Erreur plot structure: {e_stack}")


                plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.95]) # Ajuster rect pour laisser place au suptitle
                return fig
            # --- Fin Définition draw_plots_st ---

            try:
                 # Appel de la fonction de plot
                 fig_results = draw_plots_st(
                     res_fine_plot, ep_plot, l0_plot,
                     nH_plot, nL_plot, nSub_plot,
                     active_targets_plot, mse_plot,
                     is_optimized=is_optimized_plot, method_name=method_name_plot,
                     res_optim_grid=res_optim_grid_plot
                     # material_sequence=... # À passer si géré
                 )
                 st.pyplot(fig_results)
                 plt.close(fig_results) # Important pour libérer la mémoire
            except Exception as e:
                 st.error(f"Erreur lors de la génération des graphiques : {e}")
                 add_log(f"[Erreur Plot] {traceback.format_exc(limit=1)}")
        else:
             st.warning("Données de calcul manquantes ou invalides pour l'affichage des graphiques.")
    else:
        st.info("Lancez une évaluation ou une optimisation pour voir les résultats.")


# --- Colonne Logs ---
with col_log:
    st.subheader("📜 Logs")
    # Utiliser un expander pour les logs pour économiser de la place
    with st.expander("Afficher/Cacher les Logs", expanded=True):
        log_container = st.container(height=600) # Hauteur ajustable
        # Afficher les logs (plus récent en haut car ajoutés au début de la liste)
        for msg in st.session_state.get('log_messages', ["Aucun log."]):
            log_container.text(msg)

    if st.button("🧹 Effacer Logs", key="clear_logs_btn", use_container_width=True):
        st.session_state.log_messages = ["[Logs effacés]"]
        st.rerun()

# =============================================
# SECTION 4 : LOGIQUE DE MISE À JOUR ET RECALCUL
# =============================================

# Gérer le recalcul automatique si nécessaire (après undo, chargement, etc.)
if st.session_state.get('needs_rerun_calc', False):
    add_log("Déclenchement du recalcul automatique...")
    params = st.session_state.rerun_calc_params
    force_ep_val = params.get('force_ep') # Peut être None
    # Réinitialiser le flag AVANT d'appeler le wrapper pour éviter boucle infinie
    st.session_state.needs_rerun_calc = False
    st.session_state.rerun_calc_params = {}
    # Appeler le wrapper de calcul avec les paramètres sauvegardés
    run_calculation_wrapper(
        is_optimized_run=params.get('is_optimized_run', False),
        method_name=params.get('method_name', 'Recalcul Auto'),
        force_ep=force_ep_val
    )
    # Un rerun sera implicitement fait par Streamlit à la fin de ce script run

# Afficher un message d'état final (optionnel)
# st.sidebar.info(f"État: {state_desc}, MSE: {st.session_state.last_mse:.3e}" if st.session_state.last_mse else f"État: {state_desc}")
