# ============================================================
# thinfilm_optimizer_streamlit.py
# Conversion Streamlit de l'optimiseur de films minces JAX/Tkinter
# ============================================================

import streamlit as st
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.lax import scan, cond
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import json
import os
import datetime
import traceback
from scipy.optimize import minimize, OptimizeResult
import time
import functools
from typing import Union, Tuple, Dict, List, Any, Callable
import io
import copy  # Pour deep copy des états complexes si nécessaire

# --- Configuration JAX ---
# Activer le support float64 pour la précision nécessaire
jax.config.update("jax_enable_x64", True)

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Optimiseur Film Mince (JAX+Streamlit)",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Optimiseur de Films Minces\nVersion Streamlit basée sur le code original Tkinter/JAX."
    }
)

# --- CSS Personnalisé pour une interface plus compacte ---
# (Applique le style pour réduire tailles et espacements)
st.markdown("""
<style>
    /* Reduce font size for widgets and text */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div[role="button"],
    .stTextArea > div > textarea,
    .stButton > button,
    .stCheckbox > label > span,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stText"],
    div[role="tab"], /* Font size for tab labels */
    label[data-baseweb="label"] /* General labels */
     {
        font-size: 0.85rem !important; /* Smaller font */
    }
    /* Reduce padding around widgets */
    div[data-testid="stTextInput"],
    div[data-testid="stNumberInput"],
    div[data-testid="stSelectbox"],
    div[data-testid="stTextArea"],
    div[data-testid="stButton"],
    div[data-testid="stCheckbox"] {
        padding-top: 0.1rem !important;
        padding-bottom: 0.1rem !important;
        margin-bottom: -0.2rem !important; /* Reduce vertical margin */
    }
    /* Reduce padding inside buttons */
     .stButton > button {
        padding: 0.2rem 0.5rem !important;
        min-height: 1.8rem; /* Adjust button height */
        line-height: 1.4;
    }
     /* Adjust spacing in columns/sidebar specifically if needed */
    div[data-testid="stVerticalBlock"] > div[style*="gap: 1rem;"], /* Reduce gap between rows */
    div[data-testid="stSidebarNavItems"] /* Sidebar specific items if needed */
    {
         gap: 0.3rem !important; /* Smaller gap */
    }
     /* Make expander header smaller */
    .streamlit-expanderHeader {
        font-size: 0.9rem !important;
        padding: 0.3rem 0.5rem !important;
    }
     /* Reduce space after headers */
    h1, h2, h3, h4, h5, h6 {
        margin-bottom: 0.2rem !important;
        padding-bottom: 0 !important;
    }
    /* Reduce padding for target definition columns */
    div.stCheckbox { margin-right: -15px !important;} /* Pull checkbox closer */

</style>
""", unsafe_allow_html=True)


# ============================================================
# 1. CONSTANTES GLOBALES
# ============================================================
MIN_THICKNESS_PHYS_NM: float = 0.01
BASE_NEEDLE_THICKNESS_NM: float = 0.1
DEFAULT_NEEDLE_SCAN_STEP_NM: float = 2.0
AUTO_NEEDLES_PER_CYCLE: int = 5
AUTO_MAX_CYCLES: int = 5
MSE_IMPROVEMENT_TOLERANCE: float = 1e-9
EXCEL_FILE_PATH: str = "indices.xlsx" # Assurez-vous que ce fichier est accessible

# ============================================================
# 2. CŒUR MÉTIER : 20 PREMIÈRES FONCTIONS
# ============================================================

# --- Fonctions de gestion des matériaux ---

# Fonction 1: load_material_data_from_xlsx_sheet (avec cache Streamlit)
@st.cache_data(max_entries=64, ttl=3600) # Augmenter cache, TTL 1h
@functools.lru_cache(maxsize=64) # Garder lru_cache aussi
def load_material_data_from_xlsx_sheet(file_path: str, sheet_name: str) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:
    """Charge n et k depuis une feuille Excel. Gère les erreurs."""
    # Utilisation de print pour le log console (visible où Streamlit tourne)
    # car st.write/log_message ne fonctionnent pas bien dans les fonctions cachées.
    try:
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Le fichier Excel '{file_path}' est introuvable.")

        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        numeric_df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')

        if numeric_df.shape[1] < 3:
            raise ValueError(f"Feuille '{sheet_name}' requiert au moins 3 colonnes numériques (lambda, n, k).")

        # Utiliser les 3 premières colonnes, supprimer lignes avec NaN dans ces colonnes
        numeric_df = numeric_df.iloc[:, :3].dropna(subset=[0, 1, 2])
        numeric_df = numeric_df.sort_values(by=0) # Trier par longueur d'onde (colonne 0)

        l_nm = numeric_df.iloc[:, 0].values.astype(np.float64)
        n = numeric_df.iloc[:, 1].values.astype(np.float64)
        k = numeric_df.iloc[:, 2].values.astype(np.float64)

        if len(l_nm) == 0:
            raise ValueError(f"Pas de données numériques valides après filtrage dans '{sheet_name}'.")

        print(f"ST_CACHE: Données chargées depuis {file_path} (Feuille: '{sheet_name}'): {len(l_nm)} points de {l_nm.min():.1f} nm à {l_nm.max():.1f} nm")
        return l_nm, n, k

    except FileNotFoundError as fnf:
        print(f"ST_CACHE_ERROR: Fichier non trouvé: {fnf}")
        # Retourner None permet à l'appelant de gérer l'erreur
        return None, None, None
    except ValueError as ve:
        print(f"ST_CACHE_ERROR: Erreur de valeur lecture feuille '{sheet_name}': {ve}")
        return None, None, None
    except Exception as e:
        # Log plus détaillé pour les erreurs inattendues
        print(f"ST_CACHE_ERROR: Erreur inattendue lecture feuille '{sheet_name}': {type(e).__name__} - {e}")
        print(traceback.format_exc(limit=2))
        return None, None, None

# Fonction 2: get_n_fused_silica (JAX)
@jax.jit
def get_n_fused_silica(l_nm: jnp.ndarray) -> jnp.ndarray:
    """Calcule n+ik pour Fused Silica via loi de Sellmeier."""
    l_um_sq = (l_nm / 1000.0)**2
    # Éviter division par zéro ou valeurs très proches pour les pôles
    denom1 = l_um_sq - 0.0684043**2
    denom2 = l_um_sq - 0.1162414**2
    denom3 = l_um_sq - 9.896161**2
    # Remplacer les dénominations proches de zéro par une petite valeur signée
    safe_denom1 = jnp.where(jnp.abs(denom1) < 1e-12, jnp.sign(denom1) * 1e-12, denom1)
    safe_denom2 = jnp.where(jnp.abs(denom2) < 1e-12, jnp.sign(denom2) * 1e-12, denom2)
    safe_denom3 = jnp.where(jnp.abs(denom3) < 1e-12, jnp.sign(denom3) * 1e-12, denom3)
    n_sq = 1.0 + (0.6961663 * l_um_sq) / safe_denom1 + \
           (0.4079426 * l_um_sq) / safe_denom2 + \
           (0.8974794 * l_um_sq) / safe_denom3
    # Empêcher les valeurs négatives dans sqrt dues à l'instabilité numérique près des pôles
    n = jnp.sqrt(jnp.maximum(n_sq, 1e-12)) # Assurer n > 0
    k = jnp.zeros_like(n) # Pas d'absorption dans ce modèle
    return n + 1j * k

# Fonction 3: get_n_bk7 (JAX)
@jax.jit
def get_n_bk7(l_nm: jnp.ndarray) -> jnp.ndarray:
    """Calcule n+ik pour BK7 via loi de Sellmeier."""
    l_um_sq = (l_nm / 1000.0)**2
    denom1 = l_um_sq - 0.00600069867
    denom2 = l_um_sq - 0.0200179144
    denom3 = l_um_sq - 103.560653
    safe_denom1 = jnp.where(jnp.abs(denom1) < 1e-12, jnp.sign(denom1) * 1e-12, denom1)
    safe_denom2 = jnp.where(jnp.abs(denom2) < 1e-12, jnp.sign(denom2) * 1e-12, denom2)
    safe_denom3 = jnp.where(jnp.abs(denom3) < 1e-12, jnp.sign(denom3) * 1e-12, denom3)
    n_sq = 1.0 + (1.03961212 * l_um_sq) / safe_denom1 + \
           (0.231792344 * l_um_sq) / safe_denom2 + \
           (1.01046945 * l_um_sq) / safe_denom3
    n = jnp.sqrt(jnp.maximum(n_sq, 1e-12)) # Assurer n > 0
    k = jnp.zeros_like(n)
    return n + 1j * k

# Fonction 4: get_n_d263 (JAX)
@jax.jit
def get_n_d263(l_nm: jnp.ndarray) -> jnp.ndarray:
    """Retourne n+ik constant pour le verre D263."""
    n = jnp.full_like(l_nm, 1.523, dtype=jnp.float64)
    k = jnp.zeros_like(n)
    return n + 1j * k

# Fonction 5: interp_nk_cached (JAX)
@jax.jit
def interp_nk_cached(l_target: jnp.ndarray, l_data: jnp.ndarray, n_data: jnp.ndarray, k_data: jnp.ndarray) -> jnp.ndarray:
    """Interpole n et k en utilisant jnp.interp. Assure k >= 0."""
    # Assumer l_data est déjà trié (fait dans _get_nk_array_for_lambda_vec)
    n_interp = jnp.interp(l_target, l_data, n_data)
    k_interp = jnp.interp(l_target, l_data, k_data)
    # Assurer k >= 0 physiquement
    k_interp = jnp.maximum(k_interp, 0.0)
    return n_interp + 1j * k_interp

# Type Hint pour les définitions de matériaux
MaterialInputType = Union[complex, float, int, str, Tuple[np.ndarray, np.ndarray, np.ndarray]]

# Fonction 6: _get_nk_array_for_lambda_vec
def _get_nk_array_for_lambda_vec(material_definition: MaterialInputType,
                                 l_vec_target_jnp: jnp.ndarray) -> jnp.ndarray:
    """Retourne un tableau JAX d'indices complexes pour le vecteur lambda donné."""
    if isinstance(material_definition, (complex, float, int)):
        mat_complex = jnp.asarray(material_definition, dtype=jnp.complex128)
        # Assurer k >= 0
        mat_complex_corrected = jnp.where(mat_complex.imag < 0, mat_complex.real + 0j, mat_complex)
        return jnp.full(l_vec_target_jnp.shape, mat_complex_corrected)
    elif isinstance(material_definition, str):
        mat_upper = material_definition.upper()
        if mat_upper == "FUSED SILICA": return get_n_fused_silica(l_vec_target_jnp)
        if mat_upper == "BK7": return get_n_bk7(l_vec_target_jnp)
        if mat_upper == "D263": return get_n_d263(l_vec_target_jnp)
        else: # Assume nom de feuille Excel
            sheet_name = material_definition
            # Utilise la fonction cachée
            l_data, n_data, k_data = load_material_data_from_xlsx_sheet(EXCEL_FILE_PATH, sheet_name)
            if l_data is None:
                raise ValueError(f"Impossible de charger les données pour '{sheet_name}' depuis {EXCEL_FILE_PATH}")

            l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
            # Le tri est déjà fait dans la fonction load...
            min_target, max_target = jnp.min(l_vec_target_jnp), jnp.max(l_vec_target_jnp)
            min_data, max_data = l_data_jnp[0], l_data_jnp[-1]
            if min_target < min_data - 1e-6 or max_target > max_data + 1e-6:
                 print(f"AVERTISSEMENT Extrapolation: Plage cible [{min_target:.1f}-{max_target:.1f}] hors plage données [{min_data:.1f}-{max_data:.1f}] pour '{sheet_name}'.")
            return interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)
    elif isinstance(material_definition, tuple) and len(material_definition) == 3:
        l_data, n_data, k_data = material_definition
        l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
        sort_indices = jnp.argsort(l_data_jnp)
        l_data_jnp, n_data_jnp, k_data_jnp = l_data_jnp[sort_indices], n_data_jnp[sort_indices], k_data_jnp[sort_indices]
        min_target, max_target = jnp.min(l_vec_target_jnp), jnp.max(l_vec_target_jnp)
        min_data, max_data = l_data_jnp[0], l_data_jnp[-1]
        if min_target < min_data - 1e-6 or max_target > max_data + 1e-6:
             print(f"AVERTISSEMENT Extrapolation: Plage cible hors plage données tuple fournies.")
        return interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)
    else:
        raise TypeError(f"Type de définition de matériau non supporté : {type(material_definition)}")

# --- Moteur de Calcul JAX : Fonctions Core ---

# Fonction 7: _compute_layer_matrix_scan_step (JAX)
@jax.jit
def _compute_layer_matrix_scan_step(carry_matrix: jnp.ndarray, layer_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, None]:
    """Calcule la matrice d'une couche et la multiplie avec la matrice précédente (pour jax.lax.scan)."""
    thickness, Ni_complex, l_val = layer_data
    eta = Ni_complex
    safe_l_val = jnp.maximum(l_val, 1e-9)
    phi = (2 * jnp.pi / safe_l_val) * (Ni_complex * thickness)
    cos_phi = jnp.cos(phi); sin_phi = jnp.sin(phi)

    def compute_M_layer(eta_: jnp.complex128) -> jnp.ndarray:
        safe_eta = jnp.where(jnp.abs(eta_) < 1e-12, 1e-12 + 0j, eta_)
        M_layer = jnp.array([[cos_phi, (1j / safe_eta) * sin_phi], [1j * eta_ * sin_phi, cos_phi]], dtype=jnp.complex128)
        return M_layer @ carry_matrix

    def compute_identity(eta_: jnp.complex128) -> jnp.ndarray:
        return carry_matrix

    new_matrix = cond(thickness > 1e-12, compute_M_layer, compute_identity, eta)
    return new_matrix, None

# Fonction 8: compute_stack_matrix_jax (JAX)
@jax.jit
def compute_stack_matrix_jax(ep_vector: jnp.ndarray, l_val: jnp.ndarray,
                             nH_at_lval: jnp.complex128, nL_at_lval: jnp.complex128) -> jnp.ndarray:
    """Calcule la matrice caractéristique totale pour une structure H/L à une longueur d'onde."""
    num_layers = ep_vector.shape[0]
    if num_layers == 0: return jnp.eye(2, dtype=jnp.complex128)
    is_H_layer = jnp.arange(num_layers) % 2 == 0
    layer_indices = jnp.where(is_H_layer, nH_at_lval, nL_at_lval)
    l_val_broadcasted = jnp.full(num_layers, l_val)
    layers_scan_data = (ep_vector, layer_indices, l_val_broadcasted)
    M_initial = jnp.eye(2, dtype=jnp.complex128)
    M_final, _ = scan(_compute_layer_matrix_scan_step, M_initial, layers_scan_data)
    return M_final

# Fonction 9: calculate_single_wavelength_T (JAX)
@jax.jit
def calculate_single_wavelength_T(l_val: jnp.ndarray, ep_vector_contig: jnp.ndarray,
                                  nH_at_lval: jnp.complex128, nL_at_lval: jnp.complex128, nSub_at_lval: jnp.complex128) -> jnp.ndarray:
    """Calcule la Transmittance T pour une seule longueur d'onde (structure H/L)."""
    etainc = 1.0 + 0j
    etasub = nSub_at_lval

    def calculate_for_valid_l(l_: jnp.ndarray) -> jnp.ndarray:
        M = compute_stack_matrix_jax(ep_vector_contig, l_, nH_at_lval, nL_at_lval)
        m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        rs_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        rs_denominator_abs = jnp.abs(rs_denominator)
        safe_denominator = jnp.where(rs_denominator_abs < 1e-12, 1e-12 + 0j, rs_denominator)
        ts = (2.0 * etainc) / safe_denominator
        safe_real_etasub = jnp.maximum(jnp.real(etasub), 0.0)
        Ts = (safe_real_etasub / 1.0) * jnp.abs(ts)**2
        return jnp.where(rs_denominator_abs < 1e-12, jnp.nan, Ts)

    def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray:
        return jnp.nan

    Ts = cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)
    return Ts

# Fonction 10: calculate_T_from_ep_core_jax (JAX)
@jax.jit
def calculate_T_from_ep_core_jax(ep_vector: jnp.ndarray,
                                 nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                 l_vec: jnp.ndarray) -> jnp.ndarray:
    """Calcule T(lambda) pour un vecteur de lambdas (structure H/L)."""
    if not l_vec.size: return jnp.zeros(0, dtype=jnp.float64)
    ep_vector_contig = jnp.asarray(ep_vector)
    Ts_arr = vmap(calculate_single_wavelength_T, in_axes=(0, None, 0, 0, 0))(
        l_vec, ep_vector_contig, nH_arr, nL_arr, nSub_arr)
    return jnp.nan_to_num(Ts_arr, nan=0.0)

# --- Fonctions Core pour Séquence Arbitraire (JAX) ---

# Fonction 11: _compute_layer_matrix_scan_step_arbitrary (JAX)
@jax.jit
def _compute_layer_matrix_scan_step_arbitrary(carry_matrix: jnp.ndarray, layer_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, None]:
    """Étape de scan pour séquence arbitraire."""
    return _compute_layer_matrix_scan_step(carry_matrix, layer_data)

# Fonction 12: compute_stack_matrix_arbitrary_jax (JAX)
@jax.jit
def compute_stack_matrix_arbitrary_jax(ep_vector: jnp.ndarray, layer_indices_at_lval: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    """Calcule la matrice totale pour une séquence arbitraire à une longueur d'onde."""
    num_layers = ep_vector.shape[0]
    if num_layers == 0: return jnp.eye(2, dtype=jnp.complex128)
    l_val_broadcasted = jnp.full(num_layers, l_val)
    layers_scan_data = (ep_vector, layer_indices_at_lval, l_val_broadcasted)
    M_initial = jnp.eye(2, dtype=jnp.complex128)
    M_final, _ = scan(_compute_layer_matrix_scan_step_arbitrary, M_initial, layers_scan_data)
    return M_final

# Fonction 13: calculate_single_wavelength_T_arbitrary (JAX)
@jax.jit
def calculate_single_wavelength_T_arbitrary(l_val: jnp.ndarray, ep_vector_contig: jnp.ndarray,
                                            layer_indices_at_lval: jnp.ndarray, nSub_at_lval: jnp.complex128) -> jnp.ndarray:
    """Calcule T pour une seule lambda, séquence arbitraire."""
    etainc = 1.0 + 0j; etasub = nSub_at_lval
    def calculate_for_valid_l(l_: jnp.ndarray) -> jnp.ndarray:
        M = compute_stack_matrix_arbitrary_jax(ep_vector_contig, layer_indices_at_lval, l_)
        m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        rs_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        rs_denominator_abs = jnp.abs(rs_denominator)
        safe_denominator = jnp.where(rs_denominator_abs < 1e-12, 1e-12 + 0j, rs_denominator)
        ts = (2.0 * etainc) / safe_denominator
        safe_real_etasub = jnp.maximum(jnp.real(etasub), 0.0)
        Ts = (safe_real_etasub / 1.0) * jnp.abs(ts)**2
        return jnp.where(rs_denominator_abs < 1e-12, jnp.nan, Ts)
    def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray: return jnp.nan
    Ts = cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)
    return Ts

# Fonction 14: calculate_T_from_ep_arbitrary_core_jax (JAX)
@jax.jit
def calculate_T_from_ep_arbitrary_core_jax(ep_vector: jnp.ndarray,
                                           layer_indices_arr: jnp.ndarray, # Shape (num_layers, num_lambdas)
                                           nSub_arr: jnp.ndarray,          # Shape (num_lambdas,)
                                           l_vec: jnp.ndarray) -> jnp.ndarray: # Shape (num_lambdas,)
    """Calcule T(lambda) pour un vecteur lambda, séquence arbitraire."""
    if not l_vec.size: return jnp.zeros(0, dtype=jnp.float64)
    ep_vector_contig = jnp.asarray(ep_vector)
    num_layers = ep_vector_contig.shape[0]
    if num_layers == 0:
        etainc = 1.0 + 0j; etasub_batch = nSub_arr
        ts_bare = 2 * etainc / (etainc + etasub_batch)
        safe_real_etasub = jnp.maximum(jnp.real(etasub_batch), 0.0)
        Ts_arr = (safe_real_etasub / 1.0) * jnp.abs(ts_bare)**2
    else:
        # Assurer que les dimensions sont cohérentes (difficile dans JIT, mais important)
        # if layer_indices_arr.shape[0] != num_layers: -> Gérer à l'appel
        layer_indices_arr_transposed = layer_indices_arr.T # Shape (num_lambdas, num_layers)
        Ts_arr = vmap(calculate_single_wavelength_T_arbitrary, in_axes=(0, None, 0, 0))(
            l_vec, ep_vector_contig, layer_indices_arr_transposed, nSub_arr)
    return jnp.nan_to_num(Ts_arr, nan=0.0)

# Fonction 15: _get_nk_at_lambda
def _get_nk_at_lambda(material_definition: MaterialInputType, l_nm_target: float) -> complex:
    """Obtient n+ik pour un matériau donné à une longueur d'onde spécifique."""
    if isinstance(material_definition, (complex, float, int)):
        mat_complex = complex(material_definition)
        return complex(mat_complex.real, max(0.0, mat_complex.imag)) # Assurer k>=0

    l_nm_target_jnp = jnp.array([l_nm_target], dtype=jnp.float64) # Pour Sellmeier

    if isinstance(material_definition, str):
        mat_upper = material_definition.upper()
        if mat_upper == "FUSED SILICA": return complex(get_n_fused_silica(l_nm_target_jnp)[0])
        if mat_upper == "BK7": return complex(get_n_bk7(l_nm_target_jnp)[0])
        if mat_upper == "D263": return complex(get_n_d263(l_nm_target_jnp)[0])
        else: # Feuille Excel
            sheet_name = material_definition
            l_data, n_data, k_data = load_material_data_from_xlsx_sheet(EXCEL_FILE_PATH, sheet_name)
            if l_data is None: raise ValueError(f"Impossible charger données pour '{sheet_name}'")
            sort_idx = np.argsort(l_data)
            l_data_s, n_data_s, k_data_s = l_data[sort_idx], n_data[sort_idx], k_data[sort_idx]
            # Vérifier si dans la plage avant interp NumPy
            if not (l_data_s[0] <= l_nm_target <= l_data_s[-1]):
                 print(f"AVERTISSEMENT: l0={l_nm_target} hors plage données [{l_data_s[0]:.1f}-{l_data_s[-1]:.1f}] pour '{sheet_name}'. Extrapolation NumPy.")
            n_interp = np.interp(l_nm_target, l_data_s, n_data_s)
            k_interp = np.interp(l_nm_target, l_data_s, k_data_s)
            return complex(n_interp, max(0.0, k_interp))
    elif isinstance(material_definition, tuple) and len(material_definition) == 3:
        l_data, n_data, k_data = material_definition
        sort_idx = np.argsort(l_data)
        l_data_s, n_data_s, k_data_s = l_data[sort_idx], n_data[sort_idx], k_data[sort_idx]
        if not (l_data_s[0] <= l_nm_target <= l_data_s[-1]):
             print(f"AVERTISSEMENT: l0={l_nm_target} hors plage données tuple fournies. Extrapolation NumPy.")
        n_interp = np.interp(l_nm_target, l_data_s, n_data_s)
        k_interp = np.interp(l_nm_target, l_data_s, k_data_s)
        return complex(n_interp, max(0.0, k_interp))
    else:
        raise TypeError(f"Type définition matériau non supporté : {type(material_definition)}")

# Fonction 16: get_target_points_indices_jax (JAX Helper)
@jax.jit
def get_target_points_indices_jax(l_vec: jnp.ndarray, target_min: float, target_max: float) -> jnp.ndarray:
    """Trouve les indices des points de l_vec dans l'intervalle [target_min, target_max]."""
    if not l_vec.size: return jnp.empty(0, dtype=jnp.int64)
    indices_with_fill = jnp.where((l_vec >= target_min) & (l_vec <= target_max),
                                  jnp.arange(l_vec.shape[0]), -1)
    valid_indices = indices_with_fill[indices_with_fill != -1]
    return valid_indices.astype(jnp.int64)

# Fonction 17: calculate_initial_ep
def calculate_initial_ep(emp_qwot_list: List[float], l0: float,
                         nH0_material: MaterialInputType, nL0_material: MaterialInputType) -> np.ndarray:
    """Calcule les épaisseurs physiques initiales à partir des multiplicateurs QWOT."""
    num_layers = len(emp_qwot_list)
    ep_initial = np.zeros(num_layers, dtype=np.float64)
    if l0 <= 0:
        print("AVERTISSEMENT: l0 <= 0 dans calculate_initial_ep. Épaisseurs mises à 0.")
        return ep_initial
    try:
        nH_complex_at_l0 = _get_nk_at_lambda(nH0_material, l0)
        nL_complex_at_l0 = _get_nk_at_lambda(nL0_material, l0)
        nH_real_at_l0 = nH_complex_at_l0.real
        nL_real_at_l0 = nL_complex_at_l0.real
        if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
            print(f"AVERTISSEMENT: Indice réel nH({nH_real_at_l0:.3f}) ou nL({nL_real_at_l0:.3f}) à l0={l0}nm <= 0.")
    except ValueError as e:
         print(f"ERREUR obtention indices à l0={l0}nm pour calcul initial: {e}")
         raise ValueError(f"Erreur obtention indices à l0={l0}nm: {e}") from e
    except Exception as e:
        print(f"ERREUR inattendue obtention indices à l0={l0}nm : {e}")
        raise RuntimeError(f"Erreur inattendue obtention indices à l0={l0}nm : {e}") from e

    for i in range(num_layers):
        multiplier = emp_qwot_list[i]
        if multiplier < 0: # Vérifier les multiplicateurs négatifs
            raise ValueError(f"Multiplicateur QWOT négatif trouvé à l'indice {i}: {multiplier}")
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real_layer_at_l0 > 1e-9:
            ep_initial[i] = multiplier * l0 / (4.0 * n_real_layer_at_l0)
        else: ep_initial[i] = 0.0
    return ep_initial

# Fonction 18: calculate_qwot_from_ep
def calculate_qwot_from_ep(ep_vector: np.ndarray, l0: float,
                           nH0_material: MaterialInputType, nL0_material: MaterialInputType) -> np.ndarray:
    """Calcule les multiplicateurs QWOT à partir des épaisseurs physiques."""
    num_layers = len(ep_vector)
    qwot_multipliers = np.full(num_layers, np.nan, dtype=np.float64)
    if l0 <= 0:
        print("AVERTISSEMENT: l0 <= 0 dans calculate_qwot_from_ep. QWOT mis à NaN.")
        return qwot_multipliers
    try:
        nH_complex_at_l0 = _get_nk_at_lambda(nH0_material, l0)
        nL_complex_at_l0 = _get_nk_at_lambda(nL0_material, l0)
        nH_real_at_l0 = nH_complex_at_l0.real
        nL_real_at_l0 = nL_complex_at_l0.real
        if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
            print(f"AVERTISSEMENT: Indice réel nH({nH_real_at_l0:.3f}) ou nL({nL_real_at_l0:.3f}) à l0={l0}nm <= 0.")
    except ValueError as e:
         print(f"ERREUR obtention indices à l0={l0}nm pour calcul QWOT: {e}")
         raise ValueError(f"Erreur obtention indices à l0={l0}nm pour QWOT: {e}") from e
    except Exception as e:
        print(f"ERREUR inattendue obtention indices à l0={l0}nm pour QWOT: {e}")
        raise RuntimeError(f"Erreur inattendue obtention indices à l0={l0}nm pour QWOT: {e}") from e

    for i in range(num_layers):
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        # Utiliser une tolérance plus stricte pour éviter division par zéro
        if n_real_layer_at_l0 > 1e-9 and l0 > 1e-9:
            qwot_multipliers[i] = ep_vector[i] * (4.0 * n_real_layer_at_l0) / l0
    return qwot_multipliers

# Fonction 19: calculate_final_mse
def calculate_final_mse(res: Dict[str, np.ndarray], active_targets: List[Dict]) -> Tuple[Union[float, None], int]:
    """Calcule le MSE final basé sur les résultats et les cibles actives (version NumPy)."""
    total_squared_error = 0.0
    total_points_in_targets = 0
    mse = None

    if not active_targets or not res or 'Ts' not in res or res['Ts'] is None or 'l' not in res or res['l'] is None:
        print("AVERTISSEMENT: Entrées invalides pour calculate_final_mse.")
        return mse, total_points_in_targets

    res_l_np = np.asarray(res['l'])
    res_ts_np = np.asarray(res['Ts'])

    if res_l_np.size == 0 or res_ts_np.size == 0 or res_l_np.size != res_ts_np.size:
         print("AVERTISSEMENT: Données résultats vides/incohérentes pour calculate_final_mse.")
         return mse, total_points_in_targets

    for target in active_targets:
        try:
            l_min, l_max = float(target['min']), float(target['max'])
            t_min, t_max = float(target['target_min']), float(target['target_max'])
            if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0):
                 print(f"AVERTISSEMENT: Transmittance cible hors [0,1] ignorée: {target}")
                 continue
            if l_max < l_min:
                 print(f"AVERTISSEMENT: Cible avec l_max < l_min ignorée: {target}")
                 continue
        except (KeyError, ValueError, TypeError):
             print(f"AVERTISSEMENT: Cible invalide ignorée dans calculate_final_mse: {target}")
             continue

        indices = np.where((res_l_np >= l_min) & (res_l_np <= l_max))[0]
        if indices.size > 0:
            calculated_Ts_in_zone = res_ts_np[indices]
            target_lambdas_in_zone = res_l_np[indices]
            finite_mask = np.isfinite(calculated_Ts_in_zone)
            calculated_Ts_in_zone = calculated_Ts_in_zone[finite_mask]
            target_lambdas_in_zone = target_lambdas_in_zone[finite_mask]
            if calculated_Ts_in_zone.size == 0: continue

            if abs(l_max - l_min) < 1e-9:
                interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
            else:
                slope = (t_max - t_min) / (l_max - l_min)
                interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)
                interpolated_target_t = np.clip(interpolated_target_t, 0.0, 1.0)

            squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
            total_squared_error += np.sum(squared_errors)
            total_points_in_targets += len(calculated_Ts_in_zone)

    if total_points_in_targets > 0:
        mse = total_squared_error / total_points_in_targets
    elif active_targets:
        mse = np.nan

    return mse, total_points_in_targets

# Fonction 20: calculate_mse_for_optimization_penalized_jax (JAX Cost Function)
@jax.jit
def calculate_mse_for_optimization_penalized_jax(ep_vector: jnp.ndarray,
                                                nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                                l_vec_optim: jnp.ndarray,
                                                active_targets_tuple: Tuple[Tuple[float, float, float, float], ...],
                                                min_thickness_phys_nm: float) -> jnp.ndarray:
    """Fonction de coût JIT pour l'optimisation (MSE + pénalité couche mince)."""
    # Pénalité Couche Mince
    below_min_mask = (ep_vector < min_thickness_phys_nm) & (ep_vector > 1e-12)
    relative_thinness_sq = jnp.where(below_min_mask, (min_thickness_phys_nm - ep_vector) / min_thickness_phys_nm, 0.0)**2
    penalty_factor = 100.0 # Facteur de pénalité (ajustable)
    penalty_thin = jnp.sum(relative_thinness_sq) * penalty_factor

    # Calcul MSE
    ep_vector_calc = jnp.maximum(ep_vector, min_thickness_phys_nm) # Clamper épaisseurs
    Ts = calculate_T_from_ep_core_jax(ep_vector_calc, nH_arr, nL_arr, nSub_arr, l_vec_optim)

    total_squared_error = 0.0; total_points_in_targets = 0
    for i in range(len(active_targets_tuple)):
        l_min, l_max, t_min, t_max = active_targets_tuple[i]
        target_mask = (l_vec_optim >= l_min) & (l_vec_optim <= l_max)
        is_flat_target = jnp.abs(l_max - l_min) < 1e-9
        denom_slope = jnp.where(is_flat_target, 1.0, l_max - l_min)
        slope = jnp.where(is_flat_target, 0.0, (t_max - t_min) / denom_slope)
        interpolated_target_t_full = jnp.clip(t_min + slope * (l_vec_optim - l_min), 0.0, 1.0)
        ts_finite = jnp.nan_to_num(Ts, nan=0.0) # Assurer T finie
        squared_errors_full = (ts_finite - interpolated_target_t_full)**2
        masked_sq_error = jnp.where(target_mask & jnp.isfinite(squared_errors_full), squared_errors_full, 0.0)
        total_squared_error += jnp.sum(masked_sq_error)
        total_points_in_targets += jnp.sum(target_mask)

    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, 1e10) # MSE élevé si pas de points

    # Coût Final
    final_cost = mse + penalty_thin
    return jnp.nan_to_num(final_cost, nan=jnp.inf) # Robuste aux NaN

# --- FIN DES 20 PREMIÈRES FONCTIONS ---
# [PREVIOUS CODE HERE - Imports, Config, CSS, Constants, Functions 1-20]
# ... (Assume lines 1-approx 420 are present)

# ============================================================
# 2. CŒUR MÉTIER : FONCTIONS SUIVANTES (21+)
# ============================================================

# Fonction 21: calculate_mse_arbitrary_sequence_jax (JAX Cost - Non pénalisé)
@jax.jit
def calculate_mse_arbitrary_sequence_jax(ep_vector: jnp.ndarray,
                                         layer_indices_arr: jnp.ndarray, # Shape (num_layers, num_lambdas)
                                         nSub_arr: jnp.ndarray,          # Shape (num_lambdas,)
                                         l_vec_eval: jnp.ndarray,        # Shape (num_lambdas,)
                                         active_targets_tuple: Tuple[Tuple[float, float, float, float], ...]) -> jnp.ndarray:
    """Calcule le MSE pour une séquence arbitraire (sans pénalité couche mince)."""
    # Calculer Transmittance avec la fonction core arbitraire
    Ts = calculate_T_from_ep_arbitrary_core_jax(ep_vector, layer_indices_arr, nSub_arr, l_vec_eval)

    total_squared_error = 0.0
    total_points_in_targets = 0

    # Boucle sur les zones cibles actives (similaire à la fonction pénalisée)
    for i in range(len(active_targets_tuple)):
        l_min, l_max, t_min, t_max = active_targets_tuple[i]
        target_mask = (l_vec_eval >= l_min) & (l_vec_eval <= l_max)
        is_flat_target = jnp.abs(l_max - l_min) < 1e-9
        denom_slope = jnp.where(is_flat_target, 1.0, l_max - l_min)
        slope = jnp.where(is_flat_target, 0.0, (t_max - t_min) / denom_slope)
        interpolated_target_t_full = t_min + slope * (l_vec_eval - l_min)
        interpolated_target_t_full = jnp.clip(interpolated_target_t_full, 0.0, 1.0)

        ts_finite = jnp.nan_to_num(Ts, nan=0.0) # Assurer Ts finie
        squared_errors_full = (ts_finite - interpolated_target_t_full)**2
        masked_sq_error = jnp.where(target_mask & jnp.isfinite(squared_errors_full), squared_errors_full, 0.0)
        total_squared_error += jnp.sum(masked_sq_error)
        total_points_in_targets += jnp.sum(target_mask)

    # Calculer MSE, par défaut l'infinité si aucun point dans les cibles
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, jnp.inf)

    # Retourner MSE, gérant les NaN potentiels
    return jnp.nan_to_num(mse, nan=jnp.inf)


# Fonction 22: _run_core_optimization
def _run_core_optimization(ep_start_optim: np.ndarray,
                           inputs: Dict, active_targets: List[Dict],
                           nH_material: MaterialInputType,
                           nL_material: MaterialInputType,
                           nSub_material: MaterialInputType,
                           min_thickness_phys: float, log_prefix: str = "") -> Tuple[np.ndarray, bool, float, List[str], str, int, int]:
    """Exécute l'optimisation Scipy avec gradient JAX (Version adaptée retournant les logs)."""
    logs = []
    num_layers_start = len(ep_start_optim)
    final_ep = np.array(ep_start_optim, dtype=np.float64)
    optim_success = False; final_cost = np.inf; result_message_str = "Opt pas lancée."; nit_total = 0; nfev_total = 0

    if num_layers_start == 0:
        logs.append(f"{log_prefix}Structure vide, optimisation impossible.")
        return final_ep, False, np.inf, logs, "Structure vide", 0, 0

    try:
        l_min_optim, l_max_optim, l_step_optim = inputs['l_range_deb'], inputs['l_range_fin'], inputs['l_step']
        num_pts_optim = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
        l_vec_optim_np = np.geomspace(l_min_optim, l_max_optim, num_pts_optim)
        l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
        if not l_vec_optim_np.size: raise ValueError("Échec génération vecteur lambda pour optimisation.")
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)

        logs.append(f"{log_prefix} Préparation indices dispersifs pour {len(l_vec_optim_jax)} lambdas...")
        prep_start_time = time.time()
        # Utiliser les fonctions _get_nk... qui peuvent lever des erreurs si chargement échoue
        nH_arr_optim = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax)
        nL_arr_optim = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax)
        nSub_arr_optim = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax)
        logs.append(f"{log_prefix} Préparation indices terminée en {time.time() - prep_start_time:.3f}s.")

        active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
        static_args_for_jax = (nH_arr_optim, nL_arr_optim, nSub_arr_optim, l_vec_optim_jax, active_targets_tuple, min_thickness_phys)
        value_and_grad_fn = jax.jit(jax.value_and_grad(calculate_mse_for_optimization_penalized_jax))

        def scipy_obj_grad_wrapper(ep_vector_np, *args):
            ep_vector_jax = jnp.asarray(ep_vector_np)
            value_jax, grad_jax = value_and_grad_fn(ep_vector_jax, *args)
            value_np = float(np.array(value_jax))
            grad_np = np.array(grad_jax, dtype=np.float64)
            return value_np, grad_np

        lbfgsb_bounds = [(min_thickness_phys, None)] * num_layers_start
        options = {'maxiter': inputs['maxiter'], 'maxfun': inputs['maxfun'], 'disp': False, 'ftol': 1e-12, 'gtol': 1e-8}

        logs.append(f"{log_prefix}Lancement L-BFGS-B (grad JAX) - maxiter={options['maxiter']}...")
        opt_start_time = time.time()
        result = minimize(scipy_obj_grad_wrapper, np.asarray(ep_start_optim, dtype=np.float64),
                          args=static_args_for_jax, method='L-BFGS-B', jac=True, bounds=lbfgsb_bounds, options=options)
        logs.append(f"{log_prefix}L-BFGS-B (grad JAX) terminé en {time.time() - opt_start_time:.3f}s.")

        final_cost = result.fun if np.isfinite(result.fun) else np.inf
        result_message_str = result.message.decode('utf-8') if isinstance(result.message, bytes) else str(result.message)
        nit_total = result.nit if hasattr(result, 'nit') else 0
        nfev_total = result.nfev if hasattr(result, 'nfev') else 0
        is_success_or_limit = (result.success or result.status == 1) and np.isfinite(final_cost)

        if is_success_or_limit:
            final_ep = np.maximum(result.x, min_thickness_phys)
            optim_success = True
            log_status = "Limite atteinte" if result.status == 1 else "Succès"
            logs.append(f"{log_prefix}Optimisation {log_status}! Coût: {final_cost:.3e}, Iter: {nit_total}, Eval: {nfev_total}, Msg: {result_message_str}")
        else: # Echec
            optim_success = False
            final_ep = np.maximum(ep_start_optim, min_thickness_phys)
            logs.append(f"{log_prefix}Optimisation ÉCHOUÉE. Coût: {final_cost:.3e}, Statut: {result.status}, Iter: {nit_total}, Eval: {nfev_total}, Msg: {result_message_str}")
            try:
                reverted_cost, _ = scipy_obj_grad_wrapper(final_ep, *static_args_for_jax)
                logs.append(f"{log_prefix}Reverti à la structure initiale (clampée). Coût recalculé: {reverted_cost:.3e}")
                final_cost = reverted_cost if np.isfinite(reverted_cost) else np.inf
            except Exception as cost_e:
                logs.append(f"{log_prefix}Reverti. ERREUR recalcul coût: {cost_e}")
                final_cost = np.inf
            nit_total, nfev_total = 0, 0

    except ValueError as ve: # Erreurs de validation/préparation (ex: chargement matériau)
        logs.append(f"{log_prefix}ERREUR préparation optimisation: {ve}")
        # final_ep reste ep_start_optim (ou vide si erreur avant sa déf.)
        if 'ep_start_optim' in locals(): final_ep = np.maximum(ep_start_optim, min_thickness_phys)
        else: final_ep = np.array([])
        optim_success = False; final_cost = np.inf
        result_message_str = f"Erreur Prép: {ve}"
        nit_total, nfev_total = 0, 0
    except Exception as e_optim:
        logs.append(f"{log_prefix}ERREUR inattendue durant optimisation: {e_optim}\n{traceback.format_exc(limit=1)}")
        if 'ep_start_optim' in locals(): final_ep = np.maximum(ep_start_optim, min_thickness_phys)
        else: final_ep = np.array([])
        optim_success = False; final_cost = np.inf
        result_message_str = f"Exception: {e_optim}"
        nit_total, nfev_total = 0, 0

    return final_ep, optim_success, final_cost, logs, result_message_str, nit_total, nfev_total

# Fonction 23: _perform_layer_merge_or_removal_only_st (Adaptée)
def _perform_layer_merge_or_removal_only_st(ep_vector_in: np.ndarray, min_thickness_phys: float,
                                            log_prefix: str = "", target_layer_index: Union[int, None] = None,
                                            threshold_for_removal: Union[float, None] = None) -> Tuple[np.ndarray, bool, List[str]]:
    """Version adaptée pour Streamlit de la fusion/suppression de couche."""
    current_ep = ep_vector_in.copy()
    logs = []
    num_layers = len(current_ep)
    structure_changed = False
    ep_after_merge = None

    if num_layers <= 2 and target_layer_index is None:
        logs.append(f"{log_prefix}Structure <= 2 couches. Fusion/suppression impossible sans cible spécifique.")
        return current_ep, False, logs
    elif num_layers < 1:
        logs.append(f"{log_prefix}Structure vide.")
        return current_ep, False, logs

    try:
        thin_layer_index = -1
        min_thickness_found = np.inf

        # Déterminer la couche cible
        if target_layer_index is not None:
            if 0 <= target_layer_index < num_layers:
                if current_ep[target_layer_index] >= min_thickness_phys:
                    thin_layer_index = target_layer_index
                    min_thickness_found = current_ep[target_layer_index]
                    logs.append(f"{log_prefix}Ciblage couche {thin_layer_index + 1} ({min_thickness_found:.3f} nm).")
                else:
                    logs.append(f"{log_prefix}Couche cible {target_layer_index+1} trop fine (< {min_thickness_phys:.3f} nm). Recherche de la plus fine.")
                    target_layer_index = None
            else:
                 logs.append(f"{log_prefix}Indice couche cible {target_layer_index+1} invalide. Recherche de la plus fine.")
                 target_layer_index = None

        if target_layer_index is None:
            candidate_indices = np.where(current_ep >= min_thickness_phys)[0]
            if candidate_indices.size > 0:
                candidate_thicknesses = current_ep[candidate_indices]
                if threshold_for_removal is not None:
                    valid_for_removal_mask = candidate_thicknesses < threshold_for_removal
                    if np.any(valid_for_removal_mask):
                        candidate_indices = candidate_indices[valid_for_removal_mask]
                        candidate_thicknesses = candidate_thicknesses[valid_for_removal_mask]
                    else: candidate_indices = np.array([], dtype=int)
                if candidate_indices.size > 0:
                    min_idx_local = np.argmin(candidate_thicknesses)
                    thin_layer_index = candidate_indices[min_idx_local]
                    min_thickness_found = candidate_thicknesses[min_idx_local]

        # Exécuter Suppression/Fusion
        if thin_layer_index == -1:
            if threshold_for_removal is not None and target_layer_index is None: logs.append(f"{log_prefix}Aucune couche trouvée >= {min_thickness_phys:.3f} nm ET < {threshold_for_removal:.3f} nm.")
            else: logs.append(f"{log_prefix}Aucune couche valide (>= {min_thickness_phys:.3f} nm) trouvée pour suppression/fusion.")
            return current_ep, False, logs

        thin_layer_thickness = current_ep[thin_layer_index]
        log_details = f"Couche {thin_layer_index + 1} ({thin_layer_thickness:.3f} nm)"
        if threshold_for_removal is not None and target_layer_index is None : logs.append(f"{log_prefix}Trouvé couche sous seuil: {log_details}.")
        elif target_layer_index is None: logs.append(f"{log_prefix}Couche la plus fine: {log_details}.")

        merged_info = ""
        if thin_layer_index == 0: # Première couche
            if num_layers >= 2:
                ep_after_merge = current_ep[2:]; merged_info = "Supprimé couches 1 & 2."
                logs.append(f"{log_prefix}{merged_info} Nouvelle taille: {len(ep_after_merge)}."); structure_changed = True
            else: logs.append(f"{log_prefix}Impossible supprimer couches 1 & 2."); return current_ep, False, logs
        elif thin_layer_index == num_layers - 1: # Dernière couche
            ep_after_merge = current_ep[:-1]; merged_info = f"Supprimé dernière couche {num_layers}."
            logs.append(f"{log_prefix}{merged_info} Nouvelle taille: {len(ep_after_merge)}."); structure_changed = True
        else: # Couche interne
            merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
            ep_after_merge = np.concatenate((current_ep[:thin_layer_index - 1], [merged_thickness], current_ep[thin_layer_index + 2:]))
            merged_info = f"Supprimé couche {thin_layer_index+1}, fusionné {thin_layer_index} & {thin_layer_index+2} -> {merged_thickness:.3f}"
            logs.append(f"{log_prefix}{merged_info} Nouvelle taille: {len(ep_after_merge)}."); structure_changed = True

        # Finalisation
        if structure_changed and ep_after_merge is not None:
            if ep_after_merge.size > 0: ep_after_merge = np.maximum(ep_after_merge, min_thickness_phys)
            return ep_after_merge, True, logs
        else:
            if structure_changed: logs.append(f"{log_prefix}Changement structure indiqué mais résultat None.")
            return current_ep, False, logs

    except Exception as e_merge:
         logs.append(f"{log_prefix}ERREUR durant logique fusion/suppression: {e_merge}\n{traceback.format_exc(limit=1)}")
         return current_ep, False, logs

# Fonction 24: _perform_needle_insertion_scan_st (Adaptée)
def _perform_needle_insertion_scan_st(ep_vector_in: np.ndarray,
                                    nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
                                    l_vec_optim_np: np.ndarray, active_targets: List[Dict],
                                    cost_function_jax_compiled: Callable, # Passer fonction JITée
                                    min_thickness_phys: float, base_needle_thickness_nm: float,
                                    scan_step: float, l0_repr: float, log_prefix: str = "") -> Tuple[Union[np.ndarray, None], float, List[str], int]:
    """Version adaptée pour Streamlit du scan d'insertion d'aiguille."""
    logs = []
    num_layers_in = len(ep_vector_in)
    if num_layers_in == 0:
        logs.append(f"{log_prefix}Structure vide, scan aiguille impossible.")
        return None, np.inf, logs, -1

    logs.append(f"{log_prefix}Scan aiguille sur {num_layers_in} couches. Pas: {scan_step} nm, Ép. aiguille: {base_needle_thickness_nm:.3f} nm.")

    try:
        # Préparation données (indices pré-calculés à l'extérieur, passer juste cost_fn)
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        # Préparer seulement les arguments statiques nécessaires pour la fonction coût JITée
        nH_arr_optim = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax)
        nL_arr_optim = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax)
        nSub_arr_optim = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax)
        active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
        static_args_cost_fn = (nH_arr_optim, nL_arr_optim, nSub_arr_optim, l_vec_optim_jax, active_targets_tuple, min_thickness_phys)

        # Calculer coût initial
        initial_cost_jax = cost_function_jax_compiled(jnp.asarray(ep_vector_in), *static_args_cost_fn)
        initial_cost = float(np.array(initial_cost_jax))
        if not np.isfinite(initial_cost): raise ValueError("Coût initial non fini.")
        logs.append(f"{log_prefix}Coût initial: {initial_cost:.6e}")

    except Exception as e_prep:
        logs.append(f"{log_prefix}ERREUR préparation scan aiguille: {e_prep}")
        return None, np.inf, logs, -1

    best_ep_found = None; min_cost_found = initial_cost
    best_insertion_z = -1.0; best_insertion_layer_idx = -1
    tested_insertions = 0
    ep_cumsum = np.cumsum(ep_vector_in); total_thickness = ep_cumsum[-1] if num_layers_in > 0 else 0.0

    try: nH_real_l0 = _get_nk_at_lambda(nH_material, l0_repr).real
    except Exception: nH_real_l0 = -1
    try: nL_real_l0 = _get_nk_at_lambda(nL_material, l0_repr).real
    except Exception: nL_real_l0 = -1
    if nH_real_l0 < 0 or nL_real_l0 < 0: logs.append(f"{log_prefix}AVERTISSEMENT: Impossible d'obtenir n(l0={l0_repr}nm) pour type aiguille.")

    # Boucle de scan
    for z in np.arange(scan_step, total_thickness, scan_step):
        current_layer_idx = -1; layer_start_z = 0.0
        for i in range(num_layers_in):
            layer_end_z = ep_cumsum[i]
            if z > layer_start_z and z <= layer_end_z:
                t_part1 = z - layer_start_z; t_part2 = layer_end_z - z
                if t_part1 >= min_thickness_phys and t_part2 >= min_thickness_phys: current_layer_idx = i
                else: current_layer_idx = -2 # Trop près du bord
                break
            layer_start_z = layer_end_z
        if current_layer_idx < 0: continue

        tested_insertions += 1
        is_H_layer_split = (current_layer_idx % 2 == 0)
        n_real_needle = nL_real_l0 if is_H_layer_split else nH_real_l0
        if n_real_needle <= 1e-9:
            needle_type = "L" if is_H_layer_split else "H"
            logs.append(f"{log_prefix}Skip insertion z={z:.2f}. n_real({needle_type}, l0={l0_repr}nm)={n_real_needle:.3e} <= 0.")
            continue

        t_layer_split_1 = z - (ep_cumsum[current_layer_idx-1] if current_layer_idx > 0 else 0.0)
        t_layer_split_2 = ep_cumsum[current_layer_idx] - z
        ep_temp_np = np.concatenate((ep_vector_in[:current_layer_idx], [t_layer_split_1, base_needle_thickness_nm, t_layer_split_2], ep_vector_in[current_layer_idx+1:]))
        ep_temp_np = np.maximum(ep_temp_np, min_thickness_phys)

        try:
            current_cost_jax = cost_function_jax_compiled(jnp.asarray(ep_temp_np), *static_args_cost_fn)
            current_cost = float(np.array(current_cost_jax))
            if np.isfinite(current_cost) and current_cost < min_cost_found:
                min_cost_found = current_cost; best_ep_found = ep_temp_np.copy()
                best_insertion_z = z; best_insertion_layer_idx = current_layer_idx
        except Exception as e_cost: logs.append(f"{log_prefix}AVERTISSEMENT: Échec calcul coût pour z={z:.2f}. {e_cost}"); continue

    # Résultats
    if best_ep_found is not None:
        improvement = initial_cost - min_cost_found
        logs.append(f"{log_prefix}Scan terminé. Testé {tested_insertions} pts. Meilleure amélioration: {improvement:.6e}")
        logs.append(f"{log_prefix}Meilleure insertion à z={best_insertion_z:.3f} nm dans couche initiale {best_insertion_layer_idx + 1}.")
        return best_ep_found, min_cost_found, logs, best_insertion_layer_idx
    else:
        logs.append(f"{log_prefix}Scan terminé. Aucune amélioration trouvée.")
        return None, initial_cost, logs, -1

# Fonction 25: _run_needle_iterations_st (Adaptée)
def _run_needle_iterations_st(ep_start: np.ndarray, num_needles: int,
                            inputs: Dict, active_targets: List[Dict],
                            nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
                            min_thickness_phys: float, l_vec_optim_np: np.ndarray,
                            scan_step_nm: float, base_needle_thickness_nm: float,
                            log_prefix: str = "") -> Tuple[np.ndarray, float, List[str], int, int, int]:
    """Version adaptée pour Streamlit des itérations d'aiguille."""
    logs = []
    best_ep_overall = np.asarray(ep_start).copy()
    best_mse_overall = np.inf
    total_nit_needles = 0; total_nfev_needles = 0; successful_reopts_count = 0

    # Précalcul du MSE initial et compilation de la fonction coût
    try:
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        nH_arr = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax)
        nL_arr = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax)
        nSub_arr = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax)
        active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
        static_args_cost_fn = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, min_thickness_phys)
        cost_fn_compiled = jax.jit(calculate_mse_for_optimization_penalized_jax) # Utilise la fonction pénalisée
        initial_cost_jax = cost_fn_compiled(jnp.asarray(best_ep_overall), *static_args_cost_fn)
        best_mse_overall = float(np.array(initial_cost_jax))
        if not np.isfinite(best_mse_overall): raise ValueError("MSE initial non fini.")
        logs.append(f"{log_prefix} Démarrage itérations aiguille. MSE initial: {best_mse_overall:.6e}")
    except Exception as e:
        logs.append(f"{log_prefix} ERREUR calcul MSE initial pour itérations aiguille: {e}")
        return ep_start, np.inf, logs, 0, 0, 0 # Retour état initial en cas d'erreur

    l0_repr = inputs.get('l0', 500.0)

    # Boucle d'itérations
    for i in range(num_needles):
        logs.append(f"{log_prefix} Itération Aiguille {i + 1}/{num_needles}")
        current_ep_iter = best_ep_overall.copy()
        if len(current_ep_iter) == 0: logs.append(f"{log_prefix} Structure vide, arrêt."); break

        # Étape 1: Scan Aiguille (utilise fonction adaptée et cost_fn compilée)
        ep_after_scan, cost_after_scan, scan_logs, inserted_idx = _perform_needle_insertion_scan_st(
            current_ep_iter, nH_material, nL_material, nSub_material, l_vec_optim_np,
            active_targets, cost_fn_compiled, # Passer la fonction JITée
            min_thickness_phys, base_needle_thickness_nm, scan_step_nm, l0_repr,
            log_prefix=f"{log_prefix} Scan {i+1} "
        )
        logs.extend(scan_logs)
        if ep_after_scan is None:
            logs.append(f"{log_prefix} Scan aiguille {i + 1} sans amélioration. Arrêt itérations."); break

        # Étape 2: Ré-optimisation (utilise fonction core adaptée)
        logs.append(f"{log_prefix} Scan aiguille {i + 1} amélioré. Ré-optimisation...")
        ep_after_reopt, optim_success, final_cost_reopt, optim_logs, optim_status_msg, nit_reopt, nfev_reopt = \
            _run_core_optimization(ep_after_scan, inputs, active_targets, nH_material, nL_material, nSub_material,
                                   min_thickness_phys, log_prefix=f"{log_prefix} Re-Opt {i+1} ")
        logs.extend(optim_logs)
        if not optim_success:
            logs.append(f"{log_prefix} Ré-optimisation après scan {i + 1} échouée. Arrêt itérations."); break

        logs.append(f"{log_prefix} Ré-opt {i + 1} succès. MSE: {final_cost_reopt:.6e}. Iter/Eval: {nit_reopt}/{nfev_reopt}")
        total_nit_needles += nit_reopt; total_nfev_needles += nfev_reopt; successful_reopts_count += 1

        # Mise à jour du meilleur global
        if final_cost_reopt < best_mse_overall - MSE_IMPROVEMENT_TOLERANCE:
            logs.append(f"{log_prefix} MSE amélioré de {best_mse_overall:.6e}. MàJ meilleur.")
            best_ep_overall = ep_after_reopt.copy(); best_mse_overall = final_cost_reopt
        else:
            logs.append(f"{log_prefix} MSE ({final_cost_reopt:.6e}) non amélioré. Arrêt.")
            best_ep_overall = ep_after_reopt.copy(); best_mse_overall = final_cost_reopt
            break

    logs.append(f"{log_prefix} Fin itérations aiguille. Meilleur MSE final: {best_mse_overall:.6e}")
    logs.append(f"{log_prefix} Total Iter/Eval: {total_nit_needles}/{total_nfev_needles} sur {successful_reopts_count} ré-opts.")
    return best_ep_overall, best_mse_overall, logs, total_nit_needles, total_nfev_needles, successful_reopts_count

# [PREVIOUS CODE HERE - Imports, Config, CSS, Constants, Core Functions 1-26, etc.]
# ...

# Fonction 26: setup_axis_grids_st (Helper Plotting - rappel)
def setup_axis_grids_st(ax):
    """Configure les grilles standard pour un axe Matplotlib."""
    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.3, alpha=0.5)
    ax.minorticks_on()

# Fonction 27: draw_plots_st (Génération Figure Principale)
def draw_plots_st(res_plot: Dict[str, np.ndarray], # Résultats du calcul {l: array, Ts: array}
                  ep_actual: np.ndarray,           # Vecteur épaisseurs actuel (np.array)
                  nH_mat: MaterialInputType,       # Définition matériau H
                  nL_mat: MaterialInputType,       # Définition matériau L
                  nSub_mat: MaterialInputType,     # Définition matériau Substrat
                  active_targets_plot: List[Dict], # Liste des cibles actives [{'min':..},..]
                  mse: Union[float, None],         # MSE calculé (ou None)
                  is_optimized: bool = False,      # Flag si état optimisé
                  method_name: str = "",           # Nom de la méthode utilisée (pour titre)
                  res_optim_grid: Union[Dict, None] = None, # Résultats sur grille opt {l, Ts}
                  material_sequence: Union[List[str], None] = None # Séquence arbitraire (optionnel)
                  ) -> plt.Figure:
    """
    Génère la figure Matplotlib principale avec 3 sous-graphiques (Spectre, Profil n', Empilement)
    pour affichage dans Streamlit.

    Args:
        res_plot: Dictionnaire contenant 'l' et 'Ts' du calcul principal.
        ep_actual: Vecteur NumPy des épaisseurs actuelles.
        nH_mat, nL_mat, nSub_mat: Définitions des matériaux.
        active_targets_plot: Liste des dictionnaires des cibles actives.
        mse: Valeur MSE calculée ou None.
        is_optimized: Booléen indiquant si les données proviennent d'une optimisation.
        method_name: String décrivant la méthode (pour le titre).
        res_optim_grid: Optionnel, résultats {'l','Ts'} sur grille opt pour afficher les points.
        material_sequence: Optionnel, liste des noms de matériaux si séquence arbitraire.

    Returns:
        matplotlib.figure.Figure: La figure générée.
    """
    # Créer une nouvelle figure à chaque appel pour éviter problèmes état Matplotlib avec Streamlit
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5)) # Taille ajustée

    # --- Titre Général ---
    opt_method_str = f" ({method_name})" if method_name else ""
    plot_title = f'Résultats {"Optimisés" if is_optimized else "Nominaux"}{opt_method_str}'
    fig.suptitle(plot_title, fontsize=14, weight='bold')

    num_layers = len(ep_actual) if ep_actual is not None else 0
    ep_cumulative = np.cumsum(ep_actual) if num_layers > 0 else np.array([])

    # --- 1. Graphe Spectre (axes[0]) ---
    ax_spec = axes[0]
    ax_spec.set_title("Spectre T(λ)")
    ax_spec.set_xlabel("Longueur d'onde (nm)")
    ax_spec.set_ylabel('Transmittance')

    if res_plot and 'l' in res_plot and 'Ts' in res_plot and res_plot['l'] is not None and len(res_plot['l']) > 0:
        res_l_plot = np.asarray(res_plot['l'])
        res_ts_plot = np.asarray(res_plot['Ts'])

        # Tracé Transmittance
        ax_spec.plot(res_l_plot, res_ts_plot, label='Transmittance', linestyle='-', color='blue', linewidth=1.5)

        # Tracé Cibles Actives
        plotted_target_label = False
        if active_targets_plot:
            for target in active_targets_plot:
                try:
                    l_min, l_max = float(target['min']), float(target['max'])
                    t_min, t_max = float(target['target_min']), float(target['target_max'])
                    x_coords, y_coords = [l_min, l_max], [t_min, t_max]
                    label = 'Cible' if not plotted_target_label else "_nolegend_"
                    ax_spec.plot(x_coords, y_coords, 'r--', linewidth=1.0, alpha=0.8, label=label, zorder=5)
                    ax_spec.plot(x_coords, y_coords, marker='x', color='red', markersize=6, linestyle='none', zorder=6)
                    plotted_target_label = True
                except (KeyError, ValueError, TypeError): continue # Ignorer cible mal formée

            # Tracé Points Grille Optimisation (si fournis)
            if res_optim_grid and 'l' in res_optim_grid and res_optim_grid['l'].size > 0:
                 res_l_optim = np.asarray(res_optim_grid['l'])
                 # Tracer les points cibles correspondants sur la grille d'optimisation
                 for target in active_targets_plot:
                    try:
                        l_min, l_max = float(target['min']), float(target['max'])
                        t_min, t_max = float(target['target_min']), float(target['target_max'])
                        indices_optim = np.where((res_l_optim >= l_min) & (res_l_optim <= l_max))[0]
                        if indices_optim.size > 0:
                            optim_lambdas = res_l_optim[indices_optim]
                            if abs(l_max - l_min) < 1e-9: optim_target_t = np.full_like(optim_lambdas, t_min)
                            else: slope = (t_max - t_min) / (l_max - l_min); optim_target_t = t_min + slope * (optim_lambdas - l_min)
                            ax_spec.plot(optim_lambdas, np.clip(optim_target_t, 0, 1), marker='.', color='darkred', linestyle='none', markersize=3, alpha=0.7, zorder=6)
                    except (KeyError, ValueError, TypeError): continue

        # Configuration Axes Spectre
        setup_axis_grids_st(ax_spec)
        if len(res_l_plot) > 0: ax_spec.set_xlim(res_l_plot.min(), res_l_plot.max())
        ax_spec.set_ylim(-0.05, 1.05)
        if ax_spec.has_data(): ax_spec.legend(fontsize='small')

        # Annotation MSE
        if mse is not None and not np.isnan(mse) and mse != -1 : mse_text = f"MSE = {mse:.3e}"
        elif mse == -1: mse_text = "MSE: N/A (Pas ré-opt)"
        elif mse is None and active_targets_plot: mse_text = "MSE: Erreur Calc."
        elif mse is None: mse_text = "MSE: N/A (pas cible)"
        else: mse_text = "MSE: N/A (pas pts)" # Cas mse = NaN
        ax_spec.text(0.98, 0.02, mse_text, transform=ax_spec.transAxes, ha='right', va='bottom', fontsize='small', bbox=dict(boxstyle='round,pad=0.2', fc='wheat', alpha=0.7))

    else: # Pas de données spectrales
        ax_spec.text(0.5, 0.5, "Pas de données spectrales", ha='center', va='center', transform=ax_spec.transAxes)
        setup_axis_grids_st(ax_spec); ax_spec.set_ylim(-0.05, 1.05)

    # --- 2. Graphe Profil d'Indice (axes[1]) ---
    ax_idx = axes[1]
    l0_repr = 500.0 # Défaut
    try:
        if 'l0' in st.session_state: l0_repr = float(st.session_state.l0)
        if l0_repr <= 0: raise ValueError("l0 doit être positif")
    except Exception as e_l0:
        log_message(f"AVERTISSEMENT: Impossible lire/convertir l0 ('{st.session_state.get('l0', 'N/A')}') pour tracé profil. Utilisation 500nm. Erreur: {e_l0}")
        l0_repr = 500.0
    ax_idx.set_title(f"Profil Indice n' (λ={l0_repr:.0f}nm)")
    ax_idx.set_xlabel('Profondeur (depuis substrat) (nm)')
    ax_idx.set_ylabel("Partie Réelle Indice (n')")

    try:
        # Obtenir n+ik @ l0_repr
        nH_c_repr = _get_nk_at_lambda(nH_mat, l0_repr)
        nL_c_repr = _get_nk_at_lambda(nL_mat, l0_repr)
        nSub_c_repr = _get_nk_at_lambda(nSub_mat, l0_repr)
        nH_r_repr, nL_r_repr, nSub_r_repr = nH_c_repr.real, nL_c_repr.real, nSub_c_repr.real

        # Déterminer n' pour chaque couche
        n_real_layers_repr = []
        if material_sequence and len(material_sequence) == num_layers:
            # Cas séquence arbitraire (nécessite accès aux définitions via nom)
            log_message("Tracé profil indice pour séquence arbitraire (expérimental).")
            for mat_name in material_sequence:
                try: n_real_layers_repr.append(_get_nk_at_lambda(mat_name, l0_repr).real)
                except Exception as e_nk_seq:
                     n_real_layers_repr.append(np.nan); log_message(f"Erreur indice pour '{mat_name}' @{l0_repr}nm: {e_nk_seq}")
        else: # Cas HLHL...
            n_real_layers_repr = [nH_r_repr if i % 2 == 0 else nL_r_repr for i in range(num_layers)]

        # Créer coordonnées pour step plot
        total_thickness = ep_cumulative[-1] if num_layers > 0 else 0
        margin = max(50, 0.1 * total_thickness) if total_thickness > 0 else 50
        x_coords_plot = [-margin, 0] # Substrat
        y_coords_plot = [nSub_r_repr, nSub_r_repr]
        current_pos = 0.0
        for i in range(num_layers):
            layer_n_real = n_real_layers_repr[i] if i < len(n_real_layers_repr) else np.nan
            layer_thickness = ep_actual[i]
            x_coords_plot.extend([current_pos, current_pos + layer_thickness])
            y_coords_plot.extend([layer_n_real, layer_n_real])
            current_pos += layer_thickness
        x_coords_plot.extend([total_thickness, total_thickness + margin]) # Air
        y_coords_plot.extend([1.0, 1.0])

        # Filtrer les NaN potentiels pour le tracé et limites
        valid_indices_plot = ~np.isnan(y_coords_plot)
        x_coords_plot_valid = np.array(x_coords_plot)[valid_indices_plot]
        y_coords_plot_valid = np.array(y_coords_plot)[valid_indices_plot]

        if len(x_coords_plot_valid) > 1:
            ax_idx.plot(x_coords_plot_valid, y_coords_plot_valid, drawstyle='steps-post', label=f"n'(λ={l0_repr:.0f}nm)", color='purple', linewidth=1.5)

            # Limites et Annotations
            min_n_list = [1.0, nSub_r_repr] + [n for n in n_real_layers_repr if not np.isnan(n)]
            max_n_list = [1.0, nSub_r_repr] + [n for n in n_real_layers_repr if not np.isnan(n)]
            min_n = min(min_n_list) if min_n_list else 0.9
            max_n = max(max_n_list) if max_n_list else 2.5
            ax_idx.set_ylim(bottom=min_n - 0.1, top=max_n + 0.1)
            ax_idx.set_xlim(x_coords_plot_valid.min(), x_coords_plot_valid.max())

            offset = (max_n - min_n) * 0.05 + 0.02
            common_text_opts = {'ha':'center', 'va':'bottom', 'fontsize':'small', 'bbox':dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none')}
            n_sub_label = f"{nSub_c_repr.real:.3f}" + (f"{nSub_c_repr.imag:+.3f}j" if abs(nSub_c_repr.imag) > 1e-6 else "")
            ax_idx.text(-margin / 2, nSub_r_repr + offset, f"SUB\nn={n_sub_label}", **common_text_opts)
            air_x_pos = total_thickness + margin / 2
            ax_idx.text(air_x_pos, 1.0 + offset, "AIR\nn=1.0", **common_text_opts)
            if ax_idx.has_data(): ax_idx.legend(fontsize='x-small', loc='lower right')
        else:
             ax_idx.text(0.5, 0.5, "Données d'indice\nindisponibles", ha='center', va='center', transform=ax_idx.transAxes)

    except Exception as e_idx_plot:
        ax_idx.text(0.5, 0.5, f"Erreur tracé profil:\n{e_idx_plot}", ha='center', va='center', transform=ax_idx.transAxes)
        log_message(f"Erreur tracé profil indice: {e_idx_plot}")
    finally:
        setup_axis_grids_st(ax_idx) # Appliquer grille dans tous les cas


    # --- 3. Graphe Empilement (axes[2]) ---
    ax_stack = axes[2]
    ax_stack.set_title(f"Empilement ({num_layers} couches)")
    if num_layers > 0:
        try:
            # Obtenir indices complexes @ l0 pour labels
            indices_complex_repr = []
            mat_names_for_label = []
            if material_sequence and len(material_sequence) == num_layers:
                mat_names_for_label = material_sequence
                for mat_name in material_sequence:
                    try: indices_complex_repr.append(_get_nk_at_lambda(mat_name, l0_repr))
                    except Exception: indices_complex_repr.append(complex(np.nan, np.nan))
            else: # HLHL
                nH_c_repr = _get_nk_at_lambda(nH_mat, l0_repr)
                nL_c_repr = _get_nk_at_lambda(nL_mat, l0_repr)
                indices_complex_repr = [nH_c_repr if i % 2 == 0 else nL_c_repr for i in range(num_layers)]
                mat_names_for_label = [str(nH_mat)[:6] if i % 2 == 0 else str(nL_mat)[:6] for i in range(num_layers)] # Noms courts

            # Couleurs barres
            colors = ['lightblue' if i % 2 == 0 else 'lightcoral' for i in range(num_layers)] # Simple HL

            bar_pos = np.arange(num_layers)
            bars = ax_stack.barh(bar_pos, ep_actual, align='center', color=colors, edgecolor='grey', height=0.8)

            # Labels Y
            yticks_labels = []
            for i in range(num_layers):
                layer_type = mat_names_for_label[i]
                n_comp = indices_complex_repr[i]
                n_str = f"{n_comp.real:.3f}" if not np.isnan(n_comp.real) else "N/A"
                if not np.isnan(n_comp.imag) and abs(n_comp.imag) > 1e-6: n_str += f"{n_comp.imag:+.3f}j"
                yticks_labels.append(f"L{i + 1} ({layer_type}) n≈{n_str}")

            ax_stack.set_yticks(bar_pos)
            ax_stack.set_yticklabels(yticks_labels, fontsize='x-small') # Police plus petite
            ax_stack.invert_yaxis()

            # Texte épaisseurs sur barres
            max_ep = ep_actual.max() if ep_actual.size > 0 else 1.0
            fontsize_bar = max(6, 9 - num_layers // 15) # Ajuster taille texte
            for i, e_val in enumerate(ep_actual):
                ha_pos = 'left' if e_val < max_ep * 0.25 else 'right' # Ajuster seuil
                x_text_pos = e_val * 1.03 if ha_pos == 'left' else e_val * 0.97
                text_color = 'black' if ha_pos == 'left' else 'white'
                ax_stack.text(x_text_pos, i, f"{e_val:.2f}", va='center', ha=ha_pos, color=text_color, fontsize=fontsize_bar, weight='bold')

            ax_stack.set_xlabel('Épaisseur (nm)')
            ax_stack.set_ylim(bottom=num_layers - 0.5, top=-0.5)

        except Exception as e_stack_plot:
             ax_stack.text(0.5, 0.5, f"Erreur tracé empilement:\n{e_stack_plot}", ha='center', va='center', transform=ax_stack.transAxes)
             log_message(f"Erreur tracé empilement: {e_stack_plot}")
    else: # Pas de couches
        ax_stack.text(0.5, 0.5, "Pas de couches", ha='center', va='center', fontsize=10, color='grey')
        ax_stack.set_xticks([]); ax_stack.set_yticks([])

    # --- Ajustements Finaux Figure ---
    try:
        plt.tight_layout(pad=1.0, h_pad=1.5, w_pad=1.5, rect=[0, 0.03, 1, 0.95])
    except Exception:
        log_message("Avertissement: échec ajustement layout Matplotlib.")
        plt.tight_layout() # Essayer sans rect

    return fig # Retourner la figure générée



# ============================================================
# 3. STREAMLIT UI HELPER FUNCTIONS & STATE MANAGEMENT (Suite)
# ============================================================

# Fonction 28: log_message
# (Implémentation complète déjà fournie)
def log_message(message):
    """Ajoute un message formaté au log dans st.session_state."""
    now = datetime.datetime.now().strftime('%H:%M:%S')
    full_message = f"[{now}] {str(message)}"
    # Initialiser la liste de logs si elle n'existe pas
    if 'log_messages' not in st.session_state: st.session_state.log_messages = []
    st.session_state.log_messages.append(full_message)
    max_log_entries = 500 # Limiter la taille
    if len(st.session_state.log_messages) > max_log_entries:
        st.session_state.log_messages = st.session_state.log_messages[-max_log_entries:]
    print(full_message) # Afficher aussi dans la console

# Fonction 29: update_status
# (Implémentation complète déjà fournie)
def update_status(message):
    """Met à jour le message de statut dans l'état de session."""
    st.session_state.status_message = message

# Fonction 30: initialize_session_state
# (Implémentation complète déjà fournie)
def initialize_session_state(): pass # Placeholder, l'implémentation complète est cruciale

# Fonction 31: get_available_materials_from_excel_st
# (Implémentation complète déjà fournie)
def get_available_materials_from_excel_st(excel_path: str) -> List[str]: pass # Placeholder

# Fonction 32: update_available_materials
# (Implémentation complète déjà fournie)
def update_available_materials(): pass # Placeholder

# Fonction 33: validate_inputs
# (Implémentation complète déjà fournie)
def validate_inputs(require_optim_params: bool = False, require_initial_layers: bool = False) -> Dict[str, Any]: pass # Placeholder

# Fonction 34: get_active_targets_from_state
# (Implémentation complète déjà fournie)
def get_active_targets_from_state(ignore_errors=False): pass # Placeholder

# Fonction 35: get_lambda_range_from_targets
# (Implémentation complète déjà fournie)
def get_lambda_range_from_targets(active_targets): pass # Placeholder

# Fonction 36: draw_target_preview_st
# (Implémentation complète déjà fournie)
def draw_target_preview_st(): pass # Placeholder

# Fonction 37: draw_material_index_plot_st
# (Implémentation complète déjà fournie)
def draw_material_index_plot_st(): pass # Placeholder

# ============================================================
# 4. CALLBACKS STREAMLIT (pour widgets)
# ============================================================

# Fonction 38: on_material_change
# (Implémentation complète déjà fournie)
def on_material_change(): pass # Placeholder

# Fonction 39: on_qwot_change
# (Implémentation complète déjà fournie)
def on_qwot_change(): pass # Placeholder

# Fonction 40: on_initial_layer_change
# (Implémentation complète et corrigée déjà fournie)
def on_initial_layer_change(): pass # Placeholder

# --- FIN DES FONCTIONS 21-40 (LOGIQUES) ---
# [PREVIOUS CODE HERE - Imports, Config, CSS, Constants, Core Functions 1-40]
# ... (Assume lines 1-approx 920 are present, including definitions for functions #1-40)

# ============================================================
# 4. CALLBACKS STREAMLIT (Suite : Actions Principales Complexes et Autres)
# ============================================================

# Fonction 41: on_start_scan_click (Callback Bouton '1. Start Nom.')
def on_start_scan_click():
    """Action pour '1. Start Nom. (QWOT Scan+Opt)'."""
    log_message("\n" + "#"*10 + " Démarrage Scan QWOT Nominal Exhaustif + Test l0 + Opt Locale " + "#"*10)
    start_time_scan = time.time()
    update_status("Démarrage Scan QWOT + l0 + Opt...")
    st.session_state.ep_history_deque = deque(maxlen=5) # Reset historique
    log_message("Historique Undo effacé (Scan QWOT Nominal).")

    initial_candidates = []
    final_best_ep = None; final_best_mse = np.inf; final_best_l0 = None
    final_best_initial_multipliers = None
    l0_nominal_gui = None
    overall_optim_nit = 0; overall_optim_nfev = 0; successful_optim_count = 0

    # Utiliser st.spinner pour indiquer une longue opération
    with st.spinner("Scan QWOT + l0 + Optimisation Locale en cours... (Peut être long)"):
        try:
            # Valider entrées, requiert 'initial_layer_number'
            inputs = validate_inputs(require_optim_params=True, require_initial_layers=True)
            initial_layer_number = inputs['initial_layer_number'] # Déjà validé comme int > 0
            l0_nominal_gui = inputs['l0'] # l0 de l'UI

            active_targets = get_active_targets_from_state() # Erreur si invalide
            if not active_targets: raise ValueError("Scan QWOT requiert des cibles actives.")
            active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)

            nH_material = inputs['nH_material']
            nL_material = inputs['nL_material']
            nSub_material = inputs['nSub_material']

            # Préparer vecteurs lambda (complet et sparse)
            l_min, l_max, l_step = inputs['l_range_deb'], inputs['l_range_fin'], inputs['l_step']
            num_pts_eval_full = max(2, int(np.round((l_max - l_min) / l_step)) + 1)
            l_vec_eval_full_np = np.geomspace(l_min, l_max, num_pts_eval_full)
            l_vec_eval_full_np = l_vec_eval_full_np[(l_vec_eval_full_np > 0) & np.isfinite(l_vec_eval_full_np)]
            if not l_vec_eval_full_np.size: raise ValueError("Échec génération vecteur lambda évaluation.")
            l_vec_eval_sparse_np = l_vec_eval_full_np[::2] # Plus rapide pour le scan
            if not l_vec_eval_sparse_np.size: raise ValueError("Échec génération vecteur lambda sparse pour scan.")
            l_vec_eval_sparse_jax = jnp.asarray(l_vec_eval_sparse_np)
            log_message(f"Grille évaluation: {len(l_vec_eval_full_np)} pts. Grille scan rapide: {len(l_vec_eval_sparse_jax)} pts.")

            # Définir les l0 à tester
            l0_values_to_test = sorted(list(set([l0_nominal_gui, l0_nominal_gui * 1.25, l0_nominal_gui * 0.75])))
            l0_values_to_test = [l for l in l0_values_to_test if l > 1e-6]
            num_l0_tests = len(l0_values_to_test)
            num_combinations = 2**initial_layer_number
            total_evals_scan = num_combinations * num_l0_tests
            log_message(f"Démarrage Scan QWOT N={initial_layer_number} sur {num_l0_tests} l0: {[f'{l:.2f}' for l in l0_values_to_test]}.")
            log_message(f"Combinaisons par l0: {num_combinations:,}. Évaluations totales scan: {total_evals_scan:,}.")

            # Avertissement si très long
            warning_threshold_comb = 2**21 # ~2 million
            if num_combinations > warning_threshold_comb:
                 st.warning(f"**Attention:** Le scan pour N={initial_layer_number} ({num_combinations:,} comb.) sur {num_l0_tests} valeurs de l0 sera **très long** ! L'interface peut sembler figée.")
                 # Pas de pause bloquante dans Streamlit, l'utilisateur doit être patient

            # Pré-calculer indice substrat pour le scan
            nSub_arr_scan = _get_nk_array_for_lambda_vec(nSub_material, l_vec_eval_sparse_jax)

            # --- Boucle 1: Scan Rapide pour chaque l0 ---
            for l0_idx, current_l0 in enumerate(l0_values_to_test):
                log_message(f"\n--- Scan QWOT pour l0 = {current_l0:.2f} nm ({l0_idx+1}/{num_l0_tests}) ---")
                update_status(f"Scan l0={current_l0:.1f} ({l0_idx+1}/{num_l0_tests})...")

                try: # Obtenir indices H/L à ce l0
                    nH_c_l0 = _get_nk_at_lambda(nH_material, current_l0)
                    nL_c_l0 = _get_nk_at_lambda(nL_material, current_l0)
                except Exception as e_idx:
                    log_message(f"ERREUR: Impossible obtenir indices H/L à l0={current_l0:.2f} ({e_idx}). Scan pour ce l0 annulé.")
                    continue # Passer au l0 suivant

                # Exécuter le scan splitté (adapté)
                # !!! Assurez-vous que _execute_split_stack_scan_st est correctement implémenté !!!
                current_best_mse_scan, current_best_multipliers_scan, scan_logs = _execute_split_stack_scan_st(
                    current_l0, initial_layer_number, nH_c_l0, nL_c_l0,
                    nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple
                )
                for log_line in scan_logs: log_message(log_line)

                # Stocker le meilleur résultat de ce scan
                if np.isfinite(current_best_mse_scan) and current_best_multipliers_scan is not None:
                    log_message(f"Scan trouvé candidat pour l0={current_l0:.2f} avec MSE {current_best_mse_scan:.6e}.")
                    initial_candidates.append({
                        'l0': current_l0, 'mse_scan': current_best_mse_scan,
                        'multipliers': np.array(current_best_multipliers_scan)
                    })
                else:
                    log_message(f"Aucun candidat valide trouvé pour l0={current_l0:.2f} durant scan.")

            # --- Vérifier si des candidats existent ---
            if not initial_candidates:
                raise RuntimeError("Scan QWOT n'a trouvé aucun candidat initial valide.")

            # --- Boucle 2: Optimisation Locale des meilleurs candidats ---
            log_message(f"\n--- Scan QWOT terminé. Trouvé {len(initial_candidates)} candidats. Lancement Opt. Locale pour chacun. ---")
            update_status(f"Optimisation des {len(initial_candidates)} candidats...")
            initial_candidates.sort(key=lambda c: c['mse_scan']) # Trier par MSE du scan

            for idx, candidate in enumerate(initial_candidates):
                cand_l0 = candidate['l0']; cand_mult = candidate['multipliers']; cand_mse_scan = candidate['mse_scan']
                log_message(f"\n--- Optimisation Candidat {idx+1}/{len(initial_candidates)} (l0={cand_l0:.2f}, MSE scan={cand_mse_scan:.6e}) ---")
                update_status(f"Opt. Locale {idx+1}/{len(initial_candidates)} (l0={cand_l0:.1f})...")
                try:
                    # Calculer ep initial pour ce candidat
                    emp_list_cand = list(cand_mult)
                    ep_start_optim = calculate_initial_ep(emp_list_cand, cand_l0, nH_material, nL_material)
                    ep_start_optim = np.maximum(ep_start_optim, MIN_THICKNESS_PHYS_NM)
                    log_message(f"Démarrage opt. locale depuis {len(ep_start_optim)} couches.")

                    # Lancer optimisation core
                    result_ep_optim, optim_success, final_cost_optim, optim_logs, optim_status_msg, nit_optim, nfev_optim = \
                        _run_core_optimization(ep_start_optim, inputs, active_targets,
                                               nH_material, nL_material, nSub_material,
                                               MIN_THICKNESS_PHYS_NM, log_prefix=f"  [Opt Cand {idx+1}] ")
                    for log_line in optim_logs: log_message(log_line)

                    if optim_success:
                        successful_optim_count += 1; overall_optim_nit += nit_optim; overall_optim_nfev += nfev_optim
                        log_message(f"Optimisation succès pour candidat {idx+1}. MSE final: {final_cost_optim:.6e}")
                        if final_cost_optim < final_best_mse:
                            log_message(f"*** Nouveau meilleur global trouvé! MSE amélioré de {final_best_mse:.6e} à {final_cost_optim:.6e} ***")
                            final_best_mse = final_cost_optim; final_best_ep = result_ep_optim.copy()
                            final_best_l0 = cand_l0; final_best_initial_multipliers = cand_mult
                        else:
                            log_message(f"Résultat MSE {final_cost_optim:.6e} pas meilleur que le meilleur actuel {final_best_mse:.6e}.")
                    else:
                        log_message(f"Optimisation ÉCHOUÉE pour candidat {idx+1}. Msg: {optim_status_msg}, Coût: {final_cost_optim:.3e}")

                except Exception as e_optim_cand:
                    log_message(f"ERREUR durant optimisation candidat {idx+1}: {e_optim_cand}\n{traceback.format_exc(limit=1)}")

            # --- Résultats Finaux ---
            if final_best_ep is None:
                raise RuntimeError("Optimisation locale échouée pour tous les candidats du scan.")

            log_message("\n--- Meilleur Résultat Global Après Optimisations Locales ---")
            log_message(f"Meilleur MSE Final: {final_best_mse:.6e}")
            log_message(f"Provenant de l0 = {final_best_l0:.2f} nm")
            best_mult_str = ",".join([f"{m:.3f}" for m in final_best_initial_multipliers])
            log_message(f"Séquence Multiplicateur Originale ({initial_layer_number} couches): {best_mult_str}")

            st.session_state.current_optimized_ep = final_best_ep.copy()
            st.session_state.current_material_sequence = None
            st.session_state.optimization_ran = True

            if abs(final_best_l0 - l0_nominal_gui) > 1e-3:
                log_message(f"Mise à jour GUI l0 de {l0_nominal_gui:.2f} vers {final_best_l0:.2f}")
                st.session_state.l0 = final_best_l0 # Mettre à jour l'état (sera float)

            # Calculer et afficher QWOT final
            final_qwot_str = "QWOT Erreur"
            try:
                qwots = calculate_qwot_from_ep(final_best_ep, final_best_l0, nH_material, nL_material)
                if not np.any(np.isnan(qwots)): final_qwot_str = ", ".join([f"{q:.3f}" for q in qwots])
                else: final_qwot_str = "QWOT N/A"
            except Exception as qwot_e: log_message(f"Avertissement: calcul QWOT échoué ({qwot_e})")
            st.session_state.optimized_qwot_display = final_qwot_str

            avg_nit = f"{overall_optim_nit / successful_optim_count:.1f}" if successful_optim_count else "N/A"
            avg_nfev = f"{overall_optim_nfev / successful_optim_count:.1f}" if successful_optim_count else "N/A"
            final_status = f"Scan+Opt Terminé | Best MSE: {final_best_mse:.3e} | Couches: {len(final_best_ep)} | L0: {final_best_l0:.1f} | Avg Iter/Eval: {avg_nit}/{avg_nfev}"
            update_status(f"Prêt ({final_status})")

            log_message("Recalcul et tracé du résultat final...")
            run_calculation_st(ep_vector_to_use=final_best_ep, is_optimized=True, method_name=f"Scan+Opt (N={initial_layer_number}, L0={final_best_l0:.1f})")

        except (ValueError, RuntimeError, TypeError) as e:
            err_msg = f"ERREUR (Workflow Scan QWOT): {e}"
            log_message(err_msg); st.error(err_msg)
            update_status(f"Erreur Scan+Opt: {e}")
            st.session_state.current_optimized_ep = None; st.session_state.optimization_ran = False
        except Exception as e:
            err_msg = f"ERREUR Inattendue (Scan QWOT / JAX): {type(e).__name__}: {e}"
            tb_msg = traceback.format_exc()
            log_message(err_msg); log_message(tb_msg); print(err_msg); print(tb_msg)
            st.error(f"{err_msg}\nVoir console/log pour détails (possible erreur mémoire/JAX).")
            update_status("Erreur Inattendue (Scan+Opt)")
            st.session_state.current_optimized_ep = None; st.session_state.optimization_ran = False
        finally:
            log_message(f"--- Temps Total Scan QWOT + Opt: {time.time() - start_time_scan:.3f}s ---")


# Fonction 42: on_auto_mode_click (Callback Bouton '2. Auto Mode')
def on_auto_mode_click():
    """Action pour '2. Auto Mode (Needle>Thin>Opt)'."""
    log_message("\n" + "#"*10 + f" Démarrage Mode Auto (Max {AUTO_MAX_CYCLES} Cycles) " + "#"*10)
    start_time_auto = time.time()
    update_status("Démarrage Mode Auto...")
    st.session_state.ep_history_deque = deque(maxlen=5) # Reset historique
    log_message("Historique Undo effacé (Mode Auto).")

    best_ep_so_far = None; best_mse_so_far = np.inf
    num_cycles_done = 0; termination_reason = f"Max {AUTO_MAX_CYCLES} cycles atteints"
    initial_opt_needed_and_done = False
    total_iters_auto = 0; total_evals_auto = 0; optim_runs_auto = 0

    with st.spinner(f"Mode Auto en cours (Max {AUTO_MAX_CYCLES} cycles)..."):
        try:
            inputs = validate_inputs(require_optim_params=True) # Besoin params opt
            active_targets = get_active_targets_from_state()
            if not active_targets: raise ValueError("Mode Auto requiert des cibles actives.")
            threshold_from_gui = inputs.get('auto_thin_threshold', 1.0)
            log_message(f"Utilisation seuil élimination auto: {threshold_from_gui:.3f} nm")

            nH_mat = inputs['nH_material']; nL_mat = inputs['nL_material']; nSub_mat = inputs['nSub_material']; l0 = inputs['l0']

            # Préparer vecteur lambda optimisation
            l_min, l_max, l_step = inputs['l_range_deb'], inputs['l_range_fin'], inputs['l_step']
            num_pts = max(2, int(np.round((l_max - l_min) / l_step)) + 1)
            l_vec_optim_np = np.geomspace(l_min, l_max, num_pts)
            l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
            if not l_vec_optim_np.size: raise ValueError("Échec génération vecteur lambda pour Mode Auto.")

            # --- Déterminer Point de Départ & Calculer Coût Initial ---
            ep_start_auto = None
            if st.session_state.get('optimization_ran', False) and st.session_state.get('current_optimized_ep') is not None:
                 log_message("Mode Auto: Utilisation structure optimisée existante comme départ.")
                 ep_start_auto = np.asarray(st.session_state.current_optimized_ep).copy()
                 try: # Calculer MSE départ
                      l_vec_jax = jnp.asarray(l_vec_optim_np)
                      nH_arr = _get_nk_array_for_lambda_vec(nH_mat, l_vec_jax)
                      nL_arr = _get_nk_array_for_lambda_vec(nL_mat, l_vec_jax)
                      nSub_arr = _get_nk_array_for_lambda_vec(nSub_mat, l_vec_jax)
                      targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
                      cost_args = (nH_arr, nL_arr, nSub_arr, l_vec_jax, targets_tuple, MIN_THICKNESS_PHYS_NM)
                      cost_fn = jax.jit(calculate_mse_for_optimization_penalized_jax)
                      cost_val = cost_fn(jnp.asarray(ep_start_auto), *cost_args)
                      best_mse_so_far = float(np.array(cost_val))
                      if not np.isfinite(best_mse_so_far): raise ValueError("MSE départ non fini")
                 except Exception as e_cost: raise ValueError(f"Impossible calculer MSE départ: {e_cost}") from e_cost
            else:
                 log_message("Mode Auto: Utilisation structure nominale (QWOT) comme départ.")
                 try:
                     emp_list_str = [item.strip() for item in inputs['emp_str'].split(',') if item.strip()]
                     emp_list_float = [float(e) for e in emp_list_str]
                 except ValueError: raise ValueError(f"Format QWOT Nominal invalide: '{inputs['emp_str']}'")
                 ep_nominal_calc = calculate_initial_ep(emp_list_float, l0, nH_mat, nL_mat)
                 if ep_nominal_calc is None or ep_nominal_calc.size == 0: raise ValueError("Impossible déterminer structure nominale départ depuis QWOT.")
                 ep_nominal = np.asarray(ep_nominal_calc)
                 ep_nominal = np.maximum(ep_nominal, MIN_THICKNESS_PHYS_NM)
                 log_message(f"Structure nominale a {len(ep_nominal)} couches. Lancement optimisation initiale...")
                 update_status("Mode Auto - Optimisation Initiale...")

                 ep_after_initial_opt, initial_opt_success, initial_mse, initial_opt_logs, initial_opt_msg, initial_nit, initial_nfev = \
                     _run_core_optimization(ep_nominal, inputs, active_targets, nH_mat, nL_mat, nSub_mat,
                                            MIN_THICKNESS_PHYS_NM, log_prefix="  [Auto Init Opt] ")
                 for log_line in initial_opt_logs: log_message(log_line)
                 if not initial_opt_success: raise RuntimeError(f"Optimisation Initiale échouée ({initial_opt_msg})")

                 log_message(f"Mode Auto: Opt initiale terminée. MSE: {initial_mse:.6e} (Iter/Eval: {initial_nit}/{initial_nfev})")
                 ep_start_auto = ep_after_initial_opt.copy()
                 best_mse_so_far = initial_mse
                 initial_opt_needed_and_done = True
                 total_iters_auto += initial_nit; total_evals_auto += initial_nfev; optim_runs_auto += 1

            best_ep_so_far = ep_start_auto.copy()
            if not np.isfinite(best_mse_so_far): raise ValueError("MSE de départ pour cycles non fini.")
            log_message(f"Démarrage Cycles Mode Auto avec MSE: {best_mse_so_far:.6e} ({len(best_ep_so_far)} couches)")

            # --- Boucle des Cycles Auto ---
            for cycle_num in range(AUTO_MAX_CYCLES):
                 log_message(f"\n--- Cycle Auto {cycle_num + 1} / {AUTO_MAX_CYCLES} ---")
                 update_status(f"Cycle Auto {cycle_num + 1} - Début | MSE: {best_mse_so_far:.3e}")
                 mse_at_cycle_start = best_mse_so_far
                 ep_at_cycle_start = best_ep_so_far.copy()
                 cycle_improved_overall = False

                 # Pré-calculer la fonction coût JITée pour ce cycle (les arguments statiques sont constants)
                 cost_fn_compiled = jax.jit(calculate_mse_for_optimization_penalized_jax)

                 # Phase 1: Itérations Aiguille
                 log_message(f"  [Cycle {cycle_num+1}] Lancement {AUTO_NEEDLES_PER_CYCLE} itérations aiguille...")
                 update_status(f"Cycle Auto {cycle_num + 1} - Aiguille ({AUTO_NEEDLES_PER_CYCLE}x)...")
                 ep_after_needles, mse_after_needles, needle_logs, nit_needles, nfev_needles, reopts_in_needles = \
                     _run_needle_iterations_st(best_ep_so_far, AUTO_NEEDLES_PER_CYCLE, inputs, active_targets,
                                               nH_mat, nL_mat, nSub_mat, MIN_THICKNESS_PHYS_NM, l_vec_optim_np,
                                               DEFAULT_NEEDLE_SCAN_STEP_NM, BASE_NEEDLE_THICKNESS_NM,
                                               log_prefix=f"    [Cycle {cycle_num+1} Aiguille] ")
                 for log_line in needle_logs: log_message(log_line)
                 log_message(f"  [Cycle {cycle_num+1}] MSE après Aiguilles: {mse_after_needles:.6e} (Iter/Eval: {nit_needles}/{nfev_needles} sur {reopts_in_needles} ré-opts)")
                 total_iters_auto += nit_needles; total_evals_auto += nfev_needles; optim_runs_auto += reopts_in_needles

                 if mse_after_needles < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                      log_message("    Phase aiguille a amélioré MSE globalement.")
                      best_ep_so_far = ep_after_needles.copy(); best_mse_so_far = mse_after_needles; cycle_improved_overall = True
                 else:
                      log_message("    Phase aiguille n'a pas amélioré MSE significativement.")
                      best_ep_so_far = ep_after_needles.copy(); best_mse_so_far = mse_after_needles # Garder pour la suite

                 # Phase 2: Boucle Suppression Couches Fines
                 log_message(f"  [Cycle {cycle_num+1}] Lancement phase Suppression Fine + Ré-Opt (Seuil: {threshold_from_gui:.3f} nm)...")
                 update_status(f"Cycle Auto {cycle_num + 1} - Suppression+RéOpt...")
                 layers_removed_this_cycle = 0; thinning_loop_iter = 0
                 max_thinning_iters = len(best_ep_so_far) + 2 # Sécurité un peu plus grande

                 while thinning_loop_iter < max_thinning_iters:
                      thinning_loop_iter += 1
                      current_num_layers = len(best_ep_so_far)
                      if current_num_layers <= 2: log_message("    Structure trop petite pour suppression fine."); break

                      # Tenter suppression
                      ep_after_thin_rem, structure_thin_changed, thin_rem_logs = _perform_layer_merge_or_removal_only_st(
                           best_ep_so_far, MIN_THICKNESS_PHYS_NM, log_prefix=f"    [Suppr+RéOpt Iter {thinning_loop_iter}]",
                           threshold_for_removal=threshold_from_gui
                      )
                      for line in thin_rem_logs: log_message(line)

                      if structure_thin_changed:
                           layers_removed_this_cycle += 1
                           log_message(f"    Couche supprimée. Ré-optimisation {len(ep_after_thin_rem)} couches...")
                           update_status(f"Cycle Auto {cycle_num + 1} - Ré-Opt post-suppr {layers_removed_this_cycle}...")

                           # Ré-optimiser
                           ep_after_thin_reopt, thin_reopt_success, mse_after_thin_reopt, thin_reopt_logs, thin_reopt_msg, nit_thin_reopt, nfev_thin_reopt = \
                               _run_core_optimization(ep_after_thin_rem, inputs, active_targets, nH_mat, nL_mat, nSub_mat,
                                                      MIN_THICKNESS_PHYS_NM, log_prefix=f"      [SupprReOpt {layers_removed_this_cycle}] ")
                           for log_line in thin_reopt_logs: log_message(log_line)

                           if thin_reopt_success:
                                log_message(f"    Ré-opt après suppression succès. Nouveau MSE: {mse_after_thin_reopt:.6e} (Iter/Eval: {nit_thin_reopt}/{nfev_thin_reopt})")
                                total_iters_auto += nit_thin_reopt; total_evals_auto += nfev_thin_reopt; optim_runs_auto += 1
                                if mse_after_thin_reopt < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                                     log_message("      MSE amélioré globalement. Mise à jour meilleur état.")
                                     best_ep_so_far = ep_after_thin_reopt.copy(); best_mse_so_far = mse_after_thin_reopt; cycle_improved_overall = True
                                else:
                                     log_message("      MSE non amélioré globalement vs meilleur précédent. Garde résultat pour prochaine tentative suppression.")
                                     best_ep_so_far = ep_after_thin_reopt.copy(); best_mse_so_far = mse_after_thin_reopt
                                # Continuer la boucle de suppression
                           else: # Echec ré-opt
                                log_message(f"    AVERTISSEMENT: Ré-opt après suppression échouée ({thin_reopt_msg}). Arrêt phase suppression pour ce cycle.")
                                best_ep_so_far = ep_after_thin_rem.copy() # Garder état juste après suppression
                                # Recalculer MSE pour cet état
                                try:
                                     # Recalculer les arguments statiques si l'état a changé
                                     l_vec_jax_recalc = jnp.asarray(l_vec_optim_np)
                                     nH_arr_recalc = _get_nk_array_for_lambda_vec(nH_mat, l_vec_jax_recalc)
                                     nL_arr_recalc = _get_nk_array_for_lambda_vec(nL_mat, l_vec_jax_recalc)
                                     nSub_arr_recalc = _get_nk_array_for_lambda_vec(nSub_mat, l_vec_jax_recalc)
                                     targets_tuple_recalc = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
                                     cost_args_recalc = (nH_arr_recalc, nL_arr_recalc, nSub_arr_recalc, l_vec_jax_recalc, targets_tuple_recalc, MIN_THICKNESS_PHYS_NM)
                                     cost_val = cost_fn_compiled(jnp.asarray(best_ep_so_far), *cost_args_recalc)
                                     best_mse_so_far = float(np.array(cost_val))
                                     if not np.isfinite(best_mse_so_far): best_mse_so_far = np.inf
                                except Exception as cost_e: best_mse_so_far = np.inf; log_message(f"Erreur recalcul MSE post-échec ré-opt: {cost_e}")
                                log_message(f"    MSE après Suppr+Echec RéOpt (reverti): {best_mse_so_far:.6e}")
                                cycle_improved_overall = (best_mse_so_far < mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE) or cycle_improved_overall
                                break # Arrêter boucle suppression pour ce cycle
                      else: # Pas de couche fine trouvée
                           log_message("    Plus de couches trouvées sous seuil dans cette phase suppression."); break # Sortir boucle suppression

                 log_message(f"  [Cycle {cycle_num+1}] Phase Suppression Fine + Ré-Opt terminée. {layers_removed_this_cycle} couche(s) supprimée(s) ce cycle.")

                 # --- Fin de Cycle & Vérif Terminaison ---
                 if cycle_improved_overall or cycle_num == AUTO_MAX_CYCLES - 1 or initial_opt_needed_and_done:
                      log_message(f"--- Tracé Résultat Cycle Auto {cycle_num + 1} (MSE: {best_mse_so_far:.6e}) ---")
                      update_status(f"Cycle Auto {cycle_num + 1} - Tracé...")
                      initial_opt_needed_and_done = False # Reset flag
                      try:
                           run_calculation_st(ep_vector_to_use=best_ep_so_far, is_optimized=True, method_name=f"Fin Cycle Auto {cycle_num + 1}")
                      except Exception as plot_e: log_message(f"AVERTISSEMENT: Échec tracé résultat cycle {cycle_num + 1}: {plot_e}")
                 else:
                      log_message(f"--- Skip Tracé Intermédiaire Cycle {cycle_num + 1} (Pas d'amélioration globale détectée) ---")

                 num_cycles_done += 1
                 log_message(f"--- Fin Cycle Auto {cycle_num + 1} --- Meilleur MSE actuel: {best_mse_so_far:.6e} ({len(best_ep_so_far)} couches) ---")

                 if not cycle_improved_overall and not (best_mse_so_far < mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE):
                      log_message(f"Pas d'amélioration significative Cycle {cycle_num + 1} (Début MSE: {mse_at_cycle_start:.6e}, Fin MSE: {best_mse_so_far:.6e}). Arrêt Mode Auto.")
                      termination_reason = "Pas d'amélioration"
                      if best_mse_so_far > mse_at_cycle_start + MSE_IMPROVEMENT_TOLERANCE :
                           log_message(f"Retour à l'état avant Cycle {cycle_num + 1}.")
                           best_ep_so_far = ep_at_cycle_start; best_mse_so_far = mse_at_cycle_start
                      break # Sortir boucle auto mode

            # --- Mode Auto Terminé ---
            log_message(f"\n--- Mode Auto Aiguille+Fine Terminé après {num_cycles_done} cycles ---")
            log_message(f"Raison Terminaison: {termination_reason}")
            log_message(f"Meilleur MSE Final: {best_mse_so_far:.6e} avec {len(best_ep_so_far)} couches.")
            avg_nit = f"{total_iters_auto / optim_runs_auto:.1f}" if optim_runs_auto else "N/A"
            avg_nfev = f"{total_evals_auto / optim_runs_auto:.1f}" if optim_runs_auto else "N/A"
            log_message(f"Stats Opt Mode Auto: Runs={optim_runs_auto}, Total Iter={total_iters_auto}, Total Eval={total_evals_auto}, Avg Iter={avg_nit}, Avg Eval={avg_nfev}")

            st.session_state.current_optimized_ep = best_ep_so_far.copy()
            st.session_state.current_material_sequence = None
            st.session_state.optimization_ran = True

            final_qwot_str = "QWOT Erreur"
            try:
                 qwots = calculate_qwot_from_ep(best_ep_so_far, l0, nH_mat, nL_mat)
                 if not np.any(np.isnan(qwots)): final_qwot_str = ", ".join([f"{q:.3f}" for q in qwots])
                 else: final_qwot_str = "QWOT N/A"
            except Exception as qwot_e: log_message(f"Erreur calcul QWOT final mode auto: {qwot_e}")
            st.session_state.optimized_qwot_display = final_qwot_str

            final_status = f"Mode Auto Terminé ({num_cycles_done} cyc, {termination_reason}) | MSE: {best_mse_so_far:.3e} | Couches: {len(best_ep_so_far)} | Avg Iter/Eval: {avg_nit}/{avg_nfev}"
            update_status(f"Prêt ({final_status})")

            # Assurer que le dernier/meilleur résultat est tracé
            run_calculation_st(ep_vector_to_use=best_ep_so_far, is_optimized=True, method_name=f"Mode Auto Final ({num_cycles_done} cyc, {termination_reason})")

        except (ValueError, RuntimeError, TypeError) as e:
            err_msg = f"ERREUR (Workflow Mode Auto): {e}"
            log_message(err_msg); st.error(err_msg)
            update_status(f"Erreur Mode Auto: {e}")
            st.session_state.current_optimized_ep = None; st.session_state.optimization_ran = False
        except Exception as e:
            err_msg = f"ERREUR Inattendue (Mode Auto / JAX): {type(e).__name__}: {e}"
            tb_msg = traceback.format_exc()
            log_message(err_msg); log_message(tb_msg); print(err_msg); print(tb_msg)
            st.error(f"{err_msg}\nVoir console/log.")
            update_status("Erreur Inattendue (Mode Auto)")
            st.session_state.current_optimized_ep = None; st.session_state.optimization_ran = False
        finally:
             log_message(f"--- Temps Total Mode Auto: {time.time() - start_time_auto:.3f}s ---")


# --- Fonctions JAX spécifiques au Scan QWOT Rapide (Split Stack) ---

# Fonction 67: get_T_from_batch_matrix (JAX)
@jax.jit
def get_T_from_batch_matrix(M_batch: jnp.ndarray, nSub_arr: jnp.ndarray) -> jnp.ndarray:
    """Calcule la Transmittance à partir d'un batch de matrices totales."""
    etainc = 1.0 + 0j; etasub_batch = nSub_arr
    m00 = M_batch[:, 0, 0]; m01 = M_batch[:, 0, 1]
    m10 = M_batch[:, 1, 0]; m11 = M_batch[:, 1, 1]
    rs_den = (etainc * m00 + etasub_batch * m11 + etainc * etasub_batch * m01 + m10)
    rs_den_abs = jnp.abs(rs_den)
    safe_den = jnp.where(rs_den_abs < 1e-12, 1e-12 + 0j, rs_den)
    ts = (2.0 * etainc) / safe_den
    safe_real_etasub = jnp.maximum(jnp.real(etasub_batch), 0.0)
    Ts = (safe_real_etasub / 1.0) * jnp.abs(ts)**2
    return jnp.where(rs_den_abs < 1e-12, 0.0, jnp.nan_to_num(Ts, nan=0.0))

# Fonction 68: calculate_mse_basic_jax (JAX)
@jax.jit
def calculate_mse_basic_jax(Ts: jnp.ndarray, l_vec: jnp.ndarray, targets_tuple: Tuple) -> jnp.ndarray:
    """Calcule le MSE de base sans pénalités (pour le scan rapide)."""
    total_squared_error = 0.0; total_points_in_targets = 0
    for i in range(len(targets_tuple)):
        l_min, l_max, t_min, t_max = targets_tuple[i]
        target_mask = (l_vec >= l_min) & (l_vec <= l_max)
        is_flat_target = jnp.abs(l_max - l_min) < 1e-9
        denom_slope = jnp.where(is_flat_target, 1.0, l_max - l_min)
        slope = jnp.where(is_flat_target, 0.0, (t_max - t_min) / denom_slope)
        interpolated_target_t = jnp.clip(t_min + slope * (l_vec - l_min), 0.0, 1.0)
        ts_finite = jnp.nan_to_num(Ts, nan=0.0)
        squared_errors = (ts_finite - interpolated_target_t)**2
        masked_sq_error = jnp.where(target_mask & jnp.isfinite(squared_errors), squared_errors, 0.0)
        total_squared_error += jnp.sum(masked_sq_error)
        total_points_in_targets += jnp.sum(target_mask)
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, jnp.inf)
    return jnp.nan_to_num(mse, nan=jnp.inf)

# Fonction 69: combine_and_calc_mse (JAX)
@jax.jit
def combine_and_calc_mse(prod1: jnp.ndarray, prod2: jnp.ndarray,
                         nSub_arr_in: jnp.ndarray,
                         l_vec_in: jnp.ndarray, targets_tuple_in: Tuple
                         ) -> jnp.ndarray:
    """Combine les produits matriciels des moitiés, calcule T, puis calcule MSE."""
    M_total = vmap(jnp.matmul)(prod2, prod1) # M_half2 * M_half1
    Ts = get_T_from_batch_matrix(M_total, nSub_arr_in)
    mse = calculate_mse_basic_jax(Ts, l_vec_in, targets_tuple_in)
    return mse

# Fonction 70 (placeholder)
def merge_adjacent_layers_st(ep_vector: np.ndarray, material_sequence: List[str]) -> Tuple[np.ndarray, List[str]]:
    print("WARN: merge_adjacent_layers_st non implémentée.")
    return ep_vector, material_sequence

# Fonction 71 (placeholder)
def calculate_ep_for_sequence_st(material_sequence: List[str], l0: float,
                                 multipliers: Union[List[float], None] = None) -> np.ndarray:
    print("WARN: calculate_ep_for_sequence_st non implémentée.")
    return np.zeros(len(material_sequence))


# ============================================================
# 6. INITIALISATION
# ============================================================
if 'app_initialized' not in st.session_state:
    print("--- Initialisation de l'état de session ---")
    initialize_session_state()
    st.session_state.app_initialized = True

# [PREVIOUS CODE HERE - Imports, Config, CSS, Constants, Core Functions 1-71, State Init Call etc.]
# ... (Assume lines 1-approx 1680 are present, including definitions for functions #1-71)

# ============================================================
# 7. INTERFACE UTILISATEUR STREAMLIT (Sidebar)
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- Section Fichiers & Matériaux ---
    with st.expander("📁 Fichiers & Matériaux", expanded=True):
        if st.button("🔄 Recharger Matériaux (Excel)", key="reload_mat_btn_sb", help=f"Relit les matériaux depuis '{EXCEL_FILE_PATH}'"):
            load_material_data_from_xlsx_sheet.cache_clear()
            update_available_materials()
            st.toast(f"Listes de matériaux rechargées depuis {EXCEL_FILE_PATH} !", icon="✅")
            st.rerun() # Recharger l'UI avec les nouvelles listes

        uploaded_file = st.file_uploader("Charger Design (.json)", type="json", key="load_design_uploader_sb")
        if uploaded_file is not None:
            handle_load_design_st(uploaded_file)
            # Déclencher un rerun pour appliquer l'état chargé et nettoyer l'uploader
            st.rerun()

        # Préparation données sauvegarde
        save_data_nominal_dict = {}; save_data_optimized_dict = None; save_error = None
        try:
            save_data_nominal_dict = _collect_design_data_st(include_optimized=False)
            if st.session_state.get('optimization_ran') and st.session_state.get('current_optimized_ep') is not None:
                save_data_optimized_dict = _collect_design_data_st(include_optimized=True)
        except Exception as e_collect: save_error = f"Erreur collecte données: {e_collect}"; log_message(save_error)
        save_json_nominal = ""; save_json_optimized = ""
        if not save_error:
            try:
                save_json_nominal = json.dumps(save_data_nominal_dict, indent=4)
                if save_data_optimized_dict: save_json_optimized = json.dumps(save_data_optimized_dict, indent=4)
            except Exception as e_json: save_error = f"Erreur JSON: {e_json}"; log_message(save_error)
        if save_error: st.error(f"Erreur prép. sauvegarde: {save_error}")

        # Boutons Sauvegarde
        col_save1, col_save2 = st.columns(2)
        save_file_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with col_save1:
            st.download_button(label="💾 Sauver Nominal", data=save_json_nominal if not save_error else "",
                               file_name=f"design_nominal_{save_file_ts}.json", mime="application/json",
                               key="save_nom_btn_sb", disabled=bool(save_error))
        with col_save2:
            st.download_button(label="💾 Sauver Optimisé", data=save_json_optimized if not save_error and save_data_optimized_dict else "",
                               file_name=f"design_optimise_{save_file_ts}.json", mime="application/json",
                               key="save_opt_btn_sb", disabled=bool(save_error) or (save_data_optimized_dict is None))

    # --- Section Matériaux ---
with st.expander("🔬 Matériaux et Substrat", expanded=True):
    # Récupérer les listes de matériaux/substrats de manière sûre depuis l'état de session
    # Fournir une liste par défaut minimale si l'état n'est pas encore initialisé
    mats = st.session_state.get('available_materials', ["Constant"])
    subs = st.session_state.get('available_substrates', ["Constant", "Fused Silica", "BK7", "D263"])

    # --- Matériau H ---
    # Récupérer la sélection actuelle de manière sûre
    selected_h = st.session_state.get('selected_H_material', mats[0] if mats else "Constant")
    # Calculer l'index pour le selectbox, en gérant le cas où la sélection n'est plus valide
    if selected_h in mats:
        idx_H = mats.index(selected_h)
    else:
        idx_H = 0 # Se rabattre sur le premier élément ('Constant' normalement)
        # Optionnel : Corriger l'état si invalide et notifier (nécessite que log_message soit défini)
        # st.session_state.selected_H_material = mats[idx_H]
        # log_message(f"Matériau H sélectionné '{selected_h}' invalide, réinitialisé à '{mats[idx_H]}'.")

    st.selectbox(
        "Matériau H", mats, index=idx_H, key="selected_H_material",
        on_change=on_material_change, # Assurer que le callback est défini et fonctionnel
        help="Choisir le matériau à haute réflectivité (H) ou 'Constant' pour définir n et k manuellement."
    )
    # Vérifier à nouveau après l'interaction potentielle avec le selectbox
    h_is_const = st.session_state.get('selected_H_material') == "Constant"
    colH1, colH2 = st.columns(2)
    with colH1:
        st.number_input("n' H",
                        value=st.session_state.get('nH_r', 2.35), # Utiliser get avec défaut numérique float
                        min_value=0.0, step=0.01, format="%.4f", key="nH_r",
                        disabled=not h_is_const, help="Partie réelle si Matériau H = Constant")
    with colH2:
        st.number_input("k H",
                        value=st.session_state.get('nH_i', 0.0), # Utiliser get avec défaut numérique float
                        min_value=0.0, step=1e-4, format="%.4f", key="nH_i",
                        disabled=not h_is_const, help="Partie imaginaire (>=0) si Matériau H = Constant")

    # --- Matériau L ---
    selected_l = st.session_state.get('selected_L_material', mats[0] if mats else "Constant")
    if selected_l in mats:
        idx_L = mats.index(selected_l)
    else:
        idx_L = 0
        # Optionnel : Corriger l'état et notifier
        # st.session_state.selected_L_material = mats[idx_L]
        # log_message(f"Matériau L sélectionné '{selected_l}' invalide, réinitialisé à '{mats[idx_L]}'.")
    st.selectbox(
        "Matériau L", mats, index=idx_L, key="selected_L_material",
        on_change=on_material_change,
        help="Choisir le matériau à basse réflectivité (L) ou 'Constant'."
    )
    l_is_const = st.session_state.get('selected_L_material') == "Constant"
    colL1, colL2 = st.columns(2)
    with colL1:
        st.number_input("n' L",
                        value=st.session_state.get('nL_r', 1.46),
                        min_value=0.0, step=0.01, format="%.4f", key="nL_r",
                        disabled=not l_is_const, help="Partie réelle si Matériau L = Constant")
    with colL2:
        st.number_input("k L",
                        value=st.session_state.get('nL_i', 0.0),
                        min_value=0.0, step=1e-4, format="%.4f", key="nL_i",
                        disabled=not l_is_const, help="Partie imaginaire (>=0) si Matériau L = Constant")

    # --- Substrat ---
    selected_s = st.session_state.get('selected_Sub_material', subs[0] if subs else "Constant")
    if selected_s in subs:
        idx_S = subs.index(selected_s)
    else:
        idx_S = 0
        # Optionnel : Corriger l'état et notifier
        # st.session_state.selected_Sub_material = subs[idx_S]
        # log_message(f"Substrat sélectionné '{selected_s}' invalide, réinitialisé à '{subs[idx_S]}'.")
    st.selectbox(
        "Substrat", subs, index=idx_S, key="selected_Sub_material",
        on_change=on_material_change,
        help="Choisir le matériau du substrat ou 'Constant'."
    )
    sub_is_const = st.session_state.get('selected_Sub_material') == "Constant"
    colS1, colS2 = st.columns([3,1]) # Plus d'espace pour l'entrée n'
    with colS1:
        st.number_input("n' Substrat",
                        value=st.session_state.get('nSub', 1.52), # Défaut float
                        min_value=0.0, step=0.01, format="%.4f", key="nSub",
                        disabled=not sub_is_const, help="Partie réelle si Substrat = Constant (k=0 assumé)")
    with colS2:
        # Rappel de la convention n = n' + ik
        st.markdown("<p style='font-size:0.75rem; margin-top: 25px; color: gray;'>(n = n'+ik)</p>", unsafe_allow_html=True)
        
        # Affichage Graphe Indices
        label_btn_idx = "Masquer Indices n'(λ)" if st.session_state.get('show_indices_plot') else "👁️ Voir Indices n'(λ)"
        if st.button(label_btn_idx, key="toggle_indices_btn_sb"):
            st.session_state.show_indices_plot = not st.session_state.get('show_indices_plot', False)
            st.session_state.figure_indices = None
            st.rerun()

    # Affichage conditionnel graphe indices dans sidebar
    if st.session_state.get('show_indices_plot', False):
         with st.expander("Indices n'(λ) des Matériaux Sélectionnés", expanded=True):
            if st.session_state.get('figure_indices') is None:
                log_message("Génération du graphique des indices...")
                try: st.session_state.figure_indices = draw_material_index_plot_st()
                except Exception as e_idx_plot:
                    st.error(f"Erreur génération graphique indices: {e_idx_plot}")
                    log_message(f"Erreur plot indices: {e_idx_plot}\n{traceback.format_exc(limit=2)}")
                    st.session_state.figure_indices = None
            if st.session_state.figure_indices: st.pyplot(st.session_state.figure_indices, clear_figure=False)
            else: st.warning("Impossible d'afficher le graphique des indices.")

    # --- Section Définition Empilement ---
    with st.expander("🧱 Définition Empilement", expanded=True):
         st.text_area("QWOT Nominal (ex: 1,1,0.8,1.2,...)", key="emp_str", on_change=on_qwot_change, height=100)
         col_init1, col_init2 = st.columns([2,3])
         with col_init1: st.number_input("Nb Couches Initial", min_value=0, step=1, value=st.session_state.get('initial_layer_number', 1), format="%d", key="initial_layer_number", on_change=on_initial_layer_change, help="Utilisé par '1. Start Nom.'. Met aussi à jour QWOT nominal.")
         with col_init2:
            # Affichage dynamique du nombre de couches actuel
            current_ep = st.session_state.get('current_optimized_ep')
            is_opt = st.session_state.get('optimization_ran', False)
            num_layers_disp = 0 # Default value
            label_type = "Nominal" # Default label

            if is_opt and current_ep is not None:
                num_layers_disp = len(current_ep)
                label_type = "Optimisé"
            else:
                # Calculate from nominal QWOT string, handle errors
                try:
                    # Ensure emp_str exists and is a string before splitting
                    emp_str_val = st.session_state.get('emp_str', '')
                    if isinstance(emp_str_val, str):
                         num_layers_disp = len([item for item in emp_str_val.split(',') if item.strip()])
                    else: # Should not happen if state is managed well, but fallback
                         num_layers_disp = 0
                         log_message(f"AVERTISSEMENT: st.session_state.emp_str n'est pas une chaîne ({type(emp_str_val)}). Nb couches mis à 0.")
                except Exception as e_len:
                     # Catch potential errors during split/len if emp_str is unusual
                     num_layers_disp = 0
                     log_message(f"Erreur calcul nb couches depuis QWOT: {e_len}")

            label_disp = f"Couches ({label_type}): **{num_layers_disp}**"
            st.markdown(f"<div style='margin-top: 28px; font-size:0.85rem;'>{label_disp}</div>", unsafe_allow_html=True)

         st.text_input("QWOT Optimisé (readonly)", key="optimized_qwot_display", disabled=True)
         st.number_input("λ Centrage QWOT (nm)", min_value=0.1, step=1.0, value=st.session_state.get('l0', 500.0), format="%.2f", key="l0")

    # --- Section Paramètres Calcul & Optimisation ---
    with st.expander("🛠️ Paramètres Calcul & Optimisation", expanded=False):
         colP1, colP2 = st.columns(2)
         with colP1: st.number_input("Max Iter (Opt)", min_value=1, step=10, value=st.session_state.get('maxiter', 1000), format="%d", key="maxiter")
         with colP2: st.number_input("Max Eval (Opt)", min_value=1, step=10, value=st.session_state.get('maxfun', 1000), format="%d", key="maxfun")
         st.number_input("λ Step (nm) (Optim/Plot)", min_value=0.01, step=0.1, value=st.session_state.get('l_step', 10.0), format="%.2f", key="l_step")
         st.number_input("Seuil Élimination Auto (nm)", min_value=0.0, step=0.1, value=st.session_state.get('auto_thin_threshold', 1.0), format="%.3f", key="auto_thin_threshold")
# --- FIN SIDEBAR ---


# ============================================================
# 8. INTERFACE UTILISATEUR STREAMLIT (Zone Principale)
# ============================================================

st.header("Optimiseur de Films Minces (JAX Grad - Dispersif)")
st.caption(f"Heure serveur: {datetime.datetime.now(datetime.timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}") # Afficher fuseau horaire

# --- Définition des Cibles Spectrales ---
st.subheader("🎯 Cibles Spectrales (Transmittance T)")
target_cols_header = st.columns([0.5, 0.5, 1, 1, 1, 1])
headers = ["Actif", "Zone", "λ min", "λ max", "T min", "T max"]
for col, header in zip(target_cols_header, headers):
    col.markdown(f"**{header}**")

# Récupérer et afficher les lignes de cibles
targets_state = st.session_state.target_entries # C'est une liste de dicts
for i in range(len(targets_state)):
    # Créer une clé unique pour chaque ligne/widget
    row_key_prefix = f"target_{i}"
    cols = st.columns([0.5, 0.5, 1, 1, 1, 1])
    # Utiliser les valeurs de l'état de session pour initialiser les widgets
    targets_state[i]['enabled'] = cols[0].checkbox("", value=targets_state[i]['enabled'], key=f"{row_key_prefix}_enable", label_visibility="collapsed", on_change=on_target_change)
    cols[1].markdown(f"**{i+1}:**", unsafe_allow_html=True)
    targets_state[i]['min'] = cols[2].number_input("LMin", value=float(targets_state[i].get('min', 0.0)), min_value=0.0, step=1.0, format="%.1f", key=f"{row_key_prefix}_min", label_visibility="collapsed", on_change=on_target_change)
    targets_state[i]['max'] = cols[3].number_input("LMax", value=float(targets_state[i].get('max', 0.0)), min_value=0.0, step=1.0, format="%.1f", key=f"{row_key_prefix}_max", label_visibility="collapsed", on_change=on_target_change)
    targets_state[i]['target_min'] = cols[4].number_input("TMin", value=float(targets_state[i].get('target_min', 0.0)), min_value=0.0, max_value=1.0, step=0.01, format="%.3f", key=f"{row_key_prefix}_tmin", label_visibility="collapsed", on_change=on_target_change)
    targets_state[i]['target_max'] = cols[5].number_input("TMax", value=float(targets_state[i].get('target_max', 0.0)), min_value=0.0, max_value=1.0, step=0.01, format="%.3f", key=f"{row_key_prefix}_tmax", label_visibility="collapsed", on_change=on_target_change)
# Note: La modification directe de targets_state[i]['key'] = widget_value fonctionne car
# les widgets Streamlit mettent à jour la clé correspondante dans st.session_state,
# et ici targets_state est une référence à st.session_state.target_entries.

# Afficher le nombre de points estimés
try:
    active_targets_preview = get_active_targets_from_state(ignore_errors=True)
    pts_text = "Points d'optimisation estimés: N/A"
    if active_targets_preview:
        l_min_p, l_max_p = get_lambda_range_from_targets(active_targets_preview)
        l_step_p = st.session_state.get('l_step', 10.0) # Utiliser la valeur de l'état
        if l_min_p is not None and l_max_p is not None and l_step_p > 0 and l_max_p >= l_min_p:
             num_pts_preview = max(2, int(np.round((l_max_p - l_min_p) / l_step_p)) + 1)
             pts_text = f"Points d'optimisation estimés (espacement géom.): ≈ {num_pts_preview}"
        elif l_max_p is not None and l_min_p is not None and l_max_p < l_min_p:
            pts_text = "Points d'optimisation estimés: N/A (λ max < λ min)"
        else: pts_text = "Points d'optimisation estimés: N/A (plage/pas invalide)"
    else: pts_text = "Points d'optimisation estimés: N/A (pas de cibles actives)"
except ValueError: pts_text = "Points d'optimisation estimés: N/A (pas invalide)"
except Exception as e_pts: pts_text = f"Erreur calcul points: {e_pts}"
st.caption(pts_text)

# --- Prévisualisation des Cibles (Expander) ---
label_btn_preview = "Masquer Prévisu Cibles" if st.session_state.get('show_target_preview') else "📉 Voir Prévisu Cibles"
if st.button(label_btn_preview, key="toggle_preview_btn"):
    st.session_state.show_target_preview = not st.session_state.get('show_target_preview', False)
    st.session_state.figure_preview = None # Forcer regénération
    st.rerun()

if st.session_state.get('show_target_preview', False):
    with st.expander("📉 Prévisualisation des Cibles Actives", expanded=True):
        if st.session_state.get('figure_preview') is None:
            try: st.session_state.figure_preview = draw_target_preview_st()
            except Exception as e_prev_plot: st.error(f"Erreur génération prévisualisation: {e_prev_plot}"); log_message(f"Erreur plot preview: {e_prev_plot}"); st.session_state.figure_preview = None
        if st.session_state.figure_preview: st.pyplot(st.session_state.figure_preview, clear_figure=False)
        else: st.warning("Impossible d'afficher la prévisualisation des cibles.")


# --- Actions Principales ---
st.subheader("⚙️ Actions")
action_col_1, action_col_2, action_col_3, action_col_4, action_col_5 = st.columns(5)
# Colonne 1
with action_col_1:
    if st.button("📊 Evaluer", key="eval_btn_main", help="Calcule et affiche le spectre pour le QWOT nominal.", use_container_width=True): on_evaluate_button_click()
    if st.button("📈 Opt. Locale", key="local_opt_btn_main", help="Optimise la structure actuelle (nominale ou optimisée).", use_container_width=True): on_local_opt_button_click()
# Colonne 2
with action_col_2:
    if st.button("🚀 1. Start Nom.", key="start_nom_btn_main", help="Scan QWOT + Test l0 + Opt Locale (peut être long).", use_container_width=True): on_start_scan_click()
    if st.button("🤖 2. Auto Mode", key="auto_mode_btn_main", help="Mode Auto: Aiguille > Suppr. Fine > Opt (cycles).", use_container_width=True): on_auto_mode_click()
# Colonne 3
with action_col_3:
    can_remove = st.session_state.get('optimization_ran', False) and st.session_state.get('current_optimized_ep') is not None and len(st.session_state.current_optimized_ep) > 2
    if st.button("🔪 Suppr. Fine", key="remove_thin_btn_main", disabled=not can_remove, use_container_width=True, help="Supprime la couche la plus fine et ré-optimise."): on_remove_thin_click()
    can_undo = bool(st.session_state.get('ep_history_deque', deque()))
    if st.button("↩️ Annuler Sup.", key="undo_btn_main", disabled=not can_undo, use_container_width=True, help="Annule la dernière suppression de couche."): on_undo_click()
# Colonne 4
with action_col_4:
    can_set_nominal = st.session_state.get('optimization_ran', False) and st.session_state.get('current_optimized_ep') is not None
    if st.button("➡️ Opt➜Nominal", key="set_nominal_btn_main", disabled=not can_set_nominal, use_container_width=True, help="Copie QWOT optimisé vers nominal."): on_set_nominal_click()
    can_clear_opt = st.session_state.get('optimization_ran', False)
    if st.button("🗑️ Effacer Opt.", key="clear_opt_btn_main2", disabled=not can_clear_opt, use_container_width=True, help="Efface l'état optimisé actuel."): on_clear_opt_click()
# Colonne 5
with action_col_5:
    if st.button("📋 Copier Nom.", key="copy_nom_btn", help="Affiche le QWOT Nominal pour copie manuelle.", use_container_width=True): on_copy_nominal_qwot_click()
    can_copy_opt = st.session_state.get('optimization_ran', False) and st.session_state.get('optimized_qwot_display','') not in ["", "QWOT Erreur", "QWOT N/A"]
    if st.button("📋 Copier Opt.", key="copy_opt_btn", disabled=not can_copy_opt, help="Affiche le QWOT Optimisé pour copie manuelle.", use_container_width=True): on_copy_optimized_qwot_click()
    if st.button("🔄 Rafraîchir", key="refresh_plots_btn", help="Recalcule et affiche les graphiques.", use_container_width=True): on_refresh_plots_click()
    if st.button("ℹ️ À Propos", key="about_btn", help="Affiche les infos sur l'app.", use_container_width=True): on_about_click()

# Affichage Épaisseur Minimale
ep_display_final = st.session_state.get('current_optimized_ep') if st.session_state.get('optimization_ran', False) else None
if ep_display_final is None:
     try:
         inputs_thin_final = validate_inputs()
         emp_list_thin_final = [float(e.strip()) for e in inputs_thin_final['emp_str'].split(',') if e.strip()]
         ep_display_final = calculate_initial_ep(emp_list_thin_final, inputs_thin_final['l0'], inputs_thin_final['nH_material'], inputs_thin_final['nL_material'])
     except Exception: ep_display_final = None
min_thickness_display = np.inf
if ep_display_final is not None and len(ep_display_final) > 0:
    valid_thicknesses = ep_display_final[ep_display_final > 1e-12]
    if valid_thicknesses.size > 0: min_thickness_display = np.min(valid_thicknesses)
if np.isfinite(min_thickness_display): st.caption(f"Épaisseur minimale actuelle: **{min_thickness_display:.3f} nm**")
else: st.caption("Épaisseur minimale actuelle: N/A")

st.divider()

# ============================================================
# 9. AFFICHAGE DES RÉSULTATS (Graphiques)
# ============================================================
st.subheader("📈 Résultats")
plot_placeholder = st.empty() # Conteneur pour le graphique principal

results_plot_data = st.session_state.get('last_calc_results_plot')
is_opt_plot = st.session_state.get('last_calc_is_optimized', False)
ep_to_plot_final = st.session_state.get('current_optimized_ep') if is_opt_plot else None
# Recalculer ep nominal si besoin pour le plot
if not is_opt_plot and ep_to_plot_final is None:
    try:
        inputs_plot_final = validate_inputs()
        emp_list_plot_final = [float(e.strip()) for e in inputs_plot_final['emp_str'].split(',') if e.strip()]
        ep_to_plot_final = calculate_initial_ep(emp_list_plot_final, inputs_plot_final['l0'], inputs_plot_final['nH_material'], inputs_plot_final['nL_material'])
    except Exception as e_nom_plot_final: ep_to_plot_final = np.array([]); log_message(f"Erreur calcul ep nominal pour plot final: {e_nom_plot_final}")

# Afficher le graphique principal s'il y a des données et un ep valide
if results_plot_data and ep_to_plot_final is not None:
    # Ne regénérer que si la figure n'est pas dans l'état ou si elle est None
    if st.session_state.get('figure_main') is None:
        log_message("Regénération du graphique principal...")
        try:
            inputs_plot_fig = validate_inputs() # Revalider pour s'assurer que les matériaux sont corrects
            active_targets_plot_fig = get_active_targets_from_state(ignore_errors=True) or []
            # Appel de la fonction de tracé (qui crée une nouvelle figure)
            fig_main = draw_plots_st(
                res=results_plot_data, current_ep=ep_to_plot_final,
                nH_mat=inputs_plot_fig['nH_material'], nL_mat=inputs_plot_fig['nL_material'], nSub_mat=inputs_plot_fig['nSub_material'],
                active_targets_plot=active_targets_plot_fig, mse=st.session_state.get('last_calc_mse'),
                is_optimized=is_opt_plot, method_name=st.session_state.get('last_calc_method_name', ""),
                res_optim_grid=st.session_state.get('last_calc_results_optim_grid'),
                material_sequence=st.session_state.get('last_calc_material_sequence')
            )
            st.session_state.figure_main = fig_main # Stocker la nouvelle figure
        except Exception as e_plot_final:
            st.error(f"Erreur lors de la génération du graphique principal: {e_plot_final}")
            log_message(f"Erreur plot principal: {e_plot_final}\n{traceback.format_exc(limit=2)}")
            st.session_state.figure_main = None # Échec

    # Afficher la figure stockée s'elle existe
    if st.session_state.figure_main:
        plot_placeholder.pyplot(st.session_state.figure_main, clear_figure=True) # Important: clear_figure=True ici pour libérer la mémoire après affichage
        # Si on fait clear_figure=True, il faut la regénérer à chaque fois ou gérer le cache différemment.
        # Pour l'instant, on la regénère si elle est None.
    else:
        plot_placeholder.warning("Impossible d'afficher le graphique principal des résultats.")

else:
    plot_placeholder.info("Lancez un calcul ('Evaluer' ou une optimisation) pour afficher les résultats.")


# ============================================================
# 10. LOGS & STATUT
# ============================================================
st.divider()
log_expander = st.expander("📜 Log Détaillé", expanded=False)
with log_expander:
    log_content = "\n".join(st.session_state.get('log_messages', ["Log vide."]))
    st.text_area("Logs", value=log_content, height=300, key="log_display_area_final", disabled=True, max_chars=50000)
    if st.button("Vider le Log", key="clear_log_btn_final"):
        st.session_state.log_messages = [f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Log vidé."]
        st.rerun()

# Barre de statut
st.caption(f"Statut: {st.session_state.get('status_message', 'Indéfini')}")


# ============================================================
# 11. GESTION POST-CHARGEMENT / DÉCLENCHEURS
# ============================================================
if st.session_state.get('trigger_recalc', False):
    st.session_state.trigger_recalc = False # Consommer le flag immédiatement
    log_message("Déclenchement recalcul post-chargement...")
    # Il est préférable de laisser le script se ré-exécuter normalement après la mise à jour de l'état.
    # L'état étant chargé, l'affichage se mettra à jour.
    # Si un calcul est absolument nécessaire, il faudrait le déclencher via un bouton
    # ou une logique plus complexe, car un rerun ici peut entrer en conflit avec d'autres.
    st.toast("Design chargé. Relancez un calcul si nécessaire.", icon="✅")
    # Un st.rerun() ici pourrait être utile pour forcer la prise en compte immédiate de l'état chargé.
    # st.rerun()

# --- FIN DU SCRIPT STREAMLIT COMPLET ---
