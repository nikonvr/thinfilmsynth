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

# --- Configuration JAX ---
jax.config.update("jax_enable_x64", True)

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Optimiseur Film Mince (JAX+Streamlit)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Personnalisé pour une interface plus compacte ---
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
MIN_THICKNESS_PHYS_NM = 0.01
BASE_NEEDLE_THICKNESS_NM = 0.1
DEFAULT_NEEDLE_SCAN_STEP_NM = 2.0
AUTO_NEEDLES_PER_CYCLE = 5
AUTO_MAX_CYCLES = 5
MSE_IMPROVEMENT_TOLERANCE = 1e-9
EXCEL_FILE_PATH = "indices.xlsx" # Assurez-vous que ce fichier est accessible par l'app Streamlit

# ============================================================
# 2. CŒUR MÉTIER : 20 PREMIÈRES FONCTIONS
# ============================================================

# --- Fonctions de gestion des matériaux ---

# Fonction 1: load_material_data_from_xlsx_sheet (avec cache Streamlit)
@st.cache_data(max_entries=32, ttl=3600) # Cache pour 1h max
@functools.lru_cache(maxsize=32) # Garder lru_cache aussi par sécurité
def load_material_data_from_xlsx_sheet(file_path: str, sheet_name: str) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:
    """Charge n et k depuis une feuille Excel. Gère les erreurs."""
    try:
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Le fichier Excel '{file_path}' est introuvable.")

        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        numeric_df = numeric_df.dropna(how='all')
        if numeric_df.shape[1] < 3:
            raise ValueError(f"Feuille '{sheet_name}' ne contient pas 3 colonnes numériques.")
        numeric_df = numeric_df.dropna(subset=[0, 1, 2]) # Utilise les 3 premières colonnes
        numeric_df = numeric_df.sort_values(by=0)
        l_nm = numeric_df.iloc[:, 0].values.astype(np.float64)
        n = numeric_df.iloc[:, 1].values.astype(np.float64)
        k = numeric_df.iloc[:, 2].values.astype(np.float64)
        if len(l_nm) == 0:
            raise ValueError(f"Pas de données numériques valides trouvées dans la feuille '{sheet_name}' du fichier {file_path}")
        # Utiliser print pour le log console, car st.write n'est pas utilisable dans les fonctions cachées
        print(f"ST_CACHE: Données chargées depuis {file_path} (Feuille: '{sheet_name}'): {len(l_nm)} points de {l_nm.min():.1f} nm à {l_nm.max():.1f} nm")
        return l_nm, n, k
    except FileNotFoundError as fnf:
        print(f"ST_CACHE_ERROR: Fichier non trouvé: {fnf}")
        # L'appelant (e.g., _get_nk_array_for_lambda_vec) devra gérer le retour None
        return None, None, None
    except ValueError as ve:
        print(f"ST_CACHE_ERROR: Erreur de valeur lecture feuille '{sheet_name}': {ve}")
        return None, None, None
    except Exception as e:
        print(f"ST_CACHE_ERROR: Erreur inattendue lecture feuille '{sheet_name}': {type(e).__name__} - {e}\n{traceback.format_exc(limit=1)}")
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
    safe_denom1 = jnp.where(jnp.abs(denom1) < 1e-12, jnp.sign(denom1)*1e-12, denom1)
    safe_denom2 = jnp.where(jnp.abs(denom2) < 1e-12, jnp.sign(denom2)*1e-12, denom2)
    safe_denom3 = jnp.where(jnp.abs(denom3) < 1e-12, jnp.sign(denom3)*1e-12, denom3)
    n_sq = 1.0 + (0.6961663 * l_um_sq) / safe_denom1 + \
           (0.4079426 * l_um_sq) / safe_denom2 + \
           (0.8974794 * l_um_sq) / safe_denom3
    # Empêcher les valeurs négatives dans sqrt dues à l'instabilité numérique près des pôles
    n = jnp.sqrt(jnp.maximum(n_sq, 0.0))
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
    safe_denom1 = jnp.where(jnp.abs(denom1) < 1e-12, jnp.sign(denom1)*1e-12, denom1)
    safe_denom2 = jnp.where(jnp.abs(denom2) < 1e-12, jnp.sign(denom2)*1e-12, denom2)
    safe_denom3 = jnp.where(jnp.abs(denom3) < 1e-12, jnp.sign(denom3)*1e-12, denom3)
    n_sq = 1.0 + (1.03961212 * l_um_sq) / safe_denom1 + \
           (0.231792344 * l_um_sq) / safe_denom2 + \
           (1.01046945 * l_um_sq) / safe_denom3
    n = jnp.sqrt(jnp.maximum(n_sq, 0.0))
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
    """Interpole n et k en utilisant jnp.interp."""
    # Assumer l_data est déjà trié (fait dans _get_nk_array_for_lambda_vec)
    n_interp = jnp.interp(l_target, l_data, n_data)
    k_interp = jnp.interp(l_target, l_data, k_data)
    # Assurer k >= 0 (physiquement attendu)
    k_interp = jnp.maximum(k_interp, 0.0)
    return n_interp + 1j * k_interp

# Type Hint pour les définitions de matériaux
MaterialInputType = Union[complex, float, int, str, Tuple[np.ndarray, np.ndarray, np.ndarray]]

# Fonction 6: _get_nk_array_for_lambda_vec
def _get_nk_array_for_lambda_vec(material_definition: MaterialInputType,
                                 l_vec_target_jnp: jnp.ndarray) -> jnp.ndarray:
    """Retourne un tableau JAX d'indices complexes pour le vecteur lambda donné."""
    if isinstance(material_definition, (complex, float, int)):
        # Gérer n+ik constant
        # Assurer k >= 0
        mat_complex = jnp.asarray(material_definition, dtype=jnp.complex128)
        mat_complex_corrected = jnp.where(mat_complex.imag < 0, mat_complex.real + 0j, mat_complex)
        return jnp.full(l_vec_target_jnp.shape, mat_complex_corrected)
    elif isinstance(material_definition, str):
        mat_upper = material_definition.upper()
        if mat_upper == "FUSED SILICA": return get_n_fused_silica(l_vec_target_jnp)
        if mat_upper == "BK7": return get_n_bk7(l_vec_target_jnp)
        if mat_upper == "D263": return get_n_d263(l_vec_target_jnp)
        else: # Assume nom de feuille Excel
            sheet_name = material_definition
            l_data, n_data, k_data = load_material_data_from_xlsx_sheet(EXCEL_FILE_PATH, sheet_name)
            if l_data is None:
                raise ValueError(f"Impossible de charger les données pour '{sheet_name}' depuis {EXCEL_FILE_PATH}")

            # Convertir en JAX arrays et trier
            l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
            sort_indices = jnp.argsort(l_data_jnp)
            l_data_jnp = l_data_jnp[sort_indices]
            n_data_jnp = n_data_jnp[sort_indices]
            k_data_jnp = k_data_jnp[sort_indices]

            # Avertir si extrapolation
            min_target, max_target = jnp.min(l_vec_target_jnp), jnp.max(l_vec_target_jnp)
            min_data, max_data = l_data_jnp[0], l_data_jnp[-1]
            if min_target < min_data - 1e-6 or max_target > max_data + 1e-6: # Tolérance numérique
                 print(f"AVERTISSEMENT: Plage cible [{min_target:.1f}-{max_target:.1f}] hors plage [{min_data:.1f}-{max_data:.1f}] pour '{sheet_name}'. Extrapolation.")

            return interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)
    elif isinstance(material_definition, tuple) and len(material_definition) == 3:
        # Gérer tuple (l, n, k) - utile pour tests ou données ad-hoc
        l_data, n_data, k_data = material_definition
        l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
        sort_indices = jnp.argsort(l_data_jnp)
        l_data_jnp, n_data_jnp, k_data_jnp = l_data_jnp[sort_indices], n_data_jnp[sort_indices], k_data_jnp[sort_indices]
        min_target, max_target = jnp.min(l_vec_target_jnp), jnp.max(l_vec_target_jnp)
        min_data, max_data = l_data_jnp[0], l_data_jnp[-1]
        if min_target < min_data - 1e-6 or max_target > max_data + 1e-6:
             print(f"AVERTISSEMENT: Plage cible hors plage données fournies. Extrapolation.")
        return interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)
    else:
        raise TypeError(f"Type de définition de matériau non supporté : {type(material_definition)}")

# --- Moteur de Calcul JAX : Fonctions Core ---

# Fonction 7: _compute_layer_matrix_scan_step (JAX)
@jax.jit
def _compute_layer_matrix_scan_step(carry_matrix: jnp.ndarray, layer_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, None]:
    """Calcule la matrice d'une couche et la multiplie avec la matrice précédente (pour jax.lax.scan)."""
    thickness, Ni_complex, l_val = layer_data
    eta = Ni_complex # Notation standard : admitance optique (ou indice)
    safe_l_val = jnp.maximum(l_val, 1e-9) # Éviter lambda <= 0

    # Calcul robuste de phi, gérant l'épaisseur nulle ou très petite
    phi_real = (2 * jnp.pi / safe_l_val) * Ni_complex.real * thickness
    phi_imag = (2 * jnp.pi / safe_l_val) * Ni_complex.imag * thickness
    phi = phi_real + 1j * phi_imag

    # Utiliser exp plutôt que cos/sin pour potentielle meilleure stabilité numérique avec k grand
    exp_jphi = jnp.exp(1j * phi)
    exp_neg_jphi = jnp.exp(-1j * phi)
    cos_phi = 0.5 * (exp_jphi + exp_neg_jphi)
    sin_phi = -0.5j * (exp_jphi - exp_neg_jphi) # = sin(phi_real)*cosh(phi_imag) + j*cos(phi_real)*sinh(phi_imag)

    def compute_M_layer(eta_: jnp.complex128) -> jnp.ndarray:
        safe_eta = jnp.where(jnp.abs(eta_) < 1e-12, 1e-12 + 0j, eta_) # Éviter eta = 0
        M_layer = jnp.array([
            [cos_phi,             (1j / safe_eta) * sin_phi],
            [1j * eta_ * sin_phi,  cos_phi]
        ], dtype=jnp.complex128)
        # Ordre de multiplication : Nouvelle Matrice * Matrice Précédente
        return M_layer @ carry_matrix

    def compute_identity(eta_: jnp.complex128) -> jnp.ndarray:
        # Si épaisseur négligeable, la matrice est l'identité => on retourne la matrice précédente
        return carry_matrix

    # Exécution conditionnelle basée sur l'épaisseur physique
    new_matrix = cond(
        thickness > 1e-12,   # Condition: couche physiquement présente
        compute_M_layer,     # Fonction si Vrai
        compute_identity,    # Fonction si Faux
        eta                  # Opérande passé aux fonctions (juste pour la signature)
    )
    # scan attend (carry, output), l'output ici est None car on ne s'intéresse qu'au carry final
    return new_matrix, None

# Fonction 8: compute_stack_matrix_jax (JAX)
@jax.jit
def compute_stack_matrix_jax(ep_vector: jnp.ndarray, l_val: jnp.ndarray,
                             nH_at_lval: jnp.complex128, nL_at_lval: jnp.complex128) -> jnp.ndarray:
    """Calcule la matrice caractéristique totale pour une structure H/L à une longueur d'onde."""
    num_layers = ep_vector.shape[0]
    if num_layers == 0:
        # Retourne la matrice identité si pas de couches
        return jnp.eye(2, dtype=jnp.complex128)

    # Créer le tableau des indices pour chaque couche (alternance H/L)
    # Utilise jax.lax.select pour la JIT-compatibilité
    is_H_layer = jnp.arange(num_layers) % 2 == 0
    layer_indices = jnp.where(is_H_layer, nH_at_lval, nL_at_lval) # Shape (num_layers,)

    # Préparer les données pour scan: (épaisseurs, indices_complexes, lambda unique)
    # Lambda doit être broadcasté pour chaque couche
    l_val_broadcasted = jnp.full(num_layers, l_val)
    layers_scan_data = (ep_vector, layer_indices, l_val_broadcasted)

    # Matrice initiale = Identité
    M_initial = jnp.eye(2, dtype=jnp.complex128)

    # Utiliser scan pour multiplier séquentiellement les matrices de couches
    # La multiplication se fait dans l'ordre : M_N * ... * M_2 * M_1 * M_initial
    M_final, _ = scan(_compute_layer_matrix_scan_step, M_initial, layers_scan_data)
    return M_final

# Fonction 9: calculate_single_wavelength_T (JAX)
@jax.jit
def calculate_single_wavelength_T(l_val: jnp.ndarray, ep_vector_contig: jnp.ndarray,
                                  nH_at_lval: jnp.complex128, nL_at_lval: jnp.complex128, nSub_at_lval: jnp.complex128) -> jnp.ndarray:
    """Calcule la Transmittance T pour une seule longueur d'onde."""
    etainc = 1.0 + 0j  # Milieu incident (air/vide)
    etasub = nSub_at_lval # Milieu substrat (complexe)

    def calculate_for_valid_l(l_: jnp.ndarray) -> jnp.ndarray:
        # Calculer la matrice totale de l'empilement
        M = compute_stack_matrix_jax(ep_vector_contig, l_, nH_at_lval, nL_at_lval)
        m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

        # Calculer le coefficient de transmission (amplitude) 'ts'
        # Dénominateur pour le calcul de réflexion/transmission
        rs_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        rs_denominator_abs = jnp.abs(rs_denominator)
        # Éviter la division par zéro
        safe_denominator = jnp.where(rs_denominator_abs < 1e-12, 1e-12 + 0j, rs_denominator)
        ts = (2.0 * etainc) / safe_denominator

        # Calculer la Transmittance (Intensité) Ts = (Re(etasub)/Re(etainc)) * |ts|^2
        real_etasub = jnp.real(etasub)
        # S'assurer que Re(etasub) est >= 0 pour la physique
        safe_real_etasub = jnp.maximum(real_etasub, 0.0)
        real_etainc = 1.0 # Car etainc = 1.0 + 0j
        Ts = (safe_real_etasub / real_etainc) * jnp.abs(ts)**2 # Utiliser jnp.abs(ts)**2 est équivalent à ts * conj(ts) et potentiellement plus stable

        # Retourner NaN si dénominateur proche de zéro, sinon Ts (qui est réel)
        return jnp.where(rs_denominator_abs < 1e-12, jnp.nan, Ts)

    def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray:
        # Lambda invalide (zéro ou négatif)
        return jnp.nan

    # Utiliser cond pour gérer lambda invalide
    Ts = cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)
    return Ts

# Fonction 10: calculate_T_from_ep_core_jax (JAX)
@jax.jit
def calculate_T_from_ep_core_jax(ep_vector: jnp.ndarray,
                                 nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                 l_vec: jnp.ndarray) -> jnp.ndarray:
    """Calcule T(lambda) pour un vecteur de lambdas (structure H/L)."""
    if not l_vec.size:
        return jnp.zeros(0, dtype=jnp.float64) # Retourner vide si pas de lambdas

    # Assurer que ep_vector est un array JAX
    ep_vector_contig = jnp.asarray(ep_vector)

    # Vectoriser le calcul pour une seule longueur d'onde sur tout le vecteur lambda
    # vmap applique la fonction sur les axes spécifiés des tableaux d'entrée
    # Axe 0 pour l_vec, nH_arr, nL_arr, nSub_arr (varie avec lambda)
    # None pour ep_vector_contig (constant pour tous les lambdas)
    Ts_arr = vmap(calculate_single_wavelength_T, in_axes=(0, None, 0, 0, 0))(
        l_vec, ep_vector_contig, nH_arr, nL_arr, nSub_arr
    )
    # Remplacer les NaN potentiels (par ex. lambda invalide ou calcul instable) par 0.0
    return jnp.nan_to_num(Ts_arr, nan=0.0)

# --- Fonctions Core pour Séquence Arbitraire (JAX) ---

# Fonction 11: _compute_layer_matrix_scan_step_arbitrary (JAX)
@jax.jit
def _compute_layer_matrix_scan_step_arbitrary(carry_matrix: jnp.ndarray, layer_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, None]:
    """Étape de scan pour séquence arbitraire (identique à la version HL pour l'instant)."""
    # Peut différer à l'avenir si des logiques spécifiques sont nécessaires
    return _compute_layer_matrix_scan_step(carry_matrix, layer_data)

# Fonction 12: compute_stack_matrix_arbitrary_jax (JAX)
@jax.jit
def compute_stack_matrix_arbitrary_jax(ep_vector: jnp.ndarray, layer_indices_at_lval: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    """Calcule la matrice totale pour une séquence arbitraire à une longueur d'onde."""
    num_layers = ep_vector.shape[0]
    if num_layers == 0:
        return jnp.eye(2, dtype=jnp.complex128)

    # layer_indices_at_lval contient les n+ik pour chaque couche à ce lambda (shape (num_layers,))
    l_val_broadcasted = jnp.full(num_layers, l_val)
    layers_scan_data = (ep_vector, layer_indices_at_lval, l_val_broadcasted)

    M_initial = jnp.eye(2, dtype=jnp.complex128)
    # Utiliser la fonction d'étape de scan (potentiellement spécifique à l'arbitraire à l'avenir)
    M_final, _ = scan(_compute_layer_matrix_scan_step_arbitrary, M_initial, layers_scan_data)
    return M_final

# Fonction 13: calculate_single_wavelength_T_arbitrary (JAX)
@jax.jit
def calculate_single_wavelength_T_arbitrary(l_val: jnp.ndarray, ep_vector_contig: jnp.ndarray,
                                            layer_indices_at_lval: jnp.ndarray, nSub_at_lval: jnp.complex128) -> jnp.ndarray:
    """Calcule T pour une seule lambda, séquence arbitraire."""
    etainc = 1.0 + 0j
    etasub = nSub_at_lval

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

    def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray:
        return jnp.nan

    Ts = cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)
    return Ts

# Fonction 14: calculate_T_from_ep_arbitrary_core_jax (JAX)
@jax.jit
def calculate_T_from_ep_arbitrary_core_jax(ep_vector: jnp.ndarray,
                                           layer_indices_arr: jnp.ndarray, # Shape (num_layers, num_lambdas)
                                           nSub_arr: jnp.ndarray,          # Shape (num_lambdas,)
                                           l_vec: jnp.ndarray) -> jnp.ndarray: # Shape (num_lambdas,)
    """Calcule T(lambda) pour un vecteur lambda, séquence arbitraire."""
    if not l_vec.size:
        return jnp.zeros(0, dtype=jnp.float64)

    ep_vector_contig = jnp.asarray(ep_vector) # Shape (num_layers,)
    num_layers = ep_vector_contig.shape[0]

    # Vérifier la cohérence des dimensions
    if num_layers > 0 and layer_indices_arr.shape[0] != num_layers:
         # Cette vérification est difficile à faire dans JIT. Assumer que les shapes sont correctes.
         # On pourrait utiliser jax.debug.print pour afficher les shapes si nécessaire pendant le débogage.
         # jax.debug.print("Shape mismatch: ep_vector {ep} vs layer_indices {li}", ep=ep_vector_contig.shape, li=layer_indices_arr.shape)
         pass

    # La fonction vmap attend les indices pour une seule lambda à la fois.
    # L'entrée layer_indices_arr a la shape (num_layers, num_lambdas).
    # On a besoin de mapper sur la dimension lambda (axis 1). Transposer d'abord.
    # Shape transposée : (num_lambdas, num_layers)
    # Gérer le cas où il n'y a pas de couches
    if num_layers == 0:
        # Si pas de couches, T dépend seulement de l'interface air/substrat
        etainc = 1.0 + 0j
        etasub_batch = nSub_arr
        # Coefficient de transmission de Fresnel pour l'amplitude
        ts_bare = 2 * etainc / (etainc + etasub_batch)
        safe_real_etasub = jnp.maximum(jnp.real(etasub_batch), 0.0)
        Ts_arr = (safe_real_etasub / 1.0) * jnp.abs(ts_bare)**2
    else:
        layer_indices_arr_transposed = layer_indices_arr.T
        # Vmap sur:
        # l_vec (axis 0) -> fournit l_val
        # ep_vector_contig (None) -> même ep_vector pour tous les lambdas
        # layer_indices_arr_transposed (axis 0) -> fournit layer_indices_at_lval pour chaque lambda
        # nSub_arr (axis 0) -> fournit nSub_at_lval pour chaque lambda
        Ts_arr = vmap(calculate_single_wavelength_T_arbitrary, in_axes=(0, None, 0, 0))(
            l_vec, ep_vector_contig, layer_indices_arr_transposed, nSub_arr
        )

    return jnp.nan_to_num(Ts_arr, nan=0.0)

# Fonction 15: _get_nk_at_lambda
def _get_nk_at_lambda(material_definition: MaterialInputType, l_nm_target: float) -> complex:
    """Obtient n+ik pour un matériau donné à une longueur d'onde spécifique."""
    if isinstance(material_definition, (complex, float, int)):
        mat_complex = complex(material_definition)
        # Assurer k>=0
        return complex(mat_complex.real, max(0.0, mat_complex.imag))

    l_nm_target_jnp = jnp.array([l_nm_target], dtype=jnp.float64) # Pour fonctions Sellmeier JAX

    if isinstance(material_definition, str):
        mat_upper = material_definition.upper()
        if mat_upper == "FUSED SILICA": return complex(get_n_fused_silica(l_nm_target_jnp)[0])
        if mat_upper == "BK7": return complex(get_n_bk7(l_nm_target_jnp)[0])
        if mat_upper == "D263": return complex(get_n_d263(l_nm_target_jnp)[0])
        else: # Feuille Excel
            sheet_name = material_definition
            l_data, n_data, k_data = load_material_data_from_xlsx_sheet(EXCEL_FILE_PATH, sheet_name)
            if l_data is None: raise ValueError(f"Impossible de charger les données pour '{sheet_name}'")
            # Utiliser interp NumPy pour un seul point (plus simple hors JIT)
            # Assurer le tri pour interp NumPy
            sort_idx = np.argsort(l_data)
            l_data_s, n_data_s, k_data_s = l_data[sort_idx], n_data[sort_idx], k_data[sort_idx]
            n_interp = np.interp(l_nm_target, l_data_s, n_data_s)
            k_interp = np.interp(l_nm_target, l_data_s, k_data_s)
            return complex(n_interp, max(0.0, k_interp)) # Assurer k>=0
    elif isinstance(material_definition, tuple) and len(material_definition) == 3:
        l_data, n_data, k_data = material_definition
        sort_idx = np.argsort(l_data)
        l_data_s, n_data_s, k_data_s = l_data[sort_idx], n_data[sort_idx], k_data[sort_idx]
        n_interp = np.interp(l_nm_target, l_data_s, n_data_s)
        k_interp = np.interp(l_nm_target, l_data_s, k_data_s)
        return complex(n_interp, max(0.0, k_interp))
    else:
        raise TypeError(f"Type de définition de matériau non supporté : {type(material_definition)}")

# Fonction 16: get_target_points_indices_jax (JAX Helper)
@jax.jit
def get_target_points_indices_jax(l_vec: jnp.ndarray, target_min: float, target_max: float) -> jnp.ndarray:
    """Trouve les indices des points de l_vec dans l'intervalle [target_min, target_max]."""
    if not l_vec.size: return jnp.empty(0, dtype=jnp.int64)
    # Utiliser jnp.where pour trouver les indices, size= garantit une taille fixe pour JIT
    indices_with_fill = jnp.where((l_vec >= target_min) & (l_vec <= target_max),
                                  jnp.arange(l_vec.shape[0]),
                                  -1) # Marque les points hors condition avec -1
    # Filtrer les valeurs de remplissage (-1) pour obtenir seulement les indices valides
    valid_indices = indices_with_fill[indices_with_fill != -1]
    return valid_indices.astype(jnp.int64) # Assurer le type de retour

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
    except ValueError as e: # Propager l'erreur de chargement matériau
         print(f"ERREUR obtention indices à l0={l0}nm pour calcul initial: {e}")
         raise ValueError(f"Erreur obtention indices à l0={l0}nm: {e}") from e
    except Exception as e: # Autre erreur inattendue
        print(f"ERREUR inattendue obtention indices à l0={l0}nm : {e}")
        raise RuntimeError(f"Erreur inattendue obtention indices à l0={l0}nm : {e}") from e

    for i in range(num_layers):
        multiplier = emp_qwot_list[i]
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real_layer_at_l0 > 1e-9:
            # Calcul standard QWOT -> épaisseur physique
            ep_initial[i] = multiplier * l0 / (4.0 * n_real_layer_at_l0)
        else:
            ep_initial[i] = 0.0 # Épaisseur nulle si indice invalide

    return ep_initial

# Fonction 18: calculate_qwot_from_ep
def calculate_qwot_from_ep(ep_vector: np.ndarray, l0: float,
                           nH0_material: MaterialInputType, nL0_material: MaterialInputType) -> np.ndarray:
    """Calcule les multiplicateurs QWOT à partir des épaisseurs physiques."""
    num_layers = len(ep_vector)
    qwot_multipliers = np.full(num_layers, np.nan, dtype=np.float64) # Initialiser avec NaN
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
    except ValueError as e: # Propager l'erreur de chargement
         print(f"ERREUR obtention indices à l0={l0}nm pour calcul QWOT: {e}")
         # Retourner NaN en cas d'erreur, mais lever une exception serait peut-être mieux
         raise ValueError(f"Erreur obtention indices à l0={l0}nm pour QWOT: {e}") from e
    except Exception as e:
        print(f"ERREUR inattendue obtention indices à l0={l0}nm pour QWOT: {e}")
        raise RuntimeError(f"Erreur inattendue obtention indices à l0={l0}nm pour QWOT: {e}") from e

    for i in range(num_layers):
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real_layer_at_l0 > 1e-9 and l0 > 1e-9:
            # Calcul QWOT = ep * 4 * n / l0
            qwot_multipliers[i] = ep_vector[i] * (4.0 * n_real_layer_at_l0) / l0
        # else: reste NaN si indice ou l0 invalide

    return qwot_multipliers

# Fonction 19: calculate_final_mse
def calculate_final_mse(res: Dict[str, np.ndarray], active_targets: List[Dict]) -> Tuple[Union[float, None], int]:
    """Calcule le MSE final basé sur les résultats et les cibles actives (version NumPy)."""
    total_squared_error = 0.0
    total_points_in_targets = 0
    mse = None # Défaut à None

    # Validation basique des entrées
    if not active_targets or 'Ts' not in res or res['Ts'] is None or 'l' not in res or res['l'] is None:
        print("AVERTISSEMENT: Entrées invalides pour calculate_final_mse.")
        return mse, total_points_in_targets # Retourne None, 0

    res_l_np = np.asarray(res['l'])
    res_ts_np = np.asarray(res['Ts'])

    if res_l_np.size == 0 or res_ts_np.size == 0 or res_l_np.size != res_ts_np.size:
         print("AVERTISSEMENT: Données de résultats vides ou incohérentes pour calculate_final_mse.")
         return mse, total_points_in_targets

    for target in active_targets:
        # Assurer que les clés existent et sont valides
        try:
            l_min, l_max = float(target['min']), float(target['max'])
            t_min, t_max = float(target['target_min']), float(target['target_max'])
        except (KeyError, ValueError, TypeError):
             print(f"AVERTISSEMENT: Cible invalide ignorée dans calculate_final_mse: {target}")
             continue # Ignorer cette cible

        # Trouver les indices des points calculés dans la zone lambda de la cible
        indices = np.where((res_l_np >= l_min) & (res_l_np <= l_max))[0]

        if indices.size > 0:
            calculated_Ts_in_zone = res_ts_np[indices]
            target_lambdas_in_zone = res_l_np[indices]

            # Filtrer les NaN/inf potentiels dans les données calculées
            finite_mask = np.isfinite(calculated_Ts_in_zone)
            calculated_Ts_in_zone = calculated_Ts_in_zone[finite_mask]
            target_lambdas_in_zone = target_lambdas_in_zone[finite_mask]

            if calculated_Ts_in_zone.size == 0: continue # Passer à la cible suivante si pas de points finis

            # Calculer la valeur de transmittance cible pour chaque point dans la zone
            if abs(l_max - l_min) < 1e-9: # Cible constante
                interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
            else: # Cible en rampe
                slope = (t_max - t_min) / (l_max - l_min)
                interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)
                # Clipper les valeurs cibles entre 0 et 1 si la rampe sort de l'intervalle
                interpolated_target_t = np.clip(interpolated_target_t, 0.0, 1.0)

            # Calculer les erreurs quadratiques et accumuler
            squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
            total_squared_error += np.sum(squared_errors)
            total_points_in_targets += len(calculated_Ts_in_zone)

    # Calculer le MSE final
    if total_points_in_targets > 0:
        mse = total_squared_error / total_points_in_targets
    elif active_targets: # Si cibles existent mais aucun point trouvé dans les zones
        mse = np.nan # Indiquer que le MSE n'est pas calculable

    return mse, total_points_in_targets

# Fonction 20: calculate_mse_for_optimization_penalized_jax (JAX Cost Function)
@jax.jit
def calculate_mse_for_optimization_penalized_jax(ep_vector: jnp.ndarray,
                                                nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                                l_vec_optim: jnp.ndarray,
                                                active_targets_tuple: Tuple[Tuple[float, float, float, float], ...],
                                                min_thickness_phys_nm: float) -> jnp.ndarray:
    """Fonction de coût JIT pour l'optimisation (MSE + pénalité couche mince)."""
    # --- Pénalité Couche Mince ---
    # Identifier les couches trop fines (mais non nulles, qu'on ignore)
    below_min_mask = (ep_vector < min_thickness_phys_nm) & (ep_vector > 1e-12)
    # Calculer une pénalité quadratique, fortement pondérée
    # Utilise une échelle relative pour la pénalité : (1 - ep/min_thickness)^2
    relative_thinness_sq = jnp.where(below_min_mask, (min_thickness_phys_nm - ep_vector) / min_thickness_phys_nm, 0.0)**2
    # Facteur de pénalité élevé (à ajuster si besoin)
    # penalty_thin = jnp.sum(relative_thinness_sq) * 1e3 # Ancien facteur * 1e5
    # Test avec un facteur de pénalité peut-être moins agressif pour commencer
    penalty_factor = 100.0
    penalty_thin = jnp.sum(relative_thinness_sq) * penalty_factor

    # --- Calcul MSE ---
    # Pour le calcul T lui-même, clamper les épaisseurs à la valeur minimale physique
    # Évite problèmes avec épaisseurs nulles/négatives dans calcul matriciel.
    # La pénalité ci-dessus gère la pression d'optimisation contre les couches fines.
    ep_vector_calc = jnp.maximum(ep_vector, min_thickness_phys_nm)

    # Calculer Transmittance avec la fonction core JAX
    Ts = calculate_T_from_ep_core_jax(ep_vector_calc, nH_arr, nL_arr, nSub_arr, l_vec_optim)

    total_squared_error = 0.0
    total_points_in_targets = 0

    # Boucle sur les zones cibles actives (passées comme tuple pour JIT)
    for i in range(len(active_targets_tuple)):
        l_min, l_max, t_min, t_max = active_targets_tuple[i]

        # Masque booléen pour les points dans la plage lambda de la cible courante
        target_mask = (l_vec_optim >= l_min) & (l_vec_optim <= l_max)

        # Calculer la valeur de transmittance cible pour tous les points (sera masquée ensuite)
        # Gérer le cas d'une cible plate (l_max == l_min)
        is_flat_target = jnp.abs(l_max - l_min) < 1e-9
        denom_slope = jnp.where(is_flat_target, 1.0, l_max - l_min) # Éviter division par 0
        slope = jnp.where(is_flat_target, 0.0, (t_max - t_min) / denom_slope)
        interpolated_target_t_full = t_min + slope * (l_vec_optim - l_min)
        # Clipper les valeurs cibles entre 0 et 1
        interpolated_target_t_full = jnp.clip(interpolated_target_t_full, 0.0, 1.0)


        # Calculer les erreurs quadratiques pour tous les points
        # Rendre la différence robuste aux NaN potentiels dans Ts (même si nan_to_num est utilisé après)
        ts_finite = jnp.nan_to_num(Ts, nan=0.0)
        squared_errors_full = (ts_finite - interpolated_target_t_full)**2

        # Appliquer le masque : considérer seulement les erreurs dans la zone cible
        # S'assurer que les erreurs NaN ne contribuent pas (même si peu probable ici)
        masked_sq_error = jnp.where(target_mask & jnp.isfinite(squared_errors_full), squared_errors_full, 0.0)

        # Accumuler l'erreur totale et compter les points dans la cible
        total_squared_error += jnp.sum(masked_sq_error)
        total_points_in_targets += jnp.sum(target_mask) # Compter tous les points dans la zone, même si Ts était NaN

    # Calculer l'Erreur Quadratique Moyenne, utiliser une grande valeur si aucun point dans les cibles
    # Utiliser une valeur très grande comme 1e10 pour indiquer un mauvais état initial
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, 1e10)

    # --- Coût Final ---
    # Combiner MSE et pénalité couche mince
    final_cost = mse + penalty_thin

    # Retourner une valeur finie, par défaut l'infinité si NaN se produit
    # (devrait être géré par nan_to_num plus tôt, mais par sécurité)
    return jnp.nan_to_num(final_cost, nan=jnp.inf)


# ============================================================
# La suite du code (fonctions 21+) définirait :
# - Les fonctions restantes du cœur métier (optimisation, etc.)
# - Les fonctions helper Streamlit (initialisation état, logs, validation, etc.)
# - Les callbacks pour les actions UI
# - La structure de l'interface utilisateur Streamlit (sidebar, main area)
# - L'affichage des graphiques et des résultats
# ============================================================

# [PREVIOUS CODE HERE - Imports, Config, CSS, Constants, Functions 1-20]
# ... (Assume lines 1-approx 420 are present)

# ============================================================
# 2. CŒUR MÉTIER : 20 FONCTIONS SUIVANTES (Adaptées pour Streamlit)
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

        ts_finite = jnp.nan_to_num(Ts, nan=0.0)
        squared_errors_full = (ts_finite - interpolated_target_t_full)**2
        masked_sq_error = jnp.where(target_mask & jnp.isfinite(squared_errors_full), squared_errors_full, 0.0)
        total_squared_error += jnp.sum(masked_sq_error)
        total_points_in_targets += jnp.sum(target_mask)

    # Calculer MSE, par défaut l'infinité si aucun point dans les cibles
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, jnp.inf)

    # Retourner MSE, gérant les NaN potentiels
    return jnp.nan_to_num(mse, nan=jnp.inf)


# Fonction 22: _run_core_optimization (Déjà définie et adaptée dans la réponse précédente, incluse ici pour continuité)
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
        nH_arr_optim = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax)
        nL_arr_optim = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax)
        nSub_arr_optim = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax)
        logs.append(f"{log_prefix} Préparation indices terminée en {time.time() - prep_start_time:.3f}s.")

        active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
        static_args_for_jax = (nH_arr_optim, nL_arr_optim, nSub_arr_optim, l_vec_optim_jax, active_targets_tuple, min_thickness_phys)
        # Utiliser la fonction coût pénalisée
        value_and_grad_fn = jax.jit(jax.value_and_grad(calculate_mse_for_optimization_penalized_jax))

        # Wrapper Scipy (identique à la version précédente)
        def scipy_obj_grad_wrapper(ep_vector_np, *args):
            ep_vector_jax = jnp.asarray(ep_vector_np)
            value_jax, grad_jax = value_and_grad_fn(ep_vector_jax, *args)
            value_np = float(np.array(value_jax))
            grad_np = np.array(grad_jax, dtype=np.float64)
            # Log verbeux optionnel : print(f" Cost={value_np:.4e} GradNorm={np.linalg.norm(grad_np):.3e}")
            return value_np, grad_np

        lbfgsb_bounds = [(min_thickness_phys, None)] * num_layers_start
        options = {'maxiter': inputs['maxiter'], 'maxfun': inputs['maxfun'], 'disp': False, 'ftol': 1e-12, 'gtol': 1e-8}

        logs.append(f"{log_prefix}Lancement L-BFGS-B (grad JAX) - maxiter={options['maxiter']}...")
        opt_start_time = time.time()
        result = minimize(scipy_obj_grad_wrapper, np.asarray(ep_start_optim, dtype=np.float64),
                          args=static_args_for_jax, method='L-BFGS-B', jac=True, bounds=lbfgsb_bounds, options=options)
        logs.append(f"{log_prefix}L-BFGS-B (grad JAX) terminé en {time.time() - opt_start_time:.3f}s.")

        # Traitement résultats (identique à la version précédente)
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

    except Exception as e_optim:
        logs.append(f"{log_prefix}ERREUR durant optimisation (Setup JAX ou run): {e_optim}\n{traceback.format_exc(limit=1)}")
        # Assurer que ep_start_optim est défini avant de l'utiliser ici
        if 'ep_start_optim' in locals():
             final_ep = np.maximum(ep_start_optim, min_thickness_phys)
        else: # Cas où l'erreur se produit avant la définition de ep_start_optim
             final_ep = np.array([]) # Retourner un tableau vide par sécurité
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
    ep_after_merge = None # Initialiser le résultat

    # Vérifications de base
    if num_layers <= 2 and target_layer_index is None:
        logs.append(f"{log_prefix}Structure <= 2 couches. Fusion/suppression impossible sans cible spécifique.")
        return current_ep, False, logs
    elif num_layers < 1:
        logs.append(f"{log_prefix}Structure vide.")
        return current_ep, False, logs

    try:
        thin_layer_index = -1
        min_thickness_found = np.inf

        # --- Déterminer la couche cible ---
        if target_layer_index is not None: # Si une couche spécifique est ciblée
            if 0 <= target_layer_index < num_layers:
                if current_ep[target_layer_index] >= min_thickness_phys:
                    thin_layer_index = target_layer_index
                    min_thickness_found = current_ep[target_layer_index]
                    logs.append(f"{log_prefix}Ciblage couche {thin_layer_index + 1} ({min_thickness_found:.3f} nm).")
                else:
                    logs.append(f"{log_prefix}Couche cible {target_layer_index+1} trop fine (< {min_thickness_phys:.3f} nm). Recherche de la plus fine.")
                    target_layer_index = None # Fallback
            else:
                 logs.append(f"{log_prefix}Indice couche cible {target_layer_index+1} invalide. Recherche de la plus fine.")
                 target_layer_index = None # Fallback

        if target_layer_index is None: # Trouver la couche la plus fine (potentiellement sous un seuil)
            candidate_indices = np.where(current_ep >= min_thickness_phys)[0]
            if candidate_indices.size > 0:
                candidate_thicknesses = current_ep[candidate_indices]
                if threshold_for_removal is not None:
                    valid_for_removal_mask = candidate_thicknesses < threshold_for_removal
                    if np.any(valid_for_removal_mask):
                        candidate_indices = candidate_indices[valid_for_removal_mask]
                        candidate_thicknesses = candidate_thicknesses[valid_for_removal_mask]
                    else:
                        candidate_indices = np.array([], dtype=int) # Aucune couche sous le seuil

                if candidate_indices.size > 0:
                    min_idx_local = np.argmin(candidate_thicknesses)
                    thin_layer_index = candidate_indices[min_idx_local]
                    min_thickness_found = candidate_thicknesses[min_idx_local]

        # --- Exécuter Suppression/Fusion ---
        if thin_layer_index == -1:
            if threshold_for_removal is not None and target_layer_index is None:
                logs.append(f"{log_prefix}Aucune couche trouvée >= {min_thickness_phys:.3f} nm ET < {threshold_for_removal:.3f} nm.")
            else:
                logs.append(f"{log_prefix}Aucune couche valide (>= {min_thickness_phys:.3f} nm) trouvée pour suppression/fusion.")
            return current_ep, False, logs

        # Log de la couche identifiée
        thin_layer_thickness = current_ep[thin_layer_index]
        log_details = f"Couche {thin_layer_index + 1} ({thin_layer_thickness:.3f} nm)"
        if threshold_for_removal is not None and target_layer_index is None :
             logs.append(f"{log_prefix}Trouvé couche sous seuil: {log_details}.")
        elif target_layer_index is None:
             logs.append(f"{log_prefix}Couche la plus fine: {log_details}.")

        merged_info = ""
        # Cas 1: Première couche (index 0)
        if thin_layer_index == 0:
            if num_layers >= 2:
                ep_after_merge = current_ep[2:] # Supprimer couches 0 et 1
                merged_info = "Supprimé couches 1 & 2."
                logs.append(f"{log_prefix}{merged_info} Nouvelle taille: {len(ep_after_merge)}.")
                structure_changed = True
            else:
                logs.append(f"{log_prefix}Impossible supprimer couches 1 & 2 - structure trop petite.")
                return current_ep, False, logs

        # Cas 2: Dernière couche (index N-1)
        elif thin_layer_index == num_layers - 1:
             # Supprimer seulement la dernière couche
             ep_after_merge = current_ep[:-1]
             merged_info = f"Supprimé seulement la dernière couche {num_layers}."
             logs.append(f"{log_prefix}{merged_info} Nouvelle taille: {len(ep_after_merge)}.")
             structure_changed = True

        # Cas 3: Couche interne
        else:
            if thin_layer_index > 0 and thin_layer_index + 1 < num_layers:
                merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
                ep_after_merge = np.concatenate((
                    current_ep[:thin_layer_index - 1], # Avant voisin gauche
                    [merged_thickness],               # Couche fusionnée
                    current_ep[thin_layer_index + 2:] # Après voisin droit
                ))
                merged_info = f"Supprimé couche {thin_layer_index+1}, fusionné {thin_layer_index} & {thin_layer_index+2} -> {merged_thickness:.3f}"
                logs.append(f"{log_prefix}{merged_info} Nouvelle taille: {len(ep_after_merge)}.")
                structure_changed = True
            else: # Ne devrait pas arriver si interne
                logs.append(f"{log_prefix}Impossible fusionner pour couche {thin_layer_index+1} - cas limite.")
                return current_ep, False, logs

        # --- Finalisation ---
        if structure_changed and ep_after_merge is not None:
            if ep_after_merge.size > 0:
                # Assurer épaisseur minimale pour les couches restantes (surtout la fusionnée)
                ep_after_merge = np.maximum(ep_after_merge, min_thickness_phys)
            return ep_after_merge, True, logs
        else:
            if structure_changed: logs.append(f"{log_prefix}Changement structure indiqué mais résultat None. Annulation.")
            return current_ep, False, logs # Retourner original si pas de changement ou erreur

    except Exception as e_merge:
         logs.append(f"{log_prefix}ERREUR durant logique fusion/suppression: {e_merge}\n{traceback.format_exc(limit=1)}")
         return current_ep, False, logs # Retourner original en cas d'erreur inattendue

# Fonction 24: _perform_needle_insertion_scan_st (Adaptée)
# (Déjà définie dans la réponse précédente, fonction #21 logiquement)
# Placeholder pour ne pas répéter le code complet
def _perform_needle_insertion_scan_st(*args, **kwargs):
     # ... (Logique de la fonction 21 de la réponse précédente) ...
     pass

# Fonction 25: _run_needle_iterations_st (Adaptée)
# (Déjà définie dans la réponse précédente, fonction #22 logiquement)
# Placeholder pour ne pas répéter le code complet
def _run_needle_iterations_st(*args, **kwargs):
     # ... (Logique de la fonction 22 de la réponse précédente) ...
     pass

# Fonction 26: setup_axis_grids_st (Helper Plotting)
# (Déjà définie dans la réponse précédente, fonction #23 logiquement)
def setup_axis_grids_st(ax):
    """Configure les grilles standard pour un axe Matplotlib."""
    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.3, alpha=0.5)
    ax.minorticks_on()

# Fonction 27: draw_plots_st (Helper Plotting)
# (Déjà définie dans la réponse précédente, fonction #24 logiquement)
# Placeholder pour ne pas répéter le code complet
def draw_plots_st(*args, **kwargs):
     # ... (Logique de la fonction 24 de la réponse précédente) ...
     pass

# ============================================================
# 3. STREAMLIT UI HELPER FUNCTIONS & STATE MANAGEMENT (Suite)
# ============================================================

# Fonction 28: log_message
# (Déjà définie dans la réponse précédente, fonction #25 logiquement)
# Placeholder
def log_message(message): pass

# Fonction 29: update_status
# (Déjà définie dans la réponse précédente, fonction #26 logiquement)
# Placeholder
def update_status(message): pass

# Fonction 30: initialize_session_state
def initialize_session_state():
    """Initialise les variables nécessaires dans st.session_state si elles n'existent pas."""
    # Définir les valeurs par défaut
    # Vérifier d'abord si les listes de matériaux sont déjà chargées
    if 'available_materials' not in st.session_state:
         update_available_materials() # Charge depuis Excel et met à jour les listes dans l'état

    # Déterminer H/L par défaut basé sur ce qui est disponible
    available_mats = st.session_state.get('available_materials', ["Constant"])
    desired_H = "Nb2O5-Helios"; desired_L = "SiO2-Helios"; fallback = "Constant"
    default_H = desired_H if desired_H in available_mats else next((m for m in available_mats if m != fallback), fallback)
    # Logique de fallback pour L (éviter de prendre H si H est le seul dispo autre que Constant)
    potential_L = [m for m in available_mats if m != fallback and m != default_H]
    default_L = desired_L if desired_L in available_mats else (potential_L[0] if potential_L else fallback)

    defaults = {
        'selected_H_material': default_H,
        'selected_L_material': default_L,
        'selected_Sub_material': "Fused Silica" if "Fused Silica" in st.session_state.get('available_substrates',[]) else "Constant",
        'nH_r': 2.35, 'nH_i': 0.0,   # <--- Nombres (float)
        'nL_r': 1.46, 'nL_i': 0.0,   # <--- Nombres (float)
        'nSub': 1.52,               # <--- Nombre (float)
        'emp_str': "1", # QWOT initial défaut
        'initial_layer_number': 1, # Doit être string pour number_input si format %d
        'l0': 500.0,
        'l_step': 10.0,
        'maxiter': 1000, # Doit être string pour number_input si format %d
        'maxfun': 1000,  # Doit être string pour number_input si format %d
        'auto_thin_threshold': 1.0,
        'target_entries': [ # Liste de dictionnaires pour les cibles par défaut
            {'min': 400.0, 'max': 500.0, 'target_min': 1.0, 'target_max': 1.0, 'enabled': True},
            {'min': 500.0, 'max': 600.0, 'target_min': 1.0, 'target_max': 0.2, 'enabled': True},
            {'min': 600.0, 'max': 700.0, 'target_min': 0.2, 'target_max': 0.2, 'enabled': True},
            {'min': 700.0, 'max': 800.0, 'target_min': 0.2, 'target_max': 0.8, 'enabled': False},
            {'min': 800.0, 'max': 900.0, 'target_min': 0.8, 'target_max': 0.8, 'enabled': False},
        ],
        'log_messages': [f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Application démarrée."],
        'current_optimized_ep': None,
        'current_material_sequence': None,
        'optimization_ran': False,
        'ep_history_deque': deque(maxlen=5),
        'optimized_qwot_display': "",
        'status_message': "Prêt",
        'last_calc_results_plot': None,
        'last_calc_results_optim_grid': None,
        'last_calc_mse': None,
        'last_calc_is_optimized': False,
        'last_calc_method_name': "",
        'last_calc_material_sequence': None,
        'figure_main': None,
        'figure_indices': None,
        'figure_preview': None,
        'show_indices_plot': False, # Contrôle l'affichage du graphe indices
        'show_target_preview': False, # Contrôle l'affichage graphe preview
        # Note: available_materials, available_substrates, excel_materials sont déjà initialisés par update_available_materials()
    }

    # Appliquer les défauts seulement si la clé n'existe pas déjà
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # S'assurer que le nombre de couches initial correspond au QWOT initial
    try:
        num_layers_init_qwot = len([item for item in st.session_state.emp_str.split(',') if item.strip()])
        st.session_state.initial_layer_number = str(max(1, num_layers_init_qwot)) # Assurer au moins 1
    except Exception:
        st.session_state.initial_layer_number = 1


# Fonction 31: get_available_materials_from_excel_st
# (Déjà définie dans la réponse précédente, fonction #27 logiquement)
# Placeholder
def get_available_materials_from_excel_st(excel_path: str) -> List[str]: pass

# Fonction 32: update_available_materials
# (Déjà définie dans la réponse précédente, fonction #28 logiquement)
# Placeholder
def update_available_materials(): pass

# Fonction 33: validate_inputs
# (Déjà définie dans la réponse précédente, fonction #29 logiquement)
# Placeholder
def validate_inputs(require_optim_params: bool = False, require_initial_layers: bool = False) -> Dict[str, Any]: pass

# Fonction 34: get_active_targets_from_state
# (Déjà définie dans la réponse précédente, fonction #30 logiquement)
# Placeholder
def get_active_targets_from_state(ignore_errors=False): pass

# Fonction 35: get_lambda_range_from_targets
# (Déjà définie dans la réponse précédente, fonction #31 logiquement)
# Placeholder
def get_lambda_range_from_targets(active_targets): pass

# Fonction 36: draw_target_preview_st
# (Déjà définie dans la réponse précédente, fonction #32 logiquement)
# Placeholder
def draw_target_preview_st(): pass

# Fonction 37: draw_material_index_plot_st
# (Déjà définie dans la réponse précédente, fonction #33 logiquement)
# Placeholder
def draw_material_index_plot_st(): pass

# ============================================================
# 4. CALLBACKS STREAMLIT (pour widgets)
# ============================================================

# Fonction 38: on_material_change
# (Déjà définie dans la réponse précédente, fonction #34 logiquement)
# Placeholder
def on_material_change(): pass

# Fonction 39: on_qwot_change
# (Déjà définie dans la réponse précédente, fonction #35 logiquement)
# Placeholder
def on_qwot_change(): pass

# Fonction 40: on_initial_layer_change
# (Déjà définie dans la réponse précédente, fonction #36 logiquement)
# Placeholder
def on_initial_layer_change(): pass

# Callback pour les changements dans les cibles (checkbox ou valeurs)
def on_target_change():
    """Callback si une cible (valeur ou état activé) change."""
    # Forcer la regénération du graphique de prévisualisation
    st.session_state.figure_preview = None
    log_message("Cible modifiée. Mise à jour de la prévisualisation.")
    # La validation et le calcul du nb de points sont faits au moment de l'action

# --- FIN DES 20 FONCTIONS SUIVANTES ---
# [PREVIOUS CODE HERE - Imports, Config, CSS, Constants, Core Calcs, State Init, Sidebar UI, Main UI Start, Callbacks 1-21 (on_...)]
# ... (Assume lines 1-approx 920 are present)

# ============================================================
# 4. CALLBACKS STREAMLIT (Suite : Actions Principales Complexes)
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

            # Avertissement si très long (Streamlit n'a pas de messagebox bloquant natif)
            warning_threshold_comb = 2**21 # ~2 million
            if num_combinations > warning_threshold_comb:
                 st.warning(f"**Attention:** Le scan pour N={initial_layer_number} ({num_combinations:,} comb.) sur {num_l0_tests} valeurs de l0 sera **très long** ! L'interface peut sembler figée.")
                 time.sleep(3) # Laisser le temps de lire l'avertissement

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
                    emp_list_cand = list(cand_mult) # Convertir numpy array en liste pour calculate_initial_ep
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
                    # Continuer avec le candidat suivant

            # --- Résultats Finaux ---
            if final_best_ep is None:
                raise RuntimeError("Optimisation locale échouée pour tous les candidats du scan.")

            log_message("\n--- Meilleur Résultat Global Après Optimisations Locales ---")
            log_message(f"Meilleur MSE Final: {final_best_mse:.6e}")
            log_message(f"Provenant de l0 = {final_best_l0:.2f} nm")
            best_mult_str = ",".join([f"{m:.3f}" for m in final_best_initial_multipliers])
            log_message(f"Séquence Multiplicateur Originale ({initial_layer_number} couches): {best_mult_str}")

            # Stocker le meilleur résultat
            st.session_state.current_optimized_ep = final_best_ep.copy()
            st.session_state.current_material_sequence = None
            st.session_state.optimization_ran = True

            # Mettre à jour l0 dans l'UI si différent
            if abs(final_best_l0 - l0_nominal_gui) > 1e-3:
                log_message(f"Mise à jour GUI l0 de {l0_nominal_gui:.2f} vers {final_best_l0:.2f}")
                st.session_state.l0 = f"{final_best_l0:.4f}" # Mettre à jour l'état pour le widget

            # Calculer et afficher QWOT final
            final_qwot_str = "QWOT Erreur"
            try:
                qwots = calculate_qwot_from_ep(final_best_ep, final_best_l0, nH_material, nL_material)
                if not np.any(np.isnan(qwots)): final_qwot_str = ", ".join([f"{q:.3f}" for q in qwots])
                else: final_qwot_str = "QWOT N/A"
            except Exception as qwot_e: log_message(f"Avertissement: calcul QWOT échoué ({qwot_e})")
            st.session_state.optimized_qwot_display = final_qwot_str

            # Mettre à jour statut final
            avg_nit = f"{overall_optim_nit / successful_optim_count:.1f}" if successful_optim_count else "N/A"
            avg_nfev = f"{overall_optim_nfev / successful_optim_count:.1f}" if successful_optim_count else "N/A"
            final_status = f"Scan+Opt Terminé | Best MSE: {final_best_mse:.3e} | Couches: {len(final_best_ep)} | L0: {final_best_l0:.1f} | Avg Iter/Eval: {avg_nit}/{avg_nfev}"
            update_status(f"Prêt ({final_status})")

            # Tracer le résultat final
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

            # --- Déterminer Point de Départ ---
            ep_start_auto = None
            if st.session_state.get('optimization_ran', False) and st.session_state.get('current_optimized_ep') is not None:
                 log_message("Mode Auto: Utilisation structure optimisée existante comme départ.")
                 ep_start_auto = np.asarray(st.session_state.current_optimized_ep).copy()
                 # Calculer MSE de départ
                 try:
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
                 except Exception as e_cost: raise ValueError(f"Impossible calculer MSE départ: {e_cost}")
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
                 max_thinning_iters = len(best_ep_so_far) + 1 # Sécurité

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
                                     cost_val = cost_fn(jnp.asarray(best_ep_so_far), *cost_args) # Réutiliser cost_fn d'avant
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
            # Log stats opt
            avg_nit = f"{total_iters_auto / optim_runs_auto:.1f}" if optim_runs_auto else "N/A"
            avg_nfev = f"{total_evals_auto / optim_runs_auto:.1f}" if optim_runs_auto else "N/A"
            log_message(f"Stats Opt Mode Auto: Runs={optim_runs_auto}, Total Iter={total_iters_auto}, Total Eval={total_evals_auto}, Avg Iter={avg_nit}, Avg Eval={avg_nfev}")

            # Stocker résultat final
            st.session_state.current_optimized_ep = best_ep_so_far.copy()
            st.session_state.current_material_sequence = None
            st.session_state.optimization_ran = True

            # Calculer et afficher QWOT final
            final_qwot_str = "QWOT Erreur"
            try:
                 qwots = calculate_qwot_from_ep(best_ep_so_far, l0, nH_mat, nL_mat)
                 if not np.any(np.isnan(qwots)): final_qwot_str = ", ".join([f"{q:.3f}" for q in qwots])
                 else: final_qwot_str = "QWOT N/A"
            except Exception as qwot_e: log_message(f"Erreur calcul QWOT final mode auto: {qwot_e}")
            st.session_state.optimized_qwot_display = final_qwot_str

            # Mettre à jour statut final
            final_status = f"Mode Auto Terminé ({num_cycles_done} cyc, {termination_reason}) | MSE: {best_mse_so_far:.3e} | Couches: {len(best_ep_so_far)} | Avg Iter/Eval: {avg_nit}/{avg_nfev}"
            update_status(f"Prêt ({final_status})")

            # Tracer le résultat final (déjà fait dans le dernier cycle si amélioration ou dernier cycle)
            # Mais on peut forcer un dernier tracé ici si besoin ou si sorti avant dernier cycle
            if not cycle_improved_overall and cycle_num < AUTO_MAX_CYCLES -1:
                 log_message("Retraçage du résultat final du Mode Auto...")
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


# --- Fonctions JAX pour Scan QWOT Rapide (Split Stack) ---

# Fonction 43: calculate_M_for_thickness (JAX)
# (Déjà définie dans la réponse Tkinter - Fonction 84)
@jax.jit
def calculate_M_for_thickness(thickness: jnp.ndarray, n_complex_layer: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    """Calcule la matrice 2x2 caractéristique pour une seule couche."""
    # ... (code identique à la fonction 84 du code Tkinter) ...
    pass # Placeholder

# Fonction 44: calculate_M_batch_for_thickness (JAX vmap)
# (Déjà définie dans la réponse Tkinter - après Fonction 84)
calculate_M_batch_for_thickness = vmap(calculate_M_for_thickness, in_axes=(None, None, 0)) # Map sur l_val

# Fonction 45: get_layer_matrices_qwot (JAX)
# (Déjà définie dans la réponse Tkinter - Fonction 85)
@jax.jit
def get_layer_matrices_qwot(layer_idx: int, initial_layer_number: int, l0: float,
                            nH_c_l0: jnp.ndarray, nL_c_l0: jnp.ndarray,
                            l_vec: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calcule les matrices pour QWOT=1 et QWOT=2 pour une couche donnée."""
    # ... (code identique à la fonction 85 du code Tkinter) ...
    pass # Placeholder

# Fonction 46: compute_half_product (JAX)
# (Déjà définie dans la réponse Tkinter - Fonction 86)
@jax.jit
def compute_half_product(multiplier_indices: jnp.ndarray, # Shape (N_half,)
                         layer_matrices_half: jnp.ndarray # Shape (N_half, 2, L, 2, 2)
                         ) -> jnp.ndarray: # Retourne produit (L, 2, 2)
    """Calcule le produit matriciel pour une moitié de l'empilement."""
    # ... (code identique à la fonction 86 du code Tkinter) ...
    pass # Placeholder

# Fonction 47: get_T_from_batch_matrix (JAX)
# (Déjà définie dans la réponse Tkinter - Fonction 87)
@jax.jit
def get_T_from_batch_matrix(M_batch: jnp.ndarray, # Shape (L, 2, 2)
                            nSub_arr: jnp.ndarray # Shape (L,)
                            ) -> jnp.ndarray: # Retourne Ts (L,)
    """Calcule la Transmittance à partir d'un batch de matrices totales."""
    # ... (code identique à la fonction 87 du code Tkinter) ...
    pass # Placeholder

# Fonction 48: calculate_mse_basic_jax (JAX)
# (Déjà définie dans la réponse Tkinter - Fonction 88)
@jax.jit
def calculate_mse_basic_jax(Ts: jnp.ndarray, l_vec: jnp.ndarray, targets_tuple: Tuple) -> jnp.ndarray:
    """Calcule le MSE de base sans pénalités (pour le scan rapide)."""
    # ... (code identique à la fonction 88 du code Tkinter) ...
    pass # Placeholder

# Fonction 49: combine_and_calc_mse (JAX)
# (Déjà définie dans la réponse Tkinter - Fonction 89)
@jax.jit
def combine_and_calc_mse(prod1: jnp.ndarray, prod2: jnp.ndarray, # (L, 2, 2)
                         nSub_arr_in: jnp.ndarray,             # (L,)
                         l_vec_in: jnp.ndarray, targets_tuple_in: Tuple
                         ) -> jnp.ndarray: # Retourne MSE scalaire
    """Combine les produits matriciels des moitiés, calcule T, puis calcule MSE."""
    # ... (code identique à la fonction 89 du code Tkinter) ...
    pass # Placeholder

# Fonction 50: _execute_split_stack_scan_st (Adaptée)
def _execute_split_stack_scan_st(current_l0: float, initial_layer_number: int,
                                nH_c_l0: complex, nL_c_l0: complex,
                                nSub_arr_scan: jnp.ndarray, # Indice substrat sur grille sparse
                                l_vec_eval_sparse_jax: jnp.ndarray, # Lambda grille sparse
                                active_targets_tuple: Tuple) -> Tuple[float, Union[np.ndarray, None], List[str]]:
    """Exécute le scan QWOT rapide via split-stack (version Streamlit retournant logs)."""
    # --- !! Adaptation Nécessaire !! ---
    # Remplacer self.status_label... par des logs.append et update_status si interactif
    # S'assurer que les fonctions JAX appelées sont bien définies (get_layer_matrices_qwot, etc.)
    # Retourner les logs générés.

    logs = []
    L_sparse = len(l_vec_eval_sparse_jax)
    num_combinations = 2**initial_layer_number
    logs.append(f"  [Scan l0={current_l0:.2f}] Test {num_combinations:,} combinaisons QWOT (1.0 ou 2.0)...")

    # --- Précalcul Matrices ---
    logs.append(f"  [Scan l0={current_l0:.2f}] Précalcul matrices...")
    precompute_start_time = time.time()
    layer_matrices_list = []
    try:
        # Convertir nH/nL complexes (potentiellement NumPy) en JAX arrays pour JIT
        nH_c_l0_jax = jnp.asarray(nH_c_l0)
        nL_c_l0_jax = jnp.asarray(nL_c_l0)
        for i in range(initial_layer_number):
             m1, m2 = get_layer_matrices_qwot(i, initial_layer_number, current_l0,
                                             nH_c_l0_jax, nL_c_l0_jax,
                                             l_vec_eval_sparse_jax)
             layer_matrices_list.append(jnp.stack([m1, m2], axis=0)) # Shape (2, L_sparse, 2, 2)
        all_layer_matrices = jnp.stack(layer_matrices_list, axis=0) # Shape (N, 2, L_sparse, 2, 2)
        all_layer_matrices.block_until_ready() # Pour timing précis
    except Exception as e_mat:
        logs.append(f"  ERREUR Précalcul Matrices pour l0={current_l0:.2f}: {e_mat}")
        return np.inf, None, logs # Retourner état d'erreur

    logs.append(f"    Précalcul matrices (l0={current_l0:.2f}) terminé en {time.time() - precompute_start_time:.3f}s.")

    # --- Calcul Split Stack ---
    N1 = initial_layer_number // 2; N2 = initial_layer_number - N1
    num_comb1 = 2**N1; num_comb2 = 2**N2

    # Générer indices (identique à Tkinter version)
    indices1 = jnp.arange(num_comb1); indices2 = jnp.arange(num_comb2)
    powers1 = 2**jnp.arange(N1); powers2 = 2**jnp.arange(N2)
    multiplier_indices1 = jnp.not_equal(indices1[:, None] & powers1, 0).astype(jnp.int32)
    multiplier_indices2 = jnp.not_equal(indices2[:, None] & powers2, 0).astype(jnp.int32)

    matrices_half1 = all_layer_matrices[:N1]; matrices_half2 = all_layer_matrices[N1:]

    # --- Calcul Produits Partiels ---
    logs.append(f"  [Scan l0={current_l0:.2f}] Calcul produits partiels 1/2...")
    half1_start_time = time.time()
    compute_half_product_jit = jax.jit(compute_half_product)
    partial_products1 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices1, matrices_half1)
    partial_products1.block_until_ready()
    logs.append(f"    Produits partiels 1/2 (l0={current_l0:.2f}) calculés en {time.time() - half1_start_time:.3f}s.")

    logs.append(f"  [Scan l0={current_l0:.2f}] Calcul produits partiels 2/2...")
    half2_start_time = time.time()
    partial_products2 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices2, matrices_half2)
    partial_products2.block_until_ready()
    logs.append(f"    Produits partiels 2/2 (l0={current_l0:.2f}) calculés en {time.time() - half2_start_time:.3f}s.")

    # --- Combinaison et Calcul MSE ---
    logs.append(f"  [Scan l0={current_l0:.2f}] Combinaison et calcul MSE...")
    combine_start_time = time.time()
    combine_and_calc_mse_jit = jax.jit(combine_and_calc_mse)
    vmap_inner = vmap(combine_and_calc_mse_jit, in_axes=(None, 0, None, None, None)) # Map sur prod2
    vmap_outer = vmap(vmap_inner, in_axes=(0, None, None, None, None)) # Map sur prod1
    all_mses_nested = vmap_outer(partial_products1, partial_products2, nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple)
    all_mses_nested.block_until_ready()
    logs.append(f"    Combinaison et MSE (l0={current_l0:.2f}) terminés en {time.time() - combine_start_time:.3f}s.")

    # --- Trouver Meilleur Résultat ---
    all_mses_flat = all_mses_nested.reshape(-1)
    # Gérer le cas où tous les MSE sont infinis/NaN
    finite_mses = all_mses_flat[jnp.isfinite(all_mses_flat)]
    if finite_mses.size == 0:
         logs.append(f"    AVERTISSEMENT: Aucune combinaison valide trouvée pour l0={current_l0:.2f} (tous MSE infinis/NaN).")
         return np.inf, None, logs

    best_idx_flat = jnp.argmin(all_mses_flat) # Trouve l'indice du minimum (peut être Inf si tout est Inf)
    current_best_mse = float(all_mses_flat[best_idx_flat])

    # Obtenir les multiplicateurs seulement si le MSE est fini
    if np.isfinite(current_best_mse):
        best_idx_half1, best_idx_half2 = jnp.unravel_index(best_idx_flat, (num_comb1, num_comb2))
        best_indices_h1 = multiplier_indices1[best_idx_half1]
        best_indices_h2 = multiplier_indices2[best_idx_half2]
        best_multipliers_h1 = 1.0 + best_indices_h1.astype(jnp.float64)
        best_multipliers_h2 = 1.0 + best_indices_h2.astype(jnp.float64)
        current_best_multipliers = jnp.concatenate([best_multipliers_h1, best_multipliers_h2])
        logs.append(f"    Meilleur MSE pour scan l0={current_l0:.2f}: {current_best_mse:.6e}")
        return current_best_mse, np.array(current_best_multipliers), logs # Retourner NumPy array
    else:
         # Ce cas ne devrait pas être atteint grâce à la vérification finite_mses.size == 0
         logs.append(f"    AVERTISSEMENT: Meilleur MSE trouvé non fini pour l0={current_l0:.2f}.")
         return np.inf, None, logs


# ============================================================
# 5. CALLBACKS STREAMLIT (Suite : Actions Secondaires)
# ============================================================

# Fonction 51: handle_load_design_st (Logique pour file_uploader)
def handle_load_design_st(uploaded_file):
    """Charge un design depuis un fichier JSON uploadé et met à jour l'état."""
    if uploaded_file is None: return # Ne rien faire si aucun fichier

    filename = uploaded_file.name
    log_message(f"\n--- Chargement Design depuis: {filename} ---")
    update_status(f"Chargement {filename}...")
    try:
        design_data = json.loads(uploaded_file.getvalue().decode("utf-8"))

        if 'params' not in design_data or 'targets' not in design_data:
            raise ValueError("Fichier JSON invalide: section 'params' ou 'targets' manquante.")

        # --- Charger Paramètres ---
        loaded_params = design_data.get('params', {})
        for key, value in loaded_params.items():
            session_key = key
            # Gérer renommage clé 'nmax' -> 'initial_layer_number'
            if key == 'nmax' and 'initial_layer_number' in st.session_state:
                session_key = 'initial_layer_number'
            # Gérer types : certains doivent être string pour les widgets number_input(format="%d")
            if session_key in ['maxiter', 'maxfun', 'initial_layer_number']: value_str = str(int(value))
            elif session_key == 'emp_str': value_str = str(value) # QWOT reste string
            elif session_key in ['l0', 'l_step', 'auto_thin_threshold', 'nH_r', 'nH_i', 'nL_r', 'nL_i', 'nSub']:
                 value_str = f"{float(value):.6g}" # Format float avec précision raisonnable
            else: value_str = str(value) # Autres paramètres en string par défaut

            if session_key in st.session_state:
                 st.session_state[session_key] = value_str
                 log_message(f"  Param '{session_key}' chargé: {value_str}")
            # Ignorer les clés non reconnues pour éviter les erreurs

        # --- Charger Sélection Matériaux ---
        available_mats = st.session_state.available_materials
        available_subs = st.session_state.available_substrates
        # H
        h_mat_loaded = loaded_params.get('nH_material')
        if isinstance(h_mat_loaded, str) and h_mat_loaded in available_mats: st.session_state.selected_H_material = h_mat_loaded
        elif isinstance(h_mat_loaded, (dict, list)): # Ancien format complexe
             try:
                 nH_r = float(h_mat_loaded.get('real', 2.35)) if isinstance(h_mat_loaded, dict) else float(h_mat_loaded[0])
                 nH_i = float(h_mat_loaded.get('imag', 0.0)) if isinstance(h_mat_loaded, dict) else float(h_mat_loaded[1])
                 st.session_state.selected_H_material = "Constant"
                 st.session_state.nH_r = f"{nH_r:.6g}"
                 st.session_state.nH_i = f"{nH_i:.6g}"
             except Exception: st.session_state.selected_H_material = "Constant" # Fallback
        else: st.session_state.selected_H_material = "Constant" # Fallback
        log_message(f"  Matériau H chargé: {st.session_state.selected_H_material}")
        # L (similaire à H)
        l_mat_loaded = loaded_params.get('nL_material')
        # ... (logique similaire pour L) ...
        log_message(f"  Matériau L chargé: {st.session_state.selected_L_material}")
        # Sub (similaire, vérifier dans available_substrates)
        sub_mat_loaded = loaded_params.get('nSub_material')
        # ... (logique similaire pour Sub) ...
        log_message(f"  Substrat chargé: {st.session_state.selected_Sub_material}")


        # --- Charger Cibles ---
        loaded_targets = design_data.get('targets', [])
        current_targets_state = st.session_state.target_entries
        for i in range(len(current_targets_state)):
             if i < len(loaded_targets):
                 target_data = loaded_targets[i]
                 if isinstance(target_data, dict):
                      current_targets_state[i]['enabled'] = target_data.get('enabled', False)
                      current_targets_state[i]['min'] = str(target_data.get('min', ''))
                      current_targets_state[i]['max'] = str(target_data.get('max', ''))
                      current_targets_state[i]['target_min'] = str(target_data.get('target_min', ''))
                      current_targets_state[i]['target_max'] = str(target_data.get('target_max', ''))
                 else: # Données invalides, désactiver
                      current_targets_state[i]['enabled'] = False
             else: # Pas assez de données chargées, désactiver le reste
                 current_targets_state[i]['enabled'] = False
        log_message("Cibles chargées.")

        # --- Réinitialiser État et Charger EP Optimisé (si présent) ---
        st.session_state.current_optimized_ep = None
        st.session_state.current_material_sequence = None
        st.session_state.optimization_ran = False
        st.session_state.ep_history_deque = deque(maxlen=5)
        st.session_state.optimized_qwot_display = ""

        if 'optimized_ep' in design_data and isinstance(design_data['optimized_ep'], list) and design_data['optimized_ep']:
             try:
                 loaded_ep = np.array(design_data['optimized_ep'], dtype=np.float64)
                 if np.all(np.isfinite(loaded_ep)) and np.all(loaded_ep >= 0):
                      st.session_state.current_optimized_ep = loaded_ep
                      st.session_state.optimization_ran = True
                      log_message(f"État optimisé précédent chargé ({len(loaded_ep)} couches).")
                      # Charger ou recalculer QWOT optimisé affiché
                      if 'optimized_qwot_string' in design_data and isinstance(design_data['optimized_qwot_string'], str):
                           st.session_state.optimized_qwot_display = design_data['optimized_qwot_string']
                      else: # Recalculer
                          try:
                              inputs_recalc = validate_inputs()
                              qwots_recalc = calculate_qwot_from_ep(loaded_ep, inputs_recalc['l0'], inputs_recalc['nH_material'], inputs_recalc['nL_material'])
                              if np.any(np.isnan(qwots_recalc)): opt_qwot_str_recalc = "QWOT N/A"
                              else: opt_qwot_str_recalc = ", ".join([f"{q:.3f}" for q in qwots_recalc])
                              st.session_state.optimized_qwot_display = opt_qwot_str_recalc
                              log_message("QWOT optimisé recalculé.")
                          except Exception as e_qwot: log_message(f"Avertissement: recalcul QWOT optimisé échoué: {e_qwot}")

                 else: log_message("AVERTISSEMENT: Données 'optimized_ep' invalides (non finies ou négatives).")
             except Exception as e_ep: log_message(f"AVERTISSEMENT: Impossible charger 'optimized_ep': {e_ep}.")
        else: log_message("Pas d'état 'optimized_ep' trouvé. État réglé sur Nominal.")

        st.success(f"Design '{filename}' chargé avec succès.")
        update_status(f"Prêt (Design '{filename}' chargé)")
        # Déclencher recalcul après chargement
        st.session_state.trigger_recalc = True # Flag pour déclencher après rerun

    except json.JSONDecodeError as e:
        st.error(f"Erreur de décodage JSON dans '{filename}': {e}")
        log_message(f"ERREUR décodage JSON: {e}")
        update_status("Erreur chargement design (JSON)")
    except (ValueError, KeyError, TypeError) as e:
        st.error(f"Erreur lors du traitement du fichier design '{filename}': {e}")
        log_message(f"ERREUR traitement design: {e}")
        update_status(f"Erreur chargement design: {e}")
    except Exception as e:
        err_msg = f"ERREUR inattendue chargement design '{filename}': {type(e).__name__}: {e}"
        tb_msg = traceback.format_exc()
        log_message(err_msg); log_message(tb_msg); print(err_msg); print(tb_msg)
        st.error(f"{err_msg}\nVoir console/log.")
        update_status("Erreur inattendue chargement design")


# Fonction 52: _collect_design_data_st (Helper pour sauvegarde)
def _collect_design_data_st(include_optimized: bool = False) -> Dict:
    """Collecte l'état actuel de st.session_state pour sauvegarde."""
    design_data = {'params': {}, 'targets': []}
    errors = []

    # --- Collecter Paramètres ---
    param_keys_to_save = ['l0', 'l_step', 'maxiter', 'maxfun', 'emp_str', 'initial_layer_number', 'auto_thin_threshold']
    for key in param_keys_to_save:
        if key in st.session_state:
            try:
                val_str = st.session_state[key]
                if key in ['maxiter', 'maxfun', 'initial_layer_number']: val = int(val_str)
                elif key == 'emp_str': val = val_str
                else: val = float(val_str)
                design_data['params'][key] = val
            except (ValueError, TypeError): errors.append(f"Paramètre '{key}' invalide ('{val_str}')")

    # Matériaux (sauvegarder noms)
    design_data['params']['nH_material'] = st.session_state.selected_H_material
    design_data['params']['nL_material'] = st.session_state.selected_L_material
    design_data['params']['nSub_material'] = st.session_state.selected_Sub_material
    # Sauver valeurs constantes (même si non utilisées si matériau non constant est sélectionné)
    const_keys = ['nH_r', 'nH_i', 'nL_r', 'nL_i', 'nSub']
    for key in const_keys:
         if key in st.session_state: design_data['params'][key] = st.session_state[key]

    # --- Collecter Cibles ---
    targets_state = st.session_state.get('target_entries', [])
    for i, target_data_state in enumerate(targets_state):
        try:
            # Tenter de convertir en float pour validation (mais sauver comme str)
            float(target_data_state.get('min', 'nan'))
            float(target_data_state.get('max', 'nan'))
            float(target_data_state.get('target_min', 'nan'))
            float(target_data_state.get('target_max', 'nan'))
            design_data['targets'].append({
                 'enabled': target_data_state.get('enabled', False),
                 'min': target_data_state.get('min', ''),
                 'max': target_data_state.get('max', ''),
                 'target_min': target_data_state.get('target_min', ''),
                 'target_max': target_data_state.get('target_max', '')
             })
        except (ValueError, TypeError): errors.append(f"Cible {i+1} contient des valeurs non numériques.")

    # --- Collecter État Optimisé (Optionnel) ---
    if include_optimized and st.session_state.get('optimization_ran') and st.session_state.get('current_optimized_ep') is not None:
        design_data['optimized_ep'] = st.session_state.current_optimized_ep.tolist()
        design_data['optimized_qwot_string'] = st.session_state.get('optimized_qwot_display', "")

    if errors:
        # Lever une exception si des erreurs de collecte se produisent
        raise ValueError("Erreurs lors de la collecte des données pour sauvegarde:\n- " + "\n- ".join(errors))

    return design_data

# Fonction 53: on_copy_nominal_qwot_click (Callback Bouton Copier QWOT Nominal)
def on_copy_nominal_qwot_click():
    """Copie le QWOT nominal dans le presse-papier."""
    # Streamlit n'a pas d'accès direct au presse-papier pour des raisons de sécurité navigateur.
    # La meilleure approche est d'afficher la chaîne dans un st.code ou st.text_area
    # pour que l'utilisateur puisse la copier manuellement.
    qwot_str = st.session_state.get('emp_str', '')
    if qwot_str:
        st.info("QWOT Nominal (copiez manuellement):")
        st.code(qwot_str, language=None) # Pas de langage spécifique
        log_message("QWOT Nominal affiché pour copie.")
    else:
        st.warning("Pas de QWOT nominal à copier.")

# Fonction 54: on_copy_optimized_qwot_click (Callback Bouton Copier QWOT Optimisé)
def on_copy_optimized_qwot_click():
    """Affiche le QWOT optimisé pour copie manuelle."""
    qwot_str = st.session_state.get('optimized_qwot_display', '')
    is_valid = qwot_str and "Error" not in qwot_str and "N/A" not in qwot_str
    if st.session_state.get('optimization_ran') and is_valid:
        st.info("QWOT Optimisé (copiez manuellement):")
        st.code(qwot_str, language=None)
        log_message("QWOT Optimisé affiché pour copie.")
    else:
        st.warning("Pas de QWOT optimisé valide à copier.")

# Fonction 55: on_refresh_plots_click (Callback Bouton Rafraîchir Graphes)
def on_refresh_plots_click():
    """Force le recalcul et le rafraîchissement des graphiques."""
    log_message("Rafraîchissement des graphiques demandé...")
    update_status("Rafraîchissement des graphiques...")
    st.session_state.figure_main = None # Vider cache figure principale
    st.session_state.figure_indices = None # Vider cache figure indices
    st.session_state.figure_preview = None # Vider cache figure preview

    # Déterminer quel état retracer
    is_opt = st.session_state.get('optimization_ran', False)
    ep_to_plot = st.session_state.get('current_optimized_ep') if is_opt else None
    method = st.session_state.get('last_calc_method_name', "Nominal" if not is_opt else "Optimisé") + " (Rafraîchi)"

    with st.spinner("Recalcul et rafraîchissement des graphiques..."):
        try:
            run_calculation_st(ep_vector_to_use=ep_to_plot, is_optimized=is_opt, method_name=method)
            update_status("Prêt (Graphiques rafraîchis)")
            st.toast("Graphiques rafraîchis.", icon="🔄")
        except (ValueError, RuntimeError, TypeError) as e:
             err_msg = f"ERREUR (Rafraîchissement): {e}"
             log_message(err_msg); st.error(err_msg)
             update_status(f"Erreur Rafraîchissement: {e}")
        except Exception as e:
             err_msg = f"ERREUR Inattendue (Rafraîchissement): {type(e).__name__}: {e}"
             tb_msg = traceback.format_exc()
             log_message(err_msg); log_message(tb_msg); print(err_msg); print(tb_msg)
             st.error(f"{err_msg}\nVoir console/log.")
             update_status("Erreur Inattendue (Rafraîchissement)")


# Fonction 56: on_about_click (Callback Bouton 'About')
def on_about_click():
    """Affiche les informations 'About' dans un dialogue modal."""
    # Utiliser st.dialog (expérimental mais adapté ici) ou st.expander
    about_text = """
    **Optimiseur Film Mince (JAX Grad - Dispersif) - Version Streamlit**
    ---
    * **Version:** [Indiquer version, ex: 4.0 - Mai 2025]
    * **Auteur Original (Tkinter):** Fabien Lemarchand
    * **Conversion Streamlit:** Gemini
    ---
    **Fonctionnalités Principales:**
    * Calcul de spectre T(λ) via Méthode Matrice Transfert (TMM).
    * Gestion matériaux dispersifs (Excel, Lois Sellmeier, Constantes).
    * Optimisation par gradient (JAX + Scipy L-BFGS-B) sur cibles spectrales.
    * Algorithmes: Opt. Locale, Scan QWOT Initial (Split Stack), Insertion Aiguille, Suppression Couches Fines, Mode Auto.
    * Visualisation interactive (Spectre, Profil d'indice, Empilement).
    * Chargement/Sauvegarde de designs (JSON).
    * Historique d'annulation (Undo) pour suppression de couches.
    * Interface Web via Streamlit.

    **Technologies:** Python, Streamlit, JAX, Scipy, Matplotlib, Pandas, Numpy.

    **Contact (Code Original):** fabien.lemarchand@gmail.com
    """
    # Utiliser st.markdown dans le dialogue pour le formatage
    with st.dialog("À Propos", width="large"):
         st.markdown(about_text)
         if st.button("Fermer", key="close_about_dialog"):
             st.rerun() # Ferme le dialogue en relançant le script


# --- Fonctions JAX spécifiques au Scan QWOT Rapide (Split Stack) ---

# Fonction 57: calculate_M_for_thickness (JAX)
# (Déjà définie - Fonction 43 logique)
# Placeholder
@jax.jit
def calculate_M_for_thickness(*args, **kwargs): pass

# Fonction 58: calculate_M_batch_for_thickness (JAX vmap)
# (Déjà définie - Fonction 44 logique)
# Placeholder
calculate_M_batch_for_thickness = None # vmap(...)

# Fonction 59: get_layer_matrices_qwot (JAX)
# (Déjà définie - Fonction 45 logique)
# Placeholder
@jax.jit
def get_layer_matrices_qwot(*args, **kwargs): pass

# Fonction 60: compute_half_product (JAX)
# (Déjà définie - Fonction 46 logique)
# Placeholder
@jax.jit
def compute_half_product(*args, **kwargs): pass

# --- FIN DES 20 FONCTIONS SUIVANTES ---

# [PREVIOUS CODE HERE - Imports, Config, CSS, Constants, Core Calcs, State Init, Sidebar UI, Callbacks etc.]
# ... (Assume lines 1-approx 1420 are present, including definitions for functions #41-60)

# ============================================================
# 5. CALLBACKS STREAMLIT (Suite : Actions Secondaires & Helpers)
# ============================================================

# Fonction 61: handle_load_design_st (Déjà définie - Fonction #51 logique)
# Placeholder
def handle_load_design_st(uploaded_file): pass

# Fonction 62: _collect_design_data_st (Helper pour sauvegarde)
# (Déjà définie - Fonction #52 logique)
# Placeholder
def _collect_design_data_st(include_optimized: bool = False) -> Dict: pass

# Fonction 63: on_copy_nominal_qwot_click (Callback Bouton Copier QWOT Nominal)
# (Déjà définie - Fonction #53 logique)
# Placeholder
def on_copy_nominal_qwot_click(): pass

# Fonction 64: on_copy_optimized_qwot_click (Callback Bouton Copier QWOT Optimisé)
# (Déjà définie - Fonction #54 logique)
# Placeholder
def on_copy_optimized_qwot_click(): pass

# Fonction 65: on_refresh_plots_click (Callback Bouton Rafraîchir Graphes)
# (Déjà définie - Fonction #55 logique)
# Placeholder
def on_refresh_plots_click(): pass

# Fonction 66: on_about_click (Callback Bouton 'About')
# (Déjà définie - Fonction #56 logique)
# Placeholder
def on_about_click(): pass


# --- Fonctions JAX spécifiques au Scan QWOT Rapide (Split Stack) ---
# (Ces fonctions étaient numérotées 84-89 dans le code Tkinter original)
# (Elles ont été définies comme fonctions #57-60 logiques dans la réponse précédente)

# Fonction 67: get_T_from_batch_matrix (JAX)
# (Déjà définie - Fonction #47 logique)
# Placeholder
@jax.jit
def get_T_from_batch_matrix(*args, **kwargs): pass

# Fonction 68: calculate_mse_basic_jax (JAX)
# (Déjà définie - Fonction #48 logique)
# Placeholder
@jax.jit
def calculate_mse_basic_jax(*args, **kwargs): pass

# Fonction 69: combine_and_calc_mse (JAX)
# (Déjà définie - Fonction #49 logique)
# Placeholder
@jax.jit
def combine_and_calc_mse(*args, **kwargs): pass

# Fonction 70 (placeholder - correspond approx à l'ancienne fct #70 'merge_adjacent_layers', non utilisée)
def placeholder_function_70(): pass

# Fonction 71 (placeholder - correspond approx à l'ancienne fct #71 'calculate_ep_for_sequence', non utilisée)
def placeholder_function_71(): pass

# ============================================================
# 6. INITIALISATION (Appel unique au début du script)
# ============================================================
if 'app_initialized' not in st.session_state:
     print("--- Initialisation de l'état de session ---")
     initialize_session_state()
     st.session_state.app_initialized = True
     # Optionnel: Déclencher un calcul initial au premier chargement ?
     # st.session_state.trigger_recalc = True # Pourrait lancer on_evaluate_button_click

# ============================================================
# 7. INTERFACE UTILISATEUR STREAMLIT (Sidebar - Déjà définie)
# ============================================================
# ============================================================
# 7. INTERFACE UTILISATEUR STREAMLIT (Sidebar)
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- Section Fichiers & Matériaux ---
    with st.expander("📁 Fichiers & Matériaux", expanded=True):
        # Bouton pour recharger les matériaux depuis Excel
        if st.button("🔄 Recharger Matériaux (Excel)", key="reload_mat_btn", help=f"Relit les matériaux depuis '{EXCEL_FILE_PATH}'"):
            # @st.cache_data.clear() # Optionnel: vider tout le cache de données
            # Vider spécifiquement le cache de la fonction de chargement
            load_material_data_from_xlsx_sheet.cache_clear()
            update_available_materials() # Met à jour les listes dans session_state
            st.toast(f"Listes de matériaux rechargées depuis {EXCEL_FILE_PATH} !", icon="✅")
            # Pas besoin de st.rerun() ici car la mise à jour de session_state le déclenchera implicitement

        # Widget pour charger un fichier de design
        uploaded_file = st.file_uploader("Charger Design (.json)", type="json", key="load_design_uploader")
        if uploaded_file is not None:
            # Traiter le fichier chargé (appelle la logique définie précédemment)
            handle_load_design_st(uploaded_file)
            # Nettoyer l'uploader après traitement pour éviter re-traitement au prochain rerun
            st.session_state.load_design_uploader = None # Ne fonctionne pas directement, le widget se réinitialise
            # Un st.rerun() est souvent nécessaire après traitement pour rafraîchir l'UI
            st.rerun()


        # Préparation des données pour les boutons de sauvegarde
        save_data_nominal_dict = {}
        save_data_optimized_dict = None
        save_error = None
        try:
            save_data_nominal_dict = _collect_design_data_st(include_optimized=False)
            if st.session_state.get('optimization_ran') and st.session_state.get('current_optimized_ep') is not None:
                save_data_optimized_dict = _collect_design_data_st(include_optimized=True)
        except Exception as e_collect:
            save_error = f"Erreur collecte données pour sauvegarde: {e_collect}"
            log_message(save_error)

        # Afficher l'erreur de collecte si elle existe
        if save_error:
            st.error(save_error)

        # Convertir en JSON string pour le download button (seulement si pas d'erreur)
        save_json_nominal = ""
        save_json_optimized = ""
        if not save_error:
            try:
                save_json_nominal = json.dumps(save_data_nominal_dict, indent=4)
                if save_data_optimized_dict:
                    save_json_optimized = json.dumps(save_data_optimized_dict, indent=4)
            except Exception as e_json:
                 save_error = f"Erreur conversion JSON pour sauvegarde: {e_json}"
                 log_message(save_error)
                 st.error(save_error) # Afficher aussi l'erreur JSON

        # Boutons de sauvegarde (désactivés si erreur de collecte/JSON)
        col_save1, col_save2 = st.columns(2)
        with col_save1:
            st.download_button(
                label="💾 Sauver Nominal",
                data=save_json_nominal if not save_error else "",
                file_name=f"design_nominal_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                key="save_nom_btn",
                disabled=bool(save_error)
            )
        with col_save2:
            st.download_button(
                label="💾 Sauver Optimisé",
                data=save_json_optimized if not save_error and save_data_optimized_dict else "",
                file_name=f"design_optimise_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                key="save_opt_btn",
                disabled=bool(save_error) or (save_data_optimized_dict is None),
                help="Sauvegarde l'état optimisé actuel (épaisseurs incluses)."
            )

    # --- Section Matériaux ---
    with st.expander("🔬 Matériaux et Substrat", expanded=True):
        # Utiliser les listes stockées dans session_state (mises à jour par update_available_materials)
        mats = st.session_state.get('available_materials', ["Constant"])
        subs = st.session_state.get('available_substrates', ["Constant", "Fused Silica", "BK7", "D263"])

        # Matériau H
        # Trouver l'index actuel pour le selectbox, ou 0 si invalide
        try: idx_H = mats.index(st.session_state.selected_H_material)
        except ValueError: idx_H = 0
        st.selectbox("Matériau H", mats, index=idx_H, key="selected_H_material", on_change=on_material_change)
        h_is_const = st.session_state.selected_H_material == "Constant"
        colH1, colH2 = st.columns(2)
        # Utiliser number_input pour les constantes n/k pour une meilleure validation
        with colH1:
            st.number_input("n' H",
                    value=st.session_state.get('nH_r', 2.35), # <-- Utiliser .get avec défaut float
                    min_value=0.0, step=0.01, format="%.4f", key="nH_r",
                    disabled=not h_is_const, help="Partie réelle si Matériau H = Constant")
        with colH2:
            st.number_input("k H",
                    value=st.session_state.get('nH_i', 0.0), # <-- Utiliser .get avec défaut float
                    min_value=0.0, step=1e-4, format="%.4f", key="nH_i",
                    disabled=not h_is_const, help="Partie imaginaire (>=0) si Matériau H = Constant")

        # Matériau L
        try: idx_L = mats.index(st.session_state.selected_L_material)
        except ValueError: idx_L = 0
        st.selectbox("Matériau L", mats, index=idx_L, key="selected_L_material", on_change=on_material_change)
        l_is_const = st.session_state.selected_L_material == "Constant"
        colL1, colL2 = st.columns(2)
        with colL1:
            st.number_input("n' L",
                    value=st.session_state.get('nL_r', 1.45), # <-- Utiliser .get avec défaut float
                    min_value=0.0, step=0.01, format="%.4f", key="nHL_r",
                    disabled=not l_is_const, help="Partie réelle si Matériau L = Constant")
        with colL2:
            st.number_input("k L",
                    value=st.session_state.get('nL_i', 0.0), # <-- Utiliser .get avec défaut float
                    min_value=0.0, step=1e-4, format="%.4f", key="nL_i",
                    disabled=not l_is_const, help="Partie imaginaire (>=0) si Matériau L = Constant")
        # Substrat
        try: idx_S = subs.index(st.session_state.selected_Sub_material)
        except ValueError: idx_S = 0
        st.selectbox("Substrat", subs, index=idx_S, key="selected_Sub_material", on_change=on_material_change)
        sub_is_const = st.session_state.selected_Sub_material == "Constant"
        colS1, colS2 = st.columns([3,1]) # Donner plus de place à l'entrée n'
        with colS1:
             st.number_input("n' Substrat", value=float(st.session_state.nSub), min_value=0.0, step=0.01, format="%.4f", key="nSub", disabled=not sub_is_const, help="Partie réelle si Substrat = Constant (k=0 assumé)")
        with colS2:
             # Note sur la convention n+ik
             st.markdown("<p style='font-size:0.75rem; margin-top: 25px; color: gray;'>(n = n'+ik)</p>", unsafe_allow_html=True)

        # Bouton pour afficher/cacher le graphe des indices
        label_btn_idx = "Masquer Indices n'(λ)" if st.session_state.get('show_indices_plot') else "👁️ Voir Indices n'(λ)"
        if st.button(label_btn_idx, key="toggle_indices_btn"):
             st.session_state.show_indices_plot = not st.session_state.get('show_indices_plot', False)
             st.session_state.figure_indices = None # Forcer regénération si on ré-affiche
             st.rerun() # Mettre à jour l'affichage de l'expander

    # --- Affichage optionnel du graphe des indices ---
    if st.session_state.get('show_indices_plot', False):
         with st.expander("Indices n'(λ) des Matériaux Sélectionnés", expanded=True):
            if st.session_state.get('figure_indices') is None:
                log_message("Génération du graphique des indices...")
                try:
                    st.session_state.figure_indices = draw_material_index_plot_st()
                except Exception as e_idx_plot:
                    st.error(f"Erreur lors de la génération du graphique des indices: {e_idx_plot}")
                    log_message(f"Erreur plot indices: {e_idx_plot}\n{traceback.format_exc(limit=2)}")
                    st.session_state.figure_indices = None

            if st.session_state.figure_indices:
                st.pyplot(st.session_state.figure_indices, clear_figure=False) # Garder la référence
            else:
                st.warning("Impossible d'afficher le graphique des indices.")


    # --- Section Définition Empilement ---
    with st.expander("🧱 Définition Empilement", expanded=True):
         # Champ texte pour le QWOT nominal
         st.text_area("QWOT Nominal (ex: 1,1,0.8,1.2,...)", key="emp_str", on_change=on_qwot_change, height=100)

         # Champ pour le nombre initial de couches (synchronisé avec QWOT)
         col_init1, col_init2 = st.columns([2,3])
         with col_init1:
             # Utiliser number_input pour forcer un entier

             st.number_input("Nb Couches Initial", min_value=0, step=1,
                value=st.session_state.get('initial_layer_number', 1), # <-- Utiliser .get avec défaut int
                format="%d", key="initial_layer_number", on_change=on_initial_layer_change,
                help="Utilisé par '1. Start Nom.'. Met aussi à jour le QWOT nominal avec des '1'.")

             
         with col_init2:
             # Affichage dynamique du nombre de couches actuel
             current_ep = st.session_state.get('current_optimized_ep')
             is_opt = st.session_state.get('optimization_ran', False)
             num_layers_disp = 0
             label_type = "Nominal"
             if is_opt and current_ep is not None:
                 num_layers_disp = len(current_ep)
                 label_type = "Optimisé"
             else:
                 try:
                     num_layers_disp = len([item for item in st.session_state.emp_str.split(',') if item.strip()])
                 except Exception: num_layers_disp = 0
             label_disp = f"Couches ({label_type}): **{num_layers_disp}**"
             st.markdown(f"<div style='margin-top: 28px; font-size:0.85rem;'>{label_disp}</div>", unsafe_allow_html=True)

         # Affichage (readonly) du QWOT optimisé
         st.text_input("QWOT Optimisé (readonly)", key="optimized_qwot_display", disabled=True, help="QWOT calculé à partir de la dernière structure optimisée.")

         # Champ pour Lambda de centrage QWOT
         st.number_input("λ Centrage QWOT (nm)", min_value=0.1, step=1.0, key="l0", format="%.2f", help="Longueur d'onde de référence pour le calcul QWOT -> épaisseur.")

    # --- Section Paramètres Calcul & Optimisation ---
    with st.expander("🛠️ Paramètres Calcul & Optimisation", expanded=False):
         colP1, colP2 = st.columns(2)
         with colP1:
             # Utiliser number_input pour les entiers
             st.number_input("Max Iter (Opt)", min_value=1, step=10, key="maxiter", format="%d")
         with colP2:
             st.number_input("Max Eval (Opt)", min_value=1, step=10, key="maxfun", format="%d")
         st.number_input("λ Step (nm) (Optim/Plot)", min_value=0.01, step=0.1, key="l_step", format="%.2f", help="Pas pour la grille d'optimisation (géométrique) et résolution de base pour les tracés.")
         st.number_input("Seuil Élimination Auto (nm)", min_value=0.0, step=0.1, key="auto_thin_threshold", format="%.3f", help="Épaisseur max pour élimination auto en mode 'Auto'.")

# --- FIN DU BLOC SIDEBAR ---

# ============================================================
# 8. INTERFACE UTILISATEUR STREAMLIT (Zone Principale - Complétée)
# ============================================================

st.header("Optimiseur de Films Minces (JAX Grad - Dispersif)")
st.caption(f"Heure serveur: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Définition des Cibles Spectrales (Déjà définie) ---
st.subheader("🎯 Cibles Spectrales (Transmittance T)")
# ... (Code pour afficher les cibles via st.columns, st.checkbox, st.number_input) ...
# ... (Affichage des points estimés) ...

# --- Prévisualisation des Cibles (Expander) ---
with st.expander("📉 Prévisualisation des Cibles Actives", expanded=False):
    if st.session_state.get('figure_preview') is None:
        try:
            st.session_state.figure_preview = draw_target_preview_st()
        except Exception as e_prev_plot:
            st.error(f"Erreur génération prévisualisation: {e_prev_plot}")
            log_message(f"Erreur plot preview: {e_prev_plot}")
            st.session_state.figure_preview = None

    if st.session_state.figure_preview:
        st.pyplot(st.session_state.figure_preview, clear_figure=False)
    else:
        st.info("Définissez des cibles actives pour voir la prévisualisation.")


# --- Actions Principales (Complétées) ---
st.subheader("⚙️ Actions")
action_col_1, action_col_2, action_col_3, action_col_4, action_col_5 = st.columns(5)

# Colonne 1
with action_col_1:
    if st.button("📊 Evaluer", key="eval_btn_main", help="Calcule et affiche le spectre pour le QWOT nominal.", use_container_width=True):
         on_evaluate_button_click()
    if st.button("📈 Opt. Locale", key="local_opt_btn_main", help="Optimise la structure actuelle (nominale ou optimisée).", use_container_width=True):
        on_local_opt_button_click()

# Colonne 2
with action_col_2:
    if st.button("🚀 1. Start Nom.", key="start_nom_btn_main", help="Scan QWOT + Test l0 + Opt Locale (peut être long).", use_container_width=True):
        on_start_scan_click()
    if st.button("🤖 2. Auto Mode", key="auto_mode_btn_main", help="Mode Auto: Aiguille > Suppr. Fine > Opt (cycles).", use_container_width=True):
        on_auto_mode_click()

# Colonne 3
with action_col_3:
    # Le bouton 'Suppr. Couche Fine'
    can_remove = st.session_state.get('optimization_ran', False) and \
                 st.session_state.get('current_optimized_ep') is not None and \
                 len(st.session_state.current_optimized_ep) > 2
    if st.button("🔪 Suppr. Fine", key="remove_thin_btn_main", disabled=not can_remove, use_container_width=True, help="Supprime la couche la plus fine et ré-optimise."):
         on_remove_thin_click()

    # Le bouton 'Annuler Sup.'
    can_undo = bool(st.session_state.get('ep_history_deque', deque()))
    if st.button("↩️ Annuler Sup.", key="undo_btn_main", disabled=not can_undo, use_container_width=True, help="Annule la dernière suppression de couche."):
         on_undo_click()

# Colonne 4
with action_col_4:
    # Le bouton 'Opt -> Nominal'
    can_set_nominal = st.session_state.get('optimization_ran', False) and \
                      st.session_state.get('current_optimized_ep') is not None
    if st.button("➡️ Opt➜Nominal", key="set_nominal_btn_main", disabled=not can_set_nominal, use_container_width=True, help="Copie QWOT optimisé vers nominal."):
        on_set_nominal_click()

    # Le bouton 'Effacer Opt.'
    can_clear_opt = st.session_state.get('optimization_ran', False)
    if st.button("🗑️ Effacer Opt.", key="clear_opt_btn_main2", disabled=not can_clear_opt, use_container_width=True, help="Efface l'état optimisé actuel."):
        on_clear_opt_click()

# Colonne 5
with action_col_5:
    if st.button("📋 Copier Nom. QWOT", key="copy_nom_btn", help="Affiche le QWOT Nominal pour copie manuelle.", use_container_width=True):
        on_copy_nominal_qwot_click()
    can_copy_opt = st.session_state.get('optimization_ran', False) and st.session_state.get('optimized_qwot_display','') not in ["", "QWOT Erreur", "QWOT N/A"]
    if st.button("📋 Copier Opt. QWOT", key="copy_opt_btn", disabled=not can_copy_opt, help="Affiche le QWOT Optimisé pour copie manuelle.", use_container_width=True):
        on_copy_optimized_qwot_click()
    # Rafraîchir et About peuvent aller ici ou ailleurs (ex: sidebar)
    if st.button("🔄 Rafraîchir Graphes", key="refresh_plots_btn", help="Recalcule et affiche les graphiques pour l'état actuel.", use_container_width=True):
        on_refresh_plots_click()
    if st.button("ℹ️ À Propos", key="about_btn", help="Affiche les informations sur l'application.", use_container_width=True):
        on_about_click()


# --- Affichage Épaisseur Minimale (Déjà défini) ---
# ... (Code pour afficher le caption de l'épaisseur minimale) ...

st.divider()

# ============================================================
# 9. AFFICHAGE DES RÉSULTATS (Graphiques - Complété)
# ============================================================
st.subheader("📈 Résultats")

# Affichage conditionnel du graphique principal
results_plot_data = st.session_state.get('last_calc_results_plot')
is_opt_plot = st.session_state.get('last_calc_is_optimized', False)
ep_to_plot = st.session_state.get('current_optimized_ep') if is_opt_plot else None

# Recalculer ep nominal si besoin pour le plot
if not is_opt_plot:
    try:
        inputs_plot = validate_inputs()
        emp_list_plot = [float(e.strip()) for e in inputs_plot['emp_str'].split(',') if e.strip()]
        ep_to_plot = calculate_initial_ep(emp_list_plot, inputs_plot['l0'], inputs_plot['nH_material'], inputs_plot['nL_material'])
    except Exception as e_nom_plot:
        ep_to_plot = np.array([])
        log_message(f"Erreur calcul ep nominal pour plot: {e_nom_plot}")

plot_placeholder = st.empty() # Conteneur pour le graphique

if results_plot_data and ep_to_plot is not None:
    if st.session_state.get('figure_main') is None: # Générer seulement si pas déjà généré
        log_message("Génération du graphique principal...")
        try:
            inputs_plot = validate_inputs()
            active_targets_plot_final = get_active_targets_from_state(ignore_errors=True) or []
            st.session_state.figure_main = draw_plots_st(
                res=results_plot_data, current_ep=ep_to_plot,
                nH_material=inputs_plot['nH_material'], nL_material=inputs_plot['nL_material'], nSub_material=inputs_plot['nSub_material'],
                active_targets_for_plot=active_targets_plot_final,
                mse=st.session_state.get('last_calc_mse'),
                is_optimized=is_opt_plot,
                method_name=st.session_state.get('last_calc_method_name', ""),
                res_optim_grid=st.session_state.get('last_calc_results_optim_grid'),
                material_sequence=st.session_state.get('last_calc_material_sequence')
            )
        except Exception as e_plot:
            st.error(f"Erreur lors de la génération du graphique principal: {e_plot}")
            log_message(f"Erreur plot principal: {e_plot}\n{traceback.format_exc(limit=2)}")
            st.session_state.figure_main = None

    # Afficher la figure stockée dans le placeholder
    if st.session_state.figure_main:
        plot_placeholder.pyplot(st.session_state.figure_main, clear_figure=False)
        # Note : On ne ferme PAS la figure ici (clear_figure=False) pour la garder en cache session_state.
        # Elle sera regénérée si st.session_state.figure_main est remis à None (ex: après une nouvelle action).
    else:
        plot_placeholder.warning("Impossible d'afficher le graphique principal des résultats.")

else:
    plot_placeholder.info("Lancez un calcul ('Evaluer' ou une optimisation) pour afficher les résultats.")

# ============================================================
# 10. LOGS & STATUT (Complété)
# ============================================================
st.divider()
log_expander = st.expander("📜 Log Détaillé", expanded=False)
with log_expander:
    log_content = "\n".join(st.session_state.get('log_messages', ["Log vide."]))
    # Utiliser max_chars pour limiter la taille affichée si le log devient énorme
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
    st.session_state.trigger_recalc = False # Consommer le flag
    log_message("Déclenchement recalcul post-chargement...")
    if st.session_state.get('optimization_ran', False):
        on_refresh_plots_click() # Relance le calcul optimisé
    else:
        on_evaluate_button_click() # Relance le calcul nominal
    st.rerun() # Assurer que le recalcul est bien pris en compte et l'UI rafraîchie

# --- FIN DU SCRIPT STREAMLIT ---
