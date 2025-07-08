import streamlit as st
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.lax import scan, cond
import numpy as np
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import pandas as pd
import functools
from typing import Union, Tuple, Dict, List, Any, Callable, Optional
from scipy.optimize import minimize, OptimizeResult
import time
import datetime
import traceback
from collections import deque

# --- Constants ---
MIN_THICKNESS_PHYS_NM = 0.01
BASE_NEEDLE_THICKNESS_NM = 0.1
DEFAULT_NEEDLE_SCAN_STEP_NM = 2.0
AUTO_NEEDLES_PER_CYCLE = 3
AUTO_MAX_CYCLES = 3
COST_IMPROVEMENT_TOLERANCE = 1e-9
EXCEL_FILE_PATH = "indices.xlsx"
MAXITER_HARDCODED = 1000
MAXFUN_HARDCODED = 1000

# --- Logging ---
def add_log(message: Union[str, List[str]]):
    """Placeholder for a more robust logging system if needed."""
    pass

# --- Data Loading ---
@st.cache_data
def load_material_data_from_xlsx_sheet(file_path: str, sheet_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Loads and cleans material data (lambda, n, k) from a specific sheet of an Excel file."""
    logs = []
    try:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        except FileNotFoundError:
            st.error(f"Fichier Excel '{file_path}' non trouvé. Veuillez vérifier sa présence.")
            logs.append(f"Critical error: Excel file not found: {file_path}")
            return None, None, None, logs
        except Exception as e:
            st.error(f"Erreur de lecture du fichier Excel ('{file_path}', feuille '{sheet_name}'): {e}")
            logs.append(f"Unexpected Excel error ({type(e).__name__}): {e}")
            return None, None, None, logs

        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        numeric_df = numeric_df.dropna(how='all')

        if numeric_df.shape[1] >= 3:
            cols_to_check = numeric_df.columns[:3]
            numeric_df = numeric_df.dropna(subset=cols_to_check)
        else:
            logs.append(f"Avertissement: La feuille '{sheet_name}' ne contient pas 3 colonnes numériques.")
            return np.array([]), np.array([]), np.array([]), logs

        if numeric_df.empty:
            logs.append(f"Avertissement: Aucune donnée numérique valide trouvée dans '{sheet_name}' après nettoyage.")
            return np.array([]), np.array([]), np.array([]), logs

        try:
            numeric_df = numeric_df.sort_values(by=numeric_df.columns[0])
        except IndexError:
            logs.append(f"Erreur: Impossible de trier les données pour la feuille '{sheet_name}'. Colonne d'index 0 manquante ?")
            return np.array([]), np.array([]), np.array([]), logs

        l_nm = numeric_df.iloc[:, 0].values.astype(np.float64)
        n = numeric_df.iloc[:, 1].values.astype(np.float64)
        k = numeric_df.iloc[:, 2].values.astype(np.float64)

        if np.any(k < -1e-9):
            invalid_k_indices = np.where(k < -1e-9)[0]
            logs.append(f"AVERTISSEMENT: Valeurs k négatives détectées et mises à 0 pour '{sheet_name}' aux indices : {invalid_k_indices.tolist()}")
            k = np.maximum(k, 0.0)

        if len(l_nm) == 0:
            logs.append(f"Avertissement: Aucune ligne de données valide après conversion dans '{sheet_name}'.")
            return np.array([]), np.array([]), np.array([]), logs

        logs.append(f"Données chargées '{sheet_name}': {len(l_nm)} pts [{l_nm.min():.1f}-{l_nm.max():.1f} nm]")
        return l_nm, n, k, logs

    except ValueError as ve:
        logs.append(f"Erreur de valeur Excel ('{sheet_name}'): {ve}")
        return None, None, None, logs
    except Exception as e:
        logs.append(f"Erreur inattendue lors de la lecture d'Excel ('{sheet_name}'): {type(e).__name__} - {e}")
        return None, None, None, logs

def get_available_materials_from_excel(excel_path: str) -> Tuple[List[str], List[str]]:
    """Reads an Excel file to get a list of available material sheets."""
    logs = []
    try:
        xl = pd.ExcelFile(excel_path)
        sheet_names = [name for name in xl.sheet_names if not name.startswith("Sheet")]
        logs.append(f"Matériaux trouvés dans {excel_path}: {sheet_names}")
        return sheet_names, logs
    except FileNotFoundError:
        st.error(f"Fichier Excel '{excel_path}' non trouvé pour lister les matériaux.")
        logs.append(f"Erreur critique FNF: Fichier Excel {excel_path} non trouvé pour lister les matériaux.")
        return [], logs
    except Exception as e:
        st.error(f"Erreur de lecture des noms de feuilles de '{excel_path}': {e}")
        logs.append(f"Erreur de lecture des noms de feuilles de {excel_path}: {type(e).__name__} - {e}")
        return [], logs

# --- Material Definitions ---
@jax.jit
def get_n_fused_silica(l_nm: jnp.ndarray) -> jnp.ndarray:
    """Sellmeier equation for Fused Silica."""
    l_um = l_nm / 1000.0
    l_um_squared = l_um**2
    B1 = 0.6961663
    B2 = 0.4079426
    B3 = 0.8974794
    C1 = 0.0684043**2
    C2 = 0.1162414**2
    C3 = 9.896161**2
    n_squared = 1.0 + (B1 * l_um_squared) / (l_um_squared - C1) \
                    + (B2 * l_um_squared) / (l_um_squared - C2) \
                    + (B3 * l_um_squared) / (l_um_squared - C3)
    n = jnp.sqrt(n_squared)
    k_val = jnp.zeros_like(n)
    return n + 1j * k_val

@jax.jit
def get_n_bk7(l_nm: jnp.ndarray) -> jnp.ndarray:
    """Sellmeier equation for BK7 glass."""
    l_um = l_nm / 1000.0
    l_um_squared = l_um**2
    B1 = 1.03961212
    B2 = 0.231792344
    B3 = 1.01046945
    C1 = 0.00600069867
    C2 = 0.0200179144
    C3 = 103.560653
    n_squared = 1.0 + (B1 * l_um_squared) / (l_um_squared - C1) \
                    + (B2 * l_um_squared) / (l_um_squared - C2) \
                    + (B3 * l_um_squared) / (l_um_squared - C3)
    n = jnp.sqrt(n_squared)
    k_val = jnp.zeros_like(n)
    return n + 1j * k_val

@jax.jit
def get_n_d263(l_nm: jnp.ndarray) -> jnp.ndarray:
    """Constant index for D263 glass."""
    n = jnp.full_like(l_nm, 1.523, dtype=jnp.float64)
    k_val = jnp.zeros_like(n)
    return n + 1j * k_val

@jax.jit
def interp_nk_cached(l_target: jnp.ndarray, l_data: jnp.ndarray, n_data: jnp.ndarray, k_data: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled interpolation for n and k values."""
    n_interp = jnp.interp(l_target, l_data, n_data)
    k_interp_raw = jnp.interp(l_target, l_data, k_data)
    k_interp = jnp.maximum(k_interp_raw, 0.0)
    return n_interp + 1j * k_interp

MaterialInputType = Union[complex, float, int, str, Tuple[np.ndarray, np.ndarray, np.ndarray]]

def _get_nk_array_for_lambda_vec(material_definition: MaterialInputType,
                                     l_vec_target_jnp: jnp.ndarray,
                                     excel_file_path: str) -> Tuple[Optional[jnp.ndarray], List[str]]:
    """
    Resolves a material definition into a complex refractive index array for a given wavelength vector.
    Handles constants, built-in materials (Sellmeier), and data from Excel files.
    """
    logs = []
    try:
        if isinstance(material_definition, (complex, float, int)):
            nk_complex = jnp.asarray(material_definition, dtype=jnp.complex128)
            if nk_complex.real <= 0:
                logs.append(f"AVERTISSEMENT: Indice constant n'={nk_complex.real} <= 0 pour '{material_definition}'. Utilisation de n'=1.0.")
                nk_complex = complex(1.0, nk_complex.imag)
            if nk_complex.imag < 0:
                logs.append(f"AVERTISSEMENT: Indice constant k={nk_complex.imag} < 0 pour '{material_definition}'. Utilisation de k=0.0.")
                nk_complex = complex(nk_complex.real, 0.0)
            result = jnp.full(l_vec_target_jnp.shape, nk_complex)
        elif isinstance(material_definition, str):
            mat_upper = material_definition.upper()
            if mat_upper == "FUSED SILICA":
                result = get_n_fused_silica(l_vec_target_jnp)
            elif mat_upper == "BK7":
                result = get_n_bk7(l_vec_target_jnp)
            elif mat_upper == "D263":
                result = get_n_d263(l_vec_target_jnp)
            else:
                sheet_name = material_definition
                l_data, n_data, k_data, load_logs = load_material_data_from_xlsx_sheet(excel_file_path, sheet_name)
                logs.extend(load_logs)
                if l_data is None or len(l_data) == 0:
                    st.error(f"Impossible de charger ou données vides pour le matériau '{sheet_name}' depuis {excel_file_path}.")
                    logs.append(f"Erreur critique: Échec du chargement des données pour '{sheet_name}'.")
                    return None, logs
                l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
                l_target_min, l_target_max = jnp.min(l_vec_target_jnp), jnp.max(l_vec_target_jnp)
                l_data_min, l_data_max = jnp.min(l_data_jnp), jnp.max(l_data_jnp)
                if l_target_min < l_data_min - 1e-6 or l_target_max > l_data_max + 1e-6:
                    logs.append(f"AVERTISSEMENT: Interpolation pour '{sheet_name}' hors limites [{l_data_min:.1f}, {l_data_max:.1f}] nm (cible: [{l_target_min:.1f}, {l_target_max:.1f}] nm). Extrapolation utilisée.")
                result = interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)
        elif isinstance(material_definition, tuple) and len(material_definition) == 3:
            l_data, n_data, k_data = material_definition
            l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
            if not len(l_data_jnp): raise ValueError("Données de matériau brutes vides.")
            sort_indices = jnp.argsort(l_data_jnp)
            l_data_jnp, n_data_jnp, k_data_jnp = l_data_jnp[sort_indices], n_data_jnp[sort_indices], k_data_jnp[sort_indices]
            if np.any(k_data_jnp < -1e-9):
                logs.append("AVERTISSEMENT: k<0 dans les données de matériau brutes. Mise à 0.")
                k_data_jnp = jnp.maximum(k_data_jnp, 0.0)
            result = interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)
        else:
            raise TypeError(f"Type de définition de matériau non supporté: {type(material_definition)}")

        if jnp.any(jnp.isnan(result.real)) or jnp.any(result.real <= 0):
            logs.append(f"AVERTISSEMENT: n'<=0 ou NaN détecté pour '{material_definition}'. Remplacé par n'=1.")
            result = jnp.where(jnp.isnan(result.real) | (result.real <= 0), 1.0 + 1j*result.imag, result)
        if jnp.any(jnp.isnan(result.imag)) or jnp.any(result.imag < 0):
            logs.append(f"AVERTISSEMENT: k<0 ou NaN détecté pour '{material_definition}'. Remplacé par k=0.")
            result = jnp.where(jnp.isnan(result.imag) | (result.imag < 0), result.real + 0.0j, result)
        return result, logs
    except Exception as e:
        logs.append(f"Erreur lors de la préparation des données du matériau pour '{material_definition}': {e}")
        st.error(f"Erreur critique lors de la préparation du matériau '{material_definition}': {e}")
        return None, logs

def _get_nk_at_lambda(material_definition: MaterialInputType, l_nm_target: float, excel_file_path: str) -> Tuple[Optional[complex], List[str]]:
    """Gets the complex refractive index for a single target wavelength."""
    logs = []
    if l_nm_target <= 0:
        logs.append(f"Erreur: Longueur d'onde cible {l_nm_target}nm invalide pour obtenir n+ik.")
        return None, logs
    l_vec_jnp = jnp.array([l_nm_target], dtype=jnp.float64)
    nk_array, prep_logs = _get_nk_array_for_lambda_vec(material_definition, l_vec_jnp, excel_file_path)
    logs.extend(prep_logs)
    if nk_array is None:
        return None, logs
    else:
        return complex(nk_array[0]), logs

# --- Core Physics Calculation (JAX) ---
@jax.jit
def _compute_layer_matrix_scan_step_jit(carry_matrix: jnp.ndarray, layer_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, None]:
    """JAX scan step to compute and multiply one layer's characteristic matrix."""
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
    """Computes the total characteristic matrix for a stack at a single wavelength."""
    num_layers = len(ep_vector)
    layers_scan_data = (ep_vector, layer_indices, jnp.full(num_layers, l_val))
    M_initial = jnp.eye(2, dtype=jnp.complex128)
    M_final, _ = scan(_compute_layer_matrix_scan_step_jit, M_initial, layers_scan_data)
    return M_final

@jax.jit
def calculate_single_wavelength_TR_core(l_val: jnp.ndarray, ep_vector_contig: jnp.ndarray,
                                        layer_indices_at_lval: jnp.ndarray, nSub_at_lval: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculates Transmittance (T) and Reflectance (R) for a single wavelength.
    This is the core physics function that is vectorized for full spectrum calculation.
    """
    etainc = 1.0 + 0j
    etasub = nSub_at_lval

    def calculate_for_valid_l(l_: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        current_layer_indices = layer_indices_at_lval
        M = compute_stack_matrix_core_jax(ep_vector_contig, current_layer_indices, l_)
        m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

        rs_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        rs_denominator_abs = jnp.abs(rs_denominator)
        safe_denominator = jnp.where(rs_denominator_abs < 1e-12, 1e-12 + 0j, rs_denominator)

        # Reflectance (rs, Rs)
        rs_numerator = (etainc * m00 - etasub * m11 + etainc * etasub * m01 - m10)
        rs = rs_numerator / safe_denominator
        Rs_complex = rs * jnp.conj(rs)
        Rs = jnp.real(Rs_complex)

        # Transmittance (ts, Ts)
        ts = (2.0 * etainc) / safe_denominator
        real_etasub = jnp.real(etasub)
        real_etainc = jnp.real(etainc)
        safe_real_etainc = jnp.maximum(real_etainc, 1e-9)
        Ts_complex = (real_etasub / safe_real_etainc) * (ts * jnp.conj(ts))
        Ts = jnp.real(Ts_complex)

        return jnp.where(rs_denominator_abs < 1e-12, jnp.nan, Ts), jnp.where(rs_denominator_abs < 1e-12, jnp.nan, Rs)

    def calculate_for_invalid_l(l_: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return jnp.nan, jnp.nan

    return cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)


def calculate_TR_from_ep_jax(ep_vector: Union[np.ndarray, List[float]],
                             nH_material: MaterialInputType,
                             nL_material: MaterialInputType,
                             nSub_material: MaterialInputType,
                             l_vec: Union[np.ndarray, List[float]],
                             excel_file_path: str) -> Tuple[Optional[Dict[str, np.ndarray]], List[str]]:
    """
    Calculates T and R for a full spectrum for an alternating H/L stack.
    Vectorizes the single-wavelength calculation over the lambda vector.
    """
    logs = []
    l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64)
    ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)

    if not l_vec_jnp.size:
        logs.append("Vecteur lambda vide, aucun calcul T/R effectué.")
        return {'l': np.array([]), 'Ts': np.array([]), 'Rs': np.array([])}, logs

    logs.append(f"Préparation des indices pour {len(l_vec_jnp)} lambdas...")
    start_time = time.time()
    nH_arr, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_jnp, excel_file_path)
    logs.extend(logs_h)
    nL_arr, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_jnp, excel_file_path)
    logs.extend(logs_l)
    nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
    logs.extend(logs_sub)

    if nH_arr is None or nL_arr is None or nSub_arr is None:
        logs.append("Erreur critique: Échec du chargement d'un des indices de matériau.")
        return None, logs
    logs.append(f"Préparation des indices terminée en {time.time() - start_time:.3f}s.")

    if not ep_vector_jnp.size:
        logs.append("Structure vide (0 couches). Calcul pour le substrat nu.")
        etainc = 1.0 + 0j
        etasub = nSub_arr
        rs = (etainc - etasub) / (etainc + etasub)
        Rs = jnp.real(rs * jnp.conj(rs))
        Ts = 1.0 - Rs
        return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts), 'Rs': np.array(Rs)}, logs

    calculate_single_wavelength_TR_hl_jit = jax.jit(calculate_single_wavelength_TR_core)
    num_layers = len(ep_vector_jnp)
    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH_arr, nL_arr)
    indices_alternating_T = indices_alternating.T

    Ts_arr_raw, Rs_arr_raw = vmap(calculate_single_wavelength_TR_hl_jit, in_axes=(0, None, 0, 0))(
        l_vec_jnp, ep_vector_jnp, indices_alternating_T, nSub_arr
    )
    Ts_arr = jnp.nan_to_num(Ts_arr_raw, nan=0.0)
    Rs_arr = jnp.nan_to_num(Rs_arr_raw, nan=0.0)

    Ts_arr_clipped = jnp.clip(Ts_arr, 0.0, 1.0)
    Rs_arr_clipped = jnp.clip(Rs_arr, 0.0, 1.0)

    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts_arr_clipped), 'Rs': np.array(Rs_arr_clipped)}, logs


def calculate_TR_from_ep_arbitrary_jax(ep_vector: Union[np.ndarray, List[float]],
                                           material_sequence: List[str],
                                           nSub_material: MaterialInputType,
                                           l_vec: Union[np.ndarray, List[float]],
                                           excel_file_path: str) -> Tuple[Optional[Dict[str, np.ndarray]], List[str]]:
    """Calculates T and R for a stack with an arbitrary sequence of materials."""
    logs = []
    l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64)
    ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)
    num_layers = len(ep_vector_jnp)

    if num_layers != len(material_sequence):
        logs.append("Erreur: La taille de ep_vector et de material_sequence doit correspondre.")
        return None, logs

    if not l_vec_jnp.size:
        logs.append("Vecteur lambda vide.")
        return {'l': np.array([]), 'Ts': np.array([]), 'Rs': np.array([])}, logs

    logs.append(f"Préparation des indices pour une séquence arbitraire ({num_layers} couches, {len(l_vec_jnp)} lambdas)...")
    start_time = time.time()
    layer_indices_list = []
    materials_ok = True
    for i, mat_name in enumerate(material_sequence):
        nk_arr, logs_layer = _get_nk_array_for_lambda_vec(mat_name, l_vec_jnp, excel_file_path)
        logs.extend(logs_layer)
        if nk_arr is None:
            logs.append(f"Erreur critique: Échec du chargement du matériau '{mat_name}' (couche {i+1}).")
            materials_ok = False; break
        layer_indices_list.append(nk_arr)

    nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
    logs.extend(logs_sub)
    if nSub_arr is None:
        logs.append("Erreur critique: Échec du chargement du matériau du substrat.")
        materials_ok = False

    if not materials_ok: return None, logs

    if not ep_vector_jnp.size:
        logs.append("Structure vide (0 couches). Calcul pour le substrat nu.")
        etainc = 1.0 + 0j
        etasub = nSub_arr
        rs = (etainc - etasub) / (etainc + etasub)
        Rs = jnp.real(rs * jnp.conj(rs))
        Ts = 1.0 - Rs
        return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts), 'Rs': np.array(Rs)}, logs

    layer_indices_arr = jnp.stack(layer_indices_list, axis=0) if layer_indices_list else jnp.empty((0, len(l_vec_jnp)), dtype=jnp.complex128)
    logs.append(f"Préparation des indices terminée en {time.time() - start_time:.3f}s.")

    calculate_single_wavelength_TR_arb_jit = jax.jit(calculate_single_wavelength_TR_core)
    layer_indices_arr_T = layer_indices_arr.T
    Ts_arr_raw, Rs_arr_raw = vmap(calculate_single_wavelength_TR_arb_jit, in_axes=(0, None, 0, 0))(
        l_vec_jnp, ep_vector_jnp, layer_indices_arr_T, nSub_arr
    )
    Ts_arr = jnp.nan_to_num(Ts_arr_raw, nan=0.0)
    Rs_arr = jnp.nan_to_num(Rs_arr_raw, nan=0.0)

    Ts_arr_clipped = jnp.clip(Ts_arr, 0.0, 1.0)
    Rs_arr_clipped = jnp.clip(Rs_arr, 0.0, 1.0)

    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts_arr_clipped), 'Rs': np.array(Rs_arr_clipped)}, logs

# --- Structure Definition (QWOT, Thickness) ---
def calculate_initial_ep(emp: Union[List[float], Tuple[float,...]], l0: float,
                         nH0_material: MaterialInputType, nL0_material: MaterialInputType,
                         excel_file_path: str) -> Tuple[Optional[np.ndarray], List[str]]:
    """Calculates physical thicknesses from QWOT multipliers and a reference wavelength."""
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
        st.error(f"Erreur critique lors de l'obtention des indices à l0={l0}nm pour le calcul de l'épaisseur initiale.")
        return None, logs

    nH_real_at_l0, nL_real_at_l0 = nH_complex_at_l0.real, nL_complex_at_l0.real
    if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
        logs.append(f"AVERTISSEMENT: n'H({nH_real_at_l0:.3f}) ou n'L({nL_real_at_l0:.3f}) à l0={l0}nm est <= 0. Le calcul QWOT peut être incorrect.")

    for i in range(num_layers):
        multiplier = emp[i]
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real_layer_at_l0 <= 1e-9:
            ep_initial[i] = 0.0
        else:
            ep_initial[i] = multiplier * l0 / (4.0 * n_real_layer_at_l0)

    ep_initial_phys = np.where(ep_initial < MIN_THICKNESS_PHYS_NM, 0.0, ep_initial)
    num_clamped_zero = np.sum((ep_initial > 1e-12) & (ep_initial < MIN_THICKNESS_PHYS_NM))
    if num_clamped_zero > 0:
        logs.append(f"Avertissement: {num_clamped_zero} épaisseurs initiales < {MIN_THICKNESS_PHYS_NM}nm ont été mises à 0.")
        ep_initial = np.where(ep_initial < MIN_THICKNESS_PHYS_NM, 0.0, ep_initial)

    valid_indices = True
    for i in range(num_layers):
        if emp[i] > 1e-9 and ep_initial[i] < 1e-12:
            layer_type = "H" if i % 2 == 0 else "L"
            n_val = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
            logs.append(f"Erreur: Couche {i+1} ({layer_type}) a QWOT={emp[i]} mais épaisseur=0 (probablement n'({layer_type},l0)={n_val:.3f} <= 0).")
            valid_indices = False
    if not valid_indices:
        st.error("Erreur lors du calcul de l'épaisseur initiale en raison d'indices non valides à l0.")
        return None, logs

    return ep_initial, logs

def calculate_qwot_from_ep(ep_vector: np.ndarray, l0: float,
                           nH0_material: MaterialInputType, nL0_material: MaterialInputType,
                           excel_file_path: str) -> Tuple[Optional[np.ndarray], List[str]]:
    """Calculates QWOT multipliers from physical thicknesses."""
    logs = []
    num_layers = len(ep_vector)
    qwot_multipliers = np.full(num_layers, np.nan, dtype=np.float64)
    if l0 <= 0:
        logs.append(f"Avertissement: l0={l0} <= 0 dans calculate_qwot_from_ep. QWOT mis à NaN.")
        return qwot_multipliers, logs

    nH_complex_at_l0, logs_h = _get_nk_at_lambda(nH0_material, l0, excel_file_path)
    logs.extend(logs_h)
    nL_complex_at_l0, logs_l = _get_nk_at_lambda(nL0_material, l0, excel_file_path)
    logs.extend(logs_l)

    if nH_complex_at_l0 is None or nL_complex_at_l0 is None:
        logs.append(f"Erreur: Impossible d'obtenir n'H ou n'L à l0={l0}nm pour calculer le QWOT. Retour de NaN.")
        st.error(f"Erreur de calcul du QWOT: Indices H/L non trouvés à l0={l0}nm.")
        return None, logs

    nH_real_at_l0, nL_real_at_l0 = nH_complex_at_l0.real, nL_complex_at_l0.real
    if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
        logs.append(f"AVERTISSEMENT: n'H({nH_real_at_l0:.3f}) ou n'L({nL_real_at_l0:.3f}) à l0={l0}nm est <= 0. Le calcul QWOT peut être incorrect/NaN.")

    indices_ok = True
    for i in range(num_layers):
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real_layer_at_l0 <= 1e-9:
            if ep_vector[i] > 1e-9 :
                layer_type = "H" if i % 2 == 0 else "L"
                logs.append(f"Avertissement: Impossible de calculer le QWOT pour la couche {i+1} ({layer_type}) car n'({l0}nm) <= 0.")
                indices_ok = False
            else:
                qwot_multipliers[i] = 0.0
        else:
            qwot_multipliers[i] = ep_vector[i] * (4.0 * n_real_layer_at_l0) / l0
    if not indices_ok:
        st.warning("Certaines valeurs QWOT n'ont pas pu être calculées (indices non valides à l0). Elles apparaissent comme NaN.")
        return qwot_multipliers, logs
    else:
        return qwot_multipliers, logs

# --- Target and RMSE Calculation ---
def calculate_final_rmse_manual(res: Dict[str, np.ndarray], manual_targets: List[Dict]) -> Tuple[Optional[float], int]:
    """Calculates RMSE between results and manually defined target zones."""
    total_squared_error = 0.0
    total_points_in_targets = 0
    res_l_np = np.asarray(res['l'])
    res_ts_np = np.asarray(res['Ts'])

    for target in manual_targets:
        l_min, l_max = float(target['min']), float(target['max'])
        t_min, t_max = float(target['target_min']), float(target['target_max'])
        indices = np.where((res_l_np >= l_min) & (res_l_np <= l_max))[0]
        if indices.size > 0:
            calculated_Ts_in_zone = res_ts_np[indices]
            target_lambdas_in_zone = res_l_np[indices]
            if abs(l_max - l_min) < 1e-9:
                interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
            else:
                slope = (t_max - t_min) / (l_max - l_min)
                interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)
            squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
            total_squared_error += np.sum(squared_errors)
            total_points_in_targets += len(calculated_Ts_in_zone)
    if total_points_in_targets > 0:
        mse = total_squared_error / total_points_in_targets
        return np.sqrt(mse), total_points_in_targets
    return None, 0

def calculate_final_rmse(res: Dict[str, np.ndarray], target_info: Dict) -> Tuple[Optional[float], int]:
    """Calculates final RMSE for display, handling both file and manual targets."""
    if not target_info or 'type' not in target_info or 'data' not in target_info or not target_info['data']:
        return None, 0

    if target_info['type'] == 'file':
        target_l = target_info['data']['lambdas']
        target_t = target_info['data']['transmittances']
        if target_l.size == 0: return None, 0

        sim_l, sim_t = res.get('l'), res.get('Ts')
        if sim_l is None or sim_t is None or sim_l.size == 0: return None, 0

        valid_mask = (target_l >= sim_l.min()) & (target_l <= sim_l.max())
        if not np.any(valid_mask): return None, 0

        target_l_valid, target_t_valid = target_l[valid_mask], target_t[valid_mask]

        calculated_t_at_target_l = np.interp(target_l_valid, sim_l, sim_t)

        squared_errors = (calculated_t_at_target_l - target_t_valid)**2
        mse = np.mean(squared_errors)
        return np.sqrt(mse), len(target_l_valid)

    elif target_info['type'] == 'manual':
        return calculate_final_rmse_manual(res, target_info['data'])

    return None, 0

def parse_target_file(uploaded_file) -> Optional[Dict[str, np.ndarray]]:
    """Parses an uploaded Excel file for target data."""
    if uploaded_file is None: return None
    try:
        df = pd.read_excel(uploaded_file, header=None)
        if df.shape[1] < 2:
            st.error("Le fichier cible doit avoir au moins deux colonnes (lambda, T%).")
            return None

        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        df.dropna(subset=[df.columns[0], df.columns[1]], inplace=True)

        if df.empty:
            st.warning("Aucune donnée numérique valide trouvée dans le fichier cible téléchargé.")
            return {'lambdas': np.array([]), 'transmittances': np.array([])}

        df = df.sort_values(by=df.columns[0])
        l_target = df.iloc[:, 0].values.astype(np.float64)
        t_target_percent = df.iloc[:, 1].values.astype(np.float64)
        t_target = np.clip(t_target_percent / 100.0, 0.0, 1.0)

        return {'lambdas': l_target, 'transmittances': t_target}
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier cible: {e}")
        return None

def get_target_data() -> Optional[Dict[str, Any]]:
    """Gets target data from either an uploaded file or manual definitions."""
    uploaded_file = st.session_state.get("target_file_uploader_key")
    if uploaded_file:
        file_data = parse_target_file(uploaded_file)
        if file_data is not None:
            st.info(f"Utilisation de {len(file_data['lambdas'])} points cibles de '{uploaded_file.name}'. Cibles manuelles ignorées.")
            return {'type': 'file', 'data': file_data}
        return None
    else:
        manual_targets = validate_targets()
        if manual_targets is not None:
            return {'type': 'manual', 'data': manual_targets}
        return None

def validate_targets() -> Optional[List[Dict]]:
    """Validates the manual target definitions from the UI."""
    active_targets = []
    is_valid = True
    if 'targets' not in st.session_state or not isinstance(st.session_state.targets, list):
        st.error("Erreur interne: Liste de cibles manquante ou invalide dans session_state."); return None

    for i, target_state in enumerate(st.session_state.targets):
        if target_state.get('enabled', False):
            try:
                l_min, l_max = float(target_state['min']), float(target_state['max'])
                t_min, t_max = float(target_state['target_min']), float(target_state['target_max'])
                if l_max < l_min: is_valid = False; continue
                if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0): is_valid = False; continue
                active_targets.append({'min': l_min, 'max': l_max, 'target_min': t_min, 'target_max': t_max})
            except (KeyError, ValueError, TypeError):
                is_valid = False; continue

    if not is_valid:
        st.warning("Des erreurs existent dans les définitions des cibles spectrales actives. Veuillez corriger."); return None
    return active_targets

def get_lambda_range(target_info: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """Determines the min and max lambda from the active targets."""
    if not target_info or not target_info.get('data'): return None, None

    if target_info['type'] == 'file':
        lambdas = target_info['data']['lambdas']
        return (np.min(lambdas), np.max(lambdas)) if lambdas.size > 0 else (None, None)

    elif target_info['type'] == 'manual':
        manual_targets = target_info['data']
        if not manual_targets: return None, None
        all_mins = [t['min'] for t in manual_targets]
        all_maxs = [t['max'] for t in manual_targets]
        return min(all_mins), max(all_maxs)

    return None, None

def interpolate_manual_targets(l_vec: np.ndarray, manual_targets: List[Dict]) -> np.ndarray:
    """Creates a target transmittance vector by interpolating over manual target zones."""
    target_t = np.zeros_like(l_vec)
    for target in manual_targets:
        l_min, l_max = target['min'], target['max']
        t_min, t_max = target['target_min'], target['target_max']
        mask = (l_vec >= l_min) & (l_vec <= l_max)
        if np.any(mask):
            l_segment = l_vec[mask]
            if abs(l_max - l_min) < 1e-9:
                interp_t = np.full_like(l_segment, t_min)
            else:
                slope = (t_max - t_min) / (l_max - l_min)
                interp_t = t_min + slope * (l_segment - l_min)
            target_t[mask] = interp_t
    return target_t

@jax.jit
def calculate_rmse_for_optimization_unified_jax(ep_vector: jnp.ndarray,
                                                nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                                l_vec_optim: jnp.ndarray,
                                                target_t_at_l_vec: jnp.ndarray,
                                                min_thickness_phys_nm: float,
                                                use_backside_correction: bool) -> jnp.ndarray:
    """
    JAX-compiled cost function for optimization. Calculates RMSE and includes penalties for thin layers.
    """
    below_min_mask = (ep_vector < min_thickness_phys_nm) & (ep_vector > 1e-12)
    penalty_thin = jnp.sum(jnp.where(below_min_mask, (min_thickness_phys_nm - ep_vector)**2, 0.0))
    penalty_cost = penalty_thin * 1e5
    ep_vector_calc = jnp.maximum(ep_vector, min_thickness_phys_nm)

    num_layers = len(ep_vector_calc)
    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH_arr, nL_arr)
    indices_alternating_T = indices_alternating.T

    calculate_TR_single_jit = jax.jit(calculate_single_wavelength_TR_core)
    Ts_raw, Rs_raw = vmap(calculate_TR_single_jit, in_axes=(0, None, 0, 0))(
        l_vec_optim, ep_vector_calc, indices_alternating_T, nSub_arr
    )
    Ts = jnp.nan_to_num(Ts_raw, nan=0.0)
    Rs = jnp.nan_to_num(Rs_raw, nan=0.0)

    def apply_correction(T_R_nSub):
        T, R, nSub = T_R_nSub
        ns = jnp.real(nSub)
        safe_ns_plus_1 = jnp.where(jnp.abs(ns + 1.0) < 1e-9, 1e-9, ns + 1.0)
        RB = ((ns - 1.0) / safe_ns_plus_1)**2
        TB = 1.0 - RB
        denominator = 1.0 - R * RB
        safe_denominator = jnp.where(jnp.abs(denominator) < 1e-9, 1e-9, denominator)
        T_corrected = T * TB / safe_denominator
        return T_corrected

    def no_correction(T_R_nSub):
        T, _, _ = T_R_nSub
        return T

    Ts_final = cond(use_backside_correction, apply_correction, no_correction, (Ts, Rs, nSub_arr))

    squared_errors = (Ts_final - target_t_at_l_vec)**2
    mse = jnp.mean(squared_errors)

    final_cost = jnp.sqrt(mse + penalty_cost)
    return jnp.nan_to_num(final_cost, nan=jnp.inf, posinf=jnp.inf)

# --- Optimization and Structure Modification ---
def _run_core_optimization(ep_start_optim: np.ndarray,
                           validated_inputs: Dict,
                           target_info: Dict,
                           min_thickness_phys: float,
                           use_backside_correction: bool,
                           log_prefix: str = ""
                           ) -> Tuple[Optional[np.ndarray], bool, float, List[str], str]:
    """
    Runs the core L-BFGS-B optimization using Scipy and JAX for gradients.
    """
    logs = []
    num_layers_start = len(ep_start_optim)
    optim_success = False
    final_cost = np.inf
    result_message_str = "Optimisation non lancée ou échouée prématurément."
    final_ep = None

    if num_layers_start == 0:
        logs.append(f"{log_prefix}Impossible d'optimiser une structure vide.")
        return None, False, np.inf, logs, "Structure vide"

    try:
        nH_material, nL_material, nSub_material = validated_inputs['nH_material'], validated_inputs['nL_material'], validated_inputs['nSub_material']
        maxiter, maxfun = MAXITER_HARDCODED, MAXFUN_HARDCODED

        if target_info['type'] == 'file':
            l_vec_optim_np = target_info['data']['lambdas']
            target_t_at_l_vec_np = target_info['data']['transmittances']
        else: # manual
            manual_targets = target_info['data']
            l_min_optim, l_max_optim = get_lambda_range(target_info)
            l_step_optim = validated_inputs['l_step']
            num_pts_optim = min(max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1), 100)
            l_vec_optim_np = np.geomspace(l_min_optim, l_max_optim, num_pts_optim)
            target_t_at_l_vec_np = interpolate_manual_targets(l_vec_optim_np, manual_targets)

        l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
        if not l_vec_optim_np.size: raise ValueError("Échec de la génération du vecteur lambda pour l'optimisation.")
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        target_t_at_l_vec_jax = jnp.asarray(target_t_at_l_vec_np)

        logs.append(f"{log_prefix}Préparation des indices dispersifs pour {len(l_vec_optim_jax)} lambdas...")
        prep_start_time = time.time()
        nH_arr_optim, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_h)
        nL_arr_optim, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_l)
        nSub_arr_optim, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_sub)
        if nH_arr_optim is None or nL_arr_optim is None or nSub_arr_optim is None:
            raise RuntimeError("Échec du chargement des indices pour l'optimisation.")
        logs.append(f"{log_prefix} Préparation des indices terminée en {time.time() - prep_start_time:.3f}s.")

        static_args_for_jax = (
            nH_arr_optim, nL_arr_optim, nSub_arr_optim,
            l_vec_optim_jax, target_t_at_l_vec_jax,
            min_thickness_phys,
            use_backside_correction
        )
        value_and_grad_fn = jax.jit(value_and_grad(calculate_rmse_for_optimization_unified_jax))

        def scipy_obj_grad_wrapper(ep_vector_np_in, *args):
            try:
                ep_vector_jax = jnp.asarray(ep_vector_np_in, dtype=jnp.float64)
                value_jax, grad_jax = value_and_grad_fn(ep_vector_jax, *args)
                if not jnp.isfinite(value_jax):
                    value_np, grad_np = np.inf, np.zeros_like(ep_vector_np_in, dtype=np.float64)
                else:
                    value_np = float(np.array(value_jax))
                    grad_np_raw = np.array(grad_jax, dtype=np.float64)
                    grad_np = np.nan_to_num(grad_np_raw, nan=0.0, posinf=1e6, neginf=-1e6)
                return value_np, grad_np
            except Exception as e_wrap:
                print(f"Erreur dans scipy_obj_grad_wrapper: {e_wrap}")
                return np.inf, np.zeros_like(ep_vector_np_in, dtype=np.float64)

        lbfgsb_bounds = [(min_thickness_phys, None)] * num_layers_start
        options = {'maxiter': maxiter, 'maxfun': maxfun, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-8}
        logs.append(f"{log_prefix}Démarrage de L-BFGS-B avec gradient JAX...")
        opt_start_time = time.time()
        result = minimize(scipy_obj_grad_wrapper, ep_start_optim, args=static_args_for_jax,
                          method='L-BFGS-B', jac=True, bounds=lbfgsb_bounds, options=options)
        logs.append(f"{log_prefix}L-BFGS-B (grad JAX) terminé en {time.time() - opt_start_time:.3f}s.")

        final_cost = result.fun if np.isfinite(result.fun) else np.inf
        result_message_str = result.message.decode('utf-8') if isinstance(result.message, bytes) else str(result.message)
        is_success_or_limit = (result.success or result.status == 1) and np.isfinite(final_cost)

        if is_success_or_limit:
            final_ep_raw = result.x
            final_ep = np.maximum(final_ep_raw, min_thickness_phys)
            optim_success = True
            log_status = "succès" if result.success else "limite atteinte"
            logs.append(f"{log_prefix}Optimisation terminée ({log_status}). Coût final (RMSE): {final_cost:.3e}, Msg: {result_message_str}")
        else:
            optim_success = False
            final_ep = np.maximum(ep_start_optim, min_thickness_phys)
            logs.append(f"{log_prefix}Optimisation ÉCHOUÉE. Statut: {result.status}, Msg: {result_message_str}, Coût: {final_cost:.3e}")
            try:
                reverted_cost, _ = scipy_obj_grad_wrapper(final_ep, *static_args_for_jax)
                logs.append(f"{log_prefix}Retour à la structure initiale (clampée). Coût recalculé: {reverted_cost:.3e}")
                final_cost = reverted_cost if np.isfinite(reverted_cost) else np.inf
            except Exception as cost_e:
                logs.append(f"{log_prefix}Retour à la structure initiale (clampée). ERREUR de recalcul du coût: {cost_e}")
                final_cost = np.inf

    except Exception as e_optim:
        logs.append(f"{log_prefix}ERREUR majeure pendant l'optimisation JAX/Scipy: {e_optim}\n{traceback.format_exc(limit=2)}")
        st.error(f"Erreur critique pendant l'optimisation: {e_optim}")
        final_ep = np.maximum(ep_start_optim, min_thickness_phys) if ep_start_optim is not None else None
        optim_success = False
        final_cost = np.inf
        result_message_str = f"Exception: {e_optim}"

    return final_ep, optim_success, final_cost, logs, result_message_str

def _perform_layer_merge_or_removal_only(ep_vector_in: np.ndarray, min_thickness_phys: float,
                                         log_prefix: str = "", target_layer_index: Optional[int] = None,
                                         threshold_for_removal: Optional[float] = None) -> Tuple[Optional[np.ndarray], bool, List[str]]:
    """Removes the thinnest layer from a stack and merges its neighbors."""
    current_ep = ep_vector_in.copy()
    logs = []
    num_layers = len(current_ep)
    structure_changed = False
    ep_after_merge = None

    if num_layers <= 2 and target_layer_index is None:
        logs.append(f"{log_prefix}Structure <= 2 couches. Suppression/fusion impossible sans cible.")
        return current_ep, False, logs
    elif num_layers < 1:
        logs.append(f"{log_prefix}Structure vide.")
        return current_ep, False, logs

    try:
        thin_layer_index = -1
        min_thickness_found = np.inf
        if target_layer_index is not None:
            if 0 <= target_layer_index < num_layers and current_ep[target_layer_index] >= min_thickness_phys:
                thin_layer_index = target_layer_index
                min_thickness_found = current_ep[target_layer_index]
                logs.append(f"{log_prefix}Ciblage manuel de la couche {thin_layer_index + 1} ({min_thickness_found:.3f} nm).")
            else:
                logs.append(f"{log_prefix}Cible manuelle {target_layer_index+1} invalide/trop fine. Recherche automatique.")
                target_layer_index = None

        if target_layer_index is None:
            candidate_indices = np.where(current_ep >= min_thickness_phys)[0]
            if candidate_indices.size == 0:
                logs.append(f"{log_prefix}Aucune couche >= {min_thickness_phys:.3f} nm trouvée.")
                return current_ep, False, logs

            indices_to_consider = candidate_indices
            thicknesses_to_consider = current_ep[candidate_indices]
            if threshold_for_removal is not None:
                mask_below_threshold = thicknesses_to_consider < threshold_for_removal
                if np.any(mask_below_threshold):
                    indices_to_consider = indices_to_consider[mask_below_threshold]
                    thicknesses_to_consider = thicknesses_to_consider[mask_below_threshold]
                    logs.append(f"{log_prefix}Recherche parmi les couches < {threshold_for_removal:.3f} nm.")
                else:
                    logs.append(f"{log_prefix}Aucune couche éligible (< {threshold_for_removal:.3f} nm) trouvée.")
                    return current_ep, False, logs

            if indices_to_consider.size > 0:
                min_idx_local = np.argmin(thicknesses_to_consider)
                thin_layer_index = indices_to_consider[min_idx_local]
                min_thickness_found = thicknesses_to_consider[min_idx_local]
            else:
                logs.append(f"{log_prefix}Aucune couche candidate finale trouvée.")
                return current_ep, False, logs

        if thin_layer_index == -1:
            logs.append(f"{log_prefix}Échec de l'identification de la couche (cas inattendu).")
            return current_ep, False, logs

        thin_layer_thickness = current_ep[thin_layer_index]
        logs.append(f"{log_prefix}Couche identifiée pour action: Index {thin_layer_index} (Couche {thin_layer_index + 1}), épaisseur {thin_layer_thickness:.3f} nm.")

        if num_layers <= 2 and thin_layer_index == 0:
            ep_after_merge = current_ep[2:]
            merged_info = "Suppression des 2 premières couches (taille de la structure <= 2)."
            structure_changed = True
        elif num_layers <= 1 and thin_layer_index == 0:
            ep_after_merge = np.array([])
            merged_info = "Suppression de la seule couche."
            structure_changed = True
        elif num_layers <= 2 and thin_layer_index == 1:
            ep_after_merge = current_ep[:-1]
            merged_info = "Suppression de la dernière couche (taille de la structure 2)."
            structure_changed = True
        elif thin_layer_index == 0:
            ep_after_merge = current_ep[2:]
            merged_info = "Suppression des 2 premières couches."
            structure_changed = True
        elif thin_layer_index == num_layers - 1:
            if num_layers >= 1:
                ep_after_merge = current_ep[:-1]
                merged_info = f"Suppression de SEULEMENT la dernière couche (Couche {num_layers})."
                structure_changed = True
            else:
                logs.append(f"{log_prefix}Cas spécial: impossible de supprimer la dernière couche (num_layers={num_layers}).")
                return current_ep, False, logs
        else:
            merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
            ep_before = current_ep[:thin_layer_index - 1]
            ep_after = current_ep[thin_layer_index + 2:]
            ep_after_merge = np.concatenate((ep_before, [merged_thickness], ep_after))
            merged_info = f"Fusion des couches {thin_layer_index} et {thin_layer_index + 2} autour de la couche supprimée {thin_layer_index + 1} -> nouvelle épaisseur {merged_thickness:.3f} nm."
            structure_changed = True

        if structure_changed and ep_after_merge is not None:
            logs.append(f"{log_prefix}{merged_info} Nouvelle structure: {len(ep_after_merge)} couches.")
            ep_after_merge = np.maximum(ep_after_merge, min_thickness_phys)
            return ep_after_merge, True, logs
        elif structure_changed and ep_after_merge is None:
            logs.append(f"{log_prefix}Erreur logique: structure_changed=True mais ep_after_merge=None.")
            return current_ep, False, logs
        else:
            logs.append(f"{log_prefix}Aucune modification de structure effectuée.")
            return current_ep, False, logs

    except Exception as e_merge:
        logs.append(f"{log_prefix}ERREUR pendant la logique de fusion/suppression: {e_merge}\n{traceback.format_exc(limit=1)}")
        st.error(f"Erreur interne lors de la suppression/fusion de couche: {e_merge}")
        return current_ep, False, logs

def _perform_needle_insertion_scan(ep_vector_in: np.ndarray,
                                   nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
                                   l_vec_optim_np: np.ndarray, target_info: Dict,
                                   min_thickness_phys: float, base_needle_thickness_nm: float,
                                   scan_step: float, l0_repr: float,
                                   excel_file_path: str, use_backside_correction: bool,
                                   log_prefix: str = ""
                                   ) -> Tuple[Optional[np.ndarray], float, List[str], int]:
    """
    Scans through the stack to find the optimal position to insert a new "needle" layer.
    """
    logs = []
    num_layers_in = len(ep_vector_in)
    if num_layers_in == 0:
        logs.append(f"{log_prefix}Scan d'aiguille impossible sur une structure vide.")
        return None, np.inf, logs, -1

    logs.append(f"{log_prefix}Démarrage du scan d'aiguille ({num_layers_in} couches). Pas: {scan_step} nm, épaisseur aiguille: {base_needle_thickness_nm:.3f} nm.")
    try:
        if target_info['type'] == 'file':
            target_t_at_l_vec_np = np.interp(l_vec_optim_np, target_info['data']['lambdas'], target_info['data']['transmittances'])
        else: # manual
            target_t_at_l_vec_np = interpolate_manual_targets(l_vec_optim_np, target_info['data'])
        
        target_t_at_l_vec_jax = jnp.asarray(target_t_at_l_vec_np)
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        
        nH_arr_optim, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, excel_file_path)
        logs.extend(logs_h)
        nL_arr_optim, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, excel_file_path)
        logs.extend(logs_l)
        nSub_arr_optim, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, excel_file_path)
        logs.extend(logs_sub)
        if nH_arr_optim is None or nL_arr_optim is None or nSub_arr_optim is None:
            raise RuntimeError("Échec du chargement des indices pour le scan d'aiguille.")

        cost_function_jax = jax.jit(calculate_rmse_for_optimization_unified_jax)
        static_args_cost_fn = (
            nH_arr_optim, nL_arr_optim, nSub_arr_optim,
            l_vec_optim_jax, target_t_at_l_vec_jax,
            min_thickness_phys, use_backside_correction
        )
        initial_cost_jax = cost_function_jax(jnp.asarray(ep_vector_in), *static_args_cost_fn)
        initial_cost = float(np.array(initial_cost_jax))
        if not np.isfinite(initial_cost):
            logs.append(f"{log_prefix} ERREUR: Coût initial non fini ({initial_cost}). Scan annulé.")
            st.error("Erreur du scan d'aiguille: Le coût de la structure de départ n'est pas fini.")
            return None, np.inf, logs, -1
        logs.append(f"{log_prefix} Coût initial: {initial_cost:.6e}")
    except Exception as e_prep:
        logs.append(f"{log_prefix} ERREUR lors de la préparation du scan d'aiguille: {e_prep}")
        st.error(f"Erreur lors de la préparation du scan d'aiguille: {e_prep}")
        return None, np.inf, logs, -1

    best_ep_found, min_cost_found, best_insertion_idx = None, initial_cost, -1
    tested_insertions = 0
    ep_cumsum = np.cumsum(ep_vector_in)
    total_thickness = ep_cumsum[-1] if num_layers_in > 0 else 0.0

    for z in np.arange(scan_step, total_thickness, scan_step):
        current_layer_idx = -1
        layer_start_z = 0.0
        for i in range(num_layers_in):
            layer_end_z = ep_cumsum[i]
            if z > layer_start_z and z <= layer_end_z:
                t_part1, t_part2 = z - layer_start_z, layer_end_z - z
                if t_part1 >= min_thickness_phys and t_part2 >= min_thickness_phys:
                    current_layer_idx = i
                else:
                    current_layer_idx = -2 # Flag to skip
                break
            layer_start_z = layer_end_z

        if current_layer_idx < 0: continue

        tested_insertions += 1
        t_layer_split_1 = z - (ep_cumsum[current_layer_idx-1] if current_layer_idx > 0 else 0.0)
        t_layer_split_2 = ep_cumsum[current_layer_idx] - z
        ep_temp_np = np.concatenate((
            ep_vector_in[:current_layer_idx],
            [t_layer_split_1, base_needle_thickness_nm, t_layer_split_2],
            ep_vector_in[current_layer_idx+1:]
        ))
        ep_temp_np_clamped = np.maximum(ep_temp_np, min_thickness_phys)

        try:
            current_cost_jax = cost_function_jax(jnp.asarray(ep_temp_np_clamped), *static_args_cost_fn)
            current_cost = float(np.array(current_cost_jax))
            if np.isfinite(current_cost) and current_cost < min_cost_found:
                min_cost_found = current_cost
                best_ep_found = ep_temp_np_clamped.copy()
                best_insertion_idx = current_layer_idx
        except Exception as e_cost:
            logs.append(f"{log_prefix} AVERTISSEMENT: Échec du calcul du coût pour z={z:.2f}. {e_cost}")
            continue

    if best_ep_found is not None:
        improvement = initial_cost - min_cost_found
        logs.append(f"{log_prefix} Scan terminé. {tested_insertions} points testés.")
        logs.append(f"{log_prefix} Meilleure amélioration trouvée: {improvement:.6e} (RMSE {min_cost_found:.6e})")
        logs.append(f"{log_prefix} Insertion optimale dans la couche d'origine {best_insertion_idx + 1}.")
        return best_ep_found, min_cost_found, logs, best_insertion_idx
    else:
        logs.append(f"{log_prefix} Scan terminé. {tested_insertions} points testés. Aucune amélioration trouvée.")
        return None, initial_cost, logs, -1

def _run_needle_iterations(ep_start: np.ndarray, num_needles: int,
                           validated_inputs: Dict, target_info: Dict,
                           min_thickness_phys: float, l_vec_optim_np_in: np.ndarray,
                           scan_step_nm: float, base_needle_thickness_nm: float,
                           excel_file_path: str, use_backside_correction: bool,
                           log_prefix: str = ""
                           ) -> Tuple[np.ndarray, float, List[str]]:
    """
    Runs a sequence of needle insertions followed by re-optimizations.
    """
    logs = []
    best_ep_overall = np.asarray(ep_start).copy()
    best_rmse_overall = np.inf
    
    l0_repr = validated_inputs.get('l0', 500.0)
    
    try:
        _, _, initial_rmse, init_logs, _ = _run_core_optimization(best_ep_overall, validated_inputs, target_info, min_thickness_phys, use_backside_correction)
        logs.extend(init_logs)
        if not np.isfinite(initial_rmse):
            raise ValueError("Le RMSE initial pour les itérations d'aiguille n'est pas fini.")
        best_rmse_overall = initial_rmse
        logs.append(f"{log_prefix} Démarrage des itérations d'aiguille ({num_needles} max). RMSE initial: {best_rmse_overall:.6e}")
    except Exception as e_init:
        logs.append(f"{log_prefix} ERREUR lors du calcul du RMSE initial pour les itérations d'aiguille: {e_init}")
        st.error(f"Erreur lors de l'initialisation des itérations d'aiguille: {e_init}")
        return ep_start, np.inf, logs

    for i in range(num_needles):
        logs.append(f"{log_prefix} --- Itération d'aiguille {i + 1}/{num_needles} ---")
        current_ep_iter = best_ep_overall.copy()
        if len(current_ep_iter) == 0:
            logs.append(f"{log_prefix} Structure vide, arrêt des itérations d'aiguille."); break

        st.write(f"{log_prefix} Scan d'aiguille {i+1}...")
        ep_after_scan, cost_after_scan, scan_logs, inserted_idx = _perform_needle_insertion_scan(
            current_ep_iter, validated_inputs['nH_material'], validated_inputs['nL_material'], validated_inputs['nSub_material'],
            l_vec_optim_np_in, target_info,
            min_thickness_phys, base_needle_thickness_nm, scan_step_nm, l0_repr,
            excel_file_path, use_backside_correction, log_prefix=f"{log_prefix}  [Scan {i+1}] "
        )
        logs.extend(scan_logs)

        if ep_after_scan is None:
            logs.append(f"{log_prefix} Le scan d'aiguille {i + 1} n'a trouvé aucune amélioration. Arrêt des itérations d'aiguille."); break

        logs.append(f"{log_prefix} Le scan {i + 1} a trouvé une amélioration potentielle. Ré-optimisation...")
        st.write(f"{log_prefix} Ré-optimisation après l'aiguille {i+1}...")
        ep_after_reopt, optim_success, final_cost_reopt, optim_logs, optim_status_msg = \
            _run_core_optimization(ep_after_scan, validated_inputs, target_info,
                                   min_thickness_phys, use_backside_correction,
                                   log_prefix=f"{log_prefix}  [Re-Opt {i+1}] ")
        logs.extend(optim_logs)

        if not optim_success:
            logs.append(f"{log_prefix} La ré-optimisation après le scan {i + 1} a ÉCHOUÉ. Arrêt des itérations d'aiguille."); break

        logs.append(f"{log_prefix} Ré-optimisation {i + 1} réussie. Nouveau RMSE: {final_cost_reopt:.6e}.")
        if final_cost_reopt < best_rmse_overall - COST_IMPROVEMENT_TOLERANCE:
            logs.append(f"{log_prefix}  RMSE amélioré par rapport au meilleur précédent ({best_rmse_overall:.6e}). Mise à jour.")
            best_ep_overall, best_rmse_overall = ep_after_reopt.copy(), final_cost_reopt
        else:
            logs.append(f"{log_prefix}  Nouveau RMSE ({final_cost_reopt:.6e}) pas significativement meilleur que le précédent ({best_rmse_overall:.6e}). Arrêt des itérations d'aiguille.")
            best_ep_overall, best_rmse_overall = ep_after_reopt.copy(), final_cost_reopt
            break

    logs.append(f"{log_prefix} Fin des itérations d'aiguille. Meilleur RMSE final: {best_rmse_overall:.6e}")
    return best_ep_overall, best_rmse_overall, logs

def run_auto_mode(initial_ep: Optional[np.ndarray],
                  validated_inputs: Dict, target_info: Dict,
                  excel_file_path: str, use_backside_correction: bool,
                  log_callback: Callable):
    """
    Runs the full automatic optimization routine: Needle -> Thin -> Opt cycles.
    """
    logs = []
    start_time_auto = time.time()
    log_callback("#"*10 + f" Démarrage du Mode Auto (Max {AUTO_MAX_CYCLES} Cycles) " + "#"*10)
    best_ep_so_far, best_rmse_so_far = None, np.inf
    num_cycles_done = 0
    termination_reason = f"Max {AUTO_MAX_CYCLES} cycles atteint"
    threshold_for_thin_removal = validated_inputs.get('auto_thin_threshold', 1.0)
    log_callback(f"  Seuil de suppression auto: {threshold_for_thin_removal:.3f} nm")

    try:
        current_ep = None
        if initial_ep is not None:
            log_callback("  Mode Auto: Utilisation de la structure optimisée précédente.")
            current_ep = initial_ep.copy()
        else:
            log_callback("  Mode Auto: Utilisation de la structure nominale (QWOT).")
            emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
            if not emp_list: raise ValueError("QWOT nominal vide.")
            ep_nominal, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'],
                                                            validated_inputs['nH_material'], validated_inputs['nL_material'],
                                                            excel_file_path)
            log_callback(logs_ep_init)
            if ep_nominal is None: raise RuntimeError("Échec du calcul des épaisseurs nominales initiales.")
            current_ep = ep_nominal.copy()

        log_callback(f"  Démarrage de l'optimisation/évaluation initiale...")
        st.info("Mode Auto: Optimisation initiale de la structure de départ...")
        ep_after_initial_opt, initial_opt_success, initial_rmse, initial_opt_logs, initial_opt_msg = \
            _run_core_optimization(current_ep, validated_inputs, target_info,
                                   MIN_THICKNESS_PHYS_NM, use_backside_correction,
                                   log_prefix="  [Auto Init Opt] ")
        log_callback(initial_opt_logs)
        if not initial_opt_success:
            log_callback(f"ERREUR: Échec de l'optimisation initiale en Mode Auto ({initial_opt_msg}). Annulation.")
            st.error(f"Échec de l'optimisation initiale du Mode Auto: {initial_opt_msg}")
            return None, np.inf, logs
        log_callback(f"  Optimisation initiale terminée. RMSE: {initial_rmse:.6e}")
        best_ep_so_far, best_rmse_so_far = ep_after_initial_opt.copy(), initial_rmse

        if best_ep_so_far is None or not np.isfinite(best_rmse_so_far):
            raise RuntimeError("État de départ invalide pour les cycles Auto.")

        log_callback(f"--- Démarrage des Cycles Auto (RMSE de départ: {best_rmse_so_far:.6e}, {len(best_ep_so_far)} couches) ---")
        for cycle_num in range(AUTO_MAX_CYCLES):
            log_callback(f"\n--- Cycle Auto {cycle_num + 1} / {AUTO_MAX_CYCLES} ---")
            st.info(f"Cycle Auto {cycle_num + 1}/{AUTO_MAX_CYCLES} | RMSE actuel: {best_rmse_so_far:.3e}")
            rmse_at_cycle_start, ep_at_cycle_start = best_rmse_so_far, best_ep_so_far.copy()
            cycle_improved_globally = False

            log_callback(f"  [Cycle {cycle_num+1}] Phase Aiguille ({AUTO_NEEDLES_PER_CYCLE} itérations max)...")
            st.write(f"Cycle {cycle_num + 1}: Phase Aiguille...")
            l_min_optim, l_max_optim = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
            l_step_optim = validated_inputs['l_step']
            num_pts_needle_auto = min(max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1), 100)
            l_vec_optim_np_needle_auto = np.geomspace(l_min_optim, l_max_optim, num_pts_needle_auto)
            l_vec_optim_np_needle_auto = l_vec_optim_np_needle_auto[(l_vec_optim_np_needle_auto > 0) & np.isfinite(l_vec_optim_np_needle_auto)]
            if not l_vec_optim_np_needle_auto.size:
                log_callback("  ERREUR: impossible de générer un lambda pour la phase aiguille. Cycle annulé."); break

            ep_after_needles, rmse_after_needles, needle_logs = \
                _run_needle_iterations(best_ep_so_far, AUTO_NEEDLES_PER_CYCLE, validated_inputs, target_info,
                                       MIN_THICKNESS_PHYS_NM, l_vec_optim_np_needle_auto,
                                       DEFAULT_NEEDLE_SCAN_STEP_NM, BASE_NEEDLE_THICKNESS_NM,
                                       excel_file_path, use_backside_correction,
                                       log_prefix=f"    [Aiguille {cycle_num+1}] ")
            log_callback(needle_logs)
            log_callback(f"  [Cycle {cycle_num+1}] Fin de la Phase Aiguille. RMSE: {rmse_after_needles:.6e}")
            if rmse_after_needles < best_rmse_so_far - COST_IMPROVEMENT_TOLERANCE:
                log_callback(f"    Amélioration globale par la phase aiguille (vs {best_rmse_so_far:.6e}).")
                best_ep_so_far, best_rmse_so_far = ep_after_needles.copy(), rmse_after_needles
                cycle_improved_globally = True
            else:
                log_callback(f"    Pas d'amélioration globale par la phase aiguille (vs {best_rmse_so_far:.6e}).")
                best_ep_so_far, best_rmse_so_far = ep_after_needles.copy(), rmse_after_needles

            log_callback(f"  [Cycle {cycle_num+1}] Phase d'Amincissement (< {threshold_for_thin_removal:.3f} nm) + Ré-Opt...")
            st.write(f"Cycle {cycle_num + 1}: Phase d'Amincissement...")
            layers_removed_this_cycle = 0;
            max_thinning_attempts = len(best_ep_so_far) + 2
            for attempt in range(max_thinning_attempts):
                if len(best_ep_so_far) <= 2:
                    log_callback("    Structure trop petite (< 3 couches), arrêt de l'amincissement."); break

                ep_after_single_removal, structure_changed, removal_logs = \
                    _perform_layer_merge_or_removal_only(best_ep_so_far, MIN_THICKNESS_PHYS_NM,
                                                         log_prefix=f"    [Amincissement {cycle_num+1}.{attempt+1}] ",
                                                         threshold_for_removal=threshold_for_thin_removal)
                log_callback(removal_logs)
                if structure_changed and ep_after_single_removal is not None:
                    layers_removed_this_cycle += 1
                    log_callback(f"    Couche supprimée/fusionnée ({layers_removed_this_cycle} dans ce cycle). Ré-optimisation ({len(ep_after_single_removal)} couches)...")
                    st.write(f"Cycle {cycle_num + 1}: Ré-opt après suppression {layers_removed_this_cycle}...")
                    ep_after_thin_reopt, thin_reopt_success, rmse_after_thin_reopt, thin_reopt_logs, thin_reopt_msg = \
                        _run_core_optimization(ep_after_single_removal, validated_inputs, target_info,
                                               MIN_THICKNESS_PHYS_NM, use_backside_correction,
                                               log_prefix=f"      [ReOptAminci {cycle_num+1}.{attempt+1}] ")
                    log_callback(thin_reopt_logs)
                    if thin_reopt_success:
                        log_callback(f"      Ré-optimisation réussie. RMSE: {rmse_after_thin_reopt:.6e}")
                        if rmse_after_thin_reopt < best_rmse_so_far - COST_IMPROVEMENT_TOLERANCE:
                            log_callback(f"      Amélioration globale par amincissement+réopt (vs {best_rmse_so_far:.6e}).")
                            best_ep_so_far, best_rmse_so_far = ep_after_thin_reopt.copy(), rmse_after_thin_reopt
                            cycle_improved_globally = True
                        else:
                            log_callback(f"      Pas d'amélioration globale (vs {best_rmse_so_far:.6e}). Continuation avec ce résultat.")
                            best_ep_so_far, best_rmse_so_far = ep_after_thin_reopt.copy(), rmse_after_thin_reopt
                    else:
                        log_callback(f"    AVERTISSEMENT: La ré-optimisation après amincissement a ÉCHOUÉ ({thin_reopt_msg}). Arrêt de l'amincissement pour ce cycle.")
                        best_ep_so_far = ep_after_single_removal.copy()
                        _, _, best_rmse_so_far, _, _ = _run_core_optimization(best_ep_so_far, validated_inputs, target_info, MIN_THICKNESS_PHYS_NM, use_backside_correction)
                        break
                else:
                    log_callback("    Plus de couches à supprimer/fusionner dans cette phase."); break

            log_callback(f"  [Cycle {cycle_num+1}] Fin de la Phase d'Amincissement. {layers_removed_this_cycle} couche(s) supprimée(s).")
            num_cycles_done += 1
            log_callback(f"--- Fin du Cycle Auto {cycle_num + 1} --- Meilleur RMSE actuel: {best_rmse_so_far:.6e} ({len(best_ep_so_far)} couches) ---")
            if not cycle_improved_globally and best_rmse_so_far >= rmse_at_cycle_start - COST_IMPROVEMENT_TOLERANCE :
                log_callback(f"Pas d'amélioration significative dans le Cycle {cycle_num + 1} (Début: {rmse_at_cycle_start:.6e}, Fin: {best_rmse_so_far:.6e}). Arrêt du Mode Auto.")
                termination_reason = f"Pas d'amélioration (Cycle {cycle_num + 1})"
                if best_rmse_so_far > rmse_at_cycle_start + COST_IMPROVEMENT_TOLERANCE :
                    log_callback("  Le RMSE a augmenté, retour à l'état d'avant ce cycle.")
                    best_ep_so_far, best_rmse_so_far = ep_at_cycle_start.copy(), rmse_at_cycle_start
                break

        log_callback(f"\n--- Mode Auto Terminé après {num_cycles_done} cycles ---")
        log_callback(f"Raison: {termination_reason}")
        log_callback(f"Meilleur RMSE final: {best_rmse_so_far:.6e} avec {len(best_ep_so_far)} couches.")
        return best_ep_so_far, best_rmse_so_far, logs

    except (ValueError, RuntimeError, TypeError) as e:
        log_callback(f"ERREUR FATALE pendant le Mode Auto (Setup/Workflow): {e}")
        st.error(f"Erreur Mode Auto: {e}")
        return None, np.inf, logs
    except Exception as e_fatal:
        log_callback(f"ERREUR FATALE inattendue pendant le Mode Auto: {type(e_fatal).__name__}: {e_fatal}")
        st.error(f"Erreur inattendue en Mode Auto: {e_fatal}")
        traceback.print_exc()
        return None, np.inf, logs

# --- QWOT Scan Optimization ---
@jax.jit
def calculate_M_for_thickness(thickness: jnp.ndarray, n_complex_layer: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    """Calculates characteristic matrix for a single layer of given thickness."""
    eta = n_complex_layer
    safe_l_val = jnp.maximum(l_val, 1e-9)
    safe_eta = jnp.where(jnp.abs(eta) < 1e-12, 1e-12 + 0j, eta)
    phi = (2 * jnp.pi / safe_l_val) * (n_complex_layer * thickness)
    cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
    m00 = jnp.where(thickness > 1e-12, cos_phi, 1.0)
    m01 = jnp.where(thickness > 1e-12, (1j / safe_eta) * sin_phi, 0.0)
    m10 = jnp.where(thickness > 1e-12, 1j * eta * sin_phi, 0.0)
    m11 = jnp.where(thickness > 1e-12, cos_phi, 1.0)
    return jnp.array([[m00, m01], [m10, m11]], dtype=jnp.complex128)

calculate_M_batch_for_thickness = vmap(calculate_M_for_thickness, in_axes=(None, None, 0))

@jax.jit
def get_layer_matrices_qwot(layer_idx: int, initial_layer_number: int, l0: float,
                            nH_c_l0: jnp.ndarray, nL_c_l0: jnp.ndarray,
                            l_vec: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pre-calculates the matrices for 1x and 2x QWOT for a given layer type."""
    predicate_is_H = (layer_idx % 2 == 0)
    n_real_l0 = jax.lax.select(predicate_is_H, nH_c_l0.real, nL_c_l0.real)
    n_complex_for_matrix = jax.lax.select(predicate_is_H, nH_c_l0, nL_c_l0)
    denom = 4.0 * jnp.maximum(n_real_l0, 1e-9)
    safe_l0 = jnp.maximum(l0, 1e-9)
    ep1_calc, ep2_calc = 1.0 * safe_l0 / denom, 2.0 * safe_l0 / denom
    ep1 = jnp.where(n_real_l0 > 1e-9, ep1_calc, 0.0)
    ep2 = jnp.where(n_real_l0 > 1e-9, ep2_calc, 0.0)
    M_1qwot_batch = calculate_M_batch_for_thickness(ep1, n_complex_for_matrix, l_vec)
    M_2qwot_batch = calculate_M_batch_for_thickness(ep2, n_complex_for_matrix, l_vec)
    return M_1qwot_batch, M_2qwot_batch

@jax.jit
def compute_half_product(multiplier_indices: jnp.ndarray,
                         layer_matrices_half: jnp.ndarray
                         ) -> jnp.ndarray:
    """Computes the matrix product for one half of the stack for a given combination of QWOT multipliers."""
    N_half, _, L = layer_matrices_half.shape[0], layer_matrices_half.shape[1], layer_matrices_half.shape[2]
    init_prod = jnp.tile(jnp.eye(2, dtype=jnp.complex128), (L, 1, 1))
    def multiply_step(carry_prod: jnp.ndarray, layer_idx: int) -> Tuple[jnp.ndarray, None]:
        M_k = layer_matrices_half[layer_idx, multiplier_indices[layer_idx], :, :, :]
        return vmap(jnp.matmul)(M_k, carry_prod), None
    final_prod, _ = jax.lax.scan(multiply_step, init_prod, jnp.arange(N_half))
    return final_prod

@jax.jit
def get_T_from_batch_matrix(M_batch: jnp.ndarray, nSub_arr: jnp.ndarray) -> jnp.ndarray:
    """Calculates Transmittance from a batch of characteristic matrices."""
    etainc = 1.0 + 0j
    etasub_batch = nSub_arr
    m00, m01, m10, m11 = M_batch[..., 0, 0], M_batch[..., 0, 1], M_batch[..., 1, 0], M_batch[..., 1, 1]
    rs_den = (etainc * m00 + etasub_batch * m11 + etainc * etasub_batch * m01 + m10)
    rs_den_abs = jnp.abs(rs_den)
    safe_den = jnp.where(rs_den_abs < 1e-12, 1e-12 + 0j, rs_den)
    ts = (2.0 * etainc) / safe_den
    real_etasub_batch = jnp.real(etasub_batch)
    Ts_complex = (real_etasub_batch / 1.0) * (ts * jnp.conj(ts))
    Ts = jnp.real(Ts_complex)
    return jnp.where(rs_den_abs < 1e-12, 0.0, jnp.nan_to_num(Ts, nan=0.0))

@jax.jit
def calculate_rmse_basic_jax(Ts: jnp.ndarray, l_vec: jnp.ndarray,
                             targets_tuple: Tuple[Tuple[float, float, float, float], ...]) -> jnp.ndarray:
    """A basic RMSE calculation function that works with a tuple of target definitions."""
    total_squared_error, total_points_in_targets = 0.0, 0
    for i in range(len(targets_tuple)):
        l_min, l_max, t_min, t_max = targets_tuple[i]
        target_mask = (l_vec >= l_min) & (l_vec <= l_max)
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t = t_min + slope * (l_vec - l_min)
        squared_errors = (Ts - interpolated_target_t)**2
        masked_sq_error = jnp.where(target_mask, squared_errors, 0.0)
        total_squared_error += jnp.sum(masked_sq_error, axis=-1)
        total_points_in_targets += jnp.sum(target_mask)
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, jnp.inf)
    rmse = jnp.sqrt(mse)
    return jnp.nan_to_num(rmse, nan=jnp.inf, posinf=jnp.inf)

@jax.jit
def combine_and_calc_rmse(prod1: jnp.ndarray, prod2: jnp.ndarray,
                          nSub_arr_in: jnp.ndarray,
                          l_vec_in: jnp.ndarray, targets_tuple_in: Tuple) -> jnp.ndarray:
    """Combines the matrix products from two halves of the stack and calculates the final RMSE."""
    M_total = vmap(jnp.matmul)(prod2, prod1)
    Ts = get_T_from_batch_matrix(M_total, nSub_arr_in)
    return calculate_rmse_basic_jax(Ts, l_vec_in, targets_tuple_in)

def _execute_split_stack_scan(current_l0: float, initial_layer_number: int,
                              nH_c_l0: complex, nL_c_l0: complex,
                              nSub_arr_scan: jnp.ndarray,
                              l_vec_eval_sparse_jax: jnp.ndarray,
                              active_targets_tuple: Tuple,
                              log_callback: Callable) -> Tuple[float, Optional[np.ndarray], List[str]]:
    """Executes the highly optimized QWOT scan by splitting the stack in half."""
    logs = []
    num_combinations = 2**initial_layer_number
    log_callback(f"  [Scan l0={current_l0:.2f}] Test de {num_combinations:,} combinaisons QWOT (1.0x/2.0x)...")
    precompute_start_time = time.time()
    st.write(f"Scan l0={current_l0:.1f}: Pré-calcul des matrices...")
    try:
        get_layer_matrices_qwot_jit = jax.jit(get_layer_matrices_qwot)
        layer_matrices_list = [get_layer_matrices_qwot_jit(i, initial_layer_number, current_l0,
                                                           jnp.asarray(nH_c_l0), jnp.asarray(nL_c_l0),
                                                           l_vec_eval_sparse_jax) for i in range(initial_layer_number)]
        all_layer_matrices = jnp.stack([jnp.stack(mats, axis=0) for mats in layer_matrices_list], axis=0)
        all_layer_matrices.block_until_ready()
        log_callback(f"    Pré-calcul des matrices (l0={current_l0:.2f}) terminé en {time.time() - precompute_start_time:.3f}s.")
    except Exception as e_mat:
        logs.append(f"  ERREUR Pré-calcul des Matrices pour l0={current_l0:.2f}: {e_mat}")
        st.error(f"Erreur lors du pré-calcul des matrices de scan QWOT: {e_mat}")
        return np.inf, None, logs

    N, N1 = initial_layer_number, initial_layer_number // 2
    N2 = N - N1
    num_comb1, num_comb2 = 2**N1, 2**N2

    log_callback(f"    Calcul des produits partiels 1 ({num_comb1:,} combinaisons)...")
    st.write(f"Scan l0={current_l0:.1f}: Produits partiels 1...")
    half1_start_time = time.time()
    indices1 = jnp.arange(num_comb1)
    powers1 = 2**jnp.arange(N1)
    multiplier_indices1 = jnp.not_equal(indices1[:, None] & powers1, 0).astype(jnp.int32)
    matrices_half1 = all_layer_matrices[:N1]
    compute_half_product_jit = jax.jit(compute_half_product)
    partial_products1 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices1, matrices_half1)
    partial_products1.block_until_ready()
    log_callback(f"    Produits partiels 1 terminés en {time.time() - half1_start_time:.3f}s.")

    log_callback(f"    Calcul des produits partiels 2 ({num_comb2:,} combinaisons)...")
    st.write(f"Scan l0={current_l0:.1f}: Produits partiels 2...")
    half2_start_time = time.time()
    indices2 = jnp.arange(num_comb2)
    powers2 = 2**jnp.arange(N2)
    multiplier_indices2 = jnp.not_equal(indices2[:, None] & powers2, 0).astype(jnp.int32)
    matrices_half2 = all_layer_matrices[N1:]
    partial_products2 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices2, matrices_half2)
    partial_products2.block_until_ready()
    log_callback(f"    Produits partiels 2 terminés en {time.time() - half2_start_time:.3f}s.")

    log_callback(f"    Combinaison et calcul du RMSE ({num_comb1 * num_comb2:,} total)...")
    st.write(f"Scan l0={current_l0:.1f}: Combinaison & RMSE...")
    combine_start_time = time.time()
    combine_and_calc_rmse_jit = jax.jit(combine_and_calc_rmse)
    vmap_inner = vmap(combine_and_calc_rmse_jit, in_axes=(None, 0, None, None, None))
    vmap_outer = vmap(vmap_inner, in_axes=(0, None, None, None, None))
    all_rmses_nested = vmap_outer(partial_products1, partial_products2, nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple)
    all_rmses_nested.block_until_ready()
    log_callback(f"    Combinaison et RMSE terminés en {time.time() - combine_start_time:.3f}s.")

    all_rmses_flat = all_rmses_nested.reshape(-1)
    best_idx_flat = jnp.argmin(all_rmses_flat)
    current_best_rmse = float(all_rmses_flat[best_idx_flat])
    if not np.isfinite(current_best_rmse):
        logs.append(f"    Avertissement: Aucun résultat valide (RMSE fini) trouvé pour l0={current_l0:.2f}.")
        return np.inf, None, logs

    best_idx_half1, best_idx_half2 = jnp.unravel_index(best_idx_flat, (num_comb1, num_comb2))
    best_indices_h1 = multiplier_indices1[best_idx_half1]
    best_indices_h2 = multiplier_indices2[best_idx_half2]
    best_multipliers_h1 = 1.0 + best_indices_h1.astype(jnp.float64)
    best_multipliers_h2 = 1.0 + best_indices_h2.astype(jnp.float64)
    current_best_multipliers = jnp.concatenate([best_multipliers_h1, best_multipliers_h2])
    logs.append(f"    Meilleur RMSE pour scan l0={current_l0:.2f}: {current_best_rmse:.6e}")
    return current_best_rmse, np.array(current_best_multipliers), logs

# --- UI and Workflow Functions ---
def get_material_input(role: str) -> Tuple[Optional[MaterialInputType], str]:
    """Gets the material definition from the UI for a given role (H, L, Sub)."""
    sel_key, const_r_key, const_i_key = "", "", ""
    if role == 'H': sel_key, const_r_key, const_i_key = "selected_H", "nH_r", "nH_i"
    elif role == 'L': sel_key, const_r_key, const_i_key = "selected_L", "nL_r", "nL_i"
    elif role == 'Sub': sel_key, const_r_key, const_i_key = "selected_Sub", "nSub_r", None
    else:
        st.error(f"Rôle de matériau inconnu: {role}"); return None, "Erreur de Rôle"

    selection = st.session_state.get(sel_key)
    if selection == "Constant":
        n_real = st.session_state.get(const_r_key, 1.0 if role != 'Sub' else 1.5)
        n_imag = 0.0
        if const_i_key and role in ['H', 'L']: n_imag = st.session_state.get(const_i_key, 0.0)
        valid_n, valid_k = True, True
        if n_real <= 0: n_real, valid_n = 1.0, False
        if n_imag < 0: n_imag, valid_k = 0.0, False
        mat_repr = f"Constant ({n_real:.3f}{'+' if n_imag>=0 else ''}{n_imag:.3f}j)"
        if not valid_n or not valid_k: mat_repr += " (Ajusté)"
        return complex(n_real, n_imag), mat_repr
    elif isinstance(selection, str) and selection:
        return selection, selection
    else:
        st.error(f"Sélection de matériau pour '{role}' invalide ou manquante dans session_state.")
        return None, "Erreur de Sélection"

def clear_optimized_state():
    """Resets the application to its nominal, un-optimized state."""
    st.session_state.optimized_ep = None
    st.session_state.is_optimized_state = False
    st.session_state.ep_history = deque(maxlen=5)
    st.session_state.optimized_qwot_str = ""
    st.session_state.last_rmse = None

def set_optimized_as_nominal_wrapper():
    """Sets the current optimized structure as the new nominal QWOT structure."""
    if not st.session_state.get('is_optimized_state') or st.session_state.get('optimized_ep') is None:
        st.error("Aucune structure optimisée valide à définir comme nominale."); return
    try:
        l0 = st.session_state.l0
        nH_mat, _ = get_material_input('H')
        nL_mat, _ = get_material_input('L')
        if nH_mat is None or nL_mat is None:
            st.error("Impossible de récupérer les matériaux H/L pour recalculer le QWOT."); return

        optimized_qwots, logs_qwot = calculate_qwot_from_ep(st.session_state.optimized_ep, l0, nH_mat, nL_mat, EXCEL_FILE_PATH)
        if optimized_qwots is None:
            st.error("Erreur lors du recalcul du QWOT à partir de la structure optimisée."); return

        if np.any(np.isnan(optimized_qwots)):
            st.warning("Le QWOT recalculé contient des NaNs (probablement un indice invalide à l0). Le QWOT nominal n'a pas été mis à jour.")
        else:
            new_qwot_str = ",".join([f"{q:.6f}" for q in optimized_qwots])
            st.session_state.current_qwot = new_qwot_str
            st.success("Structure optimisée définie comme nouvelle Nominale (QWOT mis à jour).")
            clear_optimized_state()
    except Exception as e:
        st.error(f"Erreur inattendue lors de la définition de l'optimisé comme nominal: {e}")

def undo_remove_wrapper():
    """Reverts to the state before the last layer removal."""
    if not st.session_state.get('ep_history'):
        st.info("L'historique d'annulation est vide."); return
    try:
        last_ep = st.session_state.ep_history.pop()
        st.session_state.optimized_ep = last_ep.copy()
        st.session_state.is_optimized_state = True
        l0 = st.session_state.l0
        nH_mat, _ = get_material_input('H')
        nL_mat, _ = get_material_input('L')
        if nH_mat is not None and nL_mat is not None:
            qwots_recalc, logs_qwot = calculate_qwot_from_ep(last_ep, l0, nH_mat, nL_mat, EXCEL_FILE_PATH)
            if qwots_recalc is not None and not np.any(np.isnan(qwots_recalc)):
                st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_recalc])
            else:
                st.session_state.optimized_qwot_str = "QWOT N/A (après annulation)"
        else:
            st.session_state.optimized_qwot_str = "Erreur Matériau QWOT (après annulation)"
        st.info("État restauré. Recalcul en cours...")
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {
            'is_optimized_run': True,
            'method_name': "Optimisé (Annuler)",
            'force_ep': st.session_state.optimized_ep
            }
    except IndexError:
        st.warning("L'historique d'annulation est vide (erreur interne?).")
    except Exception as e:
        st.error(f"Erreur inattendue lors de l'annulation: {e}")
        clear_optimized_state()

def calculate_and_display_rmse_wrapper():
    """
    Calculates RMSE on demand based on the last computed results and current targets.
    Updates the session state with the new RMSE.
    """
    if 'last_calc_results' not in st.session_state or not st.session_state.last_calc_results:
        st.toast("Veuillez d'abord lancer un calcul ('Eval Nom.' ou une optimisation).", icon="📊")
        return

    target_info = get_target_data()
    if target_info is None or not target_info.get('data'):
        st.toast("Impossible de calculer le RMSE: Aucune donnée cible valide trouvée.", icon="🎯")
        st.session_state.last_rmse = None
        return

    results_data = st.session_state.last_calc_results
    res_to_use = results_data.get('res_fine')
    if res_to_use is None:
        st.toast("Erreur: Les données du dernier calcul sont invalides.", icon="🔥")
        return

    rmse, num_pts = calculate_final_rmse(res_to_use, target_info)

    st.session_state.last_rmse = rmse
    if rmse is not None and np.isfinite(rmse):
        st.toast(f"RMSE calculé: {rmse:.4e} (sur {num_pts} points)", icon="✅")
    else:
        st.toast("Impossible de calculer le RMSE. Vérifiez les cibles et la plage de calcul.", icon="⚠️")

def run_calculation_wrapper(is_optimized_run: bool, method_name: str = "", force_ep: Optional[np.ndarray] = None):
    """
    Main wrapper to run a calculation (either nominal or optimized) and update the UI.
    """
    calc_type = 'Optimisé' if is_optimized_run else 'Nominal'
    st.session_state.last_calc_results = {}
    st.session_state.last_rmse = None
    with st.spinner(f"Calcul {calc_type} en cours..."):
        try:
            target_info = get_target_data()
            if target_info is None:
                st.error("Définition de la cible invalide. Vérifiez les logs, le fichier ou les cibles manuelles et corrigez."); return

            if not target_info['data']:
                st.warning("Aucune cible active. Plage lambda par défaut utilisée (400-700nm). Le calcul du RMSE sera N/A.")
                l_min_plot, l_max_plot = 400.0, 700.0
            else:
                l_min_plot, l_max_plot = get_lambda_range(target_info)
                if l_min_plot is None or l_max_plot is None or l_max_plot < l_min_plot:
                    st.error("Impossible de déterminer une plage de longueurs d'onde valide à partir des cibles."); return

            validated_inputs = {
                'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot,
                'auto_thin_threshold': st.session_state.auto_thin_threshold,
                'l_range_deb': l_min_plot, 'l_range_fin': l_max_plot,
            }
            nH_mat, nH_repr = get_material_input('H')
            nL_mat, nL_repr = get_material_input('L')
            nSub_mat, nSub_repr = get_material_input('Sub')
            if nH_mat is None or nL_mat is None or nSub_mat is None:
                st.error("Erreur de définition du matériau. Vérifiez les sélections et/ou les fichiers Excel."); return
            validated_inputs.update({'nH_material': nH_mat, 'nL_material': nL_mat, 'nSub_material': nSub_mat})

            ep_to_calculate = None
            if force_ep is not None:
                ep_to_calculate = force_ep.copy()
            elif is_optimized_run and st.session_state.get('optimized_ep') is not None:
                ep_to_calculate = st.session_state.optimized_ep.copy()
            else:
                emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
                if not emp_list and calc_type == 'Nominal':
                    ep_to_calculate = np.array([], dtype=np.float64)
                elif not emp_list and calc_type == 'Optimisé':
                    st.error("Impossible de démarrer un calcul optimisé si le QWOT nominal est vide et qu'aucun état optimisé précédent n'existe."); return
                else:
                    ep_calc, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                    if ep_calc is None:
                        st.error("Échec du calcul des épaisseurs initiales à partir du QWOT."); return
                    ep_to_calculate = ep_calc.copy()

            st.session_state.current_ep = ep_to_calculate.copy() if ep_to_calculate is not None else None
            num_plot_points = max(501, int(np.round((l_max_plot - l_min_plot) / validated_inputs['l_step'])) * 3 + 1)
            l_vec_plot_fine_np = np.linspace(l_min_plot, l_max_plot, num_plot_points)
            l_vec_plot_fine_np = l_vec_plot_fine_np[(l_vec_plot_fine_np > 0) & np.isfinite(l_vec_plot_fine_np)]
            if not l_vec_plot_fine_np.size:
                st.error("Impossible de générer un vecteur lambda valide pour le traçage."); return

            results_fine, calc_logs = calculate_TR_from_ep_jax(
                ep_to_calculate, nH_mat, nL_mat, nSub_mat, l_vec_plot_fine_np, EXCEL_FILE_PATH
            )
            if results_fine is None:
                st.error("Le calcul principal de la transmittance a échoué."); return

            if st.session_state.get('use_backside_correction', False):
                T = results_fine['Ts']
                R = results_fine['Rs']
                nSub_arr_fine, _ = _get_nk_array_for_lambda_vec(nSub_mat, l_vec_plot_fine_np, EXCEL_FILE_PATH)
                if nSub_arr_fine is not None:
                    ns = np.real(nSub_arr_fine)
                    safe_ns_plus_1 = np.where(np.abs(ns + 1.0) < 1e-9, 1e-9, ns + 1.0)
                    RB = ((ns - 1.0) / safe_ns_plus_1)**2
                    TB = 1.0 - RB
                    denominator = 1.0 - R * RB
                    safe_denominator = np.where(np.abs(denominator) < 1e-9, 1e-9, denominator)
                    T_corrected = T * TB / safe_denominator
                    results_fine['Ts'] = np.clip(T_corrected, 0.0, 1.0)
                    method_name += " (BK)"
                else:
                    st.warning("Impossible d'appliquer la correction de la face arrière: échec de l'obtention de l'indice du substrat.")

            st.session_state.last_calc_results = {
                'res_fine': results_fine, 'method_name': method_name,
                'ep_used': ep_to_calculate.copy() if ep_to_calculate is not None else None,
                'l0_used': validated_inputs['l0'],
                'nH_used': nH_mat, 'nL_used': nL_mat, 'nSub_used': nSub_mat,
            }
            if target_info['data']:
                rmse_display, num_pts_rmse = calculate_final_rmse(results_fine, target_info)
                st.session_state.last_rmse = rmse_display

            st.session_state.is_optimized_state = is_optimized_run
            if not is_optimized_run:
                clear_optimized_state()
                st.session_state.current_ep = ep_to_calculate.copy() if ep_to_calculate is not None else None
            st.success(f"Calcul {calc_type} terminé.")
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Erreur lors du calcul {calc_type}: {e}")
        except Exception as e_fatal:
            st.error(f"Erreur inattendue lors du calcul {calc_type}: {e_fatal}")

def run_local_optimization_wrapper():
    """Wrapper for the 'Opt Local' button."""
    st.session_state.last_calc_results = {}
    st.session_state.last_rmse = None
    clear_optimized_state()
    with st.spinner("Optimisation locale en cours..."):
        try:
            target_info = get_target_data()
            if target_info is None or not target_info['data']:
                st.error("L'optimisation locale nécessite des cibles actives et valides."); return

            l_min_opt, l_max_opt = get_lambda_range(target_info)
            if l_min_opt is None:
                st.error("Impossible de déterminer la plage de longueurs d'onde pour l'optimisation."); return

            validated_inputs = {
                'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot,
                'auto_thin_threshold': st.session_state.auto_thin_threshold,
                'l_range_deb': l_min_opt, 'l_range_fin': l_max_opt,
            }
            nH_mat, nH_repr = get_material_input('H')
            nL_mat, nL_repr = get_material_input('L')
            nSub_mat, nSub_repr = get_material_input('Sub')
            if nH_mat is None or nL_mat is None or nSub_mat is None:
                st.error("Erreur de définition de matériau pour l'optimisation."); return
            validated_inputs.update({'nH_material': nH_mat, 'nL_material': nL_mat, 'nSub_material': nSub_mat})

            emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
            if not emp_list:
                st.error("QWOT nominal vide, impossible de démarrer l'optimisation locale."); return

            ep_start, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
            if ep_start is None:
                st.error("Échec du calcul de l'épaisseur initiale pour l'optimisation locale."); return

            use_backside = st.session_state.get('use_backside_correction', False)
            final_ep, success, final_cost, optim_logs, msg = \
                _run_core_optimization(ep_start, validated_inputs, target_info,
                                       MIN_THICKNESS_PHYS_NM, use_backside,
                                       log_prefix="  [Opt Local] ")

            if success and final_ep is not None:
                st.session_state.optimized_ep = final_ep.copy()
                st.session_state.current_ep = final_ep.copy()
                st.session_state.is_optimized_state = True
                st.session_state.last_rmse = final_cost
                qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                    st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                else:
                    st.session_state.optimized_qwot_str = "QWOT N/A"
                st.success(f"Optimisation locale terminée ({msg}). RMSE: {final_cost:.4e}")
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Opt Local"}
            else:
                st.error(f"L'optimisation locale a échoué: {msg}")
                clear_optimized_state()
                st.session_state.current_ep = ep_start.copy()

        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Erreur lors de l'optimisation locale: {e}")
            clear_optimized_state()
        except Exception as e_fatal:
            st.error(f"Erreur inattendue lors de l'optimisation locale: {e_fatal}")
            clear_optimized_state()

def run_scan_optimization_wrapper():
    """Wrapper for the 'Scan+Opt' button."""
    st.session_state.last_calc_results = {}
    st.session_state.last_rmse = None
    clear_optimized_state()
    with st.spinner("Scan QWOT + Double Optimisation en cours (peut être très long)..."):
        try:
            initial_layer_number_scan = len([q for q in st.session_state.current_qwot.split(',') if q.strip()])
            if 'initial_layer_number' not in st.session_state or st.session_state.initial_layer_number == 0:
                st.session_state.initial_layer_number = initial_layer_number_scan

            if st.session_state.initial_layer_number == 0:
                st.error("Le QWOT nominal est vide et le nombre initial de couches n'est pas défini / est nul."); return

            initial_layer_number_to_use = min(st.session_state.initial_layer_number, 18)
            if initial_layer_number_to_use != st.session_state.initial_layer_number:
                st.warning(f"Scan + Opt: Nombre de couches plafonné à 18 (était {st.session_state.initial_layer_number}).")
            st.session_state.initial_layer_number = initial_layer_number_to_use

            target_info = get_target_data()
            if target_info is None or not target_info['data']:
                st.error("Scan+Opt QWOT nécessite des cibles actives et valides."); return

            l_min_opt, l_max_opt = get_lambda_range(target_info)
            if l_min_opt is None:
                st.error("Impossible de déterminer la plage de longueurs d'onde pour Scan+Opt QWOT."); return

            validated_inputs = {
                'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot,
                'initial_layer_number': st.session_state.initial_layer_number,
                'auto_thin_threshold': st.session_state.auto_thin_threshold,
                'l_range_deb': l_min_opt, 'l_range_fin': l_max_opt,
            }
            nH_mat, nH_repr = get_material_input('H')
            nL_mat, nL_repr = get_material_input('L')
            nSub_mat, nSub_repr = get_material_input('Sub')
            if nH_mat is None or nL_mat is None or nSub_mat is None:
                st.error("Erreur de définition de matériau pour Scan+Opt QWOT."); return
            validated_inputs.update({'nH_material': nH_mat, 'nL_material': nL_mat, 'nSub_material': nSub_mat})

            num_pts_sparse_scan = min(max(2, int(np.round((l_max_opt - l_min_opt) / validated_inputs['l_step'])) + 1) // 2 + 1, 100)
            l_vec_eval_sparse_np = np.geomspace(l_min_opt, l_max_opt, num_pts_sparse_scan)
            l_vec_eval_sparse_np = l_vec_eval_sparse_np[(l_vec_eval_sparse_np > 0) & np.isfinite(l_vec_eval_sparse_np)]
            if not l_vec_eval_sparse_np.size: raise ValueError("Échec de la génération de lambda clairsemé pour le Scan.")
            l_vec_eval_sparse_jax = jnp.asarray(l_vec_eval_sparse_np)

            nSub_arr_scan, logs_sub_scan = _get_nk_array_for_lambda_vec(nSub_mat, l_vec_eval_sparse_jax, EXCEL_FILE_PATH)
            if nSub_arr_scan is None: raise RuntimeError("Échec de la préparation de l'indice du substrat pour le scan.")

            active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in validate_targets() or [])

            l0_nominal = validated_inputs['l0']
            l0_values_to_test = sorted(list(set([l0_nominal, l0_nominal * 1.2, l0_nominal * 0.8])))
            l0_values_to_test = [l for l in l0_values_to_test if l > 1e-6]

            scan_candidates = []
            st.write(f"Phase 1: Scan des combinaisons QWOT pour l0 = {l0_values_to_test}...")
            for l0_scan in l0_values_to_test:
                st.write(f"Scan pour l0={l0_scan:.1f}...")
                try:
                    nH_c_l0, log_h_l0 = _get_nk_at_lambda(nH_mat, l0_scan, EXCEL_FILE_PATH)
                    nL_c_l0, log_l_l0 = _get_nk_at_lambda(nL_mat, l0_scan, EXCEL_FILE_PATH)
                    if nH_c_l0 is None or nL_c_l0 is None:
                        st.warning(f"Impossible d'obtenir les indices à l0={l0_scan:.1f}, cette valeur est ignorée."); continue

                    scan_rmse, scan_multipliers, scan_logs = _execute_split_stack_scan(
                        l0_scan, validated_inputs['initial_layer_number'],
                        nH_c_l0, nL_c_l0, nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple, add_log
                    )
                    if scan_multipliers is not None and np.isfinite(scan_rmse):
                        scan_candidates.append({'l0': l0_scan, 'rmse_scan': scan_rmse, 'multipliers': scan_multipliers})
                        st.write(f"-> Meilleur résultat du scan pour l0={l0_scan:.1f}: RMSE = {scan_rmse:.4e}")
                    else:
                        st.warning(f"Le scan pour l0={l0_scan:.1f} n'a pas donné de résultat valide.")
                except Exception as e_scan_l0:
                    st.warning(f"Erreur lors du scan pour l0={l0_scan:.2f}: {e_scan_l0}")

            if not scan_candidates:
                st.error("Le Scan QWOT n'a trouvé aucun candidat initial valide après avoir scanné toutes les valeurs de l0."); return

            st.write(f"\nPhase 2: Double Optimisation pour {len(scan_candidates)} candidat(s)...")
            optimization_results = []
            use_backside = st.session_state.get('use_backside_correction', False)

            for idx, candidate in enumerate(scan_candidates):
                l0_cand, multipliers_cand = candidate['l0'], candidate['multipliers']
                st.write(f"-- Optimisation du Candidat {idx+1}/{len(scan_candidates)} (de l0={l0_cand:.1f}, RMSE scan={candidate['rmse_scan']:.4e}) --")
                ep_start_optim, logs_ep_best = calculate_initial_ep(multipliers_cand, l0_cand, nH_mat, nL_mat, EXCEL_FILE_PATH)
                if ep_start_optim is None:
                    st.warning(f"Échec du calcul de l'épaisseur pour le candidat {idx+1}. Ignoré."); continue

                st.write("    Exécution de l'Optimisation 1/2...")
                final_ep_1, success_1, final_cost_1, _, msg_1 = \
                    _run_core_optimization(ep_start_optim, validated_inputs, target_info,
                                           MIN_THICKNESS_PHYS_NM, use_backside, log_prefix=f"  [OptScan {idx+1}-1] ")
                if success_1 and final_ep_1 is not None:
                    st.write(f"    Optimisation 1/2 terminée ({msg_1}). RMSE: {final_cost_1:.4e}. Exécution de l'Optimisation 2/2...")
                    final_ep_2, success_2, final_cost_2, _, msg_2 = \
                        _run_core_optimization(final_ep_1, validated_inputs, target_info,
                                               MIN_THICKNESS_PHYS_NM, use_backside, log_prefix=f"  [OptScan {idx+1}-2] ")
                    if success_2 and final_ep_2 is not None:
                        st.write(f"    Optimisation 2/2 terminée ({msg_2}). RMSE Final: {final_cost_2:.4e}")
                        optimization_results.append({
                            'l0_origin': l0_cand, 'final_ep': final_ep_2, 'final_rmse': final_cost_2,
                            'message': f"Opt1: {msg_1} | Opt2: {msg_2}"
                        })
                    else:
                        st.warning(f"    Optimisation 2/2 ÉCHOUÉE pour le candidat {idx+1}: {msg_2}. Rejet de ce candidat.")
                else:
                    st.warning(f"    Optimisation 1/2 ÉCHOUÉE pour le candidat {idx+1}: {msg_1}. Saut de la deuxième optimisation.")

            if not optimization_results:
                st.error("Scan + Opt: Aucun candidat n'a terminé avec succès la double optimisation."); clear_optimized_state(); return

            st.write("\nPhase 3: Sélection du Meilleur Résultat Global...")
            best_overall_result = min(optimization_results, key=lambda r: r['final_rmse'])
            final_ep, final_cost, final_l0, final_msg = best_overall_result['final_ep'], best_overall_result['final_rmse'], best_overall_result['l0_origin'], best_overall_result['message']

            st.session_state.optimized_ep = final_ep.copy()
            st.session_state.current_ep = final_ep.copy()
            st.session_state.is_optimized_state = True
            st.session_state.last_rmse = final_cost
            st.session_state.l0 = final_l0
            qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, final_l0, nH_mat, nL_mat, EXCEL_FILE_PATH)
            if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
            else:
                st.session_state.optimized_qwot_str = "QWOT N/A"

            st.success(f"Scan + Double Optimisation terminé. Meilleur résultat de l0={final_l0:.1f}. RMSE Final: {final_cost:.4e}")
            st.caption(f"Détails de l'optimisation: {final_msg}")
            st.session_state.needs_rerun_calc = True
            st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Scan+Opt*2 (l0={final_l0:.1f})"}
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Erreur pendant le Scan + Double Optimisation QWOT: {e}"); clear_optimized_state()
        except Exception as e_fatal:
            st.error(f"Erreur inattendue pendant le Scan + Double Optimisation QWOT: {e_fatal}"); clear_optimized_state()

def run_auto_mode_wrapper():
    """Wrapper for the 'Auto' button."""
    st.session_state.last_calc_results = {}
    st.session_state.last_rmse = None
    with st.spinner("Mode Automatique en cours (peut être très long)..."):
        try:
            target_info = get_target_data()
            if target_info is None or not target_info['data']:
                st.error("Le Mode Auto nécessite des cibles actives et valides."); return

            l_min_opt, l_max_opt = get_lambda_range(target_info)
            if l_min_opt is None:
                st.error("Impossible de déterminer la plage de longueurs d'onde pour le Mode Auto."); return

            validated_inputs = {
                'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot,
                'auto_thin_threshold': st.session_state.auto_thin_threshold,
                'l_range_deb': l_min_opt, 'l_range_fin': l_max_opt,
            }
            nH_mat, nH_repr = get_material_input('H')
            nL_mat, nL_repr = get_material_input('L')
            nSub_mat, nSub_repr = get_material_input('Sub')
            if nH_mat is None or nL_mat is None or nSub_mat is None:
                st.error("Erreur de définition de matériau pour le Mode Auto."); return
            validated_inputs.update({'nH_material': nH_mat, 'nL_material': nL_mat, 'nSub_material': nSub_mat})

            ep_start_auto = st.session_state.optimized_ep.copy() if st.session_state.get('is_optimized_state') and st.session_state.get('optimized_ep') is not None else None
            use_backside = st.session_state.get('use_backside_correction', False)
            final_ep, final_rmse, auto_logs = run_auto_mode(
                initial_ep=ep_start_auto, validated_inputs=validated_inputs,
                target_info=target_info, excel_file_path=EXCEL_FILE_PATH,
                use_backside_correction=use_backside, log_callback=add_log
            )

            if final_ep is not None and np.isfinite(final_rmse):
                st.session_state.optimized_ep = final_ep.copy()
                st.session_state.current_ep = final_ep.copy()
                st.session_state.is_optimized_state = True
                st.session_state.last_rmse = final_rmse
                qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                    st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                else: st.session_state.optimized_qwot_str = "QWOT N/A"
                st.success(f"Mode Auto terminé. RMSE Final: {final_rmse:.4e}")
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': "Mode Auto"}
            else:
                st.error("Le Mode Automatique a échoué ou n'a pas produit de résultat valide.")
                clear_optimized_state()
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (Après échec Auto)"}
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Erreur pendant le Mode Auto: {e}")
        except Exception as e_fatal:
            st.error(f"Erreur inattendue pendant le Mode Auto: {e_fatal}")

def run_remove_thin_wrapper():
    """Wrapper for the 'Thin+ReOpt' button."""
    st.session_state.last_calc_results = {}
    st.session_state.last_rmse = None
    ep_start_removal = None
    if st.session_state.get('is_optimized_state') and st.session_state.get('optimized_ep') is not None:
        ep_start_removal = st.session_state.optimized_ep.copy()
    else:
        try:
            nH_mat_temp, _ = get_material_input('H')
            nL_mat_temp, _ = get_material_input('L')
            if nH_mat_temp is None or nL_mat_temp is None:
                st.error("Impossible de calculer la structure nominale: Erreur de définition du matériau."); return
            emp_list_temp = [float(e.strip()) for e in st.session_state.current_qwot.split(',') if e.strip()]
            if not emp_list_temp:
                ep_start_removal = np.array([], dtype=np.float64)
            else:
                ep_start_removal, logs_ep_init = calculate_initial_ep(
                    emp_list_temp, st.session_state.l0, nH_mat_temp, nL_mat_temp, EXCEL_FILE_PATH
                )
                if ep_start_removal is None:
                    st.error("Échec du calcul de la structure nominale à partir du QWOT pour la suppression."); return
            st.session_state.current_ep = ep_start_removal.copy()
        except Exception as e_nom:
            st.error(f"Erreur lors du calcul de la structure nominale pour la suppression: {e_nom}"); return

    if ep_start_removal is None:
        st.error("Impossible de déterminer une structure de départ valide pour la suppression."); return
    if len(ep_start_removal) <= 2:
        st.error("Structure trop petite (<= 2 couches) pour la suppression/fusion."); return

    with st.spinner("Suppression de la couche fine + Ré-optimisation..."):
        try:
            st.session_state.ep_history.append(ep_start_removal.copy())
            target_info = get_target_data()
            if target_info is None or not target_info['data']:
                st.session_state.ep_history.pop(); st.error("Suppression annulée: cibles invalides ou manquantes pour la ré-optimisation."); return

            l_min_opt, l_max_opt = get_lambda_range(target_info)
            if l_min_opt is None:
                st.session_state.ep_history.pop(); st.error("Suppression annulée: plage de longueurs d'onde invalide pour la ré-optimisation."); return

            validated_inputs = {
                'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot,
                'auto_thin_threshold': st.session_state.auto_thin_threshold,
                'l_range_deb': l_min_opt, 'l_range_fin': l_max_opt,
            }
            nH_mat, nH_repr = get_material_input('H')
            nL_mat, nL_repr = get_material_input('L')
            nSub_mat, nSub_repr = get_material_input('Sub')
            if nH_mat is None or nL_mat is None or nSub_mat is None:
                st.session_state.ep_history.pop(); st.error("Suppression annulée: erreur de définition de matériau."); return
            validated_inputs.update({'nH_material': nH_mat, 'nL_material': nL_mat, 'nSub_material': nSub_mat})

            ep_after_removal, structure_changed, removal_logs = _perform_layer_merge_or_removal_only(
                ep_start_removal, MIN_THICKNESS_PHYS_NM, log_prefix="  [Suppression] ", threshold_for_removal=None
            )

            if structure_changed and ep_after_removal is not None:
                st.write("Ré-optimisation après suppression...")
                use_backside = st.session_state.get('use_backside_correction', False)
                final_ep, success, final_cost, optim_logs, msg = \
                    _run_core_optimization(ep_after_removal, validated_inputs, target_info,
                                           MIN_THICKNESS_PHYS_NM, use_backside, log_prefix="  [ReOpt Aminci] ")
                if success and final_ep is not None:
                    st.session_state.optimized_ep = final_ep.copy()
                    st.session_state.current_ep = final_ep.copy()
                    st.session_state.is_optimized_state = True
                    st.session_state.last_rmse = final_cost
                    qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                    if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                        st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                    else: st.session_state.optimized_qwot_str = "QWOT N/A"
                    st.success(f"Suppression + Ré-optimisation terminée ({msg}). RMSE Final: {final_cost:.4e}")
                    st.session_state.needs_rerun_calc = True
                    st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': "Optimisé (Post-Suppression)"}
                else:
                    st.warning(f"Couche supprimée, mais la ré-optimisation a échoué ({msg}). L'état est APRÈS la suppression mais AVANT la tentative de ré-optimisation échouée.")
                    st.session_state.optimized_ep = ep_after_removal.copy()
                    st.session_state.current_ep = ep_after_removal.copy()
                    st.session_state.is_optimized_state = True
                    try:
                        _, _, fail_rmse, _, _ = _run_core_optimization(ep_after_removal, validated_inputs, target_info, MIN_THICKNESS_PHYS_NM, use_backside)
                        st.session_state.last_rmse = fail_rmse
                        qwots_fail, _ = calculate_qwot_from_ep(ep_after_removal, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                        if qwots_fail is not None and not np.any(np.isnan(qwots_fail)):
                            st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_fail])
                        else: st.session_state.optimized_qwot_str = "QWOT N/A (Échec ReOpt)"
                    except Exception as e_recalc:
                        st.session_state.last_rmse = None; st.session_state.optimized_qwot_str = "Erreur Recalcul"
                    st.session_state.needs_rerun_calc = True
                    st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': "Optimisé (Post-Suppression, Échec Re-Opt)"}
            else:
                st.info("Aucune couche n'a été supprimée (critères non remplis).")
                try: st.session_state.ep_history.pop()
                except IndexError: pass
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Erreur lors de la suppression de la couche fine: {e}")
            try: st.session_state.ep_history.pop()
            except IndexError: pass
        except Exception as e_fatal:
            st.error(f"Erreur inattendue lors de la suppression de la couche fine: {e_fatal}")
            try: st.session_state.ep_history.pop()
            except IndexError: pass

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Optimiseur de Couches Minces (Streamlit)")

if 'init_done' not in st.session_state:
    st.session_state.log_messages = ["[Initialisation] Bienvenue dans l'optimiseur Streamlit."]
    st.session_state.current_ep = None
    st.session_state.current_qwot = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    st.session_state.optimized_ep = None
    st.session_state.is_optimized_state = False
    st.session_state.optimized_qwot_str = ""
    st.session_state.material_sequence = None
    st.session_state.ep_history = deque(maxlen=5)
    st.session_state.last_rmse = None
    st.session_state.needs_rerun_calc = False
    st.session_state.rerun_calc_params = {}
    try:
        mats, logs = get_available_materials_from_excel(EXCEL_FILE_PATH)
        add_log(logs)
        st.session_state.available_materials = sorted(list(set(["Constant"] + mats)))
        base_subs = ["Constant", "Fused Silica", "BK7", "D263"]
        st.session_state.available_substrates = sorted(list(set(base_subs + st.session_state.available_materials)))
    except Exception as e:
        st.error(f"Erreur initiale lors du chargement des matériaux depuis {EXCEL_FILE_PATH}: {e}")
        st.session_state.available_materials = ["Constant"]
        st.session_state.available_substrates = ["Constant"]
    st.session_state.l0 = 500.0
    st.session_state.l_step = 10.0
    st.session_state.auto_thin_threshold = 1.0
    st.session_state.selected_H = next((m for m in ["Nb2O5-Helios", "Constant"] if m in st.session_state.available_materials), "Constant")
    st.session_state.selected_L = next((m for m in ["SiO2-Helios", "Constant"] if m in st.session_state.available_materials), "Constant")
    st.session_state.selected_Sub = next((m for m in ["Fused Silica", "Constant"] if m in st.session_state.available_substrates), "Constant")
    st.session_state.targets = [
        {'enabled': True, 'min': 400.0, 'max': 500.0, 'target_min': 1.0, 'target_max': 1.0},
        {'enabled': True, 'min': 500.0, 'max': 600.0, 'target_min': 1.0, 'target_max': 0.2},
        {'enabled': True, 'min': 600.0, 'max': 700.0, 'target_min': 0.2, 'target_max': 0.2},
        {'enabled': False, 'min': 700.0, 'max': 800.0, 'target_min': 0.0, 'target_max': 0.0},
        {'enabled': False, 'min': 800.0, 'max': 900.0, 'target_min': 0.0, 'target_max': 0.0},
    ]
    st.session_state.nH_r, st.session_state.nH_i = 2.35, 0.0
    st.session_state.nL_r, st.session_state.nL_i = 1.46, 0.0
    st.session_state.nSub_r = 1.52
    st.session_state.init_done = True
    st.session_state.needs_rerun_calc = True
    st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Chargement Initial"}

def trigger_nominal_recalc():
    """Callback to trigger a recalculation when a parameter changes."""
    if not st.session_state.get('calculating', False):
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {
            'is_optimized_run': False,
            'method_name': "Nominal (Mise à jour Param.)",
            'force_ep': None
        }

def on_nominal_structure_change():
    """
    Callback for when a fundamental nominal parameter (like the QWOT string) changes.
    It resets any optimized state and triggers a full recalculation of the new nominal design.
    """
    clear_optimized_state()
    trigger_nominal_recalc()

st.title("🔬 Optimiseur de Couches Minces (Streamlit + JAX)")
menu_cols = st.columns(9)
with menu_cols[0]:
    if st.button("📊 Eval Nom.", key="eval_nom_top", help="Évaluer la Structure Nominale"):
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (Évalué)"}
        st.rerun()
with menu_cols[1]:
    if st.button("✨ Opt Local", key="optim_local_top", help="Optimisation Locale"):
        run_local_optimization_wrapper()
with menu_cols[2]:
    if st.button("🚀 Scan+Opt", key="optim_scan_top", help="Scan Initial + Optimisation (Max 18 couches pour le scan, Max 100 pts pour le RMSE)"):
        run_scan_optimization_wrapper()
with menu_cols[3]:
    if st.button("🤖 Auto", key="optim_auto_top", help="Mode Auto (Aiguille > Amincissement > Opt) (Max 100 pts pour le RMSE)"):
        run_auto_mode_wrapper()
with menu_cols[4]:
    current_structure_for_check = st.session_state.get('optimized_ep') if st.session_state.get('is_optimized_state') else st.session_state.get('current_ep')
    can_remove_structurally = current_structure_for_check is not None and len(current_structure_for_check) > 2
    if st.button("🗑️ Amincir+ReOpt", key="remove_thin_top", help="Supprimer la couche la plus fine + Ré-optimiser", disabled=not can_remove_structurally):
        run_remove_thin_wrapper()
with menu_cols[5]:
    can_optimize_top = st.session_state.get('is_optimized_state', False) and st.session_state.get('optimized_ep') is not None
    if st.button("💾 Opt->Nom", key="set_optim_as_nom_top", help="Définir l'Optimisé comme Nominal", disabled=not can_optimize_top):
        set_optimized_as_nominal_wrapper()
        st.rerun()
with menu_cols[6]:
    can_undo_top = bool(st.session_state.get('ep_history'))
    if st.button(f"↩️ Annuler ({len(st.session_state.get('ep_history', deque()))})", key="undo_remove_top", help="Annuler la dernière suppression", disabled=not can_undo_top):
        undo_remove_wrapper()
        st.rerun()
with menu_cols[7]:
    if st.button("🔄 Recharger Mat.", key="reload_mats_top", help="Recharger les Matériaux depuis Excel"):
        st.cache_data.clear()
        try:
            mats, logs = get_available_materials_from_excel(EXCEL_FILE_PATH)
            st.session_state.available_materials = sorted(list(set(["Constant"] + mats)))
            base_subs = ["Constant", "Fused Silica", "BK7", "D263"]
            st.session_state.available_substrates = sorted(list(set(base_subs + st.session_state.available_materials)))
            if st.session_state.selected_H not in st.session_state.available_materials: st.session_state.selected_H = "Constant"
            if st.session_state.selected_L not in st.session_state.available_materials: st.session_state.selected_L = "Constant"
            if st.session_state.selected_Sub not in st.session_state.available_substrates: st.session_state.selected_Sub = "Constant"
            st.rerun()
        except Exception as e:
            st.error(f"Erreur lors du rechargement des matériaux: {e}")
with menu_cols[8]:
    can_calc_rmse = 'last_calc_results' in st.session_state and st.session_state.last_calc_results
    if st.button("Calc. RMSE", key="calc_rmse_top", help="Recalculer le RMSE entre la courbe actuelle et la cible", disabled=not can_calc_rmse):
        calculate_and_display_rmse_wrapper()
        st.rerun()

st.divider()

main_layout = st.columns([1, 3])
with main_layout[0]:
    st.subheader("Matériaux")
    st.session_state.selected_H = st.selectbox("Matériau H", options=st.session_state.available_materials, index=st.session_state.available_materials.index(st.session_state.selected_H), key="sb_H", on_change=trigger_nominal_recalc)
    if st.session_state.selected_H == "Constant":
        hc1, hc2 = st.columns(2)
        st.session_state.nH_r = hc1.number_input("n' H", value=st.session_state.nH_r, format="%.4f", key="nH_r_const", on_change=trigger_nominal_recalc)
        st.session_state.nH_i = hc2.number_input("k H", value=st.session_state.nH_i, min_value=0.0, format="%.4f", key="nH_i_const", on_change=trigger_nominal_recalc)

    st.session_state.selected_L = st.selectbox("Matériau L", options=st.session_state.available_materials, index=st.session_state.available_materials.index(st.session_state.selected_L), key="sb_L", on_change=trigger_nominal_recalc)
    if st.session_state.selected_L == "Constant":
        lc1, lc2 = st.columns(2)
        st.session_state.nL_r = lc1.number_input("n' L", value=st.session_state.nL_r, format="%.4f", key="nL_r_const", on_change=trigger_nominal_recalc)
        st.session_state.nL_i = lc2.number_input("k L", value=st.session_state.nL_i, min_value=0.0, format="%.4f", key="nL_i_const", on_change=trigger_nominal_recalc)

    st.session_state.selected_Sub = st.selectbox("Matériau Substrat", options=st.session_state.available_substrates, index=st.session_state.available_substrates.index(st.session_state.selected_Sub), key="sb_Sub", on_change=trigger_nominal_recalc)
    if st.session_state.selected_Sub == "Constant":
        st.session_state.nSub_r = st.number_input("n' Substrat", value=st.session_state.nSub_r, format="%.4f", key="nSub_const", on_change=trigger_nominal_recalc)

    st.subheader("Structure Nominale")
    st.session_state.current_qwot = st.text_area("Multiplicateurs QWOT (séparés par des virgules)", value=st.session_state.current_qwot, key="qwot_input", on_change=on_nominal_structure_change, height=100)
    num_layers_from_qwot = len([q for q in st.session_state.current_qwot.split(',') if q.strip()])
    qwot_cols = st.columns([3,2])
    with qwot_cols[0]:
        st.session_state.l0 = st.number_input("Longueur d'onde de référence λ₀ (nm) pour QWOT", value=st.session_state.l0, min_value=1.0, format="%.2f", key="l0_input", on_change=trigger_nominal_recalc)
    with qwot_cols[1]:
        init_layers_num = st.number_input("Nombre de couches à générer", min_value=0, value=num_layers_from_qwot, step=1, key="init_layers_gen_num")
        if st.button("Gén. QWOT 1.0", key="gen_qwot_btn_main", use_container_width=True, help="Générer une chaîne QWOT avec le nombre de couches spécifié, toutes à 1.0"):
            if init_layers_num > 0:
                new_qwot = ",".join(['1'] * init_layers_num)
                if new_qwot != st.session_state.current_qwot:
                    st.session_state.current_qwot = new_qwot; on_nominal_structure_change(); st.rerun()
            elif st.session_state.current_qwot != "":
                st.session_state.current_qwot = ""; on_nominal_structure_change(); st.rerun()
    st.caption(f"Couches Nominales Actuelles: {num_layers_from_qwot}")

    st.subheader("Cibles (T) & Paramètres de Calcul")
    uploaded_target_file = st.file_uploader(
        "Télécharger Fichier Cible (optionnel, .xlsx avec λ, T%)",
        type=['xlsx'], key="target_file_uploader_key", on_change=trigger_nominal_recalc,
        help="Télécharger un fichier Excel avec deux colonnes: Longueur d'onde (nm) et Transmittance Cible (%). Cela remplacera les cibles manuelles."
    )
    is_file_target = st.session_state.get("target_file_uploader_key") is not None

    st.session_state.l_step = st.number_input("Pas λ pour la grille RMSE (nm)", value=st.session_state.l_step, min_value=0.1, format="%.2f", key="l_step_input_main", on_change=trigger_nominal_recalc, help="Pas de longueur d'onde pour les points de la grille d'optimisation (max 100 points). Le tracé utilise une grille plus fine.")
    st.session_state.auto_thin_threshold = st.number_input("Seuil de suppression de couche fine auto (nm)", value=st.session_state.auto_thin_threshold, min_value=MIN_THICKNESS_PHYS_NM, format="%.3f", key="auto_thin_input_main", help="En mode Auto, les couches plus fines que cela peuvent être supprimées.")
    
    st.checkbox("Considérer la face arrière du substrat (T_corrigée = T*Tb/(1-R*Rb))",
                key="use_backside_correction",
                on_change=trigger_nominal_recalc,
                help="Applique une correction pour la réflexion incohérente de la face arrière. Cela affecte l'affichage et l'optimisation.")

    st.markdown("---")
    st.caption("Cibles Manuelles (ignorées si un fichier est téléchargé)")
    hdr_cols = st.columns([0.5, 1, 1, 1, 1])
    for c, h in zip(hdr_cols, ["On", "λmin", "λmax", "Tmin", "Tmax"]): c.caption(h)
    for i in range(len(st.session_state.targets)):
        target = st.session_state.targets[i]
        cols = st.columns([0.5, 1, 1, 1, 1])
        st.session_state.targets[i]['enabled'] = cols[0].checkbox("", value=target.get('enabled', False), key=f"target_enable_{i}_main", label_visibility="collapsed", on_change=trigger_nominal_recalc, disabled=is_file_target)
        st.session_state.targets[i]['min'] = cols[1].number_input(f"λmin Cible {i+1}", value=target.get('min', 0.0), format="%.1f", step=10.0, key=f"target_min_{i}_main", label_visibility="collapsed", on_change=trigger_nominal_recalc, disabled=is_file_target)
        st.session_state.targets[i]['max'] = cols[2].number_input(f"λmax Cible {i+1}", value=target.get('max', 0.0), format="%.1f", step=10.0, key=f"target_max_{i}_main", label_visibility="collapsed", on_change=trigger_nominal_recalc, disabled=is_file_target)
        st.session_state.targets[i]['target_min'] = cols[3].number_input(f"Tmin Cible {i+1}", value=target.get('target_min', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmin_{i}_main", label_visibility="collapsed", on_change=trigger_nominal_recalc, disabled=is_file_target)
        st.session_state.targets[i]['target_max'] = cols[4].number_input(f"Tmax Cible {i+1}", value=target.get('target_max', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmax_{i}_main", label_visibility="collapsed", on_change=trigger_nominal_recalc, disabled=is_file_target)

with main_layout[1]:
    st.subheader("Résultats")
    state_desc = "Optimisé" if st.session_state.is_optimized_state else "Nominal"
    ep_display = st.session_state.optimized_ep if st.session_state.is_optimized_state else st.session_state.current_ep
    num_layers_display = len(ep_display) if ep_display is not None else 0
    res_info_cols = st.columns(3)
    res_info_cols[0].caption(f"État: {state_desc} ({num_layers_display} couches)")
    res_info_cols[1].caption(f"RMSE: {st.session_state.last_rmse:.4e}" if st.session_state.last_rmse is not None and np.isfinite(st.session_state.last_rmse) else "RMSE: N/A")
    min_thick_str = "N/A"
    if ep_display is not None and ep_display.size > 0:
        valid_thick = ep_display[ep_display >= MIN_THICKNESS_PHYS_NM - 1e-9]
        if valid_thick.size > 0: min_thick_str = f"{np.min(valid_thick):.3f} nm"
    res_info_cols[2].caption(f"Épaisseur Min: {min_thick_str}")

    if st.session_state.is_optimized_state and st.session_state.get('optimized_qwot_str'):
        st.text_input("QWOT Optimisé (à λ₀ d'origine)", value=st.session_state.optimized_qwot_str, disabled=True, key="opt_qwot_display_main_res")

    if 'last_calc_results' in st.session_state and st.session_state.last_calc_results:
        results_data = st.session_state.last_calc_results
        res_fine_plot = results_data.get('res_fine')
        target_info_plot = get_target_data()
        rmse_plot = st.session_state.last_rmse
        method_name_plot = results_data.get('method_name', '')

        if res_fine_plot:
            fig_spec, ax_spec = plt.subplots(figsize=(12, 4))
            opt_method_str = f" ({method_name_plot})" if method_name_plot else ""
            window_title = f'Réponse Spectrale {"Optimisée" if st.session_state.is_optimized_state else "Nominale"}{opt_method_str}'
            fig_spec.suptitle(window_title, fontsize=12, weight='bold')
            line_ts = None
            try:
                if res_fine_plot and 'l' in res_fine_plot and 'Ts' in res_fine_plot and res_fine_plot['l'] is not None and len(res_fine_plot['l']) > 0:
                    res_l_plot, res_ts_plot = np.asarray(res_fine_plot['l']), np.asarray(res_fine_plot['Ts'])
                    line_ts, = ax_spec.plot(res_l_plot, res_ts_plot, label='Transmittance', linestyle='-', color='blue', linewidth=1.5)
                    plotted_target_label = False
                    if target_info_plot and target_info_plot['data']:
                        active_targets_plot = target_info_plot['data'] if target_info_plot['type'] == 'manual' else [{'min': l, 'max': l, 'target_min': t, 'target_max': t} for l, t in zip(target_info_plot['data']['lambdas'], target_info_plot['data']['transmittances'])]
                        for i, target in enumerate(active_targets_plot):
                            l_min, l_max = target['min'], target['max']
                            t_min, t_max_corr = target['target_min'], target['target_max']
                            x_coords, y_coords = [l_min, l_max], [t_min, t_max_corr]
                            label = 'Cible(s)' if not plotted_target_label else "_nolegend_"
                            ax_spec.plot(x_coords, y_coords, 'r--', linewidth=1.0, alpha=0.7, label=label, zorder=5)
                            ax_spec.plot(x_coords, y_coords, marker='x', color='red', markersize=6, linestyle='none', label='_nolegend_', zorder=6)
                            plotted_target_label = True
                ax_spec.set_xlabel("Longueur d'onde (nm)"); ax_spec.set_ylabel('Transmittance')
                ax_spec.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
                ax_spec.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')
                ax_spec.minorticks_on()
                if 'res_l_plot' in locals() and len(res_l_plot) > 0 : ax_spec.set_xlim(res_l_plot[0], res_l_plot[-1])
                ax_spec.set_ylim(-0.05, 1.05)
                if 'plotted_target_label' in locals() and (plotted_target_label or (line_ts is not None)): ax_spec.legend(fontsize=8)
                rmse_text = f"RMSE = {rmse_plot:.3e}" if rmse_plot is not None and np.isfinite(rmse_plot) else "RMSE: N/A"
                ax_spec.text(0.98, 0.98, rmse_text, transform=ax_spec.transAxes, ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
            except Exception as e_spec:
                ax_spec.text(0.5, 0.5, f"Erreur de traçage du spectre:\n{e_spec}", ha='center', va='center', transform=ax_spec.transAxes, color='red')
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            st.pyplot(fig_spec)
            plt.close(fig_spec)
        else:
            st.warning("Données de calcul manquantes ou invalides pour l'affichage du spectre.")
    else:
        st.info("Lancez une évaluation ou une optimisation pour voir les résultats.")

    plot_col1, plot_col2 = st.columns(2)
    if 'last_calc_results' in st.session_state and st.session_state.last_calc_results:
        results_data = st.session_state.last_calc_results
        ep_plot, l0_plot = results_data.get('ep_used'), results_data.get('l0_used')
        nH_plot, nL_plot, nSub_plot = results_data.get('nH_used'), results_data.get('nL_used'), results_data.get('nSub_used')
        is_optimized_plot = st.session_state.is_optimized_state
        if all(v is not None for v in [ep_plot, l0_plot, nH_plot, nL_plot, nSub_plot]):
            with plot_col1:
                fig_idx, ax_idx = plt.subplots(figsize=(6, 4))
                try:
                    nH_c_repr, _ = _get_nk_at_lambda(nH_plot, l0_plot, EXCEL_FILE_PATH)
                    nL_c_repr, _ = _get_nk_at_lambda(nL_plot, l0_plot, EXCEL_FILE_PATH)
                    nSub_c_repr, _ = _get_nk_at_lambda(nSub_plot, l0_plot, EXCEL_FILE_PATH)
                    if nH_c_repr is None or nL_c_repr is None or nSub_c_repr is None: raise ValueError("Indices à l0 non trouvés pour le tracé du profil.")
                    nH_r_repr, nL_r_repr, nSub_r_repr = nH_c_repr.real, nL_c_repr.real, nSub_c_repr.real
                    num_layers = len(ep_plot)
                    n_real_layers_repr = [nH_r_repr if i % 2 == 0 else nL_r_repr for i in range(num_layers)]

                    x_coords_plot, y_coords_plot = [], []
                    total_thickness = np.sum(ep_plot) if num_layers > 0 else 0
                    margin = max(50, 0.1 * total_thickness) if total_thickness > 0 else 50

                    x_coords_plot.extend([-margin, 0])
                    y_coords_plot.extend([nSub_r_repr, nSub_r_repr])
                    last_n = nSub_r_repr
                    current_z = 0

                    if num_layers > 0:
                        for i in range(num_layers):
                            layer_n_real = n_real_layers_repr[i]
                            x_coords_plot.extend([current_z, current_z])
                            y_coords_plot.extend([last_n, layer_n_real])
                            current_z += ep_plot[i]
                            x_coords_plot.append(current_z)
                            y_coords_plot.append(layer_n_real)
                            last_n = layer_n_real

                    x_coords_plot.extend([current_z, current_z])
                    y_coords_plot.extend([last_n, 1.0])
                    x_coords_plot.append(current_z + margin)
                    y_coords_plot.append(1.0)

                    ax_idx.plot(x_coords_plot, y_coords_plot, label=f'n\'(λ={l0_plot:.0f}nm)', color='purple', linewidth=1.5)

                    ax_idx.set_xlabel('Profondeur (depuis le substrat) (nm)')
                    ax_idx.set_ylabel("Partie réelle de l'indice (n')")
                    ax_idx.set_title(f"Profil d'indice (à λ={l0_plot:.0f}nm)", fontsize=10)
                    ax_idx.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray'); ax_idx.minorticks_on()
                    ax_idx.set_xlim(x_coords_plot[0], x_coords_plot[-1])
                    valid_n = [n for n in [1.0, nSub_r_repr] + n_real_layers_repr if np.isfinite(n)]
                    min_n, max_n = (min(valid_n), max(valid_n)) if valid_n else (0.9, 2.5)
                    y_padding = (max_n - min_n) * 0.1 + 0.05
                    ax_idx.set_ylim(bottom=min_n - y_padding, top=max_n + y_padding)
                    if ax_idx.get_legend_handles_labels()[1]: ax_idx.legend(fontsize=8)
                except Exception as e_idx:
                    ax_idx.text(0.5, 0.5, f"Erreur de traçage du profil d'indice:\n{e_idx}", ha='center', va='center', transform=ax_idx.transAxes, color='red')
                plt.tight_layout(); st.pyplot(fig_idx); plt.close(fig_idx)

            with plot_col2:
                fig_stack, ax_stack = plt.subplots(figsize=(6, 4))
                try:
                    num_layers = len(ep_plot)
                    if num_layers > 0:
                        nH_c_repr, _ = _get_nk_at_lambda(nH_plot, l0_plot, EXCEL_FILE_PATH)
                        nL_c_repr, _ = _get_nk_at_lambda(nL_plot, l0_plot, EXCEL_FILE_PATH)
                        indices_complex_repr = [nH_c_repr if i % 2 == 0 else nL_c_repr for i in range(num_layers)]
                        layer_types = ["H" if i % 2 == 0 else "L" for i in range(num_layers)]
                        colors = ['lightblue' if i % 2 == 0 else 'lightcoral' for i in range(num_layers)]
                        bar_pos = np.arange(num_layers)
                        bars = ax_stack.barh(bar_pos, ep_plot, align='center', color=colors, edgecolor='grey', height=0.8)
                        yticks_labels = []
                        for i in range(num_layers):
                            n_comp_repr = indices_complex_repr[i] if indices_complex_repr and i < len(indices_complex_repr) else complex(0,0)
                            n_str = f"{n_comp_repr.real:.3f}" if np.isfinite(n_comp_repr.real) else "N/A"
                            if np.isfinite(n_comp_repr.imag) and abs(n_comp_repr.imag) > 1e-6: n_str += f"{n_comp_repr.imag:+.3f}j"
                            yticks_labels.append(f"C{i + 1} ({layer_types[i]}) n≈{n_str}")
                        ax_stack.set_yticks(bar_pos); ax_stack.set_yticklabels(yticks_labels, fontsize=7); ax_stack.invert_yaxis()
                        max_ep_plot_val = max(ep_plot) if ep_plot.size > 0 else 1.0
                        fontsize_bar = max(6, 9 - num_layers // 15)
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ha_pos = 'left' if width < max_ep_plot_val * 0.3 else 'right'
                            x_text_pos = width * 1.02 if ha_pos == 'left' else width * 0.98
                            text_color = 'black' if ha_pos == 'left' else 'white'
                            ax_stack.text(x_text_pos, bar.get_y() + bar.get_height()/2., f"{width:.2f}", va='center', ha=ha_pos, color=text_color, fontsize=fontsize_bar, weight='bold')
                    else:
                        ax_stack.text(0.5, 0.5, "Structure Vide", ha='center', va='center', fontsize=10, color='grey', transform=ax_stack.transAxes)
                        ax_stack.set_yticks([]); ax_stack.set_xticks([])
                    ax_stack.set_xlabel('Épaisseur (nm)')
                    ax_stack.set_title(f'Structure {"Optimisée" if is_optimized_plot else "Nominale"} ({num_layers} couches)', fontsize=10)
                    max_ep_plot_val_xlim = max(ep_plot) if num_layers > 0 and ep_plot.size > 0 else 10
                    ax_stack.set_xlim(right=max_ep_plot_val_xlim * 1.1)
                except Exception as e_stack:
                    ax_stack.text(0.5, 0.5, f"Erreur de traçage de la structure:\n{e_stack}", ha='center', va='center', transform=ax_stack.transAxes, color='red')
                plt.tight_layout(); st.pyplot(fig_stack); plt.close(fig_stack)

if st.session_state.get('needs_rerun_calc', False):
    params = st.session_state.rerun_calc_params
    st.session_state.needs_rerun_calc = False
    st.session_state.rerun_calc_params = {}
    st.session_state.calculating = True
    run_calculation_wrapper(
        is_optimized_run=params.get('is_optimized_run', False),
        method_name=params.get('method_name', 'Recalcul Auto'),
        force_ep=params.get('force_ep')
    )
    st.session_state.calculating = False
    st.rerun()
