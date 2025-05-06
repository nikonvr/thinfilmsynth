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
MIN_THICKNESS_PHYS_NM = 0.01
BASE_NEEDLE_THICKNESS_NM = 0.1
DEFAULT_NEEDLE_SCAN_STEP_NM = 2.0
AUTO_NEEDLES_PER_CYCLE = 5
AUTO_MAX_CYCLES = 5
MSE_IMPROVEMENT_TOLERANCE = 1e-9
EXCEL_FILE_PATH = "indices.xlsx"
@st.cache_data
def load_material_data_from_xlsx_sheet(file_path: str, sheet_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    logs = []
    try:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        except FileNotFoundError:
            st.error(f"Fichier Excel '{file_path}' introuvable. Vérifiez sa présence.")
            logs.append(f"Erreur critique: Fichier Excel introuvable: {file_path}")
            return None, None, None, logs
        except Exception as e:
            st.error(f"Erreur lors de la lecture Excel ('{file_path}', feuille '{sheet_name}'): {e}")
            logs.append(f"Erreur Excel inattendue ({type(e).__name__}): {e}")
            return None, None, None, logs
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        numeric_df = numeric_df.dropna(how='all')
        if numeric_df.shape[1] >= 3:
            cols_to_check = numeric_df.columns[:3]
            numeric_df = numeric_df.dropna(subset=cols_to_check)
        else:
            logs.append(f"Avertissement: Feuille '{sheet_name}' ne contient pas 3 colonnes numériques.")
            return np.array([]), np.array([]), np.array([]), logs
        if numeric_df.empty:
             logs.append(f"Avertissement: Aucune donnée numérique valide trouvée dans '{sheet_name}' après nettoyage.")
             return np.array([]), np.array([]), np.array([]), logs
        try:
             numeric_df = numeric_df.sort_values(by=numeric_df.columns[0])
        except IndexError:
             logs.append(f"Erreur: Impossible de trier les données pour la feuille '{sheet_name}'. Colonne d'index 0 manquante?")
             return np.array([]), np.array([]), np.array([]), logs
        l_nm = numeric_df.iloc[:, 0].values.astype(np.float64)
        n = numeric_df.iloc[:, 1].values.astype(np.float64)
        k = numeric_df.iloc[:, 2].values.astype(np.float64)
        if np.any(k < -1e-9):
             invalid_k_indices = np.where(k < -1e-9)[0]
             logs.append(f"AVERTISSEMENT: Valeurs k négatives détectées et mises à 0 pour '{sheet_name}' aux indices: {invalid_k_indices.tolist()}")
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
        logs.append(f"Erreur inattendue lecture Excel ('{sheet_name}'): {type(e).__name__} - {e}")
        return None, None, None, logs
def get_available_materials_from_excel(excel_path: str) -> Tuple[List[str], List[str]]:
    logs = []
    try:
        xl = pd.ExcelFile(excel_path)
        sheet_names = [name for name in xl.sheet_names if not name.startswith("Sheet")]
        logs.append(f"Matériaux trouvés dans {excel_path}: {sheet_names}")
        return sheet_names, logs
    except FileNotFoundError:
        st.error(f"Fichier Excel '{excel_path}' introuvable pour lister les matériaux.")
        logs.append(f"Erreur critique FNF: Fichier Excel {excel_path} non trouvé pour lister matériaux.")
        return [], logs
    except Exception as e:
        st.error(f"Erreur lors de la lecture des noms de feuilles depuis '{excel_path}': {e}")
        logs.append(f"Erreur lecture noms feuilles depuis {excel_path}: {type(e).__name__} - {e}")
        return [], logs
@jax.jit
def get_n_fused_silica(l_nm: jnp.ndarray) -> jnp.ndarray:
    n = jnp.full_like(l_nm, 1.46, dtype=jnp.float64)
    k = jnp.zeros_like(n)
    return n + 1j * k
@jax.jit
def get_n_bk7(l_nm: jnp.ndarray) -> jnp.ndarray:
    n = jnp.full_like(l_nm, 1.523, dtype=jnp.float64)
    k = jnp.zeros_like(n)
    return n + 1j * k
@jax.jit
def get_n_d263(l_nm: jnp.ndarray) -> jnp.ndarray:
    n = jnp.full_like(l_nm, 1.523, dtype=jnp.float64)
    k = jnp.zeros_like(n)
    return n + 1j * k
@jax.jit
def interp_nk_cached(l_target: jnp.ndarray, l_data: jnp.ndarray, n_data: jnp.ndarray, k_data: jnp.ndarray) -> jnp.ndarray:
    n_interp = jnp.interp(l_target, l_data, n_data)
    k_interp_raw = jnp.interp(l_target, l_data, k_data)
    k_interp = jnp.maximum(k_interp_raw, 0.0)
    return n_interp + 1j * k_interp
MaterialInputType = Union[complex, float, int, str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
def _get_nk_array_for_lambda_vec(material_definition: MaterialInputType,
                                 l_vec_target_jnp: jnp.ndarray,
                                 excel_file_path: str) -> Tuple[Optional[jnp.ndarray], List[str]]:
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
                    logs.append(f"Erreur critique: Échec chargement données pour '{sheet_name}'.")
                    return None, logs
                l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
                l_target_min = jnp.min(l_vec_target_jnp)
                l_target_max = jnp.max(l_vec_target_jnp)
                l_data_min = jnp.min(l_data_jnp)
                l_data_max = jnp.max(l_data_jnp)
                if l_target_min < l_data_min - 1e-6 or l_target_max > l_data_max + 1e-6:
                     logs.append(f"AVERTISSEMENT: Interpolation pour '{sheet_name}' hors limites [{l_data_min:.1f}, {l_data_max:.1f}] nm (cible: [{l_target_min:.1f}, {l_target_max:.1f}] nm). Extrapolation utilisée.")
                     result = interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)
        elif isinstance(material_definition, tuple) and len(material_definition) == 3:
             l_data, n_data, k_data = material_definition
             l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
             if not len(l_data_jnp): raise ValueError("Données matériau brutes vides.")
             sort_indices = jnp.argsort(l_data_jnp)
             l_data_jnp = l_data_jnp[sort_indices]
             n_data_jnp = n_data_jnp[sort_indices]
             k_data_jnp = k_data_jnp[sort_indices]
             if np.any(k_data_jnp < -1e-9):
                 logs.append("AVERTISSEMENT: k<0 dans les données matériau brutes. Mise à 0.")
                 k_data_jnp = jnp.maximum(k_data_jnp, 0.0)
             result = interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)
        else:
            raise TypeError(f"Type de définition matériau non supporté: {type(material_definition)}")
        if jnp.any(jnp.isnan(result.real)) or jnp.any(result.real <= 0):
            logs.append(f"AVERTISSEMENT: n'<=0 ou NaN détecté pour '{material_definition}'. Remplacé par n'=1.")
            result = jnp.where(jnp.isnan(result.real) | (result.real <= 0), 1.0 + 1j*result.imag, result)
        if jnp.any(jnp.isnan(result.imag)) or jnp.any(result.imag < 0):
             logs.append(f"AVERTISSEMENT: k<0 ou NaN détecté pour '{material_definition}'. Remplacé par k=0.")
             result = jnp.where(jnp.isnan(result.imag) | (result.imag < 0), result.real + 0.0j, result)
        return result, logs
    except Exception as e:
        logs.append(f"Erreur préparation données matériau pour '{material_definition}': {e}")
        st.error(f"Erreur critique lors de la préparation du matériau '{material_definition}': {e}")
        return None, logs
def _get_nk_at_lambda(material_definition: MaterialInputType, l_nm_target: float, excel_file_path: str) -> Tuple[Optional[complex], List[str]]:
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
        nk_complex = complex(nk_array[0])
        return nk_complex, logs
@jax.jit
def _compute_layer_matrix_scan_step_jit(carry_matrix: jnp.ndarray, layer_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, None]:
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
    num_layers = len(ep_vector)
    layers_scan_data = (ep_vector, layer_indices, jnp.full(num_layers, l_val))
    M_initial = jnp.eye(2, dtype=jnp.complex128)
    M_final, _ = scan(_compute_layer_matrix_scan_step_jit, M_initial, layers_scan_data)
    return M_final
@jax.jit
def calculate_single_wavelength_T_core(l_val: jnp.ndarray, ep_vector_contig: jnp.ndarray,
                                       layer_indices_at_lval: jnp.ndarray, nSub_at_lval: jnp.ndarray) -> jnp.ndarray:
    etainc = 1.0 + 0j
    etasub = nSub_at_lval
    def calculate_for_valid_l(l_: jnp.ndarray) -> jnp.ndarray:
        current_layer_indices = layer_indices_at_lval
        M = compute_stack_matrix_core_jax(ep_vector_contig, current_layer_indices, l_)
        m00, m01 = M[0, 0], M[0, 1]
        m10, m11 = M[1, 0], M[1, 1]
        rs_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        rs_denominator_abs = jnp.abs(rs_denominator)
        safe_denominator = jnp.where(rs_denominator_abs < 1e-12, 1e-12 + 0j, rs_denominator)
        ts = (2.0 * etainc) / safe_denominator
        real_etasub = jnp.real(etasub)
        real_etainc = jnp.real(etainc)
        safe_real_etainc = jnp.maximum(real_etainc, 1e-9)
        Ts_complex = (real_etasub / safe_real_etainc) * (ts * jnp.conj(ts))
        Ts = jnp.real(Ts_complex)
        return jnp.where(rs_denominator_abs < 1e-12, jnp.nan, Ts)
    def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray:
        return jnp.nan
    Ts_result = cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)
    return Ts_result
def calculate_T_from_ep_jax(ep_vector: Union[np.ndarray, List[float]],
                            nH_material: MaterialInputType,
                            nL_material: MaterialInputType,
                            nSub_material: MaterialInputType,
                            l_vec: Union[np.ndarray, List[float]],
                            excel_file_path: str) -> Tuple[Optional[Dict[str, np.ndarray]], List[str]]:
    logs = []
    l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64)
    ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)
    if not l_vec_jnp.size:
        logs.append("Vecteur lambda vide, aucun calcul de T effectué.")
        return {'l': np.array([]), 'Ts': np.array([])}, logs
    if not ep_vector_jnp.size:
         logs.append("Structure vide (0 couches). Calcul du substrat nu.")
         nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
         logs.extend(logs_sub)
         if nSub_arr is None:
            return None, logs
         n_sub = jnp.real(nSub_arr)
         k_sub = jnp.imag(nSub_arr)
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
    if nH_arr is None or nL_arr is None or nSub_arr is None:
        logs.append("Erreur critique: Échec du chargement d'un des indices matériau.")
        return None, logs
    logs.append(f"Préparation indices terminée en {time.time() - start_time:.3f}s.")
    calculate_single_wavelength_T_hl_jit = jax.jit(calculate_single_wavelength_T_core)
    num_layers = len(ep_vector_jnp)
    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH_arr, nL_arr)
    indices_alternating_T = indices_alternating.T
    Ts_arr_raw = vmap(calculate_single_wavelength_T_hl_jit, in_axes=(0, None, 0, 0))(
        l_vec_jnp, ep_vector_jnp, indices_alternating_T, nSub_arr
    )
    Ts_arr = jnp.nan_to_num(Ts_arr_raw, nan=0.0)
    Ts_arr_clipped = jnp.clip(Ts_arr, 0.0, 1.0)
    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts_arr_clipped)}, logs
def calculate_T_from_ep_arbitrary_jax(ep_vector: Union[np.ndarray, List[float]],
                                       material_sequence: List[str],
                                       nSub_material: MaterialInputType,
                                       l_vec: Union[np.ndarray, List[float]],
                                       excel_file_path: str) -> Tuple[Optional[Dict[str, np.ndarray]], List[str]]:
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
         logs.append("Structure vide (0 couches). Calcul du substrat nu.")
         nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
         logs.extend(logs_sub)
         if nSub_arr is None: return None, logs
         Ts = jnp.ones_like(l_vec_jnp)
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
            break
        layer_indices_list.append(nk_arr)
    nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
    logs.extend(logs_sub)
    if nSub_arr is None:
        logs.append("Erreur critique: Échec chargement matériau substrat.")
        materials_ok = False
    if not materials_ok:
        return None, logs
    if layer_indices_list:
        layer_indices_arr = jnp.stack(layer_indices_list, axis=0)
    else:
         layer_indices_arr = jnp.empty((0, len(l_vec_jnp)), dtype=jnp.complex128)
    logs.append(f"Préparation indices terminée en {time.time() - start_time:.3f}s.")
    calculate_single_wavelength_T_arb_jit = jax.jit(calculate_single_wavelength_T_core)
    layer_indices_arr_T = layer_indices_arr.T
    Ts_arr_raw = vmap(calculate_single_wavelength_T_arb_jit, in_axes=(0, None, 0, 0))(
        l_vec_jnp, ep_vector_jnp, layer_indices_arr_T, nSub_arr
    )
    Ts_arr = jnp.nan_to_num(Ts_arr_raw, nan=0.0)
    Ts_arr_clipped = jnp.clip(Ts_arr, 0.0, 1.0)
    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts_arr_clipped)}, logs
def calculate_initial_ep(emp: Union[List[float], Tuple[float,...]], l0: float,
                           nH0_material: MaterialInputType, nL0_material: MaterialInputType,
                           excel_file_path: str) -> Tuple[Optional[np.ndarray], List[str]]:
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
        return None, logs
    nH_real_at_l0 = nH_complex_at_l0.real
    nL_real_at_l0 = nL_complex_at_l0.real
    if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
         logs.append(f"AVERTISSEMENT: n'H({nH_real_at_l0:.3f}) ou n'L({nL_real_at_l0:.3f}) à l0={l0}nm est <= 0. Calcul QWOT peut être incorrect.")
    for i in range(num_layers):
        multiplier = emp[i]
        is_H_layer = (i % 2 == 0)
        n_real_layer_at_l0 = nH_real_at_l0 if is_H_layer else nL_real_at_l0
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
         st.error("Erreur lors du calcul des épaisseurs initiales due à des indices invalides à l0.")
         return None, logs
    return ep_initial, logs
def calculate_qwot_from_ep(ep_vector: np.ndarray, l0: float,
                            nH0_material: MaterialInputType, nL0_material: MaterialInputType,
                            excel_file_path: str) -> Tuple[Optional[np.ndarray], List[str]]:
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
        logs.append(f"Erreur: Impossible d'obtenir n'H ou n'L à l0={l0}nm pour calculer QWOT. Retourne NaN.")
        st.error(f"Erreur calcul QWOT : indices H/L non trouvés à l0={l0}nm.")
        return None, logs
    nH_real_at_l0 = nH_complex_at_l0.real
    nL_real_at_l0 = nL_complex_at_l0.real
    if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
         logs.append(f"AVERTISSEMENT: n'H({nH_real_at_l0:.3f}) ou n'L({nL_real_at_l0:.3f}) à l0={l0}nm est <= 0. Calcul QWOT peut être incorrect/NaN.")
    indices_ok = True
    for i in range(num_layers):
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real_layer_at_l0 <= 1e-9:
            if ep_vector[i] > 1e-9:
                layer_type = "H" if i % 2 == 0 else "L"
                logs.append(f"Avertissement: Impossible de calculer QWOT pour couche {i+1} ({layer_type}) car n'({l0}nm) <= 0.")
                indices_ok = False
            else:
                qwot_multipliers[i] = 0.0
        else:
            qwot_multipliers[i] = ep_vector[i] * (4.0 * n_real_layer_at_l0) / l0
    if not indices_ok:
        st.warning("Certains QWOT n'ont pas pu être calculés (indices invalides à l0). Ils apparaissent comme NaN.")
        return qwot_multipliers, logs
    else:
        return qwot_multipliers, logs
def calculate_final_mse(res: Dict[str, np.ndarray], active_targets: List[Dict]) -> Tuple[Optional[float], int]:
    total_squared_error = 0.0
    total_points_in_targets = 0
    mse = None
    if not active_targets or 'Ts' not in res or res['Ts'] is None or 'l' not in res or res['l'] is None:
        return mse, total_points_in_targets
    res_l_np = np.asarray(res['l'])
    res_ts_np = np.asarray(res['Ts'])
    if res_l_np.size == 0 or res_ts_np.size == 0 or res_l_np.size != res_ts_np.size:
         return mse, total_points_in_targets
    for target in active_targets:
        try:
            l_min = float(target['min'])
            l_max = float(target['max'])
            t_min = float(target['target_min'])
            t_max = float(target['target_max'])
            if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0): continue
            if l_max < l_min: continue
        except (KeyError, ValueError, TypeError):
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
            squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
            total_squared_error += np.sum(squared_errors)
            total_points_in_targets += len(calculated_Ts_in_zone)
    if total_points_in_targets > 0:
        mse = total_squared_error / total_points_in_targets
    return mse, total_points_in_targets
@jax.jit
def calculate_mse_for_optimization_penalized_jax(ep_vector: jnp.ndarray,
                                                 nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                                 l_vec_optim: jnp.ndarray,
                                                 active_targets_tuple: Tuple[Tuple[float, float, float, float], ...],
                                                 min_thickness_phys: float) -> jnp.ndarray:
    below_min_mask = (ep_vector < min_thickness_phys) & (ep_vector > 1e-12)
    penalty_thin = jnp.sum(jnp.where(below_min_mask, (min_thickness_phys - ep_vector)**2, 0.0))
    penalty_weight = 1e5
    penalty_cost = penalty_thin * penalty_weight
    ep_vector_calc = jnp.maximum(ep_vector, min_thickness_phys)
    num_layers = len(ep_vector_calc)
    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH_arr, nL_arr)
    indices_alternating_T = indices_alternating.T
    calculate_T_single_jit = jax.jit(calculate_single_wavelength_T_core)
    Ts_raw = vmap(calculate_T_single_jit, in_axes=(0, None, 0, 0))(
        l_vec_optim, ep_vector_calc, indices_alternating_T, nSub_arr
    )
    Ts = jnp.nan_to_num(Ts_raw, nan=0.0)
    total_squared_error = 0.0
    total_points_in_targets = 0
    for i in range(len(active_targets_tuple)):
        l_min, l_max, t_min, t_max = active_targets_tuple[i]
        target_mask = (l_vec_optim >= l_min) & (l_vec_optim <= l_max)
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t_full = t_min + slope * (l_vec_optim - l_min)
        squared_errors_full = (Ts - interpolated_target_t_full)**2
        masked_sq_error = jnp.where(target_mask, squared_errors_full, 0.0)
        total_squared_error += jnp.sum(masked_sq_error)
        total_points_in_targets += jnp.sum(target_mask)
    mse = jnp.where(total_points_in_targets > 0,
                     total_squared_error / total_points_in_targets,
                     jnp.inf)
    final_cost = mse + penalty_cost
    return jnp.nan_to_num(final_cost, nan=jnp.inf, posinf=jnp.inf)
@jax.jit
def calculate_mse_arbitrary_sequence_jax(ep_vector: jnp.ndarray,
                                         layer_indices_arr: jnp.ndarray,
                                         nSub_arr: jnp.ndarray,
                                         l_vec_eval: jnp.ndarray,
                                         active_targets_tuple: Tuple[Tuple[float, float, float, float], ...]) -> jnp.ndarray:
    layer_indices_arr_T = layer_indices_arr.T
    calculate_T_single_jit = jax.jit(calculate_single_wavelength_T_core)
    Ts_raw = vmap(calculate_T_single_jit, in_axes=(0, None, 0, 0))(
        l_vec_eval, ep_vector, layer_indices_arr_T, nSub_arr
    )
    Ts = jnp.nan_to_num(Ts_raw, nan=0.0)
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
def _run_core_optimization(ep_start_optim: np.ndarray,
                           validated_inputs: Dict, active_targets: List[Dict],
                           min_thickness_phys: float, log_prefix: str = ""
                           ) -> Tuple[Optional[np.ndarray], bool, float, List[str], str, int, int]:
    logs = []
    num_layers_start = len(ep_start_optim)
    optim_success = False
    final_cost = np.inf
    result_message_str = "Optimisation non lancée ou échouée précocement."
    nit_total = 0
    nfev_total = 0
    final_ep = None
    if num_layers_start == 0:
        logs.append(f"{log_prefix}Impossible d'optimiser une structure vide.")
        return None, False, np.inf, logs, "Structure vide", 0, 0
    try:
        l_min_optim = validated_inputs['l_range_deb']
        l_max_optim = validated_inputs['l_range_fin']
        l_step_optim = validated_inputs['l_step']
        nH_material = validated_inputs['nH_material']
        nL_material = validated_inputs['nL_material']
        nSub_material = validated_inputs['nSub_material']
        maxiter = validated_inputs['maxiter']
        maxfun = validated_inputs['maxfun']
        num_pts_optim = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
        l_vec_optim_np = np.geomspace(l_min_optim, l_max_optim, num_pts_optim)
        l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
        if not l_vec_optim_np.size:
            raise ValueError("Échec de génération du vecteur lambda pour l'optimisation.")
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        logs.append(f"{log_prefix}Préparation indices dispersifs pour {len(l_vec_optim_jax)} lambdas...")
        prep_start_time = time.time()
        nH_arr_optim, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_h)
        nL_arr_optim, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_l)
        nSub_arr_optim, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_sub)
        if nH_arr_optim is None or nL_arr_optim is None or nSub_arr_optim is None:
             raise RuntimeError("Échec chargement indices pour l'optimisation.")
        logs.append(f"{log_prefix} Préparation indices finie en {time.time() - prep_start_time:.3f}s.")
        active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
        static_args_for_jax = (
            nH_arr_optim, nL_arr_optim, nSub_arr_optim,
            l_vec_optim_jax, active_targets_tuple,
            min_thickness_phys
        )
        value_and_grad_fn = jax.jit(jax.value_and_grad(calculate_mse_for_optimization_penalized_jax))
        def scipy_obj_grad_wrapper(ep_vector_np_in, *args):
            try:
                ep_vector_jax = jnp.asarray(ep_vector_np_in, dtype=jnp.float64)
                value_jax, grad_jax = value_and_grad_fn(ep_vector_jax, *args)
                if not jnp.isfinite(value_jax):
                    value_np = np.inf
                    grad_np = np.zeros_like(ep_vector_np_in, dtype=np.float64)
                else:
                    value_np = float(np.array(value_jax))
                    grad_np_raw = np.array(grad_jax, dtype=np.float64)
                    grad_np = np.nan_to_num(grad_np_raw, nan=0.0, posinf=1e6, neginf=-1e6)
                return value_np, grad_np
            except Exception as e_wrap:
                print(f"Erreur dans scipy_obj_grad_wrapper: {e_wrap}")
                return np.inf, np.zeros_like(ep_vector_np_in, dtype=np.float64)
        lbfgsb_bounds = [(min_thickness_phys, None)] * num_layers_start
        options = {'maxiter': maxiter, 'maxfun': maxfun,
                   'disp': False,
                   'ftol': 1e-12, 'gtol': 1e-8}
        logs.append(f"{log_prefix}Lancement L-BFGS-B avec JAX gradient...")
        opt_start_time = time.time()
        result = minimize(scipy_obj_grad_wrapper,
                          ep_start_optim,
                          args=static_args_for_jax,
                          method='L-BFGS-B',
                          jac=True,
                          bounds=lbfgsb_bounds,
                          options=options)
        logs.append(f"{log_prefix}L-BFGS-B (JAX grad) fini en {time.time() - opt_start_time:.3f}s.")
        final_cost = result.fun if np.isfinite(result.fun) else np.inf
        result_message_str = result.message.decode('utf-8') if isinstance(result.message, bytes) else str(result.message)
        nit_total = result.nit if hasattr(result, 'nit') else 0
        nfev_total = result.nfev if hasattr(result, 'nfev') else 0
        is_success_or_limit = (result.success or result.status == 1) and np.isfinite(final_cost)
        if is_success_or_limit:
            final_ep_raw = result.x
            final_ep = np.maximum(final_ep_raw, min_thickness_phys)
            optim_success = True
            log_status = "succès" if result.success else "limite atteinte"
            logs.append(f"{log_prefix}Optimisation terminée ({log_status}). Coût final: {final_cost:.3e}, Itérations: {nit_total}, Evals: {nfev_total}, Msg: {result_message_str}")
        else:
            optim_success = False
            final_ep = np.maximum(ep_start_optim, min_thickness_phys)
            logs.append(f"{log_prefix}Optimisation ÉCHOUÉE. Status: {result.status}, Msg: {result_message_str}, Coût: {final_cost:.3e}")
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
        final_ep = np.maximum(ep_start_optim, min_thickness_phys) if ep_start_optim is not None else None
        optim_success = False
        final_cost = np.inf
        result_message_str = f"Exception: {e_optim}"
        nit_total = 0
        nfev_total = 0
    return final_ep, optim_success, final_cost, logs, result_message_str, nit_total, nfev_total
def _perform_layer_merge_or_removal_only(ep_vector_in: np.ndarray, min_thickness_phys: float,
                                         log_prefix: str = "", target_layer_index: Optional[int] = None,
                                         threshold_for_removal: Optional[float] = None) -> Tuple[Optional[np.ndarray], bool, List[str]]:
    current_ep = ep_vector_in.copy()
    logs = []
    num_layers = len(current_ep)
    structure_changed = False
    ep_after_merge = None
    if num_layers <= 2 and target_layer_index is None:
        logs.append(f"{log_prefix}Structure <= 2 couches. Suppression/fusion non possible sans cible.")
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
                logs.append(f"{log_prefix}Ciblage manuel couche {thin_layer_index + 1} ({min_thickness_found:.3f} nm).")
            else:
                logs.append(f"{log_prefix}Cible manuelle {target_layer_index+1} invalide/trop fine. Recherche auto.")
                target_layer_index = None
        if target_layer_index is None:
            candidate_indices = np.where(current_ep >= min_thickness_phys)[0]
            if candidate_indices.size == 0:
                logs.append(f"{log_prefix}Aucune couche >= {min_thickness_phys:.3f} nm trouvée.")
                return current_ep, False, logs
            candidate_thicknesses = current_ep[candidate_indices]
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
                    return current_ep, False, logs
            if indices_to_consider.size > 0:
                min_idx_local = np.argmin(thicknesses_to_consider)
                thin_layer_index = indices_to_consider[min_idx_local]
                min_thickness_found = thicknesses_to_consider[min_idx_local]
            else:
                logs.append(f"{log_prefix}Aucune couche candidate finale trouvée.")
                return current_ep, False, logs
        if thin_layer_index == -1:
            logs.append(f"{log_prefix}Échec identification couche (cas inattendu).")
            return current_ep, False, logs
        thin_layer_thickness = current_ep[thin_layer_index]
        logs.append(f"{log_prefix}Couche identifiée pour action: Index {thin_layer_index} (Couche {thin_layer_index + 1}), épaisseur {thin_layer_thickness:.3f} nm.")
        if num_layers <= 2:
             logs.append(f"{log_prefix}Erreur logique: tentative de fusion sur <= 2 couches.")
             return current_ep, False, logs
        elif thin_layer_index == 0:
            ep_after_merge = current_ep[2:]
            merged_info = f"Suppression des 2 premières couches."
            structure_changed = True
        elif thin_layer_index == num_layers - 1:
             if num_layers >= 2:
                 ep_after_merge = current_ep[:-2]
                 merged_info = f"Suppression des 2 dernières couches."
                 structure_changed = True
             else:
                 logs.append(f"{log_prefix}Cas spécial: impossible supprimer 2 dernières couches (num_layers={num_layers}).")
                 return current_ep, False, logs
        else:
            merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
            ep_before = current_ep[:thin_layer_index - 1]
            ep_after = current_ep[thin_layer_index + 2:]
            ep_after_merge = np.concatenate((ep_before, [merged_thickness], ep_after))
            merged_info = f"Fusion des couches {thin_layer_index} et {thin_layer_index + 2} autour de la couche {thin_layer_index + 1} supprimée -> nouvelle épaisseur {merged_thickness:.3f} nm."
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
        logs.append(f"{log_prefix}ERREUR durant la logique de fusion/suppression: {e_merge}\n{traceback.format_exc(limit=1)}")
        st.error(f"Erreur interne lors de la suppression/fusion de couche: {e_merge}")
        return current_ep, False, logs
def _perform_needle_insertion_scan(ep_vector_in: np.ndarray,
                                   nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
                                   l_vec_optim_np: np.ndarray, active_targets: List[Dict],
                                   cost_function_jax: Callable,
                                   min_thickness_phys: float, base_needle_thickness_nm: float,
                                   scan_step: float, l0_repr: float,
                                   excel_file_path: str, log_prefix: str = ""
                                   ) -> Tuple[Optional[np.ndarray], float, List[str], int]:
    logs = []
    num_layers_in = len(ep_vector_in)
    if num_layers_in == 0:
        logs.append(f"{log_prefix}Scan aiguille impossible sur structure vide.")
        return None, np.inf, logs, -1
    logs.append(f"{log_prefix}Démarrage scan aiguille ({num_layers_in} couches). Pas: {scan_step} nm, ép. aiguille: {base_needle_thickness_nm:.3f} nm.")
    try:
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
    best_ep_found = None
    min_cost_found = initial_cost
    best_insertion_idx = -1
    tested_insertions = 0
    ep_cumsum = np.cumsum(ep_vector_in)
    total_thickness = ep_cumsum[-1] if num_layers_in > 0 else 0.0
    for z in np.arange(scan_step, total_thickness, scan_step):
        current_layer_idx = -1
        layer_start_z = 0.0
        for i in range(num_layers_in):
            layer_end_z = ep_cumsum[i]
            if z > layer_start_z and z <= layer_end_z:
                t_part1 = z - layer_start_z
                t_part2 = layer_end_z - z
                if t_part1 >= min_thickness_phys and t_part2 >= min_thickness_phys:
                    current_layer_idx = i
                else:
                    current_layer_idx = -2
                break
            layer_start_z = layer_end_z
        if current_layer_idx < 0:
            continue
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
            logs.append(f"{log_prefix} AVERTISSEMENT: Échec calcul coût pour z={z:.2f}. {e_cost}")
            continue
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
    logs = []
    best_ep_overall = np.asarray(ep_start).copy()
    best_mse_overall = np.inf
    total_nit_needles = 0
    total_nfev_needles = 0
    successful_reopts_count = 0
    nH_material = validated_inputs['nH_material']
    nL_material = validated_inputs['nL_material']
    nSub_material = validated_inputs['nSub_material']
    l0_repr = validated_inputs.get('l0', 500.0)
    cost_fn_penalized_jit = jax.jit(calculate_mse_for_optimization_penalized_jax)
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
        return ep_start, np.inf, logs, 0, 0, 0
    for i in range(num_needles):
        logs.append(f"{log_prefix} --- Itération Aiguille {i + 1}/{num_needles} ---")
        current_ep_iter = best_ep_overall.copy()
        num_layers_current = len(current_ep_iter)
        if num_layers_current == 0:
            logs.append(f"{log_prefix} Structure vide, arrêt itérations aiguilles."); break
        st.write(f"{log_prefix} Scan aiguille {i+1}...")
        l_min_optim, l_max_optim = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
        l_step_optim = validated_inputs['l_step']
        num_pts = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
        l_vec_optim_np_needle = np.geomspace(l_min_optim, l_max_optim, num_pts)
        l_vec_optim_np_needle = l_vec_optim_np_needle[(l_vec_optim_np_needle > 0) & np.isfinite(l_vec_optim_np_needle)]
        if not l_vec_optim_np_needle.size:
             logs.append("  ERREUR: impossible de générer lambda pour phase aiguille. Cycle annulé.")
             break
        ep_after_scan, cost_after_scan, scan_logs, inserted_idx = _perform_needle_insertion_scan(
            current_ep_iter,
            nH_material, nL_material, nSub_material,
            l_vec_optim_np_needle, active_targets,
            cost_fn_penalized_jit,
            min_thickness_phys, BASE_NEEDLE_THICKNESS_NM, DEFAULT_NEEDLE_SCAN_STEP_NM, l0_repr,
            excel_file_path, log_prefix=f"{log_prefix}  [Scan {i+1}] "
        )
        logs.extend(scan_logs)
        if ep_after_scan is None:
            logs.append(f"{log_prefix} Scan aiguille {i + 1} n'a pas trouvé d'amélioration. Arrêt des itérations aiguilles."); break
        logs.append(f"{log_prefix} Scan {i + 1} a trouvé amélioration potentielle. Ré-optimisation...")
        st.write(f"{log_prefix} Ré-optimisation après aiguille {i+1}...")
        ep_after_reopt, optim_success, final_cost_reopt, optim_logs, optim_status_msg, nit_reopt, nfev_reopt = \
            _run_core_optimization(ep_after_scan, validated_inputs, active_targets,
                                 min_thickness_phys, log_prefix=f"{log_prefix}  [Re-Opt {i+1}] ")
        logs.extend(optim_logs)
        if not optim_success:
            logs.append(f"{log_prefix} Ré-optimisation après scan {i + 1} ÉCHOUÉE. Arrêt des itérations aiguilles."); break
        logs.append(f"{log_prefix} Ré-optimisation {i + 1} réussie. Nouveau MSE: {final_cost_reopt:.6e}. (Iter/Eval: {nit_reopt}/{nfev_reopt})")
        total_nit_needles += nit_reopt
        total_nfev_needles += nfev_reopt
        successful_reopts_count += 1
        if final_cost_reopt < best_mse_overall - MSE_IMPROVEMENT_TOLERANCE:
            logs.append(f"{log_prefix}  MSE amélioré par rapport au meilleur précédent ({best_mse_overall:.6e}). Mise à jour.")
            best_ep_overall = ep_after_reopt.copy()
            best_mse_overall = final_cost_reopt
            cycle_improved_globally = True
        else:
            logs.append(f"{log_prefix}  Nouveau MSE ({final_cost_reopt:.6e}) pas significativement meilleur que le précédent ({best_mse_overall:.6e}). Arrêt des itérations aiguilles.")
            best_ep_overall = ep_after_reopt.copy()
            best_mse_overall = final_cost_reopt
            break
    logs.append(f"{log_prefix} Fin itérations aiguilles. Meilleur MSE final: {best_mse_overall:.6e}")
    logs.append(f"{log_prefix} Total Iter/Eval durant {successful_reopts_count} ré-optimisations réussies: {total_nit_needles}/{total_nfev_needles}")
    return best_ep_overall, best_mse_overall, logs, total_nit_needles, total_nfev_needles, successful_reopts_count
def run_auto_mode(initial_ep: Optional[np.ndarray],
                  validated_inputs: Dict, active_targets: List[Dict],
                  excel_file_path: str, log_callback: Callable):
    logs = []
    start_time_auto = time.time()
    log_callback("#"*10 + f" Démarrage Mode Auto (Max {AUTO_MAX_CYCLES} Cycles) " + "#"*10)
    best_ep_so_far = None
    best_mse_so_far = np.inf
    num_cycles_done = 0
    termination_reason = f"Max {AUTO_MAX_CYCLES} cycles atteints"
    threshold_for_thin_removal = validated_inputs.get('auto_thin_threshold', 1.0)
    log_callback(f"  Seuil suppression auto: {threshold_for_thin_removal:.3f} nm")
    total_iters_auto = 0
    total_evals_auto = 0
    optim_runs_auto = 0
    try:
        current_ep = None
        if initial_ep is not None:
             log_callback("  Mode Auto: Utilisation de la structure optimisée précédente.")
             current_ep = initial_ep.copy()
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
             log_callback(f"  MSE initial (depuis état optimisé): {best_mse_so_far:.6e}")
        else:
            log_callback("  Mode Auto: Utilisation de la structure nominale (QWOT).")
            emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
            if not emp_list: raise ValueError("QWOT nominal vide.")
            ep_nominal, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'],
                                                            validated_inputs['nH_material'], validated_inputs['nL_material'],
                                                            excel_file_path)
            log_callback(logs_ep_init)
            if ep_nominal is None: raise RuntimeError("Échec calcul épaisseurs nominales initiales.")
            log_callback(f"  Structure nominale: {len(ep_nominal)} couches. Lancement optimisation initiale...")
            st.info("Mode Auto : Optimisation initiale de la structure nominale...")
            ep_after_initial_opt, initial_opt_success, initial_mse, initial_opt_logs, initial_opt_msg, initial_nit, initial_nfev = \
                 _run_core_optimization(ep_nominal, validated_inputs, active_targets,
                                        MIN_THICKNESS_PHYS_NM, log_prefix="  [Auto Init Opt] ")
            log_callback(initial_opt_logs)
            if not initial_opt_success:
                 log_callback(f"ERREUR: Échec optimisation initiale en Mode Auto ({initial_opt_msg}). Annulation.")
                 st.error(f"Échec de l'optimisation initiale du Mode Auto: {initial_opt_msg}")
                 return None, np.inf, logs, 0, 0
            log_callback(f"  Optimisation initiale terminée. MSE: {initial_mse:.6e} (Iter/Eval: {initial_nit}/{initial_nfev})")
            best_ep_so_far = ep_after_initial_opt.copy()
            best_mse_so_far = initial_mse
            total_iters_auto += initial_nit; total_evals_auto += initial_nfev; optim_runs_auto += 1
        if best_ep_so_far is None or not np.isfinite(best_mse_so_far):
             raise RuntimeError("État de départ invalide pour les cycles Auto.")
        log_callback(f"--- Démarrage des Cycles Auto (MSE départ: {best_mse_so_far:.6e}, {len(best_ep_so_far)} couches) ---")
        for cycle_num in range(AUTO_MAX_CYCLES):
            log_callback(f"\n--- Cycle Auto {cycle_num + 1} / {AUTO_MAX_CYCLES} ---")
            st.info(f"Cycle Auto {cycle_num + 1}/{AUTO_MAX_CYCLES} | MSE actuel: {best_mse_so_far:.3e}")
            mse_at_cycle_start = best_mse_so_far
            ep_at_cycle_start = best_ep_so_far.copy()
            cycle_improved_globally = False
            log_callback(f"  [Cycle {cycle_num+1}] Phase Aiguille ({AUTO_NEEDLES_PER_CYCLE} itérations max)...")
            st.write(f"Cycle {cycle_num + 1}: Phase Aiguille...")
            l_min_optim, l_max_optim = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
            l_step_optim = validated_inputs['l_step']
            num_pts = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
            l_vec_optim_np_needle = np.geomspace(l_min_optim, l_max_optim, num_pts)
            l_vec_optim_np_needle = l_vec_optim_np_needle[(l_vec_optim_np_needle > 0) & np.isfinite(l_vec_optim_np_needle)]
            if not l_vec_optim_np_needle.size:
                 log_callback("  ERREUR: impossible de générer lambda pour phase aiguille. Cycle annulé.")
                 break
            ep_after_needles, mse_after_needles, needle_logs, nit_needles, nfev_needles, reopts_in_needles = \
                 _run_needle_iterations(best_ep_so_far, AUTO_NEEDLES_PER_CYCLE, validated_inputs, active_targets,
                                        MIN_THICKNESS_PHYS_NM, l_vec_optim_np_needle,
                                        DEFAULT_NEEDLE_SCAN_STEP_NM, BASE_NEEDLE_THICKNESS_NM,
                                        excel_file_path, log_prefix=f"    [Needle {cycle_num+1}] ")
            log_callback(needle_logs)
            log_callback(f"  [Cycle {cycle_num+1}] Fin Phase Aiguille. MSE: {mse_after_needles:.6e} (Iter/Eval: {nit_needles}/{nfev_needles})")
            total_iters_auto += nit_needles; total_evals_auto += nfev_needles; optim_runs_auto += reopts_in_needles
            if mse_after_needles < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                 log_callback(f"    Amélioration globale par phase aiguille (vs {best_mse_so_far:.6e}).")
                 best_ep_so_far = ep_after_needles.copy()
                 best_mse_so_far = mse_after_needles
                 cycle_improved_globally = True
            else:
                 log_callback(f"    Pas d'amélioration globale par phase aiguille (vs {best_mse_so_far:.6e}).")
                 best_ep_so_far = ep_after_needles.copy()
                 best_mse_so_far = mse_after_needles
            log_callback(f"  [Cycle {cycle_num+1}] Phase Suppression (< {threshold_for_thin_removal:.3f} nm) + Re-Opt...")
            st.write(f"Cycle {cycle_num + 1}: Phase Suppression...")
            layers_removed_this_cycle = 0;
            max_thinning_attempts = len(best_ep_so_far) + 2
            for attempt in range(max_thinning_attempts):
                current_num_layers_thin = len(best_ep_so_far)
                if current_num_layers_thin <= 2:
                    log_callback("    Structure trop petite (< 3 couches), arrêt suppression.")
                    break
                ep_after_single_removal, structure_changed, removal_logs = \
                    _perform_layer_merge_or_removal_only(best_ep_so_far, MIN_THICKNESS_PHYS_NM,
                                                         log_prefix=f"    [Thin {cycle_num+1}.{attempt+1}] ",
                                                         threshold_for_removal=threshold_for_thin_removal)
                log_callback(removal_logs)
                if structure_changed and ep_after_single_removal is not None:
                    layers_removed_this_cycle += 1
                    log_callback(f"    Couche supprimée/fusionnée ({layers_removed_this_cycle} dans ce cycle). Ré-optimisation ({len(ep_after_single_removal)} couches)...")
                    st.write(f"Cycle {cycle_num + 1}: Ré-opt après suppression {layers_removed_this_cycle}...")
                    ep_after_thin_reopt, thin_reopt_success, mse_after_thin_reopt, thin_reopt_logs, thin_reopt_msg, nit_thin_reopt, nfev_thin_reopt = \
                         _run_core_optimization(ep_after_single_removal, validated_inputs, active_targets,
                                                MIN_THICKNESS_PHYS_NM, log_prefix=f"      [ReOptThin {cycle_num+1}.{attempt+1}] ")
                    log_callback(thin_reopt_logs)
                    total_iters_auto += nit_thin_reopt; total_evals_auto += nfev_thin_reopt
                    if thin_reopt_success:
                        optim_runs_auto += 1
                        log_callback(f"      Ré-optimisation réussie. MSE: {mse_after_thin_reopt:.6e} (Iter/Eval: {nit_thin_reopt}/{nfev_thin_reopt})")
                        if mse_after_thin_reopt < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                            log_callback(f"      Amélioration globale par suppression+reopt (vs {best_mse_so_far:.6e}).")
                            best_ep_so_far = ep_after_thin_reopt.copy()
                            best_mse_so_far = mse_after_thin_reopt
                            cycle_improved_globally = True
                        else:
                            log_callback(f"      Pas d'amélioration globale (vs {best_mse_so_far:.6e}). On continue avec ce résultat.")
                            best_ep_so_far = ep_after_thin_reopt.copy()
                            best_mse_so_far = mse_after_thin_reopt
                    else:
                        log_callback(f"    AVERTISSEMENT: Ré-optimisation après suppression ÉCHOUÉE ({thin_reopt_msg}). Arrêt suppression pour ce cycle.")
                        best_ep_so_far = ep_after_single_removal.copy()
                        try:
                            l_min_opt_recalc, l_max_opt_recalc = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
                            l_step_opt_recalc = validated_inputs['l_step']
                            num_pts_opt_recalc = max(2, int(np.round((l_max_opt_recalc - l_min_opt_recalc) / l_step_opt_recalc)) + 1)
                            l_vec_opt_recalc_np = np.geomspace(l_min_opt_recalc, l_max_opt_recalc, num_pts_opt_recalc)
                            l_vec_opt_recalc_np = l_vec_opt_recalc_np[(l_vec_opt_recalc_np > 0) & np.isfinite(l_vec_opt_recalc_np)]
                            if l_vec_opt_recalc_np.size > 0:
                                results_fail_grid_recalc, _ = calculate_T_from_ep_jax(best_ep_so_far, validated_inputs['nH_material'], validated_inputs['nL_material'], validated_inputs['nSub_material'], l_vec_opt_recalc_np, excel_file_path)
                                if results_fail_grid_recalc:
                                    mse_fail_recalc, _ = calculate_final_mse(results_fail_grid_recalc, active_targets)
                                    best_mse_so_far = mse_fail_recalc if mse_fail_recalc is not None and np.isfinite(mse_fail_recalc) else np.inf
                                else: best_mse_so_far = np.inf
                            else: best_mse_so_far = np.inf
                        except Exception as e_cost_fail:
                            log_callback(f"      ERREUR recalcul MSE après échec re-opt: {e_cost_fail}"); best_mse_so_far = np.inf
                        break
                else:
                    log_callback("    Aucune autre couche à supprimer/fusionner dans cette phase.")
                    break
            log_callback(f"  [Cycle {cycle_num+1}] Fin Phase Suppression. {layers_removed_this_cycle} couche(s) supprimée(s).")
            num_cycles_done += 1
            log_callback(f"--- Fin Cycle Auto {cycle_num + 1} --- Meilleur MSE actuel: {best_mse_so_far:.6e} ({len(best_ep_so_far)} couches) ---")
            if not cycle_improved_globally and best_mse_so_far >= mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE:
                 log_callback(f"Aucune amélioration significative dans Cycle {cycle_num + 1} (Début: {mse_at_cycle_start:.6e}, Fin: {best_mse_so_far:.6e}). Arrêt Mode Auto.")
                 termination_reason = f"Pas d'amélioration (Cycle {cycle_num + 1})"
                 if best_mse_so_far > mse_at_cycle_start + MSE_IMPROVEMENT_TOLERANCE :
                      log_callback("  MSE a augmenté, retour à l'état précédent le cycle.")
                      best_ep_so_far = ep_at_cycle_start.copy()
                      best_mse_so_far = mse_at_cycle_start
                 break
        log_callback(f"\n--- Mode Auto Terminé après {num_cycles_done} cycles ---")
        log_callback(f"Raison: {termination_reason}")
        log_callback(f"Meilleur MSE final: {best_mse_so_far:.6e} avec {len(best_ep_so_far)} couches.")
        avg_nit_str = f"{total_iters_auto / optim_runs_auto:.1f}" if optim_runs_auto > 0 else "N/A"
        avg_nfev_str = f"{total_evals_auto / optim_runs_auto:.1f}" if optim_runs_auto > 0 else "N/A"
        log_callback(f"Stats Globales Auto: {optim_runs_auto} optimisations réussies, Sum Iter/Eval: {total_iters_auto}/{total_evals_auto}, Moy Iter/Eval: {avg_nit_str}/{avg_nfev_str}")
        return best_ep_so_far, best_mse_so_far, logs, total_iters_auto, total_evals_auto
    except (ValueError, RuntimeError, TypeError) as e:
        log_callback(f"ERREUR fatale durant le Mode Auto (Setup/Workflow): {e}")
        st.error(f"Erreur Mode Auto: {e}")
        return None, np.inf, logs, total_iters_auto, total_evals_auto
    except Exception as e_fatal:
         log_callback(f"ERREUR inattendue fatale durant le Mode Auto: {type(e_fatal).__name__}: {e_fatal}")
         st.error(f"Erreur inattendue Mode Auto: {e_fatal}")
         traceback.print_exc()
         return None, np.inf, logs, total_iters_auto, total_evals_auto
@jax.jit
def calculate_M_for_thickness(thickness: jnp.ndarray, n_complex_layer: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    eta = n_complex_layer
    safe_l_val = jnp.maximum(l_val, 1e-9)
    safe_eta = jnp.where(jnp.abs(eta) < 1e-12, 1e-12 + 0j, eta)
    phi = (2 * jnp.pi / safe_l_val) * (n_complex_layer * thickness)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    m00 = jnp.where(thickness > 1e-12, cos_phi, 1.0)
    m01 = jnp.where(thickness > 1e-12, (1j / safe_eta) * sin_phi, 0.0)
    m10 = jnp.where(thickness > 1e-12, 1j * eta * sin_phi, 0.0)
    m11 = jnp.where(thickness > 1e-12, cos_phi, 1.0)
    M_layer = jnp.array([[m00, m01], [m10, m11]], dtype=jnp.complex128)
    return M_layer
calculate_M_batch_for_thickness = vmap(calculate_M_for_thickness, in_axes=(None, None, 0))
@jax.jit
def get_layer_matrices_qwot(layer_idx: int, initial_layer_number: int, l0: float,
                            nH_c_l0: jnp.ndarray, nL_c_l0: jnp.ndarray,
                            l_vec: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    predicate_is_H = (layer_idx % 2 == 0)
    n_real_l0 = jax.lax.select(predicate_is_H, nH_c_l0.real, nL_c_l0.real)
    n_complex_for_matrix = jax.lax.select(predicate_is_H, nH_c_l0, nL_c_l0)
    denom = 4.0 * jnp.maximum(n_real_l0, 1e-9)
    safe_l0 = jnp.maximum(l0, 1e-9)
    ep1_calc = 1.0 * safe_l0 / denom
    ep2_calc = 2.0 * safe_l0 / denom
    ep1 = jnp.where(n_real_l0 > 1e-9, ep1_calc, 0.0)
    ep2 = jnp.where(n_real_l0 > 1e-9, ep2_calc, 0.0)
    M_1qwot_batch = calculate_M_batch_for_thickness(ep1, n_complex_for_matrix, l_vec)
    M_2qwot_batch = calculate_M_batch_for_thickness(ep2, n_complex_for_matrix, l_vec)
    return M_1qwot_batch, M_2qwot_batch
@jax.jit
def compute_half_product(multiplier_indices: jnp.ndarray,
                         layer_matrices_half: jnp.ndarray
                         ) -> jnp.ndarray:
    N_half = layer_matrices_half.shape[0]
    L = layer_matrices_half.shape[2]
    init_prod = jnp.tile(jnp.eye(2, dtype=jnp.complex128), (L, 1, 1))
    def multiply_step(carry_prod: jnp.ndarray, layer_idx: int) -> Tuple[jnp.ndarray, None]:
        multiplier_idx = multiplier_indices[layer_idx]
        M_k = layer_matrices_half[layer_idx, multiplier_idx, :, :, :]
        new_prod = vmap(jnp.matmul)(M_k, carry_prod)
        return new_prod, None
    final_prod, _ = jax.lax.scan(multiply_step, init_prod, jnp.arange(N_half))
    return final_prod
@jax.jit
def get_T_from_batch_matrix(M_batch: jnp.ndarray,
                            nSub_arr: jnp.ndarray
                            ) -> jnp.ndarray:
    etainc = 1.0 + 0j
    etasub_batch = nSub_arr
    m00 = M_batch[..., 0, 0]; m01 = M_batch[..., 0, 1]
    m10 = M_batch[..., 1, 0]; m11 = M_batch[..., 1, 1]
    rs_den = (etainc * m00 + etasub_batch * m11 + etainc * etasub_batch * m01 + m10)
    rs_den_abs = jnp.abs(rs_den)
    safe_den = jnp.where(rs_den_abs < 1e-12, 1e-12 + 0j, rs_den)
    ts = (2.0 * etainc) / safe_den
    real_etasub_batch = jnp.real(etasub_batch)
    safe_real_etainc = 1.0
    Ts_complex = (real_etasub_batch / safe_real_etainc) * (ts * jnp.conj(ts))
    Ts = jnp.real(Ts_complex)
    return jnp.where(rs_den_abs < 1e-12, 0.0, jnp.nan_to_num(Ts, nan=0.0))
@jax.jit
def calculate_mse_basic_jax(Ts: jnp.ndarray,
                            l_vec: jnp.ndarray,
                            targets_tuple: Tuple[Tuple[float, float, float, float], ...]
                            ) -> jnp.ndarray:
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
    mse = jnp.where(total_points_in_targets > 0,
                     total_squared_error / total_points_in_targets,
                     jnp.inf)
    return jnp.nan_to_num(mse, nan=jnp.inf, posinf=jnp.inf)
@jax.jit
def combine_and_calc_mse(prod1: jnp.ndarray, prod2: jnp.ndarray,
                         nSub_arr_in: jnp.ndarray,
                         l_vec_in: jnp.ndarray, targets_tuple_in: Tuple
                         ) -> jnp.ndarray:
    M_total = vmap(jnp.matmul)(prod2, prod1)
    Ts = get_T_from_batch_matrix(M_total, nSub_arr_in)
    mse = calculate_mse_basic_jax(Ts, l_vec_in, targets_tuple_in)
    return mse
def _execute_split_stack_scan(current_l0: float, initial_layer_number: int,
                              nH_c_l0: complex, nL_c_l0: complex,
                              nSub_arr_scan: jnp.ndarray,
                              l_vec_eval_sparse_jax: jnp.ndarray,
                              active_targets_tuple: Tuple,
                              log_callback: Callable) -> Tuple[float, Optional[np.ndarray], List[str]]:
    logs = []
    L_sparse = len(l_vec_eval_sparse_jax)
    num_combinations = 2**initial_layer_number
    log_callback(f"  [Scan l0={current_l0:.2f}] Test {num_combinations:,} comb. QWOT (1.0/2.0)...")
    precompute_start_time = time.time()
    st.write(f"Scan l0={current_l0:.1f}: Précalcul matrices...")
    layer_matrices_list = []
    try:
        get_layer_matrices_qwot_jit = jax.jit(get_layer_matrices_qwot)
        for i in range(initial_layer_number):
            m1, m2 = get_layer_matrices_qwot_jit(i, initial_layer_number, current_l0,
                                                jnp.asarray(nH_c_l0), jnp.asarray(nL_c_l0),
                                                l_vec_eval_sparse_jax)
            layer_matrices_list.append(jnp.stack([m1, m2], axis=0))
        all_layer_matrices = jnp.stack(layer_matrices_list, axis=0)
        all_layer_matrices.block_until_ready()
        log_callback(f"    Précalcul matrices (l0={current_l0:.2f}) terminé en {time.time() - precompute_start_time:.3f}s.")
    except Exception as e_mat:
        logs.append(f"  ERREUR Précalcul Matrices pour l0={current_l0:.2f}: {e_mat}")
        st.error(f"Erreur précalcul matrices QWOT scan: {e_mat}")
        return np.inf, None, logs
    N = initial_layer_number
    N1 = N // 2
    N2 = N - N1
    num_comb1 = 2**N1
    num_comb2 = 2**N2
    log_callback(f"    Calcul produits partiels 1 ({num_comb1:,} comb)...")
    st.write(f"Scan l0={current_l0:.1f}: Produits partiels 1...")
    half1_start_time = time.time()
    indices1 = jnp.arange(num_comb1)
    powers1 = 2**jnp.arange(N1)
    multiplier_indices1 = jnp.not_equal(indices1[:, None] & powers1, 0).astype(jnp.int32)
    matrices_half1 = all_layer_matrices[:N1]
    compute_half_product_jit = jax.jit(compute_half_product)
    partial_products1 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices1, matrices_half1)
    partial_products1.block_until_ready()
    log_callback(f"    Produits partiels 1 terminés en {time.time() - half1_start_time:.3f}s.")
    log_callback(f"    Calcul produits partiels 2 ({num_comb2:,} comb)...")
    st.write(f"Scan l0={current_l0:.1f}: Produits partiels 2...")
    half2_start_time = time.time()
    indices2 = jnp.arange(num_comb2)
    powers2 = 2**jnp.arange(N2)
    multiplier_indices2 = jnp.not_equal(indices2[:, None] & powers2, 0).astype(jnp.int32)
    matrices_half2 = all_layer_matrices[N1:]
    partial_products2 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices2, matrices_half2)
    partial_products2.block_until_ready()
    log_callback(f"    Produits partiels 2 terminés en {time.time() - half2_start_time:.3f}s.")
    log_callback(f"    Combinaison et calcul MSE ({num_comb1 * num_comb2:,} total)...")
    st.write(f"Scan l0={current_l0:.1f}: Combinaison & MSE...")
    combine_start_time = time.time()
    combine_and_calc_mse_jit = jax.jit(combine_and_calc_mse)
    vmap_inner = vmap(combine_and_calc_mse_jit, in_axes=(None, 0, None, None, None))
    vmap_outer = vmap(vmap_inner, in_axes=(0, None, None, None, None))
    all_mses_nested = vmap_outer(partial_products1, partial_products2, nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple)
    all_mses_nested.block_until_ready()
    log_callback(f"    Combinaison et MSE terminés en {time.time() - combine_start_time:.3f}s.")
    all_mses_flat = all_mses_nested.reshape(-1)
    best_idx_flat = jnp.argmin(all_mses_flat)
    current_best_mse = float(all_mses_flat[best_idx_flat])
    if not np.isfinite(current_best_mse):
         logs.append(f"    Avertissement: Aucun résultat valide (MSE fini) trouvé pour l0={current_l0:.2f}.")
         return np.inf, None, logs
    best_idx_half1, best_idx_half2 = jnp.unravel_index(best_idx_flat, (num_comb1, num_comb2))
    best_indices_h1 = multiplier_indices1[best_idx_half1]
    best_indices_h2 = multiplier_indices2[best_idx_half2]
    best_multipliers_h1 = 1.0 + best_indices_h1.astype(jnp.float64)
    best_multipliers_h2 = 1.0 + best_indices_h2.astype(jnp.float64)
    current_best_multipliers = jnp.concatenate([best_multipliers_h1, best_multipliers_h2])
    logs.append(f"    Meilleur MSE pour scan l0={current_l0:.2f}: {current_best_mse:.6e}")
    return current_best_mse, np.array(current_best_multipliers), logs

def add_log(message: Union[str, List[str]]):
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    MAX_LOG_LINES = 500
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    new_logs = []
    if isinstance(message, list):
        for msg in message:
            new_logs.append(f"[{timestamp}] {str(msg)}")
    else:
        new_logs.append(f"[{timestamp}] {str(message)}")
    st.session_state.log_messages = new_logs + st.session_state.log_messages
    if len(st.session_state.log_messages) > MAX_LOG_LINES:
        st.session_state.log_messages = st.session_state.log_messages[:MAX_LOG_LINES]
def get_material_input(role: str) -> Tuple[Optional[MaterialInputType], str]:
    if role == 'H':
        sel_key, const_r_key, const_i_key = "selected_H", "nH_r", "nH_i"
    elif role == 'L':
        sel_key, const_r_key, const_i_key = "selected_L", "nL_r", "nL_i"
    elif role == 'Sub':
        sel_key, const_r_key, const_i_key = "selected_Sub", "nSub_r", None
    else:
        st.error(f"Rôle matériau inconnu : {role}")
        return None, "Erreur Rôle"
    selection = st.session_state.get(sel_key)
    if selection == "Constant":
        n_real = st.session_state.get(const_r_key, 1.0 if role != 'Sub' else 1.5)
        n_imag = 0.0
        if const_i_key and role in ['H', 'L']:
             n_imag = st.session_state.get(const_i_key, 0.0)
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
        return selection, selection
    else:
        st.error(f"Sélection matériau pour '{role}' invalide ou manquante dans session_state.")
        add_log(f"Erreur critique: Sélection matériau '{role}' invalide: {selection}")
        return None, "Erreur Sélection"
def validate_targets() -> Optional[List[Dict]]:
    active_targets = []
    logs = []
    is_valid = True
    if 'targets' not in st.session_state or not isinstance(st.session_state.targets, list):
        st.error("Erreur interne : Liste de cibles manquante ou invalide dans session_state.")
        return None
    for i, target_state in enumerate(st.session_state.targets):
        if target_state.get('enabled', False):
            try:
                l_min = float(target_state['min'])
                l_max = float(target_state['max'])
                t_min = float(target_state['target_min'])
                t_max = float(target_state['target_max'])
                if l_max < l_min:
                    logs.append(f"Erreur Cible {i+1}: λ max ({l_max:.1f}) < λ min ({l_min:.1f}).")
                    is_valid = False; continue
                if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0):
                    logs.append(f"Erreur Cible {i+1}: Transmittance hors de [0, 1] (Tmin={t_min:.2f}, Tmax={t_max:.2f}).")
                    is_valid = False; continue
                active_targets.append({
                    'min': l_min, 'max': l_max,
                    'target_min': t_min, 'target_max': t_max
                })
            except (KeyError, ValueError, TypeError) as e:
                logs.append(f"Erreur Cible {i+1}: Données manquantes ou invalides ({e}).")
                is_valid = False; continue
    if not is_valid:
        add_log(["Erreurs détectées dans la définition des cibles actives:"] + logs)
        st.warning("Des erreurs existent dans la définition des cibles spectrales actives. Veuillez corriger.")
        return None
    elif not active_targets:
        add_log("Aucune cible spectrale n'est activée.")
        return []
    else:
        add_log(f"{len(active_targets)} cible(s) active(s) et valide(s) trouvée(s).")
        return active_targets
def get_lambda_range_from_targets(validated_targets: Optional[List[Dict]]) -> Tuple[Optional[float], Optional[float]]:
    overall_min, overall_max = None, None
    if validated_targets:
        all_mins = [t['min'] for t in validated_targets]
        all_maxs = [t['max'] for t in validated_targets]
        if all_mins: overall_min = min(all_mins)
        if all_maxs: overall_max = max(all_maxs)
    return overall_min, overall_max
def clear_optimized_state():
    add_log("Nettoyage de l'état optimisé et de l'historique.")
    st.session_state.optimized_ep = None
    st.session_state.is_optimized_state = False
    st.session_state.ep_history = deque(maxlen=5)
    st.session_state.optimized_qwot_str = ""
    st.session_state.last_mse = None
def set_optimized_as_nominal_wrapper():
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
        else:
             new_qwot_str = ",".join([f"{q:.6f}" for q in optimized_qwots])
             st.session_state.current_qwot = new_qwot_str
             add_log(f"QWOT Nominal mis à jour : {new_qwot_str}")
             st.success("Structure optimisée définie comme nouveau Nominal (QWOT mis à jour).")
             clear_optimized_state()
    except Exception as e:
        st.error(f"Erreur inattendue lors de la définition de l'optimisé comme nominal: {e}")
        add_log(f"Erreur inattendue (set_optimized_as_nominal): {e}\n{traceback.format_exc(limit=1)}")
def undo_remove_wrapper():
    add_log("Tentative d'annulation de la dernière suppression...")
    if not st.session_state.get('ep_history'):
        st.info("Historique d'annulation vide.")
        add_log("Historique vide, annulation impossible.")
        return
    try:
        last_ep = st.session_state.ep_history.pop()
        st.session_state.optimized_ep = last_ep.copy()
        st.session_state.is_optimized_state = True
        add_log(f"État restauré ({len(last_ep)} couches). {len(st.session_state.ep_history)} états restants dans l'historique.")
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
        st.info("État restauré. Recalcul en cours...")
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {
             'is_optimized_run': True,
             'method_name': "Optimized (Undo)",
             'force_ep': st.session_state.optimized_ep
             }
    except IndexError:
         st.warning("Historique d'annulation vide (erreur interne?).")
         add_log("Erreur: Tentative de pop sur historique vide.")
    except Exception as e:
        st.error(f"Erreur inattendue lors de l'annulation: {e}")
        add_log(f"Erreur inattendue (undo_remove): {e}\n{traceback.format_exc(limit=1)}")
        clear_optimized_state()
def run_calculation_wrapper(is_optimized_run: bool, method_name: str = "", force_ep: Optional[np.ndarray] = None):
    calc_type = 'Optimisé' if is_optimized_run else 'Nominal'
    add_log(f"\n{'='*10} Démarrage Calcul {calc_type} {'('+method_name+')' if method_name else ''} {'='*10}")
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None
    with st.spinner(f"Calcul {calc_type} en cours..."):
        try:
            active_targets = validate_targets()
            if active_targets is None:
                 st.error("Définition des cibles invalide. Vérifiez les logs et corrigez.")
                 add_log("Calcul annulé: Cibles invalides.")
                 return
            if not active_targets:
                 st.warning("Aucune cible active. Plage lambda par défaut utilisée (400-700nm). Le calcul MSE sera N/A.")
                 l_min_plot, l_max_plot = 400.0, 700.0
            else:
                 l_min_plot, l_max_plot = get_lambda_range_from_targets(active_targets)
                 if l_min_plot is None or l_max_plot is None or l_max_plot < l_min_plot:
                     st.error("Impossible de déterminer une plage lambda valide depuis les cibles.")
                     add_log("Calcul annulé: Plage lambda invalide depuis cibles.")
                     return
            validated_inputs = {
                'l0': st.session_state.l0,
                'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot,
                'maxiter': st.session_state.maxiter,
                'maxfun': st.session_state.maxfun,
                'auto_thin_threshold': st.session_state.auto_thin_threshold,
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
            ep_to_calculate = None
            if force_ep is not None:
                 ep_to_calculate = force_ep.copy()
                 add_log("Utilisation d'un vecteur ep forcé.")
            elif is_optimized_run and st.session_state.get('optimized_ep') is not None:
                 ep_to_calculate = st.session_state.optimized_ep.copy()
                 add_log("Utilisation de la structure optimisée actuelle.")
            else:
                 add_log("Utilisation de la structure nominale (QWOT).")
                 emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
                 if not emp_list and calc_type == 'Nominal':
                     add_log("QWOT Nominal vide, calcul pour substrat nu.")
                     ep_to_calculate = np.array([], dtype=np.float64)
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
            st.session_state.current_ep = ep_to_calculate.copy() if ep_to_calculate is not None else None
            num_plot_points = max(501, int(np.round((l_max_plot - l_min_plot) / validated_inputs['l_step'])) * 3 + 1)
            l_vec_plot_fine_np = np.linspace(l_min_plot, l_max_plot, num_plot_points)
            l_vec_plot_fine_np = l_vec_plot_fine_np[(l_vec_plot_fine_np > 0) & np.isfinite(l_vec_plot_fine_np)]
            if not l_vec_plot_fine_np.size:
                 st.error("Impossible de générer un vecteur lambda valide pour le tracé.")
                 add_log("Calcul annulé: vecteur lambda pour plot invalide.")
                 return
            add_log(f"Calcul T(lambda) sur {len(l_vec_plot_fine_np)} points pour le tracé [{l_min_plot:.1f}-{l_max_plot:.1f} nm].")
            start_calc_time = time.time()
            results_fine, calc_logs = calculate_T_from_ep_jax(
                ep_to_calculate, nH_mat, nL_mat, nSub_mat, l_vec_plot_fine_np, EXCEL_FILE_PATH
            )
            add_log(calc_logs)
            if results_fine is None:
                 st.error("Le calcul principal de la transmittance a échoué.")
                 add_log("Erreur critique: calculate_T_from_ep_jax a retourné None.")
                 return
            add_log(f"Calcul T(lambda) terminé en {time.time() - start_calc_time:.3f}s.")
            st.session_state.last_calc_results = {
                'res_fine': results_fine,
                'method_name': method_name,
                'ep_used': ep_to_calculate.copy() if ep_to_calculate is not None else None,
                'l0_used': validated_inputs['l0'],
                'nH_used': nH_mat, 'nL_used': nL_mat, 'nSub_used': nSub_mat,
            }
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
                          st.session_state.last_calc_results['res_optim_grid'] = results_optim_grid
                      else:
                          add_log("Échec calcul T sur grille optim pour MSE.")
                          st.session_state.last_mse = None
                 else:
                      add_log("Grille d'optimisation vide, MSE non calculé.")
                      st.session_state.last_mse = None
            else:
                 add_log("Pas de cibles actives, MSE non calculé.")
                 st.session_state.last_mse = None
            st.session_state.is_optimized_state = is_optimized_run
            if not is_optimized_run:
                 clear_optimized_state()
                 st.session_state.current_ep = ep_to_calculate.copy() if ep_to_calculate is not None else None
            st.success(f"Calcul {calc_type} terminé.")
            add_log(f"--- Fin Calcul {calc_type} ---")
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Erreur durant le calcul {calc_type}: {e}")
            add_log(f"ERREUR (Calcul {calc_type}): {e}\n{traceback.format_exc(limit=1)}")
        except Exception as e_fatal:
             st.error(f"Erreur inattendue durant le calcul {calc_type}: {e_fatal}")
             add_log(f"ERREUR FATALE (Calcul {calc_type}): {e_fatal}\n{traceback.format_exc()}")
def run_local_optimization_wrapper():
    add_log(f"\n{'='*10} Démarrage Optimisation Locale {'='*10}")
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None
    clear_optimized_state()
    with st.spinner("Optimisation locale en cours..."):
        try:
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
            validated_inputs = {
                'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                'maxiter': st.session_state.maxiter, 'maxfun': st.session_state.maxfun,
                'emp_str': st.session_state.current_qwot,
                'auto_thin_threshold': st.session_state.auto_thin_threshold,
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
            final_ep, success, final_cost, optim_logs, msg, nit, nfev = \
                _run_core_optimization(ep_start, validated_inputs, active_targets,
                                       MIN_THICKNESS_PHYS_NM, log_prefix="  [Opt Local] ")
            add_log(optim_logs)
            if success and final_ep is not None:
                 st.session_state.optimized_ep = final_ep.copy()
                 st.session_state.current_ep = final_ep.copy()
                 st.session_state.is_optimized_state = True
                 st.session_state.last_mse = final_cost
                 qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                 add_log(logs_qwot)
                 if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                     st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                 else:
                     st.session_state.optimized_qwot_str = "QWOT N/A"
                 st.success(f"Optimisation locale terminée ({msg}). MSE: {final_cost:.4e}")
                 add_log(f"--- Fin Optimisation Locale (Succès) ---")
                 st.session_state.needs_rerun_calc = True
                 st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Opt Local ({nit}/{nfev})"}
            else:
                 st.error(f"L'optimisation locale a échoué: {msg}")
                 add_log(f"--- Fin Optimisation Locale (Échec) ---")
                 st.session_state.is_optimized_state = False
                 st.session_state.optimized_ep = None
                 st.session_state.current_ep = ep_start.copy()
                 st.session_state.last_mse = None
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Erreur durant l'optimisation locale: {e}")
            add_log(f"ERREUR (Opt Locale): {e}\n{traceback.format_exc(limit=1)}")
            clear_optimized_state()
        except Exception as e_fatal:
             st.error(f"Erreur inattendue durant l'optimisation locale: {e_fatal}")
             add_log(f"ERREUR FATALE (Opt Locale): {e_fatal}\n{traceback.format_exc()}")
             clear_optimized_state()
def run_scan_optimization_wrapper():
    add_log(f"\n{'#'*10} Démarrage Scan QWOT + Optimisation {'#'*10}")
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None
    clear_optimized_state()
    with st.spinner("Scan QWOT + Optimisation en cours (peut être long)..."):
        try:
            if 'initial_layer_number' not in st.session_state:
                 st.session_state.initial_layer_number = len([q for q in st.session_state.current_qwot.split(',') if q.strip()])
                 if st.session_state.initial_layer_number == 0:
                     st.error("Le QWOT nominal est vide et le nombre initial de couches n'est pas défini.")
                     add_log("Erreur Scan+Opt: Nombre initial de couches non défini.")
                     return
            active_targets = validate_targets()
            if active_targets is None or not active_targets:
                 st.error("Scan QWOT+Opt nécessite des cibles actives et valides.")
                 add_log("Scan QWOT+Opt annulé: cibles invalides ou manquantes.")
                 return
            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None:
                 st.error("Impossible de déterminer la plage lambda pour le Scan QWOT+Opt.")
                 add_log("Scan QWOT+Opt annulé: plage lambda invalide.")
                 return
            validated_inputs = {
                 'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                 'maxiter': st.session_state.maxiter, 'maxfun': st.session_state.maxfun,
                 'emp_str': st.session_state.current_qwot,
                 'initial_layer_number': st.session_state.initial_layer_number,
                 'auto_thin_threshold': st.session_state.auto_thin_threshold,
                 'l_range_deb': l_min_opt, 'l_range_fin': l_max_opt,
              }
            nH_mat, nH_repr = get_material_input('H')
            nL_mat, nL_repr = get_material_input('L')
            nSub_mat, nSub_repr = get_material_input('Sub')
            if nH_mat is None or nL_mat is None or nSub_mat is None:
                 st.error("Erreur définition matériau pour Scan QWOT+Opt.")
                 add_log("Scan QWOT+Opt annulé: erreur matériau.")
                 return
            validated_inputs['nH_material'] = nH_mat
            validated_inputs['nL_material'] = nL_mat
            validated_inputs['nSub_material'] = nSub_mat
            add_log(f"Matériaux Scan+Opt: H={nH_repr}, L={nL_repr}, Sub={nSub_repr}")
            add_log(f"Scan pour N={validated_inputs['initial_layer_number']} couches.")
            l_vec_eval_full_np = np.geomspace(l_min_opt, l_max_opt, max(2, int(np.round((l_max_opt - l_min_opt) / validated_inputs['l_step'])) + 1))
            l_vec_eval_full_np = l_vec_eval_full_np[(l_vec_eval_full_np > 0) & np.isfinite(l_vec_eval_full_np)]
            if not l_vec_eval_full_np.size: raise ValueError("Échec génération lambda pour Scan.")
            l_vec_eval_sparse_np = l_vec_eval_full_np[::2]
            if not l_vec_eval_sparse_np.size: raise ValueError("Échec génération lambda sparse pour Scan.")
            l_vec_eval_sparse_jax = jnp.asarray(l_vec_eval_sparse_np)
            add_log(f"Utilisation grille de scan à {len(l_vec_eval_sparse_jax)} points.")
            nSub_arr_scan, logs_sub_scan = _get_nk_array_for_lambda_vec(nSub_mat, l_vec_eval_sparse_jax, EXCEL_FILE_PATH)
            add_log(logs_sub_scan)
            if nSub_arr_scan is None: raise RuntimeError("Echec préparation indice substrat pour scan.")
            active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
            l0_nominal = validated_inputs['l0']
            l0_values_to_test = sorted(list(set([l0_nominal, l0_nominal * 1.2, l0_nominal * 0.8])))
            l0_values_to_test = [l for l in l0_values_to_test if l > 1e-6]
            add_log(f"Scan QWOT testera l0 = {[f'{l:.1f}' for l in l0_values_to_test]} nm.")
            initial_candidates = []
            overall_scan_logs = []
            for l0_scan in l0_values_to_test:
                 add_log(f"--- Scan QWOT pour l0 = {l0_scan:.2f} ---")
                 st.write(f"Scan pour l0={l0_scan:.1f}...")
                 try:
                     nH_c_l0, log_h_l0 = _get_nk_at_lambda(nH_mat, l0_scan, EXCEL_FILE_PATH)
                     nL_c_l0, log_l_l0 = _get_nk_at_lambda(nL_mat, l0_scan, EXCEL_FILE_PATH)
                     overall_scan_logs.extend(log_h_l0); overall_scan_logs.extend(log_l_l0)
                     if nH_c_l0 is None or nL_c_l0 is None:
                          add_log(f"AVERTISSEMENT: Indices H/L non trouvés pour l0={l0_scan:.2f}. Scan pour ce l0 ignoré.")
                          continue
                     scan_mse, scan_multipliers, scan_logs = _execute_split_stack_scan(
                          l0_scan, validated_inputs['initial_layer_number'],
                          nH_c_l0, nL_c_l0,
                          nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple,
                          add_log
                      )
                     overall_scan_logs.extend(scan_logs)
                     if scan_multipliers is not None and np.isfinite(scan_mse):
                          initial_candidates.append({
                              'l0': l0_scan,
                              'mse_scan': scan_mse,
                              'multipliers': scan_multipliers
                          })
                          add_log(f"Candidat trouvé pour l0={l0_scan:.2f} avec MSE (scan)={scan_mse:.4e}")
                     else:
                         add_log(f"Aucun candidat valide trouvé pour l0={l0_scan:.2f}")
                 except Exception as e_scan_l0:
                      add_log(f"Erreur durant scan pour l0={l0_scan:.2f}: {e_scan_l0}")
                      st.warning(f"Erreur durant scan pour l0={l0_scan:.2f}: {e_scan_l0}")
            if not initial_candidates:
                 st.error("Le Scan QWOT n'a trouvé aucun candidat initial valide.")
                 add_log("Erreur critique: Scan QWOT vide.")
                 return
            add_log(f"\n--- QWOT Scan finished. Found {len(initial_candidates)} candidate(s). Running Local Optimization for each. ---")
            st.write("Optimisation locale du meilleur candidat du scan...")
            initial_candidates.sort(key=lambda c: c['mse_scan'])
            best_candidate = initial_candidates[0]
            add_log(f"\nMeilleur candidat du scan: l0={best_candidate['l0']:.2f}, MSE(scan)={best_candidate['mse_scan']:.4e}")
            add_log("Lancement optimisation locale pour ce candidat...")
            ep_start_optim, logs_ep_best = calculate_initial_ep(best_candidate['multipliers'], best_candidate['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
            add_log(logs_ep_best)
            if ep_start_optim is None:
                st.error("Échec calcul épaisseurs pour le meilleur candidat du scan.")
                add_log("Erreur critique: ep_start_optim est None pour le meilleur candidat.")
                return
            final_ep, success, final_cost, optim_logs, msg, nit, nfev = \
                 _run_core_optimization(ep_start_optim, validated_inputs, active_targets,
                                        MIN_THICKNESS_PHYS_NM, log_prefix="  [Opt Scan Cand] ")
            add_log(optim_logs)
            if success and final_ep is not None:
                 st.session_state.optimized_ep = final_ep.copy()
                 st.session_state.current_ep = final_ep.copy()
                 st.session_state.is_optimized_state = True
                 st.session_state.last_mse = final_cost
                 st.session_state.l0 = best_candidate['l0']
                 qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, best_candidate['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                 add_log(logs_qwot)
                 if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                     st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                 else: st.session_state.optimized_qwot_str = "QWOT N/A"
                 st.success(f"Scan QWOT + Optimisation terminé ({msg}). MSE final: {final_cost:.4e}")
                 add_log(f"--- Fin Scan QWOT + Optimisation (Succès) ---")
                 st.session_state.needs_rerun_calc = True
                 st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Scan+Opt (l0={best_candidate['l0']:.1f}, {nit}/{nfev})"}
            else:
                 st.error(f"L'optimisation locale après le scan a échoué: {msg}")
                 add_log(f"--- Fin Scan QWOT + Optimisation (Échec Optim Finale) ---")
                 clear_optimized_state()
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Erreur durant le Scan QWOT + Optimisation: {e}")
            add_log(f"ERREUR (Scan+Opt): {e}\n{traceback.format_exc(limit=1)}")
            clear_optimized_state()
        except Exception as e_fatal:
            st.error(f"Erreur inattendue durant Scan QWOT + Optimisation: {e_fatal}")
            add_log(f"ERREUR FATALE (Scan+Opt): {e_fatal}\n{traceback.format_exc()}")
            clear_optimized_state()
        finally:
            pass
def run_auto_mode_wrapper():
    add_log(f"\n{'#'*10} Démarrage Mode Auto {'#'*10}")
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None
    with st.spinner("Mode Automatique en cours (peut être très long)..."):
        try:
            active_targets = validate_targets()
            if active_targets is None or not active_targets:
                 st.error("Mode Auto nécessite des cibles actives et valides.")
                 add_log("Mode Auto annulé: cibles invalides ou manquantes.")
                 return
            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None:
                 st.error("Impossible de déterminer la plage lambda pour Mode Auto.")
                 add_log("Mode Auto annulé: plage lambda invalide.")
                 return
            validated_inputs = {
                 'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                 'maxiter': st.session_state.maxiter, 'maxfun': st.session_state.maxfun,
                 'emp_str': st.session_state.current_qwot,
                 'auto_thin_threshold': st.session_state.auto_thin_threshold,
                 'l_range_deb': l_min_opt, 'l_range_fin': l_max_opt,
              }
            nH_mat, nH_repr = get_material_input('H')
            nL_mat, nL_repr = get_material_input('L')
            nSub_mat, nSub_repr = get_material_input('Sub')
            if nH_mat is None or nL_mat is None or nSub_mat is None:
                 st.error("Erreur définition matériau pour Mode Auto.")
                 add_log("Mode Auto annulé: erreur matériau.")
                 return
            validated_inputs['nH_material'] = nH_mat
            validated_inputs['nL_material'] = nL_mat
            validated_inputs['nSub_material'] = nSub_mat
            add_log(f"Matériaux Auto: H={nH_repr}, L={nL_repr}, Sub={nSub_repr}")
            ep_start_auto = None
            if st.session_state.get('is_optimized_state') and st.session_state.get('optimized_ep') is not None:
                 ep_start_auto = st.session_state.optimized_ep.copy()
                 add_log("Mode Auto démarre depuis l'état optimisé précédent.")
            final_ep, final_mse, auto_logs, total_nit, total_nfev = run_auto_mode(
                initial_ep=ep_start_auto,
                validated_inputs=validated_inputs,
                active_targets=active_targets,
                excel_file_path=EXCEL_FILE_PATH,
                log_callback=add_log
            )
            add_log(auto_logs)
            if final_ep is not None and np.isfinite(final_mse):
                 st.session_state.optimized_ep = final_ep.copy()
                 st.session_state.current_ep = final_ep.copy()
                 st.session_state.is_optimized_state = True
                 st.session_state.last_mse = final_mse
                 qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                 add_log(logs_qwot)
                 if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                     st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                 else: st.session_state.optimized_qwot_str = "QWOT N/A"
                 st.success(f"Mode Auto terminé. MSE final: {final_mse:.4e}")
                 add_log(f"--- Fin Mode Auto (Succès) ---")
                 st.session_state.needs_rerun_calc = True
                 st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Mode Auto ({total_nit}/{total_nfev} tot)"}
            else:
                 st.error("Le Mode Automatique a échoué ou n'a pas produit de résultat valide.")
                 add_log(f"--- Fin Mode Auto (Échec) ---")
                 clear_optimized_state()
                 st.session_state.needs_rerun_calc = True
                 st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (Après Échec Auto)"}
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Erreur durant le Mode Auto: {e}")
            add_log(f"ERREUR (Mode Auto): {e}\n{traceback.format_exc(limit=1)}")
        except Exception as e_fatal:
            st.error(f"Erreur inattendue durant le Mode Auto: {e_fatal}")
            add_log(f"ERREUR FATALE (Mode Auto): {e_fatal}\n{traceback.format_exc()}")
        finally:
            pass
def run_remove_thin_wrapper():
    add_log(f"\n{'-'*10} Tentative Suppression Couche Fine + Ré-Optimisation {'-'*10}")
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None
    if not st.session_state.get('is_optimized_state') or st.session_state.get('optimized_ep') is None:
        st.error("Cette action nécessite une structure optimisée existante. Lancez d'abord une optimisation.")
        add_log("Erreur 'Remove Thin': Pas d'état optimisé.")
        return
    current_ep_optim = st.session_state.optimized_ep.copy()
    if len(current_ep_optim) <= 2:
        st.error("Structure trop petite (<= 2 couches) pour suppression/fusion.")
        add_log("Erreur 'Remove Thin': Structure trop petite.")
        return
    with st.spinner("Suppression couche fine + Ré-optimisation en cours..."):
        try:
            st.session_state.ep_history.append(current_ep_optim)
            add_log(f"  [Undo] État sauvegardé ({len(current_ep_optim)} couches). Hist: {len(st.session_state.ep_history)}")
            active_targets = validate_targets()
            if active_targets is None or not active_targets:
                 st.session_state.ep_history.pop()
                 st.error("Suppression annulée: cibles invalides ou manquantes pour ré-optimisation.")
                 add_log("Erreur 'Remove Thin': cibles invalides.")
                 return
            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None:
                 st.session_state.ep_history.pop()
                 st.error("Suppression annulée: plage lambda invalide pour ré-optimisation.")
                 add_log("Erreur 'Remove Thin': plage lambda invalide.")
                 return
            validated_inputs = {
                 'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                 'maxiter': st.session_state.maxiter, 'maxfun': st.session_state.maxfun,
                 'emp_str': st.session_state.current_qwot,
                 'auto_thin_threshold': st.session_state.auto_thin_threshold,
                 'l_range_deb': l_min_opt, 'l_range_fin': l_max_opt,
              }
            nH_mat, nH_repr = get_material_input('H')
            nL_mat, nL_repr = get_material_input('L')
            nSub_mat, nSub_repr = get_material_input('Sub')
            if nH_mat is None or nL_mat is None or nSub_mat is None:
                 st.session_state.ep_history.pop()
                 st.error("Suppression annulée: erreur définition matériau.")
                 add_log("Erreur 'Remove Thin': erreur matériau.")
                 return
            validated_inputs['nH_material'] = nH_mat
            validated_inputs['nL_material'] = nL_mat
            validated_inputs['nSub_material'] = nSub_mat
            threshold = validated_inputs['auto_thin_threshold']
            add_log(f"Recherche de la couche la plus fine >= {MIN_THICKNESS_PHYS_NM:.3f} nm...")
            ep_after_removal, structure_changed, removal_logs = _perform_layer_merge_or_removal_only(
                current_ep_optim, MIN_THICKNESS_PHYS_NM,
                log_prefix="  [Remove] ",
                threshold_for_removal=None
            )
            add_log(removal_logs)
            if structure_changed and ep_after_removal is not None:
                add_log(f"Structure modifiée ({len(ep_after_removal)} couches). Ré-optimisation...")
                st.write("Ré-optimisation après suppression...")
                final_ep, success, final_cost, optim_logs, msg, nit, nfev = \
                     _run_core_optimization(ep_after_removal, validated_inputs, active_targets,
                                            MIN_THICKNESS_PHYS_NM, log_prefix="  [ReOpt Thin] ")
                add_log(optim_logs)
                if success and final_ep is not None:
                     st.session_state.optimized_ep = final_ep.copy()
                     st.session_state.current_ep = final_ep.copy()
                     st.session_state.is_optimized_state = True
                     st.session_state.last_mse = final_cost
                     qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                     add_log(logs_qwot)
                     if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                         st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                     else: st.session_state.optimized_qwot_str = "QWOT N/A"
                     st.success(f"Suppression + Ré-optimisation terminé ({msg}). MSE final: {final_cost:.4e}")
                     add_log(f"--- Fin Suppression+Ré-Opt (Succès) ---")
                     st.session_state.needs_rerun_calc = True
                     st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Optimized (Post-Remove, {nit}/{nfev})"}
                else:
                     st.warning(f"Couche supprimée, mais la ré-optimisation a échoué ({msg}). L'état est celui APRES suppression mais AVANT tentative de ré-opt.")
                     add_log("AVERTISSEMENT: Ré-optimisation après suppression échouée.")
                     st.session_state.optimized_ep = ep_after_removal.copy()
                     st.session_state.current_ep = ep_after_removal.copy()
                     st.session_state.is_optimized_state = True
                     try:
                          l_min_opt_recalc, l_max_opt_recalc = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
                          l_step_optim = validated_inputs['l_step']
                          num_pts_optim = max(2, int(np.round((l_max_opt_recalc - l_min_opt_recalc) / l_step_optim)) + 1)
                          l_vec_optim_np_recalc = np.geomspace(l_min_opt_recalc, l_max_opt_recalc, num_pts_optim)
                          l_vec_optim_np_recalc = l_vec_optim_np_recalc[(l_vec_optim_np_recalc > 0) & np.isfinite(l_vec_optim_np_recalc)]
                          if l_vec_optim_np_recalc.size > 0:
                              results_fail_grid, _ = calculate_T_from_ep_jax(ep_after_removal, nH_mat, nL_mat, nSub_mat, l_vec_optim_np_recalc, EXCEL_FILE_PATH)
                              if results_fail_grid:
                                  mse_fail, _ = calculate_final_mse(results_fail_grid, active_targets)
                                  st.session_state.last_mse = mse_fail
                                  add_log(f"MSE (après suppression, avant re-opt échoué): {mse_fail:.4e}" if mse_fail is not None else "MSE N/A")
                              else: st.session_state.last_mse = None
                          qwots_fail, _ = calculate_qwot_from_ep(ep_after_removal, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                          if qwots_fail is not None and not np.any(np.isnan(qwots_fail)):
                              st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_fail])
                          else: st.session_state.optimized_qwot_str = "QWOT N/A (ReOpt Fail)"
                     except Exception as e_recalc:
                          add_log(f"Erreur recalcul QWOT/MSE après échec re-opt: {e_recalc}")
                          st.session_state.last_mse = None
                          st.session_state.optimized_qwot_str = "Erreur Recalc"
                     add_log(f"--- Fin Suppression+Ré-Opt (Échec Ré-Opt) ---")
                     st.session_state.needs_rerun_calc = True
                     st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': "Optimized (Post-Remove, Re-Opt Fail)"}
            else:
                 st.info("Aucune couche n'a été supprimée (critères non remplis).")
                 add_log("Structure non modifiée par la tentative de suppression.")
                 try:
                      st.session_state.ep_history.pop()
                      add_log("  [Undo] État historique inutile retiré.")
                 except IndexError: pass
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Erreur durant Suppression Couche Fine: {e}")
            add_log(f"ERREUR (Remove Thin): {e}\n{traceback.format_exc(limit=1)}")
        except Exception as e_fatal:
            st.error(f"Erreur inattendue durant Suppression Couche Fine: {e_fatal}")
            add_log(f"ERREUR FATALE (Remove Thin): {e_fatal}\n{traceback.format_exc()}")
        finally:
            pass
def draw_spectrum_plot(res: Dict, active_targets_for_plot: List[Dict], mse: Optional[float], is_optimized: bool = False, method_name: str = "", res_optim_grid: Optional[Dict] = None) -> plt.Figure:
    fig_spec, ax_spec = plt.subplots(figsize=(12, 6))
    opt_method_str = f" ({method_name})" if method_name else ""
    title = f'Spectre {"Optimisé" if is_optimized else "Nominal"}{opt_method_str}'
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
                    label = 'Cible(s)' if not plotted_target_label else "_nolegend_"
                    ax_spec.plot(x_coords, y_coords, 'r--', linewidth=1.0, alpha=0.7, label=label, zorder=5)
                    ax_spec.plot(x_coords, y_coords, marker='x', color='red', markersize=6, linestyle='none', label='_nolegend_', zorder=6)
                    plotted_target_label = True
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
            ax_spec.set_title(title)
            ax_spec.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
            ax_spec.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')
            ax_spec.minorticks_on()
            if len(res_l_plot) > 0: ax_spec.set_xlim(res_l_plot[0], res_l_plot[-1])
            ax_spec.set_ylim(-0.05, 1.05)
            if plotted_target_label or (line_ts is not None): ax_spec.legend(fontsize=8)
            if mse is not None and np.isfinite(mse): mse_text = f"MSE = {mse:.3e}"
            else: mse_text = "MSE: N/A"
            ax_spec.text(0.98, 0.98, mse_text, transform=ax_spec.transAxes, ha='right', va='top', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
        else:
            ax_spec.text(0.5, 0.5, "Pas de données spectrales", ha='center', va='center', transform=ax_spec.transAxes)
    except Exception as e_spec:
        ax_spec.text(0.5, 0.5, f"Erreur plot spectre:\n{e_spec}", ha='center', va='center', transform=ax_spec.transAxes, color='red')
        add_log(f"Erreur plot spectre: {e_spec}")
    fig_spec.tight_layout()
    return fig_spec
def draw_profile_and_stack_plots(current_ep: np.ndarray, l0_repr: float, nH_material_in: MaterialInputType, nL_material_in: MaterialInputType, nSub_material_in: MaterialInputType, is_optimized: bool = False, material_sequence: Optional[List[str]] = None) -> plt.Figure:
    fig_others, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_idx = axes[0]
    ax_stack = axes[1]
    num_layers = len(current_ep) if current_ep is not None else 0
    try:
        nH_c_repr, logs_h = _get_nk_at_lambda(nH_material_in, l0_repr, EXCEL_FILE_PATH)
        nL_c_repr, logs_l = _get_nk_at_lambda(nL_material_in, l0_repr, EXCEL_FILE_PATH)
        nSub_c_repr, logs_s = _get_nk_at_lambda(nSub_material_in, l0_repr, EXCEL_FILE_PATH)
        add_log(logs_h); add_log(logs_l); add_log(logs_s)
        if nH_c_repr is None or nL_c_repr is None or nSub_c_repr is None:
             raise ValueError("Indices à l0 non trouvés pour plot profil.")
        nH_r_repr, nL_r_repr, nSub_r_repr = nH_c_repr.real, nL_c_repr.real, nSub_c_repr.real
        if material_sequence:
            add_log("AVERTISSEMENT: Plot profil pour séquence arbitraire non implémenté.")
            n_real_layers_repr = [nH_r_repr if i % 2 == 0 else nL_r_repr for i in range(num_layers)]
        else:
            n_real_layers_repr = [nH_r_repr if i % 2 == 0 else nL_r_repr for i in range(num_layers)]
        ep_cumulative = np.cumsum(current_ep) if num_layers > 0 else np.array([0])
        total_thickness = ep_cumulative[-1] if num_layers > 0 else 0
        margin = max(50, 0.1 * total_thickness) if total_thickness > 0 else 50
        x_coords_plot = [-margin]
        y_coords_plot = [nSub_r_repr]
        if num_layers > 0:
            x_coords_plot.append(0)
            y_coords_plot.append(nSub_r_repr)
            for i in range(num_layers):
                layer_start = ep_cumulative[i-1] if i > 0 else 0
                layer_end = ep_cumulative[i]
                layer_n_real = n_real_layers_repr[i]
                x_coords_plot.extend([layer_start, layer_end])
                y_coords_plot.extend([layer_n_real, layer_n_real])
            last_layer_end = ep_cumulative[-1]
            x_coords_plot.extend([last_layer_end, last_layer_end + margin])
            y_coords_plot.extend([1.0, 1.0])
        else:
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
        valid_n = [n for n in [1.0, nSub_r_repr] + n_real_layers_repr if np.isfinite(n)]
        min_n = min(valid_n) if valid_n else 0.9
        max_n = max(valid_n) if valid_n else 2.5
        y_padding = (max_n - min_n) * 0.1 + 0.05
        ax_idx.set_ylim(bottom=min_n - y_padding, top=max_n + y_padding)
        if ax_idx.get_legend_handles_labels()[1]: ax_idx.legend(fontsize=8)
    except Exception as e_idx:
        ax_idx.text(0.5, 0.5, f"Erreur plot indice:\n{e_idx}", ha='center', va='center', transform=ax_idx.transAxes, color='red')
        add_log(f"Erreur plot indice: {e_idx}")
    try:
        if num_layers > 0:
            indices_complex_repr = []
            if material_sequence:
                add_log("AVERTISSEMENT: Plot structure pour séquence arbitraire non implémenté.")
                indices_complex_repr = [nH_c_repr if i % 2 == 0 else nL_c_repr for i in range(num_layers)]
                layer_types = [f"Mat{i+1}" for i in range(num_layers)]
            else:
                indices_complex_repr = [nH_c_repr if i % 2 == 0 else nL_c_repr for i in range(num_layers)]
                layer_types = ["H" if i % 2 == 0 else "L" for i in range(num_layers)]
            colors = ['lightblue' if i % 2 == 0 else 'lightcoral' for i in range(num_layers)]
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
            ax_stack.set_yticklabels(yticks_labels, fontsize=7)
            ax_stack.invert_yaxis()
            max_ep_plot = max(current_ep) if current_ep.size > 0 else 1.0
            fontsize_bar = max(6, 9 - num_layers // 15)
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ha_pos = 'left' if width < max_ep_plot * 0.3 else 'right'
                x_text_pos = width * 1.02 if ha_pos == 'left' else width * 0.98
                text_color = 'black' if ha_pos == 'left' else 'white'
                ax_stack.text(x_text_pos, bar.get_y() + bar.get_height()/2., f"{width:.2f}",
                             va='center', ha=ha_pos, color=text_color, fontsize=fontsize_bar, weight='bold')
        else:
            ax_stack.text(0.5, 0.5, "Structure Vide", ha='center', va='center', fontsize=10, color='grey', transform=ax_stack.transAxes)
            ax_stack.set_yticks([]); ax_stack.set_xticks([])
        ax_stack.set_xlabel('Épaisseur (nm)')
        stack_title_prefix = f'Structure {"Optimisée" if is_optimized else "Nominale"}'
        ax_stack.set_title(f"{stack_title_prefix} ({num_layers} couches)")
        max_ep_plot = max(current_ep) if num_layers > 0 else 10
        ax_stack.set_xlim(right=max_ep_plot * 1.1)
    except Exception as e_stack:
        ax_stack.text(0.5, 0.5, f"Erreur plot structure:\n{e_stack}", ha='center', va='center', transform=ax_stack.transAxes, color='red')
        add_log(f"Erreur plot structure: {e_stack}")
    fig_others.tight_layout(pad=1.5)
    return fig_others
st.set_page_config(layout="wide", page_title="Optimiseur Film Mince (Streamlit)")
st.title("🔬 Optimiseur de Films Minces (Streamlit + JAX)")
st.markdown("""
*Conversion Streamlit de l'outil Tkinter. Se concentre sur les calculs H/L.*
""")
if 'init_done' not in st.session_state:
    st.session_state.log_messages = ["[Initialisation] Bienvenue dans l'optimiseur Streamlit."]
    st.session_state.current_ep = None
    st.session_state.current_qwot = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    st.session_state.optimized_ep = None
    st.session_state.is_optimized_state = False
    st.session_state.optimized_qwot_str = ""
    st.session_state.material_sequence = None
    st.session_state.ep_history = deque(maxlen=5)
    st.session_state.last_mse = None
    st.session_state.needs_rerun_calc = False
    st.session_state.rerun_calc_params = {}
    st.session_state.maxiter = 1000
    st.session_state.maxfun = 1000
    try:
        mats, logs = get_available_materials_from_excel(EXCEL_FILE_PATH)
        add_log(logs)
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
    st.session_state.nH_r = 2.35
    st.session_state.nH_i = 0.0
    st.session_state.nL_r = 1.46
    st.session_state.nL_i = 0.0
    st.session_state.nSub_r = 1.52
    st.session_state.init_done = True
    add_log("État de session initialisé.")
    st.session_state.needs_rerun_calc = True
    st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Initial Load"}
def trigger_nominal_recalc():
    if not st.session_state.get('calculating', False):
        print("INFO: trigger_nominal_recalc called")
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {
            'is_optimized_run': False,
            'method_name': "Nominal (Param Update)",
            'force_ep': None
        }
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("Matériaux")
    st.session_state.selected_H = st.selectbox(
        "Matériau H", options=st.session_state.available_materials,
        index=st.session_state.available_materials.index(st.session_state.selected_H),
        key="sb_H",
        on_change=trigger_nominal_recalc
    )
    if st.session_state.selected_H == "Constant":
        hc1, hc2 = st.columns(2)
        st.session_state.nH_r = hc1.number_input("n' H", value=st.session_state.nH_r, format="%.4f", key="nH_r_const", on_change=trigger_nominal_recalc)
        st.session_state.nH_i = hc2.number_input("k H", value=st.session_state.nH_i, min_value=0.0, format="%.4f", key="nH_i_const", on_change=trigger_nominal_recalc)
    st.session_state.selected_L = st.selectbox(
        "Matériau L", options=st.session_state.available_materials,
        index=st.session_state.available_materials.index(st.session_state.selected_L),
        key="sb_L",
        on_change=trigger_nominal_recalc
    )
    if st.session_state.selected_L == "Constant":
         lc1, lc2 = st.columns(2)
         st.session_state.nL_r = lc1.number_input("n' L", value=st.session_state.nL_r, format="%.4f", key="nL_r_const", on_change=trigger_nominal_recalc)
         st.session_state.nL_i = lc2.number_input("k L", value=st.session_state.nL_i, min_value=0.0, format="%.4f", key="nL_i_const", on_change=trigger_nominal_recalc)
    st.session_state.selected_Sub = st.selectbox(
        "Substrat", options=st.session_state.available_substrates,
        index=st.session_state.available_substrates.index(st.session_state.selected_Sub),
        key="sb_Sub",
        on_change=trigger_nominal_recalc
    )
    if st.session_state.selected_Sub == "Constant":
         st.session_state.nSub_r = st.number_input("n' Substrat", value=st.session_state.nSub_r, format="%.4f", key="nSub_const", on_change=trigger_nominal_recalc)
    if st.button("🔄 Recharger Matériaux Excel", key="reload_mats"):
        st.cache_data.clear()
        try:
            mats, logs = get_available_materials_from_excel(EXCEL_FILE_PATH)
            add_log(logs)
            st.session_state.available_materials = sorted(list(set(["Constant"] + mats)))
            base_subs = ["Constant", "Fused Silica", "BK7", "D263"]
            st.session_state.available_substrates = sorted(list(set(base_subs + st.session_state.available_materials)))
            add_log("Liste matériaux rechargée.")
            if st.session_state.selected_H not in st.session_state.available_materials: st.session_state.selected_H = "Constant"
            if st.session_state.selected_L not in st.session_state.available_materials: st.session_state.selected_L = "Constant"
            if st.session_state.selected_Sub not in st.session_state.available_substrates: st.session_state.selected_Sub = "Constant"
            st.rerun()
        except Exception as e:
            st.error(f"Erreur rechargement matériaux: {e}")
            add_log(f"Erreur rechargement matériaux: {e}")
    st.divider()
    st.subheader("Structure Nominale")
    st.session_state.current_qwot = st.text_area(
        "QWOT Nominal (multiplicateurs séparés par ',')",
        value=st.session_state.current_qwot,
        key="qwot_input",
        on_change=clear_optimized_state
    )
    num_layers_from_qwot = len([q for q in st.session_state.current_qwot.split(',') if q.strip()])
    st.caption(f"Couches d'après QWOT: {num_layers_from_qwot}")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.session_state.l0 = st.number_input("λ₀ centrage (nm)", value=st.session_state.l0, min_value=1.0, format="%.2f", key="l0_input", on_change=trigger_nominal_recalc)
    with c2:
         init_layers_num = st.number_input("N couches:", min_value=0, value=num_layers_from_qwot, step=1, key="init_layers_gen_num", label_visibility="collapsed")
         if st.button("Générer 1s", key="gen_qwot_btn", use_container_width=True):
             if init_layers_num > 0:
                 new_qwot = ",".join(['1'] * init_layers_num)
                 if new_qwot != st.session_state.current_qwot:
                     st.session_state.current_qwot = new_qwot
                     clear_optimized_state()
                     st.session_state.needs_rerun_calc = True
                     st.session_state.rerun_calc_params = {
                         'is_optimized_run': False,
                         'method_name': "Nominal (Généré 1s)"
                     }
                     st.rerun()
             elif st.session_state.current_qwot != "":
                  st.session_state.current_qwot = ""
                  clear_optimized_state()
                  st.session_state.needs_rerun_calc = True
                  st.session_state.rerun_calc_params = {
                      'is_optimized_run': False,
                      'method_name': "Nominal (QWOT Effacé)"
                  }
                  st.rerun()
    st.divider()
    st.subheader("Cibles Spectrales (Transmission T)")
    st.session_state.l_step = st.number_input("Pas λ (nm) (optimisation)", value=st.session_state.l_step, min_value=0.1, format="%.2f", key="l_step_input_sidebar", on_change=trigger_nominal_recalc)
    header_cols = st.columns([0.5, 1.5, 1.5, 1.5, 1.5])
    headers = ["Actif", "λ min", "λ max", "T@λmin", "T@λmax"]
    for col, header in zip(header_cols, headers): col.caption(header)
    for i in range(len(st.session_state.targets)):
        target = st.session_state.targets[i]
        cols = st.columns([0.5, 1.5, 1.5, 1.5, 1.5])
        current_enabled = target.get('enabled', False)
        new_enabled = cols[0].checkbox("", value=current_enabled, key=f"target_enable_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['enabled'] = new_enabled
        st.session_state.targets[i]['min'] = cols[1].number_input("λmin", value=target.get('min', 0.0), format="%.1f", key=f"target_min_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['max'] = cols[2].number_input("λmax", value=target.get('max', 0.0), format="%.1f", key=f"target_max_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['target_min'] = cols[3].number_input("Tmin", value=target.get('target_min', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmin_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['target_max'] = cols[4].number_input("Tmax", value=target.get('target_max', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmax_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
    with st.expander("👁️ Prévisualiser Cibles Actives", expanded=False):
        active_targets_preview = validate_targets()
        if active_targets_preview is not None:
             def create_target_preview_fig(active_targets_list):
                 fig_prev, ax_prev = plt.subplots(figsize=(5, 3))
                 if not active_targets_list:
                     ax_prev.text(0.5, 0.5, "Pas de cibles actives/valides", ha='center', va='center', transform=ax_prev.transAxes)
                     ax_prev.set_ylim(-0.05, 1.05)
                 else:
                     all_l_min = min(t['min'] for t in active_targets_list)
                     all_l_max = max(t['max'] for t in active_targets_list)
                     padding = (all_l_max - all_l_min) * 0.05 + 1
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
             fig_preview = create_target_preview_fig(active_targets_preview)
             st.pyplot(fig_preview)
             plt.close(fig_preview)
        else:
             st.warning("Impossible d'afficher la prévisualisation (erreurs dans les cibles).")
col_main, col_actions = st.columns([3, 1])
with col_actions:
    st.header("▶️ Actions")
    if st.button("📊 Évaluer Structure Nominale", key="eval_nom", use_container_width=True):
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (Évalué)"}
        st.rerun()
    if st.button("✨ Optimisation Locale", key="optim_local", use_container_width=True):
        run_local_optimization_wrapper()
    if st.button("🚀 Scan Initial + Optimisation", key="optim_scan", use_container_width=True):
        run_scan_optimization_wrapper()
    st.divider()
    st.subheader("🤖 Mode Auto")
    st.session_state.auto_thin_threshold = st.number_input("Seuil suppression (nm)", value=st.session_state.auto_thin_threshold, min_value=MIN_THICKNESS_PHYS_NM, format="%.3f", key="auto_thin_input_actions")
    if st.button("▶️ Lancer Mode Auto", key="optim_auto", use_container_width=True):
        run_auto_mode_wrapper()
    st.divider()
    st.subheader("🛠️ Actions sur Optimisé")
    can_optimize = st.session_state.get('is_optimized_state', False) and st.session_state.get('optimized_ep') is not None
    ep_display_for_actions = st.session_state.optimized_ep if st.session_state.is_optimized_state else None
    can_remove = can_optimize and ep_display_for_actions is not None and len(ep_display_for_actions) > 2
    can_undo = bool(st.session_state.get('ep_history'))
    if st.button("🗑️ Suppr. Couche Fine + Ré-opt", key="remove_thin", use_container_width=True, disabled=not can_remove):
        run_remove_thin_wrapper()
    if st.button("💾 Optimisé -> Nominal", key="set_optim_as_nom", use_container_width=True, disabled=not can_optimize):
         set_optimized_as_nominal_wrapper()
         st.rerun()
    if st.button(f"↩️ Annuler Suppr. ({len(st.session_state.get('ep_history', deque()))})", key="undo_remove", use_container_width=True, disabled=not can_undo):
         undo_remove_wrapper()
         st.rerun()
    st.divider()
    st.subheader("📜 Logs")
    with st.expander("Afficher/Cacher les Logs", expanded=True):
        log_container = st.container(height=400)
        for msg in st.session_state.get('log_messages', ["Aucun log."]):
            log_container.text(msg)
    if st.button("🧹 Effacer Logs", key="clear_logs_btn", use_container_width=True):
        st.session_state.log_messages = ["[Logs effacés]"]
        st.rerun()
with col_main:
    st.header("📈 Résultats")
    state_desc = "Optimisé" if st.session_state.is_optimized_state else "Nominal"
    ep_display = st.session_state.optimized_ep if st.session_state.is_optimized_state else st.session_state.current_ep
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
             valid_thick = ep_display[ep_display >= MIN_THICKNESS_PHYS_NM - 1e-9]
             if valid_thick.size > 0:
                 min_thick_str = f"{np.min(valid_thick):.3f} nm"
         st.metric("Ép. Min", min_thick_str)
    with res_cols[2]:
        if st.session_state.is_optimized_state and st.session_state.get('optimized_qwot_str'):
             st.text_input("QWOT Opt.", value=st.session_state.optimized_qwot_str, disabled=True, key="opt_qwot_display_main")
    st.subheader("Graphiques")
    if 'last_calc_results' in st.session_state and st.session_state.last_calc_results:
        results_data = st.session_state.last_calc_results
        res_fine_plot = results_data.get('res_fine')
        ep_plot = results_data.get('ep_used')
        l0_plot = results_data.get('l0_used')
        nH_plot = results_data.get('nH_used')
        nL_plot = results_data.get('nL_used')
        nSub_plot = results_data.get('nSub_used')
        active_targets_plot = validate_targets()
        mse_plot = st.session_state.last_mse
        is_optimized_plot = st.session_state.is_optimized_state
        method_name_plot = results_data.get('method_name', '')
        res_optim_grid_plot = results_data.get('res_optim_grid')
        if res_fine_plot and ep_plot is not None and l0_plot is not None and \
           nH_plot is not None and nL_plot is not None and nSub_plot is not None and active_targets_plot is not None:
            try:
                 fig_spec = draw_spectrum_plot(res_fine_plot, active_targets_plot, mse_plot, is_optimized_plot, method_name_plot, res_optim_grid_plot)
                 st.pyplot(fig_spec)
                 plt.close(fig_spec)
                 fig_others = draw_profile_and_stack_plots(ep_plot, l0_plot, nH_plot, nL_plot, nSub_plot, is_optimized_plot, None) # Assuming no arbitrary sequence for now
                 st.pyplot(fig_others)
                 plt.close(fig_others)
            except Exception as e:
                 st.error(f"Erreur lors de la génération des graphiques : {e}")
                 add_log(f"[Erreur Plot] {traceback.format_exc(limit=1)}")
        else:
             st.warning("Données de calcul manquantes ou invalides pour l'affichage des graphiques.")
    else:
        st.info("Lancez une évaluation ou une optimisation pour voir les résultats.")
if st.session_state.get('needs_rerun_calc', False):
    add_log("Déclenchement du recalcul automatique...")
    params = st.session_state.rerun_calc_params
    force_ep_val = params.get('force_ep')
    st.session_state.needs_rerun_calc = False
    st.session_state.rerun_calc_params = {}
    st.session_state.calculating = True
    run_calculation_wrapper(
        is_optimized_run=params.get('is_optimized_run', False),
        method_name=params.get('method_name', 'Recalcul Auto'),
        force_ep=force_ep_val
    )
    st.session_state.calculating = False
    add_log("Recalcul terminé, déclenchement du rafraîchissement UI...")
    st.rerun()
    
