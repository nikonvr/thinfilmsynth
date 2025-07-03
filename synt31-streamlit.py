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
AUTO_NEEDLES_PER_CYCLE = 3
AUTO_MAX_CYCLES = 3
MSE_IMPROVEMENT_TOLERANCE = 1e-9
EXCEL_FILE_PATH = "indices.xlsx"
MAXITER_HARDCODED = 1000
MAXFUN_HARDCODED = 1000

def add_log(message: Union[str, List[str]]):
    pass

@st.cache_data
def load_material_data_from_xlsx_sheet(file_path: str, sheet_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    logs = []
    try:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        except FileNotFoundError:
            st.error(f"Excel file '{file_path}' not found. Please check its presence.")
            logs.append(f"Critical error: Excel file not found: {file_path}")
            return None, None, None, logs
        except Exception as e:
            st.error(f"Error reading Excel ('{file_path}', sheet '{sheet_name}'): {e}")
            logs.append(f"Unexpected Excel error ({type(e).__name__}): {e}")
            return None, None, None, logs

        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        numeric_df = numeric_df.dropna(how='all')

        if numeric_df.shape[1] >= 3:
            cols_to_check = numeric_df.columns[:3]
            numeric_df = numeric_df.dropna(subset=cols_to_check)
        else:
            logs.append(f"Warning: Sheet '{sheet_name}' does not contain 3 numeric columns.")
            return np.array([]), np.array([]), np.array([]), logs

        if numeric_df.empty:
            logs.append(f"Warning: No valid numeric data found in '{sheet_name}' after cleaning.")
            return np.array([]), np.array([]), np.array([]), logs

        try:
            numeric_df = numeric_df.sort_values(by=numeric_df.columns[0])
        except IndexError:
            logs.append(f"Error: Could not sort data for sheet '{sheet_name}'. Index column 0 missing?")
            return np.array([]), np.array([]), np.array([]), logs

        l_nm = numeric_df.iloc[:, 0].values.astype(np.float64)
        n = numeric_df.iloc[:, 1].values.astype(np.float64)
        k = numeric_df.iloc[:, 2].values.astype(np.float64)

        if np.any(k < -1e-9):
            invalid_k_indices = np.where(k < -1e-9)[0]
            logs.append(f"WARNING: Negative k values detected and set to 0 for '{sheet_name}' at indices: {invalid_k_indices.tolist()}")
            k = np.maximum(k, 0.0)

        if len(l_nm) == 0:
            logs.append(f"Warning: No valid data rows after conversion in '{sheet_name}'.")
            return np.array([]), np.array([]), np.array([]), logs

        logs.append(f"Data loaded '{sheet_name}': {len(l_nm)} pts [{l_nm.min():.1f}-{l_nm.max():.1f} nm]")
        return l_nm, n, k, logs

    except ValueError as ve:
        logs.append(f"Excel Value Error ('{sheet_name}'): {ve}")
        return None, None, None, logs
    except Exception as e:
        logs.append(f"Unexpected error reading Excel ('{sheet_name}'): {type(e).__name__} - {e}")
        return None, None, None, logs

def process_uploaded_target_file(uploaded_file) -> Tuple[Optional[np.ndarray], List[str]]:
    logs = []
    if uploaded_file is None:
        return None, ["No file uploaded."]
    try:
        df = pd.read_excel(uploaded_file, header=None)
        numeric_df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
        if numeric_df.shape[1] < 2:
            logs.append(f"Error: Uploaded target file must have at least 2 columns. Found {numeric_df.shape[1]}.")
            st.error("Uploaded target file must have at least 2 numeric columns.")
            return None, logs
        cols_to_check = numeric_df.columns[:2]
        numeric_df = numeric_df.dropna(subset=cols_to_check)
        if numeric_df.empty:
            logs.append("Error: No valid numeric data found in the first two columns of the uploaded target file.")
            st.error("No valid numeric data found in the uploaded target file.")
            return None, logs
        numeric_df = numeric_df.sort_values(by=numeric_df.columns[0])
        l_nm = numeric_df.iloc[:, 0].values.astype(np.float64)
        t_percent = numeric_df.iloc[:, 1].values.astype(np.float64)
        if np.any(l_nm <= 0):
            logs.append("Warning: Wavelength values <= 0 found and ignored.")
            valid_l_mask = l_nm > 0
            l_nm = l_nm[valid_l_mask]
            t_percent = t_percent[valid_l_mask]
        if np.any((t_percent < 0) | (t_percent > 100)):
            logs.append("Warning: Transmission values outside [0, 100]% detected. They will be clipped.")
            t_percent = np.clip(t_percent, 0.0, 100.0)
        t_fraction = t_percent / 100.0
        if len(l_nm) < 2:
            logs.append("Error: At least two valid data points are required in the target file.")
            st.error("Uploaded target file needs at least two valid data points.")
            return None, logs
        target_data = np.vstack((l_nm, t_fraction)).T
        logs.append(f"Successfully loaded target from file: {len(target_data)} points from {l_nm.min():.1f} to {l_nm.max():.1f} nm.")
        return target_data, logs
    except Exception as e:
        logs.append(f"Error processing uploaded target file: {type(e).__name__} - {e}")
        st.error(f"Failed to process the uploaded file: {e}")
        return None, logs

def get_available_materials_from_excel(excel_path: str) -> Tuple[List[str], List[str]]:
    logs = []
    try:
        xl = pd.ExcelFile(excel_path)
        sheet_names = [name for name in xl.sheet_names if not name.startswith("Sheet")]
        logs.append(f"Materials found in {excel_path}: {sheet_names}")
        return sheet_names, logs
    except FileNotFoundError:
        st.error(f"Excel file '{excel_path}' not found for listing materials.")
        logs.append(f"Critical FNF error: Excel file {excel_path} not found for listing materials.")
        return [], logs
    except Exception as e:
        st.error(f"Error reading sheet names from '{excel_path}': {e}")
        logs.append(f"Error reading sheet names from {excel_path}: {type(e).__name__} - {e}")
        return [], logs

@jax.jit
def get_n_fused_silica(l_nm: jnp.ndarray) -> jnp.ndarray:
    l_um = l_nm / 1000.0
    l_um_squared = l_um**2
    B1, B2, B3 = 0.6961663, 0.4079426, 0.8974794
    C1, C2, C3 = 0.0684043**2, 0.1162414**2, 9.896161**2
    n_squared = 1.0 + (B1 * l_um_squared) / (l_um_squared - C1) \
                      + (B2 * l_um_squared) / (l_um_squared - C2) \
                      + (B3 * l_um_squared) / (l_um_squared - C3)
    n = jnp.sqrt(n_squared)
    k_val = jnp.zeros_like(n)
    return n + 1j * k_val

@jax.jit
def get_n_bk7(l_nm: jnp.ndarray) -> jnp.ndarray:
    l_um = l_nm / 1000.0
    l_um_squared = l_um**2
    B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
    C1, C2, C3 = 0.00600069867, 0.0200179144, 103.560653
    n_squared = 1.0 + (B1 * l_um_squared) / (l_um_squared - C1) \
                      + (B2 * l_um_squared) / (l_um_squared - C2) \
                      + (B3 * l_um_squared) / (l_um_squared - C3)
    n = jnp.sqrt(n_squared)
    k_val = jnp.zeros_like(n)
    return n + 1j * k_val

@jax.jit
def get_n_d263(l_nm: jnp.ndarray) -> jnp.ndarray:
    n = jnp.full_like(l_nm, 1.523, dtype=jnp.float64)
    k_val = jnp.zeros_like(n)
    return n + 1j * k_val

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
                logs.append(f"WARNING: Constant index n'={nk_complex.real} <= 0. Using n'=1.0.")
                nk_complex = complex(1.0, nk_complex.imag)
            if nk_complex.imag < 0:
                logs.append(f"WARNING: Constant index k={nk_complex.imag} < 0. Using k=0.0.")
                nk_complex = complex(nk_complex.real, 0.0)
            result = jnp.full(l_vec_target_jnp.shape, nk_complex)
        elif isinstance(material_definition, str):
            mat_upper = material_definition.upper()
            if mat_upper == "FUSED SILICA": result = get_n_fused_silica(l_vec_target_jnp)
            elif mat_upper == "BK7": result = get_n_bk7(l_vec_target_jnp)
            elif mat_upper == "D263": result = get_n_d263(l_vec_target_jnp)
            else:
                sheet_name = material_definition
                l_data, n_data, k_data, load_logs = load_material_data_from_xlsx_sheet(excel_file_path, sheet_name)
                logs.extend(load_logs)
                if l_data is None or len(l_data) == 0:
                    st.error(f"Could not load or empty data for material '{sheet_name}' from {excel_file_path}.")
                    return None, logs
                l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
                if jnp.min(l_vec_target_jnp) < jnp.min(l_data_jnp) - 1e-6 or jnp.max(l_vec_target_jnp) > jnp.max(l_data_jnp) + 1e-6:
                    logs.append(f"WARNING: Interpolation for '{sheet_name}' out of bounds. Extrapolation used.")
                result = interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)
        elif isinstance(material_definition, tuple) and len(material_definition) == 3:
            l_data, n_data, k_data = material_definition
            l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
            if not len(l_data_jnp): raise ValueError("Raw material data empty.")
            sort_indices = jnp.argsort(l_data_jnp)
            l_data_jnp, n_data_jnp, k_data_jnp = l_data_jnp[sort_indices], n_data_jnp[sort_indices], k_data_jnp[sort_indices]
            result = interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)
        else:
            raise TypeError(f"Unsupported material definition type: {type(material_definition)}")
        
        if jnp.any(jnp.isnan(result.real)) or jnp.any(result.real <= 0):
            logs.append(f"WARNING: n'<=0 or NaN detected for '{material_definition}'. Replaced with n'=1.")
            result = jnp.where(jnp.isnan(result.real) | (result.real <= 0), 1.0 + 1j*result.imag, result)
        if jnp.any(jnp.isnan(result.imag)) or jnp.any(result.imag < 0):
            logs.append(f"WARNING: k<0 or NaN detected for '{material_definition}'. Replaced with k=0.")
            result = jnp.where(jnp.isnan(result.imag) | (result.imag < 0), result.real + 0.0j, result)
        return result, logs
    except Exception as e:
        logs.append(f"Error preparing material data for '{material_definition}': {e}")
        st.error(f"Critical error preparing material '{material_definition}': {e}")
        return None, logs

def _get_nk_at_lambda(material_definition: MaterialInputType, l_nm_target: float, excel_file_path: str) -> Tuple[Optional[complex], List[str]]:
    logs = []
    if l_nm_target <= 0:
        return None, [f"Error: Target wavelength {l_nm_target}nm invalid."]
    l_vec_jnp = jnp.array([l_nm_target], dtype=jnp.float64)
    nk_array, prep_logs = _get_nk_array_for_lambda_vec(material_definition, l_vec_jnp, excel_file_path)
    logs.extend(prep_logs)
    return complex(nk_array[0]) if nk_array is not None else (None, logs)

@jax.jit
def _compute_layer_matrix_scan_step_jit(carry_matrix: jnp.ndarray, layer_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, None]:
    thickness, Ni, l_val = layer_data
    eta = Ni
    safe_l_val = jnp.maximum(l_val, 1e-9)
    phi = (2 * jnp.pi / safe_l_val) * (Ni * thickness)
    cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
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
    etainc, etasub = 1.0 + 0j, nSub_at_lval
    def calculate_for_valid_l(l_: jnp.ndarray) -> jnp.ndarray:
        M = compute_stack_matrix_core_jax(ep_vector_contig, layer_indices_at_lval, l_)
        m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        rs_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        safe_denominator = jnp.where(jnp.abs(rs_denominator) < 1e-12, 1e-12 + 0j, rs_denominator)
        ts = (2.0 * etainc) / safe_denominator
        Ts_complex = (jnp.real(etasub) / jnp.real(etainc)) * (ts * jnp.conj(ts))
        return jnp.where(jnp.abs(rs_denominator) < 1e-12, jnp.nan, jnp.real(Ts_complex))
    return cond(l_val > 1e-9, calculate_for_valid_l, lambda l_: jnp.nan, l_val)

def calculate_T_from_ep_jax(ep_vector: Union[np.ndarray, List[float]],
                            nH_material: MaterialInputType, nL_material: MaterialInputType,
                            nSub_material: MaterialInputType, l_vec: Union[np.ndarray, List[float]],
                            excel_file_path: str) -> Tuple[Optional[Dict[str, np.ndarray]], List[str]]:
    logs = []
    l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64)
    ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)
    if not l_vec_jnp.size: return {'l': np.array([]), 'Ts': np.array([])}, ["Empty lambda vector."]
    if not ep_vector_jnp.size:
        return {'l': np.array(l_vec_jnp), 'Ts': np.ones_like(l_vec)}, ["Empty structure, T=1."]
    
    nH_arr, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_jnp, excel_file_path)
    nL_arr, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_jnp, excel_file_path)
    nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
    logs.extend(logs_h + logs_l + logs_sub)
    if nH_arr is None or nL_arr is None or nSub_arr is None: return None, logs

    calculate_single_wavelength_T_hl_jit = jax.jit(calculate_single_wavelength_T_core)
    num_layers = len(ep_vector_jnp)
    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH_arr, nL_arr)
    indices_alternating_T = indices_alternating.T
    
    Ts_arr_raw = vmap(calculate_single_wavelength_T_hl_jit, in_axes=(0, None, 0, 0))(
        l_vec_jnp, ep_vector_jnp, indices_alternating_T, nSub_arr
    )
    Ts_arr_clipped = jnp.clip(jnp.nan_to_num(Ts_arr_raw, nan=0.0), 0.0, 1.0)
    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts_arr_clipped)}, logs

def calculate_initial_ep(emp: Union[List[float], Tuple[float,...]], l0: float,
                         nH0_material: MaterialInputType, nL0_material: MaterialInputType,
                         excel_file_path: str) -> Tuple[Optional[np.ndarray], List[str]]:
    logs = []
    num_layers = len(emp)
    ep_initial = np.zeros(num_layers, dtype=np.float64)
    if l0 <= 0: return ep_initial, [f"Warning: l0={l0} <= 0. Initial thicknesses set to 0."]

    nH_complex_at_l0, logs_h = _get_nk_at_lambda(nH0_material, l0, excel_file_path)
    nL_complex_at_l0, logs_l = _get_nk_at_lambda(nL0_material, l0, excel_file_path)
    logs.extend(logs_h + logs_l)
    if nH_complex_at_l0 is None or nL_complex_at_l0 is None:
        st.error(f"Critical error getting indices at l0={l0}nm for initial thickness calculation.")
        return None, logs

    nH_real_at_l0, nL_real_at_l0 = nH_complex_at_l0.real, nL_complex_at_l0.real
    for i in range(num_layers):
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        ep_initial[i] = (emp[i] * l0 / (4.0 * n_real_layer_at_l0)) if n_real_layer_at_l0 > 1e-9 else 0.0
    
    ep_initial_phys = np.where(ep_initial < MIN_THICKNESS_PHYS_NM, 0.0, ep_initial)
    num_clamped_zero = np.sum((ep_initial > 1e-12) & (ep_initial < MIN_THICKNESS_PHYS_NM))
    if num_clamped_zero > 0:
        logs.append(f"Warning: {num_clamped_zero} initial thicknesses < {MIN_THICKNESS_PHYS_NM}nm were set to 0.")
    return ep_initial_phys, logs

def calculate_qwot_from_ep(ep_vector: np.ndarray, l0: float,
                           nH0_material: MaterialInputType, nL0_material: MaterialInputType,
                           excel_file_path: str) -> Tuple[Optional[np.ndarray], List[str]]:
    logs = []
    num_layers = len(ep_vector)
    qwot_multipliers = np.full(num_layers, np.nan, dtype=np.float64)
    if l0 <= 0: return qwot_multipliers, [f"Warning: l0={l0} <= 0. QWOT set to NaN."]

    nH_complex_at_l0, logs_h = _get_nk_at_lambda(nH0_material, l0, excel_file_path)
    nL_complex_at_l0, logs_l = _get_nk_at_lambda(nL0_material, l0, excel_file_path)
    logs.extend(logs_h + logs_l)
    if nH_complex_at_l0 is None or nL_complex_at_l0 is None:
        st.error(f"Error calculating QWOT: H/L indices not found at l0={l0}nm.")
        return None, logs

    nH_real_at_l0, nL_real_at_l0 = nH_complex_at_l0.real, nL_complex_at_l0.real
    for i in range(num_layers):
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real_layer_at_l0 > 1e-9:
            qwot_multipliers[i] = ep_vector[i] * (4.0 * n_real_layer_at_l0) / l0
        else:
            qwot_multipliers[i] = 0.0 if ep_vector[i] < 1e-9 else np.nan
    return qwot_multipliers, logs

def calculate_final_mse(res: Dict[str, np.ndarray], active_targets: List[Dict]) -> Tuple[Optional[float], int]:
    total_squared_error, total_points_in_targets = 0.0, 0
    if not active_targets or 'Ts' not in res or res['Ts'] is None or 'l' not in res or res['l'] is None:
        return None, 0
    res_l_np, res_ts_np = np.asarray(res['l']), np.asarray(res['Ts'])
    if res_l_np.size == 0 or res_l_np.size != res_ts_np.size: return None, 0

    for target in active_targets:
        try:
            l_min, l_max = float(target['min']), float(target['max'])
            t_min, t_max = float(target['target_min']), float(target['target_max'])
            if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0) or l_max < l_min: continue
        except (KeyError, ValueError, TypeError): continue
        
        indices = np.where((res_l_np >= l_min) & (res_l_np <= l_max))[0]
        if indices.size > 0:
            calculated_Ts_in_zone = res_ts_np[indices]
            target_lambdas_in_zone = res_l_np[indices]
            if abs(l_max - l_min) < 1e-9:
                interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
            else:
                slope = (t_max - t_min) / (l_max - l_min)
                interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)
            total_squared_error += np.sum((calculated_Ts_in_zone - interpolated_target_t)**2)
            total_points_in_targets += len(calculated_Ts_in_zone)
    return (total_squared_error / total_points_in_targets) if total_points_in_targets > 0 else None, total_points_in_targets

@jax.jit
def calculate_mse_for_optimization_penalized_jax(ep_vector: jnp.ndarray,
                                                 nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                                 l_vec_optim: jnp.ndarray,
                                                 active_targets_tuple: Tuple[Tuple[float, float, float, float], ...],
                                                 min_thickness_phys_nm: float) -> jnp.ndarray:
    below_min_mask = (ep_vector < min_thickness_phys_nm) & (ep_vector > 1e-12)
    penalty_thin = jnp.sum(jnp.where(below_min_mask, (min_thickness_phys_nm - ep_vector)**2, 0.0))
    penalty_cost = penalty_thin * 1e5
    
    ep_vector_calc = jnp.maximum(ep_vector, min_thickness_phys_nm)
    num_layers = len(ep_vector_calc)
    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH_arr, nL_arr)
    indices_alternating_T = indices_alternating.T
    
    Ts_raw = vmap(calculate_single_wavelength_T_core, in_axes=(0, None, 0, 0))(
        l_vec_optim, ep_vector_calc, indices_alternating_T, nSub_arr
    )
    Ts = jnp.nan_to_num(Ts_raw, nan=0.0)
    
    total_squared_error, total_points_in_targets = 0.0, 0
    for l_min, l_max, t_min, t_max in active_targets_tuple:
        target_mask = (l_vec_optim >= l_min) & (l_vec_optim <= l_max)
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t_full = t_min + slope * (l_vec_optim - l_min)
        squared_errors_full = (Ts - interpolated_target_t_full)**2
        total_squared_error += jnp.sum(jnp.where(target_mask, squared_errors_full, 0.0))
        total_points_in_targets += jnp.sum(target_mask)
        
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, jnp.inf)
    return jnp.nan_to_num(mse + penalty_cost, nan=jnp.inf, posinf=jnp.inf)

def _run_core_optimization(ep_start_optim: np.ndarray,
                           validated_inputs: Dict, active_targets: List[Dict],
                           min_thickness_phys: float, log_prefix: str = ""
                           ) -> Tuple[Optional[np.ndarray], bool, float, List[str], str]:
    logs = []
    if len(ep_start_optim) == 0:
        return None, False, np.inf, [f"{log_prefix}Cannot optimize an empty structure."], "Empty structure"
    try:
        l_min_optim, l_max_optim, l_step_optim = validated_inputs['l_range_deb'], validated_inputs['l_range_fin'], validated_inputs['l_step']
        nH_material, nL_material, nSub_material = validated_inputs['nH_material'], validated_inputs['nL_material'], validated_inputs['nSub_material']
        
        num_pts_optim = min(max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1), 100)
        l_vec_optim_np = np.geomspace(l_min_optim, l_max_optim, num_pts_optim)
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)

        nH_arr_optim, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        nL_arr_optim, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        nSub_arr_optim, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_h + logs_l + logs_sub)
        if nH_arr_optim is None or nL_arr_optim is None or nSub_arr_optim is None:
            raise RuntimeError("Failed to load indices for optimization.")
        
        active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
        static_args_for_jax = (nH_arr_optim, nL_arr_optim, nSub_arr_optim, l_vec_optim_jax, active_targets_tuple, min_thickness_phys)
        
        value_and_grad_fn = jax.jit(jax.value_and_grad(calculate_mse_for_optimization_penalized_jax))
        def scipy_obj_grad_wrapper(ep_vector_np_in, *args):
            value_jax, grad_jax = value_and_grad_fn(jnp.asarray(ep_vector_np_in), *args)
            value_np = float(np.array(value_jax)) if jnp.isfinite(value_jax) else np.inf
            grad_np = np.array(grad_jax, dtype=np.float64)
            return value_np, np.nan_to_num(grad_np)

        lbfgsb_bounds = [(min_thickness_phys, None)] * len(ep_start_optim)
        options = {'maxiter': MAXITER_HARDCODED, 'maxfun': MAXFUN_HARDCODED, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-8}
        
        logs.append(f"{log_prefix}Starting L-BFGS-B with JAX gradient...")
        result = minimize(scipy_obj_grad_wrapper, ep_start_optim, args=static_args_for_jax, method='L-BFGS-B', jac=True, bounds=lbfgsb_bounds, options=options)
        
        final_cost = result.fun if np.isfinite(result.fun) else np.inf
        result_message_str = result.message.decode('utf-8') if isinstance(result.message, bytes) else str(result.message)
        is_success_or_limit = (result.success or result.status == 1) and np.isfinite(final_cost)

        if is_success_or_limit:
            final_ep = np.maximum(result.x, min_thickness_phys)
            logs.append(f"{log_prefix}Optimization finished. Final cost: {final_cost:.3e}, Msg: {result_message_str}")
            return final_ep, True, final_cost, logs, result_message_str
        else:
            final_ep = np.maximum(ep_start_optim, min_thickness_phys)
            logs.append(f"{log_prefix}Optimization FAILED. Status: {result.status}, Msg: {result_message_str}")
            return final_ep, False, final_cost, logs, result_message_str
    except Exception as e_optim:
        logs.append(f"{log_prefix}Major ERROR during optimization: {e_optim}\n{traceback.format_exc(limit=2)}")
        st.error(f"Critical error during optimization: {e_optim}")
        return ep_start_optim, False, np.inf, logs, f"Exception: {e_optim}"

def _perform_layer_merge_or_removal_only(ep_vector_in: np.ndarray, min_thickness_phys: float,
                                         log_prefix: str = "", target_layer_index: Optional[int] = None,
                                         threshold_for_removal: Optional[float] = None) -> Tuple[Optional[np.ndarray], bool, List[str]]:
    current_ep = ep_vector_in.copy()
    logs = []
    num_layers = len(current_ep)
    structure_changed = False
    ep_after_merge = None 
    if num_layers <= 2 and target_layer_index is None:
        logs.append(f"{log_prefix}Structure <= 2 layers. Removal/merge not possible without target.")
        return current_ep, False, logs
    elif num_layers < 1:
        logs.append(f"{log_prefix}Empty structure.")
        return current_ep, False, logs
    try:
        thin_layer_index = -1 
        min_thickness_found = np.inf
        if target_layer_index is not None:
            if 0 <= target_layer_index < num_layers and current_ep[target_layer_index] >= min_thickness_phys:
                thin_layer_index = target_layer_index
                min_thickness_found = current_ep[target_layer_index]
                logs.append(f"{log_prefix}Manual targeting layer {thin_layer_index + 1} ({min_thickness_found:.3f} nm).")
            else:
                logs.append(f"{log_prefix}Manual target {target_layer_index+1} invalid/too thin. Auto search.")
                target_layer_index = None 
        if target_layer_index is None:
            candidate_indices = np.where(current_ep >= min_thickness_phys)[0]
            if candidate_indices.size == 0:
                logs.append(f"{log_prefix}No layer >= {min_thickness_phys:.3f} nm found.")
                return current_ep, False, logs
            candidate_thicknesses = current_ep[candidate_indices]
            indices_to_consider = candidate_indices
            thicknesses_to_consider = candidate_thicknesses
            if threshold_for_removal is not None:
                mask_below_threshold = thicknesses_to_consider < threshold_for_removal
                if np.any(mask_below_threshold):
                    indices_to_consider = indices_to_consider[mask_below_threshold]
                    thicknesses_to_consider = thicknesses_to_consider[mask_below_threshold]
                    logs.append(f"{log_prefix}Searching among layers < {threshold_for_removal:.3f} nm.")
                else:
                    logs.append(f"{log_prefix}No eligible layer (< {threshold_for_removal:.3f} nm) found.")
                    return current_ep, False, logs 
            if indices_to_consider.size > 0:
                min_idx_local = np.argmin(thicknesses_to_consider)
                thin_layer_index = indices_to_consider[min_idx_local]
                min_thickness_found = thicknesses_to_consider[min_idx_local]
            else:
                logs.append(f"{log_prefix}No final candidate layer found.")
                return current_ep, False, logs
        if thin_layer_index == -1:
            logs.append(f"{log_prefix}Failed to identify layer (unexpected case).")
            return current_ep, False, logs
        thin_layer_thickness = current_ep[thin_layer_index]
        logs.append(f"{log_prefix}Layer identified for action: Index {thin_layer_index} (Layer {thin_layer_index + 1}), thickness {thin_layer_thickness:.3f} nm.")
        if num_layers <= 2 and thin_layer_index == 0: 
            ep_after_merge = current_ep[2:] 
            merged_info = f"Removal of first 2 layers (structure size <= 2)."
            structure_changed = True
        elif num_layers <= 1 and thin_layer_index == 0: 
            ep_after_merge = np.array([]) 
            merged_info = f"Removal of the only layer."
            structure_changed = True
        elif num_layers <= 2 and thin_layer_index == 1: 
            ep_after_merge = current_ep[:-1] 
            merged_info = f"Removal of last layer (structure size 2)."
            structure_changed = True
        elif thin_layer_index == 0: 
            ep_after_merge = current_ep[2:]
            merged_info = f"Removal of first 2 layers."
            structure_changed = True
        elif thin_layer_index == num_layers - 1: 
            if num_layers >= 1: 
                ep_after_merge = current_ep[:-1] 
                merged_info = f"Removal of ONLY the last layer (Layer {num_layers})."
                structure_changed = True
            else: 
                logs.append(f"{log_prefix}Special case: cannot remove last layer (num_layers={num_layers}).")
                return current_ep, False, logs
        else: 
            merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
            ep_before = current_ep[:thin_layer_index - 1]
            ep_after = current_ep[thin_layer_index + 2:]
            ep_after_merge = np.concatenate((ep_before, [merged_thickness], ep_after))
            merged_info = f"Merging layers {thin_layer_index} and {thin_layer_index + 2} around removed layer {thin_layer_index + 1} -> new thickness {merged_thickness:.3f} nm."
            structure_changed = True
        if structure_changed and ep_after_merge is not None:
            logs.append(f"{log_prefix}{merged_info} New structure: {len(ep_after_merge)} layers.")
            ep_after_merge = np.maximum(ep_after_merge, min_thickness_phys) 
            return ep_after_merge, True, logs
        elif structure_changed and ep_after_merge is None: 
            logs.append(f"{log_prefix}Logic error: structure_changed=True but ep_after_merge=None.")
            return current_ep, False, logs
        else: 
            logs.append(f"{log_prefix}No structure modification performed.")
            return current_ep, False, logs
    except Exception as e_merge:
        logs.append(f"{log_prefix}ERROR during merge/removal logic: {e_merge}\n{traceback.format_exc(limit=1)}")
        st.error(f"Internal error during layer removal/merge: {e_merge}")
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
        logs.append(f"{log_prefix}Needle scan impossible on empty structure.")
        return None, np.inf, logs, -1
    logs.append(f"{log_prefix}Starting needle scan ({num_layers_in} layers). Step: {scan_step} nm, needle thick: {base_needle_thickness_nm:.3f} nm.")
    try:
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        nH_arr_optim, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, excel_file_path)
        logs.extend(logs_h)
        nL_arr_optim, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, excel_file_path)
        logs.extend(logs_l)
        nSub_arr_optim, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, excel_file_path)
        logs.extend(logs_sub)
        if nH_arr_optim is None or nL_arr_optim is None or nSub_arr_optim is None:
            raise RuntimeError("Failed to load indices for needle scan.")
        active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
        static_args_cost_fn = (
            nH_arr_optim, nL_arr_optim, nSub_arr_optim,
            l_vec_optim_jax, active_targets_tuple,
            min_thickness_phys
        )
        initial_cost_jax = cost_function_jax(jnp.asarray(ep_vector_in), *static_args_cost_fn)
        initial_cost = float(np.array(initial_cost_jax))
        if not np.isfinite(initial_cost):
            logs.append(f"{log_prefix} ERROR: Initial cost not finite ({initial_cost}). Scan aborted.")
            st.error("Needle Scan Error: Cost of starting structure is not finite.")
            return None, np.inf, logs, -1
        logs.append(f"{log_prefix} Initial cost: {initial_cost:.6e}")
    except Exception as e_prep:
        logs.append(f"{log_prefix} ERROR preparing needle scan: {e_prep}")
        st.error(f"Error preparing needle scan: {e_prep}")
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
            logs.append(f"{log_prefix} WARNING: Failed cost calculation for z={z:.2f}. {e_cost}")
            continue
    if best_ep_found is not None:
        improvement = initial_cost - min_cost_found
        logs.append(f"{log_prefix} Scan finished. {tested_insertions} points tested.")
        logs.append(f"{log_prefix} Best improvement found: {improvement:.6e} (MSE {min_cost_found:.6e})")
        logs.append(f"{log_prefix} Optimal insertion in original layer {best_insertion_idx + 1}.")
        return best_ep_found, min_cost_found, logs, best_insertion_idx
    else:
        logs.append(f"{log_prefix} Scan finished. {tested_insertions} points tested. No improvement found.")
        return None, initial_cost, logs, -1

def _run_needle_iterations(ep_start: np.ndarray, num_needles: int,
                           validated_inputs: Dict, active_targets: List[Dict],
                           min_thickness_phys: float, l_vec_optim_np_in: np.ndarray,
                           scan_step_nm: float, base_needle_thickness_nm: float,
                           excel_file_path: str, log_prefix: str = ""
                           ) -> Tuple[np.ndarray, float, List[str]]:
    logs = []
    best_ep_overall = np.asarray(ep_start).copy()
    best_mse_overall = np.inf
    nH_material = validated_inputs['nH_material']
    nL_material = validated_inputs['nL_material']
    nSub_material = validated_inputs['nSub_material']
    l0_repr = validated_inputs.get('l0', 500.0)
    l_step_optim = validated_inputs['l_step']
    l_min_optim_glob, l_max_optim_glob = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
    num_pts_needle = min(max(2, int(np.round((l_max_optim_glob - l_min_optim_glob) / l_step_optim)) + 1), 100)
    l_vec_optim_np = np.geomspace(l_min_optim_glob, l_max_optim_glob, num_pts_needle)
    l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
    if not l_vec_optim_np.size:
        logs.append(f"{log_prefix} ERROR: Failed to generate lambda vector for needle iterations.")
        st.error("Needle Iterations Error: Lambda vector generation failed.")
        return ep_start, np.inf, logs
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
            raise RuntimeError("Failed to load indices for needle iterations.")
        active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
        static_args_cost_fn = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, min_thickness_phys)
        initial_cost_jax = cost_fn_penalized_jit(jnp.asarray(best_ep_overall), *static_args_cost_fn)
        best_mse_overall = float(np.array(initial_cost_jax))
        if not np.isfinite(best_mse_overall):
            raise ValueError("Initial MSE for needle iterations is not finite.")
        logs.append(f"{log_prefix} Starting needle iterations ({num_needles} max). Initial MSE: {best_mse_overall:.6e}")
    except Exception as e_init:
        logs.append(f"{log_prefix} ERROR calculating initial MSE for needle iterations: {e_init}")
        st.error(f"Error initializing needle iterations: {e_init}")
        return ep_start, np.inf, logs
    for i in range(num_needles):
        logs.append(f"{log_prefix} --- Needle Iteration {i + 1}/{num_needles} ---")
        current_ep_iter = best_ep_overall.copy()
        num_layers_current = len(current_ep_iter)
        if num_layers_current == 0:
            logs.append(f"{log_prefix} Empty structure, stopping needle iterations."); break
        st.write(f"{log_prefix} Needle scan {i+1}...")
        ep_after_scan, cost_after_scan, scan_logs, inserted_idx = _perform_needle_insertion_scan(
            current_ep_iter,
            nH_material, nL_material, nSub_material,
            l_vec_optim_np, active_targets,
            cost_fn_penalized_jit,
            min_thickness_phys, base_needle_thickness_nm, scan_step_nm, l0_repr,
            excel_file_path, log_prefix=f"{log_prefix}  [Scan {i+1}] "
        )
        logs.extend(scan_logs)
        if ep_after_scan is None:
            logs.append(f"{log_prefix} Needle scan {i + 1} found no improvement. Stopping needle iterations."); break
        logs.append(f"{log_prefix} Scan {i + 1} found potential improvement. Re-optimizing...")
        st.write(f"{log_prefix} Re-optimizing after needle {i+1}...")
        ep_after_reopt, optim_success, final_cost_reopt, optim_logs, optim_status_msg = \
            _run_core_optimization(ep_after_scan, validated_inputs, active_targets,
                                   min_thickness_phys, log_prefix=f"{log_prefix}  [Re-Opt {i+1}] ")
        logs.extend(optim_logs)
        if not optim_success:
            logs.append(f"{log_prefix} Re-optimization after scan {i + 1} FAILED. Stopping needle iterations."); break
        logs.append(f"{log_prefix} Re-optimization {i + 1} successful. New MSE: {final_cost_reopt:.6e}.")
        if final_cost_reopt < best_mse_overall - MSE_IMPROVEMENT_TOLERANCE:
            logs.append(f"{log_prefix}  MSE improved compared to previous best ({best_mse_overall:.6e}). Updating.")
            best_ep_overall = ep_after_reopt.copy()
            best_mse_overall = final_cost_reopt
        else:
            logs.append(f"{log_prefix}  New MSE ({final_cost_reopt:.6e}) not significantly better than previous ({best_mse_overall:.6e}). Stopping needle iterations.")
            best_ep_overall = ep_after_reopt.copy()
            best_mse_overall = final_cost_reopt
            break
    logs.append(f"{log_prefix} End of needle iterations. Best final MSE: {best_mse_overall:.6e}")
    return best_ep_overall, best_mse_overall, logs

def run_auto_mode(initial_ep: Optional[np.ndarray],
                  validated_inputs: Dict, active_targets: List[Dict],
                  excel_file_path: str, log_callback: Callable):
    logs = []
    start_time_auto = time.time()
    log_callback("#"*10 + f" Starting Auto Mode (Max {AUTO_MAX_CYCLES} Cycles) " + "#"*10)
    best_ep_so_far = None
    best_mse_so_far = np.inf
    num_cycles_done = 0
    termination_reason = f"Max {AUTO_MAX_CYCLES} cycles reached"
    threshold_for_thin_removal = validated_inputs.get('auto_thin_threshold', 1.0)
    log_callback(f"  Auto removal threshold: {threshold_for_thin_removal:.3f} nm")
    try:
        current_ep = None
        if initial_ep is not None:
            log_callback("  Auto Mode: Using previous optimized structure.")
            current_ep = initial_ep.copy()
            l_min_optim, l_max_optim = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
            l_step_optim = validated_inputs['l_step']
            num_pts = min(max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1), 100)
            l_vec_optim_np_auto = np.geomspace(l_min_optim, l_max_optim, num_pts)
            l_vec_optim_np_auto = l_vec_optim_np_auto[(l_vec_optim_np_auto > 0) & np.isfinite(l_vec_optim_np_auto)]
            if not l_vec_optim_np_auto.size: raise ValueError("Failed to generate lambda for initial auto MSE calc.")
            l_vec_optim_jax = jnp.asarray(l_vec_optim_np_auto)
            nH_arr, log_h = _get_nk_array_for_lambda_vec(validated_inputs['nH_material'], l_vec_optim_jax, excel_file_path)
            nL_arr, log_l = _get_nk_array_for_lambda_vec(validated_inputs['nL_material'], l_vec_optim_jax, excel_file_path)
            nSub_arr, log_sub = _get_nk_array_for_lambda_vec(validated_inputs['nSub_material'], l_vec_optim_jax, excel_file_path)
            log_callback(log_h); log_callback(log_l); log_callback(log_sub)
            if nH_arr is None or nL_arr is None or nSub_arr is None: raise RuntimeError("Failed to load indices for initial auto MSE.")
            active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
            static_args = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, MIN_THICKNESS_PHYS_NM)
            cost_fn_jit = jax.jit(calculate_mse_for_optimization_penalized_jax)
            initial_mse_jax = cost_fn_jit(jnp.asarray(current_ep), *static_args)
            initial_mse = float(np.array(initial_mse_jax))
            if not np.isfinite(initial_mse): raise ValueError("Initial MSE (from optimized state) not finite.")
            best_mse_so_far = initial_mse
            best_ep_so_far = current_ep.copy()
            log_callback(f"  Initial MSE (from optimized state): {best_mse_so_far:.6e}")
        else:
            log_callback("  Auto Mode: Using nominal structure (QWOT).")
            emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
            if not emp_list: raise ValueError("Nominal QWOT empty.")
            ep_nominal, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'],
                                                            validated_inputs['nH_material'], validated_inputs['nL_material'],
                                                            excel_file_path)
            log_callback(logs_ep_init)
            if ep_nominal is None: raise RuntimeError("Failed to calculate initial nominal thicknesses.")
            log_callback(f"  Nominal structure: {len(ep_nominal)} layers. Starting initial optimization...")
            st.info("Auto Mode: Initial optimization of nominal structure...")
            ep_after_initial_opt, initial_opt_success, initial_mse, initial_opt_logs, initial_opt_msg = \
                _run_core_optimization(ep_nominal, validated_inputs, active_targets,
                                       MIN_THICKNESS_PHYS_NM, log_prefix="  [Auto Init Opt] ")
            log_callback(initial_opt_logs)
            if not initial_opt_success:
                log_callback(f"ERROR: Failed initial optimization in Auto Mode ({initial_opt_msg}). Aborting.")
                st.error(f"Failed initial optimization of Auto Mode: {initial_opt_msg}")
                return None, np.inf, logs
            log_callback(f"  Initial optimization finished. MSE: {initial_mse:.6e}")
            best_ep_so_far = ep_after_initial_opt.copy()
            best_mse_so_far = initial_mse
        if best_ep_so_far is None or not np.isfinite(best_mse_so_far):
            raise RuntimeError("Invalid starting state for Auto cycles.")
        log_callback(f"--- Starting Auto Cycles (Starting MSE: {best_mse_so_far:.6e}, {len(best_ep_so_far)} layers) ---")
        for cycle_num in range(AUTO_MAX_CYCLES):
            log_callback(f"\n--- Auto Cycle {cycle_num + 1} / {AUTO_MAX_CYCLES} ---")
            st.info(f"Auto Cycle {cycle_num + 1}/{AUTO_MAX_CYCLES} | Current MSE: {best_mse_so_far:.3e}")
            mse_at_cycle_start = best_mse_so_far
            ep_at_cycle_start = best_ep_so_far.copy()
            cycle_improved_globally = False
            log_callback(f"  [Cycle {cycle_num+1}] Needle Phase ({AUTO_NEEDLES_PER_CYCLE} max iterations)...")
            st.write(f"Cycle {cycle_num + 1}: Needle Phase...")
            l_min_optim, l_max_optim = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
            l_step_optim = validated_inputs['l_step']
            num_pts_needle_auto = min(max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1), 100)
            l_vec_optim_np_needle_auto = np.geomspace(l_min_optim, l_max_optim, num_pts_needle_auto)
            l_vec_optim_np_needle_auto = l_vec_optim_np_needle_auto[(l_vec_optim_np_needle_auto > 0) & np.isfinite(l_vec_optim_np_needle_auto)]
            if not l_vec_optim_np_needle_auto.size:
                log_callback("  ERROR: cannot generate lambda for needle phase. Cycle aborted.")
                break
            ep_after_needles, mse_after_needles, needle_logs = \
                _run_needle_iterations(best_ep_so_far, AUTO_NEEDLES_PER_CYCLE, validated_inputs, active_targets,
                                       MIN_THICKNESS_PHYS_NM, l_vec_optim_np_needle_auto,
                                       DEFAULT_NEEDLE_SCAN_STEP_NM, BASE_NEEDLE_THICKNESS_NM,
                                       excel_file_path, log_prefix=f"    [Needle {cycle_num+1}] ")
            log_callback(needle_logs)
            log_callback(f"  [Cycle {cycle_num+1}] End Needle Phase. MSE: {mse_after_needles:.6e}")
            if mse_after_needles < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                log_callback(f"    Global improvement by needle phase (vs {best_mse_so_far:.6e}).")
                best_ep_so_far = ep_after_needles.copy()
                best_mse_so_far = mse_after_needles
                cycle_improved_globally = True
            else:
                log_callback(f"    No global improvement by needle phase (vs {best_mse_so_far:.6e}).")
                best_ep_so_far = ep_after_needles.copy() 
                best_mse_so_far = mse_after_needles   
            log_callback(f"  [Cycle {cycle_num+1}] Thinning Phase (< {threshold_for_thin_removal:.3f} nm) + Re-Opt...")
            st.write(f"Cycle {cycle_num + 1}: Thinning Phase...")
            layers_removed_this_cycle = 0;
            max_thinning_attempts = len(best_ep_so_far) + 2
            for attempt in range(max_thinning_attempts):
                current_num_layers_thin = len(best_ep_so_far)
                if current_num_layers_thin <= 2:
                    log_callback("    Structure too small (< 3 layers), stopping thinning.")
                    break
                ep_after_single_removal, structure_changed, removal_logs = \
                    _perform_layer_merge_or_removal_only(best_ep_so_far, MIN_THICKNESS_PHYS_NM,
                                                         log_prefix=f"    [Thin {cycle_num+1}.{attempt+1}] ",
                                                         threshold_for_removal=threshold_for_thin_removal)
                log_callback(removal_logs)
                if structure_changed and ep_after_single_removal is not None:
                    layers_removed_this_cycle += 1
                    log_callback(f"    Layer removed/merged ({layers_removed_this_cycle} in this cycle). Re-optimizing ({len(ep_after_single_removal)} layers)...")
                    st.write(f"Cycle {cycle_num + 1}: Re-opt after removal {layers_removed_this_cycle}...")
                    ep_after_thin_reopt, thin_reopt_success, mse_after_thin_reopt, thin_reopt_logs, thin_reopt_msg = \
                        _run_core_optimization(ep_after_single_removal, validated_inputs, active_targets,
                                               MIN_THICKNESS_PHYS_NM, log_prefix=f"      [ReOptThin {cycle_num+1}.{attempt+1}] ")
                    log_callback(thin_reopt_logs)
                    if thin_reopt_success:
                        log_callback(f"      Re-optimization successful. MSE: {mse_after_thin_reopt:.6e}")
                        if mse_after_thin_reopt < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                            log_callback(f"      Global improvement by thinning+reopt (vs {best_mse_so_far:.6e}).")
                            best_ep_so_far = ep_after_thin_reopt.copy()
                            best_mse_so_far = mse_after_thin_reopt
                            cycle_improved_globally = True
                        else:
                            log_callback(f"      No global improvement (vs {best_mse_so_far:.6e}). Continuing with this result.")
                            best_ep_so_far = ep_after_thin_reopt.copy()
                            best_mse_so_far = mse_after_thin_reopt
                    else:
                        log_callback(f"    WARNING: Re-optimization after thinning FAILED ({thin_reopt_msg}). Stopping thinning for this cycle.")
                        best_ep_so_far = ep_after_single_removal.copy()
                        try:
                            current_mse_jax = cost_fn_jit(jnp.asarray(best_ep_so_far), *static_args) # Re-use static_args from Auto Mode start
                            best_mse_so_far = float(np.array(current_mse_jax))
                            if not np.isfinite(best_mse_so_far): best_mse_so_far = np.inf
                            log_callback(f"      MSE after failed re-opt (reduced structure, not opt): {best_mse_so_far:.6e}")
                        except Exception as e_cost_fail:
                            log_callback(f"      ERROR recalculating MSE after failed re-opt: {e_cost_fail}"); best_mse_so_far = np.inf
                        break
                else:
                    log_callback("    No further layers to remove/merge in this phase.")
                    break
            log_callback(f"  [Cycle {cycle_num+1}] End Thinning Phase. {layers_removed_this_cycle} layer(s) removed.")
            num_cycles_done += 1
            log_callback(f"--- End Auto Cycle {cycle_num + 1} --- Best current MSE: {best_mse_so_far:.6e} ({len(best_ep_so_far)} layers) ---")
            if not cycle_improved_globally and best_mse_so_far >= mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE :
                log_callback(f"No significant improvement in Cycle {cycle_num + 1} (Start: {mse_at_cycle_start:.6e}, End: {best_mse_so_far:.6e}). Stopping Auto Mode.")
                termination_reason = f"No improvement (Cycle {cycle_num + 1})"
                if best_mse_so_far > mse_at_cycle_start + MSE_IMPROVEMENT_TOLERANCE : # Check if MSE actually got worse
                    log_callback("  MSE increased, reverting to state before this cycle.")
                    best_ep_so_far = ep_at_cycle_start.copy()
                    best_mse_so_far = mse_at_cycle_start
                break
        log_callback(f"\n--- Auto Mode Finished after {num_cycles_done} cycles ---")
        log_callback(f"Reason: {termination_reason}")
        log_callback(f"Best final MSE: {best_mse_so_far:.6e} with {len(best_ep_so_far)} layers.")
        return best_ep_so_far, best_mse_so_far, logs
    except (ValueError, RuntimeError, TypeError) as e:
        log_callback(f"FATAL ERROR during Auto Mode (Setup/Workflow): {e}")
        st.error(f"Auto Mode Error: {e}")
        return None, np.inf, logs
    except Exception as e_fatal:
        log_callback(f"Unexpected FATAL ERROR during Auto Mode: {type(e_fatal).__name__}: {e_fatal}")
        st.error(f"Unexpected Auto Mode Error: {e_fatal}")
        traceback.print_exc()
        return None, np.inf, logs

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
        total_squared_error += jnp.sum(masked_sq_error, axis=-1)
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
    log_callback(f"  [Scan l0={current_l0:.2f}] Testing {num_combinations:,} QWOT combinations (1.0x/2.0x)...")
    precompute_start_time = time.time()
    st.write(f"Scan l0={current_l0:.1f}: Pre-calculating matrices...")
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
        log_callback(f"    Pre-calculation of matrices (l0={current_l0:.2f}) finished in {time.time() - precompute_start_time:.3f}s.")
    except Exception as e_mat:
        logs.append(f"  ERROR Pre-calculating Matrices for l0={current_l0:.2f}: {e_mat}")
        st.error(f"Error pre-calculating QWOT scan matrices: {e_mat}")
        return np.inf, None, logs
    N = initial_layer_number
    N1 = N // 2
    N2 = N - N1
    num_comb1 = 2**N1
    num_comb2 = 2**N2
    log_callback(f"    Calculating partial products 1 ({num_comb1:,} combinations)...")
    st.write(f"Scan l0={current_l0:.1f}: Partial products 1...")
    half1_start_time = time.time()
    indices1 = jnp.arange(num_comb1)
    powers1 = 2**jnp.arange(N1)
    multiplier_indices1 = jnp.not_equal(indices1[:, None] & powers1, 0).astype(jnp.int32)
    matrices_half1 = all_layer_matrices[:N1]
    compute_half_product_jit = jax.jit(compute_half_product)
    partial_products1 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices1, matrices_half1)
    partial_products1.block_until_ready()
    log_callback(f"    Partial products 1 finished in {time.time() - half1_start_time:.3f}s.")
    log_callback(f"    Calculating partial products 2 ({num_comb2:,} combinations)...")
    st.write(f"Scan l0={current_l0:.1f}: Partial products 2...")
    half2_start_time = time.time()
    indices2 = jnp.arange(num_comb2)
    powers2 = 2**jnp.arange(N2)
    multiplier_indices2 = jnp.not_equal(indices2[:, None] & powers2, 0).astype(jnp.int32)
    matrices_half2 = all_layer_matrices[N1:]
    partial_products2 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices2, matrices_half2)
    partial_products2.block_until_ready()
    log_callback(f"    Partial products 2 finished in {time.time() - half2_start_time:.3f}s.")
    log_callback(f"    Combining and calculating MSE ({num_comb1 * num_comb2:,} total)...")
    st.write(f"Scan l0={current_l0:.1f}: Combining & MSE...")
    combine_start_time = time.time()
    combine_and_calc_mse_jit = jax.jit(combine_and_calc_mse)
    vmap_inner = vmap(combine_and_calc_mse_jit, in_axes=(None, 0, None, None, None))
    vmap_outer = vmap(vmap_inner, in_axes=(0, None, None, None, None))
    all_mses_nested = vmap_outer(partial_products1, partial_products2, nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple)
    all_mses_nested.block_until_ready()
    log_callback(f"    Combination and MSE finished in {time.time() - combine_start_time:.3f}s.")
    all_mses_flat = all_mses_nested.reshape(-1)
    best_idx_flat = jnp.argmin(all_mses_flat)
    current_best_mse = float(all_mses_flat[best_idx_flat])
    if not np.isfinite(current_best_mse):
        logs.append(f"    Warning: No valid result (finite MSE) found for l0={current_l0:.2f}.")
        return np.inf, None, logs
    best_idx_half1, best_idx_half2 = jnp.unravel_index(best_idx_flat, (num_comb1, num_comb2))
    best_indices_h1 = multiplier_indices1[best_idx_half1]
    best_indices_h2 = multiplier_indices2[best_idx_half2]
    best_multipliers_h1 = 1.0 + best_indices_h1.astype(jnp.float64)
    best_multipliers_h2 = 1.0 + best_indices_h2.astype(jnp.float64)
    current_best_multipliers = jnp.concatenate([best_multipliers_h1, best_multipliers_h2])
    logs.append(f"    Best MSE for scan l0={current_l0:.2f}: {current_best_mse:.6e}")
    return current_best_mse, np.array(current_best_multipliers), logs

def get_material_input(role: str) -> Tuple[Optional[MaterialInputType], str]:
    if role == 'H':
        sel_key, const_r_key, const_i_key = "selected_H", "nH_r", "nH_i"
    elif role == 'L':
        sel_key, const_r_key, const_i_key = "selected_L", "nL_r", "nL_i"
    elif role == 'Sub':
        sel_key, const_r_key, const_i_key = "selected_Sub", "nSub_r", None
    else:
        st.error(f"Unknown material role: {role}")
        return None, "Role Error"
    selection = st.session_state.get(sel_key)
    if selection == "Constant":
        n_real = st.session_state.get(const_r_key, 1.0 if role != 'Sub' else 1.5)
        n_imag = 0.0
        if const_i_key and role in ['H', 'L']:
           n_imag = st.session_state.get(const_i_key, 0.0)
        valid_n = True
        valid_k = True
        if n_real <= 0:
            n_real = 1.0
            valid_n = False
        if n_imag < 0:
            n_imag = 0.0
            valid_k = False
        mat_repr = f"Constant ({n_real:.3f}{'+' if n_imag>=0 else ''}{n_imag:.3f}j)"
        if not valid_n or not valid_k:
            mat_repr += " (Adjusted)"
        return complex(n_real, n_imag), mat_repr
    elif isinstance(selection, str) and selection:
        return selection, selection
    else:
        st.error(f"Material selection for '{role}' invalid or missing in session_state.")
        return None, "Selection Error"

def validate_targets() -> Optional[List[Dict]]:
    active_targets, logs = [], []
    is_valid = True
    target_type = st.session_state.get("target_definition_type", "Manual Targets")

    if target_type == "Import from File":
        target_data = st.session_state.get('uploaded_target_data')
        if target_data is None or len(target_data) < 2:
            st.warning("Please upload a valid Excel file with target data.")
            return []
        for i in range(len(target_data) - 1):
            l_min, t_min = target_data[i]
            l_max, t_max = target_data[i+1]
            if l_max <= l_min:
                logs.append(f"File Target Error: Wavelengths not strictly increasing at point {i+1}.")
                is_valid = False
                continue
            active_targets.append({'min': float(l_min), 'max': float(l_max), 'target_min': float(t_min), 'target_max': float(t_max)})
        if not is_valid:
            st.error("Errors found in uploaded target file (e.g., non-increasing wavelengths).")
            return None
        return active_targets
    else:
        if 'targets' not in st.session_state or not isinstance(st.session_state.targets, list):
            st.error("Internal error: Target list missing or invalid in session_state.")
            return None
        for i, target_state in enumerate(st.session_state.targets):
            if target_state.get('enabled', False):
                try:
                    l_min = float(target_state['min'])
                    l_max = float(target_state['max'])
                    t_min = float(target_state['target_min'])
                    t_max = float(target_state['target_max'])
                    if l_max < l_min:
                        logs.append(f"Target {i+1} Error: λ max ({l_max:.1f}) < λ min ({l_min:.1f}).")
                        is_valid = False; continue
                    if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0):
                        logs.append(f"Target {i+1} Error: Transmittance out of [0, 1] (Tmin={t_min:.2f}, Tmax={t_max:.2f}).")
                        is_valid = False; continue
                    active_targets.append({
                        'min': l_min, 'max': l_max,
                        'target_min': t_min, 'target_max': t_max
                    })
                except (KeyError, ValueError, TypeError) as e:
                    logs.append(f"Target {i+1} Error: Missing or invalid data ({e}).")
                    is_valid = False; continue
        if not is_valid:
            st.warning("Errors exist in the active spectral target definitions. Please correct.")
            return None
        elif not active_targets:
            return []
        else:
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
    st.session_state.optimized_ep = None
    st.session_state.is_optimized_state = False
    st.session_state.ep_history = deque(maxlen=5)
    st.session_state.optimized_qwot_str = ""
    st.session_state.last_mse = None

def set_optimized_as_nominal_wrapper():
    if not st.session_state.get('is_optimized_state') or st.session_state.get('optimized_ep') is None:
        st.error("No valid optimized structure to set as nominal.")
        return
    try:
        l0 = st.session_state.l0
        nH_mat, _ = get_material_input('H')
        nL_mat, _ = get_material_input('L')
        if nH_mat is None or nL_mat is None:
            st.error("Cannot retrieve H/L materials to recalculate QWOT.")
            return
        optimized_qwots, logs_qwot = calculate_qwot_from_ep(st.session_state.optimized_ep, l0, nH_mat, nL_mat, EXCEL_FILE_PATH)
        if optimized_qwots is None:
            st.error("Error recalculating QWOT from the optimized structure.")
            return
        if np.any(np.isnan(optimized_qwots)):
            st.warning("Recalculated QWOT contains NaNs (likely invalid index at l0). Nominal QWOT not updated.")
        else:
            new_qwot_str = ",".join([f"{q:.6f}" for q in optimized_qwots])
            st.session_state.current_qwot = new_qwot_str
            st.success("Optimized structure set as new Nominal (QWOT updated).")
            clear_optimized_state()
    except Exception as e:
        st.error(f"Unexpected error setting optimized as nominal: {e}")

def undo_remove_wrapper():
    if not st.session_state.get('ep_history'):
        st.info("Undo history is empty.")
        return
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
                st.session_state.optimized_qwot_str = "QWOT N/A (after undo)"
        else:
            st.session_state.optimized_qwot_str = "QWOT Material Error (after undo)"
        st.info("State restored. Recalculating...")
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {
            'is_optimized_run': True,
            'method_name': "Optimized (Undo)",
            'force_ep': st.session_state.optimized_ep
            }
    except IndexError:
        st.warning("Undo history is empty (internal error?).")
    except Exception as e:
        st.error(f"Unexpected error during undo: {e}")
        clear_optimized_state()

def run_calculation_wrapper(is_optimized_run: bool, method_name: str = "", force_ep: Optional[np.ndarray] = None):
    calc_type = 'Optimized' if is_optimized_run else 'Nominal'
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None
    with st.spinner(f"{calc_type} calculation in progress..."):
        try:
            active_targets = validate_targets()
            if active_targets is None:
                st.error("Target definition invalid. Check logs and correct.")
                return
            if not active_targets:
                st.warning("No active targets. Default lambda range used (400-700nm). MSE calculation will be N/A.")
                l_min_plot, l_max_plot = 400.0, 700.0
            else:
                l_min_plot, l_max_plot = get_lambda_range_from_targets(active_targets)
                if l_min_plot is None or l_max_plot is None or l_max_plot < l_min_plot:
                    st.error("Could not determine a valid lambda range from targets.")
                    return
            validated_inputs = {
                'l0': st.session_state.l0,
                'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot,
                'auto_thin_threshold': st.session_state.auto_thin_threshold,
                'l_range_deb': l_min_plot,
                'l_range_fin': l_max_plot,
            }
            nH_mat, nH_repr = get_material_input('H')
            nL_mat, nL_repr = get_material_input('L')
            nSub_mat, nSub_repr = get_material_input('Sub')
            if nH_mat is None or nL_mat is None or nSub_mat is None:
                st.error("Material definition error. Check selections and/or Excel files.")
                return
            validated_inputs['nH_material'] = nH_mat
            validated_inputs['nL_material'] = nL_mat
            validated_inputs['nSub_material'] = nSub_mat
            ep_to_calculate = None
            if force_ep is not None:
                ep_to_calculate = force_ep.copy()
            elif is_optimized_run and st.session_state.get('optimized_ep') is not None:
                ep_to_calculate = st.session_state.optimized_ep.copy()
            else:
                emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
                if not emp_list and calc_type == 'Nominal':
                    ep_to_calculate = np.array([], dtype=np.float64)
                elif not emp_list and calc_type == 'Optimized':
                    st.error("Cannot start an optimized calculation if nominal QWOT is empty and no previous optimized state exists.")
                    return
                else:
                    ep_calc, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                    if ep_calc is None:
                        st.error("Failed to calculate initial thicknesses from QWOT.")
                        return
                    ep_to_calculate = ep_calc.copy()
            st.session_state.current_ep = ep_to_calculate.copy() if ep_to_calculate is not None else None
            num_plot_points = max(501, int(np.round((l_max_plot - l_min_plot) / validated_inputs['l_step'])) * 3 + 1)
            l_vec_plot_fine_np = np.linspace(l_min_plot, l_max_plot, num_plot_points)
            l_vec_plot_fine_np = l_vec_plot_fine_np[(l_vec_plot_fine_np > 0) & np.isfinite(l_vec_plot_fine_np)]
            if not l_vec_plot_fine_np.size:
                st.error("Could not generate a valid lambda vector for plotting.")
                return
            start_calc_time = time.time()
            results_fine, calc_logs = calculate_T_from_ep_jax(
                ep_to_calculate, nH_mat, nL_mat, nSub_mat, l_vec_plot_fine_np, EXCEL_FILE_PATH
            )
            if results_fine is None:
                st.error("Main transmittance calculation failed.")
                return
            st.session_state.last_calc_results = {
                'res_fine': results_fine,
                'method_name': method_name,
                'ep_used': ep_to_calculate.copy() if ep_to_calculate is not None else None,
                'l0_used': validated_inputs['l0'],
                'nH_used': nH_mat, 'nL_used': nL_mat, 'nSub_used': nSub_mat,
            }
            if active_targets:
                num_pts_optim_display = min(max(2, int(np.round((l_max_plot - l_min_plot) / validated_inputs['l_step'])) + 1), 100)
                l_vec_optim_np_display = np.geomspace(l_min_plot, l_max_plot, num_pts_optim_display)
                l_vec_optim_np_display = l_vec_optim_np_display[(l_vec_optim_np_display > 0) & np.isfinite(l_vec_optim_np_display)]
                if l_vec_optim_np_display.size > 0:
                    results_optim_grid, logs_mse_calc = calculate_T_from_ep_jax(
                        ep_to_calculate, nH_mat, nL_mat, nSub_mat, l_vec_optim_np_display, EXCEL_FILE_PATH
                    )
                    if results_optim_grid is not None:
                        mse_display, num_pts_mse = calculate_final_mse(results_optim_grid, active_targets)
                        st.session_state.last_mse = mse_display
                        st.session_state.last_calc_results['res_optim_grid'] = results_optim_grid
                    else:
                        st.session_state.last_mse = None
                else:
                    st.session_state.last_mse = None
            else:
                st.session_state.last_mse = None
            st.session_state.is_optimized_state = is_optimized_run
            if not is_optimized_run:
                clear_optimized_state()
                st.session_state.current_ep = ep_to_calculate.copy() if ep_to_calculate is not None else None
            st.success(f"{calc_type} calculation finished.")
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during {calc_type} calculation: {e}")
        except Exception as e_fatal:
            st.error(f"Unexpected error during {calc_type} calculation: {e_fatal}")
        finally:
            pass

def run_local_optimization_wrapper():
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None
    clear_optimized_state()
    with st.spinner("Local optimization in progress..."):
        try:
            active_targets = validate_targets()
            if active_targets is None or not active_targets:
                st.error("Local optimization requires active and valid targets.")
                return
            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None:
                st.error("Could not determine lambda range for optimization.")
                return
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
                st.error("Material definition error for optimization.")
                return
            validated_inputs['nH_material'] = nH_mat
            validated_inputs['nL_material'] = nL_mat
            validated_inputs['nSub_material'] = nSub_mat
            emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
            if not emp_list:
                st.error("Nominal QWOT empty, cannot start local optimization.")
                return
            ep_start, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
            if ep_start is None:
                st.error("Failed initial thickness calculation for local optimization.")
                return
            final_ep, success, final_cost, optim_logs, msg = \
                _run_core_optimization(ep_start, validated_inputs, active_targets,
                                       MIN_THICKNESS_PHYS_NM, log_prefix="  [Opt Local] ")
            if success and final_ep is not None:
                st.session_state.optimized_ep = final_ep.copy()
                st.session_state.current_ep = final_ep.copy()
                st.session_state.is_optimized_state = True
                st.session_state.last_mse = final_cost
                qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                    st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                else:
                    st.session_state.optimized_qwot_str = "QWOT N/A"
                st.success(f"Local optimization finished ({msg}). MSE: {final_cost:.4e}")
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Opt Local"}
            else:
                st.error(f"Local optimization failed: {msg}")
                st.session_state.is_optimized_state = False
                st.session_state.optimized_ep = None
                st.session_state.current_ep = ep_start.copy()
                st.session_state.last_mse = None
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during local optimization: {e}")
            clear_optimized_state()
        except Exception as e_fatal:
            st.error(f"Unexpected error during local optimization: {e_fatal}")
            clear_optimized_state()

def run_scan_optimization_wrapper():
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None
    clear_optimized_state()
    with st.spinner("QWOT Scan + Double Optimization in progress (can be very long)..."):
        try:
            initial_layer_number_scan = len([q for q in st.session_state.current_qwot.split(',') if q.strip()])
            if 'initial_layer_number' not in st.session_state or st.session_state.initial_layer_number == 0 :
                st.session_state.initial_layer_number = initial_layer_number_scan

            if st.session_state.initial_layer_number == 0 :
                st.error("Nominal QWOT is empty and initial layer number is not defined / is zero.")
                return
            
            initial_layer_number_to_use = min(st.session_state.initial_layer_number, 18)
            if initial_layer_number_to_use != st.session_state.initial_layer_number:
                st.warning(f"Scan + Opt: Number of layers capped at 18 (was {st.session_state.initial_layer_number}).")
            st.session_state.initial_layer_number = initial_layer_number_to_use

            active_targets = validate_targets()
            if active_targets is None or not active_targets:
                st.error("QWOT Scan+Opt requires active and valid targets.")
                return
            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None:
                st.error("Could not determine lambda range for QWOT Scan+Opt.")
                return
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
                st.error("Material definition error for QWOT Scan+Opt.")
                return
            validated_inputs['nH_material'] = nH_mat
            validated_inputs['nL_material'] = nL_mat
            validated_inputs['nSub_material'] = nSub_mat
            
            num_pts_sparse_scan = min(max(2, int(np.round((l_max_opt - l_min_opt) / validated_inputs['l_step'])) + 1) // 2 +1 , 100)
            l_vec_eval_sparse_np = np.geomspace(l_min_opt, l_max_opt, num_pts_sparse_scan)
            l_vec_eval_sparse_np = l_vec_eval_sparse_np[(l_vec_eval_sparse_np > 0) & np.isfinite(l_vec_eval_sparse_np)]

            if not l_vec_eval_sparse_np.size: raise ValueError("Failed sparse lambda generation for Scan.")
            l_vec_eval_sparse_jax = jnp.asarray(l_vec_eval_sparse_np)
            nSub_arr_scan, logs_sub_scan = _get_nk_array_for_lambda_vec(nSub_mat, l_vec_eval_sparse_jax, EXCEL_FILE_PATH)
            if nSub_arr_scan is None: raise RuntimeError("Failed substrate index preparation for scan.")
            active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
            l0_nominal = validated_inputs['l0']
            l0_values_to_test = sorted(list(set([l0_nominal, l0_nominal * 1.2, l0_nominal * 0.8])))
            l0_values_to_test = [l for l in l0_values_to_test if l > 1e-6]
            scan_candidates = []
            overall_scan_logs = []
            st.write(f"Phase 1: Scanning QWOT combinations for l0 = {l0_values_to_test}...")
            for l0_scan in l0_values_to_test:
                st.write(f"Scanning for l0={l0_scan:.1f}...")
                try:
                    nH_c_l0, log_h_l0 = _get_nk_at_lambda(nH_mat, l0_scan, EXCEL_FILE_PATH)
                    nL_c_l0, log_l_l0 = _get_nk_at_lambda(nL_mat, l0_scan, EXCEL_FILE_PATH)
                    overall_scan_logs.extend(log_h_l0); overall_scan_logs.extend(log_l_l0)
                    if nH_c_l0 is None or nL_c_l0 is None:
                        st.warning(f"Could not get indices at l0={l0_scan:.1f}, skipping this value.")
                        continue
                    scan_mse, scan_multipliers, scan_logs = _execute_split_stack_scan(
                        l0_scan, validated_inputs['initial_layer_number'],
                        nH_c_l0, nL_c_l0,
                        nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple,
                        add_log
                    )
                    overall_scan_logs.extend(scan_logs)
                    if scan_multipliers is not None and np.isfinite(scan_mse):
                        scan_candidates.append({
                            'l0': l0_scan,
                            'mse_scan': scan_mse,
                            'multipliers': scan_multipliers
                        })
                        st.write(f"-> Best scan result for l0={l0_scan:.1f}: MSE = {scan_mse:.4e}")
                    else:
                        st.warning(f"Scan for l0={l0_scan:.1f} did not yield a valid result.")
                except Exception as e_scan_l0:
                    st.warning(f"Error during scan for l0={l0_scan:.2f}: {e_scan_l0}")
            if not scan_candidates:
                st.error("QWOT Scan found no valid initial candidates after scanning all l0 values.")
                return
            st.write(f"\nPhase 2: Double Optimization for {len(scan_candidates)} candidate(s)...")
            optimization_results = []
            for idx, candidate in enumerate(scan_candidates):
                l0_cand = candidate['l0']
                multipliers_cand = candidate['multipliers']
                st.write(f"-- Optimizing Candidate {idx+1}/{len(scan_candidates)} (from l0={l0_cand:.1f}, scan MSE={candidate['mse_scan']:.4e}) --")
                ep_start_optim, logs_ep_best = calculate_initial_ep(multipliers_cand, l0_cand, nH_mat, nL_mat, EXCEL_FILE_PATH)
                if ep_start_optim is None:
                    st.warning(f"Failed thickness calculation for candidate {idx+1}. Skipping.")
                    continue
                st.write(f"    Running Optimization 1/2...")
                final_ep_1, success_1, final_cost_1, optim_logs_1, msg_1 = \
                    _run_core_optimization(ep_start_optim, validated_inputs, active_targets,
                                           MIN_THICKNESS_PHYS_NM, log_prefix=f"  [OptScan {idx+1}-1] ")
                if success_1 and final_ep_1 is not None:
                    st.write(f"    Optimization 1/2 finished ({msg_1}). MSE: {final_cost_1:.4e}. Running Optimization 2/2...")
                    final_ep_2, success_2, final_cost_2, optim_logs_2, msg_2 = \
                        _run_core_optimization(final_ep_1, validated_inputs, active_targets, 
                                               MIN_THICKNESS_PHYS_NM, log_prefix=f"  [OptScan {idx+1}-2] ")
                    if success_2 and final_ep_2 is not None:
                        st.write(f"    Optimization 2/2 finished ({msg_2}). Final MSE: {final_cost_2:.4e}")
                        optimization_results.append({
                            'l0_origin': l0_cand,
                            'final_ep': final_ep_2,
                            'final_mse': final_cost_2,
                            'message': f"Opt1: {msg_1} | Opt2: {msg_2}"
                        })
                    else:
                        st.warning(f"    Optimization 2/2 FAILED for candidate {idx+1}: {msg_2}. Discarding this candidate.")
                else:
                    st.warning(f"    Optimization 1/2 FAILED for candidate {idx+1}: {msg_1}. Skipping second optimization.")
            if not optimization_results:
                st.error("Scan + Opt: No candidates successfully completed the double optimization.")
                clear_optimized_state() 
                return
            st.write("\nPhase 3: Selecting Best Overall Result...")
            optimization_results.sort(key=lambda r: r['final_mse'])
            best_overall_result = optimization_results[0]
            final_ep = best_overall_result['final_ep']
            final_cost = best_overall_result['final_mse']
            final_l0 = best_overall_result['l0_origin']
            final_msg = best_overall_result['message']
            st.session_state.optimized_ep = final_ep.copy()
            st.session_state.current_ep = final_ep.copy()
            st.session_state.is_optimized_state = True
            st.session_state.last_mse = final_cost
            st.session_state.l0 = final_l0 
            qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, final_l0, nH_mat, nL_mat, EXCEL_FILE_PATH)
            if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
            else:
                st.session_state.optimized_qwot_str = "QWOT N/A"
            st.success(f"Scan + Double Optimization finished. Best result from l0={final_l0:.1f}. Final MSE: {final_cost:.4e}")
            st.caption(f"Optimization details: {final_msg}")
            st.session_state.needs_rerun_calc = True
            st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Scan+Opt*2 (l0={final_l0:.1f})"}
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during QWOT Scan + Double Optimization: {e}")
            clear_optimized_state()
        except Exception as e_fatal:
            st.error(f"Unexpected error during QWOT Scan + Double Optimization: {e_fatal}")
            clear_optimized_state()
        finally:
            pass

def run_auto_mode_wrapper():
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None
    with st.spinner("Automatic Mode in progress (can be very long)..."):
        try:
            active_targets = validate_targets()
            if active_targets is None or not active_targets:
                st.error("Auto Mode requires active and valid targets.")
                return
            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None:
                st.error("Could not determine lambda range for Auto Mode.")
                return
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
                st.error("Material definition error for Auto Mode.")
                return
            validated_inputs['nH_material'] = nH_mat
            validated_inputs['nL_material'] = nL_mat
            validated_inputs['nSub_material'] = nSub_mat
            ep_start_auto = None
            if st.session_state.get('is_optimized_state') and st.session_state.get('optimized_ep') is not None:
                ep_start_auto = st.session_state.optimized_ep.copy()
            final_ep, final_mse, auto_logs = run_auto_mode(
                initial_ep=ep_start_auto,
                validated_inputs=validated_inputs,
                active_targets=active_targets,
                excel_file_path=EXCEL_FILE_PATH,
                log_callback=add_log
            )
            if final_ep is not None and np.isfinite(final_mse):
                st.session_state.optimized_ep = final_ep.copy()
                st.session_state.current_ep = final_ep.copy()
                st.session_state.is_optimized_state = True
                st.session_state.last_mse = final_mse
                qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                    st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                else: st.session_state.optimized_qwot_str = "QWOT N/A"
                st.success(f"Auto Mode finished. Final MSE: {final_mse:.4e}")
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Auto Mode"}
            else:
                st.error("Automatic Mode failed or did not produce a valid result.")
                clear_optimized_state()
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (After Auto Fail)"}
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during Auto Mode: {e}")
        except Exception as e_fatal:
            st.error(f"Unexpected error during Auto Mode: {e_fatal}")
        finally:
            pass

def run_remove_thin_wrapper():
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None
    ep_start_removal = None
    is_starting_from_optimized = False
    if st.session_state.get('is_optimized_state') and st.session_state.get('optimized_ep') is not None:
        ep_start_removal = st.session_state.optimized_ep.copy()
        is_starting_from_optimized = True
        print("Remove Thin: Starting from existing optimized structure.")
    else:
        print("Remove Thin: Starting from nominal structure.")
        try:
            nH_mat_temp, _ = get_material_input('H')
            nL_mat_temp, _ = get_material_input('L')
            if nH_mat_temp is None or nL_mat_temp is None:
                st.error("Cannot calculate nominal structure: Material definition error.")
                return
            emp_list_temp = [float(e.strip()) for e in st.session_state.current_qwot.split(',') if e.strip()]
            if not emp_list_temp:
                ep_start_removal = np.array([], dtype=np.float64) 
            else:
                ep_start_removal, logs_ep_init = calculate_initial_ep(
                    emp_list_temp, st.session_state.l0, nH_mat_temp, nL_mat_temp, EXCEL_FILE_PATH
                )
                if ep_start_removal is None:
                    st.error("Failed to calculate nominal structure from QWOT for removal.")
                    return
            st.session_state.current_ep = ep_start_removal.copy() 
        except Exception as e_nom:
            st.error(f"Error calculating nominal structure for removal: {e_nom}")
            return
    if ep_start_removal is None:
        st.error("Could not determine a valid starting structure for removal.")
        return
    if len(ep_start_removal) <= 2:
        st.error("Structure too small (<= 2 layers) for removal/merge.")
        return
    with st.spinner("Removing thin layer + Re-optimizing..."):
        try:
            st.session_state.ep_history.append(ep_start_removal.copy())
            active_targets = validate_targets()
            if active_targets is None or not active_targets:
                st.session_state.ep_history.pop()
                st.error("Removal aborted: invalid or missing targets for re-optimization.")
                return
            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None:
                st.session_state.ep_history.pop()
                st.error("Removal aborted: invalid lambda range for re-optimization.")
                return
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
                st.session_state.ep_history.pop()
                st.error("Removal aborted: material definition error.")
                return
            validated_inputs['nH_material'] = nH_mat
            validated_inputs['nL_material'] = nL_mat
            validated_inputs['nSub_material'] = nSub_mat
            threshold = validated_inputs['auto_thin_threshold']
            ep_after_removal, structure_changed, removal_logs = _perform_layer_merge_or_removal_only(
                ep_start_removal, MIN_THICKNESS_PHYS_NM,
                log_prefix="  [Remove] ",
                threshold_for_removal=None 
            )
            if structure_changed and ep_after_removal is not None:
                st.write("Re-optimizing after removal...")
                final_ep, success, final_cost, optim_logs, msg = \
                    _run_core_optimization(ep_after_removal, validated_inputs, active_targets,
                                           MIN_THICKNESS_PHYS_NM, log_prefix="  [ReOpt Thin] ")
                if success and final_ep is not None:
                    st.session_state.optimized_ep = final_ep.copy()
                    st.session_state.current_ep = final_ep.copy() 
                    st.session_state.is_optimized_state = True 
                    st.session_state.last_mse = final_cost
                    qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                    if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                        st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                    else: st.session_state.optimized_qwot_str = "QWOT N/A"
                    st.success(f"Removal + Re-optimization finished ({msg}). Final MSE: {final_cost:.4e}")
                    st.session_state.needs_rerun_calc = True
                    st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Optimized (Post-Remove)"}
                else:
                    st.warning(f"Layer removed, but re-optimization failed ({msg}). State is AFTER removal but BEFORE failed re-opt attempt.")
                    st.session_state.optimized_ep = ep_after_removal.copy()
                    st.session_state.current_ep = ep_after_removal.copy()
                    st.session_state.is_optimized_state = True 
                    try:
                        l_min_opt_disp, l_max_opt_disp = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
                        l_step_optim_disp = validated_inputs['l_step']
                        num_pts_optim_disp = min(max(2, int(np.round((l_max_opt_disp - l_min_opt_disp) / l_step_optim_disp)) + 1), 100)
                        l_vec_optim_np_disp = np.geomspace(l_min_opt_disp, l_max_opt_disp, num_pts_optim_disp)
                        l_vec_optim_np_disp = l_vec_optim_np_disp[(l_vec_optim_np_disp > 0) & np.isfinite(l_vec_optim_np_disp)]
                        if l_vec_optim_np_disp.size > 0:
                            results_fail_grid, _ = calculate_T_from_ep_jax(ep_after_removal, nH_mat, nL_mat, nSub_mat, l_vec_optim_np_disp, EXCEL_FILE_PATH)
                            if results_fail_grid:
                                mse_fail, _ = calculate_final_mse(results_fail_grid, active_targets)
                                st.session_state.last_mse = mse_fail
                            else: st.session_state.last_mse = None
                        qwots_fail, _ = calculate_qwot_from_ep(ep_after_removal, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                        if qwots_fail is not None and not np.any(np.isnan(qwots_fail)):
                            st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_fail])
                        else: st.session_state.optimized_qwot_str = "QWOT N/A (ReOpt Fail)"
                    except Exception as e_recalc:
                        st.session_state.last_mse = None
                        st.session_state.optimized_qwot_str = "Recalc Error"
                    st.session_state.needs_rerun_calc = True
                    st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': "Optimized (Post-Remove, Re-Opt Fail)"}
            else:
                st.info("No layer was removed (criteria not met).")
                try:
                    st.session_state.ep_history.pop() 
                except IndexError: pass
        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during Thin Layer Removal: {e}")
            try: st.session_state.ep_history.pop()
            except IndexError: pass
        except Exception as e_fatal:
            st.error(f"Unexpected error during Thin Layer Removal: {e_fatal}")
            try: st.session_state.ep_history.pop()
            except IndexError: pass
        finally:
            pass

st.set_page_config(layout="wide", page_title="Thin Film Optimizer (Streamlit)")
if 'init_done' not in st.session_state:
    st.session_state.log_messages = ["[Initialization] Welcome to the Streamlit optimizer."]
    st.session_state.current_ep = None
    st.session_state.current_qwot = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    st.session_state.optimized_ep = None
    st.session_state.is_optimized_state = False
    st.session_state.optimized_qwot_str = ""
    st.session_state.material_sequence = None
    st.session_state.ep_history = deque(maxlen=5)
    st.session_state.last_mse = None
    st.session_state.needs_rerun_calc = False
    st.session_state.rerun_calc_params = {}
    st.session_state.target_definition_type = "Manual Targets"
    st.session_state.uploaded_target_data = None
    st.session_state.last_uploaded_target_file = None
    try:
        mats, logs = get_available_materials_from_excel(EXCEL_FILE_PATH)
        add_log(logs)
        st.session_state.available_materials = sorted(list(set(["Constant"] + mats)))
        base_subs = ["Constant", "Fused Silica", "BK7", "D263"]
        st.session_state.available_substrates = sorted(list(set(base_subs + st.session_state.available_materials)))
    except Exception as e:
        st.error(f"Initial error loading materials from {EXCEL_FILE_PATH}: {e}")
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

st.title("🔬 Thin Film Optimizer (Streamlit + JAX)")
menu_cols = st.columns(8)
with menu_cols[0]:
    if st.button("📊 Eval Nom.", key="eval_nom_top", help="Evaluate Nominal Structure"):
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (Evaluated)"}
        st.rerun()
with menu_cols[1]:
    if st.button("✨ Opt Local", key="optim_local_top", help="Local Optimization"):
        run_local_optimization_wrapper()
with menu_cols[2]:
    if st.button("🚀 Scan+Opt", key="optim_scan_top", help="Initial Scan + Optimization (Max 18 layers for scan, Max 100 pts for MSE)"):
        run_scan_optimization_wrapper()
with menu_cols[3]:
    if st.button("🤖 Auto", key="optim_auto_top", help="Auto Mode (Needle > Thin > Opt) (Max 100 pts for MSE)"):
        run_auto_mode_wrapper()
with menu_cols[4]:
    current_structure_for_check = None
    if st.session_state.get('is_optimized_state') and st.session_state.get('optimized_ep') is not None:
        current_structure_for_check = st.session_state.optimized_ep
    elif st.session_state.get('current_ep') is not None:
        current_structure_for_check = st.session_state.current_ep
    can_remove_structurally = current_structure_for_check is not None and len(current_structure_for_check) > 2
    if st.button("🗑️ Thin+ReOpt", key="remove_thin_top", help="Remove Thin Layer + Re-optimize", disabled=not can_remove_structurally):
        run_remove_thin_wrapper()
with menu_cols[5]:
    can_optimize_top = st.session_state.get('is_optimized_state', False) and st.session_state.get('optimized_ep') is not None
    if st.button("💾 Opt->Nom", key="set_optim_as_nom_top", help="Set Optimized as Nominal", disabled=not can_optimize_top):
        set_optimized_as_nominal_wrapper()
        st.rerun()
with menu_cols[6]:
    can_undo_top = bool(st.session_state.get('ep_history'))
    if st.button(f"↩️ Undo ({len(st.session_state.get('ep_history', deque()))})", key="undo_remove_top", help="Undo Last Removal", disabled=not can_undo_top):
        undo_remove_wrapper()
        st.rerun()
with menu_cols[7]:
    if st.button("🔄 Reload Mats", key="reload_mats_top", help="Reload Excel Materials"):
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
            st.error(f"Error reloading materials: {e}")
st.divider()
main_layout = st.columns([1, 3]) 
with main_layout[0]: 
    st.subheader("Materials")
    st.session_state.selected_H = st.selectbox(
        "H Material", options=st.session_state.available_materials,
        index=st.session_state.available_materials.index(st.session_state.selected_H),
        key="sb_H", on_change=trigger_nominal_recalc 
    )
    if st.session_state.selected_H == "Constant":
        hc1, hc2 = st.columns(2)
        st.session_state.nH_r = hc1.number_input("n' H", value=st.session_state.nH_r, format="%.4f", key="nH_r_const", on_change=trigger_nominal_recalc)
        st.session_state.nH_i = hc2.number_input("k H", value=st.session_state.nH_i, min_value=0.0, format="%.4f", key="nH_i_const", on_change=trigger_nominal_recalc)
    st.session_state.selected_L = st.selectbox(
        "L Material", options=st.session_state.available_materials,
        index=st.session_state.available_materials.index(st.session_state.selected_L),
        key="sb_L", on_change=trigger_nominal_recalc
    )
    if st.session_state.selected_L == "Constant":
        lc1, lc2 = st.columns(2)
        st.session_state.nL_r = lc1.number_input("n' L", value=st.session_state.nL_r, format="%.4f", key="nL_r_const", on_change=trigger_nominal_recalc)
        st.session_state.nL_i = lc2.number_input("k L", value=st.session_state.nL_i, min_value=0.0, format="%.4f", key="nL_i_const", on_change=trigger_nominal_recalc)
    st.session_state.selected_Sub = st.selectbox(
        "Substrate Material", options=st.session_state.available_substrates,
        index=st.session_state.available_substrates.index(st.session_state.selected_Sub),
        key="sb_Sub", on_change=trigger_nominal_recalc
    )
    if st.session_state.selected_Sub == "Constant":
        st.session_state.nSub_r = st.number_input("n' Substrate", value=st.session_state.nSub_r, format="%.4f", key="nSub_const", on_change=trigger_nominal_recalc)
    st.subheader("Nominal Structure")
    st.session_state.current_qwot = st.text_area(
        "QWOT Multipliers (comma-separated)", value=st.session_state.current_qwot, key="qwot_input",
        on_change=clear_optimized_state, height=100
    )
    num_layers_from_qwot = len([q for q in st.session_state.current_qwot.split(',') if q.strip()])
    qwot_cols = st.columns([3,2])
    with qwot_cols[0]:
        st.session_state.l0 = st.number_input("Reference Wavelength λ₀ (nm) for QWOT", value=st.session_state.l0, min_value=1.0, format="%.2f", key="l0_input", on_change=trigger_nominal_recalc)
    with qwot_cols[1]:
        init_layers_num = st.number_input("Number of Layers to Generate", min_value=0, value=num_layers_from_qwot, step=1, key="init_layers_gen_num")
        if st.button("Gen 1s QWOT", key="gen_qwot_btn_main", use_container_width=True, help="Generate a QWOT string with the specified number of layers, all set to 1.0"):
            if init_layers_num > 0:
                new_qwot = ",".join(['1'] * init_layers_num)
                if new_qwot != st.session_state.current_qwot:
                    st.session_state.current_qwot = new_qwot
                    clear_optimized_state()
                    st.session_state.needs_rerun_calc = True
                    st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (Generated 1s)"}
                    st.rerun()
            elif st.session_state.current_qwot != "":
                st.session_state.current_qwot = ""
                clear_optimized_state()
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (QWOT Cleared)"}
                st.rerun()
    st.caption(f"Current Nominal Layers: {num_layers_from_qwot}")
    st.subheader("Targets (T) & Calculation Parameters")
    st.session_state.l_step = st.number_input("λ Step for MSE Grid (nm)", value=st.session_state.l_step, min_value=0.1, format="%.2f", key="l_step_input_main", on_change=trigger_nominal_recalc, help="Wavelength step for optimization grid points (max 100 points). Plotting uses a finer grid.")
    st.session_state.auto_thin_threshold = st.number_input("Auto Thin Layer Removal Threshold (nm)", value=st.session_state.auto_thin_threshold, min_value=MIN_THICKNESS_PHYS_NM, format="%.3f", key="auto_thin_input_main", help="In Auto mode, layers thinner than this may be removed.")
    
    st.radio("Target Definition Method", ["Manual Targets", "Import from File"], key="target_definition_type", on_change=trigger_nominal_recalc, horizontal=True)
    
    if st.session_state.target_definition_type == "Import from File":
        uploaded_target_file = st.file_uploader("Upload Target (Excel: Col A=λ(nm), Col B=T%)", type=['xlsx', 'xls'], key="target_uploader")
        if uploaded_target_file is not None and uploaded_target_file != st.session_state.get('last_uploaded_target_file'):
            st.session_state.last_uploaded_target_file = uploaded_target_file
            target_data, logs = process_uploaded_target_file(uploaded_target_file)
            add_log(logs)
            st.session_state.uploaded_target_data = target_data
            if target_data is not None:
                st.success(f"Loaded {len(target_data)} target points.")
                trigger_nominal_recalc()
        if 'uploaded_target_data' in st.session_state and st.session_state.uploaded_target_data is not None:
             st.info(f"{len(st.session_state.uploaded_target_data)} points loaded from '{st.session_state.last_uploaded_target_file.name}'.")
    else:
        hdr_cols = st.columns([0.5, 1, 1, 1, 1])
        hdrs = ["On", "λmin", "λmax", "Tmin", "Tmax"]
        for c, h in zip(hdr_cols, hdrs): c.caption(h)
        for i in range(len(st.session_state.targets)):
            target = st.session_state.targets[i]
            cols = st.columns([0.5, 1, 1, 1, 1])
            current_enabled = target.get('enabled', False)
            new_enabled = cols[0].checkbox("", value=current_enabled, key=f"target_enable_{i}_main", label_visibility="collapsed", on_change=trigger_nominal_recalc)
            st.session_state.targets[i]['enabled'] = new_enabled
            st.session_state.targets[i]['min'] = cols[1].number_input(f"λmin Target {i+1}", value=target.get('min', 0.0), format="%.1f", step=10.0, key=f"target_min_{i}_main", label_visibility="collapsed", on_change=trigger_nominal_recalc)
            st.session_state.targets[i]['max'] = cols[2].number_input(f"λmax Target {i+1}", value=target.get('max', 0.0), format="%.1f", step=10.0, key=f"target_max_{i}_main", label_visibility="collapsed", on_change=trigger_nominal_recalc)
            st.session_state.targets[i]['target_min'] = cols[3].number_input(f"Tmin Target {i+1}", value=target.get('target_min', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmin_{i}_main", label_visibility="collapsed", on_change=trigger_nominal_recalc)
            st.session_state.targets[i]['target_max'] = cols[4].number_input(f"Tmax Target {i+1}", value=target.get('target_max', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmax_{i}_main", label_visibility="collapsed", on_change=trigger_nominal_recalc)

with main_layout[1]: 
    st.subheader("Results")
    state_desc = "Optimized" if st.session_state.is_optimized_state else "Nominal"
    ep_display = st.session_state.optimized_ep if st.session_state.is_optimized_state else st.session_state.current_ep
    num_layers_display = len(ep_display) if ep_display is not None else 0
    res_info_cols = st.columns(3)
    with res_info_cols[0]:
        st.caption(f"State: {state_desc} ({num_layers_display} layers)")
    with res_info_cols[1]:
        if st.session_state.last_mse is not None and np.isfinite(st.session_state.last_mse):
            st.caption(f"MSE: {st.session_state.last_mse:.4e}")
        else:
            st.caption("MSE: N/A")
    with res_info_cols[2]:
        min_thick_str = "N/A"
        if ep_display is not None and ep_display.size > 0:
            valid_thick = ep_display[ep_display >= MIN_THICKNESS_PHYS_NM - 1e-9]
            if valid_thick.size > 0:
                min_thick_str = f"{np.min(valid_thick):.3f} nm"
        st.caption(f"Min Thick: {min_thick_str}")
    if st.session_state.is_optimized_state and st.session_state.get('optimized_qwot_str'):
        st.text_input("Optimized QWOT (at original λ₀)", value=st.session_state.optimized_qwot_str, disabled=True, key="opt_qwot_display_main_res")
    if 'last_calc_results' in st.session_state and st.session_state.last_calc_results:
        results_data = st.session_state.last_calc_results
        res_fine_plot = results_data.get('res_fine')
        active_targets_plot = validate_targets()
        mse_plot = st.session_state.last_mse
        method_name_plot = results_data.get('method_name', '')
        res_optim_grid_plot = results_data.get('res_optim_grid')
        if res_fine_plot and active_targets_plot is not None:
            fig_spec, ax_spec = plt.subplots(figsize=(12, 4)) 
            opt_method_str = f" ({method_name_plot})" if method_name_plot else ""
            window_title = f'Spectral Response {"Optimized" if st.session_state.is_optimized_state else "Nominal"}{opt_method_str}'
            fig_spec.suptitle(window_title, fontsize=12, weight='bold')
            line_ts = None
            try:
                if res_fine_plot and 'l' in res_fine_plot and 'Ts' in res_fine_plot and res_fine_plot['l'] is not None and len(res_fine_plot['l']) > 0:
                    res_l_plot = np.asarray(res_fine_plot['l'])
                    res_ts_plot = np.asarray(res_fine_plot['Ts'])
                    line_ts, = ax_spec.plot(res_l_plot, res_ts_plot, label='Transmittance', linestyle='-', color='blue', linewidth=1.5)
                    
                    if st.session_state.get("target_definition_type") == "Import from File" and st.session_state.get('uploaded_target_data') is not None:
                        target_points = st.session_state.uploaded_target_data
                        l_target = target_points[:, 0]
                        t_target = target_points[:, 1]
                        ax_spec.plot(l_target, t_target, 'r--', label='Target (File)', linewidth=1.5, alpha=0.9, zorder=5)
                    elif active_targets_plot:
                        plotted_target_label = False
                        for i, target in enumerate(active_targets_plot):
                            l_min, l_max = target['min'], target['max']
                            t_min, t_max_corr = target['target_min'], target['target_max']
                            x_coords, y_coords = [l_min, l_max], [t_min, t_max_corr]
                            label = 'Target(s)' if not plotted_target_label else "_nolegend_"
                            ax_spec.plot(x_coords, y_coords, 'r--', linewidth=1.0, alpha=0.7, label=label, zorder=5)
                            plotted_target_label = True

                ax_spec.set_xlabel("Wavelength (nm)")
                ax_spec.set_ylabel('Transmittance')
                ax_spec.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
                ax_spec.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')
                ax_spec.minorticks_on()
                if len(res_l_plot) > 0 : ax_spec.set_xlim(res_l_plot[0], res_l_plot[-1])
                ax_spec.set_ylim(-0.05, 1.05)
                ax_spec.legend(fontsize=8)
                if mse_plot is not None and np.isfinite(mse_plot): mse_text = f"MSE = {mse_plot:.3e}"
                else: mse_text = "MSE: N/A"
                ax_spec.text(0.98, 0.98, mse_text, transform=ax_spec.transAxes, ha='right', va='top', fontsize=9,
                                     bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
            except Exception as e_spec:
                ax_spec.text(0.5, 0.5, f"Error plotting spectrum:\n{e_spec}", ha='center', va='center', transform=ax_spec.transAxes, color='red')
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            st.pyplot(fig_spec)
            plt.close(fig_spec)
        else:
            st.warning("Missing or invalid calculation data for spectrum display.")
    else:
        st.info("Run an evaluation or optimization to see results.")
    plot_col1, plot_col2 = st.columns(2)
    if 'last_calc_results' in st.session_state and st.session_state.last_calc_results:
        results_data = st.session_state.last_calc_results
        ep_plot = results_data.get('ep_used')
        l0_plot = results_data.get('l0_used')
        nH_plot = results_data.get('nH_used')
        nL_plot = results_data.get('nL_used')
        nSub_plot = results_data.get('nSub_used')
        is_optimized_plot = st.session_state.is_optimized_state
        material_sequence_plot = st.session_state.get('material_sequence')
        if ep_plot is not None and l0_plot is not None and nH_plot is not None and nL_plot is not None and nSub_plot is not None:
            with plot_col1:
                fig_idx, ax_idx = plt.subplots(figsize=(6, 4)) 
                try:
                    nH_c_repr, logs_h = _get_nk_at_lambda(nH_plot, l0_plot, EXCEL_FILE_PATH)
                    nL_c_repr, logs_l = _get_nk_at_lambda(nL_plot, l0_plot, EXCEL_FILE_PATH)
                    nSub_c_repr, logs_s = _get_nk_at_lambda(nSub_plot, l0_plot, EXCEL_FILE_PATH)
                    if nH_c_repr is None or nL_c_repr is None or nSub_c_repr is None:
                        raise ValueError("Indices at l0 not found for profile plot.")
                    nH_r_repr, nL_r_repr, nSub_r_repr = nH_c_repr.real, nL_c_repr.real, nSub_c_repr.real
                    num_layers = len(ep_plot)
                    if material_sequence_plot:
                        n_real_layers_repr = [nH_r_repr if i % 2 == 0 else nL_r_repr for i in range(num_layers)]
                    else:
                        n_real_layers_repr = [nH_r_repr if i % 2 == 0 else nL_r_repr for i in range(num_layers)]
                    ep_cumulative = np.cumsum(ep_plot) if num_layers > 0 else np.array([0])
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
                    ax_idx.plot(x_coords_plot, y_coords_plot, drawstyle='steps-post', label=f'n\'(λ={l0_plot:.0f}nm)', color='purple', linewidth=1.5)
                    ax_idx.set_xlabel('Depth (from substrate) (nm)')
                    ax_idx.set_ylabel("Real Part of Index (n')")
                    ax_idx.set_title(f"Index Profile (at λ={l0_plot:.0f}nm)", fontsize=10)
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
                    ax_idx.text(0.5, 0.5, f"Error plotting index profile:\n{e_idx}", ha='center', va='center', transform=ax_idx.transAxes, color='red')
                plt.tight_layout()
                st.pyplot(fig_idx)
                plt.close(fig_idx)
            with plot_col2:
                fig_stack, ax_stack = plt.subplots(figsize=(6, 4)) 
                try:
                    num_layers = len(ep_plot)
                    if num_layers > 0:
                        indices_complex_repr = []
                        if material_sequence_plot:
                            nH_c_repr, _ = _get_nk_at_lambda(nH_plot, l0_plot, EXCEL_FILE_PATH)
                            nL_c_repr, _ = _get_nk_at_lambda(nL_plot, l0_plot, EXCEL_FILE_PATH)
                            indices_complex_repr = [nH_c_repr if i % 2 == 0 else nL_c_repr for i in range(num_layers)]
                            layer_types = [f"Mat{i+1}" for i in range(num_layers)]
                        else:
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
                            layer_type = layer_types[i] if layer_types and i < len(layer_types) else '?'
                            n_str = f"{n_comp_repr.real:.3f}" if np.isfinite(n_comp_repr.real) else "N/A"
                            k_val = n_comp_repr.imag
                            if np.isfinite(k_val) and abs(k_val) > 1e-6: n_str += f"{k_val:+.3f}j"
                            yticks_labels.append(f"L{i + 1} ({layer_type}) n≈{n_str}")
                        ax_stack.set_yticks(bar_pos)
                        ax_stack.set_yticklabels(yticks_labels, fontsize=7)
                        ax_stack.invert_yaxis()
                        max_ep_plot_val = max(ep_plot) if ep_plot.size > 0 else 1.0
                        fontsize_bar = max(6, 9 - num_layers // 15)
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ha_pos = 'left' if width < max_ep_plot_val * 0.3 else 'right'
                            x_text_pos = width * 1.02 if ha_pos == 'left' else width * 0.98
                            text_color = 'black' if ha_pos == 'left' else 'white'
                            ax_stack.text(x_text_pos, bar.get_y() + bar.get_height()/2., f"{width:.2f}",
                                          va='center', ha=ha_pos, color=text_color, fontsize=fontsize_bar, weight='bold')
                    else:
                        ax_stack.text(0.5, 0.5, "Empty Structure", ha='center', va='center', fontsize=10, color='grey', transform=ax_stack.transAxes)
                        ax_stack.set_yticks([]); ax_stack.set_xticks([])
                    ax_stack.set_xlabel('Thickness (nm)')
                    stack_title_prefix = f'Structure {"Optimized" if is_optimized_plot else "Nominal"}'
                    ax_stack.set_title(f"{stack_title_prefix} ({num_layers} layers)", fontsize=10)
                    max_ep_plot_val_xlim = max(ep_plot) if num_layers > 0 and ep_plot.size > 0 else 10
                    ax_stack.set_xlim(right=max_ep_plot_val_xlim * 1.1)
                except Exception as e_stack:
                    ax_stack.text(0.5, 0.5, f"Error plotting structure:\n{e_stack}", ha='center', va='center', transform=ax_stack.transAxes, color='red')
                plt.tight_layout()
                st.pyplot(fig_stack)
                plt.close(fig_stack)
        else:
            st.warning("Missing data to display profiles.")
    else:
        pass

if st.session_state.get('needs_rerun_calc', False):
    params = st.session_state.rerun_calc_params
    force_ep_val = params.get('force_ep')
    st.session_state.needs_rerun_calc = False
    st.session_state.rerun_calc_params = {}
    st.session_state.calculating = True
    run_calculation_wrapper(
        is_optimized_run=params.get('is_optimized_run', False),
        method_name=params.get('method_name', 'Auto Recalc'),
        force_ep=force_ep_val
    )
    st.session_state.calculating = False
    st.rerun()
