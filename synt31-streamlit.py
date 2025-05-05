import streamlit as st
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.lax import scan, cond
import numpy as np
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from collections import deque
import json
import os
import datetime
import traceback
from scipy.optimize import minimize, OptimizeResult
import time
import pandas as pd
import functools
from typing import Union, Tuple, Dict, List, Any, Callable, Optional

st.set_page_config(layout="wide", page_title="Thin Film Optimizer (JAX)")

MIN_THICKNESS_PHYS_NM = 0.01
BASE_NEEDLE_THICKNESS_NM = 0.1
DEFAULT_NEEDLE_SCAN_STEP_NM = 2.0
AUTO_NEEDLES_PER_CYCLE = 5
AUTO_MAX_CYCLES = 5
MSE_IMPROVEMENT_TOLERANCE = 1e-9
EXCEL_FILE_PATH = "indices.xlsx"

@st.cache_data(show_spinner="Loading material data from Excel...")
def load_material_data_from_xlsx_sheet(_file_path: str, sheet_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    logs = []
    try:
        df = pd.read_excel(_file_path, sheet_name=sheet_name, header=None)
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        numeric_df = numeric_df.dropna(how='all')
        if numeric_df.shape[1] >= 3:
            numeric_df = numeric_df.dropna(subset=[0, 1, 2])
        else:
            st.error(f"Sheet '{sheet_name}' does not contain 3 numeric columns.")
            return None, None, None
        numeric_df = numeric_df.sort_values(by=0)
        l_nm = numeric_df.iloc[:, 0].values.astype(np.float64)
        n = numeric_df.iloc[:, 1].values.astype(np.float64)
        k = numeric_df.iloc[:, 2].values.astype(np.float64)
        if len(l_nm) == 0:
            st.warning(f"No valid numeric data found in sheet '{sheet_name}' of file {_file_path}")
            return None, None, None
        # Use a less intrusive way to show loading info, maybe a toast or just log
        # st.sidebar.info(f"Loaded {sheet_name}: {len(l_nm)} points ({l_nm.min():.1f}-{l_nm.max():.1f} nm)")
        return l_nm, n, k
    except FileNotFoundError:
        st.error(f"Error: Excel file not found: {_file_path}")
        return None, None, None
    except ValueError as ve:
        st.error(f"Value error reading sheet '{sheet_name}' in {_file_path}: {ve}")
        return None, None, None
    except Exception as e:
        st.error(f"Error reading sheet '{sheet_name}' in {_file_path}: {type(e).__name__} - {e}")
        return None, None, None

@jax.jit
def get_n_fused_silica(l_nm: jnp.ndarray) -> jnp.ndarray:
    l_um_sq = (l_nm / 1000.0)**2
    n_sq = 1.0 + (0.6961663 * l_um_sq) / (l_um_sq - 0.0684043**2) + \
           (0.4079426 * l_um_sq) / (l_um_sq - 0.1162414**2) + \
           (0.8974794 * l_um_sq) / (l_um_sq - 9.896161**2)
    n = jnp.sqrt(n_sq)
    k = jnp.zeros_like(n)
    return n + 1j * k

@jax.jit
def get_n_bk7(l_nm: jnp.ndarray) -> jnp.ndarray:
    l_um_sq = (l_nm / 1000.0)**2
    n_sq = 1.0 + (1.03961212 * l_um_sq) / (l_um_sq - 0.00600069867) + \
           (0.231792344 * l_um_sq) / (l_um_sq - 0.0200179144) + \
           (1.01046945 * l_um_sq) / (l_um_sq - 103.560653)
    n = jnp.sqrt(n_sq)
    k = jnp.zeros_like(n)
    return n + 1j * k

@jax.jit
def get_n_d263(l_nm: jnp.ndarray) -> jnp.ndarray:
    n = jnp.full_like(l_nm, 1.523, dtype=jnp.float64)
    k = jnp.zeros_like(n)
    return n + 1j * k

def interp_nk_cached(l_target: Union[np.ndarray, jnp.ndarray],
                      l_data: Union[np.ndarray, jnp.ndarray],
                      n_data: Union[np.ndarray, jnp.ndarray],
                      k_data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    l_target_np = np.asarray(l_target)
    l_data_np = np.asarray(l_data)
    n_data_np = np.asarray(n_data)
    k_data_np = np.asarray(k_data)
    sort_indices = np.argsort(l_data_np)
    l_data_sorted = l_data_np[sort_indices]
    n_data_sorted = n_data_np[sort_indices]
    k_data_sorted = k_data_np[sort_indices]
    n_interp = np.interp(l_target_np, l_data_sorted, n_data_sorted)
    k_interp = np.interp(l_target_np, l_data_sorted, k_data_sorted)
    return jnp.array(n_interp + 1j * k_interp)

MaterialInputType = Union[complex, float, int, str, Tuple[np.ndarray, np.ndarray, np.ndarray]]

def _get_nk_array_for_lambda_vec(material_definition: MaterialInputType,
                                  l_vec_target_jnp: jnp.ndarray,
                                  excel_file_path: str) -> Tuple[jnp.ndarray, List[str]]:
    logs = []
    try:
        if isinstance(material_definition, (complex, float, int)):
            result = jnp.full(l_vec_target_jnp.shape, jnp.asarray(material_definition, dtype=jnp.complex128))
        elif isinstance(material_definition, str):
            mat_upper = material_definition.upper()
            if mat_upper == "FUSED SILICA": result = get_n_fused_silica(l_vec_target_jnp)
            elif mat_upper == "BK7": result = get_n_bk7(l_vec_target_jnp)
            elif mat_upper == "D263": result = get_n_d263(l_vec_target_jnp)
            else:
                sheet_name = material_definition
                l_data, n_data, k_data = load_material_data_from_xlsx_sheet(excel_file_path, sheet_name)
                if l_data is None: raise ValueError(f"Could not load data for sheet '{sheet_name}' from {excel_file_path}")
                result = interp_nk_cached(l_vec_target_jnp, l_data, n_data, k_data)
        elif isinstance(material_definition, tuple) and len(material_definition) == 3:
            l_data, n_data, k_data = material_definition
            result = interp_nk_cached(l_vec_target_jnp, l_data, n_data, k_data)
        else: raise TypeError(f"Unsupported material definition type: {type(material_definition)}")
        return result, logs
    except Exception as e:
        logs.append(f"Error preparing material data for '{material_definition}': {e}")
        st.error(f"Failed to get nk array for {material_definition}: {e}")
        raise ValueError(f"Failed to get nk array: {e}")

@jax.jit
def _compute_layer_matrix_scan_step(carry_matrix: jnp.ndarray, layer_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, None]:
    thickness, Ni, l_val = layer_data
    eta = Ni
    safe_l_val = jnp.maximum(l_val, 1e-9)
    phi = (2 * jnp.pi / safe_l_val) * (Ni * thickness)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    def compute_M_layer(thickness_: jnp.ndarray) -> jnp.ndarray:
        safe_eta = jnp.where(jnp.abs(eta) < 1e-12, 1e-12 + 0j, eta)
        M_layer = jnp.array([[cos_phi, (1j / safe_eta) * sin_phi], [1j * eta * sin_phi, cos_phi]], dtype=jnp.complex128)
        return M_layer @ carry_matrix
    def compute_identity(thickness_: jnp.ndarray) -> jnp.ndarray: return carry_matrix
    new_matrix = cond(thickness > 1e-12, compute_M_layer, compute_identity, thickness)
    return new_matrix, None

@jax.jit
def compute_stack_matrix_jax(ep_vector: jnp.ndarray, l_val: jnp.ndarray, nH_at_lval: jnp.ndarray, nL_at_lval: jnp.ndarray) -> jnp.ndarray:
    num_layers = len(ep_vector)
    layer_indices = jnp.array([nH_at_lval if i % 2 == 0 else nL_at_lval for i in range(num_layers)])
    layers_scan_data = (ep_vector, layer_indices, jnp.full(num_layers, l_val))
    M_initial = jnp.eye(2, dtype=jnp.complex128)
    M_final, _ = scan(_compute_layer_matrix_scan_step, M_initial, layers_scan_data)
    return M_final

@jax.jit
def calculate_single_wavelength_T(l_val: jnp.ndarray, ep_vector_contig: jnp.ndarray,
                                  nH_at_lval: jnp.ndarray, nL_at_lval: jnp.ndarray, nSub_at_lval: jnp.ndarray) -> jnp.ndarray:
    etainc = 1.0 + 0j; etasub = nSub_at_lval
    def calculate_for_valid_l(l_: jnp.ndarray) -> jnp.ndarray:
        M = compute_stack_matrix_jax(ep_vector_contig, l_, nH_at_lval, nL_at_lval)
        m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        rs_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        rs_denominator_abs = jnp.abs(rs_denominator)
        safe_denominator = jnp.where(rs_denominator_abs < 1e-12, 1e-12 + 0j, rs_denominator)
        ts = (2.0 * etainc) / safe_denominator
        real_etasub = jnp.real(etasub); real_etainc = jnp.real(etainc)
        Ts = (real_etasub / real_etainc) * (ts * jnp.conj(ts))
        return jnp.where(rs_denominator_abs < 1e-12, jnp.nan, jnp.real(Ts))
    def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray: return jnp.nan
    Ts = cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)
    return Ts

@jax.jit
def calculate_T_from_ep_core_jax(ep_vector: jnp.ndarray,
                                  nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                  l_vec: jnp.ndarray) -> jnp.ndarray:
    if not l_vec.size: return jnp.zeros(0, dtype=jnp.float64)
    ep_vector_contig = jnp.asarray(ep_vector)
    Ts_arr = vmap(calculate_single_wavelength_T, in_axes=(0, None, 0, 0, 0))(l_vec, ep_vector_contig, nH_arr, nL_arr, nSub_arr)
    return jnp.nan_to_num(Ts_arr, nan=0.0)

def calculate_T_from_ep_jax(ep_vector: Union[np.ndarray, List[float]],
                            nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
                            l_vec: Union[np.ndarray, List[float]], excel_file_path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    logs = []; l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64); ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)
    try:
        nH_arr, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_jnp, excel_file_path); logs.extend(logs_h)
        nL_arr, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_jnp, excel_file_path); logs.extend(logs_l)
        nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path); logs.extend(logs_sub)
    except Exception as e: logs.append(f"Error preparing T calc data: {e}"); raise
    Ts = calculate_T_from_ep_core_jax(ep_vector_jnp, nH_arr, nL_arr, nSub_arr, l_vec_jnp)
    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts)}, logs

@jax.jit
def _compute_layer_matrix_scan_step_arbitrary(carry_matrix: jnp.ndarray, layer_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, None]:
     return _compute_layer_matrix_scan_step(carry_matrix, layer_data)

@jax.jit
def compute_stack_matrix_arbitrary_jax(ep_vector: jnp.ndarray, layer_indices_at_lval: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    num_layers = len(ep_vector)
    layers_scan_data = (ep_vector, layer_indices_at_lval, jnp.full(num_layers, l_val))
    M_initial = jnp.eye(2, dtype=jnp.complex128)
    M_final, _ = scan(_compute_layer_matrix_scan_step_arbitrary, M_initial, layers_scan_data)
    return M_final

@jax.jit
def calculate_single_wavelength_T_arbitrary(l_val: jnp.ndarray, ep_vector_contig: jnp.ndarray,
                                            layer_indices_at_lval: jnp.ndarray, nSub_at_lval: jnp.ndarray) -> jnp.ndarray:
    etainc = 1.0 + 0j; etasub = nSub_at_lval
    def calculate_for_valid_l(l_: jnp.ndarray) -> jnp.ndarray:
        M = compute_stack_matrix_arbitrary_jax(ep_vector_contig, layer_indices_at_lval, l_)
        m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        rs_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        rs_denominator_abs = jnp.abs(rs_denominator)
        safe_denominator = jnp.where(rs_denominator_abs < 1e-12, 1e-12 + 0j, rs_denominator)
        ts = (2.0 * etainc) / safe_denominator
        real_etasub = jnp.real(etasub); real_etainc = jnp.real(etainc)
        Ts = (real_etasub / real_etainc) * (ts * jnp.conj(ts))
        return jnp.where(rs_denominator_abs < 1e-12, jnp.nan, jnp.real(Ts))
    def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray: return jnp.nan
    Ts = cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)
    return Ts

@jax.jit
def calculate_T_from_ep_arbitrary_core_jax(ep_vector: jnp.ndarray,
                                           layer_indices_arr: jnp.ndarray,
                                           nSub_arr: jnp.ndarray,
                                           l_vec: jnp.ndarray) -> jnp.ndarray:
    if not l_vec.size: return jnp.zeros(0, dtype=jnp.float64)
    ep_vector_contig = jnp.asarray(ep_vector)
    layer_indices_arr_transposed = layer_indices_arr.T
    Ts_arr = vmap(calculate_single_wavelength_T_arbitrary, in_axes=(0, None, 0, 0))(l_vec, ep_vector_contig, layer_indices_arr_transposed, nSub_arr)
    return jnp.nan_to_num(Ts_arr, nan=0.0)

def calculate_T_from_ep_arbitrary_jax(ep_vector: Union[np.ndarray, List[float]],
                                      material_sequence: List[str],
                                      nSub_material: MaterialInputType,
                                      l_vec: Union[np.ndarray, List[float]],
                                      excel_file_path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    logs = []; l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64); ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64); num_layers = len(ep_vector)
    if len(material_sequence) != num_layers:
        if num_layers == 0 and not material_sequence: logs.append("Calculating for bare substrate.")
        else: raise ValueError(f"Length mismatch: sequence ({len(material_sequence)}) vs ep_vector ({num_layers}).")
    layer_indices_list = []
    for i, material_name in enumerate(material_sequence):
        try: nk_arr, logs_layer = _get_nk_array_for_lambda_vec(material_name, l_vec_jnp, excel_file_path); logs.extend(logs_layer); layer_indices_list.append(nk_arr)
        except Exception as e: logs.append(f"Error preparing material '{material_name}' (layer {i+1}): {e}"); raise
    if layer_indices_list: layer_indices_arr = jnp.stack(layer_indices_list, axis=0)
    else: layer_indices_arr = jnp.empty((0, len(l_vec_jnp)), dtype=jnp.complex128)
    try: nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path); logs.extend(logs_sub)
    except Exception as e: logs.append(f"Error preparing substrate: {e}"); raise
    Ts = calculate_T_from_ep_arbitrary_core_jax(ep_vector_jnp, layer_indices_arr, nSub_arr, l_vec_jnp)
    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts)}, logs

def _get_nk_at_lambda(material_definition: MaterialInputType, l_nm_target: float, excel_file_path: str) -> Tuple[complex, List[str]]:
    logs = []
    try:
        if isinstance(material_definition, (complex, float, int)): result = complex(material_definition)
        else:
            l_nm_target_jnp = jnp.array([l_nm_target], dtype=jnp.float64)
            if isinstance(material_definition, str):
                mat_upper = material_definition.upper()
                if mat_upper == "FUSED SILICA": result = complex(get_n_fused_silica(l_nm_target_jnp)[0])
                elif mat_upper == "BK7": result = complex(get_n_bk7(l_nm_target_jnp)[0])
                elif mat_upper == "D263": result = complex(get_n_d263(l_nm_target_jnp)[0])
                else:
                    sheet_name = material_definition; l_data, n_data, k_data = load_material_data_from_xlsx_sheet(excel_file_path, sheet_name)
                    if l_data is None: raise ValueError(f"Could not load data for '{sheet_name}'"); n_interp = np.interp(l_nm_target, l_data, n_data); k_interp = np.interp(l_nm_target, l_data, k_data); result = complex(n_interp, k_interp)
            elif isinstance(material_definition, tuple) and len(material_definition) == 3:
                l_data, n_data, k_data = material_definition; n_interp = np.interp(l_nm_target, l_data, n_data); k_interp = np.interp(l_nm_target, l_data, k_data); result = complex(n_interp, k_interp)
            else: raise TypeError(f"Unsupported material type: {type(material_definition)}")
        return result, logs
    except Exception as e: logs.append(f"Error getting nk @{l_nm_target}nm: {e}"); raise ValueError(f"Failed nk: {e}")

@jax.jit
def get_target_points_indices_jax(l_vec: jnp.ndarray, target_min: float, target_max: float) -> jnp.ndarray:
    if not l_vec.size: return jnp.empty(0, dtype=jnp.int64)
    indices = jnp.where((l_vec >= target_min) & (l_vec <= target_max), size=l_vec.shape[0], fill_value=-1)[0]
    return indices[indices != -1]

def calculate_initial_ep(emp: Union[List[float], Tuple[float,...]], l0: float,
                           nH0_material: MaterialInputType, nL0_material: MaterialInputType,
                           excel_file_path: str) -> Tuple[np.ndarray, List[str]]:
    logs = []; num_layers = len(emp); ep_initial = np.zeros(num_layers, dtype=np.float64)
    if l0 <= 0: logs.append("Warn: l0<=0."); return ep_initial, logs
    nH_real_at_l0 = -1.0; nL_real_at_l0 = -1.0
    try:
        nH_complex_at_l0, logs_h = _get_nk_at_lambda(nH0_material, l0, excel_file_path); logs.extend(logs_h)
        nL_complex_at_l0, logs_l = _get_nk_at_lambda(nL0_material, l0, excel_file_path); logs.extend(logs_l)
        nH_real_at_l0 = nH_complex_at_l0.real; nL_real_at_l0 = nL_complex_at_l0.real
        if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9: logs.append(f"Warn: nH/nL({l0}nm)<=0.")
    except Exception as e: logs.append(f"Error getting n({l0}nm): {e}. Thicknesses=0."); return np.zeros(num_layers, dtype=np.float64), logs
    for i in range(num_layers):
        multiplier = emp[i]; n_real = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real <= 1e-9: ep_initial[i] = 0.0
        else: ep_initial[i] = multiplier * l0 / (4.0 * n_real)
    return ep_initial, logs

def calculate_qwot_from_ep(ep_vector: np.ndarray, l0: float,
                            nH0_material: MaterialInputType, nL0_material: MaterialInputType,
                            excel_file_path: str) -> Tuple[np.ndarray, List[str]]:
    logs = []; num_layers = len(ep_vector); qwot_multipliers = np.full(num_layers, np.nan, dtype=np.float64)
    if l0 <= 0: logs.append("Warn: l0<=0. QWOT=NaN."); return qwot_multipliers, logs
    nH_real_at_l0 = -1.0; nL_real_at_l0 = -1.0
    try:
        nH_complex_at_l0, logs_h = _get_nk_at_lambda(nH0_material, l0, excel_file_path); logs.extend(logs_h)
        nL_complex_at_l0, logs_l = _get_nk_at_lambda(nL0_material, l0, excel_file_path); logs.extend(logs_l)
        nH_real_at_l0 = nH_complex_at_l0.real; nL_real_at_l0 = nL_complex_at_l0.real
        if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9: logs.append(f"Warn: nH/nL({l0}nm)<=0.")
    except Exception as e: logs.append(f"Error getting n({l0}nm): {e}"); return qwot_multipliers, logs
    for i in range(num_layers):
        n_real = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real > 1e-9: qwot_multipliers[i] = ep_vector[i] * (4.0 * n_real) / l0
    return qwot_multipliers, logs

def calculate_final_mse(res: Dict[str, np.ndarray], active_targets: List[Dict]) -> Tuple[Optional[float], int]:
    total_squared_error = 0.0; total_points_in_targets = 0; mse = None
    if not active_targets or not res or 'Ts' not in res or res['Ts'] is None or 'l' not in res or res['l'] is None: return mse, total_points_in_targets
    res_l_np = np.asarray(res['l']); res_ts_np = np.asarray(res['Ts'])
    if len(res_ts_np) == 0: return mse, total_points_in_targets
    for target in active_targets:
        l_min, l_max = target['min'], target['max']; t_min, t_max = target['target_min'], target['target_max']
        indices = np.where((res_l_np >= l_min) & (res_l_np <= l_max))[0]
        if indices.size > 0:
            valid_indices = indices[indices < len(res_ts_np)];
            if valid_indices.size == 0: continue
            calculated_Ts_in_zone = res_ts_np[valid_indices]; target_lambdas_in_zone = res_l_np[valid_indices]
            finite_mask = np.isfinite(calculated_Ts_in_zone); calculated_Ts_in_zone = calculated_Ts_in_zone[finite_mask]; target_lambdas_in_zone = target_lambdas_in_zone[finite_mask]
            if calculated_Ts_in_zone.size == 0: continue
            if abs(l_max - l_min) < 1e-9: interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
            else: slope = (t_max - t_min) / (l_max - l_min); interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)
            squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
            total_squared_error += np.sum(squared_errors); total_points_in_targets += len(calculated_Ts_in_zone)
    if total_points_in_targets > 0: mse = total_squared_error / total_points_in_targets
    elif active_targets: mse = np.nan
    return mse, total_points_in_targets

@jax.jit
def calculate_mse_for_optimization_penalized_jax(ep_vector: jnp.ndarray,
                                                  nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                                  l_vec_optim: jnp.ndarray, active_targets_tuple: Tuple,
                                                  min_thickness_phys_nm: float) -> jnp.ndarray:
    below_min_mask = (ep_vector < min_thickness_phys_nm) & (ep_vector > 1e-12)
    penalty_thin = jnp.sum(jnp.where(below_min_mask, (min_thickness_phys_nm - ep_vector)**2, 0.0)) * 1e5
    ep_vector_calc = jnp.maximum(ep_vector, min_thickness_phys_nm)
    Ts = calculate_T_from_ep_core_jax(ep_vector_calc, nH_arr, nL_arr, nSub_arr, l_vec_optim)
    total_squared_error = 0.0; total_points_in_targets = 0
    for i in range(len(active_targets_tuple)):
        l_min, l_max, t_min, t_max = active_targets_tuple[i]; target_mask = (l_vec_optim >= l_min) & (l_vec_optim <= l_max)
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t_full = t_min + slope * (l_vec_optim - l_min)
        squared_errors_full = (Ts - interpolated_target_t_full)**2
        masked_sq_error = jnp.where(target_mask, squared_errors_full, 0.0)
        total_squared_error += jnp.sum(masked_sq_error); total_points_in_targets += jnp.sum(target_mask)
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, 1e7)
    final_cost = mse + penalty_thin
    return jnp.nan_to_num(final_cost, nan=jnp.inf)

@jax.jit
def calculate_mse_arbitrary_sequence_jax(ep_vector: jnp.ndarray, layer_indices_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                           l_vec_eval: jnp.ndarray, active_targets_tuple: Tuple) -> jnp.ndarray:
    Ts = calculate_T_from_ep_arbitrary_core_jax(ep_vector, layer_indices_arr, nSub_arr, l_vec_eval)
    total_squared_error = 0.0; total_points_in_targets = 0
    for i in range(len(active_targets_tuple)):
        l_min, l_max, t_min, t_max = active_targets_tuple[i]; target_mask = (l_vec_eval >= l_min) & (l_vec_eval <= l_max)
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t_full = t_min + slope * (l_vec_eval - l_min)
        squared_errors_full = (Ts - interpolated_target_t_full)**2
        masked_sq_error = jnp.where(target_mask, squared_errors_full, 0.0)
        total_squared_error += jnp.sum(masked_sq_error); total_points_in_targets += jnp.sum(target_mask)
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, jnp.inf)
    return jnp.nan_to_num(mse, nan=jnp.inf)

def scipy_obj_grad_wrapper(ep_vector_np: np.ndarray,
                            nH_arr_jax: jnp.ndarray, nL_arr_jax: jnp.ndarray, nSub_arr_jax: jnp.ndarray,
                            l_vec_optim_jax: jnp.ndarray, active_targets_tuple_jax: Tuple, min_thickness_phys_nm_jax: float):
    ep_vector_jax = jnp.asarray(ep_vector_np)
    value_and_grad_fn = jax.value_and_grad(calculate_mse_for_optimization_penalized_jax)
    value_jax, grad_jax = value_and_grad_fn(ep_vector_jax, nH_arr_jax, nL_arr_jax, nSub_arr_jax, l_vec_optim_jax, active_targets_tuple_jax, min_thickness_phys_nm_jax)
    value_np = float(np.array(value_jax)); grad_np = np.array(grad_jax, dtype=np.float64)
    if not np.isfinite(value_np): value_np = 1e10
    if not np.all(np.isfinite(grad_np)): grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=1e5, neginf=-1e5)
    return value_np, grad_np

@st.cache_data
def get_available_material_names(excel_path):
    try:
        xl = pd.ExcelFile(excel_path); sheet_names = [name for name in xl.sheet_names if not name.startswith("Sheet")]
        return ["Constant"] + sorted(sheet_names)
    except FileNotFoundError: st.sidebar.error(f"File '{excel_path}' not found."); return ["Constant"]
    except Exception as e: st.sidebar.error(f"Error reading Excel sheets: {e}"); return ["Constant"]

def get_material_value(role: str) -> MaterialInputType:
    selected_material = st.session_state[f'selected_{role}_material']
    if selected_material == "Constant":
        try:
            n_r = st.session_state[f'n{role}_r']; n_i = st.session_state.get(f'n{role}_i', 0.0)
            if n_r <= 0: st.sidebar.warning(f"n' for {role} must be > 0. Using default."); return complex(1.5, 0.0) if role == 'Sub' else complex(2.0 if role == 'H' else 1.45, 0.0)
            if n_i < 0: st.sidebar.warning(f"k for {role} must be >= 0. Using 0."); n_i = 0.0
            return complex(n_r, n_i)
        except KeyError: st.sidebar.error(f"Constant value input missing for {role}."); return complex(1.5, 0.0) if role == 'Sub' else complex(2.0 if role == 'H' else 1.45, 0.0)
        except Exception as e: st.sidebar.error(f"Invalid constant value for {role}: {e}"); return complex(1.5, 0.0) if role == 'Sub' else complex(2.0 if role == 'H' else 1.45, 0.0)
    else: return selected_material

def get_validated_input_params() -> Optional[Dict]:
    params = {}; valid = True
    try:
        params['nH_material'] = get_material_value('H'); params['nL_material'] = get_material_value('L'); params['nSub_material'] = get_material_value('Sub')
        params['emp_str'] = st.session_state.get('nominal_qwot_str', ""); params['initial_layer_number'] = st.session_state.get('initial_layer_number', 20); params['l0'] = st.session_state.get('l0_qwot', 500.0); params['l_step'] = st.session_state.get('lambda_step', 10.0); params['maxiter'] = st.session_state.get('max_iter', 1000); params['maxfun'] = st.session_state.get('max_fun', 1000); params['auto_thin_threshold'] = st.session_state.get('auto_thin_threshold', 1.0)
        if params['l0'] <= 0: st.error("l0 > 0."); valid = False
        if params['l_step'] <= 0: st.error("λ Step > 0."); valid = False
        if params['maxiter'] <= 0: st.error("Max Iter > 0."); valid = False
        if params['maxfun'] <= 0: st.error("Max Eval > 0."); valid = False
        if params['initial_layer_number'] <= 0 and params['emp_str'].strip() == "": st.error("Layer Number > 0 if QWOT empty."); valid = False
        if params['auto_thin_threshold'] < 0: st.error("Thin Thr >= 0."); valid = False
    except KeyError as e: st.error(f"Missing input state: {e}"); return None
    except Exception as e: st.error(f"Error retrieving params: {e}"); return None
    return params if valid else None
def get_validated_active_targets_from_state() -> Optional[List[Dict]]:
    active_targets = []
    valid = True
    num_target_rows = st.session_state.get('num_target_rows', 5)
    for i in range(num_target_rows):
        target_id_prefix = f"target_{i}"
        is_enabled = st.session_state.get(f"{target_id_prefix}_enabled", False)
        if is_enabled:
            try:
                l_min = float(st.session_state[f"{target_id_prefix}_min"])
                l_max = float(st.session_state[f"{target_id_prefix}_max"])
                t_min = float(st.session_state[f"{target_id_prefix}_target_min"])
                t_max = float(st.session_state[f"{target_id_prefix}_target_max"])
                if l_max < l_min: st.error(f"Target {i+1}: λ max < λ min."); valid = False
                if not (0.0 <= t_min <= 1.0): st.error(f"Target {i+1}: T@min out of [0,1]."); valid = False
                if not (0.0 <= t_max <= 1.0): st.error(f"Target {i+1}: T@max out of [0,1]."); valid = False
                active_targets.append({'min': l_min, 'max': l_max, 'target_min': t_min, 'target_max': t_max})
            except KeyError as e: st.error(f"Target {i+1}: Missing state key {e}"); valid = False
            except ValueError as e: st.error(f"Target {i+1}: Invalid number - {e}"); valid = False
    return active_targets if valid else None

def get_lambda_range_from_targets(validated_targets: Optional[List[Dict]]) -> Tuple[Optional[float], Optional[float]]:
    overall_min, overall_max = None, None
    if validated_targets:
        try:
            all_mins = [t['min'] for t in validated_targets]
            all_maxs = [t['max'] for t in validated_targets]
            if all_mins: overall_min = min(all_mins)
            if all_maxs: overall_max = max(all_maxs)
        except (KeyError, TypeError): return None, None
    return overall_min, overall_max

def add_log_message(message: Union[str, List[str]]):
    if 'log_messages' not in st.session_state: st.session_state.log_messages = []
    now = datetime.datetime.now().strftime('%H:%M:%S')
    if isinstance(message, list): full_message = "\n".join(f"[{now}] {str(msg)}" for msg in message)
    else: full_message = f"[{now}] {str(message)}"
    st.session_state.log_messages.append(full_message)
    max_log_size = 500
    if len(st.session_state.log_messages) > max_log_size:
         st.session_state.log_messages = st.session_state.log_messages[-max_log_size:]

def _run_core_optimization(ep_start_optim: np.ndarray,
                            validated_inputs: Dict, active_targets: List[Dict],
                            min_thickness_phys: float, log_prefix: str = ""
                            ) -> Tuple[Optional[np.ndarray], bool, float, List[str], str, int, int]:
    logs = []
    num_layers_start = len(ep_start_optim)
    optim_success = False; final_cost = np.inf; result_message_str = "Opt not run/failed early."; nit_total = 0; nfev_total = 0; final_ep = None
    if num_layers_start == 0: logs.append(f"{log_prefix}Cannot optimize empty structure."); return None, False, np.inf, logs, "Empty structure", 0, 0
    try:
        l_min_optim, l_max_optim = get_lambda_range_from_targets(active_targets)
        if l_min_optim is None or l_max_optim is None: raise ValueError("Cannot get optim lambda range.")
        l_step_optim = validated_inputs['l_step']; num_pts_optim = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
        l_vec_optim_np = np.geomspace(l_min_optim, l_max_optim, num_pts_optim); l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
        if not l_vec_optim_np.size: raise ValueError("Failed lambda vector gen."); l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        logs.append(f"{log_prefix}Prep indices {len(l_vec_optim_jax)} lambdas..."); prep_start_time = time.time()
        nH_material = validated_inputs['nH_material']; nL_material = validated_inputs['nL_material']; nSub_material = validated_inputs['nSub_material']
        nH_arr_optim, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH); logs.extend(logs_h)
        nL_arr_optim, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH); logs.extend(logs_l)
        nSub_arr_optim, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH); logs.extend(logs_sub)
        logs.append(f"{log_prefix}Index prep {time.time() - prep_start_time:.3f}s.")
        active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
        static_args_for_scipy = (nH_arr_optim, nL_arr_optim, nSub_arr_optim, l_vec_optim_jax, active_targets_tuple, jnp.float64(min_thickness_phys))
        lbfgsb_bounds = [(min_thickness_phys, None)] * num_layers_start
        options = {'maxiter': validated_inputs['maxiter'], 'maxfun': validated_inputs['maxfun'], 'disp': False, 'ftol': 1e-12, 'gtol': 1e-8}
        logs.append(f"{log_prefix}Starting L-BFGS-B..."); opt_start_time = time.time()
        ep_start_bounded = np.maximum(np.asarray(ep_start_optim, dtype=np.float64), min_thickness_phys)
        result = minimize(scipy_obj_grad_wrapper, ep_start_bounded, args=static_args_for_scipy, method='L-BFGS-B', jac=True, bounds=lbfgsb_bounds, options=options)
        logs.append(f"{log_prefix}L-BFGS-B {time.time() - opt_start_time:.3f}s.")
        final_cost = result.fun if np.isfinite(result.fun) else np.inf; result_message_str = result.message.decode('utf-8') if isinstance(result.message, bytes) else str(result.message)
        nit_total = result.nit if hasattr(result, 'nit') else 0; nfev_total = result.nfev if hasattr(result, 'nfev') else 0
        is_success_or_limit = result.success or result.status in [0, 1, 2]
        if is_success_or_limit and np.isfinite(final_cost):
            final_ep = np.maximum(result.x, min_thickness_phys); optim_success = True
            log_status = "succeeded" if result.status == 0 else f"stopped (status {result.status})"
            logs.append(f"{log_prefix}Opt {log_status}! Cost:{final_cost:.3e}, Iter:{nit_total}, Eval:{nfev_total}, Msg:{result_message_str}")
        else:
            optim_success = False; final_ep = ep_start_bounded.copy()
            logs.append(f"{log_prefix}Opt FAILED. Status:{result.status}, Msg:{result_message_str}")
            try:
                reverted_cost, _ = scipy_obj_grad_wrapper(final_ep, *static_args_for_scipy)
                logs.append(f"{log_prefix}Reverted. Recalc cost:{reverted_cost:.3e}"); final_cost = reverted_cost if np.isfinite(reverted_cost) else np.inf
            except Exception as cost_e: logs.append(f"{log_prefix}Reverted. ERROR recalc cost: {cost_e}"); final_cost = np.inf
            nit_total = 0; nfev_total = 0
    except Exception as e_optim:
        logs.append(f"{log_prefix}ERROR core opt: {e_optim}\n{traceback.format_exc(limit=1)}")
        final_ep = np.maximum(np.asarray(ep_start_optim), min_thickness_phys) if ep_start_optim is not None else None
        optim_success = False; final_cost = np.inf; result_message_str = f"Exception: {e_optim}"; nit_total = 0; nfev_total = 0
    if final_ep is None and ep_start_optim is not None: final_ep = np.maximum(np.asarray(ep_start_optim), min_thickness_phys)
    return final_ep, optim_success, final_cost, logs, result_message_str, nit_total, nfev_total

def _perform_needle_insertion_scan(ep_vector_in: np.ndarray,
                                   nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
                                   l_vec_optim_np: np.ndarray, active_targets: List[Dict],
                                   cost_function_jax: Callable, min_thickness_phys: float, base_needle_thickness_nm: float,
                                   scan_step: float, l0_repr: float, log_prefix: str = ""
                                   ) -> Tuple[Optional[np.ndarray], float, List[str], int]:
    logs = []; num_layers_in = len(ep_vector_in); best_ep_found = None; min_cost_found = np.inf; best_insertion_z = -1.0; best_insertion_layer_idx = -1; tested_insertions = 0
    if num_layers_in == 0: logs.append(f"{log_prefix}Empty structure."); return None, np.inf, logs, -1
    logs.append(f"{log_prefix}Needle scan N={num_layers_in}. Step:{scan_step}nm, Needle:{base_needle_thickness_nm:.3f}nm.")
    try:
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        nH_arr_optim, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH); logs.extend(logs_h)
        nL_arr_optim, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH); logs.extend(logs_l)
        nSub_arr_optim, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH); logs.extend(logs_sub)
        active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
        static_args_cost_fn = (nH_arr_optim, nL_arr_optim, nSub_arr_optim, l_vec_optim_jax, active_targets_tuple, jnp.float64(min_thickness_phys))
        cost_fn_compiled = jax.jit(cost_function_jax)
        initial_cost_jax = cost_fn_compiled(jnp.asarray(ep_vector_in), *static_args_cost_fn); initial_cost = float(np.array(initial_cost_jax))
        if not np.isfinite(initial_cost): logs.append(f"{log_prefix}ERROR: Initial cost not finite ({initial_cost}). Abort."); return None, np.inf, logs, -1
        min_cost_found = initial_cost; logs.append(f"{log_prefix}Initial cost: {initial_cost:.6e}")
    except Exception as e_prep: logs.append(f"{log_prefix}ERROR prep needle scan: {e_prep}. Abort."); return None, np.inf, logs, -1
    ep_cumsum = np.cumsum(ep_vector_in); total_thickness = ep_cumsum[-1] if num_layers_in > 0 else 0.0
    try:
        nH_c_l0, logs_hnk = _get_nk_at_lambda(nH_material, l0_repr, EXCEL_FILE_PATH); logs.extend(logs_hnk)
        nL_c_l0, logs_lnk = _get_nk_at_lambda(nL_material, l0_repr, EXCEL_FILE_PATH); logs.extend(logs_lnk)
        nH_real_l0 = nH_c_l0.real; nL_real_l0 = nL_c_l0.real
    except Exception as e_nk: logs.append(f"{log_prefix}WARN: No n(l0={l0_repr}nm): {e_nk}"); nH_real_l0, nL_real_l0 = -1.0, -1.0
    for z in np.arange(scan_step, total_thickness, scan_step):
        current_layer_idx = -1; layer_start_z = 0.0
        for i in range(num_layers_in):
            layer_end_z = ep_cumsum[i]
            if z > layer_start_z and z <= layer_end_z:
                 t_part1 = z - layer_start_z; t_part2 = layer_end_z - z
                 if t_part1 >= min_thickness_phys and t_part2 >= min_thickness_phys: current_layer_idx = i
                 else: current_layer_idx = -2
                 break
            layer_start_z = layer_end_z
        if current_layer_idx < 0: continue
        tested_insertions += 1
        is_H_layer_split = (current_layer_idx % 2 == 0)
        if nH_real_l0 < 0 or nL_real_l0 < 0: logs.append(f"{log_prefix}Skip z={z:.2f}. No n(l0)."); continue
        t_layer_split_1 = z - (ep_cumsum[current_layer_idx-1] if current_layer_idx > 0 else 0.0)
        t_layer_split_2 = ep_cumsum[current_layer_idx] - z
        ep_temp_np = np.concatenate((ep_vector_in[:current_layer_idx], [t_layer_split_1, base_needle_thickness_nm, t_layer_split_2], ep_vector_in[current_layer_idx+1:]))
        ep_temp_np = np.maximum(ep_temp_np, min_thickness_phys)
        try:
            current_cost_jax = cost_fn_compiled(jnp.asarray(ep_temp_np), *static_args_cost_fn)
            current_cost = float(np.array(current_cost_jax))
            if np.isfinite(current_cost) and current_cost < min_cost_found:
                 min_cost_found = current_cost; best_ep_found = ep_temp_np.copy(); best_insertion_z = z; best_insertion_layer_idx = current_layer_idx
        except Exception as e_cost: logs.append(f"{log_prefix}WARN: Cost calc fail z={z:.2f}. {e_cost}"); continue
    if best_ep_found is not None:
        improvement = initial_cost - min_cost_found; logs.append(f"{log_prefix}Needle Scan Done. Tested {tested_insertions}. Best improve: {improvement:.6e}")
        logs.append(f"{log_prefix}Best insert z={best_insertion_z:.3f} nm in layer {best_insertion_layer_idx + 1}.")
        return best_ep_found, min_cost_found, logs, best_insertion_layer_idx
    else: logs.append(f"{log_prefix}Needle Scan Done. No improvement."); return None, initial_cost, logs, -1

def _perform_layer_merge_or_removal_only(ep_vector_in: np.ndarray, min_thickness_phys: float,
                                           log_prefix: str = "", target_layer_index: Optional[int] = None,
                                           threshold_for_removal: Optional[float] = None) -> Tuple[np.ndarray, bool, List[str]]:
    current_ep = ep_vector_in.copy(); logs = []; num_layers = len(current_ep); structure_changed = False; ep_after_merge = None
    if num_layers < 1: logs.append(f"{log_prefix}Empty."); return current_ep, False, logs
    if num_layers <= 2 and target_layer_index is None: logs.append(f"{log_prefix}N<=2. No auto remove.")
    try:
        thin_layer_index = -1; min_thickness_found = np.inf
        if target_layer_index is not None:
            if 0 <= target_layer_index < num_layers:
                target_thickness = current_ep[target_layer_index]
                if threshold_for_removal is None or target_thickness < threshold_for_removal:
                     if target_thickness >= min_thickness_phys: thin_layer_index = target_layer_index; min_thickness_found = target_thickness; logs.append(f"{log_prefix}Targeting L{thin_layer_index+1} ({min_thickness_found:.3f}nm).")
                     else: logs.append(f"{log_prefix}Target L{target_layer_index+1} < min phys.")
                else: logs.append(f"{log_prefix}Target L{target_layer_index+1} >= threshold.")
            else: logs.append(f"{log_prefix}Invalid target idx. Searching auto.")
            if thin_layer_index == -1: target_layer_index = None
        if target_layer_index is None:
            candidate_indices = np.where(current_ep >= min_thickness_phys)[0]
            if candidate_indices.size > 0:
                candidate_thicknesses = current_ep[candidate_indices]
                if threshold_for_removal is not None:
                    valid_for_removal_mask = candidate_thicknesses < threshold_for_removal
                    if np.any(valid_for_removal_mask): candidate_indices = candidate_indices[valid_for_removal_mask]; candidate_thicknesses = candidate_thicknesses[valid_for_removal_mask]
                    else: candidate_indices = np.array([], dtype=int)
                if candidate_indices.size > 0: min_idx_local = np.argmin(candidate_thicknesses); thin_layer_index = candidate_indices[min_idx_local]; min_thickness_found = candidate_thicknesses[min_idx_local]; logs.append(f"{log_prefix}Found thinnest L{thin_layer_index+1} ({min_thickness_found:.3f}nm).")
        if thin_layer_index == -1:
            if threshold_for_removal is not None and target_layer_index is None: logs.append(f"{log_prefix}No layer < {threshold_for_removal:.3f}nm found.")
            elif target_layer_index is None: logs.append(f"{log_prefix}No valid layer found.")
            return current_ep, False, logs
        merged_info = ""
        if num_layers <= 2:
            if thin_layer_index >= 0: ep_after_merge = np.array([], dtype=np.float64); merged_info = f"Removed remaining {num_layers} layer(s)."; structure_changed = True
            else: logs.append(f"{log_prefix}Cannot remove N<=2 without target."); return current_ep, False, logs
        elif thin_layer_index == 0: ep_after_merge = current_ep[2:]; merged_info = f"Removed L1&L2."; structure_changed = True
        elif thin_layer_index == num_layers - 1:
             if num_layers >= 2: ep_after_merge = current_ep[:-2]; merged_info = f"Removed L{num_layers-1}&L{num_layers}."; structure_changed = True
             else: logs.append(f"{log_prefix}Cannot remove last pair."); return current_ep, False, logs
        else:
             merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
             ep_after_merge = np.concatenate((current_ep[:thin_layer_index - 1], [merged_thickness], current_ep[thin_layer_index + 2:]))
             merged_info = f"Removed L{thin_layer_index+1}, merged L{thin_layer_index}&L{thin_layer_index+2} -> {merged_thickness:.3f}nm"; structure_changed = True
        if structure_changed and ep_after_merge is not None:
            logs.append(f"{log_prefix}{merged_info}. New size: {len(ep_after_merge)}.");
            if ep_after_merge.size > 0: ep_after_merge = np.maximum(ep_after_merge, min_thickness_phys)
            return ep_after_merge, True, logs
        else:
            if structure_changed: logs.append(f"{log_prefix}Merge logic error.")
            return current_ep, False, logs
    except Exception as e_merge: logs.append(f"{log_prefix}ERROR merge/remove: {e_merge}\n{traceback.format_exc(limit=1)}"); return current_ep, False, logs

# --- Placeholder for Plotting (Defined later) ---
def draw_plots(*args, **kwargs): pass
# --- Placeholder for Index Plotting (Defined later) ---
def display_material_index_plot(): pass
# --- Placeholder for Save Data Helper (Defined later) ---
def _collect_design_data_for_save(*args, **kwargs): pass

# --- Needle Iteration Runner ---
# (Helper for Auto Mode)
def _run_needle_iterations(ep_start: np.ndarray, num_needles: int,
                           validated_inputs: Dict, active_targets: List[Dict],
                           min_thickness_phys: float, l_vec_optim_np: np.ndarray,
                           scan_step_nm: float, base_needle_thickness_nm: float,
                           cost_function_jax: Callable, # Pass the JAX cost function
                           log_prefix: str = "") -> Tuple[np.ndarray, float, List[str], int, int, int]:
    logs = []
    best_ep_overall = np.asarray(ep_start).copy()
    best_mse_overall = np.inf
    total_nit_needles = 0; total_nfev_needles = 0; successful_reopts_count = 0
    nH_material = validated_inputs['nH_material']; nL_material = validated_inputs['nL_material']
    nSub_material = validated_inputs['nSub_material']
    l0_repr = validated_inputs.get('l0', 500.0)

    try:
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        nH_arr, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH); logs.extend(logs_h)
        nL_arr, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH); logs.extend(logs_l)
        nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH); logs.extend(logs_sub)
        active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
        static_args_cost_fn = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, jnp.float64(min_thickness_phys))
        cost_fn_compiled = jax.jit(cost_function_jax)
        initial_cost_jax = cost_fn_compiled(jnp.asarray(best_ep_overall), *static_args_cost_fn)
        best_mse_overall = float(np.array(initial_cost_jax))
        if not np.isfinite(best_mse_overall): raise ValueError("Initial MSE for needle iterations not finite.")
        logs.append(f"{log_prefix}Starting needle iterations. Initial MSE: {best_mse_overall:.6e}")
    except Exception as e:
        logs.append(f"{log_prefix}ERROR calculating initial MSE for needle iterations: {e}")
        return ep_start, np.inf, logs, 0, 0, 0

    for i in range(num_needles):
        logs.append(f"{log_prefix}Needle Iteration {i + 1}/{num_needles}")
        current_ep_iter = best_ep_overall.copy()
        if len(current_ep_iter) == 0: logs.append(f"{log_prefix}Empty structure, stopping."); break

        ep_after_scan, cost_after_scan, scan_logs, inserted_idx = _perform_needle_insertion_scan(
            current_ep_iter, nH_material, nL_material, nSub_material, l_vec_optim_np,
            active_targets, cost_function_jax, min_thickness_phys,
            base_needle_thickness_nm, scan_step_nm, l0_repr, log_prefix=f"{log_prefix}Scan {i+1} ")
        logs.extend(scan_logs)

        if ep_after_scan is None: logs.append(f"{log_prefix}Needle scan {i + 1} found no improvement. Stopping iterations."); break
        logs.append(f"{log_prefix}Needle scan {i + 1} found potential improvement. Re-optimizing...")

        # Need to ensure lambda range is in validated_inputs for core optimizer
        l_min_val_iter, l_max_val_iter = get_lambda_range_from_targets(active_targets)
        if l_min_val_iter is None or l_max_val_iter is None: raise ValueError("Cannot determine lambda range for needle re-opt.")
        validated_inputs_iter = validated_inputs.copy() # Avoid modifying original dict if passed by reference
        validated_inputs_iter['l_range_deb'] = l_min_val_iter
        validated_inputs_iter['l_range_fin'] = l_max_val_iter

        ep_after_reopt, optim_success, final_cost_reopt, optim_logs, _, nit_reopt, nfev_reopt = \
            _run_core_optimization(ep_after_scan, validated_inputs_iter, active_targets,
                                   min_thickness_phys, log_prefix=f"{log_prefix}Re-Opt {i+1} ")
        logs.extend(optim_logs)

        if not optim_success: logs.append(f"{log_prefix}Re-optimization after needle scan {i + 1} failed. Stopping."); break

        logs.append(f"{log_prefix}Re-optimization {i + 1} successful. New MSE: {final_cost_reopt:.6e}.")
        total_nit_needles += nit_reopt; total_nfev_needles += nfev_reopt; successful_reopts_count += 1

        if final_cost_reopt < best_mse_overall - MSE_IMPROVEMENT_TOLERANCE:
            logs.append(f"{log_prefix}MSE improved from {best_mse_overall:.6e}. Updating best result.")
            best_ep_overall = ep_after_reopt.copy(); best_mse_overall = final_cost_reopt
        else:
            logs.append(f"{log_prefix}New MSE ({final_cost_reopt:.6e}) not improved vs best ({best_mse_overall:.6e}). Stopping.")
            best_ep_overall = ep_after_reopt.copy(); best_mse_so_far = final_cost_reopt # Still update to last successful result
            break

    logs.append(f"{log_prefix}End needle iterations. Final best MSE: {best_mse_overall:.6e}")
    logs.append(f"{log_prefix}Total Iter/Eval during {successful_reopts_count} re-opts: {total_nit_needles}/{total_nfev_needles}")
    return best_ep_overall, best_mse_overall, logs, total_nit_needles, total_nfev_needles, successful_reopts_count

@jax.jit
def calculate_M_for_thickness(thickness: jnp.ndarray, n_complex_layer: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    eta = n_complex_layer; safe_l_val = jnp.maximum(l_val, 1e-9); safe_eta = jnp.where(jnp.abs(eta) < 1e-12, 1e-12 + 0j, eta)
    phi = (2 * jnp.pi / safe_l_val) * (n_complex_layer * thickness); cos_phi = jnp.cos(phi); sin_phi = jnp.sin(phi)
    m00 = jnp.where(thickness > 1e-12, cos_phi, 1.0); m01 = jnp.where(thickness > 1e-12, (1j / safe_eta) * sin_phi, 0.0)
    m10 = jnp.where(thickness > 1e-12, 1j * eta * sin_phi, 0.0); m11 = jnp.where(thickness > 1e-12, cos_phi, 1.0)
    return jnp.array([[m00, m01], [m10, m11]], dtype=jnp.complex128)

calculate_M_batch_for_thickness = vmap(calculate_M_for_thickness, in_axes=(None, None, 0))

@jax.jit
def get_layer_matrices_qwot(layer_idx: int, initial_layer_number: int, l0: float,
                            nH_c_l0: jnp.ndarray, nL_c_l0: jnp.ndarray,
                            l_vec: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    predicate_is_H = (layer_idx % 2 == 0); n_real_l0 = jax.lax.select(predicate_is_H, nH_c_l0.real, nL_c_l0.real); n_complex_for_matrix = jax.lax.select(predicate_is_H, nH_c_l0, nL_c_l0)
    denom = 4.0 * jnp.maximum(n_real_l0, 1e-9); safe_l0 = jnp.maximum(l0, 1e-9)
    ep1_calc = 1.0 * safe_l0 / denom; ep2_calc = 2.0 * safe_l0 / denom
    ep1 = jnp.where(n_real_l0 > 1e-9, ep1_calc, 0.0); ep2 = jnp.where(n_real_l0 > 1e-9, ep2_calc, 0.0)
    ep1 = jnp.maximum(ep1, MIN_THICKNESS_PHYS_NM * (ep1 > 1e-12)); ep2 = jnp.maximum(ep2, MIN_THICKNESS_PHYS_NM * (ep2 > 1e-12))
    M_1qwot_batch = calculate_M_batch_for_thickness(ep1, n_complex_for_matrix, l_vec)
    M_2qwot_batch = calculate_M_batch_for_thickness(ep2, n_complex_for_matrix, l_vec)
    return M_1qwot_batch, M_2qwot_batch

@jax.jit
def compute_half_product(multiplier_indices: jnp.ndarray, layer_matrices_half: jnp.ndarray) -> jnp.ndarray:
    N_half = layer_matrices_half.shape[0]; L = layer_matrices_half.shape[2]; init_prod = jnp.tile(jnp.eye(2, dtype=jnp.complex128), (L, 1, 1))
    def multiply_step(carry_prod: jnp.ndarray, layer_idx: int) -> Tuple[jnp.ndarray, None]:
        multiplier_idx = multiplier_indices[layer_idx]; M_k = layer_matrices_half[layer_idx, multiplier_idx, :, :, :]; new_prod = vmap(jnp.matmul)(M_k, carry_prod)
        return new_prod, None
    final_prod, _ = jax.lax.scan(multiply_step, init_prod, jnp.arange(N_half))
    return final_prod

@jax.jit
def get_T_from_batch_matrix(M_batch: jnp.ndarray, nSub_arr: jnp.ndarray) -> jnp.ndarray:
    etainc = 1.0 + 0j; etasub_batch = nSub_arr; m00 = M_batch[:, 0, 0]; m01 = M_batch[:, 0, 1]; m10 = M_batch[:, 1, 0]; m11 = M_batch[:, 1, 1]
    rs_den = (etainc * m00 + etasub_batch * m11 + etainc * etasub_batch * m01 + m10); rs_den_abs = jnp.abs(rs_den)
    safe_den = jnp.where(rs_den_abs < 1e-12, 1e-12 + 0j, rs_den); ts = (2.0 * etainc) / safe_den
    real_etasub_batch = jnp.real(etasub_batch); real_etainc = 1.0; Ts_complex = (real_etasub_batch / real_etainc) * (ts * jnp.conj(ts))
    Ts = jnp.real(Ts_complex); return jnp.where(rs_den_abs < 1e-12, 0.0, jnp.nan_to_num(Ts, nan=0.0))

@jax.jit
def calculate_mse_basic_jax(Ts: jnp.ndarray, l_vec: jnp.ndarray, targets_tuple: Tuple) -> jnp.ndarray:
    total_squared_error = 0.0; total_points_in_targets = 0
    for i in range(len(targets_tuple)):
        l_min, l_max, t_min, t_max = targets_tuple[i]; target_mask = (l_vec >= l_min) & (l_vec <= l_max)
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t = t_min + slope * (l_vec - l_min); squared_errors = (Ts - interpolated_target_t)**2
        masked_sq_error = jnp.where(target_mask, squared_errors, 0.0); total_squared_error += jnp.sum(masked_sq_error); total_points_in_targets += jnp.sum(target_mask)
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, jnp.inf)
    return jnp.nan_to_num(mse, nan=jnp.inf)

@jax.jit
def combine_and_calc_mse(prod1: jnp.ndarray, prod2: jnp.ndarray, nSub_arr_in: jnp.ndarray, l_vec_in: jnp.ndarray, targets_tuple_in: Tuple) -> jnp.ndarray:
    M_total = vmap(jnp.matmul)(prod2, prod1); Ts = get_T_from_batch_matrix(M_total, nSub_arr_in); mse = calculate_mse_basic_jax(Ts, l_vec_in, targets_tuple_in)
    return mse

def _execute_split_stack_scan(current_l0: float, initial_layer_number: int,
                              nH_c_l0: jnp.ndarray, nL_c_l0: jnp.ndarray, nSub_arr_scan: jnp.ndarray,
                              l_vec_eval_sparse_jax: jnp.ndarray, active_targets_tuple: Tuple
                              ) -> Tuple[float, Optional[np.ndarray], List[str]]:
    logs = []; num_combinations = 2**initial_layer_number; logs.append(f"  [Scan l0={current_l0:.2f}] Testing {num_combinations:,} QWOT combinations...")
    precompute_start_time = time.time(); layer_matrices_list = []
    try:
        for i in range(initial_layer_number): m1, m2 = get_layer_matrices_qwot(i, initial_layer_number, current_l0, nH_c_l0, nL_c_l0, l_vec_eval_sparse_jax); layer_matrices_list.append(jnp.stack([m1, m2], axis=0))
        all_layer_matrices = jnp.stack(layer_matrices_list, axis=0); all_layer_matrices.block_until_ready()
    except Exception as e_mat: logs.append(f"  ERROR Precomputing Matrices: {e_mat}"); return np.inf, None, logs
    logs.append(f"    Matrix precomp {time.time() - precompute_start_time:.3f}s.")
    N1 = initial_layer_number // 2; N2 = initial_layer_number - N1; num_comb1 = 2**N1; num_comb2 = 2**N2
    indices1 = jnp.arange(num_comb1); indices2 = jnp.arange(num_comb2); powers1 = 2**jnp.arange(N1); powers2 = 2**jnp.arange(N2)
    multiplier_indices1 = jnp.not_equal(indices1[:, None] & powers1, 0).astype(jnp.int32); multiplier_indices2 = jnp.not_equal(indices2[:, None] & powers2, 0).astype(jnp.int32)
    matrices_half1 = all_layer_matrices[:N1]; matrices_half2 = all_layer_matrices[N1:]
    half1_start_time = time.time(); compute_half_product_jit = jax.jit(compute_half_product); partial_products1 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices1, matrices_half1); partial_products1.block_until_ready()
    logs.append(f"    Partial products 1/2 {time.time() - half1_start_time:.3f}s.")
    half2_start_time = time.time(); partial_products2 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices2, matrices_half2); partial_products2.block_until_ready()
    logs.append(f"    Partial products 2/2 {time.time() - half2_start_time:.3f}s.")
    combine_start_time = time.time(); combine_and_calc_mse_jit = jax.jit(combine_and_calc_mse); vmap_inner = vmap(combine_and_calc_mse_jit, in_axes=(None, 0, None, None, None)); vmap_outer = vmap(vmap_inner, in_axes=(0, None, None, None, None))
    all_mses_nested = vmap_outer(partial_products1, partial_products2, nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple); all_mses_nested.block_until_ready()
    logs.append(f"    Combination & MSE {time.time() - combine_start_time:.3f}s.")
    all_mses_flat = all_mses_nested.reshape(-1); best_idx_flat = jnp.argmin(all_mses_flat); current_best_mse = float(all_mses_flat[best_idx_flat])
    if not np.isfinite(current_best_mse): logs.append(f"    Warning: No valid combination found."); return np.inf, None, logs
    best_idx_half1, best_idx_half2 = jnp.unravel_index(best_idx_flat, (num_comb1, num_comb2))
    best_indices_h1 = multiplier_indices1[best_idx_half1]; best_indices_h2 = multiplier_indices2[best_idx_half2]
    best_multipliers_h1 = 1.0 + best_indices_h1.astype(jnp.float64); best_multipliers_h2 = 1.0 + best_indices_h2.astype(jnp.float64)
    current_best_multipliers = jnp.concatenate([best_multipliers_h1, best_multipliers_h2])
    logs.append(f"    Best Scan MSE for l0={current_l0:.2f}: {current_best_mse:.6e}")
    return current_best_mse, np.array(current_best_multipliers), logs

def draw_plots(res: Optional[Dict], current_ep: Optional[np.ndarray], l0_repr: float,
               nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
               active_targets_for_plot: List[Dict], mse: Optional[float],
               is_optimized: bool = False, method_name: str = "",
               res_optim_grid: Optional[Dict] = None, material_sequence: Optional[List[str]] = None,
               fig_placeholder=None):
    if fig_placeholder is None: st.warning("Plotting requires a figure placeholder."); return
    fig = fig_placeholder['fig']; axes = fig_placeholder['axes']; fig.clear()
    if not isinstance(axes, (list, np.ndarray)) or len(axes) != 3: axes = fig.subplots(1, 3); fig_placeholder['axes'] = axes
    else:
         for ax in axes: ax.cla()
    ax_spec, ax_idx, ax_stack = axes[0], axes[1], axes[2]
    line_ts = None; overall_l_min_plot, overall_l_max_plot = None, None; plot_title_status = "Optimized" if is_optimized else "Nominal"; opt_method_str = f" ({method_name})" if method_name else ""
    if res is not None and isinstance(res, dict) and 'l' in res and 'Ts' in res and res['l'] is not None and len(res['l']) > 0:
        res_l_plot = np.asarray(res['l']); res_ts_plot = np.asarray(res['Ts']); overall_l_min_plot, overall_l_max_plot = res_l_plot.min(), res_l_plot.max()
        line_ts, = ax_spec.plot(res_l_plot, res_ts_plot, label='Transmittance', linestyle='-', color='blue', linewidth=1.5)
        plotted_target_label = False; current_target_lines = []
        if active_targets_for_plot:
            for i, target in enumerate(active_targets_for_plot):
                l_min, l_max = target['min'], target['max']; t_min, t_max = target['target_min'], target['target_max']; x_coords, y_coords = [l_min, l_max], [t_min, t_max]
                label = f'Target {i+1}' if not plotted_target_label else "_nolegend_"; line_target, = ax_spec.plot(x_coords, y_coords, 'r--', linewidth=1.5, alpha=0.8, label=label, zorder=5)
                marker_target = ax_spec.plot(x_coords, y_coords, marker='x', color='red', markersize=6, linestyle='none', label='_nolegend_', zorder=6); current_target_lines.extend([line_target] + marker_target); plotted_target_label = True
                if res_optim_grid and 'l' in res_optim_grid and res_optim_grid['l'].size > 0:
                    res_l_optim = np.asarray(res_optim_grid['l']); indices_optim = np.where((res_l_optim >= l_min) & (res_l_optim <= l_max))[0]
                    if indices_optim.size > 0:
                        optim_lambdas = res_l_optim[indices_optim]
                        if abs(l_max - l_min) < 1e-9: optim_target_t = np.full_like(optim_lambdas, t_min)
                        else: slope = (t_max - t_min) / (l_max - l_min); optim_target_t = t_min + slope * (optim_lambdas - l_min)
                        optim_markers = ax_spec.plot(optim_lambdas, optim_target_t, marker='.', color='darkred', linestyle='none', markersize=4, alpha=0.7, label='_nolegend_', zorder=6); current_target_lines.extend(optim_markers)
        ax_spec.set_xlabel("Wavelength (nm)"); ax_spec.set_ylabel('Transmittance'); ax_spec.grid(True, linestyle=':', linewidth=0.6); ax_spec.set_ylim(-0.05, 1.05)
        if overall_l_min_plot is not None: pad = (overall_l_max_plot - overall_l_min_plot) * 0.02; ax_spec.set_xlim(overall_l_min_plot - pad, overall_l_max_plot + pad)
        if plotted_target_label or (line_ts is not None): ax_spec.legend(fontsize='small')
        if mse is not None and not np.isnan(mse) and mse != -1: mse_text = f"MSE: {mse:.3e}"
        elif mse == -1: mse_text = "MSE: N/A"
        elif mse is None and active_targets_for_plot: mse_text = "MSE: Error"
        elif mse is None: mse_text = "MSE: No Target"
        else: mse_text = "MSE: No Pts"
        ax_spec.text(0.98, 0.98, mse_text, transform=ax_spec.transAxes, ha='right', va='top', fontsize='small', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
    else: ax_spec.text(0.5, 0.5, "No spectral data", ha='center', va='center', transform=ax_spec.transAxes)
    ax_spec.set_title(f"Spectrum {plot_title_status}{opt_method_str}")

    num_layers = len(current_ep) if current_ep is not None else 0; n_real_layers_repr = []; nSub_c_repr = complex(1.5, 0)
    if current_ep is not None:
        try:
            nSub_c_repr, _ = _get_nk_at_lambda(nSub_material, l0_repr, EXCEL_FILE_PATH)
            if material_sequence and len(material_sequence) == num_layers:
                 for mat_name in material_sequence:
                      try: nk_c, _ = _get_nk_at_lambda(mat_name, l0_repr, EXCEL_FILE_PATH); n_real_layers_repr.append(nk_c.real)
                      except Exception: n_real_layers_repr.append(np.nan)
            elif num_layers > 0:
                nH_c_repr, _ = _get_nk_at_lambda(nH_material, l0_repr, EXCEL_FILE_PATH); nL_c_repr, _ = _get_nk_at_lambda(nL_material, l0_repr, EXCEL_FILE_PATH)
                for i in range(num_layers): n_real_layers_repr.append(nH_c_repr.real if i % 2 == 0 else nL_c_repr.real)
        except Exception as e: add_log_message(f"Warn: Idx plot n({l0_repr}nm) error: {e}"); nSub_c_repr = complex(1.5, 0); n_real_layers_repr = [np.nan] * num_layers
        ep_cumulative = np.cumsum(current_ep) if num_layers > 0 else np.array([]); nSub_r_repr = nSub_c_repr.real; total_thickness = ep_cumulative[-1] if num_layers > 0 else 0
        margin = max(50, 0.1 * total_thickness) if total_thickness > 0 else 50; x_coords_plot = [-margin]; y_coords_plot = [nSub_r_repr]
        if num_layers > 0:
            x_coords_plot.append(0); y_coords_plot.append(nSub_r_repr)
            for i in range(num_layers): layer_start = ep_cumulative[i-1] if i > 0 else 0; layer_end = ep_cumulative[i]; layer_n_real = n_real_layers_repr[i] if i < len(n_real_layers_repr) else np.nan; x_coords_plot.extend([layer_start, layer_end]); y_coords_plot.extend([layer_n_real, layer_n_real])
            last_layer_end = ep_cumulative[-1]; x_coords_plot.extend([last_layer_end, last_layer_end + margin]); y_coords_plot.extend([1.0, 1.0])
        else: x_coords_plot.extend([0, 0, margin]); y_coords_plot.extend([nSub_r_repr, 1.0, 1.0])
        valid_indices_mask = ~np.isnan(y_coords_plot); x_coords_plot_valid = np.array(x_coords_plot)[valid_indices_mask]; y_coords_plot_valid = np.array(y_coords_plot)[valid_indices_mask]
        if len(x_coords_plot_valid) > 1 : ax_idx.plot(x_coords_plot_valid, y_coords_plot_valid, drawstyle='steps-post', label=f'n\'(λ={l0_repr:.0f}nm)', color='purple', linewidth=1.5)
        ax_idx.set_xlabel('Depth (nm)'); ax_idx.set_ylabel("n'"); ax_idx.set_title(f"Index Profile (λ={l0_repr:.0f}nm)"); ax_idx.grid(True, linestyle=':', linewidth=0.6); ax_idx.set_xlim(x_coords_plot[0], x_coords_plot[-1])
        min_n_list = [1.0, nSub_r_repr] + [n for n in n_real_layers_repr if not np.isnan(n)]; max_n_list = [1.0, nSub_r_repr] + [n for n in n_real_layers_repr if not np.isnan(n)]
        min_n = min(min_n_list) if min_n_list else 0.9; max_n = max(max_n_list) if max_n_list else 2.5; ax_idx.set_ylim(bottom=min_n - 0.1, top=max_n + 0.1)
        offset = (max_n - min_n) * 0.05 + 0.02; common_text_opts = {'ha':'center', 'va':'bottom', 'fontsize':'small', 'bbox':dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none')}
        n_sub_label = f"{nSub_c_repr.real:.3f}"; air_x_pos = (total_thickness + margin / 2) if num_layers > 0 else margin / 2
        if abs(nSub_c_repr.imag) > 1e-6: n_sub_label += f"{nSub_c_repr.imag:+.3f}j"; ax_idx.text(-margin / 2, nSub_r_repr + offset, f"SUB\nn={n_sub_label}", **common_text_opts); ax_idx.text(air_x_pos, 1.0 + offset, "AIR\nn=1.0", **common_text_opts)
        if len(ax_idx.get_legend_handles_labels()[1]) > 0: ax_idx.legend(fontsize='x-small', loc='lower right')
    else: ax_idx.text(0.5, 0.5, "No layer data", ha='center', va='center', transform=ax_idx.transAxes); ax_idx.set_title(f"Index Profile (at λ={l0_repr:.0f}nm)")

    if current_ep is not None and num_layers > 0:
        indices_complex_repr = []; layer_labels = []
        try:
            if material_sequence and len(material_sequence) == num_layers:
                 for i, mat_name in enumerate(material_sequence): nk_c, _ = _get_nk_at_lambda(mat_name, l0_repr, EXCEL_FILE_PATH); indices_complex_repr.append(nk_c); layer_labels.append(f"L{i+1} ({mat_name[:6]})")
            else:
                 nH_c_repr, _ = _get_nk_at_lambda(nH_material, l0_repr, EXCEL_FILE_PATH); nL_c_repr, _ = _get_nk_at_lambda(nL_material, l0_repr, EXCEL_FILE_PATH)
                 for i in range(num_layers): is_H = (i % 2 == 0); indices_complex_repr.append(nH_c_repr if is_H else nL_c_repr); layer_labels.append(f"L{i+1} ({'H' if is_H else 'L'})")
        except Exception: indices_complex_repr = [complex(np.nan, np.nan)] * num_layers; layer_labels = [f"L{i+1}" for i in range(num_layers)]
        colors = ['lightblue' if i % 2 == 0 else 'lightcoral' for i in range(num_layers)]; bar_pos = np.arange(num_layers)
        bars = ax_stack.barh(bar_pos, current_ep, align='center', color=colors, edgecolor='grey', height=0.8); yticks_labels_stack = []
        for i, n_comp_repr in enumerate(indices_complex_repr):
            base_label = layer_labels[i] if i < len(layer_labels) else f"L{i+1}"; n_str = f"n≈{n_comp_repr.real:.3f}" if not np.isnan(n_comp_repr.real) else "n=N/A"
            k_val = n_comp_repr.imag;
            if not np.isnan(k_val) and abs(k_val) > 1e-6: n_str += f"{k_val:+.3f}j"
            yticks_labels_stack.append(f"{base_label} {n_str}")
        ax_stack.set_yticks(bar_pos); ax_stack.set_yticklabels(yticks_labels_stack, fontsize='x-small'); ax_stack.invert_yaxis()
        max_ep = max(current_ep) if current_ep.size > 0 else 1.0; fontsize_bar = max(6, 9 - num_layers // 15)
        for i, bar in enumerate(bars):
             e_val = bar.get_width(); ha_pos = 'left' if e_val < max_ep * 0.3 else 'right'; x_text_pos = e_val * 1.05 if ha_pos == 'left' else e_val * 0.95; text_color = 'black' if ha_pos == 'left' else 'white'
             ax_stack.text(x_text_pos, bar.get_y() + bar.get_height()/2., f"{e_val:.2f} nm", va='center', ha=ha_pos, color=text_color, fontsize=fontsize_bar, weight='bold')
    else: ax_stack.text(0.5, 0.5, "No layers", ha='center', va='center'); ax_stack.set_yticks([]); ax_stack.set_xticks([])
    ax_stack.set_xlabel('Thickness (nm)'); ax_stack.set_title(f'Stack ({num_layers} layers)')
    if num_layers > 0: ax_stack.set_ylim(bottom=num_layers - 0.5, top=-0.5)
    fig.suptitle(f"Results: {plot_title_status}{opt_method_str} (vs λ={l0_repr:.0f}nm profile)", fontsize=14); plt.tight_layout(pad=1.0, rect=[0, 0.03, 1, 0.95])

# --- Initial Session State Setup ---
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    st.session_state.nominal_qwot_str = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    st.session_state.initial_layer_number = 20
    st.session_state.l0_qwot = 500.0; st.session_state.lambda_step = 10.0
    st.session_state.max_iter = 1000; st.session_state.max_fun = 1000
    st.session_state.auto_thin_threshold = 1.0
    default_targets_data = [{'id': 0, 'enabled': True, 'min': 400., 'max': 500., 'target_min': 1., 'target_max': 1.}, {'id': 1, 'enabled': True, 'min': 500., 'max': 600., 'target_min': 1., 'target_max': 0.2}, {'id': 2, 'enabled': True, 'min': 600., 'max': 700., 'target_min': 0.2, 'target_max': 0.2}, {'id': 3, 'enabled': False, 'min': 700., 'max': 800., 'target_min': 0.2, 'target_max': 0.8}, {'id': 4, 'enabled': False, 'min': 800., 'max': 900., 'target_min': 0.8, 'target_max': 0.8}]
    num_target_rows = 5; st.session_state.num_target_rows = num_target_rows
    for i in range(num_target_rows):
        target_id_prefix = f"target_{i}"; data = default_targets_data[i] if i < len(default_targets_data) else {'enabled': False, 'min': 0., 'max': 0., 'target_min': 0., 'target_max': 0.}
        st.session_state[f"{target_id_prefix}_enabled"] = data['enabled']; st.session_state[f"{target_id_prefix}_min"] = data['min']; st.session_state[f"{target_id_prefix}_max"] = data['max']; st.session_state[f"{target_id_prefix}_target_min"] = data['target_min']; st.session_state[f"{target_id_prefix}_target_max"] = data['target_max']
    st.session_state.log_messages = []; st.session_state.current_optimized_ep = None; st.session_state.current_material_sequence = None; st.session_state.optimization_ran_since_nominal_change = False; st.session_state.ep_history = deque(maxlen=5)
    st.session_state.last_calc_results = None; st.session_state.last_calc_mse = None; st.session_state.last_calc_ep = None; st.session_state.last_calc_is_optimized = False; st.session_state.last_calc_method_name = ""; st.session_state.last_calc_l0_plot = 500.0; st.session_state.status_message = "Status: Ready"; st.session_state.plot_placeholder = None
    add_log_message("Streamlit App Initialized.")

# --- Main UI Layout ---
st.title("Thin Film Optimizer (JAX - Streamlit)")

available_materials = get_available_material_names(EXCEL_FILE_PATH)
available_substrates = ["Constant", "Fused Silica", "BK7", "D263"] + [m for m in available_materials if m != "Constant"]
st.sidebar.header("Materials")
# Initialize selections if needed
if 'selected_H_material' not in st.session_state: st.session_state.selected_H_material = "Constant"
if 'selected_L_material' not in st.session_state: st.session_state.selected_L_material = "Constant"
if 'selected_Sub_material' not in st.session_state: st.session_state.selected_Sub_material = "Constant"
if 'nH_r' not in st.session_state: st.session_state.nH_r = 2.35
if 'nH_i' not in st.session_state: st.session_state.nH_i = 0.0
if 'nL_r' not in st.session_state: st.session_state.nL_r = 1.46
if 'nL_i' not in st.session_state: st.session_state.nL_i = 0.0
if 'nSub_r' not in st.session_state: st.session_state.nSub_r = 1.52
# Display Material Selectors
st.session_state.selected_H_material = st.sidebar.selectbox("H Material", options=available_materials, key="select_H")
col_h1, col_h2 = st.sidebar.columns(2)
with col_h1: st.session_state.nH_r = st.number_input("n'", key="nH_r", format="%.4f", step=0.01, disabled=(st.session_state.selected_H_material != "Constant"))
with col_h2: st.session_state.nH_i = st.number_input("k", key="nH_i", format="%.4f", min_value=0.0, step=0.001, disabled=(st.session_state.selected_H_material != "Constant"))
st.session_state.selected_L_material = st.sidebar.selectbox("L Material", options=available_materials, key="select_L")
col_l1, col_l2 = st.sidebar.columns(2)
with col_l1: st.session_state.nL_r = st.number_input("n'", key="nL_r", format="%.4f", step=0.01, disabled=(st.session_state.selected_L_material != "Constant"))
with col_l2: st.session_state.nL_i = st.number_input("k", key="nL_i", format="%.4f", min_value=0.0, step=0.001, disabled=(st.session_state.selected_L_material != "Constant"))
st.session_state.selected_Sub_material = st.sidebar.selectbox("Substrate Material", options=available_substrates, key="select_Sub")
st.session_state.nSub_r = st.sidebar.number_input("n' (k=0 assumed)", key="nSub_r", format="%.4f", step=0.01, disabled=(st.session_state.selected_Sub_material != "Constant"))

st.sidebar.divider()
with st.sidebar.expander("Indices n'(λ) des Matériaux Sélectionnés", expanded=False):
    fig_material_indices = display_material_index_plot()
    if fig_material_indices: st.pyplot(fig_material_indices); plt.close(fig_material_indices)

# --- Needle Iteration Runner ---
# (Helper for Auto Mode)
def _run_needle_iterations(ep_start: np.ndarray, num_needles: int,
                           validated_inputs: Dict, active_targets: List[Dict],
                           min_thickness_phys: float, l_vec_optim_np: np.ndarray,
                           scan_step_nm: float, base_needle_thickness_nm: float,
                           cost_function_jax: Callable, # Pass the JAX cost function
                           log_prefix: str = "") -> Tuple[np.ndarray, float, List[str], int, int, int]:
    logs = []
    best_ep_overall = np.asarray(ep_start).copy()
    best_mse_overall = np.inf
    total_nit_needles = 0; total_nfev_needles = 0; successful_reopts_count = 0
    nH_material = validated_inputs['nH_material']; nL_material = validated_inputs['nL_material']
    nSub_material = validated_inputs['nSub_material']
    l0_repr = validated_inputs.get('l0', 500.0)

    try:
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        nH_arr, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH); logs.extend(logs_h)
        nL_arr, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH); logs.extend(logs_l)
        nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH); logs.extend(logs_sub)
        active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
        static_args_cost_fn = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, jnp.float64(min_thickness_phys))
        cost_fn_compiled = jax.jit(cost_function_jax)
        initial_cost_jax = cost_fn_compiled(jnp.asarray(best_ep_overall), *static_args_cost_fn)
        best_mse_overall = float(np.array(initial_cost_jax))
        if not np.isfinite(best_mse_overall): raise ValueError("Initial MSE for needle iterations not finite.")
        logs.append(f"{log_prefix}Starting needle iterations. Initial MSE: {best_mse_overall:.6e}")
    except Exception as e:
        logs.append(f"{log_prefix}ERROR calculating initial MSE for needle iterations: {e}")
        return ep_start, np.inf, logs, 0, 0, 0

    for i in range(num_needles):
        logs.append(f"{log_prefix}Needle Iteration {i + 1}/{num_needles}")
        current_ep_iter = best_ep_overall.copy()
        if len(current_ep_iter) == 0: logs.append(f"{log_prefix}Empty structure, stopping."); break

        ep_after_scan, cost_after_scan, scan_logs, inserted_idx = _perform_needle_insertion_scan(
            current_ep_iter, nH_material, nL_material, nSub_material, l_vec_optim_np,
            active_targets, cost_function_jax, min_thickness_phys,
            base_needle_thickness_nm, scan_step_nm, l0_repr, log_prefix=f"{log_prefix}Scan {i+1} ")
        logs.extend(scan_logs)

        if ep_after_scan is None: logs.append(f"{log_prefix}Needle scan {i + 1} found no improvement. Stopping iterations."); break
        logs.append(f"{log_prefix}Needle scan {i + 1} found potential improvement. Re-optimizing...")

        l_min_val_iter, l_max_val_iter = get_lambda_range_from_targets(active_targets)
        if l_min_val_iter is None or l_max_val_iter is None: raise ValueError("Cannot determine lambda range for needle re-opt.")
        validated_inputs_iter = validated_inputs.copy()
        validated_inputs_iter['l_range_deb'] = l_min_val_iter
        validated_inputs_iter['l_range_fin'] = l_max_val_iter

        ep_after_reopt, optim_success, final_cost_reopt, optim_logs, _, nit_reopt, nfev_reopt = \
            _run_core_optimization(ep_after_scan, validated_inputs_iter, active_targets,
                                   min_thickness_phys, log_prefix=f"{log_prefix}Re-Opt {i+1} ")
        logs.extend(optim_logs)

        if not optim_success: logs.append(f"{log_prefix}Re-optimization after needle scan {i + 1} failed. Stopping."); break

        logs.append(f"{log_prefix}Re-optimization {i + 1} successful. New MSE: {final_cost_reopt:.6e}.")
        total_nit_needles += nit_reopt; total_nfev_needles += nfev_reopt; successful_reopts_count += 1

        if final_cost_reopt < best_mse_overall - MSE_IMPROVEMENT_TOLERANCE:
            logs.append(f"{log_prefix}MSE improved from {best_mse_overall:.6e}. Updating best result.")
            best_ep_overall = ep_after_reopt.copy(); best_mse_overall = final_cost_reopt
        else:
            logs.append(f"{log_prefix}New MSE ({final_cost_reopt:.6e}) not improved vs best ({best_mse_overall:.6e}). Stopping.")
            best_ep_overall = ep_after_reopt.copy(); best_mse_so_far = final_cost_reopt # Corrected variable name
            break

    logs.append(f"{log_prefix}End needle iterations. Final best MSE: {best_mse_overall:.6e}")
    logs.append(f"{log_prefix}Total Iter/Eval during {successful_reopts_count} re-opts: {total_nit_needles}/{total_nfev_needles}")
    return best_ep_overall, best_mse_overall, logs, total_nit_needles, total_nfev_needles, successful_reopts_count


@jax.jit
def calculate_M_for_thickness(thickness: jnp.ndarray, n_complex_layer: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    eta = n_complex_layer; safe_l_val = jnp.maximum(l_val, 1e-9); safe_eta = jnp.where(jnp.abs(eta) < 1e-12, 1e-12 + 0j, eta)
    phi = (2 * jnp.pi / safe_l_val) * (n_complex_layer * thickness); cos_phi = jnp.cos(phi); sin_phi = jnp.sin(phi)
    m00 = jnp.where(thickness > 1e-12, cos_phi, 1.0); m01 = jnp.where(thickness > 1e-12, (1j / safe_eta) * sin_phi, 0.0)
    m10 = jnp.where(thickness > 1e-12, 1j * eta * sin_phi, 0.0); m11 = jnp.where(thickness > 1e-12, cos_phi, 1.0)
    return jnp.array([[m00, m01], [m10, m11]], dtype=jnp.complex128)

calculate_M_batch_for_thickness = vmap(calculate_M_for_thickness, in_axes=(None, None, 0))

@jax.jit
def get_layer_matrices_qwot(layer_idx: int, initial_layer_number: int, l0: float,
                            nH_c_l0: jnp.ndarray, nL_c_l0: jnp.ndarray,
                            l_vec: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    predicate_is_H = (layer_idx % 2 == 0); n_real_l0 = jax.lax.select(predicate_is_H, nH_c_l0.real, nL_c_l0.real); n_complex_for_matrix = jax.lax.select(predicate_is_H, nH_c_l0, nL_c_l0)
    denom = 4.0 * jnp.maximum(n_real_l0, 1e-9); safe_l0 = jnp.maximum(l0, 1e-9)
    ep1_calc = 1.0 * safe_l0 / denom; ep2_calc = 2.0 * safe_l0 / denom
    ep1 = jnp.where(n_real_l0 > 1e-9, ep1_calc, 0.0); ep2 = jnp.where(n_real_l0 > 1e-9, ep2_calc, 0.0)
    ep1 = jnp.maximum(ep1, MIN_THICKNESS_PHYS_NM * (ep1 > 1e-12)); ep2 = jnp.maximum(ep2, MIN_THICKNESS_PHYS_NM * (ep2 > 1e-12))
    M_1qwot_batch = calculate_M_batch_for_thickness(ep1, n_complex_for_matrix, l_vec)
    M_2qwot_batch = calculate_M_batch_for_thickness(ep2, n_complex_for_matrix, l_vec)
    return M_1qwot_batch, M_2qwot_batch

@jax.jit
def compute_half_product(multiplier_indices: jnp.ndarray, layer_matrices_half: jnp.ndarray) -> jnp.ndarray:
    N_half = layer_matrices_half.shape[0]; L = layer_matrices_half.shape[2]; init_prod = jnp.tile(jnp.eye(2, dtype=jnp.complex128), (L, 1, 1))
    def multiply_step(carry_prod: jnp.ndarray, layer_idx: int) -> Tuple[jnp.ndarray, None]:
        multiplier_idx = multiplier_indices[layer_idx]; M_k = layer_matrices_half[layer_idx, multiplier_idx, :, :, :]; new_prod = vmap(jnp.matmul)(M_k, carry_prod)
        return new_prod, None
    final_prod, _ = jax.lax.scan(multiply_step, init_prod, jnp.arange(N_half))
    return final_prod

@jax.jit
def get_T_from_batch_matrix(M_batch: jnp.ndarray, nSub_arr: jnp.ndarray) -> jnp.ndarray:
    etainc = 1.0 + 0j; etasub_batch = nSub_arr; m00 = M_batch[:, 0, 0]; m01 = M_batch[:, 0, 1]; m10 = M_batch[:, 1, 0]; m11 = M_batch[:, 1, 1]
    rs_den = (etainc * m00 + etasub_batch * m11 + etainc * etasub_batch * m01 + m10); rs_den_abs = jnp.abs(rs_den)
    safe_den = jnp.where(rs_den_abs < 1e-12, 1e-12 + 0j, rs_den); ts = (2.0 * etainc) / safe_den
    real_etasub_batch = jnp.real(etasub_batch); real_etainc = 1.0; Ts_complex = (real_etasub_batch / real_etainc) * (ts * jnp.conj(ts))
    Ts = jnp.real(Ts_complex); return jnp.where(rs_den_abs < 1e-12, 0.0, jnp.nan_to_num(Ts, nan=0.0))

@jax.jit
def calculate_mse_basic_jax(Ts: jnp.ndarray, l_vec: jnp.ndarray, targets_tuple: Tuple) -> jnp.ndarray:
    total_squared_error = 0.0; total_points_in_targets = 0
    for i in range(len(targets_tuple)):
        l_min, l_max, t_min, t_max = targets_tuple[i]; target_mask = (l_vec >= l_min) & (l_vec <= l_max)
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t = t_min + slope * (l_vec - l_min); squared_errors = (Ts - interpolated_target_t)**2
        masked_sq_error = jnp.where(target_mask, squared_errors, 0.0); total_squared_error += jnp.sum(masked_sq_error); total_points_in_targets += jnp.sum(target_mask)
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, jnp.inf)
    return jnp.nan_to_num(mse, nan=jnp.inf)

@jax.jit
def combine_and_calc_mse(prod1: jnp.ndarray, prod2: jnp.ndarray, nSub_arr_in: jnp.ndarray, l_vec_in: jnp.ndarray, targets_tuple_in: Tuple) -> jnp.ndarray:
    M_total = vmap(jnp.matmul)(prod2, prod1); Ts = get_T_from_batch_matrix(M_total, nSub_arr_in); mse = calculate_mse_basic_jax(Ts, l_vec_in, targets_tuple_in)
    return mse

def _execute_split_stack_scan(current_l0: float, initial_layer_number: int,
                              nH_c_l0: jnp.ndarray, nL_c_l0: jnp.ndarray, nSub_arr_scan: jnp.ndarray,
                              l_vec_eval_sparse_jax: jnp.ndarray, active_targets_tuple: Tuple
                              ) -> Tuple[float, Optional[np.ndarray], List[str]]:
    logs = []; num_combinations = 2**initial_layer_number; logs.append(f"  [Scan l0={current_l0:.2f}] Testing {num_combinations:,} QWOT combos...")
    precompute_start_time = time.time(); layer_matrices_list = []
    try:
        for i in range(initial_layer_number): m1, m2 = get_layer_matrices_qwot(i, initial_layer_number, current_l0, nH_c_l0, nL_c_l0, l_vec_eval_sparse_jax); layer_matrices_list.append(jnp.stack([m1, m2], axis=0))
        all_layer_matrices = jnp.stack(layer_matrices_list, axis=0); all_layer_matrices.block_until_ready()
    except Exception as e_mat: logs.append(f"  ERROR Precomputing Matrices: {e_mat}"); return np.inf, None, logs
    logs.append(f"    Matrix precomp {time.time() - precompute_start_time:.3f}s.")
    N1 = initial_layer_number // 2; N2 = initial_layer_number - N1; num_comb1 = 2**N1; num_comb2 = 2**N2
    indices1 = jnp.arange(num_comb1); indices2 = jnp.arange(num_comb2); powers1 = 2**jnp.arange(N1); powers2 = 2**jnp.arange(N2)
    multiplier_indices1 = jnp.not_equal(indices1[:, None] & powers1, 0).astype(jnp.int32); multiplier_indices2 = jnp.not_equal(indices2[:, None] & powers2, 0).astype(jnp.int32)
    matrices_half1 = all_layer_matrices[:N1]; matrices_half2 = all_layer_matrices[N1:]
    half1_start_time = time.time(); compute_half_product_jit = jax.jit(compute_half_product); partial_products1 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices1, matrices_half1); partial_products1.block_until_ready()
    logs.append(f"    Partial products 1/2 {time.time() - half1_start_time:.3f}s.")
    half2_start_time = time.time(); partial_products2 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices2, matrices_half2); partial_products2.block_until_ready()
    logs.append(f"    Partial products 2/2 {time.time() - half2_start_time:.3f}s.")
    combine_start_time = time.time(); combine_and_calc_mse_jit = jax.jit(combine_and_calc_mse); vmap_inner = vmap(combine_and_calc_mse_jit, in_axes=(None, 0, None, None, None)); vmap_outer = vmap(vmap_inner, in_axes=(0, None, None, None, None))
    all_mses_nested = vmap_outer(partial_products1, partial_products2, nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple); all_mses_nested.block_until_ready()
    logs.append(f"    Combination & MSE {time.time() - combine_start_time:.3f}s.")
    all_mses_flat = all_mses_nested.reshape(-1); best_idx_flat = jnp.argmin(all_mses_flat); current_best_mse = float(all_mses_flat[best_idx_flat])
    if not np.isfinite(current_best_mse): logs.append(f"    Warning: No valid combo found."); return np.inf, None, logs
    best_idx_half1, best_idx_half2 = jnp.unravel_index(best_idx_flat, (num_comb1, num_comb2))
    best_indices_h1 = multiplier_indices1[best_idx_half1]; best_indices_h2 = multiplier_indices2[best_idx_half2]
    best_multipliers_h1 = 1.0 + best_indices_h1.astype(jnp.float64); best_multipliers_h2 = 1.0 + best_indices_h2.astype(jnp.float64)
    current_best_multipliers = jnp.concatenate([best_multipliers_h1, best_multipliers_h2])
    logs.append(f"    Best Scan MSE l0={current_l0:.2f}: {current_best_mse:.6e}"); return current_best_mse, np.array(current_best_multipliers), logs

def draw_plots(res: Optional[Dict], current_ep: Optional[np.ndarray], l0_repr: float,
               nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
               active_targets_for_plot: List[Dict], mse: Optional[float],
               is_optimized: bool = False, method_name: str = "",
               res_optim_grid: Optional[Dict] = None, material_sequence: Optional[List[str]] = None,
               fig_placeholder=None):
    if fig_placeholder is None: st.warning("Plotting requires placeholder."); return
    fig = fig_placeholder['fig']; axes = fig_placeholder['axes']; fig.clear()
    if not isinstance(axes, (list, np.ndarray)) or len(axes) != 3: axes = fig.subplots(1, 3); fig_placeholder['axes'] = axes
    else:
         for ax in axes: ax.cla()
    ax_spec, ax_idx, ax_stack = axes[0], axes[1], axes[2]
    line_ts = None; overall_l_min_plot, overall_l_max_plot = None, None; plot_title_status = "Optimized" if is_optimized else "Nominal"; opt_method_str = f" ({method_name})" if method_name else ""
    if res is not None and isinstance(res, dict) and 'l' in res and 'Ts' in res and res['l'] is not None and len(res['l']) > 0:
        res_l_plot = np.asarray(res['l']); res_ts_plot = np.asarray(res['Ts']); overall_l_min_plot, overall_l_max_plot = res_l_plot.min(), res_l_plot.max()
        line_ts, = ax_spec.plot(res_l_plot, res_ts_plot, label='Transmittance', linestyle='-', color='blue', linewidth=1.5); plotted_target_label = False; current_target_lines = []
        if active_targets_for_plot:
            for i, target in enumerate(active_targets_for_plot):
                l_min, l_max = target['min'], target['max']; t_min, t_max = target['target_min'], target['target_max']; x_coords, y_coords = [l_min, l_max], [t_min, t_max]
                label = f'Target {i+1}' if not plotted_target_label else "_nolegend_"; line_target, = ax_spec.plot(x_coords, y_coords, 'r--', linewidth=1.5, alpha=0.8, label=label, zorder=5)
                marker_target = ax_spec.plot(x_coords, y_coords, marker='x', color='red', markersize=6, linestyle='none', label='_nolegend_', zorder=6); current_target_lines.extend([line_target] + marker_target); plotted_target_label = True
                if res_optim_grid and 'l' in res_optim_grid and res_optim_grid['l'].size > 0:
                    res_l_optim = np.asarray(res_optim_grid['l']); indices_optim = np.where((res_l_optim >= l_min) & (res_l_optim <= l_max))[0]
                    if indices_optim.size > 0:
                        optim_lambdas = res_l_optim[indices_optim]
                        if abs(l_max - l_min) < 1e-9: optim_target_t = np.full_like(optim_lambdas, t_min)
                        else: slope = (t_max - t_min) / (l_max - l_min); optim_target_t = t_min + slope * (optim_lambdas - l_min)
                        optim_markers = ax_spec.plot(optim_lambdas, optim_target_t, marker='.', color='darkred', linestyle='none', markersize=4, alpha=0.7, label='_nolegend_', zorder=6); current_target_lines.extend(optim_markers)
        ax_spec.set_xlabel("Wavelength (nm)"); ax_spec.set_ylabel('Transmittance'); ax_spec.grid(True, linestyle=':', linewidth=0.6); ax_spec.set_ylim(-0.05, 1.05)
        if overall_l_min_plot is not None: pad = (overall_l_max_plot - overall_l_min_plot) * 0.02; ax_spec.set_xlim(overall_l_min_plot - pad, overall_l_max_plot + pad)
        if plotted_target_label or (line_ts is not None): ax_spec.legend(fontsize='small')
        if mse is not None and not np.isnan(mse) and mse != -1: mse_text = f"MSE: {mse:.3e}"
        elif mse == -1: mse_text = "MSE: N/A"; elif mse is None and active_targets_for_plot: mse_text = "MSE: Error"; elif mse is None: mse_text = "MSE: No Target"; else: mse_text = "MSE: No Pts"
        ax_spec.text(0.98, 0.98, mse_text, transform=ax_spec.transAxes, ha='right', va='top', fontsize='small', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
    else: ax_spec.text(0.5, 0.5, "No spectral data", ha='center', va='center', transform=ax_spec.transAxes)
    ax_spec.set_title(f"Spectrum {plot_title_status}{opt_method_str}")
    num_layers = len(current_ep) if current_ep is not None else 0; n_real_layers_repr = []; nSub_c_repr = complex(1.5, 0)
    if current_ep is not None:
        try:
            nSub_c_repr, _ = _get_nk_at_lambda(nSub_material, l0_repr, EXCEL_FILE_PATH)
            if material_sequence and len(material_sequence) == num_layers:
                 for mat_name in material_sequence:
                      try: nk_c, _ = _get_nk_at_lambda(mat_name, l0_repr, EXCEL_FILE_PATH); n_real_layers_repr.append(nk_c.real)
                      except Exception: n_real_layers_repr.append(np.nan)
            elif num_layers > 0:
                nH_c_repr, _ = _get_nk_at_lambda(nH_material, l0_repr, EXCEL_FILE_PATH); nL_c_repr, _ = _get_nk_at_lambda(nL_material, l0_repr, EXCEL_FILE_PATH)
                for i in range(num_layers): n_real_layers_repr.append(nH_c_repr.real if i % 2 == 0 else nL_c_repr.real)
        except Exception as e: add_log_message(f"Warn: Idx plot n({l0_repr}nm) error: {e}"); nSub_c_repr = complex(1.5, 0); n_real_layers_repr = [np.nan] * num_layers
        ep_cumulative = np.cumsum(current_ep) if num_layers > 0 else np.array([]); nSub_r_repr = nSub_c_repr.real; total_thickness = ep_cumulative[-1] if num_layers > 0 else 0
        margin = max(50, 0.1 * total_thickness) if total_thickness > 0 else 50; x_coords_plot = [-margin]; y_coords_plot = [nSub_r_repr]
        if num_layers > 0:
            x_coords_plot.append(0); y_coords_plot.append(nSub_r_repr)
            for i in range(num_layers): layer_start = ep_cumulative[i-1] if i > 0 else 0; layer_end = ep_cumulative[i]; layer_n_real = n_real_layers_repr[i] if i < len(n_real_layers_repr) else np.nan; x_coords_plot.extend([layer_start, layer_end]); y_coords_plot.extend([layer_n_real, layer_n_real])
            last_layer_end = ep_cumulative[-1]; x_coords_plot.extend([last_layer_end, last_layer_end + margin]); y_coords_plot.extend([1.0, 1.0])
        else: x_coords_plot.extend([0, 0, margin]); y_coords_plot.extend([nSub_r_repr, 1.0, 1.0])
        valid_indices_mask = ~np.isnan(y_coords_plot); x_coords_plot_valid = np.array(x_coords_plot)[valid_indices_mask]; y_coords_plot_valid = np.array(y_coords_plot)[valid_indices_mask]
        if len(x_coords_plot_valid) > 1 : ax_idx.plot(x_coords_plot_valid, y_coords_plot_valid, drawstyle='steps-post', label=f'n\'(λ={l0_repr:.0f}nm)', color='purple', linewidth=1.5)
        ax_idx.set_xlabel('Depth (nm)'); ax_idx.set_ylabel("n'"); ax_idx.set_title(f"Index Profile (λ={l0_repr:.0f}nm)"); ax_idx.grid(True, linestyle=':', linewidth=0.6); ax_idx.set_xlim(x_coords_plot[0], x_coords_plot[-1])
        min_n_list = [1.0, nSub_r_repr] + [n for n in n_real_layers_repr if not np.isnan(n)]; max_n_list = [1.0, nSub_r_repr] + [n for n in n_real_layers_repr if not np.isnan(n)]; min_n = min(min_n_list) if min_n_list else 0.9; max_n = max(max_n_list) if max_n_list else 2.5; ax_idx.set_ylim(bottom=min_n - 0.1, top=max_n + 0.1)
        offset = (max_n - min_n) * 0.05 + 0.02; common_text_opts = {'ha':'center', 'va':'bottom', 'fontsize':'small', 'bbox':dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none')}
        n_sub_label = f"{nSub_c_repr.real:.3f}"; air_x_pos = (total_thickness + margin / 2) if num_layers > 0 else margin / 2
        if abs(nSub_c_repr.imag) > 1e-6: n_sub_label += f"{nSub_c_repr.imag:+.3f}j"; ax_idx.text(-margin / 2, nSub_r_repr + offset, f"SUB\nn={n_sub_label}", **common_text_opts); ax_idx.text(air_x_pos, 1.0 + offset, "AIR\nn=1.0", **common_text_opts)
        if len(ax_idx.get_legend_handles_labels()[1]) > 0: ax_idx.legend(fontsize='x-small', loc='lower right')
    else: ax_idx.text(0.5, 0.5, "No layer data", ha='center', va='center', transform=ax_idx.transAxes); ax_idx.set_title(f"Index Profile (at λ={l0_repr:.0f}nm)")
    if current_ep is not None and num_layers > 0:
        indices_complex_repr = []; layer_labels = []
        try:
            if material_sequence and len(material_sequence) == num_layers:
                 for i, mat_name in enumerate(material_sequence): nk_c, _ = _get_nk_at_lambda(mat_name, l0_repr, EXCEL_FILE_PATH); indices_complex_repr.append(nk_c); layer_labels.append(f"L{i+1} ({mat_name[:6]})")
            else:
                 nH_c_repr, _ = _get_nk_at_lambda(nH_material, l0_repr, EXCEL_FILE_PATH); nL_c_repr, _ = _get_nk_at_lambda(nL_material, l0_repr, EXCEL_FILE_PATH)
                 for i in range(num_layers): is_H = (i % 2 == 0); indices_complex_repr.append(nH_c_repr if is_H else nL_c_repr); layer_labels.append(f"L{i+1} ({'H' if is_H else 'L'})")
        except Exception: indices_complex_repr = [complex(np.nan, np.nan)] * num_layers; layer_labels = [f"L{i+1}" for i in range(num_layers)]
        colors = ['lightblue' if i % 2 == 0 else 'lightcoral' for i in range(num_layers)]; bar_pos = np.arange(num_layers)
        bars = ax_stack.barh(bar_pos, current_ep, align='center', color=colors, edgecolor='grey', height=0.8); yticks_labels_stack = []
        for i, n_comp_repr in enumerate(indices_complex_repr):
            base_label = layer_labels[i] if i < len(layer_labels) else f"L{i+1}"; n_str = f"n≈{n_comp_repr.real:.3f}" if not np.isnan(n_comp_repr.real) else "n=N/A"; k_val = n_comp_repr.imag;
            if not np.isnan(k_val) and abs(k_val) > 1e-6: n_str += f"{k_val:+.3f}j"; yticks_labels_stack.append(f"{base_label} {n_str}")
        ax_stack.set_yticks(bar_pos); ax_stack.set_yticklabels(yticks_labels_stack, fontsize='x-small'); ax_stack.invert_yaxis()
        max_ep = max(current_ep) if current_ep.size > 0 else 1.0; fontsize_bar = max(6, 9 - num_layers // 15)
        for i, bar in enumerate(bars):
             e_val = bar.get_width(); ha_pos = 'left' if e_val < max_ep * 0.3 else 'right'; x_text_pos = e_val * 1.05 if ha_pos == 'left' else e_val * 0.95; text_color = 'black' if ha_pos == 'left' else 'white'; ax_stack.text(x_text_pos, bar.get_y() + bar.get_height()/2., f"{e_val:.2f} nm", va='center', ha=ha_pos, color=text_color, fontsize=fontsize_bar, weight='bold')
    else: ax_stack.text(0.5, 0.5, "No layers", ha='center', va='center'); ax_stack.set_yticks([]); ax_stack.set_xticks([]); num_layers = 0
    ax_stack.set_xlabel('Thickness (nm)'); ax_stack.set_title(f'Stack ({num_layers} layers)')
    if num_layers > 0: ax_stack.set_ylim(bottom=num_layers - 0.5, top=-0.5)
    fig.suptitle(f"Results: {plot_title_status}{opt_method_str} (vs λ={l0_repr:.0f}nm profile)", fontsize=14); plt.tight_layout(pad=1.0, rect=[0, 0.03, 1, 0.95])

def display_material_index_plot():
    materials_to_plot = {}; logs = []; at_least_one_non_constant = False
    selections = {"H": st.session_state.get('selected_H_material', 'Constant'), "L": st.session_state.get('selected_L_material', 'Constant'), "Sub": st.session_state.get('selected_Sub_material', 'Constant')}
    built_in_dispersive = ["FUSED SILICA", "BK7", "D263"]; l_plot_min, l_plot_max, l_plot_num = 350.0, 1500.0, 200; l_vec_plot_jnp = jnp.linspace(l_plot_min, l_plot_max, l_plot_num)
    for role, name in selections.items():
        if name != "Constant":
            at_least_one_non_constant = True; data_tuple = None
            try:
                if name.upper() in built_in_dispersive:
                    nk_complex_gen, _ = _get_nk_array_for_lambda_vec(name, l_vec_plot_jnp, EXCEL_FILE_PATH); n_data_gen = np.real(np.array(nk_complex_gen)); k_data_gen = np.imag(np.array(nk_complex_gen)); valid_mask = np.isfinite(n_data_gen) & np.isfinite(k_data_gen); data_tuple = (np.array(l_vec_plot_jnp)[valid_mask], n_data_gen[valid_mask], k_data_gen[valid_mask])
                else:
                    l_data, n_data, k_data = load_material_data_from_xlsx_sheet(EXCEL_FILE_PATH, name)
                    if l_data is not None and n_data is not None and k_data is not None: data_tuple = (l_data, n_data, k_data)
                if data_tuple and len(data_tuple[0]) > 0: materials_to_plot[f"{role}: {name}"] = data_tuple
                else: logs.append(f"No valid data for '{name}'.")
            except Exception as e: logs.append(f"ERROR processing '{name}': {e}")
    if not materials_to_plot: st.info("Select non-constant materials."); return None
    try:
        fig_idx, ax_idx = plt.subplots(figsize=(6, 4)); min_l_overall, max_l_overall = np.inf, -np.inf; min_n_overall, max_n_overall = np.inf, -np.inf; plotted_something = False
        for label, (l_data, n_data, k_data) in materials_to_plot.items():
            l_data_np = np.asarray(l_data); n_data_np = np.asarray(n_data); l_plot = l_data_np; n_plot = n_data_np
            if len(l_plot) > 0:
                ax_idx.plot(l_plot, n_plot, label=label, marker='.', markersize=2, linestyle='-'); min_l_overall = min(min_l_overall, np.min(l_plot)); max_l_overall = max(max_l_overall, np.max(l_plot)); min_n_overall = min(min_n_overall, np.min(n_plot)); max_n_overall = max(max_n_overall, np.max(n_plot)); plotted_something = True
        if plotted_something:
            ax_idx.set_xlabel("Wavelength (nm)"); ax_idx.set_ylabel("n'"); ax_idx.set_title("n'(λ) of Selected Materials"); ax_idx.grid(True, linestyle=':', linewidth=0.6); ax_idx.legend(fontsize='x-small')
            if np.isfinite(min_l_overall) and np.isfinite(max_l_overall) and max_l_overall > min_l_overall: pad_l = (max_l_overall - min_l_overall) * 0.05; ax_idx.set_xlim(min_l_overall - pad_l, max_l_overall + pad_l)
            if np.isfinite(min_n_overall) and np.isfinite(max_n_overall) and max_n_overall > min_n_overall: pad_n = (max_n_overall - min_n_overall) * 0.1; ax_idx.set_ylim(min_n_overall - pad_n, max_n_overall + pad_n)
            plt.tight_layout(); return fig_idx
        else: plt.close(fig_idx); st.sidebar.warning("No valid index data found."); return None
    except Exception as e_plot: add_log_message(f"ERROR index plot: {e_plot}"); st.sidebar.error(f"Index Plot Error: {e_plot}"); if 'fig_idx' in locals() and fig_idx: plt.close(fig_idx); return None

def _collect_design_data_for_save(include_optimized: bool = False) -> Dict:
    design_data = {'params': {}, 'targets': []}; params = get_validated_input_params()
    if not params: raise ValueError("Invalid input parameters."); param_keys_to_save = ['l0', 'l_step', 'maxiter', 'maxfun', 'initial_layer_number', 'auto_thin_threshold']
    for key in param_keys_to_save:
         if key in params: design_data['params'][key] = params[key]
    design_data['params']['emp_str'] = st.session_state.get('nominal_qwot_str', '')
    design_data['params']['nH_material'] = st.session_state.selected_H_material; design_data['params']['nL_material'] = st.session_state.selected_L_material; design_data['params']['nSub_material'] = st.session_state.selected_Sub_material
    if st.session_state.selected_H_material == "Constant": design_data['params']['nH_constant'] = {'real': st.session_state.nH_r, 'imag': st.session_state.nH_i}
    if st.session_state.selected_L_material == "Constant": design_data['params']['nL_constant'] = {'real': st.session_state.nL_r, 'imag': st.session_state.nL_i}
    if st.session_state.selected_Sub_material == "Constant": design_data['params']['nSub_constant'] = {'real': st.session_state.nSub_r, 'imag': 0.0}
    num_target_rows_save = st.session_state.get('num_target_rows', 5); targets_list_save = []
    for i in range(num_target_rows_save):
        target_id_prefix = f"target_{i}"; target_data = {'id': i, 'enabled': st.session_state.get(f"{target_id_prefix}_enabled", False), 'min': st.session_state.get(f"{target_id_prefix}_min", 0.0), 'max': st.session_state.get(f"{target_id_prefix}_max", 0.0), 'target_min': st.session_state.get(f"{target_id_prefix}_target_min", 0.0), 'target_max': st.session_state.get(f"{target_id_prefix}_target_max", 0.0)}; targets_list_save.append(target_data)
    design_data['targets'] = targets_list_save
    if include_optimized:
        if st.session_state.get('optimization_ran_since_nominal_change') and st.session_state.get('current_optimized_ep') is not None:
             design_data['optimized_ep'] = st.session_state.current_optimized_ep.tolist()
             if st.session_state.get('current_material_sequence'): design_data['optimized_material_sequence'] = st.session_state.current_material_sequence
             try:
                  opt_ep = st.session_state.current_optimized_ep; l0_save = params['l0']; nH_mat_save = params['nH_material']; nL_mat_save = params['nL_material']
                  qwots_save, _ = calculate_qwot_from_ep(opt_ep, l0_save, nH_mat_save, nL_mat_save, EXCEL_FILE_PATH)
                  if np.any(np.isnan(qwots_save)): opt_qwot_str_save = "QWOT N/A"; else: opt_qwot_str_save = ", ".join([f"{q:.6f}" for q in qwots_save])
                  design_data['optimized_qwot_string'] = opt_qwot_str_save
             except Exception: design_data['optimized_qwot_string'] = "Error calculating QWOT"
        else: add_log_message("Warn: Save Optimized requested, but no opt state found.")
    return design_data

# --- Initial Session State Setup ---
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    st.session_state.nominal_qwot_str = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    st.session_state.initial_layer_number = 20
    st.session_state.l0_qwot = 500.0; st.session_state.lambda_step = 10.0
    st.session_state.max_iter = 1000; st.session_state.max_fun = 1000
    st.session_state.auto_thin_threshold = 1.0
    default_targets_data = [{'id': 0, 'enabled': True, 'min': 400., 'max': 500., 'target_min': 1., 'target_max': 1.}, {'id': 1, 'enabled': True, 'min': 500., 'max': 600., 'target_min': 1., 'target_max': 0.2}, {'id': 2, 'enabled': True, 'min': 600., 'max': 700., 'target_min': 0.2, 'target_max': 0.2}, {'id': 3, 'enabled': False, 'min': 700., 'max': 800., 'target_min': 0.2, 'target_max': 0.8}, {'id': 4, 'enabled': False, 'min': 800., 'max': 900., 'target_min': 0.8, 'target_max': 0.8}]
    num_target_rows = 5; st.session_state.num_target_rows = num_target_rows
    for i in range(num_target_rows):
        target_id_prefix = f"target_{i}"; data = default_targets_data[i] if i < len(default_targets_data) else {'enabled': False, 'min': 0., 'max': 0., 'target_min': 0., 'target_max': 0.}
        st.session_state[f"{target_id_prefix}_enabled"] = data['enabled']; st.session_state[f"{target_id_prefix}_min"] = data['min']; st.session_state[f"{target_id_prefix}_max"] = data['max']; st.session_state[f"{target_id_prefix}_target_min"] = data['target_min']; st.session_state[f"{target_id_prefix}_target_max"] = data['target_max']
    st.session_state.log_messages = []; st.session_state.current_optimized_ep = None; st.session_state.current_material_sequence = None; st.session_state.optimization_ran_since_nominal_change = False; st.session_state.ep_history = deque(maxlen=5)
    st.session_state.last_calc_results = None; st.session_state.last_calc_mse = None; st.session_state.last_calc_ep = None; st.session_state.last_calc_is_optimized = False; st.session_state.last_calc_method_name = ""; st.session_state.last_calc_l0_plot = 500.0; st.session_state.status_message = "Status: Ready"; st.session_state.plot_placeholder = None
    add_log_message("Streamlit App Initialized.")

# --- Main UI Layout ---
st.title("Thin Film Optimizer (JAX - Streamlit)")

available_materials = get_available_material_names(EXCEL_FILE_PATH)
available_substrates = ["Constant", "Fused Silica", "BK7", "D263"] + [m for m in available_materials if m != "Constant"]
st.sidebar.header("Materials")
if 'selected_H_material' not in st.session_state: st.session_state.selected_H_material = "Constant"
if 'selected_L_material' not in st.session_state: st.session_state.selected_L_material = "Constant"
if 'selected_Sub_material' not in st.session_state: st.session_state.selected_Sub_material = "Constant"
if 'nH_r' not in st.session_state: st.session_state.nH_r = 2.35
if 'nH_i' not in st.session_state: st.session_state.nH_i = 0.0
if 'nL_r' not in st.session_state: st.session_state.nL_r = 1.46
if 'nL_i' not in st.session_state: st.session_state.nL_i = 0.0
if 'nSub_r' not in st.session_state: st.session_state.nSub_r = 1.52
st.session_state.selected_H_material = st.sidebar.selectbox("H Material", options=available_materials, key="select_H")
col_h1, col_h2 = st.sidebar.columns(2)
with col_h1: st.session_state.nH_r = st.number_input("n'", key="nH_r", format="%.4f", step=0.01, disabled=(st.session_state.selected_H_material != "Constant"))
with col_h2: st.session_state.nH_i = st.number_input("k", key="nH_i", format="%.4f", min_value=0.0, step=0.001, disabled=(st.session_state.selected_H_material != "Constant"))
st.session_state.selected_L_material = st.sidebar.selectbox("L Material", options=available_materials, key="select_L")
col_l1, col_l2 = st.sidebar.columns(2)
with col_l1: st.session_state.nL_r = st.number_input("n'", key="nL_r", format="%.4f", step=0.01, disabled=(st.session_state.selected_L_material != "Constant"))
with col_l2: st.session_state.nL_i = st.number_input("k", key="nL_i", format="%.4f", min_value=0.0, step=0.001, disabled=(st.session_state.selected_L_material != "Constant"))
st.session_state.selected_Sub_material = st.sidebar.selectbox("Substrate Material", options=available_substrates, key="select_Sub")
st.session_state.nSub_r = st.sidebar.number_input("n' (k=0 assumed)", key="nSub_r", format="%.4f", step=0.01, disabled=(st.session_state.selected_Sub_material != "Constant"))
st.sidebar.divider()
with st.sidebar.expander("Indices n'(λ) des Matériaux Sélectionnés", expanded=False): fig_material_indices = display_material_index_plot();
if fig_material_indices: st.pyplot(fig_material_indices); plt.close(fig_material_indices)
st.sidebar.divider()
st.sidebar.header("Load/Save Design")
uploaded_file = st.sidebar.file_uploader("Load Design (.json)", type="json", key="load_design_uploader")
save_nominal_data = False
if st.sidebar.button("Save Nominal Design", key="save_nom_btn", use_container_width=True):
    try: design_data_nom = _collect_design_data_for_save(include_optimized=False); save_nominal_data = json.dumps(design_data_nom, indent=4).encode('utf-8'); add_log_message("Prepared nominal design data.")
    except Exception as e: add_log_message(f"ERROR prep nominal save: {e}"); st.sidebar.error(f"Save Error: {e}")
if save_nominal_data: st.sidebar.download_button(label="Download Nominal (.json)", data=save_nominal_data, file_name=f"nom_design_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json", mime="application/json", use_container_width=True)
save_optimized_data = False; save_opt_disabled = not (st.session_state.get('optimization_ran_since_nominal_change') and st.session_state.get('current_optimized_ep') is not None)
if st.sidebar.button("Save Optimized Design", key="save_opt_btn", disabled=save_opt_disabled, use_container_width=True):
     try: design_data_opt = _collect_design_data_for_save(include_optimized=True); save_optimized_data = json.dumps(design_data_opt, indent=4).encode('utf-8'); add_log_message("Prepared optimized design data.")
     except Exception as e: add_log_message(f"ERROR prep optimized save: {e}"); st.sidebar.error(f"Save Error: {e}")
if save_optimized_data: st.sidebar.download_button(label="Download Optimized (.json)", data=save_optimized_data, file_name=f"opt_design_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json", mime="application/json", use_container_width=True)

st.subheader("Stack Definition")
col_stack1, col_stack2 = st.columns([3, 1])
with col_stack1: st.session_state.nominal_qwot_str = st.text_area("Nominal QWOT Multipliers (comma-separated)", value=st.session_state.nominal_qwot_str, key="qwot_input_area", height=75, help="Define starting stack using QWOT multipliers.")
try: current_layers = len([item for item in st.session_state.nominal_qwot_str.split(',') if item.strip()]); col_stack1.caption(f"Layers: {current_layers}")
except: col_stack1.caption("Layers: Error")
with col_stack2: st.session_state.initial_layer_number = st.number_input("Layer # (Scan)", min_value=0, value=st.session_state.initial_layer_number, step=1, key="init_layer_num_input", help="Layers for 'Start Nom. (Scan+Opt)'."); st.session_state.l0_qwot = st.number_input("Centering λ (nm)", min_value=1.0, value=st.session_state.l0_qwot, format="%.1f", step=10.0, key="l0_input", help="Wavelength for QWOT calcs.")
if st.session_state.get('optimization_ran_since_nominal_change', False) and st.session_state.get('current_optimized_ep') is not None:
     opt_ep = st.session_state.current_optimized_ep; opt_layers = len(opt_ep); st.text(f"Optimized Structure ({opt_layers} layers):")
     try:
         params = get_validated_input_params()
         if params: qwots_opt, _ = calculate_qwot_from_ep(opt_ep, params['l0'], params['nH_material'], params['nL_material'], EXCEL_FILE_PATH); opt_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt]) if not np.any(np.isnan(qwots_opt)) else "QWOT N/A"; st.text_area("Optimized QWOT", value=opt_qwot_str, height=75, disabled=True, key="opt_qwot_display")
         else: st.text_area("Optimized QWOT", value="Error retrieving params", height=75, disabled=True)
     except Exception as e: st.text_area("Optimized QWOT", value=f"Error calculating: {e}", height=75, disabled=True)
st.divider()

st.subheader("Calculation & Optimization Parameters")
col_param1, col_param2, col_param3 = st.columns(3)
with col_param1: st.session_state.lambda_step = st.number_input("λ Step (nm)", min_value=0.01, value=st.session_state.lambda_step, format="%.2f", step=1.0, key="lambda_step_input", help="λ step for optimization grid.")
with col_param2: st.session_state.max_iter = st.number_input("Max Iter (Opt)", min_value=1, value=st.session_state.max_iter, step=100, key="max_iter_input"); st.session_state.max_fun = st.number_input("Max Eval (Opt)", min_value=1, value=st.session_state.max_fun, step=100, key="max_fun_input")
with col_param3: st.session_state.auto_thin_threshold = st.number_input("Auto Thin Thr. (nm)", min_value=0.0, value=st.session_state.auto_thin_threshold, format="%.3f", step=0.1, key="auto_thin_input")
st.divider()

st.subheader("Spectral Target (Transmittance T)")
target_cols_header = st.columns([0.5, 0.5, 1, 1, 1, 1]); headers = ["Active", "Zone", "λ min", "λ max", "T @ λmin", "T @ λmax"]
for col, header in zip(target_cols_header, headers): col.caption(header)
num_target_rows = st.session_state.get('num_target_rows', 5)
for i in range(num_target_rows):
    target_id_prefix = f"target_{i}"; cols = st.columns([0.5, 0.5, 1, 1, 1, 1])
    st.checkbox("", value=st.session_state[f"{target_id_prefix}_enabled"], key=f"{target_id_prefix}_enabled", label_visibility="collapsed", help=f"Enable target {i+1}")
    cols[1].markdown(f"**{i+1}**")
    st.session_state[f"{target_id_prefix}_min"] = cols[2].number_input("λ min", value=st.session_state[f"{target_id_prefix}_min"], format="%.1f", step=10.0, key=f"{target_id_prefix}_min_input", label_visibility="collapsed")
    st.session_state[f"{target_id_prefix}_max"] = cols[3].number_input("λ max", value=st.session_state[f"{target_id_prefix}_max"], format="%.1f", step=10.0, key=f"{target_id_prefix}_max_input", label_visibility="collapsed")
    st.session_state[f"{target_id_prefix}_target_min"] = cols[4].number_input("T min", value=st.session_state[f"{target_id_prefix}_target_min"], min_value=0.0, max_value=1.0, format="%.3f", step=0.05, key=f"{target_id_prefix}_target_min_input", label_visibility="collapsed")
    st.session_state[f"{target_id_prefix}_target_max"] = cols[5].number_input("T max", value=st.session_state[f"{target_id_prefix}_target_max"], min_value=0.0, max_value=1.0, format="%.3f", step=0.05, key=f"{target_id_prefix}_target_max_input", label_visibility="collapsed")
active_targets_list = get_validated_active_targets_from_state(); active_targets_calc_range, _ = get_lambda_range_from_targets(active_targets_list); num_opt_points_str = "N/A"
if active_targets_calc_range: l_min_calc, l_max_calc = active_targets_calc_range;
if l_min_calc is not None and l_max_calc is not None and l_max_calc >= l_min_calc: l_step_calc = st.session_state.lambda_step
if l_step_calc > 0: num_pts = max(2, int(np.round((l_max_calc - l_min_calc) / l_step_calc)) + 1); num_opt_points_str = f"{num_pts}"
st.caption(f"Approx points in optim grid: {num_opt_points_str}")
st.divider()

st.subheader("Actions")
action_cols = st.columns(4)
evaluate_btn = action_cols[0].button("📊 Evaluate Nominal", use_container_width=True, help="Calculate spectrum for nominal QWOT.")
local_opt_btn = action_cols[1].button("⚙️ Local Optimizer", use_container_width=True, help="Optimize current structure.")
start_scan_opt_btn = action_cols[2].button("✨ Start Scan+Opt", type="primary", use_container_width=True, help="Exhaustive QWOT scan + l0 test + optimize.")
auto_mode_btn = action_cols[3].button("🚀 Auto Mode", use_container_width=True, help=f"Cycles of Needle>Thin>Opt (Max {AUTO_MAX_CYCLES}).")
action_cols2 = st.columns(4)
remove_thin_btn = action_cols2[0].button("🗑️ Remove Thin Layer", use_container_width=True, help="Remove thinnest layer & re-optimize.")
undo_remove_btn = action_cols2[1].button("↩️ Undo Remove", use_container_width=True, help="Revert last layer removal.")
set_nominal_btn = action_cols2[2].button("⬇️ Optimized -> Nominal", use_container_width=True, help="Set optimized stack as nominal QWOT.")
clear_opt_btn = action_cols2[3].button("🧹 Clear Optimized State", use_container_width=True, help="Clear optimized result and history.")
st.divider()

st.subheader("Results")
results_container = st.container()
with results_container:
     st.text(st.session_state.get("status_message", "Status: Ready"))
     if st.session_state.get('plot_placeholder') is None: fig, axes = plt.subplots(1, 3, figsize=(18, 5)); st.session_state.plot_placeholder = {'fig': fig, 'axes': axes}
     st.pyplot(st.session_state.plot_placeholder['fig'])
st.divider()

with st.expander("Log Messages", expanded=False):
    log_container = st.container(height=300)
    log_container.text("\n".join(st.session_state.get('log_messages', ["No logs yet."])))
    if st.button("Clear Log"): st.session_state.log_messages = []; add_log_message("Log Cleared."); st.rerun()

if 'action_running' not in st.session_state: st.session_state.action_running = False

# --- Action Handling Logic ---
current_status = st.session_state.get("status_message", "Status: Ready")

if current_status == "Status: Evaluate Clicked (Implement Logic)" and not st.session_state.action_running:
     st.session_state.action_running = True
     st.session_state.current_optimized_ep = None; st.session_state.current_material_sequence = None
     st.session_state.optimization_ran_since_nominal_change = False; st.session_state.ep_history = deque(maxlen=5)
     add_log_message("Undo history cleared (Nominal/QWOT Calculation).")
     run_calculation_and_plot(ep_vector_to_use=None, is_optimized=False, method_name="Evaluate")
     st.session_state.action_running = False
     st.rerun()

elif current_status == "Status: Starting Local Optimization..." and not st.session_state.action_running:
    st.session_state.action_running = True; final_ep_result = None
    try:
        with st.spinner("Running Local Optimization..."):
            validated_inputs = get_validated_input_params(); active_targets = get_validated_active_targets_from_state()
            if not validated_inputs or active_targets is None or not active_targets: raise ValueError("Invalid inputs/targets for optimization.")
            ep_start = None
            if st.session_state.get('optimization_ran_since_nominal_change') and st.session_state.get('current_optimized_ep') is not None:
                add_log_message("Using current opt state."); ep_start = np.asarray(st.session_state.current_optimized_ep).copy()
            else:
                add_log_message("Using nominal state."); ep_start_calc, logs_init_ep = calculate_initial_ep([float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()], validated_inputs['l0'], validated_inputs['nH_material'], validated_inputs['nL_material'], EXCEL_FILE_PATH); add_log_message(logs_init_ep)
                if ep_start_calc is None: raise ValueError("Could not calc initial thickness."); ep_start = np.asarray(ep_start_calc)
            if ep_start is None or ep_start.size == 0: raise ValueError("Cannot determine start structure."); ep_start = np.maximum(ep_start, MIN_THICKNESS_PHYS_NM); add_log_message(f"Starting opt {len(ep_start)} layers.")
            final_ep_result, optim_success, final_cost, optim_logs, optim_status_msg, total_nit, total_nfev = _run_core_optimization(ep_start, validated_inputs, active_targets, MIN_THICKNESS_PHYS_NM, log_prefix="  [Local Opt] "); add_log_message(optim_logs)
            if not optim_success: raise RuntimeError(f"Local opt failed: {optim_status_msg}, Cost: {final_cost:.3e}")
            add_log_message(f"Local Opt OK. Cost:{final_cost:.3e}, Iter/Eval:{total_nit}/{total_nfev}")
            run_calculation_and_plot(ep_vector_to_use=final_ep_result, is_optimized=True, method_name=f"Local Opt (Cost:{final_cost:.3e})")
    except (ValueError, RuntimeError, TypeError) as e: err_msg = f"ERROR (Local Opt): {e}"; add_log_message(err_msg); st.error(err_msg); st.session_state.status_message = "Status: Local Opt Failed"
    except Exception as e: err_msg = f"ERROR (Unexpected Local Opt): {type(e).__name__}: {e}"; tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e); st.session_state.status_message = "Status: Local Opt Failed (Unexpected)"
    finally: st.session_state.action_running = False; st.rerun()

elif current_status == "Status: Starting Scan+Opt..." and not st.session_state.action_running:
    st.session_state.action_running = True; final_ep_result = None; final_best_l0 = None; final_best_mse = np.inf
    try:
        start_time_scan_opt = time.time()
        with st.spinner("Running QWOT Scan + Optimization... This may take time."):
            validated_inputs = get_validated_input_params(); initial_layer_number = validated_inputs.get('initial_layer_number')
            if not validated_inputs or initial_layer_number is None or initial_layer_number <= 0: raise ValueError("Invalid inputs or Layer Number <= 0.")
            active_targets = get_validated_active_targets_from_state()
            if active_targets is None or not active_targets: raise ValueError("Invalid or no active targets.")
            l0_nominal_gui = validated_inputs['l0']; nH_material = validated_inputs['nH_material']; nL_material = validated_inputs['nL_material']; nSub_material = validated_inputs['nSub_material']; active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
            l_min_val, l_max_val = get_lambda_range_from_targets(active_targets); l_step_optim = validated_inputs['l_step'];
            if l_min_val is None or l_max_val is None: raise ValueError("Cannot determine lambda range.")
            num_pts_eval_full = max(2, int(np.round((l_max_val - l_min_val) / l_step_optim)) + 1); l_vec_eval_full_np = np.geomspace(l_min_val, l_max_val, num_pts_eval_full); l_vec_eval_full_np = l_vec_eval_full_np[(l_vec_eval_full_np > 0) & np.isfinite(l_vec_eval_full_np)]
            l_vec_eval_sparse_np = l_vec_eval_full_np[::5]; # Use sparser grid
            if not l_vec_eval_sparse_np.size: raise ValueError("Sparse lambda vector empty."); l_vec_eval_sparse_jax = jnp.asarray(l_vec_eval_sparse_np); add_log_message(f"Scan grid: {len(l_vec_eval_sparse_jax)} pts.")
            l0_values_to_test = sorted(list(set([l0_nominal_gui, l0_nominal_gui * 1.15, l0_nominal_gui * 0.85]))); l0_values_to_test = [l for l in l0_values_to_test if l > 1e-6]; num_l0_tests = len(l0_values_to_test); num_combinations = 2**initial_layer_number; total_evals_scan = num_combinations * num_l0_tests
            add_log_message(f"Scan N={initial_layer_number}, {num_l0_tests} l0s: {[f'{l:.1f}' for l in l0_values_to_test]}. Combos/l0: {num_combinations:,}."); initial_candidates = []
            nSub_arr_scan, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_eval_sparse_jax, EXCEL_FILE_PATH); add_log_message(logs_sub)
            for l0_idx, current_l0 in enumerate(l0_values_to_test):
                add_log_message(f"\n--- Scan l0 = {current_l0:.2f} nm ({l0_idx+1}/{num_l0_tests}) ---")
                try: nH_c_l0, logs_h = _get_nk_at_lambda(nH_material, current_l0, EXCEL_FILE_PATH); add_log_message(logs_h); nL_c_l0, logs_l = _get_nk_at_lambda(nL_material, current_l0, EXCEL_FILE_PATH); add_log_message(logs_l)
                except Exception as e: add_log_message(f"ERROR get n({current_l0:.2f}): {e}. Skip."); continue
                current_best_mse_scan, current_best_multipliers_scan, scan_logs = _execute_split_stack_scan(current_l0, initial_layer_number, jnp.asarray(nH_c_l0), jnp.asarray(nL_c_l0), nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple); add_log_message(scan_logs)
                if np.isfinite(current_best_mse_scan) and current_best_multipliers_scan is not None: add_log_message(f"Scan cand MSE {current_best_mse_scan:.6e}."); initial_candidates.append({'l0': current_l0, 'mse_scan': current_best_mse_scan, 'multipliers': np.array(current_best_multipliers_scan)})
            if not initial_candidates: raise RuntimeError("Scan found no candidates."); add_log_message(f"\n--- Scan done. {len(initial_candidates)} candidates. Optimizing... ---"); initial_candidates.sort(key=lambda c: c['mse_scan'])
            final_best_ep = None; final_best_initial_multipliers = None; overall_optim_nit = 0; overall_optim_nfev = 0; successful_optim_count = 0
            for idx, candidate in enumerate(initial_candidates):
                 cand_l0 = candidate['l0']; cand_mult = candidate['multipliers']; cand_mse_scan = candidate['mse_scan']; add_log_message(f"\n--- Optimizing Cand {idx+1}/{len(initial_candidates)} (l0={cand_l0:.2f}, scanMSE={cand_mse_scan:.6e}) ---")
                 try:
                     ep_start_optim, logs_ep = calculate_initial_ep(cand_mult, cand_l0, nH_material, nL_material, EXCEL_FILE_PATH); add_log_message(logs_ep); ep_start_optim = np.maximum(ep_start_optim, MIN_THICKNESS_PHYS_NM); add_log_message(f"Start local opt {len(ep_start_optim)} layers.")
                     result_ep_optim, optim_success, final_cost_optim, optim_logs, _, nit_optim, nfev_optim = _run_core_optimization(ep_start_optim, validated_inputs, active_targets, MIN_THICKNESS_PHYS_NM, log_prefix=f"  [OptCand{idx+1}] "); add_log_message(optim_logs)
                     if optim_success: successful_optim_count += 1; overall_optim_nit += nit_optim; overall_optim_nfev += nfev_optim; add_log_message(f"Opt OK. MSE: {final_cost_optim:.6e}")
                     if final_cost_optim < final_best_mse: add_log_message(f"*** New global best! ***"); final_best_mse = final_cost_optim; final_best_ep = result_ep_optim.copy(); final_best_l0 = cand_l0; final_best_initial_multipliers = cand_mult
                 except Exception as e_optim_cand: add_log_message(f"ERROR opt cand {idx+1}: {e_optim_cand}")
            if final_best_ep is None: raise RuntimeError("Opt failed for all candidates.")
            add_log_message(f"\n--- Scan+Opt Best ---"); add_log_message(f"MSE: {final_best_mse:.6e} ({len(final_best_ep)}L), L0: {final_best_l0:.2f}nm"); best_mult_list_str = ",".join([f"{m:.3f}" for m in final_best_initial_multipliers]); add_log_message(f"Orig Mult: {best_mult_list_str}")
            if abs(final_best_l0 - l0_nominal_gui) > 1e-3: add_log_message(f"Updating GUI l0 -> {final_best_l0:.2f}"); st.session_state.l0_qwot = final_best_l0
            final_ep_result = final_best_ep
    except (ValueError, RuntimeError, TypeError) as e: err_msg = f"ERROR (Scan+Opt): {e}"; add_log_message(err_msg); st.error(err_msg); st.session_state.status_message = "Status: Scan+Opt Failed"
    except Exception as e: err_msg = f"ERROR (Unexpected Scan+Opt): {type(e).__name__}: {e}"; tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e); st.session_state.status_message = "Status: Scan+Opt Failed (Unexpected)"
    finally:
        st.session_state.action_running = False
        if final_ep_result is not None: run_calculation_and_plot(ep_vector_to_use=final_ep_result, is_optimized=True, method_name=f"Scan+Opt (L0={final_best_l0:.1f})")
        st.rerun()

elif current_status == "Status: Starting Auto Mode..." and not st.session_state.action_running:
    st.session_state.action_running = True; final_ep_result = None; best_mse_so_far = np.inf; num_cycles_done = 0; termination_reason = f"Max {AUTO_MAX_CYCLES} cycles"
    try:
        start_time_auto = time.time()
        with st.spinner("Running Auto Mode..."):
            validated_inputs = get_validated_input_params(); active_targets = get_validated_active_targets_from_state()
            if not validated_inputs or active_targets is None or not active_targets: raise ValueError("Invalid inputs/targets.")
            nH_material = validated_inputs['nH_material']; nL_material = validated_inputs['nL_material']; nSub_material = validated_inputs['nSub_material']; l0 = validated_inputs['l0']; threshold_from_gui = validated_inputs.get('auto_thin_threshold', 1.0); add_log_message(f"Auto Thin Thr: {threshold_from_gui:.3f}nm")
            l_min_val, l_max_val = get_lambda_range_from_targets(active_targets); l_step_optim = validated_inputs['l_step']
            if l_min_val is None or l_max_val is None: raise ValueError("Cannot get lambda range.")
            num_pts = max(2, int(np.round((l_max_val - l_min_val) / l_step_optim)) + 1); l_vec_optim_np = np.geomspace(l_min_val, l_max_val, num_pts); l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
            if not l_vec_optim_np.size: raise ValueError("Lambda vector empty."); ep_start_auto = None; total_iters_auto = 0; total_evals_auto = 0; optim_runs_auto = 0
            if st.session_state.get('optimization_ran_since_nominal_change') and st.session_state.get('current_optimized_ep') is not None:
                 add_log_message("Auto: Using current opt state."); ep_start_auto = np.asarray(st.session_state.current_optimized_ep).copy()
                 try: l_vec_optim_jax = jnp.asarray(l_vec_optim_np); nH_arr, _ = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH); nL_arr, _ = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH); nSub_arr, _ = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH); active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets); static_args_cost_fn = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, jnp.float64(MIN_THICKNESS_PHYS_NM)); cost_fn_compiled = jax.jit(calculate_mse_for_optimization_penalized_jax); initial_cost_jax = cost_fn_compiled(jnp.asarray(ep_start_auto), *static_args_cost_fn); best_mse_so_far = float(np.array(initial_cost_jax));
                 except Exception as e: raise ValueError(f"Cannot calc initial MSE: {e}")
            else:
                add_log_message("Auto: Using nominal + initial opt."); ep_nominal_calc, logs_init_ep = calculate_initial_ep([float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()], l0, nH_material, nL_material, EXCEL_FILE_PATH); add_log_message(logs_init_ep);
                if ep_nominal_calc is None or ep_nominal_calc.size == 0: raise ValueError("Cannot get nominal start."); ep_nominal = np.maximum(np.asarray(ep_nominal_calc), MIN_THICKNESS_PHYS_NM)
                ep_after_initial_opt, initial_opt_success, initial_mse, initial_opt_logs, initial_opt_msg, initial_nit, initial_nfev = _run_core_optimization(ep_nominal, validated_inputs, active_targets, MIN_THICKNESS_PHYS_NM, log_prefix="  [AutoInitOpt] "); add_log_message(initial_opt_logs)
                if not initial_opt_success: raise RuntimeError(f"Initial opt failed: {initial_opt_msg}")
                ep_start_auto = ep_after_initial_opt.copy(); best_mse_so_far = initial_mse; total_iters_auto += initial_nit; total_evals_auto += initial_nfev; optim_runs_auto += 1
            best_ep_so_far = ep_start_auto.copy()
            if not np.isfinite(best_mse_so_far): raise ValueError("Start MSE not finite."); add_log_message(f"Start Auto Cycles. MSE: {best_mse_so_far:.6e} ({len(best_ep_so_far)}L)")
            for cycle_num in range(AUTO_MAX_CYCLES):
                add_log_message(f"\n--- Auto Cycle {cycle_num + 1}/{AUTO_MAX_CYCLES} ---"); mse_at_cycle_start = best_mse_so_far; ep_at_cycle_start = best_ep_so_far.copy(); cycle_improved_overall = False
                add_log_message(f" [C{cycle_num+1}] Needles ({AUTO_NEEDLES_PER_CYCLE}x)..."); ep_after_needles, mse_after_needles, needle_logs, nit_needles, nfev_needles, reopts_in_needles = _run_needle_iterations(best_ep_so_far, AUTO_NEEDLES_PER_CYCLE, validated_inputs, active_targets, MIN_THICKNESS_PHYS_NM, l_vec_optim_np, DEFAULT_NEEDLE_SCAN_STEP_NM, BASE_NEEDLE_THICKNESS_NM, calculate_mse_for_optimization_penalized_jax, log_prefix=f"    [NdlC{cycle_num+1}] "); add_log_message(needle_logs); total_iters_auto += nit_needles; total_evals_auto += nfev_needles; optim_runs_auto += reopts_in_needles
                if mse_after_needles < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE: best_ep_so_far = ep_after_needles.copy(); best_mse_so_far = mse_after_needles; cycle_improved_overall = True
                else: best_ep_so_far = ep_after_needles.copy(); best_mse_so_far = mse_after_needles
                add_log_message(f" [C{cycle_num+1}] Thin Removal (Thr:{threshold_from_gui:.3f}nm)..."); layers_removed_this_cycle = 0; thinning_loop_iteration = 0; max_thinning_iterations = len(best_ep_so_far) + 2
                while thinning_loop_iteration < max_thinning_iterations:
                     thinning_loop_iteration += 1; current_num_layers = len(best_ep_so_far)
                     if current_num_layers <= 2: break
                     ep_after_single_removal, structure_changed, removal_logs = _perform_layer_merge_or_removal_only(best_ep_so_far, MIN_THICKNESS_PHYS_NM, log_prefix=f"      [Rem{thinning_loop_iteration}]", threshold_for_removal=threshold_from_gui); add_log_message(removal_logs)
                     if structure_changed:
                          layers_removed_this_cycle += 1; add_log_message(f"    Layer removed. Re-opt {len(ep_after_single_removal)} layers...")
                          ep_after_thin_reopt, thin_reopt_success, mse_after_thin_reopt, thin_reopt_logs, thin_reopt_msg, nit_thin_reopt, nfev_thin_reopt = _run_core_optimization(ep_after_single_removal, validated_inputs, active_targets, MIN_THICKNESS_PHYS_NM, log_prefix=f"        [ReOpt{layers_removed_this_cycle}] "); add_log_message(thin_reopt_logs)
                          if thin_reopt_success:
                               total_iters_auto += nit_thin_reopt; total_evals_auto += nfev_thin_reopt; optim_runs_auto += 1
                               if mse_after_thin_reopt < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE: best_ep_so_far = ep_after_thin_reopt.copy(); best_mse_so_far = mse_after_thin_reopt; cycle_improved_overall = True
                               else: best_ep_so_far = ep_after_thin_reopt.copy(); best_mse_so_far = mse_after_thin_reopt
                          else:
                               add_log_message(f"    WARN: Re-opt fail ({thin_reopt_msg}). Stop removal."); best_ep_so_far = ep_after_single_removal.copy()
                               try: cost_fn_compiled = jax.jit(calculate_mse_for_optimization_penalized_jax); static_args_cost_fn = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, jnp.float64(MIN_THICKNESS_PHYS_NM)); failed_cost_jax = cost_fn_compiled(jnp.asarray(best_ep_so_far), *static_args_cost_fn); best_mse_so_far = float(np.array(failed_cost_jax));
                               except Exception: best_mse_so_far = np.inf
                               cycle_improved_overall = (best_mse_so_far < mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE) or cycle_improved_overall; break
                     else: break
                add_log_message(f" [C{cycle_num+1}] Thin Rem Phase done. {layers_removed_this_cycle} removed."); num_cycles_done += 1
                add_log_message(f"--- End Auto Cycle {cycle_num + 1}. Best MSE:{best_mse_so_far:.6e} ({len(best_ep_so_far)}L)---")
                if not cycle_improved_overall and best_mse_so_far >= mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE:
                    add_log_message(f"No improvement. Stop Auto."); termination_reason = "No improvement"
                    if best_mse_so_far > mse_at_cycle_start + MSE_IMPROVEMENT_TOLERANCE : add_log_message(f"Revert to state before Cycle {cycle_num + 1}."); best_ep_so_far = ep_at_cycle_start; best_mse_so_far = mse_at_cycle_start
                    break
            final_ep_result = best_ep_so_far
    except (ValueError, RuntimeError, TypeError) as e: err_msg = f"ERROR (Auto Mode): {e}"; add_log_message(err_msg); st.error(err_msg); st.session_state.status_message = "Status: Auto Mode Failed"
    except Exception as e: err_msg = f"ERROR (Unexpected Auto Mode): {e}"; tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e); st.session_state.status_message = "Status: Auto Mode Failed (Unexpected)"
    finally:
        st.session_state.action_running = False
        if final_ep_result is not None: add_log_message(f"\n--- Auto Mode Finished ({num_cycles_done} cyc, {termination_reason}) ---"); add_log_message(f"Final MSE: {best_mse_so_far:.6e} ({len(final_ep_result)}L)"); run_calculation_and_plot(ep_vector_to_use=final_ep_result, is_optimized=True, method_name=f"Auto Mode ({num_cycles_done} cyc)")
        st.rerun()

elif current_status == "Status: Starting Remove Thin Layer..." and not st.session_state.action_running:
    st.session_state.action_running = True; final_ep_result = None
    try:
        if not st.session_state.get('optimization_ran_since_nominal_change') or st.session_state.get('current_optimized_ep') is None: raise ValueError("Remove Thin requires opt structure.")
        current_ep_opt = np.asarray(st.session_state.current_optimized_ep)
        if len(current_ep_opt) <= 2: raise ValueError("Structure <= 2 layers.")
        with st.spinner("Removing thin layer & re-optimizing..."):
             validated_inputs = get_validated_input_params(); active_targets = get_validated_active_targets_from_state()
             if not validated_inputs or active_targets is None or not active_targets: raise ValueError("Invalid inputs/targets.")
             st.session_state.ep_history.append(current_ep_opt.copy()); add_log_message(f" [Undo] State saved. Hist:{len(st.session_state.ep_history)}")
             threshold_rem = validated_inputs.get('auto_thin_threshold', None); add_log_message(f"Remove Thr: {threshold_rem}")
             ep_after_removal, structure_changed, removal_logs = _perform_layer_merge_or_removal_only(current_ep_opt, MIN_THICKNESS_PHYS_NM, log_prefix="  [Removal] ", threshold_for_removal=threshold_rem); add_log_message(removal_logs)
             if not structure_changed: add_log_message("No layer removed."); st.info("Could not remove layer."); st.session_state.status_message = f"Status: Removal Skipped"; final_ep_result = current_ep_opt;
             else:
                 add_log_message(f"Changed to {len(ep_after_removal)} layers. Re-opt..."); ep_after_reopt, reopt_success, final_cost, reopt_logs, reopt_status_msg, _, _ = _run_core_optimization(ep_after_removal, validated_inputs, active_targets, MIN_THICKNESS_PHYS_NM, log_prefix="  [Re-Opt] "); add_log_message(reopt_logs)
                 if not reopt_success: add_log_message("ERROR: Re-opt fail."); st.warning(f"Re-opt fail: {reopt_status_msg}."); final_ep_result = ep_after_removal.copy(); st.session_state.status_message = f"Status: Removed, Re-Opt Failed"
                 else: add_log_message("Re-opt OK."); final_ep_result = ep_after_reopt.copy(); st.session_state.status_message = f"Status: Removed & Re-Opt OK | MSE: {final_cost:.3e}"
             run_calculation_and_plot(ep_vector_to_use=final_ep_result, is_optimized=True, method_name=f"Opt Post-Removal{' (Skip)' if not structure_changed else (' (Re-Opt Fail)' if not reopt_success else '')}")
    except (ValueError, RuntimeError, TypeError) as e: err_msg = f"ERROR (Remove Thin): {e}"; add_log_message(err_msg); st.error(err_msg); st.session_state.status_message = "Status: Remove Thin Failed"
    except Exception as e: err_msg = f"ERROR (Unexpected Remove): {e}"; tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e); st.session_state.status_message = "Status: Remove Thin Failed (Unexpected)"
    finally: st.session_state.action_running = False; st.rerun()

elif current_status == "Status: Undoing removal..." and not st.session_state.action_running:
     st.session_state.action_running = True
     try:
         if not st.session_state.ep_history: add_log_message("Undo history empty."); st.info("Nothing to undo."); st.session_state.status_message = "Status: Ready"
         else:
             with st.spinner("Restoring previous state..."):
                 restored_ep = st.session_state.ep_history.pop(); add_log_message(f"Undo OK. Restore {len(restored_ep)}L. Undos left: {len(st.session_state.ep_history)}")
                 run_calculation_and_plot(ep_vector_to_use=restored_ep, is_optimized=True, method_name="Optimized (Undo)")
                 st.session_state.status_message = f"Status: Undo OK | Layers: {len(restored_ep)}"
     except Exception as e: err_msg = f"ERROR (Undo): {e}"; tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e); st.session_state.status_message = "Status: Undo Failed"; st.session_state.current_optimized_ep = None; st.session_state.optimization_ran_since_nominal_change = False; st.session_state.ep_history = deque(maxlen=5)
     finally: st.session_state.action_running = False; st.rerun()

elif current_status == "Status: Setting Optimized as Nominal..." and not st.session_state.action_running:
     st.session_state.action_running = True
     try:
         if not st.session_state.get('optimization_ran_since_nominal_change') or st.session_state.get('current_optimized_ep') is None: raise ValueError("No opt structure.")
         with st.spinner("Calculating QWOT and updating..."):
              validated_inputs = get_validated_input_params(); optimized_ep = st.session_state.current_optimized_ep
              if not validated_inputs: raise ValueError("Invalid params.")
              optimized_qwots, logs_qwot = calculate_qwot_from_ep(optimized_ep, validated_inputs['l0'], validated_inputs['nH_material'], validated_inputs['nL_material'], EXCEL_FILE_PATH); add_log_message(logs_qwot)
              if np.any(np.isnan(optimized_qwots)): st.warning("NaN in QWOTs. Nominal not updated.")
              else: final_qwot_str = ",".join([f"{q:.6f}" for q in optimized_qwots]); st.session_state.nominal_qwot_str = final_qwot_str; add_log_message("Nominal QWOT updated."); st.success("Optimized set as Nominal.")
              st.session_state.current_optimized_ep = None; st.session_state.current_material_sequence = None; st.session_state.optimization_ran_since_nominal_change = False; st.session_state.ep_history = deque(maxlen=5); add_log_message("Optimized state cleared.")
              run_calculation_and_plot(ep_vector_to_use=None, is_optimized=False, method_name="Nominal (Post Set)")
              st.session_state.status_message = "Status: Optimized set as Nominal"
     except (ValueError, RuntimeError, TypeError) as e: err_msg = f"ERROR (Set Nominal): {e}"; add_log_message(err_msg); st.error(err_msg); st.session_state.status_message = "Status: Set Nominal Failed"
     except Exception as e: err_msg = f"ERROR (Unexpected Set Nominal): {e}"; tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e); st.session_state.status_message = "Status: Set Nominal Failed (Unexpected)"
     finally: st.session_state.action_running = False; st.rerun()

elif current_status == "Status: Clearing Optimized State..." and not st.session_state.action_running:
     st.session_state.action_running = True
     add_log_message("Clearing optimized state and Undo history.")
     st.session_state.current_optimized_ep = None; st.session_state.current_material_sequence = None; st.session_state.optimization_ran_since_nominal_change = False; st.session_state.ep_history = deque(maxlen=5); st.session_state.last_calc_results = None; st.session_state.last_calc_mse = None; st.session_state.last_calc_ep = None; st.session_state.last_calc_is_optimized = False
     st.session_state.status_message = "Status: Optimized state cleared."
     run_calculation_and_plot(ep_vector_to_use=None, is_optimized=False, method_name="Nominal (Cleared)")
     st.session_state.action_running = False
     st.rerun()

elif current_status == "Status: Loading Design..." and not st.session_state.action_running:
     st.session_state.action_running = True; load_error = False
     try:
         uploaded_file_obj = st.session_state.get("load_design_uploader");
         if uploaded_file_obj is None: raise ValueError("Uploaded file object missing.")
         design_data = json.load(uploaded_file_obj)
         if 'params' not in design_data or 'targets' not in design_data: raise ValueError("Invalid JSON.")
         loaded_params = design_data.get('params', {}); param_map = {'l0': 'l0_qwot', 'l_step': 'lambda_step', 'maxiter': 'max_iter', 'maxfun': 'max_fun', 'emp_str': 'nominal_qwot_str', 'initial_layer_number': 'initial_layer_number', 'auto_thin_threshold': 'auto_thin_threshold'}
         for key, value in loaded_params.items():
              state_key = param_map.get(key)
              if state_key and state_key in st.session_state: try: st.session_state[state_key] = type(st.session_state[state_key])(value) except: st.session_state[state_key] = value
              elif key in ['nH_r', 'nH_i', 'nL_r', 'nL_i', 'nSub_r'] and key in st.session_state: try: st.session_state[key] = float(value) except: pass
         def set_material_state(role, loaded_val):
             state_key = f'selected_{role}_material'; const_r = f'n{role}_r'; const_i = f'n{role}_i' if role != 'Sub' else None; opts = available_materials if role != 'Sub' else available_substrates
             if isinstance(loaded_val, str) and loaded_val in opts: st.session_state[state_key] = loaded_val
             elif isinstance(loaded_val, (dict, list, int, float)):
                  try:
                      if isinstance(loaded_val, dict): n_r=float(loaded_val.get('real',1.)); n_i=float(loaded_val.get('imag',0.))
                      elif isinstance(loaded_val, list) and len(loaded_val)>=1: n_r=float(loaded_val[0]); n_i=float(loaded_val[1]) if len(loaded_val)>1 else 0.
                      else: n_r=float(loaded_val); n_i=0.
                      st.session_state[state_key]="Constant"; st.session_state[const_r]=n_r;
                      if const_i and const_i in st.session_state: st.session_state[const_i]=n_i
                  except Exception as e: add_log_message(f"Warn: Parse const {role} fail: {e}."); st.session_state[state_key]="Constant"
             else: st.session_state[state_key]="Constant"
         set_material_state('H', loaded_params.get('nH_material', 'Constant')); set_material_state('L', loaded_params.get('nL_material', 'Constant')); set_material_state('Sub', loaded_params.get('nSub_material', 'Constant'))
         loaded_targets_raw = design_data.get('targets', []); new_targets_state = []
         num_target_rows_load = st.session_state.get('num_target_rows', 5)
         for i in range(num_target_rows_load):
             if i < len(loaded_targets_raw): t_data_raw = loaded_targets_raw[i]
             else: t_data_raw = {} # Use default if file has fewer targets
             if isinstance(t_data_raw, dict): new_targets_state.append({'id':i, 'enabled':bool(t_data_raw.get('enabled',False)), 'min':float(t_data_raw.get('min',0.)), 'max':float(t_data_raw.get('max',0.)), 'target_min':float(t_data_raw.get('target_min',0.)), 'target_max':float(t_data_raw.get('target_max',0.))})
             else: new_targets_state.append({'id': i, 'enabled': False, 'min': 0., 'max': 0., 'target_min': 0., 'target_max': 0.})
         for i, t_data in enumerate(new_targets_state): prefix = f"target_{i}"; st.session_state[f"{prefix}_enabled"] = t_data['enabled']; st.session_state[f"{prefix}_min"] = t_data['min']; st.session_state[f"{prefix}_max"] = t_data['max']; st.session_state[f"{prefix}_target_min"] = t_data['target_min']; st.session_state[f"{prefix}_target_max"] = t_data['target_max']
         st.session_state.current_optimized_ep = None; st.session_state.current_material_sequence = None; st.session_state.optimization_ran_since_nominal_change = False; st.session_state.ep_history = deque(maxlen=5)
         if 'optimized_ep' in design_data and isinstance(design_data['optimized_ep'], list) and design_data['optimized_ep']:
              try:
                   loaded_ep = np.array(design_data['optimized_ep'], dtype=np.float64)
                   if np.all(np.isfinite(loaded_ep)) and np.all(loaded_ep >= 0):
                        st.session_state.current_optimized_ep = loaded_ep; st.session_state.optimization_ran_since_nominal_change = True; add_log_message(f"Loaded opt state ({len(loaded_ep)} layers).")
                        if 'optimized_material_sequence' in design_data and isinstance(design_data['optimized_material_sequence'], list):
                            if len(design_data['optimized_material_sequence']) == len(loaded_ep): st.session_state.current_material_sequence = design_data['optimized_material_sequence']; add_log_message("Loaded opt sequence.")
                            else: add_log_message("Warn: Opt sequence length mismatch.")
                   else: raise ValueError("Invalid values in opt_ep.")
              except Exception as e_ep: add_log_message(f"WARN: Load opt_ep fail: {e_ep}."); st.warning(f"Load opt fail: {e_ep}")
         add_log_message("Design loaded. Recalculating..."); st.toast(f"Design loaded"); st.session_state.status_message = "Status: Design Loaded, Recalculating..."
     except (ValueError, json.JSONDecodeError) as e: err_msg = f"ERROR loading design: {e}"; add_log_message(err_msg); st.error(err_msg); st.session_state.status_message = "Status: Load Failed"; load_error = True
     except Exception as e: err_msg = f"ERROR (Unexpected Load): {e}"; tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e); st.session_state.status_message = "Status: Load Failed (Unexpected)"; load_error = True
     finally:
         st.session_state.action_running = False
         if not load_error:
              ep_to_calc = st.session_state.current_optimized_ep if st.session_state.optimization_ran_since_nominal_change else None; seq_to_calc = st.session_state.current_material_sequence if st.session_state.optimization_ran_since_nominal_change else None
              run_calculation_and_plot(ep_vector_to_use=ep_to_calc, is_optimized=st.session_state.optimization_ran_since_nominal_change, method_name="Loaded", material_sequence=seq_to_calc)
         st.session_state.load_design_uploader = None # Reset uploader
         st.rerun()

# --- Trigger Initial Calculation on First Load ---
if st.session_state.get('last_calc_results') is None and not st.session_state.action_running :
     add_log_message("Performing initial calculation on first load...")
     run_calculation_and_plot(ep_vector_to_use=None, is_optimized=False, method_name="Initial Load")
     st.rerun()

# --- Final cleanup: Reset action_running if stuck ---
valid_statuses_for_action = [ "Status: Evaluate Clicked (Implement Logic)", "Status: Starting Local Optimization...", "Status: Starting Scan+Opt...", "Status: Starting Auto Mode...", "Status: Starting Remove Thin Layer...", "Status: Undoing removal...", "Status: Setting Optimized as Nominal...", "Status: Clearing Optimized State...", "Status: Loading Design..." ]
if st.session_state.action_running and st.session_state.status_message not in valid_statuses_for_action:
    st.session_state.action_running = False
