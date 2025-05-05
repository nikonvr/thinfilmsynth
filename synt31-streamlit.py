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
import openpyxl
st.set_page_config(layout="wide", page_title="Thin Film Optimizer (Streamlit)")
try:
    jax.config.update("jax_enable_x64", True)
    print("JAX configured for float64.")
except Exception as e:
    st.error(f"Error configuring JAX: {e}")
    st.stop()
MIN_THICKNESS_PHYS_NM = 0.01
BASE_NEEDLE_THICKNESS_NM = 0.1
DEFAULT_NEEDLE_SCAN_STEP_NM = 2.0
AUTO_NEEDLES_PER_CYCLE = 5
AUTO_MAX_CYCLES = 5
MSE_IMPROVEMENT_TOLERANCE = 1e-9
EXCEL_FILE_PATH = "indices.xlsx"
@functools.lru_cache(maxsize=32)
def load_material_data_from_xlsx_sheet(file_path: str, sheet_name: str) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:
    if not os.path.exists(file_path):
         st.error(f"Excel file not found at specified path: {file_path}")
         return None, None, None
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        numeric_df = numeric_df.dropna(how='all')
        if numeric_df.shape[1] >= 3:
            numeric_df = numeric_df.dropna(subset=[0, 1, 2])
        else:
            raise ValueError(f"Sheet '{sheet_name}' does not contain 3 numeric columns.")
        numeric_df = numeric_df.sort_values(by=0)
        l_nm = numeric_df.iloc[:, 0].values.astype(np.float64)
        n = numeric_df.iloc[:, 1].values.astype(np.float64)
        k = numeric_df.iloc[:, 2].values.astype(np.float64)
        if len(l_nm) == 0:
            raise ValueError(f"No valid numeric data found in sheet '{sheet_name}' of file {file_path}")
        print(f"Data loaded from {file_path} (Sheet: '{sheet_name}'): {len(l_nm)} points from {l_nm.min():.1f} nm to {l_nm.max():.1f} nm")
        return l_nm, n, k
    except FileNotFoundError:
        st.error(f"Error: Excel file not found: {file_path}")
        return None, None, None
    except ValueError as ve:
        st.error(f"Value error while reading sheet '{sheet_name}' in {file_path}: {ve}")
        return None, None, None
    except Exception as e:
        st.error(f"Unexpected error while reading sheet '{sheet_name}' in {file_path}: {type(e).__name__} - {e}")
        return None, None, None
@st.cache_data(max_entries=32)
def get_available_materials_from_excel(excel_path: str) -> List[str]:
    if not os.path.exists(excel_path):
         print(f"Warning: Excel file {excel_path} not found for listing materials.")
         return []
    try:
        xl = pd.ExcelFile(excel_path)
        sheet_names = [name for name in xl.sheet_names if not name.startswith("Sheet")]
        print(f"Materials found in {excel_path}: {sheet_names}")
        return sorted(sheet_names)
    except Exception as e:
        st.error(f"Error reading sheet names from {excel_path}: {e}")
        return []
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
@jax.jit
def interp_nk_cached(l_target: jnp.ndarray, l_data: jnp.ndarray, n_data: jnp.ndarray, k_data: jnp.ndarray) -> jnp.ndarray:
    n_interp = jnp.interp(l_target, l_data, n_data)
    k_interp = jnp.interp(l_target, l_data, k_data)
    return n_interp + 1j * k_interp
MaterialInputType = Union[complex, float, int, str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
@st.cache_data(max_entries=64, show_spinner=False)
def _get_nk_array_for_lambda_vec_cached(material_definition_repr: str,
                                        l_vec_target_jnp_tuple: Tuple[float],
                                        excel_file_path: str,
                                        const_n_r: float = 0.0,
                                        const_n_i: float = 0.0
                                        ) -> jnp.ndarray:
    l_vec_target_jnp = jnp.array(l_vec_target_jnp_tuple)
    if material_definition_repr == "Constant":
        material_definition = complex(const_n_r, const_n_i)
        return jnp.full(l_vec_target_jnp.shape, jnp.asarray(material_definition, dtype=jnp.complex128))
    elif isinstance(material_definition_repr, str):
        mat_upper = material_definition_repr.upper()
        if mat_upper == "FUSED SILICA":
            return get_n_fused_silica(l_vec_target_jnp)
        elif mat_upper == "BK7":
            return get_n_bk7(l_vec_target_jnp)
        elif mat_upper == "D263":
            return get_n_d263(l_vec_target_jnp)
        else:
            sheet_name = material_definition_repr
            l_data, n_data, k_data = load_material_data_from_xlsx_sheet(excel_file_path, sheet_name)
            if l_data is None:
                raise ValueError(f"Could not load data for sheet '{sheet_name}' from {excel_file_path}")
            l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
            return interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)
    else:
        raise TypeError(f"Unsupported material definition type for caching: {type(material_definition_repr)}")
def get_nk_array(material_definition: MaterialInputType, l_vec_target_jnp: jnp.ndarray, excel_file_path: str) -> jnp.ndarray:
    l_vec_tuple = tuple(np.array(l_vec_target_jnp).tolist())
    if isinstance(material_definition, (complex, float, int)):
        return _get_nk_array_for_lambda_vec_cached("Constant", l_vec_tuple, excel_file_path,
                                                   const_n_r=float(np.real(material_definition)),
                                                   const_n_i=float(np.imag(material_definition)))
    elif isinstance(material_definition, str):
        return _get_nk_array_for_lambda_vec_cached(material_definition, l_vec_tuple, excel_file_path)
    elif isinstance(material_definition, tuple) and len(material_definition) == 3:
         l_data, n_data, k_data = material_definition
         l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
         sort_indices = jnp.argsort(l_data_jnp)
         l_data_jnp, n_data_jnp, k_data_jnp = l_data_jnp[sort_indices], n_data_jnp[sort_indices], k_data_jnp[sort_indices]
         return interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)
    else:
         raise TypeError(f"Unsupported material definition type: {type(material_definition)}")
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
        M_layer = jnp.array([
            [cos_phi,              (1j / safe_eta) * sin_phi],
            [1j * eta * sin_phi,   cos_phi]
        ], dtype=jnp.complex128)
        return M_layer @ carry_matrix
    def compute_identity(thickness_: jnp.ndarray) -> jnp.ndarray:
        return carry_matrix
    new_matrix = cond(thickness > 1e-12, compute_M_layer, compute_identity, thickness)
    return new_matrix, None
@jax.jit
def compute_stack_matrix_jax(ep_vector: jnp.ndarray, l_val: jnp.ndarray, nH_at_lval: jnp.ndarray, nL_at_lval: jnp.ndarray) -> jnp.ndarray:
    num_layers = len(ep_vector)
    is_H_layer = jnp.arange(num_layers) % 2 == 0
    layer_indices = jnp.where(is_H_layer, nH_at_lval, nL_at_lval)
    layers_scan_data = (ep_vector, layer_indices, jnp.full(num_layers, l_val))
    M_initial = jnp.eye(2, dtype=jnp.complex128)
    M_final, _ = scan(_compute_layer_matrix_scan_step, M_initial, layers_scan_data)
    return M_final
@jax.jit
def calculate_single_wavelength_T(l_val: jnp.ndarray, ep_vector_contig: jnp.ndarray,
                                  nH_at_lval: jnp.ndarray, nL_at_lval: jnp.ndarray, nSub_at_lval: jnp.ndarray) -> jnp.ndarray:
    etainc = 1.0 + 0j
    etasub = nSub_at_lval
    def calculate_for_valid_l(l_: jnp.ndarray) -> jnp.ndarray:
        M = compute_stack_matrix_jax(ep_vector_contig, l_, nH_at_lval, nL_at_lval)
        m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        rs_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        rs_denominator_abs = jnp.abs(rs_denominator)
        safe_denominator = jnp.where(rs_denominator_abs < 1e-12, 1e-12 + 0j, rs_denominator)
        ts = (2.0 * etainc) / safe_denominator
        real_etasub = jnp.real(etasub); real_etainc = jnp.real(etainc)
        Ts_complex = (real_etasub / real_etainc) * (ts * jnp.conj(ts))
        Ts = jnp.real(Ts_complex)
        return jnp.where(rs_denominator_abs < 1e-12, jnp.nan, Ts)
    def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray:
        return jnp.nan
    Ts = cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)
    return Ts
@jax.jit
def calculate_T_from_ep_core_jax(ep_vector: jnp.ndarray,
                                 nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                 l_vec: jnp.ndarray) -> jnp.ndarray:
    if not l_vec.size:
        return jnp.zeros(0, dtype=jnp.float64)
    ep_vector_contig = jnp.asarray(ep_vector)
    Ts_arr = vmap(calculate_single_wavelength_T, in_axes=(0, None, 0, 0, 0))(
        l_vec, ep_vector_contig, nH_arr, nL_arr, nSub_arr
    )
    return jnp.nan_to_num(Ts_arr, nan=0.0)
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
    etainc = 1.0 + 0j
    etasub = nSub_at_lval
    def calculate_for_valid_l(l_: jnp.ndarray) -> jnp.ndarray:
        M = compute_stack_matrix_arbitrary_jax(ep_vector_contig, layer_indices_at_lval, l_)
        m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        rs_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        rs_denominator_abs = jnp.abs(rs_denominator)
        safe_denominator = jnp.where(rs_denominator_abs < 1e-12, 1e-12 + 0j, rs_denominator)
        ts = (2.0 * etainc) / safe_denominator
        real_etasub = jnp.real(etasub); real_etainc = jnp.real(etainc)
        Ts_complex = (real_etasub / real_etainc) * (ts * jnp.conj(ts))
        Ts = jnp.real(Ts_complex)
        return jnp.where(rs_denominator_abs < 1e-12, jnp.nan, Ts)
    def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray:
        return jnp.nan
    Ts = cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)
    return Ts
@jax.jit
def calculate_T_from_ep_arbitrary_core_jax(ep_vector: jnp.ndarray,
                                           layer_indices_arr: jnp.ndarray,
                                           nSub_arr: jnp.ndarray,
                                           l_vec: jnp.ndarray) -> jnp.ndarray:
    if not l_vec.size:
        return jnp.zeros(0, dtype=jnp.float64)
    ep_vector_contig = jnp.asarray(ep_vector)
    layer_indices_arr_transposed = layer_indices_arr.T
    Ts_arr = vmap(calculate_single_wavelength_T_arbitrary, in_axes=(0, None, 0, 0))(
        l_vec, ep_vector_contig, layer_indices_arr_transposed, nSub_arr
    )
    return jnp.nan_to_num(Ts_arr, nan=0.0)
def calculate_T_from_ep_jax_wrapper(ep_vector: Union[np.ndarray, List[float]],
                              nH_material: MaterialInputType,
                              nL_material: MaterialInputType,
                              nSub_material: MaterialInputType,
                              l_vec: Union[np.ndarray, List[float]],
                              excel_file_path: str) -> Dict[str, np.ndarray]:
    l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64)
    ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)
    try:
        nH_arr = get_nk_array(nH_material, l_vec_jnp, excel_file_path)
        nL_arr = get_nk_array(nL_material, l_vec_jnp, excel_file_path)
        nSub_arr = get_nk_array(nSub_material, l_vec_jnp, excel_file_path)
    except Exception as e:
        st.error(f"Error preparing material data for calculation: {e}")
        raise
    Ts = calculate_T_from_ep_core_jax(ep_vector_jnp, nH_arr, nL_arr, nSub_arr, l_vec_jnp)
    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts)}
def calculate_T_from_ep_arbitrary_jax_wrapper(ep_vector: Union[np.ndarray, List[float]],
                                        material_sequence: List[str],
                                        nSub_material: MaterialInputType,
                                        l_vec: Union[np.ndarray, List[float]],
                                        excel_file_path: str) -> Dict[str, np.ndarray]:
    l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64)
    ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)
    num_layers = len(ep_vector)
    if len(material_sequence) != num_layers:
        raise ValueError("Length of material_sequence must match ep_vector.")
    layer_indices_list = []
    for i, material_name in enumerate(material_sequence):
        try:
            nk_arr = get_nk_array(material_name, l_vec_jnp, excel_file_path)
            layer_indices_list.append(nk_arr)
        except Exception as e:
            st.error(f"Error preparing material '{material_name}' (layer {i+1}): {e}")
            raise
    if layer_indices_list:
        layer_indices_arr = jnp.stack(layer_indices_list, axis=0)
    else:
        layer_indices_arr = jnp.empty((0, len(l_vec_jnp)), dtype=jnp.complex128)
    try:
        nSub_arr = get_nk_array(nSub_material, l_vec_jnp, excel_file_path)
    except Exception as e:
        st.error(f"Error preparing substrate material: {e}")
        raise
    Ts = calculate_T_from_ep_arbitrary_core_jax(ep_vector_jnp, layer_indices_arr, nSub_arr, l_vec_jnp)
    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts)}
def _get_nk_at_lambda(material_definition: MaterialInputType, l_nm_target: float, excel_file_path: str) -> complex:
    if isinstance(material_definition, (complex, float, int)):
        return complex(material_definition)
    l_nm_target_jnp = jnp.array([l_nm_target], dtype=jnp.float64)
    if isinstance(material_definition, str):
        mat_upper = material_definition.upper()
        if mat_upper == "FUSED SILICA": return complex(get_n_fused_silica(l_nm_target_jnp)[0])
        if mat_upper == "BK7": return complex(get_n_bk7(l_nm_target_jnp)[0])
        if mat_upper == "D263": return complex(get_n_d263(l_nm_target_jnp)[0])
        else:
            sheet_name = material_definition
            l_data, n_data, k_data = load_material_data_from_xlsx_sheet(excel_file_path, sheet_name)
            if l_data is None: raise ValueError(f"Could not load data for '{sheet_name}'")
            n_interp = np.interp(l_nm_target, l_data, n_data)
            k_interp = np.interp(l_nm_target, l_data, k_data)
            return complex(n_interp, k_interp)
    elif isinstance(material_definition, tuple) and len(material_definition) == 3:
        l_data, n_data, k_data = material_definition
        n_interp = np.interp(l_nm_target, l_data, n_data)
        k_interp = np.interp(l_nm_target, l_data, k_data)
        return complex(n_interp, k_interp)
    else:
        raise TypeError(f"Unsupported material definition type: {type(material_definition)}")
@jax.jit
def get_target_points_indices_jax(l_vec: jnp.ndarray, target_min: float, target_max: float) -> jnp.ndarray:
    if not l_vec.size: return jnp.empty(0, dtype=jnp.int64)
    indices = jnp.where((l_vec >= target_min) & (l_vec <= target_max), size=l_vec.shape[0], fill_value=-1)[0]
    return indices[indices != -1]
def calculate_initial_ep(emp: Union[List[float], Tuple[float,...]], l0: float,
                         nH0_material: MaterialInputType, nL0_material: MaterialInputType, excel_file_path: str) -> np.ndarray:
    num_layers = len(emp)
    ep_initial = np.zeros(num_layers, dtype=np.float64)
    if l0 <= 0:
        log_message("Warning: l0 <= 0 in calculate_initial_ep. Thicknesses set to 0.")
        return ep_initial
    try:
        nH_complex_at_l0 = _get_nk_at_lambda(nH0_material, l0, excel_file_path)
        nL_complex_at_l0 = _get_nk_at_lambda(nL0_material, l0, excel_file_path)
        nH_real_at_l0 = nH_complex_at_l0.real
        nL_real_at_l0 = nL_complex_at_l0.real
        if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
            log_message(f"Warning: Real index nH({nH_real_at_l0:.3f}) or nL({nL_real_at_l0:.3f}) at l0={l0}nm is <= 0. QWOT calculation might be incorrect.")
    except Exception as e:
        log_message(f"Error getting indices at l0={l0}nm for initial calculation: {e}. Initial thicknesses set to 0.")
        return np.zeros(num_layers, dtype=np.float64)
    for i in range(num_layers):
        multiplier = emp[i]
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real_layer_at_l0 <= 1e-9:
            ep_initial[i] = 0.0
        else:
            ep_initial[i] = multiplier * l0 / (4.0 * n_real_layer_at_l0)
    return ep_initial
def calculate_qwot_from_ep(ep_vector: np.ndarray, l0: float,
                           nH0_material: MaterialInputType, nL0_material: MaterialInputType, excel_file_path: str) -> np.ndarray:
    num_layers = len(ep_vector)
    qwot_multipliers = np.full(num_layers, np.nan, dtype=np.float64)
    if l0 <= 0:
        log_message("Warning: l0 <= 0 in calculate_qwot_from_ep. QWOT set to NaN.")
        return qwot_multipliers
    try:
        nH_complex_at_l0 = _get_nk_at_lambda(nH0_material, l0, excel_file_path)
        nL_complex_at_l0 = _get_nk_at_lambda(nL0_material, l0, excel_file_path)
        nH_real_at_l0 = nH_complex_at_l0.real
        nL_real_at_l0 = nL_complex_at_l0.real
        if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
            log_message(f"Warning: Real index nH({nH_real_at_l0:.3f}) or nL({nL_real_at_l0:.3f}) at l0={l0}nm is <= 0. QWOT calculation might be incorrect.")
    except Exception as e:
        log_message(f"Error getting indices at l0={l0}nm for QWOT calculation: {e}. QWOT set to NaN.")
        return qwot_multipliers
    for i in range(num_layers):
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real_layer_at_l0 <= 1e-9:
            pass
        else:
            qwot_multipliers[i] = ep_vector[i] * (4.0 * n_real_layer_at_l0) / l0
    return qwot_multipliers
def calculate_final_mse(res: Dict[str, np.ndarray], active_targets: List[Dict]) -> Tuple[Union[float, None], int]:
    total_squared_error = 0.0
    total_points_in_targets = 0
    mse = None
    if not active_targets or 'Ts' not in res or res['Ts'] is None or len(res['Ts']) == 0 or 'l' not in res or res['l'] is None:
        return mse, total_points_in_targets
    res_l_np = np.asarray(res['l'])
    res_ts_np = np.asarray(res['Ts'])
    for target in active_targets:
        l_min, l_max = target['min'], target['max']
        t_min, t_max = target['target_min'], target['target_max']
        indices = np.where((res_l_np >= l_min) & (res_l_np <= l_max))[0]
        if indices.size > 0:
            valid_indices = indices[indices < len(res_ts_np)]
            if valid_indices.size == 0: continue
            calculated_Ts_in_zone = res_ts_np[valid_indices]
            target_lambdas_in_zone = res_l_np[valid_indices]
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
    elif active_targets:
        mse = np.nan
    return mse, total_points_in_targets
@jax.jit
def calculate_mse_for_optimization_penalized_jax(ep_vector: jnp.ndarray,
                                                 nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                                 l_vec_optim: jnp.ndarray,
                                                 active_targets_tuple: Tuple[Tuple[float, float, float, float], ...],
                                                 min_thickness_phys_nm: float) -> jnp.ndarray:
    below_min_mask = (ep_vector < min_thickness_phys_nm) & (ep_vector > 1e-12)
    penalty_thin = jnp.sum(jnp.where(below_min_mask, (min_thickness_phys_nm - ep_vector)**2, 0.0)) * 1e5
    ep_vector_calc = jnp.maximum(ep_vector, min_thickness_phys_nm)
    Ts = calculate_T_from_ep_core_jax(ep_vector_calc, nH_arr, nL_arr, nSub_arr, l_vec_optim)
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
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, 1e7)
    final_cost = mse + penalty_thin
    return jnp.nan_to_num(final_cost, nan=jnp.inf)
@jax.jit
def calculate_mse_arbitrary_sequence_jax(ep_vector: jnp.ndarray,
                                         layer_indices_arr: jnp.ndarray,
                                         nSub_arr: jnp.ndarray,
                                         l_vec_eval: jnp.ndarray,
                                         active_targets_tuple: Tuple[Tuple[float, float, float, float], ...]) -> jnp.ndarray:
    Ts = calculate_T_from_ep_arbitrary_core_jax(ep_vector, layer_indices_arr, nSub_arr, l_vec_eval)
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
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, jnp.inf)
    return jnp.nan_to_num(mse, nan=jnp.inf)
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
    ep1, ep2 = 0.0, 0.0
    denom = 4.0 * jnp.maximum(n_real_l0, 1e-9)
    safe_l0 = jnp.maximum(l0, 1e-9)
    ep1_calc = 1.0 * safe_l0 / denom
    ep2_calc = 2.0 * safe_l0 / denom
    ep1 = jnp.where(n_real_l0 > 1e-9, ep1_calc, 0.0)
    ep2 = jnp.where(n_real_l0 > 1e-9, ep2_calc, 0.0)
    ep1_eff = jnp.maximum(ep1, MIN_THICKNESS_PHYS_NM * (ep1 > 1e-12))
    ep2_eff = jnp.maximum(ep2, MIN_THICKNESS_PHYS_NM * (ep2 > 1e-12))
    M_1qwot_batch = calculate_M_batch_for_thickness(ep1_eff, n_complex_for_matrix, l_vec)
    M_2qwot_batch = calculate_M_batch_for_thickness(ep2_eff, n_complex_for_matrix, l_vec)
    return M_1qwot_batch, M_2qwot_batch
@jax.jit
def compute_half_product(multiplier_indices: jnp.ndarray,
                         layer_matrices_half: jnp.ndarray
                         ) -> jnp.ndarray:
    N_half = layer_matrices_half.shape[0]; L = layer_matrices_half.shape[2]
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
    m00, m01 = M_batch[:, 0, 0], M_batch[:, 0, 1]
    m10, m11 = M_batch[:, 1, 0], M_batch[:, 1, 1]
    rs_den = (etainc * m00 + etasub_batch * m11 + etainc * etasub_batch * m01 + m10)
    rs_den_abs = jnp.abs(rs_den)
    safe_den = jnp.where(rs_den_abs < 1e-12, 1e-12 + 0j, rs_den)
    ts = (2.0 * etainc) / safe_den
    real_etasub_batch = jnp.real(etasub_batch)
    real_etainc = 1.0
    Ts_complex = (real_etasub_batch / real_etainc) * (ts * jnp.conj(ts))
    Ts = jnp.real(Ts_complex)
    return jnp.where(rs_den_abs < 1e-12, 0.0, jnp.nan_to_num(Ts, nan=0.0))
@jax.jit
def calculate_mse_basic_jax(Ts: jnp.ndarray, l_vec: jnp.ndarray, targets_tuple: Tuple) -> jnp.ndarray:
    total_squared_error = 0.0; total_points_in_targets = 0
    for i in range(len(targets_tuple)):
        l_min, l_max, t_min, t_max = targets_tuple[i]
        target_mask = (l_vec >= l_min) & (l_vec <= l_max)
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t = t_min + slope * (l_vec - l_min)
        squared_errors = (Ts - interpolated_target_t)**2
        masked_sq_error = jnp.where(target_mask, squared_errors, 0.0)
        total_squared_error += jnp.sum(masked_sq_error)
        total_points_in_targets += jnp.sum(target_mask)
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, jnp.inf)
    return jnp.nan_to_num(mse, nan=jnp.inf)
@jax.jit
def combine_and_calc_mse(prod1: jnp.ndarray, prod2: jnp.ndarray,
                         nSub_arr_in: jnp.ndarray,
                         l_vec_in: jnp.ndarray, targets_tuple_in: Tuple
                         ) -> jnp.ndarray:
    M_total = vmap(jnp.matmul)(prod2, prod1)
    Ts = get_T_from_batch_matrix(M_total, nSub_arr_in)
    mse = calculate_mse_basic_jax(Ts, l_vec_in, targets_tuple_in)
    return mse
if 'initialized' not in st.session_state:
    print("Initializing Streamlit Session State...")
    st.session_state.initialized = True
    st.session_state.current_optimized_ep = None
    st.session_state.current_material_sequence = None
    st.session_state.optimization_ran_since_nominal_change = False
    st.session_state.ep_history_deque = deque(maxlen=5)
    st.session_state.log_messages = ["Application started."]
    st.session_state.status_message = "Ready"
    st.session_state.available_materials = []
    st.session_state.available_substrates = ["Constant", "Fused Silica", "BK7", "D263"]
    st.session_state.combined_substrates = []
    st.session_state.default_l0 = 500.0
    st.session_state.default_l_step = 10.0
    st.session_state.default_maxiter = 1000
    st.session_state.default_maxfun = 1000
    st.session_state.default_nH_r = 2.35
    st.session_state.default_nH_i = 0.0
    st.session_state.default_nL_r = 1.46
    st.session_state.default_nL_i = 0.0
    st.session_state.default_nSub = 1.52
    st.session_state.default_initial_layer_number = 20
    st.session_state.default_emp_str = ",".join(['1'] * st.session_state.default_initial_layer_number)
    st.session_state.default_auto_thin_threshold = 1.0
    st.session_state.selected_H_material = "Constant"
    st.session_state.selected_L_material = "Constant"
    st.session_state.selected_Sub_material = "Fused Silica"
    st.session_state.targets = [
        {'min': 400.0, 'max': 500.0, 'target_min': 1.0, 'target_max': 1.0, 'enabled': True},
        {'min': 500.0, 'max': 600.0, 'target_min': 1.0, 'target_max': 0.2, 'enabled': True},
        {'min': 600.0, 'max': 700.0, 'target_min': 0.2, 'target_max': 0.2, 'enabled': True},
        {'min': 700.0, 'max': 800.0, 'target_min': 0.2, 'target_max': 0.8, 'enabled': True},
        {'min': 800.0, 'max': 900.0, 'target_min': 0.8, 'target_max': 0.8, 'enabled': True}
    ]
    st.session_state.optimized_qwot_string_display = ""
    st.session_state.last_plot_fig = None
    st.session_state.material_index_plot_fig = None
def log_message(message: str):
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    st.session_state.log_messages.append(full_message)
    max_log_lines = 500
    if len(st.session_state.log_messages) > max_log_lines:
        st.session_state.log_messages = st.session_state.log_messages[-max_log_lines:]
def set_status(message: str):
    st.session_state.status_message = f"Status: {message}"
if not st.session_state.available_materials:
    log_message(f"Loading available materials from: {EXCEL_FILE_PATH}")
    if os.path.exists(EXCEL_FILE_PATH):
        excel_materials = get_available_materials_from_excel(EXCEL_FILE_PATH)
        st.session_state.available_materials = sorted(list(set(["Constant"] + excel_materials)))
        st.session_state.combined_substrates = sorted(list(set(st.session_state.available_substrates + excel_materials)))
        log_message(f"Available H/L materials: {st.session_state.available_materials}")
        log_message(f"Available Substrate materials: {st.session_state.combined_substrates}")
        desired_default_H = "Nb2O5-Helios"
        desired_default_L = "SiO2-Helios"
        fallback_default = "Constant"
        avail_mats = st.session_state.available_materials
        if desired_default_H in avail_mats: default_H = desired_default_H
        else: default_H = next((mat for mat in avail_mats if mat != fallback_default), fallback_default)
        if default_H != desired_default_H: log_message(f"Warning: Desired default H '{desired_default_H}' not found. Using '{default_H}'.")
        if desired_default_L in avail_mats: default_L = desired_default_L
        else:
            non_constant_non_H = [mat for mat in avail_mats if mat != fallback_default and mat != default_H]
            if non_constant_non_H: default_L = non_constant_non_H[0]
            elif default_H != fallback_default: default_L = fallback_default
            else: default_L = next((mat for mat in avail_mats if mat != fallback_default), fallback_default)
        if default_L != desired_default_L: log_message(f"Warning: Desired default L '{desired_default_L}' not found. Using '{default_L}'.")
        st.session_state.selected_H_material = default_H
        st.session_state.selected_L_material = default_L
        if "Fused Silica" not in st.session_state.combined_substrates:
            st.session_state.selected_Sub_material = "Constant"
    else:
        log_message(f"ERROR: Material file '{EXCEL_FILE_PATH}' not found. Using 'Constant' only.")
        st.warning(f"Material definition file '{EXCEL_FILE_PATH}' not found. Only 'Constant' materials available.")
        st.session_state.available_materials = ["Constant"]
        st.session_state.combined_substrates = ["Constant", "Fused Silica", "BK7", "D263"]
st.title("Thin Film Stack Optimizer (Streamlit / JAX)")
with st.sidebar:
    st.header("Inputs & Controls")
    with st.expander("📂 File Operations", expanded=False):
        uploaded_file = st.file_uploader("Load Design (.json)", type="json", key="load_design_uploader") # Corrected Key Usage
        save_nominal_placeholder = st.empty()
        save_optimized_placeholder = st.empty()
        save_nominal_placeholder.button("Save Nominal Design", key="save_nominal_btn_ph", disabled=True, help="Save functionality needs implementation")
        save_optimized_placeholder.button("Save Optimized Design", key="save_opt_btn_ph", disabled=True, help="Save functionality needs implementation")
    with st.expander("🔬 Materials & Substrate", expanded=True):
        m_col1, m_col2, m_col3 = st.columns([2,1,1])
        with m_col1:
             selected_H = st.selectbox(
                  "H Material:", st.session_state.available_materials,
                  index=st.session_state.available_materials.index(st.session_state.selected_H_material) if st.session_state.selected_H_material in st.session_state.available_materials else 0,
                  key='selected_H_material_widget'
             )
             selected_L = st.selectbox(
                  "L Material:", st.session_state.available_materials,
                  index=st.session_state.available_materials.index(st.session_state.selected_L_material) if st.session_state.selected_L_material in st.session_state.available_materials else 0,
                  key='selected_L_material_widget'
             )
             selected_Sub = st.selectbox(
                  "Substrate:", st.session_state.combined_substrates,
                  index=st.session_state.combined_substrates.index(st.session_state.selected_Sub_material) if st.session_state.selected_Sub_material in st.session_state.combined_substrates else 0,
                  key='selected_Sub_material_widget'
             )
             st.caption("(n = n' + ik)")
             st.session_state.selected_H_material = selected_H
             st.session_state.selected_L_material = selected_L
             st.session_state.selected_Sub_material = selected_Sub
        enable_H_const = (st.session_state.selected_H_material == "Constant")
        enable_L_const = (st.session_state.selected_L_material == "Constant")
        enable_Sub_const = (st.session_state.selected_Sub_material == "Constant")
        with m_col2:
            st.session_state.input_nH_r = st.number_input("n' (H)", value=st.session_state.default_nH_r, key='nH_r', format="%.4f", step=0.01, disabled=not enable_H_const)
            st.session_state.input_nL_r = st.number_input("n' (L)", value=st.session_state.default_nL_r, key='nL_r', format="%.4f", step=0.01, disabled=not enable_L_const)
            st.session_state.input_nSub = st.number_input("n' (Sub)", value=st.session_state.default_nSub, key='nSub', format="%.4f", step=0.01, disabled=not enable_Sub_const)
        with m_col3:
            st.session_state.input_nH_i = st.number_input("k (H)", value=st.session_state.default_nH_i, min_value=0.0, key='nH_i', format="%.4f", step=0.001, disabled=not enable_H_const)
            st.session_state.input_nL_i = st.number_input("k (L)", value=st.session_state.default_nL_i, min_value=0.0, key='nL_i', format="%.4f", step=0.001, disabled=not enable_L_const)
        if st.button("Plot Selected Material Indices", key="plot_indices_btn"): # Corrected Key Usage
             st.session_state.show_material_plot = True
    with st.expander("🧱 Stack Definition", expanded=True):
         st.session_state.input_initial_layer_number = st.number_input(
             "Initial Layer Number (for Scan/QWOT Gen):", min_value=0,
             value=st.session_state.default_initial_layer_number, step=2,
             key='initial_layer_number_widget', help="Number of layers (alternating H/L) used for 'Start Nom.' or to generate initial QWOT string."
         )
         if st.button("Generate QWOT from Layer Number", key="generate_qwot_btn"): # Corrected Key Usage
              num_layers = st.session_state.input_initial_layer_number
              if num_layers > 0: st.session_state.input_emp_str = ",".join(['1'] * num_layers)
              else: st.session_state.input_emp_str = ""
              log_message(f"Generated QWOT string for {num_layers} layers.")
         st.session_state.input_emp_str = st.text_area(
             "Nominal QWOT Multipliers (comma-separated):", value=st.session_state.get('input_emp_str', st.session_state.default_emp_str),
             key='emp_str_widget', height=100, help="Enter comma-separated Quarter-Wave Optical Thickness multipliers (e.g., 1,1,1.5,1)"
         )
         st.session_state.input_l0 = st.number_input(
             "Centering λ (QWOT, nm):", min_value=1.0, value=st.session_state.default_l0,
             key='l0_widget', format="%.1f", step=10.0
         )
         st.text_input("Optimized QWOT:", value=st.session_state.optimized_qwot_string_display, disabled=True, key="opt_qwot_display")
         if st.button("Info: Copy Optimized QWOT", key="copy_opt_qwot_btn_info", disabled=(not st.session_state.optimization_ran_since_nominal_change)):
             log_message(f"Info: Optimized QWOT: {st.session_state.optimized_qwot_string_display}")
             st.info(f"Optimized QWOT:\n```\n{st.session_state.optimized_qwot_string_display}\n```\n(Please copy manually)")
    with st.expander("⚙️ Calculation & Optimization Parameters", expanded=True):
         p_col1, p_col2 = st.columns(2)
         with p_col1:
              st.session_state.input_l_step = st.number_input(
                  "λ Step (Optim/Plot, nm):", min_value=0.01, value=st.session_state.default_l_step,
                  key='l_step_widget', format="%.2f", step=1.0, help="Wavelength step for optimization grid and high-res plot generation."
              )
              st.session_state.input_maxiter = st.number_input(
                  "Max Iter (Opt):", min_value=1, value=st.session_state.default_maxiter,
                  step=100, key='maxiter_widget'
              )
         with p_col2:
             st.session_state.input_auto_thin_threshold = st.number_input(
                 "Thin Threshold (Auto Mode, nm):", min_value=MIN_THICKNESS_PHYS_NM, value=st.session_state.default_auto_thin_threshold,
                 key='auto_thin_threshold_widget', format="%.3f", step=0.1, help="Thickness below which layers are considered for removal in Auto Mode."
            )
             st.session_state.input_maxfun = st.number_input(
                  "Max Eval (Opt):", min_value=1, value=st.session_state.default_maxfun,
                  step=100, key='maxfun_widget'
             )
    with st.expander("🎯 Spectral Targets (Transmittance T)", expanded=True):
         st.caption("Define target T values over wavelength ranges. Ensure ranges don't overlap excessively.")
         cols = st.columns([0.5, 1.5, 1.5, 1.5, 1.5])
         cols[0].markdown("**Active**")
         cols[1].markdown("**λ Min (nm)**")
         cols[2].markdown("**λ Max (nm)**")
         cols[3].markdown("**T @ λ Min**")
         cols[4].markdown("**T @ λ Max**")
         if 'targets' not in st.session_state: st.session_state.targets = []
         num_target_rows = 5
         active_target_list_st = []
         target_range_error = False
         for i in range(num_target_rows):
             if i >= len(st.session_state.targets):
                 st.session_state.targets.append({'min': 0.0, 'max': 0.0, 'target_min': 0.0, 'target_max': 0.0, 'enabled': False})
             cols = st.columns([0.5, 1.5, 1.5, 1.5, 1.5])
             current_target = st.session_state.targets[i]
             enabled = cols[0].checkbox(f"##{i}", value=current_target['enabled'], key=f"target_enable_{i}")
             l_min = cols[1].number_input(f"##lmin{i}", value=current_target['min'], key=f"target_lmin_{i}", step=10.0, format="%.1f", label_visibility="collapsed")
             l_max = cols[2].number_input(f"##lmax{i}", value=current_target['max'], key=f"target_lmax_{i}", step=10.0, format="%.1f", label_visibility="collapsed")
             t_min = cols[3].number_input(f"##tmin{i}", value=current_target['target_min'], min_value=0.0, max_value=1.0, key=f"target_tmin_{i}", step=0.05, format="%.3f", label_visibility="collapsed")
             t_max = cols[4].number_input(f"##tmax{i}", value=current_target['target_max'], min_value=0.0, max_value=1.0, key=f"target_tmax_{i}", step=0.05, format="%.3f", label_visibility="collapsed")
             st.session_state.targets[i] = {'min': l_min, 'max': l_max, 'target_min': t_min, 'target_max': t_max, 'enabled': enabled}
             if enabled:
                  valid_target = True
                  if l_min <= 0 or l_max <= l_min: valid_target = False
                  if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0): valid_target = False
                  if valid_target: active_target_list_st.append(st.session_state.targets[i])
                  else: target_range_error = True
         if target_range_error: st.warning("One or more active targets have invalid ranges.")
    st.header("Actions")
    action_cols = st.columns(2)
    eval_nom_pressed = action_cols[0].button("📊 Evaluate Nominal", key="eval_nom_btn")
    local_opt_pressed = action_cols[0].button("✨ Local Optimizer", key="local_opt_btn")
    start_nom_scan_pressed = action_cols[0].button("🔍 Start Nom. (Scan+Opt)", key="start_nom_scan_btn")
    auto_mode_pressed = action_cols[0].button("🤖 Auto Mode (Needle>Thin>Opt)", key="auto_mode_btn")
    remove_thin_pressed = action_cols[1].button("🗑️ Remove Thinnest Layer & Re-Opt", key="remove_thin_btn", disabled=(not st.session_state.optimization_ran_since_nominal_change))
    undo_remove_pressed = action_cols[1].button("↩️ Undo Remove Layer", key="undo_remove_btn", disabled=(len(st.session_state.ep_history_deque) == 0))
    set_nominal_pressed = action_cols[1].button("➡️ Set Optimized as Nominal", key="set_nominal_btn", disabled=(not st.session_state.optimization_ran_since_nominal_change))
    clear_opt_state_pressed = action_cols[1].button("❌ Clear Optimized State", key="clear_opt_state_btn")
def get_active_targets_st() -> Union[List[Dict], None]:
    active_targets = []
    validation_passed = True
    for i, target_def in enumerate(st.session_state.targets):
        if target_def['enabled']:
            try:
                l_min = float(target_def['min'])
                l_max = float(target_def['max'])
                t_min = float(target_def['target_min'])
                t_max = float(target_def['target_max'])
                if l_max < l_min:
                    log_message(f"ERROR Target {i+1}: λ max < λ min.")
                    validation_passed = False
                elif not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0):
                    log_message(f"ERROR Target {i+1}: Transmittance (T) must be between 0 and 1.")
                    validation_passed = False
                else:
                    active_targets.append({'min': l_min, 'max': l_max, 'target_min': t_min, 'target_max': t_max})
            except ValueError as e:
                log_message(f"ERROR Target {i+1} Configuration: Invalid number format ({e})")
                validation_passed = False
            except KeyError:
                 log_message(f"ERROR Target {i+1} Configuration: Missing key in state.")
                 validation_passed = False
    if not validation_passed:
        st.error("Invalid target definition(s) found. Please check ranges and values.")
        return None
    if not active_targets:
        log_message("Warning: No active targets defined or enabled.")
        return []
    return active_targets
def get_lambda_range_from_targets_st() -> Tuple[Union[float, None], Union[float, None]]:
    overall_min, overall_max = None, None
    active_targets = []
    for target_def in st.session_state.targets:
         if target_def['enabled']:
             try:
                 l_min = float(target_def['min'])
                 l_max = float(target_def['max'])
                 if l_max >= l_min > 0:
                      active_targets.append({'min': l_min, 'max': l_max})
             except (ValueError, KeyError):
                 pass
    if active_targets:
        all_mins = [t['min'] for t in active_targets]
        all_maxs = [t['max'] for t in active_targets]
        if all_mins: overall_min = min(all_mins)
        if all_maxs: overall_max = max(all_maxs)
    return overall_min, overall_max
def validate_physical_inputs_st(require_optim_params: bool = False, require_initial_layers: bool = False) -> Union[Dict[str, Any], None]:
    values = {}
    errors = []
    param_map = {
        'l0': ('input_l0', float), 'l_step': ('input_l_step', float),
        'maxiter': ('input_maxiter', int), 'maxfun': ('input_maxfun', int),
        'initial_layer_number': ('input_initial_layer_number', int),
        'auto_thin_threshold': ('input_auto_thin_threshold', float)
    }
    required_base = ['l0', 'l_step']
    if require_optim_params: required_base.extend(['maxiter', 'maxfun'])
    if require_initial_layers: required_base.append('initial_layer_number')
    for key in required_base:
        state_key, type_func = param_map[key]
        try: values[key] = type_func(st.session_state[state_key])
        except (KeyError, ValueError, TypeError): errors.append(f"Invalid or missing value for '{key}'.")
    try: values['emp_str'] = st.session_state.input_emp_str
    except KeyError: errors.append("Missing QWOT string ('emp_str').")
    try:
        selected_h = st.session_state.selected_H_material
        selected_l = st.session_state.selected_L_material
        selected_sub = st.session_state.selected_Sub_material
        if selected_h == "Constant":
            values['nH_material'] = complex(st.session_state.input_nH_r, st.session_state.input_nH_i)
            if values['nH_material'].real <= 0: errors.append("Real part nH must be > 0.")
            if values['nH_material'].imag < 0: errors.append("Imaginary part k H must be >= 0.")
        else: values['nH_material'] = selected_h
        if selected_l == "Constant":
            values['nL_material'] = complex(st.session_state.input_nL_r, st.session_state.input_nL_i)
            if values['nL_material'].real <= 0: errors.append("Real part nL must be > 0.")
            if values['nL_material'].imag < 0: errors.append("Imaginary part k L must be >= 0.")
        else: values['nL_material'] = selected_l
        if selected_sub == "Constant":
            values['nSub_material'] = complex(st.session_state.input_nSub, 0.0)
            if values['nSub_material'].real <= 0: errors.append("Real index nSub must be > 0.")
        else: values['nSub_material'] = selected_sub
    except (KeyError, ValueError, TypeError): errors.append("Error reading material parameters.")
    l_min, l_max = get_lambda_range_from_targets_st()
    if l_min is None or l_max is None: errors.append("Cannot determine calculation range. Activate/define valid target(s).")
    elif l_max < l_min: errors.append("Invalid target range: λ max < λ min.")
    else: values['l_range_deb'], values['l_range_fin'] = l_min, l_max
    if 'l_step' in values and values['l_step'] <= 0: errors.append("λ step (l_step) must be > 0.")
    if 'l0' in values and values['l0'] <= 0: errors.append("Centering λ (l0) must be > 0.")
    if require_optim_params:
        if 'maxiter' in values and values['maxiter'] <= 0: errors.append("Optimizer maxiter must be > 0.")
        if 'maxfun' in values and values['maxfun'] <= 0: errors.append("Optimizer maxfun must be > 0.")
    if require_initial_layers:
        if 'initial_layer_number' in values and values['initial_layer_number'] <= 0: errors.append("Initial layer number must be > 0.")
    if 'auto_thin_threshold' not in values and 'auto_thin_threshold' in param_map:
         try: values['auto_thin_threshold'] = param_map['auto_thin_threshold'][1](st.session_state[param_map['auto_thin_threshold'][0]])
         except (KeyError, ValueError, TypeError): values['auto_thin_threshold'] = st.session_state.default_auto_thin_threshold
    if errors:
        for err in errors: st.error(err)
        log_message(f"Input validation failed: {'; '.join(errors)}")
        return None
    return values
def prepare_calculation_data_st(inputs: Dict[str, Any], ep_vector_to_use: Union[np.ndarray, None] = None) -> Tuple[np.ndarray, np.ndarray, MaterialInputType, MaterialInputType, MaterialInputType]:
    nH_material = inputs['nH_material']
    nL_material = inputs['nL_material']
    nSub_material = inputs['nSub_material']
    l0 = inputs['l0']
    l_min_plot, l_max_plot = inputs['l_range_deb'], inputs['l_range_fin']
    l_step_base = inputs['l_step']
    num_plot_points = max(500, int(np.round((l_max_plot - l_min_plot) / l_step_base)) * 5)
    l_vec_plot = np.linspace(l_min_plot, l_max_plot, num_plot_points)
    l_vec_plot = l_vec_plot[(l_vec_plot > 0) & np.isfinite(l_vec_plot)]
    if not l_vec_plot.size: raise ValueError("Lambda range/step generates no points for plotting.")
    ep_actual = None
    if ep_vector_to_use is not None:
        ep_actual = np.asarray(ep_vector_to_use, dtype=np.float64)
    else:
        try:
            emp_list = [float(e.strip()) for e in inputs['emp_str'].split(',') if e.strip()]
            if not emp_list and inputs['emp_str'].strip():
                 raise ValueError("QWOT string contains non-numeric characters or only separators.")
            ep_actual_calc = calculate_initial_ep(emp_list, l0, nH_material, nL_material, EXCEL_FILE_PATH)
            ep_actual = np.asarray(ep_actual_calc)
        except Exception as e:
            raise ValueError(f"Could not calculate initial thicknesses from QWOT: {e}")
    if ep_actual.size == 0:
        if inputs['emp_str'].strip() and ep_vector_to_use is None:
             raise ValueError("Nominal QWOT results in an empty structure.")
        else:
             log_message("Empty structure detected. Calculating for bare substrate.")
             ep_actual = np.array([], dtype=np.float64)
    elif not np.all(np.isfinite(ep_actual)):
        ep_actual = np.nan_to_num(ep_actual, nan=0.0, posinf=0.0, neginf=0.0)
        log_message("WARNING: Thicknesses contained NaN/inf, replaced with 0.")
        if not np.all(np.isfinite(ep_actual)):
             raise ValueError("Thickness vector contains uncorrectable non-finite values.")
    return l_vec_plot, ep_actual, nH_material, nL_material, nSub_material
def display_results_plot(res: Dict, current_ep: np.ndarray,
                         nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
                         active_targets_for_plot: List[Dict], mse: Union[float, None],
                         is_optimized: bool = False, method_name: str = "",
                         res_optim_grid: Union[Dict, None] = None,
                         material_sequence: Union[List[str], None] = None,
                         l0_for_index_plot: float = 500.0) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    opt_method_str = f" ({method_name})" if method_name else ""
    window_title = f'Results {"Optimized" if is_optimized else "Nominal"}{opt_method_str}'
    fig.suptitle(window_title, fontsize=14, weight='bold')
    num_layers = len(current_ep) if current_ep is not None else 0
    ep_cumulative = np.cumsum(current_ep) if num_layers > 0 else np.array([])
    ax_spec = axes[0]
    line_ts = None
    if res and 'l' in res and 'Ts' in res and res['l'] is not None and len(res['l']) > 0:
        res_l_plot = np.asarray(res['l'])
        res_ts_plot = np.asarray(res['Ts'])
        line_ts, = ax_spec.plot(res_l_plot, res_ts_plot, label='Transmittance', linestyle='-', color='blue', linewidth=2.0)
        plotted_target_label = False
        if active_targets_for_plot:
            for target in active_targets_for_plot:
                l_min, l_max = target['min'], target['max']
                t_min, t_max = target['target_min'], target['target_max']
                x_coords, y_coords = [l_min, l_max], [t_min, t_max]
                label = 'Target (Ramp)' if not plotted_target_label else "_nolegend_"
                ax_spec.plot(x_coords, y_coords, 'r--', linewidth=1.5, alpha=0.8, label=label, zorder=5)
                ax_spec.plot(x_coords, y_coords, marker='x', color='red', markersize=8, linestyle='none', label='_nolegend_', zorder=6)
                plotted_target_label = True
                if res_optim_grid and 'l' in res_optim_grid and res_optim_grid['l'].size > 0:
                     res_l_optim = np.asarray(res_optim_grid['l'])
                     indices_optim = np.where((res_l_optim >= l_min) & (res_l_optim <= l_max))[0]
                     if indices_optim.size > 0:
                         optim_lambdas = res_l_optim[indices_optim]
                         if abs(l_max - l_min) < 1e-9: optim_target_t = np.full_like(optim_lambdas, t_min)
                         else: slope = (t_max - t_min) / (l_max - l_min); optim_target_t = t_min + slope * (optim_lambdas - l_min)
                         ax_spec.plot(optim_lambdas, optim_target_t, marker='.', color='darkred', linestyle='none', markersize=5, alpha=0.7, label='_nolegend_', zorder=6)
        ax_spec.set_xlabel("Wavelength (nm)")
        ax_spec.set_ylabel('Transmittance')
        ax_spec.set_title(f"Spectrum{opt_method_str}")
        ax_spec.grid(which='major', color='grey', linestyle='-', linewidth=0.7)
        ax_spec.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)
        ax_spec.minorticks_on()
        if len(res_l_plot) > 0: ax_spec.set_xlim(res_l_plot[0], res_l_plot[-1])
        ax_spec.set_ylim(-0.05, 1.05)
        if plotted_target_label or (ax_spec.get_legend_handles_labels() and ax_spec.get_legend_handles_labels()[1]):
            ax_spec.legend(fontsize=9)
        if mse is not None and not np.isnan(mse) and mse != -1 : mse_text = f"MSE (vs Target) = {mse:.3e}"
        elif mse == -1: mse_text = "MSE: N/A (No Re-opt)"
        elif mse is None and active_targets_for_plot: mse_text = "MSE: Calculation Error"
        elif mse is None: mse_text = "MSE: N/A (no target)"
        else: mse_text = "MSE: N/A (no pts)"
        ax_spec.text(0.98, 0.98, mse_text, transform=ax_spec.transAxes, ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
    else:
        ax_spec.text(0.5, 0.5, "No spectral data", ha='center', va='center', transform=ax_spec.transAxes)
        ax_spec.set_title(f"Spectrum{opt_method_str}")
    ax_idx = axes[1]
    l0_repr = l0_for_index_plot
    try:
        nH_c_repr = _get_nk_at_lambda(nH_material, l0_repr, EXCEL_FILE_PATH)
        nL_c_repr = _get_nk_at_lambda(nL_material, l0_repr, EXCEL_FILE_PATH)
        nSub_c_repr = _get_nk_at_lambda(nSub_material, l0_repr, EXCEL_FILE_PATH)
        nH_r_repr, nL_r_repr, nSub_r_repr = nH_c_repr.real, nL_c_repr.real, nSub_c_repr.real
    except Exception as e:
        log_message(f"Error getting indices at l0={l0_repr}nm for index plot: {e}")
        nH_r_repr, nL_r_repr, nSub_r_repr = (2.0, 1.5, 1.5); nSub_c_repr = complex(nSub_r_repr, 0)
    if material_sequence and len(material_sequence) == num_layers:
        n_real_layers_repr = []
        for mat_name in material_sequence:
            try: n_real_layers_repr.append(_get_nk_at_lambda(mat_name, l0_repr, EXCEL_FILE_PATH).real)
            except Exception: n_real_layers_repr.append(np.nan)
    else:
        n_real_layers_repr = [nH_r_repr if i % 2 == 0 else nL_r_repr for i in range(num_layers)]
    total_thickness = ep_cumulative[-1] if num_layers > 0 else 0
    margin = max(50, 0.1 * total_thickness) if total_thickness > 0 else 50
    x_coords_plot = [-margin]; y_coords_plot = [nSub_r_repr]
    if num_layers > 0:
        x_coords_plot.append(0); y_coords_plot.append(nSub_r_repr)
        for i in range(num_layers):
            layer_start = ep_cumulative[i-1] if i > 0 else 0
            layer_end = ep_cumulative[i]
            layer_n_real = n_real_layers_repr[i] if i < len(n_real_layers_repr) else np.nan
            x_coords_plot.extend([layer_start, layer_end])
            y_coords_plot.extend([layer_n_real, layer_n_real])
        last_layer_end = ep_cumulative[-1]
        x_coords_plot.extend([last_layer_end, last_layer_end + margin])
        y_coords_plot.extend([1.0, 1.0])
    else:
        x_coords_plot.extend([0, 0, margin]); y_coords_plot.extend([nSub_r_repr, 1.0, 1.0])
    ax_idx.plot(x_coords_plot, y_coords_plot, drawstyle='steps-post', label=f'n\'(λ={l0_repr:.0f}nm)', color='purple', linewidth=1.5)
    ax_idx.set_xlabel('Depth (from substrate) (nm)')
    ax_idx.set_ylabel("Real Part of Index (n')")
    ax_idx.set_title(f"Index Profile (at λ={l0_repr:.0f}nm)")
    ax_idx.grid(True, linestyle=':')
    ax_idx.set_xlim(x_coords_plot[0], x_coords_plot[-1])
    min_n_list = [1.0, nSub_r_repr] + [n for n in n_real_layers_repr if not np.isnan(n)]
    max_n_list = [1.0, nSub_r_repr] + [n for n in n_real_layers_repr if not np.isnan(n)]
    min_n = min(min_n_list) if min_n_list else 0.9; max_n = max(max_n_list) if max_n_list else 2.5
    ax_idx.set_ylim(bottom=min_n - 0.1, top=max_n + 0.1)
    offset = (max_n - min_n) * 0.05 + 0.02
    common_text_opts = {'ha':'center', 'va':'bottom', 'fontsize':8, 'bbox':dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none')}
    n_sub_label = f"{nSub_c_repr.real:.3f}{nSub_c_repr.imag:+.3f}j" if abs(nSub_c_repr.imag) > 1e-6 else f"{nSub_c_repr.real:.3f}"
    ax_idx.text(-margin / 2, nSub_r_repr + offset, f"SUBSTRATE\nn={n_sub_label} @{l0_repr:.0f}nm", **common_text_opts)
    air_x_pos = (total_thickness + margin / 2) if num_layers > 0 else margin / 2
    ax_idx.text(air_x_pos, 1.0 + offset, "AIR\nn=1.0", **common_text_opts)
    if ax_idx.get_legend_handles_labels()[1]: ax_idx.legend(fontsize=8, loc='lower right')
    ax_stack = axes[2]
    if num_layers > 0:
        indices_complex_repr = []
        if material_sequence and len(material_sequence) == num_layers:
            for mat_name in material_sequence:
                try: indices_complex_repr.append(_get_nk_at_lambda(mat_name, l0_repr, EXCEL_FILE_PATH))
                except Exception: indices_complex_repr.append(complex(np.nan, np.nan))
        else:
            indices_complex_repr = [nH_c_repr if i % 2 == 0 else nL_c_repr for i in range(num_layers)]
        if material_sequence:
             colors = ['lightblue' if (i == num_layers - 1 or np.isnan(indices_complex_repr[i+1].real) or indices_complex_repr[i].real > indices_complex_repr[i+1].real) else 'lightcoral' for i in range(num_layers)]
        else:
             colors = ['lightblue' if i % 2 == 0 else 'lightcoral' for i in range(num_layers)]
        bar_pos = np.arange(num_layers)
        bars = ax_stack.barh(bar_pos, current_ep, align='center', color=colors, edgecolor='grey', height=0.8)
        yticks_labels = []
        for i, n_comp_repr in enumerate(indices_complex_repr):
            if material_sequence: layer_type = material_sequence[i][:6]
            else: layer_type = "H" if i % 2 == 0 else "L"
            n_str = f"{n_comp_repr.real:.3f}" if not np.isnan(n_comp_repr.real) else "N/A"
            k_val = n_comp_repr.imag
            if not np.isnan(k_val) and abs(k_val) > 1e-6: n_str += f"{k_val:+.3f}j"
            yticks_labels.append(f"L{i + 1} ({layer_type}) n≈{n_str}")
        ax_stack.set_yticks(bar_pos)
        ax_stack.set_yticklabels(yticks_labels, fontsize=8)
        ax_stack.invert_yaxis()
        max_ep = max(current_ep) if current_ep.size > 0 else 1.0
        fontsize = max(6, 9 - num_layers // 10)
        for i, e_val in enumerate(current_ep):
            ha_pos = 'left' if e_val < max_ep * 0.2 else 'right'
            x_text_pos = e_val * 1.05 if ha_pos == 'left' else e_val * 0.95
            text_color = 'black' if ha_pos == 'left' else 'white'
            ax_stack.text(x_text_pos, i, f"{e_val:.2f} nm", va='center', ha=ha_pos, color=text_color, fontsize=fontsize, weight='bold')
    else:
        ax_stack.text(0.5, 0.5, "No layers", ha='center', va='center', fontsize=10, color='grey', transform=ax_stack.transAxes)
        ax_stack.set_yticks([]); ax_stack.set_xticks([])
    ax_stack.set_xlabel('Thickness (nm)')
    stack_title_prefix = f'Stack {"Optimized" if is_optimized else "Nominal"}{opt_method_str}'
    ax_stack.set_title(f"{stack_title_prefix} ({num_layers} layers)")
    if num_layers > 0: ax_stack.set_ylim(bottom=num_layers - 0.5, top=-0.5)
    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.95])
    return fig
def display_material_index_plot() -> Union[plt.Figure, None]:
    materials_to_plot = {}
    selections = {
        "H": st.session_state.selected_H_material,
        "L": st.session_state.selected_L_material,
        "Sub": st.session_state.selected_Sub_material
    }
    at_least_one_non_constant = False
    for role, name in selections.items():
        if name != "Constant":
            at_least_one_non_constant = True
            log_message(f"Plot Indices: Loading/Generating data for '{name}'...")
            try:
                 if name.upper() in ["FUSED SILICA", "BK7", "D263"]:
                     l_plot_gen = np.linspace(300, 1500, 200)
                     nk_complex_gen = _get_nk_array_for_lambda_vec_cached(name, tuple(l_plot_gen), EXCEL_FILE_PATH)
                     n_data_gen, k_data_gen = np.real(nk_complex_gen), np.imag(nk_complex_gen)
                     data_tuple = (l_plot_gen, n_data_gen, k_data_gen)
                 else:
                     l_data, n_data, k_data = load_material_data_from_xlsx_sheet(EXCEL_FILE_PATH, name)
                     if l_data is None: raise ValueError("Load failed")
                     data_tuple = (l_data, n_data, k_data)
                 materials_to_plot[f"{role}: {name}"] = data_tuple
            except Exception as e:
                 log_message(f"ERROR Plot Indices: Cannot load/generate data for '{name}': {e}")
    if not at_least_one_non_constant:
        st.info("No non-constant materials selected to plot.")
        return None
    fig_idx, ax_idx = plt.subplots(figsize=(7, 5))
    min_l, max_l, min_n, max_n = np.inf, -np.inf, np.inf, -np.inf
    plotted_something = False
    for label, (l_data, n_data, k_data) in materials_to_plot.items():
        if l_data is not None and n_data is not None and len(l_data) > 0:
             valid_k_mask = (k_data >= -1e-9) if k_data is not None else np.ones_like(l_data, dtype=bool)
             valid_n_data = n_data[valid_k_mask]
             valid_l_data = l_data[valid_k_mask]
             if len(valid_l_data) > 0:
                 ax_idx.plot(valid_l_data, valid_n_data, label=label, marker='.', markersize=3, linestyle='-')
                 min_l = min(min_l, valid_l_data.min()); max_l = max(max_l, valid_l_data.max())
                 min_n = min(min_n, valid_n_data.min()); max_n = max(max_n, valid_n_data.max())
                 plotted_something = True
    if plotted_something:
        ax_idx.set_xlabel("Wavelength (nm)"); ax_idx.set_ylabel("Refractive Index (n')")
        ax_idx.set_title("Real Part of Indices vs Wavelength")
        ax_idx.grid(True, linestyle=':'); ax_idx.minorticks_on()
        ax_idx.legend(fontsize='small')
        l_padding = (max_l - min_l) * 0.05 if max_l > min_l else 10
        n_padding = (max_n - min_n) * 0.05 if max_n > min_n else 0.1
        ax_idx.set_xlim(max(0, min_l - l_padding), max_l + l_padding)
        ax_idx.set_ylim(min_n - n_padding, max_n + n_padding)
        plt.tight_layout()
        return fig_idx
    else:
        plt.close(fig_idx)
        st.warning("Could not plot any valid index data.")
        return None

def run_calculation_st(ep_vector_to_use: Union[np.ndarray, None] = None, is_optimized: bool = False,
                       method_name: str = "", l_vec_override: Union[np.ndarray, None] = None,
                       material_sequence: Union[List[str], None] = None):
    calc_type = 'Optimized' if is_optimized else 'Nominal'
    full_method_name = f"{calc_type}{' (' + method_name + ')' if method_name else ''}"
    log_message(f"\n{'='*10} Starting Calculation: {full_method_name} {'='*10}")
    set_status(f"Starting Calculation: {full_method_name}...")
    results_fig = None
    try:
        with st.spinner(f"Running {full_method_name}..."):
            inputs = validate_physical_inputs_st(require_optim_params=False, require_initial_layers=False)
            if inputs is None: raise ValueError("Input validation failed.")
            active_targets = get_active_targets_st()
            active_targets_for_plot = active_targets if active_targets is not None else []
            ep_to_calc_input = ep_vector_to_use
            if not is_optimized and st.session_state.optimization_ran_since_nominal_change:
                 ep_to_calc_input = None
                 log_message("  Nominal calculation requested, using nominal QWOT (ignoring existing optimized state).")
            elif is_optimized and ep_vector_to_use is None:
                 ep_to_calc_input = st.session_state.current_optimized_ep
            l_vec_plot, ep_actual, nH_mat, nL_mat, nSub_mat = prepare_calculation_data_st(inputs, ep_to_calc_input)
            l_vec_final_plot = l_vec_override if l_vec_override is not None else l_vec_plot
            material_sequence_to_plot = None
            if material_sequence and len(material_sequence) == len(ep_actual):
                log_message(f"  Calculating T(λ) using provided arbitrary sequence ({len(material_sequence)} layers)...")
                calc_func = calculate_T_from_ep_arbitrary_jax_wrapper
                calc_args = (ep_actual, material_sequence, nSub_mat, l_vec_final_plot, EXCEL_FILE_PATH)
                material_sequence_to_plot = material_sequence
                st.warning("Arbitrary sequence calculation and plotting might be experimental.")
            else:
                if material_sequence: log_message("WARNING: Arbitrary sequence provided but invalid. Using standard H/L.")
                log_message(f"  Calculating T(λ) using standard H/L sequence ({len(ep_actual)} layers)...")
                calc_func = calculate_T_from_ep_jax_wrapper
                calc_args = (ep_actual, nH_mat, nL_mat, nSub_mat, l_vec_final_plot, EXCEL_FILE_PATH)
                material_sequence_to_plot = None
            log_message(f"  Calculating T for {len(l_vec_final_plot)} wavelengths...")
            start_rt_time = time.time()
            res_fine = calc_func(*calc_args)
            log_message(f"  T calculation finished in {time.time() - start_rt_time:.3f}s.")
            mse_display, res_optim_grid = None, None
            if active_targets_for_plot:
                l_min_o, l_max_o = inputs['l_range_deb'], inputs['l_range_fin']
                l_step_g = inputs['l_step']
                num_pts_o = max(2, int(np.round((l_max_o - l_min_o) / l_step_g)) + 1)
                l_vec_optim_disp = np.geomspace(l_min_o, l_max_o, num_pts_o)
                l_vec_optim_disp = l_vec_optim_disp[(l_vec_optim_disp > 0) & np.isfinite(l_vec_optim_disp)]
                if l_vec_optim_disp.size > 0:
                    log_message("  Calculating T on optimization grid for MSE display...")
                    calc_args_optim = calc_args[:-2] + (l_vec_optim_disp, EXCEL_FILE_PATH)
                    res_optim_grid = calc_func(*calc_args_optim)
                    mse_display, num_pts_mse = calculate_final_mse(res_optim_grid, active_targets_for_plot)
                    if num_pts_mse > 0: log_message(f"  MSE (display) = {mse_display:.3e} over {num_pts_mse} pts.")
                    else: log_message("  No points in target zones for display MSE."); mse_display = np.nan
                else: log_message("  Optimization grid empty, display MSE not calculated."); mse_display = np.nan
            else: log_message("  No active targets, display MSE not calculated.")
            log_message("  Generating plot figure...")
            results_fig = display_results_plot(res_fine, ep_actual, nH_mat, nL_mat, nSub_mat,
                                               active_targets_for_plot, mse_display,
                                               is_optimized=is_optimized, method_name=method_name,
                                               res_optim_grid=res_optim_grid,
                                               material_sequence=material_sequence_to_plot,
                                               l0_for_index_plot=inputs['l0'])
            log_message("  Plot figure generated.")
            if is_optimized:
                st.session_state.optimization_ran_since_nominal_change = True
                st.session_state.current_optimized_ep = ep_actual.copy()
                try:
                    qwots_opt = calculate_qwot_from_ep(ep_actual, inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                    if np.any(np.isnan(qwots_opt)): opt_qwot_str = "QWOT N/A (NaN)"
                    else: opt_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                    st.session_state.optimized_qwot_string_display = opt_qwot_str
                except Exception as e_qwot:
                    log_message(f"Error calculating optimized QWOT: {e_qwot}")
                    st.session_state.optimized_qwot_string_display = "QWOT Error"
            else:
                st.session_state.optimization_ran_since_nominal_change = False
                st.session_state.current_optimized_ep = None
                st.session_state.optimized_qwot_string_display = ""
    except (ValueError, RuntimeError, TypeError) as e:
        err_msg = f"ERROR ({calc_type} Calculation): {type(e).__name__}: {e}"
        log_message(err_msg); st.error(err_msg)
        set_status(f"{calc_type} Calculation Error")
    except Exception as e:
        err_msg = f"ERROR (Unexpected {calc_type} Calculation): {type(e).__name__}: {e}"
        tb_msg = traceback.format_exc()
        log_message(err_msg); log_message(tb_msg); print(err_msg); print(tb_msg)
        st.error(f"{err_msg}\n\nSee console/log for details.")
        set_status(f"Unexpected {calc_type} Calculation Error")
    finally:
        if st.session_state.get('last_plot_fig') is not None:
            try: plt.close(st.session_state.last_plot_fig)
            except Exception: pass
        st.session_state.last_plot_fig = results_fig
        if "Error" not in st.session_state.status_message:
            set_status(f"{full_method_name} Finished.")
        log_message(f"--- {calc_type} Calculation Finished ---")
def handle_calculate_button_click_st():
    log_message("Clearing optimized state for Nominal Evaluation.")
    st.session_state.current_optimized_ep = None
    st.session_state.current_material_sequence = None
    st.session_state.optimization_ran_since_nominal_change = False
    st.session_state.ep_history_deque.clear()
    st.session_state.optimized_qwot_string_display = ""
    log_message("Undo history cleared.")
    run_calculation_st(ep_vector_to_use=None, is_optimized=False, method_name="Nominal")
def handle_clear_optimized_state_st():
    log_message("Clearing optimized state and Undo history.")
    st.session_state.current_optimized_ep = None
    st.session_state.current_material_sequence = None
    st.session_state.optimization_ran_since_nominal_change = False
    st.session_state.ep_history_deque.clear()
    st.session_state.optimized_qwot_string_display = ""
    set_status("Optimized state cleared.")
def handle_set_nominal_wrapper_st():
    log_message("\n--- Setting Optimized Structure as Nominal ---")
    if not st.session_state.optimization_ran_since_nominal_change or st.session_state.current_optimized_ep is None:
        st.error("No optimized structure available to set as nominal.")
        log_message("ERROR: No optimized structure available.")
        return
    try:
        inputs = validate_physical_inputs_st()
        if inputs is None: return
        nH_material = inputs['nH_material']
        nL_material = inputs['nL_material']
        l0 = inputs['l0']
        ep_to_set = st.session_state.current_optimized_ep
        optimized_qwots = calculate_qwot_from_ep(ep_to_set, l0, nH_material, nL_material, EXCEL_FILE_PATH)
        if np.any(np.isnan(optimized_qwots)):
            log_message("Warning: QWOT calculation produced NaN. Nominal QWOT field cleared.")
            st.warning("Cannot calculate valid QWOTs (NaN). Nominal QWOT field cleared.")
            st.session_state.input_emp_str = ""
        else:
            final_qwot_str = ",".join([f"{q:.6f}" for q in optimized_qwots])
            st.session_state.input_emp_str = final_qwot_str
            log_message(f"Nominal QWOT field updated to: {final_qwot_str}")
            st.success("Optimized structure set as new Nominal (QWOT updated). Optimized state cleared.")
        handle_clear_optimized_state_st()
        set_status("Optimized set as Nominal")
    except Exception as e:
        err_msg = f"ERROR during 'Set Nominal': {type(e).__name__}: {e}"
        log_message(err_msg); st.error(err_msg)
        set_status("Set Nominal Error")
def handle_local_optimizer_click_st():
    log_message("\n" + "="*10 + " Starting Local Optimization " + "="*10)
    set_status("Starting Local Optimization...")
    start_time = time.time()
    st.session_state.ep_history_deque.clear()
    log_message("Undo history cleared (New Local Optimization).")
    optim_success = False
    with st.spinner("Running Local Optimizer..."):
        try:
            inputs = validate_physical_inputs_st(require_optim_params=True)
            if inputs is None: raise ValueError("Input validation failed.")
            active_targets = get_active_targets_st()
            if not active_targets: raise ValueError("Local optimization requires active targets.")
            nH_material, nL_material, nSub_material = inputs['nH_material'], inputs['nL_material'], inputs['nSub_material']
            l0 = inputs['l0']
            ep_start = None
            if st.session_state.optimization_ran_since_nominal_change and st.session_state.current_optimized_ep is not None:
                log_message("  Using current optimized structure as starting point.")
                ep_start = np.asarray(st.session_state.current_optimized_ep).copy()
            else:
                log_message("  Using nominal structure (QWOT) as starting point.")
                emp_list = [float(e.strip()) for e in inputs['emp_str'].split(',') if e.strip()]
                if not emp_list and inputs['emp_str'].strip(): raise ValueError("Nominal QWOT invalid.")
                ep_start_calc = calculate_initial_ep(emp_list, l0, nH_material, nL_material, EXCEL_FILE_PATH)
                ep_start = np.asarray(ep_start_calc)
                if ep_start is None or ep_start.size == 0: raise ValueError("Cannot determine nominal starting structure.")
            ep_start = np.maximum(ep_start, MIN_THICKNESS_PHYS_NM)
            log_message(f"  Starting optimization with {len(ep_start)} layers.")
            result_ep, local_opt_success, final_cost, optim_logs, optim_status_msg, total_nit, total_nfev = \
                _run_core_optimization_st(ep_start, inputs, active_targets,
                                          nH_material, nL_material, nSub_material,
                                          MIN_THICKNESS_PHYS_NM, log_prefix="  [Local Opt] ")
            for log_line in optim_logs: log_message(log_line)
            if local_opt_success:
                optim_success = True
                st.session_state.current_optimized_ep = result_ep.copy()
                st.session_state.optimization_ran_since_nominal_change = True
                st.session_state.current_material_sequence = None
                try:
                     qwots = calculate_qwot_from_ep(result_ep, l0, nH_material, nL_material, EXCEL_FILE_PATH)
                     if np.any(np.isnan(qwots)): final_qwot_str = "QWOT N/A"
                     else: final_qwot_str = ", ".join([f"{q:.3f}" for q in qwots])
                     st.session_state.optimized_qwot_string_display = final_qwot_str
                except Exception as e: log_message(f"Error calculating QWOT: {e}"); st.session_state.optimized_qwot_string_display = "QWOT Error"
                status_detail = "OK" if "success" in optim_status_msg.lower() else ("Limit" if "limit" in optim_status_msg.lower() else "Finished")
                status_text = f"Local Opt {status_detail} | MSE: {final_cost:.3e} | Layers: {len(result_ep)} | Iter/Eval: {total_nit}/{total_nfev}"
                set_status(status_text)
                log_message(f"  Local Optimization finished successfully. {status_text}")
                run_calculation_st(ep_vector_to_use=result_ep, is_optimized=True, method_name=f"Local Opt ({status_detail})")
            else:
                log_message(f"ERROR: Local optimization failed. Msg: {optim_status_msg}")
                st.error(f"Local Optimization failed: {optim_status_msg}")
                set_status(f"Local Opt Failed | Cost: {final_cost:.3e}")
        except (ValueError, RuntimeError, TypeError) as e:
            err_msg = f"ERROR (Local Opt Setup): {e}"
            log_message(err_msg); st.error(err_msg)
            set_status("Local Opt Failed (Params/Runtime)")
        except Exception as e:
            err_msg = f"ERROR (Unexpected Local Opt): {type(e).__name__}: {e}"
            tb_msg = traceback.format_exc()
            log_message(err_msg); log_message(tb_msg); print(err_msg); print(tb_msg)
            st.error(f"Unexpected Error during Local Optimization: {err_msg}")
            set_status("Local Opt Failed (Unexpected)")
        finally:
            log_message(f"--- End Local Optimization Process in {time.time() - start_time:.3f}s ---")
def handle_remove_thinnest_layer_st():
    log_message("\n" + "-"*10 + " Attempting Thin Layer Removal + Re-Optimization " + "-"*10)
    set_status("Starting Layer Removal + Re-Opt...")
    start_time = time.time()
    removal_success = False
    if not st.session_state.optimization_ran_since_nominal_change or st.session_state.current_optimized_ep is None:
        st.error("Run an optimization successfully first to enable layer removal.")
        log_message("ERROR: Removal requires a valid optimized structure.")
        set_status("Removal Failed (No opt. structure)")
        return
    ep_before_removal = np.asarray(st.session_state.current_optimized_ep)
    if len(ep_before_removal) <= 2:
        st.error("Structure too small (<= 2 layers) to remove/merge.")
        log_message("ERROR: Structure <= 2 layers.")
        set_status("Removal Failed (Too small)")
        return
    with st.spinner("Removing layer and re-optimizing..."):
        try:
            inputs = validate_physical_inputs_st(require_optim_params=True)
            if inputs is None: raise ValueError("Input validation failed.")
            active_targets = get_active_targets_st()
            if not active_targets: raise ValueError("Re-optimization requires active targets.")
            nH_material, nL_material, nSub_material = inputs['nH_material'], inputs['nL_material'], inputs['nSub_material']
            l0 = inputs['l0']
            st.session_state.ep_history_deque.append(ep_before_removal.copy())
            log_message(f"  [Undo] State saved before removal. History size: {len(st.session_state.ep_history_deque)}")
            set_status("Performing layer removal/merge...")
            ep_after_removal, structure_changed, removal_logs = _perform_layer_merge_or_removal_only_st(
                ep_before_removal, MIN_THICKNESS_PHYS_NM, log_prefix="  [Removal] "
            )
            for line in removal_logs: log_message(line)
            if not structure_changed:
                log_message("  No layer removed or structure unchanged.")
                st.info("Could not remove a layer according to criteria, or structure unchanged.")
                set_status(f"Removal Skipped | Layers: {len(ep_before_removal)}")
                try: st.session_state.ep_history_deque.pop(); log_message("  [Undo] Unneeded history state removed.")
                except IndexError: pass
                removal_success = False
            else:
                removal_success = True
                log_message(f"  Structure changed to {len(ep_after_removal)} layers. Re-optimizing...")
                set_status(f"Re-optimizing {len(ep_after_removal)} layers...")
                ep_after_reopt, reopt_success, final_cost, reopt_logs, reopt_status_msg, reopt_nit, reopt_nfev = \
                    _run_core_optimization_st(ep_after_removal, inputs, active_targets,
                                              nH_material, nL_material, nSub_material,
                                              MIN_THICKNESS_PHYS_NM, log_prefix="  [Re-Opt] ")
                for line in reopt_logs: log_message(line)
                if not reopt_success:
                    log_message("ERROR: Re-optimization after removal failed.")
                    st.warning(f"Layer removed, but re-optimization failed: {reopt_status_msg}. Structure reverted to state *after* removal.")
                    st.session_state.current_optimized_ep = ep_after_removal.copy()
                    st.session_state.optimization_ran_since_nominal_change = True
                    st.session_state.current_material_sequence = None
                    set_status(f"Removed, Re-Opt Failed | Layers: {len(ep_after_removal)}")
                else:
                    log_message("  Re-optimization successful.")
                    st.session_state.current_optimized_ep = ep_after_reopt.copy()
                    st.session_state.optimization_ran_since_nominal_change = True
                    st.session_state.current_material_sequence = None
                    status_text = f"Removed & Re-Opt OK | MSE: {final_cost:.3e} | Layers: {len(ep_after_reopt)} | Iter/Eval: {reopt_nit}/{reopt_nfev}"
                    set_status(status_text)
                    log_message(f"  Re-Optimization finished. {status_text}")
                final_ep_to_display = st.session_state.current_optimized_ep
                final_qwot_str = "QWOT Error"
                try:
                    qwots = calculate_qwot_from_ep(final_ep_to_display, l0, nH_material, nL_material, EXCEL_FILE_PATH)
                    if np.any(np.isnan(qwots)): final_qwot_str = "QWOT N/A"
                    else: final_qwot_str = ", ".join([f"{q:.3f}" for q in qwots])
                except Exception as e: log_message(f"Error calculating QWOT: {e}")
                st.session_state.optimized_qwot_string_display = final_qwot_str
                method_suffix = "(Re-Opt OK)" if reopt_success else "(Re-Opt Failed)" if structure_changed else "(No Change)"
                run_calculation_st(ep_vector_to_use=st.session_state.current_optimized_ep, is_optimized=True, method_name=f"Post-Removal {method_suffix}")
        except (ValueError, RuntimeError, TypeError) as e:
            err_msg = f"ERROR (Removal/Re-Opt Workflow): {e}"
            log_message(err_msg); st.error(err_msg)
            set_status("Removal Failed (Params/Runtime)")
            try: st.session_state.ep_history_deque.pop(); log_message("  [Undo] History state removed due to error.")
            except IndexError: pass
        except Exception as e:
            err_msg = f"ERROR (Unexpected Removal/Re-Opt): {type(e).__name__}: {e}"
            tb_msg = traceback.format_exc()
            log_message(err_msg); log_message(tb_msg); print(err_msg); print(tb_msg)
            st.error(f"Unexpected Error during Removal/Re-Opt: {err_msg}")
            set_status("Removal Failed (Unexpected)")
            try: st.session_state.ep_history_deque.pop(); log_message("  [Undo] History state removed due to error.")
            except IndexError: pass
        finally:
            log_message(f"--- End Attempted Removal + Re-Optimization in {time.time() - start_time:.3f}s ---")
def handle_undo_remove_layer_st():
    log_message("\n" + "~"*10 + " Undoing Last Layer Removal " + "~"*10)
    set_status("Undoing removal...")
    if not st.session_state.ep_history_deque:
        st.warning("Nothing to undo.")
        log_message("Undo history empty.")
        set_status("Undo Failed (Empty History)")
        return
    with st.spinner("Undoing..."):
        try:
            restored_ep = st.session_state.ep_history_deque.pop()
            st.session_state.current_optimized_ep = restored_ep.copy()
            st.session_state.current_material_sequence = None
            st.session_state.optimization_ran_since_nominal_change = True
            num_layers_restored = len(restored_ep)
            log_message(f"  Undo successful. State restored with {num_layers_restored} layers.")
            log_message(f"  Remaining Undo steps: {len(st.session_state.ep_history_deque)}")
            inputs = validate_physical_inputs_st()
            if inputs is None: raise ValueError("Cannot validate inputs after undo.")
            nH_material, nL_material, l0 = inputs['nH_material'], inputs['nL_material'], inputs['l0']
            final_qwot_str = "QWOT Error"
            try:
                qwots = calculate_qwot_from_ep(restored_ep, l0, nH_material, nL_material, EXCEL_FILE_PATH)
                if np.any(np.isnan(qwots)): final_qwot_str = "QWOT N/A"
                else: final_qwot_str = ", ".join([f"{q:.3f}" for q in qwots])
            except Exception as e: log_message(f"Error calculating QWOT after undo: {e}")
            st.session_state.optimized_qwot_string_display = final_qwot_str
            set_status(f"Undo OK | Layers: {num_layers_restored}")
            log_message("  Recalculating and plotting restored state...")
            run_calculation_st(ep_vector_to_use=restored_ep, is_optimized=True, method_name="Optimized (Undone)")
        except Exception as e:
            err_msg = f"ERROR (Undo Operation): {type(e).__name__}: {e}"
            tb_msg = traceback.format_exc()
            log_message(err_msg); log_message(tb_msg); print(err_msg); print(tb_msg)
            st.error(f"An unexpected error occurred during undo:\n{err_msg}")
            set_status("Undo Failed (Unexpected)")
        finally:
             log_message(f"--- End Undo Operation ---")

def _run_core_optimization_st(ep_start_optim: np.ndarray,
                              inputs: Dict, active_targets: List[Dict],
                              nH_material: MaterialInputType,
                              nL_material: MaterialInputType,
                              nSub_material: MaterialInputType,
                              min_thickness_phys: float, log_prefix: str = "") -> Tuple[np.ndarray, bool, float, List[str], str, int, int]:
    logs = []
    num_layers_start = len(ep_start_optim)
    final_ep = np.array(ep_start_optim, dtype=np.float64)
    optim_success = False
    final_cost = np.inf
    result_message_str = "Optimization not run or failed early."
    nit_total, nfev_total = 0, 0
    if num_layers_start == 0:
        logs.append(f"{log_prefix}Cannot optimize an empty structure.")
        return final_ep, False, np.inf, logs, "Empty structure", 0, 0
    try:
        l_min_optim, l_max_optim, l_step_optim = inputs['l_range_deb'], inputs['l_range_fin'], inputs['l_step']
        num_pts_optim = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
        l_vec_optim_np = np.geomspace(l_min_optim, l_max_optim, num_pts_optim)
        l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
        if not l_vec_optim_np.size: raise ValueError("Failed to generate lambda vector for optimization.")
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        logs.append(f"{log_prefix} Preparing dispersive indices for {len(l_vec_optim_jax)} lambdas...")
        prep_start_time = time.time()
        nH_arr_optim = get_nk_array(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        nL_arr_optim = get_nk_array(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        nSub_arr_optim = get_nk_array(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.append(f"{log_prefix} Index preparation finished in {time.time() - prep_start_time:.3f}s.")
        active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
        static_args_for_jax = (nH_arr_optim, nL_arr_optim, nSub_arr_optim, l_vec_optim_jax, active_targets_tuple, min_thickness_phys)
        value_and_grad_fn_jit = jax.jit(jax.value_and_grad(calculate_mse_for_optimization_penalized_jax))
        def scipy_obj_grad_wrapper(ep_vector_np, *args):
            ep_vector_jax = jnp.asarray(ep_vector_np)
            value_jax, grad_jax = value_and_grad_fn_jit(ep_vector_jax, *args)
            value_np = np.float64(np.array(value_jax))
            grad_np = np.array(grad_jax, dtype=np.float64)
            if not np.isfinite(value_np): value_np = np.finfo(np.float64).max
            if not np.all(np.isfinite(grad_np)): grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=1e6, neginf=-1e6)
            return value_np, grad_np
        lbfgsb_bounds = [(min_thickness_phys, None)] * num_layers_start
        options = {'maxiter': inputs['maxiter'], 'maxfun': inputs['maxfun'], 'disp': False, 'ftol': 1e-12, 'gtol': 1e-8}
        logs.append(f"{log_prefix}Starting L-BFGS-B with JAX gradient ({num_layers_start} layers)...")
        opt_start_time = time.time()
        result = minimize(scipy_obj_grad_wrapper, np.asarray(ep_start_optim, dtype=np.float64),
                          args=static_args_for_jax, method='L-BFGS-B', jac=True, bounds=lbfgsb_bounds, options=options)
        logs.append(f"{log_prefix}L-BFGS-B (JAX grad) finished in {time.time() - opt_start_time:.3f}s.")
        final_cost = result.fun if np.isfinite(result.fun) else np.inf
        result_message_str = result.message.decode('utf-8') if isinstance(result.message, bytes) else str(result.message)
        nit_total = result.nit if hasattr(result, 'nit') else 0
        nfev_total = result.nfev if hasattr(result, 'nfev') else 0
        is_success_or_limit = (result.success or result.status == 1) and np.isfinite(final_cost)
        if is_success_or_limit:
            final_ep = np.maximum(result.x, min_thickness_phys)
            optim_success = True
            log_status = "stopped by limit" if result.status == 1 else "succeeded"
            logs.append(f"{log_prefix}Optimization {log_status}! Cost: {final_cost:.3e}, Iterations: {nit_total}, Evals: {nfev_total}, Msg: {result_message_str}")
        else:
            optim_success = False
            final_ep = np.maximum(ep_start_optim, min_thickness_phys)
            logs.append(f"{log_prefix}Optimization FAILED. Cost: {final_cost:.3e}, Status: {result.status}, Iterations: {nit_total}, Evals: {nfev_total}, Msg: {result_message_str}")
            try:
                reverted_cost, _ = scipy_obj_grad_wrapper(final_ep, *static_args_for_jax)
                logs.append(f"{log_prefix}Reverted to initial structure (clamped). Recalculated cost: {reverted_cost:.3e}")
                final_cost = reverted_cost if np.isfinite(reverted_cost) else np.inf
            except Exception as cost_e:
                logs.append(f"{log_prefix}Reverted to initial structure (clamped). ERROR recalculating cost: {cost_e}")
                final_cost = np.inf
            nit_total, nfev_total = 0, 0
    except Exception as e_optim:
        logs.append(f"{log_prefix}ERROR during core optimization: {e_optim}\n{traceback.format_exc(limit=1)}")
        final_ep = np.maximum(ep_start_optim, min_thickness_phys)
        optim_success, final_cost = False, np.inf
        result_message_str = f"Exception: {e_optim}"
        nit_total, nfev_total = 0, 0
    return final_ep, optim_success, final_cost, logs, result_message_str, nit_total, nfev_total
def _perform_needle_insertion_scan_st(ep_vector_in: np.ndarray,
                                    inputs: Dict,
                                    active_targets: List[Dict],
                                    nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
                                    min_thickness_phys: float, l_vec_optim_np: np.ndarray,
                                    scan_step_nm: float, base_needle_thickness_nm: float,
                                    log_prefix: str = "") -> Tuple[Union[np.ndarray, None], float, List[str], int]:
    logs = []
    num_layers_in = len(ep_vector_in)
    if num_layers_in == 0:
        logs.append(f"{log_prefix}Cannot run needle scan on an empty structure.")
        return None, np.inf, logs, -1
    logs.append(f"{log_prefix}Starting needle scan on {num_layers_in} layers. Step: {scan_step_nm:.2f} nm, Needle: {base_needle_thickness_nm:.3f} nm.")
    try:
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        nH_arr = get_nk_array(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        nL_arr = get_nk_array(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        nSub_arr = get_nk_array(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
        static_args_cost_fn = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, min_thickness_phys)
        cost_fn_compiled = jax.jit(calculate_mse_for_optimization_penalized_jax)
        initial_cost_jax = cost_fn_compiled(jnp.asarray(ep_vector_in), *static_args_cost_fn)
        initial_cost = float(np.array(initial_cost_jax))
        if not np.isfinite(initial_cost):
            logs.append(f"{log_prefix}ERROR: Initial cost not finite ({initial_cost}). Aborting needle scan.")
            return None, np.inf, logs, -1
        logs.append(f"{log_prefix}Initial cost: {initial_cost:.6e}")
    except Exception as e_prep:
        logs.append(f"{log_prefix}ERROR preparing indices/initial cost for needle scan: {e_prep}. Aborting.")
        return None, np.inf, logs, -1
    best_ep_found, min_cost_found = None, initial_cost
    best_insertion_z, best_insertion_layer_idx = -1.0, -1
    tested_insertions = 0
    ep_cumsum = np.cumsum(ep_vector_in); total_thickness = ep_cumsum[-1] if num_layers_in > 0 else 0.0
    l0_repr = inputs.get('l0', 500.0)
    for z in np.arange(scan_step_nm, total_thickness, scan_step_nm):
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
        t_layer_split_1 = z - (ep_cumsum[current_layer_idx-1] if current_layer_idx > 0 else 0.0)
        t_layer_split_2 = ep_cumsum[current_layer_idx] - z
        ep_temp_np = np.concatenate((
            ep_vector_in[:current_layer_idx],
            [t_layer_split_1, base_needle_thickness_nm, t_layer_split_2],
            ep_vector_in[current_layer_idx+1:]
        ))
        ep_temp_np = np.maximum(ep_temp_np, min_thickness_phys)
        try:
            current_cost_jax = cost_fn_compiled(jnp.asarray(ep_temp_np), *static_args_cost_fn)
            current_cost = float(np.array(current_cost_jax))
            if np.isfinite(current_cost) and current_cost < min_cost_found:
                min_cost_found = current_cost; best_ep_found = ep_temp_np.copy()
                best_insertion_z = z; best_insertion_layer_idx = current_layer_idx
        except Exception as e_cost:
            logs.append(f"{log_prefix}WARNING: Cost calculation failed for z={z:.2f}. {e_cost}"); continue
    if best_ep_found is not None:
        improvement = initial_cost - min_cost_found
        logs.append(f"{log_prefix}Finished. Tested {tested_insertions} points. Best improvement: {improvement:.6e}")
        logs.append(f"{log_prefix}Best insertion at z={best_insertion_z:.3f} nm in original layer {best_insertion_layer_idx + 1}. New cost: {min_cost_found:.6e}")
        return best_ep_found, min_cost_found, logs, best_insertion_layer_idx
    else:
        logs.append(f"{log_prefix}Finished. No improvement found after testing {tested_insertions} points.")
        return None, initial_cost, logs, -1
def _run_needle_iterations_st(ep_start: np.ndarray, num_needles: int,
                              inputs: Dict, active_targets: List[Dict],
                              nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
                              min_thickness_phys: float, l_vec_optim_np: np.ndarray,
                              scan_step_nm: float, base_needle_thickness_nm: float,
                              log_prefix: str = "") -> Tuple[np.ndarray, float, List[str], int, int, int]:
    logs = []
    best_ep_overall = np.asarray(ep_start).copy()
    best_mse_overall = np.inf
    total_nit_needles = 0
    total_nfev_needles = 0
    successful_reopts_count = 0
    status_updates = []
    try:
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        nH_arr = get_nk_array(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        nL_arr = get_nk_array(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        nSub_arr = get_nk_array(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
        static_args_cost_fn = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, min_thickness_phys)
        cost_fn_compiled = jax.jit(calculate_mse_for_optimization_penalized_jax)
        initial_cost_jax = cost_fn_compiled(jnp.asarray(best_ep_overall), *static_args_cost_fn)
        best_mse_overall = float(np.array(initial_cost_jax))
        if not np.isfinite(best_mse_overall): raise ValueError("Initial MSE not finite.")
        logs.append(f"{log_prefix}Starting needle iterations. Initial MSE: {best_mse_overall:.6e}")
    except Exception as e:
        logs.append(f"{log_prefix}ERROR calculating initial MSE for needle iterations: {e}")
        return ep_start, np.inf, logs, 0, 0, 0
    l0_repr = inputs.get('l0', 500.0)
    for i in range(num_needles):
        iter_log_prefix = f"{log_prefix} Needle Iter {i + 1}/{num_needles}: "
        logs.append(f"{iter_log_prefix}Starting...")
        status_updates.append(f"Needle Iter {i+1}/{num_needles} - Scanning...")
        current_ep_iter = best_ep_overall.copy()
        num_layers_current = len(current_ep_iter)
        if num_layers_current == 0:
            logs.append(f"{iter_log_prefix}Empty structure, stopping."); break
        ep_after_scan, cost_after_scan, scan_logs, inserted_idx = _perform_needle_insertion_scan_st(
            current_ep_iter, inputs, active_targets,
            nH_material, nL_material, nSub_material,
            min_thickness_phys, l_vec_optim_np,
            scan_step_nm, base_needle_thickness_nm,
            log_prefix=f"{iter_log_prefix} Scan: "
        )
        logs.extend(scan_logs)
        if ep_after_scan is None:
            logs.append(f"{iter_log_prefix}Scan found no improvement. Stopping iterations."); break
        logs.append(f"{iter_log_prefix}Scan found potential improvement. Re-optimizing {len(ep_after_scan)} layers...")
        status_updates.append(f"Needle Iter {i+1}/{num_needles} - Re-Optimizing...")
        ep_after_reopt, optim_success, final_cost_reopt, optim_logs, optim_status_msg, nit_reopt, nfev_reopt = \
            _run_core_optimization_st(ep_after_scan, inputs, active_targets,
                                      nH_material, nL_material, nSub_material,
                                      min_thickness_phys, log_prefix=f"{iter_log_prefix} Re-Opt: ")
        logs.extend(optim_logs)
        if not optim_success:
            logs.append(f"{iter_log_prefix}Re-optimization FAILED. Stopping iterations."); break
        logs.append(f"{iter_log_prefix}Re-optimization successful. New MSE: {final_cost_reopt:.6e}. Iter/Eval: {nit_reopt}/{nfev_reopt}")
        total_nit_needles += nit_reopt; total_nfev_needles += nfev_reopt; successful_reopts_count += 1
        if final_cost_reopt < best_mse_overall - MSE_IMPROVEMENT_TOLERANCE:
            logs.append(f"{iter_log_prefix}MSE significantly improved from {best_mse_overall:.6e}. Updating best result.")
            best_ep_overall = ep_after_reopt.copy(); best_mse_overall = final_cost_reopt
        else:
            logs.append(f"{iter_log_prefix}New MSE ({final_cost_reopt:.6e}) not significantly improved vs best ({best_mse_overall:.6e}). Stopping iterations.")
            best_ep_overall = ep_after_reopt.copy(); best_mse_overall = final_cost_reopt
            break
    logs.append(f"{log_prefix}End needle iterations. Final best MSE: {best_mse_overall:.6e}")
    logs.append(f"{log_prefix}Total Iter/Eval during {successful_reopts_count} successful re-opts: {total_nit_needles}/{total_nfev_needles}")
    return best_ep_overall, best_mse_overall, logs, total_nit_needles, total_nfev_needles, successful_reopts_count
def _execute_split_stack_scan_st(current_l0: float, initial_layer_number: int,
                                 inputs: Dict,
                                 active_targets: List[Dict],
                                 l_vec_eval_sparse_jax: jnp.ndarray,
                                 status_callback: Callable[[str], None]
                                 ) -> Tuple[float, Union[np.ndarray, None], List[str]]:
    logs = []
    num_combinations = 2**initial_layer_number
    logs.append(f"  [Scan l0={current_l0:.2f}] Testing {num_combinations:,} QWOT combinations (1.0 or 2.0)...")
    active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
    try:
        nH_material = inputs['nH_material']
        nL_material = inputs['nL_material']
        nSub_material = inputs['nSub_material']
        nH_c_l0 = jnp.asarray(_get_nk_at_lambda(nH_material, current_l0, EXCEL_FILE_PATH))
        nL_c_l0 = jnp.asarray(_get_nk_at_lambda(nL_material, current_l0, EXCEL_FILE_PATH))
        nSub_arr_scan = get_nk_array(nSub_material, l_vec_eval_sparse_jax, EXCEL_FILE_PATH)
        status_callback(f"Scan l0={current_l0:.1f} - Precomputing matrices ({initial_layer_number} layers)...")
        precompute_start_time = time.time()
        layer_matrices_list = []
        for i in range(initial_layer_number):
            m1, m2 = get_layer_matrices_qwot(i, initial_layer_number, current_l0, nH_c_l0, nL_c_l0, l_vec_eval_sparse_jax)
            layer_matrices_list.append(jnp.stack([m1, m2], axis=0))
        all_layer_matrices = jnp.stack(layer_matrices_list, axis=0)
        all_layer_matrices.block_until_ready()
        logs.append(f"    Matrix precomputation finished in {time.time() - precompute_start_time:.3f}s.")
        N1, N2 = initial_layer_number // 2, initial_layer_number - (initial_layer_number // 2)
        num_comb1, num_comb2 = 2**N1, 2**N2
        indices1 = jnp.arange(num_comb1); indices2 = jnp.arange(num_comb2)
        powers1 = 2**jnp.arange(N1); powers2 = 2**jnp.arange(N2)
        multiplier_indices1 = jnp.not_equal(indices1[:, None] & powers1, 0).astype(jnp.int32)
        multiplier_indices2 = jnp.not_equal(indices2[:, None] & powers2, 0).astype(jnp.int32)
        matrices_half1, matrices_half2 = all_layer_matrices[:N1], all_layer_matrices[N1:]
        status_callback(f"Scan l0={current_l0:.1f} - Partial products 1/2 ({num_comb1:,} combs)...")
        half1_start_time = time.time()
        compute_half_product_jit = jax.jit(compute_half_product)
        partial_products1 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices1, matrices_half1)
        partial_products1.block_until_ready()
        logs.append(f"    Partial products 1/2 calculated in {time.time() - half1_start_time:.3f}s.")
        status_callback(f"Scan l0={current_l0:.1f} - Partial products 2/2 ({num_comb2:,} combs)...")
        half2_start_time = time.time()
        partial_products2 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices2, matrices_half2)
        partial_products2.block_until_ready()
        logs.append(f"    Partial products 2/2 calculated in {time.time() - half2_start_time:.3f}s.")
        status_callback(f"Scan l0={current_l0:.1f} - Combining & MSE ({num_comb1*num_comb2:,} total)...")
        combine_start_time = time.time()
        combine_and_calc_mse_jit = jax.jit(combine_and_calc_mse)
        vmap_inner = vmap(combine_and_calc_mse_jit, in_axes=(None, 0, None, None, None))
        vmap_outer = vmap(vmap_inner, in_axes=(0, None, None, None, None))
        all_mses_nested = vmap_outer(partial_products1, partial_products2, nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple)
        all_mses_nested.block_until_ready()
        logs.append(f"    Combination and MSE finished in {time.time() - combine_start_time:.3f}s.")
        all_mses_flat = all_mses_nested.reshape(-1)
        best_idx_flat = jnp.argmin(all_mses_flat)
        current_best_mse = float(all_mses_flat[best_idx_flat])
        if not np.isfinite(current_best_mse):
            logs.append(f"    Warning: No valid combination found (all MSEs infinite/NaN).")
            return np.inf, None, logs
        best_idx_half1, best_idx_half2 = jnp.unravel_index(best_idx_flat, (num_comb1, num_comb2))
        best_indices_h1 = multiplier_indices1[best_idx_half1]
        best_indices_h2 = multiplier_indices2[best_idx_half2]
        best_multipliers_h1 = 1.0 + best_indices_h1.astype(jnp.float64)
        best_multipliers_h2 = 1.0 + best_indices_h2.astype(jnp.float64)
        current_best_multipliers = np.array(jnp.concatenate([best_multipliers_h1, best_multipliers_h2]))
        logs.append(f"    Best Scan MSE for l0={current_l0:.2f}: {current_best_mse:.6e}")
        return current_best_mse, current_best_multipliers, logs
    except Exception as e_scan:
        logs.append(f"  ERROR during Scan l0={current_l0:.2f}: {e_scan}\n{traceback.format_exc(limit=1)}")
        return np.inf, None, logs
def handle_start_nominal_qwot_scan_st():
    log_message("\n" + "#"*10 + " Starting Exhaustive Nominal QWOT Scan + l0 Test + Local Optimization " + "#"*10)
    set_status("Starting QWOT Scan + l0 + Opt...")
    start_time_scan = time.time()
    st.session_state.ep_history_deque.clear()
    log_message("  Undo history cleared.")
    overall_success = False
    with st.status("Initializing Scan...", expanded=True) as status_ui:
        try:
            inputs = validate_physical_inputs_st(require_optim_params=True, require_initial_layers=True)
            if inputs is None: raise ValueError("Input validation failed.")
            initial_layer_number = inputs['initial_layer_number']
            if initial_layer_number <= 0: raise ValueError("Initial Layer Number must be positive.")
            l0_nominal_gui = inputs['l0']
            active_targets = get_active_targets_st()
            if not active_targets: raise ValueError("QWOT Scan requires active targets.")
            nH_material, nL_material, nSub_material = inputs['nH_material'], inputs['nL_material'], inputs['nSub_material']
            l_min, l_max, l_step = inputs['l_range_deb'], inputs['l_range_fin'], inputs['l_step']
            num_pts_eval = max(2, int(np.round((l_max - l_min) / l_step)) + 1)
            l_vec_eval_full_np = np.geomspace(l_min, l_max, num_pts_eval)
            l_vec_eval_full_np = l_vec_eval_full_np[(l_vec_eval_full_np > 0) & np.isfinite(l_vec_eval_full_np)]
            if not l_vec_eval_full_np.size: raise ValueError("Failed to generate evaluation lambda vector.")
            l_vec_eval_sparse_np = l_vec_eval_full_np[::2]
            if not l_vec_eval_sparse_np.size: raise ValueError("Failed to generate sparse lambda vector for scan.")
            l_vec_eval_sparse_jax = jnp.asarray(l_vec_eval_sparse_np)
            log_message(f"  Evaluation grid: {len(l_vec_eval_full_np)} pts. Fast scan grid: {len(l_vec_eval_sparse_jax)} pts.")
            l0_values_to_test = sorted(list(set([l0_nominal_gui, l0_nominal_gui * 1.25, l0_nominal_gui * 0.75])))
            l0_values_to_test = [l for l in l0_values_to_test if l > 1e-6]
            num_l0_tests = len(l0_values_to_test)
            num_combinations = 2**initial_layer_number
            total_evals_scan = num_combinations * num_l0_tests
            log_message(f"  N={initial_layer_number}, {num_l0_tests} l0 values: {[f'{l:.2f}' for l in l0_values_to_test]}.")
            log_message(f"  Combinations per l0: {num_combinations:,}. Total scan evaluations: {total_evals_scan:,}.")
            warning_threshold_comb = 2**21
            if num_combinations > warning_threshold_comb:
                st.warning(f"Scan requires testing {num_combinations:,} combinations for EACH of {num_l0_tests} l0 values (Total Scan: {total_evals_scan:,}). This may take a very long time and consume significant memory/CPU.")
                log_message("WARNING: Scan size is very large.")
            initial_candidates = []
            scan_progress = st.progress(0.0)
            for l0_idx, current_l0 in enumerate(l0_values_to_test):
                status_ui.update(label=f"Scanning l0={current_l0:.1f} ({l0_idx+1}/{num_l0_tests})... Combinations: {num_combinations:,}")
                scan_start_l0 = time.time()
                log_message(f"\n--- Running QWOT Scan for l0 = {current_l0:.2f} nm ({l0_idx+1}/{num_l0_tests}) ---")
                current_best_mse_scan, current_best_multipliers_scan, scan_logs = _execute_split_stack_scan_st(
                     current_l0, initial_layer_number, inputs, active_targets, l_vec_eval_sparse_jax,
                     lambda msg: status_ui.update(label=msg)
                )
                for line in scan_logs: log_message(line)
                log_message(f"--- Scan for l0={current_l0:.2f} finished in {time.time()-scan_start_l0:.2f}s ---")
                if np.isfinite(current_best_mse_scan) and current_best_multipliers_scan is not None:
                    log_message(f"  Scan found candidate for l0={current_l0:.2f} with MSE {current_best_mse_scan:.6e}.")
                    initial_candidates.append({
                        'l0': current_l0, 'mse_scan': current_best_mse_scan,
                        'multipliers': np.array(current_best_multipliers_scan)
                    })
                else: log_message(f"  No valid candidate found for l0={current_l0:.2f}.")
                scan_progress.progress((l0_idx + 1) / num_l0_tests)
            if not initial_candidates: raise RuntimeError("QWOT Scan found no valid initial candidates.")
            initial_candidates.sort(key=lambda c: c['mse_scan'])
            log_message(f"\n--- QWOT Scan finished. Found {len(initial_candidates)} candidate(s). Running Local Optimization. ---")
            status_ui.update(label=f"Optimizing {len(initial_candidates)} candidates...")
            final_best_ep, final_best_mse, final_best_l0 = None, np.inf, None
            final_best_initial_multipliers = None
            overall_optim_nit, overall_optim_nfev, successful_optim_count = 0, 0, 0
            optim_progress = st.progress(0.0)
            for idx, candidate in enumerate(initial_candidates):
                 cand_l0, cand_mult, cand_mse_scan = candidate['l0'], candidate['multipliers'], candidate['mse_scan']
                 status_ui.update(label=f"Optimizing Candidate {idx+1}/{len(initial_candidates)} (l0={cand_l0:.1f})...")
                 log_message(f"\n--- Optimizing Candidate {idx+1}/{len(initial_candidates)} (from l0={cand_l0:.2f}, scan MSE={cand_mse_scan:.6e}) ---")
                 try:
                     ep_start_optim = calculate_initial_ep(cand_mult, cand_l0, nH_material, nL_material, EXCEL_FILE_PATH)
                     ep_start_optim = np.maximum(ep_start_optim, MIN_THICKNESS_PHYS_NM)
                     log_message(f"  Starting local optimization from {len(ep_start_optim)} layers.")
                     result_ep_optim, optim_success, final_cost_optim, optim_logs, optim_status_msg, nit_optim, nfev_optim = \
                         _run_core_optimization_st(ep_start_optim, inputs, active_targets,
                                                   nH_material, nL_material, nSub_material,
                                                   MIN_THICKNESS_PHYS_NM, log_prefix=f"  [Local Opt Cand {idx+1}] ")
                     for log_line in optim_logs: log_message(log_line)
                     if optim_success:
                         successful_optim_count += 1; overall_optim_nit += nit_optim; overall_optim_nfev += nfev_optim
                         log_message(f"  Optimization successful. Final MSE: {final_cost_optim:.6e}")
                         if final_cost_optim < final_best_mse:
                             log_message(f"  *** New global best found! MSE improved from {final_best_mse:.6e} ***")
                             final_best_mse = final_cost_optim; final_best_ep = result_ep_optim.copy()
                             final_best_l0 = cand_l0; final_best_initial_multipliers = cand_mult
                         else: log_message(f"  Result MSE not better than current best {final_best_mse:.6e}.")
                     else: log_message(f"  Optimization FAILED for candidate {idx+1}. Msg: {optim_status_msg}")
                 except Exception as e_optim_cand: log_message(f"ERROR during candidate {idx+1} optimization: {e_optim_cand}")
                 optim_progress.progress((idx + 1) / len(initial_candidates))
            if final_best_ep is None: raise RuntimeError("Local optimization failed for all candidates from the scan.")
            log_message(f"\n--- Best Global Result After Scan+Opt ---")
            log_message(f"Final Best MSE: {final_best_mse:.6e}")
            log_message(f"Originating from l0 = {final_best_l0:.2f} nm")
            best_mult_str = ",".join([f"{m:.3f}" for m in final_best_initial_multipliers])
            log_message(f"Original Multiplier Sequence ({initial_layer_number} layers): {best_mult_str}")
            st.session_state.current_optimized_ep = final_best_ep.copy()
            st.session_state.optimization_ran_since_nominal_change = True
            st.session_state.current_material_sequence = None
            if abs(final_best_l0 - l0_nominal_gui) > 1e-3:
                log_message(f"Updating Centering λ from {l0_nominal_gui:.2f} to best found value {final_best_l0:.2f}")
                st.session_state.input_l0 = final_best_l0
            final_qwot_str = "QWOT Error"
            try:
                 qwots = calculate_qwot_from_ep(final_best_ep, final_best_l0, nH_material, nL_material, EXCEL_FILE_PATH)
                 if not np.any(np.isnan(qwots)): final_qwot_str = ",".join([f"{q:.3f}" for q in qwots])
                 else: final_qwot_str = "QWOT N/A"
            except Exception as e: log_message(f"Warning: QWOT calculation failed ({e})")
            st.session_state.optimized_qwot_string_display = final_qwot_str
            avg_nit = overall_optim_nit / successful_optim_count if successful_optim_count > 0 else 0
            avg_nfev = overall_optim_nfev / successful_optim_count if successful_optim_count > 0 else 0
            status_text = f"Scan+Opt Finished | Best MSE: {final_best_mse:.3e} | Layers: {len(final_best_ep)} | L0: {final_best_l0:.1f} | Avg Iter/Eval: {avg_nit:.1f}/{avg_nfev:.1f}"
            status_ui.update(label=status_text, state="complete", expanded=False)
            set_status(status_text)
            overall_success = True
            log_message("Plotting final Scan+Opt result...")
            run_calculation_st(ep_vector_to_use=final_best_ep, is_optimized=True, method_name=f"Scan+Opt (N={initial_layer_number}, L0={final_best_l0:.1f})")
        except (ValueError, RuntimeError, TypeError) as e:
            err_msg = f"ERROR (QWOT Scan + Opt Workflow): {e}"
            log_message(err_msg); st.error(err_msg)
            set_status("Scan+Opt Failed (Params/Runtime)")
            status_ui.update(label=f"Error: {e}", state="error", expanded=True)
        except Exception as e:
            err_msg = f"ERROR (Unexpected Scan+Opt / JAX Error): {type(e).__name__}: {e}"
            tb_msg = traceback.format_exc()
            log_message(err_msg); log_message(tb_msg); print(err_msg); print(tb_msg)
            st.error(f"Unexpected Error during Scan+Opt: {err_msg}")
            set_status("Scan+Opt Failed (Unexpected/JAX)")
            status_ui.update(label=f"Unexpected Error: {e}", state="error", expanded=True)
        finally:
            log_message(f"--- Total QWOT Scan + Opt Time: {time.time() - start_time_scan:.3f}s ---")
def handle_auto_needle_mode_st():
    log_message("\n" + "#"*10 + f" Starting Auto Needle+Thin Mode (Max {AUTO_MAX_CYCLES} Cycles) " + "#"*10)
    set_status("Starting Auto Mode...")
    start_time_auto = time.time()
    st.session_state.ep_history_deque.clear()
    log_message("  Undo history cleared.")
    overall_success = False
    with st.status("Initializing Auto Mode...", expanded=True) as status_ui:
        try:
            inputs = validate_physical_inputs_st(require_optim_params=True)
            if inputs is None: raise ValueError("Input validation failed.")
            active_targets = get_active_targets_st()
            if not active_targets: raise ValueError("Auto mode requires active targets.")
            nH_material, nL_material, nSub_material = inputs['nH_material'], inputs['nL_material'], inputs['nSub_material']
            l0 = inputs['l0']
            threshold_from_gui = inputs.get('auto_thin_threshold', 1.0)
            log_message(f"  Using Auto Thin Removal Threshold: {threshold_from_gui:.3f} nm")
            l_min, l_max, l_step = inputs['l_range_deb'], inputs['l_range_fin'], inputs['l_step']
            num_pts = max(2, int(np.round((l_max - l_min) / l_step)) + 1)
            l_vec_optim_np = np.geomspace(l_min, l_max, num_pts)
            l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
            if not l_vec_optim_np.size: raise ValueError("Failed to generate lambda vector for Auto Mode.")
            ep_start_auto, best_mse_so_far = None, np.inf
            initial_opt_needed_and_done = False
            total_iters_auto, total_evals_auto, optim_runs_auto = 0, 0, 0
            cost_fn = None # Define cost function later if needed
            static_args_cost = None
            if st.session_state.optimization_ran_since_nominal_change and st.session_state.current_optimized_ep is not None:
                 log_message("  Auto Mode: Using existing optimized structure as start.")
                 ep_start_auto = np.asarray(st.session_state.current_optimized_ep).copy()
                 try:
                     l_vec_opt_jax = jnp.asarray(l_vec_optim_np)
                     nH_arr = get_nk_array(nH_material, l_vec_opt_jax, EXCEL_FILE_PATH)
                     nL_arr = get_nk_array(nL_material, l_vec_opt_jax, EXCEL_FILE_PATH)
                     nSub_arr = get_nk_array(nSub_material, l_vec_opt_jax, EXCEL_FILE_PATH)
                     active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
                     static_args_cost = (nH_arr, nL_arr, nSub_arr, l_vec_opt_jax, active_targets_tuple, MIN_THICKNESS_PHYS_NM)
                     cost_fn = jax.jit(calculate_mse_for_optimization_penalized_jax)
                     best_mse_so_far = float(np.array(cost_fn(jnp.asarray(ep_start_auto), *static_args_cost)))
                     if not np.isfinite(best_mse_so_far): raise ValueError("Initial MSE not finite")
                 except Exception as e: raise ValueError(f"Cannot calculate initial MSE for auto mode: {e}")
            else:
                 log_message("  Auto Mode: Using nominal structure (QWOT) as start.")
                 emp_list = [float(e.strip()) for e in inputs['emp_str'].split(',') if e.strip()]
                 if not emp_list and inputs['emp_str'].strip(): raise ValueError("Nominal QWOT invalid.")
                 ep_nominal_calc = calculate_initial_ep(emp_list, l0, nH_material, nL_material, EXCEL_FILE_PATH)
                 if ep_nominal_calc is None or ep_nominal_calc.size == 0: raise ValueError("Cannot determine nominal starting structure.")
                 ep_nominal = np.maximum(np.asarray(ep_nominal_calc), MIN_THICKNESS_PHYS_NM)
                 log_message(f"  Nominal structure has {len(ep_nominal)} layers. Running initial optimization...")
                 status_ui.update(label="Auto Mode - Initial Optimization...")
                 ep_after_initial_opt, initial_opt_success, initial_mse, initial_opt_logs, initial_opt_msg, initial_nit, initial_nfev = \
                     _run_core_optimization_st(ep_nominal, inputs, active_targets, nH_material, nL_material, nSub_material, MIN_THICKNESS_PHYS_NM, log_prefix="  [Auto Init Opt] ")
                 for log_line in initial_opt_logs: log_message(log_line)
                 if not initial_opt_success: raise RuntimeError(f"Initial Optimization Failed: {initial_opt_msg}")
                 log_message(f"  Initial optimization finished. MSE: {initial_mse:.6e} (Iter/Eval: {initial_nit}/{initial_nfev})")
                 ep_start_auto = ep_after_initial_opt.copy()
                 best_mse_so_far = initial_mse
                 initial_opt_needed_and_done = True
                 total_iters_auto += initial_nit; total_evals_auto += initial_nfev; optim_runs_auto += 1
            best_ep_so_far = ep_start_auto.copy()
            if not np.isfinite(best_mse_so_far): raise ValueError("Starting MSE for cycles not finite.")
            log_message(f"  Starting Auto Mode Cycles with MSE: {best_mse_so_far:.6e} ({len(best_ep_so_far)} layers)")
            num_cycles_done = 0
            auto_progress = st.progress(0.0)
            for cycle_num in range(AUTO_MAX_CYCLES):
                status_ui.update(label=f"Auto Cycle {cycle_num + 1}/{AUTO_MAX_CYCLES} - Start | MSE: {best_mse_so_far:.3e}")
                log_message(f"\n--- Auto Cycle {cycle_num + 1} / {AUTO_MAX_CYCLES} ---")
                mse_at_cycle_start = best_mse_so_far
                ep_at_cycle_start = best_ep_so_far.copy()
                cycle_improved_overall = False
                status_ui.update(label=f"Cycle {cycle_num+1} - Needle ({AUTO_NEEDLES_PER_CYCLE}x)...")
                ep_after_needles, mse_after_needles, needle_logs, nit_needles, nfev_needles, reopts_in_needles = \
                    _run_needle_iterations_st(best_ep_so_far, AUTO_NEEDLES_PER_CYCLE, inputs, active_targets,
                                              nH_material, nL_material, nSub_material, MIN_THICKNESS_PHYS_NM,
                                              l_vec_optim_np, DEFAULT_NEEDLE_SCAN_STEP_NM, BASE_NEEDLE_THICKNESS_NM,
                                              log_prefix=f"  [Auto Cycle {cycle_num+1} Needle] ")
                for log_line in needle_logs: log_message(log_line)
                log_message(f"  MSE after Needles: {mse_after_needles:.6e} (Iter/Eval Sum: {nit_needles}/{nfev_needles})")
                total_iters_auto += nit_needles; total_evals_auto += nfev_needles; optim_runs_auto += reopts_in_needles
                if mse_after_needles < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                    log_message("    Needle phase improved MSE globally.")
                    best_ep_so_far = ep_after_needles.copy(); best_mse_so_far = mse_after_needles; cycle_improved_overall = True
                else:
                     log_message("    Needle phase did not improve MSE significantly.")
                     best_ep_so_far = ep_after_needles.copy(); best_mse_so_far = mse_after_needles
                status_ui.update(label=f"Cycle {cycle_num+1} - Removal+ReOpt (Thr: {threshold_from_gui:.2f}nm)...")
                layers_removed_this_cycle = 0; thinning_loop_iteration = 0
                max_thinning_iterations = len(best_ep_so_far) + 1
                while thinning_loop_iteration < max_thinning_iterations:
                    thinning_loop_iteration += 1
                    current_num_layers = len(best_ep_so_far)
                    if current_num_layers <= 2: log_message("    Structure too small for thin removal."); break
                    ep_before_thin_iter = best_ep_so_far.copy()
                    mse_before_thin_iter = best_mse_so_far
                    ep_after_single_removal, structure_changed, removal_logs = _perform_layer_merge_or_removal_only_st(
                        best_ep_so_far, MIN_THICKNESS_PHYS_NM, log_prefix=f"    [Remove+ReOpt Iter {thinning_loop_iteration}]",
                        threshold_for_removal=threshold_from_gui
                    )
                    for line in removal_logs: log_message(line)
                    if structure_changed:
                        layers_removed_this_cycle += 1
                        status_ui.update(label=f"Cycle {cycle_num+1} - Re-Opt after removal {layers_removed_this_cycle}...")
                        log_message(f"    Layer removed. Re-optimizing {len(ep_after_single_removal)} layers...")
                        ep_after_thin_reopt, thin_reopt_success, mse_after_thin_reopt, thin_reopt_logs, thin_reopt_msg, nit_thin, nfev_thin = \
                            _run_core_optimization_st(ep_after_single_removal, inputs, active_targets, nH_material, nL_material, nSub_material, MIN_THICKNESS_PHYS_NM, log_prefix=f"      [RemoveReOpt {layers_removed_this_cycle}] ")
                        for log_line in thin_reopt_logs: log_message(log_line)
                        if thin_reopt_success:
                            log_message(f"    Re-opt successful. New MSE: {mse_after_thin_reopt:.6e} (Iter/Eval: {nit_thin}/{nfev_thin})")
                            total_iters_auto += nit_thin; total_evals_auto += nfev_thin; optim_runs_auto += 1
                            if mse_after_thin_reopt < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                                log_message("      MSE improved globally. Updating best state.")
                                best_ep_so_far = ep_after_thin_reopt.copy(); best_mse_so_far = mse_after_thin_reopt; cycle_improved_overall = True
                            else:
                                log_message("      MSE not improved globally vs previous best.")
                                best_ep_so_far = ep_after_thin_reopt.copy(); best_mse_so_far = mse_after_thin_reopt
                        else:
                            log_message(f"    WARNING: Re-opt after removal failed ({thin_reopt_msg}). Stopping removal phase.")
                            best_ep_so_far = ep_after_single_removal.copy()
                            try: best_mse_so_far = float(np.array(cost_fn(jnp.asarray(best_ep_so_far), *static_args_cost)))
                            except: best_mse_so_far = np.inf
                            log_message(f"    MSE after Remove+Failed ReOpt (reverted): {best_mse_so_far:.6e}")
                            cycle_improved_overall = (best_mse_so_far < mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE) or cycle_improved_overall
                            break
                    else: log_message("    No more layers found below threshold."); break
                log_message(f"  Thin Removal Phase finished. {layers_removed_this_cycle} layer(s) removed.")
                num_cycles_done += 1
                auto_progress.progress(num_cycles_done / AUTO_MAX_CYCLES)
                log_message(f"--- End Auto Cycle {cycle_num + 1} --- Current best MSE: {best_mse_so_far:.6e} ({len(best_ep_so_far)} layers)---")
                if not cycle_improved_overall and not (best_mse_so_far < mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE):
                    log_message(f"No significant improvement in Cycle {cycle_num + 1}. Stopping Auto Mode.")
                    if best_mse_so_far > mse_at_cycle_start + MSE_IMPROVEMENT_TOLERANCE:
                         log_message(f"Reverting to state before Cycle {cycle_num + 1}.")
                         best_ep_so_far = ep_at_cycle_start; best_mse_so_far = mse_at_cycle_start
                    break
            log_message(f"\n--- Auto Needle+Thin Mode Finished after {num_cycles_done} cycles ---")
            termination_reason = "No improvement" if num_cycles_done < AUTO_MAX_CYCLES else f"Max {AUTO_MAX_CYCLES} cycles"
            log_message(f"Termination Reason: {termination_reason}")
            log_message(f"Final Best MSE: {best_mse_so_far:.6e} with {len(best_ep_so_far)} layers.")
            st.session_state.current_optimized_ep = best_ep_so_far.copy()
            st.session_state.optimization_ran_since_nominal_change = True
            st.session_state.current_material_sequence = None
            final_qwot_str = "QWOT Error"
            try:
                 qwots = calculate_qwot_from_ep(best_ep_so_far, l0, nH_material, nL_material, EXCEL_FILE_PATH)
                 if not np.any(np.isnan(qwots)): final_qwot_str = ",".join([f"{q:.3f}" for q in qwots])
                 else: final_qwot_str = "QWOT N/A"
            except Exception as e: log_message(f"Warning: Final QWOT calculation failed ({e})")
            st.session_state.optimized_qwot_string_display = final_qwot_str
            avg_nit = total_iters_auto / optim_runs_auto if optim_runs_auto > 0 else 0
            avg_nfev = total_evals_auto / optim_runs_auto if optim_runs_auto > 0 else 0
            final_status_text = f"Auto Mode Finished ({num_cycles_done} cyc, {termination_reason}) | MSE: {best_mse_so_far:.3e} | Layers: {len(best_ep_so_far)} | Avg Iter/Eval: {avg_nit:.1f}/{avg_nfev:.1f}"
            status_ui.update(label=final_status_text, state="complete", expanded=False)
            set_status(final_status_text)
            overall_success = True
            run_calculation_st(ep_vector_to_use=best_ep_so_far, is_optimized=True, method_name=f"Auto Mode ({num_cycles_done} cyc, {termination_reason})")
        except (ValueError, RuntimeError, TypeError) as e:
            err_msg = f"ERROR (Auto Mode Workflow/Setup): {e}"
            log_message(err_msg); st.error(err_msg)
            set_status(f"Auto Mode Failed ({e})")
            status_ui.update(label=f"Error: {e}", state="error", expanded=True)
        except Exception as e:
            err_msg = f"ERROR (Unexpected Auto Mode): {type(e).__name__}: {e}"
            tb_msg = traceback.format_exc()
            log_message(err_msg); log_message(tb_msg); print(err_msg); print(tb_msg)
            st.error(f"Unexpected Error during Auto Mode: {err_msg}")
            set_status("Auto Mode Failed (Unexpected)")
            status_ui.update(label=f"Unexpected Error: {e}", state="error", expanded=True)
        finally:
            log_message(f"--- Total Auto Mode Time: {time.time() - start_time_auto:.3f}s ---")
def _collect_design_data_st(include_optimized: bool = False) -> Dict:
     design_data = {'params': {}, 'targets': []}
     try:
        param_keys_to_save = ['l0', 'l_step', 'maxiter', 'maxfun', 'initial_layer_number', 'auto_thin_threshold']
        param_state_keys = ['input_l0', 'input_l_step', 'input_maxiter', 'input_maxfun', 'input_initial_layer_number', 'input_auto_thin_threshold']
        for key, state_key in zip(param_keys_to_save, param_state_keys):
            if state_key in st.session_state: design_data['params'][key] = st.session_state[state_key]
        design_data['params']['emp_str'] = st.session_state.input_emp_str
        design_data['params']['nH_material'] = st.session_state.selected_H_material
        design_data['params']['nL_material'] = st.session_state.selected_L_material
        design_data['params']['nSub_material'] = st.session_state.selected_Sub_material
        if st.session_state.selected_H_material == "Constant":
             design_data['params']['constant_nH_r'] = st.session_state.input_nH_r
             design_data['params']['constant_nH_i'] = st.session_state.input_nH_i
        if st.session_state.selected_L_material == "Constant":
             design_data['params']['constant_nL_r'] = st.session_state.input_nL_r
             design_data['params']['constant_nL_i'] = st.session_state.input_nL_i
        if st.session_state.selected_Sub_material == "Constant":
             design_data['params']['constant_nSub'] = st.session_state.input_nSub
        design_data['targets'] = st.session_state.targets
        if include_optimized and st.session_state.current_optimized_ep is not None and st.session_state.optimization_ran_since_nominal_change:
            design_data['optimized_ep'] = st.session_state.current_optimized_ep.tolist()
            design_data['optimized_qwot_string'] = st.session_state.optimized_qwot_string_display
     except Exception as e:
          log_message(f"Error collecting design data: {e}")
          st.error(f"Error collecting design data: {e}")
          return {}
     return design_data
def handle_load_design_st(uploaded_file_obj):
     log_message(f"\n--- Loading Design from: {uploaded_file_obj.name} ---")
     try:
          design_data = json.load(uploaded_file_obj)
          if 'params' not in design_data or 'targets' not in design_data:
                raise ValueError("Invalid JSON file: missing 'params' or 'targets'.")
          loaded_params = design_data.get('params', {})
          st.session_state.input_l0 = float(loaded_params.get('l0', st.session_state.default_l0))
          st.session_state.input_l_step = float(loaded_params.get('l_step', st.session_state.default_l_step))
          st.session_state.input_maxiter = int(loaded_params.get('maxiter', st.session_state.default_maxiter))
          st.session_state.input_maxfun = int(loaded_params.get('maxfun', st.session_state.default_maxfun))
          st.session_state.input_initial_layer_number = int(loaded_params.get('initial_layer_number', loaded_params.get('nmax', st.session_state.default_initial_layer_number))) # Handle old key 'nmax'
          st.session_state.input_auto_thin_threshold = float(loaded_params.get('auto_thin_threshold', st.session_state.default_auto_thin_threshold))
          st.session_state.input_emp_str = loaded_params.get('emp_str', "") # Load nominal QWOT
          # Load material selections carefully, checking against available options
          h_mat_loaded = loaded_params.get('nH_material', "Constant")
          if isinstance(h_mat_loaded, str) and h_mat_loaded in st.session_state.available_materials:
              st.session_state.selected_H_material = h_mat_loaded
          elif h_mat_loaded == "Constant" or isinstance(h_mat_loaded, (dict, list)): # Handle constant definition
              st.session_state.selected_H_material = "Constant"
              st.session_state.input_nH_r = float(loaded_params.get('constant_nH_r', st.session_state.default_nH_r))
              st.session_state.input_nH_i = float(loaded_params.get('constant_nH_i', st.session_state.default_nH_i))
          else: st.session_state.selected_H_material = "Constant" # Fallback
          # Similar logic for L and Substrate materials...
          l_mat_loaded = loaded_params.get('nL_material', "Constant")
          if isinstance(l_mat_loaded, str) and l_mat_loaded in st.session_state.available_materials:
              st.session_state.selected_L_material = l_mat_loaded
          elif l_mat_loaded == "Constant" or isinstance(l_mat_loaded, (dict, list)):
              st.session_state.selected_L_material = "Constant"
              st.session_state.input_nL_r = float(loaded_params.get('constant_nL_r', st.session_state.default_nL_r))
              st.session_state.input_nL_i = float(loaded_params.get('constant_nL_i', st.session_state.default_nL_i))
          else: st.session_state.selected_L_material = "Constant"
          sub_mat_loaded = loaded_params.get('nSub_material', "Constant")
          if isinstance(sub_mat_loaded, str) and sub_mat_loaded in st.session_state.combined_substrates:
              st.session_state.selected_Sub_material = sub_mat_loaded
          elif sub_mat_loaded == "Constant" or isinstance(sub_mat_loaded, (dict, list, float, int)):
              st.session_state.selected_Sub_material = "Constant"
              st.session_state.input_nSub = float(loaded_params.get('constant_nSub', st.session_state.default_nSub))
          else: st.session_state.selected_Sub_material = "Constant"
          # Load Targets
          loaded_targets = design_data.get('targets', [])
          st.session_state.targets = []
          for i in range(max(len(loaded_targets), 5)):
              if i < len(loaded_targets) and isinstance(loaded_targets[i], dict):
                   st.session_state.targets.append({
                        'enabled': loaded_targets[i].get('enabled', False),
                        'min': float(loaded_targets[i].get('min', 0)),
                        'max': float(loaded_targets[i].get('max', 0)),
                        'target_min': float(loaded_targets[i].get('target_min', 0)),
                        'target_max': float(loaded_targets[i].get('target_max', 0))
                   })
              else: st.session_state.targets.append({'min': 0.0, 'max': 0.0, 'target_min': 0.0, 'target_max': 0.0, 'enabled': False})
          handle_clear_optimized_state_st()
          if 'optimized_ep' in design_data and isinstance(design_data['optimized_ep'], list) and design_data['optimized_ep']:
               loaded_ep = np.array(design_data['optimized_ep'], dtype=np.float64)
               if np.all(np.isfinite(loaded_ep)) and np.all(loaded_ep >= 0):
                   st.session_state.current_optimized_ep = loaded_ep
                   st.session_state.optimization_ran_since_nominal_change = True
                   st.session_state.optimized_qwot_string_display = design_data.get('optimized_qwot_string', "QWOT Recalc Needed")
                   log_message(f"  Previous optimized state loaded ({len(loaded_ep)} layers).")
               else: log_message("  WARNING: Loaded optimized_ep data invalid. Ignoring.")
          else: log_message("  No 'optimized_ep' found in file. State set to Nominal.")
          log_message("Design loaded. Recalculating...")
          st.success(f"Design '{uploaded_file_obj.name}' loaded.")
          set_status("Design Loaded")
          if st.session_state.optimization_ran_since_nominal_change:
               run_calculation_st(ep_vector_to_use=st.session_state.current_optimized_ep, is_optimized=True, method_name="Loaded Opt")
          else: run_calculation_st(ep_vector_to_use=None, is_optimized=False, method_name="Loaded Nom")
     except Exception as e:
          log_message(f"ERROR loading design file: {e}\n{traceback.format_exc(limit=1)}")
          st.error(f"Failed to load design file.\nError: {e}")
          set_status("Load Design Error")
if st.session_state.get('show_material_plot', False):
     fig_mat_idx = display_material_index_plot()
     if fig_mat_idx:
          if st.session_state.get('material_index_plot_fig') is not None:
              try: plt.close(st.session_state.material_index_plot_fig)
              except Exception: pass
          st.header("🔬 Material Indices (n')")
          st.pyplot(fig_mat_idx)
          st.session_state.material_index_plot_fig = fig_mat_idx
     st.session_state.show_material_plot = False
# Update Download Button Data (generate data on each run)
save_data_nominal = json.dumps(_collect_design_data_st(include_optimized=False), indent=4)
save_nominal_placeholder.download_button(
     label="💾 Save Nominal Design", data=save_data_nominal, file_name="nominal_design.json", mime="application/json", key="save_nominal_dl_btn"
)
save_data_optimized = json.dumps(_collect_design_data_st(include_optimized=True), indent=4)
save_optimized_placeholder.download_button(
     label="💾 Save Optimized Design", data=save_data_optimized, file_name="optimized_design.json", mime="application/json", key="save_opt_dl_btn", disabled=(not st.session_state.optimization_ran_since_nominal_change)
)
# Trigger actions based on button presses
if eval_nom_pressed: handle_calculate_button_click_st(); st.rerun()
if local_opt_pressed: handle_local_optimizer_click_st(); st.rerun()
if start_nom_scan_pressed: handle_start_nominal_qwot_scan_st(); st.rerun()
if auto_mode_pressed: handle_auto_needle_mode_st(); st.rerun()
if remove_thin_pressed: handle_remove_thinnest_layer_st(); st.rerun()
if undo_remove_pressed: handle_undo_remove_layer_st(); st.rerun()
if set_nominal_pressed: handle_set_nominal_wrapper_st(); st.rerun()
if clear_opt_state_pressed: handle_clear_optimized_state_st(); st.rerun()
# Handle file loading trigger
if uploaded_file is not None:
     if st.session_state.get('last_uploaded_filename') != uploaded_file.name:
         handle_load_design_st(uploaded_file)
         st.session_state.last_uploaded_filename = uploaded_file.name
         st.rerun()
# Display final plot and status
if st.session_state.get('last_plot_fig') is not None:
    plot_placeholder.pyplot(st.session_state.last_plot_fig)
else:
    plot_placeholder.info("Run 'Evaluate Nominal' or an optimization to generate plots.")
status_placeholder.status(st.session_state.status_message, expanded=False, state="complete" if "Error" not in st.session_state.status_message and "Failed" not in st.session_state.status_message else "error")


