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
MAXITER_HARDCODED = 1000
MAXFUN_HARDCODED = 1000

# Global log message list (if you want to display it in the UI later)
# For now, add_log will just print to console if st.session_state.log_messages doesn't exist
# or append if it does.
def add_log(message: Union[str, List[str]]):
    # This is a placeholder. In a real app, you'd append to st.session_state.log_messages
    # and have a UI element to display them.
    # For the purpose of this modification, we'll assume it exists or prints.
    if 'log_messages' in st.session_state:
        if isinstance(message, list):
            st.session_state.log_messages.extend(message)
        else:
            st.session_state.log_messages.append(message)
    print(message)


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
                logs.append(f"WARNING: Constant index n'={nk_complex.real} <= 0 for '{material_definition}'. Using n'=1.0.")
                nk_complex = complex(1.0, nk_complex.imag)
            if nk_complex.imag < 0:
                logs.append(f"WARNING: Constant index k={nk_complex.imag} < 0 for '{material_definition}'. Using k=0.0.")
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
                    st.error(f"Could not load or empty data for material '{sheet_name}' from {excel_file_path}.")
                    logs.append(f"Critical error: Failed to load data for '{sheet_name}'.")
                    return None, logs
                l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
                l_target_min = jnp.min(l_vec_target_jnp)
                l_target_max = jnp.max(l_vec_target_jnp)
                l_data_min = jnp.min(l_data_jnp)
                l_data_max = jnp.max(l_data_jnp)
                if l_target_min < l_data_min - 1e-6 or l_target_max > l_data_max + 1e-6:
                    logs.append(f"WARNING: Interpolation for '{sheet_name}' out of bounds [{l_data_min:.1f}, {l_data_max:.1f}] nm (target: [{l_target_min:.1f}, {l_target_max:.1f}] nm). Extrapolation used.")
                result = interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)
        elif isinstance(material_definition, tuple) and len(material_definition) == 3:
            l_data, n_data, k_data = material_definition
            l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
            if not len(l_data_jnp): raise ValueError("Raw material data empty.")
            sort_indices = jnp.argsort(l_data_jnp)
            l_data_jnp = l_data_jnp[sort_indices]
            n_data_jnp = n_data_jnp[sort_indices]
            k_data_jnp = k_data_jnp[sort_indices]
            if np.any(k_data_jnp < -1e-9):
                logs.append("WARNING: k<0 in raw material data. Setting to 0.")
                k_data_jnp = jnp.maximum(k_data_jnp, 0.0)
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
        logs.append(f"Error: Target wavelength {l_nm_target}nm invalid for getting n+ik.")
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
    safe_l_val = jnp.maximum(l_val, 1e-9) # Avoid division by zero if l_val is 0
    phi = (2 * jnp.pi / safe_l_val) * (Ni * thickness)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    # Condition for layer computation: if thickness is negligible, M_layer is identity
    def compute_M_layer(thickness_: jnp.ndarray) -> jnp.ndarray:
        safe_eta = jnp.where(jnp.abs(eta) < 1e-12, 1e-12 + 0j, eta) # Avoid division by zero if eta is 0
        m01 = (1j / safe_eta) * sin_phi
        m10 = 1j * eta * sin_phi
        M_layer = jnp.array([[cos_phi, m01], [m10, cos_phi]], dtype=jnp.complex128)
        return M_layer @ carry_matrix

    def compute_identity(thickness_: jnp.ndarray) -> jnp.ndarray:
        return carry_matrix

    # If thickness is very small, treat as no layer (identity matrix multiplication)
    new_matrix = cond(thickness > 1e-12, compute_M_layer, compute_identity, thickness)
    return new_matrix, None

@jax.jit
def compute_stack_matrix_core_jax(ep_vector: jnp.ndarray, layer_indices: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    num_layers = len(ep_vector)
    # Prepare data for scan: (thicknesses, indices_for_this_lambda, lambda_value_repeated)
    layers_scan_data = (ep_vector, layer_indices, jnp.full(num_layers, l_val))
    M_initial = jnp.eye(2, dtype=jnp.complex128) # Start with identity matrix
    # Use jax.lax.scan to iteratively multiply layer matrices
    M_final, _ = scan(_compute_layer_matrix_scan_step_jit, M_initial, layers_scan_data)
    return M_final

@jax.jit
def calculate_single_wavelength_T_core(l_val: jnp.ndarray, ep_vector_contig: jnp.ndarray,
                                         layer_indices_at_lval: jnp.ndarray, nSub_at_lval: jnp.ndarray) -> jnp.ndarray:
    etainc = 1.0 + 0j # Refractive index of incident medium (air/vacuum)
    etasub = nSub_at_lval # Refractive index of substrate

    # Define calculation for valid lambda
    def calculate_for_valid_l(l_: jnp.ndarray) -> jnp.ndarray:
        current_layer_indices = layer_indices_at_lval
        M = compute_stack_matrix_core_jax(ep_vector_contig, current_layer_indices, l_)
        m00, m01 = M[0, 0], M[0, 1]
        m10, m11 = M[1, 0], M[1, 1]

        # Transmission calculation using matrix elements
        rs_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        rs_denominator_abs = jnp.abs(rs_denominator)
        safe_denominator = jnp.where(rs_denominator_abs < 1e-12, 1e-12 + 0j, rs_denominator) # Avoid division by zero

        ts = (2.0 * etainc) / safe_denominator
        real_etasub = jnp.real(etasub)
        real_etainc = jnp.real(etainc) # Should be 1.0
        safe_real_etainc = jnp.maximum(real_etainc, 1e-9) # Avoid division by zero if somehow etainc is 0
        Ts_complex = (real_etasub / safe_real_etainc) * (ts * jnp.conj(ts))
        Ts = jnp.real(Ts_complex) # Transmittance is real
        return jnp.where(rs_denominator_abs < 1e-12, jnp.nan, Ts) # Return NaN if denominator was too small

    # Define calculation for invalid lambda (e.g., l_val <= 0)
    def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray:
        return jnp.nan # Return NaN for invalid wavelengths

    # Use cond to choose path based on l_val
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
        logs.append("Empty lambda vector, no T calculation performed.")
        return {'l': np.array([]), 'Ts': np.array([])}, logs

    # Handle empty structure (0 layers) - calculate for bare substrate
    if not ep_vector_jnp.size:
        logs.append("Empty structure (0 layers). Calculating for bare substrate.")
        nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
        logs.extend(logs_sub)
        if nSub_arr is None:
            return None, logs
        # For a bare substrate, T can be approximated as 1 if we ignore reflections at the first interface for this simplified case,
        # or calculated more accurately. Here, we'll use a simplified T=1 for an empty stack.
        # A more rigorous calculation for bare substrate T = 4*n_inc*n_sub / (n_inc+n_sub)^2 if k_sub=0
        # For simplicity in this "empty stack" context, let's assume T=1 (no layers to cause interference)
        # This part might need refinement if a precise bare substrate calculation is needed.
        etainc_arr = jnp.ones_like(l_vec_jnp) + 0j
        t_interface = (2 * etainc_arr) / (etainc_arr + nSub_arr)
        Ts_bare = (jnp.real(nSub_arr) / jnp.real(etainc_arr)) * jnp.abs(t_interface)**2
        Ts_bare = jnp.nan_to_num(Ts_bare, nan=0.0) # Handle potential NaNs if nSub_arr had issues
        Ts_bare_clipped = jnp.clip(Ts_bare, 0.0, 1.0)
        return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts_bare_clipped)}, logs


    logs.append(f"Preparing indices for {len(l_vec_jnp)} lambdas...")
    start_time = time.time()
    nH_arr, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_jnp, excel_file_path)
    logs.extend(logs_h)
    nL_arr, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_jnp, excel_file_path)
    logs.extend(logs_l)
    nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
    logs.extend(logs_sub)

    if nH_arr is None or nL_arr is None or nSub_arr is None:
        logs.append("Critical error: Failed to load one of the material indices.")
        return None, logs
    logs.append(f"Index preparation finished in {time.time() - start_time:.3f}s.")

    calculate_single_wavelength_T_hl_jit = jax.jit(calculate_single_wavelength_T_core)
    num_layers = len(ep_vector_jnp)
    # Create an array of layer indices: for each layer, it's either nH_arr or nL_arr
    # Shape: (num_layers, num_lambdas)
    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH_arr, nL_arr)
    # Transpose for vmap: (num_lambdas, num_layers)
    indices_alternating_T = indices_alternating.T

    # Vectorize the calculation over lambdas
    # in_axes: (lambda_val, None, layer_indices_for_this_lambda, nSub_for_this_lambda)
    Ts_arr_raw = vmap(calculate_single_wavelength_T_hl_jit, in_axes=(0, None, 0, 0))(
        l_vec_jnp, ep_vector_jnp, indices_alternating_T, nSub_arr
    )
    Ts_arr = jnp.nan_to_num(Ts_arr_raw, nan=0.0) # Replace any NaNs with 0
    Ts_arr_clipped = jnp.clip(Ts_arr, 0.0, 1.0) # Ensure T is between 0 and 1
    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts_arr_clipped)}, logs

def calculate_T_from_ep_arbitrary_jax(ep_vector: Union[np.ndarray, List[float]],
                                          material_sequence: List[str], # List of material names for each layer
                                          nSub_material: MaterialInputType,
                                          l_vec: Union[np.ndarray, List[float]],
                                          excel_file_path: str) -> Tuple[Optional[Dict[str, np.ndarray]], List[str]]:
    logs = []
    l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64)
    ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)
    num_layers = len(ep_vector_jnp)

    if num_layers != len(material_sequence):
        logs.append("Error: Size of ep_vector and material_sequence must match.")
        st.error("Error: Thickness vector and material sequence have different lengths.")
        return None, logs

    if not l_vec_jnp.size:
        logs.append("Empty lambda vector.")
        return {'l': np.array([]), 'Ts': np.array([])}, logs

    if not ep_vector_jnp.size: # No layers
        logs.append("Empty structure (0 layers). Calculating for bare substrate.")
        nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
        logs.extend(logs_sub)
        if nSub_arr is None: return None, logs
        etainc_arr = jnp.ones_like(l_vec_jnp) + 0j
        t_interface = (2 * etainc_arr) / (etainc_arr + nSub_arr)
        Ts_bare = (jnp.real(nSub_arr) / jnp.real(etainc_arr)) * jnp.abs(t_interface)**2
        Ts_bare = jnp.nan_to_num(Ts_bare, nan=0.0)
        Ts_bare_clipped = jnp.clip(Ts_bare, 0.0, 1.0)
        return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts_bare_clipped)}, logs

    logs.append(f"Preparing indices for arbitrary sequence ({num_layers} layers, {len(l_vec_jnp)} lambdas)...")
    start_time = time.time()
    layer_indices_list = [] # Will store n+ik arrays for each layer
    materials_ok = True
    for i, mat_name in enumerate(material_sequence):
        nk_arr, logs_layer = _get_nk_array_for_lambda_vec(mat_name, l_vec_jnp, excel_file_path)
        logs.extend(logs_layer)
        if nk_arr is None:
            logs.append(f"Critical error: Failed to load material '{mat_name}' (layer {i+1}).")
            materials_ok = False
            break
        layer_indices_list.append(nk_arr)

    nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
    logs.extend(logs_sub)
    if nSub_arr is None:
        logs.append("Critical error: Failed to load substrate material.")
        materials_ok = False

    if not materials_ok:
        return None, logs

    # Stack the list of (num_lambda,) arrays into a (num_layers, num_lambda) array
    if layer_indices_list: # Ensure it's not empty before stacking
        layer_indices_arr = jnp.stack(layer_indices_list, axis=0)
    else: # Should not happen if num_layers > 0 and materials_ok is True, but as a safeguard
        layer_indices_arr = jnp.empty((0, len(l_vec_jnp)), dtype=jnp.complex128)


    logs.append(f"Index preparation finished in {time.time() - start_time:.3f}s.")

    calculate_single_wavelength_T_arb_jit = jax.jit(calculate_single_wavelength_T_core)
    # Transpose layer_indices_arr for vmap: (num_lambdas, num_layers)
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
        logs.append(f"Warning: l0={l0} <= 0 in calculate_initial_ep. Initial thicknesses set to 0.")
        return ep_initial, logs # Return zero thicknesses

    # Get n+ik at l0 for H and L materials
    nH_complex_at_l0, logs_h = _get_nk_at_lambda(nH0_material, l0, excel_file_path)
    logs.extend(logs_h)
    nL_complex_at_l0, logs_l = _get_nk_at_lambda(nL0_material, l0, excel_file_path)
    logs.extend(logs_l)

    if nH_complex_at_l0 is None or nL_complex_at_l0 is None:
        logs.append(f"Error: Could not get H or L indices at l0={l0}nm. Initial thicknesses set to 0.")
        st.error(f"Critical error getting indices at l0={l0}nm for initial thickness calculation.")
        return None, logs # Cannot proceed

    nH_real_at_l0 = nH_complex_at_l0.real
    nL_real_at_l0 = nL_complex_at_l0.real

    if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9: # Using 1e-9 as effectively zero or negative
        logs.append(f"WARNING: n'H({nH_real_at_l0:.3f}) or n'L({nL_real_at_l0:.3f}) at l0={l0}nm is <= 0. QWOT calculation may be incorrect.")
        # Proceeding, but thicknesses might become zero or very large

    for i in range(num_layers):
        multiplier = emp[i]
        is_H_layer = (i % 2 == 0) # Assuming alternating H/L structure
        n_real_layer_at_l0 = nH_real_at_l0 if is_H_layer else nL_real_at_l0

        if n_real_layer_at_l0 <= 1e-9: # If refractive index is non-physical
            ep_initial[i] = 0.0 # Set thickness to 0 to avoid division by zero or huge numbers
        else:
            ep_initial[i] = multiplier * l0 / (4.0 * n_real_layer_at_l0)

    # Ensure physical thicknesses (non-negative and above a minimum if desired)
    # Clamp thicknesses below MIN_THICKNESS_PHYS_NM (but > 0) to 0.0
    ep_initial_phys = np.where(ep_initial < MIN_THICKNESS_PHYS_NM, 0.0, ep_initial)
    num_clamped_zero = np.sum((ep_initial > 1e-12) & (ep_initial < MIN_THICKNESS_PHYS_NM)) # Count layers that were >0 but clamped
    if num_clamped_zero > 0:
        logs.append(f"Warning: {num_clamped_zero} initial thicknesses < {MIN_THICKNESS_PHYS_NM}nm were set to 0.")
        ep_initial = np.where(ep_initial < MIN_THICKNESS_PHYS_NM, 0.0, ep_initial)


    # Check for issues where QWOT was specified but thickness became zero due to bad index
    valid_indices = True
    for i in range(num_layers):
        if emp[i] > 1e-9 and ep_initial[i] < 1e-12: # If QWOT specified but thickness became zero
            layer_type = "H" if i % 2 == 0 else "L"
            n_val = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
            logs.append(f"Error: Layer {i+1} ({layer_type}) has QWOT={emp[i]} but thickness=0 (likely n'({layer_type},l0)={n_val:.3f} <= 0).")
            valid_indices = False
    if not valid_indices:
        st.error("Error during initial thickness calculation due to invalid indices at l0.")
        return None, logs # Return None if critical errors occurred

    return ep_initial, logs


def calculate_qwot_from_ep(ep_vector: np.ndarray, l0: float,
                               nH0_material: MaterialInputType, nL0_material: MaterialInputType,
                               excel_file_path: str) -> Tuple[Optional[np.ndarray], List[str]]:
    logs = []
    num_layers = len(ep_vector)
    qwot_multipliers = np.full(num_layers, np.nan, dtype=np.float64) # Initialize with NaN

    if l0 <= 0:
        logs.append(f"Warning: l0={l0} <= 0 in calculate_qwot_from_ep. QWOT set to NaN.")
        return qwot_multipliers, logs

    nH_complex_at_l0, logs_h = _get_nk_at_lambda(nH0_material, l0, excel_file_path)
    logs.extend(logs_h)
    nL_complex_at_l0, logs_l = _get_nk_at_lambda(nL0_material, l0, excel_file_path)
    logs.extend(logs_l)

    if nH_complex_at_l0 is None or nL_complex_at_l0 is None:
        logs.append(f"Error: Could not get n'H or n'L at l0={l0}nm to calculate QWOT. Returning NaN.")
        st.error(f"Error calculating QWOT: H/L indices not found at l0={l0}nm.")
        return None, logs # Critical error, return None

    nH_real_at_l0 = nH_complex_at_l0.real
    nL_real_at_l0 = nL_complex_at_l0.real

    if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
        logs.append(f"WARNING: n'H({nH_real_at_l0:.3f}) or n'L({nL_real_at_l0:.3f}) at l0={l0}nm is <= 0. QWOT calculation may be incorrect/NaN.")

    indices_ok = True
    for i in range(num_layers):
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real_layer_at_l0 <= 1e-9: # Non-physical index
            if ep_vector[i] > 1e-9 : # If layer has thickness but index is bad, QWOT is problematic (NaN)
                layer_type = "H" if i % 2 == 0 else "L"
                logs.append(f"Warning: Cannot calculate QWOT for layer {i+1} ({layer_type}) because n'({l0}nm) <= 0.")
                indices_ok = False # QWOT will be NaN for this layer
            else: # Layer has no thickness, QWOT is 0
                qwot_multipliers[i] = 0.0
        else: # Physical index, calculate QWOT
            qwot_multipliers[i] = ep_vector[i] * (4.0 * n_real_layer_at_l0) / l0
    
    if not indices_ok:
        st.warning("Some QWOT values could not be calculated (invalid indices at l0). They appear as NaN.")
        return qwot_multipliers, logs # Return array with NaNs
    else:
        return qwot_multipliers, logs

def calculate_final_mse(res: Dict[str, np.ndarray], active_targets: List[Dict]) -> Tuple[Optional[float], int]:
    total_squared_error = 0.0
    total_points_in_targets = 0
    mse = None # Initialize mse to None

    if not active_targets or 'Ts' not in res or res['Ts'] is None or 'l' not in res or res['l'] is None:
        return mse, total_points_in_targets

    res_l_np = np.asarray(res['l'])
    res_ts_np = np.asarray(res['Ts'])

    if res_l_np.size == 0 or res_ts_np.size == 0 or res_l_np.size != res_ts_np.size:
        return mse, total_points_in_targets # No valid data

    for target in active_targets:
        try:
            l_min = float(target['min'])
            l_max = float(target['max'])
            t_min = float(target['target_min'])
            t_max = float(target['target_max'])
            # Basic validation of target values
            if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0): continue # Skip invalid T targets
            if l_max < l_min: continue # Skip invalid lambda range
        except (KeyError, ValueError, TypeError):
            continue # Skip malformed targets

        # Find indices in the result that fall within the target's lambda range
        indices = np.where((res_l_np >= l_min) & (res_l_np <= l_max))[0]
        if indices.size > 0:
            calculated_Ts_in_zone = res_ts_np[indices]
            target_lambdas_in_zone = res_l_np[indices]

            # Filter out NaNs or Infs from calculated T's if any slipped through
            finite_mask = np.isfinite(calculated_Ts_in_zone)
            calculated_Ts_in_zone = calculated_Ts_in_zone[finite_mask]
            target_lambdas_in_zone = target_lambdas_in_zone[finite_mask]

            if calculated_Ts_in_zone.size == 0: continue # No valid points after filtering

            # Interpolate target T values for the lambdas in the zone
            if abs(l_max - l_min) < 1e-9: # Single point target or very narrow
                interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min) # Use t_min (or t_max, should be same)
            else: # Linear interpolation for sloped target
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
                                                   min_thickness_phys_nm: float) -> jnp.ndarray:
    # Penalty for thicknesses below min_thickness_phys_nm (but not zero)
    # We want to allow layers to become zero (effectively removed) without penalty.
    # Penalty applies if 0 < thickness < min_thickness_phys_nm
    below_min_mask = (ep_vector < min_thickness_phys_nm) & (ep_vector > 1e-12) # 1e-12 to avoid penalizing truly zero layers
    penalty_thin = jnp.sum(jnp.where(below_min_mask, (min_thickness_phys_nm - ep_vector)**2, 0.0))
    penalty_weight = 1e5 # Weight for the penalty term
    penalty_cost = penalty_thin * penalty_weight

    # For calculation, ensure thicknesses are at least min_thickness_phys_nm or 0
    # Layers that are < 1e-12 are considered 0 and not modified.
    # Layers between 1e-12 and min_thickness_phys_nm are effectively clamped to min_thickness_phys_nm for calculation.
    ep_vector_calc = jnp.where(ep_vector < 1e-12, 0.0, jnp.maximum(ep_vector, min_thickness_phys_nm))


    num_layers = len(ep_vector_calc)
    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH_arr, nL_arr)
    indices_alternating_T = indices_alternating.T # Shape (num_lambdas, num_layers)

    calculate_T_single_jit = jax.jit(calculate_single_wavelength_T_core)
    Ts_raw = vmap(calculate_T_single_jit, in_axes=(0, None, 0, 0))(
        l_vec_optim, ep_vector_calc, indices_alternating_T, nSub_arr
    )
    Ts = jnp.nan_to_num(Ts_raw, nan=0.0) # Replace NaN with 0 for MSE calculation

    total_squared_error = 0.0
    total_points_in_targets = 0
    for i in range(len(active_targets_tuple)): # Iterate over targets
        l_min, l_max, t_min, t_max = active_targets_tuple[i]
        target_mask = (l_vec_optim >= l_min) & (l_vec_optim <= l_max)

        # Interpolate target T values for all lambdas, then mask
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t_full = t_min + slope * (l_vec_optim - l_min)

        squared_errors_full = (Ts - interpolated_target_t_full)**2
        masked_sq_error = jnp.where(target_mask, squared_errors_full, 0.0) # Apply mask

        total_squared_error += jnp.sum(masked_sq_error)
        total_points_in_targets += jnp.sum(target_mask)

    mse = jnp.where(total_points_in_targets > 0,
                    total_squared_error / total_points_in_targets,
                    jnp.inf) # Return infinity if no target points
    final_cost = mse + penalty_cost
    return jnp.nan_to_num(final_cost, nan=jnp.inf, posinf=jnp.inf) # Ensure result is not NaN

@jax.jit
def calculate_mse_arbitrary_sequence_jax(ep_vector: jnp.ndarray,
                                           layer_indices_arr: jnp.ndarray, # Shape (num_layers, num_lambdas)
                                           nSub_arr: jnp.ndarray, # Shape (num_lambdas,)
                                           l_vec_eval: jnp.ndarray, # Shape (num_lambdas,)
                                           active_targets_tuple: Tuple[Tuple[float, float, float, float], ...]) -> jnp.ndarray:
    # Transpose layer_indices_arr for vmap: (num_lambdas, num_layers)
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
                           ) -> Tuple[Optional[np.ndarray], bool, float, List[str], str]:
    logs = []
    num_layers_start = len(ep_start_optim)
    optim_success = False
    final_cost = np.inf
    result_message_str = "Optimization not launched or failed early."
    final_ep = None # Initialize final_ep

    if num_layers_start == 0:
        logs.append(f"{log_prefix}Cannot optimize an empty structure.")
        return None, False, np.inf, logs, "Empty structure"

    try:
        l_min_optim = validated_inputs['l_range_deb']
        l_max_optim = validated_inputs['l_range_fin']
        l_step_optim = validated_inputs['l_step']
        nH_material = validated_inputs['nH_material']
        nL_material = validated_inputs['nL_material']
        nSub_material = validated_inputs['nSub_material']
        maxiter = MAXITER_HARDCODED # From global constants
        maxfun = MAXFUN_HARDCODED  # From global constants

        # Generate lambda vector for optimization
        num_pts_optim = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
        l_vec_optim_np = np.geomspace(l_min_optim, l_max_optim, num_pts_optim) # Use geomspace for potentially better sampling
        l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)] # Ensure valid lambdas
        if not l_vec_optim_np.size:
            raise ValueError("Failed to generate lambda vector for optimization.")
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)

        logs.append(f"{log_prefix}Preparing dispersive indices for {len(l_vec_optim_jax)} lambdas...")
        prep_start_time = time.time()
        nH_arr_optim, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_h)
        nL_arr_optim, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_l)
        nSub_arr_optim, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_sub)
        if nH_arr_optim is None or nL_arr_optim is None or nSub_arr_optim is None:
            raise RuntimeError("Failed to load indices for optimization.")
        logs.append(f"{log_prefix} Index preparation finished in {time.time() - prep_start_time:.3f}s.")

        active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
        static_args_for_jax = ( # Arguments for the JAX cost function
            nH_arr_optim, nL_arr_optim, nSub_arr_optim,
            l_vec_optim_jax, active_targets_tuple,
            min_thickness_phys # Pass min_thickness_phys here
        )

        # JIT compile the value and gradient function
        value_and_grad_fn = jax.jit(jax.value_and_grad(calculate_mse_for_optimization_penalized_jax))

        # Wrapper for SciPy optimizer
        def scipy_obj_grad_wrapper(ep_vector_np_in, *args):
            try:
                ep_vector_jax = jnp.asarray(ep_vector_np_in, dtype=jnp.float64)
                value_jax, grad_jax = value_and_grad_fn(ep_vector_jax, *args)
                # Handle non-finite values robustly
                if not jnp.isfinite(value_jax):
                    value_np = np.inf
                    grad_np = np.zeros_like(ep_vector_np_in, dtype=np.float64) # Zero gradient if cost is inf
                else:
                    value_np = float(np.array(value_jax))
                    grad_np_raw = np.array(grad_jax, dtype=np.float64)
                    grad_np = np.nan_to_num(grad_np_raw, nan=0.0, posinf=1e6, neginf=-1e6) # Replace NaNs/Infs in gradient
                return value_np, grad_np
            except Exception as e_wrap:
                # Log or print the error for debugging
                print(f"Error in scipy_obj_grad_wrapper: {e_wrap}") # Or use logs.append
                return np.inf, np.zeros_like(ep_vector_np_in, dtype=np.float64) # Return inf cost and zero gradient

        # Bounds for L-BFGS-B: thicknesses must be >= min_thickness_phys (or 0 if layer is meant to be removed)
        # The penalty in the cost function handles layers < min_thickness_phys but > 0.
        # The direct bound here ensures optimizer doesn't try negative values.
        # Using 0.0 as lower bound, penalty handles "too thin".
        lbfgsb_bounds = [(0.0, None)] * num_layers_start # Allow layers to become zero

        options = {'maxiter': maxiter, 'maxfun': maxfun,
                   'disp': False, # Set to True for detailed optimizer output
                   'ftol': 1e-12, 'gtol': 1e-8} # Optimizer tolerances

        logs.append(f"{log_prefix}Starting L-BFGS-B with JAX gradient...")
        opt_start_time = time.time()
        result = minimize(scipy_obj_grad_wrapper,
                          ep_start_optim,
                          args=static_args_for_jax,
                          method='L-BFGS-B',
                          jac=True, # We provide the gradient
                          bounds=lbfgsb_bounds,
                          options=options)
        logs.append(f"{log_prefix}L-BFGS-B (JAX grad) finished in {time.time() - opt_start_time:.3f}s.")
        
        final_cost = result.fun if np.isfinite(result.fun) else np.inf
        result_message_str = result.message.decode('utf-8') if isinstance(result.message, bytes) else str(result.message)

        # Check for success or if iteration/function call limit was reached (status 1 for L-BFGS-B)
        is_success_or_limit = (result.success or result.status == 1) and np.isfinite(final_cost)

        if is_success_or_limit:
            final_ep_raw = result.x
            # Post-optimization clamping: ensure layers are either 0 or >= min_thickness_phys
            final_ep = np.where(final_ep_raw < min_thickness_phys, 0.0, final_ep_raw)
            final_ep = np.maximum(final_ep, 0.0) # Ensure no negative thicknesses due to numerical precision

            optim_success = True
            log_status = "success" if result.success else "limit reached"
            logs.append(f"{log_prefix}Optimization finished ({log_status}). Final cost: {final_cost:.3e}, Msg: {result_message_str}")
        else:
            optim_success = False
            # If optimization failed, revert to the starting structure (clamped)
            final_ep = np.where(ep_start_optim < min_thickness_phys, 0.0, ep_start_optim)
            final_ep = np.maximum(final_ep, 0.0)
            
            logs.append(f"{log_prefix}Optimization FAILED. Status: {result.status}, Msg: {result_message_str}, Cost: {final_cost:.3e}")
            # Recalculate cost for the reverted structure if needed for logging/consistency
            try:
                reverted_cost, _ = scipy_obj_grad_wrapper(final_ep, *static_args_for_jax)
                logs.append(f"{log_prefix}Reverted to initial (clamped) structure. Recalculated cost: {reverted_cost:.3e}")
                final_cost = reverted_cost if np.isfinite(reverted_cost) else np.inf
            except Exception as cost_e:
                logs.append(f"{log_prefix}Reverted to initial (clamped) structure. ERROR recalculating cost: {cost_e}")
                final_cost = np.inf # Fallback cost


    except Exception as e_optim:
        logs.append(f"{log_prefix}Major ERROR during JAX/Scipy optimization: {e_optim}\n{traceback.format_exc(limit=2)}")
        st.error(f"Critical error during optimization: {e_optim}")
        # Fallback: use initial structure, clamped
        if ep_start_optim is not None:
            final_ep = np.where(ep_start_optim < min_thickness_phys, 0.0, ep_start_optim)
            final_ep = np.maximum(final_ep, 0.0)
        else: # Should not happen if num_layers_start > 0
            final_ep = None
        optim_success = False
        final_cost = np.inf # Cost is unknown/infinite due to error
        result_message_str = f"Exception: {e_optim}"

    return final_ep, optim_success, final_cost, logs, result_message_str

def _perform_layer_merge_or_removal_only(ep_vector_in: np.ndarray, min_thickness_phys: float,
                                           log_prefix: str = "", target_layer_index: Optional[int] = None,
                                           threshold_for_removal: Optional[float] = None) -> Tuple[Optional[np.ndarray], bool, List[str]]:
    current_ep = ep_vector_in.copy()
    logs = []
    num_layers = len(current_ep)
    structure_changed = False
    ep_after_merge = None # Initialize

    # Basic checks for very small structures
    if num_layers <= 2 and target_layer_index is None: # Cannot merge/remove from <=2 layers without a specific target to remove one of the ends
        logs.append(f"{log_prefix}Structure <= 2 layers. Removal/merge not possible without target for end layers.")
        return current_ep, False, logs
    elif num_layers < 1:
        logs.append(f"{log_prefix}Empty structure.")
        return current_ep, False, logs

    try:
        thin_layer_index = -1 # Index of the layer to be removed (0-indexed)
        min_thickness_found = np.inf

        # 1. Determine which layer to remove/merge around
        if target_layer_index is not None: # Manual override
            if 0 <= target_layer_index < num_layers and current_ep[target_layer_index] >= min_thickness_phys : # Target must be a valid, existing layer
                thin_layer_index = target_layer_index
                min_thickness_found = current_ep[target_layer_index]
                logs.append(f"{log_prefix}Manual targeting layer {thin_layer_index + 1} ({min_thickness_found:.3f} nm).")
            else:
                logs.append(f"{log_prefix}Manual target {target_layer_index+1} invalid or already too thin. Auto search for thinnest layer.")
                target_layer_index = None # Fallback to auto search

        if target_layer_index is None: # Auto-find thinnest layer (optionally below a threshold)
            # Consider only layers that are thick enough to be physically present
            candidate_indices = np.where(current_ep >= min_thickness_phys)[0]
            if candidate_indices.size == 0:
                logs.append(f"{log_prefix}No layer >= {min_thickness_phys:.3f} nm found to consider for removal.")
                return current_ep, False, logs

            candidate_thicknesses = current_ep[candidate_indices]
            indices_to_consider = candidate_indices
            thicknesses_to_consider = candidate_thicknesses
            
            # If a threshold is specified, filter candidates further
            if threshold_for_removal is not None:
                mask_below_threshold = thicknesses_to_consider < threshold_for_removal
                if np.any(mask_below_threshold):
                    indices_to_consider = indices_to_consider[mask_below_threshold]
                    thicknesses_to_consider = thicknesses_to_consider[mask_below_threshold]
                    logs.append(f"{log_prefix}Searching for thinnest layer among those < {threshold_for_removal:.3f} nm and >= {min_thickness_phys:.3f} nm.")
                else:
                    logs.append(f"{log_prefix}No eligible layer (< {threshold_for_removal:.3f} nm but >= {min_thickness_phys:.3f} nm) found for removal.")
                    return current_ep, False, logs # No layer meets this specific criterion
            
            if indices_to_consider.size > 0:
                min_idx_local = np.argmin(thicknesses_to_consider) # Find thinnest among candidates
                thin_layer_index = indices_to_consider[min_idx_local]
                min_thickness_found = thicknesses_to_consider[min_idx_local]
            else: # Should not happen if candidate_indices was not empty, unless threshold logic filtered all out
                logs.append(f"{log_prefix}No final candidate layer found after filtering.")
                return current_ep, False, logs
        
        if thin_layer_index == -1: # Should have been set if we reached here
            logs.append(f"{log_prefix}Failed to identify a layer for removal (unexpected case).")
            return current_ep, False, logs

        thin_layer_thickness = current_ep[thin_layer_index]
        logs.append(f"{log_prefix}Layer identified for action: Index {thin_layer_index} (Layer {thin_layer_index + 1}), thickness {thin_layer_thickness:.3f} nm.")

        # 2. Perform removal/merge based on layer position
        # IMPORTANT: The logic assumes H/L/H/L... structure. Merging L1+L3 means they are of the SAME material type.
        if num_layers <= 2 and thin_layer_index == 0: # Removing first of 1 or 2 layers
            # This case implies removing the layer and its (non-existent or single) partner, effectively clearing or reducing.
            # If num_layers = 1, ep_after_merge = []
            # If num_layers = 2, ep_after_merge = current_ep[1:2] (remaining single layer)
            # The original logic was current_ep[2:], which is problematic for num_layers=1 or 2.
            if num_layers == 1: ep_after_merge = np.array([])
            else: ep_after_merge = current_ep[1:2] # Keep the second layer
            merged_info = f"Removal of layer {thin_layer_index + 1} from a {num_layers}-layer structure."
            structure_changed = True
        elif num_layers <= 2 and thin_layer_index == 1: # Removing second of 2 layers
            ep_after_merge = current_ep[:1] # Keep the first layer
            merged_info = f"Removal of layer {thin_layer_index + 1} from a 2-layer structure."
            structure_changed = True
        elif thin_layer_index == 0: # Removing the first layer (if num_layers > 2)
            # The layer at index 1 (original L2) becomes the new L1. Its thickness is combined with L0 (removed).
            # This is tricky. Original: H1 L2 H3... Remove H1. L2 becomes H1.
            # The original code `current_ep[2:]` implies removing the first TWO layers.
            # Let's assume the goal is to remove the identified `thin_layer_index` and merge its neighbors.
            # If thin_layer_index is 0, its neighbors are "incident medium" and layer 1.
            # This means layer 0 is removed, and layer 1's thickness is preserved. Stack shifts.
            ep_after_merge = current_ep[1:]
            merged_info = f"Removal of first layer (Layer {thin_layer_index + 1})."
            structure_changed = True
        elif thin_layer_index == num_layers - 1: # Removing the last layer (if num_layers > 2)
            # Layer N is removed. Layer N-1's thickness is preserved.
            ep_after_merge = current_ep[:-1]
            merged_info = f"Removal of last layer (Layer {thin_layer_index + 1})."
            structure_changed = True
        else: # Removing an internal layer: merge its neighbors
            # Layer `i` is removed. Layers `i-1` and `i+1` are merged.
            # These neighbors are of the same material type in an H/L stack.
            merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
            ep_before = current_ep[:thin_layer_index - 1]
            ep_after = current_ep[thin_layer_index + 2:]
            ep_after_merge = np.concatenate((ep_before, [merged_thickness], ep_after))
            merged_info = f"Merging layers {thin_layer_index} and {thin_layer_index + 2} around removed layer {thin_layer_index + 1} -> new thickness {merged_thickness:.3f} nm."
            structure_changed = True

        if structure_changed and ep_after_merge is not None:
            logs.append(f"{log_prefix}{merged_info} New structure: {len(ep_after_merge)} layers.")
            # Ensure resulting thicknesses are not below physical minimum (or zero)
            ep_after_merge = np.where(ep_after_merge < min_thickness_phys, 0.0, ep_after_merge)
            ep_after_merge = np.maximum(ep_after_merge, 0.0) # Ensure non-negative
            return ep_after_merge, True, logs
        elif structure_changed and ep_after_merge is None: # Should not happen with current logic
            logs.append(f"{log_prefix}Logic error: structure_changed=True but ep_after_merge=None.")
            return current_ep, False, logs
        else: # No change made
            logs.append(f"{log_prefix}No structure modification performed (e.g. layer not found or edge case).")
            return current_ep, False, logs

    except Exception as e_merge:
        logs.append(f"{log_prefix}ERROR during merge/removal logic: {e_merge}\n{traceback.format_exc(limit=1)}")
        st.error(f"Internal error during layer removal/merge: {e_merge}")
        return current_ep, False, logs # Return original on error

def _perform_needle_insertion_scan(ep_vector_in: np.ndarray,
                                     nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
                                     l_vec_optim_np: np.ndarray, active_targets: List[Dict],
                                     cost_function_jax: Callable, # The JAX cost function (e.g., calculate_mse_for_optimization_penalized_jax)
                                     min_thickness_phys: float, base_needle_thickness_nm: float,
                                     scan_step: float, l0_repr: float, # l0_repr for logging/context, not directly used in cost
                                     excel_file_path: str, log_prefix: str = ""
                                     ) -> Tuple[Optional[np.ndarray], float, List[str], int]:
    logs = []
    num_layers_in = len(ep_vector_in)
    if num_layers_in == 0: # Cannot insert into an empty structure this way
        logs.append(f"{log_prefix}Needle scan impossible on empty structure. Consider starting with a nominal structure.")
        # To insert into an empty structure, one would typically add a single layer of H or L.
        # This function assumes splitting an existing layer.
        # For now, let's allow inserting into "substrate" (0 layers) by creating H L H structure.
        # This needs careful thought. The current logic splits existing layers.
        # Let's stick to splitting existing layers or inserting at interfaces for now.
        # If num_layers_in == 0, we can try inserting a H L H structure.
        # This function is designed to SPLIT existing layers.
        # A separate function or logic branch would be needed to "start" a stack with needles.
        # For now, if num_layers_in == 0, we can't proceed with splitting.
        # A simple initial structure could be [base_needle_thickness_nm] of H material.
        # Let's assume for now it can only split existing layers.
        if num_layers_in == 0:
             logs.append(f"{log_prefix}Needle scan cannot split layers in an empty structure.")
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
        static_args_cost_fn = ( # Arguments for the JAX cost function
            nH_arr_optim, nL_arr_optim, nSub_arr_optim,
            l_vec_optim_jax, active_targets_tuple,
            min_thickness_phys # Pass min_thickness_phys
        )
        # Calculate initial cost of the current structure
        initial_cost_jax = cost_function_jax(jnp.asarray(ep_vector_in), *static_args_cost_fn)
        initial_cost = float(np.array(initial_cost_jax))
        if not np.isfinite(initial_cost): # Critical if starting cost is not finite
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
    best_insertion_idx = -1 # Index of the *original* layer where insertion was best
    tested_insertions = 0

    # Iterate through all possible insertion points (within each layer)
    ep_cumsum = np.cumsum(ep_vector_in)
    total_thickness = ep_cumsum[-1] if num_layers_in > 0 else 0.0

    # Scan insertion points 'z' from scan_step up to total_thickness
    for z in np.arange(scan_step, total_thickness, scan_step):
        current_layer_idx = -1 # Index of the layer being split
        layer_start_z = 0.0
        for i in range(num_layers_in):
            layer_end_z = ep_cumsum[i]
            if z > layer_start_z and z <= layer_end_z: # z is within layer i
                # Check if splitting at z leaves two physically meaningful parts
                t_part1 = z - layer_start_z
                t_part2 = layer_end_z - z
                if t_part1 >= min_thickness_phys and t_part2 >= min_thickness_phys:
                    current_layer_idx = i
                else:
                    current_layer_idx = -2 # Invalid split point (too close to edge)
                break
            layer_start_z = layer_end_z

        if current_layer_idx < 0: # z is not a valid split point (or -2 if too close to edge)
            continue

        tested_insertions += 1
        # Thickness of the first part of the split layer
        t_layer_split_1 = z - (ep_cumsum[current_layer_idx-1] if current_layer_idx > 0 else 0.0)
        # Thickness of the second part of the split layer
        t_layer_split_2 = ep_cumsum[current_layer_idx] - z

        # Create temporary structure with inserted needle
        # Needle material alternates: if current_layer_idx is H (even), needle is L (odd relative to new structure).
        # This assumes the needle is of the *opposite* type of the layer it's inserted into.
        # For simplicity, let's assume the needle is always H if inserted into L, and L if inserted into H.
        # The cost function `calculate_mse_for_optimization_penalized_jax` assumes alternating H/L.
        # When we insert a layer, the parity of subsequent layers changes.
        # Example: H1, L2, H3. Insert in L2 (idx 1).
        # H1, (L2_part1, NEW_NEEDLE_H, L2_part2), H3_becomes_L4
        # This requires careful handling of indices in the cost function if it's not robust to this.
        # The provided cost function `calculate_mse_for_optimization_penalized_jax` uses `jnp.arange(num_layers)[:, None] % 2 == 0`
        # This means it strictly alternates H, L, H, L... based on the *new* layer index.
        # So, if we insert a needle, the structure becomes N+2 layers, and the H/L nature is determined by new index.
        # This is standard needle technique.

        ep_temp_np = np.concatenate((
            ep_vector_in[:current_layer_idx], # Layers before the split one
            [t_layer_split_1, base_needle_thickness_nm, t_layer_split_2], # Split layer + needle
            ep_vector_in[current_layer_idx+1:] # Layers after the split one
        ))
        # Clamp all layers in the temporary structure (including the new needle parts)
        ep_temp_np_clamped = np.where(ep_temp_np < min_thickness_phys, 0.0, ep_temp_np)
        ep_temp_np_clamped = np.maximum(ep_temp_np_clamped, 0.0)


        try:
            current_cost_jax = cost_function_jax(jnp.asarray(ep_temp_np_clamped), *static_args_cost_fn)
            current_cost = float(np.array(current_cost_jax))
            if np.isfinite(current_cost) and current_cost < min_cost_found:
                min_cost_found = current_cost
                best_ep_found = ep_temp_np_clamped.copy()
                best_insertion_idx = current_layer_idx # Store index of original layer
        except Exception as e_cost:
            logs.append(f"{log_prefix} WARNING: Failed cost calculation for z={z:.2f}. {e_cost}")
            continue # Skip this insertion point

    if best_ep_found is not None:
        improvement = initial_cost - min_cost_found
        logs.append(f"{log_prefix} Scan finished. {tested_insertions} points tested.")
        logs.append(f"{log_prefix} Best improvement found: {improvement:.6e} (MSE {min_cost_found:.6e})")
        logs.append(f"{log_prefix} Optimal insertion in original layer {best_insertion_idx + 1}.")
        return best_ep_found, min_cost_found, logs, best_insertion_idx
    else:
        logs.append(f"{log_prefix} Scan finished. {tested_insertions} points tested. No improvement found.")
        return None, initial_cost, logs, -1 # Return None for ep, original cost, and -1 for index

def _run_needle_iterations(ep_start: np.ndarray, num_needles: int,
                             validated_inputs: Dict, active_targets: List[Dict],
                             min_thickness_phys: float, l_vec_optim_np: np.ndarray,
                             scan_step_nm: float, base_needle_thickness_nm: float,
                             excel_file_path: str, log_prefix: str = ""
                             ) -> Tuple[np.ndarray, float, List[str]]:
    logs = []
    best_ep_overall = np.asarray(ep_start).copy()
    best_mse_overall = np.inf

    # Extract materials from validated_inputs
    nH_material = validated_inputs['nH_material']
    nL_material = validated_inputs['nL_material']
    nSub_material = validated_inputs['nSub_material']
    l0_repr = validated_inputs.get('l0', 500.0) # For logging context in needle scan

    # JIT compile the cost function once
    cost_fn_penalized_jit = jax.jit(calculate_mse_for_optimization_penalized_jax)

    try:
        # Pre-calculate n,k arrays for the cost function
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

        # Calculate initial MSE
        initial_cost_jax = cost_fn_penalized_jit(jnp.asarray(best_ep_overall), *static_args_cost_fn)
        best_mse_overall = float(np.array(initial_cost_jax))
        if not np.isfinite(best_mse_overall):
            raise ValueError("Initial MSE for needle iterations is not finite.")
        logs.append(f"{log_prefix} Starting needle iterations ({num_needles} max). Initial MSE: {best_mse_overall:.6e}")
    except Exception as e_init:
        logs.append(f"{log_prefix} ERROR calculating initial MSE for needle iterations: {e_init}")
        st.error(f"Error initializing needle iterations: {e_init}")
        return ep_start, np.inf, logs # Return original state on error

    for i in range(num_needles):
        logs.append(f"{log_prefix} --- Needle Iteration {i + 1}/{num_needles} ---")
        current_ep_iter = best_ep_overall.copy()
        num_layers_current = len(current_ep_iter)
        if num_layers_current == 0: # Should not happen if initial check is done, but safeguard
            logs.append(f"{log_prefix} Empty structure, stopping needle iterations."); break

        # 1. Perform Needle Insertion Scan
        st.write(f"{log_prefix} Needle scan {i+1}...") # UI update
        ep_after_scan, cost_after_scan, scan_logs, inserted_idx = _perform_needle_insertion_scan(
            current_ep_iter,
            nH_material, nL_material, nSub_material, # Pass materials
            l_vec_optim_np, active_targets,
            cost_fn_penalized_jit, # Pass the JITted cost function
            min_thickness_phys, base_needle_thickness_nm, scan_step_nm, l0_repr,
            excel_file_path, log_prefix=f"{log_prefix}  [Scan {i+1}] "
        )
        logs.extend(scan_logs)

        if ep_after_scan is None: # No improvement found by scan
            logs.append(f"{log_prefix} Needle scan {i + 1} found no improvement. Stopping needle iterations."); break

        # 2. Re-optimize the structure with the inserted needle
        logs.append(f"{log_prefix} Scan {i + 1} found potential improvement. Re-optimizing...")
        st.write(f"{log_prefix} Re-optimizing after needle {i+1}...") # UI update
        ep_after_reopt, optim_success, final_cost_reopt, optim_logs, optim_status_msg = \
            _run_core_optimization(ep_after_scan, validated_inputs, active_targets,
                                   min_thickness_phys, log_prefix=f"{log_prefix}  [Re-Opt {i+1}] ")
        logs.extend(optim_logs)

        if not optim_success:
            logs.append(f"{log_prefix} Re-optimization after scan {i + 1} FAILED. Stopping needle iterations."); break

        logs.append(f"{log_prefix} Re-optimization {i + 1} successful. New MSE: {final_cost_reopt:.6e}.")

        # 3. Check for overall improvement
        if final_cost_reopt < best_mse_overall - MSE_IMPROVEMENT_TOLERANCE:
            logs.append(f"{log_prefix}  MSE improved compared to previous best ({best_mse_overall:.6e}). Updating.")
            best_ep_overall = ep_after_reopt.copy()
            best_mse_overall = final_cost_reopt
        else:
            logs.append(f"{log_prefix}  New MSE ({final_cost_reopt:.6e}) not significantly better than previous ({best_mse_overall:.6e}). Stopping needle iterations.")
            # Even if not significantly better, update to the latest optimized structure if it's at least as good.
            # This can help escape local minima sometimes if the structure is different.
            # However, the original logic was to stop. Let's stick to stopping if no significant improvement.
            # If we want to keep the result even if not better, uncomment below and remove break.
            # best_ep_overall = ep_after_reopt.copy()
            # best_mse_overall = final_cost_reopt
            break

    logs.append(f"{log_prefix} End of needle iterations. Best final MSE: {best_mse_overall:.6e}")
    return best_ep_overall, best_mse_overall, logs

def run_auto_mode(initial_ep: Optional[np.ndarray],
                  validated_inputs: Dict, active_targets: List[Dict],
                  excel_file_path: str, log_callback: Callable):
    logs = [] # Local logs for this function, will be returned
    start_time_auto = time.time()
    log_callback("#"*10 + f" Starting Auto Mode (Max {AUTO_MAX_CYCLES} Cycles) " + "#"*10)

    best_ep_so_far = None
    best_mse_so_far = np.inf
    num_cycles_done = 0
    termination_reason = f"Max {AUTO_MAX_CYCLES} cycles reached"
    threshold_for_thin_removal = validated_inputs.get('auto_thin_threshold', 1.0) # Default if not in inputs
    log_callback(f"  Auto removal threshold: {threshold_for_thin_removal:.3f} nm")

    try:
        current_ep = None # Will hold the structure being worked on in each cycle
        # Setup initial structure and MSE for Auto Mode
        if initial_ep is not None: # Starting from a previously optimized state
            log_callback("  Auto Mode: Using previous optimized structure as starting point.")
            current_ep = initial_ep.copy()
            # Calculate MSE for this initial_ep
            l_min_optim, l_max_optim = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
            l_step_optim = validated_inputs['l_step']
            num_pts = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
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
            cost_fn_jit = jax.jit(calculate_mse_for_optimization_penalized_jax) # JIT the cost function
            initial_mse_jax = cost_fn_jit(jnp.asarray(current_ep), *static_args)
            initial_mse = float(np.array(initial_mse_jax))
            if not np.isfinite(initial_mse): raise ValueError("Initial MSE (from optimized state) not finite.")
            best_mse_so_far = initial_mse
            best_ep_so_far = current_ep.copy()
            log_callback(f"  Initial MSE (from provided optimized state): {best_mse_so_far:.6e}")
        else: # Starting from nominal QWOT structure
            log_callback("  Auto Mode: Using nominal structure (QWOT) as starting point.")
            emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
            if not emp_list: raise ValueError("Nominal QWOT for Auto Mode is empty.")
            ep_nominal, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'],
                                                            validated_inputs['nH_material'], validated_inputs['nL_material'],
                                                            excel_file_path)
            log_callback(logs_ep_init)
            if ep_nominal is None: raise RuntimeError("Failed to calculate initial nominal thicknesses for Auto Mode.")
            log_callback(f"  Nominal structure: {len(ep_nominal)} layers. Starting initial optimization...")
            st.info("Auto Mode: Initial optimization of nominal structure...") # UI Feedback
            ep_after_initial_opt, initial_opt_success, initial_mse, initial_opt_logs, initial_opt_msg = \
                _run_core_optimization(ep_nominal, validated_inputs, active_targets,
                                       MIN_THICKNESS_PHYS_NM, log_prefix="  [Auto Init Opt] ")
            log_callback(initial_opt_logs)
            if not initial_opt_success:
                log_callback(f"ERROR: Failed initial optimization in Auto Mode ({initial_opt_msg}). Aborting.")
                st.error(f"Failed initial optimization of Auto Mode: {initial_opt_msg}")
                return None, np.inf, logs # Return logs collected so far
            log_callback(f"  Initial optimization finished. MSE: {initial_mse:.6e}")
            best_ep_so_far = ep_after_initial_opt.copy()
            best_mse_so_far = initial_mse

        if best_ep_so_far is None or not np.isfinite(best_mse_so_far): # Should be caught by earlier checks
            raise RuntimeError("Invalid starting state for Auto cycles (ep or MSE is invalid).")

        # --- Auto Mode Cycles ---
        log_callback(f"--- Starting Auto Cycles (Starting MSE: {best_mse_so_far:.6e}, {len(best_ep_so_far)} layers) ---")
        for cycle_num in range(AUTO_MAX_CYCLES):
            log_callback(f"\n--- Auto Cycle {cycle_num + 1} / {AUTO_MAX_CYCLES} ---")
            st.info(f"Auto Cycle {cycle_num + 1}/{AUTO_MAX_CYCLES} | Current MSE: {best_mse_so_far:.3e}, Layers: {len(best_ep_so_far)}") # UI Feedback
            mse_at_cycle_start = best_mse_so_far
            ep_at_cycle_start = best_ep_so_far.copy() # To revert if cycle worsens things
            cycle_improved_globally = False # Track if this cycle made any net improvement

            # 1. Needle Phase
            log_callback(f"  [Cycle {cycle_num+1}] Needle Phase ({AUTO_NEEDLES_PER_CYCLE} max iterations)...")
            st.write(f"Cycle {cycle_num + 1}: Needle Phase...") # UI Feedback
            # Prepare lambda vector for needle iterations (can be same as initial auto MSE calc)
            l_min_optim, l_max_optim = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
            l_step_optim = validated_inputs['l_step']
            num_pts_needle = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
            l_vec_optim_np_needle = np.geomspace(l_min_optim, l_max_optim, num_pts_needle)
            l_vec_optim_np_needle = l_vec_optim_np_needle[(l_vec_optim_np_needle > 0) & np.isfinite(l_vec_optim_np_needle)]
            if not l_vec_optim_np_needle.size:
                log_callback("  ERROR: cannot generate lambda for needle phase. Cycle aborted.")
                break # Exit auto mode cycles

            ep_after_needles, mse_after_needles, needle_logs_iter = \
                _run_needle_iterations(best_ep_so_far, AUTO_NEEDLES_PER_CYCLE, validated_inputs, active_targets,
                                       MIN_THICKNESS_PHYS_NM, l_vec_optim_np_needle,
                                       DEFAULT_NEEDLE_SCAN_STEP_NM, BASE_NEEDLE_THICKNESS_NM,
                                       excel_file_path, log_prefix=f"    [Needle {cycle_num+1}] ")
            log_callback(needle_logs_iter)
            log_callback(f"  [Cycle {cycle_num+1}] End Needle Phase. MSE: {mse_after_needles:.6e}, Layers: {len(ep_after_needles)}")

            if mse_after_needles < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                log_callback(f"    Global improvement by needle phase (vs {best_mse_so_far:.6e}).")
                best_ep_so_far = ep_after_needles.copy()
                best_mse_so_far = mse_after_needles
                cycle_improved_globally = True
            else: # No significant improvement, or it got worse
                log_callback(f"    No significant global improvement by needle phase (vs {best_mse_so_far:.6e}).")
                # Keep the result from needles if it's not much worse, as structure might be better for thinning
                if mse_after_needles < best_mse_so_far + MSE_IMPROVEMENT_TOLERANCE * 10: # Allow slight worsening
                     best_ep_so_far = ep_after_needles.copy()
                     best_mse_so_far = mse_after_needles
                # else, best_ep_so_far and best_mse_so_far remain from before needle phase


            # 2. Thinning Phase (Remove thin layers and re-optimize)
            log_callback(f"  [Cycle {cycle_num+1}] Thinning Phase (target layers < {threshold_for_thin_removal:.3f} nm) + Re-Opt...")
            st.write(f"Cycle {cycle_num + 1}: Thinning Phase...") # UI Feedback
            layers_removed_this_cycle = 0;
            # Max attempts to remove layers = current number of layers + a margin, to prevent infinite loops if removal fails to reduce layers
            max_thinning_attempts = len(best_ep_so_far) + 2
            for attempt in range(max_thinning_attempts):
                current_num_layers_thin = len(best_ep_so_far)
                if current_num_layers_thin <= 2: # Cannot thin further if only 2 layers or less
                    log_callback("    Structure too small (< 3 layers), stopping thinning for this cycle.")
                    break

                ep_after_single_removal, structure_changed, removal_logs_iter = \
                    _perform_layer_merge_or_removal_only(best_ep_so_far, MIN_THICKNESS_PHYS_NM,
                                                         log_prefix=f"    [Thin {cycle_num+1}.{attempt+1}] ",
                                                         threshold_for_removal=threshold_for_thin_removal) # Use the threshold
                log_callback(removal_logs_iter)

                if structure_changed and ep_after_single_removal is not None:
                    layers_removed_this_cycle += (current_num_layers_thin - len(ep_after_single_removal)) # Count actual layers removed
                    log_callback(f"    Layer(s) removed/merged. Re-optimizing ({len(ep_after_single_removal)} layers)...")
                    st.write(f"Cycle {cycle_num + 1}: Re-opt after removal {layers_removed_this_cycle}...") # UI Feedback
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
                        else: # No global improvement, but accept the new structure if MSE is comparable
                            log_callback(f"      No significant global improvement (vs {best_mse_so_far:.6e}). Continuing with this thinned+reopt structure.")
                            best_ep_so_far = ep_after_thin_reopt.copy()
                            best_mse_so_far = mse_after_thin_reopt
                    else: # Re-optimization after thinning failed
                        log_callback(f"    WARNING: Re-optimization after thinning FAILED ({thin_reopt_msg}). Stopping thinning for this cycle.")
                        # Revert to state before this specific removal attempt if re-opt failed badly.
                        # Or, keep ep_after_single_removal and its (likely poor) MSE.
                        # For now, keep the removed structure and try to get its MSE.
                        best_ep_so_far = ep_after_single_removal.copy()
                        try:
                            # Recalculate MSE for the structure that failed re-optimization
                            current_mse_jax = cost_fn_jit(jnp.asarray(best_ep_so_far), *static_args) # Use pre-calculated static_args
                            best_mse_so_far = float(np.array(current_mse_jax))
                            if not np.isfinite(best_mse_so_far): best_mse_so_far = np.inf
                            log_callback(f"      MSE after failed re-opt (reduced structure, not opt): {best_mse_so_far:.6e}")
                        except Exception as e_cost_fail:
                            log_callback(f"      ERROR recalculating MSE after failed re-opt: {e_cost_fail}"); best_mse_so_far = np.inf
                        break # Stop thinning for this cycle if re-opt fails
                else: # No layer was removed (e.g., no layer met thinning criteria)
                    log_callback("    No further layers to remove/merge in this thinning phase attempt.")
                    break # Exit thinning attempts for this cycle
            log_callback(f"  [Cycle {cycle_num+1}] End Thinning Phase. {layers_removed_this_cycle} layer(s) effectively removed in this cycle.")

            num_cycles_done += 1
            log_callback(f"--- End Auto Cycle {cycle_num + 1} --- Best current MSE: {best_mse_so_far:.6e} ({len(best_ep_so_far)} layers) ---")

            # Check for termination condition for the entire Auto Mode
            if not cycle_improved_globally and best_mse_so_far >= mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE :
                log_callback(f"No significant improvement in Cycle {cycle_num + 1} (Start: {mse_at_cycle_start:.6e}, End: {best_mse_so_far:.6e}). Stopping Auto Mode.")
                termination_reason = f"No improvement (Cycle {cycle_num + 1})"
                # If MSE actually got worse, revert to the state before this cycle
                if best_mse_so_far > mse_at_cycle_start + MSE_IMPROVEMENT_TOLERANCE : # Check if it got significantly worse
                    log_callback("  MSE significantly increased in this cycle, reverting to state before this cycle.")
                    best_ep_so_far = ep_at_cycle_start.copy()
                    best_mse_so_far = mse_at_cycle_start
                break # Exit Auto Mode

        log_callback(f"\n--- Auto Mode Finished after {num_cycles_done} cycles ---")
        log_callback(f"Reason: {termination_reason}")
        log_callback(f"Best final MSE: {best_mse_so_far:.6e} with {len(best_ep_so_far)} layers.")
        return best_ep_so_far, best_mse_so_far, logs

    except (ValueError, RuntimeError, TypeError) as e:
        log_callback(f"FATAL ERROR during Auto Mode (Setup/Workflow): {e}")
        st.error(f"Auto Mode Error: {e}")
        return None, np.inf, logs # Return logs collected so far
    except Exception as e_fatal: # Catch any other unexpected errors
        log_callback(f"Unexpected FATAL ERROR during Auto Mode: {type(e_fatal).__name__}: {e_fatal}")
        st.error(f"Unexpected Auto Mode Error: {e_fatal}")
        traceback.print_exc() # Print stack trace to console for debugging
        return None, np.inf, logs

@jax.jit
def calculate_M_for_thickness(thickness: jnp.ndarray, n_complex_layer: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    eta = n_complex_layer
    safe_l_val = jnp.maximum(l_val, 1e-9)
    safe_eta = jnp.where(jnp.abs(eta) < 1e-12, 1e-12 + 0j, eta) # Avoid division by zero if eta is 0
    phi = (2 * jnp.pi / safe_l_val) * (n_complex_layer * thickness) # n_complex_layer already includes k
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    # If thickness is negligible, M_layer is identity
    m00 = jnp.where(thickness > 1e-12, cos_phi, 1.0)
    m01 = jnp.where(thickness > 1e-12, (1j / safe_eta) * sin_phi, 0.0)
    m10 = jnp.where(thickness > 1e-12, 1j * eta * sin_phi, 0.0) # use original eta here
    m11 = jnp.where(thickness > 1e-12, cos_phi, 1.0)
    M_layer = jnp.array([[m00, m01], [m10, m11]], dtype=jnp.complex128)
    return M_layer

# Batch calculation of M_layer over different lambda values (l_vec)
calculate_M_batch_for_thickness = vmap(calculate_M_for_thickness, in_axes=(None, None, 0))

@jax.jit
def get_layer_matrices_qwot(layer_idx: int, initial_layer_number: int, l0: float,
                                nH_c_l0: jnp.ndarray, nL_c_l0: jnp.ndarray,
                                l_vec: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Determine if the current layer (by index) is H or L
    predicate_is_H = (layer_idx % 2 == 0)
    # Get the real part of the index at l0 for the layer type
    n_real_l0 = jax.lax.select(predicate_is_H, nH_c_l0.real, nL_c_l0.real)
    # Get the full complex index at l0 for the layer type (needed for matrix calculation)
    n_complex_for_matrix = jax.lax.select(predicate_is_H, nH_c_l0, nL_c_l0)

    # Calculate denominator for QWOT thickness, ensuring it's not zero
    denom = 4.0 * jnp.maximum(n_real_l0, 1e-9)
    safe_l0 = jnp.maximum(l0, 1e-9) # Ensure l0 is not zero

    # Calculate 1x QWOT and 2x QWOT thicknesses
    ep1_calc = 1.0 * safe_l0 / denom
    ep2_calc = 2.0 * safe_l0 / denom

    # Set thickness to 0 if the index was non-physical (<=0)
    ep1 = jnp.where(n_real_l0 > 1e-9, ep1_calc, 0.0)
    ep2 = jnp.where(n_real_l0 > 1e-9, ep2_calc, 0.0)

    # Calculate the characteristic matrices for both thicknesses across all lambdas in l_vec
    M_1qwot_batch = calculate_M_batch_for_thickness(ep1, n_complex_for_matrix, l_vec)
    M_2qwot_batch = calculate_M_batch_for_thickness(ep2, n_complex_for_matrix, l_vec)
    return M_1qwot_batch, M_2qwot_batch

@jax.jit
def compute_half_product(multiplier_indices: jnp.ndarray, # (N_half,) array of 0s (1x QWOT) or 1s (2x QWOT)
                           layer_matrices_half: jnp.ndarray # (N_half, 2, L, 2, 2) array [layer, multiplier_idx, lambda, mat_row, mat_col]
                           ) -> jnp.ndarray: # Returns (L, 2, 2) product matrix for this combination for all lambdas
    N_half = layer_matrices_half.shape[0] # Number of layers in this half
    L = layer_matrices_half.shape[2] # Number of lambdas

    # Initial product matrix (identity) for each lambda
    init_prod = jnp.tile(jnp.eye(2, dtype=jnp.complex128), (L, 1, 1)) # Shape (L, 2, 2)

    # Scan function to iteratively multiply matrices
    def multiply_step(carry_prod: jnp.ndarray, layer_idx: int) -> Tuple[jnp.ndarray, None]:
        # carry_prod shape: (L, 2, 2)
        # layer_idx: current layer index (0 to N_half-1)
        multiplier_idx = multiplier_indices[layer_idx] # 0 or 1
        # Select the correct pre-calculated matrix (1x or 2x QWOT) for this layer and all lambdas
        # M_k shape: (L, 2, 2)
        M_k = layer_matrices_half[layer_idx, multiplier_idx, :, :, :]
        # Perform batch matrix multiplication: M_k @ carry_prod for each lambda
        new_prod = vmap(jnp.matmul)(M_k, carry_prod) # vmap over the lambda dimension (axis 0)
        return new_prod, None

    # Scan over the layers in this half
    final_prod, _ = jax.lax.scan(multiply_step, init_prod, jnp.arange(N_half))
    # final_prod shape: (L, 2, 2)
    return final_prod

@jax.jit
def get_T_from_batch_matrix(M_batch: jnp.ndarray, # Shape (..., L, 2, 2) where ... is batch dims
                              nSub_arr: jnp.ndarray # Shape (L,)
                              ) -> jnp.ndarray: # Returns Ts array shape (..., L)
    etainc = 1.0 + 0j # Incident medium index (assumed air)
    etasub_batch = nSub_arr # Substrate index (complex) for each lambda

    # Extract matrix elements, keeping batch dimensions
    m00 = M_batch[..., 0, 0]; m01 = M_batch[..., 0, 1]
    m10 = M_batch[..., 1, 0]; m11 = M_batch[..., 1, 1]

    # Calculate denominator, avoiding division by zero
    rs_den = (etainc * m00 + etasub_batch * m11 + etainc * etasub_batch * m01 + m10)
    rs_den_abs = jnp.abs(rs_den)
    safe_den = jnp.where(rs_den_abs < 1e-12, 1e-12 + 0j, rs_den)

    # Calculate transmission coefficient (amplitude)
    ts = (2.0 * etainc) / safe_den
    # Calculate Transmittance (intensity)
    real_etasub_batch = jnp.real(etasub_batch)
    safe_real_etainc = 1.0 # Real part of incident index (air)
    Ts_complex = (real_etasub_batch / safe_real_etainc) * (ts * jnp.conj(ts))
    Ts = jnp.real(Ts_complex) # Transmittance is real

    # Handle cases where denominator was near zero -> T=0, also handle potential NaNs
    return jnp.where(rs_den_abs < 1e-12, 0.0, jnp.nan_to_num(Ts, nan=0.0))

@jax.jit
def calculate_mse_basic_jax(Ts: jnp.ndarray, # Shape (..., L)
                              l_vec: jnp.ndarray, # Shape (L,)
                              targets_tuple: Tuple[Tuple[float, float, float, float], ...]
                              ) -> jnp.ndarray: # Returns MSE array shape (...)
    total_squared_error = 0.0
    total_points_in_targets = 0

    for i in range(len(targets_tuple)):
        l_min, l_max, t_min, t_max = targets_tuple[i]
        target_mask = (l_vec >= l_min) & (l_vec <= l_max) # Shape (L,)

        # Interpolate target T values across the lambda vector
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t = t_min + slope * (l_vec - l_min) # Shape (L,)

        # Calculate squared errors for all lambdas
        squared_errors = (Ts - interpolated_target_t)**2 # Shape (..., L)
        # Apply mask to consider only errors within the target range
        masked_sq_error = jnp.where(target_mask, squared_errors, 0.0) # Shape (..., L)

        # Sum errors over the lambda dimension (L)
        total_squared_error += jnp.sum(masked_sq_error, axis=-1) # Shape (...)
        total_points_in_targets += jnp.sum(target_mask) # Scalar: total number of points across all targets

    # Calculate MSE, handle case with zero target points
    mse = jnp.where(total_points_in_targets > 0,
                    total_squared_error / total_points_in_targets,
                    jnp.inf) # Return Inf if no points in target
    # Ensure final MSE is not NaN or Inf (replace with Inf)
    return jnp.nan_to_num(mse, nan=jnp.inf, posinf=jnp.inf)

@jax.jit
def combine_and_calc_mse(prod1: jnp.ndarray, # Shape (L, 2, 2) - Product of first half matrices
                           prod2: jnp.ndarray, # Shape (L, 2, 2) - Product of second half matrices
                           nSub_arr_in: jnp.ndarray, # Shape (L,)
                           l_vec_in: jnp.ndarray, # Shape (L,)
                           targets_tuple_in: Tuple # Tuple of target tuples
                           ) -> jnp.ndarray: # Returns scalar MSE
    # Combine the products: M_total = M_half2 @ M_half1
    M_total = vmap(jnp.matmul)(prod2, prod1) # vmap over lambda dimension, shape (L, 2, 2)
    # Calculate Transmittance from the total matrix
    Ts = get_T_from_batch_matrix(M_total, nSub_arr_in) # Shape (L,)
    # Calculate MSE
    mse = calculate_mse_basic_jax(Ts, l_vec_in, targets_tuple_in) # Scalar result
    return mse

def _execute_split_stack_scan(current_l0: float, initial_layer_number: int,
                                nH_c_l0: complex, nL_c_l0: complex, # Indices at the specific l0 being scanned
                                nSub_arr_scan: jnp.ndarray, # Substrate index array for eval lambdas
                                l_vec_eval_sparse_jax: jnp.ndarray, # Sparse lambda vector for evaluation
                                active_targets_tuple: Tuple, # Targets for MSE calculation
                                log_callback: Callable) -> Tuple[float, Optional[np.ndarray], List[str]]:
    logs = []
    L_sparse = len(l_vec_eval_sparse_jax)
    num_combinations = 2**initial_layer_number
    log_callback(f"  [Scan l0={current_l0:.2f}] Testing {num_combinations:,} QWOT combinations (1.0x/2.0x)...")

    precompute_start_time = time.time()
    st.write(f"Scan l0={current_l0:.1f}: Pre-calculating matrices...") # UI Feedback
    layer_matrices_list = []
    try:
        get_layer_matrices_qwot_jit = jax.jit(get_layer_matrices_qwot)
        # Pre-calculate M_1qwot and M_2qwot for each layer at this l0 across all eval lambdas
        for i in range(initial_layer_number):
            m1, m2 = get_layer_matrices_qwot_jit(i, initial_layer_number, current_l0,
                                                  jnp.asarray(nH_c_l0), jnp.asarray(nL_c_l0), # Use indices for current l0
                                                  l_vec_eval_sparse_jax) # Calculate for sparse eval lambdas
            # Stack the 1x and 2x QWOT matrices for this layer: shape (2, L_sparse, 2, 2)
            layer_matrices_list.append(jnp.stack([m1, m2], axis=0))
        # Stack matrices for all layers: shape (N, 2, L_sparse, 2, 2)
        all_layer_matrices = jnp.stack(layer_matrices_list, axis=0)
        all_layer_matrices.block_until_ready() # Ensure calculation is complete
        log_callback(f"    Pre-calculation of matrices (l0={current_l0:.2f}) finished in {time.time() - precompute_start_time:.3f}s.")
    except Exception as e_mat:
        logs.append(f"  ERROR Pre-calculating Matrices for l0={current_l0:.2f}: {e_mat}")
        st.error(f"Error pre-calculating QWOT scan matrices: {e_mat}")
        return np.inf, None, logs

    # Split stack computation (meet-in-the-middle)
    N = initial_layer_number
    N1 = N // 2 # Number of layers in first half
    N2 = N - N1 # Number of layers in second half
    num_comb1 = 2**N1
    num_comb2 = 2**N2

    log_callback(f"    Calculating partial products 1 ({num_comb1:,} combinations)...")
    st.write(f"Scan l0={current_l0:.1f}: Partial products 1...") # UI Feedback
    half1_start_time = time.time()
    # Generate all 2^N1 combinations of multiplier indices (0 or 1) for the first half
    indices1 = jnp.arange(num_comb1)
    powers1 = 2**jnp.arange(N1)
    # Shape (num_comb1, N1) - each row is a combination of 0s and 1s
    multiplier_indices1 = jnp.not_equal(indices1[:, None] & powers1, 0).astype(jnp.int32)
    # Get the pre-calculated matrices for the first half
    matrices_half1 = all_layer_matrices[:N1] # Shape (N1, 2, L_sparse, 2, 2)
    compute_half_product_jit = jax.jit(compute_half_product)
    # Calculate product matrices for all combinations in the first half
    # vmap over the combinations (axis 0 of multiplier_indices1)
    # partial_products1 shape: (num_comb1, L_sparse, 2, 2)
    partial_products1 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices1, matrices_half1)
    partial_products1.block_until_ready()
    log_callback(f"    Partial products 1 finished in {time.time() - half1_start_time:.3f}s.")

    log_callback(f"    Calculating partial products 2 ({num_comb2:,} combinations)...")
    st.write(f"Scan l0={current_l0:.1f}: Partial products 2...") # UI Feedback
    half2_start_time = time.time()
    # Generate combinations for the second half
    indices2 = jnp.arange(num_comb2)
    powers2 = 2**jnp.arange(N2)
    multiplier_indices2 = jnp.not_equal(indices2[:, None] & powers2, 0).astype(jnp.int32) # Shape (num_comb2, N2)
    matrices_half2 = all_layer_matrices[N1:] # Shape (N2, 2, L_sparse, 2, 2)
    # partial_products2 shape: (num_comb2, L_sparse, 2, 2)
    partial_products2 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices2, matrices_half2)
    partial_products2.block_until_ready()
    log_callback(f"    Partial products 2 finished in {time.time() - half2_start_time:.3f}s.")

    log_callback(f"    Combining and calculating MSE ({num_comb1 * num_comb2:,} total)...")
    st.write(f"Scan l0={current_l0:.1f}: Combining & MSE...") # UI Feedback
    combine_start_time = time.time()
    combine_and_calc_mse_jit = jax.jit(combine_and_calc_mse)

    # Combine products and calculate MSE using nested vmap for efficiency
    # Outer vmap iterates through prod1 (from first half combinations)
    # Inner vmap iterates through prod2 (from second half combinations) for each prod1
    vmap_inner = vmap(combine_and_calc_mse_jit, in_axes=(None, 0, None, None, None)) # vmap over prod2
    vmap_outer = vmap(vmap_inner, in_axes=(0, None, None, None, None)) # vmap over prod1

    # all_mses_nested shape: (num_comb1, num_comb2)
    all_mses_nested = vmap_outer(partial_products1, partial_products2, nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple)
    all_mses_nested.block_until_ready()
    log_callback(f"    Combination and MSE finished in {time.time() - combine_start_time:.3f}s.")

    # Find the best combination
    all_mses_flat = all_mses_nested.reshape(-1) # Flatten the MSE array
    best_idx_flat = jnp.argmin(all_mses_flat) # Find index of minimum MSE
    current_best_mse = float(all_mses_flat[best_idx_flat]) # Get the minimum MSE value

    if not np.isfinite(current_best_mse):
        logs.append(f"    Warning: No valid result (finite MSE) found for l0={current_l0:.2f}.")
        return np.inf, None, logs

    # Find the indices corresponding to the best combination in the original nested structure
    best_idx_half1, best_idx_half2 = jnp.unravel_index(best_idx_flat, (num_comb1, num_comb2))

    # Get the multiplier indices (0s and 1s) for the best combination
    best_indices_h1 = multiplier_indices1[best_idx_half1] # Shape (N1,)
    best_indices_h2 = multiplier_indices2[best_idx_half2] # Shape (N2,)

    # Convert indices (0, 1) to multipliers (1.0, 2.0)
    best_multipliers_h1 = 1.0 + best_indices_h1.astype(jnp.float64)
    best_multipliers_h2 = 1.0 + best_indices_h2.astype(jnp.float64)

    # Concatenate multipliers for the full stack
    current_best_multipliers = jnp.concatenate([best_multipliers_h1, best_multipliers_h2]) # Shape (N,)
    logs.append(f"    Best MSE for scan l0={current_l0:.2f}: {current_best_mse:.6e}")
    return current_best_mse, np.array(current_best_multipliers), logs # Return best MSE and corresponding multipliers

def get_material_input(role: str) -> Tuple[Optional[MaterialInputType], str]:
    # Helper to get material definition based on UI selection
    if role == 'H':
        sel_key, const_r_key, const_i_key = "selected_H", "nH_r", "nH_i"
    elif role == 'L':
        sel_key, const_r_key, const_i_key = "selected_L", "nL_r", "nL_i"
    elif role == 'Sub':
        sel_key, const_r_key, const_i_key = "selected_Sub", "nSub_r", None # Substrate usually has k=0 input simplified
    else:
        st.error(f"Unknown material role: {role}")
        return None, "Role Error"

    selection = st.session_state.get(sel_key)
    if selection == "Constant":
        n_real = st.session_state.get(const_r_key, 1.0 if role != 'Sub' else 1.5)
        n_imag = 0.0
        if const_i_key and role in ['H', 'L']: # Only H, L have explicit k input here
           n_imag = st.session_state.get(const_i_key, 0.0)

        # Validate and potentially adjust constant values
        valid_n = True
        valid_k = True
        if n_real <= 0:
            n_real = 1.0 # Default to 1 if non-physical
            valid_n = False
        if n_imag < 0:
            n_imag = 0.0 # Default to 0 if non-physical
            valid_k = False

        mat_repr = f"Constant ({n_real:.3f}{'+' if n_imag>=0 else ''}{n_imag:.3f}j)"
        if not valid_n or not valid_k:
            mat_repr += " (Adjusted)"
        return complex(n_real, n_imag), mat_repr
    elif isinstance(selection, str) and selection: # Material selected from Excel/Built-in
        return selection, selection
    else: # Error case
        st.error(f"Material selection for '{role}' invalid or missing in session_state.")
        return None, "Selection Error"

def validate_targets() -> Optional[List[Dict]]:
    # Validates the target definitions entered in the UI
    active_targets = []
    logs = [] # For potential error messages
    is_valid = True
    if 'targets' not in st.session_state or not isinstance(st.session_state.targets, list):
        st.error("Internal error: Target list missing or invalid in session_state.")
        return None

    for i, target_state in enumerate(st.session_state.targets):
        if target_state.get('enabled', False): # Only consider enabled targets
            try:
                l_min = float(target_state['min'])
                l_max = float(target_state['max'])
                t_min = float(target_state['target_min'])
                t_max = float(target_state['target_max'])

                # Add validation checks
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
        # Display errors/warnings to the user
        for log in logs: st.warning(log)
        st.error("Errors exist in the active spectral target definitions. Please correct.")
        return None
    elif not active_targets:
        # st.warning("No active targets defined.") # Optional warning
        return [] # Return empty list if none are active
    else:
        return active_targets

def get_lambda_range_from_targets(validated_targets: Optional[List[Dict]]) -> Tuple[Optional[float], Optional[float]]:
    # Determines the overall min/max lambda from all active targets
    overall_min, overall_max = None, None
    if validated_targets: # Check if list is not None and not empty
        all_mins = [t['min'] for t in validated_targets]
        all_maxs = [t['max'] for t in validated_targets]
        if all_mins: overall_min = min(all_mins)
        if all_maxs: overall_max = max(all_maxs)
    return overall_min, overall_max

def clear_optimized_state():
    # Resets the state related to optimization results
    st.session_state.optimized_ep = None
    st.session_state.is_optimized_state = False
    st.session_state.ep_history = deque(maxlen=5) # Keep history for undo
    st.session_state.optimized_qwot_str = ""
    st.session_state.last_mse = None
    # Do NOT clear current_qwot here, as that's the nominal definition

def set_optimized_as_nominal_wrapper():
    # Callback for the "Opt->Nom" button
    if not st.session_state.get('is_optimized_state') or st.session_state.get('optimized_ep') is None:
        st.error("No valid optimized structure to set as nominal.")
        return

    try:
        l0 = st.session_state.l0 # Use the current l0
        nH_mat, _ = get_material_input('H')
        nL_mat, _ = get_material_input('L')
        if nH_mat is None or nL_mat is None:
            st.error("Cannot retrieve H/L materials to recalculate QWOT.")
            return

        # Recalculate QWOT based on the optimized thicknesses and current l0/materials
        optimized_qwots, logs_qwot = calculate_qwot_from_ep(st.session_state.optimized_ep, l0, nH_mat, nL_mat, EXCEL_FILE_PATH)
        add_log(logs_qwot) # Log potential issues

        if optimized_qwots is None:
            st.error("Error recalculating QWOT from the optimized structure.")
            return

        if np.any(np.isnan(optimized_qwots)):
            st.warning("Recalculated QWOT contains NaNs (likely invalid index at l0). Nominal QWOT not updated.")
        else:
            # Update the nominal QWOT string
            new_qwot_str = ",".join([f"{q:.6f}" for q in optimized_qwots]) # Use more precision
            st.session_state.current_qwot = new_qwot_str
            st.success("Optimized structure set as new Nominal (QWOT updated).")
            # Clear the optimized state as it's now the nominal one
            clear_optimized_state()
            # Trigger a recalculation of the (now new) nominal structure
            st.session_state.needs_rerun_calc = True
            st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (Set from Opt)"}
    except Exception as e:
        st.error(f"Unexpected error setting optimized as nominal: {e}")

def undo_remove_wrapper():
    # Callback for the "Undo" button
    if not st.session_state.get('ep_history'):
        st.info("Undo history is empty.")
        return

    try:
        # Restore the previous thickness vector from history
        last_ep = st.session_state.ep_history.pop()
        st.session_state.optimized_ep = last_ep.copy()
        st.session_state.is_optimized_state = True # Mark as optimized state (even if it was intermediate)
        st.session_state.current_ep = last_ep.copy() # Update current_ep for display consistency

        # Recalculate the corresponding QWOT string for display
        l0 = st.session_state.l0
        nH_mat, _ = get_material_input('H')
        nL_mat, _ = get_material_input('L')
        if nH_mat is not None and nL_mat is not None:
            qwots_recalc, logs_qwot = calculate_qwot_from_ep(last_ep, l0, nH_mat, nL_mat, EXCEL_FILE_PATH)
            add_log(logs_qwot)
            if qwots_recalc is not None and not np.any(np.isnan(qwots_recalc)):
                st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_recalc])
            else:
                st.session_state.optimized_qwot_str = "QWOT N/A (after undo)"
        else:
            st.session_state.optimized_qwot_str = "QWOT Material Error (after undo)"

        st.info("State restored. Recalculating...")
        # Trigger recalculation of the restored state
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {
            'is_optimized_run': True, # Treat restored state as 'optimized' for display
            'method_name': "Optimized (Undo)",
            'force_ep': st.session_state.optimized_ep # Use the restored ep
            }
    except IndexError:
        st.warning("Undo history is empty (internal error?).")
    except Exception as e:
        st.error(f"Unexpected error during undo: {e}")
        clear_optimized_state() # Clear state on error

def run_calculation_wrapper(is_optimized_run: bool, method_name: str = "", force_ep: Optional[np.ndarray] = None):
    # Central function to calculate and display spectral results
    calc_type = 'Optimized' if is_optimized_run else 'Nominal'
    st.session_state.last_calc_results = {} # Clear previous results
    st.session_state.last_mse = None

    with st.spinner(f"{calc_type} calculation in progress..."):
        try:
            active_targets = validate_targets()
            if active_targets is None: # Validation failed
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

            # Gather all necessary inputs
            validated_inputs = {
                'l0': st.session_state.l0,
                'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot, # Use nominal QWOT definition
                'auto_thin_threshold': st.session_state.auto_thin_threshold,
                'l_range_deb': l_min_plot, # Use range from targets for consistency
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

            # Determine the thickness vector to use for calculation
            ep_to_calculate = None
            if force_ep is not None: # If specific thicknesses are provided (e.g., after undo)
                ep_to_calculate = force_ep.copy()
            elif is_optimized_run and st.session_state.get('optimized_ep') is not None: # Use stored optimized result
                ep_to_calculate = st.session_state.optimized_ep.copy()
            else: # Calculate nominal thicknesses from QWOT string
                emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
                if not emp_list and calc_type == 'Nominal': # Handle empty nominal QWOT
                    ep_to_calculate = np.array([], dtype=np.float64)
                    add_log("Calculating for empty nominal structure (0 layers).")
                elif not emp_list and calc_type == 'Optimized': # Should not happen if logic is correct
                    st.error("Cannot calculate 'Optimized' state if nominal QWOT is empty and no previous optimized state exists.")
                    return
                else: # Calculate from non-empty QWOT
                    ep_calc, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                    add_log(logs_ep_init)
                    if ep_calc is None:
                        st.error("Failed to calculate initial thicknesses from QWOT.")
                        return
                    ep_to_calculate = ep_calc.copy()

            # Store the thicknesses being used for this calculation in session state (for display)
            st.session_state.current_ep = ep_to_calculate.copy() if ep_to_calculate is not None else None

            # Generate lambda vector for fine plotting
            num_plot_points = max(501, int(np.round((l_max_plot - l_min_plot) / validated_inputs['l_step'])) * 3 + 1)
            l_vec_plot_fine_np = np.linspace(l_min_plot, l_max_plot, num_plot_points)
            l_vec_plot_fine_np = l_vec_plot_fine_np[(l_vec_plot_fine_np > 0) & np.isfinite(l_vec_plot_fine_np)]
            if not l_vec_plot_fine_np.size:
                st.error("Could not generate a valid lambda vector for plotting.")
                return

            # Perform the main transmittance calculation
            start_calc_time = time.time()
            results_fine, calc_logs = calculate_T_from_ep_jax(
                ep_to_calculate, nH_mat, nL_mat, nSub_mat, l_vec_plot_fine_np, EXCEL_FILE_PATH
            )
            add_log(calc_logs)
            add_log(f"Transmittance calculation ({len(l_vec_plot_fine_np)} pts) took {time.time()-start_calc_time:.3f}s")
            if results_fine is None:
                st.error("Main transmittance calculation failed.")
                return

            # Store results for plotting
            st.session_state.last_calc_results = {
                'res_fine': results_fine,
                'method_name': method_name,
                'ep_used': ep_to_calculate.copy() if ep_to_calculate is not None else None,
                'l0_used': validated_inputs['l0'],
                'nH_used': nH_mat, 'nL_used': nL_mat, 'nSub_used': nSub_mat,
            }

            # Calculate MSE for display if targets exist
            if active_targets:
                # Use the same lambda grid as optimization for fair comparison
                num_pts_optim = max(2, int(np.round((l_max_plot - l_min_plot) / validated_inputs['l_step'])) + 1)
                l_vec_optim_np = np.geomspace(l_min_plot, l_max_plot, num_pts_optim)
                l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
                if l_vec_optim_np.size > 0:
                    # Calculate T on the optimization grid
                    results_optim_grid, logs_mse_calc = calculate_T_from_ep_jax(
                        ep_to_calculate, nH_mat, nL_mat, nSub_mat, l_vec_optim_np, EXCEL_FILE_PATH
                    )
                    add_log(logs_mse_calc)
                    if results_optim_grid is not None:
                        # Calculate final MSE using the results on the optimization grid
                        mse_display, num_pts_mse = calculate_final_mse(results_optim_grid, active_targets)
                        st.session_state.last_mse = mse_display
                        # Store the grid results too if needed later
                        st.session_state.last_calc_results['res_optim_grid'] = results_optim_grid
                    else:
                        st.session_state.last_mse = None # Calculation failed
                else:
                    st.session_state.last_mse = None # Lambda grid invalid
            else:
                st.session_state.last_mse = None # No targets to calculate MSE

            # Update optimization state flag
            st.session_state.is_optimized_state = is_optimized_run
            if not is_optimized_run: # If it was a nominal calculation, clear any old optimized state
                clear_optimized_state()
                # Ensure current_ep reflects the nominal calculation just performed
                st.session_state.current_ep = ep_to_calculate.copy() if ep_to_calculate is not None else None

            st.success(f"{calc_type} calculation finished.")

        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during {calc_type} calculation: {e}")
            traceback.print_exc() # Log traceback for debugging
        except Exception as e_fatal:
            st.error(f"Unexpected error during {calc_type} calculation: {e_fatal}")
            traceback.print_exc() # Log traceback for debugging
        finally:
            pass # Cleanup if needed

def run_local_optimization_wrapper():
    # Callback for the "Opt Local" button
    st.session_state.last_calc_results = {} # Clear previous results
    st.session_state.last_mse = None
    clear_optimized_state() # Start fresh optimization state

    with st.spinner("Local optimization in progress..."):
        try:
            active_targets = validate_targets()
            if active_targets is None or not active_targets:
                st.error("Local optimization requires active and valid targets.")
                return

            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None: # Should be caught by active_targets check, but safeguard
                st.error("Could not determine lambda range for optimization.")
                return

            # Gather inputs
            validated_inputs = {
                'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot, # Start from current nominal
                'auto_thin_threshold': st.session_state.auto_thin_threshold, # Needed by core opt function args? No.
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

            # Calculate starting thicknesses from nominal QWOT
            emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
            if not emp_list:
                st.error("Nominal QWOT empty, cannot start local optimization.")
                return

            ep_start, logs_ep_init = calculate_initial_ep(emp_list, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
            add_log(logs_ep_init)
            if ep_start is None:
                st.error("Failed initial thickness calculation for local optimization.")
                return

            # Run the core optimization
            final_ep, success, final_cost, optim_logs, msg = \
                _run_core_optimization(ep_start, validated_inputs, active_targets,
                                       MIN_THICKNESS_PHYS_NM, log_prefix="  [Opt Local] ")
            add_log(optim_logs)

            if success and final_ep is not None:
                # Store results in session state
                st.session_state.optimized_ep = final_ep.copy()
                st.session_state.current_ep = final_ep.copy() # Update current_ep to optimized result
                st.session_state.is_optimized_state = True
                st.session_state.last_mse = final_cost
                # Calculate QWOT string for display
                qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                add_log(logs_qwot)
                if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                    st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                else:
                    st.session_state.optimized_qwot_str = "QWOT N/A"
                st.success(f"Local optimization finished ({msg}). MSE: {final_cost:.4e}")
                # Trigger recalculation to display the optimized result
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Opt Local"}
            else: # Optimization failed
                st.error(f"Local optimization failed: {msg}")
                st.session_state.is_optimized_state = False # Ensure state is not marked as optimized
                st.session_state.optimized_ep = None
                st.session_state.current_ep = ep_start.copy() # Revert current_ep to the starting nominal
                st.session_state.last_mse = None # No valid MSE
                # Optionally trigger a recalc of the nominal state if desired after failure
                # st.session_state.needs_rerun_calc = True
                # st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (After Opt Fail)"}


        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during local optimization: {e}")
            clear_optimized_state() # Clear state on error
        except Exception as e_fatal:
            st.error(f"Unexpected error during local optimization: {e_fatal}")
            clear_optimized_state() # Clear state on error

def run_scan_optimization_wrapper():
    # Callback for "Scan+Opt" button
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None
    clear_optimized_state() # Start fresh

    with st.spinner("QWOT Scan + Double Optimization in progress (can be very long)..."):
        try:
            # ***MODIFICATION START: Explicitly get layer number from nominal QWOT***
            nominal_qwot_str = st.session_state.current_qwot
            initial_emp_list = [q for q in nominal_qwot_str.split(',') if q.strip()]
            initial_layer_number_scan = len(initial_emp_list)
            if initial_layer_number_scan == 0:
                st.error("Nominal QWOT is empty. Cannot perform Scan+Opt.")
                return
            # Store this fixed number for the scan process
            st.session_state.initial_layer_number_scan = initial_layer_number_scan
            add_log(f"Scan+Opt: Starting with {initial_layer_number_scan} layers based on nominal QWOT.")
            # ***MODIFICATION END***

            active_targets = validate_targets()
            if active_targets is None or not active_targets:
                st.error("QWOT Scan+Opt requires active and valid targets.")
                return

            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None:
                st.error("Could not determine lambda range for QWOT Scan+Opt.")
                return

            # Gather inputs
            validated_inputs = {
                'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot, # Keep nominal QWOT for reference if needed
                # Use the explicitly determined layer number for the scan function
                'initial_layer_number': initial_layer_number_scan,
                'auto_thin_threshold': st.session_state.auto_thin_threshold, # Not directly used but part of standard inputs
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

            # Prepare lambda vectors and substrate indices for scan evaluation
            l_vec_eval_full_np = np.geomspace(l_min_opt, l_max_opt, max(2, int(np.round((l_max_opt - l_min_opt) / validated_inputs['l_step'])) + 1))
            l_vec_eval_full_np = l_vec_eval_full_np[(l_vec_eval_full_np > 0) & np.isfinite(l_vec_eval_full_np)]
            if not l_vec_eval_full_np.size: raise ValueError("Failed lambda generation for Scan.")

            l_vec_eval_sparse_np = l_vec_eval_full_np[::2] # Use a sparser grid for the initial scan phase
            if not l_vec_eval_sparse_np.size: raise ValueError("Failed sparse lambda generation for Scan.")
            l_vec_eval_sparse_jax = jnp.asarray(l_vec_eval_sparse_np)

            nSub_arr_scan, logs_sub_scan = _get_nk_array_for_lambda_vec(nSub_mat, l_vec_eval_sparse_jax, EXCEL_FILE_PATH)
            add_log(logs_sub_scan)
            if nSub_arr_scan is None: raise RuntimeError("Failed substrate index preparation for scan.")

            active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)

            # --- Phase 1: Scan different l0 values ---
            l0_nominal = validated_inputs['l0']
            l0_values_to_test = sorted(list(set([l0_nominal, l0_nominal * 1.2, l0_nominal * 0.8]))) # Test nominal and +/- 20%
            l0_values_to_test = [l for l in l0_values_to_test if l > 1e-6] # Filter out invalid l0

            scan_candidates = []
            overall_scan_logs = []
            st.write(f"Phase 1: Scanning QWOT combinations for l0 = {l0_values_to_test}...")
            for l0_scan in l0_values_to_test:
                st.write(f"Scanning for l0={l0_scan:.1f}...")
                try:
                    # Get indices at the specific l0 being scanned
                    nH_c_l0, log_h_l0 = _get_nk_at_lambda(nH_mat, l0_scan, EXCEL_FILE_PATH)
                    nL_c_l0, log_l_l0 = _get_nk_at_lambda(nL_mat, l0_scan, EXCEL_FILE_PATH)
                    overall_scan_logs.extend(log_h_l0); overall_scan_logs.extend(log_l_l0)
                    if nH_c_l0 is None or nL_c_l0 is None:
                        st.warning(f"Could not get indices at l0={l0_scan:.1f}, skipping this value.")
                        continue

                    # Execute the scan for this l0
                    scan_mse, scan_multipliers, scan_logs_l0 = _execute_split_stack_scan(
                        l0_scan, initial_layer_number_scan, # Use the fixed layer number
                        nH_c_l0, nL_c_l0,
                        nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple,
                        add_log # Use the logging callback
                    )
                    overall_scan_logs.extend(scan_logs_l0)
                    if scan_multipliers is not None and np.isfinite(scan_mse):
                        # Store the result as a candidate for optimization
                        scan_candidates.append({
                            'l0': l0_scan,
                            'mse_scan': scan_mse,
                            'multipliers': scan_multipliers # QWOT multipliers (1.0 or 2.0)
                        })
                        st.write(f"-> Best scan result for l0={l0_scan:.1f}: MSE = {scan_mse:.4e}")
                    else:
                        st.warning(f"Scan for l0={l0_scan:.1f} did not yield a valid result.")
                except Exception as e_scan_l0:
                    st.warning(f"Error during scan for l0={l0_scan:.2f}: {e_scan_l0}")
            add_log(overall_scan_logs) # Add all scan logs

            if not scan_candidates:
                st.error("QWOT Scan found no valid initial candidates after scanning all l0 values.")
                return

            # --- Phase 2: Double Optimization for each candidate ---
            st.write(f"\nPhase 2: Double Optimization for {len(scan_candidates)} candidate(s)...")
            optimization_results = []
            for idx, candidate in enumerate(scan_candidates):
                l0_cand = candidate['l0']
                multipliers_cand = candidate['multipliers'] # QWOT multipliers from scan
                st.write(f"-- Optimizing Candidate {idx+1}/{len(scan_candidates)} (from l0={l0_cand:.1f}, scan MSE={candidate['mse_scan']:.4e}) --")

                # Calculate initial physical thicknesses for this candidate
                ep_start_optim, logs_ep_best = calculate_initial_ep(multipliers_cand, l0_cand, nH_mat, nL_mat, EXCEL_FILE_PATH)
                add_log(logs_ep_best)
                if ep_start_optim is None:
                    st.warning(f"Failed thickness calculation for candidate {idx+1}. Skipping.")
                    continue

                # First Optimization
                st.write(f"    Running Optimization 1/2...")
                final_ep_1, success_1, final_cost_1, optim_logs_1, msg_1 = \
                    _run_core_optimization(ep_start_optim, validated_inputs, active_targets,
                                           MIN_THICKNESS_PHYS_NM, log_prefix=f"  [OptScan {idx+1}-1] ")
                add_log(optim_logs_1)

                if success_1 and final_ep_1 is not None:
                    st.write(f"    Optimization 1/2 finished ({msg_1}). MSE: {final_cost_1:.4e}. Running Optimization 2/2...")
                    # Second Optimization (start from the result of the first)
                    final_ep_2, success_2, final_cost_2, optim_logs_2, msg_2 = \
                        _run_core_optimization(final_ep_1, validated_inputs, active_targets,
                                               MIN_THICKNESS_PHYS_NM, log_prefix=f"  [OptScan {idx+1}-2] ")
                    add_log(optim_logs_2)

                    if success_2 and final_ep_2 is not None:
                        st.write(f"    Optimization 2/2 finished ({msg_2}). Final MSE: {final_cost_2:.4e}")
                        # Store the successful double-optimized result
                        optimization_results.append({
                            'l0_origin': l0_cand, # The l0 that generated this candidate
                            'final_ep': final_ep_2,
                            'final_mse': final_cost_2,
                            'message': f"Opt1: {msg_1} | Opt2: {msg_2}"
                        })
                    else:
                        st.warning(f"    Optimization 2/2 FAILED for candidate {idx+1}: {msg_2}. Discarding this candidate.")
                else:
                    st.warning(f"    Optimization 1/2 FAILED for candidate {idx+1}: {msg_1}. Skipping second optimization.")

            # --- Phase 3: Final Selection ---
            if not optimization_results:
                st.error("Scan + Opt: No candidates successfully completed the double optimization.")
                clear_optimized_state() # Clear any potential intermediate state
                return

            st.write("\nPhase 3: Selecting Best Overall Result...")
            # Sort results by final MSE
            optimization_results.sort(key=lambda r: r['final_mse'])
            best_overall_result = optimization_results[0]

            # Extract best result details
            final_ep = best_overall_result['final_ep']
            final_cost = best_overall_result['final_mse']
            final_l0 = best_overall_result['l0_origin'] # Use the l0 that led to the best result
            final_msg = best_overall_result['message']

            # Update session state with the best result
            st.session_state.optimized_ep = final_ep.copy()
            st.session_state.current_ep = final_ep.copy()
            st.session_state.is_optimized_state = True
            st.session_state.last_mse = final_cost
            st.session_state.l0 = final_l0 # ***MODIFICATION: Update main l0 to the best one found***

            # Calculate QWOT string for the optimized result (for display)
            qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, final_l0, nH_mat, nL_mat, EXCEL_FILE_PATH)
            add_log(logs_qwot)

            # ***MODIFICATION START: Update nominal QWOT if possible***
            if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                new_qwot_str = ", ".join([f"{q:.6f}" for q in qwots_opt]) # More precision
                st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt]) # Display string
                # Check if the number of layers matches the initial scan number
                if len(qwots_opt) == initial_layer_number_scan:
                    st.session_state.current_qwot = new_qwot_str # Update nominal QWOT
                    add_log("Scan+Opt: Updated nominal QWOT string with the optimized result.")
                else:
                    # This case should ideally not happen if optimization doesn't change layer count
                    st.warning(f"Scan+Opt: Optimized layer count ({len(qwots_opt)}) differs from initial ({initial_layer_number_scan}). Nominal QWOT not updated.")
                    st.session_state.optimized_qwot_str = "QWOT (Layer# Mismatch)"
            else:
                st.session_state.optimized_qwot_str = "QWOT N/A"
                st.warning("Scan+Opt: Could not calculate valid QWOT for the optimized result. Nominal QWOT not updated.")
            # ***MODIFICATION END***

            st.success(f"Scan + Double Optimization finished. Best result from l0={final_l0:.1f}. Final MSE: {final_cost:.4e}")
            st.caption(f"Optimization details: {final_msg}")
            # Trigger recalculation to display the final result
            st.session_state.needs_rerun_calc = True
            st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Scan+Opt*2 (l0={final_l0:.1f})"}

        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during QWOT Scan + Double Optimization: {e}")
            clear_optimized_state()
        except Exception as e_fatal:
            st.error(f"Unexpected error during QWOT Scan + Double Optimization: {e_fatal}")
            clear_optimized_state()
        finally:
            # Clean up temporary state if any was added
            if 'initial_layer_number_scan' in st.session_state:
                del st.session_state['initial_layer_number_scan']


def run_auto_mode_wrapper():
    # Callback for the "Auto" button
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None
    # Auto mode can start from optimized OR nominal, so don't clear optimized state here

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

            # Gather inputs
            validated_inputs = {
                'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot, # Pass nominal QWOT
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

            # Determine starting point for Auto mode
            ep_start_auto = None
            if st.session_state.get('is_optimized_state') and st.session_state.get('optimized_ep') is not None:
                ep_start_auto = st.session_state.optimized_ep.copy()
                add_log("Auto Mode: Starting from current optimized state.")
            else:
                add_log("Auto Mode: Starting from current nominal state (will perform initial optimization).")

            # Run the main Auto Mode logic
            final_ep, final_mse, auto_logs = run_auto_mode(
                initial_ep=ep_start_auto, # Pass starting structure (or None)
                validated_inputs=validated_inputs,
                active_targets=active_targets,
                excel_file_path=EXCEL_FILE_PATH,
                log_callback=add_log
            )
            add_log(auto_logs) # Add logs from the auto process

            if final_ep is not None and np.isfinite(final_mse):
                # Update session state with the final result from Auto Mode
                st.session_state.optimized_ep = final_ep.copy()
                st.session_state.current_ep = final_ep.copy()
                st.session_state.is_optimized_state = True
                st.session_state.last_mse = final_mse
                # Calculate QWOT string for display
                qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                add_log(logs_qwot)
                if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                    st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                else: st.session_state.optimized_qwot_str = "QWOT N/A"
                st.success(f"Auto Mode finished. Final MSE: {final_mse:.4e}")
                # Trigger recalculation to display the final result
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Auto Mode"}
            else: # Auto mode failed
                st.error("Automatic Mode failed or did not produce a valid result.")
                # Decide what state to revert to. Maybe the state before Auto was run?
                # For now, clear optimized state and trigger nominal recalc.
                clear_optimized_state()
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (After Auto Fail)"}

        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during Auto Mode: {e}")
        except Exception as e_fatal:
            st.error(f"Unexpected error during Auto Mode: {e_fatal}")
        finally:
            pass # Cleanup if needed

def run_remove_thin_wrapper():
    # Callback for "Thin+ReOpt" button
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None

    # Determine the starting structure: optimized if available, otherwise nominal
    ep_start_removal = None
    is_starting_from_optimized = False
    if st.session_state.get('is_optimized_state') and st.session_state.get('optimized_ep') is not None:
        ep_start_removal = st.session_state.optimized_ep.copy()
        is_starting_from_optimized = True
        add_log("Remove Thin: Starting from existing optimized structure.")
    else:
        add_log("Remove Thin: Starting from nominal structure.")
        # Need to calculate nominal ep first
        try:
            nH_mat_temp, _ = get_material_input('H')
            nL_mat_temp, _ = get_material_input('L')
            if nH_mat_temp is None or nL_mat_temp is None:
                st.error("Cannot calculate nominal structure: Material definition error.")
                return
            emp_list_temp = [float(e.strip()) for e in st.session_state.current_qwot.split(',') if e.strip()]
            if not emp_list_temp:
                 ep_start_removal = np.array([], dtype=np.float64) # Handle empty QWOT case
                 add_log("Remove Thin: Nominal QWOT is empty.")
            else:
                ep_start_removal, logs_ep_init = calculate_initial_ep(
                    emp_list_temp, st.session_state.l0, nH_mat_temp, nL_mat_temp, EXCEL_FILE_PATH
                )
                add_log(logs_ep_init)
                if ep_start_removal is None:
                    st.error("Failed to calculate nominal structure from QWOT for removal.")
                    return
            st.session_state.current_ep = ep_start_removal.copy() # Update current_ep if starting from nominal
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
            # Store the state *before* removal in history
            st.session_state.ep_history.append(ep_start_removal.copy())
            add_log(f"Remove Thin: Saved state with {len(ep_start_removal)} layers to undo history.")

            active_targets = validate_targets()
            if active_targets is None or not active_targets:
                st.session_state.ep_history.pop() # Remove saved state if aborted
                add_log("Remove Thin: Aborted, popped saved state from history.")
                st.error("Removal aborted: invalid or missing targets for re-optimization.")
                return

            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None:
                st.session_state.ep_history.pop()
                add_log("Remove Thin: Aborted, popped saved state from history.")
                st.error("Removal aborted: invalid lambda range for re-optimization.")
                return

            # Gather inputs for re-optimization
            validated_inputs = {
                'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot, # Pass current nominal QWOT for context if needed
                'auto_thin_threshold': st.session_state.auto_thin_threshold, # Pass threshold
                'l_range_deb': l_min_opt, 'l_range_fin': l_max_opt,
            }
            nH_mat, nH_repr = get_material_input('H')
            nL_mat, nL_repr = get_material_input('L')
            nSub_mat, nSub_repr = get_material_input('Sub')
            if nH_mat is None or nL_mat is None or nSub_mat is None:
                st.session_state.ep_history.pop()
                add_log("Remove Thin: Aborted, popped saved state from history.")
                st.error("Removal aborted: material definition error.")
                return
            validated_inputs['nH_material'] = nH_mat
            validated_inputs['nL_material'] = nL_mat
            validated_inputs['nSub_material'] = nSub_mat

            # Perform the removal/merge operation
            # Use threshold=None to find the absolute thinnest layer >= min_thickness_phys
            ep_after_removal, structure_changed, removal_logs = _perform_layer_merge_or_removal_only(
                ep_start_removal, MIN_THICKNESS_PHYS_NM,
                log_prefix="  [Remove] ",
                threshold_for_removal=None # Find absolute thinnest valid layer
            )
            add_log(removal_logs)

            if structure_changed and ep_after_removal is not None:
                st.write("Re-optimizing after removal...")
                # Re-optimize the structure after removal
                final_ep, success, final_cost, optim_logs, msg = \
                    _run_core_optimization(ep_after_removal, validated_inputs, active_targets,
                                           MIN_THICKNESS_PHYS_NM, log_prefix="  [ReOpt Thin] ")
                add_log(optim_logs)

                if success and final_ep is not None:
                    # Update state with successfully re-optimized structure
                    st.session_state.optimized_ep = final_ep.copy()
                    st.session_state.current_ep = final_ep.copy() # Also update current_ep
                    st.session_state.is_optimized_state = True # Mark state as optimized
                    st.session_state.last_mse = final_cost
                    # Calculate QWOT string for display
                    qwots_opt, logs_qwot = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                    add_log(logs_qwot)
                    if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                        st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                    else: st.session_state.optimized_qwot_str = "QWOT N/A"
                    st.success(f"Removal + Re-optimization finished ({msg}). Final MSE: {final_cost:.4e}")
                    # Trigger recalculation to display the new optimized state
                    st.session_state.needs_rerun_calc = True
                    st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Optimized (Post-Remove)"}
                else: # Re-optimization failed
                    st.warning(f"Layer removed, but re-optimization failed ({msg}). State is AFTER removal but BEFORE failed re-opt attempt.")
                    # Keep the state after removal, but before the failed re-opt
                    st.session_state.optimized_ep = ep_after_removal.copy()
                    st.session_state.current_ep = ep_after_removal.copy()
                    st.session_state.is_optimized_state = True # State is 'optimized' post-removal, even if re-opt failed
                    # Try to calculate MSE and QWOT for this intermediate state
                    try:
                        l_min_opt, l_max_opt = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
                        l_step_optim = validated_inputs['l_step']
                        num_pts_optim = max(2, int(np.round((l_max_opt - l_min_opt) / l_step_optim)) + 1)
                        l_vec_optim_np = np.geomspace(l_min_opt, l_max_opt, num_pts_optim)
                        l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
                        if l_vec_optim_np.size > 0:
                            results_fail_grid, _ = calculate_T_from_ep_jax(ep_after_removal, nH_mat, nL_mat, nSub_mat, l_vec_optim_np, EXCEL_FILE_PATH)
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
                    # Trigger recalculation to display this intermediate state
                    st.session_state.needs_rerun_calc = True
                    st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': "Optimized (Post-Remove, Re-Opt Fail)"}
            else: # No layer was removed
                st.info("No layer was removed (criteria not met or structure too small).")
                try:
                    st.session_state.ep_history.pop() # Remove the state saved if no change occurred
                    add_log("Remove Thin: No change, popped saved state from history.")
                except IndexError: pass # History might be empty if attempted on invalid state

        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during Thin Layer Removal: {e}")
            try:
                st.session_state.ep_history.pop() # Attempt to pop history on error
                add_log("Remove Thin: Error occurred, popped saved state from history.")
            except IndexError: pass
        except Exception as e_fatal:
            st.error(f"Unexpected error during Thin Layer Removal: {e_fatal}")
            try:
                st.session_state.ep_history.pop()
                add_log("Remove Thin: Fatal error occurred, popped saved state from history.")
            except IndexError: pass
        finally:
            pass # Cleanup if needed

# --- UI Layout ---
st.set_page_config(layout="wide", page_title="Thin Film Optimizer (Streamlit)")

# --- Initialization ---
if 'init_done' not in st.session_state:
    st.session_state.log_messages = ["[Initialization] Welcome to the Streamlit optimizer."]
    st.session_state.current_ep = None # Thicknesses of the currently displayed structure (nominal or optimized)
    st.session_state.current_qwot = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1" # Nominal QWOT definition
    st.session_state.optimized_ep = None # Thicknesses of the last successful optimization
    st.session_state.is_optimized_state = False # Flag indicating if the current display shows an optimized result
    st.session_state.optimized_qwot_str = "" # QWOT string corresponding to optimized_ep (for display)
    st.session_state.material_sequence = None # For arbitrary sequence mode (not used here)
    st.session_state.ep_history = deque(maxlen=5) # History for undoing removals
    st.session_state.last_mse = None # Last calculated MSE for display
    st.session_state.needs_rerun_calc = False # Flag to trigger recalculation
    st.session_state.rerun_calc_params = {} # Parameters for the triggered recalc
    st.session_state.calculating = False # Lock to prevent concurrent calculations

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

    # Default UI values
    st.session_state.l0 = 500.0
    st.session_state.l_step = 10.0
    st.session_state.auto_thin_threshold = 1.0

    # Default material selections (try specific names first, fallback to Constant)
    st.session_state.selected_H = next((m for m in ["Nb2O5-Helios", "Constant"] if m in st.session_state.available_materials), "Constant")
    st.session_state.selected_L = next((m for m in ["SiO2-Helios", "Constant"] if m in st.session_state.available_materials), "Constant")
    st.session_state.selected_Sub = next((m for m in ["Fused Silica", "Constant"] if m in st.session_state.available_substrates), "Constant")

    # Default targets
    st.session_state.targets = [
        {'enabled': True, 'min': 400.0, 'max': 500.0, 'target_min': 1.0, 'target_max': 1.0},
        {'enabled': True, 'min': 500.0, 'max': 600.0, 'target_min': 1.0, 'target_max': 0.2},
        {'enabled': True, 'min': 600.0, 'max': 700.0, 'target_min': 0.2, 'target_max': 0.2},
        {'enabled': False, 'min': 700.0, 'max': 800.0, 'target_min': 0.0, 'target_max': 0.0},
        {'enabled': False, 'min': 800.0, 'max': 900.0, 'target_min': 0.0, 'target_max': 0.0},
    ]

    # Default constant refractive indices
    st.session_state.nH_r = 2.35
    st.session_state.nH_i = 0.0
    st.session_state.nL_r = 1.46
    st.session_state.nL_i = 0.0
    st.session_state.nSub_r = 1.52

    st.session_state.init_done = True
    st.session_state.needs_rerun_calc = True # Trigger initial calculation on load
    st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Initial Load"}

# --- Callbacks ---
def trigger_nominal_recalc():
    # Called when nominal parameters change (materials, l0, targets, etc.)
    if not st.session_state.get('calculating', False):
        print("INFO: trigger_nominal_recalc called")
        # Clear any previous optimization result as nominal parameters changed
        clear_optimized_state()
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {
            'is_optimized_run': False, # Always calculate nominal after param change
            'method_name': "Nominal (Param Update)",
            'force_ep': None # Calculate from current_qwot
        }
        # No rerun here, wait for Streamlit's natural flow after widget change

# --- UI Definition ---
st.title("🔬 Thin Film Optimizer (Streamlit + JAX)")

# --- Top Menu Bar ---
menu_cols = st.columns(8)
with menu_cols[0]:
    if st.button("📊 Eval Nom.", key="eval_nom_top", help="Evaluate Nominal Structure", use_container_width=True):
        clear_optimized_state() # Ensure we evaluate the current nominal definition
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (Evaluated)"}
        st.rerun()
with menu_cols[1]:
    if st.button("✨ Opt Local", key="optim_local_top", help="Local Optimization (starts from Nominal)", use_container_width=True):
        run_local_optimization_wrapper()
        # Rerun might be triggered by needs_rerun_calc inside the wrapper
with menu_cols[2]:
    if st.button("🚀 Scan+Opt", key="optim_scan_top", help="QWOT Scan + Double Optimization (starts from Nominal Layer Count)", use_container_width=True):
        run_scan_optimization_wrapper()
with menu_cols[3]:
    if st.button("🤖 Auto", key="optim_auto_top", help="Auto Mode (Needle > Thin > Opt cycles)", use_container_width=True):
        run_auto_mode_wrapper()
with menu_cols[4]:
    # Enable "Thin+ReOpt" only if a structure exists (nominal or optimized) with > 2 layers
    current_structure_for_check = st.session_state.get('current_ep') # Check the currently displayed structure
    can_remove_structurally = current_structure_for_check is not None and len(current_structure_for_check) > 2
    if st.button("🗑️ Thin+ReOpt", key="remove_thin_top", help="Remove Thinnest Layer + Re-optimize", disabled=not can_remove_structurally, use_container_width=True):
        run_remove_thin_wrapper()
with menu_cols[5]:
    # Enable "Opt->Nom" only if there is a valid optimized state
    can_set_opt_as_nom = st.session_state.get('is_optimized_state', False) and st.session_state.get('optimized_ep') is not None
    if st.button("💾 Opt->Nom", key="set_optim_as_nom_top", help="Set Optimized as Nominal", disabled=not can_set_opt_as_nom, use_container_width=True):
        set_optimized_as_nominal_wrapper()
        st.rerun() # Rerun needed to reflect the change in nominal QWOT input
with menu_cols[6]:
    # Enable "Undo" only if history is not empty
    can_undo_top = bool(st.session_state.get('ep_history'))
    if st.button(f"↩️ Undo ({len(st.session_state.get('ep_history', deque()))})", key="undo_remove_top", help="Undo Last Removal", disabled=not can_undo_top, use_container_width=True):
        undo_remove_wrapper()
        st.rerun() # Rerun needed to display the restored state
with menu_cols[7]:
    if st.button("🔄 Reload Mats", key="reload_mats_top", help="Reload Excel Materials", use_container_width=True):
        st.cache_data.clear() # Clear cache for material loading
        try:
            mats, logs = get_available_materials_from_excel(EXCEL_FILE_PATH)
            add_log(logs)
            st.session_state.available_materials = sorted(list(set(["Constant"] + mats)))
            base_subs = ["Constant", "Fused Silica", "BK7", "D263"]
            st.session_state.available_substrates = sorted(list(set(base_subs + st.session_state.available_materials)))
            # Reset selections if they became invalid after reload
            if st.session_state.selected_H not in st.session_state.available_materials: st.session_state.selected_H = "Constant"
            if st.session_state.selected_L not in st.session_state.available_materials: st.session_state.selected_L = "Constant"
            if st.session_state.selected_Sub not in st.session_state.available_substrates: st.session_state.selected_Sub = "Constant"
            st.success("Materials reloaded.")
            trigger_nominal_recalc() # Trigger recalc with potentially new materials
            st.rerun()
        except Exception as e:
            st.error(f"Error reloading materials: {e}")

st.divider()

# --- Main Content Area ---
main_layout = st.columns([1, 3]) # Configuration on left, Results on right

with main_layout[0]: # Configuration Column
    st.subheader("Materials")
    # Material Selectors
    st.selectbox(
        "H Material", options=st.session_state.available_materials,
        key="selected_H", on_change=trigger_nominal_recalc
    )
    if st.session_state.selected_H == "Constant":
        hc1, hc2 = st.columns(2)
        hc1.number_input("n' H", value=st.session_state.nH_r, format="%.4f", key="nH_r", on_change=trigger_nominal_recalc)
        hc2.number_input("k H", value=st.session_state.nH_i, min_value=0.0, format="%.4f", key="nH_i", on_change=trigger_nominal_recalc)

    st.selectbox(
        "L Material", options=st.session_state.available_materials,
        key="selected_L", on_change=trigger_nominal_recalc
    )
    if st.session_state.selected_L == "Constant":
        lc1, lc2 = st.columns(2)
        lc1.number_input("n' L", value=st.session_state.nL_r, format="%.4f", key="nL_r", on_change=trigger_nominal_recalc)
        lc2.number_input("k L", value=st.session_state.nL_i, min_value=0.0, format="%.4f", key="nL_i", on_change=trigger_nominal_recalc)

    st.selectbox(
        "Substrate", options=st.session_state.available_substrates,
        key="selected_Sub", on_change=trigger_nominal_recalc
    )
    if st.session_state.selected_Sub == "Constant":
        st.number_input("n' Sub", value=st.session_state.nSub_r, format="%.4f", key="nSub_r", on_change=trigger_nominal_recalc)

    st.subheader("Nominal Structure")
    # QWOT Input Area
    st.text_area(
        "QWOT Multipliers (comma-separated)", value=st.session_state.current_qwot, key="qwot_input_area",
        on_change=trigger_nominal_recalc, # Recalculate nominal if QWOT changes
        help="Enter Quarter Wave Optical Thickness multipliers (e.g., 1,1,1 or 1,2,1). Assumes H/L/H... starting from incident medium.",
        height=100
    )
    # Update session state from text_area (needed because on_change doesn't update immediately for text_area)
    st.session_state.current_qwot = st.session_state.qwot_input_area

    # Calculate number of layers from the current QWOT string for display
    num_layers_from_qwot = len([q for q in st.session_state.current_qwot.split(',') if q.strip()])

    # l0 and Layer Generation
    qwot_cols = st.columns([3,2])
    with qwot_cols[0]:
        st.number_input("Reference λ₀ (nm)", value=st.session_state.l0, min_value=1.0, format="%.2f", key="l0", on_change=trigger_nominal_recalc)
    with qwot_cols[1]:
        # Input for generating '1's - value reflects current QWOT length
        init_layers_num = st.number_input("N Layers", min_value=0, value=num_layers_from_qwot, step=1, key="init_layers_gen_num_ui", label_visibility="collapsed", help="Number of layers to generate QWOT=1 for.")
        if st.button("Gen QWOT=1", key="gen_qwot_btn_main", use_container_width=True, help="Generate QWOT string with N layers, all set to 1."):
            if init_layers_num > 0:
                new_qwot = ",".join(['1'] * init_layers_num)
                if new_qwot != st.session_state.current_qwot:
                    st.session_state.current_qwot = new_qwot
                    st.session_state.qwot_input_area = new_qwot # Update text area too
                    trigger_nominal_recalc()
                    st.rerun()
            elif st.session_state.current_qwot != "": # If N=0, clear QWOT
                st.session_state.current_qwot = ""
                st.session_state.qwot_input_area = ""
                trigger_nominal_recalc()
                st.rerun()
    st.caption(f"Nominal Layers: {num_layers_from_qwot}")

    st.subheader("Targets & Parameters")
    param_cols = st.columns(2)
    with param_cols[0]:
      st.number_input("λ Step (nm)", value=st.session_state.l_step, min_value=0.1, format="%.2f", key="l_step", on_change=trigger_nominal_recalc, help="Step size for optimization/MSE grid and fine plot density.")
    with param_cols[1]:
      st.number_input("Thin Thresh.", value=st.session_state.auto_thin_threshold, min_value=MIN_THICKNESS_PHYS_NM, format="%.3f", key="auto_thin_threshold", on_change=trigger_nominal_recalc, help="Threshold for auto thin layer removal (nm) in Auto mode.")


    # Target Definition Table
    hdr_cols = st.columns([0.5, 1, 1, 1, 1])
    hdrs = ["On", "λ Min", "λ Max", "T Min", "T Max"]
    for c, h in zip(hdr_cols, hdrs): c.caption(h)

    for i in range(len(st.session_state.targets)):
        target = st.session_state.targets[i]
        cols = st.columns([0.5, 1, 1, 1, 1])
        # Use unique keys for each widget
        current_enabled = target.get('enabled', False)
        new_enabled = cols[0].checkbox("", value=current_enabled, key=f"target_enable_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['enabled'] = new_enabled
        st.session_state.targets[i]['min'] = cols[1].number_input("λmin", value=target.get('min', 0.0), format="%.1f", step=10.0, key=f"target_min_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['max'] = cols[2].number_input("λmax", value=target.get('max', 0.0), format="%.1f", step=10.0, key=f"target_max_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['target_min'] = cols[3].number_input("Tmin", value=target.get('target_min', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmin_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['target_max'] = cols[4].number_input("Tmax", value=target.get('target_max', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmax_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)


with main_layout[1]: # Results Column
    st.subheader("Results")
    # Display current state information
    state_desc = "Optimized" if st.session_state.is_optimized_state else "Nominal"
    # Use current_ep which reflects the structure used for the last calculation
    ep_display = st.session_state.get('current_ep')
    num_layers_display = len(ep_display) if ep_display is not None else 0

    res_info_cols = st.columns(3)
    with res_info_cols[0]:
         st.metric(label="Current State", value=state_desc)
    with res_info_cols[1]:
        mse_val = st.session_state.get('last_mse')
        mse_str = f"{mse_val:.4e}" if mse_val is not None and np.isfinite(mse_val) else "N/A"
        st.metric(label="MSE", value=mse_str)
    with res_info_cols[2]:
        min_thick_str = "N/A"
        if ep_display is not None and ep_display.size > 0:
            # Find min thickness among layers > 0 (or very close to 0)
            valid_thick = ep_display[ep_display > 1e-9] # Consider layers > 0
            if valid_thick.size > 0:
                min_thick_val = np.min(valid_thick)
                min_thick_str = f"{min_thick_val:.3f} nm"
        st.metric(label="Min Thickness", value=min_thick_str)


    # Display QWOT string if in optimized state
    if st.session_state.is_optimized_state and st.session_state.get('optimized_qwot_str'):
        st.text_input("Optimized QWOT (approx.)", value=st.session_state.optimized_qwot_str, disabled=True, key="opt_qwot_display_main_res", help=f"Calculated from optimized thicknesses using l0={st.session_state.l0:.1f} nm")

    # --- Plotting Area ---
    if 'last_calc_results' in st.session_state and st.session_state.last_calc_results:
        results_data = st.session_state.last_calc_results
        res_fine_plot = results_data.get('res_fine')
        active_targets_plot = validate_targets() # Get current valid targets for plotting
        mse_plot = st.session_state.last_mse
        method_name_plot = results_data.get('method_name', '')
        res_optim_grid_plot = results_data.get('res_optim_grid') # Results on the coarser grid

        if res_fine_plot and active_targets_plot is not None: # Need results and valid targets config
            fig_spec, ax_spec = plt.subplots(figsize=(12, 4)) # Adjusted height

            # Plot Title
            opt_method_str = f" ({method_name_plot})" if method_name_plot else ""
            window_title = f'Spectral Response - {state_desc}{opt_method_str}' # Use state_desc
            # fig_spec.suptitle(window_title, fontsize=12, weight='bold') # Removed suptitle for cleaner look

            line_ts = None
            try:
                # Plot main transmittance curve (fine grid)
                if res_fine_plot and 'l' in res_fine_plot and 'Ts' in res_fine_plot and res_fine_plot['l'] is not None and len(res_fine_plot['l']) > 0:
                    res_l_plot = np.asarray(res_fine_plot['l'])
                    res_ts_plot = np.asarray(res_fine_plot['Ts'])
                    line_ts, = ax_spec.plot(res_l_plot, res_ts_plot, label='Transmittance', linestyle='-', color='blue', linewidth=1.5)

                    # Plot Targets
                    plotted_target_label = False
                    if active_targets_plot: # Check if list is not empty
                        for i, target in enumerate(active_targets_plot):
                            l_min, l_max = target['min'], target['max']
                            t_min, t_max_corr = target['target_min'], target['target_max'] # t_max_corr used for plotting line
                            x_coords, y_coords = [l_min, l_max], [t_min, t_max_corr]
                            label = 'Target(s)' if not plotted_target_label else "_nolegend_" # Only label first target
                            # Plot target line segment
                            line_target, = ax_spec.plot(x_coords, y_coords, 'r--', linewidth=1.0, alpha=0.7, label=label, zorder=5)
                            # Plot markers at target endpoints
                            marker_target = ax_spec.plot(x_coords, y_coords, marker='x', color='red', markersize=6, linestyle='none', label='_nolegend_', zorder=6)
                            plotted_target_label = True

                            # Optionally plot points used for MSE calculation (if available)
                            # This helps visualize the grid used for optimization/MSE
                            if res_optim_grid_plot and 'l' in res_optim_grid_plot and res_optim_grid_plot['l'].size > 0:
                                res_l_optim = np.asarray(res_optim_grid_plot['l'])
                                indices_optim = np.where((res_l_optim >= l_min) & (res_l_optim <= l_max))[0]
                                if indices_optim.size > 0:
                                    optim_lambdas = res_l_optim[indices_optim]
                                    # Calculate target T at these specific grid points
                                    if abs(l_max - l_min) < 1e-9: optim_target_t = np.full_like(optim_lambdas, t_min)
                                    else: slope = (t_max_corr - t_min) / (l_max - l_min); optim_target_t = t_min + slope * (optim_lambdas - l_min)
                                    # Plot small dots for the target points on the optimization grid
                                    # ax_spec.plot(optim_lambdas, optim_target_t, marker='.', color='darkred', linestyle='none', markersize=4, alpha=0.5, label='_nolegend_', zorder=6)

                                                # Plot Formatting
                ax_spec.set_xlabel("Wavelength (nm)")
                ax_spec.set_ylabel('Transmittance')
                ax_spec.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
                ax_spec.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')
                ax_spec.minorticks_on()
                # Ensure res_l_plot exists and is not empty before accessing its elements
                if 'res_l_plot' in locals() and res_l_plot is not None and len(res_l_plot) > 0 :
                    ax_spec.set_xlim(res_l_plot[0], res_l_plot[-1])
                ax_spec.set_ylim(-0.05, 1.05)
                if 'plotted_target_label' in locals() and 'line_ts' in locals() and (plotted_target_label or (line_ts is not None)):
                     ax_spec.legend(fontsize=8)

                # Display MSE on plot
                if 'mse_plot' in locals() and mse_plot is not None and np.isfinite(mse_plot):
                     mse_text = f"MSE = {mse_plot:.3e}"
                else: mse_text = "MSE: N/A"
                ax_spec.text(0.98, 0.98, mse_text, transform=ax_spec.transAxes, ha='right', va='top', fontsize=9,
                                 bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
            except Exception as e_spec:
                # Ensure ax_spec exists before trying to plot error text
                if 'ax_spec' in locals():
                    ax_spec.text(0.5, 0.5, f"Error plotting spectrum:\n{e_spec}", ha='center', va='center', transform=ax_spec.transAxes, color='red')
                else:
                    st.error(f"Error preparing spectrum plot: {e_spec}")


            # Ensure fig_spec exists before trying to adjust layout or plot
            if 'fig_spec' in locals():
                plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout slightly
                st.pyplot(fig_spec)
                plt.close(fig_spec) # Close figure to free memory
        else:
            st.warning("Missing or invalid calculation data for spectrum display.")

    # --- Subplots for Index Profile and Structure ---
    plot_col1, plot_col2 = st.columns(2)

    if 'last_calc_results' in st.session_state and st.session_state.last_calc_results:
        # Extract data needed for the subplots
        results_data = st.session_state.last_calc_results
        ep_plot = results_data.get('ep_used') # Thicknesses used for the last calculation
        l0_plot = results_data.get('l0_used') # l0 used for the last calculation
        nH_plot = results_data.get('nH_used') # H material definition used
        nL_plot = results_data.get('nL_used') # L material definition used
        nSub_plot = results_data.get('nSub_used') # Substrate definition used
        is_optimized_plot = st.session_state.is_optimized_state # Was it an optimized state?
        # material_sequence_plot = st.session_state.get('material_sequence') # Not used in current main paths

        # Check if all necessary data is available
        if ep_plot is not None and l0_plot is not None and nH_plot is not None and nL_plot is not None and nSub_plot is not None:

            # --- Index Profile Plot ---
            with plot_col1:
                fig_idx, ax_idx = plt.subplots(figsize=(6, 4)) # Adjusted height
                try:
                    # Get n+ik at l0 for plotting the profile
                    # Use _get_nk_at_lambda which should be defined earlier in the full script
                    nH_c_repr, logs_h = _get_nk_at_lambda(nH_plot, l0_plot, EXCEL_FILE_PATH)
                    nL_c_repr, logs_l = _get_nk_at_lambda(nL_plot, l0_plot, EXCEL_FILE_PATH)
                    nSub_c_repr, logs_s = _get_nk_at_lambda(nSub_plot, l0_plot, EXCEL_FILE_PATH)

                    if nH_c_repr is None or nL_c_repr is None or nSub_c_repr is None:
                        raise ValueError("Indices at l0 not found for profile plot.")

                    nH_r_repr, nL_r_repr, nSub_r_repr = nH_c_repr.real, nL_c_repr.real, nSub_c_repr.real
                    num_layers = len(ep_plot)

                    # Assume alternating H/L for index profile plot
                    n_real_layers_repr = [nH_r_repr if i % 2 == 0 else nL_r_repr for i in range(num_layers)]

                    # Calculate cumulative thickness for x-axis
                    ep_cumulative = np.cumsum(ep_plot) if num_layers > 0 else np.array([0])
                    total_thickness = ep_cumulative[-1] if num_layers > 0 else 0
                    # Add margin before substrate and after last layer (air)
                    margin = max(50, 0.1 * total_thickness) if total_thickness > 0 else 50

                    # Prepare coordinates for steps plot
                    x_coords_plot = [-margin] # Start in substrate
                    y_coords_plot = [nSub_r_repr]
                    if num_layers > 0:
                        x_coords_plot.append(0) # Substrate ends at 0
                        y_coords_plot.append(nSub_r_repr)
                        for i in range(num_layers):
                            layer_start = ep_cumulative[i-1] if i > 0 else 0
                            layer_end = ep_cumulative[i]
                            layer_n_real = n_real_layers_repr[i]
                            x_coords_plot.extend([layer_start, layer_end]) # Step at start, step at end
                            y_coords_plot.extend([layer_n_real, layer_n_real]) # Constant index within layer
                        last_layer_end = ep_cumulative[-1]
                        x_coords_plot.extend([last_layer_end, last_layer_end + margin]) # Step into air
                        y_coords_plot.extend([1.0, 1.0]) # Air index = 1.0
                    else: # Handle empty structure case
                        x_coords_plot.extend([0, 0, margin]) # Substrate -> Air transition at 0
                        y_coords_plot.extend([nSub_r_repr, 1.0, 1.0])

                    # Plot the index profile
                    ax_idx.plot(x_coords_plot, y_coords_plot, drawstyle='steps-post', label=f'n\'(λ={l0_plot:.0f}nm)', color='purple', linewidth=1.5)
                    ax_idx.set_xlabel('Depth (from substrate) (nm)')
                    ax_idx.set_ylabel("Real Part of Index (n')")
                    ax_idx.set_title(f"Index Profile (at λ={l0_plot:.0f}nm)", fontsize=10)
                    ax_idx.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
                    ax_idx.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')
                    ax_idx.minorticks_on()
                    ax_idx.set_xlim(x_coords_plot[0], x_coords_plot[-1]) # Set x-limits based on data+margin

                    # Adjust y-limits for better visualization
                    valid_n = [n for n in [1.0, nSub_r_repr] + n_real_layers_repr if np.isfinite(n)]
                    min_n = min(valid_n) if valid_n else 0.9
                    max_n = max(valid_n) if valid_n else 2.5
                    y_padding = (max_n - min_n) * 0.1 + 0.05 # Add padding
                    ax_idx.set_ylim(bottom=min_n - y_padding, top=max_n + y_padding)

                    if ax_idx.get_legend_handles_labels()[1]: ax_idx.legend(fontsize=8) # Show legend if labels exist
                except Exception as e_idx:
                    ax_idx.text(0.5, 0.5, f"Error plotting index profile:\n{e_idx}", ha='center', va='center', transform=ax_idx.transAxes, color='red')
                plt.tight_layout()
                st.pyplot(fig_idx)
                plt.close(fig_idx)

            # --- Structure Plot (Bar Chart) ---
            with plot_col2:
                fig_stack, ax_stack = plt.subplots(figsize=(6, 4)) # Adjusted height
                try:
                    num_layers = len(ep_plot)
                    if num_layers > 0:
                        # Get complex indices at l0 for labeling bars
                        indices_complex_repr = []
                        # Assuming alternating H/L structure for bar chart labeling
                        # Use _get_nk_at_lambda which should be defined earlier in the full script
                        nH_c_repr, _ = _get_nk_at_lambda(nH_plot, l0_plot, EXCEL_FILE_PATH)
                        nL_c_repr, _ = _get_nk_at_lambda(nL_plot, l0_plot, EXCEL_FILE_PATH)
                        indices_complex_repr = [nH_c_repr if i % 2 == 0 else nL_c_repr for i in range(num_layers)]
                        layer_types = ["H" if i % 2 == 0 else "L" for i in range(num_layers)]

                        # Create horizontal bar chart
                        colors = ['lightblue' if i % 2 == 0 else 'lightcoral' for i in range(num_layers)]
                        bar_pos = np.arange(num_layers)
                        bars = ax_stack.barh(bar_pos, ep_plot, align='center', color=colors, edgecolor='grey', height=0.8)

                        # Create labels for y-axis ticks
                        yticks_labels = []
                        for i in range(num_layers):
                            n_comp_repr = indices_complex_repr[i] if indices_complex_repr and i < len(indices_complex_repr) else complex(0,0)
                            layer_type = layer_types[i] if layer_types and i < len(layer_types) else '?'
                            # Format n+ik string
                            n_str = f"{n_comp_repr.real:.3f}" if np.isfinite(n_comp_repr.real) else "N/A"
                            k_val = n_comp_repr.imag
                            if np.isfinite(k_val) and abs(k_val) > 1e-6: n_str += f"{k_val:+.3f}j"
                            yticks_labels.append(f"L{i + 1} ({layer_type}) n≈{n_str}")

                        ax_stack.set_yticks(bar_pos)
                        ax_stack.set_yticklabels(yticks_labels, fontsize=7)
                        ax_stack.invert_yaxis() # Layer 1 at the top

                        # Add thickness labels to bars
                        max_ep_plot = max(ep_plot) if ep_plot.size > 0 else 1.0
                        fontsize_bar = max(6, 9 - num_layers // 15) # Adjust font size based on number of layers
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            # Position label inside or outside bar based on width
                            ha_pos = 'left' if width < max_ep_plot * 0.3 else 'right'
                            x_text_pos = width * 1.02 if ha_pos == 'left' else width * 0.98
                            text_color = 'black' if ha_pos == 'left' else 'white'
                            ax_stack.text(x_text_pos, bar.get_y() + bar.get_height()/2., f"{width:.2f}",
                                          va='center', ha=ha_pos, color=text_color, fontsize=fontsize_bar, weight='bold')
                    else: # Handle empty structure
                        ax_stack.text(0.5, 0.5, "Empty Structure", ha='center', va='center', fontsize=10, color='grey', transform=ax_stack.transAxes)
                        ax_stack.set_yticks([]); ax_stack.set_xticks([])


                    # Plot Formatting
                    ax_stack.set_xlabel('Thickness (nm)')
                    # Use state_desc which is defined earlier ('Nominal' or 'Optimized')
                    state_desc = "Optimized" if st.session_state.is_optimized_state else "Nominal"
                    stack_title_prefix = f'Structure - {state_desc}'
                    ax_stack.set_title(f"{stack_title_prefix} ({num_layers} layers)", fontsize=10)
                    max_ep_plot = max(ep_plot) if num_layers > 0 else 10 # Adjust xlim based on max thickness
                    ax_stack.set_xlim(right=max_ep_plot * 1.1) # Add padding to x-axis
                except Exception as e_stack:
                    ax_stack.text(0.5, 0.5, f"Error plotting structure:\n{e_stack}", ha='center', va='center', transform=ax_stack.transAxes, color='red')
                plt.tight_layout()
                st.pyplot(fig_stack)
                plt.close(fig_stack)
        else:
            st.warning("Missing data to display profile/structure plots.")
    else:
        st.info("Run an evaluation or optimization to see results and plots.")


# --- Recalculation Trigger Logic ---
# This block checks if a recalculation is needed (e.g., after param change, undo, optimization)
if st.session_state.get('needs_rerun_calc', False) and not st.session_state.get('calculating', False):
    params = st.session_state.rerun_calc_params
    force_ep_val = params.get('force_ep') # Get specific ep if provided (e.g., after undo)

    st.session_state.needs_rerun_calc = False # Reset flag
    st.session_state.rerun_calc_params = {}
    st.session_state.calculating = True # Set lock

    # Run the main calculation function
    # Ensure run_calculation_wrapper is defined earlier in the full script
    run_calculation_wrapper(
        is_optimized_run=params.get('is_optimized_run', False),
        method_name=params.get('method_name', 'Auto Recalc'),
        force_ep=force_ep_val
    )
    st.session_state.calculating = False # Release lock
    st.rerun() # Rerun the script to update the UI with new results/plots

# --- UI Definition Continuation (from the point where the error occurred) ---

# This section should be placed within the `with main_layout[0]:` block,
# replacing the original QWOT input and generation logic.

with main_layout[0]: # Configuration Column (Continuing...)
    # ... (Material Selectors defined earlier) ...

    st.subheader("Nominal Structure")
    # QWOT Input Area - Value comes directly from session state
    st.text_area(
        "QWOT Multipliers (comma-separated)",
        value=st.session_state.current_qwot, # Use current_qwot as the source
        key="qwot_input_display", # Use a different key if needed, maybe just for display? Or rely on current_qwot
        on_change=trigger_nominal_recalc, # Trigger recalc if user manually edits
        help="Enter Quarter Wave Optical Thickness multipliers (e.g., 1,1,1 or 1,2,1). Assumes H/L/H... starting from incident medium.",
        height=100
    )
    # Update current_qwot if the user manually edits the text area
    if st.session_state.qwot_input_display != st.session_state.current_qwot:
         st.session_state.current_qwot = st.session_state.qwot_input_display
         # Optional: trigger_nominal_recalc() could be called here too if not handled by on_change
         # clear_optimized_state() # Clear optimization if nominal is manually changed


    # Calculate number of layers from the current QWOT string for display
    num_layers_from_qwot = len([q for q in st.session_state.current_qwot.split(',') if q.strip()])

    # l0 and Layer Generation
    qwot_cols = st.columns([3,2])
    with qwot_cols[0]:
        # Ensure 'l0' key exists in session state before accessing
        l0_value = st.session_state.get('l0', 500.0) # Provide default if missing
        st.number_input("Reference λ₀ (nm)", value=l0_value, min_value=1.0, format="%.2f", key="l0", on_change=trigger_nominal_recalc)
    with qwot_cols[1]:
        # Input for generating '1's - value reflects current QWOT length
        init_layers_num = st.number_input("N Layers", min_value=0, value=num_layers_from_qwot, step=1, key="init_layers_gen_num_ui", label_visibility="collapsed", help="Number of layers to generate QWOT=1 for.")
        if st.button("Gen QWOT=1", key="gen_qwot_btn_main", use_container_width=True, help="Generate QWOT string with N layers, all set to 1."):
            if init_layers_num > 0:
                new_qwot = ",".join(['1'] * init_layers_num)
                if new_qwot != st.session_state.current_qwot:
                    st.session_state.current_qwot = new_qwot
                    # REMOVED: st.session_state.qwot_input_area = new_qwot
                    trigger_nominal_recalc() # Trigger recalc because nominal changed
                    st.rerun() # Rerun to update the text_area display
            elif st.session_state.current_qwot != "": # If N=0, clear QWOT
                st.session_state.current_qwot = ""
                # REMOVED: st.session_state.qwot_input_area = ""
                trigger_nominal_recalc() # Trigger recalc because nominal changed
                st.rerun() # Rerun to update the text_area display
    st.caption(f"Nominal Layers: {num_layers_from_qwot}")

    st.subheader("Targets & Parameters")
    param_cols = st.columns(2)
    with param_cols[0]:
      # Ensure 'l_step' key exists
      l_step_value = st.session_state.get('l_step', 10.0)
      st.number_input("λ Step (nm)", value=l_step_value, min_value=0.1, format="%.2f", key="l_step", on_change=trigger_nominal_recalc, help="Step size for optimization/MSE grid and fine plot density.")
    with param_cols[1]:
      # Ensure 'auto_thin_threshold' key exists
      thin_thresh_value = st.session_state.get('auto_thin_threshold', 1.0)
      st.number_input("Thin Thresh.", value=thin_thresh_value, min_value=MIN_THICKNESS_PHYS_NM, format="%.3f", key="auto_thin_threshold", on_change=trigger_nominal_recalc, help="Threshold for auto thin layer removal (nm) in Auto mode.")


    # Target Definition Table
    hdr_cols = st.columns([0.5, 1, 1, 1, 1])
    hdrs = ["On", "λ Min", "λ Max", "T Min", "T Max"]
    for c, h in zip(hdr_cols, hdrs): c.caption(h)

    # Ensure 'targets' exists and is a list
    if 'targets' not in st.session_state or not isinstance(st.session_state.targets, list):
        st.session_state.targets = [] # Initialize if missing

    # Assuming 5 targets are always defined or initialized
    num_target_rows = 5
    if len(st.session_state.targets) < num_target_rows:
         # Initialize missing target rows if needed
         for _ in range(num_target_rows - len(st.session_state.targets)):
              st.session_state.targets.append({'enabled': False, 'min': 0.0, 'max': 0.0, 'target_min': 0.0, 'target_max': 0.0})


    for i in range(num_target_rows): # Iterate up to the expected number of rows
        target = st.session_state.targets[i]
        cols = st.columns([0.5, 1, 1, 1, 1])
        # Use unique keys for each widget
        current_enabled = target.get('enabled', False)
        new_enabled = cols[0].checkbox("", value=current_enabled, key=f"target_enable_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['enabled'] = new_enabled
        st.session_state.targets[i]['min'] = cols[1].number_input("λmin", value=target.get('min', 0.0), format="%.1f", step=10.0, key=f"target_min_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['max'] = cols[2].number_input("λmax", value=target.get('max', 0.0), format="%.1f", step=10.0, key=f"target_max_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['target_min'] = cols[3].number_input("Tmin", value=target.get('target_min', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmin_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['target_max'] = cols[4].number_input("Tmax", value=target.get('target_max', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmax_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)


# --- Optional: Display Logs ---
# with st.expander("Logs"):
#    log_content = "\n".join(st.session_state.get('log_messages', ["No logs yet."]))
#    st.text_area("Log Messages", value=log_content, height=200, disabled=True, key="log_display_area")


