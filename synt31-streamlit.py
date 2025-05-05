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
        st.sidebar.info(f"Loaded {sheet_name}: {len(l_nm)} points ({l_nm.min():.1f}-{l_nm.max():.1f} nm)")
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

# Using numpy interpolation directly as JAX JIT might be problematic with caching complex logic inside
# Removed @jax.jit, will rely on JAX for core matrix calcs later
# Also, lru_cache might interfere with st.cache_data if not careful. Rely on st.cache_data for file loading.
def interp_nk_cached(l_target: Union[np.ndarray, jnp.ndarray],
                      l_data: Union[np.ndarray, jnp.ndarray],
                      n_data: Union[np.ndarray, jnp.ndarray],
                      k_data: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
    # Ensure inputs are numpy for interpolation if they aren't already JAX arrays handled elsewhere
    l_target_np = np.asarray(l_target)
    l_data_np = np.asarray(l_data)
    n_data_np = np.asarray(n_data)
    k_data_np = np.asarray(k_data)

    # Sort data just in case (interpolation requires monotonic x)
    sort_indices = np.argsort(l_data_np)
    l_data_sorted = l_data_np[sort_indices]
    n_data_sorted = n_data_np[sort_indices]
    k_data_sorted = k_data_np[sort_indices]

    n_interp = np.interp(l_target_np, l_data_sorted, n_data_sorted)
    k_interp = np.interp(l_target_np, l_data_sorted, k_data_sorted)
    # Return as JAX array for consistency with JAX functions
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
            if mat_upper == "FUSED SILICA":
                result = get_n_fused_silica(l_vec_target_jnp)
            elif mat_upper == "BK7":
                result = get_n_bk7(l_vec_target_jnp)
            elif mat_upper == "D263":
                 result = get_n_d263(l_vec_target_jnp)
            else:
                sheet_name = material_definition
                l_data, n_data, k_data = load_material_data_from_xlsx_sheet(excel_file_path, sheet_name)
                if l_data is None:
                    raise ValueError(f"Could not load data for sheet '{sheet_name}' from {excel_file_path}")
                # No need to convert to JAX here, interp_nk_cached handles it
                result = interp_nk_cached(l_vec_target_jnp, l_data, n_data, k_data)
        elif isinstance(material_definition, tuple) and len(material_definition) == 3:
            l_data, n_data, k_data = material_definition
            # Data provided directly, no need to convert to JAX here
            result = interp_nk_cached(l_vec_target_jnp, l_data, n_data, k_data)
        else:
            raise TypeError(f"Unsupported material definition type: {type(material_definition)}")
        return result, logs
    except Exception as e:
        logs.append(f"Error preparing material data for '{material_definition}': {e}")
        st.error(f"Failed to get nk array for {material_definition}: {e}")
        raise ValueError(f"Failed to get nk array: {e}")

# --- Function Definitions from Original Code (Calculation Core) ---
# These functions calculate the transmittance based on layer thicknesses and refractive indices.
# They heavily use JAX for performance and automatic differentiation (though grad isn't used here yet).

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
            [cos_phi,           (1j / safe_eta) * sin_phi],
            [1j * eta * sin_phi, cos_phi]
        ], dtype=jnp.complex128)
        # Matrix multiplication order matters: M_layer @ carry_matrix
        return M_layer @ carry_matrix

    def compute_identity(thickness_: jnp.ndarray) -> jnp.ndarray:
        return carry_matrix

    new_matrix = cond(
        thickness > 1e-12,
        compute_M_layer,
        compute_identity,
        thickness
    )
    return new_matrix, None

@jax.jit
def compute_stack_matrix_jax(ep_vector: jnp.ndarray, l_val: jnp.ndarray, nH_at_lval: jnp.ndarray, nL_at_lval: jnp.ndarray) -> jnp.ndarray:
    num_layers = len(ep_vector)
    layer_indices = jnp.array([nH_at_lval if i % 2 == 0 else nL_at_lval for i in range(num_layers)])

    layers_scan_data = (ep_vector, layer_indices, jnp.full(num_layers, l_val))

    M_initial = jnp.eye(2, dtype=jnp.complex128)
    # Use scan for efficient sequential matrix multiplication
    M_final, _ = scan(_compute_layer_matrix_scan_step, M_initial, layers_scan_data)
    return M_final

@jax.jit
def calculate_single_wavelength_T(l_val: jnp.ndarray, ep_vector_contig: jnp.ndarray,
                                  nH_at_lval: jnp.ndarray, nL_at_lval: jnp.ndarray, nSub_at_lval: jnp.ndarray) -> jnp.ndarray:
    etainc = 1.0 + 0j # Air
    etasub = nSub_at_lval

    def calculate_for_valid_l(l_: jnp.ndarray) -> jnp.ndarray:
        M = compute_stack_matrix_jax(ep_vector_contig, l_, nH_at_lval, nL_at_lval)
        m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

        # Transmission coefficient (amplitude) calculation
        rs_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        rs_denominator_abs = jnp.abs(rs_denominator)
        safe_denominator = jnp.where(rs_denominator_abs < 1e-12, 1e-12 + 0j, rs_denominator)
        ts = (2.0 * etainc) / safe_denominator

        # Transmittance (intensity) T = (n_sub / n_inc) * |ts|^2
        real_etasub = jnp.real(etasub)
        real_etainc = jnp.real(etainc) # Should be 1.0
        Ts = (real_etasub / real_etainc) * (ts * jnp.conj(ts))

        # Return real part, handle potential division by zero
        return jnp.where(rs_denominator_abs < 1e-12, jnp.nan, jnp.real(Ts))

    def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray:
        return jnp.nan # Return NaN for invalid wavelengths

    # Use cond to handle potentially zero or negative wavelengths
    Ts = cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)
    return Ts


@jax.jit
def calculate_T_from_ep_core_jax(ep_vector: jnp.ndarray,
                                  nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                  l_vec: jnp.ndarray) -> jnp.ndarray:
    if not l_vec.size:
        return jnp.zeros(0, dtype=jnp.float64)

    ep_vector_contig = jnp.asarray(ep_vector) # Ensure it's a JAX array

    # Vectorize the single wavelength calculation over the lambda vector
    Ts_arr = vmap(calculate_single_wavelength_T, in_axes=(0, None, 0, 0, 0))(
        l_vec, ep_vector_contig, nH_arr, nL_arr, nSub_arr
    )

    # Replace any NaNs (e.g., from invalid wavelengths) with 0.0
    return jnp.nan_to_num(Ts_arr, nan=0.0)

# --- Start of Streamlit UI Definition ---

st.title("Thin Film Optimizer (JAX - Streamlit)")

st.sidebar.header("Materials")

# Load available materials from Excel (cached)
@st.cache_data
def get_available_material_names(excel_path):
    try:
        xl = pd.ExcelFile(excel_path)
        # Exclude default sheet names like "Sheet1", "Sheet2" etc.
        sheet_names = [name for name in xl.sheet_names if not name.startswith("Sheet")]
        return ["Constant"] + sorted(sheet_names)
    except FileNotFoundError:
        st.sidebar.error(f"Excel file '{excel_path}' not found.")
        return ["Constant"]
    except Exception as e:
        st.sidebar.error(f"Error reading Excel sheets: {e}")
        return ["Constant"]

available_materials = get_available_material_names(EXCEL_FILE_PATH)
available_substrates = ["Constant", "Fused Silica", "BK7", "D263"] + [m for m in available_materials if m != "Constant"]


# Initialize session state for selections if they don't exist
if 'selected_H_material' not in st.session_state:
    default_H = "Nb2O5-Helios" if "Nb2O5-Helios" in available_materials else "Constant"
    st.session_state.selected_H_material = default_H
if 'selected_L_material' not in st.session_state:
    default_L = "SiO2-Helios" if "SiO2-Helios" in available_materials else "Constant"
    st.session_state.selected_L_material = default_L
if 'selected_Sub_material' not in st.session_state:
    st.session_state.selected_Sub_material = "Fused Silica" if "Fused Silica" in available_substrates else "Constant"

# Material Selection UI (Sidebar)
st.session_state.selected_H_material = st.sidebar.selectbox(
    "H Material",
    options=available_materials,
    index=available_materials.index(st.session_state.selected_H_material) # Use state for persistence
)

# Continuation from the previous ~200 lines.

# --- Sidebar Material UI (Continued) ---

# Helper function to get complex material value from session state or constant inputs
def get_material_value(role: str) -> MaterialInputType:
    selected_material = st.session_state[f'selected_{role}_material']
    if selected_material == "Constant":
        try:
            n_r = st.session_state[f'n{role}_r']
            n_i = st.session_state.get(f'n{role}_i', 0.0) # Substrate might not have 'i'
            if n_r <= 0:
                st.sidebar.warning(f"n' for {role} must be > 0. Using default.")
                # Provide a fallback default to avoid errors downstream
                return complex(1.5, 0.0) if role == 'Sub' else complex(2.0 if role == 'H' else 1.45, 0.0)
            if n_i < 0:
                 st.sidebar.warning(f"k for {role} must be >= 0. Using 0.")
                 n_i = 0.0
            return complex(n_r, n_i)
        except KeyError:
             st.sidebar.error(f"Constant value input missing for {role}.")
             # Provide a fallback default
             return complex(1.5, 0.0) if role == 'Sub' else complex(2.0 if role == 'H' else 1.45, 0.0)
        except Exception as e:
             st.sidebar.error(f"Invalid constant value for {role}: {e}")
             # Provide a fallback default
             return complex(1.5, 0.0) if role == 'Sub' else complex(2.0 if role == 'H' else 1.45, 0.0)
    else:
        return selected_material


# Initialize constant value session states if they don't exist
if 'nH_r' not in st.session_state: st.session_state.nH_r = 2.35
if 'nH_i' not in st.session_state: st.session_state.nH_i = 0.0
if 'nL_r' not in st.session_state: st.session_state.nL_r = 1.46
if 'nL_i' not in st.session_state: st.session_state.nL_i = 0.0
if 'nSub_r' not in st.session_state: st.session_state.nSub_r = 1.52
# Note: nSub_i is assumed 0 if Constant is chosen, unless explicitly added later

col_h1, col_h2 = st.sidebar.columns(2)
with col_h1:
    st.session_state.nH_r = st.number_input("n'", key="nH_r_input", value=st.session_state.nH_r, format="%.4f", step=0.01, disabled=(st.session_state.selected_H_material != "Constant"))
with col_h2:
    st.session_state.nH_i = st.number_input("k", key="nH_i_input", value=st.session_state.nH_i, format="%.4f", min_value=0.0, step=0.001, disabled=(st.session_state.selected_H_material != "Constant"))

st.session_state.selected_L_material = st.sidebar.selectbox(
    "L Material",
    options=available_materials,
    index=available_materials.index(st.session_state.selected_L_material)
)
col_l1, col_l2 = st.sidebar.columns(2)
with col_l1:
    st.session_state.nL_r = st.number_input("n'", key="nL_r_input", value=st.session_state.nL_r, format="%.4f", step=0.01, disabled=(st.session_state.selected_L_material != "Constant"))
with col_l2:
    st.session_state.nL_i = st.number_input("k", key="nL_i_input", value=st.session_state.nL_i, format="%.4f", min_value=0.0, step=0.001, disabled=(st.session_state.selected_L_material != "Constant"))

st.session_state.selected_Sub_material = st.sidebar.selectbox(
    "Substrate Material",
    options=available_substrates,
    index=available_substrates.index(st.session_state.selected_Sub_material)
)
st.session_state.nSub_r = st.sidebar.number_input("n' (k=0 assumed)", key="nSub_r_input", value=st.session_state.nSub_r, format="%.4f", step=0.01, disabled=(st.session_state.selected_Sub_material != "Constant"))

st.sidebar.divider()


# --- More Core Functions (Ported) ---

def calculate_T_from_ep_jax(ep_vector: Union[np.ndarray, List[float]],
                            nH_material: MaterialInputType,
                            nL_material: MaterialInputType,
                            nSub_material: MaterialInputType,
                            l_vec: Union[np.ndarray, List[float]],
                            excel_file_path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    logs = []
    l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64)
    ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)
    try:
        nH_arr, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_jnp, excel_file_path)
        logs.extend(logs_h)
        nL_arr, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_jnp, excel_file_path)
        logs.extend(logs_l)
        nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
        logs.extend(logs_sub)
    except Exception as e:
        logs.append(f"Error preparing material data for T calculation: {e}")
        raise # Re-raise to indicate failure
    Ts = calculate_T_from_ep_core_jax(ep_vector_jnp, nH_arr, nL_arr, nSub_arr, l_vec_jnp)
    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts)}, logs


# Arbitrary Sequence Calculation Functions
@jax.jit
def _compute_layer_matrix_scan_step_arbitrary(carry_matrix: jnp.ndarray, layer_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, None]:
     # This function is identical to the standard one, just renamed for clarity in the arbitrary path
     return _compute_layer_matrix_scan_step(carry_matrix, layer_data)

@jax.jit
def compute_stack_matrix_arbitrary_jax(ep_vector: jnp.ndarray, layer_indices_at_lval: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    num_layers = len(ep_vector)
    # layer_indices_at_lval must have shape (num_layers,) for a single wavelength
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
        real_etasub = jnp.real(etasub)
        real_etainc = jnp.real(etainc)
        Ts = (real_etasub / real_etainc) * (ts * jnp.conj(ts))
        return jnp.where(rs_denominator_abs < 1e-12, jnp.nan, jnp.real(Ts))

    def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray:
        return jnp.nan

    Ts = cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)
    return Ts

@jax.jit
def calculate_T_from_ep_arbitrary_core_jax(ep_vector: jnp.ndarray,
                                           layer_indices_arr: jnp.ndarray, # Shape (num_layers, num_lambdas)
                                           nSub_arr: jnp.ndarray,          # Shape (num_lambdas,)
                                           l_vec: jnp.ndarray) -> jnp.ndarray:          # Shape (num_lambdas,)
    if not l_vec.size:
        return jnp.zeros(0, dtype=jnp.float64)

    ep_vector_contig = jnp.asarray(ep_vector)

    # Transpose layer_indices_arr for vmap: needs shape (num_lambdas, num_layers)
    layer_indices_arr_transposed = layer_indices_arr.T

    # vmap over lambda, passing the corresponding column of layer indices each time
    Ts_arr = vmap(calculate_single_wavelength_T_arbitrary, in_axes=(0, None, 0, 0))(
        l_vec, ep_vector_contig, layer_indices_arr_transposed, nSub_arr
    )
    return jnp.nan_to_num(Ts_arr, nan=0.0)


def calculate_T_from_ep_arbitrary_jax(ep_vector: Union[np.ndarray, List[float]],
                                      material_sequence: List[str],
                                      nSub_material: MaterialInputType,
                                      l_vec: Union[np.ndarray, List[float]],
                                      excel_file_path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    logs = []
    l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64)
    ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)
    num_layers = len(ep_vector)

    if len(material_sequence) != num_layers:
         # Handle empty case gracefully
        if num_layers == 0 and not material_sequence:
             logs.append("Calculating for bare substrate (empty sequence and ep_vector).")
        else:
            raise ValueError(f"Length mismatch: material_sequence ({len(material_sequence)}) vs ep_vector ({num_layers}).")

    layer_indices_list = []
    for i, material_name in enumerate(material_sequence):
        try:
            material_def = material_name
            nk_arr, logs_layer = _get_nk_array_for_lambda_vec(material_def, l_vec_jnp, excel_file_path)
            logs.extend(logs_layer)
            layer_indices_list.append(nk_arr)
        except Exception as e:
            logs.append(f"Error preparing material '{material_name}' (layer {i+1}): {e}")
            raise # Re-raise to indicate failure

    # Stack the indices: result shape (num_layers, num_lambdas)
    if layer_indices_list:
        layer_indices_arr = jnp.stack(layer_indices_list, axis=0)
    else:
        # Handle empty stack case for jax function
        layer_indices_arr = jnp.empty((0, len(l_vec_jnp)), dtype=jnp.complex128)


    try:
        nSub_arr, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
        logs.extend(logs_sub)
    except Exception as e:
        logs.append(f"Error preparing substrate material: {e}")
        raise # Re-raise to indicate failure

    Ts = calculate_T_from_ep_arbitrary_core_jax(ep_vector_jnp, layer_indices_arr, nSub_arr, l_vec_jnp)
    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts)}, logs

# Function to get n, k at a single lambda (might not need caching with st.cache_data on file load)
def _get_nk_at_lambda(material_definition: MaterialInputType, l_nm_target: float, excel_file_path: str) -> Tuple[complex, List[str]]:
    logs = []
    try:
        if isinstance(material_definition, (complex, float, int)):
            result = complex(material_definition)
        else:
            l_nm_target_jnp = jnp.array([l_nm_target], dtype=jnp.float64)
            if isinstance(material_definition, str):
                mat_upper = material_definition.upper()
                if mat_upper == "FUSED SILICA":
                    result = complex(get_n_fused_silica(l_nm_target_jnp)[0])
                elif mat_upper == "BK7":
                    result = complex(get_n_bk7(l_nm_target_jnp)[0])
                elif mat_upper == "D263":
                     result = complex(get_n_d263(l_nm_target_jnp)[0])
                else:
                    sheet_name = material_definition
                    l_data, n_data, k_data = load_material_data_from_xlsx_sheet(excel_file_path, sheet_name)
                    if l_data is None:
                        raise ValueError(f"Could not load data for '{sheet_name}'")
                    # Use numpy for single point interpolation
                    n_interp = np.interp(l_nm_target, l_data, n_data)
                    k_interp = np.interp(l_nm_target, l_data, k_data)
                    result = complex(n_interp, k_interp)
            elif isinstance(material_definition, tuple) and len(material_definition) == 3:
                l_data, n_data, k_data = material_definition
                n_interp = np.interp(l_nm_target, l_data, n_data)
                k_interp = np.interp(l_nm_target, l_data, k_data)
                result = complex(n_interp, k_interp)
            else:
                raise TypeError(f"Unsupported material definition type: {type(material_definition)}")
        return result, logs
    except Exception as e:
        logs.append(f"Error getting nk for '{material_definition}' at {l_nm_target} nm: {e}")
        raise ValueError(f"Failed to get nk: {e}") # Re-raise for calling function

@jax.jit
def get_target_points_indices_jax(l_vec: jnp.ndarray, target_min: float, target_max: float) -> jnp.ndarray:
    if not l_vec.size: return jnp.empty(0, dtype=jnp.int64)
    # size=l_vec.shape[0] ensures the output size matches input, fill_value handles cases where fewer points match
    indices = jnp.where((l_vec >= target_min) & (l_vec <= target_max), size=l_vec.shape[0], fill_value=-1)[0]
    # Filter out the fill values
    return indices[indices != -1]

def calculate_initial_ep(emp: Union[List[float], Tuple[float,...]], l0: float,
                           nH0_material: MaterialInputType, nL0_material: MaterialInputType,
                           excel_file_path: str) -> Tuple[np.ndarray, List[str]]:
    logs = []
    num_layers = len(emp)
    ep_initial = np.zeros(num_layers, dtype=np.float64)

    if l0 <= 0:
        logs.append("Warning: l0 <= 0 in calculate_initial_ep. Thicknesses set to 0.")
        return ep_initial, logs

    nH_real_at_l0 = -1.0
    nL_real_at_l0 = -1.0

    try:
        nH_complex_at_l0, logs_h = _get_nk_at_lambda(nH0_material, l0, excel_file_path)
        logs.extend(logs_h)
        nL_complex_at_l0, logs_l = _get_nk_at_lambda(nL0_material, l0, excel_file_path)
        logs.extend(logs_l)

        nH_real_at_l0 = nH_complex_at_l0.real
        nL_real_at_l0 = nL_complex_at_l0.real

        if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
             logs.append(f"Warning: Real index nH({nH_real_at_l0:.3f}) or nL({nL_real_at_l0:.3f}) at l0={l0}nm is <= 0. QWOT calculation might be incorrect.")

    except Exception as e:
        logs.append(f"Error getting indices at l0={l0}nm for initial calculation: {e}")
        logs.append("Initial thicknesses will be set to 0.")
        return np.zeros(num_layers, dtype=np.float64), logs

    for i in range(num_layers):
        multiplier = emp[i]
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real_layer_at_l0 <= 1e-9:
             ep_initial[i] = 0.0 # Avoid division by zero or negative thickness
        else:
             ep_initial[i] = multiplier * l0 / (4.0 * n_real_layer_at_l0)

    return ep_initial, logs

def calculate_qwot_from_ep(ep_vector: np.ndarray, l0: float,
                            nH0_material: MaterialInputType, nL0_material: MaterialInputType,
                            excel_file_path: str) -> Tuple[np.ndarray, List[str]]:
    logs = []
    num_layers = len(ep_vector)
    qwot_multipliers = np.full(num_layers, np.nan, dtype=np.float64) # Initialize with NaN

    if l0 <= 0:
        logs.append("Warning: l0 <= 0 in calculate_qwot_from_ep. QWOT set to NaN.")
        return qwot_multipliers, logs

    nH_real_at_l0 = -1.0
    nL_real_at_l0 = -1.0

    try:
        nH_complex_at_l0, logs_h = _get_nk_at_lambda(nH0_material, l0, excel_file_path)
        logs.extend(logs_h)
        nL_complex_at_l0, logs_l = _get_nk_at_lambda(nL0_material, l0, excel_file_path)
        logs.extend(logs_l)

        nH_real_at_l0 = nH_complex_at_l0.real
        nL_real_at_l0 = nL_complex_at_l0.real

        if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
             logs.append(f"Warning: Real index nH({nH_real_at_l0:.3f}) or nL({nL_real_at_l0:.3f}) at l0={l0}nm is <= 0. QWOT calculation might be incorrect.")

    except Exception as e:
        logs.append(f"Error getting indices at l0={l0}nm for QWOT calculation: {e}")
        # Return array initialized with NaN
        return qwot_multipliers, logs

    for i in range(num_layers):
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real_layer_at_l0 <= 1e-9:
             pass # Keep the NaN value
        else:
             qwot_multipliers[i] = ep_vector[i] * (4.0 * n_real_layer_at_l0) / l0

    return qwot_multipliers, logs

def calculate_final_mse(res: Dict[str, np.ndarray], active_targets: List[Dict]) -> Tuple[Optional[float], int]:
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
            # Ensure indices are within the bounds of Ts array
            valid_indices = indices[indices < len(res_ts_np)]
            if valid_indices.size == 0: continue # Skip if no valid points in range

            calculated_Ts_in_zone = res_ts_np[valid_indices]
            target_lambdas_in_zone = res_l_np[valid_indices]

            # Filter out potential NaN/inf values in calculated Ts
            finite_mask = np.isfinite(calculated_Ts_in_zone)
            calculated_Ts_in_zone = calculated_Ts_in_zone[finite_mask]
            target_lambdas_in_zone = target_lambdas_in_zone[finite_mask]

            if calculated_Ts_in_zone.size == 0: continue # Skip if no finite points

            # Calculate target transmittance values at the exact lambda points
            if abs(l_max - l_min) < 1e-9: # Handle single point target
                interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
            else: # Handle ramp target
                slope = (t_max - t_min) / (l_max - l_min)
                interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)

            squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
            total_squared_error += np.sum(squared_errors)
            total_points_in_targets += len(calculated_Ts_in_zone)

    if total_points_in_targets > 0:
        mse = total_squared_error / total_points_in_targets
    elif active_targets: # If targets exist but no points fell within them
        mse = np.nan # Indicate calculation happened but no points matched

    return mse, total_points_in_targets

@jax.jit
def calculate_mse_for_optimization_penalized_jax(ep_vector: jnp.ndarray,
                                                  nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                                  l_vec_optim: jnp.ndarray,
                                                  active_targets_tuple: Tuple[Tuple[float, float, float, float], ...],
                                                  min_thickness_phys_nm: float) -> jnp.ndarray:

    # Penalty for layers slightly below the physical minimum (but > 0)
    # Helps discourage the optimizer from collapsing layers completely if not necessary
    below_min_mask = (ep_vector < min_thickness_phys_nm) & (ep_vector > 1e-12)
    penalty_thin = jnp.sum(jnp.where(below_min_mask, (min_thickness_phys_nm - ep_vector)**2, 0.0)) * 1e5 # High penalty factor

    # Clamp thicknesses for the actual T calculation (L-BFGS-B bounds handle the hard minimum)
    ep_vector_calc = jnp.maximum(ep_vector, min_thickness_phys_nm)

    Ts = calculate_T_from_ep_core_jax(ep_vector_calc, nH_arr, nL_arr, nSub_arr, l_vec_optim)

    total_squared_error = 0.0
    total_points_in_targets = 0

    # Iterate through targets using standard Python loop (compiles fine in JAX)
    for i in range(len(active_targets_tuple)):
        l_min, l_max, t_min, t_max = active_targets_tuple[i]
        target_mask = (l_vec_optim >= l_min) & (l_vec_optim <= l_max)

        # Calculate target values across the whole lambda vector for the current target zone
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t_full = t_min + slope * (l_vec_optim - l_min)

        # Calculate squared errors for all points
        squared_errors_full = (Ts - interpolated_target_t_full)**2

        # Apply mask to sum errors only within the target zone
        masked_sq_error = jnp.where(target_mask, squared_errors_full, 0.0)
        total_squared_error += jnp.sum(masked_sq_error)
        total_points_in_targets += jnp.sum(target_mask) # Count points in this zone

    # Calculate MSE, handle case with zero points
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, 1e7) # Large MSE if no points

    # Add penalty
    final_cost = mse + penalty_thin

    return jnp.nan_to_num(final_cost, nan=jnp.inf) # Ensure optimizer gets a finite value


@jax.jit
def calculate_mse_arbitrary_sequence_jax(ep_vector: jnp.ndarray,
                                           layer_indices_arr: jnp.ndarray, # Shape (num_layers, num_lambdas)
                                           nSub_arr: jnp.ndarray,          # Shape (num_lambdas,)
                                           l_vec_eval: jnp.ndarray,          # Shape (num_lambdas,)
                                           active_targets_tuple: Tuple[Tuple[float, float, float, float], ...]) -> jnp.ndarray:
    # Calculate Transmittance using arbitrary sequence core function
    Ts = calculate_T_from_ep_arbitrary_core_jax(ep_vector, layer_indices_arr, nSub_arr, l_vec_eval)

    total_squared_error = 0.0
    total_points_in_targets = 0

    # Iterate through targets
    for i in range(len(active_targets_tuple)):
        l_min, l_max, t_min, t_max = active_targets_tuple[i]
        target_mask = (l_vec_eval >= l_min) & (l_vec_eval <= l_max)

        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t_full = t_min + slope * (l_vec_eval - l_min)

        squared_errors_full = (Ts - interpolated_target_t_full)**2
        masked_sq_error = jnp.where(target_mask, squared_errors_full, 0.0)
        total_squared_error += jnp.sum(masked_sq_error)
        total_points_in_targets += jnp.sum(target_mask)

    # Calculate MSE, handle zero points case
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, jnp.inf) # Use infinity if no points

    return jnp.nan_to_num(mse, nan=jnp.inf) # Ensure a finite value is returned


# --- Helper Functions for Streamlit UI and Logic ---

def get_validated_input_params() -> Optional[Dict]:
    params = {}
    valid = True
    try:
        # Materials (retrieved from session state via helper)
        params['nH_material'] = get_material_value('H')
        params['nL_material'] = get_material_value('L')
        params['nSub_material'] = get_material_value('Sub')

        # Stack & Calculation Params from session state
        params['emp_str'] = st.session_state.get('nominal_qwot_str', "")
        params['initial_layer_number'] = st.session_state.get('initial_layer_number', 20)
        params['l0'] = st.session_state.get('l0_qwot', 500.0)
        params['l_step'] = st.session_state.get('lambda_step', 10.0)
        params['maxiter'] = st.session_state.get('max_iter', 1000)
        params['maxfun'] = st.session_state.get('max_fun', 1000)
        params['auto_thin_threshold'] = st.session_state.get('auto_thin_threshold', 1.0)

        # Basic validation
        if params['l0'] <= 0: st.error("Centering λ (l0) must be > 0."); valid = False
        if params['l_step'] <= 0: st.error("λ Step must be > 0."); valid = False
        if params['maxiter'] <= 0: st.error("Max Iter (Opt) must be > 0."); valid = False
        if params['maxfun'] <= 0: st.error("Max Eval (Opt) must be > 0."); valid = False
        if params['initial_layer_number'] <= 0 and params['emp_str'].strip() == "":
             st.error("Initial Layer Number must be > 0 if Nominal QWOT is empty."); valid = False
        if params['auto_thin_threshold'] < 0: st.error("Auto Thin Threshold must be >= 0."); valid = False

    except KeyError as e:
        st.error(f"Missing input parameter in session state: {e}")
        return None
    except Exception as e:
        st.error(f"Error retrieving input parameters: {e}")
        return None

    return params if valid else None

def get_validated_active_targets_from_state() -> Optional[List[Dict]]:
    active_targets = []
    targets_in_state = st.session_state.get('targets', [])
    valid = True
    if not targets_in_state:
         st.warning("No targets defined in session state.")
         return [] # Return empty list, not None, if simply no targets

    for i, target_state in enumerate(targets_in_state):
        if target_state.get('enabled', False):
            try:
                l_min = float(target_state['min'])
                l_max = float(target_state['max'])
                t_min = float(target_state['target_min'])
                t_max = float(target_state['target_max'])

                if l_max < l_min:
                    st.error(f"Target {i+1}: λ max ({l_max}) < λ min ({l_min}).")
                    valid = False
                if not (0.0 <= t_min <= 1.0):
                    st.error(f"Target {i+1}: T @ λmin ({t_min}) must be between 0 and 1.")
                    valid = False
                if not (0.0 <= t_max <= 1.0):
                     st.error(f"Target {i+1}: T @ λmax ({t_max}) must be between 0 and 1.")
                     valid = False

                active_targets.append({
                    'min': l_min, 'max': l_max,
                    'target_min': t_min, 'target_max': t_max
                })
            except ValueError as e:
                st.error(f"Target {i+1}: Invalid number format - {e}")
                valid = False
            except KeyError as e:
                 st.error(f"Target {i+1}: Missing value for {e}")
                 valid = False

    return active_targets if valid else None

def get_lambda_range_from_targets(validated_targets: Optional[List[Dict]]) -> Tuple[Optional[float], Optional[float]]:
    overall_min, overall_max = None, None
    if validated_targets: # Check if list is not None and not empty
        try:
            all_mins = [t['min'] for t in validated_targets]
            all_maxs = [t['max'] for t in validated_targets]
            if all_mins: overall_min = min(all_mins)
            if all_maxs: overall_max = max(all_maxs)
        except (KeyError, TypeError):
             # Handle cases where target dicts might be malformed
             return None, None
    return overall_min, overall_max

def add_log_message(message: Union[str, List[str]]):
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    now = datetime.datetime.now().strftime('%H:%M:%S')
    if isinstance(message, list):
         full_message = "\n".join(f"[{now}] {str(msg)}" for msg in message)
    else:
         full_message = f"[{now}] {str(message)}"
    st.session_state.log_messages.append(full_message)
    # Keep log size manageable (e.g., last 500 messages)
    max_log_size = 500
    if len(st.session_state.log_messages) > max_log_size:
         st.session_state.log_messages = st.session_state.log_messages[-max_log_size:]

# --- SciPy Optimizer Wrapper (using JAX gradient) ---
# This function remains largely the same, it bridges Scipy's interface with JAX's gradient calculation.
def scipy_obj_grad_wrapper(ep_vector_np: np.ndarray,
                            # Pass all necessary JAX arrays and static params as *args
                            nH_arr_jax: jnp.ndarray, nL_arr_jax: jnp.ndarray, nSub_arr_jax: jnp.ndarray,
                            l_vec_optim_jax: jnp.ndarray,
                            active_targets_tuple_jax: Tuple,
                            min_thickness_phys_nm_jax: float):
    # Convert numpy array from scipy optimizer back to JAX array
    ep_vector_jax = jnp.asarray(ep_vector_np)

    # Define the JAX function to compute value and gradient
    # We can define it inside or outside, but ensure it captures the args correctly
    value_and_grad_fn = jax.value_and_grad(calculate_mse_for_optimization_penalized_jax)

    # Call the JAX function
    value_jax, grad_jax = value_and_grad_fn(
        ep_vector_jax,
        nH_arr_jax, nL_arr_jax, nSub_arr_jax,
        l_vec_optim_jax,
        active_targets_tuple_jax,
        min_thickness_phys_nm_jax
    )

    # Convert results back to numpy float64 for scipy
    value_np = float(np.array(value_jax))
    grad_np = np.array(grad_jax, dtype=np.float64)

    # Optionally add checks for NaN/inf in value or gradient if needed
    if not np.isfinite(value_np):
        # st.warning(f"Objective function returned non-finite value: {value_np}")
        value_np = 1e10 # Return a large finite number
    if not np.all(np.isfinite(grad_np)):
         # st.warning("Gradient contained non-finite values. Clipping.")
         grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=1e5, neginf=-1e5)

    return value_np, grad_np


# --- Core Optimization Logic ---
# Encapsulates the call to scipy.minimize using the JAX wrapper
def _run_core_optimization(ep_start_optim: np.ndarray,
                            validated_inputs: Dict, active_targets: List[Dict],
                            min_thickness_phys: float, log_prefix: str = ""
                            ) -> Tuple[Optional[np.ndarray], bool, float, List[str], str, int, int]:
    logs = []
    num_layers_start = len(ep_start_optim)
    optim_success = False
    final_cost = np.inf
    result_message_str = "Optimization not run or failed early."
    nit_total = 0
    nfev_total = 0
    final_ep = None # Initialize as None

    if num_layers_start == 0:
        logs.append(f"{log_prefix}Cannot optimize an empty structure.")
        return None, False, np.inf, logs, "Empty structure", 0, 0

    try:
        l_min_optim, l_max_optim = get_lambda_range_from_targets(active_targets)
        if l_min_optim is None or l_max_optim is None:
            raise ValueError("Cannot determine optimization lambda range from targets.")

        l_step_optim = validated_inputs['l_step']
        num_pts_optim = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
        l_vec_optim_np = np.geomspace(l_min_optim, l_max_optim, num_pts_optim)
        l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]

        if not l_vec_optim_np.size:
            raise ValueError("Failed to generate lambda vector for optimization.")

        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        logs.append(f"{log_prefix}Preparing dispersive indices for {len(l_vec_optim_jax)} lambdas...")
        prep_start_time = time.time()

        nH_material = validated_inputs['nH_material']
        nL_material = validated_inputs['nL_material']
        nSub_material = validated_inputs['nSub_material']

        nH_arr_optim, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_h)
        nL_arr_optim, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_l)
        nSub_arr_optim, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_sub)
        logs.append(f"{log_prefix}Index preparation finished in {time.time() - prep_start_time:.3f}s.")

        active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)

        # --- Args for scipy_obj_grad_wrapper ---
        # Must match the order expected by the wrapper function
        static_args_for_scipy = (
            nH_arr_optim, nL_arr_optim, nSub_arr_optim,
            l_vec_optim_jax,
            active_targets_tuple,
            jnp.float64(min_thickness_phys) # Pass as JAX type if needed by JAX func
        )

        # Bounds for L-BFGS-B: thickness >= min_thickness_phys
        lbfgsb_bounds = [(min_thickness_phys, None)] * num_layers_start

        options = {
            'maxiter': validated_inputs['maxiter'],
            'maxfun': validated_inputs['maxfun'],
            'disp': False, # Display convergence messages? Set to True for debugging
            'ftol': 1e-12, # Tolerance for function value change
            'gtol': 1e-8   # Tolerance for gradient norm
            }

        logs.append(f"{log_prefix}Starting L-BFGS-B with JAX gradient...")
        opt_start_time = time.time()

        # Ensure starting point respects bounds
        ep_start_bounded = np.maximum(np.asarray(ep_start_optim, dtype=np.float64), min_thickness_phys)

        result = minimize(scipy_obj_grad_wrapper,
                          ep_start_bounded,
                          args=static_args_for_scipy,
                          method='L-BFGS-B',
                          jac=True, # We provide the gradient
                          bounds=lbfgsb_bounds,
                          options=options)

        logs.append(f"{log_prefix}L-BFGS-B (JAX grad) finished in {time.time() - opt_start_time:.3f}s.")

        final_cost = result.fun if np.isfinite(result.fun) else np.inf
        # Decode message if bytes (common with L-BFGS-B)
        result_message_str = result.message.decode('utf-8') if isinstance(result.message, bytes) else str(result.message)
        nit_total = result.nit if hasattr(result, 'nit') else 0
        nfev_total = result.nfev if hasattr(result, 'nfev') else 0

        # Check for success (status 0) or tolerance reached (status 1 or 2 often indicate this)
        is_success_or_limit = result.success or result.status in [0, 1, 2]

        if is_success_or_limit and np.isfinite(final_cost):
            # Ensure final result also respects bounds strictly
            final_ep = np.maximum(result.x, min_thickness_phys)
            optim_success = True
            log_status = "succeeded" if result.status == 0 else f"stopped (status {result.status})"
            logs.append(f"{log_prefix}Optimization {log_status}! Cost: {final_cost:.3e}, Iter: {nit_total}, Evals: {nfev_total}, Msg: {result_message_str}")

        else:
            optim_success = False
            # Revert to the starting point (bounded) if optimization failed
            final_ep = ep_start_bounded.copy()
            logs.append(f"{log_prefix}Optimization FAILED or produced non-finite cost. Status: {result.status}, Msg: {result_message_str}")
            try:
                # Recalculate cost at the starting point for reference
                reverted_cost, _ = scipy_obj_grad_wrapper(final_ep, *static_args_for_scipy)
                logs.append(f"{log_prefix}Reverted to initial structure (bounded). Recalculated cost: {reverted_cost:.3e}")
                final_cost = reverted_cost if np.isfinite(reverted_cost) else np.inf
            except Exception as cost_e:
                logs.append(f"{log_prefix}Reverted to initial structure (bounded). ERROR recalculating cost: {cost_e}")
                final_cost = np.inf
            # Reset counts as the optimization run itself failed
            nit_total = 0
            nfev_total = 0

    except Exception as e_optim:
        logs.append(f"{log_prefix}ERROR during core optimization setup or run: {e_optim}\n{traceback.format_exc(limit=1)}")
        final_ep = np.maximum(np.asarray(ep_start_optim), min_thickness_phys) if ep_start_optim is not None else None # Fallback
        optim_success = False
        final_cost = np.inf
        result_message_str = f"Exception: {e_optim}"
        nit_total = 0
        nfev_total = 0

    # Ensure final_ep is never None if optimization was attempted
    if final_ep is None and ep_start_optim is not None:
         final_ep = np.maximum(np.asarray(ep_start_optim), min_thickness_phys)

    return final_ep, optim_success, final_cost, logs, result_message_str, nit_total, nfev_total


# --- Needle Optimization Logic ---
# Slightly adapted to fit into the Streamlit context (logging, potential status updates)
def _perform_needle_insertion_scan(ep_vector_in: np.ndarray,
                                   nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
                                   l_vec_optim_np: np.ndarray, active_targets: List[Dict],
                                   cost_function_jax: Callable, # Pass the JAX cost function (e.g., calculate_mse_for_optimization_penalized_jax)
                                   min_thickness_phys: float, base_needle_thickness_nm: float,
                                   scan_step: float, l0_repr: float, log_prefix: str = ""
                                   ) -> Tuple[Optional[np.ndarray], float, List[str], int]:
    logs = []
    num_layers_in = len(ep_vector_in)
    best_ep_found = None
    min_cost_found = np.inf
    best_insertion_z = -1.0
    best_insertion_layer_idx = -1
    tested_insertions = 0

    if num_layers_in == 0:
        logs.append(f"{log_prefix}Cannot run needle scan on an empty structure.")
        return None, np.inf, logs, -1

    logs.append(f"{log_prefix}Starting needle scan on {num_layers_in} layers. Step: {scan_step} nm, Base needle: {base_needle_thickness_nm:.3f} nm.")

    try:
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        nH_arr_optim, logs_h = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_h)
        nL_arr_optim, logs_l = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_l)
        nSub_arr_optim, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        logs.extend(logs_sub)

        active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)

        # Prepare static args for the cost function
        static_args_cost_fn = (
            nH_arr_optim, nL_arr_optim, nSub_arr_optim,
            l_vec_optim_jax, active_targets_tuple,
            jnp.float64(min_thickness_phys) # Ensure correct type
        )

        # Compile the cost function for efficiency
        cost_fn_compiled = jax.jit(cost_function_jax)

        # Calculate initial cost
        initial_cost_jax = cost_fn_compiled(jnp.asarray(ep_vector_in), *static_args_cost_fn)
        initial_cost = float(np.array(initial_cost_jax))

        if not np.isfinite(initial_cost):
            logs.append(f"{log_prefix}ERROR: Initial cost not finite ({initial_cost}). Aborting needle scan.")
            return None, np.inf, logs, -1

        min_cost_found = initial_cost # Initialize best cost found so far
        logs.append(f"{log_prefix}Initial cost: {initial_cost:.6e}")

    except Exception as e_prep:
        logs.append(f"{log_prefix}ERROR preparing indices/initial cost for needle scan: {e_prep}. Aborting.")
        return None, np.inf, logs, -1

    # Calculate cumulative thickness for scan range
    ep_cumsum = np.cumsum(ep_vector_in)
    total_thickness = ep_cumsum[-1] if num_layers_in > 0 else 0.0

    # Get indices at l0 for determining needle type
    try:
        nH_c_l0, logs_hnk = _get_nk_at_lambda(nH_material, l0_repr, EXCEL_FILE_PATH)
        logs.extend(logs_hnk)
        nL_c_l0, logs_lnk = _get_nk_at_lambda(nL_material, l0_repr, EXCEL_FILE_PATH)
        logs.extend(logs_lnk)
        nH_real_l0 = nH_c_l0.real
        nL_real_l0 = nL_c_l0.real
    except Exception as e_nk:
        logs.append(f"{log_prefix}WARNING: Could not get n(l0={l0_repr}nm) for needle type: {e_nk}")
        nH_real_l0, nL_real_l0 = -1.0, -1.0 # Indicate failure

    # Iterate through potential insertion depths (z)
    for z in np.arange(scan_step, total_thickness, scan_step):
        current_layer_idx = -1
        layer_start_z = 0.0

        # Find which layer the insertion depth z falls into
        for i in range(num_layers_in):
            layer_end_z = ep_cumsum[i]
            if z > layer_start_z and z <= layer_end_z:
                 # Check if splitting the layer results in valid thicknesses
                 t_part1 = z - layer_start_z
                 t_part2 = layer_end_z - z
                 if t_part1 >= min_thickness_phys and t_part2 >= min_thickness_phys:
                     current_layer_idx = i
                 else:
                     # Cannot split here as one part would be too thin
                     current_layer_idx = -2 # Use -2 to indicate invalid split
                 break # Found the layer interval
            layer_start_z = layer_end_z

        # Skip if z is not within a splittable layer
        if current_layer_idx < 0: continue

        tested_insertions += 1

        # Determine needle material (opposite of the layer being split)
        is_H_layer_split = (current_layer_idx % 2 == 0)
        # Note: This assumes standard H/L alternation. Needs modification for arbitrary sequences.
        # For now, stick to H/L assumption as in original code here.

        # Check if n(l0) was successfully retrieved
        if nH_real_l0 < 0 or nL_real_l0 < 0:
             logs.append(f"{log_prefix}Skipping insertion z={z:.2f}. Failed to get n(l0) earlier.")
             continue

        # Construct the temporary ep vector with the inserted needle
        t_layer_split_1 = z - (ep_cumsum[current_layer_idx-1] if current_layer_idx > 0 else 0.0)
        t_layer_split_2 = ep_cumsum[current_layer_idx] - z

        ep_temp_np = np.concatenate((
            ep_vector_in[:current_layer_idx],
            [t_layer_split_1, base_needle_thickness_nm, t_layer_split_2],
            ep_vector_in[current_layer_idx+1:]
        ))

        # Ensure all layers in the temporary vector meet minimum thickness
        ep_temp_np = np.maximum(ep_temp_np, min_thickness_phys)

        # Calculate cost for this temporary structure
        try:
            current_cost_jax = cost_fn_compiled(jnp.asarray(ep_temp_np), *static_args_cost_fn)
            current_cost = float(np.array(current_cost_jax))

            # Check if this insertion resulted in a lower cost
            if np.isfinite(current_cost) and current_cost < min_cost_found:
                 min_cost_found = current_cost
                 best_ep_found = ep_temp_np.copy() # Store the promising structure
                 best_insertion_z = z
                 best_insertion_layer_idx = current_layer_idx

        except Exception as e_cost:
            logs.append(f"{log_prefix}WARNING: Cost calculation failed for needle at z={z:.2f}. {e_cost}")
            continue # Skip to next insertion point

    # --- Scan finished ---
    if best_ep_found is not None:
        improvement = initial_cost - min_cost_found
        logs.append(f"{log_prefix}Needle Scan Finished. Tested {tested_insertions} points.")
        logs.append(f"{log_prefix}Best improvement found: {improvement:.6e} (New Cost: {min_cost_found:.6e})")
        logs.append(f"{log_prefix}Best insertion at z={best_insertion_z:.3f} nm in original layer {best_insertion_layer_idx + 1}.")
        return best_ep_found, min_cost_found, logs, best_insertion_layer_idx
    else:
        logs.append(f"{log_prefix}Needle Scan Finished. Tested {tested_insertions} points. No improvement found.")
        return None, initial_cost, logs, -1

# --- Thin Layer Removal/Merging Logic ---
def _perform_layer_merge_or_removal_only(ep_vector_in: np.ndarray, min_thickness_phys: float,
                                           log_prefix: str = "", target_layer_index: Optional[int] = None,
                                           threshold_for_removal: Optional[float] = None) -> Tuple[np.ndarray, bool, List[str]]:
    # Note: This function assumes standard H/L alternation for merging logic.
    # It needs significant changes to handle arbitrary sequences correctly.
    current_ep = ep_vector_in.copy()
    logs = []
    num_layers = len(current_ep)
    structure_changed = False
    ep_after_merge = None # Initialize result

    # Basic checks
    if num_layers < 1:
        logs.append(f"{log_prefix}Structure is empty. Cannot remove.")
        return current_ep, False, logs
    if num_layers <= 2 and target_layer_index is None:
         # Allow removal if a specific layer (>=0) is targeted below, otherwise prevent merging/removal
         logs.append(f"{log_prefix}Structure has <= 2 layers. Cannot merge/remove automatically.")
         # Fall through to check if a specific target_layer_index allows removal
         # return current_ep, False, logs # Original logic stopped here

    try:
        thin_layer_index = -1
        min_thickness_found = np.inf

        # --- Identify the layer to remove ---
        if target_layer_index is not None:
            # User explicitly targets a layer for removal
            if 0 <= target_layer_index < num_layers:
                # Check if the target layer itself meets the threshold if one is provided
                target_thickness = current_ep[target_layer_index]
                if threshold_for_removal is None or target_thickness < threshold_for_removal:
                     # Allow removal if no threshold or if it's below threshold
                     # Also check if it's above the absolute minimum physical thickness
                     if target_thickness >= min_thickness_phys:
                           thin_layer_index = target_layer_index
                           min_thickness_found = target_thickness
                           logs.append(f"{log_prefix}Targeting layer {thin_layer_index + 1} ({min_thickness_found:.3f} nm) for removal.")
                     else:
                           logs.append(f"{log_prefix}Target layer {target_layer_index+1} ({target_thickness:.3f} nm) is below min physical thickness {min_thickness_phys:.3f} nm. Cannot remove.")
                else:
                     logs.append(f"{log_prefix}Target layer {target_layer_index+1} ({target_thickness:.3f} nm) is >= removal threshold {threshold_for_removal:.3f} nm. Not removed.")
            else:
                logs.append(f"{log_prefix}Invalid target layer index: {target_layer_index+1}. Searching automatically.")
            # If target wasn't valid or wasn't removed, fall through to automatic search
            if thin_layer_index == -1: target_layer_index = None # Reset target index

        if target_layer_index is None:
            # Automatic search for the thinnest layer (respecting threshold if given)
            candidate_indices = np.where(current_ep >= min_thickness_phys)[0] # Consider only layers above absolute min
            if candidate_indices.size > 0:
                candidate_thicknesses = current_ep[candidate_indices]
                # Apply threshold if provided
                if threshold_for_removal is not None:
                    valid_for_removal_mask = candidate_thicknesses < threshold_for_removal
                    if np.any(valid_for_removal_mask):
                        candidate_indices = candidate_indices[valid_for_removal_mask]
                        candidate_thicknesses = candidate_thicknesses[valid_for_removal_mask]
                    else:
                         # No layers are below the threshold (but above phys min)
                         candidate_indices = np.array([], dtype=int)

                # Find the thinnest among remaining candidates
                if candidate_indices.size > 0:
                    min_idx_local = np.argmin(candidate_thicknesses)
                    thin_layer_index = candidate_indices[min_idx_local]
                    min_thickness_found = candidate_thicknesses[min_idx_local]
                    logs.append(f"{log_prefix}Found thinnest layer meeting criteria: Layer {thin_layer_index + 1} ({min_thickness_found:.3f} nm).")

        # --- Perform Removal/Merge if a layer was identified ---
        if thin_layer_index == -1:
            if threshold_for_removal is not None and target_layer_index is None:
                 logs.append(f"{log_prefix}No layer found >= {min_thickness_phys:.3f} nm AND < {threshold_for_removal:.3f} nm.")
            elif target_layer_index is None:
                 logs.append(f"{log_prefix}No valid layer (>= {min_thickness_phys:.3f} nm) found for removal/merging.")
            # No layer identified for removal
            return current_ep, False, logs

        # --- Removal/Merge Logic (Assumes H/L Alternation) ---
        thin_layer_thickness = current_ep[thin_layer_index]
        merged_info = ""

        if num_layers <= 2:
            # If we only have 1 or 2 layers, and targeted one, remove both
            if thin_layer_index >= 0:
                 ep_after_merge = np.array([], dtype=np.float64)
                 merged_info = f"Removed remaining {num_layers} layer(s) (Layer {thin_layer_index+1} was targeted/thinnest)."
                 structure_changed = True
            else: # Should not happen given earlier checks, but safeguard
                 logs.append(f"{log_prefix}Cannot remove from <=2 layers without valid target.")
                 return current_ep, False, logs
        elif thin_layer_index == 0:
            # Remove first layer: merge its thickness into the (now first) layer [original index 1]
            # This assumes the second layer [idx 1] is the same material type - WRONG for H/L
            # Correct logic: Remove layer 0 AND layer 1 (assuming they form a pair to be removed)
            ep_after_merge = current_ep[2:]
            merged_info = f"Removed layers 1 & 2 (Layer 1 was thinnest)."
            structure_changed = True
        elif thin_layer_index == num_layers - 1:
             # Remove last layer: merge its thickness into the (now last) layer [original index N-2]
             # Correct logic: Remove layer N-1 AND layer N-2 (assuming they form a pair)
             if num_layers >= 2:
                   ep_after_merge = current_ep[:-2]
                   merged_info = f"Removed layers {num_layers-1} & {num_layers} (Layer {num_layers} was thinnest)."
                   structure_changed = True
             else: # Should not happen
                   logs.append(f"{log_prefix}Cannot remove last layer pair - structure too small.")
                   return current_ep, False, logs
        else:
             # Remove layer i: merge thickness of i-1 and i+1
             # Assumes i-1 and i+1 are the same material type (correct for H/L structure if i is L or H)
             merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
             ep_after_merge = np.concatenate((
                  current_ep[:thin_layer_index - 1],
                  [merged_thickness],
                  current_ep[thin_layer_index + 2:]
             ))
             merged_info = f"Removed layer {thin_layer_index+1}, merged {thin_layer_index} & {thin_layer_index+2} -> {merged_thickness:.3f} nm"
             structure_changed = True

        # Final checks and return
        if structure_changed and ep_after_merge is not None:
            logs.append(f"{log_prefix}{merged_info}. New size: {len(ep_after_merge)} layers.")
            # Ensure resulting layers still meet minimum thickness
            if ep_after_merge.size > 0:
                ep_after_merge = np.maximum(ep_after_merge, min_thickness_phys)
            return ep_after_merge, True, logs
        else:
            # If structure_changed was True but ep_after_merge is None, something went wrong
            if structure_changed:
                 logs.append(f"{log_prefix}Structure change indicated but result is None. Error in logic.")
            # Otherwise, no change was made
            return current_ep, False, logs

    except Exception as e_merge:
        logs.append(f"{log_prefix}ERROR during merge/removal logic: {e_merge}\n{traceback.format_exc(limit=1)}")
        return current_ep, False, logs # Return original state on error


# --- Placeholder for Plotting ---
# The actual plotting logic will be called within action handlers
def draw_plots(res: Dict, current_ep: Optional[np.ndarray], l0_repr: float,
               nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
               active_targets_for_plot: List[Dict], mse: Optional[float],
               is_optimized: bool = False, method_name: str = "",
               res_optim_grid: Optional[Dict] = None,
               material_sequence: Optional[List[str]] = None,
               fig_placeholder=None): # Placeholder to draw on

    if fig_placeholder is None:
         st.warning("Plotting requires a figure placeholder.")
         return

    # Reuse matplotlib logic, but target the placeholder figure/axes
    # (Assuming fig_placeholder is a dict {'fig': fig, 'axes': axes})
    fig = fig_placeholder['fig']
    axes = fig_placeholder['axes']
    fig.clear() # Clear previous plots

    # Re-add subplots after clearing
    # Check if axes is iterable and has expected length
    if not isinstance(axes, (list, np.ndarray)) or len(axes) != 3:
        axes = fig.subplots(1, 3) # Recreate if necessary
        fig_placeholder['axes'] = axes # Update placeholder
    else:
         # Ensure axes are clear if fig.clear() didn't handle it (sometimes needed)
         for ax in axes:
              ax.cla()


    ax_spec, ax_idx, ax_stack = axes[0], axes[1], axes[2]

    # --- Spectrum Plot (axes[0]) ---
    line_ts = None
    overall_l_min_plot, overall_l_max_plot = None, None
    if res and 'l' in res and 'Ts' in res and res['l'] is not None and len(res['l']) > 0:
        res_l_plot = np.asarray(res['l'])
        res_ts_plot = np.asarray(res['Ts'])
        overall_l_min_plot, overall_l_max_plot = res_l_plot.min(), res_l_plot.max()

        line_ts, = ax_spec.plot(res_l_plot, res_ts_plot, label='Transmittance', linestyle='-', color='blue', linewidth=1.5)
        plotted_target_label = False
        current_target_lines = [] # Store target lines locally for this plot instance

        if active_targets_for_plot:
            for i, target in enumerate(active_targets_for_plot):
                l_min, l_max = target['min'], target['max']
                t_min, t_max = target['target_min'], target['target_max']
                x_coords, y_coords = [l_min, l_max], [t_min, t_max]
                label = f'Target {i+1}' if not plotted_target_label else "_nolegend_"
                line_target, = ax_spec.plot(x_coords, y_coords, 'r--', linewidth=1.5, alpha=0.8, label=label, zorder=5)
                marker_target = ax_spec.plot(x_coords, y_coords, marker='x', color='red', markersize=6, linestyle='none', label='_nolegend_', zorder=6)
                current_target_lines.extend([line_target] + marker_target)
                plotted_target_label = True

                # Plot optimization grid points if available
                if res_optim_grid and 'l' in res_optim_grid and res_optim_grid['l'].size > 0:
                    res_l_optim = np.asarray(res_optim_grid['l'])
                    indices_optim = np.where((res_l_optim >= l_min) & (res_l_optim <= l_max))[0]
                    if indices_optim.size > 0:
                        optim_lambdas = res_l_optim[indices_optim]
                        if abs(l_max - l_min) < 1e-9: optim_target_t = np.full_like(optim_lambdas, t_min)
                        else: slope = (t_max - t_min) / (l_max - l_min); optim_target_t = t_min + slope * (optim_lambdas - l_min)
                        optim_markers = ax_spec.plot(optim_lambdas, optim_target_t, marker='.', color='darkred', linestyle='none', markersize=4, alpha=0.7, label='_nolegend_', zorder=6)
                        current_target_lines.extend(optim_markers)

        ax_spec.set_xlabel("Wavelength (nm)")
        ax_spec.set_ylabel('Transmittance')
        ax_spec.grid(True, linestyle=':', linewidth=0.6)
        ax_spec.set_ylim(-0.05, 1.05)
        if overall_l_min_plot is not None:
             pad = (overall_l_max_plot - overall_l_min_plot) * 0.02
             ax_spec.set_xlim(overall_l_min_plot - pad, overall_l_max_plot + pad)

        if plotted_target_label or (line_ts is not None):
             ax_spec.legend(fontsize='small')

        if mse is not None and not np.isnan(mse) and mse != -1: mse_text = f"MSE: {mse:.3e}"
        elif mse == -1: mse_text = "MSE: N/A"
        elif mse is None and active_targets_for_plot: mse_text = "MSE: Error"
        elif mse is None: mse_text = "MSE: No Target"
        else: mse_text = "MSE: No Pts"
        ax_spec.text(0.98, 0.98, mse_text, transform=ax_spec.transAxes, ha='right', va='top', fontsize='small', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

    else:
        ax_spec.text(0.5, 0.5, "No spectral data calculated", ha='center', va='center', transform=ax_spec.transAxes)

    opt_method_str = f" ({method_name})" if method_name else ""
    plot_title_status = "Optimized" if is_optimized else "Nominal"
    ax_spec.set_title(f"Spectrum {plot_title_status}{opt_method_str}")

    # --- Index Profile Plot (axes[1]) ---
    num_layers = len(current_ep) if current_ep is not None else 0
    ep_cumulative = np.cumsum(current_ep) if num_layers > 0 else np.array([])
    n_real_layers_repr = []
    nSub_c_repr = complex(1.5, 0) # Default

    try:
        nSub_c_repr, _ = _get_nk_at_lambda(nSub_material, l0_repr, EXCEL_FILE_PATH)
        if material_sequence and len(material_sequence) == num_layers:
             for mat_name in material_sequence:
                  try:
                       nk_c, _ = _get_nk_at_lambda(mat_name, l0_repr, EXCEL_FILE_PATH)
                       n_real_layers_repr.append(nk_c.real)
                  except Exception:
                       n_real_layers_repr.append(np.nan)
        elif num_layers > 0: # Assume H/L if no sequence
            nH_c_repr, _ = _get_nk_at_lambda(nH_material, l0_repr, EXCEL_FILE_PATH)
            nL_c_repr, _ = _get_nk_at_lambda(nL_material, l0_repr, EXCEL_FILE_PATH)
            for i in range(num_layers):
                 n_real_layers_repr.append(nH_c_repr.real if i % 2 == 0 else nL_c_repr.real)

    except Exception as e:
        add_log_message(f"Warning: Error getting indices at l0={l0_repr}nm for index plot: {e}")
        # Use defaults if index lookup fails
        nSub_c_repr = complex(1.5, 0)
        if num_layers > 0 and not material_sequence:
             n_real_layers_repr = [2.0 if i % 2 == 0 else 1.46 for i in range(num_layers)]
        elif num_layers > 0:
              n_real_layers_repr = [np.nan] * num_layers


    nSub_r_repr = nSub_c_repr.real
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
        y_coords_plot.extend([1.0, 1.0]) # Air index = 1.0
    else:
        x_coords_plot.extend([0, 0, margin])
        y_coords_plot.extend([nSub_r_repr, 1.0, 1.0])

    # Filter out NaN values for plotting and ylim calculation
    valid_indices_mask = ~np.isnan(y_coords_plot)
    x_coords_plot_valid = np.array(x_coords_plot)[valid_indices_mask]
    y_coords_plot_valid = np.array(y_coords_plot)[valid_indices_mask]

    if len(x_coords_plot_valid) > 1 :
        ax_idx.plot(x_coords_plot_valid, y_coords_plot_valid, drawstyle='steps-post', label=f'n\'(λ={l0_repr:.0f}nm)', color='purple', linewidth=1.5)

    ax_idx.set_xlabel('Depth (from substrate) (nm)')
    ax_idx.set_ylabel("Real Part of Index (n')")
    ax_idx.set_title(f"Index Profile (at λ={l0_repr:.0f}nm)")
    ax_idx.grid(True, linestyle=':', linewidth=0.6)
    ax_idx.set_xlim(x_coords_plot[0], x_coords_plot[-1])

    min_n_list = [1.0, nSub_r_repr] + [n for n in n_real_layers_repr if not np.isnan(n)]
    max_n_list = [1.0, nSub_r_repr] + [n for n in n_real_layers_repr if not np.isnan(n)]
    min_n = min(min_n_list) if min_n_list else 0.9
    max_n = max(max_n_list) if max_n_list else 2.5
    ax_idx.set_ylim(bottom=min_n - 0.1, top=max_n + 0.1)

    # Add labels for Substrate and Air
    offset = (max_n - min_n) * 0.05 + 0.02
    common_text_opts = {'ha':'center', 'va':'bottom', 'fontsize':'small', 'bbox':dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none')}
    n_sub_label = f"{nSub_c_repr.real:.3f}"
    if abs(nSub_c_repr.imag) > 1e-6: n_sub_label += f"{nSub_c_repr.imag:+.3f}j"
    ax_idx.text(-margin / 2, nSub_r_repr + offset, f"SUB\nn={n_sub_label}", **common_text_opts)
    air_x_pos = (total_thickness + margin / 2) if num_layers > 0 else margin / 2
    ax_idx.text(air_x_pos, 1.0 + offset, "AIR\nn=1.0", **common_text_opts)

    if len(ax_idx.get_legend_handles_labels()[1]) > 0:
         ax_idx.legend(fontsize='x-small', loc='lower right')


    # --- Stack Plot (axes[2]) ---
    if num_layers > 0 and current_ep is not None:
        # Get indices again for labels (might be redundant but ensures consistency)
        indices_complex_repr = []
        layer_labels = []
        try:
            if material_sequence and len(material_sequence) == num_layers:
                 for i, mat_name in enumerate(material_sequence):
                      nk_c, _ = _get_nk_at_lambda(mat_name, l0_repr, EXCEL_FILE_PATH)
                      indices_complex_repr.append(nk_c)
                      layer_labels.append(f"L{i+1} ({mat_name[:6]})")
            else: # Assume H/L
                 nH_c_repr, _ = _get_nk_at_lambda(nH_material, l0_repr, EXCEL_FILE_PATH)
                 nL_c_repr, _ = _get_nk_at_lambda(nL_material, l0_repr, EXCEL_FILE_PATH)
                 for i in range(num_layers):
                      is_H = (i % 2 == 0)
                      indices_complex_repr.append(nH_c_repr if is_H else nL_c_repr)
                      layer_labels.append(f"L{i+1} ({'H' if is_H else 'L'})")
        except Exception:
            indices_complex_repr = [complex(np.nan, np.nan)] * num_layers
            if not material_sequence:
                 layer_labels = [f"L{i+1} ({'H' if i%2==0 else 'L'})" for i in range(num_layers)]
            else:
                 layer_labels = [f"L{i+1} ({material_sequence[i][:6]})" if i < len(material_sequence) else f"L{i+1}" for i in range(num_layers)]


        # Define colors based on alternation or sequence if possible
        colors = ['lightblue' if i % 2 == 0 else 'lightcoral' for i in range(num_layers)] # Default H/L coloring

        bar_pos = np.arange(num_layers)
        bars = ax_stack.barh(bar_pos, current_ep, align='center', color=colors, edgecolor='grey', height=0.8)

        yticks_labels_stack = []
        for i, n_comp_repr in enumerate(indices_complex_repr):
            base_label = layer_labels[i] if i < len(layer_labels) else f"L{i+1}"
            n_str = f"n≈{n_comp_repr.real:.3f}" if not np.isnan(n_comp_repr.real) else "n=N/A"
            k_val = n_comp_repr.imag
            if not np.isnan(k_val) and abs(k_val) > 1e-6: n_str += f"{k_val:+.3f}j"
            yticks_labels_stack.append(f"{base_label} {n_str}")

        ax_stack.set_yticks(bar_pos)
        ax_stack.set_yticklabels(yticks_labels_stack, fontsize='x-small')
        ax_stack.invert_yaxis() # Layer 1 at top

        # Add thickness labels to bars
        max_ep = max(current_ep) if current_ep.size > 0 else 1.0
        fontsize_bar = max(6, 9 - num_layers // 15)
        for i, bar in enumerate(bars):
             e_val = bar.get_width()
             ha_pos = 'left' if e_val < max_ep * 0.3 else 'right'
             x_text_pos = e_val * 1.05 if ha_pos == 'left' else e_val * 0.95
             text_color = 'black' if ha_pos == 'left' else 'white'
             ax_stack.text(x_text_pos, bar.get_y() + bar.get_height()/2., f"{e_val:.2f} nm",
                           va='center', ha=ha_pos, color=text_color, fontsize=fontsize_bar, weight='bold')

    else:
        ax_stack.text(0.5, 0.5, "No layers in stack", ha='center', va='center', fontsize=10, color='grey', transform=ax_stack.transAxes)
        ax_stack.set_yticks([]); ax_stack.set_xticks([])

    ax_stack.set_xlabel('Thickness (nm)')
    ax_stack.set_title(f'Stack ({num_layers} layers)')
    if num_layers > 0: ax_stack.set_ylim(bottom=num_layers - 0.5, top=-0.5)

    # Final layout adjustments
    fig.suptitle(f"Results: {plot_title_status}{opt_method_str} (vs λ={l0_repr:.0f}nm profile)", fontsize=14) # Add reference lambda to title
    plt.tight_layout(pad=1.0, rect=[0, 0.03, 1, 0.95]) # Adjust rect to prevent suptitle overlap


# --- Initial Session State Setup ---
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    st.session_state.nominal_qwot_str = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1" # Default 20 layers
    st.session_state.initial_layer_number = 20
    st.session_state.l0_qwot = 500.0
    st.session_state.lambda_step = 10.0
    st.session_state.max_iter = 1000
    st.session_state.max_fun = 1000
    st.session_state.auto_thin_threshold = 1.0

    # Target defaults
    st.session_state.targets = [
        {'id': 0, 'enabled': True, 'min': 400.0, 'max': 500.0, 'target_min': 1.0, 'target_max': 1.0},
        {'id': 1, 'enabled': True, 'min': 500.0, 'max': 600.0, 'target_min': 1.0, 'target_max': 0.2},
        {'id': 2, 'enabled': True, 'min': 600.0, 'max': 700.0, 'target_min': 0.2, 'target_max': 0.2},
        {'id': 3, 'enabled': False, 'min': 700.0, 'max': 800.0, 'target_min': 0.2, 'target_max': 0.8},
        {'id': 4, 'enabled': False, 'min': 800.0, 'max': 900.0, 'target_min': 0.8, 'target_max': 0.8}
    ]

    st.session_state.log_messages = []
    st.session_state.current_optimized_ep = None
    st.session_state.current_material_sequence = None # For arbitrary sequences
    st.session_state.optimization_ran_since_nominal_change = False
    st.session_state.ep_history = deque(maxlen=5) # For undo
    st.session_state.last_calc_results = None # Store results for plotting
    st.session_state.last_calc_mse = None
    st.session_state.last_calc_ep = None
    st.session_state.last_calc_is_optimized = False
    st.session_state.last_calc_method_name = ""
    st.session_state.last_calc_l0_plot = 500.0
    st.session_state.status_message = "Status: Ready"
    st.session_state.plot_placeholder = None # To hold the figure object

    add_log_message("Streamlit App Initialized.")

# --- Main UI Layout ---

# Stack Definition Area
st.subheader("Stack Definition")
col_stack1, col_stack2 = st.columns([3, 1])
with col_stack1:
    st.session_state.nominal_qwot_str = st.text_area(
        "Nominal QWOT Multipliers (comma-separated)",
        value=st.session_state.nominal_qwot_str,
        key="qwot_input_area",
        height=75,
        help="Define the starting stack using Quarter-Wave Optical Thickness multipliers at the Centering λ. Example: 1,1,1,1"
    )
    # Derive initial layer number from QWOT string if possible
    try:
        current_layers = len([item for item in st.session_state.nominal_qwot_str.split(',') if item.strip()])
        st.caption(f"Layers based on QWOT string: {current_layers}")
        # Update the number input if QWOT changes, but don't overwrite user input in number field? Tricky.
        # Let's update it for consistency.
        # if current_layers != st.session_state.get('initial_layer_number', 0):
        #      st.session_state.initial_layer_number = current_layers
        # TODO: Decide on interaction between QWOT string and layer number input. Let's keep them separate for now.
    except:
        st.caption("Layers based on QWOT string: Error")

with col_stack2:
     # This number input is primarily for the QWOT scan function ('Start Nom.')
     st.session_state.initial_layer_number = st.number_input(
         "Layer # (for QWOT Scan)",
         min_value=0, value=st.session_state.initial_layer_number, step=1,
         key="init_layer_num_input",
         help="Number of layers (all QWOT=1) to generate for the 'Start Nom. (Scan+Opt)' function."
     )
     st.session_state.l0_qwot = st.number_input(
         "Centering λ (QWOT, nm)",
         min_value=1.0, value=st.session_state.l0_qwot, format="%.1f", step=10.0,
         key="l0_input",
         help="Wavelength used for QWOT calculations (initial thickness and displaying QWOT strings)."
     )

# Display Optimized QWOT if available
if st.session_state.get('optimization_ran_since_nominal_change', False) and st.session_state.get('current_optimized_ep') is not None:
     opt_ep = st.session_state.current_optimized_ep
     opt_layers = len(opt_ep)
     st.text(f"Optimized Structure ({opt_layers} layers):")
     # Recalculate QWOT based on current parameters for display
     try:
         params = get_validated_input_params()
         if params:
              qwots_opt, logs_qwot = calculate_qwot_from_ep(opt_ep, params['l0'], params['nH_material'], params['nL_material'], EXCEL_FILE_PATH)
              add_log_message(logs_qwot[-1:]) # Log only last message maybe
              if np.any(np.isnan(qwots_opt)): opt_qwot_str = "QWOT N/A (check l0/materials)"
              else: opt_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
              st.text_area("Optimized QWOT", value=opt_qwot_str, height=75, disabled=True, key="opt_qwot_display")
         else:
              st.text_area("Optimized QWOT", value="Error retrieving params", height=75, disabled=True)
     except Exception as e:
          st.text_area("Optimized QWOT", value=f"Error calculating: {e}", height=75, disabled=True)

st.divider()

# Parameters Area
st.subheader("Calculation & Optimization Parameters")
col_param1, col_param2, col_param3 = st.columns(3)
with col_param1:
     st.session_state.lambda_step = st.number_input(
         "λ Step (nm)", min_value=0.01, value=st.session_state.lambda_step, format="%.2f", step=1.0,
         key="lambda_step_input", help="Wavelength step used for optimization grid and MSE calculation."
     )
with col_param2:
    st.session_state.max_iter = st.number_input(
        "Max Iter (Opt)", min_value=1, value=st.session_state.max_iter, step=100,
        key="max_iter_input", help="Maximum iterations for the L-BFGS-B optimizer."
    )
    st.session_state.max_fun = st.number_input(
        "Max Eval (Opt)", min_value=1, value=st.session_state.max_fun, step=100,
        key="max_fun_input", help="Maximum function/gradient evaluations for the L-BFGS-B optimizer."
    )
with col_param3:
     st.session_state.auto_thin_threshold = st.number_input(
         "Auto Thin Thr. (nm)", min_value=0.0, value=st.session_state.auto_thin_threshold, format="%.3f", step=0.1,
         key="auto_thin_input", help="Thickness threshold below which layers are targeted for removal in Auto Mode."
     )

st.divider()

# Target Definition Area
st.subheader("Spectral Target (Transmittance T)")

target_cols = st.columns([0.5, 0.5, 1, 1, 1, 1]) # Adjust widths as needed
headers = ["Active", "Zone", "λ min", "λ max", "T @ λmin", "T @ λmax"]
for col, header in zip(target_cols, headers):
    col.caption(header)

# Ensure targets list exists in state
if 'targets' not in st.session_state or not isinstance(st.session_state.targets, list):
     st.session_state.targets = [] # Should have been initialized above, but safeguard

num_target_rows = 5
# Ensure state has enough rows
while len(st.session_state.targets) < num_target_rows:
     st.session_state.targets.append({'id': len(st.session_state.targets), 'enabled': False, 'min': 0.0, 'max': 0.0, 'target_min': 0.0, 'target_max': 0.0})

# Trim excess rows if any somehow exist
st.session_state.targets = st.session_state.targets[:num_target_rows]

# Display target rows
for i in range(num_target_rows):
     target_id = f"target_{i}"
     target_data = st.session_state.targets[i]
     cols = st.columns([0.5, 0.5, 1, 1, 1, 1])

     target_data['enabled'] = cols[0].checkbox("", value=target_data['enabled'], key=f"{target_id}_enabled")
     cols[1].markdown(f"**{i+1}**") # Zone number
     target_data['min'] = cols[2].number_input("λ min", value=target_data['min'], format="%.1f", step=10.0, key=f"{target_id}_min", label_visibility="collapsed")
     target_data['max'] = cols[3].number_input("λ max", value=target_data['max'], format="%.1f", step=10.0, key=f"{target_id}_max", label_visibility="collapsed")
     target_data['target_min'] = cols[4].number_input("T min", value=target_data['target_min'], min_value=0.0, max_value=1.0, format="%.3f", step=0.05, key=f"{target_id}_target_min", label_visibility="collapsed")
     target_data['target_max'] = cols[5].number_input("T max", value=target_data['target_max'], min_value=0.0, max_value=1.0, format="%.3f", step=0.05, key=f"{target_id}_target_max", label_visibility="collapsed")
     # Update the list in session state
     st.session_state.targets[i] = target_data


# --- Calculate number of points in optimization grid based on targets ---
active_targets_calc, _ = get_lambda_range_from_targets(get_validated_active_targets_from_state())
num_opt_points_str = "N/A"
if active_targets_calc:
     l_min_calc, l_max_calc = active_targets_calc
     if l_min_calc is not None and l_max_calc is not None and l_max_calc >= l_min_calc:
          l_step_calc = st.session_state.lambda_step
          if l_step_calc > 0:
               num_pts = max(2, int(np.round((l_max_calc - l_min_calc) / l_step_calc)) + 1)
               num_opt_points_str = f"{num_pts}"

st.caption(f"Approximate points in optimization grid based on active targets and λ step: {num_opt_points_str}")

st.divider()

# --- Action Buttons ---
st.subheader("Actions")
action_cols = st.columns(4)

# Row 1
evaluate_btn = action_cols[0].button("📊 Evaluate Nominal", use_container_width=True, help="Calculate and plot the spectrum for the nominal QWOT string.")
local_opt_btn = action_cols[1].button("⚙️ Local Optimizer", use_container_width=True, help="Run L-BFGS-B optimization starting from the current structure (nominal or last optimized).")
start_scan_opt_btn = action_cols[2].button("✨ Start Scan+Opt", type="primary", use_container_width=True, help="Perform exhaustive QWOT scan, test multiple l0, then optimize the best candidates.")
auto_mode_btn = action_cols[3].button("🚀 Auto Mode", use_container_width=True, help=f"Run cycles of Needle Insertion -> Thin Removal -> Local Optimization (Max {AUTO_MAX_CYCLES} cycles).")

# Row 2
action_cols2 = st.columns(4)
remove_thin_btn = action_cols2[0].button("🗑️ Remove Thin Layer", use_container_width=True, help="Remove the thinnest layer (respecting threshold) from the last optimized result and re-optimize.")
undo_remove_btn = action_cols2[1].button("↩️ Undo Remove", use_container_width=True, help="Revert to the state before the last 'Remove Thin Layer' operation.")
set_nominal_btn = action_cols2[2].button("⬇️ Optimized -> Nominal", use_container_width=True, help="Set the last optimized layer thicknesses as the new Nominal QWOT string.")
clear_opt_btn = action_cols2[3].button("🧹 Clear Optimized State", use_container_width=True, help="Clear the stored optimized result and undo history.")


st.divider()

# --- Results Display Area ---
st.subheader("Results")
results_container = st.container()
with results_container:
     st.text(st.session_state.get("status_message", "Status: Ready"))
     # Placeholder for the plot
     if st.session_state.get('plot_placeholder') is None:
         fig, axes = plt.subplots(1, 3, figsize=(18, 5)) # Create figure object
         st.session_state.plot_placeholder = {'fig': fig, 'axes': axes}
     # Display the plot using the figure object stored in session state
     st.pyplot(st.session_state.plot_placeholder['fig'])

st.divider()

# --- Log Display ---
with st.expander("Log Messages", expanded=False):
    log_container = st.container(height=300)
    log_container.text("\n".join(st.session_state.get('log_messages', ["No logs yet."])))
    if st.button("Clear Log"):
         st.session_state.log_messages = []
         add_log_message("Log Cleared.")
         st.rerun()


# --- Action Handling Logic ---
# This part needs to be carefully structured to handle button clicks and update state.
# This will come after the function definitions and UI layout.

# Continuation from the previous ~1000 lines.

def run_calculation_and_plot(ep_vector_to_use: Optional[np.ndarray] = None,
                              is_optimized: bool = False,
                              method_name: str = "",
                              material_sequence: Optional[List[str]] = None):
    calc_type = 'Optimized' if is_optimized else 'Nominal'
    add_log_message(f"\n{'='*10} Starting {calc_type} Calculation {'('+method_name+')' if method_name else ''} {'='*10}")
    # Update status immediately - This might require a rerun structure or careful placement
    # For now, just log. Status message will be set at the end.
    # st.session_state.status_message = f"Status: Running {calc_type} Calculation..." # Causes immediate rerun if uncommented

    res_optim_grid = None # Store results on optimization grid for MSE calculation
    final_ep_to_plot = None
    mse_display = None
    num_pts_mse = 0
    plot_l0 = 500.0 # Default

    try:
        # 1. Get validated inputs from session state
        validated_inputs = get_validated_input_params()
        if not validated_inputs:
            raise ValueError("Invalid input parameters. Check sidebar/inputs.")
        plot_l0 = validated_inputs['l0'] # Use l0 for plotting index profile

        # 2. Get validated targets from session state
        active_targets = get_validated_active_targets_from_state()
        if active_targets is None: # Validation failed
            raise ValueError("Invalid target definition(s). Check target inputs.")
        # No error if active_targets is an empty list, but plotting range might fail

        # 3. Determine lambda range for plotting and optimization grid
        l_min_plot, l_max_plot = get_lambda_range_from_targets(active_targets)
        if l_min_plot is None or l_max_plot is None or l_max_plot < l_min_plot:
             # If no targets, fallback to a default range maybe? Or raise error.
             # Let's use a default if no targets are active, but error if targets are invalid.
             if not active_targets: # No targets are *active*
                 l_min_plot, l_max_plot = 400.0, 900.0
                 add_log_message("No active targets found. Using default plot range 400-900 nm.")
             else: # Targets were defined but somehow resulted in invalid range (shouldn't happen with validation)
                 raise ValueError("Cannot determine a valid lambda range from active targets.")

        validated_inputs['l_range_deb'] = l_min_plot
        validated_inputs['l_range_fin'] = l_max_plot

        # Generate lambda vectors
        l_step_calc = validated_inputs['l_step']
        # Ensure a reasonable number of points for plotting
        num_plot_points = max(500, int(np.round((l_max_plot - l_min_plot) / l_step_calc) * 2) + 1)
        l_vec_plot_np = np.linspace(l_min_plot, l_max_plot, num_plot_points)
        l_vec_plot_np = l_vec_plot_np[(l_vec_plot_np > 0) & np.isfinite(l_vec_plot_np)]
        if not l_vec_plot_np.size: raise ValueError("Plot lambda vector is empty.")

        # Optimization grid lambda vector (for consistent MSE calc)
        num_optim_points = max(2, int(np.round((l_max_plot - l_min_plot) / l_step_calc)) + 1)
        l_vec_optim_np = np.geomspace(l_min_plot, l_max_plot, num_optim_points)
        l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
        # No error needed if optim grid is empty, MSE calc will just be skipped

        # 4. Determine ep_vector to use
        if ep_vector_to_use is not None:
            final_ep_to_plot = np.asarray(ep_vector_to_use, dtype=np.float64)
        else: # Calculate from nominal QWOT string
            add_log_message("Calculating initial thickness from nominal QWOT string...")
            ep_nominal_calc, logs_init_ep = calculate_initial_ep(
                [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()],
                validated_inputs['l0'],
                validated_inputs['nH_material'], validated_inputs['nL_material'], EXCEL_FILE_PATH)
            add_log_message(logs_init_ep)
            if ep_nominal_calc is None: raise ValueError("Could not calculate initial thicknesses.")
            final_ep_to_plot = np.asarray(ep_nominal_calc)

        # Ensure ep vector is valid
        if not np.all(np.isfinite(final_ep_to_plot)):
            add_log_message("Warning: Thicknesses contained NaN/inf, replaced with 0.")
            final_ep_to_plot = np.nan_to_num(final_ep_to_plot, nan=0.0, posinf=0.0, neginf=0.0)
        final_ep_to_plot = np.maximum(final_ep_to_plot, 0.0) # Ensure non-negative

        add_log_message(f"Calculating for {len(final_ep_to_plot)} layers.")

        # 5. Select calculation function and run calculations
        nH_mat = validated_inputs['nH_material']
        nL_mat = validated_inputs['nL_material']
        nSub_mat = validated_inputs['nSub_material']

        # Choose function based on material_sequence presence and validity
        calc_func = None
        if material_sequence and len(material_sequence) == len(final_ep_to_plot):
            calc_func = calculate_T_from_ep_arbitrary_jax
            calc_args_base = (final_ep_to_plot, material_sequence, nSub_mat)
            add_log_message(f"Using arbitrary material sequence ({len(material_sequence)} layers).")
        else:
            if material_sequence: # Sequence provided but invalid length
                add_log_message("Warning: Invalid material_sequence provided. Using standard H/L materials.")
                st.warning("Material sequence length mismatch, using H/L materials instead.")
                st.session_state.current_material_sequence = None # Clear invalid sequence
            calc_func = calculate_T_from_ep_jax
            calc_args_base = (final_ep_to_plot, nH_mat, nL_mat, nSub_mat)
            add_log_message("Using standard H/L materials.")


        # Calculate on fine grid for plotting
        add_log_message(f"Calculating T for {len(l_vec_plot_np)} wavelengths (Plot Grid)...")
        start_rt_time = time.time()
        calc_args_plot = calc_args_base + (l_vec_plot_np, EXCEL_FILE_PATH)
        res_fine, calc_logs_fine = calc_func(*calc_args_plot)
        add_log_message(calc_logs_fine)
        add_log_message(f"T calculation (Plot Grid) finished in {time.time() - start_rt_time:.3f}s.")

        # Calculate on optimization grid for MSE display
        if active_targets and l_vec_optim_np.size > 0:
            add_log_message(f"Calculating T for {len(l_vec_optim_np)} wavelengths (Optim Grid for MSE)...")
            start_mse_time = time.time()
            calc_args_optim = calc_args_base + (l_vec_optim_np, EXCEL_FILE_PATH)
            res_optim_grid, calc_logs_optim = calc_func(*calc_args_optim)
            add_log_message(calc_logs_optim)
            add_log_message(f"T calculation (Optim Grid) finished in {time.time() - start_mse_time:.3f}s.")

            # Calculate final MSE using the results on the optimization grid
            mse_display, num_pts_mse = calculate_final_mse(res_optim_grid, active_targets)
            if num_pts_mse > 0:
                add_log_message(f"MSE (display) = {mse_display:.3e} over {num_pts_mse} pts.")
            else:
                add_log_message("No points in target zones for display MSE.")
                mse_display = np.nan
        elif not active_targets:
             add_log_message("No active targets defined, skipping MSE calculation.")
             mse_display = None
        else: # optim grid empty
             add_log_message("Optimization grid empty, skipping MSE calculation.")
             mse_display = np.nan

        # 6. Update session state with results for plotting
        st.session_state.last_calc_results = res_fine
        st.session_state.last_calc_mse = mse_display
        st.session_state.last_calc_ep = final_ep_to_plot # Store the ep vector that was plotted
        st.session_state.last_calc_is_optimized = is_optimized
        st.session_state.last_calc_method_name = method_name
        st.session_state.last_calc_l0_plot = plot_l0
        # Store the material sequence used for the plot
        st.session_state.last_calc_material_sequence = material_sequence if calc_func == calculate_T_from_ep_arbitrary_jax else None

        # 7. Trigger plot update (drawing happens based on session state)
        add_log_message("Calculation successful. Preparing plot data...")
        st.session_state.status_message = f"Status: {calc_type} Calculation Finished"

        # 8. Update optimized state flags if necessary
        if is_optimized:
            st.session_state.current_optimized_ep = final_ep_to_plot.copy()
            st.session_state.current_material_sequence = material_sequence # Store sequence if optimized
            st.session_state.optimization_ran_since_nominal_change = True
        else: # If it was a nominal calculation, ensure optimized state is cleared
             if ep_vector_to_use is None: # Only clear if we explicitly ran nominal from QWOT
                 st.session_state.current_optimized_ep = None
                 st.session_state.current_material_sequence = None
                 st.session_state.optimization_ran_since_nominal_change = False
                 st.session_state.ep_history = deque(maxlen=5) # Clear history on fresh nominal calc


    except (ValueError, RuntimeError, TypeError) as e:
        err_msg = f"ERROR ({calc_type} Calculation): {type(e).__name__}: {e}"
        add_log_message(err_msg)
        st.error(err_msg)
        st.session_state.status_message = f"Status: {calc_type} Calculation Error"
        # Clear potentially inconsistent results state
        st.session_state.last_calc_results = None
        st.session_state.last_calc_mse = None
        st.session_state.last_calc_ep = None
    except Exception as e:
        err_msg = f"ERROR (Unexpected {calc_type} Calculation): {type(e).__name__}: {e}"
        tb_msg = traceback.format_exc()
        add_log_message(err_msg)
        add_log_message(tb_msg)
        st.error(err_msg)
        st.exception(e) # Show full traceback in Streamlit app
        st.session_state.status_message = f"Status: Unexpected {calc_type} Calculation Error"
        st.session_state.last_calc_results = None
        st.session_state.last_calc_mse = None
        st.session_state.last_calc_ep = None

    # Update the plot explicitly after calculations and state updates
    # Retrieve necessary data from session state for plotting
    plot_data = st.session_state.get('last_calc_results')
    plot_ep = st.session_state.get('last_calc_ep')
    plot_mse = st.session_state.get('last_calc_mse')
    plot_is_optimized = st.session_state.get('last_calc_is_optimized', False)
    plot_method_name = st.session_state.get('last_calc_method_name', "")
    plot_l0_ref = st.session_state.get('last_calc_l0_plot', 500.0)
    plot_material_seq = st.session_state.get('last_calc_material_sequence')
    plot_placeholder = st.session_state.get('plot_placeholder')

    if plot_data and plot_ep is not None and plot_placeholder:
         add_log_message("Updating plot...")
         try:
             # Get materials from current state for the plot legends/profile
             current_params = get_validated_input_params()
             if current_params:
                 draw_plots(
                     res=plot_data,
                     current_ep=plot_ep,
                     l0_repr=plot_l0_ref,
                     nH_material=current_params['nH_material'],
                     nL_material=current_params['nL_material'],
                     nSub_material=current_params['nSub_material'],
                     active_targets_for_plot=active_targets if active_targets is not None else [],
                     mse=plot_mse,
                     is_optimized=plot_is_optimized,
                     method_name=plot_method_name,
                     material_sequence=plot_material_seq,
                     fig_placeholder=plot_placeholder
                 )
                 add_log_message("Plot updated.")
             else:
                 add_log_message("Plot update skipped: Could not retrieve current material parameters.")
                 st.warning("Could not update plot - failed to get current parameters.")

         except Exception as plot_e:
              add_log_message(f"ERROR drawing plot: {plot_e}")
              st.error(f"Failed to draw plot: {plot_e}")
              # Attempt to clear the plot placeholder on error
              try:
                  st.session_state.plot_placeholder['fig'].clear()
                  st.session_state.plot_placeholder['axes'] = st.session_state.plot_placeholder['fig'].subplots(1, 3)
                  st.pyplot(st.session_state.plot_placeholder['fig']) # Show cleared plot
              except Exception: pass # Ignore errors during error handling
    else:
         add_log_message("Plot update skipped: Missing calculation results or plot placeholder.")
         # Optionally clear the plot if results are missing
         if plot_placeholder:
             try:
                 plot_placeholder['fig'].clear()
                 plot_placeholder['axes'] = plot_placeholder['fig'].subplots(1, 3)
                 # Add text indicating no data
                 plot_placeholder['axes'][0].text(0.5, 0.5, "Calculation needed or failed", ha='center', va='center')
                 st.pyplot(plot_placeholder['fig'])
             except Exception: pass


# --- Button Action Implementations ---

# Use a flag to prevent re-running actions on natural reruns after button press
if 'action_running' not in st.session_state:
     st.session_state.action_running = False

if evaluate_btn and not st.session_state.action_running:
     st.session_state.action_running = True
     st.session_state.current_optimized_ep = None
     st.session_state.current_material_sequence = None
     st.session_state.optimization_ran_since_nominal_change = False
     st.session_state.ep_history = deque(maxlen=5)
     add_log_message("Undo history cleared (Nominal/QWOT Calculation).")
     run_calculation_and_plot(ep_vector_to_use=None, is_optimized=False, method_name="Evaluate")
     st.session_state.action_running = False
     st.rerun() # Rerun to show the final plot and status

if local_opt_btn and not st.session_state.action_running:
     st.session_state.action_running = True
     add_log_message("\n" + "="*10 + " Starting Local Optimization " + "="*10)
     st.session_state.ep_history = deque(maxlen=5)
     add_log_message("Undo history cleared (New Local Optimization).")
     st.session_state.status_message = "Status: Starting Local Optimization..."
     st.rerun() # Show status update

if start_scan_opt_btn and not st.session_state.action_running:
     st.session_state.action_running = True
     add_log_message("\n" + "#"*10 + " Starting Scan+Opt " + "#"*10)
     st.session_state.ep_history = deque(maxlen=5)
     add_log_message("Undo history cleared (Nominal QWOT Scan).")
     st.session_state.status_message = "Status: Starting Scan+Opt..."
     st.rerun() # Show status update

if auto_mode_btn and not st.session_state.action_running:
     st.session_state.action_running = True
     add_log_message("\n" + "#"*10 + " Starting Auto Mode " + "#"*10)
     st.session_state.ep_history = deque(maxlen=5)
     add_log_message("Undo history cleared (Auto Mode).")
     st.session_state.status_message = "Status: Starting Auto Mode..."
     st.rerun() # Show status update

if remove_thin_btn and not st.session_state.action_running:
     st.session_state.action_running = True
     add_log_message("\n" + "-"*10 + " Attempting Thin Layer Removal + Re-Opt " + "-"*10)
     st.session_state.status_message = "Status: Starting Remove Thin Layer..."
     st.rerun() # Show status update

if undo_remove_btn and not st.session_state.action_running:
     st.session_state.action_running = True
     add_log_message("\n" + "~"*10 + " Undoing Last Layer Removal " + "~"*10)
     st.session_state.status_message = "Status: Undoing removal..."
     st.rerun() # Show status update

if set_nominal_btn and not st.session_state.action_running:
     st.session_state.action_running = True
     add_log_message("\n--- Setting Optimized Structure as Nominal ---")
     st.session_state.status_message = "Status: Setting Optimized as Nominal..."
     st.rerun() # Show status update

if clear_opt_btn and not st.session_state.action_running:
     st.session_state.action_running = True
     add_log_message("Clearing optimized state and Undo history.")
     st.session_state.current_optimized_ep = None
     st.session_state.current_material_sequence = None
     st.session_state.optimization_ran_since_nominal_change = False
     st.session_state.ep_history = deque(maxlen=5)
     st.session_state.last_calc_results = None # Clear plot data
     st.session_state.last_calc_mse = None
     st.session_state.last_calc_ep = None
     st.session_state.last_calc_is_optimized = False
     st.session_state.status_message = "Status: Optimized state cleared."
     # Optional: Rerun nominal calculation after clearing
     # run_calculation_and_plot(ep_vector_to_use=None, is_optimized=False, method_name="Nominal (Cleared)")
     st.session_state.action_running = False
     st.rerun()


# --- Logic to Execute Within Button Handlers (Needs to be implemented) ---

if st.session_state.get('status_message') == "Status: Starting Local Optimization...":
     # --- Local Optimizer Implementation ---
     final_ep_result = None
     try:
         with st.spinner("Running Local Optimization..."):
             validated_inputs = get_validated_input_params()
             if not validated_inputs: raise ValueError("Invalid input parameters.")
             active_targets = get_validated_active_targets_from_state()
             if active_targets is None: raise ValueError("Invalid targets.")
             if not active_targets: raise ValueError("Local optimization requires active targets.")

             ep_start = None
             if st.session_state.get('optimization_ran_since_nominal_change') and st.session_state.get('current_optimized_ep') is not None:
                  add_log_message("Using current optimized structure as starting point.")
                  ep_start = np.asarray(st.session_state.current_optimized_ep).copy()
             else:
                  add_log_message("Using nominal structure (QWOT) as starting point.")
                  ep_start_calc, logs_init_ep = calculate_initial_ep(
                      [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()],
                      validated_inputs['l0'],
                      validated_inputs['nH_material'], validated_inputs['nL_material'], EXCEL_FILE_PATH)
                  add_log_message(logs_init_ep)
                  if ep_start_calc is None: raise ValueError("Could not calculate initial thickness from QWOT.")
                  ep_start = np.asarray(ep_start_calc)

             if ep_start is None or ep_start.size == 0: raise ValueError("Cannot determine starting structure.")
             ep_start = np.maximum(ep_start, MIN_THICKNESS_PHYS_NM) # Ensure minimum thickness
             add_log_message(f"Starting optimization with {len(ep_start)} layers.")

             final_ep_result, optim_success, final_cost, optim_logs, optim_status_msg, total_nit, total_nfev = \
                 _run_core_optimization(ep_start, validated_inputs, active_targets,
                                        MIN_THICKNESS_PHYS_NM, log_prefix="  [Local Opt] ")
             add_log_message(optim_logs)

             if not optim_success:
                  raise RuntimeError(f"Local optimization failed. Msg: {optim_status_msg}, Cost: {final_cost:.3e}")

             add_log_message(f"Local Optimization finished successfully. Cost: {final_cost:.3e}, Iter/Eval: {total_nit}/{total_nfev}")
             # Update state and trigger plot for the successful result
             run_calculation_and_plot(ep_vector_to_use=final_ep_result, is_optimized=True, method_name=f"Local Opt (Cost:{final_cost:.3e})")

     except (ValueError, RuntimeError, TypeError) as e:
          err_msg = f"ERROR (Local Opt): {e}"
          add_log_message(err_msg); st.error(err_msg)
          st.session_state.status_message = "Status: Local Opt Failed"
     except Exception as e:
          err_msg = f"ERROR (Unexpected Local Opt): {type(e).__name__}: {e}"
          tb_msg = traceback.format_exc()
          add_log_message(err_msg); add_log_message(tb_msg); st.exception(e)
          st.session_state.status_message = "Status: Local Opt Failed (Unexpected)"
     finally:
         st.session_state.action_running = False # Allow other actions now
         # No rerun here, run_calculation_and_plot handles the final update/plot implicitly


if st.session_state.get('status_message') == "Status: Starting Scan+Opt...":
    # --- Scan+Opt Implementation ---
    final_ep_result = None
    try:
        with st.spinner("Running QWOT Scan + Optimization... This may take time."):
            validated_inputs = get_validated_input_params()
            if not validated_inputs: raise ValueError("Invalid input parameters.")
            # Check initial layer number specifically
            initial_layer_number = validated_inputs.get('initial_layer_number')
            if initial_layer_number is None or initial_layer_number <= 0:
                 raise ValueError("Initial Layer Number (for QWOT Scan) must be > 0.")

            active_targets = get_validated_active_targets_from_state()
            if active_targets is None: raise ValueError("Invalid targets.")
            if not active_targets: raise ValueError("Scan+Opt requires active targets.")

            l0_nominal_gui = validated_inputs['l0']
            nH_material = validated_inputs['nH_material']
            nL_material = validated_inputs['nL_material']
            nSub_material = validated_inputs['nSub_material']
            active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)

            l_min_val, l_max_val = get_lambda_range_from_targets(active_targets)
            if l_min_val is None or l_max_val is None: raise ValueError("Cannot determine lambda range.")
            l_step_optim = validated_inputs['l_step']
            # Use sparse grid for scan evaluation
            num_pts_eval_full = max(2, int(np.round((l_max_val - l_min_val) / l_step_optim)) + 1)
            l_vec_eval_full_np = np.geomspace(l_min_val, l_max_val, num_pts_eval_full)
            l_vec_eval_full_np = l_vec_eval_full_np[(l_vec_eval_full_np > 0) & np.isfinite(l_vec_eval_full_np)]
            l_vec_eval_sparse_np = l_vec_eval_full_np[::2] # Use sparser grid
            if not l_vec_eval_sparse_np.size: raise ValueError("Failed to generate sparse lambda vector for scan.")
            l_vec_eval_sparse_jax = jnp.asarray(l_vec_eval_sparse_np)
            add_log_message(f"Scan evaluation grid: {len(l_vec_eval_sparse_jax)} pts.")

            l0_values_to_test = sorted(list(set([l0_nominal_gui, l0_nominal_gui * 1.15, l0_nominal_gui * 0.85]))) # Adjusted factors maybe
            l0_values_to_test = [l for l in l0_values_to_test if l > 1e-6]
            num_l0_tests = len(l0_values_to_test)
            num_combinations = 2**initial_layer_number
            total_evals_scan = num_combinations * num_l0_tests
            add_log_message(f"Starting Exhaustive QWOT Scan N={initial_layer_number}, {num_l0_tests} l0s: {[f'{l:.1f}' for l in l0_values_to_test]}.")
            add_log_message(f"Combinations per l0: {num_combinations:,}. Total scan evals: {total_evals_scan:,}.")

            initial_candidates = []
            nSub_arr_scan, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_eval_sparse_jax, EXCEL_FILE_PATH)
            add_log_message(logs_sub)

            for l0_idx, current_l0 in enumerate(l0_values_to_test):
                add_log_message(f"\n--- Scan l0 = {current_l0:.2f} nm ({l0_idx+1}/{num_l0_tests}) ---")
                # Update spinner text - needs rerun or different approach
                # st.session_state.status_message = f"Status: Scanning l0={current_l0:.1f}..."
                try:
                    nH_c_l0, logs_h = _get_nk_at_lambda(nH_material, current_l0, EXCEL_FILE_PATH); add_log_message(logs_h)
                    nL_c_l0, logs_l = _get_nk_at_lambda(nL_material, current_l0, EXCEL_FILE_PATH); add_log_message(logs_l)
                except Exception as e:
                    add_log_message(f"ERROR getting indices at l0={current_l0:.2f} ({e}). Skipping scan for this l0."); continue

                current_best_mse_scan, current_best_multipliers_scan, scan_logs = _execute_split_stack_scan(
                    current_l0, initial_layer_number, jnp.asarray(nH_c_l0), jnp.asarray(nL_c_l0),
                    nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple )
                add_log_message(scan_logs)

                if np.isfinite(current_best_mse_scan) and current_best_multipliers_scan is not None:
                    add_log_message(f"Scan candidate found for l0={current_l0:.2f}. MSE {current_best_mse_scan:.6e}.")
                    initial_candidates.append({'l0': current_l0, 'mse_scan': current_best_mse_scan, 'multipliers': np.array(current_best_multipliers_scan)})

            if not initial_candidates: raise RuntimeError("QWOT Scan found no valid candidates.")
            add_log_message(f"\n--- QWOT Scan finished. Found {len(initial_candidates)} candidate(s). Optimizing... ---")
            initial_candidates.sort(key=lambda c: c['mse_scan'])

            final_best_ep = None; final_best_mse = np.inf; final_best_l0 = None; final_best_initial_multipliers = None
            overall_optim_nit = 0; overall_optim_nfev = 0; successful_optim_count = 0

            for idx, candidate in enumerate(initial_candidates):
                 cand_l0 = candidate['l0']; cand_mult = candidate['multipliers']; cand_mse_scan = candidate['mse_scan']
                 add_log_message(f"\n--- Optimizing Candidate {idx+1}/{len(initial_candidates)} (l0={cand_l0:.2f}, scan MSE={cand_mse_scan:.6e}) ---")
                 # st.session_state.status_message = f"Status: Local Opt {idx+1}/{len(initial_candidates)}..." # Update status
                 try:
                     ep_start_optim, logs_ep = calculate_initial_ep(cand_mult, cand_l0, nH_material, nL_material, EXCEL_FILE_PATH); add_log_message(logs_ep)
                     ep_start_optim = np.maximum(ep_start_optim, MIN_THICKNESS_PHYS_NM)
                     add_log_message(f"Starting local opt from {len(ep_start_optim)} layers.")

                     result_ep_optim, optim_success, final_cost_optim, optim_logs, optim_status_msg, nit_optim, nfev_optim = \
                         _run_core_optimization(ep_start_optim, validated_inputs, active_targets, MIN_THICKNESS_PHYS_NM, log_prefix=f"  [Opt Cand {idx+1}] ")
                     add_log_message(optim_logs)

                     if optim_success:
                          successful_optim_count += 1; overall_optim_nit += nit_optim; overall_optim_nfev += nfev_optim
                          add_log_message(f"Opt successful for candidate {idx+1}. Final MSE: {final_cost_optim:.6e}")
                          if final_cost_optim < final_best_mse:
                               add_log_message(f"*** New global best found! MSE improved from {final_best_mse:.6e} ***")
                               final_best_mse = final_cost_optim; final_best_ep = result_ep_optim.copy();
                               final_best_l0 = cand_l0; final_best_initial_multipliers = cand_mult
                     else:
                          add_log_message(f"Opt FAILED for candidate {idx+1}. Msg: {optim_status_msg}")
                 except Exception as e_optim_cand:
                      add_log_message(f"ERROR optimizing candidate {idx+1}: {e_optim_cand}")

            if final_best_ep is None: raise RuntimeError("Local optimization failed for all scan candidates.")

            add_log_message(f"\n--- Scan+Opt Best Result ---")
            add_log_message(f"Final Best MSE: {final_best_mse:.6e} ({len(final_best_ep)} layers)")
            add_log_message(f"Originating l0 = {final_best_l0:.2f} nm")
            best_mult_list_str = ",".join([f"{m:.3f}" for m in final_best_initial_multipliers])
            add_log_message(f"Original Multipliers: {best_mult_list_str}")

            # Update l0 in UI if it changed
            if abs(final_best_l0 - l0_nominal_gui) > 1e-3:
                 add_log_message(f"Updating GUI l0 from {l0_nominal_gui:.2f} to best found value {final_best_l0:.2f}")
                 st.session_state.l0_qwot = final_best_l0 # Update state

            final_ep_result = final_best_ep # Store result to plot outside spinner

    except (ValueError, RuntimeError, TypeError) as e:
          err_msg = f"ERROR (Scan+Opt): {e}"
          add_log_message(err_msg); st.error(err_msg)
          st.session_state.status_message = "Status: Scan+Opt Failed"
    except Exception as e:
          err_msg = f"ERROR (Unexpected Scan+Opt): {type(e).__name__}: {e}"
          tb_msg = traceback.format_exc()
          add_log_message(err_msg); add_log_message(tb_msg); st.exception(e)
          st.session_state.status_message = "Status: Scan+Opt Failed (Unexpected)"
    finally:
        st.session_state.action_running = False
        if final_ep_result is not None:
            # Plot the final best result
            run_calculation_and_plot(ep_vector_to_use=final_ep_result, is_optimized=True, method_name=f"Scan+Opt (L0={final_best_l0:.1f}, MSE:{final_best_mse:.3e})")
            st.rerun() # Show final plot
        else:
            st.rerun() # Rerun to show error status


if st.session_state.get('status_message') == "Status: Starting Auto Mode...":
     # --- Auto Mode Implementation ---
     final_ep_result = None
     try:
          start_time_auto = time.time()
          best_ep_so_far = None
          best_mse_so_far = np.inf
          num_cycles_done = 0
          termination_reason = f"Max {AUTO_MAX_CYCLES} cycles reached"
          total_iters_auto = 0; total_evals_auto = 0; optim_runs_auto = 0

          with st.spinner("Running Auto Mode... This may take several minutes."):
                validated_inputs = get_validated_input_params()
                if not validated_inputs: raise ValueError("Invalid input parameters.")
                active_targets = get_validated_active_targets_from_state()
                if active_targets is None: raise ValueError("Invalid targets.")
                if not active_targets: raise ValueError("Auto mode requires active targets.")

                nH_material = validated_inputs['nH_material']; nL_material = validated_inputs['nL_material']
                nSub_material = validated_inputs['nSub_material']; l0 = validated_inputs['l0']
                threshold_from_gui = validated_inputs.get('auto_thin_threshold', 1.0)
                add_log_message(f"Auto Thin Removal Threshold: {threshold_from_gui:.3f} nm")

                l_min_val, l_max_val = get_lambda_range_from_targets(active_targets)
                if l_min_val is None or l_max_val is None: raise ValueError("Cannot determine lambda range.")
                l_step_optim = validated_inputs['l_step']
                num_pts = max(2, int(np.round((l_max_val - l_min_val) / l_step_optim)) + 1)
                l_vec_optim_np = np.geomspace(l_min_val, l_max_val, num_pts)
                l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
                if not l_vec_optim_np.size: raise ValueError("Failed to generate lambda vector for Auto Mode.")

                ep_start_auto = None
                initial_opt_needed = False
                if st.session_state.get('optimization_ran_since_nominal_change') and st.session_state.get('current_optimized_ep') is not None:
                     add_log_message("Auto Mode: Using existing optimized structure as start.")
                     ep_start_auto = np.asarray(st.session_state.current_optimized_ep).copy()
                     # Calculate initial MSE
                     try:
                         l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
                         nH_arr, _ = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH)
                         nL_arr, _ = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH)
                         nSub_arr, _ = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH)
                         active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
                         static_args_cost_fn = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, jnp.float64(MIN_THICKNESS_PHYS_NM))
                         cost_fn_compiled = jax.jit(calculate_mse_for_optimization_penalized_jax)
                         initial_cost_jax = cost_fn_compiled(jnp.asarray(ep_start_auto), *static_args_cost_fn)
                         best_mse_so_far = float(np.array(initial_cost_jax))
                         if not np.isfinite(best_mse_so_far): raise ValueError("Initial MSE not finite")
                     except Exception as e: raise ValueError(f"Cannot calculate initial MSE for auto mode: {e}")
                else:
                    add_log_message("Auto Mode: Using nominal structure (QWOT) as start.")
                    initial_opt_needed = True
                    ep_nominal_calc, logs_init_ep = calculate_initial_ep(
                        [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()],
                        l0, nH_material, nL_material, EXCEL_FILE_PATH)
                    add_log_message(logs_init_ep)
                    if ep_nominal_calc is None or ep_nominal_calc.size == 0: raise ValueError("Cannot get nominal starting structure.")
                    ep_nominal = np.maximum(np.asarray(ep_nominal_calc), MIN_THICKNESS_PHYS_NM)
                    add_log_message(f"Nominal structure has {len(ep_nominal)} layers. Running initial optimization...")

                    ep_after_initial_opt, initial_opt_success, initial_mse, initial_opt_logs, initial_opt_msg, initial_nit, initial_nfev = \
                        _run_core_optimization(ep_nominal, validated_inputs, active_targets, MIN_THICKNESS_PHYS_NM, log_prefix="  [Auto Init Opt] ")
                    add_log_message(initial_opt_logs)
                    if not initial_opt_success: raise RuntimeError(f"Initial optimization failed: {initial_opt_msg}")
                    add_log_message(f"Initial opt finished. MSE: {initial_mse:.6e} (Iter/Eval: {initial_nit}/{initial_nfev})")
                    ep_start_auto = ep_after_initial_opt.copy()
                    best_mse_so_far = initial_mse
                    total_iters_auto += initial_nit; total_evals_auto += initial_nfev; optim_runs_auto += 1

                best_ep_so_far = ep_start_auto.copy()
                if not np.isfinite(best_mse_so_far): raise ValueError("Starting MSE for cycles not finite.")
                add_log_message(f"Starting Auto Mode Cycles with MSE: {best_mse_so_far:.6e} ({len(best_ep_so_far)} layers)")

                # --- Auto Mode Cycle Loop ---
                for cycle_num in range(AUTO_MAX_CYCLES):
                    add_log_message(f"\n--- Auto Cycle {cycle_num + 1} / {AUTO_MAX_CYCLES} ---")
                    # Update spinner text - tricky without reruns, log is primary feedback
                    mse_at_cycle_start = best_mse_so_far
                    ep_at_cycle_start = best_ep_so_far.copy()
                    cycle_improved_overall = False

                    # 1. Needle Phase
                    add_log_message(f" [Cycle {cycle_num+1}] Running {AUTO_NEEDLES_PER_CYCLE} needle iterations...")
                    ep_after_needles, mse_after_needles, needle_logs, nit_needles, nfev_needles, reopts_in_needles = \
                        _run_needle_iterations(best_ep_so_far, AUTO_NEEDLES_PER_CYCLE, validated_inputs, active_targets,
                                               MIN_THICKNESS_PHYS_NM, l_vec_optim_np, DEFAULT_NEEDLE_SCAN_STEP_NM,
                                               BASE_NEEDLE_THICKNESS_NM, calculate_mse_for_optimization_penalized_jax, # Pass cost func
                                               log_prefix=f"    [Auto Cycle {cycle_num+1} Needle] ")
                    add_log_message(needle_logs)
                    add_log_message(f" [Cycle {cycle_num+1}] MSE after Needles: {mse_after_needles:.6e} (Iter/Eval Sum: {nit_needles}/{nfev_needles})")
                    total_iters_auto += nit_needles; total_evals_auto += nfev_needles; optim_runs_auto += reopts_in_needles

                    if mse_after_needles < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                         add_log_message("    Needle phase improved MSE globally.")
                         best_ep_so_far = ep_after_needles.copy(); best_mse_so_far = mse_after_needles; cycle_improved_overall = True
                    else:
                         add_log_message("    Needle phase did not improve MSE significantly vs global best.")
                         best_ep_so_far = ep_after_needles.copy(); best_mse_so_far = mse_after_needles # Still update state

                    # 2. Thin Removal Phase
                    add_log_message(f" [Cycle {cycle_num+1}] Running Thin Removal + Re-Opt Phase (Threshold: {threshold_from_gui:.3f} nm)...")
                    layers_removed_this_cycle = 0; thinning_loop_iteration = 0
                    max_thinning_iterations = len(best_ep_so_far) + 2 # Safety limit

                    while thinning_loop_iteration < max_thinning_iterations:
                         thinning_loop_iteration += 1
                         current_num_layers = len(best_ep_so_far)
                         if current_num_layers <= 2: add_log_message("    Structure too small for thin removal."); break

                         ep_before_thin_iter = best_ep_so_far.copy()
                         mse_before_thin_iter = best_mse_so_far

                         ep_after_single_removal, structure_changed, removal_logs = _perform_layer_merge_or_removal_only(
                              best_ep_so_far, MIN_THICKNESS_PHYS_NM, log_prefix=f"      [Remove Iter {thinning_loop_iteration}]",
                              threshold_for_removal=threshold_from_gui)
                         add_log_message(removal_logs)

                         if structure_changed:
                              layers_removed_this_cycle += 1
                              add_log_message(f"    Layer removed. Re-optimizing {len(ep_after_single_removal)} layers...")
                              ep_after_thin_reopt, thin_reopt_success, mse_after_thin_reopt, thin_reopt_logs, thin_reopt_msg, nit_thin_reopt, nfev_thin_reopt = \
                                   _run_core_optimization(ep_after_single_removal, validated_inputs, active_targets, MIN_THICKNESS_PHYS_NM, log_prefix=f"        [RemoveReOpt {layers_removed_this_cycle}] ")
                              add_log_message(thin_reopt_logs)

                              if thin_reopt_success:
                                   add_log_message(f"    Re-opt successful. MSE: {mse_after_thin_reopt:.6e} (Iter/Eval: {nit_thin_reopt}/{nfev_thin_reopt})")
                                   total_iters_auto += nit_thin_reopt; total_evals_auto += nfev_thin_reopt; optim_runs_auto += 1
                                   if mse_after_thin_reopt < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                                        add_log_message("        MSE improved globally. Updating best state.")
                                        best_ep_so_far = ep_after_thin_reopt.copy(); best_mse_so_far = mse_after_thin_reopt; cycle_improved_overall = True
                                   else:
                                        add_log_message("        MSE not improved globally. Keeping result for next removal attempt.")
                                        best_ep_so_far = ep_after_thin_reopt.copy(); best_mse_so_far = mse_after_thin_reopt # Update current state
                              else:
                                   add_log_message(f"    WARNING: Re-opt after removal failed ({thin_reopt_msg}). Stopping removal phase.")
                                   best_ep_so_far = ep_after_single_removal.copy() # Keep the state after removal but before failed re-opt
                                   # Try to calculate MSE for this state
                                   try:
                                       cost_fn_compiled = jax.jit(calculate_mse_for_optimization_penalized_jax) # Reuse compiled func if possible
                                       static_args_cost_fn = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, jnp.float64(MIN_THICKNESS_PHYS_NM)) # Reuse arrays
                                       failed_cost_jax = cost_fn_compiled(jnp.asarray(best_ep_so_far), *static_args_cost_fn)
                                       best_mse_so_far = float(np.array(failed_cost_jax))
                                       if not np.isfinite(best_mse_so_far): best_mse_so_far = np.inf
                                   except Exception as cost_e: best_mse_so_far = np.inf
                                   add_log_message(f"    MSE after Remove+Failed ReOpt (reverted): {best_mse_so_far:.6e}")
                                   cycle_improved_overall = (best_mse_so_far < mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE) or cycle_improved_overall
                                   break # Stop thinning loop for this cycle
                         else:
                              add_log_message("    No more layers found below threshold."); break # End thinning loop for this cycle

                    add_log_message(f" [Cycle {cycle_num+1}] Thin Removal Phase finished. {layers_removed_this_cycle} layer(s) removed.")
                    num_cycles_done += 1

                    # 3. Check for Improvement and Termination
                    add_log_message(f"--- End Auto Cycle {cycle_num + 1} --- Current best MSE: {best_mse_so_far:.6e} ({len(best_ep_so_far)} layers)---")
                    if not cycle_improved_overall and best_mse_so_far >= mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE:
                         add_log_message(f"No significant improvement in Cycle {cycle_num + 1}. Stopping Auto Mode.")
                         termination_reason = "No improvement"
                         # Check if MSE actually worsened, if so, revert
                         if best_mse_so_far > mse_at_cycle_start + MSE_IMPROVEMENT_TOLERANCE :
                             add_log_message(f"Reverting to state before Cycle {cycle_num + 1} (MSE {mse_at_cycle_start:.6e}).")
                             best_ep_so_far = ep_at_cycle_start; best_mse_so_far = mse_at_cycle_start
                         break # Exit cycle loop

          # --- Auto Mode Finished ---
          final_ep_result = best_ep_so_far # Store final result
          add_log_message(f"\n--- Auto Mode Finished after {num_cycles_done} cycles ---")
          add_log_message(f"Termination Reason: {termination_reason}")
          add_log_message(f"Final Best MSE: {best_mse_so_far:.6e} with {len(final_ep_result)} layers.")
          # Log stats
          avg_nit_str = f"{total_iters_auto / optim_runs_auto:.1f}" if optim_runs_auto > 0 else "N/A"
          avg_nfev_str = f"{total_evals_auto / optim_runs_auto:.1f}" if optim_runs_auto > 0 else "N/A"
          add_log_message(f"Total successful optimizations: {optim_runs_auto}")
          add_log_message(f"Total iterations sum: {total_iters_auto}")
          add_log_message(f"Total func/grad evaluations sum: {total_evals_auto}")
          add_log_message(f"Average Iterations / successful run: {avg_nit_str}")
          add_log_message(f"Average Evaluations / successful run: {avg_nfev_str}")


     except (ValueError, RuntimeError, TypeError) as e:
          err_msg = f"ERROR (Auto Mode): {e}"
          add_log_message(err_msg); st.error(err_msg)
          st.session_state.status_message = "Status: Auto Mode Failed"
     except Exception as e:
          err_msg = f"ERROR (Unexpected Auto Mode): {type(e).__name__}: {e}"
          tb_msg = traceback.format_exc()
          add_log_message(err_msg); add_log_message(tb_msg); st.exception(e)
          st.session_state.status_message = "Status: Auto Mode Failed (Unexpected)"
     finally:
         st.session_state.action_running = False
         if final_ep_result is not None:
             # Plot the final best result
             run_calculation_and_plot(ep_vector_to_use=final_ep_result, is_optimized=True, method_name=f"Auto Mode ({num_cycles_done} cyc, {termination_reason})")
             st.rerun()
         else:
             st.rerun()


if st.session_state.get('status_message') == "Status: Starting Remove Thin Layer...":
    # --- Remove Thin Layer Implementation ---
    final_ep_result = None
    try:
        if not st.session_state.get('optimization_ran_since_nominal_change') or st.session_state.get('current_optimized_ep') is None:
            raise ValueError("Remove Thin Layer requires a valid optimized structure. Run optimization first.")

        current_ep_opt = np.asarray(st.session_state.current_optimized_ep)
        if len(current_ep_opt) <= 2:
             raise ValueError("Structure has <= 2 layers. Cannot remove/merge further.")

        with st.spinner("Removing thin layer and re-optimizing..."):
             validated_inputs = get_validated_input_params()
             if not validated_inputs: raise ValueError("Invalid input parameters.")
             active_targets = get_validated_active_targets_from_state()
             if active_targets is None: raise ValueError("Invalid targets.")
             if not active_targets: raise ValueError("Cannot re-optimize without active targets.")

             # Save state for undo
             st.session_state.ep_history.append(current_ep_opt.copy())
             add_log_message(f" [Undo] State saved before removal. History size: {len(st.session_state.ep_history)}")

             ep_before_removal = current_ep_opt.copy()
             threshold_rem = validated_inputs.get('auto_thin_threshold', None) # Use threshold from UI if available
             add_log_message(f"Attempting removal with threshold: {threshold_rem if threshold_rem is not None else 'None'}")

             ep_after_removal, structure_changed, removal_logs = _perform_layer_merge_or_removal_only(
                  ep_before_removal, MIN_THICKNESS_PHYS_NM, log_prefix="  [Removal] ", threshold_for_removal=threshold_rem
             )
             add_log_message(removal_logs)

             if not structure_changed:
                 add_log_message("No layer removed or structure unchanged.")
                 st.info("Could not remove a layer according to criteria, or structure unchanged.")
                 st.session_state.status_message = f"Status: Removal Skipped | Layers: {len(ep_before_removal)}"
                 # Remove the unneeded history state
                 if st.session_state.ep_history:
                      try: st.session_state.ep_history.pop(); add_log_message(" [Undo] Unneeded history state removed.")
                      except IndexError: pass
                 final_ep_result = ep_before_removal # Keep the original structure
                 run_calculation_and_plot(ep_vector_to_use=final_ep_result, is_optimized=True, method_name="Optimized (Removal Skipped)")

             else:
                 add_log_message(f"Structure changed to {len(ep_after_removal)} layers. Re-optimizing...")
                 ep_after_reopt, reopt_success, final_cost, reopt_logs, reopt_status_msg, reopt_nit, reopt_nfev = \
                       _run_core_optimization(ep_after_removal, validated_inputs, active_targets, MIN_THICKNESS_PHYS_NM, log_prefix="  [Re-Opt] ")
                 add_log_message(reopt_logs)

                 if not reopt_success:
                      add_log_message("ERROR: Re-optimization after removal failed.")
                      st.warning(f"Layer removed, but re-optimization failed: {reopt_status_msg}. Displaying structure *after* removal.")
                      final_ep_result = ep_after_removal.copy() # Keep state after removal
                      st.session_state.status_message = f"Status: Removed, Re-Opt Failed | Layers: {len(final_ep_result)}"
                 else:
                      add_log_message("Re-optimization successful.")
                      final_ep_result = ep_after_reopt.copy()
                      st.session_state.status_message = f"Status: Layer Removed & Re-Opt OK | MSE: {final_cost:.3e} | Layers: {len(final_ep_result)}"

                 # Update state and plot the result (either from failed or successful re-opt)
                 run_calculation_and_plot(ep_vector_to_use=final_ep_result, is_optimized=True, method_name=f"Optimized Post-Removal{' (Re-Opt Failed)' if not reopt_success else ''}")

    except (ValueError, RuntimeError, TypeError) as e:
          err_msg = f"ERROR (Remove Thin): {e}"
          add_log_message(err_msg); st.error(err_msg)
          st.session_state.status_message = "Status: Remove Thin Failed"
          # Remove potentially bad undo state if error occurred before successful removal
          # if 'structure_changed' not in locals() or not structure_changed:
          #      if st.session_state.ep_history:
          #           try: st.session_state.ep_history.pop(); add_log_message(" [Undo] History state removed due to error during removal prep.")
          #           except IndexError: pass
    except Exception as e:
          err_msg = f"ERROR (Unexpected Remove Thin): {type(e).__name__}: {e}"
          tb_msg = traceback.format_exc()
          add_log_message(err_msg); add_log_message(tb_msg); st.exception(e)
          st.session_state.status_message = "Status: Remove Thin Failed (Unexpected)"
    finally:
        st.session_state.action_running = False
        st.rerun() # Rerun to show final state/plot


if st.session_state.get('status_message') == "Status: Undoing removal...":
     # --- Undo Remove Implementation ---
     try:
         if not st.session_state.ep_history:
             add_log_message("Undo history empty."); st.info("Nothing to undo.")
             st.session_state.status_message = "Status: Ready"
         else:
             with st.spinner("Restoring previous state..."):
                 restored_ep = st.session_state.ep_history.pop()
                 add_log_message(f"Undo successful. Restoring state with {len(restored_ep)} layers.")
                 add_log_message(f"Remaining Undo steps: {len(st.session_state.ep_history)}")
                 # Run calculation to plot the restored state
                 run_calculation_and_plot(ep_vector_to_use=restored_ep, is_optimized=True, method_name="Optimized (Undo)")
                 st.session_state.status_message = f"Status: Undo OK | Layers: {len(restored_ep)}"
     except Exception as e:
          err_msg = f"ERROR (Undo Operation): {type(e).__name__}: {e}"
          tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e)
          st.session_state.status_message = "Status: Undo Failed (Unexpected)"
          st.session_state.current_optimized_ep = None # Clear state on error maybe
          st.session_state.optimization_ran_since_nominal_change = False
          st.session_state.ep_history = deque(maxlen=5)
     finally:
          st.session_state.action_running = False
          st.rerun()


if st.session_state.get('status_message') == "Status: Setting Optimized as Nominal...":
     # --- Set Optimized as Nominal Implementation ---
     try:
         if not st.session_state.get('optimization_ran_since_nominal_change') or st.session_state.get('current_optimized_ep') is None:
              raise ValueError("No optimized structure available to set as nominal.")

         with st.spinner("Calculating QWOT and updating nominal..."):
              validated_inputs = get_validated_input_params()
              if not validated_inputs: raise ValueError("Invalid input parameters.")
              optimized_ep = st.session_state.current_optimized_ep
              l0 = validated_inputs['l0']
              nH_mat = validated_inputs['nH_material']
              nL_mat = validated_inputs['nL_material']

              optimized_qwots, logs_qwot = calculate_qwot_from_ep(optimized_ep, l0, nH_mat, nL_mat, EXCEL_FILE_PATH)
              add_log_message(logs_qwot)

              if np.any(np.isnan(optimized_qwots)):
                   add_log_message("Warning: QWOT calculation produced NaN. Cannot set nominal string."); st.warning("Cannot calculate valid QWOTs (NaN). Nominal QWOT field not updated.")
              else:
                   final_qwot_str = ",".join([f"{q:.6f}" for q in optimized_qwots]) # Use higher precision
                   st.session_state.nominal_qwot_str = final_qwot_str # Update the state variable
                   add_log_message("Nominal QWOT string updated in session state.")
                   st.success("Optimized structure set as new Nominal Structure (QWOT updated). Optimized state cleared.")

              # Clear the optimized state regardless of QWOT success
              st.session_state.current_optimized_ep = None
              st.session_state.current_material_sequence = None
              st.session_state.optimization_ran_since_nominal_change = False
              st.session_state.ep_history = deque(maxlen=5)
              add_log_message("Optimized state and Undo history cleared.")

              # Run calculation for the new nominal state
              run_calculation_and_plot(ep_vector_to_use=None, is_optimized=False, method_name="Nominal (Post Set)")
              st.session_state.status_message = "Status: Optimized set as Nominal"

     except (ValueError, RuntimeError, TypeError) as e:
          err_msg = f"ERROR (Set Nominal): {e}"; add_log_message(err_msg); st.error(err_msg)
          st.session_state.status_message = "Status: Set Nominal Failed"
     except Exception as e:
          err_msg = f"ERROR (Unexpected Set Nominal): {type(e).__name__}: {e}"; tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e)
          st.session_state.status_message = "Status: Set Nominal Failed (Unexpected)"
     finally:
          st.session_state.action_running = False
          st.rerun()


# --- File I/O ---
st.sidebar.divider()
st.sidebar.header("Load/Save Design")

# Load Design
uploaded_file = st.sidebar.file_uploader("Load Design (.json)", type="json", key="load_design_uploader")
if uploaded_file is not None and not st.session_state.action_running:
     st.session_state.action_running = True # Prevent other actions during load
     add_log_message(f"\n--- Loading Design from: {uploaded_file.name} ---")
     st.session_state.status_message = "Status: Loading Design..."
     st.rerun()

if st.session_state.get('status_message') == "Status: Loading Design...":
     # --- Load Implementation ---
     load_error = False
     try:
          design_data = json.load(uploaded_file)
          if 'params' not in design_data or 'targets' not in design_data:
               raise ValueError("Invalid JSON: missing 'params' or 'targets'.")

          # Load Params
          loaded_params = design_data.get('params', {})
          param_map = {'l0': 'l0_qwot', 'l_step': 'lambda_step', 'maxiter': 'max_iter', 'maxfun': 'max_fun',
                       'emp_str': 'nominal_qwot_str', 'initial_layer_number': 'initial_layer_number',
                       'auto_thin_threshold': 'auto_thin_threshold'}
          for key, value in loaded_params.items():
               state_key = param_map.get(key)
               if state_key and state_key in st.session_state:
                    # Ensure correct type if possible
                    current_type = type(st.session_state[state_key])
                    try: st.session_state[state_key] = current_type(value)
                    except: st.session_state[state_key] = value # Fallback to loaded value type
               # Also handle direct constant n/k values if present (older format?)
               elif key in ['nH_r', 'nH_i', 'nL_r', 'nL_i', 'nSub_r'] and key in st.session_state:
                   try: st.session_state[key] = float(value)
                   except: pass # Ignore if conversion fails


          # Load Materials
          def set_material_state(role, loaded_val):
              state_key = f'selected_{role}_material'
              constant_r_key = f'n{role}_r'
              constant_i_key = f'n{role}_i' if role != 'Sub' else None

              options = available_materials if role != 'Sub' else available_substrates
              if isinstance(loaded_val, str) and loaded_val in options:
                   st.session_state[state_key] = loaded_val
              elif isinstance(loaded_val, (dict, list, int, float)): # Handle constant definition
                   try:
                        if isinstance(loaded_val, dict):
                             n_r = float(loaded_val.get('real', 1.0))
                             n_i = float(loaded_val.get('imag', 0.0))
                        elif isinstance(loaded_val, list) and len(loaded_val) >= 1:
                             n_r = float(loaded_val[0])
                             n_i = float(loaded_val[1]) if len(loaded_val) > 1 else 0.0
                        else: # Assume float/int is real part only
                             n_r = float(loaded_val)
                             n_i = 0.0
                        st.session_state[state_key] = "Constant"
                        if constant_r_key in st.session_state: st.session_state[constant_r_key] = n_r
                        if constant_i_key and constant_i_key in st.session_state: st.session_state[constant_i_key] = n_i
                   except Exception as e:
                        add_log_message(f"Warning: Could not parse constant {role} material '{loaded_val}': {e}. Defaulting.")
                        st.session_state[state_key] = "Constant" # Default to constant on error
              else: # Fallback
                   st.session_state[state_key] = "Constant"

          set_material_state('H', loaded_params.get('nH_material', 'Constant'))
          set_material_state('L', loaded_params.get('nL_material', 'Constant'))
          set_material_state('Sub', loaded_params.get('nSub_material', 'Constant'))

          # Load Targets
          loaded_targets_raw = design_data.get('targets', [])
          new_targets_state = []
          for i in range(num_target_rows): # Load up to the number of UI rows
              if i < len(loaded_targets_raw):
                   t_data_raw = loaded_targets_raw[i]
                   if isinstance(t_data_raw, dict):
                        new_targets_state.append({
                            'id': i,
                            'enabled': bool(t_data_raw.get('enabled', False)),
                            'min': float(t_data_raw.get('min', 0.0)),
                            'max': float(t_data_raw.get('max', 0.0)),
                            'target_min': float(t_data_raw.get('target_min', 0.0)),
                            'target_max': float(t_data_raw.get('target_max', 0.0)),
                        })
                   else: # Add default if format is wrong
                        new_targets_state.append({'id': i, 'enabled': False, 'min': 0.0, 'max': 0.0, 'target_min': 0.0, 'target_max': 0.0})
              else: # Pad with default disabled targets if save file has fewer
                  new_targets_state.append({'id': i, 'enabled': False, 'min': 0.0, 'max': 0.0, 'target_min': 0.0, 'target_max': 0.0})
          st.session_state.targets = new_targets_state

          # Load Optimized State (if present)
          st.session_state.current_optimized_ep = None
          st.session_state.current_material_sequence = None
          st.session_state.optimization_ran_since_nominal_change = False
          st.session_state.ep_history = deque(maxlen=5) # Clear history on load

          if 'optimized_ep' in design_data and isinstance(design_data['optimized_ep'], list) and design_data['optimized_ep']:
               try:
                    loaded_ep = np.array(design_data['optimized_ep'], dtype=np.float64)
                    if np.all(np.isfinite(loaded_ep)) and np.all(loaded_ep >= 0):
                         st.session_state.current_optimized_ep = loaded_ep
                         st.session_state.optimization_ran_since_nominal_change = True
                         add_log_message(f"Loaded previous optimized state ({len(loaded_ep)} layers).")
                         # Load sequence if present
                         if 'optimized_material_sequence' in design_data and isinstance(design_data['optimized_material_sequence'], list):
                             if len(design_data['optimized_material_sequence']) == len(loaded_ep):
                                  st.session_state.current_material_sequence = design_data['optimized_material_sequence']
                                  add_log_message("Loaded optimized material sequence.")
                             else:
                                  add_log_message("Warning: Optimized material sequence length mismatch. Ignoring.")
                    else: raise ValueError("Invalid values in optimized_ep.")
               except Exception as e_ep:
                    add_log_message(f"WARNING: Could not load optimized_ep: {e_ep}. State set to Nominal.")
                    st.warning(f"Could not load optimized data: {e_ep}")

          add_log_message("Design loaded. Recalculating...")
          st.toast(f"Design loaded from {uploaded_file.name}")
          st.session_state.status_message = "Status: Design Loaded, Recalculating..."

     except (ValueError, json.JSONDecodeError) as e:
          err_msg = f"ERROR loading design file: {e}"; add_log_message(err_msg); st.error(err_msg)
          st.session_state.status_message = "Status: Load Design Failed"
          load_error = True
     except Exception as e:
          err_msg = f"ERROR (Unexpected Load): {type(e).__name__}: {e}"; tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e)
          st.session_state.status_message = "Status: Load Design Failed (Unexpected)"
          load_error = True
     finally:
          st.session_state.action_running = False
          if not load_error:
               # Trigger recalculation after successful load
               ep_to_calc = st.session_state.current_optimized_ep if st.session_state.optimization_ran_since_nominal_change else None
               seq_to_calc = st.session_state.current_material_sequence if st.session_state.optimization_ran_since_nominal_change else None
               run_calculation_and_plot(
                    ep_vector_to_use=ep_to_calc,
                    is_optimized=st.session_state.optimization_ran_since_nominal_change,
                    method_name="Loaded",
                    material_sequence=seq_to_calc
               )
          st.rerun() # Rerun to reflect loaded state


# Save Design Helper
def _collect_design_data_for_save(include_optimized: bool = False) -> Dict:
    design_data = {'params': {}, 'targets': []}
    params = get_validated_input_params() # Get current params
    if not params: raise ValueError("Cannot save: Invalid input parameters.")

    # Save core parameters
    param_keys_to_save = ['l0', 'l_step', 'maxiter', 'maxfun', 'initial_layer_number', 'auto_thin_threshold']
    for key in param_keys_to_save:
         if key in params: design_data['params'][key] = params[key]
    design_data['params']['emp_str'] = st.session_state.get('nominal_qwot_str', '') # Save the string directly

    # Save materials
    design_data['params']['nH_material'] = st.session_state.selected_H_material
    design_data['params']['nL_material'] = st.session_state.selected_L_material
    design_data['params']['nSub_material'] = st.session_state.selected_Sub_material
    if st.session_state.selected_H_material == "Constant":
         design_data['params']['nH_constant'] = {'real': st.session_state.nH_r, 'imag': st.session_state.nH_i}
    if st.session_state.selected_L_material == "Constant":
         design_data['params']['nL_constant'] = {'real': st.session_state.nL_r, 'imag': st.session_state.nL_i}
    if st.session_state.selected_Sub_material == "Constant":
         design_data['params']['nSub_constant'] = {'real': st.session_state.nSub_r, 'imag': 0.0}

    # Save Targets
    targets_in_state = st.session_state.get('targets', [])
    for target_state in targets_in_state:
         design_data['targets'].append(target_state.copy()) # Save a copy

    # Save Optimized State if requested and available
    if include_optimized:
        if st.session_state.get('optimization_ran_since_nominal_change') and st.session_state.get('current_optimized_ep') is not None:
             design_data['optimized_ep'] = st.session_state.current_optimized_ep.tolist()
             if st.session_state.get('current_material_sequence'):
                  design_data['optimized_material_sequence'] = st.session_state.current_material_sequence
             # Recalculate QWOT string for saving based on current params
             try:
                  opt_ep = st.session_state.current_optimized_ep
                  l0_save = params['l0']
                  nH_mat_save = params['nH_material']
                  nL_mat_save = params['nL_material']
                  qwots_save, _ = calculate_qwot_from_ep(opt_ep, l0_save, nH_mat_save, nL_mat_save, EXCEL_FILE_PATH)
                  if np.any(np.isnan(qwots_save)): opt_qwot_str_save = "QWOT N/A"
                  else: opt_qwot_str_save = ", ".join([f"{q:.6f}" for q in qwots_save]) # Higher precision for save
                  design_data['optimized_qwot_string'] = opt_qwot_str_save
             except Exception:
                  design_data['optimized_qwot_string'] = "Error calculating QWOT"
        else:
             add_log_message("Warning: Save Optimized requested, but no optimized state found.")

    return design_data

# Save Nominal Button
save_nominal_data = False
if st.sidebar.button("Save Nominal Design", key="save_nom_btn", use_container_width=True):
    try:
        design_data_nom = _collect_design_data_for_save(include_optimized=False)
        save_nominal_data = json.dumps(design_data_nom, indent=4).encode('utf-8')
        add_log_message("Prepared nominal design data for download.")
    except Exception as e:
        add_log_message(f"ERROR preparing nominal design data for save: {e}"); st.sidebar.error(f"Save Error: {e}")

if save_nominal_data:
     st.sidebar.download_button(
          label="Download Nominal Design (.json)",
          data=save_nominal_data,
          file_name=f"nominal_design_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json",
          mime="application/json",
          use_container_width=True
     )

# Save Optimized Button
save_optimized_data = False
save_opt_disabled = not (st.session_state.get('optimization_ran_since_nominal_change') and st.session_state.get('current_optimized_ep') is not None)
if st.sidebar.button("Save Optimized Design", key="save_opt_btn", disabled=save_opt_disabled, use_container_width=True):
     try:
          design_data_opt = _collect_design_data_for_save(include_optimized=True)
          save_optimized_data = json.dumps(design_data_opt, indent=4).encode('utf-8')
          add_log_message("Prepared optimized design data for download.")
     except Exception as e:
          add_log_message(f"ERROR preparing optimized design data for save: {e}"); st.sidebar.error(f"Save Error: {e}")

if save_optimized_data:
     st.sidebar.download_button(
          label="Download Optimized Design (.json)",
          data=save_optimized_data,
          file_name=f"optimized_design_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json",
          mime="application/json",
          use_container_width=True
     )

# Continuation from the previous ~1000 lines.

# --- Logic to Execute Within Button Handlers ---

# --- Evaluate Button ---
if st.session_state.get('status_message') == "Status: Evaluate Clicked (Implement Logic)":
    run_calculation_and_plot(ep_vector_to_use=None, is_optimized=False, method_name="Evaluate")
    st.session_state.action_running = False
    st.rerun()

# --- Local Optimizer Button ---
if st.session_state.get('status_message') == "Status: Starting Local Optimization...":
    final_ep_result = None
    try:
        with st.spinner("Running Local Optimization..."):
            validated_inputs = get_validated_input_params()
            if not validated_inputs: raise ValueError("Invalid input parameters.")
            active_targets = get_validated_active_targets_from_state()
            if active_targets is None: raise ValueError("Invalid targets.")
            if not active_targets: raise ValueError("Local optimization requires active targets.")

            ep_start = None
            if st.session_state.get('optimization_ran_since_nominal_change') and st.session_state.get('current_optimized_ep') is not None:
                add_log_message("Using current optimized structure as starting point.")
                ep_start = np.asarray(st.session_state.current_optimized_ep).copy()
            else:
                add_log_message("Using nominal structure (QWOT) as starting point.")
                ep_start_calc, logs_init_ep = calculate_initial_ep(
                    [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()],
                    validated_inputs['l0'],
                    validated_inputs['nH_material'], validated_inputs['nL_material'], EXCEL_FILE_PATH)
                add_log_message(logs_init_ep)
                if ep_start_calc is None: raise ValueError("Could not calculate initial thickness from QWOT.")
                ep_start = np.asarray(ep_start_calc)

            if ep_start is None or ep_start.size == 0: raise ValueError("Cannot determine starting structure.")
            ep_start = np.maximum(ep_start, MIN_THICKNESS_PHYS_NM)
            add_log_message(f"Starting optimization with {len(ep_start)} layers.")

            final_ep_result, optim_success, final_cost, optim_logs, optim_status_msg, total_nit, total_nfev = \
                _run_core_optimization(ep_start, validated_inputs, active_targets,
                                       MIN_THICKNESS_PHYS_NM, log_prefix="  [Local Opt] ")
            add_log_message(optim_logs)

            if not optim_success:
                raise RuntimeError(f"Local optimization failed. Msg: {optim_status_msg}, Cost: {final_cost:.3e}")

            add_log_message(f"Local Optimization finished successfully. Cost: {final_cost:.3e}, Iter/Eval: {total_nit}/{total_nfev}")
            # Trigger plot update with the successful result
            run_calculation_and_plot(ep_vector_to_use=final_ep_result, is_optimized=True, method_name=f"Local Opt (Cost:{final_cost:.3e})")

    except (ValueError, RuntimeError, TypeError) as e:
        err_msg = f"ERROR (Local Opt): {e}"; add_log_message(err_msg); st.error(err_msg)
        st.session_state.status_message = "Status: Local Opt Failed"
    except Exception as e:
        err_msg = f"ERROR (Unexpected Local Opt): {type(e).__name__}: {e}"; tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e)
        st.session_state.status_message = "Status: Local Opt Failed (Unexpected)"
    finally:
        st.session_state.action_running = False
        st.rerun()

# --- Scan+Opt Button ---
if st.session_state.get('status_message') == "Status: Starting Scan+Opt...":
    final_ep_result = None
    try:
        start_time_scan_opt = time.time()
        with st.spinner("Running QWOT Scan + Optimization... This may take time."):
            validated_inputs = get_validated_input_params()
            if not validated_inputs: raise ValueError("Invalid input parameters.")
            initial_layer_number = validated_inputs.get('initial_layer_number')
            if initial_layer_number is None or initial_layer_number <= 0: raise ValueError("Initial Layer Number (for QWOT Scan) must be > 0.")

            active_targets = get_validated_active_targets_from_state()
            if active_targets is None: raise ValueError("Invalid targets.")
            if not active_targets: raise ValueError("Scan+Opt requires active targets.")

            l0_nominal_gui = validated_inputs['l0']
            nH_material = validated_inputs['nH_material']; nL_material = validated_inputs['nL_material']
            nSub_material = validated_inputs['nSub_material']
            active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)

            l_min_val, l_max_val = get_lambda_range_from_targets(active_targets)
            if l_min_val is None or l_max_val is None: raise ValueError("Cannot determine lambda range.")
            l_step_optim = validated_inputs['l_step']
            num_pts_eval_full = max(2, int(np.round((l_max_val - l_min_val) / l_step_optim)) + 1)
            l_vec_eval_full_np = np.geomspace(l_min_val, l_max_val, num_pts_eval_full)
            l_vec_eval_full_np = l_vec_eval_full_np[(l_vec_eval_full_np > 0) & np.isfinite(l_vec_eval_full_np)]
            l_vec_eval_sparse_np = l_vec_eval_full_np[::5] # Use sparser grid
            if not l_vec_eval_sparse_np.size: raise ValueError("Failed to generate sparse lambda vector for scan.")
            l_vec_eval_sparse_jax = jnp.asarray(l_vec_eval_sparse_np)
            add_log_message(f"Scan evaluation grid: {len(l_vec_eval_sparse_jax)} pts.")

            l0_values_to_test = sorted(list(set([l0_nominal_gui, l0_nominal_gui * 1.15, l0_nominal_gui * 0.85])))
            l0_values_to_test = [l for l in l0_values_to_test if l > 1e-6]
            num_l0_tests = len(l0_values_to_test)
            num_combinations = 2**initial_layer_number
            total_evals_scan = num_combinations * num_l0_tests
            add_log_message(f"Starting Exhaustive QWOT Scan N={initial_layer_number}, {num_l0_tests} l0s: {[f'{l:.1f}' for l in l0_values_to_test]}. Combinations/l0: {num_combinations:,}.")

            initial_candidates = []
            nSub_arr_scan, logs_sub = _get_nk_array_for_lambda_vec(nSub_material, l_vec_eval_sparse_jax, EXCEL_FILE_PATH); add_log_message(logs_sub)

            for l0_idx, current_l0 in enumerate(l0_values_to_test):
                add_log_message(f"\n--- Scan l0 = {current_l0:.2f} nm ({l0_idx+1}/{num_l0_tests}) ---")
                try:
                    nH_c_l0, logs_h = _get_nk_at_lambda(nH_material, current_l0, EXCEL_FILE_PATH); add_log_message(logs_h)
                    nL_c_l0, logs_l = _get_nk_at_lambda(nL_material, current_l0, EXCEL_FILE_PATH); add_log_message(logs_l)
                except Exception as e:
                    add_log_message(f"ERROR getting indices at l0={current_l0:.2f} ({e}). Skipping."); continue

                current_best_mse_scan, current_best_multipliers_scan, scan_logs = _execute_split_stack_scan(
                    current_l0, initial_layer_number, jnp.asarray(nH_c_l0), jnp.asarray(nL_c_l0),
                    nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple )
                add_log_message(scan_logs)
                if np.isfinite(current_best_mse_scan) and current_best_multipliers_scan is not None:
                    add_log_message(f"Scan candidate found. MSE {current_best_mse_scan:.6e}.")
                    initial_candidates.append({'l0': current_l0, 'mse_scan': current_best_mse_scan, 'multipliers': np.array(current_best_multipliers_scan)})

            if not initial_candidates: raise RuntimeError("QWOT Scan found no valid initial candidates.")
            add_log_message(f"\n--- QWOT Scan finished. Found {len(initial_candidates)} candidate(s). Optimizing... ---")
            initial_candidates.sort(key=lambda c: c['mse_scan'])

            final_best_ep = None; final_best_mse = np.inf; final_best_l0 = None; final_best_initial_multipliers = None
            overall_optim_nit = 0; overall_optim_nfev = 0; successful_optim_count = 0

            for idx, candidate in enumerate(initial_candidates):
                 cand_l0 = candidate['l0']; cand_mult = candidate['multipliers']; cand_mse_scan = candidate['mse_scan']
                 add_log_message(f"\n--- Optimizing Candidate {idx+1}/{len(initial_candidates)} (l0={cand_l0:.2f}, scan MSE={cand_mse_scan:.6e}) ---")
                 try:
                     ep_start_optim, logs_ep = calculate_initial_ep(cand_mult, cand_l0, nH_material, nL_material, EXCEL_FILE_PATH); add_log_message(logs_ep)
                     ep_start_optim = np.maximum(ep_start_optim, MIN_THICKNESS_PHYS_NM)
                     add_log_message(f"Starting local opt from {len(ep_start_optim)} layers.")
                     result_ep_optim, optim_success, final_cost_optim, optim_logs, _, nit_optim, nfev_optim = \
                         _run_core_optimization(ep_start_optim, validated_inputs, active_targets, MIN_THICKNESS_PHYS_NM, log_prefix=f"  [Opt Cand {idx+1}] ")
                     add_log_message(optim_logs)
                     if optim_success:
                          successful_optim_count += 1; overall_optim_nit += nit_optim; overall_optim_nfev += nfev_optim
                          add_log_message(f"Opt successful. MSE: {final_cost_optim:.6e}")
                          if final_cost_optim < final_best_mse:
                               add_log_message(f"*** New global best! MSE {final_best_mse:.6e} -> {final_cost_optim:.6e} ***")
                               final_best_mse = final_cost_optim; final_best_ep = result_ep_optim.copy();
                               final_best_l0 = cand_l0; final_best_initial_multipliers = cand_mult
                 except Exception as e_optim_cand:
                      add_log_message(f"ERROR optimizing candidate {idx+1}: {e_optim_cand}")

            if final_best_ep is None: raise RuntimeError("Local optimization failed for all scan candidates.")

            add_log_message(f"\n--- Scan+Opt Best Result ---")
            add_log_message(f"Final MSE: {final_best_mse:.6e} ({len(final_best_ep)} layers), L0: {final_best_l0:.2f} nm")
            best_mult_list_str = ",".join([f"{m:.3f}" for m in final_best_initial_multipliers])
            add_log_message(f"Original Multipliers: {best_mult_list_str}")

            if abs(final_best_l0 - l0_nominal_gui) > 1e-3:
                 add_log_message(f"Updating GUI l0 -> {final_best_l0:.2f}")
                 st.session_state.l0_qwot = final_best_l0

            final_ep_result = final_best_ep # Store result to plot outside spinner
            # Update state immediately for plotting
            st.session_state.current_optimized_ep = final_ep_result.copy()
            st.session_state.current_material_sequence = None
            st.session_state.optimization_ran_since_nominal_change = True


    except (ValueError, RuntimeError, TypeError) as e:
        err_msg = f"ERROR (Scan+Opt): {e}"; add_log_message(err_msg); st.error(err_msg)
        st.session_state.status_message = "Status: Scan+Opt Failed"
    except Exception as e:
        err_msg = f"ERROR (Unexpected Scan+Opt): {type(e).__name__}: {e}"; tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e)
        st.session_state.status_message = "Status: Scan+Opt Failed (Unexpected)"
    finally:
        st.session_state.action_running = False
        if final_ep_result is not None:
            run_calculation_and_plot(ep_vector_to_use=final_ep_result, is_optimized=True, method_name=f"Scan+Opt (L0={final_best_l0:.1f})")
        st.rerun()

# --- Auto Mode Button ---
if st.session_state.get('status_message') == "Status: Starting Auto Mode...":
    final_ep_result = None
    try:
        start_time_auto = time.time()
        best_ep_so_far = None; best_mse_so_far = np.inf; num_cycles_done = 0
        termination_reason = f"Max {AUTO_MAX_CYCLES} cycles reached"
        total_iters_auto = 0; total_evals_auto = 0; optim_runs_auto = 0

        with st.spinner("Running Auto Mode... This may take several minutes."):
            validated_inputs = get_validated_input_params()
            if not validated_inputs: raise ValueError("Invalid input parameters.")
            active_targets = get_validated_active_targets_from_state()
            if active_targets is None: raise ValueError("Invalid targets.")
            if not active_targets: raise ValueError("Auto mode requires active targets.")

            nH_material = validated_inputs['nH_material']; nL_material = validated_inputs['nL_material']
            nSub_material = validated_inputs['nSub_material']; l0 = validated_inputs['l0']
            threshold_from_gui = validated_inputs.get('auto_thin_threshold', 1.0)
            add_log_message(f"Auto Thin Removal Threshold: {threshold_from_gui:.3f} nm")

            l_min_val, l_max_val = get_lambda_range_from_targets(active_targets)
            if l_min_val is None or l_max_val is None: raise ValueError("Cannot determine lambda range.")
            l_step_optim = validated_inputs['l_step']
            num_pts = max(2, int(np.round((l_max_val - l_min_val) / l_step_optim)) + 1)
            l_vec_optim_np = np.geomspace(l_min_val, l_max_val, num_pts)
            l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
            if not l_vec_optim_np.size: raise ValueError("Failed to generate lambda vector for Auto Mode.")

            ep_start_auto = None
            if st.session_state.get('optimization_ran_since_nominal_change') and st.session_state.get('current_optimized_ep') is not None:
                 add_log_message("Auto Mode: Using existing optimized structure as start.")
                 ep_start_auto = np.asarray(st.session_state.current_optimized_ep).copy()
                 try: # Calculate initial MSE
                     l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
                     nH_arr, _ = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH)
                     nL_arr, _ = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH)
                     nSub_arr, _ = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH)
                     active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)
                     static_args_cost_fn = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, jnp.float64(MIN_THICKNESS_PHYS_NM))
                     cost_fn_compiled = jax.jit(calculate_mse_for_optimization_penalized_jax)
                     initial_cost_jax = cost_fn_compiled(jnp.asarray(ep_start_auto), *static_args_cost_fn)
                     best_mse_so_far = float(np.array(initial_cost_jax))
                     if not np.isfinite(best_mse_so_far): raise ValueError("Initial MSE not finite")
                 except Exception as e: raise ValueError(f"Cannot calculate initial MSE: {e}")
            else: # Start from nominal + initial optimization
                add_log_message("Auto Mode: Using nominal structure (QWOT) as start + initial opt.")
                ep_nominal_calc, logs_init_ep = calculate_initial_ep([float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()], l0, nH_material, nL_material, EXCEL_FILE_PATH)
                add_log_message(logs_init_ep)
                if ep_nominal_calc is None or ep_nominal_calc.size == 0: raise ValueError("Cannot get nominal starting structure.")
                ep_nominal = np.maximum(np.asarray(ep_nominal_calc), MIN_THICKNESS_PHYS_NM)
                ep_after_initial_opt, initial_opt_success, initial_mse, initial_opt_logs, initial_opt_msg, initial_nit, initial_nfev = \
                    _run_core_optimization(ep_nominal, validated_inputs, active_targets, MIN_THICKNESS_PHYS_NM, log_prefix="  [Auto Init Opt] ")
                add_log_message(initial_opt_logs)
                if not initial_opt_success: raise RuntimeError(f"Initial optimization failed: {initial_opt_msg}")
                ep_start_auto = ep_after_initial_opt.copy()
                best_mse_so_far = initial_mse
                total_iters_auto += initial_nit; total_evals_auto += initial_nfev; optim_runs_auto += 1

            best_ep_so_far = ep_start_auto.copy()
            if not np.isfinite(best_mse_so_far): raise ValueError("Starting MSE for cycles not finite.")
            add_log_message(f"Starting Auto Cycles. Initial MSE: {best_mse_so_far:.6e} ({len(best_ep_so_far)} layers)")

            # --- Cycle Loop ---
            for cycle_num in range(AUTO_MAX_CYCLES):
                add_log_message(f"\n--- Auto Cycle {cycle_num + 1} / {AUTO_MAX_CYCLES} ---")
                mse_at_cycle_start = best_mse_so_far
                ep_at_cycle_start = best_ep_so_far.copy()
                cycle_improved_overall = False

                # 1. Needle Phase
                add_log_message(f" [Cycle {cycle_num+1}] Needles ({AUTO_NEEDLES_PER_CYCLE}x)...")
                ep_after_needles, mse_after_needles, needle_logs, nit_needles, nfev_needles, reopts_in_needles = \
                    _run_needle_iterations(best_ep_so_far, AUTO_NEEDLES_PER_CYCLE, validated_inputs, active_targets, MIN_THICKNESS_PHYS_NM, l_vec_optim_np, DEFAULT_NEEDLE_SCAN_STEP_NM, BASE_NEEDLE_THICKNESS_NM, calculate_mse_for_optimization_penalized_jax, log_prefix=f"    [Needle {cycle_num+1}] ")
                add_log_message(needle_logs)
                total_iters_auto += nit_needles; total_evals_auto += nfev_needles; optim_runs_auto += reopts_in_needles
                if mse_after_needles < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                    best_ep_so_far = ep_after_needles.copy(); best_mse_so_far = mse_after_needles; cycle_improved_overall = True
                else: best_ep_so_far = ep_after_needles.copy(); best_mse_so_far = mse_after_needles

                # 2. Thin Removal Phase
                add_log_message(f" [Cycle {cycle_num+1}] Thin Removal (Thr: {threshold_from_gui:.3f} nm)...")
                layers_removed_this_cycle = 0; thinning_loop_iteration = 0; max_thinning_iterations = len(best_ep_so_far) + 2
                while thinning_loop_iteration < max_thinning_iterations:
                     thinning_loop_iteration += 1
                     if len(best_ep_so_far) <= 2: break
                     ep_after_single_removal, structure_changed, removal_logs = _perform_layer_merge_or_removal_only(best_ep_so_far, MIN_THICKNESS_PHYS_NM, log_prefix=f"      [Remove {thinning_loop_iteration}]", threshold_for_removal=threshold_from_gui)
                     add_log_message(removal_logs)
                     if structure_changed:
                          layers_removed_this_cycle += 1
                          add_log_message(f"    Layer removed. Re-optimizing {len(ep_after_single_removal)} layers...")
                          ep_after_thin_reopt, thin_reopt_success, mse_after_thin_reopt, thin_reopt_logs, thin_reopt_msg, nit_thin_reopt, nfev_thin_reopt = \
                               _run_core_optimization(ep_after_single_removal, validated_inputs, active_targets, MIN_THICKNESS_PHYS_NM, log_prefix=f"        [ReOpt {layers_removed_this_cycle}] ")
                          add_log_message(thin_reopt_logs)
                          if thin_reopt_success:
                               total_iters_auto += nit_thin_reopt; total_evals_auto += nfev_thin_reopt; optim_runs_auto += 1
                               if mse_after_thin_reopt < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                                    best_ep_so_far = ep_after_thin_reopt.copy(); best_mse_so_far = mse_after_thin_reopt; cycle_improved_overall = True
                               else: best_ep_so_far = ep_after_thin_reopt.copy(); best_mse_so_far = mse_after_thin_reopt
                          else: # Re-opt failed
                               add_log_message(f"    WARNING: Re-opt failed ({thin_reopt_msg}). Stopping removal phase."); best_ep_so_far = ep_after_single_removal.copy()
                               try: # Recalculate MSE
                                   cost_fn_compiled = jax.jit(calculate_mse_for_optimization_penalized_jax); static_args_cost_fn = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, jnp.float64(MIN_THICKNESS_PHYS_NM))
                                   failed_cost_jax = cost_fn_compiled(jnp.asarray(best_ep_so_far), *static_args_cost_fn); best_mse_so_far = float(np.array(failed_cost_jax));
                                   if not np.isfinite(best_mse_so_far): best_mse_so_far = np.inf
                               except Exception: best_mse_so_far = np.inf
                               cycle_improved_overall = (best_mse_so_far < mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE) or cycle_improved_overall
                               break
                     else: break # No more layers below threshold
                add_log_message(f" [Cycle {cycle_num+1}] Thin Removal Phase finished. {layers_removed_this_cycle} layer(s) removed.")
                num_cycles_done += 1

                # 3. Check for Termination
                add_log_message(f"--- End Auto Cycle {cycle_num + 1}. Best MSE: {best_mse_so_far:.6e} ({len(best_ep_so_far)} layers)---")
                if not cycle_improved_overall and best_mse_so_far >= mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE:
                    add_log_message(f"No significant improvement in Cycle {cycle_num + 1}. Stopping Auto Mode.")
                    termination_reason = "No improvement"
                    if best_mse_so_far > mse_at_cycle_start + MSE_IMPROVEMENT_TOLERANCE :
                         add_log_message(f"Reverting to state before Cycle {cycle_num + 1} (MSE {mse_at_cycle_start:.6e}).")
                         best_ep_so_far = ep_at_cycle_start; best_mse_so_far = mse_at_cycle_start
                    break # Exit cycle loop

            final_ep_result = best_ep_so_far # Store final result

    except (ValueError, RuntimeError, TypeError) as e:
        err_msg = f"ERROR (Auto Mode): {e}"; add_log_message(err_msg); st.error(err_msg)
        st.session_state.status_message = "Status: Auto Mode Failed"
    except Exception as e:
        err_msg = f"ERROR (Unexpected Auto Mode): {type(e).__name__}: {e}"; tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e)
        st.session_state.status_message = "Status: Auto Mode Failed (Unexpected)"
    finally:
        st.session_state.action_running = False
        if final_ep_result is not None:
            add_log_message(f"\n--- Auto Mode Finished ({num_cycles_done} cycles, {termination_reason}) ---")
            add_log_message(f"Final Best MSE: {best_mse_so_far:.6e} ({len(final_ep_result)} layers)")
            run_calculation_and_plot(ep_vector_to_use=final_ep_result, is_optimized=True, method_name=f"Auto Mode ({num_cycles_done} cyc)")
        st.rerun()

# --- Remove Thin Layer Button ---
if st.session_state.get('status_message') == "Status: Starting Remove Thin Layer...":
    final_ep_result = None
    try:
        if not st.session_state.get('optimization_ran_since_nominal_change') or st.session_state.get('current_optimized_ep') is None:
            raise ValueError("Remove Thin Layer requires a valid optimized structure.")
        current_ep_opt = np.asarray(st.session_state.current_optimized_ep)
        if len(current_ep_opt) <= 2: raise ValueError("Structure <= 2 layers. Cannot remove.")

        with st.spinner("Removing thin layer and re-optimizing..."):
             validated_inputs = get_validated_input_params(); active_targets = get_validated_active_targets_from_state()
             if not validated_inputs or active_targets is None or not active_targets: raise ValueError("Invalid inputs/targets for re-optimization.")

             st.session_state.ep_history.append(current_ep_opt.copy())
             add_log_message(f" [Undo] State saved. History size: {len(st.session_state.ep_history)}")
             threshold_rem = validated_inputs.get('auto_thin_threshold', None)

             ep_after_removal, structure_changed, removal_logs = _perform_layer_merge_or_removal_only(current_ep_opt, MIN_THICKNESS_PHYS_NM, log_prefix="  [Removal] ", threshold_for_removal=threshold_rem)
             add_log_message(removal_logs)

             if not structure_changed:
                 add_log_message("No layer removed."); st.info("Could not remove a layer (criteria not met).")
                 st.session_state.status_message = f"Status: Removal Skipped"; final_ep_result = current_ep_opt
                 if st.session_state.ep_history: try: st.session_state.ep_history.pop(); add_log_message(" [Undo] Unneeded history removed.") except IndexError: pass
                 run_calculation_and_plot(ep_vector_to_use=final_ep_result, is_optimized=True, method_name="Optimized (Removal Skipped)")
             else:
                 add_log_message(f"Structure changed to {len(ep_after_removal)} layers. Re-optimizing...")
                 ep_after_reopt, reopt_success, final_cost, reopt_logs, reopt_status_msg, _, _ = \
                       _run_core_optimization(ep_after_removal, validated_inputs, active_targets, MIN_THICKNESS_PHYS_NM, log_prefix="  [Re-Opt] ")
                 add_log_message(reopt_logs)
                 if not reopt_success:
                      add_log_message("ERROR: Re-opt failed."); st.warning(f"Re-opt failed: {reopt_status_msg}. Showing structure post-removal.")
                      final_ep_result = ep_after_removal.copy(); st.session_state.status_message = f"Status: Removed, Re-Opt Failed"
                 else:
                      add_log_message("Re-opt successful."); final_ep_result = ep_after_reopt.copy()
                      st.session_state.status_message = f"Status: Removed & Re-Opt OK | MSE: {final_cost:.3e}"
                 run_calculation_and_plot(ep_vector_to_use=final_ep_result, is_optimized=True, method_name=f"Optimized Post-Removal{' (Re-Opt Failed)' if not reopt_success else ''}")

    except (ValueError, RuntimeError, TypeError) as e:
        err_msg = f"ERROR (Remove Thin): {e}"; add_log_message(err_msg); st.error(err_msg); st.session_state.status_message = "Status: Remove Thin Failed"
    except Exception as e:
        err_msg = f"ERROR (Unexpected Remove Thin): {type(e).__name__}: {e}"; tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e); st.session_state.status_message = "Status: Remove Thin Failed (Unexpected)"
    finally:
        st.session_state.action_running = False
        st.rerun()

# --- Undo Remove Button ---
if st.session_state.get('status_message') == "Status: Undoing removal...":
    try:
        if not st.session_state.ep_history:
            add_log_message("Undo history empty."); st.info("Nothing to undo.")
            st.session_state.status_message = "Status: Ready" # Reset status
        else:
            with st.spinner("Restoring previous state..."):
                restored_ep = st.session_state.ep_history.pop()
                add_log_message(f"Undo successful. Restoring state ({len(restored_ep)} layers). Remaining undos: {len(st.session_state.ep_history)}")
                run_calculation_and_plot(ep_vector_to_use=restored_ep, is_optimized=True, method_name="Optimized (Undo)")
                st.session_state.status_message = f"Status: Undo OK | Layers: {len(restored_ep)}"
    except Exception as e:
        err_msg = f"ERROR (Undo): {type(e).__name__}: {e}"; tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e); st.session_state.status_message = "Status: Undo Failed (Unexpected)"
        st.session_state.current_optimized_ep = None; st.session_state.optimization_ran_since_nominal_change = False; st.session_state.ep_history = deque(maxlen=5)
    finally:
        st.session_state.action_running = False
        st.rerun()

# --- Set Optimized as Nominal Button ---
if st.session_state.get('status_message') == "Status: Setting Optimized as Nominal...":
    try:
        if not st.session_state.get('optimization_ran_since_nominal_change') or st.session_state.get('current_optimized_ep') is None:
             raise ValueError("No optimized structure available.")
        with st.spinner("Calculating QWOT and updating..."):
             validated_inputs = get_validated_input_params(); optimized_ep = st.session_state.current_optimized_ep
             if not validated_inputs: raise ValueError("Invalid parameters.")
             optimized_qwots, logs_qwot = calculate_qwot_from_ep(optimized_ep, validated_inputs['l0'], validated_inputs['nH_material'], validated_inputs['nL_material'], EXCEL_FILE_PATH)
             add_log_message(logs_qwot)
             if np.any(np.isnan(optimized_qwots)): st.warning("Calculated QWOTs contain NaN. Nominal string not updated.")
             else:
                  final_qwot_str = ",".join([f"{q:.6f}" for q in optimized_qwots]); st.session_state.nominal_qwot_str = final_qwot_str; add_log_message("Nominal QWOT updated.")
                  st.success("Optimized structure set as Nominal.")
             st.session_state.current_optimized_ep = None; st.session_state.current_material_sequence = None; st.session_state.optimization_ran_since_nominal_change = False; st.session_state.ep_history = deque(maxlen=5); add_log_message("Optimized state cleared.")
             run_calculation_and_plot(ep_vector_to_use=None, is_optimized=False, method_name="Nominal (Post Set)")
             st.session_state.status_message = "Status: Optimized set as Nominal"
    except (ValueError, RuntimeError, TypeError) as e:
        err_msg = f"ERROR (Set Nominal): {e}"; add_log_message(err_msg); st.error(err_msg); st.session_state.status_message = "Status: Set Nominal Failed"
    except Exception as e:
        err_msg = f"ERROR (Unexpected Set Nominal): {type(e).__name__}: {e}"; tb_msg = traceback.format_exc(); add_log_message(err_msg); add_log_message(tb_msg); st.exception(e); st.session_state.status_message = "Status: Set Nominal Failed (Unexpected)"
    finally:
        st.session_state.action_running = False
        st.rerun()

# --- Trigger Initial Calculation on First Load ---
if st.session_state.get('last_calc_results') is None and not st.session_state.action_running :
     add_log_message("Performing initial calculation on first load...")
     run_calculation_and_plot(ep_vector_to_use=None, is_optimized=False, method_name="Initial Load")
     st.rerun() # Rerun once after initial calculation to display plot

     
# Continuation from the previous response.

# --- Helper functions for QWOT Scan (_execute_split_stack_scan) ---

@jax.jit
def calculate_M_for_thickness(thickness: jnp.ndarray, n_complex_layer: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    # Calculates the 2x2 characteristic matrix for a single layer
    eta = n_complex_layer
    safe_l_val = jnp.maximum(l_val, 1e-9)
    safe_eta = jnp.where(jnp.abs(eta) < 1e-12, 1e-12 + 0j, eta) # Avoid division by zero if k is large negative or n is near zero
    phi = (2 * jnp.pi / safe_l_val) * (n_complex_layer * thickness)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    # Handle zero thickness case explicitly to return identity matrix
    m00 = jnp.where(thickness > 1e-12, cos_phi, 1.0)
    m01 = jnp.where(thickness > 1e-12, (1j / safe_eta) * sin_phi, 0.0)
    m10 = jnp.where(thickness > 1e-12, 1j * eta * sin_phi, 0.0)
    m11 = jnp.where(thickness > 1e-12, cos_phi, 1.0)

    M_layer = jnp.array([[m00, m01], [m10, m11]], dtype=jnp.complex128)
    return M_layer

# Vectorize the matrix calculation over wavelengths for a single thickness/index
calculate_M_batch_for_thickness = vmap(calculate_M_for_thickness, in_axes=(None, None, 0))


@jax.jit
def get_layer_matrices_qwot(layer_idx: int, initial_layer_number: int, l0: float,
                            nH_c_l0: jnp.ndarray, nL_c_l0: jnp.ndarray,
                            l_vec: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Precompute characteristic matrices for QWOT=1 and QWOT=2 for a specific layer index
    predicate_is_H = (layer_idx % 2 == 0)
    n_real_l0 = jax.lax.select(predicate_is_H, nH_c_l0.real, nL_c_l0.real)
    # Use the complex index at l0 for the calculation across all wavelengths (approximation)
    n_complex_for_matrix = jax.lax.select(predicate_is_H, nH_c_l0, nL_c_l0)

    # Calculate physical thicknesses for QWOT=1 and QWOT=2
    denom = 4.0 * jnp.maximum(n_real_l0, 1e-9) # Avoid division by zero
    safe_l0 = jnp.maximum(l0, 1e-9)
    ep1_calc = 1.0 * safe_l0 / denom
    ep2_calc = 2.0 * safe_l0 / denom

    # Ensure minimum thickness constraint
    ep1 = jnp.where(n_real_l0 > 1e-9, ep1_calc, 0.0)
    ep2 = jnp.where(n_real_l0 > 1e-9, ep2_calc, 0.0)
    ep1 = jnp.maximum(ep1, MIN_THICKNESS_PHYS_NM * (ep1 > 1e-12)) # Apply min only if > 0
    ep2 = jnp.maximum(ep2, MIN_THICKNESS_PHYS_NM * (ep2 > 1e-12)) # Apply min only if > 0


    # Calculate matrices for both thicknesses across all wavelengths in l_vec
    M_1qwot_batch = calculate_M_batch_for_thickness(ep1, n_complex_for_matrix, l_vec)
    M_2qwot_batch = calculate_M_batch_for_thickness(ep2, n_complex_for_matrix, l_vec)

    return M_1qwot_batch, M_2qwot_batch # Shape: (num_lambda, 2, 2)


@jax.jit
def compute_half_product(multiplier_indices: jnp.ndarray, # Shape (N_half,) - 0 or 1 for each layer
                         layer_matrices_half: jnp.ndarray # Shape (N_half, 2, num_lambda, 2, 2)
                         ) -> jnp.ndarray: # Return shape (num_lambda, 2, 2)
    # Computes the matrix product for one half of the stack based on multiplier choices
    N_half = layer_matrices_half.shape[0]
    L = layer_matrices_half.shape[2] # Number of lambdas

    # Initialize product matrices (identity for each lambda)
    init_prod = jnp.tile(jnp.eye(2, dtype=jnp.complex128), (L, 1, 1)) # Shape (num_lambda, 2, 2)

    # Define the step for jax.lax.scan
    def multiply_step(carry_prod: jnp.ndarray, # Current product (num_lambda, 2, 2)
                      layer_idx: int         # Index of the layer (0 to N_half-1)
                      ) -> Tuple[jnp.ndarray, None]:
        # Get the multiplier index (0 or 1) for this layer
        multiplier_idx = multiplier_indices[layer_idx]
        # Select the corresponding precomputed matrix batch (QWOT=1 or QWOT=2)
        # Shape: (num_lambda, 2, 2)
        M_k = layer_matrices_half[layer_idx, multiplier_idx, :, :, :]
        # Multiply M_k with the carry_prod for each lambda
        # Matrix multiplication order: M_k @ carry_prod
        new_prod = vmap(jnp.matmul)(M_k, carry_prod)
        return new_prod, None

    # Scan through layers 0 to N_half-1
    final_prod, _ = jax.lax.scan(multiply_step, init_prod, jnp.arange(N_half))
    return final_prod


@jax.jit
def get_T_from_batch_matrix(M_batch: jnp.ndarray, # Shape (num_lambda, 2, 2)
                            nSub_arr: jnp.ndarray  # Shape (num_lambda,)
                            ) -> jnp.ndarray:     # Return shape (num_lambda,)
    # Calculates Transmittance from a batch of total characteristic matrices
    etainc = 1.0 + 0j # Air
    etasub_batch = nSub_arr # Substrate index (complex) for each lambda

    # Extract matrix elements for all lambdas
    m00 = M_batch[:, 0, 0]; m01 = M_batch[:, 0, 1]
    m10 = M_batch[:, 1, 0]; m11 = M_batch[:, 1, 1]

    # Calculate denominator for transmission coefficient (ts)
    rs_den = (etainc * m00 + etasub_batch * m11 + etainc * etasub_batch * m01 + m10)
    rs_den_abs = jnp.abs(rs_den)
    safe_den = jnp.where(rs_den_abs < 1e-12, 1e-12 + 0j, rs_den) # Avoid division by zero

    # Transmission coefficient (amplitude)
    ts = (2.0 * etainc) / safe_den

    # Transmittance (intensity)
    real_etasub_batch = jnp.real(etasub_batch)
    real_etainc = 1.0
    Ts_complex = (real_etasub_batch / real_etainc) * (ts * jnp.conj(ts))

    # Return real part, handle potential division by zero during ts calculation
    Ts = jnp.real(Ts_complex)
    return jnp.where(rs_den_abs < 1e-12, 0.0, jnp.nan_to_num(Ts, nan=0.0))


@jax.jit
def calculate_mse_basic_jax(Ts: jnp.ndarray, l_vec: jnp.ndarray, targets_tuple: Tuple) -> jnp.ndarray:
    # Calculates basic MSE without penalties (used in QWOT scan)
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

    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, jnp.inf) # Use infinity if no points
    return jnp.nan_to_num(mse, nan=jnp.inf)


@jax.jit
def combine_and_calc_mse(prod1: jnp.ndarray,        # Shape (num_lambda, 2, 2) - Product for first half
                         prod2: jnp.ndarray,        # Shape (num_lambda, 2, 2) - Product for second half
                         nSub_arr_in: jnp.ndarray,  # Shape (num_lambda,)
                         l_vec_in: jnp.ndarray,     # Shape (num_lambda,)
                         targets_tuple_in: Tuple   # Targets
                         ) -> jnp.ndarray:          # Return scalar MSE
    # Combine products from two halves and calculate MSE
    # Matrix multiplication order: M_total = M_half2 @ M_half1
    M_total = vmap(jnp.matmul)(prod2, prod1) # vmap over the lambda dimension
    # Calculate Transmittance from the total matrix batch
    Ts = get_T_from_batch_matrix(M_total, nSub_arr_in)
    # Calculate MSE
    mse = calculate_mse_basic_jax(Ts, l_vec_in, targets_tuple_in)
    return mse

# --- Main QWOT Scan Execution Function ---
def _execute_split_stack_scan(current_l0: float, initial_layer_number: int,
                              nH_c_l0: jnp.ndarray, nL_c_l0: jnp.ndarray,
                              nSub_arr_scan: jnp.ndarray,
                              l_vec_eval_sparse_jax: jnp.ndarray,
                              active_targets_tuple: Tuple
                              ) -> Tuple[float, Optional[np.ndarray], List[str]]:
    # Performs the split-stack QWOT scan for a given l0
    logs = []
    num_combinations = 2**initial_layer_number
    logs.append(f"  [Scan l0={current_l0:.2f}] Testing {num_combinations:,} QWOT combinations (1.0 or 2.0)...")

    # 1. Precompute Layer Matrices
    precompute_start_time = time.time()
    layer_matrices_list = []
    try:
        for i in range(initial_layer_number):
            m1, m2 = get_layer_matrices_qwot(i, initial_layer_number, current_l0, nH_c_l0, nL_c_l0, l_vec_eval_sparse_jax)
            layer_matrices_list.append(jnp.stack([m1, m2], axis=0)) # Stack QWOT=1 and QWOT=2 matrices
        # Shape: (num_layers, 2, num_lambda, 2, 2)
        all_layer_matrices = jnp.stack(layer_matrices_list, axis=0)
        all_layer_matrices.block_until_ready() # Ensure computation finishes
    except Exception as e_mat:
        logs.append(f"  ERROR Precomputing Matrices for l0={current_l0:.2f}: {e_mat}")
        return np.inf, None, logs
    logs.append(f"    Matrix precomputation (l0={current_l0:.2f}) finished in {time.time() - precompute_start_time:.3f}s.")

    # 2. Split stack and compute partial products using vmap and scan
    N1 = initial_layer_number // 2; N2 = initial_layer_number - N1
    num_comb1 = 2**N1; num_comb2 = 2**N2
    indices1 = jnp.arange(num_comb1); indices2 = jnp.arange(num_comb2)
    powers1 = 2**jnp.arange(N1); powers2 = 2**jnp.arange(N2)

    # Generate indices (0 or 1) for each layer in each combination half
    # Shape: (num_comb_half, N_half)
    multiplier_indices1 = jnp.not_equal(indices1[:, None] & powers1, 0).astype(jnp.int32)
    multiplier_indices2 = jnp.not_equal(indices2[:, None] & powers2, 0).astype(jnp.int32)

    # Split the precomputed matrices
    matrices_half1 = all_layer_matrices[:N1]; matrices_half2 = all_layer_matrices[N1:]

    # Compute partial products for the first half
    half1_start_time = time.time()
    compute_half_product_jit = jax.jit(compute_half_product)
    # vmap over combinations for the first half
    # Shape: (num_comb1, num_lambda, 2, 2)
    partial_products1 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices1, matrices_half1)
    partial_products1.block_until_ready()
    logs.append(f"    Partial products 1/2 calculated in {time.time() - half1_start_time:.3f}s.")

    # Compute partial products for the second half
    half2_start_time = time.time()
    # vmap over combinations for the second half
    # Shape: (num_comb2, num_lambda, 2, 2)
    partial_products2 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices2, matrices_half2)
    partial_products2.block_until_ready()
    logs.append(f"    Partial products 2/2 calculated in {time.time() - half2_start_time:.3f}s.")

    # 3. Combine products and calculate MSE for all combinations
    combine_start_time = time.time()
    combine_and_calc_mse_jit = jax.jit(combine_and_calc_mse)
    # Vmap over combinations of half1 and half2
    vmap_inner = vmap(combine_and_calc_mse_jit, in_axes=(None, 0, None, None, None)) # Vmap over prod2 for fixed prod1
    vmap_outer = vmap(vmap_inner, in_axes=(0, None, None, None, None))     # Vmap over prod1
    # Shape: (num_comb1, num_comb2)
    all_mses_nested = vmap_outer(partial_products1, partial_products2, nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple)
    all_mses_nested.block_until_ready() # Ensure calculation is done
    logs.append(f"    Combination and MSE calculation finished in {time.time() - combine_start_time:.3f}s.")

    # 4. Find the best combination
    all_mses_flat = all_mses_nested.reshape(-1) # Flatten the MSE results
    best_idx_flat = jnp.argmin(all_mses_flat)   # Find index of minimum MSE
    current_best_mse = float(all_mses_flat[best_idx_flat]) # Get the best MSE value

    if not np.isfinite(current_best_mse):
        logs.append(f"    Warning: No valid combination found (all MSEs infinite/NaN).")
        return np.inf, None, logs

    # Find the indices corresponding to the best combination in each half
    best_idx_half1, best_idx_half2 = jnp.unravel_index(best_idx_flat, (num_comb1, num_comb2))

    # Retrieve the multiplier indices (0 or 1) for the best combination
    best_indices_h1 = multiplier_indices1[best_idx_half1]
    best_indices_h2 = multiplier_indices2[best_idx_half2]

    # Convert indices (0 or 1) to multipliers (1.0 or 2.0)
    best_multipliers_h1 = 1.0 + best_indices_h1.astype(jnp.float64)
    best_multipliers_h2 = 1.0 + best_indices_h2.astype(jnp.float64)
    current_best_multipliers = jnp.concatenate([best_multipliers_h1, best_multipliers_h2])

    logs.append(f"    Best Scan MSE for l0={current_l0:.2f}: {current_best_mse:.6e}")
    return current_best_mse, np.array(current_best_multipliers), logs


# --- End of File ---
# No `if __name__ == "__main__":` block needed for Streamlit apps.
# The script runs from top to bottom.
# The action logic within the `if button_pressed:` blocks handles the execution flow.
