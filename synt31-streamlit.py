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

@st.cache_data
def load_material_data_from_xlsx_sheet(file_path: str, sheet_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        except FileNotFoundError:
            st.error(f"Excel file '{file_path}' not found. Please check its presence.")
            return None, None, None
        except Exception as e:
            st.error(f"Error reading Excel ('{file_path}', sheet '{sheet_name}'): {e}")
            return None, None, None

        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        numeric_df = numeric_df.dropna(how='all')

        if numeric_df.shape[1] >= 3:
            cols_to_check = numeric_df.columns[:3]
            numeric_df = numeric_df.dropna(subset=cols_to_check)
        else:
            return np.array([]), np.array([]), np.array([])

        if numeric_df.empty:
            return np.array([]), np.array([]), np.array([])
        try:
            numeric_df = numeric_df.sort_values(by=numeric_df.columns[0])
        except IndexError:
            return np.array([]), np.array([]), np.array([])

        l_nm = numeric_df.iloc[:, 0].values.astype(np.float64)
        n = numeric_df.iloc[:, 1].values.astype(np.float64)
        k = numeric_df.iloc[:, 2].values.astype(np.float64)

        if np.any(k < -1e-9):
            k = np.maximum(k, 0.0)

        if len(l_nm) == 0:
            return np.array([]), np.array([]), np.array([])

        return l_nm, n, k

    except ValueError as ve:
        st.error(f"Excel Value Error for sheet '{sheet_name}': {ve}")
        return None, None, None
    except Exception as e:
        st.error(f"Unexpected error reading Excel sheet '{sheet_name}': {type(e).__name__} - {e}")
        return None, None, None

def get_available_materials_from_excel(excel_path: str) -> List[str]:
    try:
        xl = pd.ExcelFile(excel_path)
        sheet_names = [name for name in xl.sheet_names if not name.startswith("Sheet")]
        return sheet_names
    except FileNotFoundError:
        st.error(f"Excel file '{excel_path}' not found for listing materials.")
        return []
    except Exception as e:
        st.error(f"Error reading sheet names from '{excel_path}': {e}")
        return []

@jax.jit
def get_n_fused_silica(l_nm: jnp.ndarray) -> jnp.ndarray:
    n = jnp.full_like(l_nm, 1.46, dtype=jnp.float64)
    k_val = jnp.zeros_like(n)
    return n + 1j * k_val

@jax.jit
def get_n_bk7(l_nm: jnp.ndarray) -> jnp.ndarray:
    n = jnp.full_like(l_nm, 1.523, dtype=jnp.float64)
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
                                   excel_file_path: str) -> Optional[jnp.ndarray]:
    try:
        if isinstance(material_definition, (complex, float, int)):
            nk_complex = jnp.asarray(material_definition, dtype=jnp.complex128)
            if nk_complex.real <= 0:
                nk_complex = complex(1.0, nk_complex.imag)
            if nk_complex.imag < 0:
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
                l_data, n_data, k_data = load_material_data_from_xlsx_sheet(excel_file_path, sheet_name)
                if l_data is None or len(l_data) == 0:
                    st.error(f"Could not load or empty data for material '{sheet_name}' from {excel_file_path}.")
                    return None
                l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
                l_target_min = jnp.min(l_vec_target_jnp)
                l_target_max = jnp.max(l_vec_target_jnp)
                l_data_min = jnp.min(l_data_jnp)
                l_data_max = jnp.max(l_data_jnp)
                if l_target_min < l_data_min - 1e-6 or l_target_max > l_data_max + 1e-6:
                    pass
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
                k_data_jnp = jnp.maximum(k_data_jnp, 0.0)
            result = interp_nk_cached(l_vec_target_jnp, l_data_jnp, n_data_jnp, k_data_jnp)
        else:
            raise TypeError(f"Unsupported material definition type: {type(material_definition)}")

        if jnp.any(jnp.isnan(result.real)) or jnp.any(result.real <= 0):
            result = jnp.where(jnp.isnan(result.real) | (result.real <= 0), 1.0 + 1j*result.imag, result)
        if jnp.any(jnp.isnan(result.imag)) or jnp.any(result.imag < 0):
            result = jnp.where(jnp.isnan(result.imag) | (result.imag < 0), result.real + 0.0j, result)
        return result
    except Exception as e:
        st.error(f"Critical error preparing material data for '{material_definition}': {e}")
        return None

def _get_nk_at_lambda(material_definition: MaterialInputType, l_nm_target: float, excel_file_path: str) -> Optional[complex]:
    if l_nm_target <= 0:
        st.error(f"Target wavelength {l_nm_target}nm is invalid for getting n+ik.")
        return None
    l_vec_jnp = jnp.array([l_nm_target], dtype=jnp.float64)
    nk_array = _get_nk_array_for_lambda_vec(material_definition, l_vec_jnp, excel_file_path)
    if nk_array is None:
        return None
    else:
        return complex(nk_array[0])

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
        return carry_matrix # Pass through if thickness is effectively zero

    # If thickness is very small, effectively skip this layer by returning identity matrix product
    new_matrix = cond(thickness > 1e-12, compute_M_layer, compute_identity, thickness)
    return new_matrix, None


@jax.jit
def compute_stack_matrix_core_jax(ep_vector: jnp.ndarray, layer_indices: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    """Computes the total characteristic matrix for a stack of layers at a single wavelength."""
    num_layers = len(ep_vector)
    # Prepare data for scan: (thicknesses, n+ik_values, repeated_lambda_value)
    layers_scan_data = (ep_vector, layer_indices, jnp.full(num_layers, l_val))
    M_initial = jnp.eye(2, dtype=jnp.complex128) # Start with identity matrix (for air/substrate interface)
    # Use jax.lax.scan for efficient sequential computation over layers
    M_final, _ = scan(_compute_layer_matrix_scan_step_jit, M_initial, layers_scan_data)
    return M_final


@jax.jit
def calculate_single_wavelength_T_core(l_val: jnp.ndarray, ep_vector_contig: jnp.ndarray,
                                         layer_indices_at_lval: jnp.ndarray, nSub_at_lval: jnp.ndarray) -> jnp.ndarray:
    """Calculates Transmittance (Ts) for a single wavelength."""
    etainc = 1.0 + 0j  # Refractive index of incident medium (Air)
    etasub = nSub_at_lval # Refractive index of substrate

    # Conditional calculation: only proceed if lambda is valid
    def calculate_for_valid_l(l_: jnp.ndarray) -> jnp.ndarray:
        current_layer_indices = layer_indices_at_lval
        M = compute_stack_matrix_core_jax(ep_vector_contig, current_layer_indices, l_)
        m00, m01 = M[0, 0], M[0, 1]
        m10, m11 = M[1, 0], M[1, 1]

        # Denominator for transmission coefficient; ensure it's not too small
        rs_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        rs_denominator_abs = jnp.abs(rs_denominator)
        safe_denominator = jnp.where(rs_denominator_abs < 1e-12, 1e-12 + 0j, rs_denominator)

        ts = (2.0 * etainc) / safe_denominator # Transmission amplitude coefficient
        # Transmittance calculation
        real_etasub = jnp.real(etasub)
        real_etainc = jnp.real(etainc)
        safe_real_etainc = jnp.maximum(real_etainc, 1e-9) # Avoid division by zero for Re(etainc)
        Ts_complex = (real_etasub / safe_real_etainc) * (ts * jnp.conj(ts))
        Ts = jnp.real(Ts_complex)
        return jnp.where(rs_denominator_abs < 1e-12, jnp.nan, Ts) # Return NaN if denominator was too small

    def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray:
        return jnp.nan # Return NaN for invalid lambda

    Ts_result = cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)
    return Ts_result


def calculate_T_from_ep_jax(ep_vector: Union[np.ndarray, List[float]],
                              nH_material: MaterialInputType,
                              nL_material: MaterialInputType,
                              nSub_material: MaterialInputType,
                              l_vec: Union[np.ndarray, List[float]],
                              excel_file_path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Calculates Transmittance T(lambda) for a given structure (ep_vector)
    and H/L/Substrate materials over a vector of wavelengths (l_vec).
    """
    l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64)
    ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)

    if not l_vec_jnp.size: # No wavelengths to calculate for
        return {'l': np.array([]), 'Ts': np.array([])}

    if not ep_vector_jnp.size: # Bare substrate case
        nSub_arr = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
        if nSub_arr is None: return None
        # For bare substrate, T is more complex due to reflections, but often approximated as 1
        # A more accurate calculation for bare substrate T would be:
        # T = (4 * n_inc * n_sub.real) / ((n_inc + n_sub.real)**2 + n_sub.imag**2)
        # For simplicity here, assuming T=1 for bare substrate (common in some contexts or if reflections are ignored)
        # Or, more simply, if we assume no absorption in substrate and matched incident medium:
        # ts = (2 * 1.0) / (1.0 + nSub_arr)
        # Ts = (nSub_arr.real / 1.0) * jnp.abs(ts)**2
        # For now, returning 1s, can be refined.
        Ts = jnp.ones_like(l_vec_jnp) # Simplified: assumes perfect transmission for bare substrate
        return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts)}

    # Load refractive indices for H, L, Substrate materials
    nH_arr = _get_nk_array_for_lambda_vec(nH_material, l_vec_jnp, excel_file_path)
    nL_arr = _get_nk_array_for_lambda_vec(nL_material, l_vec_jnp, excel_file_path)
    nSub_arr = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)

    if nH_arr is None or nL_arr is None or nSub_arr is None:
        st.error("Failed to load one or more material indices for T(lambda) calculation.")
        return None

    # JIT compile the single wavelength calculation function
    calculate_single_wavelength_T_hl_jit = jax.jit(calculate_single_wavelength_T_core)

    num_layers = len(ep_vector_jnp)
    # Create an array of layer indices: (num_layers, num_wavelengths)
    # Alternating H and L materials for each layer
    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH_arr, nL_arr)
    # Transpose to: (num_wavelengths, num_layers) for vmap compatibility
    indices_alternating_T = indices_alternating.T

    # Use vmap to apply the single wavelength calculation across all wavelengths
    Ts_arr_raw = vmap(calculate_single_wavelength_T_hl_jit, in_axes=(0, None, 0, 0))(
        l_vec_jnp, ep_vector_jnp, indices_alternating_T, nSub_arr
    )

    # Handle potential NaNs (e.g., from division by zero) and clip to [0, 1]
    Ts_arr = jnp.nan_to_num(Ts_arr_raw, nan=0.0) # Replace NaN with 0
    Ts_arr_clipped = jnp.clip(Ts_arr, 0.0, 1.0) # Ensure T is physically valid

    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts_arr_clipped)}


def calculate_T_from_ep_arbitrary_jax(ep_vector: Union[np.ndarray, List[float]],
                                        material_sequence: List[str],
                                        nSub_material: MaterialInputType,
                                        l_vec: Union[np.ndarray, List[float]],
                                        excel_file_path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Calculates T(lambda) for a structure with an arbitrary sequence of materials.
    """
    l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64)
    ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)
    num_layers = len(ep_vector_jnp)

    if num_layers != len(material_sequence):
        st.error("Error: Size of ep_vector and material_sequence must match for arbitrary calculation.")
        return None

    if not l_vec_jnp.size:
        return {'l': np.array([]), 'Ts': np.array([])}

    if not ep_vector_jnp.size: # Bare substrate
        nSub_arr = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
        if nSub_arr is None: return None
        Ts = jnp.ones_like(l_vec_jnp) # Simplified T=1
        return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts)}

    # Load indices for each layer in the sequence
    layer_indices_list = []
    materials_ok = True
    for i, mat_name in enumerate(material_sequence):
        nk_arr = _get_nk_array_for_lambda_vec(mat_name, l_vec_jnp, excel_file_path)
        if nk_arr is None:
            st.error(f"Critical error: Failed to load material '{mat_name}' (layer {i+1}).")
            materials_ok = False
            break
        layer_indices_list.append(nk_arr)

    nSub_arr = _get_nk_array_for_lambda_vec(nSub_material, l_vec_jnp, excel_file_path)
    if nSub_arr is None:
        st.error("Critical error: Failed to load substrate material.")
        materials_ok = False

    if not materials_ok:
        return None

    if layer_indices_list:
        # Stack layer indices: (num_layers, num_wavelengths)
        layer_indices_arr = jnp.stack(layer_indices_list, axis=0)
    else: # Should not happen if num_layers > 0, but safeguard
        layer_indices_arr = jnp.empty((0, len(l_vec_jnp)), dtype=jnp.complex128)

    # JIT compile the core calculation function
    calculate_single_wavelength_T_arb_jit = jax.jit(calculate_single_wavelength_T_core)
    # Transpose indices for vmap: (num_wavelengths, num_layers)
    layer_indices_arr_T = layer_indices_arr.T

    # Calculate T for all wavelengths using vmap
    Ts_arr_raw = vmap(calculate_single_wavelength_T_arb_jit, in_axes=(0, None, 0, 0))(
        l_vec_jnp, ep_vector_jnp, layer_indices_arr_T, nSub_arr
    )

    # Clean up results
    Ts_arr = jnp.nan_to_num(Ts_arr_raw, nan=0.0)
    Ts_arr_clipped = jnp.clip(Ts_arr, 0.0, 1.0)

    return {'l': np.array(l_vec_jnp), 'Ts': np.array(Ts_arr_clipped)}


def calculate_initial_ep(emp: Union[List[float], Tuple[float,...]], l0: float,
                           nH0_material: MaterialInputType, nL0_material: MaterialInputType,
                           excel_file_path: str) -> Optional[np.ndarray]:
    """Calculates initial physical thicknesses based on QWOT multipliers and indices at l0."""
    num_layers = len(emp)
    ep_initial = np.zeros(num_layers, dtype=np.float64)

    if l0 <= 0:
        st.warning(f"l0={l0} <= 0 in calculate_initial_ep. Initial thicknesses set to 0.")
        return ep_initial

    # Get complex indices at the reference wavelength l0
    nH_complex_at_l0 = _get_nk_at_lambda(nH0_material, l0, excel_file_path)
    nL_complex_at_l0 = _get_nk_at_lambda(nL0_material, l0, excel_file_path)

    if nH_complex_at_l0 is None or nL_complex_at_l0 is None:
        st.error(f"Critical error getting H or L indices at l0={l0}nm for initial thickness calculation.")
        return None

    nH_real_at_l0 = nH_complex_at_l0.real
    nL_real_at_l0 = nL_complex_at_l0.real

    if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
        st.warning(f"n'H({nH_real_at_l0:.3f}) or n'L({nL_real_at_l0:.3f}) at l0={l0}nm is <= 0. QWOT calculation may be incorrect.")

    # Calculate physical thickness for each layer
    valid_indices = True
    for i in range(num_layers):
        multiplier = emp[i]
        is_H_layer = (i % 2 == 0)
        n_real_layer_at_l0 = nH_real_at_l0 if is_H_layer else nL_real_at_l0

        if n_real_layer_at_l0 <= 1e-9: # Avoid division by zero/very small number
            ep_initial[i] = 0.0
            if multiplier > 1e-9: # If QWOT was non-zero but index is bad
                layer_type = "H" if is_H_layer else "L"
                st.error(f"Layer {i+1} ({layer_type}) has QWOT={multiplier} but thickness=0 (likely n'({layer_type},l0)={n_real_layer_at_l0:.3f} <= 0).")
                valid_indices = False
        else:
            ep_initial[i] = multiplier * l0 / (4.0 * n_real_layer_at_l0)

    if not valid_indices:
        st.error("Error during initial thickness calculation due to invalid indices at l0.")
        return None

    # Apply minimum physical thickness constraint
    ep_initial_phys = np.where(ep_initial < MIN_THICKNESS_PHYS_NM, 0.0, ep_initial)
    num_clamped_zero = np.sum((ep_initial > 1e-12) & (ep_initial < MIN_THICKNESS_PHYS_NM))
    if num_clamped_zero > 0:
        # st.warning(f"{num_clamped_zero} initial thicknesses < {MIN_THICKNESS_PHYS_NM}nm were set to 0.")
        ep_initial = np.where(ep_initial < MIN_THICKNESS_PHYS_NM, 0.0, ep_initial)

    return ep_initial


def calculate_qwot_from_ep(ep_vector: np.ndarray, l0: float,
                             nH0_material: MaterialInputType, nL0_material: MaterialInputType,
                             excel_file_path: str) -> Optional[np.ndarray]:
    """Calculates QWOT multipliers from physical thicknesses and indices at l0."""
    num_layers = len(ep_vector)
    qwot_multipliers = np.full(num_layers, np.nan, dtype=np.float64)

    if l0 <= 0:
        st.warning(f"l0={l0} <= 0 in calculate_qwot_from_ep. QWOT set to NaN.")
        return qwot_multipliers

    # Get indices at l0
    nH_complex_at_l0 = _get_nk_at_lambda(nH0_material, l0, excel_file_path)
    nL_complex_at_l0 = _get_nk_at_lambda(nL0_material, l0, excel_file_path)

    if nH_complex_at_l0 is None or nL_complex_at_l0 is None:
        st.error(f"Error calculating QWOT: H/L indices not found at l0={l0}nm.")
        return None

    nH_real_at_l0 = nH_complex_at_l0.real
    nL_real_at_l0 = nL_complex_at_l0.real

    if nH_real_at_l0 <= 1e-9 or nL_real_at_l0 <= 1e-9:
        st.warning(f"n'H({nH_real_at_l0:.3f}) or n'L({nL_real_at_l0:.3f}) at l0={l0}nm is <= 0. QWOT calculation may be incorrect/NaN.")

    # Calculate QWOT for each layer
    indices_ok = True
    for i in range(num_layers):
        n_real_layer_at_l0 = nH_real_at_l0 if i % 2 == 0 else nL_real_at_l0
        if n_real_layer_at_l0 <= 1e-9:
            if ep_vector[i] > 1e-9 : # If thickness is non-zero but index is bad
                layer_type = "H" if i % 2 == 0 else "L"
                # st.warning(f"Cannot calculate QWOT for layer {i+1} ({layer_type}) because n'({l0}nm) <= 0.")
                indices_ok = False # QWOT will remain NaN
            else: # thickness is zero, so QWOT is zero
                qwot_multipliers[i] = 0.0
        else:
            qwot_multipliers[i] = ep_vector[i] * (4.0 * n_real_layer_at_l0) / l0

    if not indices_ok:
        st.warning("Some QWOT values could not be calculated (invalid indices at l0). They appear as NaN.")
        return qwot_multipliers # Return array with NaNs
    else:
        return qwot_multipliers


def calculate_final_mse(res: Dict[str, np.ndarray], active_targets: List[Dict]) -> Tuple[Optional[float], int]:
    """Calculates the Mean Squared Error (MSE) between calculated results and targets."""
    total_squared_error = 0.0
    total_points_in_targets = 0
    mse = None

    # Basic validation of inputs
    if not active_targets or 'Ts' not in res or res['Ts'] is None or 'l' not in res or res['l'] is None:
        return mse, total_points_in_targets

    res_l_np = np.asarray(res['l'])
    res_ts_np = np.asarray(res['Ts'])

    if res_l_np.size == 0 or res_ts_np.size == 0 or res_l_np.size != res_ts_np.size:
        return mse, total_points_in_targets

    # Iterate through each active target zone
    for target in active_targets:
        try:
            # Extract and validate target parameters
            l_min = float(target['min'])
            l_max = float(target['max'])
            t_min = float(target['target_min'])
            t_max = float(target['target_max'])
            if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0): continue # Skip if T outside [0,1]
            if l_max < l_min: continue # Skip if lambda range invalid
        except (KeyError, ValueError, TypeError):
            continue # Skip malformed targets

        # Find indices of calculated results within the target lambda range
        indices = np.where((res_l_np >= l_min) & (res_l_np <= l_max))[0]
        if indices.size > 0:
            calculated_Ts_in_zone = res_ts_np[indices]
            target_lambdas_in_zone = res_l_np[indices]

            # Ignore NaN values in calculated Ts
            finite_mask = np.isfinite(calculated_Ts_in_zone)
            calculated_Ts_in_zone = calculated_Ts_in_zone[finite_mask]
            target_lambdas_in_zone = target_lambdas_in_zone[finite_mask]

            if calculated_Ts_in_zone.size == 0: continue # Skip if no valid points left

            # Determine the target transmittance value(s) for these points
            if abs(l_max - l_min) < 1e-9: # Single point target (constant T)
                interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
            else: # Linear interpolation for target T across the range
                slope = (t_max - t_min) / (l_max - l_min)
                interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)

            # Calculate squared errors and accumulate
            squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
            total_squared_error += np.sum(squared_errors)
            total_points_in_targets += len(calculated_Ts_in_zone)

    # Calculate final MSE if any points were included
    if total_points_in_targets > 0:
        mse = total_squared_error / total_points_in_targets
    return mse, total_points_in_targets


@jax.jit
def calculate_mse_for_optimization_penalized_jax(ep_vector: jnp.ndarray,
                                                 nH_arr: jnp.ndarray, nL_arr: jnp.ndarray, nSub_arr: jnp.ndarray,
                                                 l_vec_optim: jnp.ndarray,
                                                 active_targets_tuple: Tuple[Tuple[float, float, float, float], ...],
                                                 min_thickness_phys_nm: float) -> jnp.ndarray:
    """
    JAX-jitted MSE calculation function used by the optimizer.
    Includes a penalty for thicknesses below the physical minimum.
    """
    # Penalty for layers slightly below the minimum thickness (but not zero)
    below_min_mask = (ep_vector < min_thickness_phys_nm) & (ep_vector > 1e-12)
    penalty_thin = jnp.sum(jnp.where(below_min_mask, (min_thickness_phys_nm - ep_vector)**2, 0.0))
    penalty_weight = 1e5 # High weight to strongly discourage thin layers
    penalty_cost = penalty_thin * penalty_weight

    # Use clamped thicknesses for the actual optical calculation
    ep_vector_calc = jnp.maximum(ep_vector, min_thickness_phys_nm)

    num_layers = len(ep_vector_calc)
    # Prepare layer indices (alternating H/L)
    indices_alternating = jnp.where(jnp.arange(num_layers)[:, None] % 2 == 0, nH_arr, nL_arr)
    indices_alternating_T = indices_alternating.T # Transpose for vmap

    # Calculate Transmittance across wavelengths using vmap
    calculate_T_single_jit = jax.jit(calculate_single_wavelength_T_core)
    Ts_raw = vmap(calculate_T_single_jit, in_axes=(0, None, 0, 0))(
        l_vec_optim, ep_vector_calc, indices_alternating_T, nSub_arr
    )
    Ts = jnp.nan_to_num(Ts_raw, nan=0.0) # Replace potential NaNs

    # Calculate MSE against targets
    total_squared_error = 0.0
    total_points_in_targets = 0
    for i in range(len(active_targets_tuple)):
        l_min, l_max, t_min, t_max = active_targets_tuple[i]
        target_mask = (l_vec_optim >= l_min) & (l_vec_optim <= l_max) # Mask for this target zone

        # Calculate target T values (linear interpolation)
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t_full = t_min + slope * (l_vec_optim - l_min)

        # Calculate squared errors within the target mask
        squared_errors_full = (Ts - interpolated_target_t_full)**2
        masked_sq_error = jnp.where(target_mask, squared_errors_full, 0.0)

        total_squared_error += jnp.sum(masked_sq_error)
        total_points_in_targets += jnp.sum(target_mask)

    # Calculate MSE, handle case with zero target points
    mse = jnp.where(total_points_in_targets > 0,
                    total_squared_error / total_points_in_targets,
                    jnp.inf) # Infinite MSE if no points in target

    # Add penalty cost to MSE
    final_cost = mse + penalty_cost
    # Ensure final cost is finite (replace NaN/inf with large number)
    return jnp.nan_to_num(final_cost, nan=jnp.inf, posinf=jnp.inf)


@jax.jit
def calculate_mse_arbitrary_sequence_jax(ep_vector: jnp.ndarray,
                                           layer_indices_arr: jnp.ndarray, # Shape (num_layers, num_lambdas)
                                           nSub_arr: jnp.ndarray,
                                           l_vec_eval: jnp.ndarray,
                                           active_targets_tuple: Tuple[Tuple[float, float, float, float], ...]) -> jnp.ndarray:
    """JAX-jitted MSE calculation for arbitrary material sequences (no penalty)."""
    layer_indices_arr_T = layer_indices_arr.T # Transpose to (num_lambdas, num_layers)
    calculate_T_single_jit = jax.jit(calculate_single_wavelength_T_core)

    # Calculate T across wavelengths
    Ts_raw = vmap(calculate_T_single_jit, in_axes=(0, None, 0, 0))(
        l_vec_eval, ep_vector, layer_indices_arr_T, nSub_arr
    )
    Ts = jnp.nan_to_num(Ts_raw, nan=0.0)

    # Calculate MSE against targets (same logic as penalized version, without penalty)
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


# --- Optimization and Structure Modification Functions ---

def _run_core_optimization(ep_start_optim: np.ndarray,
                           validated_inputs: Dict, active_targets: List[Dict],
                           min_thickness_phys: float, log_prefix: str = ""
                           ) -> Tuple[Optional[np.ndarray], bool, float, str]:
    """Runs the core L-BFGS-B optimization using SciPy and JAX gradients."""
    num_layers_start = len(ep_start_optim)
    optim_success = False
    final_cost = np.inf
    result_message_str = "Optimization not launched or failed early."
    final_ep = None

    if num_layers_start == 0:
        # Cannot optimize an empty structure
        return None, False, np.inf, "Empty structure"

    try:
        # Extract necessary parameters from validated inputs
        l_min_optim = validated_inputs['l_range_deb']
        l_max_optim = validated_inputs['l_range_fin']
        l_step_optim = validated_inputs['l_step']
        nH_material = validated_inputs['nH_material']
        nL_material = validated_inputs['nL_material']
        nSub_material = validated_inputs['nSub_material']
        maxiter = MAXITER_HARDCODED
        maxfun = MAXFUN_HARDCODED

        # Generate lambda vector for optimization grid (geometric spacing often good for optics)
        num_pts_optim = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
        l_vec_optim_np = np.geomspace(l_min_optim, l_max_optim, num_pts_optim)
        l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)] # Ensure valid lambdas
        if not l_vec_optim_np.size:
            raise ValueError("Failed to generate valid lambda vector for optimization.")
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)

        # Prepare dispersive indices for the optimization lambda grid
        nH_arr_optim = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        nL_arr_optim = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        nSub_arr_optim = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, EXCEL_FILE_PATH)
        if nH_arr_optim is None or nL_arr_optim is None or nSub_arr_optim is None:
            raise RuntimeError("Failed to load indices for optimization lambda grid.")

        # Prepare static arguments for the JAX cost function
        active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
        static_args_for_jax = (
            nH_arr_optim, nL_arr_optim, nSub_arr_optim,
            l_vec_optim_jax, active_targets_tuple,
            min_thickness_phys
        )

        # Get the JIT-compiled value-and-gradient function
        value_and_grad_fn = jax.jit(jax.value_and_grad(calculate_mse_for_optimization_penalized_jax))

        # Wrapper function for SciPy's minimize, returning value and gradient
        def scipy_obj_grad_wrapper(ep_vector_np_in, *args):
            try:
                ep_vector_jax = jnp.asarray(ep_vector_np_in, dtype=jnp.float64)
                value_jax, grad_jax = value_and_grad_fn(ep_vector_jax, *args)
                # Handle non-finite values which can crash the optimizer
                if not jnp.isfinite(value_jax):
                    value_np = np.inf
                    grad_np = np.zeros_like(ep_vector_np_in, dtype=np.float64) # Return zero grad if cost is inf
                else:
                    value_np = float(np.array(value_jax))
                    grad_np_raw = np.array(grad_jax, dtype=np.float64)
                    # Clean gradient: replace NaN/inf with large/small numbers or zero
                    grad_np = np.nan_to_num(grad_np_raw, nan=0.0, posinf=1e6, neginf=-1e6)
                return value_np, grad_np
            except Exception as e_wrap:
                # Fallback in case of unexpected error within JAX calculation
                print(f"Error in scipy_obj_grad_wrapper: {e_wrap}") # Keep this print for debugging
                return np.inf, np.zeros_like(ep_vector_np_in, dtype=np.float64)

        # Define bounds for the optimizer (thickness >= min_physical_thickness)
        lbfgsb_bounds = [(min_thickness_phys, None)] * num_layers_start
        # Optimizer options
        options = {'maxiter': maxiter, 'maxfun': maxfun,
                   'disp': False, # Set to True for verbose optimizer output
                   'ftol': 1e-12, 'gtol': 1e-8} # Tolerances for termination

        # Run the optimization
        result = minimize(scipy_obj_grad_wrapper,
                          ep_start_optim,
                          args=static_args_for_jax,
                          method='L-BFGS-B',
                          jac=True, # Provide gradient function
                          bounds=lbfgsb_bounds,
                          options=options)

        # Process results
        final_cost = result.fun if np.isfinite(result.fun) else np.inf
        result_message_str = result.message.decode('utf-8') if isinstance(result.message, bytes) else str(result.message)

        # Check if optimization was successful or hit iteration limit (often acceptable)
        is_success_or_limit = (result.success or result.status == 1) and np.isfinite(final_cost)

        if is_success_or_limit:
            final_ep_raw = result.x
            # Ensure final thicknesses respect the minimum physical constraint
            final_ep = np.maximum(final_ep_raw, min_thickness_phys)
            optim_success = True
        else: # Optimization failed
            optim_success = False
            # Revert to the starting structure, but clamped to minimum thickness
            final_ep = np.maximum(ep_start_optim, min_thickness_phys)
            # Try to recalculate the cost of this reverted structure
            try:
                reverted_cost, _ = scipy_obj_grad_wrapper(final_ep, *static_args_for_jax)
                final_cost = reverted_cost if np.isfinite(reverted_cost) else np.inf
            except Exception:
                final_cost = np.inf # Failed to recalculate cost

    except Exception as e_optim:
        st.error(f"Critical error during optimization: {e_optim}")
        print(f"{log_prefix}Major ERROR during JAX/Scipy optimization: {e_optim}\n{traceback.format_exc(limit=2)}") # Keep for debug
        # Fallback to initial structure if possible
        final_ep = np.maximum(ep_start_optim, min_thickness_phys) if ep_start_optim is not None else None
        optim_success = False
        final_cost = np.inf
        result_message_str = f"Exception: {e_optim}"

    return final_ep, optim_success, final_cost, result_message_str


def _perform_layer_merge_or_removal_only(ep_vector_in: np.ndarray, min_thickness_phys: float,
                                         log_prefix: str = "", target_layer_index: Optional[int] = None,
                                         threshold_for_removal: Optional[float] = None) -> Tuple[Optional[np.ndarray], bool]:
    """
    Identifies the thinnest layer (optionally below a threshold or a specific target)
    and performs removal/merging based on its position (first, last, middle).
    Returns the modified ep_vector and a flag indicating if a change occurred.
    Internal logs are removed, returns only essential results.
    """
    current_ep = ep_vector_in.copy()
    num_layers = len(current_ep)
    structure_changed = False
    ep_after_merge = None # Initialize result vector

    # --- Basic Checks ---
    if num_layers <= 2 and target_layer_index is None:
        return current_ep, False # Cannot remove/merge if only 2 layers or less without target
    elif num_layers < 1:
        return current_ep, False # Empty structure

    try:
        # --- Identify Target Layer ---
        thin_layer_index = -1 # Sentinel value
        min_thickness_found = np.inf

        # 1. Check for manually targeted layer
        if target_layer_index is not None:
            if 0 <= target_layer_index < num_layers and current_ep[target_layer_index] >= min_thickness_phys:
                thin_layer_index = target_layer_index
            else: # Invalid manual target, fallback to auto search
                target_layer_index = None

        # 2. Auto-search if no valid manual target
        if target_layer_index is None:
            candidate_indices = np.where(current_ep >= min_thickness_phys)[0]
            if candidate_indices.size == 0:
                return current_ep, False # No valid layers found

            candidate_thicknesses = current_ep[candidate_indices]
            indices_to_consider = candidate_indices
            thicknesses_to_consider = candidate_thicknesses

            # Apply optional threshold filter
            if threshold_for_removal is not None:
                mask_below_threshold = thicknesses_to_consider < threshold_for_removal
                if np.any(mask_below_threshold):
                    indices_to_consider = indices_to_consider[mask_below_threshold]
                    thicknesses_to_consider = thicknesses_to_consider[mask_below_threshold]
                else:
                    return current_ep, False # No layers below threshold found

            # Find the thinnest among the remaining candidates
            if indices_to_consider.size > 0:
                min_idx_local = np.argmin(thicknesses_to_consider)
                thin_layer_index = indices_to_consider[min_idx_local]
            else:
                return current_ep, False # No final candidate found

        # --- Perform Action Based on Identified Layer ---
        if thin_layer_index == -1:
            return current_ep, False # Failed to identify layer

        # Check again for <=2 layers case if target was specified
        if num_layers <= 2 and thin_layer_index == 0:
             ep_after_merge = current_ep[2:]
             structure_changed = True
        elif num_layers <= 1 and thin_layer_index == 0:
             ep_after_merge = np.array([])
             structure_changed = True
        elif num_layers <= 2 and thin_layer_index == 1:
             ep_after_merge = current_ep[:-1]
             structure_changed = True
        # Standard cases for num_layers > 2
        elif thin_layer_index == 0: # Thinnest is the first layer
             ep_after_merge = current_ep[2:] # Remove first two
             structure_changed = True
        elif thin_layer_index == num_layers - 1: # Thinnest is the last layer
             if num_layers >= 1:
                 ep_after_merge = current_ep[:-1] # Remove only the last
                 structure_changed = True
             else: # Should not happen
                 return current_ep, False
        else: # Thinnest is a middle layer
             merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
             ep_before = current_ep[:thin_layer_index - 1]
             ep_after = current_ep[thin_layer_index + 2:]
             ep_after_merge = np.concatenate((ep_before, [merged_thickness], ep_after))
             structure_changed = True

        # --- Return Result ---
        if structure_changed and ep_after_merge is not None:
            # Ensure final thicknesses meet minimum requirement
            ep_after_merge = np.maximum(ep_after_merge, min_thickness_phys)
            return ep_after_merge, True
        elif structure_changed and ep_after_merge is None: # Logic error safeguard
            return current_ep, False
        else: # No change was made
            return current_ep, False

    except Exception as e_merge:
        st.error(f"Internal error during layer removal/merge: {e_merge}")
        print(f"{log_prefix}ERROR during merge/removal logic: {e_merge}\n{traceback.format_exc(limit=1)}") # Keep for debug
        return current_ep, False # Return original state on error


def _perform_needle_insertion_scan(ep_vector_in: np.ndarray,
                                   nH_material: MaterialInputType, nL_material: MaterialInputType, nSub_material: MaterialInputType,
                                   l_vec_optim_np: np.ndarray, active_targets: List[Dict],
                                   cost_function_jax: Callable,
                                   min_thickness_phys: float, base_needle_thickness_nm: float,
                                   scan_step: float, l0_repr: float,
                                   excel_file_path: str, log_prefix: str = ""
                                   ) -> Tuple[Optional[np.ndarray], float, int]:
    """
    Performs a needle scan to find the best position to insert a thin layer.
    Returns the best structure found, its cost, and the index where insertion occurred.
    """
    num_layers_in = len(ep_vector_in)
    if num_layers_in == 0:
        return None, np.inf, -1 # Cannot scan empty structure

    try:
        # Prepare indices and static args for cost function
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        nH_arr_optim = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, excel_file_path)
        nL_arr_optim = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, excel_file_path)
        nSub_arr_optim = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, excel_file_path)
        if nH_arr_optim is None or nL_arr_optim is None or nSub_arr_optim is None:
            raise RuntimeError("Failed to load indices for needle scan.")

        active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
        static_args_cost_fn = (
            nH_arr_optim, nL_arr_optim, nSub_arr_optim,
            l_vec_optim_jax, active_targets_tuple,
            min_thickness_phys
        )
        # Calculate initial cost
        initial_cost_jax = cost_function_jax(jnp.asarray(ep_vector_in), *static_args_cost_fn)
        initial_cost = float(np.array(initial_cost_jax))
        if not np.isfinite(initial_cost):
            st.error("Needle Scan Error: Cost of starting structure is not finite.")
            return None, np.inf, -1
    except Exception as e_prep:
        st.error(f"Error preparing needle scan: {e_prep}")
        return None, np.inf, -1

    best_ep_found = None
    min_cost_found = initial_cost
    best_insertion_idx = -1
    tested_insertions = 0

    ep_cumsum = np.cumsum(ep_vector_in)
    total_thickness = ep_cumsum[-1] if num_layers_in > 0 else 0.0

    # Iterate through potential insertion points (z)
    for z in np.arange(scan_step, total_thickness, scan_step):
        current_layer_idx = -1
        layer_start_z = 0.0
        # Find which layer the insertion point z falls into
        for i in range(num_layers_in):
            layer_end_z = ep_cumsum[i]
            if z > layer_start_z and z <= layer_end_z:
                # Check if splitting the layer creates two valid thicknesses
                t_part1 = z - layer_start_z
                t_part2 = layer_end_z - z
                if t_part1 >= min_thickness_phys and t_part2 >= min_thickness_phys:
                    current_layer_idx = i
                else: # Split point too close to edge
                    current_layer_idx = -2 # Mark as invalid split
                break
            layer_start_z = layer_end_z

        if current_layer_idx < 0: continue # Invalid split point or outside layers

        tested_insertions += 1
        # Calculate thicknesses of the split layer parts
        t_layer_split_1 = z - (ep_cumsum[current_layer_idx-1] if current_layer_idx > 0 else 0.0)
        t_layer_split_2 = ep_cumsum[current_layer_idx] - z

        # Create the temporary structure with the inserted needle layer
        ep_temp_np = np.concatenate((
            ep_vector_in[:current_layer_idx],
            [t_layer_split_1, base_needle_thickness_nm, t_layer_split_2],
            ep_vector_in[current_layer_idx+1:]
        ))
        # Ensure minimum thickness for the new structure
        ep_temp_np_clamped = np.maximum(ep_temp_np, min_thickness_phys)

        # Calculate the cost of this temporary structure
        try:
            current_cost_jax = cost_function_jax(jnp.asarray(ep_temp_np_clamped), *static_args_cost_fn)
            current_cost = float(np.array(current_cost_jax))
            # If this cost is the best found so far, store it
            if np.isfinite(current_cost) and current_cost < min_cost_found:
                min_cost_found = current_cost
                best_ep_found = ep_temp_np_clamped.copy()
                best_insertion_idx = current_layer_idx
        except Exception:
            # Silently ignore errors during cost calculation for a single point
            continue

    if best_ep_found is not None:
        # Improvement found
        return best_ep_found, min_cost_found, best_insertion_idx
    else:
        # No improvement found
        return None, initial_cost, -1


def _run_needle_iterations(ep_start: np.ndarray, num_needles: int,
                           validated_inputs: Dict, active_targets: List[Dict],
                           min_thickness_phys: float, l_vec_optim_np: np.ndarray,
                           scan_step_nm: float, base_needle_thickness_nm: float,
                           excel_file_path: str, log_prefix: str = ""
                           ) -> Tuple[np.ndarray, float]:
    """Runs multiple iterations of needle insertion followed by optimization."""
    best_ep_overall = np.asarray(ep_start).copy()
    best_mse_overall = np.inf

    nH_material = validated_inputs['nH_material']
    nL_material = validated_inputs['nL_material']
    nSub_material = validated_inputs['nSub_material']
    l0_repr = validated_inputs.get('l0', 500.0) # Get l0 for logging/repr if needed

    cost_fn_penalized_jit = jax.jit(calculate_mse_for_optimization_penalized_jax)

    # Calculate initial MSE
    try:
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        nH_arr = _get_nk_array_for_lambda_vec(nH_material, l_vec_optim_jax, excel_file_path)
        nL_arr = _get_nk_array_for_lambda_vec(nL_material, l_vec_optim_jax, excel_file_path)
        nSub_arr = _get_nk_array_for_lambda_vec(nSub_material, l_vec_optim_jax, excel_file_path)
        if nH_arr is None or nL_arr is None or nSub_arr is None:
            raise RuntimeError("Failed to load indices for needle iterations.")

        active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
        static_args_cost_fn = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, min_thickness_phys)

        initial_cost_jax = cost_fn_penalized_jit(jnp.asarray(best_ep_overall), *static_args_cost_fn)
        best_mse_overall = float(np.array(initial_cost_jax))
        if not np.isfinite(best_mse_overall):
            raise ValueError("Initial MSE for needle iterations is not finite.")
    except Exception as e_init:
        st.error(f"Error initializing needle iterations: {e_init}")
        return ep_start, np.inf # Return original state if init fails

    # --- Needle Iteration Loop ---
    for i in range(num_needles):
        current_ep_iter = best_ep_overall.copy()
        num_layers_current = len(current_ep_iter)
        if num_layers_current == 0: break # Stop if structure becomes empty

        # 1. Perform Needle Scan
        st.write(f"{log_prefix} Needle scan {i+1}...")
        ep_after_scan, cost_after_scan, inserted_idx = _perform_needle_insertion_scan(
            current_ep_iter,
            nH_material, nL_material, nSub_material,
            l_vec_optim_np, active_targets,
            cost_fn_penalized_jit, # Pass the JITted cost function
            min_thickness_phys, base_needle_thickness_nm, scan_step_nm, l0_repr,
            excel_file_path, log_prefix=f"{log_prefix}  [Scan {i+1}] "
        )

        if ep_after_scan is None:
             # No improvement found by scan
             break # Stop needle iterations

        # 2. Re-optimize the structure after needle insertion
        st.write(f"{log_prefix} Re-optimizing after needle {i+1}...")
        ep_after_reopt, optim_success, final_cost_reopt, optim_status_msg = \
            _run_core_optimization(ep_after_scan, validated_inputs, active_targets,
                                   min_thickness_phys, log_prefix=f"{log_prefix}  [Re-Opt {i+1}] ")

        if not optim_success:
             # Re-optimization failed
             break # Stop needle iterations

        # 3. Check for improvement and update best result
        if final_cost_reopt < best_mse_overall - MSE_IMPROVEMENT_TOLERANCE:
             # Significant improvement found
             best_ep_overall = ep_after_reopt.copy()
             best_mse_overall = final_cost_reopt
        else:
             # No significant improvement, stop iterations
             best_ep_overall = ep_after_reopt.copy() # Still keep the result of the last successful re-opt
             best_mse_overall = final_cost_reopt
             break

    return best_ep_overall, best_mse_overall


def run_auto_mode(initial_ep: Optional[np.ndarray],
                  validated_inputs: Dict, active_targets: List[Dict],
                  excel_file_path: str, log_callback: Callable):
    """Runs the full automatic optimization cycle (Needle -> Thin -> Re-Opt)."""
    start_time_auto = time.time()
    # log_callback("#"*10 + f" Starting Auto Mode (Max {AUTO_MAX_CYCLES} Cycles) " + "#"*10)

    best_ep_so_far = None
    best_mse_so_far = np.inf
    num_cycles_done = 0
    termination_reason = f"Max {AUTO_MAX_CYCLES} cycles reached"
    threshold_for_thin_removal = validated_inputs.get('auto_thin_threshold', 1.0)
    # log_callback(f"  Auto removal threshold: {threshold_for_thin_removal:.3f} nm")

    try:
        # --- Determine Starting Structure ---
        current_ep = None
        if initial_ep is not None: # Start from previous optimized state if available
            # log_callback("  Auto Mode: Using previous optimized structure.")
            current_ep = initial_ep.copy()
            # Calculate initial MSE for this state
            l_min_optim, l_max_optim = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
            l_step_optim = validated_inputs['l_step']
            num_pts = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
            l_vec_optim_np_auto = np.geomspace(l_min_optim, l_max_optim, num_pts)
            l_vec_optim_np_auto = l_vec_optim_np_auto[(l_vec_optim_np_auto > 0) & np.isfinite(l_vec_optim_np_auto)]
            if not l_vec_optim_np_auto.size: raise ValueError("Failed to generate lambda for initial auto MSE calc.")

            l_vec_optim_jax = jnp.asarray(l_vec_optim_np_auto)
            nH_arr = _get_nk_array_for_lambda_vec(validated_inputs['nH_material'], l_vec_optim_jax, excel_file_path)
            nL_arr = _get_nk_array_for_lambda_vec(validated_inputs['nL_material'], l_vec_optim_jax, excel_file_path)
            nSub_arr = _get_nk_array_for_lambda_vec(validated_inputs['nSub_material'], l_vec_optim_jax, excel_file_path)
            if nH_arr is None or nL_arr is None or nSub_arr is None: raise RuntimeError("Failed to load indices for initial auto MSE.")

            active_targets_tuple = tuple((float(t['min']), float(t['max']), float(t['target_min']), float(t['target_max'])) for t in active_targets)
            static_args = (nH_arr, nL_arr, nSub_arr, l_vec_optim_jax, active_targets_tuple, MIN_THICKNESS_PHYS_NM)
            cost_fn_jit = jax.jit(calculate_mse_for_optimization_penalized_jax)
            initial_mse_jax = cost_fn_jit(jnp.asarray(current_ep), *static_args)
            initial_mse = float(np.array(initial_mse_jax))
            if not np.isfinite(initial_mse): raise ValueError("Initial MSE (from optimized state) not finite.")
            best_mse_so_far = initial_mse
            best_ep_so_far = current_ep.copy()
            # log_callback(f"  Initial MSE (from optimized state): {best_mse_so_far:.6e}")
        else: # Start from nominal QWOT structure
            # log_callback("  Auto Mode: Using nominal structure (QWOT).")
            emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
            if not emp_list: raise ValueError("Nominal QWOT empty, cannot start Auto Mode.")
            ep_nominal = calculate_initial_ep(emp_list, validated_inputs['l0'],
                                                validated_inputs['nH_material'], validated_inputs['nL_material'],
                                                excel_file_path)
            if ep_nominal is None: raise RuntimeError("Failed to calculate initial nominal thicknesses.")
            # log_callback(f"  Nominal structure: {len(ep_nominal)} layers. Starting initial optimization...")
            st.info("Auto Mode: Initial optimization of nominal structure...")
            ep_after_initial_opt, initial_opt_success, initial_mse, initial_opt_msg = \
                _run_core_optimization(ep_nominal, validated_inputs, active_targets,
                                       MIN_THICKNESS_PHYS_NM, log_prefix="  [Auto Init Opt] ")
            if not initial_opt_success:
                st.error(f"Failed initial optimization of Auto Mode: {initial_opt_msg}")
                return None, np.inf
            # log_callback(f"  Initial optimization finished. MSE: {initial_mse:.6e}")
            best_ep_so_far = ep_after_initial_opt.copy()
            best_mse_so_far = initial_mse

        if best_ep_so_far is None or not np.isfinite(best_mse_so_far):
            raise RuntimeError("Invalid starting state for Auto cycles.")

        # log_callback(f"--- Starting Auto Cycles (Starting MSE: {best_mse_so_far:.6e}, {len(best_ep_so_far)} layers) ---")

        # --- Auto Mode Cycle Loop ---
        for cycle_num in range(AUTO_MAX_CYCLES):
            # log_callback(f"\n--- Auto Cycle {cycle_num + 1} / {AUTO_MAX_CYCLES} ---")
            st.info(f"Auto Cycle {cycle_num + 1}/{AUTO_MAX_CYCLES} | Current MSE: {best_mse_so_far:.3e}")
            mse_at_cycle_start = best_mse_so_far
            ep_at_cycle_start = best_ep_so_far.copy()
            cycle_improved_globally = False

            # 1. Needle Phase
            # log_callback(f"  [Cycle {cycle_num+1}] Needle Phase ({AUTO_NEEDLES_PER_CYCLE} max iterations)...")
            st.write(f"Cycle {cycle_num + 1}: Needle Phase...")
            l_min_optim, l_max_optim = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
            l_step_optim = validated_inputs['l_step']
            num_pts_needle = max(2, int(np.round((l_max_optim - l_min_optim) / l_step_optim)) + 1)
            l_vec_optim_np_needle = np.geomspace(l_min_optim, l_max_optim, num_pts_needle)
            l_vec_optim_np_needle = l_vec_optim_np_needle[(l_vec_optim_np_needle > 0) & np.isfinite(l_vec_optim_np_needle)]
            if not l_vec_optim_np_needle.size:
                # log_callback("  ERROR: cannot generate lambda for needle phase. Cycle aborted.")
                break

            ep_after_needles, mse_after_needles = \
                _run_needle_iterations(best_ep_so_far, AUTO_NEEDLES_PER_CYCLE, validated_inputs, active_targets,
                                       MIN_THICKNESS_PHYS_NM, l_vec_optim_np_needle,
                                       DEFAULT_NEEDLE_SCAN_STEP_NM, BASE_NEEDLE_THICKNESS_NM,
                                       excel_file_path, log_prefix=f"    [Needle {cycle_num+1}] ")
            # log_callback(f"  [Cycle {cycle_num+1}] End Needle Phase. MSE: {mse_after_needles:.6e}")

            # Update best result if needle phase improved it
            if mse_after_needles < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                # log_callback(f"    Global improvement by needle phase (vs {best_mse_so_far:.6e}).")
                best_ep_so_far = ep_after_needles.copy()
                best_mse_so_far = mse_after_needles
                cycle_improved_globally = True
            else:
                # log_callback(f"    No global improvement by needle phase (vs {best_mse_so_far:.6e}).")
                best_ep_so_far = ep_after_needles.copy() # Still update to the result of needle phase
                best_mse_so_far = mse_after_needles

            # 2. Thinning Phase
            # log_callback(f"  [Cycle {cycle_num+1}] Thinning Phase (< {threshold_for_thin_removal:.3f} nm) + Re-Opt...")
            st.write(f"Cycle {cycle_num + 1}: Thinning Phase...")
            layers_removed_this_cycle = 0;
            max_thinning_attempts = len(best_ep_so_far) + 2 # Safety break
            for attempt in range(max_thinning_attempts):
                current_num_layers_thin = len(best_ep_so_far)
                if current_num_layers_thin <= 2:
                    # log_callback("    Structure too small (< 3 layers), stopping thinning.")
                    break

                # Attempt to remove the thinnest layer
                ep_after_single_removal, structure_changed = \
                    _perform_layer_merge_or_removal_only(best_ep_so_far, MIN_THICKNESS_PHYS_NM,
                                                         log_prefix=f"    [Thin {cycle_num+1}.{attempt+1}] ",
                                                         threshold_for_removal=threshold_for_thin_removal)

                if structure_changed and ep_after_single_removal is not None:
                    layers_removed_this_cycle += 1
                    # log_callback(f"    Layer removed/merged ({layers_removed_this_cycle} in this cycle). Re-optimizing ({len(ep_after_single_removal)} layers)...")
                    st.write(f"Cycle {cycle_num + 1}: Re-opt after removal {layers_removed_this_cycle}...")
                    # Re-optimize after removal
                    ep_after_thin_reopt, thin_reopt_success, mse_after_thin_reopt, thin_reopt_msg = \
                        _run_core_optimization(ep_after_single_removal, validated_inputs, active_targets,
                                               MIN_THICKNESS_PHYS_NM, log_prefix=f"      [ReOptThin {cycle_num+1}.{attempt+1}] ")

                    if thin_reopt_success:
                        # log_callback(f"      Re-optimization successful. MSE: {mse_after_thin_reopt:.6e}")
                        # Check if this thinning + re-opt improved the overall best MSE
                        if mse_after_thin_reopt < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                            # log_callback(f"      Global improvement by thinning+reopt (vs {best_mse_so_far:.6e}).")
                            best_ep_so_far = ep_after_thin_reopt.copy()
                            best_mse_so_far = mse_after_thin_reopt
                            cycle_improved_globally = True
                        else:
                            # log_callback(f"      No global improvement (vs {best_mse_so_far:.6e}). Continuing with this result.")
                            best_ep_so_far = ep_after_thin_reopt.copy()
                            best_mse_so_far = mse_after_thin_reopt
                    else: # Re-optimization failed after thinning
                        # log_callback(f"    WARNING: Re-optimization after thinning FAILED ({thin_reopt_msg}). Stopping thinning for this cycle.")
                        best_ep_so_far = ep_after_single_removal.copy() # Keep the thinned structure
                        # Try to recalculate MSE for this intermediate state
                        try:
                            current_mse_jax = cost_fn_penalized_jit(jnp.asarray(best_ep_so_far), *static_args)
                            best_mse_so_far = float(np.array(current_mse_jax))
                            if not np.isfinite(best_mse_so_far): best_mse_so_far = np.inf
                            # log_callback(f"      MSE after failed re-opt (reduced structure, not opt): {best_mse_so_far:.6e}")
                        except Exception as e_cost_fail:
                            # log_callback(f"      ERROR recalculating MSE after failed re-opt: {e_cost_fail}"); best_mse_so_far = np.inf
                        break # Stop thinning attempts for this cycle
                else: # No structure change from removal function
                    # log_callback("    No further layers to remove/merge in this phase.")
                    break # Stop thinning attempts for this cycle
            # log_callback(f"  [Cycle {cycle_num+1}] End Thinning Phase. {layers_removed_this_cycle} layer(s) removed.")

            num_cycles_done += 1
            # log_callback(f"--- End Auto Cycle {cycle_num + 1} --- Best current MSE: {best_mse_so_far:.6e} ({len(best_ep_so_far)} layers) ---")

            # Check for termination condition (no significant improvement in the cycle)
            if not cycle_improved_globally and best_mse_so_far >= mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE :
                # log_callback(f"No significant improvement in Cycle {cycle_num + 1} (Start: {mse_at_cycle_start:.6e}, End: {best_mse_so_far:.6e}). Stopping Auto Mode.")
                termination_reason = f"No improvement (Cycle {cycle_num + 1})"
                if best_mse_so_far > mse_at_cycle_start + MSE_IMPROVEMENT_TOLERANCE : # If MSE actually got worse, revert
                    # log_callback("  MSE increased, reverting to state before this cycle.")
                    best_ep_so_far = ep_at_cycle_start.copy()
                    best_mse_so_far = mse_at_cycle_start
                break # Exit cycle loop

        # log_callback(f"\n--- Auto Mode Finished after {num_cycles_done} cycles ---")
        # log_callback(f"Reason: {termination_reason}")
        # log_callback(f"Best final MSE: {best_mse_so_far:.6e} with {len(best_ep_so_far)} layers.")
        return best_ep_so_far, best_mse_so_far

    except (ValueError, RuntimeError, TypeError) as e:
        st.error(f"Auto Mode Error: {e}")
        # log_callback(f"FATAL ERROR during Auto Mode (Setup/Workflow): {e}")
        return None, np.inf
    except Exception as e_fatal:
        st.error(f"Unexpected Auto Mode Error: {e_fatal}")
        # log_callback(f"Unexpected FATAL ERROR during Auto Mode: {type(e_fatal).__name__}: {e_fatal}")
        traceback.print_exc()
        return None, np.inf


@jax.jit
def calculate_M_for_thickness(thickness: jnp.ndarray, n_complex_layer: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
    """Calculates the characteristic matrix for a single layer thickness."""
    eta = n_complex_layer
    safe_l_val = jnp.maximum(l_val, 1e-9)
    safe_eta = jnp.where(jnp.abs(eta) < 1e-12, 1e-12 + 0j, eta) # Avoid division by zero
    phi = (2 * jnp.pi / safe_l_val) * (n_complex_layer * thickness)
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    # Handle zero thickness case explicitly for stability
    m00 = jnp.where(thickness > 1e-12, cos_phi, 1.0)
    m01 = jnp.where(thickness > 1e-12, (1j / safe_eta) * sin_phi, 0.0)
    m10 = jnp.where(thickness > 1e-12, 1j * eta * sin_phi, 0.0)
    m11 = jnp.where(thickness > 1e-12, cos_phi, 1.0)
    M_layer = jnp.array([[m00, m01], [m10, m11]], dtype=jnp.complex128)
    return M_layer

# Vmap the single layer matrix calculation over wavelengths
calculate_M_batch_for_thickness = vmap(calculate_M_for_thickness, in_axes=(None, None, 0))


@jax.jit
def get_layer_matrices_qwot(layer_idx: int, initial_layer_number: int, l0: float,
                              nH_c_l0: jnp.ndarray, nL_c_l0: jnp.ndarray,
                              l_vec: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculates the characteristic matrices for 1xQWOT and 2xQWOT thicknesses
    for a given layer index (H or L) across a vector of wavelengths.
    """
    predicate_is_H = (layer_idx % 2 == 0)
    # Use real part of index at l0 for QWOT thickness calculation
    n_real_l0 = jax.lax.select(predicate_is_H, nH_c_l0.real, nL_c_l0.real)
    # Use complex index at l0 for the matrix calculation itself (phase includes k)
    n_complex_for_matrix = jax.lax.select(predicate_is_H, nH_c_l0, nL_c_l0)

    # Calculate 1x and 2x QWOT physical thickness
    denom = 4.0 * jnp.maximum(n_real_l0, 1e-9) # Avoid division by zero
    safe_l0 = jnp.maximum(l0, 1e-9)
    ep1_calc = 1.0 * safe_l0 / denom
    ep2_calc = 2.0 * safe_l0 / denom

    # Set thickness to 0 if n_real_l0 was invalid
    ep1 = jnp.where(n_real_l0 > 1e-9, ep1_calc, 0.0)
    ep2 = jnp.where(n_real_l0 > 1e-9, ep2_calc, 0.0)

    # Calculate matrices for both thicknesses across all wavelengths
    M_1qwot_batch = calculate_M_batch_for_thickness(ep1, n_complex_for_matrix, l_vec)
    M_2qwot_batch = calculate_M_batch_for_thickness(ep2, n_complex_for_matrix, l_vec)
    return M_1qwot_batch, M_2qwot_batch


@jax.jit
def compute_half_product(multiplier_indices: jnp.ndarray, # (N_half_layers,) bool array (0 for 1xQWOT, 1 for 2xQWOT)
                         layer_matrices_half: jnp.ndarray # (N_half_layers, 2_multipliers, L_lambdas, 2, 2)
                         ) -> jnp.ndarray: # Result shape (L_lambdas, 2, 2)
    """Computes the matrix product for one half of the stack using jax.lax.scan."""
    N_half = layer_matrices_half.shape[0] # Number of layers in this half
    L = layer_matrices_half.shape[2] # Number of lambdas

    # Initial product matrix (identity) for each lambda
    init_prod = jnp.tile(jnp.eye(2, dtype=jnp.complex128), (L, 1, 1)) # Shape (L, 2, 2)

    # Scan function to iteratively multiply layer matrices
    def multiply_step(carry_prod: jnp.ndarray, layer_idx: int) -> Tuple[jnp.ndarray, None]:
        multiplier_idx = multiplier_indices[layer_idx] # 0 (1xQWOT) or 1 (2xQWOT)
        # Select the appropriate pre-calculated matrix (1x or 2x QWOT) for this layer
        M_k = layer_matrices_half[layer_idx, multiplier_idx, :, :, :] # Shape (L, 2, 2)
        # Multiply with the carry product (vmap ensures batching over lambda)
        new_prod = vmap(jnp.matmul)(M_k, carry_prod) # Shape (L, 2, 2)
        return new_prod, None

    # Execute the scan over the layers in this half
    final_prod, _ = jax.lax.scan(multiply_step, init_prod, jnp.arange(N_half))
    return final_prod


@jax.jit
def get_T_from_batch_matrix(M_batch: jnp.ndarray, # (L_lambdas, 2, 2) or (N_comb, L_lambdas, 2, 2)
                              nSub_arr: jnp.ndarray # (L_lambdas)
                              ) -> jnp.ndarray: # Result shape (L_lambdas) or (N_comb, L_lambdas)
    """Calculates Transmittance from a batch of characteristic matrices."""
    etainc = 1.0 + 0j # Incident medium (Air)
    etasub_batch = nSub_arr # Substrate index (will broadcast if M_batch has N_comb dimension)

    # Extract matrix elements (supports batch dimension)
    m00 = M_batch[..., 0, 0]; m01 = M_batch[..., 0, 1]
    m10 = M_batch[..., 1, 0]; m11 = M_batch[..., 1, 1]

    # Calculate transmission denominator, avoid division by zero
    rs_den = (etainc * m00 + etasub_batch * m11 + etainc * etasub_batch * m01 + m10)
    rs_den_abs = jnp.abs(rs_den)
    safe_den = jnp.where(rs_den_abs < 1e-12, 1e-12 + 0j, rs_den)

    # Calculate transmission amplitude and then transmittance (Ts)
    ts = (2.0 * etainc) / safe_den
    real_etasub_batch = jnp.real(etasub_batch)
    safe_real_etainc = 1.0 # jnp.real(etainc) is just 1.0
    Ts_complex = (real_etasub_batch / safe_real_etainc) * (ts * jnp.conj(ts))
    Ts = jnp.real(Ts_complex)
    # Return 0 if denominator was near zero, handle NaNs
    return jnp.where(rs_den_abs < 1e-12, 0.0, jnp.nan_to_num(Ts, nan=0.0))


@jax.jit
def calculate_mse_basic_jax(Ts: jnp.ndarray, # (L_lambdas) or (N_comb, L_lambdas)
                            l_vec: jnp.ndarray, # (L_lambdas)
                            targets_tuple: Tuple[Tuple[float, float, float, float], ...]
                            ) -> jnp.ndarray: # scalar or (N_comb)
    """Basic MSE calculation (JITted, no penalties)."""
    total_squared_error = 0.0
    total_points_in_targets = 0

    # Iterate through target zones
    for i in range(len(targets_tuple)):
        l_min, l_max, t_min, t_max = targets_tuple[i]
        target_mask = (l_vec >= l_min) & (l_vec <= l_max) # Boolean mask for this zone

        # Calculate interpolated target T values across the lambda vector
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        interpolated_target_t = t_min + slope * (l_vec - l_min) # Shape (L_lambdas)

        # Calculate squared errors, applying the mask
        squared_errors = (Ts - interpolated_target_t)**2 # Shape (L_lambdas) or (N_comb, L_lambdas)
        masked_sq_error = jnp.where(target_mask, squared_errors, 0.0)

        # Sum errors over the lambda dimension
        total_squared_error += jnp.sum(masked_sq_error, axis=-1) # Sums over L_lambdas
        total_points_in_targets += jnp.sum(target_mask) # Count points in this target zone

    # Calculate MSE, handle division by zero
    mse = jnp.where(total_points_in_targets > 0,
                    total_squared_error / total_points_in_targets,
                    jnp.inf) # Return Inf if no target points
    # Ensure result is finite
    return jnp.nan_to_num(mse, nan=jnp.inf, posinf=jnp.inf)


@jax.jit
def combine_and_calc_mse(prod1: jnp.ndarray, prod2: jnp.ndarray, # both (L, 2, 2)
                         nSub_arr_in: jnp.ndarray, # (L)
                         l_vec_in: jnp.ndarray, targets_tuple_in: Tuple # (L)
                         ) -> jnp.ndarray: # scalar MSE
    """Combines two half-stack matrix products and calculates the MSE."""
    # Combine the two halves: M_total = M_half2 * M_half1
    M_total = vmap(jnp.matmul)(prod2, prod1) # Shape (L, 2, 2)
    # Calculate Transmittance from the total matrix
    Ts = get_T_from_batch_matrix(M_total, nSub_arr_in) # Shape (L)
    # Calculate MSE
    mse = calculate_mse_basic_jax(Ts, l_vec_in, targets_tuple_in) # scalar
    return mse


def _execute_split_stack_scan(current_l0: float, initial_layer_number: int,
                              nH_c_l0: complex, nL_c_l0: complex,
                              nSub_arr_scan: jnp.ndarray, # (L_sparse)
                              l_vec_eval_sparse_jax: jnp.ndarray, # (L_sparse)
                              active_targets_tuple: Tuple,
                              log_callback: Callable) -> Tuple[float, Optional[np.ndarray]]:
    """
    Executes the QWOT scan using the split-stack matrix multiplication approach.
    Returns the best MSE found and the corresponding QWOT multiplier array.
    """
    L_sparse = len(l_vec_eval_sparse_jax)
    num_combinations = 2**initial_layer_number
    # log_callback(f"  [Scan l0={current_l0:.2f}] Testing {num_combinations:,} QWOT combinations (1.0x/2.0x)...")

    precompute_start_time = time.time()
    # st.write(f"Scan l0={current_l0:.1f}: Pre-calculating matrices...")
    layer_matrices_list = []
    try:
        # Pre-calculate matrices for 1x and 2x QWOT for each layer
        get_layer_matrices_qwot_jit = jax.jit(get_layer_matrices_qwot)
        for i in range(initial_layer_number):
            m1, m2 = get_layer_matrices_qwot_jit(i, initial_layer_number, current_l0,
                                                 jnp.asarray(nH_c_l0), jnp.asarray(nL_c_l0),
                                                 l_vec_eval_sparse_jax)
            # Stack the 1x and 2x QWOT matrices for this layer
            layer_matrices_list.append(jnp.stack([m1, m2], axis=0)) # Shape (2, L_sparse, 2, 2)
        # Combine all layers: Shape (N_layers, 2_multipliers, L_sparse, 2, 2)
        all_layer_matrices = jnp.stack(layer_matrices_list, axis=0)
        all_layer_matrices.block_until_ready() # Ensure calculation is complete
        # log_callback(f"    Pre-calculation of matrices (l0={current_l0:.2f}) finished in {time.time() - precompute_start_time:.3f}s.")
    except Exception as e_mat:
        st.error(f"Error pre-calculating QWOT scan matrices: {e_mat}")
        return np.inf, None

    # Split stack calculation setup
    N = initial_layer_number
    N1 = N // 2 # Size of first half
    N2 = N - N1 # Size of second half
    num_comb1 = 2**N1 # Number of combinations for first half
    num_comb2 = 2**N2 # Number of combinations for second half

    # log_callback(f"    Calculating partial products 1 ({num_comb1:,} combinations)...")
    # st.write(f"Scan l0={current_l0:.1f}: Partial products 1...")
    half1_start_time = time.time()
    # Generate all binary combinations for the first half multipliers (0=1x, 1=2x)
    indices1 = jnp.arange(num_comb1)
    powers1 = 2**jnp.arange(N1)
    multiplier_indices1 = jnp.not_equal(indices1[:, None] & powers1, 0).astype(jnp.int32) # Shape (num_comb1, N1)
    matrices_half1 = all_layer_matrices[:N1] # Matrices for the first half
    # Compute all partial products for the first half using vmap
    compute_half_product_jit = jax.jit(compute_half_product)
    partial_products1 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices1, matrices_half1) # Shape (num_comb1, L, 2, 2)
    partial_products1.block_until_ready()
    # log_callback(f"    Partial products 1 finished in {time.time() - half1_start_time:.3f}s.")

    # log_callback(f"    Calculating partial products 2 ({num_comb2:,} combinations)...")
    # st.write(f"Scan l0={current_l0:.1f}: Partial products 2...")
    half2_start_time = time.time()
    # Generate all binary combinations for the second half multipliers
    indices2 = jnp.arange(num_comb2)
    powers2 = 2**jnp.arange(N2)
    multiplier_indices2 = jnp.not_equal(indices2[:, None] & powers2, 0).astype(jnp.int32) # Shape (num_comb2, N2)
    matrices_half2 = all_layer_matrices[N1:] # Matrices for the second half
    # Compute all partial products for the second half using vmap
    partial_products2 = vmap(compute_half_product_jit, in_axes=(0, None))(multiplier_indices2, matrices_half2) # Shape (num_comb2, L, 2, 2)
    partial_products2.block_until_ready()
    # log_callback(f"    Partial products 2 finished in {time.time() - half2_start_time:.3f}s.")

    # log_callback(f"    Combining and calculating MSE ({num_comb1 * num_comb2:,} total)...")
    # st.write(f"Scan l0={current_l0:.1f}: Combining & MSE...")
    combine_start_time = time.time()
    # JIT the combination and MSE calculation function
    combine_and_calc_mse_jit = jax.jit(combine_and_calc_mse)

    # Use nested vmap to efficiently combine all pairs of half-products and calculate MSE
    vmap_inner = vmap(combine_and_calc_mse_jit, in_axes=(None, 0, None, None, None)) # vmap over partial_products2
    vmap_outer = vmap(vmap_inner, in_axes=(0, None, None, None, None))    # vmap over partial_products1

    # Execute the nested vmap
    all_mses_nested = vmap_outer(partial_products1, partial_products2, nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple) # Shape (num_comb1, num_comb2)
    all_mses_nested.block_until_ready()
    # log_callback(f"    Combination and MSE finished in {time.time() - combine_start_time:.3f}s.")

    # Find the best combination (minimum MSE)
    all_mses_flat = all_mses_nested.reshape(-1) # Flatten the MSE results
    best_idx_flat = jnp.argmin(all_mses_flat) # Find index of minimum MSE
    current_best_mse = float(all_mses_flat[best_idx_flat])

    if not np.isfinite(current_best_mse):
        # log_callback(f"    Warning: No valid result (finite MSE) found for l0={current_l0:.2f}.")
        return np.inf, None

    # Convert flat index back to indices for each half
    best_idx_half1, best_idx_half2 = jnp.unravel_index(best_idx_flat, (num_comb1, num_comb2))

    # Get the multiplier indices (0 or 1) for the best combination
    best_indices_h1 = multiplier_indices1[best_idx_half1] # Shape (N1,)
    best_indices_h2 = multiplier_indices2[best_idx_half2] # Shape (N2,)

    # Convert indices (0/1) to actual multipliers (1.0/2.0)
    best_multipliers_h1 = 1.0 + best_indices_h1.astype(jnp.float64)
    best_multipliers_h2 = 1.0 + best_indices_h2.astype(jnp.float64)

    # Concatenate the multipliers for the full stack
    current_best_multipliers = jnp.concatenate([best_multipliers_h1, best_multipliers_h2]) # Shape (N,)
    # log_callback(f"    Best MSE for scan l0={current_l0:.2f}: {current_best_mse:.6e}")
    return current_best_mse, np.array(current_best_multipliers)


# --- UI Helper Functions ---

def add_log(message: Union[str, List[str]]):
    """Dummy function to replace logging calls."""
    pass # No operation

def get_material_input(role: str) -> Tuple[Optional[MaterialInputType], str]:
    """Gets the selected material definition (constant, named, or raw) from session state."""
    # Determine session state keys based on role (H, L, Sub)
    if role == 'H':
        sel_key, const_r_key, const_i_key = "selected_H", "nH_r", "nH_i"
    elif role == 'L':
        sel_key, const_r_key, const_i_key = "selected_L", "nL_r", "nL_i"
    elif role == 'Sub':
        sel_key, const_r_key, const_i_key = "selected_Sub", "nSub_r", None # Substrate usually has k=0 assumed
    else:
        st.error(f"Unknown material role requested: {role}")
        return None, "Role Error"

    selection = st.session_state.get(sel_key)
    if selection == "Constant":
        # Retrieve constant n and k values from session state
        n_real = st.session_state.get(const_r_key, 1.0 if role != 'Sub' else 1.5)
        n_imag = 0.0
        if const_i_key and role in ['H', 'L']: # Only H and L can have non-zero k here
           n_imag = st.session_state.get(const_i_key, 0.0)

        # Validate and adjust constant values
        valid_n, valid_k = True, True
        if n_real <= 0:
            n_real = 1.0; valid_n = False
        if n_imag < 0:
            n_imag = 0.0; valid_k = False

        # Format representation string
        mat_repr = f"Constant ({n_real:.3f}{'+' if n_imag>=0 else ''}{n_imag:.3f}j)"
        if not valid_n or not valid_k: mat_repr += " (Adjusted)"
        return complex(n_real, n_imag), mat_repr
    elif isinstance(selection, str) and selection: # Material selected from Excel list
        return selection, selection
    else: # Error case: invalid selection
        st.error(f"Material selection for '{role}' invalid or missing in session_state.")
        return None, "Selection Error"

def validate_targets() -> Optional[List[Dict]]:
    """Validates the spectral targets defined in the UI."""
    active_targets = []
    is_valid = True
    if 'targets' not in st.session_state or not isinstance(st.session_state.targets, list):
        st.error("Internal error: Target list missing or invalid in session_state.")
        return None

    # Iterate through targets defined in the UI
    for i, target_state in enumerate(st.session_state.targets):
        if target_state.get('enabled', False): # Only consider enabled targets
            try:
                # Extract and convert target values
                l_min = float(target_state['min'])
                l_max = float(target_state['max'])
                t_min = float(target_state['target_min'])
                t_max = float(target_state['target_max'])

                # Validate target values
                if l_max < l_min:
                    st.warning(f"Target {i+1} Error: λ max ({l_max:.1f}) < λ min ({l_min:.1f}). Target ignored.")
                    is_valid = False; continue
                if not (0.0 <= t_min <= 1.0 and 0.0 <= t_max <= 1.0):
                    st.warning(f"Target {i+1} Error: Transmittance out of [0, 1] (Tmin={t_min:.2f}, Tmax={t_max:.2f}). Target ignored.")
                    is_valid = False; continue

                # Add valid target to the list
                active_targets.append({
                    'min': l_min, 'max': l_max,
                    'target_min': t_min, 'target_max': t_max
                })
            except (KeyError, ValueError, TypeError) as e:
                st.warning(f"Target {i+1} Error: Missing or invalid data ({e}). Target ignored.")
                is_valid = False; continue

    if not is_valid:
        st.warning("Errors exist in some active spectral target definitions. Invalid targets are ignored.")
        # Return the list of valid targets found so far, even if some were invalid
        return active_targets
    elif not active_targets:
        # st.info("No spectral targets are enabled.") # Inform user if no targets are active
        return []
    else:
        return active_targets
def get_lambda_range_from_targets(validated_targets: Optional[List[Dict]]) -> Tuple[Optional[float], Optional[float]]:
    """Determines the overall min and max lambda from a list of validated targets."""
    overall_min, overall_max = None, None
    if validated_targets: # Check if the list is not None and not empty
        all_mins = [t['min'] for t in validated_targets]
        all_maxs = [t['max'] for t in validated_targets]
        if all_mins: overall_min = min(all_mins)
        if all_maxs: overall_max = max(all_maxs)
    return overall_min, overall_max

def clear_optimized_state():
    """Resets session state variables related to optimization results."""
    st.session_state.optimized_ep = None
    st.session_state.is_optimized_state = False
    st.session_state.ep_history = deque(maxlen=5) # Keep history limited
    st.session_state.optimized_qwot_str = ""
    st.session_state.last_mse = None

def set_optimized_as_nominal_wrapper():
    """Callback to set the current optimized structure as the new nominal QWOT structure."""
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

        # Recalculate QWOT multipliers from the optimized physical thicknesses
        optimized_qwots = calculate_qwot_from_ep(st.session_state.optimized_ep, l0, nH_mat, nL_mat, EXCEL_FILE_PATH)
        if optimized_qwots is None:
            st.error("Error recalculating QWOT from the optimized structure.")
            return

        if np.any(np.isnan(optimized_qwots)):
            st.warning("Recalculated QWOT contains NaNs (likely invalid index at l0). Nominal QWOT not updated.")
        else:
            # Update the nominal QWOT string in session state
            new_qwot_str = ",".join([f"{q:.6f}" for q in optimized_qwots]) # Use higher precision
            st.session_state.current_qwot = new_qwot_str
            st.success("Optimized structure set as new Nominal (QWOT updated).")
            clear_optimized_state() # Clear the 'optimized' state flags
    except Exception as e:
        st.error(f"Unexpected error setting optimized as nominal: {e}")
        traceback.print_exc() # Print traceback for debugging

def undo_remove_wrapper():
    """Callback to undo the last 'Remove Thin Layer' action."""
    if not st.session_state.get('ep_history'):
        st.info("Undo history is empty.")
        return

    try:
        # Restore the previous state from history
        last_ep = st.session_state.ep_history.pop()
        st.session_state.optimized_ep = last_ep.copy()
        st.session_state.is_optimized_state = True # Mark state as optimized (even if reverted)

        # Recalculate QWOT string for the restored state for display purposes
        l0 = st.session_state.l0
        nH_mat, _ = get_material_input('H')
        nL_mat, _ = get_material_input('L')
        if nH_mat is not None and nL_mat is not None:
            qwots_recalc = calculate_qwot_from_ep(last_ep, l0, nH_mat, nL_mat, EXCEL_FILE_PATH)
            if qwots_recalc is not None and not np.any(np.isnan(qwots_recalc)):
                st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_recalc])
            else:
                st.session_state.optimized_qwot_str = "QWOT N/A (after undo)"
        else:
            st.session_state.optimized_qwot_str = "QWOT Material Error (after undo)"

        st.info("State restored. Recalculating results...")
        # Trigger a recalculation of the results for the restored state
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {
            'is_optimized_run': True,
            'method_name': "Optimized (Undo)",
            'force_ep': st.session_state.optimized_ep # Force calculation with the restored ep
            }
    except IndexError:
        st.warning("Undo history is empty (internal error?).")
    except Exception as e:
        st.error(f"Unexpected error during undo: {e}")
        traceback.print_exc()
        clear_optimized_state() # Clear state on error

# --- Calculation Wrappers (Called by UI Buttons) ---

def run_calculation_wrapper(is_optimized_run: bool, method_name: str = "", force_ep: Optional[np.ndarray] = None):
    """Wrapper function to run T(lambda) calculation and update results."""
    calc_type = 'Optimized' if is_optimized_run else 'Nominal'
    st.session_state.last_calc_results = {} # Clear previous results
    st.session_state.last_mse = None

    with st.spinner(f"{calc_type} calculation in progress..."):
        try:
            # Validate targets and determine lambda range for plotting/MSE
            active_targets = validate_targets()
            if active_targets is None: # Validation function returned None -> error occurred
                 st.error("Target definition invalid. Calculation aborted.")
                 return
            if not active_targets: # No targets enabled
                 st.warning("No active targets. Default lambda range used (400-700nm). MSE calculation will be N/A.")
                 l_min_plot, l_max_plot = 400.0, 700.0
            else: # Determine range from targets
                 l_min_plot, l_max_plot = get_lambda_range_from_targets(active_targets)
                 if l_min_plot is None or l_max_plot is None or l_max_plot < l_min_plot:
                     st.error("Could not determine a valid lambda range from targets. Calculation aborted.")
                     return

            # Gather validated inputs from session state
            validated_inputs = {
                'l0': st.session_state.l0,
                'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot,
                'auto_thin_threshold': st.session_state.auto_thin_threshold, # Needed for re-opt calls
                'l_range_deb': l_min_plot, # Use plot range for consistency
                'l_range_fin': l_max_plot,
            }
            nH_mat, nH_repr = get_material_input('H')
            nL_mat, nL_repr = get_material_input('L')
            nSub_mat, nSub_repr = get_material_input('Sub')
            if nH_mat is None or nL_mat is None or nSub_mat is None:
                st.error("Material definition error. Calculation aborted.")
                return
            validated_inputs['nH_material'] = nH_mat
            validated_inputs['nL_material'] = nL_mat
            validated_inputs['nSub_material'] = nSub_mat

            # Determine the ep_vector to use for calculation
            ep_to_calculate = None
            if force_ep is not None: # Use forced ep (e.g., after undo)
                ep_to_calculate = force_ep.copy()
            elif is_optimized_run and st.session_state.get('optimized_ep') is not None: # Use current optimized state
                ep_to_calculate = st.session_state.optimized_ep.copy()
            else: # Use nominal structure derived from QWOT
                emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
                if not emp_list: # Handle empty QWOT string (bare substrate)
                    ep_to_calculate = np.array([], dtype=np.float64)
                else:
                    ep_calc = calculate_initial_ep(emp_list, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                    if ep_calc is None:
                        st.error("Failed to calculate initial thicknesses from QWOT. Calculation aborted.")
                        return
                    ep_to_calculate = ep_calc.copy()

            st.session_state.current_ep = ep_to_calculate.copy() if ep_to_calculate is not None else None

            # Generate lambda vector for fine plot
            num_plot_points = max(501, int(np.round((l_max_plot - l_min_plot) / validated_inputs['l_step'])) * 3 + 1) # Ensure enough points for smooth plot
            l_vec_plot_fine_np = np.linspace(l_min_plot, l_max_plot, num_plot_points)
            l_vec_plot_fine_np = l_vec_plot_fine_np[(l_vec_plot_fine_np > 0) & np.isfinite(l_vec_plot_fine_np)]
            if not l_vec_plot_fine_np.size:
                st.error("Could not generate a valid lambda vector for plotting. Calculation aborted.")
                return

            # --- Perform the main T(lambda) calculation ---
            results_fine = calculate_T_from_ep_jax(
                ep_to_calculate, nH_mat, nL_mat, nSub_mat, l_vec_plot_fine_np, EXCEL_FILE_PATH
            )
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

            # Calculate MSE for display if targets are active
            if active_targets:
                # Use the optimization grid lambda vector for MSE calculation
                num_pts_optim = max(2, int(np.round((l_max_plot - l_min_plot) / validated_inputs['l_step'])) + 1)
                l_vec_optim_np = np.geomspace(l_min_plot, l_max_plot, num_pts_optim)
                l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
                if l_vec_optim_np.size > 0:
                    # Calculate T on the optimization grid
                    results_optim_grid = calculate_T_from_ep_jax(
                        ep_to_calculate, nH_mat, nL_mat, nSub_mat, l_vec_optim_np, EXCEL_FILE_PATH
                    )
                    if results_optim_grid is not None:
                        # Calculate MSE using the results on the optim grid
                        mse_display, num_pts_mse = calculate_final_mse(results_optim_grid, active_targets)
                        st.session_state.last_mse = mse_display
                        st.session_state.last_calc_results['res_optim_grid'] = results_optim_grid # Store grid results for plot markers
                    else:
                        st.session_state.last_mse = None # Failed T calc on optim grid
                else:
                    st.session_state.last_mse = None # Optim grid empty
            else:
                 st.session_state.last_mse = None # No active targets

            # Update state flags
            st.session_state.is_optimized_state = is_optimized_run
            if not is_optimized_run: # If it was a nominal calculation, clear any previous optimized state
                clear_optimized_state()
                st.session_state.current_ep = ep_to_calculate.copy() if ep_to_calculate is not None else None

            st.success(f"{calc_type} calculation finished.")

        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during {calc_type} calculation: {e}")
            traceback.print_exc()
        except Exception as e_fatal:
             st.error(f"Unexpected error during {calc_type} calculation: {e_fatal}")
             traceback.print_exc()


def run_local_optimization_wrapper():
    """Wrapper for the Local Optimization action."""
    st.session_state.last_calc_results = {} # Clear previous results
    st.session_state.last_mse = None
    clear_optimized_state() # Start fresh optimization

    with st.spinner("Local optimization in progress..."):
        try:
            # Validate targets and setup inputs
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

            # Get starting structure from nominal QWOT
            emp_list = [float(e.strip()) for e in validated_inputs['emp_str'].split(',') if e.strip()]
            if not emp_list:
                st.error("Nominal QWOT empty, cannot start local optimization.")
                return

            ep_start = calculate_initial_ep(emp_list, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
            if ep_start is None:
                st.error("Failed initial thickness calculation for local optimization.")
                return

            # Run the core optimization
            final_ep, success, final_cost, msg = \
                _run_core_optimization(ep_start, validated_inputs, active_targets,
                                       MIN_THICKNESS_PHYS_NM, log_prefix="  [Opt Local] ")

            # Process results
            if success and final_ep is not None:
                st.session_state.optimized_ep = final_ep.copy()
                st.session_state.current_ep = final_ep.copy() # Update current ep to optimized
                st.session_state.is_optimized_state = True
                st.session_state.last_mse = final_cost
                # Calculate QWOT string for display
                qwots_opt = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                    st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                else:
                    st.session_state.optimized_qwot_str = "QWOT N/A"
                st.success(f"Local optimization finished ({msg}). MSE: {final_cost:.4e}")
                # Trigger recalculation to update plots
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Opt Local"}
            else: # Optimization failed
                st.error(f"Local optimization failed: {msg}")
                st.session_state.is_optimized_state = False # Revert to nominal state conceptually
                st.session_state.optimized_ep = None
                st.session_state.current_ep = ep_start.copy() # Keep the starting nominal ep
                st.session_state.last_mse = None

        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during local optimization: {e}")
            traceback.print_exc()
            clear_optimized_state()
        except Exception as e_fatal:
             st.error(f"Unexpected error during local optimization: {e_fatal}")
             traceback.print_exc()
             clear_optimized_state()


def run_scan_optimization_wrapper():
    """Wrapper for the QWOT Scan + Optimization action."""
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None
    clear_optimized_state()

    with st.spinner("QWOT Scan + Optimization in progress (can be long)..."):
        try:
            # Define initial layer number based on current QWOT if not set
            if 'initial_layer_number' not in st.session_state:
                 st.session_state.initial_layer_number = len([q for q in st.session_state.current_qwot.split(',') if q.strip()])
                 if st.session_state.initial_layer_number == 0:
                     st.error("Nominal QWOT is empty. Define a structure first.")
                     return

            # Validate targets and setup inputs
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
                'emp_str': st.session_state.current_qwot, # Needed for materials
                'initial_layer_number': st.session_state.initial_layer_number,
                'auto_thin_threshold': st.session_state.auto_thin_threshold, # Needed for re-opt
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

            # Prepare lambda grid for scan (sparse) and substrate indices
            l_vec_eval_full_np = np.geomspace(l_min_opt, l_max_opt, max(2, int(np.round((l_max_opt - l_min_opt) / validated_inputs['l_step'])) + 1))
            l_vec_eval_full_np = l_vec_eval_full_np[(l_vec_eval_full_np > 0) & np.isfinite(l_vec_eval_full_np)]
            if not l_vec_eval_full_np.size: raise ValueError("Failed lambda generation for Scan.")
            l_vec_eval_sparse_np = l_vec_eval_full_np[::2] # Use sparser grid for speed
            if not l_vec_eval_sparse_np.size: raise ValueError("Failed sparse lambda generation for Scan.")
            l_vec_eval_sparse_jax = jnp.asarray(l_vec_eval_sparse_np)

            nSub_arr_scan = _get_nk_array_for_lambda_vec(nSub_mat, l_vec_eval_sparse_jax, EXCEL_FILE_PATH)
            if nSub_arr_scan is None: raise RuntimeError("Failed substrate index preparation for scan.")

            active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)

            # Define l0 values to test (nominal +/- 20%)
            l0_nominal = validated_inputs['l0']
            l0_values_to_test = sorted(list(set([l0_nominal, l0_nominal * 1.2, l0_nominal * 0.8])))
            l0_values_to_test = [l for l in l0_values_to_test if l > 1e-6] # Ensure positive l0

            # --- Execute Scan for each l0 ---
            initial_candidates = []
            for l0_scan in l0_values_to_test:
                st.write(f"Scanning for l0={l0_scan:.1f}...")
                try:
                    # Get indices at the current l0_scan
                    nH_c_l0 = _get_nk_at_lambda(nH_mat, l0_scan, EXCEL_FILE_PATH)
                    nL_c_l0 = _get_nk_at_lambda(nL_mat, l0_scan, EXCEL_FILE_PATH)
                    if nH_c_l0 is None or nL_c_l0 is None:
                        st.warning(f"H/L indices not found for l0={l0_scan:.2f}. Scan for this l0 skipped.")
                        continue

                    # Execute the split-stack scan
                    scan_mse, scan_multipliers = _execute_split_stack_scan(
                        l0_scan, validated_inputs['initial_layer_number'],
                        nH_c_l0, nL_c_l0,
                        nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple,
                        add_log # Pass dummy log function
                    )
                    # Store valid candidates
                    if scan_multipliers is not None and np.isfinite(scan_mse):
                        initial_candidates.append({
                            'l0': l0_scan,
                            'mse_scan': scan_mse,
                            'multipliers': scan_multipliers
                        })
                except Exception as e_scan_l0:
                    st.warning(f"Error during scan for l0={l0_scan:.2f}: {e_scan_l0}")

            if not initial_candidates:
                st.error("QWOT Scan found no valid initial candidates.")
                return

            # --- Optimize the Best Candidate ---
            st.write("Local optimization of best scan candidate...")
            initial_candidates.sort(key=lambda c: c['mse_scan']) # Sort by scan MSE
            best_candidate = initial_candidates[0]

            # Calculate initial thickness for the best candidate
            ep_start_optim = calculate_initial_ep(best_candidate['multipliers'], best_candidate['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
            if ep_start_optim is None:
                st.error("Failed thickness calculation for best scan candidate.")
                return

            # Run core optimization on the best candidate
            final_ep, success, final_cost, msg = \
                _run_core_optimization(ep_start_optim, validated_inputs, active_targets,
                                       MIN_THICKNESS_PHYS_NM, log_prefix="  [Opt Scan Cand] ")

            # Process final result
            if success and final_ep is not None:
                st.session_state.optimized_ep = final_ep.copy()
                st.session_state.current_ep = final_ep.copy()
                st.session_state.is_optimized_state = True
                st.session_state.last_mse = final_cost
                st.session_state.l0 = best_candidate['l0'] # Update l0 in UI to the best one found
                # Calculate QWOT string for display
                qwots_opt = calculate_qwot_from_ep(final_ep, best_candidate['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                    st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                else: st.session_state.optimized_qwot_str = "QWOT N/A"
                st.success(f"QWOT Scan + Optimization finished ({msg}). Final MSE: {final_cost:.4e}")
                # Trigger recalculation to update plots
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Scan+Opt (l0={best_candidate['l0']:.1f})"}
            else:
                st.error(f"Local optimization after scan failed: {msg}")
                clear_optimized_state()

        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during QWOT Scan + Optimization: {e}")
            traceback.print_exc()
            clear_optimized_state()
        except Exception as e_fatal:
             st.error(f"Unexpected error during QWOT Scan + Optimization: {e_fatal}")
             traceback.print_exc()
             clear_optimized_state()


def run_auto_mode_wrapper():
    """Wrapper for the Auto Mode action."""
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None
    # Don't clear optimized state here, Auto Mode can start from previous optimization

    with st.spinner("Automatic Mode in progress (can be very long)..."):
        try:
            # Validate targets and setup inputs
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

            # Determine starting point: previous optimized state or nominal
            ep_start_auto = None
            if st.session_state.get('is_optimized_state') and st.session_state.get('optimized_ep') is not None:
                ep_start_auto = st.session_state.optimized_ep.copy()

            # Run the main auto mode function
            final_ep, final_mse = run_auto_mode(
                initial_ep=ep_start_auto,
                validated_inputs=validated_inputs,
                active_targets=active_targets,
                excel_file_path=EXCEL_FILE_PATH,
                log_callback=add_log # Pass dummy log function
            )

            # Process results
            if final_ep is not None and np.isfinite(final_mse):
                st.session_state.optimized_ep = final_ep.copy()
                st.session_state.current_ep = final_ep.copy()
                st.session_state.is_optimized_state = True
                st.session_state.last_mse = final_mse
                # Calculate QWOT string for display
                qwots_opt = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                    st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                else: st.session_state.optimized_qwot_str = "QWOT N/A"
                st.success(f"Auto Mode finished. Final MSE: {final_mse:.4e}")
                # Trigger recalculation
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Auto Mode"}
            else:
                st.error("Automatic Mode failed or did not produce a valid result.")
                clear_optimized_state() # Clear state on failure
                # Trigger nominal recalc to show starting point again
                st.session_state.needs_rerun_calc = True
                st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (After Auto Fail)"}

        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during Auto Mode: {e}")
            traceback.print_exc()
        except Exception as e_fatal:
             st.error(f"Unexpected error during Auto Mode: {e_fatal}")
             traceback.print_exc()


def run_remove_thin_wrapper():
    """Wrapper for the 'Remove Thin Layer + Re-opt' action."""
    st.session_state.last_calc_results = {}
    st.session_state.last_mse = None

    # Check if there is an optimized state to modify
    if not st.session_state.get('is_optimized_state') or st.session_state.get('optimized_ep') is None:
        st.error("This action requires an existing optimized structure. Run an optimization first.")
        return

    current_ep_optim = st.session_state.optimized_ep.copy()
    if len(current_ep_optim) <= 2:
        st.error("Structure too small (<= 2 layers) for removal/merge.")
        return

    with st.spinner("Removing thin layer + Re-optimizing..."):
        try:
            # Save current state to history for undo
            st.session_state.ep_history.append(current_ep_optim)

            # Validate targets and setup inputs for re-optimization
            active_targets = validate_targets()
            if active_targets is None or not active_targets:
                st.session_state.ep_history.pop() # Remove saved state if cannot proceed
                st.error("Removal aborted: invalid or missing targets for re-optimization.")
                return

            l_min_opt, l_max_opt = get_lambda_range_from_targets(active_targets)
            if l_min_opt is None:
                st.session_state.ep_history.pop()
                st.error("Removal aborted: invalid lambda range for re-optimization.")
                return

            validated_inputs = {
                'l0': st.session_state.l0, 'l_step': st.session_state.l_step,
                'emp_str': st.session_state.current_qwot, # Needed for material loading
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

            # Perform the layer removal/merge operation
            ep_after_removal, structure_changed = _perform_layer_merge_or_removal_only(
                current_ep_optim, MIN_THICKNESS_PHYS_NM,
                log_prefix="  [Remove] ",
                threshold_for_removal=None # Find thinnest overall layer >= min_phys
            )

            if structure_changed and ep_after_removal is not None:
                st.write("Re-optimizing after removal...")
                # Re-optimize the modified structure
                final_ep, success, final_cost, msg = \
                    _run_core_optimization(ep_after_removal, validated_inputs, active_targets,
                                           MIN_THICKNESS_PHYS_NM, log_prefix="  [ReOpt Thin] ")

                if success and final_ep is not None:
                    # Update state with successfully re-optimized structure
                    st.session_state.optimized_ep = final_ep.copy()
                    st.session_state.current_ep = final_ep.copy()
                    st.session_state.is_optimized_state = True
                    st.session_state.last_mse = final_cost
                    # Update QWOT string
                    qwots_opt = calculate_qwot_from_ep(final_ep, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                    if qwots_opt is not None and not np.any(np.isnan(qwots_opt)):
                        st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_opt])
                    else: st.session_state.optimized_qwot_str = "QWOT N/A"
                    st.success(f"Removal + Re-optimization finished ({msg}). Final MSE: {final_cost:.4e}")
                    # Trigger plot update
                    st.session_state.needs_rerun_calc = True
                    st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': f"Optimized (Post-Remove)"}
                else: # Re-optimization failed
                    st.warning(f"Layer removed, but re-optimization failed ({msg}). State is AFTER removal but BEFORE failed re-opt attempt.")
                    # Keep the state after removal, but before the failed re-opt
                    st.session_state.optimized_ep = ep_after_removal.copy()
                    st.session_state.current_ep = ep_after_removal.copy()
                    st.session_state.is_optimized_state = True # Still consider it 'optimized'
                    # Try to calculate MSE and QWOT for this intermediate state
                    try:
                        l_min_opt, l_max_opt = validated_inputs['l_range_deb'], validated_inputs['l_range_fin']
                        l_step_optim = validated_inputs['l_step']
                        num_pts_optim = max(2, int(np.round((l_max_opt - l_min_opt) / l_step_optim)) + 1)
                        l_vec_optim_np = np.geomspace(l_min_opt, l_max_opt, num_pts_optim)
                        l_vec_optim_np = l_vec_optim_np[(l_vec_optim_np > 0) & np.isfinite(l_vec_optim_np)]
                        if l_vec_optim_np.size > 0:
                            results_fail_grid = calculate_T_from_ep_jax(ep_after_removal, nH_mat, nL_mat, nSub_mat, l_vec_optim_np, EXCEL_FILE_PATH)
                            if results_fail_grid:
                                mse_fail, _ = calculate_final_mse(results_fail_grid, active_targets)
                                st.session_state.last_mse = mse_fail
                        qwots_fail = calculate_qwot_from_ep(ep_after_removal, validated_inputs['l0'], nH_mat, nL_mat, EXCEL_FILE_PATH)
                        if qwots_fail is not None and not np.any(np.isnan(qwots_fail)):
                            st.session_state.optimized_qwot_str = ", ".join([f"{q:.3f}" for q in qwots_fail])
                        else: st.session_state.optimized_qwot_str = "QWOT N/A (ReOpt Fail)"
                    except Exception as e_recalc:
                        st.session_state.last_mse = None
                        st.session_state.optimized_qwot_str = "Recalc Error"
                    # Trigger plot update for the intermediate state
                    st.session_state.needs_rerun_calc = True
                    st.session_state.rerun_calc_params = {'is_optimized_run': True, 'method_name': "Optimized (Post-Remove, Re-Opt Fail)"}
            else: # No structure change occurred during removal attempt
                st.info("No layer was removed (criteria not met or structure too small).")
                try:
                    st.session_state.ep_history.pop() # Remove unnecessary saved state
                except IndexError: pass # Should not happen

        except (ValueError, RuntimeError, TypeError) as e:
            st.error(f"Error during Thin Layer Removal: {e}")
            traceback.print_exc()
            # Attempt to restore from history if possible
            if st.session_state.ep_history:
                try: st.session_state.ep_history.pop() # Remove the state added at the start of this failed attempt
                except IndexError: pass
        except Exception as e_fatal:
             st.error(f"Unexpected error during Thin Layer Removal: {e_fatal}")
             traceback.print_exc()
             # Attempt to restore from history
             if st.session_state.ep_history:
                 try: st.session_state.ep_history.pop()
                 except IndexError: pass


# --- Streamlit UI Definition ---
st.set_page_config(layout="wide", page_title="Thin Film Optimizer")
st.title("🔬 Thin Film Optimizer (Streamlit + JAX)")
# st.markdown("""*Streamlit conversion of the Tkinter tool. Focuses on H/L calculations.*""") # Optional description

# --- Initialize Session State ---
if 'init_done' not in st.session_state:
    # Initialize state variables only once
    st.session_state.current_ep = None # Holds nominal ep or optimized ep
    st.session_state.current_qwot = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1" # Default QWOT
    st.session_state.optimized_ep = None # Holds result of optimization/modification
    st.session_state.is_optimized_state = False # Flag: True if current_ep holds optimized result
    st.session_state.optimized_qwot_str = "" # QWOT string for optimized_ep
    st.session_state.material_sequence = None # Not used in H/L mode currently
    st.session_state.ep_history = deque(maxlen=5) # Undo history
    st.session_state.last_mse = None # Last calculated MSE
    st.session_state.needs_rerun_calc = False # Flag to trigger recalc
    st.session_state.rerun_calc_params = {} # Params for triggered recalc
    st.session_state.calculating = False # Flag to prevent re-entry during calculation

    # Load materials from Excel
    try:
        mats = get_available_materials_from_excel(EXCEL_FILE_PATH)
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

    # Default material selections
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
    st.session_state.nH_r = 2.35; st.session_state.nH_i = 0.0
    st.session_state.nL_r = 1.46; st.session_state.nL_i = 0.0
    st.session_state.nSub_r = 1.52

    st.session_state.init_done = True
    st.session_state.needs_rerun_calc = True # Trigger initial calculation
    st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Initial Load"}

# --- Callback for Parameter Changes ---
def trigger_nominal_recalc():
    """Callback function to trigger a recalculation when nominal parameters change."""
    if not st.session_state.get('calculating', False): # Avoid re-entry
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {
            'is_optimized_run': False, # Always recalculate nominal state
            'method_name': "Nominal (Param Update)",
            'force_ep': None # Use current QWOT
        }

# --- Sidebar Definition ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- Materials Section ---
    st.subheader("Materials")
    st.session_state.selected_H = st.selectbox(
        "Material H", options=st.session_state.available_materials,
        key="selected_H", # Use key directly
        on_change=trigger_nominal_recalc
    )
    if st.session_state.selected_H == "Constant":
        hc1, hc2 = st.columns(2)
        st.session_state.nH_r = hc1.number_input("n' H", value=st.session_state.nH_r, format="%.4f", key="nH_r", on_change=trigger_nominal_recalc)
        st.session_state.nH_i = hc2.number_input("k H", value=st.session_state.nH_i, min_value=0.0, format="%.4f", key="nH_i", on_change=trigger_nominal_recalc)

    st.session_state.selected_L = st.selectbox(
        "Material L", options=st.session_state.available_materials,
        key="selected_L",
        on_change=trigger_nominal_recalc
    )
    if st.session_state.selected_L == "Constant":
        lc1, lc2 = st.columns(2)
        st.session_state.nL_r = lc1.number_input("n' L", value=st.session_state.nL_r, format="%.4f", key="nL_r", on_change=trigger_nominal_recalc)
        st.session_state.nL_i = lc2.number_input("k L", value=st.session_state.nL_i, min_value=0.0, format="%.4f", key="nL_i", on_change=trigger_nominal_recalc)

    st.session_state.selected_Sub = st.selectbox(
        "Substrate", options=st.session_state.available_substrates,
        key="selected_Sub",
        on_change=trigger_nominal_recalc
    )
    if st.session_state.selected_Sub == "Constant":
        st.session_state.nSub_r = st.number_input("n' Substrate", value=st.session_state.nSub_r, format="%.4f", key="nSub_r", on_change=trigger_nominal_recalc)

    if st.button("🔄 Reload Excel Materials", key="reload_mats_sidebar"):
        st.cache_data.clear() # Clear cache for material loading
        try:
            mats = get_available_materials_from_excel(EXCEL_FILE_PATH)
            st.session_state.available_materials = sorted(list(set(["Constant"] + mats)))
            base_subs = ["Constant", "Fused Silica", "BK7", "D263"]
            st.session_state.available_substrates = sorted(list(set(base_subs + st.session_state.available_materials)))
            # Reset selections if they are no longer available
            if st.session_state.selected_H not in st.session_state.available_materials: st.session_state.selected_H = "Constant"
            if st.session_state.selected_L not in st.session_state.available_materials: st.session_state.selected_L = "Constant"
            if st.session_state.selected_Sub not in st.session_state.available_substrates: st.session_state.selected_Sub = "Constant"
            st.rerun() # Rerun to update selectbox options
        except Exception as e:
            st.error(f"Error reloading materials: {e}")

    st.divider()

    # --- Nominal Structure Section ---
    st.subheader("Nominal Structure")
    st.session_state.current_qwot = st.text_area(
        "Nominal QWOT (multipliers separated by ',')",
        value=st.session_state.current_qwot,
        key="qwot_input_sidebar",
        on_change=clear_optimized_state # Changing nominal QWOT clears optimized state
    )
    num_layers_from_qwot = len([q for q in st.session_state.current_qwot.split(',') if q.strip()])
    st.caption(f"Layers from QWOT: {num_layers_from_qwot}")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.session_state.l0 = c1.number_input("λ₀ Center (nm)", value=st.session_state.l0, min_value=1.0, format="%.2f", key="l0_input_sidebar", on_change=trigger_nominal_recalc)
    with c2:
        init_layers_num = st.number_input("N layers:", min_value=0, value=num_layers_from_qwot, step=1, key="init_layers_gen_num_sidebar", label_visibility="collapsed")
        if st.button("Generate 1s", key="gen_qwot_btn_sidebar", use_container_width=True):
            if init_layers_num > 0:
                new_qwot = ",".join(['1'] * init_layers_num)
                if new_qwot != st.session_state.current_qwot:
                    st.session_state.current_qwot = new_qwot
                    clear_optimized_state()
                    trigger_nominal_recalc() # Trigger recalc after generating
                    st.rerun()
            elif st.session_state.current_qwot != "": # Clear if N=0
                st.session_state.current_qwot = ""
                clear_optimized_state()
                trigger_nominal_recalc()
                st.rerun()

    st.divider()

    # --- Targets Section ---
    st.subheader("🎯 Spectral Targets (T)")
    st.session_state.l_step = st.number_input("λ Step (nm) (Opt/MSE)", value=st.session_state.l_step, min_value=0.1, format="%.2f", key="l_step_input_sidebar", on_change=trigger_nominal_recalc)

    header_cols = st.columns([1, 2, 2, 2, 2])
    headers = ["On", "λ min", "λ max", "T@λmin", "T@λmax"]
    for col, header in zip(header_cols, headers): col.caption(header)

    for i in range(len(st.session_state.targets)):
        target = st.session_state.targets[i]
        cols = st.columns([1, 2, 2, 2, 2])
        current_enabled = target.get('enabled', False)
        new_enabled = cols[0].checkbox("", value=current_enabled, key=f"target_enable_sidebar_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['enabled'] = new_enabled
        st.session_state.targets[i]['min'] = cols[1].number_input("λmin", value=target.get('min', 0.0), format="%.1f", key=f"target_min_sidebar_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['max'] = cols[2].number_input("λmax", value=target.get('max', 0.0), format="%.1f", key=f"target_max_sidebar_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['target_min'] = cols[3].number_input("Tmin", value=target.get('target_min', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmin_sidebar_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)
        st.session_state.targets[i]['target_max'] = cols[4].number_input("Tmax", value=target.get('target_max', 0.0), min_value=0.0, max_value=1.0, format="%.3f", step=0.01, key=f"target_tmax_sidebar_{i}", label_visibility="collapsed", on_change=trigger_nominal_recalc)

    st.divider()

    # --- Actions Section ---
    st.header("▶️ Actions")

    if st.button("📊 Evaluate Nominal Structure", key="eval_nom_sidebar", use_container_width=True):
        st.session_state.needs_rerun_calc = True
        st.session_state.rerun_calc_params = {'is_optimized_run': False, 'method_name': "Nominal (Evaluated)"}
        st.rerun()

    if st.button("✨ Local Optimization", key="optim_local_sidebar", use_container_width=True):
        run_local_optimization_wrapper()
        st.rerun() # Rerun to update plots after optimization

    if st.button("🚀 Initial Scan + Optimization", key="optim_scan_sidebar", use_container_width=True):
        run_scan_optimization_wrapper()
        st.rerun()

    auto_cols = st.columns([3,2])
    with auto_cols[0]:
        if st.button("🤖 Auto Mode", key="optim_auto_sidebar", use_container_width=True, help="Needle Insertion -> Thin Layer Removal -> Optimization Cycles"):
            run_auto_mode_wrapper()
            st.rerun()
    with auto_cols[1]:
        st.session_state.auto_thin_threshold = st.number_input("Thin Thr. (nm)", value=st.session_state.auto_thin_threshold, min_value=MIN_THICKNESS_PHYS_NM, format="%.3f", key="auto_thin_input_action_sidebar", help="Thickness threshold for automatic thin layer removal in Auto Mode.", label_visibility="collapsed")

    st.divider()
    st.subheader("🛠️ Actions on Optimized")

    can_optimize = st.session_state.get('is_optimized_state', False) and st.session_state.get('optimized_ep') is not None
    can_remove = can_optimize and len(st.session_state.optimized_ep) > 2
    can_undo = bool(st.session_state.get('ep_history'))

    if st.button("🗑️ Remove Thin Layer + Re-opt", key="remove_thin_sidebar", use_container_width=True, disabled=not can_remove):
        run_remove_thin_wrapper()
        st.rerun()

    if st.button("💾 Set Optimized -> Nominal", key="set_optim_as_nom_sidebar", use_container_width=True, disabled=not can_optimize):
        set_optimized_as_nominal_wrapper()
        st.rerun() # Rerun to reflect the change in nominal QWOT

    if st.button(f"↩️ Undo Removal ({len(st.session_state.get('ep_history', deque()))})", key="undo_remove_sidebar", use_container_width=True, disabled=not can_undo):
        undo_remove_wrapper()
        st.rerun() # Rerun to reflect the undone state


# --- Main Area (Results Display) ---
st.header("📈 Results")
state_desc = "Optimized" if st.session_state.is_optimized_state else "Nominal"
ep_display = st.session_state.optimized_ep if st.session_state.is_optimized_state else st.session_state.current_ep
num_layers_display = len(ep_display) if ep_display is not None else 0
st.subheader(f"Current State: {state_desc} ({num_layers_display} layers)")

# Display Metrics
res_cols = st.columns(3)
with res_cols[0]:
    if st.session_state.last_mse is not None and np.isfinite(st.session_state.last_mse):
        st.metric("MSE", f"{st.session_state.last_mse:.4e}")
    else:
         st.metric("MSE", "N/A")
with res_cols[1]:
    min_thick_str = "N/A"
    if ep_display is not None and ep_display.size > 0:
        valid_thick = ep_display[ep_display >= MIN_THICKNESS_PHYS_NM - 1e-9] # Allow tolerance
        if valid_thick.size > 0:
            min_thick_str = f"{np.min(valid_thick):.3f} nm"
    st.metric("Min Thick.", min_thick_str)
with res_cols[2]:
     if st.session_state.is_optimized_state and st.session_state.get('optimized_qwot_str'):
         st.text_input("Opt. QWOT", value=st.session_state.optimized_qwot_str, disabled=True, key="opt_qwot_display_main_area")


# Display Plots
st.subheader("Spectral Response")
if 'last_calc_results' in st.session_state and st.session_state.last_calc_results:
    results_data = st.session_state.last_calc_results
    res_fine_plot = results_data.get('res_fine')
    active_targets_plot = validate_targets() # Get current valid targets for plotting
    mse_plot = st.session_state.last_mse
    method_name_plot = results_data.get('method_name', '')
    res_optim_grid_plot = results_data.get('res_optim_grid') # Get data used for MSE calc

    if res_fine_plot and active_targets_plot is not None:
        fig_spec, ax_spec = plt.subplots(figsize=(12, 5))

        opt_method_str = f" ({method_name_plot})" if method_name_plot else ""
        window_title = f'Spectral Response {"Optimized" if st.session_state.is_optimized_state else "Nominal"}{opt_method_str}'
        # fig_spec.suptitle(window_title, fontsize=14, weight='bold') # Title can crowd plot

        line_ts = None
        try:
            if res_fine_plot and 'l' in res_fine_plot and 'Ts' in res_fine_plot and res_fine_plot['l'] is not None and len(res_fine_plot['l']) > 0:
                res_l_plot = np.asarray(res_fine_plot['l'])
                res_ts_plot = np.asarray(res_fine_plot['Ts'])
                line_ts, = ax_spec.plot(res_l_plot, res_ts_plot, label='Transmittance', linestyle='-', color='blue', linewidth=1.5)

                plotted_target_label = False
                if active_targets_plot: # Check if list is not empty
                    for i, target in enumerate(active_targets_plot):
                        l_min, l_max = target['min'], target['max']
                        t_min, t_max_corr = target['target_min'], target['target_max']
                        x_coords, y_coords = [l_min, l_max], [t_min, t_max_corr]
                        label = 'Target(s)' if not plotted_target_label else "_nolegend_"
                        line_target, = ax_spec.plot(x_coords, y_coords, 'r--', linewidth=1.0, alpha=0.7, label=label, zorder=5)
                        marker_target = ax_spec.plot(x_coords, y_coords, marker='x', color='red', markersize=6, linestyle='none', label='_nolegend_', zorder=6)
                        plotted_target_label = True

                        # Plot points used for MSE calculation within this target zone
                        if res_optim_grid_plot and 'l' in res_optim_grid_plot and res_optim_grid_plot['l'].size > 0:
                            res_l_optim = np.asarray(res_optim_grid_plot['l'])
                            indices_optim = np.where((res_l_optim >= l_min) & (res_l_optim <= l_max))[0]
                            if indices_optim.size > 0:
                                optim_lambdas = res_l_optim[indices_optim]
                                if abs(l_max - l_min) < 1e-9: optim_target_t = np.full_like(optim_lambdas, t_min)
                                else: slope = (t_max_corr - t_min) / (l_max - l_min); optim_target_t = t_min + slope * (optim_lambdas - l_min)
                                ax_spec.plot(optim_lambdas, optim_target_t, marker='.', color='darkred', linestyle='none', markersize=4, alpha=0.5, label='_nolegend_', zorder=6)

                ax_spec.set_xlabel("Wavelength (nm)")
                ax_spec.set_ylabel('Transmittance')
                ax_spec.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
                ax_spec.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')
                ax_spec.minorticks_on()
                if len(res_l_plot) > 0: ax_spec.set_xlim(res_l_plot[0], res_l_plot[-1])
                ax_spec.set_ylim(-0.05, 1.05)
                if plotted_target_label or (line_ts is not None): ax_spec.legend(fontsize=8)

                if mse_plot is not None and np.isfinite(mse_plot): mse_text = f"MSE = {mse_plot:.3e}"
                else: mse_text = "MSE: N/A"
                ax_spec.text(0.98, 0.98, mse_text, transform=ax_spec.transAxes, ha='right', va='top', fontsize=9,
                             bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
            else:
                ax_spec.text(0.5, 0.5, "No spectral data", ha='center', va='center', transform=ax_spec.transAxes)
        except Exception as e_spec:
            ax_spec.text(0.5, 0.5, f"Error plotting spectrum:\n{e_spec}", ha='center', va='center', transform=ax_spec.transAxes, color='red')

        plt.tight_layout() #rect=[0, 0, 1, 0.95]) # Adjust layout
        st.pyplot(fig_spec)
        plt.close(fig_spec) # Close the figure to free memory
    else:
        st.warning("Missing or invalid calculation data for spectrum display.")
else:
    st.info("Run an evaluation or optimization to see results.")


st.subheader("Index Profile & Structure")
plot_col1, plot_col2 = st.columns(2)

if 'last_calc_results' in st.session_state and st.session_state.last_calc_results:
    results_data = st.session_state.last_calc_results
    ep_plot = results_data.get('ep_used')
    l0_plot = results_data.get('l0_used')
    nH_plot = results_data.get('nH_used')
    nL_plot = results_data.get('nL_used')
    nSub_plot = results_data.get('nSub_used')
    is_optimized_plot = st.session_state.is_optimized_state
    material_sequence_plot = st.session_state.get('material_sequence') # Check if arbitrary sequence was used

    if ep_plot is not None and l0_plot is not None and nH_plot is not None and nL_plot is not None and nSub_plot is not None:
        with plot_col1:
            fig_idx, ax_idx = plt.subplots(figsize=(6, 5))
            try:
                nH_c_repr = _get_nk_at_lambda(nH_plot, l0_plot, EXCEL_FILE_PATH)
                nL_c_repr = _get_nk_at_lambda(nL_plot, l0_plot, EXCEL_FILE_PATH)
                nSub_c_repr = _get_nk_at_lambda(nSub_plot, l0_plot, EXCEL_FILE_PATH)

                if nH_c_repr is None or nL_c_repr is None or nSub_c_repr is None:
                    raise ValueError("Indices at l0 not found for profile plot.")

                nH_r_repr, nL_r_repr, nSub_r_repr = nH_c_repr.real, nL_c_repr.real, nSub_c_repr.real
                num_layers = len(ep_plot)

                if material_sequence_plot:
                    # Fallback for arbitrary sequence plot
                    n_real_layers_repr = [nH_r_repr if i % 2 == 0 else nL_r_repr for i in range(num_layers)]
                else:
                    n_real_layers_repr = [nH_r_repr if i % 2 == 0 else nL_r_repr for i in range(num_layers)]

                ep_cumulative = np.cumsum(ep_plot) if num_layers > 0 else np.array([0])
                total_thickness = ep_cumulative[-1] if num_layers > 0 else 0
                margin = max(50, 0.1 * total_thickness) if total_thickness > 0 else 50

                # Build coordinates for step plot
                x_coords_plot = [-margin]; y_coords_plot = [nSub_r_repr] # Start in substrate
                if num_layers > 0:
                    x_coords_plot.append(0); y_coords_plot.append(nSub_r_repr) # Interface Sub/L1
                    for i in range(num_layers):
                        layer_start = ep_cumulative[i-1] if i > 0 else 0
                        layer_end = ep_cumulative[i]
                        layer_n_real = n_real_layers_repr[i]
                        x_coords_plot.extend([layer_start, layer_end])
                        y_coords_plot.extend([layer_n_real, layer_n_real])
                    last_layer_end = ep_cumulative[-1]
                    x_coords_plot.extend([last_layer_end, last_layer_end + margin]) # Interface LN/Air
                    y_coords_plot.extend([1.0, 1.0]) # Air index
                else: # Bare substrate
                    x_coords_plot.extend([0, 0, margin])
                    y_coords_plot.extend([nSub_r_repr, 1.0, 1.0])

                ax_idx.plot(x_coords_plot, y_coords_plot, drawstyle='steps-post', label=f'n\'(λ={l0_plot:.0f}nm)', color='purple', linewidth=1.5)
                ax_idx.set_xlabel('Depth (from substrate) (nm)')
                ax_idx.set_ylabel("Real Part of Index (n')")
                ax_idx.set_title(f"Index Profile (at λ={l0_plot:.0f}nm)")
                ax_idx.grid(True, which='major', linestyle='-', linewidth='0.5', color='gray')
                ax_idx.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgray')
                ax_idx.minorticks_on()
                ax_idx.set_xlim(x_coords_plot[0], x_coords_plot[-1])

                # Adjust y-limits
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
            fig_stack, ax_stack = plt.subplots(figsize=(6, 5))
            try:
                num_layers = len(ep_plot)
                if num_layers > 0:
                    indices_complex_repr = []
                    if material_sequence_plot:
                        # Fallback using H/L assumption for display
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
                    ax_stack.invert_yaxis() # Layer 1 at top

                    # Add thickness labels
                    max_ep_plot = max(ep_plot) if ep_plot.size > 0 else 1.0
                    fontsize_bar = max(6, 9 - num_layers // 15)
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ha_pos = 'left' if width < max_ep_plot * 0.3 else 'right'
                        x_text_pos = width * 1.02 if ha_pos == 'left' else width * 0.98
                        text_color = 'black' if ha_pos == 'left' else 'white'
                        ax_stack.text(x_text_pos, bar.get_y() + bar.get_height()/2., f"{width:.2f}",
                                      va='center', ha=ha_pos, color=text_color, fontsize=fontsize_bar, weight='bold')
                else: # Empty structure
                    ax_stack.text(0.5, 0.5, "Empty Structure", ha='center', va='center', fontsize=10, color='grey', transform=ax_stack.transAxes)
                    ax_stack.set_yticks([]); ax_stack.set_xticks([])

                ax_stack.set_xlabel('Thickness (nm)')
                stack_title_prefix = f'Structure {"Optimized" if is_optimized_plot else "Nominal"}'
                ax_stack.set_title(f"{stack_title_prefix} ({num_layers} layers)")
                max_ep_plot_lim = max(ep_plot) if num_layers > 0 else 10
                ax_stack.set_xlim(right=max_ep_plot_lim * 1.1) # Add padding
            except Exception as e_stack:
                ax_stack.text(0.5, 0.5, f"Error plotting structure:\n{e_stack}", ha='center', va='center', transform=ax_stack.transAxes, color='red')
            plt.tight_layout()
            st.pyplot(fig_stack)
            plt.close(fig_stack)
    else:
        st.warning("Missing data to display profiles.")
else: # No calculation results yet
    pass # Don't display plot area if no results


# --- Automatic Recalculation Trigger ---
if st.session_state.get('needs_rerun_calc', False):
    params = st.session_state.rerun_calc_params
    force_ep_val = params.get('force_ep')

    st.session_state.needs_rerun_calc = False # Reset flag
    st.session_state.rerun_calc_params = {}
    st.session_state.calculating = True # Set lock

    # Run the calculation based on stored parameters
    run_calculation_wrapper(
        is_optimized_run=params.get('is_optimized_run', False),
        method_name=params.get('method_name', 'Auto Recalc'),
        force_ep=force_ep_val
    )
    st.session_state.calculating = False # Release lock
    st.rerun() # Rerun Streamlit to update the UI

