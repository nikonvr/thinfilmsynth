# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
import datetime
import traceback
import math
import copy
import time
import multiprocessing
import sys
import io
from functools import partial
from typing import List, Tuple, Dict, Optional, Any, Callable, Union

from scipy.optimize import minimize, OptimizeResult # type: ignore
from scipy.stats import qmc # type: ignore

try:
    import numba
    from numba import prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    st.warning("Numba not found. Calculations will be significantly slower.")
    class NumbaDummy:
        def njit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
        def prange(self, *args, **kwargs):
            return range(*args, **kwargs)
    numba = NumbaDummy()
    prange = numba.prange

# --- Constants ---
OPTIMIZATION_MODES: Dict[str, Dict[str, str]] = {
    "Slow": {"n_samples": "32768", "p_best": "50", "n_passes": "5"},
    "Medium": {"n_samples": "8192", "p_best": "15", "n_passes": "3"},
    "Fast": {"n_samples": "4096", "p_best": "10", "n_passes": "2"}
}
DEFAULT_MODE: str = "Fast"
MIN_THICKNESS_PHYS_NM: float = 0.01
DEFAULT_QWOT: str = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
DEFAULT_TARGETS: List[Dict[str, Union[str, bool]]] = [
    {'min': "480.0", 'max': "500.0", 'target_min': "0.0", 'target_max': "1.0", 'enabled': True},
    {'min': "500.0", 'max': "600.0", 'target_min': "1.0", 'target_max': "1.0", 'enabled': True},
    {'min': "600.0", 'max': "630.0", 'target_min': "1.0", 'target_max': "0.0", 'enabled': True},
    {'min': "400.0", 'max': "480.0", 'target_min': "0.0", 'target_max': "0.0", 'enabled': True},
    {'min': "630.0", 'max': "700.0", 'target_min': "0.0", 'target_max': "0.0", 'enabled': True}
]
DEFAULT_OPTIM_PARAMS: Dict[str, str] = OPTIMIZATION_MODES[DEFAULT_MODE]
ZERO_THRESHOLD: float = 1e-12
RT_DENOMINATOR_THRESHOLD: float = 1e-12
QWOT_N_THRESHOLD: float = 1e-9
MSE_TARGET_LAMBDA_DIFF_THRESHOLD: float = 1e-9
PENALTY_BASE: float = 1e6
PENALTY_FACTOR: float = 1e8
HIGH_COST_PENALTY: float = 1e8
HIGH_COST_MSE_ZERO_POINTS: float = 1e7
LBFGSB_WORKER_OPTIONS: Dict[str, Any] = {'maxiter': 99, 'ftol': 1.e-10, 'gtol': 1e-7, 'disp': False}
LBFGSB_REOPT_OPTIONS: Dict[str, Any] = {'maxiter': 199, 'ftol': 1e-10, 'gtol': 1e-7, 'disp': False}
PLOT_LAMBDA_STEP_DIVISOR: float = 10.0
MIN_SCALING_NM: float = 0.1
AUTO_REMOVE_THRESHOLD_NM: float = 1.0
INITIAL_SOBOL_REL_SCALE_LOWER: float = 0.1
INITIAL_SOBOL_REL_SCALE_UPPER: float = 2.0
PHASE1BIS_SCALING_NM: float = 10.0
PASS_P_BEST_REDUCTION_FACTOR: float = 0.8
PASS_SCALING_REDUCTION_BASE: float = 1.8

# --- Helper Functions ---
def upper_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    exponent = math.floor(math.log2(n)) + 1
    return 1 << exponent

if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

def log_message(message: Any) -> None:
    full_message = str(message)
    st.session_state.log_messages.append(full_message)

def log_with_elapsed_time(message: Any) -> None:
    prefix = ""
    if 'optim_start_time' in st.session_state and st.session_state.optim_start_time is not None:
        try:
            elapsed = time.time() - st.session_state.optim_start_time
            prefix = f"[{elapsed:8.2f}s] "
        except TypeError:
             prefix = "[?:??s] " # Fallback if time calculation fails
    full_message = prefix + str(message)
    st.session_state.log_messages.append(full_message)

def clear_log() -> None:
    st.session_state.log_messages = [f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Log cleared."]

# --- Numba Accelerated Core ---
@numba.njit(fastmath=True, cache=True)
def compute_stack_matrix(
    ep_vector: np.ndarray,
    l_val: float,
    nH_complex: complex,
    nL_complex: complex
) -> np.ndarray:
    M = np.eye(2, dtype=np.complex128)
    for i in range(len(ep_vector)):
        thickness = ep_vector[i]
        if thickness <= ZERO_THRESHOLD: continue
        Ni = nH_complex if i % 2 == 0 else nL_complex
        eta = Ni
        phi = (2 * np.pi / l_val) * (Ni * thickness)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        M_layer_00 = cos_phi
        M_layer_01 = (1j / eta) * sin_phi
        M_layer_10 = 1j * eta * sin_phi
        M_layer_11 = cos_phi
        # Manual construction faster than np.array inside loop for Numba?
        M_layer = np.empty((2, 2), dtype=np.complex128)
        M_layer[0, 0] = M_layer_00
        M_layer[0, 1] = M_layer_01
        M_layer[1, 0] = M_layer_10
        M_layer[1, 1] = M_layer_11
        M = M_layer @ M
    return M

@numba.njit(parallel=True, fastmath=True, cache=True)
def calculate_RT_from_ep_core(
    ep_vector: np.ndarray,
    nH_complex: complex,
    nL_complex: complex,
    nSub_complex: complex,
    l_vec: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if not l_vec.size:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    Rs_arr = np.zeros(len(l_vec), dtype=np.float64)
    Ts_arr = np.zeros(len(l_vec), dtype=np.float64)
    etainc = 1.0 + 0j
    etasub = nSub_complex
    real_etasub = np.real(etasub)
    real_etainc = np.real(etainc)

    for i_l in prange(len(l_vec)):
        l_val = l_vec[i_l]
        if l_val <= 0:
            Rs_arr[i_l], Ts_arr[i_l] = np.nan, np.nan
            continue

        # Ensure contiguous array for compute_stack_matrix if needed,
        # though ep_vector passed in should ideally already be contiguous.
        # ep_vector_contig = np.ascontiguousarray(ep_vector) # Might be redundant if input is always contiguous
        M = compute_stack_matrix(ep_vector, l_val, nH_complex, nL_complex)
        m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

        rs_num = (etainc * m00 - etasub * m11 + etainc * etasub * m01 - m10)
        rs_den = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)

        if np.abs(rs_den) < RT_DENOMINATOR_THRESHOLD:
            Rs_arr[i_l], Ts_arr[i_l] = np.nan, np.nan # Use NaN, handle later
        else:
            rs = rs_num / rs_den
            ts = (2 * etainc) / rs_den
            Rs_arr[i_l] = np.abs(rs)**2
            if real_etainc == 0: # Should not happen for etainc=1.0
                 Ts_arr[i_l] = np.nan
            else:
                 Ts_arr[i_l] = (real_etasub / real_etainc) * np.abs(ts)**2

    # Replace NaN results with 0.0 after the parallel loop
    for i in range(len(Rs_arr)):
        if np.isnan(Rs_arr[i]): Rs_arr[i] = 0.0
        if np.isnan(Ts_arr[i]): Ts_arr[i] = 0.0

    return Rs_arr, Ts_arr

@numba.njit(fastmath=True, cache=True)
def get_target_points_indices(l_vec: np.ndarray, target_min: float, target_max: float) -> np.ndarray:
    if not l_vec.size: return np.empty(0, dtype=np.int64)
    # Ensure target_min <= target_max before calling? Or handle here?
    # Assuming valid input where target_min <= target_max
    return np.where((l_vec >= target_min) & (l_vec <= target_max))[0]

# --- Main Calculation & Interface Functions ---
def calculate_RT_from_ep(
    ep_vector: Union[np.ndarray, List[float]],
    nH: Union[float, complex],
    nL: Union[float, complex],
    nSub: Union[float, complex],
    l_vec: Union[np.ndarray, List[float]]
) -> Optional[Dict[str, np.ndarray]]:

    try:
        nH_complex = complex(nH)
        nL_complex = complex(nL)
        nSub_complex = complex(nSub)

        # Ensure numpy arrays and correct types
        ep_vector_np = np.ascontiguousarray(ep_vector, dtype=np.float64)
        l_vec_np = np.ascontiguousarray(l_vec, dtype=np.float64)

        if not np.all(np.isfinite(ep_vector_np)):
            log_message("Warning: Non-finite values found in thickness vector for RT calculation.")
            # Option: Clamp or return None? Returning None might be safer.
            return None
        if not np.all(np.isfinite(l_vec_np)) or np.any(l_vec_np <= 0):
             log_message("Warning: Non-finite or non-positive values found in lambda vector for RT calculation.")
             return None
        if ep_vector_np.ndim != 1 or l_vec_np.ndim != 1:
             log_message("Warning: Input vectors must be 1-dimensional for RT calculation.")
             return None

        Rs, Ts = calculate_RT_from_ep_core(ep_vector_np, nH_complex, nL_complex, nSub_complex, l_vec_np)

        return {'l': l_vec_np, 'Rs': Rs, 'Ts': Ts}

    except Exception as e:
        log_message(f"Error during calculate_RT_from_ep: {type(e).__name__}: {e}")
        # Optionally log traceback for debugging
        # log_message(traceback.format_exc(limit=2))
        return None

def calculate_initial_ep(
    qwot_multipliers: Union[Tuple[float, ...], List[float]],
    l0: float,
    nH_real: float,
    nL_real: float
) -> np.ndarray:

    num_layers = len(qwot_multipliers)
    ep_initial = np.zeros(num_layers, dtype=np.float64)

    if l0 <= 0: raise ValueError("Centering wavelength l0 must be positive for QWOT calculation")
    if nH_real <= QWOT_N_THRESHOLD or nL_real <= QWOT_N_THRESHOLD:
         raise ValueError(f"Real parts of nH ({nH_real}) and nL ({nL_real}) must be > {QWOT_N_THRESHOLD}.")

    for i in range(num_layers):
        multiplier = qwot_multipliers[i]
        n_real = nH_real if i % 2 == 0 else nL_real
        # Calculation requires n_real > 0, checked above.
        ep_initial[i] = multiplier * l0 / (4 * n_real)

    # Check for issues after calculation (shouldn't happen with checks above)
    if not np.all(np.isfinite(ep_initial)):
         log_message("WARNING: Initial QWOT calculation produced NaN/inf despite checks. Replacing with 0.")
         ep_initial = np.nan_to_num(ep_initial, nan=0.0, posinf=0.0, neginf=0.0)

    return ep_initial

def get_initial_ep(
    emp_str: str,
    l0: float,
    nH: Union[float, complex],
    nL: Union[float, complex]
) -> Tuple[np.ndarray, List[float]]:

    try:
        qwot_multipliers = [float(e.strip()) for e in emp_str.split(',') if e.strip()]
    except ValueError:
        raise ValueError("Invalid QWOT format. Use numbers separated by commas (e.g., 1, 0.5, 1).")

    if any(e < 0 for e in qwot_multipliers):
        raise ValueError("QWOT multipliers cannot be negative.")

    if not qwot_multipliers:
        return np.array([], dtype=np.float64), []

    nH_r = np.real(complex(nH))
    nL_r = np.real(complex(nL))

    # calculate_initial_ep will raise ValueError for invalid nH_r, nL_r, l0
    ep_initial = calculate_initial_ep(tuple(qwot_multipliers), l0, nH_r, nL_r)

    return ep_initial, qwot_multipliers

def calculate_qwot_from_ep(
    ep_vector: np.ndarray,
    l0: float,
    nH_r: float,
    nL_r: float
) -> np.ndarray:
    num_layers = len(ep_vector)
    qwot_multipliers = np.zeros(num_layers, dtype=np.float64)

    if l0 <= 0:
        log_message("Warning: Cannot calculate QWOT, l0 must be positive.")
        qwot_multipliers[:] = np.nan
        return qwot_multipliers
    if nH_r <= QWOT_N_THRESHOLD or nL_r <= QWOT_N_THRESHOLD:
        log_message(f"Warning: Cannot calculate QWOT, real indices nH_r ({nH_r}) and nL_r ({nL_r}) must be > {QWOT_N_THRESHOLD}.")
        qwot_multipliers[:] = np.nan
        return qwot_multipliers

    for i in range(num_layers):
        n_real = nH_r if i % 2 == 0 else nL_r
        # Calculation requires n_real > QWOT_N_THRESHOLD, checked above.
        qwot_multipliers[i] = ep_vector[i] * (4 * n_real) / l0

    return qwot_multipliers

# --- Cost/Objective Functions ---
def calculate_final_mse(
    res: Optional[Dict[str, np.ndarray]],
    active_targets: List[Dict[str, float]]
) -> Tuple[Optional[float], int]:
    total_squared_error: float = 0.0
    total_points_in_targets: int = 0
    mse: Optional[float] = None

    if not active_targets:
        return None, 0
    if not isinstance(res, dict) or 'Ts' not in res or res['Ts'] is None or not isinstance(res['Ts'], np.ndarray) or 'l' not in res or res['l'] is None or not isinstance(res['l'], np.ndarray):
         log_message("MSE Calc Warning: Invalid 'res' dictionary input.")
         return None, 0

    calculated_lambdas = res['l']
    calculated_Ts = res['Ts']

    if calculated_lambdas.size == 0 or calculated_Ts.size == 0:
         return None, 0 # No data points
    if len(calculated_lambdas) != len(calculated_Ts):
        log_message("MSE Calc Warning: Lambda and Ts arrays have different lengths.")
        return None, 0

    # Pre-filter NaNs from calculated Ts
    finite_mask_all = np.isfinite(calculated_Ts)
    if not np.all(finite_mask_all):
        log_message(f"MSE Calc Warning: Found {np.sum(~finite_mask_all)} non-finite T values.")
        # Decide whether to calculate MSE on remaining points or fail
        # Let's calculate on the finite points
        calculated_lambdas = calculated_lambdas[finite_mask_all]
        calculated_Ts = calculated_Ts[finite_mask_all]
        if calculated_lambdas.size == 0:
            return None, 0

    for i, target in enumerate(active_targets):
        try:
            l_min = target['min']
            l_max = target['max']
            t_min = target['target_min']
            t_max = target['target_max']
        except KeyError:
             log_message(f"MSE Calc Warning: Target {i+1} is missing required keys.")
             continue

        indices = get_target_points_indices(calculated_lambdas, l_min, l_max)

        if indices.size > 0:
            # Indices are already relative to the filtered calculated_lambdas/Ts
            calculated_Ts_in_zone = calculated_Ts[indices]
            target_lambdas_in_zone = calculated_lambdas[indices]

            # Interpolate target T values
            if abs(l_max - l_min) < MSE_TARGET_LAMBDA_DIFF_THRESHOLD:
                 interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
            else:
                 slope = (t_max - t_min) / (l_max - l_min)
                 interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)

            squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
            total_squared_error += np.sum(squared_errors)
            total_points_in_targets += len(calculated_Ts_in_zone)

    if total_points_in_targets > 0:
        mse = total_squared_error / total_points_in_targets
    elif active_targets: # Targets exist, but no points found in zones
        mse = np.nan

    return mse, total_points_in_targets

def calculate_mse_for_optimization_penalized(
    ep_vector: Union[np.ndarray, List[float]],
    nH: complex,
    nL: complex,
    nSub: complex,
    l_vec_optim: np.ndarray,
    active_targets: List[Dict[str, float]],
    min_thickness_phys_nm: float
) -> float:

    if not isinstance(ep_vector, np.ndarray):
        ep_vector = np.array(ep_vector, dtype=np.float64)

    if ep_vector.ndim != 1:
        log_message(f"Warning: ep_vector in cost function is not 1D (shape: {ep_vector.shape}).")
        return np.inf

    ep_vector_calc = np.maximum(ep_vector, min_thickness_phys_nm)
    below_min_mask = ep_vector < min_thickness_phys_nm
    penalty: float = 0.0
    if np.any(below_min_mask):
        penalty = PENALTY_BASE + np.sum((min_thickness_phys_nm - ep_vector[below_min_mask])**2) * PENALTY_FACTOR

    try:
        res = calculate_RT_from_ep(ep_vector_calc, nH, nL, nSub, l_vec_optim)

        if res is None:
             # log_message(f"RT calculation failed in cost function for ep={ep_vector[:3]}...") # Potentially too verbose
             return np.inf + penalty

        calculated_lambdas = res['l']
        calculated_Ts = res['Ts']

        if np.any(~np.isfinite(calculated_Ts)):
            # log_message(f"Non-finite T values in cost function for ep={ep_vector[:3]}...") # Potentially too verbose
            return HIGH_COST_PENALTY + penalty

        total_squared_error: float = 0.0
        total_points_in_targets: int = 0

        for target in active_targets:
            l_min = target['min']
            l_max = target['max']
            t_min = target['target_min']
            t_max = target['target_max']

            indices = get_target_points_indices(calculated_lambdas, l_min, l_max)

            if indices.size > 0:
                calculated_Ts_in_zone = calculated_Ts[indices]
                target_lambdas_in_zone = calculated_lambdas[indices]

                if abs(l_max - l_min) < MSE_TARGET_LAMBDA_DIFF_THRESHOLD:
                    interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
                else:
                    slope = (t_max - t_min) / (l_max - l_min)
                    interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)

                squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
                total_squared_error += np.sum(squared_errors)
                total_points_in_targets += len(indices)

        if total_points_in_targets == 0:
            mse = HIGH_COST_MSE_ZERO_POINTS # High cost if no points overlap targets
        else:
            mse = total_squared_error / total_points_in_targets

        final_cost = mse + penalty

        if not np.isfinite(final_cost):
            # log_message(f"Warning: Non-finite final_cost ({final_cost}) for ep={ep_vector[:3]}...") # Potentially too verbose
            return np.inf

        return final_cost

    except ValueError as ve:
        # log_message(f"ValueError in cost function: {ve} for ep={ep_vector[:3]}...") # Potentially too verbose
        return np.inf
    except Exception as e_rt:
        # log_message(f"Error in cost function RT calc: {type(e_rt).__name__} for ep={ep_vector[:3]}...") # Potentially too verbose
        return np.inf

# --- Optimization Logic ---
def perform_single_thin_layer_removal(
    ep_vector_in: np.ndarray,
    min_thickness_phys: float,
    cost_function: Callable[..., float],
    args_for_cost: Tuple,
    log_prefix: str = "",
    target_layer_index: Optional[int] = None
) -> Tuple[np.ndarray, bool, float, List[str]]:

    current_ep = ep_vector_in.copy()
    logs: List[str] = []
    num_layers = len(current_ep)

    if num_layers <= 1:
        logs.append(f"{log_prefix}Structure has {num_layers} layers. Cannot merge/delete further.")
        try:
            initial_cost = cost_function(current_ep, *args_for_cost) if num_layers > 0 else np.inf
        except Exception as e:
             initial_cost = np.inf
             logs.append(f"{log_prefix}Error calculating initial cost for {num_layers} layers: {e}")
        return current_ep, False, initial_cost, logs

    thin_layer_index: int = -1
    min_thickness_found: float = np.inf

    if target_layer_index is not None:
        if 0 <= target_layer_index < num_layers:
            thin_layer_index = target_layer_index
            min_thickness_found = current_ep[target_layer_index]
            logs.append(f"{log_prefix}Targeting specified layer {thin_layer_index + 1} ({min_thickness_found:.3f} nm) for potential removal.")
        else:
            logs.append(f"{log_prefix}Specified target layer index {target_layer_index+1} is invalid for {num_layers} layers. Finding absolute thinnest.")
            target_layer_index = None

    if target_layer_index is None:
        eligible_indices = np.where(current_ep >= min_thickness_phys)[0]
        if eligible_indices.size > 0:
            min_idx_within_eligible = np.argmin(current_ep[eligible_indices])
            thin_layer_index = eligible_indices[min_idx_within_eligible]
            min_thickness_found = current_ep[thin_layer_index]
            logs.append(f"{log_prefix}Identified thinnest eligible layer: Layer {thin_layer_index + 1} ({min_thickness_found:.3f} nm).")
        else:
            # No layers meet the minimum thickness requirement
             logs.append(f"{log_prefix}No eligible layer found >= {min_thickness_phys:.3f} nm.")
             # Fall through, no removal possible

    if thin_layer_index == -1:
        logs.append(f"{log_prefix}No suitable layer identified for removal.")
        try:
            initial_cost = cost_function(current_ep, *args_for_cost)
        except Exception as e:
            initial_cost = np.inf
            logs.append(f"{log_prefix}Error calculating cost (no removal): {e}")
        return current_ep, False, initial_cost, logs

    ep_after_merge: Optional[np.ndarray] = None
    merged_info: str = ""
    structure_changed: bool = False

    if thin_layer_index == 0:
        if num_layers >= 2:
            ep_after_merge = current_ep[2:].copy()
            merged_info = f"Removing layer 1 (index 0) and layer 2 (index 1)."
            logs.append(f"{log_prefix}{merged_info} New structure: {len(ep_after_merge)} layers.")
            structure_changed = True
        else: # num_layers is 1, but thin_layer_index must be 0. Should have been caught earlier.
             logs.append(f"{log_prefix}Cannot remove layer 1 (index 0) - only 1 layer present (should not happen here).")
             # Fall through

    elif thin_layer_index == num_layers - 1:
        # Remove the last layer. Requires at least 1 layer.
        ep_after_merge = current_ep[:-1].copy()
        merged_info = f"Removed last layer {num_layers} (index {num_layers-1}) ({current_ep[-1]:.3f} nm)."
        logs.append(f"{log_prefix}{merged_info} New structure: {len(ep_after_merge)} layers.")
        structure_changed = True

    else: # 0 < thin_layer_index < num_layers - 1. Requires at least 3 layers.
        if num_layers >= 3:
            merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
            ep_after_merge = np.concatenate((
                current_ep[:thin_layer_index - 1],
                np.array([merged_thickness]),
                current_ep[thin_layer_index + 2:]
            ))
            merged_info = (f"Removed layer {thin_layer_index + 1} (index {thin_layer_index}), "
                           f"merged layer {thin_layer_index} (index {thin_layer_index - 1}) ({current_ep[thin_layer_index - 1]:.3f}) "
                           f"with layer {thin_layer_index + 2} (index {thin_layer_index + 1}) ({current_ep[thin_layer_index + 1]:.3f}) -> {merged_thickness:.3f}")
            logs.append(f"{log_prefix}{merged_info} New structure: {len(ep_after_merge)} layers.")
            structure_changed = True
        else: # num_layers is 2, thin_layer_index must be 0 or 1. Caught by other cases.
             logs.append(f"{log_prefix}Cannot merge around layer {thin_layer_index+1} - structure too small (needs >= 3 layers).")
             # Fall through

    final_ep = current_ep
    final_cost = np.inf
    success_overall = False

    if structure_changed and ep_after_merge is not None:
        num_layers_reopt = len(ep_after_merge)

        if num_layers_reopt == 0:
            logs.append(f"{log_prefix}Empty structure after merge/removal. Returning empty.")
            return np.array([]), True, np.inf, logs

        # Ensure minimum thickness constraint on merged layer(s)
        ep_after_merge = np.maximum(ep_after_merge, min_thickness_phys)
        reopt_bounds = [(min_thickness_phys, None)] * num_layers_reopt
        x0_reopt = ep_after_merge

        logs.append(f"{log_prefix}Starting local re-optimization (L-BFGS-B, maxiter={LBFGSB_REOPT_OPTIONS['maxiter']}) on {num_layers_reopt} layers...")
        reopt_start_time = time.time()
        reopt_args = args_for_cost # Use the same cost function arguments

        try:
            reopt_result = minimize(cost_function, x0_reopt, args=reopt_args,
                                    method='L-BFGS-B', bounds=reopt_bounds,
                                    options=LBFGSB_REOPT_OPTIONS)
            reopt_time = time.time() - reopt_start_time

            reopt_success = reopt_result.success and np.isfinite(reopt_result.fun)
            reopt_cost = reopt_result.fun if reopt_success else np.inf
            reopt_iters = reopt_result.nit

            logs.append(f"{log_prefix}Re-optimization finished in {reopt_time:.3f}s. Success: {reopt_success}, Cost: {reopt_cost:.3e}, Iters: {reopt_iters}")

            if reopt_success:
                final_ep = np.maximum(reopt_result.x.copy(), min_thickness_phys) # Ensure bounds post-optim
                final_cost = reopt_cost
                success_overall = True
                logs.append(f"{log_prefix}Re-optimization successful.")
            else:
                logs.append(f"{log_prefix}Re-optimization failed. Using merged structure without re-opt.")
                final_ep = np.maximum(x0_reopt.copy(), min_thickness_phys) # Use the clamped merged structure
                success_overall = False # Indicate re-opt failure but structure change occurred
                try:
                    final_cost = cost_function(final_ep, *reopt_args)
                    logs.append(f"{log_prefix}Recalculated cost (non-re-opt): {final_cost:.3e}")
                except Exception as e_cost:
                    final_cost = np.inf
                    logs.append(f"{log_prefix}Error calculating cost of merged structure: {e_cost}")

        except Exception as e_reopt:
            logs.append(f"{log_prefix}ERROR during L-BFGS-B re-optimization: {e_reopt}\n{traceback.format_exc(limit=2)}")
            logs.append(f"{log_prefix}Using merged structure without re-optimization attempt due to error.")
            final_ep = np.maximum(x0_reopt.copy(), min_thickness_phys) # Use the clamped merged structure
            success_overall = False
            try:
                 final_cost = cost_function(final_ep, *reopt_args)
                 logs.append(f"{log_prefix}Recalculated cost (non-re-opt after error): {final_cost:.3e}")
            except Exception: final_cost = np.inf

    else: # No structure change occurred
        logs.append(f"{log_prefix}Merge/removal not performed or structure unchanged. No re-optimization.")
        success_overall = False # No change was made successfully
        final_ep = current_ep # Return original
        try:
             final_cost = cost_function(final_ep, *args_for_cost)
        except Exception: final_cost = np.inf

    return final_ep, success_overall, final_cost, logs


# --- Parallel Local Search Worker ---
def local_search_worker(
    start_ep: np.ndarray,
    cost_function: Callable[..., float],
    args_for_cost: Tuple,
    lbfgsb_bounds: List[Tuple[Optional[float], Optional[float]]],
    min_thickness_phys_nm: float
) -> Dict[str, Any]:

    worker_pid = os.getpid()
    start_time = time.time()
    log_lines: List[str] = []
    log_lines.append(f"  [PID:{worker_pid} Start]: Starting local search (L-BFGS-B only).")

    initial_cost: float = np.inf
    start_ep_checked: np.ndarray = np.array([])
    iter_count: int = 0
    result: OptimizeResult = OptimizeResult(x=start_ep, success=False, fun=np.inf, message="Worker did not run", nit=0) # Default result

    try:
        # Ensure input is numpy array and apply min thickness
        start_ep_local = np.asarray(start_ep)
        if start_ep_local.ndim != 1 or start_ep_local.size == 0:
             raise ValueError(f"Start EP is invalid (shape: {start_ep_local.shape}).")
        start_ep_checked = np.maximum(start_ep_local, min_thickness_phys_nm)

        # Check bounds compatibility
        if len(lbfgsb_bounds) != len(start_ep_checked):
            raise ValueError(f"Dimension mismatch: start_ep ({len(start_ep_checked)}) vs bounds ({len(lbfgsb_bounds)})")

        # Calculate initial cost robustly
        initial_cost = cost_function(start_ep_checked, *args_for_cost)
        if not np.isfinite(initial_cost):
             log_lines.append(f"  [PID:{worker_pid} Start]: Warning - Initial cost is not finite ({initial_cost:.3e}). Treating as inf.")
             initial_cost = np.inf # Treat non-finite initial cost as infinity for comparison

    except Exception as e:
        log_lines.append(f"  [PID:{worker_pid} Start]: ERROR preparing/calculating initial cost: {type(e).__name__}: {e}")
        # traceback.print_exc(file=sys.stderr) # For debugging pool issues
        result = OptimizeResult(x=start_ep, success=False, fun=np.inf, message=f"Worker Initial Cost Error: {e}", nit=0)
        end_time = time.time()
        worker_summary = f"Worker {worker_pid} finished in {end_time - start_time:.2f}s. StartCost=ERROR, FinalCost=inf, Success=False, Msg='Initial Cost Error'"
        log_lines.append(f"  [PID:{worker_pid} Summary]: {worker_summary}")
        return {'result': result, 'final_ep': start_ep, 'final_cost': np.inf, 'success': False, 'start_ep': start_ep, 'pid': worker_pid, 'log_lines': log_lines, 'initial_cost': np.inf}

    # Run the optimization
    try:
        result = minimize(
            cost_function, start_ep_checked, args=args_for_cost,
            method='L-BFGS-B', bounds=lbfgsb_bounds,
            options=LBFGSB_WORKER_OPTIONS,
        )
        iter_count = result.nit if hasattr(result, 'nit') else 0

        # Ensure minimum thickness constraint after optimization
        if result.x is not None:
            result.x = np.maximum(result.x, min_thickness_phys_nm)
            # Optionally recalculate cost if clamping changed the vector significantly? Usually not necessary.

    except Exception as e:
        end_time = time.time()
        error_message = f"Worker {worker_pid} Exception during minimize after {end_time-start_time:.2f}s: {type(e).__name__}: {e}"
        log_lines.append(f"  [PID:{worker_pid} ERROR]: {error_message}\n{traceback.format_exc(limit=2)}")
        result = OptimizeResult(x=start_ep_checked, success=False, fun=np.inf, message=f"Worker Exception: {e}", nit=iter_count)

    # Process results
    end_time = time.time()
    final_cost_raw = result.fun if hasattr(result, 'fun') else np.inf
    final_cost = final_cost_raw if np.isfinite(final_cost_raw) else np.inf
    lbfgsb_success = result.success if hasattr(result, 'success') else False
    # Consider success only if LBFGSB reported success AND the final cost is finite
    success_status = lbfgsb_success and np.isfinite(final_cost)
    iterations = iter_count
    message_raw = result.message if hasattr(result, 'message') else "No message"
    message = message_raw.decode('utf-8', errors='ignore') if isinstance(message_raw, bytes) else str(message_raw)

    worker_summary = (
        f"Worker {worker_pid} finished in {end_time - start_time:.2f}s. "
        f"StartCost={initial_cost:.3e}, FinalCost={final_cost:.3e}, "
        f"Success={success_status} (LBFGSB Success: {lbfgsb_success}, Finite Cost: {np.isfinite(final_cost)}), "
        f"LBFGSB_Iters={iterations}, Msg='{message}'"
    )
    log_lines.append(f"  [PID:{worker_pid} Summary]: {worker_summary}")

    # Return the best available thickness vector
    final_x = result.x if success_status and result.x is not None else start_ep_checked

    return {'result': result, 'final_ep': final_x, 'final_cost': final_cost,
            'success': success_status, 'start_ep': start_ep, 'pid': worker_pid,
            'log_lines': log_lines, 'initial_cost': initial_cost}


# --- Optimization Phases ---
def _run_sobol_evaluation(
    num_layers: int,
    n_samples: int,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    cost_function_partial_map: Callable[[np.ndarray], float],
    min_thickness_phys_nm: float,
    phase_name: str = "Phase 1",
    progress_hook: Optional[Callable[[float], None]] = None
) -> List[Tuple[float, np.ndarray]]:

    log_with_elapsed_time(f"\n--- {phase_name}: Initial Evaluation (Sobol, N={n_samples}) ---")
    start_time_eval = time.time()

    if lower_bounds is None or upper_bounds is None or len(lower_bounds) != num_layers or len(upper_bounds) != num_layers:
        raise ValueError(f"_run_sobol_evaluation requires valid bounds matching num_layers ({num_layers}) (phase: {phase_name}).")
    if num_layers == 0:
         log_with_elapsed_time("Warning: Number of layers is 0, cannot run Sobol evaluation.")
         return []

    # Ensure bounds are valid and respect minimum thickness
    if np.any(lower_bounds > upper_bounds):
        log_with_elapsed_time(f"Warning: Sobol lower bounds > upper bounds found in {phase_name}. Clamping.")
        upper_bounds = np.maximum(lower_bounds, upper_bounds)
    # Ensure lower bounds meet physical minimum, then ensure upper > lower
    lower_bounds = np.maximum(min_thickness_phys_nm, lower_bounds)
    upper_bounds = np.maximum(lower_bounds + ZERO_THRESHOLD, upper_bounds) # Ensure upper > lower slightly

    log_with_elapsed_time(f"  Bounds (min={min_thickness_phys_nm:.3f}): L={np.array2string(lower_bounds, precision=3)}, U={np.array2string(upper_bounds, precision=3)}")

    ep_candidates: List[np.ndarray] = []
    if n_samples > 0:
        try:
            sampler = qmc.Sobol(d=num_layers, scramble=True)
            points_unit_cube = sampler.random(n=n_samples)
            # Scale requires l_bounds and u_bounds to be 1D array-like of size d
            ep_candidates_raw = qmc.scale(points_unit_cube, lower_bounds, upper_bounds)
            # Ensure minimum thickness constraint for all candidates
            ep_candidates = [np.maximum(min_thickness_phys_nm, cand) for cand in ep_candidates_raw]
        except Exception as e_sobol:
             log_with_elapsed_time(f"Error during Sobol sampling: {type(e_sobol).__name__}: {e_sobol}")
             return [] # Cannot proceed without samples

    if not ep_candidates:
        log_with_elapsed_time("  No Sobol candidates generated.")
        return []

    costs: List[float] = []
    initial_results: List[Tuple[float, np.ndarray]] = []
    num_workers_eval = min(n_samples, os.cpu_count()) # Use min(N, cpu_count) workers
    log_with_elapsed_time(f"  Evaluating cost (MSE) for {n_samples} candidates (max {num_workers_eval} workers)...")
    eval_start_pool = time.time()

    try:
        # Ensure the cost function can be pickled (partial should be fine)
        with multiprocessing.Pool(processes=num_workers_eval) as pool:
            costs = pool.map(cost_function_partial_map, ep_candidates)
        eval_pool_time = time.time() - eval_start_pool
        log_with_elapsed_time(f"  Parallel evaluation finished in {eval_pool_time:.2f}s.")

        if len(costs) != len(ep_candidates):
            log_with_elapsed_time(f"  Warning: Mismatch in cost results length ({len(costs)}) vs candidates ({len(ep_candidates)}). Truncating.")
            costs = costs[:len(ep_candidates)]
        initial_results = list(zip(costs, ep_candidates))

    except Exception as e_pool:
        log_with_elapsed_time(f"  ERROR during parallel evaluation: {type(e_pool).__name__}: {e_pool}\n{traceback.format_exc(limit=2)}")
        log_with_elapsed_time("  Switching to sequential evaluation...")
        costs = []
        eval_start_seq = time.time()
        total_candidates = len(ep_candidates)
        for i, cand in enumerate(ep_candidates):
            try:
                cost = cost_function_partial_map(cand)
                costs.append(cost)
            except Exception as e_seq:
                costs.append(np.inf) # Assign inf cost if sequential evaluation fails
                log_with_elapsed_time(f"  Error evaluating candidate {i+1} sequentially: {e_seq}")

            if (i + 1) % 200 == 0 or i == total_candidates - 1:
                log_with_elapsed_time(f"  ... Evaluated {i + 1}/{total_candidates} sequentially...")
                if progress_hook:
                     current_progress = (i + 1) / total_candidates
                     progress_hook(current_progress) # Update progress

        eval_seq_time = time.time() - eval_start_seq
        initial_results = list(zip(costs, ep_candidates))
        log_with_elapsed_time(f"  Sequential evaluation finished in {eval_seq_time:.2f}s.")

    # Filter out non-finite costs before sorting
    valid_initial_results = [(c, p) for c, p in initial_results if np.isfinite(c) and c < HIGH_COST_PENALTY] # Filter high costs too?
    num_invalid = len(initial_results) - len(valid_initial_results)
    if num_invalid > 0:
         log_with_elapsed_time(f"  Filtering: {num_invalid} invalid/high initial costs discarded.")

    if not valid_initial_results:
        log_with_elapsed_time(f"Warning: No valid initial costs found after {phase_name} evaluation.")
        return []

    # Sort by cost (ascending)
    valid_initial_results.sort(key=lambda x: x[0])
    eval_time = time.time() - start_time_eval
    log_with_elapsed_time(f"--- End {phase_name} (Initial Evaluation) in {eval_time:.2f}s ---")
    if valid_initial_results:
        log_with_elapsed_time(f"  Found {len(valid_initial_results)} valid points. Best initial cost: {valid_initial_results[0][0]:.3e}")

    return valid_initial_results


def _run_parallel_local_search(
    p_best_starts: List[np.ndarray],
    cost_function: Callable[..., float],
    args_for_cost_tuple: Tuple,
    lbfgsb_bounds: List[Tuple[Optional[float], Optional[float]]],
    min_thickness_phys_nm: float,
    progress_hook: Optional[Callable[[float], None]] = None # Progress hook might be hard to implement here
) -> List[Dict[str, Any]]:

    p_best_actual = len(p_best_starts)
    if p_best_actual == 0:
        log_with_elapsed_time("WARNING: No starting points provided for local search. Skipping Phase 2.")
        return []

    log_with_elapsed_time(f"\n--- Phase 2: Local Search (L-BFGS-B) (P={p_best_actual}) ---")
    start_time_local = time.time()

    # Create the partial function for the worker
    local_search_partial = partial(local_search_worker,
                                   cost_function=cost_function,
                                   args_for_cost=args_for_cost_tuple,
                                   lbfgsb_bounds=lbfgsb_bounds,
                                   min_thickness_phys_nm=min_thickness_phys_nm)

    local_results_raw: List[Dict[str, Any]] = []
    num_workers_local = min(p_best_actual, os.cpu_count())
    log_with_elapsed_time(f"  Starting {p_best_actual} local searches in parallel (max {num_workers_local} workers)...")
    local_start_pool = time.time()

    try:
        with multiprocessing.Pool(processes=num_workers_local) as pool:
            # Use map to apply the partial function to each starting point
            local_results_raw = list(pool.map(local_search_partial, p_best_starts))
        local_pool_time = time.time() - local_start_pool
        log_with_elapsed_time(f"  Parallel local searches finished in {local_pool_time:.2f}s.")
    except Exception as e_pool_local:
        log_with_elapsed_time(f"  ERROR during parallel local search pool execution: {type(e_pool_local).__name__}: {e_pool_local}\n{traceback.format_exc(limit=2)}")
        local_results_raw = [] # Ensure it's an empty list on error

    local_time = time.time() - start_time_local
    log_with_elapsed_time(f"--- End Phase 2 (Local Search) in {local_time:.2f}s ---")

    # Basic check on results structure
    if len(local_results_raw) != p_best_actual:
         log_with_elapsed_time(f"Warning: Number of local search results ({len(local_results_raw)}) does not match number of starts ({p_best_actual}).")

    return local_results_raw


def _process_optimization_results(
    local_results_raw: List[Dict[str, Any]],
    initial_best_ep: Optional[np.ndarray],
    initial_best_cost: float
) -> Tuple[Optional[np.ndarray], float, OptimizeResult]:

    log_with_elapsed_time(f"\n--- Processing {len(local_results_raw)} local search results ---")

    overall_best_cost: float = initial_best_cost if np.isfinite(initial_best_cost) else np.inf
    overall_best_ep: Optional[np.ndarray] = initial_best_ep.copy() if initial_best_ep is not None else None
    # Default result object based on the initial best (if any)
    overall_best_result_obj: OptimizeResult = OptimizeResult(
        x=overall_best_ep,
        fun=overall_best_cost,
        success=(overall_best_ep is not None and np.isfinite(overall_best_cost)),
        message="Initial best (pre-local search)",
        nit=0
    )
    best_run_info: Dict[str, Any] = {'index': -1, 'cost': overall_best_cost}
    processed_results_count: int = 0

    if overall_best_ep is None:
        log_with_elapsed_time("Warning: Initial best EP for processing is None.")

    for i, worker_output in enumerate(local_results_raw):
        run_idx = i + 1
        # Log worker messages first
        if isinstance(worker_output, dict) and 'log_lines' in worker_output:
            worker_log_lines = worker_output.get('log_lines', [])
            for line in worker_log_lines:
                # Indent worker logs for clarity
                log_with_elapsed_time(f"    (Worker Log Run {run_idx}): {line.strip()}")
        else:
            log_with_elapsed_time(f"  Result {run_idx}: Unexpected format from worker, cannot log details. Output: {worker_output}")
            continue # Skip processing this result

        # Check if the result dictionary has the expected keys
        if 'result' in worker_output and 'final_ep' in worker_output and 'final_cost' in worker_output and 'success' in worker_output:
            res_obj = worker_output['result']
            final_cost_run = worker_output['final_cost'] # Already checked for finite inside worker? Assume worker returns finite or inf.
            final_ep_run = worker_output['final_ep']
            success_run = worker_output['success'] # Indicates LBFGSB success AND finite cost
            processed_results_count += 1

            if success_run and final_ep_run is not None and final_ep_run.size > 0 and final_cost_run < overall_best_cost:
                 log_with_elapsed_time(f"  ==> New overall best cost found by Result {run_idx}! Cost: {final_cost_run:.3e} (Previous: {overall_best_cost:.3e})")
                 overall_best_cost = final_cost_run
                 overall_best_ep = final_ep_run.copy() # Take a copy
                 overall_best_result_obj = res_obj # Store the corresponding OptimizeResult
                 best_run_info = {'index': run_idx, 'cost': final_cost_run}
            # Optional: Log if a run was successful but didn't improve the best
            # elif success_run:
            #      log_with_elapsed_time(f"  Result {run_idx}: Successful run, cost {final_cost_run:.3e} (did not improve best {overall_best_cost:.3e})")

        else:
            log_with_elapsed_time(f"  Result {run_idx}: Worker output dictionary missing required keys.")
            # Optionally log the content of worker_output for debugging

    log_with_elapsed_time(f"--- Finished processing local search results ({processed_results_count} processed) ---")

    if processed_results_count == 0 and best_run_info['index'] == -1:
        log_with_elapsed_time("WARNING: No valid local search results were processed. Keeping pre-local best result.")
    elif best_run_info['index'] > 0:
        log_with_elapsed_time(f"Overall best result found by local search Run {best_run_info['index']} (Cost {best_run_info['cost']:.3e}).")
    elif best_run_info['index'] == -1: # Initial cost was finite, but no local search improved it
         log_with_elapsed_time(f"No local search improved the pre-local search best result (Cost {overall_best_cost:.3e}).")
    # Case: initial cost was inf, no local search found a finite result -> handled by default values


    # Final check: Ensure we have a valid thickness vector if possible
    if overall_best_ep is None or len(overall_best_ep) == 0:
        log_with_elapsed_time("CRITICAL WARNING: Result processing finished with an empty or None best thickness vector!")
        # Try falling back to the absolute initial state if it existed
        if initial_best_ep is not None and initial_best_ep.size > 0:
             log_with_elapsed_time("Falling back to initial state provided to processing function.")
             overall_best_ep = initial_best_ep.copy()
             overall_best_cost = initial_best_cost if np.isfinite(initial_best_cost) else np.inf
             overall_best_result_obj = OptimizeResult(x=overall_best_ep, fun=overall_best_cost, success=False, message="Fell back to initial state", nit=0)
        else:
             # This should ideally not happen if the process starts correctly
             log_with_elapsed_time("Error: Cannot fall back, initial state was also invalid.")
             # Return None/inf and default result object
             overall_best_ep = None
             overall_best_cost = np.inf
             overall_best_result_obj = OptimizeResult(x=None, fun=np.inf, success=False, message="Processing failed, no valid state", nit=0)


    return overall_best_ep, overall_best_cost, overall_best_result_obj


def run_optimization_process(
    inputs: Dict[str, Any],
    active_targets: List[Dict[str, float]],
    ep_nominal_for_cost: Optional[np.ndarray],
    p_best_this_pass: int,
    ep_reference_for_sobol: Optional[np.ndarray] = None,
    current_scaling: Optional[float] = None,
    run_number: int = 1,
    total_passes: int = 1,
    progress_bar_hook: Optional[Any] = None, # Streamlit progress bar object
    status_placeholder: Optional[Any] = None, # Streamlit empty object
    current_best_mse: float = np.inf,
    current_best_layers: int = 0
) -> Tuple[Optional[np.ndarray], float, OptimizeResult]:

    run_id_str = f"Pass {run_number}/{total_passes}"
    log_with_elapsed_time("\n" + "*"*10 + f" STARTING OPTIMIZATION - {run_id_str} " + "*"*10)
    start_time_run = time.time()

    # Extract necessary parameters
    n_samples = inputs['n_samples'] # Base N samples from input
    nH = inputs['nH_r'] + 1j * inputs['nH_i']
    nL = inputs['nL_r'] + 1j * inputs['nL_i']
    nSub_c = inputs['nSub'] + 0j
    l_step_gui = inputs['l_step']
    l_min_overall = inputs['l_range_deb']
    l_max_overall = inputs['l_range_fin']

    num_layers: int = 0
    lower_bounds: np.ndarray = np.array([])
    upper_bounds: np.ndarray = np.array([])

    pass_status_prefix = f"Status: {run_id_str}"
    status_suffix = f"| Best MSE: {current_best_mse:.3e} | Layers: {current_best_layers}"

    if status_placeholder: status_placeholder.info(f"{pass_status_prefix} - Phase 1 (Sobol Sampling) {status_suffix}")

    # Determine Sobol bounds and num_layers based on pass number
    if run_number == 1:
        if ep_nominal_for_cost is None or ep_nominal_for_cost.size == 0:
            raise ValueError(f"Nominal thickness vector ({run_id_str}) is empty or None.")
        ep_ref_pass1 = np.maximum(MIN_THICKNESS_PHYS_NM, ep_nominal_for_cost)
        num_layers = len(ep_ref_pass1)
        lower_bounds = np.maximum(MIN_THICKNESS_PHYS_NM, ep_ref_pass1 * INITIAL_SOBOL_REL_SCALE_LOWER)
        upper_bounds = ep_ref_pass1 * INITIAL_SOBOL_REL_SCALE_UPPER
        upper_bounds = np.maximum(upper_bounds, lower_bounds + ZERO_THRESHOLD) # Ensure upper > lower
        log_with_elapsed_time(f"  Using relative scaling [{INITIAL_SOBOL_REL_SCALE_LOWER:.2f}, {INITIAL_SOBOL_REL_SCALE_UPPER:.2f}] around nominal for Sobol ({run_id_str}).")
    else: # Passes > 1
        if ep_reference_for_sobol is None or ep_reference_for_sobol.size == 0:
             raise ValueError(f"Reference thickness vector ({run_id_str}) is empty or None for absolute scaling.")
        if current_scaling is None or current_scaling <= 0:
             raise ValueError(f"Invalid current_scaling value ({current_scaling}) for absolute Sobol ({run_id_str}).")
        ep_ref = np.maximum(MIN_THICKNESS_PHYS_NM, ep_reference_for_sobol)
        num_layers = len(ep_ref)
        lower_bounds = np.maximum(MIN_THICKNESS_PHYS_NM, ep_ref - current_scaling)
        upper_bounds = ep_ref + current_scaling
        upper_bounds = np.maximum(upper_bounds, lower_bounds + ZERO_THRESHOLD) # Ensure upper > lower
        log_with_elapsed_time(f"  Using absolute scaling +/- {current_scaling:.3f} nm (min {MIN_THICKNESS_PHYS_NM:.3f} nm) around previous best for Sobol ({run_id_str}).")

    if num_layers == 0: raise ValueError(f"Could not determine number of layers for Sobol ({run_id_str}).")

    # --- Prepare for Cost Evaluation ---
    if l_min_overall <= 0 or l_max_overall <= 0: raise ValueError("Wavelength range limits must be positive.")
    if l_max_overall <= l_min_overall: raise ValueError("Wavelength range max must be greater than min.")
    num_points_approx = max(2, int(np.round((l_max_overall - l_min_overall) / l_step_gui)) + 1)
    # Using geomspace for optimization grid - ensure endpoints are included if needed
    try:
        l_vec_optim = np.geomspace(l_min_overall, l_max_overall, num_points_approx)
        l_vec_optim = l_vec_optim[(l_vec_optim > 0) & np.isfinite(l_vec_optim)] # Filter invalid values
        if not l_vec_optim.size: raise ValueError("Geomspace resulted in empty vector.")
    except Exception as e_geom:
         raise ValueError(f"Failed to generate geomspace optimization wavelength vector: {e_geom}")

    log_with_elapsed_time(f"  Generated {len(l_vec_optim)} optimization points geometrically for cost evaluation.")

    args_for_cost_tuple: Tuple = (nH, nL, nSub_c, l_vec_optim, active_targets, MIN_THICKNESS_PHYS_NM)
    cost_function_partial_map = partial(calculate_mse_for_optimization_penalized,
                                        nH=nH, nL=nL, nSub=nSub_c, l_vec_optim=l_vec_optim,
                                        active_targets=active_targets,
                                        min_thickness_phys_nm=MIN_THICKNESS_PHYS_NM)

    # --- Phase 1: Sobol Evaluation ---
    sobol_progress = None # Define how progress is reported if needed
    if progress_bar_hook:
         def sobol_progress_update(frac):
             # Assuming phase 1 takes ~50% of the time for this pass
             progress_bar_hook.progress( (0.0 + frac * 0.5) / total_passes) # Rough estimate
         sobol_progress = sobol_progress_update

    valid_initial_results_p1 = _run_sobol_evaluation(
        num_layers, n_samples, lower_bounds, upper_bounds,
        cost_function_partial_map, MIN_THICKNESS_PHYS_NM,
        phase_name=f"{run_id_str} Phase 1",
        progress_hook=sobol_progress
    )

    # --- Phase 1bis (Only for Pass 1) ---
    top_p_results_combined: List[Tuple[float, np.ndarray]] = []
    if run_number == 1:
        if status_placeholder: status_placeholder.info(f"{pass_status_prefix} - Phase 1bis (Refining Starts) {status_suffix}")
        log_with_elapsed_time(f"\n--- {run_id_str} Phase 1bis: Refining Top P Starts ---")
        num_to_select_p1 = min(p_best_this_pass, len(valid_initial_results_p1))

        if num_to_select_p1 == 0:
            log_with_elapsed_time("WARNING: Phase 1 yielded no valid starting points. Skipping refinement.")
            top_p_results_combined = []
        else:
            top_p_results_p1 = valid_initial_results_p1[:num_to_select_p1]
            top_p_starts_p1 = [ep for cost, ep in top_p_results_p1]

            # Calculate samples per point for refinement
            n_samples_per_point_raw = max(1, n_samples // p_best_this_pass) if p_best_this_pass > 0 else n_samples
            n_samples_per_point = upper_power_of_2(n_samples_per_point_raw) # Use power of 2 for Sobol?
            log_with_elapsed_time(f"  Generating {n_samples_per_point} samples around each of the top {num_to_select_p1} points (scaling +/- {PHASE1BIS_SCALING_NM} nm).")

            phase1bis_candidates: List[np.ndarray] = []
            for i, ep_start in enumerate(top_p_starts_p1):
                lower_bounds_1bis = np.maximum(MIN_THICKNESS_PHYS_NM, ep_start - PHASE1BIS_SCALING_NM)
                upper_bounds_1bis = ep_start + PHASE1BIS_SCALING_NM
                upper_bounds_1bis = np.maximum(upper_bounds_1bis, lower_bounds_1bis + ZERO_THRESHOLD)
                try:
                    sampler_1bis = qmc.Sobol(d=num_layers, scramble=True, seed=int(time.time()) + i) # Vary seed
                    points_unit_1bis = sampler_1bis.random(n=n_samples_per_point)
                    new_candidates_raw = qmc.scale(points_unit_1bis, lower_bounds_1bis, upper_bounds_1bis)
                    new_candidates = [np.maximum(MIN_THICKNESS_PHYS_NM, cand) for cand in new_candidates_raw]
                    phase1bis_candidates.extend(new_candidates)
                except Exception as e_sobol1bis:
                     log_with_elapsed_time(f"Error during Sobol sampling for point {i+1} in Phase 1bis: {e_sobol1bis}")
                     # Continue with other points

            log_with_elapsed_time(f"  Generated {len(phase1bis_candidates)} total candidates for Phase 1bis evaluation.")

            # Evaluate Phase 1bis candidates (similar to _run_sobol_evaluation's parallel part)
            results_1bis_raw_pairs: List[Tuple[float, np.ndarray]] = []
            if phase1bis_candidates:
                num_workers_1bis = min(len(phase1bis_candidates), os.cpu_count())
                log_with_elapsed_time(f"  Evaluating cost for {len(phase1bis_candidates)} Phase 1bis candidates (max {num_workers_1bis} workers)...")
                eval_start_pool_1bis = time.time()
                try:
                    with multiprocessing.Pool(processes=num_workers_1bis) as pool:
                        costs_1bis = pool.map(cost_function_partial_map, phase1bis_candidates)
                    eval_pool_time_1bis = time.time() - eval_start_pool_1bis
                    log_with_elapsed_time(f"  Parallel evaluation (Phase 1bis) finished in {eval_pool_time_1bis:.2f}s.")
                    if len(costs_1bis) == len(phase1bis_candidates):
                        results_1bis_raw_pairs = list(zip(costs_1bis, phase1bis_candidates))
                    else:
                         log_with_elapsed_time("Warning: Mismatch in Phase 1bis results length.")
                except Exception as e_pool_1bis:
                    log_with_elapsed_time(f"  ERROR during parallel evaluation (Phase 1bis): {e_pool_1bis}")
                    # Potentially fallback to sequential or just use Phase 1 results

            # Combine and select best P for Phase 2
            valid_results_1bis = [(c, p) for c, p in results_1bis_raw_pairs if np.isfinite(c) and c < HIGH_COST_PENALTY]
            num_invalid_1bis = len(results_1bis_raw_pairs) - len(valid_results_1bis)
            if num_invalid_1bis > 0: log_with_elapsed_time(f"  Filtering (Phase 1bis): {num_invalid_1bis} invalid/high costs discarded.")

            combined_results = top_p_results_p1 + valid_results_1bis
            log_with_elapsed_time(f"  Combined Phase 1 ({len(top_p_results_p1)}) and Phase 1bis ({len(valid_results_1bis)}) valid results: {len(combined_results)} total.")

            if not combined_results:
                log_with_elapsed_time("WARNING: No valid results found after combining Phase 1 and Phase 1bis.")
                top_p_results_combined = []
            else:
                combined_results.sort(key=lambda x: x[0])
                num_to_select_final = min(p_best_this_pass, len(combined_results))
                log_with_elapsed_time(f"  Selecting final top {num_to_select_final} points from combined results for Phase 2.")
                top_p_results_combined = combined_results[:num_to_select_final]
        log_with_elapsed_time(f"--- End {run_id_str} Phase 1bis ---")

    else: # Passes > 1, just take top P from Phase 1
        num_to_select = min(p_best_this_pass, len(valid_initial_results_p1))
        log_with_elapsed_time(f"  Selecting top {num_to_select} points from Phase 1 for Phase 2.")
        top_p_results_combined = valid_initial_results_p1[:num_to_select]


    # --- Prepare for Phase 2 ---
    selected_starts = [ep for cost, ep in top_p_results_combined]
    selected_costs = [cost for cost, ep in top_p_results_combined]

    if status_placeholder: status_placeholder.info(f"{pass_status_prefix} - Phase 2 (Local Search) {status_suffix}")

    overall_best_ep: Optional[np.ndarray] = None
    overall_best_cost: float = np.inf
    overall_best_result_obj: OptimizeResult = OptimizeResult(x=None, fun=np.inf, success=False, message="Optimization did not yield a valid result yet.")

    if not selected_starts:
        log_with_elapsed_time(f"WARNING: No starting points available for Phase 2 ({run_id_str}). Skipping.")
        # If no starts, the result is the best from previous pass (or nominal if pass 1)
        # This case needs careful handling in the main optimization loop calling this function.
        # We return None/inf here, the caller must decide what to keep.
        overall_best_ep = None
        overall_best_cost = np.inf
        overall_best_result_obj = OptimizeResult(x=None, fun=np.inf, success=False, message=f"Phase 2 skipped - no starts ({run_id_str})", nit=0)
    else:
        # --- Phase 2: Parallel Local Search ---
        initial_best_ep_for_processing = selected_starts[0].copy() # Best guess before local search
        initial_best_cost_for_processing = selected_costs[0]
        log_with_elapsed_time(f"  Top {len(selected_starts)} points selected for Phase 2:")
        for i in range(min(5, len(selected_starts))):
            cost_str = f"{selected_costs[i]:.3e}" if np.isfinite(selected_costs[i]) else "inf"
            log_with_elapsed_time(f"    Point {i+1}: Cost={cost_str}") # Log first few starting costs
        if len(selected_starts) > 5: log_with_elapsed_time("    ...")

        lbfgsb_bounds = [(MIN_THICKNESS_PHYS_NM, None)] * num_layers

        local_results_raw = _run_parallel_local_search(
            selected_starts,
            calculate_mse_for_optimization_penalized, # Pass the penalized cost func directly
            args_for_cost_tuple,
            lbfgsb_bounds,
            MIN_THICKNESS_PHYS_NM,
            progress_hook=None # Progress hook for map is tricky
        )

        # --- Process Phase 2 Results ---
        if status_placeholder: status_placeholder.info(f"{pass_status_prefix} - Processing L-BFGS-B results... {status_suffix}")
        overall_best_ep, overall_best_cost, overall_best_result_obj = _process_optimization_results(
            local_results_raw,
            initial_best_ep_for_processing,
            initial_best_cost_for_processing
        )

    # --- Log Pass Summary ---
    run_time_pass = time.time() - start_time_run
    log_with_elapsed_time(f"--- End Optimization {run_id_str} in {run_time_pass:.2f}s ---")
    log_with_elapsed_time(f"Best cost found this pass ({run_id_str}): {overall_best_cost:.3e}")
    if overall_best_ep is not None and overall_best_ep.size > 0:
        layers_this_pass = len(overall_best_ep)
        ep_str = ", ".join([f"{th:.3f}" for th in overall_best_ep[:10]])
        if layers_this_pass > 10: ep_str += "..."
        log_with_elapsed_time(f"Thicknesses this pass ({run_id_str}, {layers_this_pass} layers): [{ep_str}]")
    else:
        log_with_elapsed_time(f"WARNING: No valid thickness vector found for {run_id_str}.")

    return overall_best_ep, overall_best_cost, overall_best_result_obj


# --- Plotting Function ---
def setup_axis_grids(ax: plt.Axes) -> None:
    """Adds major and minor grids to a matplotlib axis."""
    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.7, alpha=0.7)
    ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.minorticks_on()

def tracer_graphiques(
    res: Optional[Dict[str, np.ndarray]],
    ep_actual: Optional[np.ndarray],
    nH_r: float, nH_i: float, nL_r: float, nL_i: float, nSub: float, # Pass nSub real part for consistency
    active_targets_for_plot: List[Dict[str, float]],
    mse: Optional[float],
    is_optimized: bool = False,
    method_name: str = "",
    res_optim_grid: Optional[Dict[str, np.ndarray]] = None
) -> plt.Figure:
    """Generates the summary plot figure."""

    # --- Initialize Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    opt_method_str = f" ({method_name})" if method_name else ""
    window_title = f'Stack Results{opt_method_str}' if is_optimized else 'Nominal Stack Calculation Results'
    fig.suptitle(window_title, fontsize=14, weight='bold')

    # --- Plot Data Preparation ---
    num_layers = len(ep_actual) if ep_actual is not None else 0
    ep_cum = np.cumsum(ep_actual) if num_layers > 0 and ep_actual is not None else np.array([])
    total_thickness = ep_cum[-1] if num_layers > 0 else 0

    # --- 1. Spectral Plot (Transmittance) ---
    ax_spec = axes[0]
    ax_spec.set_title(f"Spectral Plot{opt_method_str}")
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel('Transmittance')
    ax_spec.set_ylim(bottom=-0.05, top=1.05)
    setup_axis_grids(ax_spec)

    if res and 'l' in res and 'Ts' in res and res['l'] is not None and res['Ts'] is not None and res['l'].size > 0:
        line_ts, = ax_spec.plot(res['l'], res['Ts'], label='Transmittance (Plot Grid)', linestyle='-', color='blue', linewidth=1.5)
        if len(res['l']) > 0: ax_spec.set_xlim(res['l'][0], res['l'][-1])

        # Plot Targets
        target_lines_drawn = False
        if active_targets_for_plot:
            plotted_label = False
            for target in active_targets_for_plot:
                l_min, l_max = target['min'], target['max']
                t_min, t_max = target['target_max'] # Typo in original? Should be t_max
                target_t_min = target['target_min'] # Get the correct key

                x_coords = [l_min, l_max]
                y_coords = [target_t_min, t_max] # Use correct variables
                label = 'Target Ramp' if not plotted_label else "_nolegend_"
                ax_spec.plot(x_coords, y_coords, 'r--', linewidth=1.5, alpha=0.8, label=label, zorder=5)
                plotted_label = True
                ax_spec.plot(x_coords, y_coords, marker='x', color='red', markersize=8, linestyle='none', label='_nolegend_', zorder=6)
                target_lines_drawn = True

                # Plot target points on optimization grid if available
                if res_optim_grid and 'l' in res_optim_grid and res_optim_grid['l'].size > 0 and 'Ts' in res_optim_grid:
                    indices_in_zone_optim = np.where((res_optim_grid['l'] >= l_min) & (res_optim_grid['l'] <= l_max))[0]
                    if indices_in_zone_optim.size > 0:
                        optim_lambdas_in_zone = res_optim_grid['l'][indices_in_zone_optim]
                        # Interpolate target T values on optim grid
                        if abs(l_max - l_min) < MSE_TARGET_LAMBDA_DIFF_THRESHOLD:
                             optim_target_t_in_zone = np.full_like(optim_lambdas_in_zone, target_t_min)
                        else:
                             slope = (t_max - target_t_min) / (l_max - l_min)
                             optim_target_t_in_zone = target_t_min + slope * (optim_lambdas_in_zone - l_min)
                        ax_spec.plot(optim_lambdas_in_zone, optim_target_t_in_zone,
                                     marker='.', color='orangered', linestyle='none', markersize=4,
                                     alpha=0.7, label='_nolegend_', zorder=6) # Changed color slightly

        # Add legend if needed
        handles, labels = ax_spec.get_legend_handles_labels()
        if handles: # Check if there are any labels to show
             ax_spec.legend(fontsize=9)

        # Add MSE text
        mse_text = "MSE: N/A"
        if mse is not None and not np.isnan(mse):
             mse_text = f"MSE (vs Target, Optim grid) = {mse:.3e}"
        elif mse is None and active_targets_for_plot: mse_text = "MSE: Calculation Error"
        elif active_targets_for_plot: mse_text = "MSE: N/A (no target points)" # If mse is NaN
        else: mse_text = "MSE: N/A (no target defined)"

        ax_spec.text(0.98, 0.98, mse_text, transform=ax_spec.transAxes, ha='right', va='top', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
    else:
        ax_spec.text(0.5, 0.5, "No spectral data available", ha='center', va='center', transform=ax_spec.transAxes)

    # --- 2. Refractive Index Profile ---
    ax_idx = axes[1]
    ax_idx.set_title("Refractive Index Profile")
    ax_idx.set_xlabel('Depth (from substrate) (nm)')
    ax_idx.set_ylabel("Real part of index (n')")
    setup_axis_grids(ax_idx)

    nSub_c = complex(nSub) # Assuming nSub passed is real part, make it complex
    nH_c_calc = complex(nH_r, nH_i)
    nL_c_calc = complex(nL_r, nL_i)

    indices_complex = [(nH_c_calc if i % 2 == 0 else nL_c_calc) for i in range(num_layers)]
    n_real_layers = [np.real(n) for n in indices_complex]

    margin = max(50, 0.1 * total_thickness) if total_thickness > 0 else 50
    x_coords_plot = [-margin]
    y_coords_plot = [np.real(nSub_c)]
    x_coords_plot.append(0)
    y_coords_plot.append(np.real(nSub_c))

    if num_layers > 0 and ep_actual is not None:
        current_depth = 0.0
        for i in range(num_layers):
            layer_n_real = n_real_layers[i]
            layer_thickness = ep_actual[i]
            x_coords_plot.append(current_depth)
            y_coords_plot.append(layer_n_real)
            current_depth += layer_thickness
            x_coords_plot.append(current_depth)
            y_coords_plot.append(layer_n_real)

        # Transition to Air
        last_layer_end_depth = current_depth
        x_coords_plot.append(last_layer_end_depth)
        y_coords_plot.append(1.0) # Air index
        x_coords_plot.append(last_layer_end_depth + margin)
        y_coords_plot.append(1.0)
    else: # No layers
        x_coords_plot.append(0)
        y_coords_plot.append(1.0)
        x_coords_plot.append(margin)
        y_coords_plot.append(1.0)

    ax_idx.plot(x_coords_plot, y_coords_plot, drawstyle='steps-post', label='Real n', color='purple', linewidth=1.5)
    ax_idx.set_xlim(x_coords_plot[0], x_coords_plot[-1])

    # Set Y limits based on actual indices
    min_n_list = [1.0, np.real(nSub_c)] + n_real_layers
    max_n_list = [1.0, np.real(nSub_c)] + n_real_layers
    min_n = min(min_n_list) if min_n_list else 0.9
    max_n = max(max_n_list) if max_n_list else 2.5
    ax_idx.set_ylim(bottom=min_n - 0.1, top=max_n + 0.1)

    # Add text labels for Substrate and Air
    offset = (max_n - min_n) * 0.05 + 0.02
    common_text_opts: Dict[str, Any] = {'ha':'center', 'va':'bottom', 'fontsize':8, 'bbox':dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none')}
    sub_text = f"SUBSTRATE\nn={nSub_c.real:.3f}"
    if abs(nSub_c.imag) > ZERO_THRESHOLD: sub_text += f"{nSub_c.imag:+.3f}j"
    ax_idx.text(-margin / 2, np.real(nSub_c) + offset, sub_text, **common_text_opts)

    air_x_pos = (total_thickness + margin / 2) if num_layers > 0 else margin / 2
    ax_idx.text(air_x_pos, 1.0 + offset, "AIR\nn=1.0", **common_text_opts)


    # --- 3. Stack Bar Chart ---
    ax_stack = axes[2]
    stack_title_prefix = f'Optimized Stack{opt_method_str}' if is_optimized else 'Nominal Stack'
    ax_stack.set_title(f"{stack_title_prefix} ({num_layers} layers)\n(Substrate at bottom -> Air at top)")
    ax_stack.set_xlabel('Thickness (nm)')

    if num_layers > 0 and ep_actual is not None:
        colors = ['lightblue' if i % 2 == 0 else 'lightcoral' for i in range(num_layers)]
        bar_pos = np.arange(num_layers)
        bars = ax_stack.barh(bar_pos, ep_actual, align='center', color=colors, edgecolor='grey', height=0.8)

        yticks_labels: List[str] = []
        for i, n_comp in enumerate(indices_complex):
            layer_type = "H" if i % 2 == 0 else "L"
            n_str = f"{np.real(n_comp):.3f}"
            k_val = np.imag(n_comp)
            if abs(k_val) > ZERO_THRESHOLD: # Use threshold for displaying k
                 n_str += f"{k_val:+.3f}j"
            label = f"L{i + 1} ({layer_type}) n={n_str}"
            yticks_labels.append(label)

        ax_stack.set_yticks(bar_pos)
        ax_stack.set_yticklabels(yticks_labels, fontsize=8)
        ax_stack.invert_yaxis() # Layer 1 at the top (closer to Air)

        # Add thickness labels on bars
        max_ep_val = np.max(ep_actual) if ep_actual.size > 0 else 1.0
        fontsize = max(6, 9 - num_layers // 10) # Dynamic font size
        for i, bar in enumerate(bars):
            e_val = bar.get_width()
            # Position text inside or outside bar based on width
            ha_pos = 'left' if e_val < max_ep_val * 0.2 else 'right'
            x_text_pos = e_val * 1.05 if ha_pos == 'left' else e_val * 0.95
            text_color = 'black' if ha_pos == 'left' else 'white'
            ax_stack.text(x_text_pos, bar.get_y() + bar.get_height() / 2, f"{e_val:.2f} nm",
                          va='center', ha=ha_pos, color=text_color, fontsize=fontsize, weight='bold')

        ax_stack.set_ylim(bottom=num_layers - 0.5, top=-0.5) # Adjust y-limits for bars

    else: # No layers
         ax_stack.text(0.5, 0.5, "No layers defined", ha='center', va='center', fontsize=10, color='grey', transform=ax_stack.transAxes)
         ax_stack.set_yticks([])
         ax_stack.set_xticks([])

    # --- Final Adjustments ---
    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap
    return fig

# --- Streamlit UI and Logic ---

# Helper functions for Streamlit state and validation
def get_active_targets_from_state() -> Optional[List[Dict[str, float]]]:
    active_targets: List[Dict[str, float]] = []
    if 'targets' not in st.session_state:
        log_message("ERROR: Target definitions not found in session state.")
        st.error("Target definitions not found in session state.")
        return None

    overall_lambda_min: Optional[float] = None
    overall_lambda_max: Optional[float] = None

    for i, target_def in enumerate(st.session_state.targets):
        if target_def.get('enabled', False):
            try:
                l_min_str = target_def.get('min', '')
                l_max_str = target_def.get('max', '')
                t_min_str = target_def.get('target_min', '')
                t_max_str = target_def.get('target_max', '')

                if not all([l_min_str, l_max_str, t_min_str, t_max_str]):
                    raise ValueError(f"Target Zone {i+1}: Required fields missing or empty.")

                l_min = float(l_min_str)
                l_max = float(l_max_str)
                t_min = float(t_min_str)
                t_max = float(t_max_str)

                if l_max < l_min: raise ValueError(f"Target Zone {i+1}: λ max ({l_max}) must be >= λ min ({l_min}).")
                if not (0.0 <= t_min <= 1.0): raise ValueError(f"Target Zone {i+1}: Target T @ min ({t_min}) must be between 0 and 1.")
                if not (0.0 <= t_max <= 1.0): raise ValueError(f"Target Zone {i+1}: Target T @ max ({t_max}) must be between 0 and 1.")
                if l_min <=0 or l_max <=0: raise ValueError(f"Target Zone {i+1}: Wavelengths ({l_min}, {l_max}) must be > 0.")

                active_targets.append({'min': l_min, 'max': l_max, 'target_min': t_min, 'target_max': t_max})

                # Track overall range of active targets
                if overall_lambda_min is None or l_min < overall_lambda_min: overall_lambda_min = l_min
                if overall_lambda_max is None or l_max > overall_lambda_max: overall_lambda_max = l_max

            except (ValueError, TypeError) as e:
                log_message(f"ERROR Spectral Target Configuration {i+1}: {e}")
                st.error(f"Error in Spectral Target Zone {i+1}: {e}")
                return None # Fail validation if any target is invalid

    # Optional: Validate target range against calculation range
    try:
        calc_l_min = float(st.session_state.l_range_deb_input)
        calc_l_max = float(st.session_state.l_range_fin_input)
        if overall_lambda_min is not None and overall_lambda_max is not None:
             if overall_lambda_min < calc_l_min or overall_lambda_max > calc_l_max:
                 st.warning(f"Calculation range [{calc_l_min:.1f}-{calc_l_max:.1f} nm] does not fully cover active target range [{overall_lambda_min:.1f}-{overall_lambda_max:.1f} nm]. Results might be suboptimal or MSE misleading.")
    except (ValueError, KeyError, AttributeError):
         # Handle cases where inputs might not be present or valid yet
         st.warning("Could not validate target range against calculation range due to invalid/missing range inputs.")
    except Exception as e:
         st.warning(f"Error during target/calculation range check: {e}")

    return active_targets


def _validate_physical_inputs_from_state(require_optim_params: bool = True) -> Dict[str, Any]:
    """Validates and converts UI inputs from session state."""
    values: Dict[str, Any] = {}
    # Define mappings from target key to session state key and type
    field_map: Dict[str, Tuple[str, type]] = {
        'nH_r': ('nH_r', float), 'nH_i': ('nH_i', float),
        'nL_r': ('nL_r', float), 'nL_i': ('nL_i', float),
        'nSub': ('nSub', float), 'l0': ('l0_input', float),
        'l_range_deb': ('l_range_deb_input', float),
        'l_range_fin': ('l_range_fin_input', float),
        'l_step': ('l_step_input', float),
        'scaling_nm': ('scaling_nm_input', float),
        'emp_str': ('emp_str_input', str)
    }
    if require_optim_params:
        field_map.update({
            'n_samples': ('n_samples_input', int),
            'p_best': ('p_best_input', int),
            'n_passes': ('n_passes_input', int),
        })

    error_messages = []
    for target_key, (state_key, field_type) in field_map.items():
        if state_key not in st.session_state:
             # This indicates a programming error (key mismatch)
             raise ValueError(f"GUI State Error: Input key '{state_key}' not found in session state. App might need restart.")

        raw_val = st.session_state[state_key]
        try:
            if field_type == str:
                 values[target_key] = str(raw_val).strip()
            else:
                 values[target_key] = field_type(raw_val) # Convert to target type

            # Add specific validation rules
            if target_key in ['n_samples', 'p_best', 'n_passes'] and values[target_key] < 1:
                 error_messages.append(f"'{target_key.replace('_', ' ').title()}' ({state_key}) must be >= 1.")
            if target_key == 'scaling_nm' and values[target_key] < 0:
                 error_messages.append(f"'Scaling (nm)' ({state_key}) must be >= 0.")
            if target_key == 'l_step' and values[target_key] <= 0:
                 error_messages.append(f"'λ Step' ({state_key}) must be > 0.")
            if target_key in ['l_range_deb', 'l0'] and values[target_key] <= 0:
                 error_messages.append(f"'{target_key.replace('_', ' ').title()}' ({state_key}) must be > 0.")
            if target_key == 'nH_r' and values[target_key] <= 0: # Allow n=0? Usually n>=1
                 error_messages.append(f"'Material H (n real)' ({state_key}) must be > 0.")
            if target_key == 'nL_r' and values[target_key] <= 0:
                 error_messages.append(f"'Material L (n real)' ({state_key}) must be > 0.")
            if target_key == 'nSub' and values[target_key] <= 0:
                 error_messages.append(f"'Substrate (n real)' ({state_key}) must be > 0.")
            if target_key in ['nH_i', 'nL_i'] and values[target_key] < 0:
                 error_messages.append(f"Imaginary parts (k) ({state_key}) must be >= 0.")

        except (ValueError, TypeError) as e:
            error_messages.append(f"Invalid value for '{state_key}': '{raw_val}'. Expected: {field_type.__name__}. Error: {e}")

    # Cross-field validation
    if 'l_range_fin' in values and 'l_range_deb' in values:
         if values['l_range_fin'] < values['l_range_deb']:
              error_messages.append(f"λ End ({values['l_range_fin']}) must be >= λ Start ({values['l_range_deb']}).")
    if require_optim_params and 'p_best' in values and 'n_samples' in values:
         if values['p_best'] > values['n_samples']:
              error_messages.append(f"P Starts ({values['p_best']}) must be <= N Samples ({values['n_samples']}).")

    if error_messages:
        # Combine all errors into a single exception
        raise ValueError("Input validation failed:\n- " + "\n- ".join(error_messages))

    return values

def _prepare_calculation_data_st(
    inputs: Dict[str, Any],
    ep_vector_to_use: Optional[np.ndarray] = None
) -> Tuple[complex, complex, complex, np.ndarray, np.ndarray, np.ndarray]:
    nH = complex(inputs['nH_r'], inputs['nH_i'])
    nL = complex(inputs['nL_r'], inputs['nL_i'])
    nSub_c = complex(inputs['nSub'])

    l_step_gui = inputs['l_step']
    l_step_plot = max(ZERO_THRESHOLD, l_step_gui / PLOT_LAMBDA_STEP_DIVISOR)
    l_vec_plot = np.linspace(inputs['l_range_deb'], inputs['l_range_fin'],
                             max(2, int(np.round((inputs['l_range_fin'] - inputs['l_range_deb']) / l_step_plot)) + 1))
    if not l_vec_plot.size: raise ValueError("Spectral range/step yields no calculation points for plotting.")

    ep_actual_orig: np.ndarray
    if ep_vector_to_use is not None:
        ep_actual_orig = np.asarray(ep_vector_to_use, dtype=np.float64)
    else:
        try:
            ep_actual_orig, _ = get_initial_ep(inputs['emp_str'], inputs['l0'], nH, nL)
        except ValueError as e:
             raise ValueError(f"Error processing nominal QWOT: {e}")

    if ep_actual_orig.size == 0:
        if inputs['emp_str'].strip():
            raise ValueError("Nominal QWOT provided but resulted in an empty structure.")
        else:
            log_message("Empty structure (no QWOT provided). Calculating R/T for bare substrate.")
    elif not np.all(np.isfinite(ep_actual_orig)):
        ep_corrected = np.nan_to_num(ep_actual_orig, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.array_equal(ep_actual_orig, ep_corrected):
             log_message("WARNING: Original thicknesses contained NaN/inf, replaced with 0.")
             ep_actual_orig = ep_corrected
        if not np.all(np.isfinite(ep_actual_orig)):
             raise ValueError(f"Thickness vector contains non-finite values even after correction: {ep_actual_orig}")

    if ep_actual_orig.size > 0:
        ep_actual_calc = np.maximum(ep_actual_orig, MIN_THICKNESS_PHYS_NM)
    else:
        ep_actual_calc = ep_actual_orig

    return nH, nL, nSub_c, l_vec_plot, ep_actual_calc, ep_actual_orig


def update_display_info(ep_vector_source: Optional[np.ndarray] = None) -> None:
    num_layers: Union[int, str] = 0
    prefix = "Layers (Nominal): "
    ep_for_thin_check: Optional[np.ndarray] = None

    is_optimized = st.session_state.get('optimization_ran_since_nominal_change', False)
    current_opt_ep = st.session_state.get('current_optimized_ep')

    if is_optimized and current_opt_ep is not None:
        num_layers = len(current_opt_ep)
        prefix = "Layers (Optimized): "
        ep_for_thin_check = current_opt_ep
    elif ep_vector_source is not None:
        num_layers = len(ep_vector_source)
        prefix = "Layers (Nominal): "
        ep_for_thin_check = ep_vector_source
    else:
        try:
            emp_str = st.session_state.get('emp_str_input', '')
            emp_list = [item for item in emp_str.split(',') if item.strip()]
            num_layers = len(emp_list)
            ep_for_thin_check = None
        except Exception:
            num_layers = "Error"
        prefix = "Layers (Nominal): "

    st.session_state.num_layers_display = f"{prefix}{num_layers}"

    min_thickness_str = "- nm"
    if ep_for_thin_check is not None and len(ep_for_thin_check) > 0:
        valid_thicknesses = ep_for_thin_check[ep_for_thin_check >= MIN_THICKNESS_PHYS_NM]
        if valid_thicknesses.size > 0:
            min_thickness = np.min(valid_thicknesses)
            min_thickness_str = f"{min_thickness:.3f} nm"
        else:
            min_thickness_str = f"None ≥ {MIN_THICKNESS_PHYS_NM} nm"
    st.session_state.thinnest_layer_display = min_thickness_str


def run_calculation_st(
    ep_vector_to_use: Optional[np.ndarray] = None,
    is_optimized: bool = False,
    method_name: str = "",
    l_vec_override: Optional[np.ndarray] = None
) -> None:
    global status_placeholder, progress_placeholder, plot_placeholder # Allow modification

    log_message(f"\n{'='*20} Starting {'Optimized' if is_optimized else 'Nominal'} Calculation {'('+method_name+')' if method_name else ''} {'='*20}")
    st.session_state.current_status = f"Status: Running {'Optimized' if is_optimized else 'Nominal'} Calculation..."
    if status_placeholder: status_placeholder.info(st.session_state.current_status)
    progress_bar = None
    if progress_placeholder: progress_bar = progress_placeholder.progress(0)

    res_optim_grid: Optional[Dict[str, np.ndarray]] = None
    final_fig: Optional[plt.Figure] = None
    ep_calculated: Optional[np.ndarray] = None
    mse_display: Optional[float] = np.nan
    final_result_ep: Optional[np.ndarray] = None # Store the ep used for this run

    try:
        inputs = _validate_physical_inputs_from_state(require_optim_params=False)
        active_targets = get_active_targets_from_state()
        if active_targets is None:
             raise ValueError("Failed to get valid active targets.")
        active_targets_for_plot = active_targets

        if progress_bar: progress_bar.progress(10)

        ep_source_for_calc: Optional[np.ndarray] = None
        current_optimized_ep = st.session_state.get('current_optimized_ep')
        optim_ran = st.session_state.get('optimization_ran_since_nominal_change', False)

        if ep_vector_to_use is not None:
            ep_source_for_calc = ep_vector_to_use
        elif current_optimized_ep is not None and optim_ran:
            ep_source_for_calc = current_optimized_ep
        # If neither is available, _prepare_calculation_data_st will use nominal QWOT

        if progress_bar: progress_bar.progress(15)

        nH, nL, nSub_c, l_vec_plot_default, ep_actual_calc, ep_actual_orig = _prepare_calculation_data_st(inputs, ep_vector_to_use=ep_source_for_calc)
        final_result_ep = ep_actual_orig.copy() # Store the vector used (original thicknesses)
        ep_calculated = ep_actual_orig # For potential use later

        l_vec_final_plot = l_vec_override if l_vec_override is not None else l_vec_plot_default

        log_message("  Calculating T(λ) for plotting...")
        if progress_bar: progress_bar.progress(25)
        start_rt_time = time.time()
        res_fine = calculate_RT_from_ep(ep_actual_calc, nH, nL, nSub_c, l_vec_final_plot)
        rt_time = time.time() - start_rt_time
        log_message(f"  T calculation (plot) finished in {rt_time:.3f}s.")
        if progress_bar: progress_bar.progress(50)

        if res_fine is None:
            raise RuntimeError("Failed to calculate spectral data for plotting.")

        if active_targets_for_plot:
            l_min_overall = inputs['l_range_deb']
            l_max_overall = inputs['l_range_fin']
            l_step_gui = inputs['l_step']
            num_points_approx = max(2, int(np.round((l_max_overall - l_min_overall) / l_step_gui)) + 1)
            # Use geomspace for consistency with optimization cost calc
            try:
                l_vec_optim_display = np.geomspace(l_min_overall, l_max_overall, num_points_approx)
                l_vec_optim_display = l_vec_optim_display[(l_vec_optim_display > 0) & np.isfinite(l_vec_optim_display)]
            except Exception:
                 l_vec_optim_display = np.array([]) # Fallback to empty

            if l_vec_optim_display.size > 0:
                 log_message("  Calculating T(λ) on optimization grid for MSE display...")
                 res_optim_grid = calculate_RT_from_ep(ep_actual_calc, nH, nL, nSub_c, l_vec_optim_display)
                 log_message("  T calculation (optim grid) finished.")
                 if progress_bar: progress_bar.progress(75)

                 if res_optim_grid:
                     mse_display, num_pts_mse = calculate_final_mse(res_optim_grid, active_targets_for_plot)
                     if num_pts_mse > 0:
                         log_message(f"  MSE (display, Optim grid) = {mse_display:.3e} over {num_pts_mse} points.")
                     else:
                         log_message("  No points found in target zones (Optim grid) to calculate display MSE.")
                         mse_display = np.nan # Explicitly NaN if no points
                 else:
                     log_message("  Failed to calculate T on optim grid, cannot calculate display MSE.")
                     mse_display = None # Indicate calculation error

            else:
                 log_message("  Optim grid empty or invalid, cannot calculate display MSE.")
                 mse_display = np.nan
        else:
            log_message("  No active spectral targets, display MSE not calculated.")
            mse_display = None

        log_message("  Generating plots...")
        if progress_bar: progress_bar.progress(90)
        final_fig = tracer_graphiques(res_fine, ep_actual_orig,
                                      inputs['nH_r'], inputs['nH_i'], inputs['nL_r'], inputs['nL_i'], inputs['nSub'], # Pass nSub real part
                                      active_targets_for_plot, mse_display,
                                      is_optimized=is_optimized, method_name=method_name,
                                      res_optim_grid=res_optim_grid)
        log_message("  Plot generation finished.")
        if progress_bar: progress_bar.progress(100)

        if is_optimized:
            st.session_state.optimization_ran_since_nominal_change = True
            # Update current_optimized_ep only if the calculation was for an optimized state
            st.session_state.current_optimized_ep = ep_actual_orig.copy() if ep_actual_orig is not None else None

        st.session_state.current_status = f"Status: {'Optimized' if is_optimized else 'Nominal'} Calculation Complete"
        if status_placeholder: status_placeholder.success(st.session_state.current_status)
        log_message(f"--- Finished {'Optimized' if is_optimized else 'Nominal'} Calculation ---")

        # Store results for potential redraw
        st.session_state.last_run_calculation_results = {
             'res': res_fine, 'ep': final_result_ep, # Store the ep used for *this* calculation
             'mse': mse_display, 'res_optim_grid': res_optim_grid, 'is_optimized': is_optimized,
             'method_name': method_name, 'inputs': inputs, 'active_targets': active_targets_for_plot
        }
        update_display_info(final_result_ep) # Update display based on the calculated vector

    except (ValueError, RuntimeError) as e:
        err_msg = f"ERROR (Input/Logic) in calculation: {e}"
        log_message(err_msg)
        if status_placeholder: st.error(err_msg)
        st.session_state.current_status = f"Status: Calculation Failed (Input Error)"
        if status_placeholder: status_placeholder.error(st.session_state.current_status)
        if plot_placeholder: plot_placeholder.empty()
        st.session_state.last_run_calculation_results = {} # Clear last results on error
    except Exception as e:
        err_msg = f"ERROR (Unexpected) in calculation: {type(e).__name__}: {e}"
        tb_msg = traceback.format_exc()
        log_message(err_msg); log_message(tb_msg); print(err_msg); print(tb_msg)
        if status_placeholder: st.error(f"{err_msg}. See log/console for details.")
        st.session_state.current_status = f"Status: Calculation Failed (Unexpected Error)"
        if status_placeholder: status_placeholder.error(st.session_state.current_status)
        if plot_placeholder: plot_placeholder.empty()
        st.session_state.last_run_calculation_results = {} # Clear last results on error
    finally:
        if final_fig and plot_placeholder:
             plot_placeholder.pyplot(final_fig)
             plt.close(final_fig) # Close the figure to free memory
        elif plot_placeholder:
             # Ensure placeholder is cleared if no figure was generated or if error occurred before plot
             # plot_placeholder.empty() # This might clear error messages too early if placed here
             pass
        if progress_placeholder: progress_placeholder.empty() # Clear progress bar


def run_optimization_st() -> None:
    """Handles the multi-pass optimization process triggered by the 'Optimize' button."""
    global status_placeholder, progress_placeholder, plot_placeholder # Allow modification

    st.session_state.optim_start_time = time.time()
    clear_log()
    log_with_elapsed_time(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Starting Multi-Pass Optimization Process ===")

    # Reset optimization state variables
    st.session_state.current_optimized_ep = None
    st.session_state.optimization_ran_since_nominal_change = False
    st.session_state.optimized_qwot_display = ""
    st.session_state.current_status = "Status: Initializing Optimization..."
    if status_placeholder: status_placeholder.info(st.session_state.current_status)
    progress_bar = None
    if progress_placeholder: progress_bar = progress_placeholder.progress(0)

    ep_nominal_glob: Optional[np.ndarray] = None
    overall_best_ep_final: Optional[np.ndarray] = None
    overall_best_cost_final: float = np.inf
    overall_best_layers: int = 0
    initial_nominal_cost: float = np.inf
    initial_nominal_layers: int = 0
    final_successful_result_obj: Optional[OptimizeResult] = None
    ep_ref_for_next_pass: Optional[np.ndarray] = None
    n_passes: int = 0
    optimization_successful: bool = False
    auto_removed_count: int = 0

    try:
        inputs = _validate_physical_inputs_from_state(require_optim_params=True)
        n_samples_base = inputs['n_samples']
        initial_p_best = inputs['p_best']
        n_passes = inputs['n_passes']
        initial_scaling_nm = inputs['scaling_nm']

        # Estimate progress steps (rough)
        # Pass 1: Sobol + Refine(Sobol+Eval) + LocalSearch(Pool+Process) = 3 major steps?
        # Pass 2+: Sobol(Eval) + LocalSearch(Pool+Process) = 2 major steps?
        # Post-processing: Auto-remove loop = 1 major step?
        # progress_max_steps = (3 if n_passes >= 1 else 0) + max(0, (n_passes - 1) * 2) + 1
        # Simpler: Each pass is roughly equal + 1 for post-processing
        progress_max_steps = n_passes + 1
        current_progress_step = 0

        active_targets = get_active_targets_from_state()
        if active_targets is None: raise ValueError("Failed to retrieve/validate spectral targets.")
        if not active_targets: raise ValueError("No active spectral targets defined. Optimization requires at least one target.")
        log_with_elapsed_time(f"{len(active_targets)} active target zone(s) found.")

        # Get initial nominal structure
        nH_complex_nom = complex(inputs['nH_r'], inputs['nH_i'])
        nL_complex_nom = complex(inputs['nL_r'], inputs['nL_i'])
        ep_nominal_glob, _ = get_initial_ep(inputs['emp_str'], inputs['l0'], nH_complex_nom, nL_complex_nom)
        if ep_nominal_glob.size == 0: raise ValueError("Initial nominal QWOT stack is empty or invalid.")
        initial_nominal_layers = len(ep_nominal_glob)
        overall_best_layers = initial_nominal_layers

        # Calculate initial cost
        try:
            nSub_c = complex(inputs['nSub'])
            l_min_overall = inputs['l_range_deb']; l_max_overall = inputs['l_range_fin']; l_step_gui = inputs['l_step']
            num_points_approx = max(2, int(np.round((l_max_overall - l_min_overall) / l_step_gui)) + 1)
            l_vec_optim_init = np.geomspace(l_min_overall, l_max_overall, num_points_approx)
            l_vec_optim_init = l_vec_optim_init[(l_vec_optim_init > 0) & np.isfinite(l_vec_optim_init)]
            if l_vec_optim_init.size == 0: raise ValueError("Failed to create optim vec for initial cost.")

            initial_nominal_cost = calculate_mse_for_optimization_penalized(ep_nominal_glob, nH_complex_nom, nL_complex_nom, nSub_c, l_vec_optim_init, active_targets, MIN_THICKNESS_PHYS_NM)
            overall_best_cost_final = initial_nominal_cost if np.isfinite(initial_nominal_cost) else np.inf
            log_with_elapsed_time(f"Initial nominal cost: {overall_best_cost_final:.3e}, Layers: {initial_nominal_layers}")

            # Set initial best to nominal
            if np.isfinite(overall_best_cost_final):
                 overall_best_ep_final = ep_nominal_glob.copy()
                 ep_ref_for_next_pass = ep_nominal_glob.copy() # Start Pass 2 from nominal if Pass 1 fails badly
            else:
                 # If nominal cost is inf, we need a valid starting point from Pass 1
                 overall_best_ep_final = None
                 ep_ref_for_next_pass = ep_nominal_glob.copy() # Still use nominal shape for Pass 2 bounds

            st.session_state.current_status = f"Status: Initial | Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
            if status_placeholder: status_placeholder.info(st.session_state.current_status)

        except Exception as e_init_cost:
            log_with_elapsed_time(f"Warning: Could not calculate initial nominal cost: {e_init_cost}. Starting optimization with Inf cost.")
            overall_best_cost_final = np.inf
            overall_best_ep_final = None # Cannot rely on nominal if cost failed
            ep_ref_for_next_pass = ep_nominal_glob.copy() # Use nominal shape for Pass 2 bounds
            st.session_state.current_status = f"Status: Initial | Best MSE: N/A | Layers: {initial_nominal_layers}"
            if status_placeholder: status_placeholder.warning(st.session_state.current_status)


        # --- Optimization Passes ---
        for pass_num in range(1, n_passes + 1):
            run_id_str = f"Pass {pass_num}/{n_passes}"
            pass_status_prefix = f"Status: {run_id_str}"
            status_suffix = f"| Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
            st.session_state.current_status = f"{pass_status_prefix} - Starting... {status_suffix}"
            if status_placeholder: status_placeholder.info(st.session_state.current_status)
            if progress_bar: progress_bar.progress( current_progress_step / progress_max_steps )

            # Adjust parameters for the current pass
            p_best_reduction = PASS_P_BEST_REDUCTION_FACTOR ** (pass_num - 1)
            p_best_for_this_pass = max(2, int(np.round(initial_p_best * p_best_reduction)))
            # Adjust N samples? Maybe keep constant or reduce less aggressively.
            # Using n_samples_base (constant) for now.
            n_samples_this_pass = n_samples_base

            current_scaling_pass: Optional[float] = None
            ep_ref_sobol_pass: Optional[np.ndarray] = None
            if pass_num == 1:
                current_scaling_pass = None # Use relative scaling
                ep_ref_sobol_pass = None
            else:
                scale_reduction_factor = PASS_SCALING_REDUCTION_BASE**(pass_num - 2) # Starts at 1 for pass 2
                current_scaling_pass = max(MIN_SCALING_NM, initial_scaling_nm / scale_reduction_factor)
                if ep_ref_for_next_pass is None:
                     # This should only happen if initial nominal cost failed AND pass 1 failed
                     raise RuntimeError(f"Cannot start {run_id_str}: No reference structure available from previous pass.")
                ep_ref_sobol_pass = ep_ref_for_next_pass # Use best from previous pass

            # Run the core optimization pass
            pass_inputs = inputs.copy()
            pass_inputs['n_samples'] = n_samples_this_pass # Use potentially adjusted N samples
            ep_this_pass, cost_this_pass, result_obj_this_pass = run_optimization_process(
                inputs=pass_inputs,
                active_targets=active_targets,
                ep_nominal_for_cost=ep_nominal_glob, # Pass original nominal for relative bounds in pass 1
                p_best_this_pass=p_best_for_this_pass,
                ep_reference_for_sobol=ep_ref_sobol_pass, # Pass previous best for bounds in pass 2+
                current_scaling=current_scaling_pass,
                run_number=pass_num,
                total_passes=n_passes,
                progress_bar_hook=None, # Pass specific hook if needed
                status_placeholder=status_placeholder,
                current_best_mse=overall_best_cost_final,
                current_best_layers=overall_best_layers
            )

            current_progress_step += 1 # Increment progress after each pass
            if progress_bar: progress_bar.progress(current_progress_step / progress_max_steps)

            # Process results of the pass
            new_best_found_this_pass = False
            if ep_this_pass is not None and ep_this_pass.size > 0 and np.isfinite(cost_this_pass):
                if cost_this_pass < overall_best_cost_final:
                    log_with_elapsed_time(f"*** New overall best cost found in {run_id_str}: {cost_this_pass:.3e} (Previous: {overall_best_cost_final:.3e}) ***")
                    overall_best_cost_final = cost_this_pass
                    overall_best_ep_final = ep_this_pass.copy()
                    overall_best_layers = len(overall_best_ep_final)
                    final_successful_result_obj = result_obj_this_pass # Store the result object
                    new_best_found_this_pass = True

                # Update reference for the *next* pass based on the *current best overall*
                ep_ref_for_next_pass = overall_best_ep_final.copy() if overall_best_ep_final is not None else ep_nominal_glob.copy()

                status_suffix = f"| Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
                if new_best_found_this_pass:
                    st.session_state.current_status = f"{pass_status_prefix} - Completed. New Best! {status_suffix}"
                    if status_placeholder: status_placeholder.info(st.session_state.current_status)
                else:
                    st.session_state.current_status = f"{pass_status_prefix} - Completed. No improvement. {status_suffix}"
                    if status_placeholder: status_placeholder.info(st.session_state.current_status)
            else:
                # Pass failed to return a valid solution
                log_with_elapsed_time(f"ERROR: {run_id_str} did not return a valid solution (cost: {cost_this_pass}, ep size: {ep_this_pass.size if ep_this_pass is not None else 'None'}).")
                st.session_state.current_status = f"{pass_status_prefix} - FAILED. {status_suffix}"
                if status_placeholder: status_placeholder.warning(st.session_state.current_status)
                # Keep using the previous best reference if this pass failed
                if ep_ref_for_next_pass is None:
                    # This implies initial nominal cost failed AND pass 1 failed. Abort.
                    raise RuntimeError(f"{run_id_str} failed and no prior successful result exists. Optimization aborted.")
                log_with_elapsed_time("Continuing optimization using previous best result as reference.")


        # --- Post-Processing: Auto-Removal ---
        if progress_bar: progress_bar.progress(current_progress_step / progress_max_steps) # Mark start of post-processing
        st.session_state.current_status = f"Status: Post-processing (Auto-Remove)... | Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
        if status_placeholder: status_placeholder.info(st.session_state.current_status)

        if overall_best_ep_final is None:
             raise RuntimeError("Optimization finished, but no valid result found before post-processing.")

        log_with_elapsed_time(f"\n--- Checking for automatic thin layer removal (< {AUTO_REMOVE_THRESHOLD_NM} nm) ---")
        max_auto_removals = len(overall_best_ep_final) # Max possible removals

        # Need cost function args for re-optimization during removal
        temp_inputs_ar = _validate_physical_inputs_from_state(require_optim_params=False) # Don't need optim params here
        temp_active_targets_ar = get_active_targets_from_state()
        if temp_active_targets_ar is None: raise ValueError("Failed to get targets for auto-removal cost func.")
        temp_nH_ar = complex(temp_inputs_ar['nH_r'], temp_inputs_ar['nH_i'])
        temp_nL_ar = complex(temp_inputs_ar['nL_r'], temp_inputs_ar['nL_i'])
        temp_nSub_c_ar = complex(temp_inputs_ar['nSub'])
        temp_l_min_ar = temp_inputs_ar['l_range_deb']; temp_l_max_ar = temp_inputs_ar['l_range_fin']; temp_l_step_ar = temp_inputs_ar['l_step']
        temp_num_pts_ar = max(2, int(np.round((temp_l_max_ar - temp_l_min_ar) / temp_l_step_ar)) + 1)
        try:
            temp_l_vec_optim_ar = np.geomspace(temp_l_min_ar, temp_l_max_ar, temp_num_pts_ar)
            temp_l_vec_optim_ar = temp_l_vec_optim_ar[(temp_l_vec_optim_ar > 0) & np.isfinite(temp_l_vec_optim_ar)]
            if not temp_l_vec_optim_ar.size: raise ValueError("Geomspace AR vector empty.")
        except Exception as e_geom_ar:
             raise ValueError(f"Failed to generate geomspace AR wavelength vector: {e_geom_ar}")

        temp_args_for_cost_ar: Tuple = (temp_nH_ar, temp_nL_ar, temp_nSub_c_ar, temp_l_vec_optim_ar, temp_active_targets_ar, MIN_THICKNESS_PHYS_NM)

        current_ep_for_removal = overall_best_ep_final.copy()

        while auto_removed_count < max_auto_removals:
            if current_ep_for_removal is None or len(current_ep_for_removal) <= 1:
                log_with_elapsed_time("  Structure too short for further auto-removal.")
                break

            # Find thinnest layer that is >= MIN_THICKNESS and < AUTO_REMOVE_THRESHOLD
            eligible_indices = np.where((current_ep_for_removal >= MIN_THICKNESS_PHYS_NM) &
                                        (current_ep_for_removal < AUTO_REMOVE_THRESHOLD_NM))[0]

            if eligible_indices.size > 0:
                # Find the index corresponding to the minimum value among eligible layers
                thinnest_eligible_value = np.min(current_ep_for_removal[eligible_indices])
                # Find the first occurrence of this value in the original array
                thinnest_below_threshold_idx = np.where(current_ep_for_removal == thinnest_eligible_value)[0][0]
                thinnest_below_threshold_val = thinnest_eligible_value

                log_with_elapsed_time(f"  Auto-removing layer {thinnest_below_threshold_idx + 1} (thickness {thinnest_below_threshold_val:.3f} nm < {AUTO_REMOVE_THRESHOLD_NM} nm)")
                status_suffix = f"| Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
                st.session_state.current_status = f"Status: Auto-removing layer {auto_removed_count + 1}... {status_suffix}"
                if status_placeholder: status_placeholder.info(st.session_state.current_status)

                new_ep, success, cost_after, removal_logs = perform_single_thin_layer_removal(
                    current_ep_for_removal, MIN_THICKNESS_PHYS_NM,
                    calculate_mse_for_optimization_penalized, temp_args_for_cost_ar,
                    log_prefix="    [Auto Removal] ", target_layer_index=thinnest_below_threshold_idx
                )
                for log_line in removal_logs: log_with_elapsed_time(log_line)

                # Check if removal and re-optimization were successful AND changed the structure
                structure_actually_changed = (success and new_ep is not None and len(new_ep) < len(current_ep_for_removal))

                if structure_actually_changed:
                    current_ep_for_removal = new_ep.copy()
                    overall_best_ep_final = new_ep.copy() # Update the main best result
                    overall_best_cost_final = cost_after if np.isfinite(cost_after) else np.inf
                    overall_best_layers = len(overall_best_ep_final)
                    auto_removed_count += 1
                    log_with_elapsed_time(f"  Auto-removal successful. New cost: {overall_best_cost_final:.3e}, Layers: {overall_best_layers}")
                    # Loop continues to check for more layers to remove
                else:
                    log_with_elapsed_time("    Auto-removal/re-optimization failed or structure unchanged this iteration. Stopping auto-removal.")
                    break # Stop the auto-removal loop
            else:
                log_with_elapsed_time(f"  No layers found below {AUTO_REMOVE_THRESHOLD_NM} nm (and >= {MIN_THICKNESS_PHYS_NM} nm).")
                break # No more layers to remove

        if auto_removed_count > 0: log_with_elapsed_time(f"--- Finished automatic removal ({auto_removed_count} layer(s) removed) ---")

        # --- Finalize Optimization ---
        optimization_successful = True
        st.session_state.current_optimized_ep = overall_best_ep_final.copy() if overall_best_ep_final is not None else None
        st.session_state.optimization_ran_since_nominal_change = True

        # Calculate final QWOT for display
        final_qwot_str = "QWOT Error"
        if overall_best_ep_final is not None and len(overall_best_ep_final) > 0:
            try:
                l0_val = inputs['l0']; nH_r_val = inputs['nH_r']; nL_r_val = inputs['nL_r']
                optimized_qwots = calculate_qwot_from_ep(overall_best_ep_final, l0_val, nH_r_val, nL_r_val)
                if np.any(np.isnan(optimized_qwots)):
                    final_qwot_str = "QWOT N/A (NaN)"
                else:
                     # Use more precision for QWOT output? 3 might be too low.
                     final_qwot_str = ", ".join([f"{q:.4f}" for q in optimized_qwots])
            except Exception as qwot_calc_error:
                log_with_elapsed_time(f"Error calculating final QWOTs: {qwot_calc_error}")
        else: final_qwot_str = "N/A (Empty Structure)"
        st.session_state.optimized_qwot_display = final_qwot_str

        # --- Log Final Summary ---
        final_method_name = f"{n_passes}-Pass Opt" + (f" + {auto_removed_count} AutoRm" if auto_removed_count > 0 else "")
        log_with_elapsed_time("\n" + "="*60)
        log_with_elapsed_time(f"--- Overall Optimization ({final_method_name}) Finished ---")
        log_with_elapsed_time(f"Best Final Cost Found (MSE): {overall_best_cost_final:.3e}")
        log_with_elapsed_time(f"Final number of layers: {overall_best_layers}")
        log_with_elapsed_time(f"Final Optimized QWOT (λ₀={inputs['l0']}nm): {final_qwot_str}")
        if overall_best_ep_final is not None:
            ep_str_list = [f"L{i+1}:{th:.4f}" for i, th in enumerate(overall_best_ep_final)]
            log_with_elapsed_time(f"Final optimized thicknesses ({overall_best_layers} layers, nm): [{', '.join(ep_str_list)}]")
        if final_successful_result_obj and isinstance(final_successful_result_obj, OptimizeResult):
             best_res_nit = getattr(final_successful_result_obj, 'nit', 'N/A')
             best_res_msg_raw = getattr(final_successful_result_obj, 'message', 'N/A')
             best_res_msg = best_res_msg_raw.decode('utf-8', errors='ignore') if isinstance(best_res_msg_raw, bytes) else str(best_res_msg_raw)
             log_with_elapsed_time(f"Best Result Info: Iters={best_res_nit}, Msg='{best_res_msg}'")
        log_with_elapsed_time("="*60 + "\n")

        if progress_bar: progress_bar.progress(1.0) # Ensure progress reaches 100%
        final_status_text = f"Status: Opt Complete{' (+AutoRm)' if auto_removed_count>0 else ''} | MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
        st.session_state.current_status = final_status_text
        if status_placeholder: status_placeholder.success(st.session_state.current_status)

        # Run final calculation and display results
        run_calculation_st(ep_vector_to_use=overall_best_ep_final, is_optimized=True, method_name=final_method_name)

    except (ValueError, RuntimeError) as e:
        err_msg = f"ERROR (Input/Logic) during optimization: {e}"
        log_with_elapsed_time(err_msg)
        if status_placeholder: st.error(err_msg)
        st.session_state.current_status = f"Status: Optimization Failed (Input/Logic Error)"
        if status_placeholder: status_placeholder.error(st.session_state.current_status)
        optimization_successful = False
        st.session_state.current_optimized_ep = None
        st.session_state.optimization_ran_since_nominal_change = False
        st.session_state.optimized_qwot_display = "Error Input/Optim Logic"
    except Exception as e:
        err_msg = f"ERROR (Unexpected) during optimization: {type(e).__name__}: {e}"
        tb_msg = traceback.format_exc()
        log_with_elapsed_time(err_msg); log_with_elapsed_time(tb_msg); print(err_msg); print(tb_msg)
        if status_placeholder: st.error(f"{err_msg}. See log/console for details.")
        st.session_state.current_status = f"Status: Optimization Failed (Unexpected Error)"
        if status_placeholder: status_placeholder.error(st.session_state.current_status)
        optimization_successful = False
        st.session_state.current_optimized_ep = None
        st.session_state.optimization_ran_since_nominal_change = False
        st.session_state.optimized_qwot_display = "Optim Error Unexpected"
    finally:
        if progress_placeholder: progress_placeholder.empty()
        st.session_state.optim_start_time = None # Clear start time
        # Update display based on final state
        update_display_info(st.session_state.current_optimized_ep if optimization_successful else None)
        # Rerun needed to reflect QWOT display update and button states
        st.rerun()


def run_remove_layer_st() -> None:
    """Handles the 'Remove Thinnest Layer' button action."""
    global status_placeholder, progress_placeholder, plot_placeholder # Allow modification

    log_message("\n" + "-"*10 + " Attempting Manual Thin Layer Removal " + "-"*10)
    st.session_state.current_status = "Status: Removing thinnest layer..."
    if status_placeholder: status_placeholder.info(st.session_state.current_status)
    progress_bar = None
    if progress_placeholder: progress_bar = progress_placeholder.progress(0)

    current_ep = st.session_state.get('current_optimized_ep')
    optim_ran = st.session_state.get('optimization_ran_since_nominal_change', False)
    removal_successful: bool = False
    final_ep_after_removal: Optional[np.ndarray] = None

    if current_ep is None or not optim_ran or len(current_ep) <= 1:
        log_message("ERROR: No valid optimized structure with >= 2 layers available for removal.")
        if status_placeholder: st.error("No valid optimized structure (must have >= 2 layers) found to modify.")
        st.session_state.current_status = "Status: Removal Failed (No Structure/Not Optimized)"
        if status_placeholder: status_placeholder.error(st.session_state.current_status)
    else:
        final_ep_after_removal = current_ep.copy() # Keep track of the current state
        try:
            if progress_bar: progress_bar.progress(10)
            inputs = _validate_physical_inputs_from_state(require_optim_params=False)
            active_targets = get_active_targets_from_state()
            if active_targets is None: raise ValueError("Failed to retrieve/validate spectral targets for re-optimization.")
            if not active_targets: raise ValueError("No active spectral targets defined. Cannot re-optimize after removal.")

            if progress_bar: progress_bar.progress(20)
            nH = complex(inputs['nH_r'], inputs['nH_i']); nL = complex(inputs['nL_r'], inputs['nL_i']); nSub_c = complex(inputs['nSub'])
            l_min = inputs['l_range_deb']; l_max = inputs['l_range_fin']; l_step = inputs['l_step']
            num_pts = max(2, int(np.round((l_max - l_min) / l_step)) + 1)
            try:
                 l_vec_optim = np.geomspace(l_min, l_max, num_pts)
                 l_vec_optim = l_vec_optim[(l_vec_optim > 0) & np.isfinite(l_vec_optim)]
                 if not l_vec_optim.size: raise ValueError("Geomspace remove vector empty.")
            except Exception as e_geom_rem:
                 raise ValueError(f"Failed to generate optim vector for removal: {e_geom_rem}")

            args_for_cost_tuple: Tuple = (nH, nL, nSub_c, l_vec_optim, active_targets, MIN_THICKNESS_PHYS_NM)
            if progress_bar: progress_bar.progress(30)

            log_message("  Starting removal and re-optimization...")
            new_ep, success, final_cost, removal_logs = perform_single_thin_layer_removal(
                current_ep, MIN_THICKNESS_PHYS_NM, calculate_mse_for_optimization_penalized,
                args_for_cost_tuple, log_prefix="  [Manual Removal] "
            )
            for log_line in removal_logs: log_message(log_line)
            if progress_bar: progress_bar.progress(70)

            structure_actually_changed = (success and new_ep is not None and len(new_ep) < len(current_ep))

            if structure_actually_changed:
                log_message("  Layer removal successful. Updating structure and display.")
                st.session_state.current_optimized_ep = new_ep.copy()
                final_ep_after_removal = new_ep.copy() # Store the successful result
                removal_successful = True

                # Update QWOT display
                final_qwot_str = "QWOT Error"
                try:
                    l0_val = inputs['l0']; nH_r_val = inputs['nH_r']; nL_r_val = inputs['nL_r']
                    optimized_qwots = calculate_qwot_from_ep(new_ep, l0_val, nH_r_val, nL_r_val)
                    if np.any(np.isnan(optimized_qwots)):
                         final_qwot_str = "QWOT N/A (NaN)"
                    else: final_qwot_str = ", ".join([f"{q:.4f}" for q in optimized_qwots])
                except Exception as qwot_calc_error: log_message(f"Error calculating QWOTs after removal: {qwot_calc_error}")
                st.session_state.optimized_qwot_display = final_qwot_str

                log_message(f"  Cost after removal/re-opt: {final_cost:.3e}")
                layers_after = len(new_ep)
                st.session_state.current_status = f"Status: Layer Removed | MSE: {final_cost:.3e} | Layers: {layers_after}"
                if status_placeholder: status_placeholder.success(st.session_state.current_status)

                # Redraw plot with the new structure
                run_calculation_st(ep_vector_to_use=new_ep, is_optimized=True, method_name="Optimized (Post-Removal)")
                if progress_bar: progress_bar.progress(100)

            else:
                log_message("  No layer removed or structure unchanged.")
                if status_placeholder: st.info("Could not find suitable layer or removal/re-optimization failed/unchanged.")
                current_ep_len = len(current_ep) if current_ep is not None else 0
                st.session_state.current_status = f"Status: Removal Skipped/Failed | Layers: {current_ep_len}"
                if status_placeholder: status_placeholder.warning(st.session_state.current_status)

        except (ValueError, RuntimeError) as e:
            err_msg = f"ERROR (Input/Logic) during layer removal: {e}"
            log_message(err_msg)
            if status_placeholder: st.error(err_msg)
            st.session_state.current_status = "Status: Removal Failed (Input/Logic Error)"
            if status_placeholder: status_placeholder.error(st.session_state.current_status)
        except Exception as e:
            err_msg = f"ERROR (Unexpected) in layer removal: {type(e).__name__}: {e}"
            tb_msg = traceback.format_exc()
            log_message(err_msg); log_message(tb_msg); print(err_msg); print(tb_msg)
            if status_placeholder: st.error(f"{err_msg}. See log/console for details.")
            st.session_state.current_status = "Status: Removal Failed (Unexpected Error)"
            if status_placeholder: status_placeholder.error(st.session_state.current_status)
        finally:
            if progress_placeholder: progress_placeholder.empty()
            # Update display based on whether removal was successful
            update_display_info(final_ep_after_removal)
            st.rerun()


def run_set_nominal_st() -> None:
    """Handles the 'Set Current as Nominal' button action."""
    global status_placeholder, plot_placeholder # Allow modification

    log_message("\n--- Setting Current Optimized Design as Nominal ---")
    st.session_state.current_status = "Status: Setting current as Nominal..."
    if status_placeholder: status_placeholder.info(st.session_state.current_status)

    current_ep = st.session_state.get('current_optimized_ep')
    optim_ran = st.session_state.get('optimization_ran_since_nominal_change', False)

    if current_ep is None or not optim_ran or len(current_ep) == 0:
        log_message("ERROR: No optimized design available to set as nominal.")
        if status_placeholder: st.error("No valid optimized design is currently loaded.")
        st.session_state.current_status = "Status: Set Nominal Failed (No Design)"
        if status_placeholder: status_placeholder.error(st.session_state.current_status)
    else:
        try:
            inputs = _validate_physical_inputs_from_state(require_optim_params=False)
            l0_val = inputs['l0']; nH_r_val = inputs['nH_r']; nL_r_val = inputs['nL_r']

            optimized_qwots = calculate_qwot_from_ep(current_ep, l0_val, nH_r_val, nL_r_val)

            if np.any(np.isnan(optimized_qwots)):
                final_qwot_str = "" # Don't update QWOT field if calculation fails
                log_message("Warning: QWOT calculation resulted in NaN. Cannot set nominal QWOT string.")
                if status_placeholder: st.warning("Could not calculate valid QWOT multipliers (resulted in NaN). Nominal QWOT field not updated.")
            else:
                # Use sufficient precision for the QWOT string representation
                final_qwot_str = ", ".join([f"{q:.6f}" for q in optimized_qwots])
                # Update the *input* session state variable for the QWOT text area
                st.session_state.emp_str_input = final_qwot_str
                log_message(f"Nominal QWOT string updated in state: {final_qwot_str}")
                if status_placeholder: st.success("Current optimized design set as new Nominal Structure (QWOT). Optimized state cleared.")

            # Reset optimization state
            st.session_state.current_optimized_ep = None
            st.session_state.optimization_ran_since_nominal_change = False
            st.session_state.optimized_qwot_display = "" # Clear optimized display field
            st.session_state.current_status = "Status: Idle (New Nominal Set)"
            if status_placeholder: status_placeholder.info(st.session_state.current_status)
            st.session_state.last_run_calculation_results = {} # Clear last calculation result

        except Exception as e:
            err_msg = f"ERROR during 'Set Nominal' operation: {type(e).__name__}: {e}"
            log_message(err_msg)
            if status_placeholder: st.error(f"An error occurred setting nominal: {e}")
            st.session_state.current_status = "Status: Set Nominal Failed (Error)"
            if status_placeholder: status_placeholder.error(st.session_state.current_status)
        finally:
            # Rerun to update the QWOT input field and clear optimization state displays
            update_display_info(None) # Update display based on cleared state
            st.rerun()


def main() -> None:
    """Sets up the Streamlit UI and handles user interactions."""
    global status_placeholder, progress_placeholder, plot_placeholder # Make placeholders accessible

    # --- Initialize Session State ---
    # Ensure all session state keys used throughout the app are initialized here
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = [f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Log Initialized."]
    if 'current_optimized_ep' not in st.session_state:
        st.session_state.current_optimized_ep = None
    if 'optimization_ran_since_nominal_change' not in st.session_state:
        st.session_state.optimization_ran_since_nominal_change = False
    if 'optim_start_time' not in st.session_state:
        st.session_state.optim_start_time = None
    if 'last_run_calculation_results' not in st.session_state:
        st.session_state.last_run_calculation_results = {} # Use empty dict
    if 'optim_mode' not in st.session_state:
        st.session_state.optim_mode = DEFAULT_MODE
    if 'current_status' not in st.session_state:
        st.session_state.current_status = "Status: Idle"
    if 'current_progress' not in st.session_state: # Not currently used?
        st.session_state.current_progress = 0
    if 'num_layers_display' not in st.session_state:
        st.session_state.num_layers_display = "Layers (Nominal): ?"
    if 'thinnest_layer_display' not in st.session_state:
        st.session_state.thinnest_layer_display = "- nm"
    if 'optimized_qwot_display' not in st.session_state:
        st.session_state.optimized_qwot_display = ""
    if 'targets' not in st.session_state:
        st.session_state.targets = copy.deepcopy(DEFAULT_TARGETS)
    # Ensure input keys are initialized if not handled by widget defaults
    # (Streamlit usually handles this for widgets with values)
    if 'emp_str_input' not in st.session_state:
         st.session_state.emp_str_input = DEFAULT_QWOT


    # --- Page Config ---
    st.set_page_config(layout="wide", page_title="Thin Film Optimizer")
    st.title("Thin Film Stack Optimizer v2.25-LinTarget-GeomSpace (Streamlit)")

    # --- UI Definition ---
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Materials and Substrate", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("Material H (n real)", min_value=0.001, value=2.35, step=0.01, format="%.3f", key='nH_r', help="Real part of high refractive index (must be > 0)")
                st.number_input("Material L (n real)", min_value=0.001, value=1.46, step=0.01, format="%.3f", key='nL_r', help="Real part of low refractive index (must be > 0)")
                st.number_input("Substrate (n real)", min_value=0.001, value=1.52, step=0.01, format="%.3f", key='nSub', help="Real part of substrate refractive index (must be > 0)")
            with c2:
                st.number_input("Material H (k imag)", min_value=0.0, value=0.0, step=0.001, format="%.4f", key='nH_i', help="Imaginary part (k) of high index (>= 0)")
                st.number_input("Material L (k imag)", min_value=0.0, value=0.0, step=0.001, format="%.4f", key='nL_i', help="Imaginary part (k) of low index (>= 0)")
                st.caption("(n = n' + ik', k>=0)") # Clarify k represents absorption

        with st.expander("Stack (Nominal Definition)", expanded=True):
            st.text_area("Nominal Structure (QWOT Multipliers, comma-separated)", value=DEFAULT_QWOT, key='emp_str_input', help="Define the starting structure using Quarter-Wave Optical Thickness multipliers relative to λ₀.")
            st.number_input("Centering λ₀ (QWOT, nm)", min_value=0.1, value=500.0, step=1.0, format="%.1f", key='l0_input', help="Reference wavelength for QWOT calculation (must be > 0).")
            # Use st.session_state directly for the read-only display
            st.text_input("Optimized QWOT (Read-only)", value=st.session_state.optimized_qwot_display, disabled=True, key='opt_qwot_ro', help="QWOT multipliers of the last successfully optimized structure.")

    with col2:
        with st.expander("Calculation & Optimization Parameters", expanded=True):
            # Use st.session_state.optim_mode to set the index correctly
            optim_mode_options = list(OPTIMIZATION_MODES.keys())
            current_mode_index = optim_mode_options.index(st.session_state.optim_mode) if st.session_state.optim_mode in optim_mode_options else 0
            st.radio(
                "Optimization Mode Preset",
                options=optim_mode_options,
                index=current_mode_index,
                key='optim_mode_radio', # Use a distinct key for the widget
                horizontal=True,
                help="Select preset optimization parameters. Changes update fields below."
            )
            # Update session state based on radio button AND update derived params if mode changed
            if st.session_state.optim_mode != st.session_state.optim_mode_radio:
                 st.session_state.optim_mode = st.session_state.optim_mode_radio
                 selected_params = OPTIMIZATION_MODES[st.session_state.optim_mode]
                 # Update the input fields based on the newly selected mode
                 st.session_state.n_samples_input = selected_params['n_samples']
                 st.session_state.p_best_input = selected_params['p_best']
                 st.session_state.n_passes_input = selected_params['n_passes']
                 # Rerun to reflect changes in the text inputs below
                 st.rerun()

            selected_params = OPTIMIZATION_MODES[st.session_state.optim_mode]

            st.markdown("---") # Separator

            c1, c2, c3 = st.columns(3)
            with c1:
                st.number_input("λ Start (nm)", min_value=0.1, value=400.0, step=1.0, format="%.1f", key='l_range_deb_input', help="Start wavelength for calculations and plots (must be > 0).")
            with c2:
                st.number_input("λ End (nm)", min_value=0.1, value=700.0, step=1.0, format="%.1f", key='l_range_fin_input', help="End wavelength (must be >= λ Start).")
            with c3:
                st.number_input("λ Step (nm, Optim Grid)", min_value=0.01, value=10.0, step=0.1, format="%.2f", key='l_step_input', help="Wavelength step for optimization cost calculation grid (must be > 0).")
                st.caption(f"Plot uses λ Step / {PLOT_LAMBDA_STEP_DIVISOR}") # Use constant

            c1, c2, c3 = st.columns(3)
            with c1:
                st.text_input("N Samples (Sobol)", value=selected_params['n_samples'], key='n_samples_input', help="Number of initial Sobol samples per pass (integer >= 1).")
            with c2:
                st.text_input("P Starts (L-BFGS-B)", value=selected_params['p_best'], key='p_best_input', help="Number of best Sobol points to start local search (integer >= 1, <= N Samples).")
            with c3:
                st.text_input("Optim Passes", value=selected_params['n_passes'], key='n_passes_input', help="Number of optimization passes (integer >= 1).")

            st.number_input("Scaling (nm, Pass 2+)", min_value=0.0, value=10.0, step=0.1, format="%.2f", key='scaling_nm_input', help="Absolute Sobol search range (+/- nm) around previous best for passes > 1 (>= 0).")

        with st.expander("Spectral Target (Optimization on Transmittance T)", expanded=True):
            st.caption("Define target transmittance ramps (T vs λ). Optimization minimizes Mean Squared Error (MSE) against active targets using the 'Optim Grid' step.")

            # Header row for targets
            headers = st.columns([0.5, 0.5, 1, 1, 1, 1])
            headers[0].markdown("**Enable**")
            headers[1].markdown("**Zone**")
            headers[2].markdown("**λ min (nm)**")
            headers[3].markdown("**λ max (nm)**")
            headers[4].markdown("**T @ λmin**")
            headers[5].markdown("**T @ λmax**")

            # Display target input rows
            active_target_list = [] # To check if any are enabled
            for i in range(len(st.session_state.targets)):
                cols = st.columns([0.5, 0.5, 1, 1, 1, 1])
                target_state = st.session_state.targets[i]

                # Use unique keys for each widget within the loop
                enabled = cols[0].checkbox("", value=target_state['enabled'], key=f"target_{i}_enabled")
                cols[1].markdown(f"**{i+1}**") # Display zone number
                l_min_str = cols[2].text_input("", value=str(target_state['min']), key=f"target_{i}_min", label_visibility="collapsed")
                l_max_str = cols[3].text_input("", value=str(target_state['max']), key=f"target_{i}_max", label_visibility="collapsed")
                t_min_str = cols[4].text_input("", value=str(target_state['target_min']), key=f"target_{i}_tmin", label_visibility="collapsed")
                t_max_str = cols[5].text_input("", value=str(target_state['target_max']), key=f"target_{i}_tmax", label_visibility="collapsed")

                # Update the session state for this target directly (triggers rerun on change)
                st.session_state.targets[i]['enabled'] = enabled
                st.session_state.targets[i]['min'] = l_min_str
                st.session_state.targets[i]['max'] = l_max_str
                st.session_state.targets[i]['target_min'] = t_min_str
                st.session_state.targets[i]['target_max'] = t_max_str

                if enabled:
                    active_target_list.append(i)

            # Add button to add/remove targets if needed (more complex UI)

    st.markdown("---")

    # --- Placeholders for dynamic content ---
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    plot_placeholder = st.empty() # Placeholder for the plots

    # --- Action Buttons ---
    action_cols = st.columns([1, 1, 1.5, 1.5, 1]) # Adjust column widths as needed

    calc_button_pressed = action_cols[0].button("Calculate Nominal", key="calc_nom", use_container_width=True, help="Calculate and plot the spectrum for the nominal QWOT structure defined above.")
    opt_button_pressed = action_cols[1].button("Optimize N Passes", key="opt", use_container_width=True, help="Run the multi-pass optimization process based on the current parameters and active spectral targets.")

    # Determine if remove/set buttons should be enabled
    can_modify_optimized = (st.session_state.current_optimized_ep is not None and
                            len(st.session_state.current_optimized_ep) > 0 and
                            st.session_state.optimization_ran_since_nominal_change)
    can_remove_layer = can_modify_optimized and len(st.session_state.current_optimized_ep) > 1

    with action_cols[2]:
        remove_button_pressed = st.button("Remove Thinnest Layer", key="remove_thin", disabled=not can_remove_layer, use_container_width=True, help=f"Removes the thinnest layer (≥{MIN_THICKNESS_PHYS_NM}nm) from the current optimized structure and re-optimizes locally (requires ≥ 2 layers).")
        # Display thinnest layer info using session state variable
        st.info(f"Thinnest ≥ {MIN_THICKNESS_PHYS_NM}nm: {st.session_state.thinnest_layer_display}")

    with action_cols[3]:
        set_nominal_button_pressed = st.button("Set Current as Nominal", key="set_nom", disabled=not can_modify_optimized, use_container_width=True, help="Copies the current optimized structure's QWOT multipliers to the 'Nominal Structure' input field.")
        # Display layer count using session state variable
        st.info(st.session_state.num_layers_display)

    # Log display area
    log_expander = st.expander("Log Messages")
    with log_expander:
        clear_log_pressed = st.button("Clear Log", key="clear_log_btn")
        log_content = "\n".join(st.session_state.log_messages)
        st.text_area("Log", value=log_content, height=300, key="log_display_area", disabled=True)

    st.markdown("---")
    st.subheader("Results")
    # plot_placeholder defined above

    # --- Handle Button Actions ---
    if calc_button_pressed:
        # Reset optimization state when calculating nominal
        st.session_state.current_optimized_ep = None
        st.session_state.optimization_ran_since_nominal_change = False
        st.session_state.optimized_qwot_display = ""
        # Run calculation using nominal definition (ep_vector_to_use=None)
        run_calculation_st(ep_vector_to_use=None, is_optimized=False)
        st.rerun() # Rerun to update plot and button states

    elif opt_button_pressed:
        run_optimization_st() # This function now handles its own rerun at the end

    elif remove_button_pressed:
        run_remove_layer_st() # This function handles its own rerun

    elif set_nominal_button_pressed:
        run_set_nominal_st() # This function handles its own rerun

    elif clear_log_pressed:
        clear_log()
        st.rerun() # Rerun to show cleared log

    else:
        # --- No button pressed: Redraw last plot if available ---
        last_res_data = st.session_state.get('last_run_calculation_results')
        if last_res_data and isinstance(last_res_data, dict) and last_res_data.get('res') is not None and last_res_data.get('ep') is not None:
            try:
                # Ensure necessary keys exist before trying to plot
                required_keys = ['res', 'ep', 'inputs', 'active_targets', 'mse', 'is_optimized', 'method_name', 'res_optim_grid']
                if all(key in last_res_data for key in required_keys):
                    inputs = last_res_data['inputs']
                    active_targets = last_res_data['active_targets']
                    fig = tracer_graphiques(
                        res=last_res_data['res'], ep_actual=last_res_data['ep'],
                        nH_r=inputs['nH_r'], nH_i=inputs['nH_i'], nL_r=inputs['nL_r'], nL_i=inputs['nL_i'], nSub=inputs['nSub'],
                        active_targets_for_plot=active_targets, mse=last_res_data['mse'],
                        is_optimized=last_res_data['is_optimized'], method_name=last_res_data['method_name'],
                        res_optim_grid=last_res_data['res_optim_grid']
                    )
                    plot_placeholder.pyplot(fig)
                    plt.close(fig)
                else:
                    # This case indicates the stored result is incomplete
                    # plot_placeholder.info("Previous result data is incomplete. Calculate or Optimize again.")
                    pass # Avoid showing message on every rerun
            except Exception as e_plot:
                # Avoid crashing the app if redraw fails
                st.warning(f"Could not redraw previous plot: {e_plot}")
                plot_placeholder.empty()
        else:
             # Initial state or after clearing results
             plot_placeholder.info("Click 'Calculate Nominal' or 'Optimize N Passes' to generate results.")

        # --- Update display info and status on every rerun (if no action taken) ---
        current_ep_display: Optional[np.ndarray] = None
        if st.session_state.get('optimization_ran_since_nominal_change') and st.session_state.get('current_optimized_ep') is not None:
            current_ep_display = st.session_state.current_optimized_ep
        elif st.session_state.get('last_run_calculation_results', {}).get('ep') is not None:
             # Display info based on the last plotted result if not optimized
             current_ep_display = st.session_state.last_run_calculation_results['ep']
        update_display_info(current_ep_display)

        # Update status display from session state
        current_status_message = st.session_state.get("current_status", "Status: Idle")
        # Use the placeholder to display status correctly (success, error, info)
        if "Failed" in current_status_message or "Error" in current_status_message:
            status_placeholder.error(current_status_message)
        elif "Complete" in current_status_message or "Removed" in current_status_message or "Set" in current_status_message or "Successful" in current_status_message :
            status_placeholder.success(current_status_message)
        elif "Idle" not in current_status_message: # Show info for running/processing states
             status_placeholder.info(current_status_message)
        # else: # Idle state - placeholder remains empty or shows nothing


# --- Main Execution Guard ---
if __name__ == "__main__":
    # Setup for Multiprocessing (especially on Windows, also good practice)
    # This needs to be called outside the main function logic
    multiprocessing.freeze_support()

    # Run the Streamlit App defined in the main function
    main()
