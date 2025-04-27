# streamlit_app.py
import streamlit as st
import numpy as np
import matplotlib
# matplotlib.use('TkAgg') # No longer needed for Streamlit
import matplotlib.pyplot as plt
# import tkinter as tk # Removed Tkinter
# from tkinter import ttk, Toplevel, scrolledtext, messagebox # Removed Tkinter
# try:
#     import mplcursors # Removed mplcursors - Streamlit handles plots differently
# except ImportError:
#     mplcursors = None
import os
import datetime
import traceback
from scipy.optimize import minimize, OptimizeResult
from scipy.stats import qmc
import numba
from numba import prange
import time
import multiprocessing
from functools import partial
import sys
import io
import math
import copy # For deep copying session state items if needed

# --- Configuration ---
OPTIMIZATION_MODES = {
    "Slow": {"n_samples": "32768", "p_best": "50", "n_passes": "5"},
    "Medium": {"n_samples": "8192", "p_best": "15", "n_passes": "3"},
    "Fast": {"n_samples": "4096", "p_best": "10", "n_passes": "2"}
}
DEFAULT_MODE = "Fast"
MIN_THICKNESS_PHYS_NM = 0.01

# --- Helper Functions ---

def upper_power_of_2(n):
    if n <= 0:
        return 1
    exponent = math.floor(math.log2(n))+1
    return 1 << exponent

# --- Logging ---
# Initialize log in session state if it doesn't exist
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

def log_message(message):
    """Appends a message to the log in session state."""
    full_message = str(message)
    st.session_state.log_messages.append(full_message)
    # Optionally print to console during development
    # print(full_message)

def log_with_elapsed_time(message):
    """Appends a message with elapsed time since optimization start to the log."""
    prefix = ""
    if 'optim_start_time' in st.session_state and st.session_state.optim_start_time is not None:
         elapsed = time.time() - st.session_state.optim_start_time
         prefix = f"[{elapsed:8.2f}s] "

    full_message = prefix + str(message)
    st.session_state.log_messages.append(full_message)
    # Optionally print to console
    # print(full_message)

def clear_log():
    """Clears the log messages in session state."""
    st.session_state.log_messages = [f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Log cleared."]

# --- Core Calculation Logic (Mostly Unchanged) ---

@numba.njit(fastmath=True, cache=True)
def compute_stack_matrix(ep_vector, l_val, nH_complex, nL_complex):
    M = np.eye(2, dtype=np.complex128)
    for i in range(len(ep_vector)):
        thickness = ep_vector[i]
        if thickness <= 1e-12: continue
        Ni = nH_complex if i % 2 == 0 else nL_complex
        eta = Ni
        phi = (2 * np.pi / l_val) * (Ni * thickness)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        M_layer_00 = cos_phi
        M_layer_01 = (1j / eta) * sin_phi
        M_layer_10 = 1j * eta * sin_phi
        M_layer_11 = cos_phi
        M_layer = np.array([[M_layer_00, M_layer_01], [M_layer_10, M_layer_11]], dtype=np.complex128)
        M = M_layer @ M
    return M

@numba.njit(parallel=True, fastmath=True, cache=True)
def calculate_RT_from_ep_core(ep_vector, nH_complex, nL_complex, nSub_complex, l_vec):
    if not l_vec.size: return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    Rs_arr = np.zeros(len(l_vec), dtype=np.float64)
    Ts_arr = np.zeros(len(l_vec), dtype=np.float64)
    etainc = 1.0 + 0j
    etasub = nSub_complex
    for i_l in prange(len(l_vec)):
        l_val = l_vec[i_l]
        if l_val <= 0:
            Rs_arr[i_l], Ts_arr[i_l] = np.nan, np.nan
            continue
        # Ensure ep_vector is contiguous within the parallel loop if needed, although it should be fine if passed correctly
        ep_vector_contig = np.ascontiguousarray(ep_vector)
        M = compute_stack_matrix(ep_vector_contig, l_val, nH_complex, nL_complex)
        m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        rs_num = (etainc * m00 - etasub * m11 + etainc * etasub * m01 - m10)
        rs_den = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        if np.abs(rs_den) < 1e-12:
            Rs_arr[i_l], Ts_arr[i_l] = np.nan, np.nan
        else:
            rs = rs_num / rs_den
            ts = (2 * etainc) / rs_den
            Rs_arr[i_l] = np.abs(rs)**2
            real_etasub = np.real(etasub)
            real_etainc = np.real(etainc)
            if real_etainc == 0: Ts_arr[i_l] = np.nan
            else: Ts_arr[i_l] = (real_etasub / real_etainc) * np.abs(ts)**2
    # Post-process NaNs outside parallel region
    for i in range(len(Rs_arr)):
        if np.isnan(Rs_arr[i]): Rs_arr[i] = 0.0
        if np.isnan(Ts_arr[i]): Ts_arr[i] = 0.0
    return Rs_arr, Ts_arr

@numba.njit(fastmath=True, cache=True)
def get_target_points_indices(l_vec, target_min, target_max):
    if not l_vec.size: return np.empty(0, dtype=np.int64)
    # Add a small epsilon for float comparisons if needed, though usually fine
    # epsilon = 1e-9
    # return np.where((l_vec >= target_min - epsilon) & (l_vec <= target_max + epsilon))[0]
    return np.where((l_vec >= target_min) & (l_vec <= target_max))[0]

def calculate_RT_from_ep(ep_vector, nH, nL, nSub, l_vec):
    # Ensure inputs are complex numbers
    nH_complex = nH + 0j if isinstance(nH, (int, float)) else nH
    nL_complex = nL + 0j if isinstance(nL, (int, float)) else nL
    nSub_complex = nSub + 0j if isinstance(nSub, (int, float)) else nSub

    # Ensure arrays are contiguous and have the right dtype
    ep_vector_np = np.ascontiguousarray(ep_vector, dtype=np.float64)
    l_vec_np = np.ascontiguousarray(l_vec, dtype=np.float64)

    Rs, Ts = calculate_RT_from_ep_core(ep_vector_np, nH_complex, nL_complex, nSub_complex, l_vec_np)
    return {'l': l_vec, 'Rs': Rs, 'Ts': Ts}


def calculate_initial_ep(emp, l0, nH_real, nL_real):
    num_layers = len(emp)
    ep_initial = np.zeros(num_layers, dtype=np.float64)
    for i in range(num_layers):
        multiplier = emp[i]
        n_real = nH_real if i % 2 == 0 else nL_real
        if n_real <= 1e-9:
            # Handle potential division by zero or near-zero
            ep_initial[i] = 0.0 # Or raise an error, depending on desired behavior
        else:
            if l0 <= 0: raise ValueError("Centering wavelength l0 must be positive for QWOT calculation")
            ep_initial[i] = multiplier * l0 / (4 * n_real)
    return ep_initial

def get_initial_ep(emp_str, l0, nH, nL):
    try:
        # Strip whitespace from each element AND filter empty strings robustly
        emp = [float(e.strip()) for e in emp_str.split(',') if e.strip()]
    except ValueError:
        raise ValueError("Invalid QWOT format. Use numbers separated by commas (e.g., 1, 0.5, 1).")
    if any(e < 0 for e in emp): raise ValueError("QWOT multipliers cannot be negative.")
    if not emp: return np.array([], dtype=np.float64), [] # Handle empty QWOT string

    # Use real parts for calculation
    nH_r = np.real(nH)
    nL_r = np.real(nL)
    if nH_r <= 0 or nL_r <=0:
        raise ValueError(f"Real parts of nH ({nH_r}) and nL ({nL_r}) must be > 0.")
    if l0 <= 0:
        raise ValueError(f"Centering wavelength l0 ({l0}) must be > 0.")

    ep_initial = calculate_initial_ep(tuple(emp), l0, nH_r, nL_r)

    # Check for NaN/inf after calculation (robustness)
    if not np.all(np.isfinite(ep_initial)):
        log_message("WARNING: Initial QWOT calculation produced NaN/inf. Replaced with 0.")
        ep_initial = np.nan_to_num(ep_initial, nan=0.0, posinf=0.0, neginf=0.0)

    return ep_initial, emp

def calculate_qwot_from_ep(ep_vector, l0, nH_r, nL_r):
    num_layers = len(ep_vector)
    qwot_multipliers = np.zeros(num_layers, dtype=np.float64)

    if l0 <= 0:
        log_message("Warning: Cannot calculate QWOT, l0 must be positive.")
        qwot_multipliers[:] = np.nan
        return qwot_multipliers

    if nH_r <=0 or nL_r <= 0:
        log_message(f"Warning: Cannot calculate QWOT, real indices nH_r ({nH_r}) and nL_r ({nL_r}) must be positive.")
        qwot_multipliers[:] = np.nan
        return qwot_multipliers

    for i in range(num_layers):
        n_real = nH_r if i % 2 == 0 else nL_r
        if n_real <= 1e-9: # Avoid division by zero
            # log_message(f"Warning: Layer {i+1} has near-zero real index ({n_real}), QWOT multiplier set to NaN.")
            qwot_multipliers[i] = np.nan # Cannot determine multiplier
        else:
            qwot_multipliers[i] = ep_vector[i] * (4 * n_real) / l0
    return qwot_multipliers


def calculate_final_mse(res, active_targets):
    total_squared_error = 0.0; total_points_in_targets = 0; mse = None

    # Basic validation of input 'res'
    if not active_targets or not isinstance(res, dict) or 'Ts' not in res or res['Ts'] is None or not isinstance(res['Ts'], np.ndarray) or len(res['Ts'])==0 or 'l' not in res or res['l'] is None or not isinstance(res['l'], np.ndarray):
        # log_message("MSE Calc: Invalid input 'res' or no active targets.")
        return mse, total_points_in_targets # Return None, 0

    calculated_lambdas = res['l']
    calculated_Ts = res['Ts']

    if len(calculated_lambdas) != len(calculated_Ts):
         log_message("MSE Calc Warning: Lambda and Ts arrays have different lengths.")
         return None, 0 # Inconsistent data

    for i, target in enumerate(active_targets):
        # Ensure target keys exist and values are reasonable
        l_min = target.get('min')
        l_max = target.get('max')
        t_min = target.get('target_min')
        t_max = target.get('target_max')

        if None in [l_min, l_max, t_min, t_max]:
            log_message(f"MSE Calc Warning: Target {i+1} is missing required keys.")
            continue # Skip this target

        # Basic type/value checks if needed (already done in get_active_targets)
        # if not isinstance(l_min, (int, float)) ... etc.

        # Find indices within the target lambda range
        indices = get_target_points_indices(calculated_lambdas, l_min, l_max)

        if indices.size > 0:
            # Ensure indices are valid for the Ts array (should be if lengths match)
            valid_indices = indices[indices < len(calculated_Ts)]
            if valid_indices.size == 0:
                # log_message(f"MSE Calc Info: No calculated points found within target zone {i+1} [{l_min:.1f}-{l_max:.1f} nm].")
                continue # No points in this zone

            calculated_Ts_in_zone = calculated_Ts[valid_indices]
            target_lambdas_in_zone = calculated_lambdas[valid_indices]

            # Check for non-finite values *before* calculations
            finite_mask = np.isfinite(calculated_Ts_in_zone)
            calculated_Ts_in_zone = calculated_Ts_in_zone[finite_mask]
            target_lambdas_in_zone = target_lambdas_in_zone[finite_mask]

            if calculated_Ts_in_zone.size == 0:
                # log_message(f"MSE Calc Info: All points in target zone {i+1} were non-finite.")
                continue # No valid points left

            # Calculate the target T value for each lambda point in the zone
            if abs(l_max - l_min) < 1e-9: # Handle case of single point target or very narrow range
                interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
            else:
                # Calculate slope only once
                slope = (t_max - t_min) / (l_max - l_min)
                interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)

            # Clip interpolated target T to [0, 1] just in case? Usually handled by input validation.
            # interpolated_target_t = np.clip(interpolated_target_t, 0.0, 1.0)

            squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
            total_squared_error += np.sum(squared_errors)
            total_points_in_targets += len(calculated_Ts_in_zone) # Use the length after filtering NaNs


    # Calculate final MSE
    if total_points_in_targets > 0:
        mse = total_squared_error / total_points_in_targets
    elif active_targets: # Targets were defined, but no points fell into them
        mse = np.nan # Indicate no points were evaluated
        # log_message("MSE Calc Info: No valid calculated points found across all active target zones.")

    return mse, total_points_in_targets

# Note: The `ep_initial_for_penalty` argument seems unused here. Removed it. If needed, add back.
def calculate_mse_for_optimization_penalized(ep_vector, nH, nL, nSub, l_vec_optim, active_targets, min_thickness_phys_nm):
    # Ensure ep_vector is a numpy array for vectorized operations
    if not isinstance(ep_vector, np.ndarray): ep_vector = np.array(ep_vector)

    # --- Penalty for thickness below minimum ---
    # Apply minimum thickness constraint *before* calculating R/T
    # We don't modify the input vector directly for the optimizer, but calculate cost *as if* it were clamped.
    # Or, alternatively, the optimizer's bounds should handle this.
    # L-BFGS-B bounds handle minimum, but let's add a penalty just in case it tries to go below during its steps.
    ep_vector_calc = np.maximum(ep_vector, min_thickness_phys_nm)
    below_min_mask = ep_vector < min_thickness_phys_nm
    penalty = 0.0
    if np.any(below_min_mask):
        # Significant penalty for violating the hard constraint
        penalty = 1e6 + np.sum((min_thickness_phys_nm - ep_vector[below_min_mask])**2) * 1e8 # Strong penalty


    # --- Calculate R/T ---
    try:
        # Ensure complex refractive indices
        nH_complex = nH + 0j if isinstance(nH, (int, float)) else nH
        nL_complex = nL + 0j if isinstance(nL, (int, float)) else nL
        nSub_complex = nSub + 0j if isinstance(nSub, (int, float)) else nSub

        # Calculate R/T using the potentially clamped thickness vector
        res = calculate_RT_from_ep(ep_vector_calc, nH_complex, nL_complex, nSub_complex, l_vec_optim)

    except Exception as e_rt:
        # Penalize heavily if R/T calculation fails
        # log_message(f"DEBUG: R/T Calc Error in cost func: {e_rt} for ep={ep_vector}") # Debug only
        return np.inf # Or a very large number like 1e12

    # --- Calculate MSE based on Targets ---
    total_squared_error = 0.0
    total_points_in_targets = 0

    calculated_lambdas = res['l']
    calculated_Ts = res['Ts']

    # Check for NaN/inf in the *entire* Ts result early?
    if np.any(~np.isfinite(calculated_Ts)):
         # log_message(f"DEBUG: Non-finite Ts found in cost func: {calculated_Ts}") # Debug only
         return 1e8 + penalty # Penalize non-physical results

    for target in active_targets:
        l_min = target['min']
        l_max = target['max']
        t_min = target['target_min']
        t_max = target['target_max']

        indices = get_target_points_indices(calculated_lambdas, l_min, l_max)

        if indices.size > 0:
            valid_indices = indices[indices < len(calculated_Ts)] # Should be redundant if lengths match
            if valid_indices.size == 0: continue

            calculated_Ts_in_zone = calculated_Ts[valid_indices]
            target_lambdas_in_zone = calculated_lambdas[valid_indices]

            # We already checked for NaNs globally, but maybe check again per zone? Redundant.
            # if np.any(np.isnan(calculated_Ts_in_zone)):
            #     return 1e8 + penalty # Penalize if calculation failed for this zone

            # Calculate interpolated target T
            if abs(l_max - l_min) < 1e-9:
                interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
            else:
                slope = (t_max - t_min) / (l_max - l_min)
                interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)

            squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
            total_squared_error += np.sum(squared_errors)
            total_points_in_targets += len(valid_indices) # Count points contributing to MSE

    # --- Final Cost ---
    if total_points_in_targets == 0:
        # If no targets were hit (e.g., lambda range mismatch), penalize slightly?
        # Or maybe the physics calculation failed entirely, already handled.
        # If targets exist but no points calculated fall in them, it's likely an issue
        # with l_vec_optim vs target ranges. Return a high cost.
        mse = 1e7 # High cost if no points were evaluated within targets
    else:
        mse = total_squared_error / total_points_in_targets

    final_cost = mse + penalty # Add thickness penalty
    return final_cost


def perform_single_thin_layer_removal(ep_vector_in, min_thickness_phys,
                                      cost_function, args_for_cost,
                                      log_prefix="", target_layer_index=None):
    """
    Attempts to remove the thinnest layer (or a specified target layer)
    and re-optimize the structure locally.

    Returns:
        tuple: (
            new_ep_vector or original_ep_vector,
            bool indicating if re-optimization was successful (implies structure changed),
            cost of the resulting structure (potentially inf),
            list of log messages generated during the process
        )
    """
    current_ep = ep_vector_in.copy()
    logs = []
    num_layers = len(current_ep)

    # --- Basic Checks ---
    if num_layers <= 1: # Need at least 2 layers to potentially merge/remove
        logs.append(f"{log_prefix}Structure has {num_layers} layers. Cannot merge/delete further.")
        # Return original structure, unsuccessful, original cost (or inf if cannot calculate)
        try: initial_cost = cost_function(current_ep, *args_for_cost)
        except Exception: initial_cost = np.inf
        return current_ep, False, initial_cost, logs

    # --- Identify Layer to Remove ---
    thin_layer_index = -1
    min_thickness_found = np.inf

    if target_layer_index is not None:
        if 0 <= target_layer_index < num_layers:
            # Check if the target layer actually needs removal based on thickness?
            # Original logic targets *any* specified valid index. Let's keep that.
            # if current_ep[target_layer_index] < min_thickness_phys: # Check if it's "thin"
            thin_layer_index = target_layer_index
            min_thickness_found = current_ep[target_layer_index]
            logs.append(f"{log_prefix}Targeting specified layer {thin_layer_index + 1} ({min_thickness_found:.3f} nm) for potential removal.")
            # else:
            #     logs.append(f"{log_prefix}Specified target layer {target_layer_index + 1} ({current_ep[target_layer_index]:.3f} nm) is not below min thickness ({min_thickness_phys:.3f} nm). Finding thinnest instead.")
            #     target_layer_index = None # Fallback to finding the actual thinnest
        else:
            logs.append(f"{log_prefix}Specified target layer index {target_layer_index+1} is invalid. Finding absolute thinnest.")
            target_layer_index = None # Fallback

    if target_layer_index is None: # Find the actual thinnest layer >= physical min
        # Find the minimum thickness among layers that are *at least* the physical minimum
        # This allows removing the "thinnest valid" layer, not necessarily sub-nm layers unless specified
        eligible_indices = np.where(current_ep >= min_thickness_phys)[0]
        if eligible_indices.size > 0:
             min_idx_within_eligible = np.argmin(current_ep[eligible_indices])
             thin_layer_index = eligible_indices[min_idx_within_eligible]
             min_thickness_found = current_ep[thin_layer_index]
        # else: No layers meet the minimum thickness criteria - should not happen if bounds work

    if thin_layer_index == -1:
        logs.append(f"{log_prefix}No suitable layer found for removal (thinnest >= {min_thickness_phys:.3f} nm).")
        try: initial_cost = cost_function(current_ep, *args_for_cost)
        except Exception: initial_cost = np.inf
        return current_ep, False, initial_cost, logs

    # Log which layer is being targeted if not explicitly specified before
    if target_layer_index is None:
        logs.append(f"{log_prefix}Identified thinnest eligible layer: Layer {thin_layer_index + 1} ({min_thickness_found:.3f} nm).")

    # --- Perform Removal/Merge ---
    ep_after_merge = None
    merged_info = ""
    structure_changed = False

    # Case 1: Removing the first layer (index 0) - requires at least 3 layers to merge 1+3 around 2.
    # If removing layer 0, the logic should be based on merging with layer 2? No, the original code removed 0 and 1. Let's stick to that logic for now.
    # Removing layer 0 and 1 means the new stack starts at original layer 2.
    if thin_layer_index == 0:
        if num_layers >= 2: # Can remove first two layers
            ep_after_merge = current_ep[2:] # New stack starts from the third layer
            merged_info = f"Removing layer 1 (index 0) and layer 2 (index 1)." # Indices are 0-based
            logs.append(f"{log_prefix}{merged_info} New structure: {len(ep_after_merge)} layers.")
            structure_changed = True
        else: # Only 1 layer, cannot remove 0 and 1
             logs.append(f"{log_prefix}Cannot remove layer 1 (index 0) - structure too small (needs >= 2 layers).")
             # Return original
             try: cost = cost_function(current_ep, *args_for_cost)
             except Exception: cost = np.inf
             return current_ep, False, cost, logs

    # Case 2: Removing the last layer (index num_layers - 1)
    elif thin_layer_index == num_layers - 1:
         # Original code seemed to just remove the last layer without merging. Let's stick to that interpretation.
         # Need at least 1 layer to remove the last one.
         if num_layers >= 1:
              ep_after_merge = current_ep[:-1] # Remove the last element
              merged_info = f"Removed last layer {num_layers} (index {num_layers-1}) ({current_ep[-1]:.3f} nm)."
              logs.append(f"{log_prefix}{merged_info} New structure: {len(ep_after_merge)} layers.")
              structure_changed = True
         # else: num_layers is 0, already handled at the start

    # Case 3: Removing an intermediate layer (index > 0 and < num_layers - 1)
    # Requires merging layers i-1 and i+1. Needs at least 3 layers total.
    else: # thin_layer_index is between 1 and num_layers-2 (inclusive)
        if num_layers >= 3:
            merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
            ep_after_merge = np.concatenate((
                current_ep[:thin_layer_index - 1], # Layers before the one to the left
                [merged_thickness],              # The new merged layer
                current_ep[thin_layer_index + 2:]  # Layers after the one to the right
            ))
            merged_info = (f"Removed layer {thin_layer_index + 1} (index {thin_layer_index}), "
                           f"merged layer {thin_layer_index} (index {thin_layer_index - 1}) ({current_ep[thin_layer_index - 1]:.3f}) "
                           f"with layer {thin_layer_index + 2} (index {thin_layer_index + 1}) ({current_ep[thin_layer_index + 1]:.3f}) -> {merged_thickness:.3f}")
            logs.append(f"{log_prefix}{merged_info} New structure: {len(ep_after_merge)} layers.")
            structure_changed = True
        else: # Should not happen if index is intermediate, but as safeguard
             logs.append(f"{log_prefix}Cannot merge around layer {thin_layer_index+1} - structure too small.")
             try: cost = cost_function(current_ep, *args_for_cost)
             except Exception: cost = np.inf
             return current_ep, False, cost, logs

    # --- Re-optimize if structure changed ---
    final_ep = current_ep # Default to original if merge fails or no change
    final_cost = np.inf
    success_overall = False

    if structure_changed and ep_after_merge is not None:
        # Apply min thickness to the merged structure *before* re-optimization
        if ep_after_merge.size > 0:
            ep_after_merge = np.maximum(ep_after_merge, min_thickness_phys)

        num_layers_reopt = len(ep_after_merge)

        if num_layers_reopt == 0:
            logs.append(f"{log_prefix}Empty structure after merge/removal. Returning empty.")
            return np.array([]), True, np.inf, logs # Changed to empty, cost is inf

        # Setup for local optimization
        reopt_bounds = [(min_thickness_phys, None)] * num_layers_reopt
        x0_reopt = ep_after_merge # Start from the merged structure

        logs.append(f"{log_prefix}Starting local re-optimization (L-BFGS-B, maxiter=199) on {num_layers_reopt} layers...")
        reopt_start_time = time.time()
        reopt_args = args_for_cost # Use the same cost function args

        try:
            reopt_result = minimize(cost_function, x0_reopt, args=reopt_args,
                                    method='L-BFGS-B', bounds=reopt_bounds,
                                    options={'maxiter': 199, 'ftol': 1e-10, 'gtol': 1e-7, 'disp': False}) # disp=False for less console noise
            reopt_time = time.time() - reopt_start_time

            reopt_success = reopt_result.success and np.isfinite(reopt_result.fun)
            reopt_cost = reopt_result.fun if reopt_success else np.inf
            reopt_iters = reopt_result.nit

            logs.append(f"{log_prefix}Re-optimization finished in {reopt_time:.3f}s. Success: {reopt_success}, Cost: {reopt_cost:.3e}, Iters: {reopt_iters}")

            if reopt_success:
                # Apply min thickness *after* optimization as well
                final_ep = np.maximum(reopt_result.x.copy(), min_thickness_phys)
                final_cost = reopt_cost # Use the cost reported by minimize
                success_overall = True
                logs.append(f"{log_prefix}Re-optimization successful.")
            else:
                # Re-opt failed, return the merged structure without re-optimization
                logs.append(f"{log_prefix}Re-optimization failed. Returning merged/removed, non-re-optimized structure.")
                final_ep = np.maximum(ep_after_merge.copy(), min_thickness_phys) # Ensure min thickness on the fallback
                success_overall = False # Re-opt failed
                # Calculate the cost of this non-reoptimized structure
                try:
                    final_cost = cost_function(final_ep, *reopt_args)
                    logs.append(f"{log_prefix}Recalculated cost (non-re-opt): {final_cost:.3e}")
                except Exception as e_cost:
                    final_cost = np.inf
                    logs.append(f"{log_prefix}Error calculating cost of merged/removed structure: {e_cost}")

        except Exception as e_reopt:
            logs.append(f"{log_prefix}ERROR during L-BFGS-B re-optimization: {e_reopt}\n{traceback.format_exc(limit=2)}")
            logs.append(f"{log_prefix}Returning merged/removed structure without re-optimization attempt due to error.")
            final_ep = np.maximum(ep_after_merge.copy(), min_thickness_phys) # Ensure min thickness
            success_overall = False
            try: final_cost = cost_function(final_ep, *reopt_args)
            except Exception: final_cost = np.inf # Assign inf if cost calc fails too

    else: # Structure not changed (or merge resulted in None - should not happen)
        logs.append(f"{log_prefix}Merge/removal not performed or structure unchanged. No re-optimization.")
        success_overall = False
        final_ep = current_ep # Already set as default
        # Calculate cost of original structure if needed (e.g., if called standalone)
        try: final_cost = cost_function(final_ep, *args_for_cost)
        except Exception: final_cost = np.inf


    return final_ep, success_overall, final_cost, logs



# --- Optimization Functions ---

def local_search_worker(start_ep, cost_function, args_for_cost, lbfgsb_bounds, min_thickness_phys_nm):
    """Worker function for parallel local search."""
    worker_pid = os.getpid()
    start_time = time.time()
    log_lines = [] # Collect logs to return
    log_lines.append(f"  [PID:{worker_pid} Start]: Starting local search (L-BFGS-B only).")

    initial_cost = np.inf
    start_ep_checked = np.array([]) # Initialize

    try:
        # Ensure starting point respects minimum thickness for initial cost eval
        start_ep_checked = np.maximum(np.asarray(start_ep), min_thickness_phys_nm)
        if start_ep_checked.size == 0: raise ValueError("Start EP is empty.")
        initial_cost = cost_function(start_ep_checked, *args_for_cost)
        if not np.isfinite(initial_cost):
             log_lines.append(f"  [PID:{worker_pid} Start]: Warning - Initial cost is not finite ({initial_cost:.3e}).")
             initial_cost = np.inf # Treat as infinite if non-finite

    except Exception as e:
        log_lines.append(f"  [PID:{worker_pid} Start]: ERROR calculating initial cost: {e}")
        # Return a structure indicating failure
        result = OptimizeResult(x=np.asarray(start_ep), success=False, fun=np.inf, message=f"Worker Initial Cost Error: {e}", nit=0)
        end_time = time.time()
        worker_summary = f"Worker {worker_pid} finished in {end_time - start_time:.2f}s. StartCost=ERROR, FinalCost=inf, Success=False, Msg='Initial Cost Error'"
        log_lines.append(f"  [PID:{worker_pid} Summary]: {worker_summary}")
        return {'result': result, 'start_ep': start_ep, 'pid': worker_pid, 'log_lines': log_lines, 'initial_cost': np.inf}

    iter_count = 0
    # Callback for logging progress (optional, can add overhead)
    # def optimization_callback(xk):
    #     nonlocal iter_count
    #     iter_count += 1
    #     # Log less frequently to reduce overhead
    #     if iter_count % 50 == 0 or iter_count == 1:
    #         try:
    #             current_cost = cost_function(xk, *args_for_cost)
    #             log_lines.append(f"    [PID:{worker_pid} Iter {iter_count}]: Cost={current_cost:.3e}")
    #         except Exception:
    #              log_lines.append(f"    [PID:{worker_pid} Iter {iter_count}]: Error calculating cost in callback.")

    result = OptimizeResult(x=start_ep_checked, success=False, fun=initial_cost, message="Worker optimization did not run or failed early", nit=0)
    try:
        # Ensure bounds match dimensions
        if len(lbfgsb_bounds) != len(start_ep_checked):
             raise ValueError(f"Dimension mismatch: start_ep ({len(start_ep_checked)}) vs bounds ({len(lbfgsb_bounds)})")

        result = minimize(
            cost_function,
            start_ep_checked, # Start from the checked vector
            args=args_for_cost,
            method='L-BFGS-B',
            bounds=lbfgsb_bounds,
            options={'disp': False, 'maxiter': 99, 'ftol': 1.e-10, 'gtol': 1e-7}, # Standard options
            # callback=optimization_callback # Uncomment to enable callback logging
        )

        # Ensure final result also respects minimum thickness
        if result.x is not None:
             result.x = np.maximum(result.x, min_thickness_phys_nm)
             # Recalculate final cost after potential clamping? Optional, minimize cost is usually sufficient.
             # try:
             #    result.fun = cost_function(result.x, *args_for_cost)
             # except Exception: result.fun = np.inf


    except Exception as e:
        end_time = time.time()
        error_message = f"Worker {worker_pid} Exception during minimize after {end_time-start_time:.2f}s: {type(e).__name__}: {e}"
        log_lines.append(f"  [PID:{worker_pid} ERROR]: {error_message}\n{traceback.format_exc(limit=2)}")
        # Ensure result reflects failure
        result = OptimizeResult(x=start_ep_checked, success=False, fun=np.inf, message=f"Worker Exception: {e}", nit=iter_count) # Use start_ep_checked as fallback x


    # --- Prepare return data ---
    end_time = time.time()
    final_cost = result.fun if np.isfinite(result.fun) else np.inf
    # Consider success only if cost is finite and improvemed? Or just rely on result.success?
    # Let's rely on finite cost as a primary indicator of practical success.
    success_status = result.success and np.isfinite(final_cost)
    iterations = result.nit
    message = result.message.decode('utf-8') if isinstance(result.message, bytes) else str(result.message)


    worker_summary = (
        f"Worker {worker_pid} finished in {end_time - start_time:.2f}s. "
        f"StartCost={initial_cost:.3e}, FinalCost={final_cost:.3e}, "
        f"Success={success_status} (LBFGSB Success: {result.success}, Finite Cost: {np.isfinite(final_cost)}), " # More detailed success info
        f"LBFGSB_Iters={iterations}, Msg='{message}'"
    )
    log_lines.append(f"  [PID:{worker_pid} Summary]: {worker_summary}")

    # Ensure returned 'x' is the best found, even if optimization failed mid-way
    final_x = result.x if success_status and result.x is not None else start_ep_checked

    return {'result': result, # Contains the OptimizeResult object
            'final_ep': final_x, # Explicitly return the final vector used
            'final_cost': final_cost, # Explicitly return the final cost
            'success': success_status, # Our definition of success
            'start_ep': start_ep, # Original start point
            'pid': worker_pid,
            'log_lines': log_lines,
            'initial_cost': initial_cost}


def _run_sobol_evaluation(num_layers, n_samples, lower_bounds, upper_bounds,
                          cost_function_partial_map, min_thickness_phys_nm, phase_name="Phase 1",
                          progress_hook=None):
    log_with_elapsed_time(f"\n--- {phase_name}: Initial Evaluation (Sobol, N={n_samples}) ---")

    start_time_eval = time.time()
    if lower_bounds is None or upper_bounds is None or len(lower_bounds) != num_layers or len(upper_bounds) != num_layers:
        raise ValueError(f"_run_sobol_evaluation requires valid lower and upper bounds matching num_layers ({num_layers}) (phase: {phase_name}).")

    # Ensure bounds are valid (lower <= upper)
    if np.any(lower_bounds > upper_bounds):
         log_with_elapsed_time(f"Warning: Sobol lower bounds > upper bounds found. Clamping.")
         upper_bounds = np.maximum(lower_bounds, upper_bounds)

    # Ensure bounds respect physical minimum
    lower_bounds = np.maximum(min_thickness_phys_nm, lower_bounds)
    upper_bounds = np.maximum(lower_bounds, upper_bounds) # Ensure upper >= lower after clamping lower

    log_with_elapsed_time(f"  Bounds (min={min_thickness_phys_nm:.3f}): L={np.array2string(lower_bounds, precision=3)}, U={np.array2string(upper_bounds, precision=3)}")


    # Use Sobol sequence for quasi-random sampling
    sampler = qmc.Sobol(d=num_layers, scramble=True)
    points_unit_cube = sampler.random(n=n_samples)

    # Scale unit cube points to the actual bounds
    ep_candidates = qmc.scale(points_unit_cube, lower_bounds, upper_bounds)

    # Apply minimum thickness constraint *after* scaling
    # This is crucial as scaling might produce values below the minimum
    ep_candidates = [np.maximum(min_thickness_phys_nm, cand) for cand in ep_candidates]


    costs = []
    initial_results = [] # List of (cost, ep_vector) tuples
    num_workers_eval = min(n_samples, os.cpu_count()) # Limit workers
    log_with_elapsed_time(f"  Evaluating cost (MSE) for {n_samples} candidates (max {num_workers_eval} workers)...")

    eval_start_pool = time.time()
    try:
        if ep_candidates:
            # Use multiprocessing pool for parallel evaluation
            with multiprocessing.Pool(processes=num_workers_eval) as pool:
                costs = pool.map(cost_function_partial_map, ep_candidates)
            eval_pool_time = time.time() - eval_start_pool
            log_with_elapsed_time(f"  Parallel evaluation finished in {eval_pool_time:.2f}s.")

            # Basic check on results length
            if len(costs) != len(ep_candidates):
                 log_with_elapsed_time(f"  Warning: Mismatch in cost results length ({len(costs)}) vs candidates ({len(ep_candidates)}). Truncating.")
                 costs = costs[:len(ep_candidates)] # Truncate results if necessary

            initial_results = list(zip(costs, ep_candidates))
        else:
            log_with_elapsed_time("  No candidates generated to evaluate.")
            initial_results = []

    except Exception as e_pool:
        log_with_elapsed_time(f"  ERROR during parallel evaluation: {type(e_pool).__name__}: {e_pool}\n{traceback.format_exc(limit=2)}")
        log_with_elapsed_time("  Switching to sequential evaluation...")
        costs = []
        eval_start_seq = time.time()
        for i, cand in enumerate(ep_candidates):
            try:
                cost = cost_function_partial_map(cand)
                costs.append(cost)
            except Exception as e_seq:
                costs.append(np.inf) # Assign inf cost if sequential evaluation fails
                # log_with_elapsed_time(f"  Error evaluating candidate {i+1} sequentially: {e_seq}") # Optional: log specific errors
            # Progress update for sequential (can be slow)
            if (i + 1) % 200 == 0 or i == len(ep_candidates) - 1:
                 log_with_elapsed_time(f"    Evaluated {i + 1}/{n_samples} sequentially...")
                 if progress_hook:
                      progress_hook( (i + 1) / n_samples ) # Update progress bar

        eval_seq_time = time.time() - eval_start_seq
        initial_results = list(zip(costs, ep_candidates))
        log_with_elapsed_time(f"  Sequential evaluation finished in {eval_seq_time:.2f}s.")

    # Filter results: Keep only those with finite cost below a reasonable threshold (e.g., < 1e9)
    valid_initial_results = [(c, p) for c, p in initial_results if np.isfinite(c) and c < 1e9]
    num_invalid = len(initial_results) - len(valid_initial_results)
    if num_invalid > 0: log_with_elapsed_time(f"  Filtering: {num_invalid} invalid initial costs (non-finite or >= 1e9) discarded.")

    if not valid_initial_results:
        log_with_elapsed_time(f"Warning: No valid initial costs found after {phase_name} evaluation.")
        # Depending on context, might want to raise an error or return empty list
        return [] # Return empty list if no valid points

    # Sort the valid results by cost (ascending)
    valid_initial_results.sort(key=lambda x: x[0])

    eval_time = time.time() - start_time_eval
    log_with_elapsed_time(f"--- End {phase_name} (Initial Evaluation) in {eval_time:.2f}s ---")
    log_with_elapsed_time(f"  Found {len(valid_initial_results)} valid points. Best initial cost: {valid_initial_results[0][0]:.3e}")

    return valid_initial_results # Return sorted list of (cost, ep_vector)


def _run_parallel_local_search(p_best_starts, cost_function, args_for_cost_tuple, lbfgsb_bounds, min_thickness_phys_nm, progress_hook=None):
    """Runs local search (L-BFGS-B) in parallel for the given starting points."""
    p_best_actual = len(p_best_starts)
    if p_best_actual == 0:
        log_with_elapsed_time("WARNING: No starting points provided for local search. Skipping Phase 2.")
        return [] # Return empty list if no starts

    log_with_elapsed_time(f"\n--- Phase 2: Local Search (L-BFGS-B) (P={p_best_actual}) ---")
    start_time_local = time.time()

    # Create a partial function for the worker, fixing arguments
    local_search_partial = partial(local_search_worker,
                                   cost_function=cost_function,
                                   args_for_cost=args_for_cost_tuple,
                                   lbfgsb_bounds=lbfgsb_bounds,
                                   min_thickness_phys_nm=min_thickness_phys_nm)

    local_results_raw = [] # List to store results from workers
    num_workers_local = min(p_best_actual, os.cpu_count()) # Limit workers
    log_with_elapsed_time(f"  Starting {p_best_actual} local searches in parallel (max {num_workers_local} workers)...")

    local_start_pool = time.time()
    try:
        if p_best_starts:
            with multiprocessing.Pool(processes=num_workers_local) as pool:
                 # Use imap_unordered for potentially better progress tracking if needed, but map is simpler
                 # Wrap in list to ensure all results are gathered before proceeding
                 local_results_raw = list(pool.map(local_search_partial, p_best_starts))

            local_pool_time = time.time() - local_start_pool
            log_with_elapsed_time(f"  Parallel local searches finished in {local_pool_time:.2f}s.")
        else:
            # This case is already handled by the check at the beginning
            pass

    except Exception as e_pool_local:
        log_with_elapsed_time(f"  ERROR during parallel local search pool execution: {type(e_pool_local).__name__}: {e_pool_local}\n{traceback.format_exc(limit=2)}")
        # Depending on desired robustness, could try sequential execution here
        # For simplicity now, we return an empty list indicating failure
        local_results_raw = [] # Indicate failure

    local_time = time.time() - start_time_local
    log_with_elapsed_time(f"--- End Phase 2 (Local Search) in {local_time:.2f}s ---")

    return local_results_raw # Return list of dictionaries from workers


def _process_optimization_results(local_results_raw, initial_best_ep, initial_best_cost):
    """Processes results from parallel local search workers."""
    log_with_elapsed_time(f"\n--- Processing {len(local_results_raw)} local search results ---")

    # Initialize overall best with the best *before* local search started
    overall_best_cost = initial_best_cost
    overall_best_ep = initial_best_ep.copy() if initial_best_ep is not None else None
    # Create a default result object for the initial best
    overall_best_result_obj = OptimizeResult(x=overall_best_ep, fun=overall_best_cost, success=(overall_best_ep is not None and np.isfinite(overall_best_cost)), message="Initial best (pre-local search)", nit=0)
    best_run_info = {'index': -1, 'cost': overall_best_cost} # -1 indicates initial best
    processed_results_count = 0

    if overall_best_ep is None:
         log_with_elapsed_time("Warning: Initial best EP for processing is None.")

    for i, worker_output in enumerate(local_results_raw):
        run_idx = i + 1 # 1-based index for logging

        # Log messages from the worker first
        if isinstance(worker_output, dict) and 'log_lines' in worker_output:
            worker_log_lines = worker_output.get('log_lines', [])
            for line in worker_log_lines:
                log_with_elapsed_time(f"  (Worker Log Run {run_idx}): {line.strip()}")
        else:
            log_with_elapsed_time(f"  Result {run_idx}: Unexpected format from worker, cannot log details. Output: {worker_output}")
            continue # Skip processing this result

        # Now process the actual result data from the worker
        if 'result' in worker_output and 'final_ep' in worker_output and 'final_cost' in worker_output and 'success' in worker_output:
            res_obj = worker_output['result'] # The scipy OptimizeResult object
            final_cost_run = worker_output['final_cost'] # Worker's reported final cost
            final_ep_run = worker_output['final_ep']     # Worker's reported final EP vector
            success_run = worker_output['success']       # Worker's reported success status

            processed_results_count += 1

            # Check if the result is valid and better than the current overall best
            if success_run and final_ep_run is not None and final_ep_run.size > 0 and final_cost_run < overall_best_cost:
                log_with_elapsed_time(f"  ==> New overall best cost found by Result {run_idx}! Cost: {final_cost_run:.3e} (Previous: {overall_best_cost:.3e})")
                overall_best_cost = final_cost_run
                overall_best_ep = final_ep_run.copy() # Important to copy
                overall_best_result_obj = res_obj # Store the corresponding OptimizeResult object
                best_run_info = {'index': run_idx, 'cost': final_cost_run}
            # else: # Log if run succeeded but wasn't better (optional, can be verbose)
            #    if success_run:
            #        log_with_elapsed_time(f"  Result {run_idx}: Succeeded (Cost={final_cost_run:.3e}) but did not improve best ({overall_best_cost:.3e}).")
            #    elif final_ep_run is None or final_ep_run.size == 0:
            #         log_with_elapsed_time(f"  Result {run_idx}: Invalid result (empty ep vector).")
            #    else: # Not successful
            #         log_with_elapsed_time(f"  Result {run_idx}: Did not succeed (Cost={final_cost_run:.3e}).")


        else:
            log_with_elapsed_time(f"  Result {run_idx}: Worker output dictionary missing required keys ('result', 'final_ep', 'final_cost', 'success').")


    log_with_elapsed_time(f"--- Finished processing local search results ---")
    if processed_results_count == 0 and best_run_info['index'] == -1:
        log_with_elapsed_time("WARNING: No valid local search results were processed. Keeping pre-local best result.")
    elif best_run_info['index'] > 0:
        log_with_elapsed_time(f"Overall best result found by local search Run {best_run_info['index']} (Cost {best_run_info['cost']:.3e}).")
    elif best_run_info['index'] == -1: # Initial remained best
        log_with_elapsed_time(f"No local search improved the pre-local search best result (Cost {best_run_info['cost']:.3e}).")

    # Final sanity check
    if overall_best_ep is None or len(overall_best_ep) == 0:
         log_with_elapsed_time("CRITICAL WARNING: Result processing finished with an empty or None best thickness vector!")
         # Attempt to fall back to the absolute initial state if possible? Risky.
         # overall_best_ep = initial_best_ep.copy() if initial_best_ep is not None else None
         # overall_best_cost = initial_best_cost if overall_best_ep is not None else np.inf
         # If even initial_best_ep was None, we have a bigger problem upstream.
         if initial_best_ep is None:
             raise RuntimeError("Result processing failed: Final thickness vector is empty, and initial was also empty.")
         else:
             # Fallback to initial state if processing somehow lost the vector
             log_with_elapsed_time("Falling back to initial state due to empty final vector.")
             overall_best_ep = initial_best_ep.copy()
             overall_best_cost = initial_best_cost
             overall_best_result_obj = OptimizeResult(x=overall_best_ep, fun=overall_best_cost, success=False, message="Fell back to initial state", nit=0)


    return overall_best_ep, overall_best_cost, overall_best_result_obj


# --- Plotting ---

def setup_axis_grids(ax):
    """Helper to add major and minor grids."""
    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.7, alpha=0.7)
    ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.minorticks_on()

def tracer_graphiques(res, ep_actual, nH_r, nH_i, nL_r, nL_i, nSub, active_targets_for_plot, mse,
                      is_optimized=False, method_name="", res_optim_grid=None):
    """Generates the Matplotlib figure with spectral, index, and stack plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Create figure and axes
    opt_method_str = f" ({method_name})" if method_name else ""
    window_title = f'Stack Results{opt_method_str}' if is_optimized else 'Nominal Stack Calculation Results'
    fig.suptitle(window_title, fontsize=14, weight='bold') # Set title for the whole figure

    num_layers = len(ep_actual) if ep_actual is not None else 0
    ep_cum = np.cumsum(ep_actual) if num_layers > 0 else np.array([])

    # 1. Spectral Plot (Transmittance)
    ax_spec = axes[0]
    if res and 'l' in res and 'Ts' in res and res['l'] is not None and res['Ts'] is not None:
        line_ts, = ax_spec.plot(res['l'], res['Ts'], label='Transmittance (Plot Grid)', linestyle='-', color='blue', linewidth=1.5)

        # Draw Target Ramps/Points
        target_lines_drawn = False
        if active_targets_for_plot:
            plotted_label = False
            for target in active_targets_for_plot:
                l_min, l_max = target['min'], target['max']
                t_min, t_max = target['target_min'], target['target_max']
                x_coords = [l_min, l_max]
                y_coords = [t_min, t_max]

                label = 'Target Ramp' if not plotted_label else "_nolegend_"
                ax_spec.plot(x_coords, y_coords, 'r--', linewidth=1.5, alpha=0.8, label=label, zorder=5)
                plotted_label = True

                # Mark endpoints of the target ramp
                ax_spec.plot(x_coords, y_coords, marker='x', color='red', markersize=8, linestyle='none', label='_nolegend_', zorder=6)

                # Plot target points on the *optimization* grid if available
                if res_optim_grid and 'l' in res_optim_grid and res_optim_grid['l'].size > 0 and 'Ts' in res_optim_grid:
                    # Find optimization lambda points within this target's range
                    indices_in_zone_optim = np.where((res_optim_grid['l'] >= l_min) & (res_optim_grid['l'] <= l_max))[0]

                    if indices_in_zone_optim.size > 0:
                        optim_lambdas_in_zone = res_optim_grid['l'][indices_in_zone_optim]

                        # Calculate the *target* T value at these specific lambda points
                        if abs(l_max - l_min) < 1e-9:
                             optim_target_t_in_zone = np.full_like(optim_lambdas_in_zone, t_min)
                        else:
                             slope = (t_max - t_min) / (l_max - l_min)
                             optim_target_t_in_zone = t_min + slope * (optim_lambdas_in_zone - l_min)

                        # Plot these target points (where MSE is calculated)
                        ax_spec.plot(optim_lambdas_in_zone, optim_target_t_in_zone,
                                     marker='.', color='red', linestyle='none', markersize=4,
                                     alpha=0.7, label='_nolegend_', zorder=6)
            target_lines_drawn = True

        ax_spec.set_xlabel("Wavelength (nm)")
        ax_spec.set_ylabel('Transmittance')
        ax_spec.set_title(f"Spectral Plot{opt_method_str}") # Removed mention of grid
        setup_axis_grids(ax_spec)
        ax_spec.set_ylim(bottom=-0.05, top=1.05)
        if len(res['l']) > 0: ax_spec.set_xlim(res['l'][0], res['l'][-1])
        if target_lines_drawn or ax_spec.get_legend_handles_labels()[1]: # Add legend if targets or Ts plotted
             ax_spec.legend(fontsize=9)

        # Display MSE value on the plot
        if mse is not None and not np.isnan(mse):
            mse_text = f"MSE (vs Target, Optim grid) = {mse:.3e}"
        elif mse is None and active_targets_for_plot:
            mse_text = "MSE: Calculation Error"
        elif mse is None:
            mse_text = "MSE: N/A (no target)"
        else: # mse is NaN
            mse_text = "MSE: N/A (no target points)"
        ax_spec.text(0.98, 0.98, mse_text, transform=ax_spec.transAxes, ha='right', va='top', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))

        # mplcursors removed - no interactive hover effects in basic conversion
        # if mplcursors:
        #      cursor_spec = mplcursors.cursor([line_ts], hover=True)
        #      @cursor_spec.connect("add")
        #      def on_add_spec(sel): # ... (callback definition)

    else: # No spectral data
        ax_spec.text(0.5, 0.5, "No spectral data available", ha='center', va='center', transform=ax_spec.transAxes)
        ax_spec.set_title(f"Spectral Plot{opt_method_str}")


    # 2. Refractive Index Profile Plot
    ax_idx = axes[1]
    nSub_c = nSub + 0j # Ensure complex
    nH_c_calc = nH_r + 1j * nH_i
    nL_c_calc = nL_r + 1j * nL_i

    # Build index list: [n_layer_1, n_layer_2, ...]
    indices_complex = [nH_c_calc if i % 2 == 0 else nL_c_calc for i in range(num_layers)]
    n_real_layers = [np.real(n) for n in indices_complex]

    total_thickness = ep_cum[-1] if num_layers > 0 else 0
    margin = max(50, 0.1 * total_thickness) if total_thickness > 0 else 50 # Margin for air/substrate visualization

    # Coordinates for steps plot: (depth, n_real)
    # Start deep in substrate, move towards air (depth 0 = substrate interface)
    x_coords_plot = [-margin] # Start in substrate
    y_coords_plot = [np.real(nSub_c)]

    x_coords_plot.append(0) # Substrate ends at depth 0
    y_coords_plot.append(np.real(nSub_c))

    if num_layers > 0:
        for i in range(num_layers):
            layer_start_depth = ep_cum[i-1] if i > 0 else 0 # Depth starts at 0 for first layer
            layer_end_depth = ep_cum[i]
            layer_n_real = n_real_layers[i]

            # Step up to the new index at the start of the layer
            x_coords_plot.append(layer_start_depth)
            y_coords_plot.append(layer_n_real)
            # Stay at that index until the end of the layer
            x_coords_plot.append(layer_end_depth)
            y_coords_plot.append(layer_n_real)

        # After last layer, step into air (n=1.0)
        last_layer_end_depth = ep_cum[-1]
        x_coords_plot.append(last_layer_end_depth)
        y_coords_plot.append(1.0) # Step to air index

        # Extend into air for margin
        x_coords_plot.append(last_layer_end_depth + margin)
        y_coords_plot.append(1.0)

    else: # No layers, just substrate to air transition at depth 0
        x_coords_plot.append(0) # Step to air index at depth 0
        y_coords_plot.append(1.0)
        x_coords_plot.append(margin) # Extend into air
        y_coords_plot.append(1.0)

    # Plot the index profile
    ax_idx.plot(x_coords_plot, y_coords_plot, drawstyle='steps-post', label='Real n', color='purple', linewidth=1.5)

    ax_idx.set_xlabel('Depth (from substrate) (nm)')
    ax_idx.set_ylabel("Real part of index (n')")
    ax_idx.set_title("Refractive Index Profile")
    setup_axis_grids(ax_idx)
    ax_idx.set_xlim(x_coords_plot[0], x_coords_plot[-1]) # Fit x-axis to data range including margins

    # Set y-axis limits dynamically based on index values
    min_n_list = [1.0, np.real(nSub_c)] + n_real_layers
    max_n_list = [1.0, np.real(nSub_c)] + n_real_layers
    min_n = min(min_n_list) if min_n_list else 0.9
    max_n = max(max_n_list) if max_n_list else 2.5
    ax_idx.set_ylim(bottom=min_n - 0.1, top=max_n + 0.1) # Add padding

    # Add text labels for substrate and air
    offset = (max_n - min_n) * 0.05 + 0.02 # Small offset for text
    common_text_opts = {'ha':'center', 'va':'bottom', 'fontsize':8, 'bbox':dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none')}

    # Substrate label position (center of the substrate margin)
    ax_idx.text(-margin / 2, np.real(nSub_c) + offset, f"SUBSTRATE\nn={nSub_c.real:.3f}{nSub_c.imag:+.3f}j" if nSub_c.imag != 0 else f"SUBSTRATE\nn={nSub_c.real:.3f}", **common_text_opts)

    # Air label position (center of the air margin)
    air_x_pos = (total_thickness + margin / 2) if num_layers > 0 else margin / 2
    ax_idx.text(air_x_pos, 1.0 + offset, "AIR\nn=1.0", **common_text_opts)


    # 3. Stack Bar Chart Plot
    ax_stack = axes[2]
    if num_layers > 0:
        colors = ['lightblue' if i % 2 == 0 else 'lightcoral' for i in range(num_layers)] # H=blue, L=red
        bar_pos = np.arange(num_layers) # y-positions for bars

        # Create horizontal bars
        bars = ax_stack.barh(bar_pos, ep_actual, align='center', color=colors, edgecolor='grey', height=0.8)

        # Create y-tick labels with layer info
        yticks_labels = []
        for i, n_comp in enumerate(indices_complex):
            layer_type = "H" if i % 2 == 0 else "L"
            n_str = f"{np.real(n_comp):.3f}"
            k_val = np.imag(n_comp)
            if abs(k_val) > 1e-6: # Show imaginary part if significant
                 n_str += f"{k_val:+.3f}j"
            label = f"L{i + 1} ({layer_type}) n={n_str}"
            yticks_labels.append(label)

        ax_stack.set_yticks(bar_pos)
        ax_stack.set_yticklabels(yticks_labels, fontsize=8)
        ax_stack.invert_yaxis() # Show layer 1 at the top

        # Add thickness labels on bars
        max_ep = max(ep_actual) if ep_actual.size > 0 else 1.0 # Find max thickness for text placement
        fontsize = max(6, 9 - num_layers // 10) # Adjust font size based on number of layers
        for i, bar in enumerate(bars):
             e_val = bar.get_width()
             # Decide text alignment and position based on bar width relative to max width
             ha_pos = 'left' if e_val < max_ep * 0.2 else 'right'
             x_text_pos = e_val * 1.05 if ha_pos == 'left' else e_val * 0.95 # Position text inside or outside
             text_color = 'black' if ha_pos == 'left' else 'white' # Contrast color
             ax_stack.text(x_text_pos, bar.get_y() + bar.get_height() / 2, f"{e_val:.2f} nm",
                           va='center', ha=ha_pos, color=text_color, fontsize=fontsize, weight='bold')

    else: # No layers to plot
         ax_stack.text(0.5, 0.5, "No layers defined", ha='center', va='center', fontsize=10, color='grey', transform=ax_stack.transAxes)
         ax_stack.set_yticks([]) # No y-ticks if no bars
         ax_stack.set_xticks([]) # No x-ticks either

    ax_stack.set_xlabel('Thickness (nm)')
    stack_title_prefix = f'Optimized Stack{opt_method_str}' if is_optimized else 'Nominal Stack'
    ax_stack.set_title(f"{stack_title_prefix} ({num_layers} layers)\n(Substrate at bottom -> Air at top)")
    if num_layers > 0: ax_stack.set_ylim(bottom=num_layers - 0.5, top=-0.5) # Adjust y-limits for bars

    # --- Final Figure Adjustments ---
    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.95]) # Adjust layout to prevent overlap, leave space for suptitle
    # Don't call plt.show() here, return the figure instead
    return fig


# --- Streamlit GUI Setup and Logic ---

# Initialize Session State Variables (if they don't exist)
# --- Inputs ---
default_qwot = "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
default_targets = [
    {'min': "480.0", 'max': "500.0", 'target_min': "0.0", 'target_max': "1.0", 'enabled': True},
    {'min': "500.0", 'max': "600.0", 'target_min': "1.0", 'target_max': "1.0", 'enabled': True},
    {'min': "600.0", 'max': "630.0", 'target_min': "1.0", 'target_max': "0.0", 'enabled': True},
    {'min': "400.0", 'max': "480.0", 'target_min': "0.0", 'target_max': "0.0", 'enabled': True},
    {'min': "630.0", 'max': "700.0", 'target_min': "0.0", 'target_max': "0.0", 'enabled': True}
]
# Optimization Params - initialize based on default mode
default_optim_params = OPTIMIZATION_MODES[DEFAULT_MODE]

# State related to results and UI control
if 'current_optimized_ep' not in st.session_state:
    st.session_state.current_optimized_ep = None # Stores the numpy array of optimized thicknesses
if 'optimization_ran_since_nominal_change' not in st.session_state:
    st.session_state.optimization_ran_since_nominal_change = False
if 'optim_start_time' not in st.session_state:
    st.session_state.optim_start_time = None
if 'last_run_calculation_results' not in st.session_state:
    st.session_state.last_run_calculation_results = {'res': None, 'ep': None, 'mse': None, 'res_optim_grid': None}
if 'optim_mode' not in st.session_state:
    st.session_state.optim_mode = DEFAULT_MODE
if 'current_status' not in st.session_state:
    st.session_state.current_status = "Status: Idle"
if 'current_progress' not in st.session_state:
    st.session_state.current_progress = 0
if 'num_layers_display' not in st.session_state:
    st.session_state.num_layers_display = "Layers (Nominal): 0"
if 'thinnest_layer_display' not in st.session_state:
    st.session_state.thinnest_layer_display = "- nm"
if 'optimized_qwot_display' not in st.session_state:
    st.session_state.optimized_qwot_display = ""

# --- GUI Definition ---
st.set_page_config(layout="wide")
st.title("Thin Film Stack Optimizer v2.25-LinTarget-GeomSpace (Streamlit)")

# --- Input Sections ---
# Use columns for better layout
col1, col2 = st.columns(2)

with col1:
    with st.expander("Materials and Substrate", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.nH_r = st.number_input("Material H (n real)", min_value=0.0, value=2.35, step=0.01, format="%.3f", key='nH_r')
            st.session_state.nL_r = st.number_input("Material L (n real)", min_value=0.0, value=1.46, step=0.01, format="%.3f", key='nL_r')
            st.session_state.nSub = st.number_input("Substrate (n real)", min_value=0.0, value=1.52, step=0.01, format="%.3f", key='nSub')
        with c2:
            st.session_state.nH_i = st.number_input("Material H (k imag)", min_value=0.0, value=0.0, step=0.001, format="%.4f", key='nH_i')
            st.session_state.nL_i = st.number_input("Material L (k imag)", min_value=0.0, value=0.0, step=0.001, format="%.4f", key='nL_i')
            st.caption("(n = n' + ik)") # Placeholder for substrate k if needed later

    with st.expander("Stack (Nominal Definition)", expanded=True):
        st.session_state.emp_str = st.text_area("Nominal Structure (QWOT Multipliers, comma-separated)", value=default_qwot, key='emp_str_input')
        st.session_state.l0 = st.number_input("Centering λ (QWOT, nm)", min_value=0.1, value=500.0, step=1.0, format="%.1f", key='l0_input')
        st.text_input("Optimized QWOT (Read-only)", value=st.session_state.optimized_qwot_display, disabled=True, key='opt_qwot_ro')


with col2:
    with st.expander("Calculation & Optimization Parameters", expanded=True):
        st.session_state.optim_mode = st.radio(
            "Optimization Mode Preset",
            options=list(OPTIMIZATION_MODES.keys()),
            index=list(OPTIMIZATION_MODES.keys()).index(st.session_state.optim_mode), # Set default index
            key='optim_mode_radio',
            horizontal=True
        )
        # Update params based on mode (needs to be done *after* radio button interaction)
        selected_params = OPTIMIZATION_MODES[st.session_state.optim_mode]

        st.markdown("---") # Separator

        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state.l_range_deb = st.number_input("λ Start (nm)", min_value=0.1, value=400.0, step=1.0, format="%.1f", key='l_range_deb_input')
        with c2:
            st.session_state.l_range_fin = st.number_input("λ End (nm)", min_value=0.1, value=700.0, step=1.0, format="%.1f", key='l_range_fin_input')
        with c3:
            st.session_state.l_step = st.number_input("λ Step (nm, Optim Grid)", min_value=0.01, value=10.0, step=0.1, format="%.2f", key='l_step_input')
            st.caption("Plot uses λ Step / 10")


        c1, c2, c3 = st.columns(3)
        with c1:
             # Use text_input for params that come from preset, allowing manual override
            st.session_state.n_samples = st.text_input("N Samples (Sobol)", value=selected_params['n_samples'], key='n_samples_input')
        with c2:
            st.session_state.p_best = st.text_input("P Starts (L-BFGS-B)", value=selected_params['p_best'], key='p_best_input')
        with c3:
            st.session_state.n_passes = st.text_input("Optim Passes", value=selected_params['n_passes'], key='n_passes_input')

        st.session_state.scaling_nm = st.number_input("Scaling (nm, Pass 2+)", min_value=0.0, value=10.0, step=0.1, format="%.2f", key='scaling_nm_input')


    with st.expander("Spectral Target (Optimization on Transmittance T)", expanded=True):
        st.caption("Define target transmittance ramps (T vs λ). Optimization minimizes MSE against these targets.")
        # Initialize targets in session state if not present
        if 'targets' not in st.session_state:
            st.session_state.targets = copy.deepcopy(default_targets) # Use deepcopy

        headers = st.columns([0.5, 0.5, 1, 1, 1, 1])
        headers[0].markdown("**Enable**")
        headers[1].markdown("**Zone**")
        headers[2].markdown("**λ min (nm)**")
        headers[3].markdown("**λ max (nm)**")
        headers[4].markdown("**T @ λmin**")
        headers[5].markdown("**T @ λmax**")

        active_target_list = [] # Build this on the fly from UI state
        for i in range(len(st.session_state.targets)):
            cols = st.columns([0.5, 0.5, 1, 1, 1, 1])
            target_state = st.session_state.targets[i] # Get current state for this target

            # Use widget values directly, update session state internally via keys
            enabled = cols[0].checkbox("", value=target_state['enabled'], key=f"target_{i}_enabled")
            cols[1].markdown(f"**{i+1}**")
            l_min_str = cols[2].text_input("", value=target_state['min'], key=f"target_{i}_min", label_visibility="collapsed")
            l_max_str = cols[3].text_input("", value=target_state['max'], key=f"target_{i}_max", label_visibility="collapsed")
            t_min_str = cols[4].text_input("", value=target_state['target_min'], key=f"target_{i}_tmin", label_visibility="collapsed")
            t_max_str = cols[5].text_input("", value=target_state['target_max'], key=f"target_{i}_tmax", label_visibility="collapsed")

            # Update session state based on widgets *after* they are drawn
            st.session_state.targets[i]['enabled'] = enabled
            st.session_state.targets[i]['min'] = l_min_str
            st.session_state.targets[i]['max'] = l_max_str
            st.session_state.targets[i]['target_min'] = t_min_str
            st.session_state.targets[i]['target_max'] = t_max_str


# --- Action Buttons and Status ---
st.markdown("---")
status_placeholder = st.empty() # For displaying status messages
progress_placeholder = st.empty() # For displaying progress bar

action_cols = st.columns([1, 1, 1.5, 1.5, 1]) # Adjust widths as needed

calc_button = action_cols[0].button("Calculate Nominal", key="calc_nom", use_container_width=True)
opt_button = action_cols[1].button("Optimize N Passes", key="opt", use_container_width=True)

# Layer removal section
with action_cols[2]:
    remove_layer_enabled = (st.session_state.current_optimized_ep is not None and
                           len(st.session_state.current_optimized_ep) > 2 and
                           st.session_state.optimization_ran_since_nominal_change)
    remove_button = st.button("Remove Thinnest Layer", key="remove_thin", disabled=not remove_layer_enabled, use_container_width=True)
    # Display thinnest layer info next to button
    st.info(f"Thinnest ≥ {MIN_THICKNESS_PHYS_NM}nm: {st.session_state.thinnest_layer_display}")


# Set nominal section
with action_cols[3]:
    set_nominal_enabled = (st.session_state.current_optimized_ep is not None and
                           st.session_state.optimization_ran_since_nominal_change)
    set_nominal_button = st.button("Set Current as Nominal", key="set_nom", disabled=not set_nominal_enabled, use_container_width=True)
    # Display layer count info
    st.info(st.session_state.num_layers_display)

log_expander = st.expander("Log Messages")
with log_expander:
    if st.button("Clear Log", key="clear_log_btn"):
        clear_log()
        st.rerun() # Rerun to reflect cleared log immediately
    # Display log messages - join list into a single string
    log_content = "\n".join(st.session_state.log_messages)
    st.text_area("Log", value=log_content, height=300, key="log_display_area", disabled=True)

# --- Plot Area ---
st.markdown("---")
st.subheader("Results")
plot_placeholder = st.empty()

# --- Helper Functions dependent on Streamlit state ---

# streamlit_app.py
# ... (Keep all the previous Streamlit code: imports, config, helpers, core logic, GUI definition up to plot_placeholder) ...

# --- Helper Functions dependent on Streamlit state ---

def get_active_targets_from_state():
    """Reads target definitions from st.session_state and validates them."""
    active_targets = []
    if 'targets' not in st.session_state:
        log_message("ERROR: Target definitions not found in session state.")
        st.error("Target definitions not found in session state.")
        return None

    overall_lambda_min = None
    overall_lambda_max = None

    for i, target_def in enumerate(st.session_state.targets):
        if target_def.get('enabled', False): # Check if key exists and is True
            try:
                l_min_str = target_def.get('min', '')
                l_max_str = target_def.get('max', '')
                t_min_str = target_def.get('target_min', '')
                t_max_str = target_def.get('target_max', '')

                if not all([l_min_str, l_max_str, t_min_str, t_max_str]):
                    raise ValueError(f"Target Zone {i+1}: Required fields missing.")

                l_min = float(l_min_str)
                l_max = float(l_max_str)
                t_min = float(t_min_str)
                t_max = float(t_max_str)

                if l_max < l_min: raise ValueError(f"Target Zone {i+1}: λ max ({l_max}) must be >= λ min ({l_min}).")
                if not (0.0 <= t_min <= 1.0): raise ValueError(f"Target Zone {i+1}: Target T @ min ({t_min}) must be between 0 and 1.")
                if not (0.0 <= t_max <= 1.0): raise ValueError(f"Target Zone {i+1}: Target T @ max ({t_max}) must be between 0 and 1.")
                if l_min <=0 or l_max <=0: raise ValueError(f"Target Zone {i+1}: Wavelengths ({l_min}, {l_max}) must be > 0.")

                active_targets.append({'min': l_min, 'max': l_max, 'target_min': t_min, 'target_max': t_max})

                # Track overall range covered by *active* targets
                if overall_lambda_min is None or l_min < overall_lambda_min:
                    overall_lambda_min = l_min
                if overall_lambda_max is None or l_max > overall_lambda_max:
                    overall_lambda_max = l_max

            except (ValueError, TypeError) as e:
                log_message(f"ERROR Spectral Target Configuration {i+1}: {e}")
                st.error(f"Error in Spectral Target Zone {i+1}: {e}")
                return None # Indicate validation failure

    # Optional: Compare overall target range with calculation range
    try:
        # Use the keys assigned to the widgets
        calc_l_min = float(st.session_state.l_range_deb_input)
        calc_l_max = float(st.session_state.l_range_fin_input)
        if overall_lambda_min is not None and overall_lambda_max is not None:
            # Check if calculation range fully covers the active target range
            if overall_lambda_min < calc_l_min or overall_lambda_max > calc_l_max:
                st.warning(f"Calculation range [{calc_l_min:.1f}-{calc_l_max:.1f} nm] does not fully cover active target range [{overall_lambda_min:.1f}-{overall_lambda_max:.1f} nm]. Results might be suboptimal.")
            # Suggestion logic could be added here if desired
    except (ValueError, KeyError): # Catch potential errors if inputs are invalid or keys change
        st.warning("Could not validate target range against calculation range due to invalid/missing range inputs.")
    except Exception as e:
        st.warning(f"Error during target/calculation range check: {e}")

    return active_targets

def _validate_physical_inputs_from_state(require_optim_params=True):
    """Reads and validates physical parameters from st.session_state using widget keys."""
    values = {}

    # Map internal logic keys to session state keys (widget keys) and types
    field_map = {
        'nH_r': ('nH_r', float), 'nH_i': ('nH_i', float),
        'nL_r': ('nL_r', float), 'nL_i': ('nL_i', float),
        'nSub': ('nSub', float), 'l0': ('l0_input', float),
        'l_range_deb': ('l_range_deb_input', float),
        'l_range_fin': ('l_range_fin_input', float),
        'l_step': ('l_step_input', float),
        'scaling_nm': ('scaling_nm_input', float),
        'emp_str': ('emp_str_input', str) # QWOT string is special
    }
    if require_optim_params:
        field_map.update({
            'n_samples': ('n_samples_input', int),
            'p_best': ('p_best_input', int),
            'n_passes': ('n_passes_input', int),
        })

    # Retrieve and validate each required field
    for target_key, (state_key, field_type) in field_map.items():
        if state_key not in st.session_state:
             raise ValueError(f"GUI State Error: Input key '{state_key}' not found in session state. App might need restart.")
        raw_val = st.session_state[state_key]

        try:
            if field_type == str:
                values[target_key] = str(raw_val).strip() # Store as string, strip whitespace
            else:
                values[target_key] = field_type(raw_val) # Attempt conversion

            # --- Specific Value Constraints ---
            if target_key in ['n_samples', 'p_best', 'n_passes'] and values[target_key] < 1:
                raise ValueError(f"Parameter '{target_key}' must be >= 1 (value: {values[target_key]}).")
            if target_key == 'scaling_nm' and values[target_key] < 0:
                 raise ValueError(f"Parameter '{target_key}' must be >= 0 (value: {values[target_key]}).")
            if target_key == 'l_step' and values[target_key] <= 0:
                raise ValueError("λ Step must be > 0.")
            if target_key == 'l_range_deb' and values[target_key] <= 0:
                raise ValueError("λ Start must be > 0.")
            if target_key == 'l0' and values[target_key] <= 0:
                raise ValueError("Centering λ must be > 0.")
            if target_key == 'nH_r' and values[target_key] <= 0:
                 raise ValueError("Real part nH_r must be > 0.")
            if target_key == 'nL_r' and values[target_key] <= 0:
                 raise ValueError("Real part nL_r must be > 0.")
            if target_key == 'nH_i' and values[target_key] < 0:
                 raise ValueError("Imaginary part nH_i (k) must be >= 0.")
            if target_key == 'nL_i' and values[target_key] < 0:
                 raise ValueError("Imaginary part nL_i (k) must be >= 0.")
            if target_key == 'nSub' and values[target_key] <= 0:
                raise ValueError("Substrate index nSub must be > 0.")

        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value for '{state_key}': '{raw_val}'. Expected type: {field_type.__name__}. Error: {e}")

    # --- Cross-Field Validations ---
    if 'l_range_fin' in values and 'l_range_deb' in values and values['l_range_fin'] < values['l_range_deb']:
        raise ValueError(f"λ End ({values['l_range_fin']}) must be >= λ Start ({values['l_range_deb']}).")

    if require_optim_params and 'p_best' in values and 'n_samples' in values and values['p_best'] > values['n_samples']:
        raise ValueError(f"P Starts (p_best: {values['p_best']}) must be <= N Samples (n_samples: {values['n_samples']}).")

    return values


def _prepare_calculation_data_st(inputs, ep_vector_to_use=None):
    """Prepares data using validated inputs from state."""
    nH = inputs['nH_r'] + 1j * inputs['nH_i']
    nL = inputs['nL_r'] + 1j * inputs['nL_i']
    nSub_c = inputs['nSub'] + 0j
    l_step_gui = inputs['l_step'] # This is for optim grid now

    # Plotting grid (finer)
    l_step_plot = max(0.01, l_step_gui / 10.0) # Use 1/10th of optim step for plotting
    l_vec_plot = np.arange(inputs['l_range_deb'], inputs['l_range_fin'] + l_step_plot/2.0, l_step_plot)
    if not l_vec_plot.size: raise ValueError("Spectral range/step yields no calculation points for plotting.")

    # Determine the actual ep_vector to use
    if ep_vector_to_use is not None:
        ep_actual_orig = np.asarray(ep_vector_to_use, dtype=np.float64)
        log_message("  Using provided ep_vector for calculation.")
    else: # Get nominal from QWOT string
        log_message("  Calculating nominal ep_vector from QWOT string.")
        try:
             # Pass complex nH, nL directly to get_initial_ep, it uses real parts inside
             ep_actual_orig, _ = get_initial_ep(inputs['emp_str'], inputs['l0'], nH, nL)
        except ValueError as e:
             raise ValueError(f"Error processing nominal QWOT: {e}")


    # Validate and sanitize ep_actual_orig
    if ep_actual_orig.size == 0:
        # Check if QWOT string was actually empty or just yielded no layers
        if inputs['emp_str'].strip():
            raise ValueError("Nominal QWOT provided but resulted in an empty structure.")
        else:
            log_message("Empty structure (no QWOT provided). Calculating R/T for bare substrate.")
            # ep_actual_orig remains empty, which is handled by calculate_RT_from_ep

    elif not np.all(np.isfinite(ep_actual_orig)):
        ep_corrected = np.nan_to_num(ep_actual_orig, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.array_equal(ep_actual_orig, ep_corrected):
            log_message("WARNING: Original thicknesses contained NaN/inf, replaced with 0 for calculation.")
            ep_actual_orig = ep_corrected # Keep corrected version for display maybe? Or keep original non-finite? Let's keep corrected.
        # Double check after correction
        if not np.all(np.isfinite(ep_actual_orig)):
            raise ValueError(f"Thickness vector contains non-finite values that cannot be corrected: {ep_actual_orig}")

    # Apply minimum physical thickness constraint *for calculation*
    # We do this *after* getting the initial vector, just before calculating R/T
    if ep_actual_orig.size > 0:
        ep_actual_calc = np.maximum(ep_actual_orig, MIN_THICKNESS_PHYS_NM)
        num_below = np.sum(ep_actual_orig < MIN_THICKNESS_PHYS_NM)
        # Optionally log if clamping occurs, but can be noisy
        # if num_below > 0:
        #     log_message(f"  (Calculation clamped {num_below} thickness(es) to {MIN_THICKNESS_PHYS_NM} nm)")
    else:
        ep_actual_calc = ep_actual_orig # Use as is if empty

    # Return both calculation-ready and original vectors
    return nH, nL, nSub_c, l_vec_plot, ep_actual_calc, ep_actual_orig


def update_display_info(ep_vector_source=None):
    """Updates layer count and thinnest layer info in session state for display."""
    num_layers = 0
    prefix = "Layers (Nominal): "
    ep_for_thin_check = None # Vector to use for finding thinnest layer

    # Determine source based on current app state
    is_optimized = st.session_state.get('optimization_ran_since_nominal_change', False)
    current_opt_ep = st.session_state.get('current_optimized_ep')

    if is_optimized and current_opt_ep is not None:
        num_layers = len(current_opt_ep)
        prefix = "Layers (Optimized): "
        ep_for_thin_check = current_opt_ep
    elif ep_vector_source is not None: # Use provided source (e.g., from nominal calc)
        num_layers = len(ep_vector_source)
        prefix = "Layers (Nominal): "
        ep_for_thin_check = ep_vector_source
    else: # Fallback: try parsing QWOT string length if nothing else is available
        try:
            emp_str = st.session_state.get('emp_str_input', '')
            emp_list = [item for item in emp_str.split(',') if item.strip()]
            num_layers = len(emp_list)
            # Cannot determine thinnest layer without calculating ep vector here
            ep_for_thin_check = None # Signal no vector available for thin check
        except Exception:
            num_layers = "Error"
        prefix = "Layers (Nominal): "


    st.session_state.num_layers_display = f"{prefix}{num_layers}"

    # Update thinnest layer display based on the determined vector
    min_thickness_str = "- nm"
    if ep_for_thin_check is not None and len(ep_for_thin_check) > 0:
        # Find thinnest layer that is >= MIN_THICKNESS_PHYS_NM
        valid_thicknesses = ep_for_thin_check[ep_for_thin_check >= MIN_THICKNESS_PHYS_NM]
        if valid_thicknesses.size > 0:
            min_thickness = np.min(valid_thicknesses)
            min_thickness_str = f"{min_thickness:.3f} nm"
        else:
            min_thickness_str = f"None ≥ {MIN_THICKNESS_PHYS_NM} nm"


    st.session_state.thinnest_layer_display = min_thickness_str

# --- run_optimization_process Modification ---
# Modify run_optimization_process to accept status/progress placeholders
# And call hooks within _run_sobol_evaluation and _run_parallel_local_search if needed
# (Assuming the existing _run_sobol/_run_parallel signatures are updated or wrappers are used)
# For simplicity, we'll pass the placeholders directly and update status within run_optimization_process

def run_optimization_process(inputs, active_targets,
                             ep_nominal_for_cost, # Keep original nominal for reference if needed
                             p_best_this_pass,
                             ep_reference_for_sobol=None, # Best from previous pass (or None for Pass 1)
                             current_scaling=None, # Absolute scaling value for Pass 2+
                             run_number=1, total_passes=1,
                             progress_bar_hook=None, # Placeholder for progress bar
                             status_placeholder=None, # Placeholder for status text
                             current_best_mse=np.inf, # Pass current global best for status updates
                             current_best_layers=0):

    global MIN_THICKNESS_PHYS_NM
    run_id_str = f"Pass {run_number}/{total_passes}"
    log_with_elapsed_time("\n" + "*"*10 + f" STARTING OPTIMIZATION - {run_id_str} " + "*"*10)
    start_time_run = time.time()

    n_samples = inputs['n_samples']
    nH = inputs['nH_r'] + 1j * inputs['nH_i']
    nL = inputs['nL_r'] + 1j * inputs['nL_i']
    nSub_c = inputs['nSub'] + 0j
    l_step_gui = inputs['l_step']
    l_min_overall = inputs['l_range_deb']
    l_max_overall = inputs['l_range_fin']

    # --- Prepare for Sobol ---
    num_layers = 0
    lower_bounds = np.array([])
    upper_bounds = np.array([])

    pass_status_prefix = f"Status: {run_id_str}"
    status_suffix = f"| Best MSE: {current_best_mse:.3e} | Layers: {current_best_layers}"

    if status_placeholder: status_placeholder.info(f"{pass_status_prefix} - Phase 1 (Sobol Sampling) {status_suffix}")
    # if progress_bar_hook: progress_bar_hook(0.0) # Hook expects 0-1 for phase progress

    if ep_reference_for_sobol is None: # Pass 1: Relative scaling around nominal
        ep_ref_pass1 = ep_nominal_for_cost
        if ep_ref_pass1 is None or ep_ref_pass1.size == 0: raise ValueError(f"Nominal thickness vector ({run_id_str}) is empty.")
        num_layers = len(ep_ref_pass1)
        # Ensure reference respects minimum thickness before scaling
        ep_ref_pass1 = np.maximum(MIN_THICKNESS_PHYS_NM, ep_ref_pass1)
        scale_lower = 0.1; scale_upper = 2.0 # Relative factors
        lower_bounds = np.maximum(MIN_THICKNESS_PHYS_NM, ep_ref_pass1 * scale_lower)
        upper_bounds = ep_ref_pass1 * scale_upper
        upper_bounds = np.maximum(upper_bounds, lower_bounds + 0.1) # Ensure upper > lower
        log_with_elapsed_time(f"  Using relative scaling [{scale_lower:.2f}, {scale_upper:.2f}] around nominal for Sobol ({run_id_str}).")
    else: # Pass 2+: Absolute scaling around previous best
        ep_ref = ep_reference_for_sobol
        if ep_ref is None or ep_ref.size == 0: raise ValueError(f"Reference thickness vector ({run_id_str}) is empty for absolute scaling.")
        if current_scaling is None or current_scaling <=0: raise ValueError(f"Invalid current_scaling value ({current_scaling}) for absolute Sobol ({run_id_str}).")
        num_layers = len(ep_ref)
        # Ensure reference respects minimum thickness
        ep_ref = np.maximum(MIN_THICKNESS_PHYS_NM, ep_ref)
        lower_bounds = np.maximum(MIN_THICKNESS_PHYS_NM, ep_ref - current_scaling)
        upper_bounds = ep_ref + current_scaling
        upper_bounds = np.maximum(upper_bounds, lower_bounds + 0.1) # Ensure upper > lower
        log_with_elapsed_time(f"  Using absolute scaling +/- {current_scaling:.3f} nm (min {MIN_THICKNESS_PHYS_NM:.3f} nm) around previous best for Sobol ({run_id_str}).")

    if num_layers == 0: raise ValueError(f"Could not determine number of layers for Sobol ({run_id_str}).")

    # --- Prepare Optimization Grid and Cost Function ---
    if l_min_overall <= 0 or l_max_overall <=0: raise ValueError("Wavelength range limits must be positive.")
    if l_max_overall <= l_min_overall: raise ValueError("Wavelength range max must be greater than min.")
    num_points_approx = max(2, int(np.round((l_max_overall - l_min_overall) / l_step_gui)) + 1)
    l_vec_optim = np.geomspace(l_min_overall, l_max_overall, num_points_approx)
    l_vec_optim = l_vec_optim[(l_vec_optim > 0) & np.isfinite(l_vec_optim)]
    if not l_vec_optim.size: raise ValueError("Failed to generate valid geomspace optimization wavelength vector.")
    log_with_elapsed_time(f"  Generated {len(l_vec_optim)} optimization points geometrically for cost evaluation.")

    # Args tuple for the cost function (Note: ep_initial_for_penalty removed from penalized func)
    args_for_cost_tuple = (nH, nL, nSub_c, l_vec_optim, active_targets, MIN_THICKNESS_PHYS_NM)
    # Partial function for mapping (doesn't need penalty reference anymore)
    cost_function_partial_map = partial(calculate_mse_for_optimization_penalized,
                                         nH=nH, nL=nL, nSub=nSub_c, l_vec_optim=l_vec_optim,
                                         active_targets=active_targets,
                                         min_thickness_phys_nm=MIN_THICKNESS_PHYS_NM)

    # --- Run Sobol Evaluation ---
    valid_initial_results_p1 = _run_sobol_evaluation(
        num_layers, n_samples, lower_bounds, upper_bounds,
        cost_function_partial_map, MIN_THICKNESS_PHYS_NM,
        phase_name=f"{run_id_str} Phase 1",
        # progress_hook=progress_bar_hook # Pass hook if _run_sobol supports it
    )
    # if progress_bar_hook: progress_bar_hook(0.33) # Approx progress after Sobol eval

    # --- Phase 1bis: Refine Top P Starts (Only for Pass 1) ---
    selected_starts = []
    selected_costs = []
    top_p_results_combined = [] # Holds the final starting points for L-BFGS-B

    if run_number == 1:
        if status_placeholder: status_placeholder.info(f"{pass_status_prefix} - Phase 1bis (Refining Starts) {status_suffix}")

        log_with_elapsed_time(f"\n--- {run_id_str} Phase 1bis: Refining Top P Starts ---")
        num_to_select_p1 = min(p_best_this_pass, len(valid_initial_results_p1))

        if num_to_select_p1 == 0:
            log_with_elapsed_time("WARNING: Phase 1 yielded no valid starting points. Skipping refinement.")
            top_p_results_combined = [] # No points to refine
        else:
            top_p_results_p1 = valid_initial_results_p1[:num_to_select_p1]
            top_p_starts_p1 = [ep for cost, ep in top_p_results_p1]

            # Define refinement parameters
            phase1bis_scaling = 10.0 # nm - explore neighborhood
            # Use N samples / P best for refinement sampling density
            n_samples_per_point_raw = max(1, n_samples // p_best_this_pass) if p_best_this_pass > 0 else n_samples
            n_samples_per_point = upper_power_of_2(n_samples_per_point_raw) # Use power of 2 for Sobol efficiency

            log_with_elapsed_time(f"  Generating {n_samples_per_point} samples around each of the top {num_to_select_p1} points (scaling +/- {phase1bis_scaling} nm).")

            phase1bis_candidates = []
            for i, ep_start in enumerate(top_p_starts_p1):
                # Define bounds for refinement around this point
                lower_bounds_1bis = np.maximum(MIN_THICKNESS_PHYS_NM, ep_start - phase1bis_scaling)
                upper_bounds_1bis = ep_start + phase1bis_scaling
                upper_bounds_1bis = np.maximum(upper_bounds_1bis, lower_bounds_1bis + 0.1) # Ensure upper > lower

                sampler_1bis = qmc.Sobol(d=num_layers, scramble=True, seed=int(time.time()) + i) # Seed for variety
                points_unit_1bis = sampler_1bis.random(n=n_samples_per_point)
                new_candidates = qmc.scale(points_unit_1bis, lower_bounds_1bis, upper_bounds_1bis)
                # Clamp after scaling
                new_candidates = [np.maximum(MIN_THICKNESS_PHYS_NM, cand) for cand in new_candidates]
                phase1bis_candidates.extend(new_candidates)

            log_with_elapsed_time(f"  Generated {len(phase1bis_candidates)} total candidates for Phase 1bis evaluation.")

            # Evaluate these refined candidates
            costs_1bis = []
            results_1bis_raw_pairs = []
            num_workers_1bis = min(len(phase1bis_candidates), os.cpu_count())
            log_with_elapsed_time(f"  Evaluating cost for {len(phase1bis_candidates)} Phase 1bis candidates (max {num_workers_1bis} workers)...")
            eval_start_pool_1bis = time.time()
            try:
                if phase1bis_candidates:
                    with multiprocessing.Pool(processes=num_workers_1bis) as pool:
                        costs_1bis = pool.map(cost_function_partial_map, phase1bis_candidates)
                    eval_pool_time_1bis = time.time() - eval_start_pool_1bis
                    log_with_elapsed_time(f"  Parallel evaluation (Phase 1bis) finished in {eval_pool_time_1bis:.2f}s.")
                    if len(costs_1bis) != len(phase1bis_candidates): costs_1bis = costs_1bis[:len(phase1bis_candidates)]
                    results_1bis_raw_pairs = list(zip(costs_1bis, phase1bis_candidates))
                else: results_1bis_raw_pairs = []
            except Exception as e_pool_1bis:
                log_with_elapsed_time(f"  ERROR during parallel evaluation (Phase 1bis): {e_pool_1bis}")
                # Fallback to sequential is possible but skipped here for brevity
                results_1bis_raw_pairs = [] # Assume failure if pool fails

            # Filter and combine results
            valid_results_1bis = [(c, p) for c, p in results_1bis_raw_pairs if np.isfinite(c) and c < 1e9]
            num_invalid_1bis = len(results_1bis_raw_pairs) - len(valid_results_1bis)
            if num_invalid_1bis > 0: log_with_elapsed_time(f"  Filtering (Phase 1bis): {num_invalid_1bis} invalid costs discarded.")

            # Combine original top P and the new valid refined points
            combined_results = top_p_results_p1 + valid_results_1bis
            log_with_elapsed_time(f"  Combined Phase 1 ({len(top_p_results_p1)}) and Phase 1bis ({len(valid_results_1bis)}) results: {len(combined_results)} total.")

            if not combined_results:
                log_with_elapsed_time("WARNING: No valid results found after combining Phase 1 and Phase 1bis.")
                top_p_results_combined = []
            else:
                combined_results.sort(key=lambda x: x[0]) # Sort all combined results
                num_to_select_final = min(p_best_this_pass, len(combined_results)) # Select the best P from combined pool
                log_with_elapsed_time(f"  Selecting final top {num_to_select_final} points from combined results for Phase 2.")
                top_p_results_combined = combined_results[:num_to_select_final]

        log_with_elapsed_time(f"--- End {run_id_str} Phase 1bis ---")
        # if progress_bar_hook: progress_bar_hook(0.66) # Approx progress after refinement

    else: # Not Pass 1 - just take top P from initial Sobol evaluation
        num_to_select = min(p_best_this_pass, len(valid_initial_results_p1))
        log_with_elapsed_time(f"  Selecting top {num_to_select} points from Phase 1 for Phase 2.")
        top_p_results_combined = valid_initial_results_p1[:num_to_select]
        # if progress_bar_hook: progress_bar_hook(0.5) # Approx progress after Sobol eval (no refine phase)


    # Prepare starting points for L-BFGS-B
    selected_starts = [ep for cost, ep in top_p_results_combined]
    selected_costs = [cost for cost, ep in top_p_results_combined]

    # --- Phase 2: Local Search ---
    if status_placeholder: status_placeholder.info(f"{pass_status_prefix} - Phase 2 (Local Search) {status_suffix}")

    overall_best_ep = None
    overall_best_cost = np.inf
    overall_best_result_obj = OptimizeResult(x=None, fun=np.inf, success=False, message="Optimization did not yield a valid result yet.")

    if not selected_starts:
        log_with_elapsed_time(f"WARNING: No starting points available for Phase 2 ({run_id_str}). Skipping.")
        # Try to use the reference EP as a fallback result if available
        if ep_reference_for_sobol is not None:
             overall_best_ep = ep_reference_for_sobol.copy()
             try: # Calculate its cost
                 overall_best_cost = cost_function_partial_map(overall_best_ep)
             except Exception: overall_best_cost = np.inf
        elif ep_nominal_for_cost is not None: # Or the initial nominal
             overall_best_ep = ep_nominal_for_cost.copy()
             try: overall_best_cost = cost_function_partial_map(overall_best_ep)
             except Exception: overall_best_cost = np.inf
        else: # No fallback possible
            overall_best_ep = None
            overall_best_cost = np.inf
        # Create a placeholder result object
        overall_best_result_obj = OptimizeResult(x=overall_best_ep, fun=overall_best_cost, success=False, message=f"Phase 2 skipped - no starts ({run_id_str})", nit=0)

    else:
        # Initial best for processing is the best point found *before* LBFGSB
        initial_best_ep_for_processing = selected_starts[0].copy()
        initial_best_cost_for_processing = selected_costs[0]

        log_with_elapsed_time(f"  Top {len(selected_starts)} points selected for Phase 2:")
        # Limit logging details
        for i in range(min(5, len(selected_starts))): # Log first few
            cost_str = f"{selected_costs[i]:.3e}" if np.isfinite(selected_costs[i]) else "inf"
            log_with_elapsed_time(f"    Point {i+1}: Cost={cost_str}")
        if len(selected_starts) > 5: log_with_elapsed_time("    ...")

        # Define bounds for L-BFGS-B
        lbfgsb_bounds = [(MIN_THICKNESS_PHYS_NM, None)] * num_layers

        # Run local search in parallel
        local_results_raw = _run_parallel_local_search(selected_starts,
                                                        calculate_mse_for_optimization_penalized, # The cost function
                                                        args_for_cost_tuple,
                                                        lbfgsb_bounds,
                                                        MIN_THICKNESS_PHYS_NM,
                                                        # progress_hook=progress_bar_hook # Pass hook if _run_parallel supports it
                                                        )

        if status_placeholder: status_placeholder.info(f"{pass_status_prefix} - Processing L-BFGS-B results... {status_suffix}")

        # Process the results from the workers
        overall_best_ep, overall_best_cost, overall_best_result_obj = _process_optimization_results(
            local_results_raw, initial_best_ep_for_processing, initial_best_cost_for_processing
        )

    # if progress_bar_hook: progress_bar_hook(1.0) # Mark phase as complete

    # --- Finalize Pass ---
    run_time_pass = time.time() - start_time_run
    log_with_elapsed_time(f"--- End Optimization {run_id_str} in {run_time_pass:.2f}s ---")
    log_with_elapsed_time(f"Best cost found this pass ({run_id_str}): {overall_best_cost:.3e}")
    if overall_best_ep is not None and overall_best_ep.size > 0:
        log_with_elapsed_time(f"Thicknesses this pass ({run_id_str}, {len(overall_best_ep)} layers): {[f'{th:.3f}' for th in overall_best_ep[:10]]}{'...' if len(overall_best_ep)>10 else ''}")
    else:
        log_with_elapsed_time(f"WARNING: No valid thickness vector found for {run_id_str}.")

    return overall_best_ep, overall_best_cost, overall_best_result_obj


# --- Button Click Handlers (Continued) ---

if calc_button:
    # Reset optimized state when calculating nominal
    st.session_state.current_optimized_ep = None
    st.session_state.optimization_ran_since_nominal_change = False
    st.session_state.optimized_qwot_display = ""
    run_calculation_st(ep_vector_to_use=None, is_optimized=False)
    st.rerun() # Update button states etc.

if opt_button:
    st.session_state.optim_start_time = time.time() # Record start time
    clear_log() # Clear log for new optimization run
    log_with_elapsed_time(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Starting Multi-Pass Optimization Process ===")

    # Reset state before optimization
    st.session_state.current_optimized_ep = None
    st.session_state.optimization_ran_since_nominal_change = False
    st.session_state.optimized_qwot_display = ""
    st.session_state.current_status = "Status: Initializing Optimization..."
    status_placeholder.info(st.session_state.current_status)
    progress_bar = progress_placeholder.progress(0) # Show progress bar

    min_scaling_nm = 0.1 # Minimum scaling for later passes
    ep_nominal_glob = None
    overall_best_ep_final = None
    overall_best_cost_final = np.inf
    overall_best_layers = 0
    initial_nominal_cost = np.inf
    initial_nominal_layers = 0
    final_successful_result_obj = None
    ep_ref_for_next_pass = None
    n_passes = 0
    optimization_successful = False
    auto_removed_count = 0 # Count auto-removed layers

    try:
        inputs = _validate_physical_inputs_from_state(require_optim_params=True)
        n_samples = inputs['n_samples']
        initial_p_best = inputs['p_best']
        n_passes = inputs['n_passes']
        initial_scaling_nm = inputs['scaling_nm']

        # Total steps for progress: Pass 1 (Sobol, [Refine], LBFGSB Proc) = 3 steps, Other passes (Sobol, LBFGSB Proc) = 2 steps
        # Plus 1 for final processing/plotting/auto-remove
        progress_max_steps = (3 if n_passes >= 1 else 0) + max(0, (n_passes - 1) * 2) + 1
        current_progress_step = 0 # Track current progress step index

        active_targets = get_active_targets_from_state()
        if active_targets is None: raise ValueError("Failed to retrieve/validate spectral targets.")
        if not active_targets: raise ValueError("No active spectral targets defined. Optimization requires at least one target.")
        log_with_elapsed_time(f"{len(active_targets)} active target zone(s) found.")

        # Get nominal structure
        nH_complex_nom = inputs['nH_r'] + 1j*inputs['nH_i']
        nL_complex_nom = inputs['nL_r'] + 1j*inputs['nL_i']
        ep_nominal_glob, _ = get_initial_ep(inputs['emp_str'], inputs['l0'], nH_complex_nom, nL_complex_nom)

        if ep_nominal_glob.size == 0:
            raise ValueError("Initial nominal QWOT stack is empty or invalid.")
        initial_nominal_layers = len(ep_nominal_glob)

        # Initial best is the nominal structure
        last_successful_ep = ep_nominal_glob.copy()
        overall_best_ep_final = last_successful_ep.copy()
        ep_ref_for_next_pass = None # Start Pass 1 relative to nominal
        overall_best_layers = initial_nominal_layers

        # Calculate initial cost (for baseline and display)
        try:
            nSub_c = inputs['nSub'] + 0j
            l_min_overall = inputs['l_range_deb']
            l_max_overall = inputs['l_range_fin']
            l_step_gui = inputs['l_step']
            num_points_approx = max(2, int(np.round((l_max_overall - l_min_overall) / l_step_gui)) + 1)
            l_vec_optim_init = np.geomspace(l_min_overall, l_max_overall, num_points_approx)
            l_vec_optim_init = l_vec_optim_init[(l_vec_optim_init > 0) & np.isfinite(l_vec_optim_init)]
            if l_vec_optim_init.size == 0: raise ValueError("Failed to create optim vec for initial cost.")

            initial_nominal_cost = calculate_mse_for_optimization_penalized(
                 ep_nominal_glob, nH_complex_nom, nL_complex_nom, nSub_c, l_vec_optim_init, active_targets, MIN_THICKNESS_PHYS_NM
            )
            overall_best_cost_final = initial_nominal_cost if np.isfinite(initial_nominal_cost) else np.inf
            log_with_elapsed_time(f"Initial nominal cost: {overall_best_cost_final:.3e}, Layers: {initial_nominal_layers}")
            st.session_state.current_status = f"Status: Initial | Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
            status_placeholder.info(st.session_state.current_status)

        except Exception as e_init_cost:
            log_with_elapsed_time(f"Warning: Could not calculate initial nominal cost: {e_init_cost}")
            overall_best_cost_final = np.inf # Start with inf if baseline fails
            st.session_state.current_status = f"Status: Initial | Best MSE: N/A | Layers: {initial_nominal_layers}"
            status_placeholder.warning(st.session_state.current_status)


        # --- Optimization Passes Loop ---
        for pass_num in range(1, n_passes + 1):
            run_id_str = f"Pass {pass_num}/{n_passes}"
            pass_status_prefix = f"Status: {run_id_str}"
            status_suffix = f"| Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"

            # --- Update Status and Progress ---
            st.session_state.current_status = f"{pass_status_prefix} - Starting... {status_suffix}"
            status_placeholder.info(st.session_state.current_status)
            progress_bar.progress( current_progress_step / progress_max_steps )


            # Determine P_Starts and Scaling for this pass
            reduction_factor = 0.8 ** (pass_num - 1) # Decrease P for later passes
            p_best_for_this_pass = max(2, int(np.round(initial_p_best * reduction_factor)))

            current_scaling = None # For Sobol bounds calculation
            ep_ref_sobol = None
            if pass_num == 1:
                current_scaling = None # Use relative scaling based on nominal
                ep_ref_sobol = None # Signal run_optim_process to use nominal
            else:
                # Use absolute scaling based on previous best result
                scale_factor = 1.8**(pass_num - 2) # Decrease scaling for later passes (increase divisor)
                current_scaling = max(min_scaling_nm, initial_scaling_nm / scale_factor)
                ep_ref_sobol = ep_ref_for_next_pass # Reference is best from previous pass

            # --- Run the Optimization Pass ---
            # Progress Hook (simplified: just increments steps)
            def progress_update_hook(phase_fraction): # phase_fraction is 0 to 1 within a phase
                 # We don't have fine-grained phase steps here, just step after each major part
                 pass # Let the main loop handle step increments

            # Pass the actual placeholders/widgets for status/progress updates
            ep_this_pass, cost_this_pass, result_obj_this_pass = run_optimization_process(
                inputs=inputs, active_targets=active_targets,
                ep_nominal_for_cost=ep_nominal_glob,
                p_best_this_pass=p_best_for_this_pass,
                ep_reference_for_sobol=ep_ref_sobol, current_scaling=current_scaling,
                run_number=pass_num, total_passes=n_passes,
                progress_bar_hook=progress_bar, # Pass the actual progress bar widget/placeholder
                status_placeholder=status_placeholder, # Pass status placeholder
                current_best_mse=overall_best_cost_final,
                current_best_layers=overall_best_layers
            )

            # Increment progress step after pass completion
            current_progress_step += (3 if pass_num == 1 else 2)
            progress_bar.progress(current_progress_step / progress_max_steps)


            # --- Process Pass Results ---
            new_best_found_this_pass = False
            if ep_this_pass is not None and ep_this_pass.size > 0 and np.isfinite(cost_this_pass):
                if cost_this_pass < overall_best_cost_final:
                    log_with_elapsed_time(f"*** New overall best cost found in {run_id_str}: {cost_this_pass:.3e} (Previous: {overall_best_cost_final:.3e}) ***")
                    overall_best_cost_final = cost_this_pass
                    overall_best_ep_final = ep_this_pass.copy()
                    overall_best_layers = len(overall_best_ep_final)
                    final_successful_result_obj = result_obj_this_pass # Store result object of the best
                    new_best_found_this_pass = True
                # else: # Log if not improved (optional)
                    # log_with_elapsed_time(f"{run_id_str}: Cost {cost_this_pass:.3e} did not improve best ({overall_best_cost_final:.3e}).")

                # Reference for the *next* pass is the best found *so far*
                ep_ref_for_next_pass = overall_best_ep_final.copy()

                # Update status if new best found
                if new_best_found_this_pass:
                    status_suffix = f"| Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
                    st.session_state.current_status = f"{pass_status_prefix} - Completed. New Best! {status_suffix}"
                    status_placeholder.info(st.session_state.current_status) # Update status display
                else: # Pass completed but no improvement
                     st.session_state.current_status = f"{pass_status_prefix} - Completed. No improvement. {status_suffix}"
                     status_placeholder.info(st.session_state.current_status)

            else: # Pass failed to return a valid solution
                log_with_elapsed_time(f"ERROR: {run_id_str} did not return a valid solution (cost: {cost_this_pass}, ep: {ep_this_pass}).")
                st.session_state.current_status = f"{pass_status_prefix} - FAILED. {status_suffix}"
                status_placeholder.warning(st.session_state.current_status) # Show warning
                if overall_best_ep_final is not None:
                    log_with_elapsed_time("Continuing optimization using previous best result.")
                    ep_ref_for_next_pass = overall_best_ep_final.copy() # Use last known good
                else:
                    # Critical failure - cannot continue
                    raise RuntimeError(f"{run_id_str} failed and no prior successful result exists. Optimization aborted.")


        # --- Post-Optimization Processing (Auto Removal) ---
        current_progress_step +=1 # Increment step for post-processing
        progress_bar.progress(current_progress_step / progress_max_steps)
        st.session_state.current_status = f"Status: Post-processing (Auto-Remove)... | Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
        status_placeholder.info(st.session_state.current_status)


        if overall_best_ep_final is None:
             raise RuntimeError("Optimization finished, but no valid result found before post-processing.")

        log_with_elapsed_time("\n--- Checking for automatic thin layer removal (< 1.0 nm) ---")
        auto_removal_threshold = 1.0
        max_auto_removals = len(overall_best_ep_final) # Max attempts

        # Need cost function args for removal re-optimization
        temp_inputs_ar = _validate_physical_inputs_from_state(require_optim_params=False) # Get current params
        temp_active_targets_ar = get_active_targets_from_state()
        if temp_active_targets_ar is None: raise ValueError("Failed to get targets for auto-removal cost func.")

        temp_nH_ar = temp_inputs_ar['nH_r'] + 1j * temp_inputs_ar['nH_i']
        temp_nL_ar = temp_inputs_ar['nL_r'] + 1j * temp_inputs_ar['nL_i']
        temp_nSub_c_ar = temp_inputs_ar['nSub'] + 0j

        temp_l_min_ar = temp_inputs_ar['l_range_deb']; temp_l_max_ar = temp_inputs_ar['l_range_fin']
        temp_l_step_ar = temp_inputs_ar['l_step']
        temp_num_pts_ar = max(2, int(np.round((temp_l_max_ar - temp_l_min_ar) / temp_l_step_ar)) + 1)
        temp_l_vec_optim_ar = np.geomspace(temp_l_min_ar, temp_l_max_ar, temp_num_pts_ar)
        temp_l_vec_optim_ar = temp_l_vec_optim_ar[(temp_l_vec_optim_ar > 0) & np.isfinite(temp_l_vec_optim_ar)]
        if not temp_l_vec_optim_ar.size: raise ValueError("Failed to generate valid geomspace AR wavelength vector.")

        # Cost function args for auto-removal
        temp_args_for_cost_ar = (temp_nH_ar, temp_nL_ar, temp_nSub_c_ar, temp_l_vec_optim_ar, temp_active_targets_ar, MIN_THICKNESS_PHYS_NM)

        current_ep_for_removal = overall_best_ep_final.copy() # Work on a copy

        while auto_removed_count < max_auto_removals:
            if current_ep_for_removal is None or len(current_ep_for_removal) <= 1: # Need >1 layer to remove
                log_with_elapsed_time("  Structure too short for further auto-removal.")
                break

            # Find the index of the thinnest layer that is < threshold BUT >= physical min
            eligible_indices = np.where(
                (current_ep_for_removal >= MIN_THICKNESS_PHYS_NM) &
                (current_ep_for_removal < auto_removal_threshold)
            )[0]

            if eligible_indices.size > 0:
                # Find the minimum thickness among these eligible layers
                min_val_in_eligible = np.min(current_ep_for_removal[eligible_indices])
                # Find the first index that matches this minimum value (arbitrary choice if multiple)
                thinnest_below_threshold_idx = np.where(current_ep_for_removal == min_val_in_eligible)[0][0]
                thinnest_below_threshold_val = min_val_in_eligible

                log_with_elapsed_time(f"  Auto-removing layer {thinnest_below_threshold_idx + 1} (thickness {thinnest_below_threshold_val:.3f} nm < {auto_removal_threshold} nm)")
                status_suffix = f"| Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}" # Use last known best cost for status
                st.session_state.current_status = f"Status: Auto-removing layer {auto_removed_count + 1}... {status_suffix}"
                status_placeholder.info(st.session_state.current_status)

                new_ep, success, cost_after, removal_logs = perform_single_thin_layer_removal(
                    current_ep_for_removal, MIN_THICKNESS_PHYS_NM,
                    calculate_mse_for_optimization_penalized, temp_args_for_cost_ar,
                    log_prefix="    [Auto Removal] ",
                    target_layer_index=thinnest_below_threshold_idx # Target specific layer
                )
                for log_line in removal_logs: log_with_elapsed_time(log_line)

                # Check if removal was successful and structure changed
                if success and new_ep is not None and len(new_ep) < len(current_ep_for_removal):
                    current_ep_for_removal = new_ep.copy() # Update the structure for the next iteration
                    # Update overall best result with the auto-removed structure
                    overall_best_ep_final = new_ep.copy()
                    overall_best_cost_final = cost_after if np.isfinite(cost_after) else np.inf
                    overall_best_layers = len(overall_best_ep_final)
                    auto_removed_count += 1
                    log_with_elapsed_time(f"  Auto-removal successful. New cost: {overall_best_cost_final:.3e}, Layers: {overall_best_layers}")
                else:
                    log_with_elapsed_time("    Auto-removal/re-optimization failed or structure unchanged. Stopping auto-removal.")
                    break # Stop the while loop

            else: # No layers found below threshold
                log_with_elapsed_time(f"  No layers found below {auto_removal_threshold} nm (and >= {MIN_THICKNESS_PHYS_NM} nm).")
                break # Stop the while loop

        if auto_removed_count > 0:
             log_with_elapsed_time(f"--- Finished automatic removal ({auto_removed_count} layer(s) removed) ---")


        # --- Finalize Optimization ---
        optimization_successful = True
        st.session_state.current_optimized_ep = overall_best_ep_final.copy() if overall_best_ep_final is not None else None
        st.session_state.optimization_ran_since_nominal_change = True

        # Calculate and display final QWOT
        final_qwot_str = "QWOT Error"
        if overall_best_ep_final is not None and len(overall_best_ep_final) > 0:
             try:
                 l0_val = inputs['l0']; nH_r_val = inputs['nH_r']; nL_r_val = inputs['nL_r']
                 optimized_qwots = calculate_qwot_from_ep(overall_best_ep_final, l0_val, nH_r_val, nL_r_val)
                 if np.any(np.isnan(optimized_qwots)): final_qwot_str = "QWOT N/A (NaN)";
                 else: final_qwot_str = ", ".join([f"{q:.3f}" for q in optimized_qwots]);
             except Exception as qwot_calc_error: log_with_elapsed_time(f"Error calculating final QWOTs: {qwot_calc_error}")
        else: final_qwot_str = "N/A (Empty Structure)"
        st.session_state.optimized_qwot_display = final_qwot_str # Update state for display

        # --- Log Final Summary ---
        final_method_name = f"{n_passes}-Pass Opt"
        if auto_removed_count > 0: final_method_name += f" + {auto_removed_count} AutoRm"

        log_with_elapsed_time("\n" + "="*60)
        log_with_elapsed_time(f"--- Overall Optimization ({final_method_name}) Finished ---")
        log_with_elapsed_time(f"Best Final Cost Found (MSE): {overall_best_cost_final:.3e}")
        log_with_elapsed_time(f"Final number of layers: {overall_best_layers}")
        log_with_elapsed_time(f"Final Optimized QWOT (λ₀={inputs['l0']}nm): {final_qwot_str}")
        ep_str_list = [f"L{i+1}:{th:.4f}" for i, th in enumerate(overall_best_ep_final)] if overall_best_ep_final is not None else []
        log_with_elapsed_time(f"Final optimized thicknesses ({overall_best_layers} layers, nm): [{', '.join(ep_str_list)}]")
        # Log info from the best result object if available
        if final_successful_result_obj and isinstance(final_successful_result_obj, OptimizeResult):
             best_res_nit = getattr(final_successful_result_obj, 'nit', 'N/A')
             best_res_msg = getattr(final_successful_result_obj, 'message', 'N/A')
             if isinstance(best_res_msg, bytes): best_res_msg = best_res_msg.decode('utf-8', errors='ignore')
             log_with_elapsed_time(f"Best Result Info (From Best Pass): LBFGSB Iters={best_res_nit}, Msg='{best_res_msg}'")
        log_with_elapsed_time("="*60 + "\n")

        # Update final status and progress
        progress_bar.progress(1.0)
        final_status_text = f"Status: Opt Complete{' (+AutoRm)' if auto_removed_count>0 else ''} | MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
        st.session_state.current_status = final_status_text
        status_placeholder.success(st.session_state.current_status)

        # Run final calculation for plotting the best result
        run_calculation_st(ep_vector_to_use=overall_best_ep_final,
                            is_optimized=True, method_name=final_method_name)

    # --- Error Handling for Optimization ---
    except (ValueError, RuntimeError) as e:
        err_msg = f"ERROR (Input/Logic) during {n_passes}-Pass optimization: {e}"
        log_with_elapsed_time(err_msg)
        st.error(err_msg)
        st.session_state.current_status = f"Status: Optimization Failed (Input/Logic Error)"
        status_placeholder.error(st.session_state.current_status)
        optimization_successful = False
        st.session_state.current_optimized_ep = None
        st.session_state.optimization_ran_since_nominal_change = False
        st.session_state.optimized_qwot_display = "Error Input/Optim Logic"

    except Exception as e:
        err_msg = f"ERROR (Unexpected) during optimization: {type(e).__name__}: {e}"
        tb_msg = traceback.format_exc()
        log_with_elapsed_time(err_msg); log_with_elapsed_time(tb_msg)
        print(err_msg); print(tb_msg)
        st.error(f"{err_msg}. See log/console for details.")
        st.session_state.current_status = f"Status: Optimization Failed (Unexpected Error)"
        status_placeholder.error(st.session_state.current_status)
        optimization_successful = False
        st.session_state.current_optimized_ep = None
        st.session_state.optimization_ran_since_nominal_change = False
        st.session_state.optimized_qwot_display = "Optim Error Unexpected"

    finally:
        # Clean up progress bar and reset start time
        progress_placeholder.empty()
        st.session_state.optim_start_time = None
        # Update display info based on the final state
        update_display_info(st.session_state.current_optimized_ep if optimization_successful else None)
        # Rerun to update button states based on final optimization status
        st.rerun()


if remove_button:
    log_message("\n" + "-"*10 + " Attempting Manual Thin Layer Removal " + "-"*10)
    st.session_state.current_status = "Status: Removing thinnest layer..."
    status_placeholder.info(st.session_state.current_status)
    progress_bar = progress_placeholder.progress(0) # Show progress briefly

    current_ep = st.session_state.get('current_optimized_ep')
    optim_ran = st.session_state.get('optimization_ran_since_nominal_change', False)

    # Check if removal is possible (optimized state exists, >1 layer)
    if current_ep is None or not optim_ran or len(current_ep) <= 1:
        log_message("ERROR: No valid optimized structure with >= 2 layers available for removal.")
        st.error("No valid optimized structure (must have >= 2 layers) found to modify.")
        st.session_state.current_status = "Status: Removal Failed (No Structure/Not Optimized)"
        status_placeholder.error(st.session_state.current_status)
        progress_placeholder.empty()
    else:
        removal_successful = False
        try:
            progress_bar.progress(10)
            inputs = _validate_physical_inputs_from_state(require_optim_params=False)
            active_targets = get_active_targets_from_state()
            if active_targets is None: raise ValueError("Failed to retrieve/validate spectral targets for re-optimization.")
            if not active_targets: raise ValueError("No active spectral targets defined. Cannot re-optimize after removal.")

            # Prepare cost function arguments
            nH = inputs['nH_r'] + 1j * inputs['nH_i']
            nL = inputs['nL_r'] + 1j * inputs['nL_i']
            nSub_c = inputs['nSub'] + 0j
            l_min = inputs['l_range_deb']; l_max = inputs['l_range_fin']; l_step = inputs['l_step']
            num_pts = max(2, int(np.round((l_max - l_min) / l_step)) + 1)
            l_vec_optim = np.geomspace(l_min, l_max, num_pts)
            l_vec_optim = l_vec_optim[(l_vec_optim > 0) & np.isfinite(l_vec_optim)]
            if not l_vec_optim.size: raise ValueError("Failed to generate optim vector for removal.")

            args_for_cost_tuple = (nH, nL, nSub_c, l_vec_optim, active_targets, MIN_THICKNESS_PHYS_NM)
            progress_bar.progress(30)

            log_message("  Starting removal and re-optimization...")
            new_ep, success, final_cost, removal_logs = perform_single_thin_layer_removal(
                current_ep, MIN_THICKNESS_PHYS_NM,
                calculate_mse_for_optimization_penalized, args_for_cost_tuple,
                log_prefix="  [Manual Removal] "
                # target_layer_index=None # Let it find the thinnest by default
            )
            for log_line in removal_logs: log_message(log_line)
            progress_bar.progress(70)

            structure_actually_changed = (success and new_ep is not None and len(new_ep) < len(current_ep))

            if structure_actually_changed:
                log_message("  Layer removal successful. Updating structure and display.")
                st.session_state.current_optimized_ep = new_ep.copy()
                # Keep optimization_ran_since_nominal_change as True
                removal_successful = True

                # Calculate and display final QWOT for the new structure
                final_qwot_str = "QWOT Error"
                try:
                    l0_val = inputs['l0']; nH_r_val = inputs['nH_r']; nL_r_val = inputs['nL_r']
                    optimized_qwots = calculate_qwot_from_ep(new_ep, l0_val, nH_r_val, nL_r_val)
                    if np.any(np.isnan(optimized_qwots)): final_qwot_str = "QWOT N/A (NaN)";
                    else: final_qwot_str = ", ".join([f"{q:.3f}" for q in optimized_qwots]);
                except Exception as qwot_calc_error: log_message(f"Error calculating final QWOTs after removal: {qwot_calc_error}")
                st.session_state.optimized_qwot_display = final_qwot_str

                log_message(f"  Cost after removal/re-opt: {final_cost:.3e}")
                st.session_state.current_status = f"Status: Layer Removed | MSE: {final_cost:.3e} | Layers: {len(new_ep)}"
                status_placeholder.success(st.session_state.current_status)

                # Run calculation to plot the new result
                run_calculation_st(ep_vector_to_use=new_ep,
                                   is_optimized=True, # Still considered optimized
                                   method_name="Optimized (Post-Removal)")
                progress_bar.progress(100)

            else: # Removal failed or didn't change structure
                log_message("  No layer removed or structure unchanged.")
                st.info("Could not find a suitable thin layer to remove, or removal/re-optimization failed/unchanged.")
                current_ep_len = len(current_ep) if current_ep is not None else 0
                st.session_state.current_status = f"Status: Removal Skipped/Failed | Layers: {current_ep_len}"
                status_placeholder.warning(st.session_state.current_status)
                # Optionally re-display previous plot if removal failed
                # (run_calculation_st already handles displaying last successful result)

        except (ValueError, RuntimeError) as e:
            err_msg = f"ERROR (Input/Logic) during layer removal: {e}"
            log_message(err_msg); st.error(err_msg)
            st.session_state.current_status = "Status: Removal Failed (Input/Logic Error)"
            status_placeholder.error(st.session_state.current_status)
        except Exception as e:
            err_msg = f"ERROR (Unexpected) in layer removal: {type(e).__name__}: {e}"
            tb_msg = traceback.format_exc()
            log_message(err_msg); log_message(tb_msg); print(err_msg); print(tb_msg)
            st.error(f"{err_msg}. See log/console for details.")
            st.session_state.current_status = "Status: Removal Failed (Unexpected Error)"
            status_placeholder.error(st.session_state.current_status)
        finally:
            progress_placeholder.empty()
            # Update display info based on the potentially changed structure
            update_display_info(st.session_state.current_optimized_ep if removal_successful else current_ep)
            st.rerun() # Rerun to update button states and display info


if set_nominal_button:
    log_message("\n--- Setting Current Optimized Design as Nominal ---")
    st.session_state.current_status = "Status: Setting current as Nominal..."
    status_placeholder.info(st.session_state.current_status)

    current_ep = st.session_state.get('current_optimized_ep')
    optim_ran = st.session_state.get('optimization_ran_since_nominal_change', False)

    if current_ep is None or not optim_ran or len(current_ep) == 0:
        log_message("ERROR: No optimized design available to set as nominal.")
        st.error("No valid optimized design is currently loaded.")
        st.session_state.current_status = "Status: Set Nominal Failed (No Design)"
        status_placeholder.error(st.session_state.current_status)
    else:
        try:
            inputs = _validate_physical_inputs_from_state(require_optim_params=False)
            l0_val = inputs['l0']
            nH_r_val = inputs['nH_r']
            nL_r_val = inputs['nL_r']

            optimized_qwots = calculate_qwot_from_ep(current_ep, l0_val, nH_r_val, nL_r_val)

            if np.any(np.isnan(optimized_qwots)):
                final_qwot_str = ""
                log_message("Warning: QWOT calculation resulted in NaN. Cannot set nominal QWOT string.")
                st.warning("Could not calculate valid QWOT multipliers (resulted in NaN). Nominal QWOT field not updated.")
                # Do not update the input field if calculation failed
            else:
                # Format with sufficient precision for reuse
                final_qwot_str = ", ".join([f"{q:.6f}" for q in optimized_qwots])
                # Update the session state variable bound to the text_area *directly*
                # This is the standard way to programmatically change an input widget's value
                st.session_state.emp_str_input = final_qwot_str
                log_message(f"Nominal QWOT string updated in state: {final_qwot_str}")
                st.success("Current optimized design has been set as the new Nominal Structure (QWOT). Optimized state cleared.")

            # Clear the optimized state regardless of QWOT calculation success
            st.session_state.current_optimized_ep = None
            st.session_state.optimization_ran_since_nominal_change = False
            st.session_state.optimized_qwot_display = "" # Clear read-only display
            st.session_state.current_status = "Status: Idle (New Nominal Set)"
            status_placeholder.info(st.session_state.current_status)
            update_display_info(None) # Update display to nominal (ep_vector_source=None)

        except Exception as e:
            err_msg = f"ERROR during 'Set Nominal' operation: {type(e).__name__}: {e}"
            log_message(err_msg)
            st.error(f"An error occurred setting nominal: {e}")
            st.session_state.current_status = "Status: Set Nominal Failed (Error)"
            status_placeholder.error(st.session_state.current_status)
        finally:
            # Rerun needed to reflect the change in the emp_str_input widget
            # and update button states/display info.
            st.rerun()


# --- Initial Plot Display / Update on Rerun ---
# This block runs on every script rerun *unless* a button was just pressed and handled above.
# It displays the plot corresponding to the *last successful calculation* stored in session state.
if not (calc_button or opt_button or remove_button or set_nominal_button):
    last_res_data = st.session_state.get('last_run_calculation_results')
    if last_res_data and last_res_data.get('res') is not None and last_res_data.get('ep') is not None:
        try:
            # Regenerate the plot from stored data to ensure it's always displayed
            # This avoids losing the plot on simple reruns (e.g., window resize)
            inputs = last_res_data['inputs']
            active_targets = last_res_data['active_targets'] # Get targets used for that calc
            fig = tracer_graphiques(
                res=last_res_data['res'],
                ep_actual=last_res_data['ep'], # Use the stored ep vector
                nH_r=inputs['nH_r'], nH_i=inputs['nH_i'],
                nL_r=inputs['nL_r'], nL_i=inputs['nL_i'],
                nSub=inputs['nSub'],
                active_targets_for_plot=active_targets,
                mse=last_res_data['mse'],
                is_optimized=last_res_data['is_optimized'],
                method_name=last_res_data['method_name'],
                res_optim_grid=last_res_data['res_optim_grid']
            )
            plot_placeholder.pyplot(fig)
            plt.close(fig) # Close the figure after displaying
        except Exception as e_plot:
            st.warning(f"Could not redraw previous plot: {e_plot}")
            plot_placeholder.empty() # Clear placeholder if redraw fails
    else:
        # If no previous results, display a message
        plot_placeholder.info("Click 'Calculate Nominal' or 'Optimize N Passes' to generate results.")

    # Update display info on every rerun if no button was pressed
    # Use the appropriate ep vector based on current state
    current_ep_display = None
    if st.session_state.get('optimization_ran_since_nominal_change') and st.session_state.get('current_optimized_ep') is not None:
        current_ep_display = st.session_state.current_optimized_ep
    elif st.session_state.get('last_run_calculation_results', {}).get('ep') is not None:
         # Use the ep from the last calculation if not currently in optimized state
         current_ep_display = st.session_state.last_run_calculation_results['ep']
    # No else needed, update_display_info handles None or parses QWOT as fallback

    update_display_info(current_ep_display)


# --- Final status update ---
# Display the current status message at the end of the script run
current_status_message = st.session_state.get("current_status", "Status: Idle")
if "Failed" in current_status_message or "Error" in current_status_message:
    status_placeholder.error(current_status_message)
elif "Complete" in current_status_message or "Removed" in current_status_message or "Set" in current_status_message :
    status_placeholder.success(current_status_message)
elif "Idle" not in current_status_message: # Don't overwrite Idle if nothing else happened
     status_placeholder.info(current_status_message)
# Else: keep the placeholder empty or show Idle if needed

# --- Multiprocessing setup (usually not needed for Streamlit deployment) ---
# if __name__ == "__main__":
#    multiprocessing.freeze_support() # Keep if running as script, remove/comment for deployment
# No mainloop equivalent in Streamlit, the script just runs top-to-bottom on interaction.
