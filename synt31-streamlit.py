import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
import math
import copy

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
    if n <= 0: return 1
    exponent = math.floor(math.log2(n))+1
    return 1 << exponent

if 'log_messages' not in st.session_state: st.session_state.log_messages = []

def log_message(message):
    st.session_state.log_messages.append(str(message))

def log_with_elapsed_time(message):
    prefix = ""
    if 'optim_start_time' in st.session_state and st.session_state.optim_start_time is not None:
       elapsed = time.time() - st.session_state.optim_start_time
       prefix = f"[{elapsed:8.2f}s] "
    st.session_state.log_messages.append(prefix + str(message))

def clear_log():
    st.session_state.log_messages = [f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Log cleared."]

# --- Core Calculation Logic ---
@numba.njit(fastmath=True, cache=True)
def compute_stack_matrix(ep_vector, l_val, nH_complex, nL_complex):
    M = np.eye(2, dtype=np.complex128)
    for i in range(len(ep_vector)):
        thickness = ep_vector[i]
        if thickness <= 1e-12: continue
        Ni = nH_complex if i % 2 == 0 else nL_complex
        eta = Ni
        phi = (2 * np.pi / l_val) * (Ni * thickness)
        cos_phi = np.cos(phi); sin_phi = np.sin(phi)
        M_layer = np.array([[cos_phi, (1j / eta) * sin_phi], [1j * eta * sin_phi, cos_phi]], dtype=np.complex128)
        M = M_layer @ M
    return M

@numba.njit(parallel=True, fastmath=True, cache=True)
def calculate_RT_from_ep_core(ep_vector, nH_complex, nL_complex, nSub_complex, l_vec):
    if not l_vec.size: return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    Rs_arr = np.zeros(len(l_vec), dtype=np.float64); Ts_arr = np.zeros(len(l_vec), dtype=np.float64)
    etainc = 1.0 + 0j; etasub = nSub_complex
    for i_l in prange(len(l_vec)):
        l_val = l_vec[i_l]
        if l_val <= 0: Rs_arr[i_l], Ts_arr[i_l] = np.nan, np.nan; continue
        ep_vector_contig = np.ascontiguousarray(ep_vector)
        M = compute_stack_matrix(ep_vector_contig, l_val, nH_complex, nL_complex)
        m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        rs_num = (etainc * m00 - etasub * m11 + etainc * etasub * m01 - m10)
        rs_den = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
        if np.abs(rs_den) < 1e-12: Rs_arr[i_l], Ts_arr[i_l] = np.nan, np.nan
        else:
            rs = rs_num / rs_den; ts = (2 * etainc) / rs_den
            Rs_arr[i_l] = np.abs(rs)**2; real_etasub = np.real(etasub); real_etainc = np.real(etainc)
            if real_etainc == 0: Ts_arr[i_l] = np.nan
            else: Ts_arr[i_l] = (real_etasub / real_etainc) * np.abs(ts)**2
    for i in range(len(Rs_arr)):
        if np.isnan(Rs_arr[i]): Rs_arr[i] = 0.0
        if np.isnan(Ts_arr[i]): Ts_arr[i] = 0.0
    return Rs_arr, Ts_arr

@numba.njit(fastmath=True, cache=True)
def get_target_points_indices(l_vec, target_min, target_max):
    if not l_vec.size: return np.empty(0, dtype=np.int64)
    return np.where((l_vec >= target_min) & (l_vec <= target_max))[0]

def calculate_RT_from_ep(ep_vector, nH, nL, nSub, l_vec):
    nH_complex = nH + 0j if isinstance(nH, (int, float)) else nH
    nL_complex = nL + 0j if isinstance(nL, (int, float)) else nL
    nSub_complex = nSub + 0j if isinstance(nSub, (int, float)) else nSub
    ep_vector_np = np.ascontiguousarray(ep_vector, dtype=np.float64)
    l_vec_np = np.ascontiguousarray(l_vec, dtype=np.float64)
    Rs, Ts = calculate_RT_from_ep_core(ep_vector_np, nH_complex, nL_complex, nSub_complex, l_vec_np)
    return {'l': l_vec, 'Rs': Rs, 'Ts': Ts}

def calculate_initial_ep(emp, l0, nH_real, nL_real):
    num_layers = len(emp); ep_initial = np.zeros(num_layers, dtype=np.float64)
    for i in range(num_layers):
        multiplier = emp[i]; n_real = nH_real if i % 2 == 0 else nL_real
        if n_real <= 1e-9: ep_initial[i] = 0.0
        else:
            if l0 <= 0: raise ValueError("l0 must be positive")
            ep_initial[i] = multiplier * l0 / (4 * n_real)
    return ep_initial

def get_initial_ep(emp_str, l0, nH, nL):
    try: emp = [float(e.strip()) for e in emp_str.split(',') if e.strip()]
    except ValueError: raise ValueError("Invalid QWOT format")
    if any(e < 0 for e in emp): raise ValueError("QWOT multipliers cannot be negative")
    if not emp: return np.array([], dtype=np.float64), []
    nH_r = np.real(nH); nL_r = np.real(nL)
    if nH_r <= 0 or nL_r <=0: raise ValueError(f"Real parts nH ({nH_r}) and nL ({nL_r}) must be > 0")
    if l0 <= 0: raise ValueError(f"l0 ({l0}) must be > 0")
    ep_initial = calculate_initial_ep(tuple(emp), l0, nH_r, nL_r)
    if not np.all(np.isfinite(ep_initial)):
        log_message("WARNING: Initial QWOT calc produced NaN/inf. Replaced with 0.")
        ep_initial = np.nan_to_num(ep_initial, nan=0.0, posinf=0.0, neginf=0.0)
    return ep_initial, emp

def calculate_qwot_from_ep(ep_vector, l0, nH_r, nL_r):
    num_layers = len(ep_vector); qwot_multipliers = np.zeros(num_layers, dtype=np.float64)
    if l0 <= 0: log_message("Warning: Cannot calc QWOT, l0 must be positive."); qwot_multipliers[:] = np.nan; return qwot_multipliers
    if nH_r <=0 or nL_r <= 0: log_message(f"Warning: Cannot calc QWOT, nH_r ({nH_r}) and nL_r ({nL_r}) must be positive."); qwot_multipliers[:] = np.nan; return qwot_multipliers
    for i in range(num_layers):
        n_real = nH_r if i % 2 == 0 else nL_r
        if n_real <= 1e-9: qwot_multipliers[i] = np.nan
        else: qwot_multipliers[i] = ep_vector[i] * (4 * n_real) / l0
    return qwot_multipliers

def calculate_final_mse(res, active_targets):
    total_squared_error = 0.0; total_points_in_targets = 0; mse = None
    if not active_targets or not isinstance(res, dict) or 'Ts' not in res or res['Ts'] is None or not isinstance(res['Ts'], np.ndarray) or len(res['Ts'])==0 or 'l' not in res or res['l'] is None or not isinstance(res['l'], np.ndarray): return mse, total_points_in_targets
    calculated_lambdas = res['l']; calculated_Ts = res['Ts']
    if len(calculated_lambdas) != len(calculated_Ts): log_message("MSE Calc Warning: Lambda/Ts length mismatch."); return None, 0
    for i, target in enumerate(active_targets):
        l_min = target.get('min'); l_max = target.get('max'); t_min = target.get('target_min'); t_max = target.get('target_max')
        if None in [l_min, l_max, t_min, t_max]: log_message(f"MSE Calc Warning: Target {i+1} missing keys."); continue
        indices = get_target_points_indices(calculated_lambdas, l_min, l_max)
        if indices.size > 0:
            valid_indices = indices[indices < len(calculated_Ts)]
            if valid_indices.size == 0: continue
            calculated_Ts_in_zone = calculated_Ts[valid_indices]; target_lambdas_in_zone = calculated_lambdas[valid_indices]
            finite_mask = np.isfinite(calculated_Ts_in_zone); calculated_Ts_in_zone = calculated_Ts_in_zone[finite_mask]; target_lambdas_in_zone = target_lambdas_in_zone[finite_mask]
            if calculated_Ts_in_zone.size == 0: continue
            if abs(l_max - l_min) < 1e-9: interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
            else: slope = (t_max - t_min) / (l_max - l_min); interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)
            squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
            total_squared_error += np.sum(squared_errors); total_points_in_targets += len(calculated_Ts_in_zone)
    if total_points_in_targets > 0: mse = total_squared_error / total_points_in_targets
    elif active_targets: mse = np.nan
    return mse, total_points_in_targets

def calculate_mse_for_optimization_penalized(ep_vector, nH, nL, nSub, l_vec_optim, active_targets, min_thickness_phys_nm, debug_log=False):
    if debug_log: log_message(f"[DEBUG COST] Input ep_vector (len {len(ep_vector)}): {np.array2string(ep_vector, precision=4, max_line_width=120)}")
    if not isinstance(ep_vector, np.ndarray): ep_vector = np.array(ep_vector)
    ep_vector_calc = np.maximum(ep_vector, min_thickness_phys_nm)
    below_min_mask = ep_vector < min_thickness_phys_nm
    penalty = 0.0
    if np.any(below_min_mask):
        thickness_penalty = 1e6 + np.sum((min_thickness_phys_nm - ep_vector[below_min_mask])**2) * 1e8 # Reduced for debug
        penalty += thickness_penalty
        if debug_log: log_message(f"[DEBUG COST] Thickness penalty: {thickness_penalty:.4e}. Violated: {np.array2string(ep_vector[below_min_mask], precision=4)}")
    res = None; nan_penalty = 0.0
    try:
        nH_complex = nH + 0j if isinstance(nH, (int, float)) else nH; nL_complex = nL + 0j if isinstance(nL, (int, float)) else nL; nSub_complex = nSub + 0j if isinstance(nSub, (int, float)) else nSub
        res = calculate_RT_from_ep(ep_vector_calc, nH_complex, nL_complex, nSub_complex, l_vec_optim)
        if res is None or 'Ts' not in res or np.any(~np.isfinite(res['Ts'])):
             nan_penalty = 1e5 # Reduced for debug
             penalty += nan_penalty
             if debug_log: log_message(f"[DEBUG COST] NaN detected! NaN penalty: {nan_penalty:.4e}. Returning cost: {penalty:.4e}")
             return penalty
    except Exception as e_rt:
        if debug_log: log_message(f"[DEBUG COST] ERROR in calculate_RT_from_ep: {e_rt}")
        return np.inf
    mse = 1e7; total_points_in_targets = 0; total_squared_error = 0.0
    if 'Ts' in res and 'l' in res:
        calculated_lambdas = res['l']; calculated_Ts = res['Ts']
        for target in active_targets:
            l_min, l_max = target['min'], target['max']; t_min, t_max = target['target_min'], target['target_max']
            indices = get_target_points_indices(calculated_lambdas, l_min, l_max)
            if indices.size > 0:
                valid_indices = indices[indices < len(calculated_Ts)]
                if valid_indices.size == 0: continue
                calculated_Ts_in_zone = calculated_Ts[valid_indices]; target_lambdas_in_zone = calculated_lambdas[valid_indices]
                if abs(l_max - l_min) < 1e-9: interpolated_target_t = np.full_like(target_lambdas_in_zone, t_min)
                else: slope = (t_max - t_min) / (l_max - l_min); interpolated_target_t = t_min + slope * (target_lambdas_in_zone - l_min)
                squared_errors = (calculated_Ts_in_zone - interpolated_target_t)**2
                total_squared_error += np.sum(squared_errors); total_points_in_targets += len(valid_indices)
        if total_points_in_targets > 0: mse = total_squared_error / total_points_in_targets
    if debug_log: log_message(f"[DEBUG COST] MSE: {mse:.4e} (Points: {total_points_in_targets}), Penalty: {penalty:.4e}")
    final_cost = mse + penalty
    if debug_log: log_message(f"[DEBUG COST] Returning final_cost: {final_cost:.4e}")
    return final_cost

def perform_single_thin_layer_removal(ep_vector_in, min_thickness_phys, cost_function, args_for_cost, log_prefix="", target_layer_index=None):
    current_ep = ep_vector_in.copy(); logs = []; num_layers = len(current_ep)
    if num_layers <= 1:
        logs.append(f"{log_prefix}Structure has {num_layers} layers. Cannot merge/delete further.")
        try: initial_cost = cost_function(current_ep, *args_for_cost); success_overall = np.isfinite(initial_cost)
        except Exception: initial_cost = np.inf; success_overall = False
        return current_ep, success_overall, initial_cost, logs
    thin_layer_index = -1; min_thickness_found = np.inf
    if target_layer_index is not None:
        if 0 <= target_layer_index < num_layers: thin_layer_index = target_layer_index; min_thickness_found = current_ep[target_layer_index]; logs.append(f"{log_prefix}Targeting layer {thin_layer_index + 1} ({min_thickness_found:.3f} nm).")
        else: logs.append(f"{log_prefix}Invalid target index {target_layer_index+1}. Finding thinnest."); target_layer_index = None
    if target_layer_index is None:
        eligible_indices = np.where(current_ep >= min_thickness_phys)[0]
        if eligible_indices.size > 0: min_idx_within_eligible = np.argmin(current_ep[eligible_indices]); thin_layer_index = eligible_indices[min_idx_within_eligible]; min_thickness_found = current_ep[thin_layer_index]
    if thin_layer_index == -1:
        logs.append(f"{log_prefix}No suitable layer found for removal (thinnest >= {min_thickness_phys:.3f} nm).")
        try: initial_cost = cost_function(current_ep, *args_for_cost); success_overall = np.isfinite(initial_cost)
        except Exception: initial_cost = np.inf; success_overall = False
        return current_ep, success_overall, initial_cost, logs
    if target_layer_index is None: logs.append(f"{log_prefix}Identified thinnest eligible layer: {thin_layer_index + 1} ({min_thickness_found:.3f} nm).")
    ep_after_merge = None; merged_info = ""; structure_changed = False
    if thin_layer_index == 0:
        if num_layers >= 2: ep_after_merge = current_ep[2:]; merged_info = f"Removing layer 1 & 2."; logs.append(f"{log_prefix}{merged_info} New: {len(ep_after_merge)} layers."); structure_changed = True
        else: logs.append(f"{log_prefix}Cannot remove layer 1, structure too small."); try: cost = cost_function(current_ep, *args_for_cost); success = np.isfinite(cost); except Exception: cost = np.inf; success=False; return current_ep, success, cost, logs
    elif thin_layer_index == num_layers - 1:
        if num_layers >= 1: ep_after_merge = current_ep[:-1]; merged_info = f"Removed last layer {num_layers} ({current_ep[-1]:.3f} nm)."; logs.append(f"{log_prefix}{merged_info} New: {len(ep_after_merge)} layers."); structure_changed = True
    else:
        if num_layers >= 3:
            merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
            ep_after_merge = np.concatenate((current_ep[:thin_layer_index - 1], [merged_thickness], current_ep[thin_layer_index + 2:]))
            merged_info = (f"Removed layer {thin_layer_index + 1}, merged {thin_layer_index}({current_ep[thin_layer_index - 1]:.3f}) + {thin_layer_index + 2}({current_ep[thin_layer_index + 1]:.3f}) -> {merged_thickness:.3f}")
            logs.append(f"{log_prefix}{merged_info} New: {len(ep_after_merge)} layers."); structure_changed = True
        else: logs.append(f"{log_prefix}Cannot merge around layer {thin_layer_index+1}."); try: cost = cost_function(current_ep, *args_for_cost); success=np.isfinite(cost); except Exception: cost = np.inf; success=False; return current_ep, success, cost, logs
    final_ep = current_ep; final_cost = np.inf; success_overall = False
    if structure_changed and ep_after_merge is not None:
        if ep_after_merge.size > 0: ep_after_merge = np.maximum(ep_after_merge, min_thickness_phys)
        num_layers_reopt = len(ep_after_merge)
        if num_layers_reopt == 0: logs.append(f"{log_prefix}Empty structure after merge."); return np.array([]), True, np.inf, logs
        reopt_bounds = [(min_thickness_phys, None)] * num_layers_reopt; x0_reopt = ep_after_merge
        logs.append(f"{log_prefix}Re-optimizing {num_layers_reopt} layers (L-BFGS-B)...")
        reopt_start_time = time.time(); reopt_args = args_for_cost
        try:
            reopt_result = minimize(cost_function, x0_reopt, args=reopt_args, method='L-BFGS-B', bounds=reopt_bounds, options={'maxiter': 199, 'ftol': 1e-10, 'gtol': 1e-7, 'disp': False})
            reopt_time = time.time() - reopt_start_time; reopt_success = reopt_result.success and np.isfinite(reopt_result.fun)
            reopt_cost = reopt_result.fun if reopt_success else np.inf; reopt_iters = reopt_result.nit
            logs.append(f"{log_prefix}Re-opt finished in {reopt_time:.3f}s. Success: {reopt_success}, Cost: {reopt_cost:.3e}, Iters: {reopt_iters}")
            if reopt_success: final_ep = np.maximum(reopt_result.x.copy(), min_thickness_phys); final_cost = reopt_cost; success_overall = True; logs.append(f"{log_prefix}Re-opt successful.")
            else:
                logs.append(f"{log_prefix}Re-opt failed. Returning merged structure."); final_ep = np.maximum(ep_after_merge.copy(), min_thickness_phys); success_overall = False
                try: final_cost = cost_function(final_ep, *reopt_args); logs.append(f"{log_prefix}Recalculated cost: {final_cost:.3e}")
                except Exception as e_cost: final_cost = np.inf; logs.append(f"{log_prefix}Error calculating fallback cost: {e_cost}")
        except Exception as e_reopt:
            logs.append(f"{log_prefix}ERROR during re-opt: {e_reopt}\n{traceback.format_exc(limit=2)}"); logs.append(f"{log_prefix}Returning merged structure."); final_ep = np.maximum(ep_after_merge.copy(), min_thickness_phys); success_overall = False
            try: final_cost = cost_function(final_ep, *reopt_args)
            except Exception: final_cost = np.inf
    else:
        logs.append(f"{log_prefix}Structure not changed. No re-optimization."); success_overall = False; final_ep = current_ep
        try: final_cost = cost_function(final_ep, *args_for_cost)
        except Exception: final_cost = np.inf
    return final_ep, success_overall, final_cost, logs

def local_search_worker(start_ep, cost_function, args_for_cost, lbfgsb_bounds, min_thickness_phys_nm):
    worker_pid = os.getpid(); start_time = time.time(); log_lines = []; log_lines.append(f"  [PID:{worker_pid} Start]: Starting local search.")
    initial_cost = np.inf; start_ep_checked = np.array([])
    try:
        start_ep_checked = np.maximum(np.asarray(start_ep), min_thickness_phys_nm)
        if start_ep_checked.size == 0: raise ValueError("Start EP empty.")
        initial_cost = cost_function(start_ep_checked, *args_for_cost)
        if not np.isfinite(initial_cost): log_lines.append(f"  [PID:{worker_pid} Start Warning]: Initial cost {initial_cost:.3e}."); initial_cost = np.inf
    except Exception as e:
        log_lines.append(f"  [PID:{worker_pid} Start ERROR]: calculating initial cost: {e}"); result = OptimizeResult(x=np.asarray(start_ep), success=False, fun=np.inf, message=f"Worker Initial Cost Error: {e}", nit=0)
        end_time = time.time(); worker_summary = f"Worker {worker_pid} finished {end_time - start_time:.2f}s. StartCost=ERROR, FinalCost=inf, Success=False"; log_lines.append(f"  [PID:{worker_pid} Summary]: {worker_summary}")
        return {'result': result, 'final_ep': np.asarray(start_ep), 'final_cost': np.inf, 'success': False, 'start_ep': start_ep, 'pid': worker_pid, 'log_lines': log_lines, 'initial_cost': np.inf}
    result = OptimizeResult(x=start_ep_checked, success=False, fun=initial_cost, message="Worker opt failed early", nit=0)
    try:
        if len(lbfgsb_bounds) != len(start_ep_checked): raise ValueError(f"Dim mismatch: ep ({len(start_ep_checked)}) vs bounds ({len(lbfgsb_bounds)})")
        result = minimize(cost_function, start_ep_checked, args=args_for_cost, method='L-BFGS-B', bounds=lbfgsb_bounds, options={'disp': False, 'maxiter': 99, 'ftol': 1.e-10, 'gtol': 1e-7})
        if result.x is not None: result.x = np.maximum(result.x, min_thickness_phys_nm)
    except Exception as e:
        end_time = time.time(); error_message = f"Worker {worker_pid} Exception in minimize {end_time-start_time:.2f}s: {type(e).__name__}: {e}"
        log_lines.append(f"  [PID:{worker_pid} ERROR]: {error_message}\n{traceback.format_exc(limit=2)}"); result = OptimizeResult(x=start_ep_checked, success=False, fun=np.inf, message=f"Worker Exception: {e}", nit=0)
    end_time = time.time(); final_cost = result.fun if np.isfinite(result.fun) else np.inf
    success_status = result.success and np.isfinite(final_cost); iterations = result.nit
    message = result.message.decode('utf-8', errors='ignore') if isinstance(result.message, bytes) else str(result.message)
    worker_summary = (f"Worker {worker_pid} {end_time - start_time:.2f}s. Start={initial_cost:.3e}, Final={final_cost:.3e}, Success={success_status} (LBFGSB:{result.success}, Finite:{np.isfinite(final_cost)}), Iters={iterations}, Msg='{message}'")
    log_lines.append(f"  [PID:{worker_pid} Summary]: {worker_summary}")
    final_x = result.x if success_status and result.x is not None else start_ep_checked
    return {'result': result, 'final_ep': final_x, 'final_cost': final_cost, 'success': success_status, 'start_ep': start_ep, 'pid': worker_pid, 'log_lines': log_lines, 'initial_cost': initial_cost}

def _run_sobol_evaluation(num_layers, n_samples, lower_bounds, upper_bounds, cost_function_partial_map, min_thickness_phys_nm, phase_name="Phase 1", progress_hook=None):
    log_with_elapsed_time(f"\n--- {phase_name}: Initial Evaluation (Sobol, N={n_samples}) ---"); start_time_eval = time.time()
    if lower_bounds is None or upper_bounds is None or len(lower_bounds) != num_layers or len(upper_bounds) != num_layers: raise ValueError(f"Sobol needs valid bounds matching num_layers ({num_layers})")
    if np.any(lower_bounds > upper_bounds): log_with_elapsed_time(f"Warning: Sobol bounds L>U. Clamping."); upper_bounds = np.maximum(lower_bounds, upper_bounds)
    lower_bounds = np.maximum(min_thickness_phys_nm, lower_bounds); upper_bounds = np.maximum(lower_bounds, upper_bounds)
    log_with_elapsed_time(f"  Bounds (min={min_thickness_phys_nm:.3f}): L={np.array2string(lower_bounds, precision=3)}, U={np.array2string(upper_bounds, precision=3)}")
    sampler = qmc.Sobol(d=num_layers, scramble=True); points_unit_cube = sampler.random(n=n_samples)
    ep_candidates = qmc.scale(points_unit_cube, lower_bounds, upper_bounds); ep_candidates = [np.maximum(min_thickness_phys_nm, cand) for cand in ep_candidates]
    costs = []; initial_results = []; num_workers_eval = min(n_samples, os.cpu_count())
    log_with_elapsed_time(f"  Evaluating cost for {n_samples} candidates (max {num_workers_eval} workers)..."); eval_start_pool = time.time()
    try:
        if ep_candidates:
            with multiprocessing.Pool(processes=num_workers_eval) as pool: costs = pool.map(cost_function_partial_map, ep_candidates)
            eval_pool_time = time.time() - eval_start_pool; log_with_elapsed_time(f"  Parallel evaluation finished in {eval_pool_time:.2f}s.")
            if len(costs) != len(ep_candidates): log_with_elapsed_time(f"Warning: Cost/candidate length mismatch ({len(costs)} vs {len(ep_candidates)})."); costs = costs[:len(ep_candidates)]
            initial_results = list(zip(costs, ep_candidates))
        else: log_with_elapsed_time("  No candidates generated."); initial_results = []
    except Exception as e_pool:
        log_with_elapsed_time(f"  ERROR during parallel eval: {e_pool}\n{traceback.format_exc(limit=2)}"); log_with_elapsed_time("  Switching to sequential..."); costs = []
        eval_start_seq = time.time()
        for i, cand in enumerate(ep_candidates):
            try: cost = cost_function_partial_map(cand); costs.append(cost)
            except Exception: costs.append(np.inf)
            if (i + 1) % 200 == 0 or i == len(ep_candidates) - 1: log_with_elapsed_time(f"    Evaluated {i + 1}/{n_samples} sequentially..."); #if progress_hook: progress_hook((i + 1) / n_samples)
        eval_seq_time = time.time() - eval_start_seq; initial_results = list(zip(costs, ep_candidates)); log_with_elapsed_time(f"  Sequential evaluation finished in {eval_seq_time:.2f}s.")
    valid_initial_results = [(c, p) for c, p in initial_results if np.isfinite(c) and c < 1e9]
    num_invalid = len(initial_results) - len(valid_initial_results)
    if num_invalid > 0: log_with_elapsed_time(f"  Filtering: {num_invalid} invalid costs discarded.")
    if not valid_initial_results: log_with_elapsed_time(f"Warning: No valid initial costs found after {phase_name} eval."); return []
    valid_initial_results.sort(key=lambda x: x[0]); eval_time = time.time() - start_time_eval
    log_with_elapsed_time(f"--- End {phase_name} (Initial Evaluation) in {eval_time:.2f}s ---")
    if valid_initial_results: log_with_elapsed_time(f"  Found {len(valid_initial_results)} valid points. Best initial cost: {valid_initial_results[0][0]:.3e}")
    return valid_initial_results

def _run_parallel_local_search(p_best_starts, cost_function, args_for_cost_tuple, lbfgsb_bounds, min_thickness_phys_nm, progress_hook=None):
    p_best_actual = len(p_best_starts)
    if p_best_actual == 0: log_with_elapsed_time("WARNING: No starts for local search. Skipping."); return []
    log_with_elapsed_time(f"\n--- Phase 2: Local Search (L-BFGS-B) (P={p_best_actual}) ---"); start_time_local = time.time()
    local_search_partial = partial(local_search_worker, cost_function=cost_function, args_for_cost=args_for_cost_tuple, lbfgsb_bounds=lbfgsb_bounds, min_thickness_phys_nm=min_thickness_phys_nm)
    local_results_raw = []; num_workers_local = min(p_best_actual, os.cpu_count())
    log_with_elapsed_time(f"  Starting {p_best_actual} local searches in parallel (max {num_workers_local} workers)..."); local_start_pool = time.time()
    try:
        if p_best_starts:
            with multiprocessing.Pool(processes=num_workers_local) as pool: local_results_raw = list(pool.map(local_search_partial, p_best_starts))
            local_pool_time = time.time() - local_start_pool; log_with_elapsed_time(f"  Parallel local searches finished in {local_pool_time:.2f}s.")
    except Exception as e_pool_local: log_with_elapsed_time(f"  ERROR during parallel local search pool: {e_pool_local}\n{traceback.format_exc(limit=2)}"); local_results_raw = []
    local_time = time.time() - start_time_local; log_with_elapsed_time(f"--- End Phase 2 (Local Search) in {local_time:.2f}s ---")
    return local_results_raw

def _process_optimization_results(local_results_raw, initial_best_ep, initial_best_cost):
    log_with_elapsed_time(f"\n--- Processing {len(local_results_raw)} local search results ---")
    overall_best_cost = initial_best_cost; overall_best_ep = initial_best_ep.copy() if initial_best_ep is not None else None
    overall_best_result_obj = OptimizeResult(x=overall_best_ep, fun=overall_best_cost, success=(overall_best_ep is not None and np.isfinite(overall_best_cost)), message="Initial best (pre-local search)", nit=0)
    best_run_info = {'index': -1, 'cost': overall_best_cost}; processed_results_count = 0
    if overall_best_ep is None: log_with_elapsed_time("Warning: Initial best EP for processing is None.")
    for i, worker_output in enumerate(local_results_raw):
        run_idx = i + 1
        if isinstance(worker_output, dict) and 'log_lines' in worker_output:
            for line in worker_output.get('log_lines', []): log_with_elapsed_time(f"    (Worker Log Run {run_idx}): {line.strip()}")
        else: log_with_elapsed_time(f"  Result {run_idx}: Unexpected worker format."); continue
        if 'result' in worker_output and 'final_ep' in worker_output and 'final_cost' in worker_output and 'success' in worker_output:
            res_obj = worker_output['result']; final_cost_run = worker_output['final_cost']; final_ep_run = worker_output['final_ep']; success_run = worker_output['success']
            processed_results_count += 1
            if success_run and final_ep_run is not None and final_ep_run.size > 0 and final_cost_run < overall_best_cost:
                log_with_elapsed_time(f"  ==> New overall best by Result {run_idx}! Cost: {final_cost_run:.3e} (Prev: {overall_best_cost:.3e})")
                overall_best_cost = final_cost_run; overall_best_ep = final_ep_run.copy(); overall_best_result_obj = res_obj; best_run_info = {'index': run_idx, 'cost': final_cost_run}
        else: log_with_elapsed_time(f"  Result {run_idx}: Worker dict missing required keys.")
    log_with_elapsed_time(f"--- Finished processing local search results ---")
    if processed_results_count == 0 and best_run_info['index'] == -1: log_with_elapsed_time("WARNING: No valid local search results processed. Keeping pre-local best.")
    elif best_run_info['index'] > 0: log_with_elapsed_time(f"Overall best result found by local search Run {best_run_info['index']} (Cost {best_run_info['cost']:.3e}).")
    elif best_run_info['index'] == -1: log_with_elapsed_time(f"No local search improved pre-local best (Cost {best_run_info['cost']:.3e}).")
    if overall_best_ep is None or len(overall_best_ep) == 0:
         log_with_elapsed_time("CRITICAL WARNING: Final best EP is empty!");
         if initial_best_ep is None: raise RuntimeError("Result processing failed: Final EP empty, initial was also empty.")
         else: log_with_elapsed_time("Falling back to initial state."); overall_best_ep = initial_best_ep.copy(); overall_best_cost = initial_best_cost; overall_best_result_obj = OptimizeResult(x=overall_best_ep, fun=overall_best_cost, success=False, message="Fell back to initial", nit=0)
    return overall_best_ep, overall_best_cost, overall_best_result_obj

# --- Streamlit UI and Logic ---
st.set_page_config(layout="wide")
st.title("Thin Film Stack Optimizer (Streamlit v2.25-DBG)")

if 'log_messages' not in st.session_state: st.session_state.log_messages = []
if 'current_optimized_ep' not in st.session_state: st.session_state.current_optimized_ep = None
if 'optimization_ran_since_nominal_change' not in st.session_state: st.session_state.optimization_ran_since_nominal_change = False
if 'optim_start_time' not in st.session_state: st.session_state.optim_start_time = None
if 'last_run_calculation_results' not in st.session_state: st.session_state.last_run_calculation_results = {}
if 'optim_mode' not in st.session_state: st.session_state.optim_mode = DEFAULT_MODE
if 'current_status' not in st.session_state: st.session_state.current_status = "Status: Idle"
if 'current_progress' not in st.session_state: st.session_state.current_progress = 0
if 'num_layers_display' not in st.session_state: st.session_state.num_layers_display = "Layers (Nominal): 0"
if 'thinnest_layer_display' not in st.session_state: st.session_state.thinnest_layer_display = "- nm"
if 'optimized_qwot_display' not in st.session_state: st.session_state.optimized_qwot_display = ""
if 'targets' not in st.session_state: st.session_state.targets = copy.deepcopy(default_targets)

col1, col2 = st.columns(2)
with col1:
    with st.expander("Materials and Substrate", expanded=True):
        c1, c2 = st.columns(2)
        with c1: st.number_input("Material H (n real)", min_value=0.01, value=2.35, step=0.01, format="%.3f", key='nH_r')
        with c1: st.number_input("Material L (n real)", min_value=0.01, value=1.46, step=0.01, format="%.3f", key='nL_r')
        with c1: st.number_input("Substrate (n real)", min_value=0.01, value=1.52, step=0.01, format="%.3f", key='nSub')
        with c2: st.number_input("Material H (k imag)", min_value=0.0, value=0.0, step=0.001, format="%.4f", key='nH_i')
        with c2: st.number_input("Material L (k imag)", min_value=0.0, value=0.0, step=0.001, format="%.4f", key='nL_i')
        with c2: st.caption("(n = n' + ik)")
    with st.expander("Stack (Nominal Definition)", expanded=True):
        st.text_area("Nominal Structure (QWOT Multipliers, comma-separated)", value=default_qwot, key='emp_str_input')
        st.number_input("Centering λ (QWOT, nm)", min_value=0.1, value=500.0, step=1.0, format="%.1f", key='l0_input')
        st.text_input("Optimized QWOT (Read-only)", value=st.session_state.optimized_qwot_display, disabled=True, key='opt_qwot_ro')
with col2:
    with st.expander("Calculation & Optimization Parameters", expanded=True):
        st.radio("Optimization Mode Preset", options=list(OPTIMIZATION_MODES.keys()), index=list(OPTIMIZATION_MODES.keys()).index(st.session_state.optim_mode), key='optim_mode_radio', horizontal=True)
        st.session_state.optim_mode = st.session_state.optim_mode_radio
        selected_params = OPTIMIZATION_MODES[st.session_state.optim_mode]
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1: st.number_input("λ Start (nm)", min_value=0.1, value=400.0, step=1.0, format="%.1f", key='l_range_deb_input')
        with c2: st.number_input("λ End (nm)", min_value=0.1, value=700.0, step=1.0, format="%.1f", key='l_range_fin_input')
        with c3: st.number_input("λ Step (nm, Optim Grid)", min_value=0.01, value=10.0, step=0.1, format="%.2f", key='l_step_input'); st.caption("Plot uses λ Step / 10")
        c1, c2, c3 = st.columns(3)
        with c1: st.text_input("N Samples (Sobol)", value=selected_params['n_samples'], key='n_samples_input')
        with c2: st.text_input("P Starts (L-BFGS-B)", value=selected_params['p_best'], key='p_best_input')
        with c3: st.text_input("Optim Passes", value=selected_params['n_passes'], key='n_passes_input')
        st.number_input("Scaling (nm, Pass 2+)", min_value=0.0, value=10.0, step=0.1, format="%.2f", key='scaling_nm_input')
    with st.expander("Spectral Target (Optimization on Transmittance T)", expanded=True):
        st.caption("Define target transmittance ramps (T vs λ).")
        headers = st.columns([0.5, 0.5, 1, 1, 1, 1])
        headers[0].markdown("**Enable**"); headers[1].markdown("**Zone**"); headers[2].markdown("**λ min (nm)**"); headers[3].markdown("**λ max (nm)**"); headers[4].markdown("**T @ λmin**"); headers[5].markdown("**T @ λmax**")
        for i in range(len(st.session_state.targets)):
            cols = st.columns([0.5, 0.5, 1, 1, 1, 1]); target_state = st.session_state.targets[i]
            enabled = cols[0].checkbox("", value=target_state['enabled'], key=f"target_{i}_enabled"); cols[1].markdown(f"**{i+1}**")
            l_min_str = cols[2].text_input("", value=target_state['min'], key=f"target_{i}_min", label_visibility="collapsed"); l_max_str = cols[3].text_input("", value=target_state['max'], key=f"target_{i}_max", label_visibility="collapsed")
            t_min_str = cols[4].text_input("", value=target_state['target_min'], key=f"target_{i}_tmin", label_visibility="collapsed"); t_max_str = cols[5].text_input("", value=target_state['target_max'], key=f"target_{i}_tmax", label_visibility="collapsed")
            st.session_state.targets[i]['enabled'] = enabled; st.session_state.targets[i]['min'] = l_min_str; st.session_state.targets[i]['max'] = l_max_str; st.session_state.targets[i]['target_min'] = t_min_str; st.session_state.targets[i]['target_max'] = t_max_str

st.markdown("---")
status_placeholder = st.empty()
progress_placeholder = st.empty()

st.checkbox("Run Single Verbose L-BFGS-B Debug", key="debug_lbfgsb_run", value=False)

action_cols = st.columns([1, 1, 1.5, 1.5, 1])
calc_button = action_cols[0].button("Calculate Nominal", key="calc_nom", use_container_width=True)
opt_button = action_cols[1].button("Optimize N Passes", key="opt", use_container_width=True)
with action_cols[2]:
    remove_layer_enabled = (st.session_state.current_optimized_ep is not None and len(st.session_state.current_optimized_ep) > 1 and st.session_state.optimization_ran_since_nominal_change)
    remove_button = st.button("Remove Thinnest Layer", key="remove_thin", disabled=not remove_layer_enabled, use_container_width=True)
    st.info(f"Thinnest ≥ {MIN_THICKNESS_PHYS_NM}nm: {st.session_state.thinnest_layer_display}")
with action_cols[3]:
    set_nominal_enabled = (st.session_state.current_optimized_ep is not None and len(st.session_state.current_optimized_ep) > 0 and st.session_state.optimization_ran_since_nominal_change)
    set_nominal_button = st.button("Set Current as Nominal", key="set_nom", disabled=not set_nominal_enabled, use_container_width=True)
    st.info(st.session_state.num_layers_display)

log_expander = st.expander("Log Messages")
with log_expander:
    if st.button("Clear Log", key="clear_log_btn"): clear_log(); st.rerun()
    log_content = "\n".join(st.session_state.log_messages)
    st.text_area("Log", value=log_content, height=300, key="log_display_area", disabled=True)

st.markdown("---")
st.subheader("Results")
plot_placeholder = st.empty()

if calc_button:
    st.session_state.current_optimized_ep = None; st.session_state.optimization_ran_since_nominal_change = False; st.session_state.optimized_qwot_display = ""
    run_calculation_st(ep_vector_to_use=None, is_optimized=False); st.rerun()

if opt_button:
    st.session_state.optim_start_time = time.time(); clear_log(); log_with_elapsed_time(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === Starting Multi-Pass Optimization Process ===")
    st.session_state.current_optimized_ep = None; st.session_state.optimization_ran_since_nominal_change = False; st.session_state.optimized_qwot_display = ""
    st.session_state.current_status = "Status: Initializing Optimization..."; status_placeholder.info(st.session_state.current_status); progress_bar = progress_placeholder.progress(0)
    min_scaling_nm = 0.1; ep_nominal_glob = None; overall_best_ep_final = None; overall_best_cost_final = np.inf; overall_best_layers = 0; initial_nominal_cost = np.inf
    initial_nominal_layers = 0; final_successful_result_obj = None; ep_ref_for_next_pass = None; n_passes = 0; optimization_successful = False; auto_removed_count = 0
    try:
        inputs = _validate_physical_inputs_from_state(require_optim_params=True)
        n_samples = inputs['n_samples']; initial_p_best = inputs['p_best']; n_passes = inputs['n_passes']; initial_scaling_nm = inputs['scaling_nm']
        progress_max_steps = (3 if n_passes >= 1 else 0) + max(0, (n_passes - 1) * 2) + 1; current_progress_step = 0
        active_targets = get_active_targets_from_state()
        if active_targets is None: raise ValueError("Failed to retrieve/validate spectral targets.")
        if not active_targets: raise ValueError("No active spectral targets defined. Optimization requires at least one target.")
        log_with_elapsed_time(f"{len(active_targets)} active target zone(s) found.")
        nH_complex_nom = inputs['nH_r'] + 1j*inputs['nH_i']; nL_complex_nom = inputs['nL_r'] + 1j*inputs['nL_i']
        ep_nominal_glob, _ = get_initial_ep(inputs['emp_str'], inputs['l0'], nH_complex_nom, nL_complex_nom)
        if ep_nominal_glob.size == 0: raise ValueError("Initial nominal QWOT stack is empty or invalid.")
        initial_nominal_layers = len(ep_nominal_glob)
        last_successful_ep = ep_nominal_glob.copy(); overall_best_ep_final = last_successful_ep.copy(); ep_ref_for_next_pass = None; overall_best_layers = initial_nominal_layers
        try:
            nSub_c = inputs['nSub'] + 0j; l_min_overall = inputs['l_range_deb']; l_max_overall = inputs['l_range_fin']; l_step_gui = inputs['l_step']
            num_points_approx = max(2, int(np.round((l_max_overall - l_min_overall) / l_step_gui)) + 1)
            l_vec_optim_init = np.geomspace(l_min_overall, l_max_overall, num_points_approx); l_vec_optim_init = l_vec_optim_init[(l_vec_optim_init > 0) & np.isfinite(l_vec_optim_init)]
            if l_vec_optim_init.size == 0: raise ValueError("Failed to create optim vec for initial cost.")
            initial_nominal_cost = calculate_mse_for_optimization_penalized(ep_nominal_glob, nH_complex_nom, nL_complex_nom, nSub_c, l_vec_optim_init, active_targets, MIN_THICKNESS_PHYS_NM, debug_log=False)
            overall_best_cost_final = initial_nominal_cost if np.isfinite(initial_nominal_cost) else np.inf
            log_with_elapsed_time(f"Initial nominal cost: {overall_best_cost_final:.3e}, Layers: {initial_nominal_layers}")
            st.session_state.current_status = f"Status: Initial | Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"; status_placeholder.info(st.session_state.current_status)
        except Exception as e_init_cost:
            log_with_elapsed_time(f"Warning: Could not calculate initial nominal cost: {e_init_cost}"); overall_best_cost_final = np.inf
            st.session_state.current_status = f"Status: Initial | Best MSE: N/A | Layers: {initial_nominal_layers}"; status_placeholder.warning(st.session_state.current_status)
        l_vec_optim = l_vec_optim_init

        for pass_num in range(1, n_passes + 1):
            run_id_str = f"Pass {pass_num}/{n_passes}"; pass_status_prefix = f"Status: {run_id_str}"; status_suffix = f"| Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
            st.session_state.current_status = f"{pass_status_prefix} - Starting... {status_suffix}"; status_placeholder.info(st.session_state.current_status); progress_bar.progress( current_progress_step / progress_max_steps )
            reduction_factor = 0.8 ** (pass_num - 1); p_best_for_this_pass = max(2, int(np.round(initial_p_best * reduction_factor)))
            current_scaling = None; ep_ref_sobol = None
            if pass_num == 1: current_scaling = None; ep_ref_sobol = ep_nominal_glob # Base Pass 1 Sobol on nominal
            else: scale_factor = 1.8**(pass_num - 2); current_scaling = max(min_scaling_nm, initial_scaling_nm / scale_factor); ep_ref_sobol = ep_ref_for_next_pass
            current_num_layers = len(ep_ref_sobol) if ep_ref_sobol is not None else initial_nominal_layers
            if current_num_layers == 0: raise ValueError("Cannot run Sobol with 0 layers")

            if pass_num == 1:
                scale_lower = 0.1; scale_upper = 2.0; ep_ref_clamped = np.maximum(MIN_THICKNESS_PHYS_NM, ep_ref_sobol)
                lower_bounds = np.maximum(MIN_THICKNESS_PHYS_NM, ep_ref_clamped * scale_lower)
                upper_bounds = ep_ref_clamped * scale_upper; upper_bounds = np.maximum(upper_bounds, lower_bounds + 0.1)
            else:
                if ep_ref_sobol is None: raise ValueError("ep_ref_sobol cannot be None for Pass > 1")
                ep_ref_clamped = np.maximum(MIN_THICKNESS_PHYS_NM, ep_ref_sobol)
                lower_bounds = np.maximum(MIN_THICKNESS_PHYS_NM, ep_ref_clamped - current_scaling)
                upper_bounds = ep_ref_clamped + current_scaling; upper_bounds = np.maximum(upper_bounds, lower_bounds + 0.1)

            args_for_cost_tuple_sobol = (inputs['nH_r'] + 1j*inputs['nH_i'], inputs['nL_r'] + 1j*inputs['nL_i'], inputs['nSub'] + 0j, l_vec_optim, active_targets, MIN_THICKNESS_PHYS_NM, False)
            cost_function_partial_map = partial(calculate_mse_for_optimization_penalized, nH=args_for_cost_tuple_sobol[0], nL=args_for_cost_tuple_sobol[1], nSub=args_for_cost_tuple_sobol[2], l_vec_optim=args_for_cost_tuple_sobol[3], active_targets=args_for_cost_tuple_sobol[4], min_thickness_phys_nm=args_for_cost_tuple_sobol[5], debug_log=args_for_cost_tuple_sobol[6])
            valid_initial_results_p1 = _run_sobol_evaluation(current_num_layers, n_samples, lower_bounds, upper_bounds, cost_function_partial_map, MIN_THICKNESS_PHYS_NM, phase_name=f"{run_id_str} Phase 1")
            top_p_results_combined = []
            if pass_num == 1:
                status_placeholder.info(f"{pass_status_prefix} - Phase 1bis (Refining Starts) {status_suffix}"); log_with_elapsed_time(f"\n--- {run_id_str} Phase 1bis: Refining Top P Starts ---")
                num_to_select_p1 = min(p_best_for_this_pass, len(valid_initial_results_p1))
                if num_to_select_p1 == 0: log_with_elapsed_time("WARNING: Phase 1 yielded no valid starting points. Skipping refinement."); top_p_results_combined = []
                else:
                    top_p_results_p1 = valid_initial_results_p1[:num_to_select_p1]; top_p_starts_p1 = [ep for cost, ep in top_p_results_p1]
                    phase1bis_scaling = 10.0; n_samples_per_point_raw = max(1, n_samples // p_best_for_this_pass) if p_best_for_this_pass > 0 else n_samples
                    n_samples_per_point = upper_power_of_2(n_samples_per_point_raw); log_with_elapsed_time(f"  Generating {n_samples_per_point} samples around top {num_to_select_p1} points (+/- {phase1bis_scaling} nm).")
                    phase1bis_candidates = []
                    for i, ep_start in enumerate(top_p_starts_p1):
                        lower_bounds_1bis = np.maximum(MIN_THICKNESS_PHYS_NM, ep_start - phase1bis_scaling); upper_bounds_1bis = ep_start + phase1bis_scaling; upper_bounds_1bis = np.maximum(upper_bounds_1bis, lower_bounds_1bis + 0.1)
                        sampler_1bis = qmc.Sobol(d=current_num_layers, scramble=True, seed=int(time.time()) + i); points_unit_1bis = sampler_1bis.random(n=n_samples_per_point)
                        new_candidates = qmc.scale(points_unit_1bis, lower_bounds_1bis, upper_bounds_1bis); new_candidates = [np.maximum(MIN_THICKNESS_PHYS_NM, cand) for cand in new_candidates]; phase1bis_candidates.extend(new_candidates)
                    log_with_elapsed_time(f"  Generated {len(phase1bis_candidates)} total candidates for Phase 1bis eval.")
                    costs_1bis = []; results_1bis_raw_pairs = []; num_workers_1bis = min(len(phase1bis_candidates), os.cpu_count())
                    log_with_elapsed_time(f"  Evaluating cost for {len(phase1bis_candidates)} Phase 1bis candidates (max {num_workers_1bis} workers)..."); eval_start_pool_1bis = time.time()
                    try:
                        if phase1bis_candidates:
                            with multiprocessing.Pool(processes=num_workers_1bis) as pool: costs_1bis = pool.map(cost_function_partial_map, phase1bis_candidates)
                            eval_pool_time_1bis = time.time() - eval_start_pool_1bis; log_with_elapsed_time(f"  Parallel eval (Phase 1bis) finished in {eval_pool_time_1bis:.2f}s.")
                            if len(costs_1bis) != len(phase1bis_candidates): costs_1bis = costs_1bis[:len(phase1bis_candidates)]
                            results_1bis_raw_pairs = list(zip(costs_1bis, phase1bis_candidates))
                        else: results_1bis_raw_pairs = []
                    except Exception as e_pool_1bis: log_with_elapsed_time(f"  ERROR during parallel eval (Phase 1bis): {e_pool_1bis}"); results_1bis_raw_pairs = []
                    valid_results_1bis = [(c, p) for c, p in results_1bis_raw_pairs if np.isfinite(c) and c < 1e9]
                    num_invalid_1bis = len(results_1bis_raw_pairs) - len(valid_results_1bis)
                    if num_invalid_1bis > 0: log_with_elapsed_time(f"  Filtering (Phase 1bis): {num_invalid_1bis} invalid costs discarded.")
                    combined_results = top_p_results_p1 + valid_results_1bis; log_with_elapsed_time(f"  Combined Phase 1 ({len(top_p_results_p1)}) + 1bis ({len(valid_results_1bis)}): {len(combined_results)} total.")
                    if not combined_results: log_with_elapsed_time("WARNING: No valid results after Phase 1bis."); top_p_results_combined = []
                    else: combined_results.sort(key=lambda x: x[0]); num_to_select_final = min(p_best_for_this_pass, len(combined_results)); log_with_elapsed_time(f"  Selecting final top {num_to_select_final} points for Phase 2."); top_p_results_combined = combined_results[:num_to_select_final]
                log_with_elapsed_time(f"--- End {run_id_str} Phase 1bis ---")
            else: num_to_select = min(p_best_for_this_pass, len(valid_initial_results_p1)); log_with_elapsed_time(f"  Selecting top {num_to_select} points from Phase 1 for Phase 2."); top_p_results_combined = valid_initial_results_p1[:num_to_select]

            if st.session_state.debug_lbfgsb_run and pass_num == 1 and top_p_results_combined:
                log_with_elapsed_time("\n--- STARTING SINGLE L-BFGS-B DEBUG RUN ---")
                debug_start_ep = top_p_results_combined[0][1].copy(); debug_cost_start = top_p_results_combined[0][0]
                log_with_elapsed_time(f"  Debug Start EP (Cost ~{debug_cost_start:.3e}): {np.array2string(debug_start_ep, precision=4, max_line_width=120)}")
                debug_args_for_cost_tuple = (inputs['nH_r'] + 1j*inputs['nH_i'], inputs['nL_r'] + 1j*inputs['nL_i'], inputs['nSub'] + 0j, l_vec_optim, active_targets, MIN_THICKNESS_PHYS_NM, True)
                debug_num_layers = len(debug_start_ep); debug_lbfgsb_bounds = [(MIN_THICKNESS_PHYS_NM, None)] * debug_num_layers
                try:
                    debug_result = minimize(calculate_mse_for_optimization_penalized, debug_start_ep, args=debug_args_for_cost_tuple, method='L-BFGS-B', bounds=debug_lbfgsb_bounds, options={'disp': 15, 'maxiter': 99, 'ftol': 1e-10, 'gtol': 1e-7})
                    log_with_elapsed_time("--- DEBUG L-BFGS-B Run Result ---"); log_message(f"Result Object:\n{debug_result}")
                except Exception as e_debug: log_with_elapsed_time(f"--- DEBUG L-BFGS-B Run FAILED ---"); log_message(f"Error: {e_debug}\n{traceback.format_exc()}")
                log_with_elapsed_time("--- END SINGLE L-BFGS-B DEBUG RUN ---")

            selected_starts = [ep for cost, ep in top_p_results_combined]; selected_costs = [cost for cost, ep in top_p_results_combined]
            status_placeholder.info(f"{pass_status_prefix} - Phase 2 (Local Search) {status_suffix}")
            if not selected_starts:
                log_with_elapsed_time(f"WARNING: No starts for Phase 2 ({run_id_str}). Skipping."); ep_this_pass = ep_ref_for_next_pass.copy() if ep_ref_for_next_pass is not None else overall_best_ep_final.copy() if overall_best_ep_final is not None else ep_nominal_glob.copy()
                try: cost_this_pass = cost_function_partial_map(ep_this_pass)
                except Exception: cost_this_pass = np.inf
                result_obj_this_pass = OptimizeResult(x=ep_this_pass, fun=cost_this_pass, success=False, message=f"Phase 2 skipped - no starts ({run_id_str})", nit=0)
            else:
                initial_best_ep_for_processing = selected_starts[0].copy(); initial_best_cost_for_processing = selected_costs[0]
                log_with_elapsed_time(f"  Top {len(selected_starts)} points for Phase 2:");
                for i in range(min(5, len(selected_starts))): log_with_elapsed_time(f"    Point {i+1}: Cost={selected_costs[i]:.3e}" if np.isfinite(selected_costs[i]) else "inf")
                if len(selected_starts) > 5: log_with_elapsed_time("    ...")
                num_layers_local = len(initial_best_ep_for_processing); lbfgsb_bounds = [(MIN_THICKNESS_PHYS_NM, None)] * num_layers_local
                args_for_cost_tuple_parallel = (inputs['nH_r'] + 1j*inputs['nH_i'], inputs['nL_r'] + 1j*inputs['nL_i'], inputs['nSub'] + 0j, l_vec_optim, active_targets, MIN_THICKNESS_PHYS_NM, False)
                local_results_raw = _run_parallel_local_search(selected_starts, calculate_mse_for_optimization_penalized, args_for_cost_tuple_parallel, lbfgsb_bounds, MIN_THICKNESS_PHYS_NM)
                status_placeholder.info(f"{pass_status_prefix} - Processing L-BFGS-B results... {status_suffix}")
                ep_this_pass, cost_this_pass, result_obj_this_pass = _process_optimization_results(local_results_raw, initial_best_ep_for_processing, initial_best_cost_for_processing)

            current_progress_step += (3 if pass_num == 1 else 2); progress_bar.progress(current_progress_step / progress_max_steps)
            new_best_found_this_pass = False
            if ep_this_pass is not None and ep_this_pass.size > 0 and np.isfinite(cost_this_pass):
                if cost_this_pass < overall_best_cost_final:
                    log_with_elapsed_time(f"*** New overall best in {run_id_str}: {cost_this_pass:.3e} (Prev: {overall_best_cost_final:.3e}) ***")
                    overall_best_cost_final = cost_this_pass; overall_best_ep_final = ep_this_pass.copy(); overall_best_layers = len(overall_best_ep_final); final_successful_result_obj = result_obj_this_pass; new_best_found_this_pass = True
                ep_ref_for_next_pass = overall_best_ep_final.copy()
                status_suffix = f"| Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
                if new_best_found_this_pass: st.session_state.current_status = f"{pass_status_prefix} - Completed. New Best! {status_suffix}"; status_placeholder.info(st.session_state.current_status)
                else: st.session_state.current_status = f"{pass_status_prefix} - Completed. No improvement. {status_suffix}"; status_placeholder.info(st.session_state.current_status)
            else:
                log_with_elapsed_time(f"ERROR: {run_id_str} failed (cost: {cost_this_pass})."); st.session_state.current_status = f"{pass_status_prefix} - FAILED. {status_suffix}"; status_placeholder.warning(st.session_state.current_status)
                if overall_best_ep_final is not None: log_with_elapsed_time("Continuing with previous best."); ep_ref_for_next_pass = overall_best_ep_final.copy()
                else: raise RuntimeError(f"{run_id_str} failed and no prior result exists. Aborting.")

        current_progress_step +=1; progress_bar.progress(current_progress_step / progress_max_steps)
        st.session_state.current_status = f"Status: Post-processing (Auto-Remove)... | Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"; status_placeholder.info(st.session_state.current_status)
        if overall_best_ep_final is None: raise RuntimeError("Optimization finished, but no valid result before post-processing.")
        log_with_elapsed_time("\n--- Checking for auto thin layer removal (< 1.0 nm) ---")
        auto_removal_threshold = 1.0; max_auto_removals = len(overall_best_ep_final)
        temp_inputs_ar = _validate_physical_inputs_from_state(require_optim_params=False); temp_active_targets_ar = get_active_targets_from_state()
        if temp_active_targets_ar is None: raise ValueError("Failed targets for auto-removal cost func.")
        temp_nH_ar = temp_inputs_ar['nH_r'] + 1j * temp_inputs_ar['nH_i']; temp_nL_ar = temp_inputs_ar['nL_r'] + 1j * temp_inputs_ar['nL_i']; temp_nSub_c_ar = temp_inputs_ar['nSub'] + 0j
        temp_l_min_ar = temp_inputs_ar['l_range_deb']; temp_l_max_ar = temp_inputs_ar['l_range_fin']; temp_l_step_ar = temp_inputs_ar['l_step']
        temp_num_pts_ar = max(2, int(np.round((temp_l_max_ar - temp_l_min_ar) / temp_l_step_ar)) + 1)
        temp_l_vec_optim_ar = np.geomspace(temp_l_min_ar, temp_l_max_ar, temp_num_pts_ar); temp_l_vec_optim_ar = temp_l_vec_optim_ar[(temp_l_vec_optim_ar > 0) & np.isfinite(temp_l_vec_optim_ar)]
        if not temp_l_vec_optim_ar.size: raise ValueError("Failed geomspace AR wavelength vector.")
        temp_args_for_cost_ar = (temp_nH_ar, temp_nL_ar, temp_nSub_c_ar, temp_l_vec_optim_ar, temp_active_targets_ar, MIN_THICKNESS_PHYS_NM, False)
        current_ep_for_removal = overall_best_ep_final.copy()
        while auto_removed_count < max_auto_removals:
            if current_ep_for_removal is None or len(current_ep_for_removal) <= 1: log_with_elapsed_time("  Structure too short for auto-removal."); break
            eligible_indices = np.where((current_ep_for_removal >= MIN_THICKNESS_PHYS_NM) & (current_ep_for_removal < auto_removal_threshold))[0]
            if eligible_indices.size > 0:
                min_val_in_eligible = np.min(current_ep_for_removal[eligible_indices]); thinnest_below_threshold_idx = np.where(current_ep_for_removal == min_val_in_eligible)[0][0]; thinnest_below_threshold_val = min_val_in_eligible
                log_with_elapsed_time(f"  Auto-removing layer {thinnest_below_threshold_idx + 1} ({thinnest_below_threshold_val:.3f} nm < {auto_removal_threshold} nm)")
                status_suffix = f"| Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"; st.session_state.current_status = f"Status: Auto-removing layer {auto_removed_count + 1}... {status_suffix}"; status_placeholder.info(st.session_state.current_status)
                new_ep, success, cost_after, removal_logs = perform_single_thin_layer_removal(current_ep_for_removal, MIN_THICKNESS_PHYS_NM, calculate_mse_for_optimization_penalized, temp_args_for_cost_ar, log_prefix="    [Auto Removal] ", target_layer_index=thinnest_below_threshold_idx)
                for log_line in removal_logs: log_with_elapsed_time(log_line)
                if success and new_ep is not None and len(new_ep) < len(current_ep_for_removal):
                    current_ep_for_removal = new_ep.copy(); overall_best_ep_final = new_ep.copy()
                    overall_best_cost_final = cost_after if np.isfinite(cost_after) else np.inf; overall_best_layers = len(overall_best_ep_final)
                    auto_removed_count += 1; log_with_elapsed_time(f"  Auto-removal successful. New cost: {overall_best_cost_final:.3e}, Layers: {overall_best_layers}")
                else: log_with_elapsed_time("    Auto-removal/re-opt failed/unchanged. Stopping."); break
            else: log_with_elapsed_time(f"  No layers found below {auto_removal_threshold} nm."); break
        if auto_removed_count > 0: log_with_elapsed_time(f"--- Finished automatic removal ({auto_removed_count} layer(s) removed) ---")

        optimization_successful = True
        st.session_state.current_optimized_ep = overall_best_ep_final.copy() if overall_best_ep_final is not None else None
        st.session_state.optimization_ran_since_nominal_change = True
        final_qwot_str = "QWOT Error"
        if overall_best_ep_final is not None and len(overall_best_ep_final) > 0:
             try: l0_val = inputs['l0']; nH_r_val = inputs['nH_r']; nL_r_val = inputs['nL_r']; optimized_qwots = calculate_qwot_from_ep(overall_best_ep_final, l0_val, nH_r_val, nL_r_val)
                 if np.any(np.isnan(optimized_qwots)): final_qwot_str = "QWOT N/A (NaN)"; else: final_qwot_str = ", ".join([f"{q:.3f}" for q in optimized_qwots]);
             except Exception as qwot_calc_error: log_with_elapsed_time(f"Error calculating final QWOTs: {qwot_calc_error}")
        else: final_qwot_str = "N/A (Empty Structure)"
        st.session_state.optimized_qwot_display = final_qwot_str
        final_method_name = f"{n_passes}-Pass Opt" + (f" + {auto_removed_count} AutoRm" if auto_removed_count > 0 else "")
        log_with_elapsed_time("\n" + "="*60); log_with_elapsed_time(f"--- Overall Optimization ({final_method_name}) Finished ---"); log_with_elapsed_time(f"Best Final Cost Found (MSE): {overall_best_cost_final:.3e}")
        log_with_elapsed_time(f"Final number of layers: {overall_best_layers}"); log_with_elapsed_time(f"Final Optimized QWOT (λ₀={inputs['l0']}nm): {final_qwot_str}")
        ep_str_list = [f"L{i+1}:{th:.4f}" for i, th in enumerate(overall_best_ep_final)] if overall_best_ep_final is not None else []; log_with_elapsed_time(f"Final optimized thicknesses ({overall_best_layers} layers, nm): [{', '.join(ep_str_list)}]")
        if final_successful_result_obj and isinstance(final_successful_result_obj, OptimizeResult):
             best_res_nit = getattr(final_successful_result_obj, 'nit', 'N/A'); best_res_msg = getattr(final_successful_result_obj, 'message', 'N/A')
             if isinstance(best_res_msg, bytes): best_res_msg = best_res_msg.decode('utf-8', errors='ignore')
             log_with_elapsed_time(f"Best Result Info: Iters={best_res_nit}, Msg='{best_res_msg}'")
        log_with_elapsed_time("="*60 + "\n"); progress_bar.progress(1.0); final_status_text = f"Status: Opt Complete{' (+AutoRm)' if auto_removed_count>0 else ''} | MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
        st.session_state.current_status = final_status_text; status_placeholder.success(st.session_state.current_status)
        run_calculation_st(ep_vector_to_use=overall_best_ep_final, is_optimized=True, method_name=final_method_name)
    except (ValueError, RuntimeError) as e:
        err_msg = f"ERROR (Input/Logic) during optimization: {e}"; log_with_elapsed_time(err_msg); st.error(err_msg); st.session_state.current_status = f"Status: Optimization Failed (Input/Logic Error)"
        status_placeholder.error(st.session_state.current_status); optimization_successful = False; st.session_state.current_optimized_ep = None; st.session_state.optimization_ran_since_nominal_change = False; st.session_state.optimized_qwot_display = "Error Input/Optim Logic"
    except Exception as e:
        err_msg = f"ERROR (Unexpected) during optimization: {type(e).__name__}: {e}"; tb_msg = traceback.format_exc(); log_with_elapsed_time(err_msg); log_with_elapsed_time(tb_msg); print(err_msg); print(tb_msg); st.error(f"{err_msg}. See log/console for details.")
        st.session_state.current_status = f"Status: Optimization Failed (Unexpected Error)"; status_placeholder.error(st.session_state.current_status); optimization_successful = False; st.session_state.current_optimized_ep = None; st.session_state.optimization_ran_since_nominal_change = False; st.session_state.optimized_qwot_display = "Optim Error Unexpected"
    finally: progress_placeholder.empty(); st.session_state.optim_start_time = None; update_display_info(st.session_state.current_optimized_ep if optimization_successful else None); st.rerun()

if remove_button:
    log_message("\n" + "-"*10 + " Attempting Manual Thin Layer Removal " + "-"*10); st.session_state.current_status = "Status: Removing thinnest layer..."; status_placeholder.info(st.session_state.current_status); progress_bar = progress_placeholder.progress(0)
    current_ep = st.session_state.get('current_optimized_ep'); optim_ran = st.session_state.get('optimization_ran_since_nominal_change', False)
    if current_ep is None or not optim_ran or len(current_ep) <= 1: log_message("ERROR: No valid optimized structure with >= 2 layers available."); st.error("No valid optimized structure (>= 2 layers) found."); st.session_state.current_status = "Status: Removal Failed (No Structure/Not Optimized)"; status_placeholder.error(st.session_state.current_status); progress_placeholder.empty()
    else:
        removal_successful = False
        try:
            progress_bar.progress(10); inputs = _validate_physical_inputs_from_state(require_optim_params=False); active_targets = get_active_targets_from_state()
            if active_targets is None: raise ValueError("Failed targets for re-optimization.");
            if not active_targets: raise ValueError("No active targets defined. Cannot re-optimize.")
            progress_bar.progress(20); nH = inputs['nH_r'] + 1j * inputs['nH_i']; nL = inputs['nL_r'] + 1j * inputs['nL_i']; nSub_c = inputs['nSub'] + 0j
            l_min = inputs['l_range_deb']; l_max = inputs['l_range_fin']; l_step = inputs['l_step']; num_pts = max(2, int(np.round((l_max - l_min) / l_step)) + 1)
            l_vec_optim = np.geomspace(l_min, l_max, num_pts); l_vec_optim = l_vec_optim[(l_vec_optim > 0) & np.isfinite(l_vec_optim)]
            if not l_vec_optim.size: raise ValueError("Failed optim vector for removal.")
            args_for_cost_tuple = (nH, nL, nSub_c, l_vec_optim, active_targets, MIN_THICKNESS_PHYS_NM, False)
            progress_bar.progress(30); log_message("  Starting removal and re-optimization...")
            new_ep, success, final_cost, removal_logs = perform_single_thin_layer_removal(current_ep, MIN_THICKNESS_PHYS_NM, calculate_mse_for_optimization_penalized, args_for_cost_tuple, log_prefix="  [Manual Removal] ")
            for log_line in removal_logs: log_message(log_line)
            progress_bar.progress(70); structure_actually_changed = (success and new_ep is not None and len(new_ep) < len(current_ep))
            if structure_actually_changed:
                log_message("  Layer removal successful."); st.session_state.current_optimized_ep = new_ep.copy(); removal_successful = True
                final_qwot_str = "QWOT Error"
                try: l0_val = inputs['l0']; nH_r_val = inputs['nH_r']; nL_r_val = inputs['nL_r']; optimized_qwots = calculate_qwot_from_ep(new_ep, l0_val, nH_r_val, nL_r_val)
                    if np.any(np.isnan(optimized_qwots)): final_qwot_str = "QWOT N/A (NaN)"; else: final_qwot_str = ", ".join([f"{q:.3f}" for q in optimized_qwots]);
                except Exception as qwot_calc_error: log_message(f"Error calculating QWOTs after removal: {qwot_calc_error}")
                st.session_state.optimized_qwot_display = final_qwot_str; log_message(f"  Cost after removal/re-opt: {final_cost:.3e}")
                st.session_state.current_status = f"Status: Layer Removed | MSE: {final_cost:.3e} | Layers: {len(new_ep)}"; status_placeholder.success(st.session_state.current_status)
                run_calculation_st(ep_vector_to_use=new_ep, is_optimized=True, method_name="Optimized (Post-Removal)"); progress_bar.progress(100)
            else:
                log_message("  No layer removed or structure unchanged."); st.info("Could not find suitable layer or removal/re-opt failed/unchanged.")
                current_ep_len = len(current_ep) if current_ep is not None else 0; st.session_state.current_status = f"Status: Removal Skipped/Failed | Layers: {current_ep_len}"; status_placeholder.warning(st.session_state.current_status)
        except (ValueError, RuntimeError) as e: err_msg = f"ERROR (Input/Logic) during layer removal: {e}"; log_message(err_msg); st.error(err_msg); st.session_state.current_status = "Status: Removal Failed (Input/Logic Error)"; status_placeholder.error(st.session_state.current_status)
        except Exception as e: err_msg = f"ERROR (Unexpected) in layer removal: {type(e).__name__}: {e}"; tb_msg = traceback.format_exc(); log_message(err_msg); log_message(tb_msg); print(err_msg); print(tb_msg); st.error(f"{err_msg}. See log/console."); st.session_state.current_status = "Status: Removal Failed (Unexpected Error)"; status_placeholder.error(st.session_state.current_status)
        finally: progress_placeholder.empty(); update_display_info(st.session_state.current_optimized_ep if removal_successful else current_ep); st.rerun()

if set_nominal_button:
    log_message("\n--- Setting Current Optimized Design as Nominal ---"); st.session_state.current_status = "Status: Setting current as Nominal..."; status_placeholder.info(st.session_state.current_status)
    current_ep = st.session_state.get('current_optimized_ep'); optim_ran = st.session_state.get('optimization_ran_since_nominal_change', False)
    if current_ep is None or not optim_ran or len(current_ep) == 0: log_message("ERROR: No optimized design available."); st.error("No valid optimized design loaded."); st.session_state.current_status = "Status: Set Nominal Failed (No Design)"; status_placeholder.error(st.session_state.current_status)
    else:
        try:
            inputs = _validate_physical_inputs_from_state(require_optim_params=False); l0_val = inputs['l0']; nH_r_val = inputs['nH_r']; nL_r_val = inputs['nL_r']
            optimized_qwots = calculate_qwot_from_ep(current_ep, l0_val, nH_r_val, nL_r_val)
            if np.any(np.isnan(optimized_qwots)): log_message("Warning: QWOT resulted in NaN."); st.warning("Could not calculate valid QWOT multipliers (NaN). Nominal QWOT field not updated.")
            else: final_qwot_str = ", ".join([f"{q:.6f}" for q in optimized_qwots]); st.session_state.emp_str_input = final_qwot_str; log_message(f"Nominal QWOT updated: {final_qwot_str}"); st.success("Optimized design set as new Nominal. Optimized state cleared.")
            st.session_state.current_optimized_ep = None; st.session_state.optimization_ran_since_nominal_change = False; st.session_state.optimized_qwot_display = ""
            st.session_state.current_status = "Status: Idle (New Nominal Set)"; status_placeholder.info(st.session_state.current_status); update_display_info(None)
        except Exception as e: err_msg = f"ERROR during Set Nominal: {e}"; log_message(err_msg); st.error(f"Error setting nominal: {e}"); st.session_state.current_status = "Status: Set Nominal Failed (Error)"; status_placeholder.error(st.session_state.current_status)
        finally: st.rerun()

if not (calc_button or opt_button or remove_button or set_nominal_button):
    last_res_data = st.session_state.get('last_run_calculation_results')
    if last_res_data and last_res_data.get('res') is not None and last_res_data.get('ep') is not None:
        try:
            inputs = last_res_data['inputs']; active_targets = last_res_data['active_targets']
            fig = tracer_graphiques(res=last_res_data['res'], ep_actual=last_res_data['ep'], nH_r=inputs['nH_r'], nH_i=inputs['nH_i'], nL_r=inputs['nL_r'], nL_i=inputs['nL_i'], nSub=inputs['nSub'], active_targets_for_plot=active_targets, mse=last_res_data['mse'], is_optimized=last_res_data['is_optimized'], method_name=last_res_data['method_name'], res_optim_grid=last_res_data['res_optim_grid'])
            plot_placeholder.pyplot(fig); plt.close(fig)
        except Exception as e_plot: st.warning(f"Could not redraw previous plot: {e_plot}"); plot_placeholder.empty()
    else: plot_placeholder.info("Click 'Calculate Nominal' or 'Optimize N Passes' to generate results.")
    current_ep_display = None
    if st.session_state.get('optimization_ran_since_nominal_change') and st.session_state.get('current_optimized_ep') is not None: current_ep_display = st.session_state.current_optimized_ep
    elif st.session_state.get('last_run_calculation_results', {}).get('ep') is not None: current_ep_display = st.session_state.last_run_calculation_results['ep']
    update_display_info(current_ep_display)

current_status_message = st.session_state.get("current_status", "Status: Idle")
if "Failed" in current_status_message or "Error" in current_status_message: status_placeholder.error(current_status_message)
elif "Complete" in current_status_message or "Removed" in current_status_message or "Set" in current_status_message : status_placeholder.success(current_status_message)
elif "Idle" not in current_status_message: status_placeholder.info(current_status_message)
