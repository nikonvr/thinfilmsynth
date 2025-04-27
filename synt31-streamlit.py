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
LBFGSB_WORKER_OPTIONS: Dict[str, Any] = {
    'maxiter': 99,
    'ftol': 1.e-10,
    'gtol': 1e-7,
    'disp': False,
    'eps': 1.49e-8
}
LBFGSB_REOPT_OPTIONS: Dict[str, Any] = {
    'maxiter': 199,
    'ftol': 1e-10,
    'gtol': 1e-7,
    'disp': False,
    'eps': 1.49e-8
    }
PLOT_LAMBDA_STEP_DIVISOR: float = 10.0
MIN_SCALING_NM: float = 0.1
AUTO_REMOVE_THRESHOLD_NM: float = 1.0
INITIAL_SOBOL_REL_SCALE_LOWER: float = 0.1
INITIAL_SOBOL_REL_SCALE_UPPER: float = 2.0
PHASE1BIS_SCALING_NM: float = 10.0
PASS_P_BEST_REDUCTION_FACTOR: float = 0.8
PASS_SCALING_REDUCTION_BASE: float = 1.8

DEFAULT_NH_R: float = 2.35
DEFAULT_NH_I: float = 0.0
DEFAULT_NL_R: float = 1.46
DEFAULT_NL_I: float = 0.0
DEFAULT_NSUB: float = 1.52
DEFAULT_L0: float = 500.0
DEFAULT_L_RANGE_DEB: float = 400.0
DEFAULT_L_RANGE_FIN: float = 700.0
DEFAULT_L_STEP: float = 10.0
DEFAULT_SCALING_NM: float = 10.0

WORKER_COST_ARGS: Optional[Tuple] = None
WORKER_LBFGSB_ARGS: Optional[Tuple] = None

def init_worker_cost(cost_args_tuple: Tuple):
    global WORKER_COST_ARGS
    WORKER_COST_ARGS = cost_args_tuple

def init_worker_lbfgsb(cost_args_tuple: Tuple, lbfgsb_bounds: List[Tuple[Optional[float], Optional[float]]], min_thickness: float):
    global WORKER_COST_ARGS, WORKER_LBFGSB_ARGS
    WORKER_COST_ARGS = cost_args_tuple
    WORKER_LBFGSB_ARGS = (lbfgsb_bounds, min_thickness)

def cost_function_wrapper_for_map(ep_vector: np.ndarray) -> float:
    if WORKER_COST_ARGS is None:
        print(f"[PID:{os.getpid()}] ERROR: WORKER_COST_ARGS not initialized!", file=sys.stderr)
        return np.inf
    nH, nL, nSub_c, l_vec_optim, active_targets, min_thickness_phys_nm = WORKER_COST_ARGS
    return calculate_mse_for_optimization_penalized(
        ep_vector, nH, nL, nSub_c, l_vec_optim, active_targets, min_thickness_phys_nm
    )

def local_search_worker_wrapper_for_map(start_ep: np.ndarray) -> Dict[str, Any]:
    if WORKER_COST_ARGS is None or WORKER_LBFGSB_ARGS is None:
        print(f"[PID:{os.getpid()}] ERROR: WORKER_COST_ARGS or WORKER_LBFGSB_ARGS not initialized!", file=sys.stderr)
        return {'result': None, 'final_ep': start_ep, 'final_cost': np.inf,
                'success': False, 'start_ep': start_ep, 'pid': os.getpid(),
                'log_lines': ["[PID:{os.getpid()}] ERROR: Worker not initialized properly."], 'initial_cost': np.inf}
    cost_args_tuple = WORKER_COST_ARGS
    lbfgsb_bounds, min_thickness_phys_nm = WORKER_LBFGSB_ARGS
    return local_search_worker(
        start_ep=start_ep,
        cost_function=cost_function_wrapper_for_map,
        args_for_cost=(),
        lbfgsb_bounds=lbfgsb_bounds,
        min_thickness_phys_nm=min_thickness_phys_nm
    )

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
             prefix = "[?:??s] "
    full_message = prefix + str(message)
    st.session_state.log_messages.append(full_message)

def clear_log() -> None:
    st.session_state.log_messages = [f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Log cleared."]

def upper_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    exponent = math.floor(math.log2(n)) + 1
    return 1 << exponent

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
        M_layer = np.empty((2, 2), dtype=np.complex128)
        M_layer[0, 0] = M_layer_00
        M_layer[0, 1] = M_layer_01
        M_layer[1, 0] = M_layer_10
        M_layer[1, 1] = M_layer_11
        M = M_layer @ M
    return M

@numba.njit(fastmath=True, cache=True)
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

    for i_l in range(len(l_vec)):
        l_val = l_vec[i_l]
        if l_val <= 0:
            Rs_arr[i_l], Ts_arr[i_l] = np.nan, np.nan
            continue

        try:
            M = compute_stack_matrix(ep_vector, l_val, nH_complex, nL_complex)
            m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

            rs_num = (etainc * m00 - etasub * m11 + etainc * etasub * m01 - m10)
            rs_den = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)

            if np.abs(rs_den) < RT_DENOMINATOR_THRESHOLD:
                Rs_arr[i_l], Ts_arr[i_l] = np.nan, np.nan
            else:
                rs = rs_num / rs_den
                ts = (2 * etainc) / rs_den
                Rs_arr[i_l] = np.abs(rs)**2
                if real_etainc <= ZERO_THRESHOLD:
                     Ts_arr[i_l] = np.nan
                else:
                     Ts_arr[i_l] = (real_etasub / real_etainc) * np.abs(ts)**2
                     if not np.isfinite(Ts_arr[i_l]): Ts_arr[i_l] = np.nan

        except Exception:
            Rs_arr[i_l], Ts_arr[i_l] = np.nan, np.nan

    for i in range(len(Rs_arr)):
        if np.isnan(Rs_arr[i]): Rs_arr[i] = 0.0
        if np.isnan(Ts_arr[i]): Ts_arr[i] = 0.0

    return Rs_arr, Ts_arr

@numba.njit(fastmath=True, cache=True)
def get_target_points_indices(l_vec: np.ndarray, target_min: float, target_max: float) -> np.ndarray:
    if not l_vec.size: return np.empty(0, dtype=np.int64)
    return np.where((l_vec >= target_min) & (l_vec <= target_max))[0]

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

        ep_vector_np = np.ascontiguousarray(ep_vector, dtype=np.float64)
        l_vec_np = np.ascontiguousarray(l_vec, dtype=np.float64)

        if not np.all(np.isfinite(ep_vector_np)):
             return None
        if not np.all(np.isfinite(l_vec_np)) or np.any(l_vec_np <= 0):
              return None
        if ep_vector_np.ndim != 1 or l_vec_np.ndim != 1:
              return None

        Rs, Ts = calculate_RT_from_ep_core(ep_vector_np, nH_complex, nL_complex, nSub_complex, l_vec_np)

        if np.any(~np.isfinite(Rs)) or np.any(~np.isfinite(Ts)):
             pass

        return {'l': l_vec_np, 'Rs': Rs, 'Ts': Ts}
    except Exception as e:
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
        ep_initial[i] = multiplier * l0 / (4 * n_real)

    if not np.all(np.isfinite(ep_initial)):
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
        qwot_multipliers[:] = np.nan
        return qwot_multipliers
    if nH_r <= QWOT_N_THRESHOLD or nL_r <= QWOT_N_THRESHOLD:
        qwot_multipliers[:] = np.nan
        return qwot_multipliers

    for i in range(num_layers):
        n_real = nH_r if i % 2 == 0 else nL_r
        qwot_multipliers[i] = ep_vector[i] * (4 * n_real) / l0

    return qwot_multipliers

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
         return None, 0

    calculated_lambdas = res['l']
    calculated_Ts = res['Ts']

    if calculated_lambdas.size == 0 or calculated_Ts.size == 0:
         return None, 0
    if len(calculated_lambdas) != len(calculated_Ts):
        return None, 0

    finite_mask_all = np.isfinite(calculated_Ts)
    if not np.all(finite_mask_all):
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
             continue

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
            total_points_in_targets += len(calculated_Ts_in_zone)

    if total_points_in_targets > 0:
        mse = total_squared_error / total_points_in_targets
    elif active_targets:
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
        return np.inf

    below_min_mask = ep_vector < min_thickness_phys_nm
    penalty: float = 0.0
    if np.any(below_min_mask):
        penalty = PENALTY_BASE + np.sum((min_thickness_phys_nm - ep_vector[below_min_mask])**2) * PENALTY_FACTOR

    ep_vector_calc = np.maximum(ep_vector, min_thickness_phys_nm)

    try:
        res = calculate_RT_from_ep(ep_vector_calc, nH, nL, nSub, l_vec_optim)

        if res is None:
             return HIGH_COST_PENALTY + penalty

        calculated_lambdas = res['l']
        calculated_Ts = res['Ts']

        if np.any(~np.isfinite(calculated_Ts)):
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
            mse = HIGH_COST_MSE_ZERO_POINTS
        else:
            mse = total_squared_error / total_points_in_targets

        final_cost = mse + penalty

        if not np.isfinite(final_cost):
            return np.inf

        return final_cost

    except Exception as e:
        return np.inf + penalty

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
        try:
            initial_cost = cost_function(current_ep, *args_for_cost) if num_layers > 0 else np.inf
        except Exception as e:
             initial_cost = np.inf
        return current_ep, False, initial_cost, logs

    thin_layer_index: int = -1
    min_thickness_found: float = np.inf

    if target_layer_index is not None:
        if 0 <= target_layer_index < num_layers:
            thin_layer_index = target_layer_index
            min_thickness_found = current_ep[target_layer_index]
        else:
            target_layer_index = None

    if target_layer_index is None:
        eligible_indices = np.where(current_ep >= min_thickness_phys)[0]
        if eligible_indices.size > 0:
            min_idx_within_eligible = np.argmin(current_ep[eligible_indices])
            thin_layer_index = eligible_indices[min_idx_within_eligible]
            min_thickness_found = current_ep[thin_layer_index]

    if thin_layer_index == -1:
        try:
            initial_cost = cost_function(current_ep, *args_for_cost)
        except Exception as e:
            initial_cost = np.inf
        return current_ep, False, initial_cost, logs

    ep_after_merge: Optional[np.ndarray] = None
    merged_info: str = ""
    structure_changed: bool = False

    if thin_layer_index == 0:
        if num_layers >= 2:
            ep_after_merge = current_ep[2:].copy()
            structure_changed = True
        else:
             pass

    elif thin_layer_index == num_layers - 1:
        ep_after_merge = current_ep[:-1].copy()
        structure_changed = True

    else:
        if num_layers >= 3:
            merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
            ep_after_merge = np.concatenate((
                current_ep[:thin_layer_index - 1],
                np.array([merged_thickness]),
                current_ep[thin_layer_index + 2:]
            ))
            structure_changed = True
        else:
             pass

    final_ep = current_ep
    final_cost = np.inf
    success_overall = False

    if structure_changed and ep_after_merge is not None:
        num_layers_reopt = len(ep_after_merge)

        if num_layers_reopt == 0:
            return np.array([]), True, np.inf, logs

        ep_after_merge = np.maximum(ep_after_merge, min_thickness_phys)
        reopt_bounds = [(min_thickness_phys, None)] * num_layers_reopt
        x0_reopt = ep_after_merge
        reopt_args = args_for_cost

        try:
            reopt_result = minimize(cost_function, x0_reopt, args=reopt_args,
                                    method='L-BFGS-B', bounds=reopt_bounds,
                                    options=LBFGSB_REOPT_OPTIONS)
            reopt_success = reopt_result.success and np.isfinite(reopt_result.fun)
            reopt_cost = reopt_result.fun if reopt_success else np.inf

            if reopt_success:
                final_ep = np.maximum(reopt_result.x.copy(), min_thickness_phys)
                final_cost = reopt_cost
                success_overall = True
            else:
                final_ep = np.maximum(x0_reopt.copy(), min_thickness_phys)
                success_overall = False
                try:
                    final_cost = cost_function(final_ep, *reopt_args)
                except Exception as e_cost:
                    final_cost = np.inf

        except Exception as e_reopt:
            final_ep = np.maximum(x0_reopt.copy(), min_thickness_phys)
            success_overall = False
            try:
                 final_cost = cost_function(final_ep, *reopt_args)
            except Exception: final_cost = np.inf

    else:
        success_overall = False
        final_ep = current_ep
        try:
             final_cost = cost_function(final_ep, *args_for_cost)
        except Exception: final_cost = np.inf

    return final_ep, success_overall, final_cost, logs

def local_search_worker(
    start_ep: np.ndarray,
    cost_function: Callable[[np.ndarray], float],
    args_for_cost: Tuple,
    lbfgsb_bounds: List[Tuple[Optional[float], Optional[float]]],
    min_thickness_phys_nm: float
) -> Dict[str, Any]:
    worker_pid = os.getpid()
    start_time = time.time()
    log_lines: List[str] = []
    initial_cost: float = np.inf
    start_ep_checked: np.ndarray = np.array([])
    iter_count: int = 0
    result: OptimizeResult = OptimizeResult(x=start_ep, success=False, fun=np.inf, message="Worker did not run", nit=0)

    try:
        start_ep_local = np.asarray(start_ep)
        if start_ep_local.ndim != 1 or start_ep_local.size == 0:
             raise ValueError(f"Start EP is invalid (shape: {start_ep_local.shape}).")
        start_ep_checked = np.maximum(start_ep_local, min_thickness_phys_nm)

        if len(lbfgsb_bounds) != len(start_ep_checked):
            raise ValueError(f"Dimension mismatch: start_ep ({len(start_ep_checked)}) vs bounds ({len(lbfgsb_bounds)})")
        initial_cost = cost_function(start_ep_checked)

    except Exception as e:
        log_lines.append(f"  [PID:{worker_pid} ERROR Prep]: Preparing/calculating initial cost failed: {type(e).__name__}: {e}")
        result = OptimizeResult(x=start_ep, success=False, fun=np.inf, message=f"Worker Initial Cost Error: {e}", nit=0)
        end_time = time.time()
        worker_summary = f"Worker {worker_pid} finished in {end_time - start_time:.2f}s. StartCost=ERROR, FinalCost=inf, Success=False, Msg='Initial Cost Error'"
        log_lines.append(f"  [PID:{worker_pid} Summary]: {worker_summary}")
        return {'result': result, 'final_ep': start_ep, 'final_cost': np.inf, 'success': False, 'start_ep': start_ep, 'pid': worker_pid, 'log_lines': log_lines, 'initial_cost': np.inf}

    if not np.isfinite(initial_cost):
         result = OptimizeResult(x=start_ep_checked, success=False, fun=np.inf, message="Initial cost was non-finite", nit=0)
    else:
        try:
            minimize_start_time = time.time()
            result = minimize(
                cost_function,
                start_ep_checked,
                args=(),
                method='L-BFGS-B',
                bounds=lbfgsb_bounds,
                options=LBFGSB_WORKER_OPTIONS,
            )
            minimize_duration = time.time() - minimize_start_time
            minimize_success = getattr(result, 'success', 'N/A')
            minimize_nit = getattr(result, 'nit', 'N/A')
            minimize_fun = getattr(result, 'fun', 'N/A')
            iter_count = result.nit if hasattr(result, 'nit') else 0

            if result.x is not None:
                result.x = np.maximum(result.x, min_thickness_phys_nm)

        except Exception as e:
            end_time = time.time()
            result = OptimizeResult(x=start_ep_checked, success=False, fun=np.inf, message=f"Worker Exception during minimize: {e}", nit=iter_count)

    end_time = time.time()
    final_cost_raw = result.fun if hasattr(result, 'fun') else np.inf
    final_cost = final_cost_raw if np.isfinite(final_cost_raw) else np.inf
    lbfgsb_success = result.success if hasattr(result, 'success') else False
    success_status = lbfgsb_success and np.isfinite(final_cost)
    iterations = iter_count
    message_raw = result.message if hasattr(result, 'message') else "No message"
    message = message_raw.decode('utf-8', errors='ignore') if isinstance(message_raw, bytes) else str(message_raw)
    final_x = result.x if success_status and result.x is not None else start_ep_checked

    return {'result': result, 'final_ep': final_x, 'final_cost': final_cost,
            'success': success_status, 'start_ep': start_ep, 'pid': worker_pid,
            'log_lines': log_lines, 'initial_cost': initial_cost}


def _run_sobol_evaluation(
    num_layers: int,
    n_samples: int,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    min_thickness_phys_nm: float,
    cost_args_tuple: Tuple,
    phase_name: str = "Phase 1",
    progress_hook: Optional[Callable[[float], None]] = None
) -> List[Tuple[float, np.ndarray]]:
    start_time_eval = time.time()

    if lower_bounds is None or upper_bounds is None or len(lower_bounds) != num_layers or len(upper_bounds) != num_layers:
        raise ValueError(f"_run_sobol_evaluation requires valid bounds matching num_layers ({num_layers}) (phase: {phase_name}).")
    if num_layers == 0:
         return []
    if np.any(lower_bounds > upper_bounds):
        upper_bounds = np.maximum(lower_bounds, upper_bounds)
    lower_bounds = np.maximum(min_thickness_phys_nm, lower_bounds)
    upper_bounds = np.maximum(lower_bounds + ZERO_THRESHOLD, upper_bounds)

    ep_candidates: List[np.ndarray] = []
    if n_samples > 0:
        try:
            sampler = qmc.Sobol(d=num_layers, scramble=True)
            points_unit_cube = sampler.random(n=n_samples)
            ep_candidates_raw = qmc.scale(points_unit_cube, lower_bounds, upper_bounds)
            ep_candidates = [np.maximum(min_thickness_phys_nm, cand) for cand in ep_candidates_raw]
        except Exception as e_sobol:
             return []

    if not ep_candidates:
        return []

    costs: List[float] = []
    initial_results: List[Tuple[float, np.ndarray]] = []
    num_workers_eval = min(len(ep_candidates), os.cpu_count())
    eval_start_pool = time.time()

    try:
        with multiprocessing.Pool(processes=num_workers_eval,
                                  initializer=init_worker_cost,
                                  initargs=(cost_args_tuple,)) as pool:
            costs = pool.map(cost_function_wrapper_for_map, ep_candidates)
        eval_pool_time = time.time() - eval_start_pool

        if len(costs) != len(ep_candidates):
            costs = costs[:len(ep_candidates)]
        initial_results = list(zip(costs, ep_candidates))

    except Exception as e_pool:
        costs = []
        eval_start_seq = time.time()
        total_candidates = len(ep_candidates)
        nH_s, nL_s, nSub_c_s, l_vec_optim_s, active_targets_s, min_thickness_phys_nm_s = cost_args_tuple
        for i, cand in enumerate(ep_candidates):
            try:
                cost = calculate_mse_for_optimization_penalized(cand, nH_s, nL_s, nSub_c_s, l_vec_optim_s, active_targets_s, min_thickness_phys_nm_s)
                costs.append(cost)
            except Exception as e_seq:
                costs.append(np.inf)

            if (i + 1) % 50 == 0 or i == total_candidates - 1:
                seq_elapsed = time.time() - eval_start_seq
                if progress_hook:
                     current_progress = (i + 1) / total_candidates
                     progress_hook(current_progress)

        eval_seq_time = time.time() - eval_start_seq
        initial_results = list(zip(costs, ep_candidates))

    valid_initial_results = [(c, p) for c, p in initial_results if np.isfinite(c) and c < HIGH_COST_PENALTY]
    num_invalid = len(initial_results) - len(valid_initial_results)

    if not valid_initial_results:
        return []
    valid_initial_results.sort(key=lambda x: x[0])
    eval_time = time.time() - start_time_eval

    return valid_initial_results


def _run_parallel_local_search(
    p_best_starts: List[np.ndarray],
    args_for_cost_tuple: Tuple,
    lbfgsb_bounds: List[Tuple[Optional[float], Optional[float]]],
    min_thickness_phys_nm: float,
    progress_hook: Optional[Callable[[float], None]] = None
) -> List[Dict[str, Any]]:

    p_best_actual = len(p_best_starts)
    if p_best_actual == 0:
        return []
    start_time_local = time.time()
    lbfgsb_init_args = (args_for_cost_tuple, lbfgsb_bounds, min_thickness_phys_nm)
    local_results_raw: List[Dict[str, Any]] = []
    num_workers_local = min(p_best_actual, os.cpu_count())
    local_start_pool = time.time()

    try:
        with multiprocessing.Pool(processes=num_workers_local,
                                  initializer=init_worker_lbfgsb,
                                  initargs=lbfgsb_init_args) as pool:
            local_results_raw = list(pool.map(local_search_worker_wrapper_for_map, p_best_starts))
        local_pool_time = time.time() - local_start_pool
    except Exception as e_pool_local:
        log_with_elapsed_time(f"  ERROR during parallel local search pool execution: {type(e_pool_local).__name__}: {e_pool_local}\n{traceback.format_exc(limit=2)}")
        local_results_raw = []

    local_time = time.time() - start_time_local
    return local_results_raw


def _process_optimization_results(
    local_results_raw: List[Dict[str, Any]],
    initial_best_ep: Optional[np.ndarray],
    initial_best_cost: float
) -> Tuple[Optional[np.ndarray], float, OptimizeResult]:
    overall_best_cost: float = initial_best_cost if np.isfinite(initial_best_cost) else np.inf
    overall_best_ep: Optional[np.ndarray] = initial_best_ep.copy() if initial_best_ep is not None else None
    overall_best_result_obj: OptimizeResult = OptimizeResult(
        x=overall_best_ep,
        fun=overall_best_cost,
        success=(overall_best_ep is not None and np.isfinite(overall_best_cost)),
        message="Initial best (pre-local search)",
        nit=0
    )
    best_run_info: Dict[str, Any] = {'index': -1, 'cost': overall_best_cost}
    processed_results_count: int = 0

    for i, worker_output in enumerate(local_results_raw):
        run_idx = i + 1
        if isinstance(worker_output, dict) and 'log_lines' in worker_output:
            worker_log_lines = worker_output.get('log_lines', [])
        else:
            continue

        if 'result' in worker_output and 'final_ep' in worker_output and 'final_cost' in worker_output and 'success' in worker_output:
            res_obj = worker_output['result']
            final_cost_run = worker_output['final_cost']
            final_ep_run = worker_output['final_ep']
            success_run = worker_output['success']
            processed_results_count += 1

            if success_run and final_ep_run is not None and final_ep_run.size > 0 and final_cost_run < overall_best_cost:
                 overall_best_cost = final_cost_run
                 overall_best_ep = final_ep_run.copy()
                 overall_best_result_obj = res_obj
                 best_run_info = {'index': run_idx, 'cost': final_cost_run}
        else:
            pass

    if overall_best_ep is None or len(overall_best_ep) == 0:
        if initial_best_ep is not None and initial_best_ep.size > 0:
             overall_best_ep = initial_best_ep.copy()
             overall_best_cost = initial_best_cost if np.isfinite(initial_best_cost) else np.inf
             overall_best_result_obj = OptimizeResult(x=overall_best_ep, fun=overall_best_cost, success=False, message="Fell back to initial state", nit=0)
        else:
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
    progress_bar_hook: Optional[Any] = None,
    status_placeholder: Optional[Any] = None,
    current_best_mse: float = np.inf,
    current_best_layers: int = 0
) -> Tuple[Optional[np.ndarray], float, OptimizeResult]:

    run_id_str = f"Pass {run_number}/{total_passes}"
    start_time_run = time.time()
    nH = inputs['nH_r'] + 1j * inputs['nH_i']
    nL = inputs['nL_r'] + 1j * inputs['nL_i']
    nSub_c = inputs['nSub'] + 0j
    l_step_gui = inputs['l_step']
    l_min_overall = inputs['l_range_deb']
    l_max_overall = inputs['l_range_fin']
    n_samples = inputs['n_samples']

    num_layers: int = 0
    lower_bounds: np.ndarray = np.array([])
    upper_bounds: np.ndarray = np.array([])

    if run_number == 1:
        if ep_nominal_for_cost is None or ep_nominal_for_cost.size == 0: raise ValueError(f"Nominal thickness vector ({run_id_str}) is empty or None.")
        ep_ref_pass1 = np.maximum(MIN_THICKNESS_PHYS_NM, ep_nominal_for_cost)
        num_layers = len(ep_ref_pass1)
        lower_bounds = np.maximum(MIN_THICKNESS_PHYS_NM, ep_ref_pass1 * INITIAL_SOBOL_REL_SCALE_LOWER)
        upper_bounds = ep_ref_pass1 * INITIAL_SOBOL_REL_SCALE_UPPER
        upper_bounds = np.maximum(upper_bounds, lower_bounds + ZERO_THRESHOLD)
    else:
        if ep_reference_for_sobol is None or ep_reference_for_sobol.size == 0: raise ValueError(f"Reference thickness vector ({run_id_str}) is empty or None for absolute scaling.")
        if current_scaling is None or current_scaling <= 0: raise ValueError(f"Invalid current_scaling value ({current_scaling}) for absolute Sobol ({run_id_str}).")
        ep_ref = np.maximum(MIN_THICKNESS_PHYS_NM, ep_reference_for_sobol)
        num_layers = len(ep_ref)
        lower_bounds = np.maximum(MIN_THICKNESS_PHYS_NM, ep_ref - current_scaling)
        upper_bounds = ep_ref + current_scaling
        upper_bounds = np.maximum(upper_bounds, lower_bounds + ZERO_THRESHOLD)
    if num_layers == 0: raise ValueError(f"Could not determine number of layers for Sobol ({run_id_str}).")

    if l_min_overall <= 0 or l_max_overall <= 0: raise ValueError("Wavelength range limits must be positive.")
    if l_max_overall <= l_min_overall: raise ValueError("Wavelength range max must be greater than min.")
    num_points_approx = max(2, int(np.round((l_max_overall - l_min_overall) / l_step_gui)) + 1)
    try:
        l_vec_optim = np.geomspace(l_min_overall, l_max_overall, num_points_approx)
        l_vec_optim = l_vec_optim[(l_vec_optim > 0) & np.isfinite(l_vec_optim)]
        if not l_vec_optim.size: raise ValueError("Geomspace resulted in empty vector.")
    except Exception as e_geom:
         raise ValueError(f"Failed to generate geomspace optimization wavelength vector: {e_geom}")

    args_for_cost_tuple: Tuple = (nH, nL, nSub_c, l_vec_optim, active_targets, MIN_THICKNESS_PHYS_NM)

    valid_initial_results_p1 = _run_sobol_evaluation(
        num_layers, n_samples, lower_bounds, upper_bounds,
        MIN_THICKNESS_PHYS_NM,
        args_for_cost_tuple,
        phase_name=f"{run_id_str} Phase 1",
        progress_hook=None
    )

    top_p_results_combined: List[Tuple[float, np.ndarray]] = []
    if run_number == 1:
        num_to_select_p1 = min(p_best_this_pass, len(valid_initial_results_p1))
        if num_to_select_p1 == 0:
            top_p_results_combined = []
        else:
            top_p_results_p1 = valid_initial_results_p1[:num_to_select_p1]
            top_p_starts_p1 = [ep for cost, ep in top_p_results_p1]
            n_samples_per_point_raw = max(1, n_samples // p_best_this_pass) if p_best_this_pass > 0 else n_samples
            n_samples_per_point = upper_power_of_2(n_samples_per_point_raw)
            phase1bis_candidates: List[np.ndarray] = []
            for i, ep_start in enumerate(top_p_starts_p1):
                 lower_bounds_1bis = np.maximum(MIN_THICKNESS_PHYS_NM, ep_start - PHASE1BIS_SCALING_NM)
                 upper_bounds_1bis = ep_start + PHASE1BIS_SCALING_NM
                 upper_bounds_1bis = np.maximum(upper_bounds_1bis, lower_bounds_1bis + ZERO_THRESHOLD)
                 try:
                    sampler_1bis = qmc.Sobol(d=num_layers, scramble=True, seed=int(time.time()) + i)
                    points_unit_1bis = sampler_1bis.random(n=n_samples_per_point)
                    new_candidates_raw = qmc.scale(points_unit_1bis, lower_bounds_1bis, upper_bounds_1bis)
                    new_candidates = [np.maximum(MIN_THICKNESS_PHYS_NM, cand) for cand in new_candidates_raw]
                    phase1bis_candidates.extend(new_candidates)
                 except Exception as e_sobol1bis: pass

            results_1bis_raw_pairs: List[Tuple[float, np.ndarray]] = []
            if phase1bis_candidates:
                num_workers_1bis = min(len(phase1bis_candidates), os.cpu_count())
                try:
                    with multiprocessing.Pool(processes=num_workers_1bis, initializer=init_worker_cost, initargs=(args_for_cost_tuple,)) as pool:
                        costs_1bis = pool.map(cost_function_wrapper_for_map, phase1bis_candidates)
                    if len(costs_1bis) == len(phase1bis_candidates): results_1bis_raw_pairs = list(zip(costs_1bis, phase1bis_candidates))
                except Exception as e_pool_1bis: pass
            valid_results_1bis = [(c, p) for c, p in results_1bis_raw_pairs if np.isfinite(c) and c < HIGH_COST_PENALTY]
            combined_results = top_p_results_p1 + valid_results_1bis
            if combined_results:
                 combined_results.sort(key=lambda x: x[0])
                 num_to_select_final = min(p_best_this_pass, len(combined_results))
                 top_p_results_combined = combined_results[:num_to_select_final]
            else:
                 top_p_results_combined = []
    else:
        num_to_select = min(p_best_this_pass, len(valid_initial_results_p1))
        top_p_results_combined = valid_initial_results_p1[:num_to_select]

    selected_starts = [ep for cost, ep in top_p_results_combined]
    selected_costs = [cost for cost, ep in top_p_results_combined]

    overall_best_ep: Optional[np.ndarray] = None
    overall_best_cost: float = np.inf
    overall_best_result_obj: OptimizeResult = OptimizeResult(x=None, fun=np.inf, success=False, message="Optimization did not yield a valid result yet.")

    if not selected_starts:
        overall_best_ep = None
        overall_best_cost = np.inf
        overall_best_result_obj = OptimizeResult(x=None, fun=np.inf, success=False, message=f"Phase 2 skipped - no starts ({run_id_str})", nit=0)
    else:
        initial_best_ep_for_processing = selected_starts[0].copy()
        initial_best_cost_for_processing = selected_costs[0]
        lbfgsb_bounds = [(MIN_THICKNESS_PHYS_NM, None)] * num_layers

        local_results_raw = _run_parallel_local_search(
            selected_starts,
            args_for_cost_tuple,
            lbfgsb_bounds,
            MIN_THICKNESS_PHYS_NM,
            progress_hook=None
        )
        overall_best_ep, overall_best_cost, overall_best_result_obj = _process_optimization_results(
            local_results_raw,
            initial_best_ep_for_processing,
            initial_best_cost_for_processing
        )

    run_time_pass = time.time() - start_time_run
    return overall_best_ep, overall_best_cost, overall_best_result_obj

def setup_axis_grids(ax: plt.Axes) -> None:
    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.7, alpha=0.7)
    ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.minorticks_on()

def tracer_graphiques(
    res: Optional[Dict[str, np.ndarray]],
    ep_actual: Optional[np.ndarray],
    nH_r: float, nH_i: float, nL_r: float, nL_i: float, nSub: float,
    active_targets_for_plot: List[Dict[str, float]],
    mse: Optional[float],
    is_optimized: bool = False,
    method_name: str = "",
    res_optim_grid: Optional[Dict[str, np.ndarray]] = None
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    opt_method_str = f" ({method_name})" if method_name else ""
    window_title = f'Stack Results{opt_method_str}' if is_optimized else 'Nominal Stack Calculation Results'
    fig.suptitle(window_title, fontsize=14, weight='bold')

    num_layers = len(ep_actual) if ep_actual is not None else 0
    ep_cum = np.cumsum(ep_actual) if num_layers > 0 and ep_actual is not None else np.array([])
    total_thickness = ep_cum[-1] if num_layers > 0 else 0

    ax_spec = axes[0]
    ax_spec.set_title(f"Spectral Plot{opt_method_str}")
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel('Transmittance')
    ax_spec.set_ylim(bottom=-0.05, top=1.05)
    setup_axis_grids(ax_spec)

    if res and 'l' in res and 'Ts' in res and res['l'] is not None and res['Ts'] is not None and res['l'].size > 0:
        line_ts, = ax_spec.plot(res['l'], res['Ts'], label='Transmittance (Plot Grid)', linestyle='-', color='blue', linewidth=1.5)
        if len(res['l']) > 0: ax_spec.set_xlim(res['l'][0], res['l'][-1])

        target_lines_drawn = False
        if active_targets_for_plot:
            plotted_label = False
            for target in active_targets_for_plot:
                l_min, l_max = target['min'], target['max']
                t_max = target['target_max']
                target_t_min = target['target_min']

                x_coords = [l_min, l_max]
                y_coords = [target_t_min, t_max]
                label = 'Target Ramp' if not plotted_label else "_nolegend_"
                ax_spec.plot(x_coords, y_coords, 'r--', linewidth=1.5, alpha=0.8, label=label, zorder=5)
                plotted_label = True
                ax_spec.plot(x_coords, y_coords, marker='x', color='red', markersize=8, linestyle='none', label='_nolegend_', zorder=6)
                target_lines_drawn = True

                if res_optim_grid and 'l' in res_optim_grid and res_optim_grid['l'].size > 0 and 'Ts' in res_optim_grid:
                    indices_in_zone_optim = np.where((res_optim_grid['l'] >= l_min) & (res_optim_grid['l'] <= l_max))[0]
                    if indices_in_zone_optim.size > 0:
                        optim_lambdas_in_zone = res_optim_grid['l'][indices_in_zone_optim]
                        if abs(l_max - l_min) < MSE_TARGET_LAMBDA_DIFF_THRESHOLD:
                             optim_target_t_in_zone = np.full_like(optim_lambdas_in_zone, target_t_min)
                        else:
                             slope = (t_max - target_t_min) / (l_max - l_min)
                             optim_target_t_in_zone = target_t_min + slope * (optim_lambdas_in_zone - l_min)
                        ax_spec.plot(optim_lambdas_in_zone, optim_target_t_in_zone,
                                     marker='.', color='orangered', linestyle='none', markersize=4,
                                     alpha=0.7, label='_nolegend_', zorder=6)

        handles, labels = ax_spec.get_legend_handles_labels()
        if handles:
             ax_spec.legend(fontsize=9)

        mse_text = "MSE: N/A"
        if mse is not None and not np.isnan(mse):
             mse_text = f"MSE (vs Target, Optim grid) = {mse:.3e}"
        elif mse is None and active_targets_for_plot: mse_text = "MSE: Calculation Error"
        elif active_targets_for_plot: mse_text = "MSE: N/A (no target points)"
        else: mse_text = "MSE: N/A (no target defined)"

        ax_spec.text(0.98, 0.98, mse_text, transform=ax_spec.transAxes, ha='right', va='top', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
    else:
        ax_spec.text(0.5, 0.5, "No spectral data available", ha='center', va='center', transform=ax_spec.transAxes)

    ax_idx = axes[1]
    ax_idx.set_title("Refractive Index Profile")
    ax_idx.set_xlabel('Depth (from substrate) (nm)')
    ax_idx.set_ylabel("Real part of index (n')")
    setup_axis_grids(ax_idx)

    nSub_c = complex(nSub)
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

        last_layer_end_depth = current_depth
        x_coords_plot.append(last_layer_end_depth)
        y_coords_plot.append(1.0)
        x_coords_plot.append(last_layer_end_depth + margin)
        y_coords_plot.append(1.0)
    else:
        x_coords_plot.append(0)
        y_coords_plot.append(1.0)
        x_coords_plot.append(margin)
        y_coords_plot.append(1.0)

    ax_idx.plot(x_coords_plot, y_coords_plot, drawstyle='steps-post', label='Real n', color='purple', linewidth=1.5)
    ax_idx.set_xlim(x_coords_plot[0], x_coords_plot[-1])

    min_n_list = [1.0, np.real(nSub_c)] + n_real_layers
    max_n_list = [1.0, np.real(nSub_c)] + n_real_layers
    min_n = min(min_n_list) if min_n_list else 0.9
    max_n = max(max_n_list) if max_n_list else 2.5
    ax_idx.set_ylim(bottom=min_n - 0.1, top=max_n + 0.1)

    offset = (max_n - min_n) * 0.05 + 0.02
    common_text_opts: Dict[str, Any] = {'ha':'center', 'va':'bottom', 'fontsize':8, 'bbox':dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none')}
    sub_text = f"SUBSTRATE\nn={nSub_c.real:.3f}"
    if abs(nSub_c.imag) > ZERO_THRESHOLD: sub_text += f"{nSub_c.imag:+.3f}j"
    ax_idx.text(-margin / 2, np.real(nSub_c) + offset, sub_text, **common_text_opts)

    air_x_pos = (total_thickness + margin / 2) if num_layers > 0 else margin / 2
    ax_idx.text(air_x_pos, 1.0 + offset, "AIR\nn=1.0", **common_text_opts)

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
            if abs(k_val) > ZERO_THRESHOLD:
                 n_str += f"{k_val:+.3f}j"
            label = f"L{i + 1} ({layer_type}) n={n_str}"
            yticks_labels.append(label)

        ax_stack.set_yticks(bar_pos)
        ax_stack.set_yticklabels(yticks_labels, fontsize=8)
        ax_stack.invert_yaxis()

        max_ep_val = np.max(ep_actual) if ep_actual.size > 0 else 1.0
        fontsize = max(6, 9 - num_layers // 10)
        for i, bar in enumerate(bars):
            e_val = bar.get_width()
            ha_pos = 'left' if e_val < max_ep_val * 0.2 else 'right'
            x_text_pos = e_val * 1.05 if ha_pos == 'left' else e_val * 0.95
            text_color = 'black' if ha_pos == 'left' else 'white'
            ax_stack.text(x_text_pos, bar.get_y() + bar.get_height() / 2, f"{e_val:.2f} nm",
                          va='center', ha=ha_pos, color=text_color, fontsize=fontsize, weight='bold')

        ax_stack.set_ylim(bottom=num_layers - 0.5, top=-0.5)

    else:
         ax_stack.text(0.5, 0.5, "No layers defined", ha='center', va='center', fontsize=10, color='grey', transform=ax_stack.transAxes)
         ax_stack.set_yticks([])
         ax_stack.set_xticks([])

    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.95])
    return fig

def get_active_targets_from_state() -> Optional[List[Dict[str, float]]]:
    active_targets: List[Dict[str, float]] = []
    if 'targets' not in st.session_state:
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

                if overall_lambda_min is None or l_min < overall_lambda_min: overall_lambda_min = l_min
                if overall_lambda_max is None or l_max > overall_lambda_max: overall_lambda_max = l_max

            except (ValueError, TypeError) as e:
                st.error(f"Error in Spectral Target Zone {i+1}: {e}")
                return None

    try:
        calc_l_min_str = st.session_state.get('l_range_deb_input')
        calc_l_max_str = st.session_state.get('l_range_fin_input')
        if calc_l_min_str is not None and calc_l_max_str is not None:
            calc_l_min = float(calc_l_min_str)
            calc_l_max = float(calc_l_max_str)
            if overall_lambda_min is not None and overall_lambda_max is not None:
                 if overall_lambda_min < calc_l_min or overall_lambda_max > calc_l_max:
                     st.warning(f"Calculation range [{calc_l_min:.1f}-{calc_l_max:.1f} nm] does not fully cover active target range [{overall_lambda_min:.1f}-{overall_lambda_max:.1f} nm]. Results might be suboptimal or MSE misleading.")
    except (ValueError, TypeError, AttributeError):
         pass
    except Exception as e:
         pass

    return active_targets

def _validate_physical_inputs_from_state(require_optim_params: bool = True) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
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
             raise ValueError(f"GUI State Error: Input key '{state_key}' not found in session state.")

        raw_val = st.session_state[state_key]
        try:
            if field_type == str:
                 values[target_key] = str(raw_val).strip()
            else:
                 values[target_key] = field_type(raw_val)

            if target_key in ['n_samples', 'p_best', 'n_passes'] and values[target_key] < 1:
                 error_messages.append(f"'{target_key.replace('_', ' ').title()}' ({state_key}) must be >= 1.")
            if target_key == 'scaling_nm' and values[target_key] < 0:
                 error_messages.append(f"'Scaling (nm)' ({state_key}) must be >= 0.")
            if target_key == 'l_step' and values[target_key] <= 0:
                 error_messages.append(f"'λ Step' ({state_key}) must be > 0.")
            if target_key in ['l_range_deb', 'l0'] and values[target_key] <= 0:
                 error_messages.append(f"'{target_key.replace('_', ' ').title()}' ({state_key}) must be > 0.")
            if target_key == 'nH_r' and values[target_key] <= 0:
                 error_messages.append(f"'Material H (n real)' ({state_key}) must be > 0.")
            if target_key == 'nL_r' and values[target_key] <= 0:
                 error_messages.append(f"'Material L (n real)' ({state_key}) must be > 0.")
            if target_key == 'nSub' and values[target_key] <= 0:
                 error_messages.append(f"'Substrate (n real)' ({state_key}) must be > 0.")
            if target_key in ['nH_i', 'nL_i'] and values[target_key] < 0:
                 error_messages.append(f"Imaginary parts (k) ({state_key}) must be >= 0.")

        except (ValueError, TypeError) as e:
            error_messages.append(f"Invalid value for '{state_key}': '{raw_val}'. Expected: {field_type.__name__}. Error: {e}")

    if 'l_range_fin' in values and 'l_range_deb' in values:
         if values['l_range_fin'] < values['l_range_deb']:
              error_messages.append(f"λ End ({values['l_range_fin']:.1f}) must be >= λ Start ({values['l_range_deb']:.1f}).")
    if require_optim_params and 'p_best' in values and 'n_samples' in values:
         if values['p_best'] > values['n_samples']:
              error_messages.append(f"P Starts ({values['p_best']}) must be <= N Samples ({values['n_samples']}).")

    if error_messages:
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
             pass
    elif not np.all(np.isfinite(ep_actual_orig)):
        ep_corrected = np.nan_to_num(ep_actual_orig, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.array_equal(ep_actual_orig, ep_corrected):
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
    global status_placeholder, progress_placeholder, plot_placeholder

    st.session_state.current_status = f"Status: Running {'Optimized' if is_optimized else 'Nominal'} Calculation..."
    if status_placeholder: status_placeholder.info(st.session_state.current_status)
    progress_bar = None
    if progress_placeholder: progress_bar = progress_placeholder.progress(0)

    res_optim_grid: Optional[Dict[str, np.ndarray]] = None
    final_fig: Optional[plt.Figure] = None
    ep_calculated: Optional[np.ndarray] = None
    mse_display: Optional[float] = np.nan
    final_result_ep: Optional[np.ndarray] = None

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

        if progress_bar: progress_bar.progress(15)

        nH, nL, nSub_c, l_vec_plot_default, ep_actual_calc, ep_actual_orig = _prepare_calculation_data_st(inputs, ep_vector_to_use=ep_source_for_calc)
        final_result_ep = ep_actual_orig.copy()
        ep_calculated = ep_actual_orig

        l_vec_final_plot = l_vec_override if l_vec_override is not None else l_vec_plot_default

        if progress_bar: progress_bar.progress(25)
        start_rt_time = time.time()
        res_fine = calculate_RT_from_ep(ep_actual_calc, nH, nL, nSub_c, l_vec_final_plot)
        rt_time = time.time() - start_rt_time
        if progress_bar: progress_bar.progress(50)

        if res_fine is None:
            raise RuntimeError("Failed to calculate spectral data for plotting.")

        if active_targets_for_plot:
            l_min_overall = inputs['l_range_deb']
            l_max_overall = inputs['l_range_fin']
            l_step_gui = inputs['l_step']
            num_points_approx = max(2, int(np.round((l_max_overall - l_min_overall) / l_step_gui)) + 1)
            try:
                l_vec_optim_display = np.geomspace(l_min_overall, l_max_overall, num_points_approx)
                l_vec_optim_display = l_vec_optim_display[(l_vec_optim_display > 0) & np.isfinite(l_vec_optim_display)]
            except Exception:
                 l_vec_optim_display = np.array([])

            if l_vec_optim_display.size > 0:
                 res_optim_grid = calculate_RT_from_ep(ep_actual_calc, nH, nL, nSub_c, l_vec_optim_display)
                 if progress_bar: progress_bar.progress(75)

                 if res_optim_grid:
                     mse_display, num_pts_mse = calculate_final_mse(res_optim_grid, active_targets_for_plot)
                     if num_pts_mse <= 0 : mse_display = np.nan
                 else:
                     mse_display = None
            else:
                 mse_display = np.nan
        else:
            mse_display = None

        if progress_bar: progress_bar.progress(90)
        final_fig = tracer_graphiques(res_fine, ep_actual_orig,
                                      inputs['nH_r'], inputs['nH_i'], inputs['nL_r'], inputs['nL_i'], inputs['nSub'],
                                      active_targets_for_plot, mse_display,
                                      is_optimized=is_optimized, method_name=method_name,
                                      res_optim_grid=res_optim_grid)
        if progress_bar: progress_bar.progress(100)

        if is_optimized:
            st.session_state.optimization_ran_since_nominal_change = True
            st.session_state.current_optimized_ep = ep_actual_orig.copy() if ep_actual_orig is not None else None

        st.session_state.current_status = f"Status: {'Optimized' if is_optimized else 'Nominal'} Calculation Complete"
        if status_placeholder: status_placeholder.success(st.session_state.current_status)

        st.session_state.last_run_calculation_results = {
             'res': res_fine, 'ep': final_result_ep,
             'mse': mse_display, 'res_optim_grid': res_optim_grid, 'is_optimized': is_optimized,
             'method_name': method_name, 'inputs': inputs, 'active_targets': active_targets_for_plot
        }
        update_display_info(final_result_ep)

    except (ValueError, RuntimeError) as e:
        err_msg = f"ERROR (Input/Logic) in calculation: {e}"
        if status_placeholder: st.error(err_msg)
        st.session_state.current_status = f"Status: Calculation Failed (Input Error)"
        if status_placeholder: status_placeholder.error(st.session_state.current_status)
        if plot_placeholder: plot_placeholder.empty()
        st.session_state.last_run_calculation_results = {}
    except Exception as e:
        err_msg = f"ERROR (Unexpected) in calculation: {type(e).__name__}: {e}"
        tb_msg = traceback.format_exc()
        if status_placeholder: st.error(f"{err_msg}. See log/console for details.")
        st.session_state.current_status = f"Status: Calculation Failed (Unexpected Error)"
        if status_placeholder: status_placeholder.error(st.session_state.current_status)
        if plot_placeholder: plot_placeholder.empty()
        st.session_state.last_run_calculation_results = {}
    finally:
        if final_fig and plot_placeholder:
             plot_placeholder.pyplot(final_fig)
             plt.close(final_fig)
        elif plot_placeholder and 'err_msg' in locals():
             plot_placeholder.empty()
        if progress_placeholder: progress_placeholder.empty()


def run_optimization_st() -> None:
    global status_placeholder, progress_placeholder, plot_placeholder

    st.session_state.optim_start_time = time.time()
    clear_log()

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

        progress_max_steps = n_passes + 1
        current_progress_step = 0

        active_targets = get_active_targets_from_state()
        if active_targets is None: raise ValueError("Failed to retrieve/validate spectral targets.")
        if not active_targets: raise ValueError("No active spectral targets defined. Optimization requires at least one target.")

        nH_complex_nom = complex(inputs['nH_r'], inputs['nH_i'])
        nL_complex_nom = complex(inputs['nL_r'], inputs['nL_i'])
        ep_nominal_glob, _ = get_initial_ep(inputs['emp_str'], inputs['l0'], nH_complex_nom, nL_complex_nom)
        if ep_nominal_glob.size == 0: raise ValueError("Initial nominal QWOT stack is empty or invalid.")
        initial_nominal_layers = len(ep_nominal_glob)
        overall_best_layers = initial_nominal_layers

        try:
            nSub_c = complex(inputs['nSub'])
            l_min_overall = inputs['l_range_deb']; l_max_overall = inputs['l_range_fin']; l_step_gui = inputs['l_step']
            num_points_approx = max(2, int(np.round((l_max_overall - l_min_overall) / l_step_gui)) + 1)
            l_vec_optim_init = np.geomspace(l_min_overall, l_max_overall, num_points_approx)
            l_vec_optim_init = l_vec_optim_init[(l_vec_optim_init > 0) & np.isfinite(l_vec_optim_init)]
            if l_vec_optim_init.size == 0: raise ValueError("Failed to create optim vec for initial cost.")

            initial_nominal_cost = calculate_mse_for_optimization_penalized(ep_nominal_glob, nH_complex_nom, nL_complex_nom, nSub_c, l_vec_optim_init, active_targets, MIN_THICKNESS_PHYS_NM)
            overall_best_cost_final = initial_nominal_cost if np.isfinite(initial_nominal_cost) else np.inf

            if np.isfinite(overall_best_cost_final):
                 overall_best_ep_final = ep_nominal_glob.copy()
                 ep_ref_for_next_pass = ep_nominal_glob.copy()
            else:
                 overall_best_ep_final = None
                 ep_ref_for_next_pass = ep_nominal_glob.copy()

            st.session_state.current_status = f"Status: Initial | Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
            if status_placeholder: status_placeholder.info(st.session_state.current_status)

        except Exception as e_init_cost:
            overall_best_cost_final = np.inf
            overall_best_ep_final = None
            ep_ref_for_next_pass = ep_nominal_glob.copy()
            st.session_state.current_status = f"Status: Initial | Best MSE: N/A | Layers: {initial_nominal_layers}"
            if status_placeholder: status_placeholder.warning(st.session_state.current_status)

        for pass_num in range(1, n_passes + 1):
            run_id_str = f"Pass {pass_num}/{n_passes}"
            pass_status_prefix = f"Status: {run_id_str}"
            status_suffix = f"| Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
            st.session_state.current_status = f"{pass_status_prefix} - Starting... {status_suffix}"
            if status_placeholder: status_placeholder.info(st.session_state.current_status)
            if progress_bar: progress_bar.progress( current_progress_step / progress_max_steps )

            p_best_reduction = PASS_P_BEST_REDUCTION_FACTOR ** (pass_num - 1)
            p_best_for_this_pass = max(2, int(np.round(initial_p_best * p_best_reduction)))
            n_samples_this_pass = n_samples_base

            current_scaling_pass: Optional[float] = None
            ep_ref_sobol_pass: Optional[np.ndarray] = None
            if pass_num == 1:
                current_scaling_pass = None
                ep_ref_sobol_pass = None
            else:
                scale_reduction_factor = PASS_SCALING_REDUCTION_BASE**(pass_num - 2)
                current_scaling_pass = max(MIN_SCALING_NM, initial_scaling_nm / scale_reduction_factor)
                if ep_ref_for_next_pass is None:
                     raise RuntimeError(f"Cannot start {run_id_str}: No reference structure available from previous pass.")
                ep_ref_sobol_pass = ep_ref_for_next_pass

            pass_inputs = inputs.copy()
            pass_inputs['n_samples'] = n_samples_this_pass
            ep_this_pass, cost_this_pass, result_obj_this_pass = run_optimization_process(
                inputs=pass_inputs,
                active_targets=active_targets,
                ep_nominal_for_cost=ep_nominal_glob,
                p_best_this_pass=p_best_for_this_pass,
                ep_reference_for_sobol=ep_ref_sobol_pass,
                current_scaling=current_scaling_pass,
                run_number=pass_num,
                total_passes=n_passes,
                progress_bar_hook=None,
                status_placeholder=status_placeholder,
                current_best_mse=overall_best_cost_final,
                current_best_layers=overall_best_layers
            )

            current_progress_step += 1
            if progress_bar: progress_bar.progress(current_progress_step / progress_max_steps)

            new_best_found_this_pass = False
            if ep_this_pass is not None and ep_this_pass.size > 0 and np.isfinite(cost_this_pass):
                if cost_this_pass < overall_best_cost_final:
                    overall_best_cost_final = cost_this_pass
                    overall_best_ep_final = ep_this_pass.copy()
                    overall_best_layers = len(overall_best_ep_final)
                    final_successful_result_obj = result_obj_this_pass
                    new_best_found_this_pass = True

                ep_ref_for_next_pass = overall_best_ep_final.copy() if overall_best_ep_final is not None else ep_nominal_glob.copy()

                status_suffix = f"| Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
                if new_best_found_this_pass:
                    st.session_state.current_status = f"{pass_status_prefix} - Completed. New Best! {status_suffix}"
                    if status_placeholder: status_placeholder.info(st.session_state.current_status)
                else:
                    st.session_state.current_status = f"{pass_status_prefix} - Completed. No improvement. {status_suffix}"
                    if status_placeholder: status_placeholder.info(st.session_state.current_status)
            else:
                st.session_state.current_status = f"{pass_status_prefix} - FAILED. {status_suffix}"
                if status_placeholder: status_placeholder.warning(st.session_state.current_status)
                if ep_ref_for_next_pass is None:
                    raise RuntimeError(f"{run_id_str} failed and no prior successful result exists. Optimization aborted.")

        if progress_bar: progress_bar.progress(current_progress_step / progress_max_steps)
        st.session_state.current_status = f"Status: Post-processing (Auto-Remove)... | Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
        if status_placeholder: status_placeholder.info(st.session_state.current_status)

        if overall_best_ep_final is None:
             raise RuntimeError("Optimization finished, but no valid result found before post-processing.")

        max_auto_removals = len(overall_best_ep_final)

        temp_inputs_ar = _validate_physical_inputs_from_state(require_optim_params=False)
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
                break

            eligible_indices = np.where((current_ep_for_removal >= MIN_THICKNESS_PHYS_NM) &
                                        (current_ep_for_removal < AUTO_REMOVE_THRESHOLD_NM))[0]

            if eligible_indices.size > 0:
                thinnest_eligible_value = np.min(current_ep_for_removal[eligible_indices])
                thinnest_below_threshold_idx = np.where(current_ep_for_removal == thinnest_eligible_value)[0][0]
                thinnest_below_threshold_val = thinnest_eligible_value

                status_suffix = f"| Best MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
                st.session_state.current_status = f"Status: Auto-removing layer {auto_removed_count + 1}... {status_suffix}"
                if status_placeholder: status_placeholder.info(st.session_state.current_status)

                new_ep, success, cost_after, removal_logs = perform_single_thin_layer_removal(
                    current_ep_for_removal, MIN_THICKNESS_PHYS_NM,
                    calculate_mse_for_optimization_penalized, temp_args_for_cost_ar,
                    log_prefix="    [Auto Removal] ", target_layer_index=thinnest_below_threshold_idx
                )

                structure_actually_changed = (success and new_ep is not None and len(new_ep) < len(current_ep_for_removal))

                if structure_actually_changed:
                    current_ep_for_removal = new_ep.copy()
                    overall_best_ep_final = new_ep.copy()
                    overall_best_cost_final = cost_after if np.isfinite(cost_after) else np.inf
                    overall_best_layers = len(overall_best_ep_final)
                    auto_removed_count += 1
                else:
                    break
            else:
                break

        optimization_successful = True
        st.session_state.current_optimized_ep = overall_best_ep_final.copy() if overall_best_ep_final is not None else None
        st.session_state.optimization_ran_since_nominal_change = True

        final_qwot_str = "QWOT Error"
        if overall_best_ep_final is not None and len(overall_best_ep_final) > 0:
            try:
                l0_val = inputs['l0']; nH_r_val = inputs['nH_r']; nL_r_val = inputs['nL_r']
                optimized_qwots = calculate_qwot_from_ep(overall_best_ep_final, l0_val, nH_r_val, nL_r_val)
                if np.any(np.isnan(optimized_qwots)):
                    final_qwot_str = "QWOT N/A (NaN)"
                else:
                     final_qwot_str = ", ".join([f"{q:.4f}" for q in optimized_qwots])
            except Exception as qwot_calc_error:
                 pass
        else: final_qwot_str = "N/A (Empty Structure)"
        st.session_state.optimized_qwot_display = final_qwot_str

        final_method_name = f"{n_passes}-Pass Opt" + (f" + {auto_removed_count} AutoRm" if auto_removed_count > 0 else "")

        if progress_bar: progress_bar.progress(1.0)
        final_status_text = f"Status: Opt Complete{' (+AutoRm)' if auto_removed_count>0 else ''} | MSE: {overall_best_cost_final:.3e} | Layers: {overall_best_layers}"
        st.session_state.current_status = final_status_text
        if status_placeholder: status_placeholder.success(st.session_state.current_status)

        run_calculation_st(ep_vector_to_use=overall_best_ep_final, is_optimized=True, method_name=final_method_name)

    except (ValueError, RuntimeError) as e:
        err_msg = f"ERROR (Input/Logic) during optimization: {e}"
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
        if status_placeholder: st.error(f"{err_msg}. See log/console for details.")
        st.session_state.current_status = f"Status: Optimization Failed (Unexpected Error)"
        if status_placeholder: status_placeholder.error(st.session_state.current_status)
        optimization_successful = False
        st.session_state.current_optimized_ep = None
        st.session_state.optimization_ran_since_nominal_change = False
        st.session_state.optimized_qwot_display = "Optim Error Unexpected"
    finally:
        if progress_placeholder: progress_placeholder.empty()
        st.session_state.optim_start_time = None
        update_display_info(st.session_state.current_optimized_ep if optimization_successful else None)
        st.rerun()


def run_remove_layer_st() -> None:
    global status_placeholder, progress_placeholder, plot_placeholder

    st.session_state.current_status = "Status: Removing thinnest layer..."
    if status_placeholder: status_placeholder.info(st.session_state.current_status)
    progress_bar = None
    if progress_placeholder: progress_bar = progress_placeholder.progress(0)

    current_ep = st.session_state.get('current_optimized_ep')
    optim_ran = st.session_state.get('optimization_ran_since_nominal_change', False)
    removal_successful: bool = False
    final_ep_after_removal: Optional[np.ndarray] = None

    if current_ep is None or not optim_ran or len(current_ep) <= 1:
        if status_placeholder: st.error("No valid optimized structure (must have >= 2 layers) found to modify.")
        st.session_state.current_status = "Status: Removal Failed (No Structure/Not Optimized)"
        if status_placeholder: status_placeholder.error(st.session_state.current_status)
    else:
        final_ep_after_removal = current_ep.copy()
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

            new_ep, success, final_cost, removal_logs = perform_single_thin_layer_removal(
                current_ep, MIN_THICKNESS_PHYS_NM, calculate_mse_for_optimization_penalized,
                args_for_cost_tuple, log_prefix="  [Manual Removal] "
            )
            if progress_bar: progress_bar.progress(70)

            structure_actually_changed = (success and new_ep is not None and len(new_ep) < len(current_ep))

            if structure_actually_changed:
                st.session_state.current_optimized_ep = new_ep.copy()
                final_ep_after_removal = new_ep.copy()
                removal_successful = True

                final_qwot_str = "QWOT Error"
                try:
                    l0_val = inputs['l0']; nH_r_val = inputs['nH_r']; nL_r_val = inputs['nL_r']
                    optimized_qwots = calculate_qwot_from_ep(new_ep, l0_val, nH_r_val, nL_r_val)
                    if np.any(np.isnan(optimized_qwots)):
                         final_qwot_str = "QWOT N/A (NaN)"
                    else: final_qwot_str = ", ".join([f"{q:.4f}" for q in optimized_qwots])
                except Exception as qwot_calc_error: pass
                st.session_state.optimized_qwot_display = final_qwot_str

                layers_after = len(new_ep)
                st.session_state.current_status = f"Status: Layer Removed | MSE: {final_cost:.3e} | Layers: {layers_after}"
                if status_placeholder: status_placeholder.success(st.session_state.current_status)

                run_calculation_st(ep_vector_to_use=new_ep, is_optimized=True, method_name="Optimized (Post-Removal)")
                if progress_bar: progress_bar.progress(100)

            else:
                if status_placeholder: st.info("Could not find suitable layer or removal/re-optimization failed/unchanged.")
                current_ep_len = len(current_ep) if current_ep is not None else 0
                st.session_state.current_status = f"Status: Removal Skipped/Failed | Layers: {current_ep_len}"
                if status_placeholder: status_placeholder.warning(st.session_state.current_status)

        except (ValueError, RuntimeError) as e:
            err_msg = f"ERROR (Input/Logic) during layer removal: {e}"
            if status_placeholder: st.error(err_msg)
            st.session_state.current_status = "Status: Removal Failed (Input/Logic Error)"
            if status_placeholder: status_placeholder.error(st.session_state.current_status)
        except Exception as e:
            err_msg = f"ERROR (Unexpected) in layer removal: {type(e).__name__}: {e}"
            tb_msg = traceback.format_exc()
            if status_placeholder: st.error(f"{err_msg}. See log/console for details.")
            st.session_state.current_status = "Status: Removal Failed (Unexpected Error)"
            if status_placeholder: status_placeholder.error(st.session_state.current_status)
        finally:
            if progress_placeholder: progress_placeholder.empty()
            update_display_info(final_ep_after_removal)
            st.rerun()

def run_set_nominal_st() -> None:
    global status_placeholder, plot_placeholder

    st.session_state.current_status = "Status: Setting current as Nominal..."
    if status_placeholder: status_placeholder.info(st.session_state.current_status)

    current_ep = st.session_state.get('current_optimized_ep')
    optim_ran = st.session_state.get('optimization_ran_since_nominal_change', False)

    if current_ep is None or not optim_ran or len(current_ep) == 0:
        if status_placeholder: st.error("No valid optimized design is currently loaded.")
        st.session_state.current_status = "Status: Set Nominal Failed (No Design)"
        if status_placeholder: status_placeholder.error(st.session_state.current_status)
    else:
        try:
            inputs = _validate_physical_inputs_from_state(require_optim_params=False)
            l0_val = inputs['l0']; nH_r_val = inputs['nH_r']; nL_r_val = inputs['nL_r']

            optimized_qwots = calculate_qwot_from_ep(current_ep, l0_val, nH_r_val, nL_r_val)

            if np.any(np.isnan(optimized_qwots)):
                final_qwot_str = ""
                if status_placeholder: st.warning("Could not calculate valid QWOT multipliers (resulted in NaN). Nominal QWOT field not updated.")
            else:
                final_qwot_str = ", ".join([f"{q:.6f}" for q in optimized_qwots])
                st.session_state.emp_str_input = final_qwot_str
                if status_placeholder: st.success("Current optimized design set as new Nominal Structure (QWOT). Optimized state cleared.")

            st.session_state.current_optimized_ep = None
            st.session_state.optimization_ran_since_nominal_change = False
            st.session_state.optimized_qwot_display = ""
            st.session_state.current_status = "Status: Idle (New Nominal Set)"
            if status_placeholder: status_placeholder.info(st.session_state.current_status)
            st.session_state.last_run_calculation_results = {}

        except Exception as e:
            err_msg = f"ERROR during 'Set Nominal' operation: {type(e).__name__}: {e}"
            if status_placeholder: st.error(f"An error occurred setting nominal: {e}")
            st.session_state.current_status = "Status: Set Nominal Failed (Error)"
            if status_placeholder: status_placeholder.error(st.session_state.current_status)
        finally:
            update_display_info(None)
            st.rerun()

status_placeholder: Optional[Any] = None
progress_placeholder: Optional[Any] = None
plot_placeholder: Optional[Any] = None

def main() -> None:
    """Sets up the Streamlit UI and handles user interactions."""
    global status_placeholder, progress_placeholder, plot_placeholder # Allow modification by action functions

    # --- Initialize Session State (if not already set) ---
    # Ensures default values are set on the first run and preserved
    default_ui_values = {
        'nH_r': DEFAULT_NH_R, 'nH_i': DEFAULT_NH_I,
        'nL_r': DEFAULT_NL_R, 'nL_i': DEFAULT_NL_I,
        'nSub': DEFAULT_NSUB,
        'emp_str_input': DEFAULT_QWOT,
        'l0_input': DEFAULT_L0,
        'l_range_deb_input': DEFAULT_L_RANGE_DEB,
        'l_range_fin_input': DEFAULT_L_RANGE_FIN,
        'l_step_input': DEFAULT_L_STEP,
        'scaling_nm_input': DEFAULT_SCALING_NM,
        'optim_mode': DEFAULT_MODE,
        'targets': copy.deepcopy(DEFAULT_TARGETS),
        'log_messages': [f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Log Initialized."],
        'current_optimized_ep': None,
        'optimization_ran_since_nominal_change': False,
        'optim_start_time': None,
        'last_run_calculation_results': {},
        'current_status': "Status: Idle",
        'current_progress': 0,
        'num_layers_display': "Layers (Nominal): ?",
        'thinnest_layer_display': "- nm",
        'optimized_qwot_display': ""
    }
    default_params_for_mode = OPTIMIZATION_MODES.get(DEFAULT_MODE, list(OPTIMIZATION_MODES.values())[0]) # Fallback if default invalid
    default_ui_values['n_samples_input'] = default_params_for_mode['n_samples']
    default_ui_values['p_best_input'] = default_params_for_mode['p_best']
    default_ui_values['n_passes_input'] = default_params_for_mode['n_passes']

    for key, value in default_ui_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
        # Ensure numeric inputs stored as strings are corrected on load if needed
        if key in ['n_samples_input', 'p_best_input', 'n_passes_input']:
            try:
                 int(st.session_state[key])
            except (ValueError, TypeError):
                 st.session_state[key] = default_params_for_mode[key.replace('_input','')]


    # --- Page Config ---
    st.set_page_config(layout="wide", page_title="Thin Film Optimizer")
    st.title("Thin Film Stack Optimizer v2.25-LinTarget-GeomSpace (Streamlit)")

    # --- UI Definition ---
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Materials and Substrate", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                # Use st.session_state for values
                st.number_input("Material H (n real)", min_value=0.001, value=st.session_state.nH_r, step=0.01, format="%.3f", key='nH_r', help="Real part of high refractive index (must be > 0)")
                st.number_input("Material L (n real)", min_value=0.001, value=st.session_state.nL_r, step=0.01, format="%.3f", key='nL_r', help="Real part of low refractive index (must be > 0)")
                st.number_input("Substrate (n real)", min_value=0.001, value=st.session_state.nSub, step=0.01, format="%.3f", key='nSub', help="Real part of substrate refractive index (must be > 0)")
            with c2:
                st.number_input("Material H (k imag)", min_value=0.0, value=st.session_state.nH_i, step=0.001, format="%.4f", key='nH_i', help="Imaginary part (k) of high index (>= 0)")
                st.number_input("Material L (k imag)", min_value=0.0, value=st.session_state.nL_i, step=0.001, format="%.4f", key='nL_i', help="Imaginary part (k) of low index (>= 0)")
                st.caption("(n = n' + ik', k>=0)") # Clarify k represents absorption

        with st.expander("Stack (Nominal Definition)", expanded=True):
            st.text_area("Nominal Structure (QWOT Multipliers, comma-separated)", value=st.session_state.emp_str_input, key='emp_str_input', help="Define the starting structure using Quarter-Wave Optical Thickness multipliers relative to λ₀.")
            st.number_input("Centering λ₀ (QWOT, nm)", min_value=0.1, value=st.session_state.l0_input, step=1.0, format="%.1f", key='l0_input', help="Reference wavelength for QWOT calculation (must be > 0).")
            # Use st.session_state directly for the read-only display
            st.text_input("Optimized QWOT (Read-only)", value=st.session_state.optimized_qwot_display, disabled=True, key='opt_qwot_ro', help="QWOT multipliers of the last successfully optimized structure.")

    with col2:
        with st.expander("Calculation & Optimization Parameters", expanded=True):
            # Use st.session_state.optim_mode to set the index correctly
            optim_mode_options = list(OPTIMIZATION_MODES.keys())
            current_mode = st.session_state.optim_mode if st.session_state.optim_mode in optim_mode_options else DEFAULT_MODE
            current_mode_index = optim_mode_options.index(current_mode)

            # Define callback for radio button to update dependent fields
            def optim_mode_changed():
                st.session_state.optim_mode = st.session_state.optim_mode_radio # Update the mode first
                selected_params = OPTIMIZATION_MODES[st.session_state.optim_mode]
                # Update the input fields based on the newly selected mode
                st.session_state.n_samples_input = selected_params['n_samples']
                st.session_state.p_best_input = selected_params['p_best']
                st.session_state.n_passes_input = selected_params['n_passes']

            st.radio(
                "Optimization Mode Preset",
                options=optim_mode_options,
                index=current_mode_index,
                key='optim_mode_radio',
                horizontal=True,
                on_change=optim_mode_changed, # Use callback
                help="Select preset optimization parameters. Changes update fields below."
            )

            st.markdown("---") # Separator

            c1, c2, c3 = st.columns(3)
            with c1:
                st.number_input("λ Start (nm)", min_value=0.1, value=st.session_state.l_range_deb_input, step=1.0, format="%.1f", key='l_range_deb_input', help="Start wavelength for calculations and plots (must be > 0).")
            with c2:
                st.number_input("λ End (nm)", min_value=0.1, value=st.session_state.l_range_fin_input, step=1.0, format="%.1f", key='l_range_fin_input', help="End wavelength (must be >= λ Start).")
            with c3:
                st.number_input("λ Step (nm, Optim Grid)", min_value=0.01, value=st.session_state.l_step_input, step=0.1, format="%.2f", key='l_step_input', help="Wavelength step for optimization cost calculation grid (must be > 0).")
                st.caption(f"Plot uses λ Step / {PLOT_LAMBDA_STEP_DIVISOR}") # Use constant

            c1, c2, c3 = st.columns(3)
            with c1:
                # Use text_input here as defined previously, relies on validation later
                st.text_input("N Samples (Sobol)", value=st.session_state.n_samples_input, key='n_samples_input', help="Number of initial Sobol samples per pass (integer >= 1).")
            with c2:
                st.text_input("P Starts (L-BFGS-B)", value=st.session_state.p_best_input, key='p_best_input', help="Number of best Sobol points to start local search (integer >= 1, <= N Samples).")
            with c3:
                st.text_input("Optim Passes", value=st.session_state.n_passes_input, key='n_passes_input', help="Number of optimization passes (integer >= 1).")

            st.number_input("Scaling (nm, Pass 2+)", min_value=0.0, value=st.session_state.scaling_nm_input, step=0.1, format="%.2f", key='scaling_nm_input', help="Absolute Sobol search range (+/- nm) around previous best for passes > 1 (>= 0).")

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

                # Read directly from session state for display, changes trigger rerun
                enabled = cols[0].checkbox("", value=target_state['enabled'], key=f"target_{i}_enabled")
                cols[1].markdown(f"**{i+1}**") # Display zone number
                l_min_str = cols[2].text_input("", value=str(target_state['min']), key=f"target_{i}_min", label_visibility="collapsed")
                l_max_str = cols[3].text_input("", value=str(target_state['max']), key=f"target_{i}_max", label_visibility="collapsed")
                t_min_str = cols[4].text_input("", value=str(target_state['target_min']), key=f"target_{i}_tmin", label_visibility="collapsed")
                t_max_str = cols[5].text_input("", value=str(target_state['target_max']), key=f"target_{i}_tmax", label_visibility="collapsed")

                # Update the session state for this target directly (widgets handle this implicitly via their key)
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
    # The action functions (e.g., run_calculation_st) now handle their own UI updates
    # and rerun if necessary.
    if calc_button_pressed:
        st.session_state.current_optimized_ep = None
        st.session_state.optimization_ran_since_nominal_change = False
        st.session_state.optimized_qwot_display = ""
        run_calculation_st(ep_vector_to_use=None, is_optimized=False)
        # No rerun needed here typically, run_calculation_st updates placeholders

    elif opt_button_pressed:
        run_optimization_st() # Handles rerun internally

    elif remove_button_pressed:
        run_remove_layer_st() # Handles rerun internally

    elif set_nominal_button_pressed:
        run_set_nominal_st() # Handles rerun internally

    elif clear_log_pressed:
        clear_log()
        st.rerun()

    else:
        # --- No button pressed: Redraw last plot if available ---
        last_res_data = st.session_state.get('last_run_calculation_results')
        if last_res_data and isinstance(last_res_data, dict) and last_res_data.get('res') is not None and last_res_data.get('ep') is not None:
            try:
                required_keys = ['res', 'ep', 'inputs', 'active_targets', 'mse', 'is_optimized', 'method_name', 'res_optim_grid']
                # Check if the stored result has all necessary components
                if all(key in last_res_data for key in required_keys) and last_res_data['inputs'] is not None:
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
                    pass
            except Exception as e_plot:
                st.warning(f"Could not redraw previous plot: {e_plot}")
                plot_placeholder.empty()
        else:
             plot_placeholder.info("Click 'Calculate Nominal' or 'Optimize N Passes' to generate results.")

        # --- Update display info and status on every rerun (if no action taken) ---
        current_ep_display: Optional[np.ndarray] = None
        if st.session_state.get('optimization_ran_since_nominal_change') and st.session_state.get('current_optimized_ep') is not None:
            current_ep_display = st.session_state.current_optimized_ep
        elif st.session_state.get('last_run_calculation_results', {}).get('ep') is not None:
             current_ep_display = st.session_state.last_run_calculation_results['ep']
        # Update display info (layer count, thinnest) based on current state
        update_display_info(current_ep_display)

        # Update status display from session state
        current_status_message = st.session_state.get("current_status", "Status: Idle")
        if status_placeholder: # Check if placeholder exists before using
            if "Failed" in current_status_message or "Error" in current_status_message:
                status_placeholder.error(current_status_message)
            elif "Complete" in current_status_message or "Removed" in current_status_message or "Set" in current_status_message or "Successful" in current_status_message :
                status_placeholder.success(current_status_message)
            elif "Idle" not in current_status_message:
                 status_placeholder.info(current_status_message)
            else:
                status_placeholder.empty()


# --- Main Execution Guard ---
if __name__ == "__main__":
    # Setup for Multiprocessing (important for compatibility, especially Windows)
    multiprocessing.freeze_support()

    # Run the Streamlit App defined in the main function
    main()
