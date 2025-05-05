import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax.lax import scan, cond
import numpy as np
import pandas as pd
import functools
from typing import Union, Tuple, Dict, List, Any, Callable, Optional, Sequence
from scipy.optimize import minimize, OptimizeResult
import json
import os
import datetime
import traceback
import time

# --- Configuration Constants ---

jax.config.update("jax_enable_x64", True)

MIN_THICKNESS_PHYS_NM = 0.01
BASE_NEEDLE_THICKNESS_NM = 0.1
DEFAULT_NEEDLE_SCAN_STEP_NM = 2.0
AUTO_NEEDLES_PER_CYCLE = 5
AUTO_MAX_CYCLES = 5
MSE_IMPROVEMENT_TOLERANCE = 1e-9
EXCEL_FILE_PATH = "indices.xlsx" # Consider making this configurable

# --- Type Definitions ---

MaterialIdentifier = Union[str, complex, float, int]
MaterialInputType = Union[MaterialIdentifier, Tuple[np.ndarray, np.ndarray, np.ndarray]]
LayerVector = Union[np.ndarray, List[float], Tuple[float, ...]]
WavelengthVector = Union[np.ndarray, List[float], Tuple[float, ...]]
TargetDict = Dict[str, float] # E.g., {'min': 400, 'max': 500, 'target_min': 1.0, 'target_max': 1.0}

# --- Core Material Handling ---

class MaterialManager:
    """Handles loading, caching, and retrieving refractive indices."""

    def __init__(self, excel_file_path: str = EXCEL_FILE_PATH):
        self.excel_file_path = excel_file_path
        self._material_cache = {} # Cache raw data loaded from Excel
        self._predefined_funcs = {
            "FUSED SILICA": self._get_n_fused_silica_static,
            "BK7": self._get_n_bk7_static,
            "D263": self._get_n_d263_static,
            "AIR": lambda l_nm: jnp.ones_like(l_nm, dtype=jnp.complex128) # Add Air explicitly
        }
        self._lru_load_excel = functools.lru_cache(maxsize=32)(self._load_material_data_from_xlsx_sheet_impl)
        self._lru_interp_nk = jax.jit(functools.lru_cache(maxsize=128)(self._interp_nk_impl))

    @staticmethod
    @jax.jit
    def _get_n_fused_silica_static(l_nm: jnp.ndarray) -> jnp.ndarray:
        l_um_sq = (l_nm / 1000.0)**2
        n_sq = 1.0 + (0.6961663 * l_um_sq) / (l_um_sq - 0.0684043**2) + \
                     (0.4079426 * l_um_sq) / (l_um_sq - 0.1162414**2) + \
                     (0.8974794 * l_um_sq) / (l_um_sq - 9.896161**2)
        n = jnp.sqrt(n_sq)
        k = jnp.zeros_like(n)
        return n + 1j * k

    @staticmethod
    @jax.jit
    def _get_n_bk7_static(l_nm: jnp.ndarray) -> jnp.ndarray:
        l_um_sq = (l_nm / 1000.0)**2
        n_sq = 1.0 + (1.03961212 * l_um_sq) / (l_um_sq - 0.00600069867) + \
                     (0.231792344 * l_um_sq) / (l_um_sq - 0.0200179144) + \
                     (1.01046945 * l_um_sq) / (l_um_sq - 103.560653)
        n = jnp.sqrt(n_sq)
        k = jnp.zeros_like(n)
        return n + 1j * k

    @staticmethod
    @jax.jit
    def _get_n_d263_static(l_nm: jnp.ndarray) -> jnp.ndarray:
        n = jnp.full_like(l_nm, 1.523, dtype=jnp.float64)
        k = jnp.zeros_like(n)
        return n + 1j * k

    def _load_material_data_from_xlsx_sheet_impl(self, sheet_name: str) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Loads and caches material data from a specific sheet."""
        logs = [] # Logging would need a separate mechanism outside this class
        try:
            df = pd.read_excel(self.excel_file_path, sheet_name=sheet_name, header=None)
            numeric_df = df.apply(pd.to_numeric, errors='coerce')
            numeric_df = numeric_df.dropna(how='all')
            if numeric_df.shape[1] < 3:
                raise ValueError(f"Sheet '{sheet_name}' needs at least 3 columns (Wavelength, n, k).")
            numeric_df = numeric_df.dropna(subset=[0, 1, 2])
            numeric_df = numeric_df.sort_values(by=0) # Sort by wavelength

            l_nm = numeric_df.iloc[:, 0].values.astype(np.float64)
            n = numeric_df.iloc[:, 1].values.astype(np.float64)
            k = numeric_df.iloc[:, 2].values.astype(np.float64)

            if len(l_nm) == 0:
                raise ValueError(f"No valid numeric data found in sheet '{sheet_name}'.")

            # Store in cache (as JAX arrays for direct use)
            l_nm_jnp, n_jnp, k_jnp = map(jnp.asarray, (l_nm, n, k))
            #print(f"Loaded {sheet_name}: {len(l_nm_jnp)} points.") # Replace with proper logging
            return l_nm_jnp, n_jnp, k_jnp

        except FileNotFoundError:
            raise FileNotFoundError(f"Material file not found: {self.excel_file_path}") from None
        except ValueError as ve:
            raise ValueError(f"Value error reading sheet '{sheet_name}': {ve}") from ve
        except Exception as e:
            raise RuntimeError(f"Error reading sheet '{sheet_name}': {type(e).__name__} - {e}") from e

    @staticmethod
    @jax.jit
    def _interp_nk_impl(l_target: jnp.ndarray, l_data: jnp.ndarray, n_data: jnp.ndarray, k_data: jnp.ndarray) -> jnp.ndarray:
        """Interpolates n and k using JAX."""
        # Ensure data is sorted (should be done at load time, but double-check)
        # Note: JAX interp requires target points to be within the data range,
        # consider adding extrapolation or bounds handling if needed.
        n_interp = jnp.interp(l_target, l_data, n_data)
        k_interp = jnp.interp(l_target, l_data, k_data)
        return n_interp + 1j * k_interp

    def get_nk_array(self, material_definition: MaterialInputType, l_vec_target: jnp.ndarray) -> jnp.ndarray:
        """
        Gets the complex refractive index array for target wavelengths.

        Handles constants, predefined materials (strings), Excel sheets (strings),
        or provided data tuples (l_np, n_np, k_np).
        """
        if isinstance(material_definition, (complex, float, int)):
            return jnp.full(l_vec_target.shape, jnp.asarray(material_definition, dtype=jnp.complex128))

        elif isinstance(material_definition, str):
            mat_upper = material_definition.upper()
            if mat_upper in self._predefined_funcs:
                return self._predefined_funcs[mat_upper](l_vec_target)
            else:
                # Assume it's an Excel sheet name
                try:
                    l_data, n_data, k_data = self._lru_load_excel(material_definition)
                    # Ensure data arrays are treated as static by JAX for interpolation cache
                    return self._lru_interp_nk(l_vec_target, l_data, n_data, k_data)
                except Exception as e:
                    # Log error appropriately
                    raise ValueError(f"Could not load or process material '{material_definition}': {e}") from e

        elif isinstance(material_definition, tuple) and len(material_definition) == 3:
            l_data, n_data, k_data = material_definition
            # Convert to JAX arrays and ensure sorted
            try:
                l_data_jnp, n_data_jnp, k_data_jnp = map(jnp.asarray, (l_data, n_data, k_data))
                # Sort if not already sorted (important for interp)
                sort_indices = jnp.argsort(l_data_jnp)
                l_data_jnp_sorted = l_data_jnp[sort_indices]
                n_data_jnp_sorted = n_data_jnp[sort_indices]
                k_data_jnp_sorted = k_data_jnp[sort_indices]
                # Ensure data arrays are treated as static by JAX for interpolation cache
                return self._lru_interp_nk(l_vec_target, l_data_jnp_sorted, n_data_jnp_sorted, k_data_jnp_sorted)
            except Exception as e:
                raise TypeError(f"Invalid data format in material tuple: {e}") from e
        else:
            raise TypeError(f"Unsupported material definition type: {type(material_definition)}")

    def get_nk_at_lambda(self, material_definition: MaterialInputType, l_nm_target: float) -> complex:
        """Gets the complex refractive index at a single wavelength."""
        l_vec_target_jnp = jnp.array([l_nm_target], dtype=jnp.float64)
        nk_array = self.get_nk_array(material_definition, l_vec_target_jnp)
        return complex(nk_array[0])

    def get_available_materials_from_excel(self) -> List[str]:
        """Lists available material sheets from the Excel file."""
        try:
            xl = pd.ExcelFile(self.excel_file_path)
            # Basic filtering, might need refinement
            sheet_names = [name for name in xl.sheet_names if not name.lower().startswith("sheet")]
            return sheet_names
        except FileNotFoundError:
            # Log appropriately
            return []
        except Exception as e:
            # Log appropriately
            print(f"Error reading sheet names from {self.excel_file_path}: {e}")
            return []

    def clear_cache(self):
        """Clears LRU caches."""
        self._lru_load_excel.cache_clear()
        self._lru_interp_nk.cache_clear()
        print("MaterialManager caches cleared.") # Replace with logging


# --- Core Optical Calculations ---

class OpticalCalculator:
    """Performs optical calculations using the transfer matrix method."""

    def __init__(self, material_manager: MaterialManager):
        self.material_manager = material_manager

    @staticmethod
    @jax.jit
    def _compute_layer_matrix_scan_step(carry_matrix: jnp.ndarray, layer_data: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, None]:
        """JAX scan function step for multiplying layer matrices."""
        thickness, Ni, l_val = layer_data
        eta = Ni
        safe_l_val = jnp.maximum(l_val, 1e-9) # Avoid division by zero
        phi = (2 * jnp.pi / safe_l_val) * (Ni * thickness)
        cos_phi = jnp.cos(phi)
        sin_phi = jnp.sin(phi)

        def compute_M_layer(thickness_: jnp.ndarray) -> jnp.ndarray:
            # Avoid division by zero or near-zero eta
            safe_eta = jnp.where(jnp.abs(eta) < 1e-12, 1e-12 + 0j, eta)
            M_layer = jnp.array([
                [cos_phi,           (1j / safe_eta) * sin_phi],
                [1j * eta * sin_phi, cos_phi]
            ], dtype=jnp.complex128)
            # Matrix multiplication order: M_layer comes first for light from substrate->air
            # Original code was M_layer @ carry_matrix which implies light from air->substrate
            # Sticking to original code's convention:
            return M_layer @ carry_matrix # Assumes light incidence from air (index 0)

        def compute_identity(thickness_: jnp.ndarray) -> jnp.ndarray:
            # If thickness is negligible, matrix is identity
            return carry_matrix

        # Apply cond based on thickness being significant
        new_matrix = cond(
            thickness > 1e-12, # Use a small tolerance instead of zero
            compute_M_layer,
            compute_identity,
            thickness # Operand passed to the functions
        )
        return new_matrix, None # Scan requires returning carry and an optional output per step

    @classmethod
    @jax.jit
    def compute_stack_matrix(cls, ep_vector: jnp.ndarray, layer_indices_at_lval: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
        """Computes the total characteristic matrix for a stack at one wavelength."""
        num_layers = len(ep_vector)
        # Prepare data for scan: (thicknesses, indices, wavelength repeated)
        # Wavelength needs to be repeated for each layer if scan step expects it
        layers_scan_data = (ep_vector, layer_indices_at_lval, jnp.full(num_layers, l_val))

        # Initial matrix is identity matrix
        M_initial = jnp.eye(2, dtype=jnp.complex128)

        # Use jax.lax.scan to iteratively multiply matrices
        # The order of layers in scan corresponds to light propagation direction
        # Assuming ep_vector[0] is the first layer light hits after incident medium
        M_final, _ = scan(cls._compute_layer_matrix_scan_step, M_initial, layers_scan_data)

        return M_final

    @classmethod
    @jax.jit
    def calculate_single_wavelength_T(cls,
                                      l_val: jnp.ndarray,
                                      ep_vector_contig: jnp.ndarray,
                                      layer_indices_at_lval: jnp.ndarray,
                                      nSub_at_lval: jnp.ndarray,
                                      nInc_at_lval: jnp.ndarray = jnp.array(1.0 + 0j, dtype=jnp.complex128)
                                      ) -> jnp.ndarray:
        """Calculates Transmittance (T) for a single wavelength."""
        etasub = nSub_at_lval
        etainc = nInc_at_lval # Allow specifying incident medium index

        def calculate_for_valid_l(l_: jnp.ndarray) -> jnp.ndarray:
            # Compute the total matrix for this wavelength
            M = cls.compute_stack_matrix(ep_vector_contig, layer_indices_at_lval, l_)

            # Extract matrix elements
            m00, m01, m10, m11 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

            # Calculate transmission coefficient (ts) denominator
            ts_denominator = (etainc * m00 + etasub * m11 + etainc * etasub * m01 + m10)
            ts_denominator_abs = jnp.abs(ts_denominator)

            # Avoid division by zero
            safe_denominator = jnp.where(ts_denominator_abs < 1e-12, 1e-12 + 0j, ts_denominator)

            # Transmission coefficient
            ts = (2.0 * etainc) / safe_denominator

            # Transmittance calculation (ensure using real parts for power flow)
            real_etasub = jnp.real(etasub)
            real_etainc = jnp.real(etainc)

            # Avoid division by zero if incident medium is absorbing (real_etainc -> 0)
            safe_real_etainc = jnp.where(jnp.abs(real_etainc) < 1e-9, 1e-9, real_etainc)

            Ts = (real_etasub / safe_real_etainc) * jnp.real(ts * jnp.conj(ts)) # Take real part at the end

            # Return NaN or 0 if denominator was zero
            return jnp.where(ts_denominator_abs < 1e-12, jnp.nan, Ts)

        def calculate_for_invalid_l(l_: jnp.ndarray) -> jnp.ndarray:
            # Return NaN if wavelength is invalid (e.g., zero or negative)
            return jnp.nan

        # Use cond to handle potentially invalid wavelength values
        Ts_result = cond(l_val > 1e-9, calculate_for_valid_l, calculate_for_invalid_l, l_val)

        return Ts_result

    # Note: Added nInc_arr parameter
    @functools.partial(jax.jit, static_argnums=(0,)) # Indicate 'self' is static
    def calculate_T_spectrum(self,
                             ep_vector: LayerVector,
                             material_sequence: Sequence[MaterialIdentifier],
                             nSub_material: MaterialIdentifier,
                             l_vec: WavelengthVector,
                             nInc_material: MaterialIdentifier = "AIR" # Default incident medium
                             ) -> Dict[str, np.ndarray]:
        """
        Calculates the Transmittance (T) spectrum for a given stack design.

        Args:
            ep_vector: Sequence of physical layer thicknesses (nm).
            material_sequence: Sequence of material identifiers corresponding to ep_vector.
            nSub_material: Material identifier for the substrate.
            l_vec: Sequence of wavelengths (nm) to calculate for.
            nInc_material: Material identifier for the incident medium (default: Air).

        Returns:
            A dictionary {'l': wavelengths_np, 'Ts': transmittance_np}.
        """
        l_vec_jnp = jnp.asarray(l_vec, dtype=jnp.float64)
        ep_vector_jnp = jnp.asarray(ep_vector, dtype=jnp.float64)

        if not l_vec_jnp.size:
            return {'l': np.array([]), 'Ts': np.array([])}
        if len(ep_vector_jnp) != len(material_sequence):
            raise ValueError("Length mismatch: ep_vector and material_sequence.")

        try:
            # Get substrate and incident medium index arrays
            nSub_arr = self.material_manager.get_nk_array(nSub_material, l_vec_jnp)
            nInc_arr = self.material_manager.get_nk_array(nInc_material, l_vec_jnp)

            # Get layer index arrays
            if ep_vector_jnp.size == 0:
                # Handle empty stack case (bare substrate)
                layer_indices_arr_transposed = jnp.empty((len(l_vec_jnp), 0), dtype=jnp.complex128)
            else:
                layer_indices_list = []
                for mat_def in material_sequence:
                    layer_indices_list.append(self.material_manager.get_nk_array(mat_def, l_vec_jnp))
                # Stack along layers, then transpose for vmap: shape becomes (num_wavelengths, num_layers)
                layer_indices_arr_transposed = jnp.stack(layer_indices_list, axis=0).T

            # Vectorize the single wavelength calculation
            # in_axes specifies how arguments map over the vectorized dimension (wavelength)
            # None means the argument is broadcasted (ep_vector)
            Ts_arr = vmap(self.calculate_single_wavelength_T, in_axes=(0, None, 0, 0, 0))(
                l_vec_jnp, ep_vector_jnp, layer_indices_arr_transposed, nSub_arr, nInc_arr
            )

            # Convert NaNs resulting from calculations (e.g., zero denominator) to 0.0
            Ts_final = jnp.nan_to_num(Ts_arr, nan=0.0)

        except Exception as e:
            # Log error appropriately
            raise RuntimeError(f"Error during T spectrum calculation: {e}") from e

        return {'l': np.asarray(l_vec_jnp), 'Ts': np.asarray(Ts_final)}


# --- Design Utilities ---

def calculate_initial_ep(material_manager: MaterialManager,
                         multipliers: LayerVector,
                         l0: float,
                         material_sequence: Sequence[MaterialIdentifier]
                         ) -> np.ndarray:
    """Calculates initial physical thicknesses from QWOT multipliers."""
    num_layers = len(multipliers)
    if len(material_sequence) != num_layers:
         raise ValueError("Length mismatch: multipliers and material_sequence.")

    ep_initial = np.zeros(num_layers, dtype=np.float64)
    if l0 <= 0:
        print("Warning: l0 <= 0 in calculate_initial_ep. Thicknesses set to 0.") # Log properly
        return ep_initial

    valid_indices = True
    n_reals_at_l0 = []
    for i, mat_name in enumerate(material_sequence):
        try:
            n_complex_l0 = material_manager.get_nk_at_lambda(mat_name, l0)
            n_real_l0 = n_complex_l0.real
            if n_real_l0 <= 1e-9:
                print(f"Warning: n_real(l0={l0}) for '{mat_name}' <= 0.") # Log properly
                n_reals_at_l0.append(0.0) # Treat as zero index for thickness calc
            else:
                n_reals_at_l0.append(n_real_l0)
        except Exception as e:
            print(f"Error getting n(l0={l0}) for '{mat_name}': {e}. Cannot calculate ep.") # Log properly
            valid_indices = False
            break

    if not valid_indices:
        return np.zeros(num_layers, dtype=np.float64)

    safe_l0 = max(l0, 1e-9)
    multipliers_np = np.asarray(multipliers)

    for i in range(num_layers):
        n_real_l0 = n_reals_at_l0[i]
        if n_real_l0 > 1e-9:
            denom = 4.0 * n_real_l0
            ep_initial[i] = multipliers_np[i] * safe_l0 / denom
        else:
            ep_initial[i] = 0.0 # Assign zero thickness if index was invalid

    # Apply minimum physical thickness constraint AFTER calculating ideal thickness
    # Ensure layers intended to be zero remain zero.
    non_zero_mask = ep_initial > 1e-12
    ep_initial[non_zero_mask] = np.maximum(ep_initial[non_zero_mask], MIN_THICKNESS_PHYS_NM)

    return ep_initial


def calculate_qwot_from_ep(material_manager: MaterialManager,
                           ep_vector: LayerVector,
                           l0: float,
                           material_sequence: Sequence[MaterialIdentifier]
                           ) -> np.ndarray:
    """Calculates QWOT multipliers from physical thicknesses."""
    num_layers = len(ep_vector)
    if len(material_sequence) != num_layers:
         raise ValueError("Length mismatch: ep_vector and material_sequence.")

    qwot_multipliers = np.full(num_layers, np.nan, dtype=np.float64)
    if l0 <= 0:
        print("Warning: l0 <= 0 in calculate_qwot_from_ep. QWOT set to NaN.") # Log properly
        return qwot_multipliers

    valid_indices = True
    n_reals_at_l0 = []
    for i, mat_name in enumerate(material_sequence):
         try:
            n_complex_l0 = material_manager.get_nk_at_lambda(mat_name, l0)
            n_real_l0 = n_complex_l0.real
            if n_real_l0 <= 1e-9:
                 print(f"Warning: n_real(l0={l0}) for '{mat_name}' <= 0. QWOT may be incorrect.") # Log
                 n_reals_at_l0.append(0.0) # Use 0 to likely produce NaN/Inf below
            else:
                 n_reals_at_l0.append(n_real_l0)
         except Exception as e:
            print(f"Error getting n(l0={l0}) for '{mat_name}': {e}. Cannot calculate QWOT.") # Log
            valid_indices = False
            break

    if not valid_indices:
        return np.full(num_layers, np.nan, dtype=np.float64)

    safe_l0 = max(l0, 1e-9)
    ep_vector_np = np.asarray(ep_vector)

    for i in range(num_layers):
        n_real_l0 = n_reals_at_l0[i]
        if n_real_l0 > 1e-9:
             qwot_multipliers[i] = ep_vector_np[i] * (4.0 * n_real_l0) / safe_l0
        # Else: remains NaN

    return qwot_multipliers


# --- Optimization Related Functions (Placeholders / Basic Structure) ---

# Potential location for cost functions, optimization wrappers etc.
# These will depend heavily on the specific requirements for Streamlit interaction

@functools.partial(jax.jit, static_argnames=("targets_tuple",))
def calculate_mse_basic_jax(Ts: jnp.ndarray, l_vec: jnp.ndarray, targets_tuple: Tuple[TargetDict, ...]) -> jnp.ndarray:
    """Calculates basic MSE against target specifications."""
    # This implementation differs slightly from the original's JIT version
    # Original passed tuple of tuples, this uses tuple of dicts (more readable)
    # Re-implementing the core logic:
    total_squared_error = 0.0
    total_points_in_targets = 0

    for target in targets_tuple: # Iterate through list of target dicts
        l_min, l_max = target['min'], target['max']
        t_min, t_max = target['target_min'], target['target_max']

        target_mask = (l_vec >= l_min) & (l_vec <= l_max)
        # Calculate slope for linear interpolation within the target zone
        slope = jnp.where(jnp.abs(l_max - l_min) < 1e-9, 0.0, (t_max - t_min) / (l_max - l_min))
        # Calculate the target transmittance value for each point in l_vec based on the slope
        interpolated_target_t_full = t_min + slope * (l_vec - l_min)

        # Calculate squared errors for all points
        squared_errors_full = (Ts - interpolated_target_t_full)**2

        # Apply the mask: only consider errors within the target zone
        masked_sq_error = jnp.where(target_mask, squared_errors_full, 0.0)

        # Sum the squared errors within the zone
        total_squared_error += jnp.sum(masked_sq_error)
        # Count the number of points within the zone
        total_points_in_targets += jnp.sum(target_mask)

    # Calculate MSE, handle case where no points fall into any target zone
    mse = jnp.where(total_points_in_targets > 0, total_squared_error / total_points_in_targets, jnp.inf)

    # Handle potential NaNs (e.g., if Ts contained NaNs)
    return jnp.nan_to_num(mse, nan=jnp.inf)


# --- Example Usage Placeholder ---
if __name__ == '__main__':

    # This section would be replaced by Streamlit UI and logic
    print("Refactored code structure. Running basic example...")

    # 1. Initialize Managers
    try:
        material_mgr = MaterialManager(excel_file_path=EXCEL_FILE_PATH)
        calculator = OpticalCalculator(material_mgr)
    except Exception as e:
        print(f"Error initializing managers: {e}")
        exit()

    # 2. Define a simple design
    # Example: Quarter-wave stack at 550 nm
    l0_design = 550.0
    num_layers = 5
    # Use identifiers known to the MaterialManager
    # Example: Using strings for materials defined in Excel or predefined
    mat_h = "Nb2O5-Helios" # Assumes this sheet exists in indices.xlsx
    mat_l = "SiO2-Helios"  # Assumes this sheet exists in indices.xlsx
    mat_sub = "Fused Silica" # Use a predefined material

    test_material_sequence = [mat_h if i % 2 == 0 else mat_l for i in range(num_layers)]
    test_multipliers = [1.0] * num_layers # Simple QWOT

    try:
        test_ep_vector = calculate_initial_ep(material_mgr, test_multipliers, l0_design, test_material_sequence)
        print(f"Calculated ep_vector for {num_layers} layers: {test_ep_vector}")

        # 3. Define calculation parameters
        wavelengths = np.linspace(400, 800, 201)

        # 4. Perform calculation
        results = calculator.calculate_T_spectrum(
            ep_vector=test_ep_vector,
            material_sequence=test_material_sequence,
            nSub_material=mat_sub,
            l_vec=wavelengths
            # nInc_material defaults to "AIR"
        )

        print(f"Calculation successful for {len(results['l'])} wavelengths.")
        # In a real app, you would plot results['l'] vs results['Ts']
        # print(f"Example T at ~400nm: {results['Ts'][0]:.4f}")
        # print(f"Example T at ~600nm: {results['Ts'][100]:.4f}")
        # print(f"Example T at ~800nm: {results['Ts'][-1]:.4f}")

        # 5. Example: Calculate QWOT back (verification)
        recalculated_qwots = calculate_qwot_from_ep(material_mgr, test_ep_vector, l0_design, test_material_sequence)
        print(f"Recalculated QWOTs: {recalculated_qwots}")


    except Exception as e:
        print(f"\n--- An error occurred during the example run ---")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()

    # Example of clearing cache (might be useful in long-running app)
    # material_mgr.clear_cache()

# --- Cost Functions for Optimization ---

@functools.partial(jax.jit, static_argnames=("targets_tuple", "min_thickness_phys_nm"))
def calculate_mse_penalized_jax(ep_vector: jnp.ndarray,
                                optical_calculator: 'OpticalCalculator', # Pass calculator instance
                                layer_indices_arr_transposed: jnp.ndarray, # Precomputed n,k vs lambda, transposed
                                nSub_arr: jnp.ndarray,
                                nInc_arr: jnp.ndarray,
                                l_vec_optim: jnp.ndarray,
                                targets_tuple: Tuple[TargetDict, ...],
                                min_thickness_phys_nm: float,
                                penalty_factor: float = 1e5 # Penalty strength for thin layers
                               ) -> jnp.ndarray:
    """
    Calculates MSE cost with a penalty for layers thinner than the physical minimum.
    This version requires precomputed index arrays for efficiency within optimization loops.
    """
    # Penalty for layers that are very thin but not zero
    # Encourages optimizer to either make layers substantial or remove them (set to ~0)
    below_min_mask = (ep_vector < min_thickness_phys_nm) & (ep_vector > 1e-12)
    # Quadratic penalty increases sharply as thickness approaches zero below the minimum
    penalty_thin = jnp.sum(jnp.where(below_min_mask, (min_thickness_phys_nm - ep_vector)**2, 0.0)) * penalty_factor

    # Calculate Transmittance using the clamped thickness vector for the optical model
    # We clamp here so the optical calculation doesn't see unphysically thin layers,
    # but the penalty is based on the original ep_vector from the optimizer.
    ep_vector_calc = jnp.maximum(ep_vector, min_thickness_phys_nm)

    # Calculate Transmittance using the core vectorized function
    # Note: calculate_single_wavelength_T expects layer_indices[wavelength, layer]
    # We already have layer_indices_arr_transposed which is [wavelength, layer]
    Ts = vmap(optical_calculator.calculate_single_wavelength_T, in_axes=(0, None, 0, 0, 0))(
        l_vec_optim, ep_vector_calc, layer_indices_arr_transposed, nSub_arr, nInc_arr
    )
    Ts = jnp.nan_to_num(Ts, nan=0.0) # Ensure no NaNs propagate from T calculation

    # Calculate the basic MSE using the calculated Transmittance
    mse = calculate_mse_basic_jax(Ts, l_vec_optim, targets_tuple)

    # Final cost is MSE plus the penalty
    final_cost = mse + penalty_thin

    # Ensure the final cost is always a finite number for the optimizer
    return jnp.nan_to_num(final_cost, nan=jnp.inf, posinf=jnp.inf, neginf=jnp.inf)


# --- SciPy Optimizer Interface ---

def scipy_optimizer_wrapper(ep_vector_np: np.ndarray,
                            jax_value_and_grad_func: Callable,
                            static_args: tuple
                           ) -> Tuple[float, np.ndarray]:
    """
    Wrapper to connect JAX gradient calculation with SciPy's minimize function.

    Args:
        ep_vector_np: Current thickness vector (NumPy array from SciPy).
        jax_value_and_grad_func: JAX function compiled with value_and_grad().
        static_args: Tuple of other arguments needed by the JAX cost function.

    Returns:
        A tuple containing the cost (float) and the gradient (NumPy array).
    """
    ep_vector_jax = jnp.asarray(ep_vector_np)
    value_jax, grad_jax = jax_value_and_grad_func(ep_vector_jax, *static_args)

    # Convert results back to NumPy format suitable for SciPy
    value_np = float(np.asarray(value_jax))
    grad_np = np.asarray(grad_jax, dtype=np.float64)

    # Basic check for non-finite values which can halt the optimizer
    if not np.isfinite(value_np):
        print(f"Warning: Non-finite cost encountered: {value_np}. Replacing with large value.") # Log properly
        value_np = 1e20 # Use a large finite number instead of infinity
    if not np.all(np.isfinite(grad_np)):
        print(f"Warning: Non-finite gradient encountered. Replacing with zeros.") # Log properly
        grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=0.0, neginf=0.0) # Replace non-finite grads

    return value_np, grad_np

# --- Core Optimization Runner ---

def run_core_optimization(ep_start_optim: np.ndarray,
                          material_manager: MaterialManager,
                          optical_calculator: OpticalCalculator,
                          material_sequence: Sequence[MaterialIdentifier],
                          nSub_material: MaterialIdentifier,
                          nInc_material: MaterialIdentifier,
                          active_targets: Sequence[TargetDict],
                          l_vec_optim_np: np.ndarray, # Wavelength grid for optimization
                          optimizer_options: Dict[str, Any],
                          min_thickness_phys: float = MIN_THICKNESS_PHYS_NM
                         ) -> Tuple[Optional[np.ndarray], bool, float, str, int, int, List[str]]:
    """
    Runs the core L-BFGS-B optimization using JAX gradients.

    Args:
        ep_start_optim: Initial thickness vector.
        material_manager: Instance for nk lookups.
        optical_calculator: Instance for T calculations.
        material_sequence: List of material identifiers for layers.
        nSub_material: Substrate material identifier.
        nInc_material: Incident medium material identifier.
        active_targets: List of validated target dictionaries.
        l_vec_optim_np: Wavelength points for optimization cost evaluation.
        optimizer_options: Dictionary of options for scipy.optimize.minimize (e.g., maxiter).
        min_thickness_phys: Minimum allowed physical thickness for layers.

    Returns:
        Tuple: (final_ep_vector, success_flag, final_cost, message, iterations, func_evals, logs)
               final_ep_vector is None if optimization fails critically.
    """
    logs = []
    num_layers_start = len(ep_start_optim)
    if num_layers_start == 0:
        logs.append("Cannot optimize an empty structure.")
        return None, False, np.inf, "Empty structure", 0, 0, logs

    if not l_vec_optim_np.size:
         logs.append("Optimization lambda vector is empty.")
         return ep_start_optim, False, np.inf, "No wavelengths for optimization", 0, 0, logs

    logs.append(f"Starting core optimization for {num_layers_start} layers.")
    l_vec_optim_jax = jnp.asarray(l_vec_optim_np)

    try:
        # --- Precompute Index Arrays ---
        t_start_indices = time.time()
        logs.append("Precomputing refractive indices for optimization...")
        nSub_arr = material_manager.get_nk_array(nSub_material, l_vec_optim_jax)
        nInc_arr = material_manager.get_nk_array(nInc_material, l_vec_optim_jax)

        if num_layers_start > 0:
             layer_indices_list = [material_manager.get_nk_array(mat, l_vec_optim_jax) for mat in material_sequence]
             # Shape: (num_layers, num_wavelengths) -> Transpose to (num_wavelengths, num_layers) for cost func
             layer_indices_arr_transposed = jnp.stack(layer_indices_list, axis=0).T
        else:
             # Still need a placeholder with correct dimensions if stack is empty
             layer_indices_arr_transposed = jnp.empty((len(l_vec_optim_jax), 0), dtype=jnp.complex128)

        logs.append(f"Index precomputation finished in {time.time() - t_start_indices:.3f}s.")

        # --- Prepare Cost Function and Arguments ---
        # Convert active targets from list of dicts to tuple of tuples for JAX compatibility
        active_targets_tuple = tuple(
            (t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets
        )

        # Static arguments for the JAX cost function
        # These arguments won't change during the optimization process
        static_args_for_cost = (
            optical_calculator, # Pass the instance itself
            layer_indices_arr_transposed,
            nSub_arr,
            nInc_arr,
            l_vec_optim_jax,
            active_targets_tuple,
            min_thickness_phys
            # penalty_factor could also be passed here if needed
        )

        # Compile the value-and-gradient function with JAX
        # Only the first argument (ep_vector) changes during optimization
        value_and_grad_cost_func = jax.jit(value_and_grad(calculate_mse_penalized_jax, argnums=0))

        # --- Define Bounds ---
        # Lower bound for thickness is the minimum physical thickness
        lbfgsb_bounds = [(min_thickness_phys, None)] * num_layers_start

        # --- Run Optimization ---
        logs.append(f"Starting L-BFGS-B with JAX gradient... Options: {optimizer_options}")
        opt_start_time = time.time()

        # Ensure starting point respects bounds (just in case)
        ep_start_np_safe = np.maximum(np.asarray(ep_start_optim, dtype=np.float64), min_thickness_phys)

        result = minimize(
            scipy_optimizer_wrapper,
            ep_start_np_safe, # Start with NumPy array
            args=(value_and_grad_cost_func, static_args_for_cost), # Pass JAX func and static args
            method='L-BFGS-B',
            jac=True, # Indicate that the function returns both value and gradient
            bounds=lbfgsb_bounds,
            options=optimizer_options
        )
        logs.append(f"L-BFGS-B finished in {time.time() - opt_start_time:.3f}s.")

        # --- Process Results ---
        final_cost = float(result.fun) if np.isfinite(result.fun) else np.inf
        result_message_str = result.message.decode('utf-8') if isinstance(result.message, bytes) else str(result.message)
        nit_total = int(result.nit) if hasattr(result, 'nit') else 0
        nfev_total = int(result.nfev) if hasattr(result, 'nfev') else 0
        optim_success = bool(result.success) # Success flag from SciPy

        # Consider termination due to limits (status=1) also a "success" in terms of usability
        is_success_or_limit = (optim_success or result.status == 1) and np.isfinite(final_cost)

        if is_success_or_limit:
            final_ep = np.maximum(result.x, min_thickness_phys) # Ensure bounds after optimization
            log_status = "converged" if optim_success else "stopped by limit"
            logs.append(f"Optimization {log_status}. Cost: {final_cost:.6e}, Iterations: {nit_total}, Evals: {nfev_total}, Msg: {result_message_str}")
            return final_ep, True, final_cost, result_message_str, nit_total, nfev_total, logs
        else:
            logs.append(f"Optimization FAILED. Status: {result.status}, Cost: {final_cost:.6e}, Msg: {result_message_str}")
            # Try to return the clamped starting vector if optimization failed badly
            final_ep_fail = np.maximum(ep_start_optim, min_thickness_phys)
            # Recalculate cost for the starting point for reference
            try:
                 cost_fail, _ = scipy_optimizer_wrapper(final_ep_fail, value_and_grad_cost_func, static_args_for_cost)
                 logs.append(f"Returning clamped starting vector. Recalculated cost: {cost_fail:.6e}")
            except Exception as cost_e:
                 logs.append(f"Returning clamped starting vector. Failed to recalculate cost: {cost_e}")
                 cost_fail = np.inf # Mark cost as infinite if recalc failed

            return final_ep_fail, False, cost_fail, f"Optimization Failed: {result_message_str}", 0, 0, logs

    except Exception as e_optim:
        logs.append(f"ERROR during core optimization setup or run: {type(e_optim).__name__} - {e_optim}")
        logs.append(traceback.format_exc(limit=2))
        final_ep_exc = np.maximum(ep_start_optim, min_thickness_phys) # Fallback
        return final_ep_exc, False, np.inf, f"Exception: {e_optim}", 0, 0, logs


# --- Thin Layer Removal / Merge ---

def perform_layer_merge_or_removal(ep_vector_in: np.ndarray,
                                   min_thickness_phys: float,
                                   threshold_for_removal: Optional[float] = None
                                   ) -> Tuple[Optional[np.ndarray], bool, List[str], Optional[int]]:
    """
    Identifies the thinnest layer (optionally below a threshold) and removes/merges it.

    Handles merging adjacent layers if the removed layer is internal,
    or removing pairs/single layers at the edges.

    Args:
        ep_vector_in: Current thickness vector.
        min_thickness_phys: Absolute minimum thickness for any layer.
        threshold_for_removal: If set, only layers *below* this threshold
                               (but >= min_thickness_phys) are considered.

    Returns:
        Tuple: (new_ep_vector, structure_changed_flag, logs, removed_layer_original_index)
               new_ep_vector is None if no layer could be removed/merged.
               removed_layer_original_index is the index (0-based) of the layer targeted for removal.
    """
    current_ep = ep_vector_in.copy()
    logs = []
    num_layers = len(current_ep)
    structure_changed = False
    ep_after_merge = None
    removed_layer_idx = None

    if num_layers < 1:
        logs.append("Structure is empty. Cannot remove/merge.")
        return None, False, logs, None
    # Need at least 2 layers to merge/remove meaningfully in most cases
    if num_layers <= 1 and threshold_for_removal is None:
        logs.append(f"Structure has {num_layers} layer(s). Cannot merge/remove further.")
        return None, False, logs, None

    try:
        # --- Find the target layer ---
        thin_layer_index = -1
        min_thickness_found = np.inf

        # Identify candidate layers: those above the absolute minimum physical thickness
        candidate_indices = np.where(current_ep >= min_thickness_phys)[0]

        if candidate_indices.size == 0:
            logs.append(f"No layers found >= minimum thickness {min_thickness_phys:.3f} nm.")
            return None, False, logs, None

        candidate_thicknesses = current_ep[candidate_indices]

        # If a threshold is specified, filter candidates further
        if threshold_for_removal is not None:
            valid_for_removal_mask = candidate_thicknesses < threshold_for_removal
            if np.any(valid_for_removal_mask):
                candidate_indices = candidate_indices[valid_for_removal_mask]
                candidate_thicknesses = candidate_thicknesses[valid_for_removal_mask]
                logs.append(f"Found {len(candidate_indices)} candidate layers below threshold {threshold_for_removal:.3f} nm.")
            else:
                logs.append(f"No layers found >= {min_thickness_phys:.3f} nm AND < threshold {threshold_for_removal:.3f} nm.")
                return None, False, logs, None # No layers meet the threshold criteria

        # Find the thinnest among the (potentially filtered) candidates
        if candidate_indices.size > 0:
            min_idx_local = np.argmin(candidate_thicknesses)
            thin_layer_index = candidate_indices[min_idx_local]
            min_thickness_found = candidate_thicknesses[min_idx_local]
            logs.append(f"Targeting thinnest valid layer: Index {thin_layer_index} (Thickness: {min_thickness_found:.3f} nm).")
        else:
            # This case should ideally not be reached if threshold wasn't met earlier
             logs.append(f"No suitable layer found for removal/merging.")
             return None, False, logs, None

        # --- Perform Removal/Merge based on index ---
        removed_layer_idx = thin_layer_index # Store the index of the layer we acted upon
        thin_layer_thickness = current_ep[thin_layer_index]

        # Special case: Removing the first layer (index 0)
        if thin_layer_index == 0:
            if num_layers >= 2:
                # Remove the first two layers (thin layer + adjacent)
                # Assumes alternating materials, where removing one makes the next one incompatible
                ep_after_merge = current_ep[2:]
                logs.append(f"Removed layers 0 & 1 (layer 0 was target). New size: {len(ep_after_merge)}.")
                structure_changed = True
            elif num_layers == 1: # Removing the only layer
                 ep_after_merge = np.array([], dtype=current_ep.dtype)
                 logs.append("Removed the only layer.")
                 structure_changed = True
            else: # Should not happen due to earlier checks
                 logs.append("Cannot remove layers 0 & 1 - unexpected structure size.")
                 return None, False, logs, removed_layer_idx

        # Special case: Removing the last layer (index num_layers - 1)
        elif thin_layer_index == num_layers - 1:
             # Remove only the last layer
             ep_after_merge = current_ep[:-1]
             logs.append(f"Removed only the last layer {num_layers - 1}. New size: {len(ep_after_merge)}.")
             structure_changed = True

        # General case: Removing an internal layer
        else:
             # Merge the layers on either side of the thin layer
             merged_thickness = current_ep[thin_layer_index - 1] + current_ep[thin_layer_index + 1]
             ep_after_merge = np.concatenate((
                 current_ep[:thin_layer_index - 1],    # Layers before the left neighbor
                 [merged_thickness],                   # The merged layer
                 current_ep[thin_layer_index + 2:]     # Layers after the right neighbor
             ))
             logs.append(f"Removed layer {thin_layer_index}, merged neighbours {thin_layer_index-1} & {thin_layer_index+1} -> {merged_thickness:.3f}. New size: {len(ep_after_merge)}.")
             structure_changed = True

        # --- Finalize and return ---
        if structure_changed and ep_after_merge is not None:
             # Ensure remaining layers meet minimum thickness requirement
             if ep_after_merge.size > 0:
                  ep_after_merge = np.maximum(ep_after_merge, min_thickness_phys)
             return ep_after_merge, True, logs, removed_layer_idx
        else:
             # If structure_changed was true but ep_after_merge is None, something went wrong
             if structure_changed:
                 logs.append("Internal Logic Error: Structure change indicated but result is None.")
             return current_ep, False, logs, None # Return original, no change

    except Exception as e_merge:
        logs.append(f"ERROR during merge/removal logic: {e_merge}\n{traceback.format_exc(limit=1)}")
        return ep_vector_in, False, logs, None # Return original on error


# --- Needle Optimization Functions ---

def perform_needle_insertion_scan(ep_vector_in: np.ndarray,
                                  material_manager: MaterialManager,
                                  optical_calculator: OpticalCalculator,
                                  material_sequence_in: Sequence[MaterialIdentifier],
                                  nSub_material: MaterialIdentifier,
                                  nInc_material: MaterialIdentifier,
                                  l_vec_optim_np: np.ndarray,
                                  active_targets: Sequence[TargetDict],
                                  l0_for_materials: float, # Reference lambda for choosing H/L needle type
                                  min_thickness_phys: float,
                                  base_needle_thickness_nm: float,
                                  scan_step_nm: float
                                  ) -> Tuple[Optional[np.ndarray], Optional[List[MaterialIdentifier]], float, List[str], int]:
    """
    Scans through the stack depth, inserts a thin 'needle' layer, and finds the
    insertion point that yields the lowest cost (MSE).

    Args:
        ep_vector_in: Current thickness vector.
        material_manager: Instance for nk lookups.
        optical_calculator: Instance for T calculations.
        material_sequence_in: Current material sequence.
        nSub_material: Substrate material identifier.
        nInc_material: Incident medium material identifier.
        l_vec_optim_np: Wavelength grid for cost evaluation.
        active_targets: List of validated target dictionaries.
        l0_for_materials: Reference wavelength to decide needle type (H or L).
        min_thickness_phys: Minimum allowed physical thickness.
        base_needle_thickness_nm: Thickness of the needle layer to insert.
        scan_step_nm: Step size for scanning insertion depth.

    Returns:
        Tuple: (best_ep_found, best_mat_seq_found, min_cost_found, logs, best_insertion_layer_idx)
               best_ep_found/best_mat_seq_found are None if no improvement found.
               best_insertion_layer_idx is the index of the *original* layer where insertion occurred.
    """
    logs = []
    num_layers_in = len(ep_vector_in)
    if num_layers_in == 0:
        logs.append("[Needle Scan] Cannot run on an empty structure.")
        return None, None, np.inf, logs, -1
    if len(material_sequence_in) != num_layers_in:
         logs.append("[Needle Scan] ERROR: ep_vector and material_sequence length mismatch.")
         return None, None, np.inf, logs, -1

    logs.append(f"[Needle Scan] Starting scan on {num_layers_in} layers. Step: {scan_step_nm:.2f} nm, Needle: {base_needle_thickness_nm:.3f} nm.")
    l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
    active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)

    # --- Precompute indices and initial cost ---
    try:
        t_start_indices = time.time()
        nSub_arr = material_manager.get_nk_array(nSub_material, l_vec_optim_jax)
        nInc_arr = material_manager.get_nk_array(nInc_material, l_vec_optim_jax)

        layer_indices_list = [material_manager.get_nk_array(mat, l_vec_optim_jax) for mat in material_sequence_in]
        layer_indices_arr_transposed = jnp.stack(layer_indices_list, axis=0).T
        logs.append(f"  Index precomputation finished in {time.time() - t_start_indices:.3f}s.")

        # Compile the penalized cost function
        cost_fn_compiled = jax.jit(
            functools.partial(calculate_mse_penalized_jax,
                              optical_calculator=optical_calculator,
                              nSub_arr=nSub_arr,
                              nInc_arr=nInc_arr,
                              l_vec_optim=l_vec_optim_jax,
                              targets_tuple=active_targets_tuple,
                              min_thickness_phys_nm=min_thickness_phys)
        )

        # Calculate initial cost (using the precomputed transposed indices)
        initial_cost_jax = cost_fn_compiled(jnp.asarray(ep_vector_in), layer_indices_arr_transposed=layer_indices_arr_transposed)
        initial_cost = float(np.array(initial_cost_jax))

        if not np.isfinite(initial_cost):
            logs.append(f"[Needle Scan] ERROR: Initial cost is not finite ({initial_cost}). Aborting.")
            return None, None, np.inf, logs, -1
        logs.append(f"  Initial cost: {initial_cost:.6e}")

        # Get H and L material identifiers based on the *first* layer of the input sequence
        # Assumes an alternating structure for choosing needle type. This might need refinement
        # if the input sequence is arbitrary.
        mat_H_ident = material_sequence_in[0] if num_layers_in > 0 else None
        mat_L_ident = material_sequence_in[1] if num_layers_in > 1 else mat_H_ident # Fallback

        # Get n(l0) to determine needle material type later (assumes nH > nL)
        nH_real_l0, nL_real_l0 = -1.0, -1.0
        if mat_H_ident: nH_real_l0 = material_manager.get_nk_at_lambda(mat_H_ident, l0_for_materials).real
        if mat_L_ident: nL_real_l0 = material_manager.get_nk_at_lambda(mat_L_ident, l0_for_materials).real
        if nH_real_l0 <= 0 or nL_real_l0 <= 0:
            logs.append(f"  WARNING: Could not get valid nH/nL at l0={l0_for_materials}. Needle type may be wrong.")
            # Use default indices if lookup failed, just to proceed
            if nH_real_l0 <=0 : nH_real_l0 = 2.0
            if nL_real_l0 <=0 : nL_real_l0 = 1.5


    except Exception as e_prep:
        logs.append(f"[Needle Scan] ERROR preparing indices/initial cost: {e_prep}. Aborting.")
        logs.append(traceback.format_exc(limit=1))
        return None, None, np.inf, logs, -1

    # --- Scan Loop ---
    best_ep_found: Optional[np.ndarray] = None
    best_mat_seq_found: Optional[List[MaterialIdentifier]] = None
    min_cost_found = initial_cost
    best_insertion_original_idx = -1
    tested_insertions = 0

    ep_cumsum = np.cumsum(ep_vector_in)
    total_thickness = ep_cumsum[-1] if num_layers_in > 0 else 0.0

    # Iterate through possible insertion depths (z)
    for z in np.arange(scan_step_nm, total_thickness, scan_step_nm):
        current_layer_idx = -1
        layer_start_z = 0.0
        # Find which original layer the depth 'z' falls into
        for i in range(num_layers_in):
            layer_end_z = ep_cumsum[i]
            if z > layer_start_z and z <= layer_end_z:
                # Check if splitting the layer results in two parts >= min_thickness
                t_part1 = z - layer_start_z
                t_part2 = layer_end_z - z
                if t_part1 >= min_thickness_phys and t_part2 >= min_thickness_phys:
                    current_layer_idx = i
                    break # Found the layer to split
                else:
                    # Splitting here would create a sub-minimal layer
                    current_layer_idx = -2 # Mark as invalid split point
                    break
            layer_start_z = layer_end_z

        if current_layer_idx < 0: # Either before first layer, after last, or invalid split
            # logs.append(f"  Skipping z={z:.2f}: Invalid split point (layer {current_layer_idx}).")
            continue

        tested_insertions += 1

        # --- Construct the temporary ep vector and material sequence ---
        original_layer_material = material_sequence_in[current_layer_idx]
        # Determine needle material: opposite of the layer being split
        # This assumes alternating H/L based on index lookup at l0.
        # A more robust way might be needed for arbitrary sequences.
        is_split_layer_H = (original_layer_material == mat_H_ident) or (material_manager.get_nk_at_lambda(original_layer_material, l0_for_materials).real > nL_real_l0 + 1e-3)

        needle_material = mat_L_ident if is_split_layer_H else mat_H_ident
        if needle_material is None:
            logs.append(f"  Skipping z={z:.2f}: Could not determine needle material.")
            continue # Cannot determine needle type


        # Thicknesses of the two parts of the split layer
        layer_start_z_for_split = ep_cumsum[current_layer_idx-1] if current_layer_idx > 0 else 0.0
        t_layer_split_1 = z - layer_start_z_for_split
        t_layer_split_2 = ep_cumsum[current_layer_idx] - z

        # Build new ep vector
        ep_temp_list = (list(ep_vector_in[:current_layer_idx]) +
                        [t_layer_split_1, base_needle_thickness_nm, t_layer_split_2] +
                        list(ep_vector_in[current_layer_idx+1:]))
        ep_temp_np = np.array(ep_temp_list)
        # Ensure minimum thickness constraint is met immediately
        ep_temp_np = np.maximum(ep_temp_np, min_thickness_phys)

        # Build new material sequence
        mat_seq_temp = (list(material_sequence_in[:current_layer_idx]) +
                        [original_layer_material, needle_material, original_layer_material] +
                        list(material_sequence_in[current_layer_idx+1:]))

        # --- Evaluate cost ---
        try:
            # Recalculate transposed indices array for the new structure
            # This is necessary because the cost function expects it precomputed
            layer_indices_list_temp = [material_manager.get_nk_array(mat, l_vec_optim_jax) for mat in mat_seq_temp]
            indices_arr_transposed_temp = jnp.stack(layer_indices_list_temp, axis=0).T

            current_cost_jax = cost_fn_compiled(jnp.asarray(ep_temp_np), layer_indices_arr_transposed=indices_arr_transposed_temp)
            current_cost = float(np.array(current_cost_jax))

            if np.isfinite(current_cost) and current_cost < min_cost_found:
                min_cost_found = current_cost
                best_ep_found = ep_temp_np.copy()
                best_mat_seq_found = list(mat_seq_temp) # Copy the list
                best_insertion_original_idx = current_layer_idx
                # logs.append(f"  Found new best cost {min_cost_found:.6e} at z={z:.2f} (orig layer {current_layer_idx})")

        except Exception as e_cost:
            logs.append(f"  WARNING: Cost calculation failed for z={z:.2f}. {type(e_cost).__name__}: {e_cost}")
            continue # Skip this insertion point

    # --- Final Reporting ---
    if best_ep_found is not None:
        improvement = initial_cost - min_cost_found
        logs.append(f"[Needle Scan] Finished. Tested {tested_insertions} points. Best improvement: {improvement:.6e} -> {min_cost_found:.6e}")
        logs.append(f"  Best insertion was in original layer index {best_insertion_original_idx}.")
        # Perform a sanity check on the returned sequence length
        if len(best_ep_found) != len(best_mat_seq_found):
             logs.append("[Needle Scan] INTERNAL ERROR: Mismatch between best_ep and best_mat_seq lengths!")
             return None, None, initial_cost, logs, -1 # Return no improvement if inconsistent
        return best_ep_found, best_mat_seq_found, min_cost_found, logs, best_insertion_original_idx
    else:
        logs.append("[Needle Scan] Finished. No improvement found.")
        return None, None, initial_cost, logs, -1

# --- Needle Iteration Loop ---

def run_needle_iterations(ep_start: np.ndarray,
                          material_sequence_start: Sequence[MaterialIdentifier],
                          num_needles: int,
                          material_manager: MaterialManager,
                          optical_calculator: OpticalCalculator,
                          nSub_material: MaterialIdentifier,
                          nInc_material: MaterialIdentifier,
                          active_targets: Sequence[TargetDict],
                          l_vec_optim_np: np.ndarray,
                          optimizer_options: Dict[str, Any],
                          l0_for_materials: float,
                          min_thickness_phys: float,
                          scan_step_nm: float = DEFAULT_NEEDLE_SCAN_STEP_NM,
                          base_needle_thickness_nm: float = BASE_NEEDLE_THICKNESS_NM,
                          log_prefix: str = ""
                          ) -> Tuple[np.ndarray, List[MaterialIdentifier], float, List[str], int, int, int]:
    """
    Performs multiple cycles of needle insertion scan followed by core optimization.

    Args:
        # ... (arguments similar to previous functions)
        num_needles: Number of needle insertion/optimization cycles to perform.

    Returns:
        Tuple: (best_ep_overall, best_mat_seq_overall, best_mse_overall, logs,
                total_iterations, total_func_evals, successful_reopts_count)
    """
    logs = []
    best_ep_overall = np.asarray(ep_start).copy()
    best_mat_seq_overall = list(material_sequence_start)
    best_mse_overall = np.inf
    total_nit_needles = 0
    total_nfev_needles = 0
    successful_reopts_count = 0

    # --- Calculate initial MSE ---
    try:
        l_vec_optim_jax = jnp.asarray(l_vec_optim_np)
        nSub_arr = material_manager.get_nk_array(nSub_material, l_vec_optim_jax)
        nInc_arr = material_manager.get_nk_array(nInc_material, l_vec_optim_jax)
        layer_indices_list = [material_manager.get_nk_array(mat, l_vec_optim_jax) for mat in best_mat_seq_overall]
        layer_indices_arr_transposed = jnp.stack(layer_indices_list, axis=0).T
        active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)

        cost_fn_compiled = jax.jit(
            functools.partial(calculate_mse_penalized_jax,
                              optical_calculator=optical_calculator,
                              nSub_arr=nSub_arr, nInc_arr=nInc_arr,
                              l_vec_optim=l_vec_optim_jax, targets_tuple=active_targets_tuple,
                              min_thickness_phys_nm=min_thickness_phys)
        )
        initial_cost_jax = cost_fn_compiled(jnp.asarray(best_ep_overall), layer_indices_arr_transposed=layer_indices_arr_transposed)
        best_mse_overall = float(np.array(initial_cost_jax))
        if not np.isfinite(best_mse_overall): raise ValueError("Initial MSE for needle iterations not finite.")
        logs.append(f"{log_prefix}Starting needle iterations. Initial MSE: {best_mse_overall:.6e}")
    except Exception as e:
        logs.append(f"{log_prefix}ERROR calculating initial MSE for needle iterations: {e}")
        logs.append(traceback.format_exc(limit=1))
        # Return start state if initial calculation fails
        return best_ep_overall, best_mat_seq_overall, np.inf, logs, 0, 0, 0

    # --- Iteration Loop ---
    for i in range(num_needles):
        logs.append(f"{log_prefix}Needle Iteration {i + 1}/{num_needles}")
        ep_before_iter = best_ep_overall.copy()
        mat_seq_before_iter = list(best_mat_seq_overall) # Important to copy list
        mse_before_iter = best_mse_overall

        # 1. Perform Needle Scan
        ep_after_scan, mat_seq_after_scan, cost_after_scan, scan_logs, inserted_idx = \
            perform_needle_insertion_scan(
                ep_before_iter, material_manager, optical_calculator, mat_seq_before_iter,
                nSub_material, nInc_material, l_vec_optim_np, active_targets,
                l0_for_materials, min_thickness_phys, base_needle_thickness_nm, scan_step_nm
            )
        logs.extend(scan_logs)

        if ep_after_scan is None:
            logs.append(f"{log_prefix}Needle scan {i + 1} found no improvement. Stopping iterations.")
            break # Exit loop if scan doesn't find a better place

        # 2. Perform Core Optimization on the structure with the inserted needle
        logs.append(f"{log_prefix}Needle scan {i + 1} found potential improvement. Re-optimizing {len(ep_after_scan)} layers...")

        # Ensure the material sequence corresponds to the ep_vector after scan
        if len(ep_after_scan) != len(mat_seq_after_scan):
             logs.append(f"{log_prefix}CRITICAL ERROR: Length mismatch after needle scan ({len(ep_after_scan)} vs {len(mat_seq_after_scan)}). Aborting iteration.")
             break

        ep_after_reopt, optim_success, final_cost_reopt, optim_msg, nit_reopt, nfev_reopt, optim_logs = \
            run_core_optimization(
                ep_after_scan, material_manager, optical_calculator,
                mat_seq_after_scan, # Use the sequence corresponding to ep_after_scan
                nSub_material, nInc_material, active_targets,
                l_vec_optim_np, optimizer_options, min_thickness_phys
            )
        logs.extend(optim_logs)

        if not optim_success:
            logs.append(f"{log_prefix}Re-optimization after needle scan {i + 1} failed ({optim_msg}). Stopping iterations.")
            # Keep the result from *before* this failed optimization attempt
            best_ep_overall = ep_before_iter
            best_mat_seq_overall = mat_seq_before_iter
            best_mse_overall = mse_before_iter
            break

        # 3. Check for Improvement
        logs.append(f"{log_prefix}Re-optimization {i + 1} successful. New MSE: {final_cost_reopt:.6e}. Iter/Eval: {nit_reopt}/{nfev_reopt}")
        total_nit_needles += nit_reopt
        total_nfev_needles += nfev_reopt
        successful_reopts_count += 1

        if final_cost_reopt < best_mse_overall - MSE_IMPROVEMENT_TOLERANCE:
            logs.append(f"{log_prefix}  MSE improved from {best_mse_overall:.6e}. Updating best result.")
            best_ep_overall = ep_after_reopt.copy()
            best_mat_seq_overall = list(mat_seq_after_scan) # Store the sequence corresponding to the best ep
            best_mse_overall = final_cost_reopt
        else:
            logs.append(f"{log_prefix}  New MSE ({final_cost_reopt:.6e}) not significantly improved vs best ({best_mse_overall:.6e}). Stopping needle iterations.")
            # Even if not improved vs *best*, the re-opt was successful, so update state to this latest result before stopping
            best_ep_overall = ep_after_reopt.copy()
            best_mat_seq_overall = list(mat_seq_after_scan)
            best_mse_overall = final_cost_reopt
            break # Stop iterating

    # --- Final Report ---
    logs.append(f"{log_prefix}End needle iterations. Final best MSE: {best_mse_overall:.6e} ({len(best_ep_overall)} layers)")
    logs.append(f"{log_prefix}Total Iter/Eval during {successful_reopts_count} successful re-opts: {total_nit_needles}/{total_nfev_needles}")

    return best_ep_overall, best_mat_seq_overall, best_mse_overall, logs, total_nit_needles, total_nfev_needles, successful_reopts_count


# --- QWOT Exhaustive Scan Related Functions ---

# Note: These functions are highly specialized for the QWOT=1 or 2 scan
# and rely heavily on JAX for performance.

@functools.partial(jax.jit, static_argnames=("initial_layer_number",))
def get_layer_matrices_qwot(layer_idx: int, initial_layer_number: int, l0: float,
                            nH_c_l0: jnp.ndarray, nL_c_l0: jnp.ndarray,
                            l_vec: jnp.ndarray
                           ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precomputes the transfer matrices for a specific layer index assuming
    either 1*QWOT or 2*QWOT thickness based on l0.

    Returns matrices for all wavelengths in l_vec for both 1*QWOT and 2*QWOT.
    Output shape per matrix: (num_wavelengths, 2, 2)
    """
    # Determine if the layer is High or Low index based on its position
    predicate_is_H = (layer_idx % 2 == 0) # Assumes layer 0 is H

    # Select the complex index at l0 and the real part at l0
    n_complex_for_matrix = jax.lax.select(predicate_is_H, nH_c_l0, nL_c_l0)
    n_real_l0 = jnp.real(n_complex_for_matrix) # Real part used for QWOT thickness calc

    # Calculate 1*QWOT and 2*QWOT physical thicknesses
    # Use safe denominators to avoid division by zero
    denom = 4.0 * jnp.maximum(n_real_l0, 1e-9)
    safe_l0 = jnp.maximum(l0, 1e-9)

    ep1_calc = 1.0 * safe_l0 / denom
    ep2_calc = 2.0 * safe_l0 / denom

    # Only calculate if n_real_l0 is valid, otherwise thickness is 0
    ep1 = jnp.where(n_real_l0 > 1e-9, ep1_calc, 0.0)
    ep2 = jnp.where(n_real_l0 > 1e-9, ep2_calc, 0.0)

    # Apply minimum physical thickness constraint IF the calculated thickness > 0
    ep1 = jnp.maximum(ep1, MIN_THICKNESS_PHYS_NM * (ep1 > 1e-12))
    ep2 = jnp.maximum(ep2, MIN_THICKNESS_PHYS_NM * (ep2 > 1e-12))

    # Vectorized function to calculate matrix M for a given thickness and index over all wavelengths
    # Need the base function to calculate M for one thickness, one index, one wavelength first
    @jax.jit
    def calculate_M_for_single_thickness(thickness: jnp.ndarray, n_complex_layer: jnp.ndarray, l_val: jnp.ndarray) -> jnp.ndarray:
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
        return jnp.array([[m00, m01], [m10, m11]], dtype=jnp.complex128)

    # Vmap over the wavelength vector (l_vec)
    calculate_M_batch_for_thickness = vmap(calculate_M_for_single_thickness, in_axes=(None, None, 0))

    # Calculate matrices for all wavelengths for 1 QWOT and 2 QWOT
    M_1qwot_batch = calculate_M_batch_for_thickness(ep1, n_complex_for_matrix, l_vec)
    M_2qwot_batch = calculate_M_batch_for_thickness(ep2, n_complex_for_matrix, l_vec)

    # Return shape: (num_wavelengths, 2, 2), (num_wavelengths, 2, 2)
    return M_1qwot_batch, M_2qwot_batch


@jax.jit
def compute_half_product(multiplier_indices: jnp.ndarray, # Shape: (N_half,) -> 0 for 1*QWOT, 1 for 2*QWOT
                         layer_matrices_half: jnp.ndarray  # Shape: (N_half, 2, num_wavelengths, 2, 2)
                         ) -> jnp.ndarray: # Output shape: (num_wavelengths, 2, 2)
    """
    Computes the matrix product for one half of the stack for a given combination
    of 1*QWOT / 2*QWOT choices.
    """
    N_half = layer_matrices_half.shape[0]
    num_wavelengths = layer_matrices_half.shape[2]

    # Initial product is identity matrix, tiled for each wavelength
    init_prod = jnp.tile(jnp.eye(2, dtype=jnp.complex128), (num_wavelengths, 1, 1))

    # Scan function to multiply matrices layer by layer
    def multiply_step(carry_prod: jnp.ndarray, # Shape: (num_wavelengths, 2, 2)
                      layer_idx: int
                      ) -> Tuple[jnp.ndarray, None]:
        # Get the index (0 or 1) for the current layer from multiplier_indices
        multiplier_idx = multiplier_indices[layer_idx]
        # Select the corresponding precomputed matrices (either 1*QWOT or 2*QWOT)
        # Shape M_k: (num_wavelengths, 2, 2)
        M_k = layer_matrices_half[layer_idx, multiplier_idx, :, :, :]
        # Multiply: M_k * carry_prod for each wavelength
        # Order assumes light direction air->substrate, matrices applied right-to-left
        new_prod = vmap(jnp.matmul)(M_k, carry_prod)
        return new_prod, None

    # Iterate through the layers in this half using scan
    final_prod, _ = jax.lax.scan(multiply_step, init_prod, jnp.arange(N_half))

    return final_prod # Shape: (num_wavelengths, 2, 2)


@functools.partial(jax.jit, static_argnames=("targets_tuple_in",))
def combine_and_calc_mse(prod1: jnp.ndarray, # Shape: (num_wavelengths, 2, 2) - Product for first half
                         prod2: jnp.ndarray, # Shape: (num_wavelengths, 2, 2) - Product for second half
                         nSub_arr_in: jnp.ndarray, # Shape: (num_wavelengths,)
                         l_vec_in: jnp.ndarray, # Shape: (num_wavelengths,)
                         targets_tuple_in: Tuple[TargetDict, ...]
                         ) -> jnp.ndarray: # Output: scalar MSE
    """
    Combines the matrix products of the two halves, calculates Transmittance,
    and computes the MSE against targets.
    """
    # Combine the products: M_total = M_half2 * M_half1
    # Order assumes light direction air->substrate
    M_total = vmap(jnp.matmul)(prod2, prod1) # Vmap over wavelength dimension

    # Calculate Transmittance from the total matrix M_total
    # Need a function similar to OpticalCalculator.calculate_single_wavelength_T
    # but operating on a batch of matrices.

    # Simplified T calculation directly here:
    etainc = 1.0 + 0j
    etasub_batch = nSub_arr_in
    m00 = M_total[:, 0, 0]; m01 = M_total[:, 0, 1]
    m10 = M_total[:, 1, 0]; m11 = M_total[:, 1, 1]
    ts_den = (etainc * m00 + etasub_batch * m11 + etainc * etasub_batch * m01 + m10)
    ts_den_abs = jnp.abs(ts_den)
    safe_den = jnp.where(ts_den_abs < 1e-12, 1e-12 + 0j, ts_den)
    ts = (2.0 * etainc) / safe_den
    real_etasub_batch = jnp.real(etasub_batch)
    safe_real_etainc = 1.0 # Assuming real(nInc) is 1 and > 1e-9
    Ts_complex = (real_etasub_batch / safe_real_etainc) * (ts * jnp.conj(ts))
    Ts = jnp.real(Ts_complex)
    Ts = jnp.where(ts_den_abs < 1e-12, 0.0, jnp.nan_to_num(Ts, nan=0.0)) # Shape: (num_wavelengths,)

    # Calculate MSE using the basic MSE function
    mse = calculate_mse_basic_jax(Ts, l_vec_in, targets_tuple_in)
    return mse # Scalar MSE value


def execute_qwot_scan(current_l0: float,
                      initial_layer_number: int,
                      material_manager: MaterialManager, # Need this to get nk at l0
                      nH_material_id: MaterialIdentifier,
                      nL_material_id: MaterialIdentifier,
                      nSub_material_id: MaterialIdentifier,
                      l_vec_eval_sparse_np: np.ndarray, # Sparse grid for faster scan
                      active_targets: Sequence[TargetDict]
                      ) -> Tuple[float, Optional[np.ndarray], List[str]]:
    """
    Executes the exhaustive QWOT scan (1.0 or 2.0 multipliers) for a *single* l0.
    Uses the split-stack approach with JAX for performance.

    Args:
        current_l0: The centering wavelength to use for QWOT calculation.
        initial_layer_number: The number of layers (N).
        # ... other managers and identifiers ...
        l_vec_eval_sparse_np: The sparser wavelength grid for evaluation.
        active_targets: Validated active targets.

    Returns:
        Tuple: (best_mse_found_for_l0, best_multipliers_found_for_l0, logs)
               multipliers are None if no valid result found.
    """
    logs = []
    l_vec_eval_sparse_jax = jnp.asarray(l_vec_eval_sparse_np)
    num_combinations = 2**initial_layer_number
    logs.append(f"[Scan l0={current_l0:.2f}] Preparing scan for {num_combinations:,} combinations...")

    # --- Get indices at l0 and precompute layer matrices ---
    try:
        nH_c_l0 = material_manager.get_nk_at_lambda(nH_material_id, current_l0)
        nL_c_l0 = material_manager.get_nk_at_lambda(nL_material_id, current_l0)
        nSub_arr_scan = material_manager.get_nk_array(nSub_material_id, l_vec_eval_sparse_jax)

        t_mat_start = time.time()
        layer_matrices_list = []
        for i in range(initial_layer_number):
             m1, m2 = get_layer_matrices_qwot(i, initial_layer_number, current_l0,
                                               jnp.asarray(nH_c_l0), jnp.asarray(nL_c_l0),
                                               l_vec_eval_sparse_jax)
             layer_matrices_list.append(jnp.stack([m1, m2], axis=0)) # Stack the 1xQWOT and 2xQWOT results

        # Shape: (N, 2, L, 2, 2) -> N layers, 2=multiplier choice, L wavelengths
        all_layer_matrices = jnp.stack(layer_matrices_list, axis=0)
        all_layer_matrices.block_until_ready() # Ensure computation is done
        logs.append(f"  Matrix precomputation (l0={current_l0:.2f}) took {time.time() - t_mat_start:.3f}s.")

    except Exception as e_prep:
        logs.append(f"  ERROR preparing scan for l0={current_l0:.2f}: {e_prep}")
        return np.inf, None, logs

    # --- Split Stack Calculation ---
    N1 = initial_layer_number // 2
    N2 = initial_layer_number - N1
    num_comb1 = 2**N1
    num_comb2 = 2**N2

    # Generate indices for selecting 1x or 2x QWOT for each layer in each half
    # Shape: (num_combX, NX) -> e.g., [[0,0,0], [1,0,0], [0,1,0], ...]
    indices1 = jnp.arange(num_comb1)
    indices2 = jnp.arange(num_comb2)
    powers1 = 2**jnp.arange(N1)
    powers2 = 2**jnp.arange(N2)
    multiplier_indices1 = jnp.not_equal(indices1[:, None] & powers1, 0).astype(jnp.int32)
    multiplier_indices2 = jnp.not_equal(indices2[:, None] & powers2, 0).astype(jnp.int32)

    # Split the precomputed matrices
    matrices_half1 = all_layer_matrices[:N1] # Shape: (N1, 2, L, 2, 2)
    matrices_half2 = all_layer_matrices[N1:] # Shape: (N2, 2, L, 2, 2)

    # --- Calculate products for each half ---
    try:
        t_half1_start = time.time()
        # Vmap over all combinations for the first half
        # Input shapes: (num_comb1, N1), (N1, 2, L, 2, 2)
        # Output shape: (num_comb1, L, 2, 2)
        partial_products1 = vmap(compute_half_product, in_axes=(0, None))(multiplier_indices1, matrices_half1)
        partial_products1.block_until_ready()
        logs.append(f"  Partial products 1/2 (l0={current_l0:.2f}, {num_comb1} combos) took {time.time() - t_half1_start:.3f}s.")

        t_half2_start = time.time()
        # Vmap over all combinations for the second half
        # Input shapes: (num_comb2, N2), (N2, 2, L, 2, 2)
        # Output shape: (num_comb2, L, 2, 2)
        partial_products2 = vmap(compute_half_product, in_axes=(0, None))(multiplier_indices2, matrices_half2)
        partial_products2.block_until_ready()
        logs.append(f"  Partial products 2/2 (l0={current_l0:.2f}, {num_comb2} combos) took {time.time() - t_half2_start:.3f}s.")

    except Exception as e_half:
         logs.append(f"  ERROR calculating partial products for l0={current_l0:.2f}: {e_half}")
         logs.append(traceback.format_exc(limit=1)) # Add traceback
         return np.inf, None, logs

    # --- Combine halves and calculate MSE ---
    try:
        t_combine_start = time.time()
        active_targets_tuple = tuple((t['min'], t['max'], t['target_min'], t['target_max']) for t in active_targets)

        # Vmap the combine_and_calc_mse function over all pairs of partial products
        # Inner vmap iterates over prod2 for a fixed prod1
        # Outer vmap iterates over prod1
        vmap_inner = vmap(combine_and_calc_mse, in_axes=(None, 0, None, None, None)) # vmap over prod2
        vmap_outer = vmap(vmap_inner, in_axes=(0, None, None, None, None)) # vmap over prod1

        # Input shapes: (num_comb1, L, 2, 2), (num_comb2, L, 2, 2), (L,), (L,), static
        # Output shape: (num_comb1, num_comb2) -> MSE for each combination
        all_mses_nested = vmap_outer(partial_products1, partial_products2, nSub_arr_scan, l_vec_eval_sparse_jax, active_targets_tuple)
        all_mses_nested.block_until_ready()
        logs.append(f"  Combination and MSE calculation (l0={current_l0:.2f}, {num_combinations} combos) took {time.time() - t_combine_start:.3f}s.")

        # Find the best combination
        all_mses_flat = all_mses_nested.reshape(-1) # Flatten the MSE array
        if not jnp.any(jnp.isfinite(all_mses_flat)):
             logs.append(f"  WARNING: All MSE values were non-finite for l0={current_l0:.2f}.")
             return np.inf, None, logs

        best_idx_flat = jnp.argmin(all_mses_flat) # Index of the minimum MSE in the flattened array
        current_best_mse = float(all_mses_flat[best_idx_flat]) # The minimum MSE value

        # Convert flat index back to 2D index (for half1 and half2 combinations)
        best_idx_half1, best_idx_half2 = jnp.unravel_index(best_idx_flat, (num_comb1, num_comb2))

        # Retrieve the multiplier indices (0 or 1) for the best combination
        best_indices_h1 = multiplier_indices1[best_idx_half1] # Shape: (N1,)
        best_indices_h2 = multiplier_indices2[best_idx_half2] # Shape: (N2,)

        # Convert indices (0/1) to multipliers (1.0/2.0)
        best_multipliers_h1 = 1.0 + best_indices_h1.astype(jnp.float64)
        best_multipliers_h2 = 1.0 + best_indices_h2.astype(jnp.float64)

        # Combine the multipliers for the full stack
        current_best_multipliers = jnp.concatenate([best_multipliers_h1, best_multipliers_h2])

        logs.append(f"  Best Scan MSE for l0={current_l0:.2f}: {current_best_mse:.6e}")
        # logs.append(f"  Best Multipliers: {current_best_multipliers}") # Maybe too verbose

        return current_best_mse, np.asarray(current_best_multipliers), logs

    except Exception as e_combine:
        logs.append(f"  ERROR combining/calculating MSE for l0={current_l0:.2f}: {e_combine}")
        logs.append(traceback.format_exc(limit=1))
        return np.inf, None, logs

# --- High-Level Workflows (Examples) ---

def run_local_optimization_workflow(
                          initial_design: Dict, # Contains ep_vector, materials, etc.
                          material_manager: MaterialManager,
                          optical_calculator: OpticalCalculator,
                          active_targets: Sequence[TargetDict],
                          l_vec_optim_np: np.ndarray,
                          optimizer_options: Dict[str, Any],
                          min_thickness_phys: float = MIN_THICKNESS_PHYS_NM
                          ) -> Dict:
    """Orchestrates a single local optimization run."""
    print("\n--- Starting Local Optimization Workflow ---") # Use logging
    results = {}
    logs = []
    try:
        ep_start = np.asarray(initial_design['ep_vector'])
        mat_seq = initial_design['material_sequence']
        nSub = initial_design['nSub_material']
        nInc = initial_design.get('nInc_material', "AIR") # Use default if not provided

        final_ep, success, cost, msg, nit, nfev, core_logs = run_core_optimization(
            ep_start, material_manager, optical_calculator, mat_seq, nSub, nInc,
            active_targets, l_vec_optim_np, optimizer_options, min_thickness_phys
        )
        logs.extend(core_logs)
        results = {
            'ep_vector': final_ep.tolist() if final_ep is not None else None,
            'material_sequence': mat_seq, # Sequence doesn't change here
            'nSub_material': nSub,
            'nInc_material': nInc,
            'success': success,
            'final_cost': cost,
            'message': msg,
            'iterations': nit,
            'evaluations': nfev,
            'logs': logs
        }
        print(f"--- Local Optimization Workflow Finished: Success={success}, Cost={cost:.6e} ---")

    except Exception as e:
        print(f"ERROR in local optimization workflow: {e}") # Use logging
        logs.append(f"Workflow Error: {e}")
        logs.append(traceback.format_exc())
        results = {'success': False, 'message': str(e), 'logs': logs}

    return results


# --- Design Persistence ---

def save_design_json(filepath: str, design_params: Dict, design_targets: List[Dict], optimized_results: Optional[Dict] = None):
    """Saves design parameters, targets, and optionally optimized results to JSON."""
    save_data = {
        'params': design_params,
        'targets': design_targets,
        'save_timestamp': datetime.datetime.now().isoformat()
    }
    if optimized_results:
        # Only include serializable parts of the optimization results
        opt_data_to_save = {
            'ep_vector': optimized_results.get('ep_vector'), # Should be list already if saved correctly
            'material_sequence': optimized_results.get('material_sequence'),
            'success': optimized_results.get('success'),
            'final_cost': optimized_results.get('final_cost'),
            'message': optimized_results.get('message'),
            'iterations': optimized_results.get('iterations'),
            'evaluations': optimized_results.get('evaluations')
            # Avoid saving large objects like logs directly if not needed
        }
        save_data['optimized_results'] = opt_data_to_save

    try:
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=4)
        print(f"Design saved successfully to: {filepath}") # Use logging
    except Exception as e:
        print(f"ERROR saving design to {filepath}: {e}") # Use logging
        raise # Re-raise error for caller to handle


def load_design_json(filepath: str) -> Dict:
    """Loads design data from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        # Basic validation
        if 'params' not in loaded_data or 'targets' not in loaded_data:
            raise ValueError("Invalid design file: missing 'params' or 'targets' section.")
        print(f"Design loaded successfully from: {filepath}") # Use logging
        return loaded_data
    except FileNotFoundError:
        print(f"ERROR: Design file not found: {filepath}") # Use logging
        raise
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON format in {filepath}: {e}") # Use logging
        raise
    except Exception as e:
        print(f"ERROR loading design from {filepath}: {e}") # Use logging
        raise

# --- Main Execution Block (Example) ---
if __name__ == '__main__':

    print("\n--- Running Refactored Example (Continued) ---")

    # (Re-initialize managers as in the previous block if needed)
    try:
        material_mgr = MaterialManager(excel_file_path=EXCEL_FILE_PATH)
        calculator = OpticalCalculator(material_mgr)
    except Exception as e:
        print(f"Error initializing managers: {e}")
        exit()

    # Define design parameters (replace with loading from file or Streamlit inputs)
    design_params = {
        'l0': 550.0,
        'initial_layer_number': 5,
        'nH_material': "Nb2O5-Helios",
        'nL_material': "SiO2-Helios",
        'nSub_material': "Fused Silica",
        'nInc_material': "AIR"
        # Add other relevant params like optimizer settings if needed
    }

    # Define targets (replace with loading or Streamlit inputs)
    active_targets = [
        {'min': 450, 'max': 500, 'target_min': 0.0, 'target_max': 0.0}, # High reflection band
        {'min': 540, 'max': 560, 'target_min': 1.0, 'target_max': 1.0}, # High transmission point
        {'min': 600, 'max': 700, 'target_min': 0.0, 'target_max': 0.0}  # High reflection band
    ]

    # Define optimization settings
    optimizer_options = {'maxiter': 50, 'maxfun': 75, 'disp': False, 'ftol': 1e-9, 'gtol': 1e-6}
    l_vec_optim_np = np.linspace(400, 800, 101) # Wavelengths for optimization

    # --- Generate Initial Design ---
    try:
        initial_num_layers = design_params['initial_layer_number']
        initial_l0 = design_params['l0']
        initial_mat_seq = [design_params['nH_material'] if i % 2 == 0 else design_params['nL_material']
                           for i in range(initial_num_layers)]
        initial_multipliers = [1.0] * initial_num_layers
        initial_ep = calculate_initial_ep(material_mgr, initial_multipliers, initial_l0, initial_mat_seq)

        initial_design_state = {
            'ep_vector': initial_ep,
            'material_sequence': initial_mat_seq,
            'nSub_material': design_params['nSub_material'],
            'nInc_material': design_params['nInc_material']
        }
        print(f"Generated initial design with {len(initial_ep)} layers.")

        # --- Run Local Optimization ---
        opt_results = run_local_optimization_workflow(
            initial_design=initial_design_state,
            material_manager=material_mgr,
            optical_calculator=calculator,
            active_targets=active_targets,
            l_vec_optim_np=l_vec_optim_np,
            optimizer_options=optimizer_options
        )

        # --- Plot Optimized Result (Conceptual) ---
        if opt_results.get('success'):
            print("Optimization reported success. Calculating final spectrum...")
            final_ep = opt_results.get('ep_vector')
            if final_ep:
                 l_vec_plot = np.linspace(400, 800, 401) # Finer grid for plotting
                 final_spectrum = calculator.calculate_T_spectrum(
                     ep_vector=final_ep,
                     material_sequence=opt_results['material_sequence'],
                     nSub_material=opt_results['nSub_material'],
                     l_vec=l_vec_plot,
                     nInc_material=opt_results['nInc_material']
                 )
                 print(f"Final spectrum calculated. Final Cost was: {opt_results.get('final_cost'):.6e}")
                 # In Streamlit, you would use st.line_chart or matplotlib here
                 # e.g., plt.plot(final_spectrum['l'], final_spectrum['Ts'])
            else:
                 print("Optimized ep_vector not found in results.")
        else:
            print(f"Optimization failed: {opt_results.get('message')}")

        # --- Save Design Example ---
        # save_design_json("refactored_design_example.json", design_params, active_targets, opt_results)

        # --- Load Design Example ---
        # loaded_data = load_design_json("refactored_design_example.json")
        # print("\nLoaded Design Params:", loaded_data.get('params'))
        # print("Loaded Optimized Cost:", loaded_data.get('optimized_results', {}).get('final_cost'))


    except Exception as e:
        print(f"\n--- An error occurred during the refactored example run ---")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()

# --- High-Level Workflow Implementations ---

def run_auto_mode(initial_design: Dict,
                  material_manager: MaterialManager,
                  optical_calculator: OpticalCalculator,
                  active_targets: Sequence[TargetDict],
                  l_vec_optim_np: np.ndarray,
                  optimizer_options: Dict[str, Any],
                  l0_for_materials: float,
                  min_thickness_phys: float = MIN_THICKNESS_PHYS_NM,
                  thin_removal_threshold_nm: Optional[float] = 1.0, # Threshold from GUI
                  max_cycles: int = AUTO_MAX_CYCLES,
                  needles_per_cycle: int = AUTO_NEEDLES_PER_CYCLE,
                  scan_step_nm: float = DEFAULT_NEEDLE_SCAN_STEP_NM,
                  base_needle_thickness_nm: float = BASE_NEEDLE_THICKNESS_NM
                  ) -> Dict:
    """
    Runs the automatic optimization mode: cycles of Needle -> Thin Removal -> Opt.

    Args:
        initial_design: Dict containing starting 'ep_vector', 'material_sequence', etc.
        # ... other managers, calculators, targets, optimization settings ...
        thin_removal_threshold_nm: Layers thinner than this (but > min_phys) are targeted for removal.
        max_cycles: Maximum number of full cycles (Needle + Thin Removal) to run.
        needles_per_cycle: Number of needle insertion attempts within each cycle.
        # ... other needle parameters ...

    Returns:
        A dictionary containing the final design state and optimization summary.
    """
    print("\n" + "#"*10 + f" Starting Auto Mode (Max {max_cycles} Cycles) " + "#"*10) # Use logging
    overall_logs = []
    start_time_auto = time.time()

    best_ep_so_far = np.asarray(initial_design['ep_vector']).copy()
    best_mat_seq_so_far = list(initial_design['material_sequence'])
    best_mse_so_far = np.inf
    num_cycles_done = 0
    termination_reason = f"Max {max_cycles} cycles reached"
    initial_opt_done = False # Flag if initial optimization was performed

    total_iters_auto = 0
    total_evals_auto = 0
    optim_runs_auto = 0 # Count successful optimization runs (core, needle re-opt, thin re-opt)

    nSub = initial_design['nSub_material']
    nInc = initial_design.get('nInc_material', "AIR")

    # --- Initial State & Cost ---
    try:
        # Ensure the starting point is optimized (run core optimization once if needed)
        # This establishes a baseline MSE and ensures the start isn't just nominal QWOT
        print("  Auto Mode: Running initial optimization check/run...")
        initial_opt_results = run_core_optimization(
            best_ep_so_far, material_manager, optical_calculator, best_mat_seq_so_far,
            nSub, nInc, active_targets, l_vec_optim_np, optimizer_options, min_thickness_phys
        )
        ep_after_initial, initial_success, initial_mse, initial_msg, initial_nit, initial_nfev, initial_logs = initial_opt_results
        overall_logs.extend(initial_logs)

        if not initial_success:
            raise RuntimeError(f"Initial optimization failed in Auto Mode setup: {initial_msg}")

        best_ep_so_far = ep_after_initial.copy()
        best_mse_so_far = initial_mse
        initial_opt_done = True
        total_iters_auto += initial_nit
        total_evals_auto += initial_nfev
        optim_runs_auto += 1
        print(f"  Initial optimization finished. Start MSE: {best_mse_so_far:.6e} ({len(best_ep_so_far)} layers)")

        if not np.isfinite(best_mse_so_far):
            raise ValueError("Starting MSE for auto mode cycles is not finite.")

        # --- Auto Mode Cycles ---
        for cycle_num in range(max_cycles):
            cycle_logs = []
            cycle_logs.append(f"\n--- Auto Cycle {cycle_num + 1} / {max_cycles} ---")
            print(cycle_logs[-1]) # Use logging
            mse_at_cycle_start = best_mse_so_far
            ep_at_cycle_start = best_ep_so_far.copy()
            mat_seq_at_cycle_start = list(best_mat_seq_so_far)
            cycle_improved_globally = False # Did this cycle improve vs the absolute best MSE?

            # --- 1. Needle Phase ---
            cycle_logs.append(f"  [Cycle {cycle_num+1}] Running {needles_per_cycle} needle iterations...")
            print(cycle_logs[-1]) # Use logging
            needle_results = run_needle_iterations(
                best_ep_so_far, best_mat_seq_so_far, needles_per_cycle,
                material_manager, optical_calculator, nSub, nInc, active_targets,
                l_vec_optim_np, optimizer_options, l0_for_materials, min_thickness_phys,
                scan_step_nm, base_needle_thickness_nm, log_prefix=f"    [Auto Cycle {cycle_num+1} Needle] "
            )
            ep_after_needles, mat_seq_after_needles, mse_after_needles, needle_logs, nit_needles, nfev_needles, reopts_needles = needle_results
            cycle_logs.extend(needle_logs)
            total_iters_auto += nit_needles
            total_evals_auto += nfev_needles
            optim_runs_auto += reopts_needles

            # Update current state after needle phase
            best_ep_so_far = ep_after_needles.copy()
            best_mat_seq_so_far = list(mat_seq_after_needles)
            current_mse_cycle = mse_after_needles # MSE at this point in the cycle

            if current_mse_cycle < mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE:
                 cycle_logs.append(f"    Needle phase improved MSE within cycle ({mse_at_cycle_start:.6e} -> {current_mse_cycle:.6e}).")
                 if current_mse_cycle < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                      cycle_improved_globally = True # Also improved vs global best
            else:
                 cycle_logs.append(f"    Needle phase did not significantly improve MSE within cycle ({mse_at_cycle_start:.6e} -> {current_mse_cycle:.6e}).")

            # Store the best MSE found so far globally (could be from this needle phase or previous cycles)
            best_mse_so_far = min(best_mse_so_far, current_mse_cycle)

            # --- 2. Thin Removal Phase ---
            cycle_logs.append(f"  [Cycle {cycle_num+1}] Running Thin Removal + Re-Opt Phase (Threshold: {thin_removal_threshold_nm} nm)...")
            print(cycle_logs[-1]) # Use logging
            layers_removed_this_cycle = 0
            max_thinning_iterations = len(best_ep_so_far) + 1 # Safety limit
            for thin_iter in range(max_thinning_iterations):
                num_layers_before_removal = len(best_ep_so_far)
                if num_layers_before_removal <= 1:
                    cycle_logs.append("    Structure too small for further thin removal."); break

                # Attempt to remove one layer
                removal_results = perform_layer_merge_or_removal(
                    best_ep_so_far, min_thickness_phys, thin_removal_threshold_nm
                )
                ep_after_single_removal, structure_changed, removal_logs, removed_idx = removal_results
                cycle_logs.extend(removal_logs)

                if structure_changed:
                    # Need to update the material sequence based on the removal logic
                    # This assumes the removal logic consistently handles indices and merging
                    original_idx_removed = removed_idx
                    if original_idx_removed == 0: # Removed layers 0 and 1
                        best_mat_seq_so_far = best_mat_seq_so_far[2:]
                    elif original_idx_removed == num_layers_before_removal - 1: # Removed last layer
                        best_mat_seq_so_far = best_mat_seq_so_far[:-1]
                    else: # Merged layers around original_idx_removed
                        best_mat_seq_so_far = (best_mat_seq_so_far[:original_idx_removed - 1] + # Before left neighbour
                                             [best_mat_seq_so_far[original_idx_removed - 1]] + # Left neighbour (material type doesn't change)
                                             best_mat_seq_so_far[original_idx_removed + 2:])   # After right neighbour

                    # Check consistency after modifying sequence
                    if len(ep_after_single_removal) != len(best_mat_seq_so_far):
                         cycle_logs.append("    CRITICAL ERROR: Material sequence length mismatch after removal logic! Stopping thin removal.")
                         # Revert state to before this problematic removal attempt
                         best_ep_so_far = ep_at_cycle_start # Revert to start of cycle for safety
                         best_mat_seq_so_far = mat_seq_at_cycle_start
                         best_mse_so_far = mse_at_cycle_start
                         break # Stop thinning loop

                    layers_removed_this_cycle += 1
                    cycle_logs.append(f"    Layer removed. Re-optimizing {len(ep_after_single_removal)} layers...")
                    print(cycle_logs[-1]) # Use logging

                    # Re-optimize the structure after removal
                    reopt_results = run_core_optimization(
                        ep_after_single_removal, material_manager, optical_calculator,
                        best_mat_seq_so_far, # Use updated sequence
                        nSub, nInc, active_targets, l_vec_optim_np, optimizer_options, min_thickness_phys
                    )
                    ep_after_thin_reopt, thin_reopt_success, mse_after_thin_reopt, thin_reopt_msg, nit_thin, nfev_thin, thin_reopt_logs = reopt_results
                    cycle_logs.extend(thin_reopt_logs)

                    if thin_reopt_success:
                        total_iters_auto += nit_thin
                        total_evals_auto += nfev_thin
                        optim_runs_auto += 1
                        cycle_logs.append(f"      Re-optimization after removal {layers_removed_this_cycle} successful. New MSE: {mse_after_thin_reopt:.6e}")
                        best_ep_so_far = ep_after_thin_reopt.copy() # Update current state
                        # Material sequence already updated before re-opt
                        current_mse_cycle = mse_after_thin_reopt
                        if current_mse_cycle < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                             cycle_logs.append("        MSE improved globally. Updating best state.")
                             best_mse_so_far = current_mse_cycle
                             cycle_improved_globally = True
                        # Continue thinning loop with the newly optimized structure
                    else:
                        cycle_logs.append(f"    WARNING: Re-optimization after removal {layers_removed_this_cycle} failed ({thin_reopt_msg}). Stopping removal phase.")
                        # Revert to the state *after* removal but *before* the failed re-opt attempt
                        best_ep_so_far = ep_after_single_removal.copy()
                        # Material sequence remains as updated after removal
                        # Recalculate MSE for this reverted state
                        try:
                            reverted_cost_jax = cost_fn_compiled(jnp.asarray(best_ep_so_far), layer_indices_arr_transposed=indices_arr_transposed_temp) # Reuse indices? Needs recompute!
                            # Need to recompute indices here for the reverted state!
                            layer_indices_list_revert = [material_manager.get_nk_array(mat, l_vec_optim_jax) for mat in best_mat_seq_so_far]
                            indices_arr_transposed_revert = jnp.stack(layer_indices_list_revert, axis=0).T
                            reverted_cost_jax = cost_fn_compiled(jnp.asarray(best_ep_so_far), layer_indices_arr_transposed=indices_arr_transposed_revert)
                            current_mse_cycle = float(np.array(reverted_cost_jax))
                            if not np.isfinite(current_mse_cycle): current_mse_cycle = np.inf
                            cycle_logs.append(f"      MSE after failed re-opt (reverted): {current_mse_cycle:.6e}")
                        except Exception as revert_e:
                            cycle_logs.append(f"      ERROR recalculating MSE after failed re-opt: {revert_e}")
                            current_mse_cycle = np.inf
                        # Check if reverted state is globally best
                        if current_mse_cycle < best_mse_so_far - MSE_IMPROVEMENT_TOLERANCE:
                             best_mse_so_far = current_mse_cycle
                             cycle_improved_globally = True
                        break # Stop thinning loop for this cycle

                else: # Structure not changed by removal attempt
                    cycle_logs.append("    No more layers found below threshold or removal failed. Finishing thin removal phase.")
                    break # Exit thinning loop for this cycle

            cycle_logs.append(f"  [Cycle {cycle_num+1}] Thin Removal + Re-Opt Phase finished. {layers_removed_this_cycle} layer(s) removed.")
            num_cycles_done += 1
            overall_logs.extend(cycle_logs)

            # --- Check for cycle improvement ---
            if not cycle_improved_globally and not (current_mse_cycle < mse_at_cycle_start - MSE_IMPROVEMENT_TOLERANCE):
                print(f"No significant improvement in Cycle {cycle_num + 1} (Start MSE: {mse_at_cycle_start:.6e}, End MSE: {current_mse_cycle:.6e}). Stopping Auto Mode.") # Log
                termination_reason = "No improvement"
                # Check if cycle actually made things worse compared to start of cycle
                if current_mse_cycle > mse_at_cycle_start + MSE_IMPROVEMENT_TOLERANCE:
                     print(f"Reverting to state before Cycle {cycle_num + 1}.") # Log
                     best_ep_so_far = ep_at_cycle_start
                     best_mat_seq_so_far = mat_seq_at_cycle_start
                     best_mse_so_far = mse_at_cycle_start
                break # Exit main auto mode loop

    except (ValueError, RuntimeError) as e:
        print(f"ERROR during Auto Mode workflow: {e}") # Log
        overall_logs.append(f"Workflow Error: {e}")
        overall_logs.append(traceback.format_exc())
        termination_reason = f"Error: {e}"
    except Exception as e:
        print(f"UNEXPECTED ERROR during Auto Mode workflow: {e}") # Log
        overall_logs.append(f"Unexpected Workflow Error: {e}")
        overall_logs.append(traceback.format_exc())
        termination_reason = f"Unexpected Error: {e}"

    # --- Final Report ---
    final_num_layers = len(best_ep_so_far) if best_ep_so_far is not None else 0
    print("\n" + "#"*10 + f" Auto Mode Finished ({num_cycles_done} Cycles) " + "#"*10) # Log
    print(f"Termination Reason: {termination_reason}") # Log
    print(f"Final Best MSE: {best_mse_so_far:.6e} with {final_num_layers} layers.") # Log
    avg_nit_str = f"{total_iters_auto / optim_runs_auto:.1f}" if optim_runs_auto > 0 else "N/A"
    avg_nfev_str = f"{total_evals_auto / optim_runs_auto:.1f}" if optim_runs_auto > 0 else "N/A"
    print(f"Total Optimizations: {optim_runs_auto}, Avg Iter/Eval: {avg_nit_str}/{avg_nfev_str}") # Log
    print(f"Total Auto Mode Time: {time.time() - start_time_auto:.3f}s") # Log

    return {
        'ep_vector': best_ep_so_far.tolist() if best_ep_so_far is not None else None,
        'material_sequence': best_mat_seq_so_far if best_ep_so_far is not None else None,
        'nSub_material': nSub,
        'nInc_material': nInc,
        'success': np.isfinite(best_mse_so_far), # Define success based on finite MSE
        'final_cost': best_mse_so_far,
        'message': f"Auto mode finished after {num_cycles_done} cycles. Reason: {termination_reason}",
        'iterations_sum': total_iters_auto,
        'evaluations_sum': total_evals_auto,
        'optim_runs': optim_runs_auto,
        'logs': overall_logs
    }


def run_qwot_scan_and_optimize(design_params: Dict,
                              material_manager: MaterialManager,
                              optical_calculator: OpticalCalculator,
                              active_targets: Sequence[TargetDict],
                              l_vec_eval_full_np: np.ndarray, # Grid for final optim
                              l_vec_eval_sparse_np: np.ndarray, # Grid for fast scan
                              optimizer_options: Dict[str, Any],
                              l0_values_to_test: Optional[Sequence[float]] = None,
                              min_thickness_phys: float = MIN_THICKNESS_PHYS_NM
                              ) -> Dict:
    """
    Runs the exhaustive QWOT scan (1 or 2 multiplier) over specified l0 values,
    then locally optimizes the best candidate(s) found.

    Args:
        design_params: Dict with 'initial_layer_number', 'l0', material IDs, etc.
        # ... managers, calculator, targets, grids, optimizer options ...
        l0_values_to_test: Optional list of l0 values. If None, uses default +/- 25% around l0 in design_params.

    Returns:
        A dictionary containing the best final design state and optimization summary.
    """
    print("\n" + "#"*10 + " Starting Exhaustive QWOT Scan + Optimize Workflow " + "#"*10) # Use logging
    overall_logs = []
    start_time_scan = time.time()
    final_best_ep = None
    final_best_mat_seq = None
    final_best_mse = np.inf
    final_best_l0 = None
    final_best_initial_multipliers = None
    overall_optim_nit = 0
    overall_optim_nfev = 0
    successful_optim_count = 0

    try:
        # --- Validate Inputs ---
        initial_layer_number = design_params.get('initial_layer_number')
        l0_nominal = design_params.get('l0')
        nH_id = design_params.get('nH_material')
        nL_id = design_params.get('nL_material')
        nSub_id = design_params.get('nSub_material')
        nInc_id = design_params.get('nInc_material', "AIR")

        if not all([initial_layer_number, l0_nominal, nH_id, nL_id, nSub_id]):
            raise ValueError("Missing required parameters in design_params for QWOT scan.")
        if initial_layer_number <= 0:
             raise ValueError("Initial Layer Number must be positive.")
        if not active_targets:
             raise ValueError("QWOT Scan requires active targets.")

        # --- Determine l0 values ---
        if l0_values_to_test is None:
             l0_center = float(l0_nominal)
             l0_test_list = sorted(list(set([l0_center, l0_center * 1.25, l0_center * 0.75])))
        else:
             l0_test_list = sorted([float(l) for l in l0_values_to_test])
        l0_test_list = [l for l in l0_test_list if l > 1e-6] # Filter out invalid values
        num_l0_tests = len(l0_test_list)
        if num_l0_tests == 0:
             raise ValueError("No valid l0 values provided or generated for testing.")

        # --- Run Scan for each l0 ---
        num_combinations = 2**initial_layer_number
        total_scan_evals = num_combinations * num_l0_tests
        overall_logs.append(f"Starting QWOT Scan for N={initial_layer_number} over {num_l0_tests} l0 values: {[f'{l:.2f}' for l in l0_test_list]}.")
        overall_logs.append(f"Total scan combinations: {total_scan_evals:,}.")
        print(overall_logs[-2]); print(overall_logs[-1]) # Use logging

        scan_candidates = []
        for l0_idx, current_l0 in enumerate(l0_test_list):
            print(f"\n--- Running QWOT Scan for l0 = {current_l0:.2f} nm ({l0_idx+1}/{num_l0_tests}) ---") # Log
            mse_scan, multipliers_scan, scan_logs = execute_qwot_scan(
                current_l0, initial_layer_number, material_manager,
                nH_id, nL_id, nSub_id, l_vec_eval_sparse_np, active_targets
            )
            overall_logs.extend(scan_logs)
            if np.isfinite(mse_scan) and multipliers_scan is not None:
                 overall_logs.append(f"  Scan Candidate Found: l0={current_l0:.2f}, MSE={mse_scan:.6e}")
                 scan_candidates.append({
                     'l0': current_l0,
                     'mse_scan': mse_scan,
                     'multipliers': multipliers_scan # NumPy array
                 })
            else:
                 overall_logs.append(f"  No valid candidate found for l0={current_l0:.2f}")

        if not scan_candidates:
            raise RuntimeError("QWOT Scan found no valid initial candidates over all tested l0 values.")

        # --- Optimize the best candidate(s) ---
        scan_candidates.sort(key=lambda c: c['mse_scan'])
        num_to_optimize = min(len(scan_candidates), max(1, num_l0_tests)) # Optimize at least 1, up to num_l0 tested
        overall_logs.append(f"\n--- QWOT Scan finished. Optimizing top {num_to_optimize} candidate(s) ---")
        print(overall_logs[-1]) # Use logging

        for idx, candidate in enumerate(scan_candidates[:num_to_optimize]):
            cand_l0 = candidate['l0']
            cand_mult = candidate['multipliers']
            cand_mse_scan = candidate['mse_scan']
            print(f"\n--- Optimizing Candidate {idx+1}/{num_to_optimize} (from l0={cand_l0:.2f}, scan MSE={cand_mse_scan:.6e}) ---") # Log
            overall_logs.append(f"Optimizing Candidate {idx+1}/{num_to_optimize} (l0={cand_l0:.2f})")

            try:
                # Define material sequence based on H/L pattern
                cand_mat_seq = [nH_id if i % 2 == 0 else nL_id for i in range(initial_layer_number)]
                # Calculate starting ep vector for this candidate
                ep_start_optim = calculate_initial_ep(material_manager, cand_mult, cand_l0, cand_mat_seq)

                # Run core optimization
                opt_results = run_core_optimization(
                    ep_start_optim, material_manager, optical_calculator, cand_mat_seq,
                    nSub_id, nInc_id, active_targets, l_vec_eval_full_np, # Use full grid here
                    optimizer_options, min_thickness_phys
                )
                res_ep, res_success, res_cost, res_msg, res_nit, res_nfev, res_logs = opt_results
                overall_logs.extend(res_logs)

                if res_success:
                    successful_optim_count += 1
                    overall_optim_nit += res_nit
                    overall_optim_nfev += res_nfev
                    overall_logs.append(f"  Optimization successful for candidate {idx+1}. Final MSE: {res_cost:.6e}")
                    print(f"  Optimization successful for candidate {idx+1}. Final MSE: {res_cost:.6e}") # Log
                    if res_cost < final_best_mse:
                         overall_logs.append(f"  *** New global best found! MSE improved from {final_best_mse:.6e} to {res_cost:.6e} ***")
                         print(overall_logs[-1]) # Log
                         final_best_mse = res_cost
                         final_best_ep = res_ep.copy()
                         final_best_mat_seq = list(cand_mat_seq) # Store sequence corresponding to best ep
                         final_best_l0 = cand_l0
                         final_best_initial_multipliers = cand_mult.copy()
                else:
                    overall_logs.append(f"  Optimization FAILED for candidate {idx+1}. Msg: {res_msg}, Cost: {res_cost:.3e}")
                    print(f"  Optimization FAILED for candidate {idx+1}.") # Log

            except Exception as e_optim_cand:
                overall_logs.append(f"ERROR optimizing candidate {idx+1}: {e_optim_cand}")
                overall_logs.append(traceback.format_exc(limit=1))
                print(f"ERROR optimizing candidate {idx+1}: {e_optim_cand}") # Log

        # --- Final Result ---
        if final_best_ep is None:
            raise RuntimeError("Local optimization failed for all candidates from QWOT scan.")

        print("\n--- QWOT Scan + Optimize Finished ---") # Log
        print(f"Final Best MSE: {final_best_mse:.6e} ({len(final_best_ep)} layers)") # Log
        print(f"Originating from l0 = {final_best_l0:.2f} nm") # Log
        avg_nit_str = f"{overall_optim_nit / successful_optim_count:.1f}" if successful_optim_count > 0 else "N/A"
        avg_nfev_str = f"{overall_optim_nfev / successful_optim_count:.1f}" if successful_optim_count > 0 else "N/A"
        print(f"Total successful optimizations: {successful_optim_count}, Avg Iter/Eval: {avg_nit_str}/{avg_nfev_str}") # Log
        print(f"Total Scan+Opt Time: {time.time() - start_time_scan:.3f}s") # Log

        return {
            'ep_vector': final_best_ep.tolist(),
            'material_sequence': final_best_mat_seq,
            'nSub_material': nSub_id,
            'nInc_material': nInc_id,
            'success': True,
            'final_cost': final_best_mse,
            'message': f"QWOT Scan + Opt finished. Best result from l0={final_best_l0:.2f}.",
            'origin_l0': final_best_l0,
            'origin_multipliers': final_best_initial_multipliers.tolist(),
            'iterations_sum': overall_optim_nit,
            'evaluations_sum': overall_optim_nfev,
            'optim_runs': successful_optim_count,
            'logs': overall_logs
        }

    except (ValueError, RuntimeError) as e:
        print(f"ERROR during QWOT Scan + Opt workflow: {e}") # Log
        overall_logs.append(f"Workflow Error: {e}")
        overall_logs.append(traceback.format_exc())
        return {'success': False, 'message': str(e), 'logs': overall_logs}
    except Exception as e:
        print(f"UNEXPECTED ERROR during QWOT Scan + Opt workflow: {e}") # Log
        overall_logs.append(f"Unexpected Workflow Error: {e}")
        overall_logs.append(traceback.format_exc())
        return {'success': False, 'message': f"Unexpected Error: {e}", 'logs': overall_logs}


# --- Updated Main Execution Block ---
if __name__ == '__main__':

    print("\n--- Running Refactored Examples ---")

    try:
        # --- Initialization ---
        material_mgr = MaterialManager(excel_file_path=EXCEL_FILE_PATH)
        calculator = OpticalCalculator(material_mgr)
        print("MaterialManager and OpticalCalculator initialized.")

        # --- Design & Optimization Setup ---
        design_params = {
            'l0': 550.0,
            'initial_layer_number': 7, # Smaller number for faster example
            'nH_material': "Nb2O5-Helios", # Assumes these exist in Excel
            'nL_material': "SiO2-Helios",
            'nSub_material': "Fused Silica",
            'nInc_material': "AIR"
        }
        active_targets = [
            {'min': 450, 'max': 500, 'target_min': 0.0, 'target_max': 0.0},
            {'min': 540, 'max': 560, 'target_min': 1.0, 'target_max': 1.0},
            {'min': 600, 'max': 700, 'target_min': 0.0, 'target_max': 0.0}
        ]
        optimizer_options = {'maxiter': 30, 'maxfun': 50, 'disp': False, 'ftol': 1e-8, 'gtol': 1e-5} # Reduced iterations for example
        l_vec_optim_np = np.linspace(400, 800, 81) # Coarser grid for faster example optim
        l_vec_scan_np = np.linspace(400, 800, 41) # Even coarser for QWOT scan example
        l_vec_plot_np = np.linspace(400, 800, 201) # Finer for plotting

        # --- Example 1: Generate Initial Design ---
        print("\n--- Example 1: Initial Design Calculation ---")
        initial_num_layers = design_params['initial_layer_number']
        initial_l0 = design_params['l0']
        initial_mat_seq = [design_params['nH_material'] if i % 2 == 0 else design_params['nL_material']
                           for i in range(initial_num_layers)]
        initial_multipliers = [1.0] * initial_num_layers
        initial_ep = calculate_initial_ep(material_mgr, initial_multipliers, initial_l0, initial_mat_seq)
        print(f"Initial EP: {initial_ep}")
        initial_design_state = {
            'ep_vector': initial_ep, 'material_sequence': initial_mat_seq,
            'nSub_material': design_params['nSub_material'], 'nInc_material': design_params['nInc_material']
        }
        # Calculate initial spectrum
        initial_spectrum = calculator.calculate_T_spectrum(
            ep_vector=initial_ep, material_sequence=initial_mat_seq,
            nSub_material=design_params['nSub_material'], l_vec=l_vec_plot_np
        )
        print("Initial Spectrum Calculated.") # Add plot command here in real use

        # --- Example 2: Run Local Optimization ---
        print("\n--- Example 2: Local Optimization ---")
        local_opt_results = run_local_optimization_workflow(
            initial_design=initial_design_state, material_manager=material_mgr,
            optical_calculator=calculator, active_targets=active_targets,
            l_vec_optim_np=l_vec_optim_np, optimizer_options=optimizer_options
        )
        if local_opt_results['success']:
            print(f"Local Opt Cost: {local_opt_results['final_cost']:.6e}")
            # Plot final spectrum...
            opt_spectrum = calculator.calculate_T_spectrum(
                ep_vector=local_opt_results['ep_vector'], material_sequence=local_opt_results['material_sequence'],
                nSub_material=local_opt_results['nSub_material'], l_vec=l_vec_plot_np
            )
            print("Optimized Spectrum Calculated.") # Add plot command
        else:
            print(f"Local Optimization Failed: {local_opt_results.get('message')}")

        # --- Example 3: Run Auto Mode (using the locally optimized result as start) ---
        if local_opt_results['success']:
            print("\n--- Example 3: Auto Mode ---")
            auto_mode_results = run_auto_mode(
                initial_design=local_opt_results, # Start from the previous result
                material_manager=material_mgr, optical_calculator=calculator,
                active_targets=active_targets, l_vec_optim_np=l_vec_optim_np,
                optimizer_options=optimizer_options, l0_for_materials=design_params['l0'],
                max_cycles=2, needles_per_cycle=3 # Short example
            )
            if auto_mode_results['success']:
                 print(f"Auto Mode Final Cost: {auto_mode_results['final_cost']:.6e}, Layers: {len(auto_mode_results['ep_vector'])}")
                 # Plot final spectrum...
                 auto_spectrum = calculator.calculate_T_spectrum(
                     ep_vector=auto_mode_results['ep_vector'], material_sequence=auto_mode_results['material_sequence'],
                     nSub_material=auto_mode_results['nSub_material'], l_vec=l_vec_plot_np
                 )
                 print("Auto Mode Spectrum Calculated.") # Add plot command
            else:
                 print(f"Auto Mode Failed: {auto_mode_results.get('message')}")


        # --- Example 4: Run QWOT Scan + Optimize ---
        # print("\n--- Example 4: QWOT Scan + Optimize ---")
        # Note: This can be very slow depending on initial_layer_number
        # design_params_scan = design_params.copy()
        # design_params_scan['initial_layer_number'] = 5 # Use a small N for example
        # qwot_scan_results = run_qwot_scan_and_optimize(
        #     design_params=design_params_scan, material_manager=material_mgr,
        #     optical_calculator=calculator, active_targets=active_targets,
        #     l_vec_eval_full_np=l_vec_optim_np, l_vec_eval_sparse_np=l_vec_scan_np,
        #     optimizer_options=optimizer_options
        # )
        # if qwot_scan_results['success']:
        #      print(f"QWOT Scan Best Cost: {qwot_scan_results['final_cost']:.6e}, Layers: {len(qwot_scan_results['ep_vector'])}, Origin l0: {qwot_scan_results['origin_l0']:.1f}")
        #      # Plot final spectrum...
        # else:
        #      print(f"QWOT Scan Failed: {qwot_scan_results.get('message')}")


    except Exception as e:
        print(f"\n--- An error occurred during the main refactored examples ---")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()        
