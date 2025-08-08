#%% Import packages
#import time
#start = time.time()

# Standard libraries
import inspect
from IPython.display import display, Math
# Numerical computing
import numpy as np
from numba import njit
from scipy.interpolate import interp1d, griddata
from copy import deepcopy
# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import colorsys
# Sequence-Jacobian framework
from sequence_jacobian import grids, interpolate
#from sequence_jacobian.blocks.stage_block import StageBlock
import sequence_jacobian as sj
# Custom utilities
from SSJ_Fun.utils import make_d_grid, LogitChoiceDurables, ExogenousMaker, Continuous1D_Durables, StageBlockDurables
from Fun.my_funs import *

#%% Interactive plot
#%matplotlib qt

#%% Some useful functions for debugging

# #Function to vizualize policy function
# def policy_functions(
#     ss,
#     amax=150,
#     amin=0,
#     d_tilde_list=[0],
#     d_list=[0],
#     ie_list=[3],
#     figsize=0.6,
#     models=['baseline']
# ):
#     a_grid = ss['baseline'].internals['hh']['a_grid']

#     a, da, c, P, V = dict(), dict(), dict(), dict(), dict()

#     for model in models:
#         a[model] = ss[model].internals['hh']['consav']['a']
#         da[model] = a[model] - a_grid
#         c[model] = ss[model].internals['hh']['consav']['c']
#         P[model] = ss[model].internals['hh']['durables']['law_of_motion'].P
#         V[model] = ss[model].internals['hh']['durables']['V']

#     fig, axes = plt.subplots(1, 3, figsize=(12 * figsize, 4 * figsize))
#     ax = axes.flatten()

#     # Define line styles for each iz (cycle if fewer styles than ie_list)
#     linestyles = ['-', '--', '-.', ':']
#     linestyle_map = {
#         iz: linestyles[i % len(linestyles)]
#         for i, iz in enumerate(ie_list)
#     }

#     n = len(d_tilde_list)
#     color_map = {}
#     for i, d_tilde in enumerate(d_tilde_list):
#         hue = i / n  # equally spaced hues on color wheel
#         r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)  # full saturation and brightness
#         color_map[d_tilde] = (r, g, b)  # matplotlib accepts RGB tuples (0-1 range)

#     # Define alpha values for d_list, normalized between 0.3 and 1 for visibility
#     if len(d_list) > 1:
#         alphas = np.linspace(0.3, 1.0, len(d_list))
#     else:
#         alphas = [1.0]
#     alpha_map = {
#         d_val: alpha
#         for d_val, alpha in zip(d_list, alphas)
#     }

#     # Plot
#     for model in models:
#         for d_tilde in d_tilde_list:
#             for d in d_list:
#                 for iz in ie_list:
#                     linestyle = linestyle_map[iz]
#                     color = color_map[d_tilde]
#                     alpha = alpha_map[d]

#                     label = f"{model} ($\\tilde{{d}}$={d_tilde}, d={d}, z={iz})"

#                     # Asset policy function
#                     ax[0].plot(
#                         a_grid[:amax],
#                         a[model][d_tilde, d, iz, :amax],
#                         label=label,
#                         linewidth=2,
#                         color=color,
#                         linestyle=linestyle,
#                         alpha=alpha
#                     )

#                     # Consumption policy function
#                     ax[1].plot(
#                         a_grid[:amax],
#                         c[model][d_tilde, d, iz, :amax],
#                         label=label,
#                         linewidth=2,
#                         color=color,
#                         linestyle=linestyle,
#                         alpha=alpha
#                     )

#                     # Discrete choice probability
#                     ax[2].plot(
#                         a_grid[amin:amax],
#                         P[model][d_tilde, d, iz, amin:amax],
#                         label=label,
#                         linewidth=2,
#                         color=color,
#                         linestyle=linestyle,
#                         alpha=alpha
#                     )

#     # Reference lines for assets plot
#     ax[0].plot(a_grid[:amax], a_grid[:amax], color='gray', linestyle=':')
#     ax[0].axhline(0, color='gray', linestyle=':')

#     # Titles
#     ax[0].set_title(r'Assets ($a^*(\tilde{d},\, d,\, z,\, a^{-})$)')
#     ax[1].set_title(r'Consumption ($c^*(\tilde{d},\, d,\, z,\, a^{-})$)')
#     ax[2].set_title(r'Discrete choice ($Pr(\tilde{d}^*(d,\, z,\, a^{-})=1)$)')

#     for axis in ax:
#         axis.set_xlabel('assets')
#         axis.legend(frameon=False)

#     plt.tight_layout()
#     plt.show()

# def plot_heatmap(Pi, title="Matrix Heatmap", fmt=".2f"):
#     """
#     Plots a heatmap of a 2D matrix using matplotlib.

#     - Uses red→yellow→green if values are in [0,1].
#     - Otherwise uses 'viridis' colormap.

#     Parameters:
#     - Pi: 2D array-like
#     - title: plot title
#     - fmt: string format for values in the cells
#     """
#     Pi = np.array(Pi)
#     n_rows, n_cols = Pi.shape

#     # Check range
#     min_val, max_val = np.min(Pi), np.max(Pi)
#     values_in_unit_interval = 0 <= min_val and max_val <= 1

#     # Choose colormap
#     #if values_in_unit_interval:
#      #   cmap = LinearSegmentedColormap.from_list("red_yellow_green", ["red", "yellow", "green"])
#     #    vmin, vmax = 0, 1
#     #else:
#     cmap = "viridis"
#     vmin, vmax = None, None  # auto-scale

#     fig, ax = plt.subplots(figsize=(1 + n_cols, 1 + n_rows))
#     cax = ax.imshow(Pi, cmap=cmap, vmin=vmin, vmax=vmax)

#     # Add colorbar
#     fig.colorbar(cax)

#     # Axis ticks and labels
#     ax.set_xticks(np.arange(n_cols))
#     ax.set_yticks(np.arange(n_rows))
#     ax.set_xticklabels([f"{j}" for j in range(n_cols)])
#     ax.set_yticklabels([f"{i}" for i in range(n_rows)])
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#     # Annotate values
#     for i in range(n_rows):
#         for j in range(n_cols):
#             val = Pi[i, j]
#             ax.text(j, i, format(val, fmt),
#                     ha="center", va="center",
#                     color="black" if not values_in_unit_interval or val > 0.7 or val < 0.3 else "white")

#     ax.set_title(title)
#     ax.set_xlabel("Column Index")
#     ax.set_ylabel("Row Index")
#     plt.tight_layout()
#     plt.show()

# def pplot(array, discrete_choice=1, prod_type=0, a_min=0, a_max=199):
#     # Try to infer the variable name passed to `array`
#     try:
#         frame = inspect.currentframe().f_back
#         array_name = next((name for name, val in frame.f_locals.items() if val is array), "array")
#     finally:
#         del frame  # Clean up to avoid reference cycles

#     # Extract the data slice
#     y_vals = array[discrete_choice, prod_type, a_min:a_max]

#     # Plot
#     plt.plot(range(a_min, a_max), y_vals)
#     plt.title(f"{array_name}: discr_choice={discrete_choice}, prod_type={prod_type}, a ∈ [{a_min}, {a_max})")
#     plt.xlabel("Asset index")
#     plt.ylabel("Value")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# def ppplot(array, discrete_choice=1, prod_type=0, a_min=0, a_max=199):
#     # Try to infer the variable name passed to `array`
#     try:
#         frame = inspect.currentframe().f_back
#         array_name = next((name for name, val in frame.f_locals.items() if val is array), "array")
#     finally:
#         del frame  # Clean up to avoid reference cycles

#     # Extract the data slice
#     y_vals = array[discrete_choice*7 +  prod_type, a_min:a_max]

#     # Plot
#     plt.plot(range(a_min, a_max), y_vals)
#     plt.title(f"{array_name}: discr_choice={discrete_choice}, prod_type={prod_type}, a ∈ [{a_min}, {a_max})")
#     plt.xlabel("Asset index")
#     plt.ylabel("Value")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

def make_strictly_decreasing(uc):
    uc_fixed = uc.copy()
    shape = uc.shape
    ndim = uc.ndim

    # Iterate over all indices except the last one
    it = np.nditer(uc[..., 0], flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index  # Tuple of all dimensions except the last
        row = uc[idx]  # This is a 1D array (the last axis)

        # Fix infinite values at the start
        if np.isinf(row[0]):
            first_finite = np.argmax(~np.isinf(row))
            if first_finite > 0:
                decrement = 1.0
                for k in range(first_finite - 1, -1, -1):
                    row[k] = row[k + 1] + decrement

        # Make strictly decreasing
        for k in range(1, row.shape[0]):
            if row[k] >= row[k - 1]:
                row[k] = row[k - 1] - 1e-8

        # Assign back to uc_fixed
        uc_fixed[idx] = row

        it.iternext()

    return uc_fixed

# def make_strictly_increasing(uc):
#     uc_fixed = uc.copy()
#     shape = uc.shape

#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             row = uc[i, j, :]

#             # Replace -inf at the beginning with decreasing values
#             if np.isneginf(row[0]):
#                 first_finite = np.argmax(~np.isneginf(row))
#                 if first_finite == 0:
#                     continue  # No -inf at start
#                 increment = 1.0  # adjust based on scale if needed
#                 for k in range(first_finite - 1, -1, -1):
#                     row[k] = row[k + 1] - increment

#             # Make strictly increasing
#             for k in range(1, shape[2]):
#                 if row[k] <= row[k - 1]:
#                     row[k] = row[k - 1] + 1e-8  # enforce strict increase

#             uc_fixed[i, j, :] = row

#     return uc_fixed

# def analyze_steady_state(param_name, param_values, cali, hh, n_d):
#     """
#     Vary a calibration parameter and plot steady-state outcomes for DD_i and DD_TILDE_i, A, and C.

#     Args:
#         param_name (str): Name of the parameter in cali['baseline'] to vary.
#         param_values (array-like): Grid of values to assign to the parameter.
#         cali (dict): Dictionary containing the baseline calibration.
#         hh (module/object): Object with a method `steady_state(cali_dict)`.
#         n_d (int): Number of durable states (e.g. 1 + n_b + n_g)

#     Returns:
#         dict: Dictionary with raw and interpolated results for DD_i, A, and C.
#     """
#     # Store DD and DD_TILDE as list of lists, one per vintage i
#     dd_vals = [[] for _ in range(n_d)]
#     dd_tilde_vals = [[] for _ in range(n_d)]
#     a_vals = []
#     c_vals = []
#     success_flags = []

#     for val in param_values:
#         cali_try = cali['baseline'].copy()
#         cali_try[param_name] = val
#         try:
#             ss_try = hh.steady_state(cali_try)

#             for i in range(n_d):
#                 dd_vals[i].append(ss_try[f'DD_{i}'])
#                 dd_tilde_vals[i].append(ss_try[f'DD_TILDE_{i}'])

#             a_vals.append(ss_try['A'])
#             c_vals.append(ss_try['C'])
#             success_flags.append(True)
#             print(f"SS found for {param_name} = {val:.3f}!")
#         except Exception as e:
#             print(f"Failed for {param_name} = {val:.3f}: {e}")
#             for i in range(n_d):
#                 dd_vals[i].append(np.nan)
#                 dd_tilde_vals[i].append(np.nan)
#             a_vals.append(np.nan)
#             c_vals.append(np.nan)
#             success_flags.append(False)

#     # Convert to arrays
#     param_values = np.array(param_values)
#     dd_vals = [np.array(v) for v in dd_vals]
#     dd_tilde_vals = [np.array(v) for v in dd_tilde_vals]
#     a_vals = np.array(a_vals)
#     c_vals = np.array(c_vals)

#     # Plotting DD_i and DD_TILDE_i
#     fig, axs = plt.subplots(1, n_d, figsize=(3 * n_d, 2), sharex=True)
#     if n_d == 1:
#         axs = [axs]  # ensure list-like even for 1 subplot

#     for i in range(n_d):
#         mask_dd = ~np.isnan(dd_vals[i])
#         mask_dd_tilde = ~np.isnan(dd_tilde_vals[i])

#         # Interpolate missing values
#         interp_dd = interp1d(param_values[mask_dd], dd_vals[i][mask_dd], kind='linear', fill_value="extrapolate")
#         interp_dd_tilde = interp1d(param_values[mask_dd_tilde], dd_tilde_vals[i][mask_dd_tilde], kind='linear', fill_value="extrapolate")
#         dd_vals_interp = dd_vals[i].copy()
#         dd_tilde_vals_interp = dd_tilde_vals[i].copy()
#         dd_vals_interp[~mask_dd] = interp_dd(param_values[~mask_dd])
#         dd_tilde_vals_interp[~mask_dd_tilde] = interp_dd_tilde(param_values[~mask_dd_tilde])

#         # Plot both
#         axs[i].plot(param_values, dd_vals_interp, '--', color='blue', label=f'DD_{i}')
#         axs[i].plot(param_values, dd_tilde_vals_interp, '--', color='green', label=f'DD_TILDE_{i}')
#         axs[i].plot(param_values[mask_dd], dd_vals[i][mask_dd], 'o', color='blue')
#         axs[i].plot(param_values[mask_dd_tilde], dd_tilde_vals[i][mask_dd_tilde], 's', color='green')
#         axs[i].set_title(f'Durable state {i}')
#         axs[i].set_xlabel(param_name)
#         axs[i].set_ylim(0, 1)
#         axs[i].grid(True)
#         axs[i].legend()

#     axs[0].set_ylabel("Durable Ownership Share")

#     # A and C
#     fig_ac, axs_ac = plt.subplots(1, 2, figsize=(8, 3), sharex=True)

#     # A
#     mask_a = ~np.isnan(a_vals)
#     interp_a = interp1d(param_values[mask_a], a_vals[mask_a], kind='linear', fill_value="extrapolate")
#     a_vals_interp = a_vals.copy()
#     a_vals_interp[~mask_a] = interp_a(param_values[~mask_a])
#     axs_ac[0].plot(param_values, a_vals_interp, '--', color='gray')
#     axs_ac[0].plot(param_values[mask_a], a_vals[mask_a], 'o', color='blue')
#     axs_ac[0].plot(param_values[~mask_a], a_vals_interp[~mask_a], 'x', color='red')
#     axs_ac[0].set_ylabel("A (Assets)")
#     axs_ac[0].set_title("Steady-state A")

#     # C
#     mask_c = ~np.isnan(c_vals)
#     interp_c = interp1d(param_values[mask_c], c_vals[mask_c], kind='linear', fill_value="extrapolate")
#     c_vals_interp = c_vals.copy()
#     c_vals_interp[~mask_c] = interp_c(param_values[~mask_c])
#     axs_ac[1].plot(param_values, c_vals_interp, '--', color='gray')
#     axs_ac[1].plot(param_values[mask_c], c_vals[mask_c], 'o', color='blue')
#     axs_ac[1].plot(param_values[~mask_c], c_vals_interp[~mask_c], 'x', color='red')
#     axs_ac[1].set_ylabel("C (Consumption)")
#     axs_ac[1].set_title("Steady-state C")

#     for ax in axs_ac:
#         ax.set_xlabel(param_name)
#         ax.grid(True)

#     plt.tight_layout()
#     plt.show()

#     return {
#         'param_values': param_values,
#         'dd_vals': dd_vals,
#         'dd_tilde_vals': dd_tilde_vals,
#         'a_vals': a_vals,
#         'c_vals': c_vals,
#         'success_flags': success_flags,
#     }

# def analyze_steady_state_3d(param1, values1, param2, values2, cali, hh, n_d, results=None):
#     """
#     Vary two calibration parameters and plot steady-state outcomes:
#     - One big figure for DD_k and DD_TILDE_k for each durable choice (vintage).
#     - One second figure for A and C.

#     Args:
#         param1 (str): First parameter name.
#         values1 (array-like): Grid for the first parameter.
#         param2 (str): Second parameter name.
#         values2 (array-like): Grid for the second parameter.
#         cali (dict): Calibration dictionary with 'baseline'.
#         hh (object): Must have method `steady_state(cali_dict)`.
#         n_d (int): Number of durable choices (vintages).
#         results (dict, optional): Previous results object to reuse.

#     Returns:
#         dict: Grid and results for DD_k, A, and C.
#     """

#     values1 = np.array(values1)
#     values2 = np.array(values2)
#     X, Y = np.meshgrid(values1, values2, indexing='ij')

#     if results is None:
#         DD_all = [np.full_like(X, np.nan, dtype=float) for _ in range(n_d)]
#         DD_tilde_all = [np.full_like(X, np.nan, dtype=float) for _ in range(n_d)]
#         A = np.full_like(X, np.nan, dtype=float)
#         C = np.full_like(X, np.nan, dtype=float)
#         success = np.full_like(X, False, dtype=bool)

#         for i, v1 in enumerate(values1):
#             for j, v2 in enumerate(values2):
#                 cali_try = cali['baseline'].copy()
#                 cali_try[param1] = v1
#                 cali_try[param2] = v2
#                 try:
#                     ss_try = hh.steady_state(cali_try)
#                     for k in range(n_d):
#                         DD_all[k][i, j] = ss_try[f'DD_{k}']
#                         DD_tilde_all[k][i, j] = ss_try[f'DD_TILDE_{k}']
#                     A[i, j] = ss_try['A']
#                     C[i, j] = ss_try['C']
#                     success[i, j] = True
#                     print(f"Success: {param1}={v1:.2f}, {param2}={v2:.2f}")
#                 except Exception as e:
#                     print(f"Fail: {param1}={v1:.2f}, {param2}={v2:.2f} | {e}")
#     else:
#         DD_all = results['DD_all']
#         DD_tilde_all = results['DD_tilde_all']
#         A = results['A']
#         C = results['C']
#         success = results['success']

#     # === Interpolation ===
#     def interpolate_missing(Z):
#         points = np.column_stack((X[~np.isnan(Z)], Y[~np.isnan(Z)]))
#         values = Z[~np.isnan(Z)]
#         return griddata(points, values, (X, Y), method='linear')

#     DD_interp_all = [interpolate_missing(DD) for DD in DD_all]
#     DD_tilde_interp_all = [interpolate_missing(DD_tilde) for DD_tilde in DD_tilde_all]
#     A_interp = interpolate_missing(A)
#     C_interp = interpolate_missing(C)

#     # === First figure: DD and DD_TILDE by vintage ===
#     ncols = 4
#     nrows = int(np.ceil(n_d / ncols))
#     #fig1 = plt.figure(figsize=(6 * ncols, 5 * nrows))
#     fig1 = plt.figure(figsize=(18,9))


#     for k in range(n_d):
#         ax = fig1.add_subplot(nrows, ncols, k + 1, projection='3d')

#         ax.plot_surface(X, Y, DD_interp_all[k], cmap='cividis', alpha=0.60, edgecolor='none')
#         ax.plot_surface(X, Y, DD_tilde_interp_all[k], cmap='viridis', alpha=0.60, edgecolor='none')
#         ax.scatter(X[success], Y[success], DD_all[k][success], color='blue', s=10, label='DD')
#         ax.scatter(X[success], Y[success], DD_tilde_all[k][success], color='green', s=10, label='DD_TILDE')

#         ax.set_xlabel(param1)
#         ax.set_ylabel(param2)
#         ax.set_zlabel(f'DD / DD_TILDE')
#         ax.set_title(f'Durable state {k}')
#         if k == 0:
#             ax.legend()

#     plt.tight_layout()
#     plt.show()

#     # === Second figure: A and C ===
#     #fig2 = plt.figure(figsize=(12, 6))
#     fig2 = plt.figure(figsize=(8, 4))

#     ax1 = fig2.add_subplot(1, 2, 1, projection='3d')
#     ax1.plot_surface(X, Y, A_interp, cmap='Oranges', alpha=0.7, edgecolor='none')
#     ax1.scatter(X[success], Y[success], A[success], color='darkorange', s=10)
#     ax1.set_xlabel(param1)
#     ax1.set_ylabel(param2)
#     ax1.set_zlabel('A')
#     ax1.set_title('Assets (A)')

#     ax2 = fig2.add_subplot(1, 2, 2, projection='3d')
#     ax2.plot_surface(X, Y, C_interp, cmap='Purples', alpha=0.7, edgecolor='none')
#     ax2.scatter(X[success], Y[success], C[success], color='indigo', s=10)
#     ax2.set_xlabel(param1)
#     ax2.set_ylabel(param2)
#     ax2.set_zlabel('C')
#     ax2.set_title('Consumption (C)')

#     plt.tight_layout()
#     plt.show()

#     return {
#         'X': X, 'Y': Y,
#         'DD_all': DD_all,
#         'DD_tilde_all': DD_tilde_all,
#         'DD_interp_all': DD_interp_all,
#         'DD_tilde_interp_all': DD_tilde_interp_all,
#         'A': A, 'C': C,
#         'A_interp': A_interp,
#         'C_interp': C_interp,
#         'success': success
#     }

# # Define the plotting function
# def show_irfs(irfs_list, variables, labels=[" "], ylabel=r"Percentage points (dev. from ss)", T_plot=50, figsize=(18, 6)):
#     if len(irfs_list) != len(labels):
#         labels = [" "] * len(irfs_list)

#     n_var = len(variables)
#     fig, ax = plt.subplots(1, n_var, figsize=figsize, sharex=True)
#     if n_var == 1:
#         ax = [ax]  # Ensure ax is iterable

#     for i in range(n_var):
#         var = variables[i]

#         for j, irf in enumerate(irfs_list):
#             if var in irf:
#                 data = 100 * np.array(irf[var][:T_plot])
#             else:
#                 data = np.zeros(T_plot)
#             ax[i].plot(data, label=labels[j])

#         ax[i].set_title(var)
#         ax[i].set_xlabel(r"$t$")
#         if i == 0:
#             ax[i].set_ylabel(ylabel)
#         ax[i].legend()

#     plt.tight_layout()
#     plt.show()

# def plot_linear_irfs(shocks_list, unknowns_td, targets_td, ha, ss, outputs,
#               rho=None, e=None, T=300, figsize=(18, 3), ylabel=r"Percentage points (dev. from ss)", labels=None):
#     # Default values if not provided
#     if rho is None:
#         rho = {shock: 0.8 for shock in shocks_list}
#     if e is None:
#         e = {shock: 0.01 for shock in shocks_list}
#     # Build shocks dictionary with time series
#     shocks = {
#         shock: e[shock] * rho[shock] ** np.arange(T)
#         for shock in shocks_list
#     }
#     # Solve the system
#     irfs = ha.solve_impulse_linear(ss, unknowns_td, targets_td, shocks)
#     # Default label
#     if labels is None:
#         labels = [" + ".join(shocks_list)]
#     # Plot
#     show_irfs([irfs], outputs, labels=labels, ylabel=ylabel, T_plot=T, figsize=figsize)

# def evaluate_param_changes(param_name, values_list):
#     # Get baseline calibration and steady state
#     baseline_calib = deepcopy(cali['baseline'])
#     baseline_ss = ha.steady_state(baseline_calib)

#     # Variables to report in SS output
#     ss_vars = ['goods_mkt', 'asset_mkt', 'Tax', 'r', 'beta', 'G', 'B', 'N', 'Y', 'Z']

#     # Store results
#     calibration_results = []
#     ss_results = []

#     # Baseline row
#     calibration_results.append(('Baseline', baseline_calib.get(param_name, 'N/A')))
#     ss_results.append(('Baseline', [baseline_ss[v] if v in baseline_ss else 'N/A' for v in ss_vars]))

#     # Loop over alternative values
#     for val in values_list:
#         # Create new calibration
#         modified_calib = deepcopy(baseline_calib)
#         modified_calib[param_name] = val

#         # Compute steady state
#         ss = ha.steady_state(modified_calib)

#         # Store results
#         case_name = f"{param_name} = {val}"
#         calibration_results.append((case_name, val))
#         ss_results.append((case_name, [ss[v] if v in ss else 'N/A' for v in ss_vars]))

#     # Print Calibration Table
#     print(f"\n🔧 Parameter sweep for '{param_name}'")
#     print("\n📌 Calibration Values:")
#     print(f"{'Case':<20} | {param_name}")
#     print("-" * 35)
#     for case, val in calibration_results:
#         print(f"{case:<20} | {val:.5f}" if isinstance(val, (float, int)) else f"{case:<20} | {val}")

#     # Print SS Table
#     print("\n📈 Steady-State Outcomes:")
#     header = f"{'Case':<20} | " + " | ".join([f"{v:<10}" for v in ss_vars])
#     print(header)
#     print("-" * len(header))
#     for case, row in ss_results:
#         row_str = " | ".join(
#             [f"{x:>10.5f}" if isinstance(x, (float, int)) else f"{x:>10}" for x in row]
#         )
#         print(f"{case:<20} | {row_str}")


#%% Stage 1 - Productivity shock (Expected value function given initial state of individual prod. level e_)

#Initialize Stage 1a
prod_stage = ExogenousMaker(markov_name='e_markov', index=1, name='prod')


#Initialize Stage 1b
depreciation_stage = ExogenousMaker(markov_name='d_markov', index=0, name='durable')


#%% Stage 2 - Discrete choice (Labor participation)

#Initialize Stage 2
#`value`: name of value function, SSJ has to know which object to apply the logsum formula to
#`backward`: names of other variables that have to be propagated backward, typically this is partial value function needed for EGM in continuous choice stage
#`index`: axis of correspoinding state
#`name`: name of stage
#`taste_shock_scale`: name of $\sigma_\varepsilon$ parameter, needed for all formulas
#`f`: (optional) function that implements additive utility cost on expanded state $(n| n_-, z, a_-)$. This is useful to implement costs that depend on origin as well as destination $(n|n_-)$. Setting some costs to infinity implements constraints on discrete choice (more on this below).

durables_stage = LogitChoiceDurables(value='V', backward='Va', index=0, name='durables',
                           taste_shock_scale='taste_shock')

#%% Stage 3 - Consumption-Savings Continuous Choice
#Discrete Choice - Endogenous Grid point Method. Performs single step of backward iteration.
def dcegm(V, Va, a_grid, disp_inc, adj_matrix, z_grid, r, T, beta, eis, shifters):
    """DC-EGM algorithm"""
    n_d = adj_matrix.shape[0] #Number of discrete choices
    # use all FOCs on endogenous grid
    W = beta * V                                                  # end-of-stage vfun
    W = np.stack([W] * n_d, axis=0)                               # Add first dimension to match the dimensions
    uc_endo = beta * Va                                           # envelope condition
    c_endo = uc_endo** (-eis)                                     # Euler equation
    a_endo = (c_endo[np.newaxis, ...]
              + a_grid[np.newaxis, np.newaxis, np.newaxis, ...]
              + adj_matrix[..., np.newaxis,np.newaxis]
              - z_grid[np.newaxis, np.newaxis, ..., np.newaxis]
              - T[np.newaxis, np.newaxis, ..., np.newaxis]
              ) / (1 + r)     # budget constraint

    #d_bool = np.zeros_like(a_endo)
    #d_bool[1,:,:,:] = 1 #Decide to have a car (either keeping the existing car or buying)

    # Mark the presence of each durable
    d_type = np.zeros_like(a_endo)
    for d in range(0, n_d):
        d_type[d, :, :, :] = d

    # interpolate with upper envelope, enforce borrowing limit
    V, c, a = upperenv(W, a_endo, disp_inc, a_grid, d_type, eis, shifters)

    # update Va on exogenous grid
    uc = c ** (-1 / eis)                                          # Euler equation
    uc = make_strictly_decreasing(uc)                             # Correct for the infinite values.
    Va = (1 + r) * uc                                             # envelope condition

    return V, Va, a, c



#Simple wrapper to make it independent of the size of the state space. Temporarily collapse states associated with all other stages into a single axis.
def upperenv(W, a_endo, disp_inc, a_grid, d_type, *args):
    # collapse (d_tilde, d, z, a) into (b, a)
    shape = W.shape
    W = W.reshape((-1, shape[-1]))
    a_endo = a_endo.reshape((-1, shape[-1]))
    d_type = d_type.reshape((-1, shape[-1]))
    disp_inc = disp_inc.reshape((-1, shape[-1]))
    V, c, a = upperenv_vec(W, a_endo, disp_inc, a_grid, d_type, *args)

    # report on (d_tilde, d, z, a)
    return V.reshape(shape), c.reshape(shape), a.reshape(shape)


#Core upper envelope step:
# Consider every segment of the endogenous grid $(a_{j}^{endo}, a_{j+1}^{endo})$ and find all the exogenous gridpoints $a^{grid}_i$ that fall into that segment.
# Interpolate there to get a candidate solution $a_i$.
# Since the endogenous grid is non-monotonic, the same point $a^{grid}_i$ may be bracketed by another segment $(a_{\tilde j}^{endo}, a_{\tilde j+1}^{endo}).$
# When this happens, we keep the solution that gives higher value.
@njit
def upperenv_vec(W, a_endo, disp_inc, a_grid, d_type, *args):
    """Interpolate value function and consumption to exogenous grid."""
    n_b, n_a = W.shape
    a = np.zeros_like(W)
    c = np.zeros_like(W)
    V = -np.inf * np.ones_like(W)

    # loop over other states, collapsed into single axis
    for ib in range(n_b):
        #d = min(ib * 2 // n_b, 2 - 1)
        d = int(d_type[ib,0])
        # loop over segments of endogenous asset grid from EGM (not necessarily increasing)
        for ja in range(n_a - 1):
            a_low, a_high = a_endo[ib, ja], a_endo[ib, ja + 1]
            W_low, W_high = W[ib, ja], W[ib, ja + 1]
            ap_low, ap_high = a_grid[ja], a_grid[ja + 1]

           # loop over exogenous asset grid (increasing)
            for ia in range(n_a):
                acur = a_grid[ia]
                coh_cur = disp_inc[ib, ia]

                interp = (a_low <= acur <= a_high)
                extrap = (ja == n_a - 2) and (acur > a_endo[ib, n_a - 1])

                # exploit that a_grid is increasing
                if (a_high < acur < a_endo[ib, n_a - 1]):
                    break

                if interp or extrap:
                    W0 = interpolate.interpolate_point(acur, a_low, a_high, W_low, W_high)
                    a0 = interpolate.interpolate_point(acur, a_low, a_high, ap_low, ap_high)
                    c0 = coh_cur - a0
                    V0 = util(c0, d, *args) + W0

                    # upper envelope, update if new is better
                    if V0 > V[ib, ia]:
                        a[ib, ia] = a0
                        c[ib, ia] = c0
                        V[ib, ia] = V0

        # Enforce borrowing constraint
        ia = 0
        while ia < n_a and a_grid[ia] <= a_endo[ib, 0]:
            a[ib, ia] = a_grid[0]
            c[ib, ia] = max(0.0001,disp_inc[ib, ia]) # Correct for negative values. Replace by small consumption (Unlikely to choose this consumption)
            V[ib, ia] = util(c[ib, ia], d, *args) + W[ib, 0]
            ia += 1

    return V, c, a

# %% Utilitz function
@njit
def util(c, d, eis, shifters):
    """
    General utility function for arbitrary discrete states.

    Parameters:
    - c: consumption (scalar)
    - d: discrete state index (integer)
    - eis: elasticity of intertemporal substitution
    - shifters: 1D array of utility shifters per d-state
    """
    # Basic bounds check (without exception)
    #if d < 0 or d >= shifters.shape[0]:
    #    return -1e10  # or some other penalizing value instead of raising an error

    if eis == 1.0:
        u = np.log(c) + shifters[d]
    else:
        u = c ** (1 - 1 / eis) / (1 - 1 / eis) + shifters[d]

    return u


#Report the aggregate demand for d
def D_demand(c):
    shape = c.shape
    D = shape[0]
    dd_tilde_list = []
    dd_list = []
    for d in range(D):
        dd_tilde = np.zeros(shape, dtype=c.dtype)
        dd = np.zeros(shape, dtype=c.dtype)
        dd_tilde[d, ...] = 1
        dd[:, d, ...] = 1
        dd_tilde_list.append(dd_tilde)
        dd_list.append(dd)
    # Dynamically assign to variables in local scope
    out_vars = []
    for i in range(D):
        globals()[f'dd_tilde_{i}'] = dd_tilde_list[i]
        globals()[f'dd_{i}'] = dd_list[i]
        out_vars.append(dd_tilde_list[i])
    for i in range(D):
        out_vars.append(dd_list[i])

    d_t_N, d_N, d_t_BN, d_BN, d_t_BO, d_BO, d_t_GN, d_GN, d_t_GO, d_GO = (dd_tilde_0, dd_0, dd_tilde_1, dd_1, dd_tilde_2, dd_2, dd_tilde_3, dd_3, dd_tilde_4, dd_4)
    #d_t_N, d_N, d_t_BN, d_BN, d_t_BM, d_BM, d_t_BO, d_BO, d_t_GN, d_GN, d_t_GM, d_GM, d_t_GO, d_GO = (dd_tilde_0, dd_0, dd_tilde_1, dd_1, dd_tilde_2, dd_2, dd_tilde_3, dd_3, dd_tilde_4, dd_4, dd_tilde_5, dd_5, dd_tilde_6, dd_6)

    d_B = d_BN + d_BO
    d_G = d_GN + d_GO
#return d_t_N, d_N, d_t_BN, d_BN, d_t_BM, d_BM, d_t_BO, d_BO, d_t_GN, d_GN, d_t_GM, d_GM, d_t_GO, d_GO
    return d_t_N, d_N, d_t_BN, d_BN, d_t_BO, d_BO, d_t_GN, d_GN, d_t_GO, d_GO, d_B, d_G
#return dd_tilde_0, dd_0, dd_tilde_1, dd_1, dd_tilde_2, dd_2, dd_tilde_3, dd_3, dd_tilde_4, dd_4



def compute_distr(c):
    distr = np.ones_like(c)
    return distr

#Initialize Stage 3
consav_stage = Continuous1D_Durables(backward=['V', 'Va'], policy='a', f=dcegm,
                            name='consav', hetoutputs=[D_demand, compute_distr])

# %% Other basic necessary functions
# hh_init: function that constructs the initial guess for backward variables
def hh_init(disp_inc, a_grid, eis, shifters):
    V = util(disp_inc-np.min(disp_inc)+1, 0, eis, shifters)         #Avoid strange behaviour due to negative values. Not too important as only for first guess.
    V = (V[0,:,:,:] + V[1,:,:,:])/2                                 #Get rid of first dimension
    Va = np.empty_like(V)
    Va[..., 1:-1] = (V[..., 2:] - V[..., :-2]) / (a_grid[2:] - a_grid[:-2])
    Va[..., 0] = (V[..., 1] - V[..., 0]) / (a_grid[1] - a_grid[0])
    Va[..., -1] = (V[..., -1] - V[..., -2]) / (a_grid[-1] - a_grid[-2])
    return V, Va

#construct Markov process for productivity, for depreciation of durables and the assets grid
def make_grids(rho_e, sd_e, n_e, min_a, max_a, n_a, n_b, n_g, lifetime_b, lifetime_g):
    e_grid, e_dist, e_markov = grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    a_grid = grids.agrid(max_a, n_a, min_a)
    d_grid, d_markov, d_grid_name = make_d_grid(n_b, n_g, lifetime_b, lifetime_g)
    return e_grid, e_dist, e_markov, a_grid, d_grid, d_markov, d_grid_name

#def income_grid(e_grid, tau, w, N):
def income_grid(e_grid, Z):
    #z_grid = (1 - tau) * w * N * e_grid
    z_grid = Z * e_grid
    return z_grid

def transfers(e_dist, Div, Tax, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter
    #tax_rule, div_rule = e_grid, e_grid
    tax_rule, div_rule = np.ones_like(e_grid), np.ones_like(e_grid)               #Lump-Sum
    div = Div / np.sum(e_dist * div_rule) * div_rule
    tax = Tax / np.sum(e_dist * tax_rule) * tax_rule
    T = div - tax
    return T

#Construct the adjustment costs matrix between durables
def adj_costs(p_d, chi):
    adj_matrix = p_d[:, None] - (1 - chi) * p_d
    np.fill_diagonal(adj_matrix, 0)                            # set diagonal to 0 (no cost if no switching)
    return adj_matrix

#Define the disposable income
def disp_inc_f(a_grid, z_grid, T, r, adj_matrix):                 #Disposable income for consumption and assets after buying the durable good
    # Disposable income is:
    # asset income          + labor income         - durable adjustment cost
    disp_inc = (
        (1 + r) * a_grid[np.newaxis, np.newaxis, np.newaxis, :]           # asset income
        + z_grid[np.newaxis, np.newaxis, ..., np.newaxis]                 # labor income
        + T[np.newaxis, np.newaxis, ..., np.newaxis]                      # Transfers
        - adj_matrix[..., np.newaxis, np.newaxis]                         # adjustment costs
    )                                                         # on (nd, nd, e, a)
    return disp_inc

#Construct the utility shifter for durables
#def make_shifters(n_b, n_g, gamma_b, gamma_g):
#    shifters = np.array([0.0] + [gamma_b] * n_b + [gamma_g] * n_g)
#    return shifters

def make_shifters(n_b, n_g, gamma_b, gamma_g, dep_util_frac_b, dep_util_frac_g):
    dep_rate_b = 1 - (dep_util_frac_b) ** (1 / (n_b - 1))        # Depreciation rate for good b
    vintages_b = np.arange(n_b)
    gammas_b_vector = gamma_b * (1 - dep_rate_b) ** vintages_b
    dep_rate_g = 1 - (dep_util_frac_g) ** (1 / (n_g - 1))        # Depreciation rate for good g
    vintages_g = np.arange(n_g)
    gammas_g_vector = gamma_g * (1 - dep_rate_g) ** vintages_g
    # Combine
    shifters = np.array([0.0] + list(gammas_b_vector) + list(gammas_g_vector))
    return shifters

def make_prices_durables(p_b, dep_frac_b, n_b, p_g, dep_frac_g, n_g):
    dep_rate_b = 1 - (dep_frac_b) ** (1 / (n_b - 1))        # Depreciation rate for good b
    vintages_b = np.arange(n_b)
    p_b_vector = p_b * (1 - dep_rate_b) ** vintages_b
    dep_rate_g = 1 - (dep_frac_g) ** (1 / (n_g - 1))        # Depreciation rate for good g
    vintages_g = np.arange(n_g)
    p_g_vector = p_g * (1 - dep_rate_g) ** vintages_g
    # Combine
    p_d = np.array([0.0] + list(p_b_vector) + list(p_g_vector))
    return p_d


#%% Assemble the HH block (staged block)
hh = StageBlockDurables([depreciation_stage, prod_stage, durables_stage, consav_stage], name='hh',
                backward_init=hh_init,
                hetinputs=[make_grids, income_grid, transfers, adj_costs, disp_inc_f, make_shifters, make_prices_durables])

print(hh)
print(f"Inputs: {hh.inputs}")
print(f"Outputs: {hh.outputs}")

#%%
# -------------------------------
# --Solving the baseline hh block --
# -------------------------------

#%% Calibration
# === Calibration dictionary ===
cali = {}
cali["baseline"] = {
    # Preferences and taste shocks
    "taste_shock": 1e-1,       # Idiosyncratic taste shock
    "vphi": 0.0,               # Value function penalty parameter
    "beta": 0.97,              # Discount factor
    "eis": 0.5,                # Elasticity of intertemporal substitution
    "r": 0.02 / 4,             # Interest rate (quarterly)
    # Productivity process
    "rho_e": 0.95,             # Persistence of productivity shocks
    "sd_e": 0.5,               # Std. deviation of productivity shocks
    "n_e": 5,                  # Number of productivity grid points
    # Asset grid
    "min_a": 0.0,              # Minimum asset level
    "max_a": 100,              # Maximum asset level
    "n_a": 20,                 # Number of asset grid points
    # Labor market
    #"w": 1.0,                  # Wage level
    "N": 1.0,                  # Labor supply
    "tau":0,                   # Labor income tax
    # Durable goods
    "p_b": 0.80,               # Initial price of brown durable
    "dep_frac_b": 0.25,        # Depreciation green (Fraction of oldest vintage relative to newest)
    "n_b": 2,                  # Number of brown vintages
    "p_g": 0.90,               # Initial price of green durable
    "dep_frac_g": 0.25,        # Depreciation green (Fraction of oldest vintage relative to newest)
    "n_g": 2,                  # Number of green vintages
    #"n_d": 1 + n_b + n_g,      # Total durable states
    "chi": 0.5,                # Resale loss (fraction)
    "gamma_b": 1.0,            # Utility from brown durable
    "dep_util_frac_b": 1,    # Depreciation utility brown (Fraction of oldest vintage relative to newest)
    "gamma_g": 1.2,            # Utility from green durable
    "dep_util_frac_g": 1,    # Depreciation utility green (Fraction of oldest vintage relative to newest)
    "lifetime_b": 60,          # Average lifetime of brown durables (quarters)
    "lifetime_g": 60,          # Average lifetime of green durables (quarters)
    # Firms
    "alpha": 1,                # Share of labor in prod. function
    "Div": 0,                  # Dividends from firms
    "Tax": 0.5,                # Total tax
    #Government
    #"Y" : 1,                   # Output
    "B" : 4,                   # Stock of debt
    "G" : 0.3,                 # Government spendings
}

#TO DELETE IN FINAL VERSION. ONLY FOR DEBUGGING
for k, v in cali["baseline"].items():
    globals()[k] = v


# %% Only useful for debugging
# e_grid, e_dist, e_markov, a_grid, d_grid, d_markov, d_grid_name = make_grids(rho_e, sd_e, n_e, min_a, max_a, n_a, n_b, n_g, lifetime_b, lifetime_g)
# z_grid = income_grid(e_grid, tau, w, N)
# T = transfers(e_dist, Div, Tax, e_grid)
# adj_matrix = adj_costs(p_d, chi)
# disp_inc = disp_inc_f(a_grid, z_grid, T, r, adj_matrix)
# shifters = make_shifters(n_b, n_g, gamma_b, gamma_g)

# V, Va = hh_init(disp_inc, a_grid, eis, shifters)

# #%% Baseline model

# ss = dict()
# ss['baseline'] = hh.steady_state(cali['baseline'])
# print(ss['baseline']['A'])
# print('Proportion of people with a brown car at the end of the period (choice variable)',ss['baseline']['DD_TILDE_1'])
# print('Proportion of people with a green car at the end of the period (choice variable)',ss['baseline']['DD_TILDE_2'])
# print('Proportion of people with a brown car at the beginning of the period (state variable)',ss['baseline']['DD_1'])
# print('Proportion of people with a green car at the beginning of the period (state variable)',ss['baseline']['DD_2'])
# print('Ratio of DD_1/DD_TILDE_1: ',ss['baseline']['DD_1'] / ss['baseline']['DD_TILDE_1'])
# print('Ratio of DD_2/DD_TILDE_2: ',ss['baseline']['DD_2'] / ss['baseline']['DD_TILDE_2'])
# print(ss['baseline']['C'])
# #%% Policy functions
# policy_functions(ss, amax=150, d_tilde_list=ss['baseline'].internals['hh']['d_grid'] ,d_list = [0],ie_list=[0], figsize=0.8, models = ['baseline'])

# #%% Comparative statics of SS - 2d
# results = analyze_steady_state('chi', np.linspace(0.1, 0.5, 3), cali, hh, n_d)

# #%% Comparative statics of SS - 3d
# CS_dep_rate_chi_2 = analyze_steady_state_3d(
#     param1='dep_rate',
#     values1=np.linspace(0.05, 0.75, 5),
#     param2='chi',
#     values2=np.linspace(0.1, 0.8, 5),
#     cali=cali,
#     hh=hh,
#     n_d=n_d
#     )

# #%% Comparative statics of SS - 3d
# CS_lifetime = analyze_steady_state_3d(
#     param1='lifetime_b',
#     values1=np.linspace(30, 90, 5),
#     param2='lifetime_g',
#     values2=np.linspace(30, 90, 5),
#     cali=cali,
#     hh=hh,
#     n_d=n_d
#     )

# #%% Comparative statics of SS - 3d
# #Would be nice to analzye how it changes by changing gamma_g and gamma_b
# CS_gammas = analyze_steady_state_3d(
#     param1='gamma_b',
#     values1=np.linspace(0.9, 1.1, 5),
#     param2='gamma_g',
#     values2=np.linspace(0.9, 1.1, 5),
#     cali=cali,
#     hh=hh,
#     n_d=n_d
#     )
# # %%



#%% Add other blocks

@sj.simple
def fiscal(B, r, G, Y):
    Tax = (1 + r) * B(-1) + G - B  # total tax burden
    Z = Y - Tax
    deficit = G - Tax
    return Tax, deficit, Z


@sj.simple
def mkt_clearing(A, B, Y, C, G):
    asset_mkt = A - B
    goods_mkt = Y - C - G
    return asset_mkt, goods_mkt

@sj.simple
def prod(N, alpha):
    Y = N**alpha
    w = N**(alpha-1)
    return Y, w

#%% Create the model
ha = sj.create_model([hh, fiscal, mkt_clearing, prod], name="Simple HA Model")
print(ha)
print('It has inputs: ' + str(ha.inputs))
print('It has outputs: ' + str(ha.outputs))
# %% Evalaute model with basic calibration
cali['no_ss'] = deepcopy(cali['baseline'])
cali['no_ss']['r'] = cali['baseline']['r'] + 0
cali['no_ss']['G'] = cali['baseline']['G'] + 0.0
cali['no_ss']['beta'] = cali['baseline']['beta'] + 0.00
cali['no_ss']['B'] = cali['baseline']['B'] + 0
cali['no_ss']['N'] = cali['baseline']['N'] + 0.0




no_ss = ha.steady_state(cali['no_ss'])
# Print the result
print("Evaluating steady state with arbitrary calibration (no equilibrium solving):")
print(f"  Given beta = {cali['no_ss']['beta']}")
print(f"  Given r    = {cali['no_ss']['r']}")
print(f"  Given G    = {cali['no_ss']['G']}")
print(f"  Given B    = {cali['no_ss']['B']}")
print("Resulting market clearing residuals:")
print(f"  Goods market:  {np.round(no_ss['goods_mkt'], 5)}")
print(f"  Asset market:  {np.round(no_ss['asset_mkt'], 5)}")
print(f"  Tax:  {np.round(no_ss['Tax'], 5)}")






#%% Find the values for SS
#unknowns_ss = {'beta': 0.97, 'G': 0.3}
#unknowns_ss = {'r':0.005, 'G': 0.3}
#unknowns_ss = {'r':0.005, 'beta': 0.97}
unknowns_ss = {'beta':0.91}
targets_ss = {'asset_mkt'}

ss = ha.solve_steady_state(cali['baseline'], unknowns_ss, targets_ss, solver='hybr')
print(f"To attain SS, we need beta={np.round(ss['beta'],4)}")
print(f"To attain SS, we need Y={np.round(ss['Y'],4)}")

print(f"Check: Goods market clearing: {np.round(ss['goods_mkt'],5)}")
print(f"Check: Assets market clearing: {np.round(ss['asset_mkt'],5)}")

#%%
def display_ss_durables(ss):
    #display(Math(r"\tilde{D}^{None} = " + str(np.round(ss['D_T_N'], 3))))
    display(Math(r"D^{None} = " + str(np.round(ss['D_N'], 3))))

    #display(Math(r"\tilde{D}^{Brown, New} = " + str(np.round(ss['D_T_BN'], 3))))
    display(Math(r"D^{Brown, New} = " + str(np.round(ss['D_BN'], 3))))

    #display(Math(r"\tilde{D}^{Brown, Medium} = " + str(np.round(ss['D_T_BM'], 3))))
    #display(Math(r"D^{Brown, Medium} = " + str(np.round(ss['D_BM'], 3))))

    #display(Math(r"\tilde{D}^{Brown, Old} = " + str(np.round(ss['D_T_BO'], 3))))
    display(Math(r"D^{Brown, Old} = " + str(np.round(ss['D_BO'], 3))))
    display(Math(r"D^{Brown} = " + str(np.round(ss['D_B'], 3))))


    #display(Math(r"\tilde{D}^{Green, New} = " + str(np.round(ss['D_T_GN'], 3))))
    display(Math(r"D^{Green, New} = " + str(np.round(ss['D_GN'], 3))))

    #display(Math(r"\tilde{D}^{Green, Medium} = " + str(np.round(ss['D_T_GM'], 3))))
    #display(Math(r"D^{Green, Medium} = " + str(np.round(ss['D_GM'], 3))))

    #display(Math(r"\tilde{D}^{Green, Old} = " + str(np.round(ss['D_T_GO'], 3))))
    display(Math(r"D^{Green, Old} = " + str(np.round(ss['D_GO'], 3))))
    display(Math(r"D^{Green} = " + str(np.round(ss['D_G'], 3))))


    #display(Math(r"Check. Total(Tilde) = " + str(ss['D_T_N'] + ss['D_T_BN']+ ss['D_T_BM']+ ss['D_T_BO']+ ss['D_T_GN']+ ss['D_T_GM'] + ss['D_T_GO'])))
    #display(Math(r"Check. Total = " + str(ss['D_N'] + ss['D_BN']+ ss['D_BM']+ ss['D_BO']+ ss['D_GN']+ ss['D_GM'] + ss['D_GO'])))
    #display(Math(r"Check. Total(Tilde) = " + str(ss['D_T_N'] + ss['D_T_BN']+ ss['D_T_BO']+ ss['D_T_GN'] + ss['D_T_GO'])))
    display(Math(r"Check. Total = " + str(ss['D_N'] + ss['D_BN']+ ss['D_BO']+ ss['D_GN'] + ss['D_GO'])))
    
def display_calibrated_from_unknowns(ss_dict, unknowns_dict):
    """
    Display calibrated parameters from ss_dict,
    only for keys in unknowns_dict.
    Works with SteadyStateDict (no .get method).
    """
    print(f"{'Parameter':<10} | {'Calibrated Value':>15}")
    print("-" * 30)
    for param in unknowns_dict.keys():
        try:
            value = np.round(ss_dict[param], 3)
        except KeyError:
            value = 'N/A'
        print(f"{param:<10} | {value:>15}")


#%% Calibrate to attain the empirical fractions of cars 

targets_ss = {'asset_mkt': 0.,
    'D_N': 1-0.075 - 0.55,
    'D_G': 0.075,
    'D_B':0.55,
    #'D_BN':0.007,
    }  # <-- with a dict rather than a list, we can specify specific targets for output variables

unknowns_ss = {
    'beta': (0.80, 0.881, 0.95),
    'p_b': (0.01, 0.273, 10),
    'p_g': (0.01, 0.9, 10),
    'gamma_g': (0,1.243,100),
    #'dep_util_frac_b': (0.1,0.99,1)
}

ss_DD = ha.solve_steady_state(cali['baseline'], unknowns_ss, targets_ss, solver = 'broyden_custom')

display_ss_durables(ss_DD)
display_calibrated_from_unknowns(ss_DD, unknowns_ss)

#%%
cali['ss_DD'] = ha.steady_state(ss_DD)
cali['ss_DD_mod'] = deepcopy(cali['ss_DD'])
cali['ss_DD_mod']['dep_util_frac_b'] = cali['ss_DD_mod']['dep_util_frac_b']*0.50
#cali['ss_DD_mod']['gamma_b'] = cali['ss_DD_mod']['gamma_b']*1.50


print('Original model:')
display_ss_durables(ha.steady_state(cali['ss_DD']))
print('Modified model:')
display_ss_durables(ha.steady_state(cali['ss_DD_mod']))


#%%
for key, value in cali['ss_DD'].items():
    if 'D_' in key:
        print(key, ":", np.round(value, 2))


#%% Use the ss
cali['ss'] = ha.steady_state(ss)
#%%
for key, value in ss.items():
    print(key, ":", np.round(value,2))

# %%
T = 300
#breakpoint()
J_ha = hh.jacobian(ss, inputs=['r'], T=T)

# %%
s_to_plot = [0, 50, 100, 150]
for s in s_to_plot:
   plt.plot(J_ha['A']['r'][:, s], label =f's={s}')
plt.legend()
plt.show()

#%%
# %% IRFs
T = 300  # <-- the length of the IRF
rho_r = 0.8
eR = 0.01
rho_B = 0.8
eB = 0.01*0
dr = eR * rho_r ** np.arange(T)
dB = eB * rho_B ** np.arange(T)
shocks = {"r": dr, "B": dB}
unknowns_td = ['N']
targets_td = ["asset_mkt"]
irfs = ha.solve_impulse_linear(ss, unknowns_td, targets_td, shocks)
irfs_alt = ha.solve_impulse_linear(ss_DD, unknowns_td, targets_td, shocks)
show_irfs([irfs, irfs_alt], ["N","w","C","Y", "A", "goods_mkt", "asset_mkt"],  labels=["Default Calib","Calibrated"], figsize=(18,3))
show_irfs([irfs, irfs_alt], ["D_N","D_BO","D_BN","D_GO","D_GN"],  labels=["Default Calib","Calibrated"], figsize=(18,3))


#%% Compute and plot directly
plot_linear_irfs(
    shocks_list=['r'],
    unknowns_td=['G','N'],
    targets_td=['asset_mkt',"goods_mkt"],
    ha=ha,
    ss=ss,
    outputs=["N", "G", "Tax","r", "B", "w", "C", "Y", "A", "goods_mkt", "asset_mkt"]
)

#%%
plot_linear_irfs(
    shocks_list=['tau'],
    unknowns_td=['G','N'],
    targets_td=['asset_mkt',"goods_mkt"],
    ha=ha,
    ss=ss,
    outputs=["N", "G", "Tax","r", "B", "w", "C", "Y", "A", "goods_mkt", "asset_mkt"]
)

# %% IRFs
T = 300  # <-- the length of the IRF
rho_r = 0.8
dr = 0.01 * rho_r ** np.arange(T)
shocks = {"G": dr}
unknowns_td = ['N']
targets_td = ["asset_mkt"]
irfs = ha.solve_impulse_linear(ss, unknowns_td, targets_td, shocks)

# %% IRFs
T = 300  # <-- the length of the IRF
dB = 0.01 * 0.8 ** np.arange(T)
shocks = {"G": dr, "B": dB}
unknowns_td = ['N']
targets_td = ["asset_mkt"]
irfs_B = ha.solve_impulse_linear(ss, unknowns_td, targets_td, shocks)

#%% Plot IRFs
show_irfs([irfs, irfs_B], ["r","C","Y", "A", "goods_mkt", "asset_mkt", "B"],  labels=["..."], figsize=(18,3))



# %%
print(f"Execution time: {time.time() - start:.2f} seconds")
# %%
