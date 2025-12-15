import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Math
from copy import deepcopy
from scipy.interpolate import interp1d, griddata
import colorsys
import time
import math


def policy_functions(
    ss,
    xmax=10,
    xmin=0,
    d_tilde_list=[0],
    d_list=[0],
    ie_list=[3],
    figsize=0.6,
    models=['baseline'],
    plots=['assets','da','cons','disc'],
    vintage_groups=None,
    save_path=None
):
    """
    Plot household policy functions (assets, Δassets, consumption, discrete choice).

    Optional vintage aggregation:
    -----------------------------
    vintage_groups: dict or None
        Example:
        {
            "Brown": [1, 2],   # New Brown + Old Brown
            "Green": [3, 4]    # New Green + Old Green
        }

        If None, each d_tilde is plotted separately (default behavior).
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # ---- 0. Extract common grids/objects -------------------------------------
    a_grid = ss['baseline'].internals['hh']['a_grid']
    A = ss['baseline']['A']

    amin_idx = np.searchsorted(a_grid, xmin * A, side='left')
    amax_idx = np.searchsorted(a_grid, xmax * A, side='right')
    amax_idx = min(amax_idx, len(a_grid))

    # ---- 1. Extract policy objects -------------------------------------------
    a, da, c, P = {}, {}, {}, {}

    for model in models:
        hh = ss[model].internals['hh']
        a[model]  = hh['consav']['a']
        da[model] = a[model] - a_grid
        c[model]  = hh['consav']['c']
        P[model]  = hh['durables']['law_of_motion'].P

    # ---- 2. Which plots to show ----------------------------------------------
    plot_map = {'assets': 0, 'da': 1, 'cons': 2, 'disc': 3}
    selected = [plot_map[p] for p in plots if p in plot_map]

    titles = {
        0: r'Assets ($a^*(\tilde{d},\,d,\,z,\,a^{-})$)',
        1: r'Savings',
        2: r'Consumption',
        3: r'Durable Adoption Probability'
    }
    ylabels = {0: "Assets", 1: "Δ Assets", 2: "Consumption", 3: "Probability"}

    # ---- 3. Figure -----------------------------------------------------------
    fig, axes = plt.subplots(
        1, len(selected),
        figsize=(6 * figsize * len(selected), 5 * figsize)
    )
    if len(selected) == 1:
        axes = [axes]

    # ---- 4. Style maps -------------------------------------------------------
    base_lw = 2.0
    lw_map = {m: base_lw + 0.8*i for i, m in enumerate(models)}

    linestyles = ['-', '--', '-.', ':']
    linestyle_map = {iz: linestyles[i_ % len(linestyles)]
                     for i_, iz in enumerate(ie_list)}

    fixed_colors = {
        0: "#808080",   # None
        1: "#8B4513",   # New Brown
        2: "#C4A484",   # Old Brown
        3: "#228B22",   # New Green
        4: "#90EE90"    # Old Green
    }

    fixed_labels = {
        0: "None",
        1: "New Brown",
        2: "Old Brown",
        3: "New Green",
        4: "Old Green"
    }

    group_colors = {
        "None": "#808080",
        "Brown": "#8B4513",
        "Green": "#228B22"
    }

    if len(d_list) > 1:
        alphas = np.linspace(0.3, 1.0, len(d_list))
    else:
        alphas = [1.0]
    alpha_map = {d: a for d, a in zip(d_list, alphas)}

    single_model = len(models) == 1

    # ---- 5. Define plotting groups -------------------------------------------
    if vintage_groups is None:
        plot_groups = {fixed_labels[dt]: [dt] for dt in d_tilde_list}
        group_color = {fixed_labels[dt]: fixed_colors[dt] for dt in d_tilde_list}
    else:
        plot_groups = vintage_groups
        group_color = group_colors

    # ---- 6. Plot loops -------------------------------------------------------
    for model in models:
        for group_name, d_tildes in plot_groups.items():
            for d in d_list:
                for iz in ie_list:

                    lw = lw_map[model]
                    ls = linestyle_map[iz]
                    col = group_color[group_name]
                    alpha = alpha_map[d]

                    label = group_name
                    if not single_model:
                        label = f"{model} – {label}"

                    # ---- Sum over vintages ---------------------------------
                    a_sum  = sum(a[model][dt, d, iz, amin_idx:amax_idx]
                                 for dt in d_tildes)
                    da_sum = sum(da[model][dt, d, iz, amin_idx:amax_idx]
                                 for dt in d_tildes)
                    c_sum  = sum(c[model][dt, d, iz, amin_idx:amax_idx]
                                 for dt in d_tildes)
                    P_sum  = sum(P[model][dt, d, iz, amin_idx:amax_idx]
                                 for dt in d_tildes)

                    x = a_grid[amin_idx:amax_idx] / A

                    if 0 in selected:
                        axes[selected.index(0)].plot(
                            x, a_sum, color=col, linestyle=ls,
                            linewidth=lw, alpha=alpha, label=label
                        )

                    if 1 in selected:
                        axes[selected.index(1)].plot(
                            x, da_sum, color=col, linestyle=ls,
                            linewidth=lw, alpha=alpha, label=label
                        )

                    if 2 in selected:
                        axes[selected.index(2)].plot(
                            x, c_sum, color=col, linestyle=ls,
                            linewidth=lw, alpha=alpha, label=label
                        )

                    if 3 in selected:
                        axes[selected.index(3)].plot(
                            x, P_sum, color=col, linestyle=ls,
                            linewidth=lw, alpha=alpha, label=label
                        )

    # ---- 7. Formatting -------------------------------------------------------
    for idx, ax in zip(selected, axes):

        if idx == 0:
            ax.plot(x, a_grid[amin_idx:amax_idx],
                    color='gray', linestyle=':', linewidth=1.2)

        if idx == 1:
            ax.axhline(0, color='gray', linestyle=':', linewidth=1.2)

        ax.set_title(titles[idx])
        ax.set_ylabel(ylabels[idx])
        ax.set_xlabel('Ratio of ind. wealth to avg. wealth')
        ax.set_xlim([xmin, xmax-1])

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()






def plot_heatmap(Pi, title="Matrix Heatmap", fmt=".2f"):
    """
    Plots a heatmap of a 2D matrix using matplotlib.

    - Uses red→yellow→green if values are in [0,1].
    - Otherwise uses 'viridis' colormap.

    Parameters:
    - Pi: 2D array-like
    - title: plot title
    - fmt: string format for values in the cells
    """
    Pi = np.array(Pi)
    n_rows, n_cols = Pi.shape

    # Check range
    min_val, max_val = np.min(Pi), np.max(Pi)
    values_in_unit_interval = 0 <= min_val and max_val <= 1

    # Choose colormap
    #if values_in_unit_interval:
     #   cmap = LinearSegmentedColormap.from_list("red_yellow_green", ["red", "yellow", "green"])
    #    vmin, vmax = 0, 1
    #else:
    cmap = "viridis"
    vmin, vmax = None, None  # auto-scale

    fig, ax = plt.subplots(figsize=(1 + n_cols, 1 + n_rows))
    cax = ax.imshow(Pi, cmap=cmap, vmin=vmin, vmax=vmax)

    # Add colorbar
    fig.colorbar(cax)

    # Axis ticks and labels
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels([f"{j}" for j in range(n_cols)])
    ax.set_yticklabels([f"{i}" for i in range(n_rows)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate values
    for i in range(n_rows):
        for j in range(n_cols):
            val = Pi[i, j]
            ax.text(j, i, format(val, fmt),
                    ha="center", va="center",
                    color="black" if not values_in_unit_interval or val > 0.7 or val < 0.3 else "white")

    ax.set_title(title)
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")
    plt.tight_layout()
    plt.show()

def pplot(array, discrete_choice=1, prod_type=0, a_min=0, a_max=199):
    # Try to infer the variable name passed to `array`
    try:
        frame = inspect.currentframe().f_back
        array_name = next((name for name, val in frame.f_locals.items() if val is array), "array")
    finally:
        del frame  # Clean up to avoid reference cycles

    # Extract the data slice
    y_vals = array[discrete_choice, prod_type, a_min:a_max]

    # Plot
    plt.plot(range(a_min, a_max), y_vals)
    plt.title(f"{array_name}: discr_choice={discrete_choice}, prod_type={prod_type}, a ∈ [{a_min}, {a_max})")
    plt.xlabel("Asset index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def ppplot(array, discrete_choice=1, prod_type=0, a_min=0, a_max=199):
    # Try to infer the variable name passed to `array`
    try:
        frame = inspect.currentframe().f_back
        array_name = next((name for name, val in frame.f_locals.items() if val is array), "array")
    finally:
        del frame  # Clean up to avoid reference cycles

    # Extract the data slice
    y_vals = array[discrete_choice*7 +  prod_type, a_min:a_max]

    # Plot
    plt.plot(range(a_min, a_max), y_vals)
    plt.title(f"{array_name}: discr_choice={discrete_choice}, prod_type={prod_type}, a ∈ [{a_min}, {a_max})")
    plt.xlabel("Asset index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
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

def make_strictly_increasing(uc):
    uc_fixed = uc.copy()
    shape = uc.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            row = uc[i, j, :]

            # Replace -inf at the beginning with decreasing values
            if np.isneginf(row[0]):
                first_finite = np.argmax(~np.isneginf(row))
                if first_finite == 0:
                    continue  # No -inf at start
                increment = 1.0  # adjust based on scale if needed
                for k in range(first_finite - 1, -1, -1):
                    row[k] = row[k + 1] - increment

            # Make strictly increasing
            for k in range(1, shape[2]):
                if row[k] <= row[k - 1]:
                    row[k] = row[k - 1] + 1e-8  # enforce strict increase

            uc_fixed[i, j, :] = row

    return uc_fixed




def analyze_steady_state(param_name, param_values, cali, hh, variables, n_d=None):
    """
    Vary a calibration parameter and plot steady-state outcomes for chosen variables.

    Args:
        param_name (str): Name of the parameter in cali to vary.
        param_values (array-like): Grid of values to assign to the parameter.
        cali (dict): Dictionary containing the baseline calibration.
        hh (object): Object with a method `steady_state(cali_dict)` returning dict-like steady-state values.
        variables (list[str]): List of variable names to track (e.g., ['A', 'C', 'DD_0', 'DD_TILDE_2']).
        n_d (int, optional): Number of durable states (if plotting DD/DD_TILDE across vintages).

    Returns:
        dict: Dictionary with results {var_name: np.array of results}.
    """
    # Storage for each variable
    results = {var: [] for var in variables}
    success_flags = []

    for val in param_values:
        cali_try = cali.copy()
        cali_try[param_name] = val
        try:
            ss_try = hh.steady_state(cali_try)

            for var in variables:
                if var in ss_try:
                    results[var].append(ss_try[var])
                else:
                    results[var].append(np.nan)

            success_flags.append(True)
            print(f"SS found for {param_name} = {val:.3f}!")
        except Exception as e:
            print(f"Failed for {param_name} = {val:.3f}: {e}")
            for var in variables:
                results[var].append(np.nan)
            success_flags.append(False)

    # Convert to arrays
    param_values = np.array(param_values)
    for var in variables:
        results[var] = np.array(results[var])

    # --- Plotting ---
    n_vars = len(variables)
    fig, axs = plt.subplots(1, n_vars, figsize=(4 * n_vars, 3), sharex=True)
    if n_vars == 1:
        axs = [axs]

    for ax, var in zip(axs, variables):
        vals = results[var]
        mask = ~np.isnan(vals)

        if mask.sum() >= 2:  # enough points for interpolation
            interp_func = interp1d(param_values[mask], vals[mask], kind='linear', fill_value="extrapolate")
            vals_interp = vals.copy()
            vals_interp[~mask] = interp_func(param_values[~mask])
        else:
            vals_interp = vals

        ax.plot(param_values, vals_interp, '--', label=var)
        ax.plot(param_values[mask], vals[mask], 'o', label=f"{var} (ok)")
        ax.plot(param_values[~mask], vals_interp[~mask], 'x', color='red', label=f"{var} (interp)")
        ax.set_title(var)
        ax.set_xlabel(param_name)
        ax.grid(True)
        ax.legend()

    axs[0].set_ylabel("Value")
    plt.tight_layout()
    plt.show()

    return {
        'param_values': param_values,
        'results': results,
        'success_flags': success_flags,
    }



def analyze_steady_state_3d(param1, values1, param2, values2, cali, hh, variables, n_d=None, results=None):
    """
    Vary two calibration parameters and plot steady-state outcomes
    for chosen variables (not hardcoded).

    Args:
        param1 (str): First parameter name.
        values1 (array-like): Grid for the first parameter.
        param2 (str): Second parameter name.
        values2 (array-like): Grid for the second parameter.
        cali (dict): Calibration dictionary.
        hh (object): Must have method `steady_state(cali_dict)`.
        variables (list[str]): List of variable names to track (e.g., ['A','C','DD_0','DD_TILDE_2']).
        n_d (int, optional): Number of durable choices (for shorthand expansion).
        results (dict, optional): Previous results object to reuse.

    Returns:
        dict: Grid and results {var_name: 2D arrays}.
    """

    values1 = np.array(values1)
    values2 = np.array(values2)
    X, Y = np.meshgrid(values1, values2, indexing='ij')

    # Storage
    if results is None:
        results = {var: np.full_like(X, np.nan, dtype=float) for var in variables}
        success = np.full_like(X, False, dtype=bool)

        for i, v1 in enumerate(values1):
            for j, v2 in enumerate(values2):
                cali_try = cali.copy()
                cali_try[param1] = v1
                cali_try[param2] = v2
                try:
                    ss_try = hh.steady_state(cali_try)
                    for var in variables:
                        if var in ss_try:
                            results[var][i, j] = ss_try[var]
                        else:
                            results[var][i, j] = np.nan
                    success[i, j] = True
                    print(f"Success: {param1}={v1:.2f}, {param2}={v2:.2f}")
                except Exception as e:
                    print(f"Fail: {param1}={v1:.2f}, {param2}={v2:.2f} | {e}")
    else:
        success = results['success']

    # === Interpolation helper ===
    def interpolate_missing(Z):
        mask = ~np.isnan(Z)
        if mask.sum() < 3:  # not enough points for griddata
            return Z
        points = np.column_stack((X[mask], Y[mask]))
        values = Z[mask]
        return griddata(points, values, (X, Y), method='linear')

    # Interpolated versions
    interp_results = {var: interpolate_missing(results[var]) for var in variables}

    # === Plotting ===
    n_vars = len(variables)
    ncols = min(4, n_vars)
    nrows = int(np.ceil(n_vars / ncols))
    fig = plt.figure(figsize=(5 * ncols, 4 * nrows))

    for k, var in enumerate(variables):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection='3d')
        ax.plot_surface(X, Y, interp_results[var], cmap='viridis', alpha=0.7, edgecolor='none')
        ax.scatter(X[success], Y[success], results[var][success], s=10, color='black')
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_zlabel(var)
        ax.set_title(var)

    plt.tight_layout()
    plt.show()

    # Return both raw and interpolated
    return {
        'X': X, 'Y': Y,
        'results': results,
        'interp_results': interp_results,
        'success': success
    }


def show_irfs(irfs_list, variables, labels=None, ylabel=r"PP (dev. from ss)",
              T_plot=50, figsize=(18, 6), save_path=None, titles=None):
    """
    Plot impulse response functions (IRFs) for multiple variables and scenarios.

    Parameters
    ----------
    irfs_list : list of dict
        Each dict contains IRFs for variables.
    variables : list of str
        List of variable names to plot.
    labels : list of str, optional
        Labels for each IRF scenario.
    ylabel : str
        Y-axis label.
    T_plot : int
        Number of periods to plot.
    figsize : tuple
        Figure size.
    save_path : str or None
        If provided, save the figure to this path.
    titles : list or dict, optional
        Custom LaTeX titles for each variable subplot.
        If list, must have same length as `variables`.
        If dict, keys are variable names.
    """

    if labels is None or len(irfs_list) != len(labels):
        labels = ["Scenario {}".format(i+1) for i in range(len(irfs_list))]

    n_var = len(variables)
    
    # Dynamically choose rows and columns
    n_cols = min(3, n_var)
    n_rows = math.ceil(n_var / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = np.array(axes).reshape(-1)  # Flatten axes array

    for i, var in enumerate(variables):
        for j, irf in enumerate(irfs_list):
            if var in irf:
                data = 100 * np.array(irf[var][:T_plot])
            else:
                data = np.zeros(T_plot)
            #axes[i].plot(data, label=labels[j])
            axes[i].plot(data)

        # Use custom title if provided
        if titles is not None:
            if isinstance(titles, dict) and var in titles:
                axes[i].set_title(titles[var], usetex=True, fontsize=16)
            elif isinstance(titles, list) and i < len(titles):
                axes[i].set_title(titles[i], usetex=True, fontsize=16)
            else:
                axes[i].set_title(var, fontsize=16)
        else:
            axes[i].set_title(var, fontsize=16)

        axes[i].set_xlabel(r"quarter")
        axes[i].set_ylabel(ylabel)
        axes[i].grid(True)
        #axes[i].legend()

    # Remove empty subplots
    for k in range(n_var, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


def plot_linear_irfs(shocks_list, unknowns_td, targets_td, ha, ss, outputs, T_plot=50,
                     rho=None, e=None, T=300, figsize=(18, 6), ylabel=r"PP (dev. from ss)",
                     labels=None, save_path=None, titles=None):
    """
    Compute linear IRFs and plot them.
    """

    # Default values if not provided
    if rho is None:
        rho = {shock: 0.8 for shock in shocks_list}
    if e is None:
        e = {shock: 0.01 for shock in shocks_list}

    # Build shocks dictionary with time series
    shocks = {shock: e[shock] * rho[shock]**np.arange(T) for shock in shocks_list}

    # Solve the system
    irfs = ha.solve_impulse_linear(ss, unknowns_td, targets_td, shocks)

    # Default label
    if labels is None:
        labels = [" + ".join(shocks_list)]

    # Plot with custom titles
    show_irfs([irfs], outputs, labels=labels, ylabel=ylabel, T_plot=T_plot,
              figsize=figsize, save_path=save_path, titles=titles)
    
    return irfs


def evaluate_param_changes(param_name, values_list, ha, cali,
                           ss_vars=['goods_mkt', 'asset_mkt', 'Tax', 'r', 'beta', 'G', 'B', 'N', 'Y', 'Z']):
    # Get baseline calibration and steady state
    baseline_calib = deepcopy(cali)
    try:
        baseline_ss = ha.steady_state(baseline_calib)
    except Exception as e:
        print(f"❌ Error computing baseline steady state: {e}")
        return  # Abort entirely if baseline itself is invalid

    # Helper: safe extraction from steady state dict-like object
    def safe_ss_get(ss_obj, key):
        try:
            return ss_obj[key]
        except Exception:
            return 'N/A'

    # Store results
    calibration_results = []
    ss_results = []

    # Baseline row
    try:
        baseline_value = baseline_calib[param_name]
    except Exception:
        baseline_value = 'N/A'

    calibration_results.append(('Baseline', baseline_value))
    ss_results.append(('Baseline', [safe_ss_get(baseline_ss, v) for v in ss_vars]))

    # Loop over alternative values
    for val in values_list:
        case_name = f"{param_name} = {val}"
        modified_calib = deepcopy(baseline_calib)
        modified_calib[param_name] = val

        try:
            ss = ha.steady_state(modified_calib)
            ss_values = [safe_ss_get(ss, v) for v in ss_vars]
        except Exception as e:
            ss_values = ['ERROR'] * len(ss_vars)
            case_name += " (Error)"
            print(f"⚠️  Error for {param_name}={val}: {e}")

        calibration_results.append((case_name, val))
        ss_results.append((case_name, ss_values))

    # Print Calibration Table
    print(f"\n🔧 Parameter sweep for '{param_name}'")
    print("\n📌 Calibration Values:")
    print(f"{'Case':<30} | {param_name}")
    print("-" * 45)
    for case, val in calibration_results:
        if isinstance(val, (float, int)):
            print(f"{case:<30} | {val:.5f}")
        else:
            print(f"{case:<30} | {val}")

    # Print SS Table
    print("\n📈 Steady-State Outcomes:")
    header = f"{'Case':<30} | " + " | ".join([f"{v:<10}" for v in ss_vars])
    print(header)
    print("-" * len(header))
    for case, row in ss_results:
        formatted_row = []
        for x in row:
            if isinstance(x, (float, int)):
                formatted_row.append(f"{x:>10.5f}")
            else:
                formatted_row.append(f"{str(x):>10}")
        print(f"{case:<30} | " + " | ".join(formatted_row))
        
        
        
def evaluate_two_param_changes(param1, values1, param2, values2, ha, cali,
                               ss_vars=['goods_mkt', 'asset_mkt', 'Tax', 'r', 'beta', 'G', 'B', 'N', 'Y', 'Z']):
    # --- Baseline calibration ---
    baseline_calib = deepcopy(cali)
    try:
        baseline_ss = ha.steady_state(baseline_calib)
    except Exception as e:
        print(f"❌ Error computing baseline steady state: {e}")
        return  # stop early if even baseline fails

    # --- Safe extraction helper ---
    def safe_ss_get(ss_obj, key):
        try:
            return ss_obj[key]
        except Exception:
            return 'N/A'

    # --- Store results ---
    results = []

    # Baseline
    try:
        base_val1 = baseline_calib[param1]
    except Exception:
        base_val1 = 'N/A'
    try:
        base_val2 = baseline_calib[param2]
    except Exception:
        base_val2 = 'N/A'

    results.append(('Baseline', base_val1, base_val2,
                    [safe_ss_get(baseline_ss, v) for v in ss_vars]))

    # --- Loop over parameter pairs ---
    for v1 in values1:
        for v2 in values2:
            case_name = f"{param1}={v1}, {param2}={v2}"
            modified_calib = deepcopy(baseline_calib)
            modified_calib[param1] = v1
            modified_calib[param2] = v2

            try:
                ss = ha.steady_state(modified_calib)
                ss_values = [safe_ss_get(ss, v) for v in ss_vars]
            except Exception as e:
                ss_values = ['N/A'] * len(ss_vars)
                print(f"⚠️  Error for {case_name}: {e}")

            results.append((case_name, v1, v2, ss_values))

    # --- Print Calibration Table ---
    print(f"\n🔧 2D Parameter sweep for '{param1}' and '{param2}'")
    print("\n📌 Calibration Values:")
    print(f"{'Case':<40} | {param1:<10} | {param2:<10}")
    print("-" * 65)
    for case_name, v1, v2, _ in results:
        v1_str = f"{v1:.5f}" if isinstance(v1, (int, float)) else str(v1)
        v2_str = f"{v2:.5f}" if isinstance(v2, (int, float)) else str(v2)
        print(f"{case_name:<40} | {v1_str:<10} | {v2_str:<10}")

    # --- Print Steady-State Table ---
    print("\n📈 Steady-State Outcomes:")
    header = f"{'Case':<40} | " + " | ".join([f"{v:<10}" for v in ss_vars])
    print(header)
    print("-" * len(header))
    for case_name, _, _, ss_values in results:
        formatted_row = [
            f"{x:>10.5f}" if isinstance(x, (float, int)) else f"{str(x):>10}"
            for x in ss_values
        ]
        print(f"{case_name:<40} | " + " | ".join(formatted_row))

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Math

def display_ss_durables(ss, title="Durable Shares", save_path=None, show_plot=False, durables_to_plot="full"):
    """
    Display and optionally plot durable goods aggregates from a steady state dictionary.

    Parameters:
    ss : dict
        Dictionary containing steady-state values with keys like 'D_N', 'D_BN', 'D_BO', etc.
    title : str
        Title for the plot.
    save_path : str or None
        If provided, the plot will be saved to this path.
    show_plot : bool
        If True, the plot will be displayed.
    durables_to_plot : str or list
        "full" for all 5 durables, "restricted" for D_N, D_B, D_G, or a custom list of keys.
    """

    # === Determine which durables to display/plot ===
    if durables_to_plot == "full":
        display_keys = ['D_N', 'D_BN', 'D_BO', 'D_GN', 'D_GO']
        display_labels = ['None', 'Brown, New', 'Brown, Old', 'Green, New', 'Green, Old']
        colors = ["#808080", "#8B4513", "#C4A484", "#228B22", "#90EE90"]
    elif durables_to_plot == "restricted":
        display_keys = ['D_N', 'D_B', 'D_G']
        display_labels = ['None', 'Brown', 'Green']
        colors = ["#808080", "#8B4513", "#228B22"]
    elif isinstance(durables_to_plot, list):
        display_keys = durables_to_plot
        display_labels = durables_to_plot  # Default labels are keys; user can modify later if needed
        colors = plt.cm.tab10.colors[:len(display_keys)]  # Auto-select colors
    else:
        raise ValueError("durables_to_plot must be 'full', 'restricted', or a list of keys.")

    # === Display ===
    for k, label in zip(display_keys, display_labels):
        if k in ss:
            display(Math(f"{label} = {np.round(ss[k], 3)}"))

    # Check total (optional: sum of selected durables)
    total = sum(ss[k] for k in display_keys if k in ss)
    display(Math(f"Check. Total = {np.round(total, 3)}"))

    # === Plotting ===
    if show_plot or save_path:
        values = [ss[k] for k in display_keys if k in ss]
        plt.figure(figsize=(8,5))
        plt.bar(display_labels, values, color=colors)
        plt.ylabel(title)
        plt.title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        
        if show_plot:
            plt.show()
        else:
            plt.close()



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

import numpy as np
import matplotlib.pyplot as plt

def plot_distribution(
    SS_object,
    lines_dim=None,       # dimension to split lines (0,1,2), or None
    truncate_at=None,     # float, ratio of assets to average wealth at which to truncate
    labels=None,          # list of labels
    normalize=False,      # normalize each line to sum to 1
    save_path=None        # str or None, file path to save the figure
):
    """
    Plot 4D distribution D along assets (x-axis).

    Parameters
    ----------
    SS_object : object
        Object containing D at SS_object.internals['hh']['consav']['D']
        and a_grid at SS_object.internals['hh']['a_grid']
    lines_dim : int or None
        Dimension to separate lines (0,1,2) or None for total marginal
    truncate_at : float or None
        If specified, all asset levels >= truncate_at * average wealth are combined in last bin
    labels : list or None
        List of labels for the lines_dim
    normalize : bool
        If True, normalize each line to sum to 1
    save_path : str or None
        If provided, save the figure to this path (e.g., "distribution.png")
    """

    D = np.asarray(SS_object.internals['hh']['consav']['D'])
    a_grid = np.asarray(SS_object.internals['hh']['a_grid'])
    A = SS_object['A']
    assert D.ndim == 4, "D must be 4D (choice, durable state, productivity, assets)"

    # Determine lines
    if lines_dim is None:
        lines = [D.sum(axis=(0,1,2))]  # marginalize all except assets
        line_labels = ["Total"]
    else:
        axes_to_sum = tuple(ax for ax in range(4) if ax not in (3, lines_dim))
        lines = D.sum(axis=axes_to_sum)

        if lines_dim != 3:
            lines = [lines[i, :] for i in range(lines.shape[0])]
        else:
            lines = [lines[i] for i in range(lines.shape[0])]

        if labels is None:
            line_labels = [f"{lines_dim}={i}" for i in range(len(lines))]
        else:
            line_labels = labels
            assert len(line_labels) == len(lines), "Number of labels must match number of lines"

    # Truncate assets if requested
    if truncate_at is not None:
        truncate_val = truncate_at * A
        truncate_idx = np.searchsorted(a_grid, truncate_val, side='right') - 1
        truncated_lines = []
        for line in lines:
            if truncate_idx < len(line):
                new_line = np.zeros(truncate_idx+1)
                new_line[:-1] = line[:truncate_idx]
                new_line[-1] = line[truncate_idx:].sum()
                truncated_lines.append(new_line)
            else:
                truncated_lines.append(line)
        lines = truncated_lines
        x_axis = a_grid[:truncate_idx+1]/A
    else:
        x_axis = a_grid/A

    # Normalize each line
    if normalize:
        lines = [line / line.sum() for line in lines]

    # Plot
    plt.figure(figsize=(8,5))
    for line, lbl in zip(lines, line_labels):
        plt.plot(x_axis, line, label=lbl)

    plt.xlabel("Ratio of ind. wealth to avg. wealth")
    plt.ylabel("%" if normalize else "% (sum = 100)")
    if lines_dim is not None:
        plt.legend()
    plt.title("Stationary Distribution")
    plt.grid(True)

    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


def check_resource_constraint(ss):
    """
    Check key equilibrium conditions.
    Checks:
      - Aggregate resource constraint
      - Labor market clearing
      - Asset market clearing
    Returns a dictionary with differences.
    """
    results = {}
    # Household objects
    D = ss.internals['hh']['consav']['D']
    p_bundle = ss.internals['hh']['p_bundle']
    c = ss.internals['hh']['consav']['c']
    a_choice = ss.internals['hh']['consav']['a']
    a_grid = ss.internals['hh']['a_grid']
    adj_matrix = ss.internals['hh']['adj_matrix']
    r = ss['r']
    N = ss['N']
    w = ss['w']
    e_grid = ss.internals['hh']['e_grid']
    T = ss.internals['hh']['T']
    A = ss['A']
    Tax = ss['Tax']
    T_E = ss['T_E']

    p_core = ss['p_core']
    c_core = ss.internals['hh']['consav']['c_core']
    C_core = np.sum(c_core * D)
    c_E = ss.internals['hh']['consav']['c_E']
    C_E = np.sum(c_E * D, axis=(1,2,3))
    p_E = ss.internals['hh']['p_e']
    #p_d = ss.internals['hh']['p_d']
    d = ss.internals['hh']['d']
    Y = ss['Y']
    #Y_d = ss['Y_d']
    #Y_d = np.array([ss['Y_d0'], ss['Y_d1'], ss['Y_d2'], ss['Y_d3'], ss['Y_d4']])
    G = ss['G']
    chi = ss['chi']
    tau_vec = ss.internals['hh']['tau_vec']
    eps_vec = ss.internals['hh']['eps_vec']

    # Individual BC (0)
    lhs = p_bundle[...,np.newaxis,np.newaxis,np.newaxis] * c \
        + a_choice \
        + adj_matrix[..., np.newaxis,np.newaxis]
    rhs = ((1 + r) * a_grid)[np.newaxis,np.newaxis,np.newaxis,...] \
        + w * N * e_grid[np.newaxis, np.newaxis, :, np.newaxis] \
        + T[np.newaxis, np.newaxis, :, np.newaxis]
    #print("Max individual BC deviation:", np.max(np.abs(lhs - rhs)))
        
    # Aggregated BC (0 agg)
    LHS_0 = np.sum(lhs * D)
    RHS_0 = np.sum(rhs * D)
    print("Aggregating the individual BC (0 agg):", LHS_0 - RHS_0)
    
    target = np.sum((adj_matrix[..., np.newaxis,np.newaxis]) * D)
    
    # Agg (1)
    # Get X flows  
    S = np.sum(D, axis=(2, 3)) 
    X_plus = np.sum(d.reshape(-1, 1)  * (S * (1 - np.eye(S.shape[0]))), axis=1)
    X_minus = np.sum(d* (S * (1 - np.eye(S.shape[0]))), axis=0)

    LHS_1 = p_core*C_core + np.sum((1+tau_vec) * (1+eps_vec) * p_E * C_E) + A + np.sum(X_plus - X_minus * (1-chi)) 
    RHS_1 = np.mean((1+r)*A + w*N + T)
    print("Aggregated BC (with flows of durables) (1 agg)", LHS_1 - RHS_1)

    # Include GBC (2)
    # Government BC
    LHS_2 = p_core*C_core + np.sum((1+eps_vec) * p_E * C_E) + np.sum(X_plus - X_minus * chi) + G
    RHS_2 = w*N
    print("Aggregated BC (including the government BC) (2 with GBC)", LHS_2 - RHS_2)

    # Labor clearing checks
    #print("Labor clearing (market):", w*N)
    #print("Labor clearing (core+d):", w*(ss['N_core'] + np.sum(ss['N_d'])))
    #print("Labor clearing (prod fn):", w * (Y_core / ss['Z_core'] + np.sum(Y_d / (mu_Z_d * np.mean(ss['Z_d'])))))
    #print("Labor clearing (pricing eq):", w*(p_core * Y_core / w + np.sum(p_d * Y_d / w)))

    # Final aggregate resource constraint (3)
    LHS_3 = p_core*C_core \
        + np.sum((1+eps_vec) * p_E * C_E) \
        + np.sum( X_plus  - (1-chi)*X_minus) \
        + G
    RHS_3 = p_core * Y

    print("Final resource constraint difference (LHS - RHS) (3 Final Resource Cstrt):", LHS_3 - RHS_3)
    #print("Final resource constraint, LHS:", LHS_3)
    
    #print("Final resource constraint, LHS_1:",  p_core*C_core )
    #print("Final resource constraint, LHS_2:",  np.sum((1+eps_vec) * p_E * C_E))
    #print("Final resource constraint, LHS_3:", np.sum(X_plus * p_d))
    #print("Final resource constraint, LHS_4:", G)

    
    #print("Final resource constraint, RHS:", RHS_3)
    results["resource_constraint"] = LHS_3 - RHS_3
    
    #Other market clearing conditions
    # --- (2) Labor clearing ---
    N = ss['N']
    N_Y = ss['N_Y']
    results["labor_clearing"] = N - N_Y

    # --- (3) Asset clearing ---
    A = ss['A']
    B = ss['B']
    results["asset_clearing"] = A - B

    print("\n=== MARKET CLEARING SUMMARY ===")
    print(f"{'Condition':<30} {'Value (LHS - RHS)':>20} {'Status':>12}")
    print("-" * 65)

    def status(x, tol=1e-4):
        return "✅ OK" if abs(x) < tol else "⚠️  FAIL"

    print(f"{'Resource Constraint':<30} {results['resource_constraint']:>20.4e} {status(results['resource_constraint']):>12}")
    print(f"{'Labor Clearing':<30} {results['labor_clearing']:>20.4e} {status(results['labor_clearing']):>12}")
    print(f"{'Asset Clearing':<30} {results['asset_clearing']:>20.4e} {status(results['asset_clearing']):>12}")

    print("=" * 65)
    
    #return LHS_3 - RHS_3


def comparative_statics_plot(ha, ss_base, param_grid, unknowns_ss, targets_ss, outputs, 
                             solver='hybr', plot_deviation=True, save_path=None):
    """
    Run comparative statics over a grid of parameter values and plot 
    how outputs vary relative to the baseline, one subplot per output.
    Handles solver failures by interpolating/extrapolating and marking them.
    """

    # --- Check param_grid ---
    if len(param_grid) != 1:
        raise NotImplementedError("Plotting supports only one varied parameter at a time.")
    param, grid = list(param_grid.items())[0]

    # --- Baseline values ---
    baseline_values = {key: ss_base[key] for key in outputs}

    # --- Storage ---
    results = {key: [] for key in outputs}
    success_flags = []  # Track which points succeeded

    # --- Runtime estimate ---
    n_points = len(grid)
    print(f"Running comparative statics for parameter '{param}' with {n_points} points...")
    print("Estimating runtime with one trial solve...")
    calib_test = deepcopy(ss_base)
    calib_test[param] = grid[0]
    t0 = time.time()
    try:
        ha.solve_steady_state(calib_test, unknowns_ss, targets_ss, solver=solver)
    except Exception:
        print("⚠️ Warning: first trial solve failed, runtime estimate may be unreliable.")
        t1 = t0 + 0.01
    else:
        t1 = time.time()
    est_total = (t1 - t0) * n_points
    print(f"Estimated runtime: ~{est_total:.1f} seconds ({est_total/60:.1f} minutes)")

    # --- Loop over grid ---
    for i, val in enumerate(grid, start=1):
        calib_dev = deepcopy(ss_base)
        calib_dev[param] = val
        try:
            ss_dev = ha.solve_steady_state(calib_dev, unknowns_ss, targets_ss, solver=solver)
            for key in outputs:
                results[key].append(ss_dev[key])
            success_flags.append(True)
            print(f"Solved {i}/{n_points} steady states for {param} = {val:.3f}")
        except Exception as e:
            for key in outputs:
                results[key].append(np.nan)  # placeholder for failed solve
            success_flags.append(False)
            print(f"❌ Failed to solve {i}/{n_points} steady states for {param} = {val:.3f} ({e})")

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    # --- Interpolate missing values ---
    results_interp = {}
    x = np.array(grid)
    for key in outputs:
        y = results[key]
        mask = ~np.isnan(y)
        if mask.sum() >= 2:  # need at least 2 points for interpolation
            y_interp = np.interp(x, x[mask], y[mask])
        else:
            y_interp = y  # leave as is if too few valid points
        results_interp[key] = y_interp

    # --- Plot: subplots, one per output ---
    n_outputs = len(outputs)
    ncols = 2 if n_outputs > 1 else 1
    nrows = int(np.ceil(n_outputs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)

    for idx, key in enumerate(outputs):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        if plot_deviation:
            y_values = results_interp[key] - baseline_values[key]
            ylabel = f"{key} deviation"
        else:
            y_values = results_interp[key]
            ylabel = f"{key} level"

        # Plot interpolated curve
        ax.plot(grid, y_values, marker='o', label='Interpolated')

        # Mark failures with red crosses
        failed_idx = np.where(~np.array(success_flags))[0]
        if failed_idx.size > 0:
            ax.scatter(np.array(grid)[failed_idx],
                       y_values[failed_idx],
                       color='red', marker='x', s=80, label='Failed solve')

        if plot_deviation:
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

        ax.set_xlabel(param)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{key} vs {param}")
        ax.legend()

    # Remove any empty subplots
    for idx in range(len(outputs), nrows*ncols):
        r, c = divmod(idx, ncols)
        fig.delaxes(axes[r, c])

    fig.suptitle(f"Comparative Statics: varying {param}", fontsize=14, y=1.02)
    
    # Save if requested
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    fig.tight_layout()
    plt.show()

    return results, results_interp, success_flags


def comparative_statics_plot_shares(
    ha, ss_base, param_grid, unknowns_ss, targets_ss, outputs,
    solver='hybr', save_path=None,
    line_labels=None,        # custom labels for outputs
    x_label=None,            # custom x-axis label
    title=None,              # custom plot title
    x_values=None,           # custom x-axis values instead of param values
    line_colors=None         # custom colors for each stacked area
):
    """
    Run comparative statics over a grid of parameter values and plot
    the outputs as shares (summing to 1). Produces a single stacked area plot.
    """

    if len(param_grid) != 1:
        raise NotImplementedError("Plotting supports only one varied parameter at a time.")
    param, grid = list(param_grid.items())[0]

    # --- Storage ---
    results = {key: [] for key in outputs}
    success_flags = []

    # --- Runtime estimate ---
    n_points = len(grid)
    print(f"Running comparative statics for parameter '{param}' with {n_points} points...")
    print("Estimating runtime with one trial solve...")
    calib_test = deepcopy(ss_base)
    calib_test[param] = grid[0]
    t0 = time.time()
    try:
        ha.solve_steady_state(calib_test, unknowns_ss, targets_ss, solver=solver)
    except Exception:
        print("⚠️ Warning: first trial solve failed, runtime estimate may be unreliable.")
        t1 = t0 + 0.01
    else:
        t1 = time.time()
    est_total = (t1 - t0) * n_points
    print(f"Estimated runtime: ~{est_total:.1f} seconds ({est_total/60:.1f} minutes)")

    # --- Loop over the parameter grid ---
    for i, val in enumerate(grid, start=1):
        calib_dev = deepcopy(ss_base)
        calib_dev[param] = val
        try:
            ss_dev = ha.solve_steady_state(calib_dev, unknowns_ss, targets_ss, solver=solver)
            for key in outputs:
                results[key].append(ss_dev[key])
            success_flags.append(True)
            print(f"Solved {i}/{n_points} steady states for {param} = {val:.3f}")
        except Exception as e:
            for key in outputs:
                results[key].append(np.nan)
            success_flags.append(False)
            print(f"❌ Failed {i}/{n_points} for {param} = {val:.3f} ({e})")

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    # --- Interpolate missing values ---
    results_interp = {}
    x = np.array(grid)
    for key in outputs:
        y = results[key]
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            y_interp = np.interp(x, x[mask], y[mask])
        else:
            y_interp = y
        results_interp[key] = y_interp

    # --- Normalize to shares ---
    stacked_values = np.vstack([results_interp[key] for key in outputs])
    row_sums = stacked_values.sum(axis=0)
    shares = stacked_values / row_sums  # ensures they sum to 1

    # --- Plot as stacked area ---
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # --- Default labels & colors when there are 5 outputs ---
    if len(outputs) == 5:
        if line_colors is None:
            line_colors = ["#808080", "#8B4513", "#C4A484", "#228B22", "#90EE90"]
        if line_labels is None:
            line_labels = ["None", "Brown New", "Brown Old", "Green New", "Green Old"]
            
    if len(outputs) == 3:
        if line_colors is None:
            line_colors = ["#808080", "#8B4513", "#228B22"]
        if line_labels is None:
            line_labels = ["None", "Brown", "Green"]


    # Custom labels if provided
    labels = line_labels if line_labels is not None else outputs

    ax.stackplot(
        x_values if x_values is not None else grid,
        shares,
        labels=labels,
        colors=line_colors,   # <-- new argument used here
        alpha=0.8
    )
    
    # --- Visual marker for failed solves ---
    x_plot = x_values if x_values is not None else grid
    failed = np.array(success_flags) == False

    if failed.any():
        # Plot a thin red bar at y=0 for failed points
        ax.scatter(
            x_plot[failed],
            np.zeros(sum(failed)),
            color="red",
            marker="X",
            s=60,
            label="Solve failed"
        )


    # Labels and title
    ax.set_xlabel(x_label if x_label is not None else param)
    ax.set_ylabel("Share of Population")
    ax.set_title(title if title is not None else f"Shares of outputs vs {param}")
    ax.legend(loc="center left")
    ax.set_ylim(0, 1)

    # Save if requested
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    plt.show()

    return results, results_interp, shares, success_flags

def compute_gini(D, x, plot=False, return_lorenz=False):
    """
    Compute the weighted Gini coefficient from a 3D household distribution.
    Works for variables defined on fewer dimensions (e.g. income[d,e,1]).
    
    Parameters
    ----------
    D : ndarray
        3D array (n_d, n_e, n_a), the joint distribution of households.
        Will be normalized if it does not sum to 1.
    x : ndarray
        Variable of interest (e.g. asset, income, consumption).
        Can be:
          - 1D array (len = n_a)
          - 3D array (same shape as D)
          - Lower-dimensional array broadcastable to D.shape
            (e.g. shape (n_d, n_e, 1))
    plot : bool, optional
        If True, plot the Lorenz curve.
    return_lorenz : bool, optional
        If True, also return Lorenz curve data.
        
    Returns
    -------
    gini : float
        Weighted Gini coefficient.
    (cumw, cumxw) : tuple of arrays, optional
        Only if return_lorenz=True.
    """
    # --- Check normalization ---
    D_sum = np.sum(D)
    if not np.isclose(D_sum, 1.0, atol=1e-6):
        warnings.warn(f"Distribution does not sum to 1 (sum = {D_sum:.6f}). Normalizing automatically.")
        D = D / D_sum

    # --- Try to broadcast x to D's shape ---
    try:
        x = np.broadcast_to(x, D.shape)
    except ValueError:
        raise ValueError(f"x with shape {x.shape} cannot be broadcast to D's shape {D.shape}")

    # --- Flatten and clean ---
    values = x.ravel()
    weights = D.ravel()
    mask = np.isfinite(values) & np.isfinite(weights)
    values, weights = values[mask], weights[mask]
    weights /= np.sum(weights)

    # --- Sort and compute Lorenz curve ---
    sorted_idx = np.argsort(values)
    x_sorted = values[sorted_idx]
    w_sorted = weights[sorted_idx]
    cumw = np.cumsum(w_sorted)
    cumxw = np.cumsum(x_sorted * w_sorted)
    cumw /= cumw[-1]
    cumxw /= cumxw[-1]

    # --- Gini ---
    B = np.trapezoid(cumxw, cumw)
    gini = 1 - 2 * B

    # --- Plot ---
    if plot:
        plt.figure(figsize=(5,5))
        plt.plot(cumw, cumxw, label='Lorenz Curve', lw=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Equality Line')
        plt.fill_between(cumw, cumxw, cumw, color='gray', alpha=0.3)
        plt.xlabel('Cumulative Population Share')
        plt.ylabel('Cumulative Share of Variable')
        plt.title(f'Lorenz Curve (Gini = {gini:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    if return_lorenz:
        return gini, (cumw, cumxw)
    return gini


def plot_durable_choice_shares(
    SS_object,
    truncate_at=None,          # e.g., 5 → keep up to 5×average wealth
    title="Durable Choice Shares by Normalized Wealth Level",
    save_path=None             # str or None, path to save the figure
):
    """
    Plot, for each normalized wealth level, the fraction of households
    choosing each durable type. Shares sum to 1 at each wealth point.

    Default labels and colors are applied.

    Parameters
    ----------
    SS_object : object
        Must contain:
        - SS_object.internals['hh']['consav']['D'] : distribution (choice, durable, prod, assets)
        - SS_object.internals['hh']['a_grid'] : asset grid
        - SS_object['A'] : average wealth

    truncate_at : float or None
        If given, keep all a <= truncate_at * average wealth.

    title : str
        Plot title.

    save_path : str or None
        File path to save the figure. If None, figure is not saved.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # Fixed labels & colors as default
    fixed_colors = {
        0: "#808080",   # None
        1: "#8B4513",   # New Brown
        2: "#C4A484",   # Old Brown
        3: "#228B22",   # New Green
        4: "#90EE90"    # Old Green
    }

    fixed_labels = {
        0: "None",
        1: "New Brown",
        2: "Old Brown",
        3: "New Green",
        4: "Old Green"
    }

    # Load distribution and asset grid
    D = np.asarray(SS_object.internals['hh']['consav']['D'])  # (choice, durable, prod, assets)
    a_grid = np.asarray(SS_object.internals['hh']['a_grid'])
    assert D.ndim == 4, "D must be 4D: (choice, durable, prod, assets)"

    # 1. Compute actual average wealth
    mass = D.sum(axis=(0,1,2))  # total mass by assets
    A_actual = np.sum(mass * a_grid)

    # 2. Collapse over choice & productivity → (durable, assets)
    M = D.sum(axis=(0, 2))

    # 3. Truncate tail if requested
    if truncate_at is not None:
        cutoff_val = truncate_at * A_actual
        cutoff_idx = np.searchsorted(a_grid, cutoff_val, side='right') - 1
        cutoff_idx = max(1, cutoff_idx)
        M = M[:, :cutoff_idx+1]
        a_grid = a_grid[:cutoff_idx+1]

    # 4. Normalize wealth
    x_axis = a_grid / A_actual

    # 5. Convert counts to shares at each wealth level
    totals = M.sum(axis=0)
    totals[totals == 0] = np.nan
    shares = M / totals[None, :]

    Nd = shares.shape[0]

    # 6. Plot
    plt.figure(figsize=(10,6))

    for d in range(Nd):
        color = fixed_colors.get(d, None)
        label = fixed_labels.get(d, f"durable {d}")
        plt.plot(x_axis, shares[d], label=label, color=color, linewidth=2)

    plt.xlabel("Wealth / Average Wealth")
    plt.ylabel("Share choosing durable (sums to 1 at each wealth level)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 7. Save figure if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
