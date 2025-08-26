import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Math
from copy import deepcopy
from scipy.interpolate import interp1d, griddata
import colorsys
import time




#Function to vizualize policy function
def policy_functions(
    ss,
    amax=150,
    amin=0,
    d_tilde_list=[0],
    d_list=[0],
    ie_list=[3],
    figsize=0.6,
    models=['baseline']
):
    a_grid = ss['baseline'].internals['hh']['a_grid']

    a, da, c, P, V = dict(), dict(), dict(), dict(), dict()

    for model in models:
        a[model] = ss[model].internals['hh']['consav']['a']
        da[model] = a[model] - a_grid
        c[model] = ss[model].internals['hh']['consav']['c']
        P[model] = ss[model].internals['hh']['durables']['law_of_motion'].P
        V[model] = ss[model].internals['hh']['durables']['V']

    fig, axes = plt.subplots(1, 3, figsize=(12 * figsize, 4 * figsize))
    ax = axes.flatten()

    # Define line styles for each iz (cycle if fewer styles than ie_list)
    linestyles = ['-', '--', '-.', ':']
    linestyle_map = {
        iz: linestyles[i % len(linestyles)]
        for i, iz in enumerate(ie_list)
    }

    n = len(d_tilde_list)
    color_map = {}
    for i, d_tilde in enumerate(d_tilde_list):
        hue = i / n  # equally spaced hues on color wheel
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)  # full saturation and brightness
        color_map[d_tilde] = (r, g, b)  # matplotlib accepts RGB tuples (0-1 range)

    # Define alpha values for d_list, normalized between 0.3 and 1 for visibility
    if len(d_list) > 1:
        alphas = np.linspace(0.3, 1.0, len(d_list))
    else:
        alphas = [1.0]
    alpha_map = {
        d_val: alpha
        for d_val, alpha in zip(d_list, alphas)
    }

    # Plot
    for model in models:
        for d_tilde in d_tilde_list:
            for d in d_list:
                for iz in ie_list:
                    linestyle = linestyle_map[iz]
                    color = color_map[d_tilde]
                    alpha = alpha_map[d]

                    label = f"{model} ($\\tilde{{d}}$={d_tilde}, d={d}, z={iz})"

                    # Asset policy function
                    ax[0].plot(
                        a_grid[:amax],
                        a[model][d_tilde, d, iz, :amax],
                        label=label,
                        linewidth=2,
                        color=color,
                        linestyle=linestyle,
                        alpha=alpha
                    )

                    # Consumption policy function
                    ax[1].plot(
                        a_grid[:amax],
                        c[model][d_tilde, d, iz, :amax],
                        label=label,
                        linewidth=2,
                        color=color,
                        linestyle=linestyle,
                        alpha=alpha
                    )

                    # Discrete choice probability
                    ax[2].plot(
                        a_grid[amin:amax],
                        P[model][d_tilde, d, iz, amin:amax],
                        label=label,
                        linewidth=2,
                        color=color,
                        linestyle=linestyle,
                        alpha=alpha
                    )

    # Reference lines for assets plot
    ax[0].plot(a_grid[:amax], a_grid[:amax], color='gray', linestyle=':')
    ax[0].axhline(0, color='gray', linestyle=':')

    # Titles
    ax[0].set_title(r'Assets ($a^*(\tilde{d},\, d,\, z,\, a^{-})$)')
    ax[1].set_title(r'Consumption ($c^*(\tilde{d},\, d,\, z,\, a^{-})$)')
    ax[2].set_title(r'Discrete choice ($Pr(\tilde{d}^*(d,\, z,\, a^{-})=1)$)')

    for axis in ax:
        axis.set_xlabel('assets')
        axis.legend(frameon=False)

    plt.tight_layout()
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


# Define the plotting function
def show_irfs(irfs_list, variables, labels=[" "], ylabel=r"Percentage points (dev. from ss)", T_plot=50, figsize=(18, 6)):
    if len(irfs_list) != len(labels):
        labels = [" "] * len(irfs_list)

    n_var = len(variables)
    fig, ax = plt.subplots(1, n_var, figsize=figsize, sharex=True)
    if n_var == 1:
        ax = [ax]  # Ensure ax is iterable

    for i in range(n_var):
        var = variables[i]

        for j, irf in enumerate(irfs_list):
            if var in irf:
                data = 100 * np.array(irf[var][:T_plot])
            else:
                data = np.zeros(T_plot)
            ax[i].plot(data, label=labels[j])

        ax[i].set_title(var)
        ax[i].set_xlabel(r"$t$")
        if i == 0:
            ax[i].set_ylabel(ylabel)
        ax[i].legend()

    plt.tight_layout()
    plt.show()

def plot_linear_irfs(shocks_list, unknowns_td, targets_td, ha, ss, outputs, T_plot=50,
              rho=None, e=None, T=300, figsize=(18, 3), ylabel=r"Percentage points (dev. from ss)", labels=None):
    # Default values if not provided
    if rho is None:
        rho = {shock: 0.8 for shock in shocks_list}
    if e is None:
        e = {shock: 0.01 for shock in shocks_list}
    # Build shocks dictionary with time series
    shocks = {
        shock: e[shock] * rho[shock] ** np.arange(T)
        for shock in shocks_list
    }
    # Solve the system
    irfs = ha.solve_impulse_linear(ss, unknowns_td, targets_td, shocks)
    # Default label
    if labels is None:
        labels = [" + ".join(shocks_list)]
    # Plot
    show_irfs([irfs], outputs, labels=labels, ylabel=ylabel, T_plot=T_plot, figsize=figsize)

def evaluate_param_changes(param_name, values_list, ha, cali,
                           ss_vars=['goods_mkt', 'asset_mkt', 'Tax', 'r', 'beta', 'G', 'B', 'N', 'Y', 'Z']):
     # Get baseline calibration and steady state
    baseline_calib = deepcopy(cali)
    baseline_ss = ha.steady_state(baseline_calib)

    # Store results
    calibration_results = []
    ss_results = []

    # Baseline row
    try:
        baseline_value = (
            baseline_calib[param_name]
            if param_name in baseline_calib
            else 'N/A'
        )
    except TypeError:
        try:
            baseline_value = baseline_calib[param_name]
        except KeyError:
            baseline_value = 'N/A'

    calibration_results.append(('Baseline', baseline_value))
    ss_results.append(('Baseline', [baseline_ss[v] if v in baseline_ss else 'N/A' for v in ss_vars]))

    # Loop over alternative values
    for val in values_list:
        modified_calib = deepcopy(baseline_calib)
        modified_calib[param_name] = val
        ss = ha.steady_state(modified_calib)

        case_name = f"{param_name} = {val}"
        calibration_results.append((case_name, val))
        ss_results.append((case_name, [ss[v] if v in ss else 'N/A' for v in ss_vars]))

    # Print Calibration Table
    print(f"\n🔧 Parameter sweep for '{param_name}'")
    print("\n📌 Calibration Values:")
    print(f"{'Case':<20} | {param_name}")
    print("-" * 35)
    for case, val in calibration_results:
        print(f"{case:<20} | {val:.5f}" if isinstance(val, (float, int)) else f"{case:<20} | {val}")

    # Print SS Table
    print("\n📈 Steady-State Outcomes:")
    header = f"{'Case':<20} | " + " | ".join([f"{v:<10}" for v in ss_vars])
    print(header)
    print("-" * len(header))
    for case, row in ss_results:
        row_str = " | ".join(
            [f"{x:>10.5f}" if isinstance(x, (float, int)) else f"{x:>10}" for x in row]
        )
        print(f"{case:<20} | {row_str}")

#Display the aggregates for durables given a SS.
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



def plot_distribution(
    SS_object,
    lines_dim=None,       # dimension to split lines (0,1,2), or None
    truncate_at=None,     # float, asset level at which to truncate
    labels=None           # list of labels
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
        If specified, all asset levels >= truncate_at are combined in last bin
    labels : list or None
        List of labels for the lines_dim
    """

    D = np.asarray(SS_object.internals['hh']['consav']['D'])
    a_grid = np.asarray(SS_object.internals['hh']['a_grid'])
    assert D.ndim == 4, "D must be 4D (choice, durable state, productivity, assets)"

    # Determine lines
    if lines_dim is None:
        lines = [D.sum(axis=(0,1,2))]  # marginalize all except assets
        line_labels = ["Total"]
    else:
        axes_to_sum = tuple(ax for ax in range(4) if ax not in (3, lines_dim))
        lines = D.sum(axis=axes_to_sum)  # KEEP your working line

        # Ensure lines is iterable along the lines_dim
        if lines_dim != 3:  # assets is 3rd dim
            lines = [lines[i, :] for i in range(lines.shape[0])]
        else:
            lines = [lines[i] for i in range(lines.shape[0])]

        # Set labels
        if labels is None:
            line_labels = [f"{lines_dim}={i}" for i in range(len(lines))]
        else:
            line_labels = labels
            assert len(line_labels) == len(lines), "Number of labels must match number of lines"

    # Truncate assets if requested
    if truncate_at is not None:
        # Find the index corresponding to the truncation asset level
        truncate_idx = np.searchsorted(a_grid, truncate_at, side='right') - 1
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
        x_axis = a_grid[:truncate_idx+1]
    else:
        x_axis = a_grid

    # Normalize each line
    lines = [line / line.sum() for line in lines]

    # Plot
    plt.figure(figsize=(8,5))
    for line, lbl in zip(lines, line_labels):
        plt.plot(x_axis, line, marker='o', label=lbl)

    plt.xlabel("Assets")
    plt.ylabel("Probability (conditional)")
    if lines_dim is not None:
        plt.legend()
    plt.title("Distribution along assets")
    plt.grid(True)
    plt.show()

def check_resource_constraint(ss):
    """
    Production version: check key equilibrium conditions.
    Checks:
      - Aggregate resource constraint
      - Labor market clearing
      - Asset market clearing
    Returns a dictionary with differences.
    """
    results = {}

    # Household objects
    D = ss.internals['hh']['consav']['D']
    c_core = ss.internals['hh']['consav']['c_core']
    C_core = np.sum(c_core * D)
    c_E = ss.internals['hh']['consav']['c_E']
    C_E = np.sum(c_E * D, axis=(1,2,3))

    # Prices
    p_core = ss['p_core']
    p_E = ss.internals['hh']['p_e']
    p_d = ss.internals['hh']['p_d']

    # Adjustment matrix: aggregate inflows/outflows
    S = np.sum(D, axis=(2, 3)) 
    X_plus = np.sum(S * (1 - np.eye(S.shape[0])), axis=1)
    X_minus = np.sum(S * (1 - np.eye(S.shape[0])), axis=0)

    # Other aggregates
    Y_core = ss['Y_core']
    #Y_d = ss['Y_d']
    Y_d = ss.internals['hh']['Y_d']
    G = ss['G']
    chi = ss['chi']
    tau_b = ss['tau_b']

    # --- (1) Aggregate resource constraint ---
    LHS = p_core * C_core \
        + np.sum((1+tau_b)**(-1) * p_E * C_E) \
        + np.sum(X_plus * p_d) \
        + G
    RHS = p_core * Y_core \
        + np.sum(p_d * (Y_d + (1-chi) * X_minus))
    results["resource_constraint"] = LHS - RHS

    # --- (2) Labor clearing ---
    N = ss['N']
    N_core = ss['N_core']
    N_d = np.sum(ss['N_d'])
    results["labor_clearing"] = N - (N_core + N_d)

    # --- (3) Asset clearing ---
    A = ss['A']
    B = ss['B']
    results["asset_clearing"] = A - B

    # Print summary
    print("Resource constraint difference (LHS - RHS):", results["resource_constraint"])
    print("Labor clearing difference (N - (N_core + sum(N_d))):", results["labor_clearing"])
    print("Asset clearing difference (A - B):", results["asset_clearing"])

    #return results


def check_resource_constraint_debug(ss):
    """
    Debug version: compute and print intermediate steps as well as the final RC.
    Useful for diagnosing inconsistencies.
    """
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

    p_core = ss['p_core']
    c_core = ss.internals['hh']['consav']['c_core']
    C_core = np.sum(c_core * D)
    c_E = ss.internals['hh']['consav']['c_E']
    C_E = np.sum(c_E * D, axis=(1,2,3))
    p_E = ss.internals['hh']['p_e']
    p_d = ss.internals['hh']['p_d']
    Y_core = ss['Y_core']
    Y_d = ss['Y_d']
    G = ss['G']
    chi = ss['chi']
    tau_b = ss['tau_b']
    mu_Z_d = ss['mu_Z_d']

    # Individual BC
    lhs = p_bundle[...,np.newaxis,np.newaxis,np.newaxis] * c \
        + a_choice \
        + adj_matrix[..., np.newaxis,np.newaxis]
    rhs = ((1 + r) * a_grid)[np.newaxis,np.newaxis,np.newaxis,...] \
        + w * N * e_grid[np.newaxis, np.newaxis, :, np.newaxis] \
        + T[np.newaxis, np.newaxis, :, np.newaxis]
    print("Max individual BC deviation:", np.max(np.abs(lhs - rhs)))

    # Aggregated BC
    LHS_0 = np.sum(lhs * D)
    RHS_0 = np.sum(rhs * D)
    print("Aggregating the individual BC:", LHS_0 - RHS_0)

    # Get X flows
    S = np.sum(D, axis=(2, 3)) 
    X_plus = np.sum(S * (1 - np.eye(S.shape[0])), axis=1)
    X_minus = np.sum(S * (1 - np.eye(S.shape[0])), axis=0)

    # Government BC
    LHS_1 = p_core*C_core + np.sum(p_E * C_E) + np.sum(X_plus * p_d - X_minus * chi * p_d) + G
    RHS_1 = w*N
    print("Aggregated BC (with flows of durables and government BC)", LHS_1 - RHS_1)

    # Labor clearing checks
    #print("Labor clearing (market):", w*N)
    #print("Labor clearing (core+d):", w*(ss['N_core'] + np.sum(ss['N_d'])))
    #print("Labor clearing (prod fn):", w * (Y_core / ss['Z_core'] + np.sum(Y_d / (mu_Z_d * np.mean(ss['Z_d'])))))
    #print("Labor clearing (pricing eq):", w*(p_core * Y_core / w + np.sum(p_d * Y_d / w)))

    # Final aggregate resource constraint
    LHS_2 = p_core*C_core \
        + np.sum((1+tau_b)**(-1) * p_E * C_E) \
        + np.sum(X_plus * p_d) \
        + G
    RHS_2 = p_core * Y_core \
        + np.sum(p_d * (Y_d + (1-chi)*X_minus))

    print("Final resource constraint difference (LHS - RHS):", LHS_2 - RHS_2)
    return LHS_2 - RHS_2


def comparative_statics_plot(ha, ss_base, param_grid, unknowns_ss, targets_ss, outputs, 
                             solver='hybr', plot_deviation=True):
    """
    Run comparative statics over a grid of parameter values and plot 
    how outputs vary relative to the baseline, one subplot per output.

    Parameters
    ----------
    ha : object
        Model object with a method solve_steady_state(calib, unknowns, targets, solver).
    ss_base : dict
        Baseline steady state.
    param_grid : dict
        Dictionary of parameters and the grid of values to loop over.
        Example: {'B': np.linspace(2.0, 4.0, 5)}
        Only supports one parameter at a time.
    unknowns_ss, targets_ss : dict
        Inputs to ha.solve_steady_state.
    outputs : list of str
        Outputs to track and plot.
    solver : str
        Solver for steady state.
    plot_deviation : bool
        If True, plots deviation from baseline. If False, plots absolute levels.

    Returns
    -------
    results : dict
        Dictionary with arrays of output values across the grid.
    """

    # --- Check param_grid ---
    if len(param_grid) != 1:
        raise NotImplementedError("Plotting supports only one varied parameter at a time.")
    param, grid = list(param_grid.items())[0]

    # --- Baseline values ---
    baseline_values = {key: ss_base[key] for key in outputs}

    # --- Storage ---
    results = {key: [] for key in outputs}

    # --- Runtime estimate ---
    n_points = len(grid)
    print(f"Running comparative statics for parameter '{param}' with {n_points} points...")
    print("Estimating runtime with one trial solve...")
    calib_test = deepcopy(ss_base)
    calib_test[param] = grid[0]
    t0 = time.time()
    ha.solve_steady_state(calib_test, unknowns_ss, targets_ss, solver=solver)
    t1 = time.time()
    est_total = (t1 - t0) * n_points
    print(f"Estimated runtime: ~{est_total:.1f} seconds ({est_total/60:.1f} minutes)")

    # --- Loop over grid ---
    for i, val in enumerate(grid, start=1):
        calib_dev = deepcopy(ss_base)
        calib_dev[param] = val
        ss_dev = ha.solve_steady_state(calib_dev, unknowns_ss, targets_ss, solver=solver)

        for key in outputs:
            results[key].append(ss_dev[key])

        print(f"Solved {i}/{n_points} steady states for {param} = {val:.3f}")

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    # --- Plot: subplots, one per output ---
    n_outputs = len(outputs)
    ncols = 2 if n_outputs > 1 else 1
    nrows = int(np.ceil(n_outputs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows), squeeze=False)

    for idx, key in enumerate(outputs):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        if plot_deviation:
            y_values = results[key] - baseline_values[key]
            ylabel = f"{key} deviation"
        else:
            y_values = results[key]
            ylabel = f"{key} level"

        ax.plot(grid, y_values, marker='o')
        if plot_deviation:
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel(param)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{key} vs {param}")

    # Remove any empty subplots
    for idx in range(len(outputs), nrows*ncols):
        r, c = divmod(idx, ncols)
        fig.delaxes(axes[r, c])

    fig.suptitle(f"Comparative Statics: varying {param}", fontsize=14, y=1.02)
    fig.tight_layout()
    plt.show()

    return results