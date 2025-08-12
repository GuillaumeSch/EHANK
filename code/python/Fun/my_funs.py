import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Math
from copy import deepcopy



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

def analyze_steady_state(param_name, param_values, cali, hh, n_d):
    """
    Vary a calibration parameter and plot steady-state outcomes for DD_i and DD_TILDE_i, A, and C.

    Args:
        param_name (str): Name of the parameter in cali['baseline'] to vary.
        param_values (array-like): Grid of values to assign to the parameter.
        cali (dict): Dictionary containing the baseline calibration.
        hh (module/object): Object with a method `steady_state(cali_dict)`.
        n_d (int): Number of durable states (e.g. 1 + n_b + n_g)

    Returns:
        dict: Dictionary with raw and interpolated results for DD_i, A, and C.
    """
    # Store DD and DD_TILDE as list of lists, one per vintage i
    dd_vals = [[] for _ in range(n_d)]
    dd_tilde_vals = [[] for _ in range(n_d)]
    a_vals = []
    c_vals = []
    success_flags = []

    for val in param_values:
        cali_try = cali['baseline'].copy()
        cali_try[param_name] = val
        try:
            ss_try = hh.steady_state(cali_try)

            for i in range(n_d):
                dd_vals[i].append(ss_try[f'DD_{i}'])
                dd_tilde_vals[i].append(ss_try[f'DD_TILDE_{i}'])

            a_vals.append(ss_try['A'])
            c_vals.append(ss_try['C'])
            success_flags.append(True)
            print(f"SS found for {param_name} = {val:.3f}!")
        except Exception as e:
            print(f"Failed for {param_name} = {val:.3f}: {e}")
            for i in range(n_d):
                dd_vals[i].append(np.nan)
                dd_tilde_vals[i].append(np.nan)
            a_vals.append(np.nan)
            c_vals.append(np.nan)
            success_flags.append(False)

    # Convert to arrays
    param_values = np.array(param_values)
    dd_vals = [np.array(v) for v in dd_vals]
    dd_tilde_vals = [np.array(v) for v in dd_tilde_vals]
    a_vals = np.array(a_vals)
    c_vals = np.array(c_vals)

    # Plotting DD_i and DD_TILDE_i
    fig, axs = plt.subplots(1, n_d, figsize=(3 * n_d, 2), sharex=True)
    if n_d == 1:
        axs = [axs]  # ensure list-like even for 1 subplot

    for i in range(n_d):
        mask_dd = ~np.isnan(dd_vals[i])
        mask_dd_tilde = ~np.isnan(dd_tilde_vals[i])

        # Interpolate missing values
        interp_dd = interp1d(param_values[mask_dd], dd_vals[i][mask_dd], kind='linear', fill_value="extrapolate")
        interp_dd_tilde = interp1d(param_values[mask_dd_tilde], dd_tilde_vals[i][mask_dd_tilde], kind='linear', fill_value="extrapolate")
        dd_vals_interp = dd_vals[i].copy()
        dd_tilde_vals_interp = dd_tilde_vals[i].copy()
        dd_vals_interp[~mask_dd] = interp_dd(param_values[~mask_dd])
        dd_tilde_vals_interp[~mask_dd_tilde] = interp_dd_tilde(param_values[~mask_dd_tilde])

        # Plot both
        axs[i].plot(param_values, dd_vals_interp, '--', color='blue', label=f'DD_{i}')
        axs[i].plot(param_values, dd_tilde_vals_interp, '--', color='green', label=f'DD_TILDE_{i}')
        axs[i].plot(param_values[mask_dd], dd_vals[i][mask_dd], 'o', color='blue')
        axs[i].plot(param_values[mask_dd_tilde], dd_tilde_vals[i][mask_dd_tilde], 's', color='green')
        axs[i].set_title(f'Durable state {i}')
        axs[i].set_xlabel(param_name)
        axs[i].set_ylim(0, 1)
        axs[i].grid(True)
        axs[i].legend()

    axs[0].set_ylabel("Durable Ownership Share")

    # A and C
    fig_ac, axs_ac = plt.subplots(1, 2, figsize=(8, 3), sharex=True)

    # A
    mask_a = ~np.isnan(a_vals)
    interp_a = interp1d(param_values[mask_a], a_vals[mask_a], kind='linear', fill_value="extrapolate")
    a_vals_interp = a_vals.copy()
    a_vals_interp[~mask_a] = interp_a(param_values[~mask_a])
    axs_ac[0].plot(param_values, a_vals_interp, '--', color='gray')
    axs_ac[0].plot(param_values[mask_a], a_vals[mask_a], 'o', color='blue')
    axs_ac[0].plot(param_values[~mask_a], a_vals_interp[~mask_a], 'x', color='red')
    axs_ac[0].set_ylabel("A (Assets)")
    axs_ac[0].set_title("Steady-state A")

    # C
    mask_c = ~np.isnan(c_vals)
    interp_c = interp1d(param_values[mask_c], c_vals[mask_c], kind='linear', fill_value="extrapolate")
    c_vals_interp = c_vals.copy()
    c_vals_interp[~mask_c] = interp_c(param_values[~mask_c])
    axs_ac[1].plot(param_values, c_vals_interp, '--', color='gray')
    axs_ac[1].plot(param_values[mask_c], c_vals[mask_c], 'o', color='blue')
    axs_ac[1].plot(param_values[~mask_c], c_vals_interp[~mask_c], 'x', color='red')
    axs_ac[1].set_ylabel("C (Consumption)")
    axs_ac[1].set_title("Steady-state C")

    for ax in axs_ac:
        ax.set_xlabel(param_name)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    return {
        'param_values': param_values,
        'dd_vals': dd_vals,
        'dd_tilde_vals': dd_tilde_vals,
        'a_vals': a_vals,
        'c_vals': c_vals,
        'success_flags': success_flags,
    }

def analyze_steady_state_3d(param1, values1, param2, values2, cali, hh, n_d, results=None):
    """
    Vary two calibration parameters and plot steady-state outcomes:
    - One big figure for DD_k and DD_TILDE_k for each durable choice (vintage).
    - One second figure for A and C.

    Args:
        param1 (str): First parameter name.
        values1 (array-like): Grid for the first parameter.
        param2 (str): Second parameter name.
        values2 (array-like): Grid for the second parameter.
        cali (dict): Calibration dictionary with 'baseline'.
        hh (object): Must have method `steady_state(cali_dict)`.
        n_d (int): Number of durable choices (vintages).
        results (dict, optional): Previous results object to reuse.

    Returns:
        dict: Grid and results for DD_k, A, and C.
    """

    values1 = np.array(values1)
    values2 = np.array(values2)
    X, Y = np.meshgrid(values1, values2, indexing='ij')

    if results is None:
        DD_all = [np.full_like(X, np.nan, dtype=float) for _ in range(n_d)]
        DD_tilde_all = [np.full_like(X, np.nan, dtype=float) for _ in range(n_d)]
        A = np.full_like(X, np.nan, dtype=float)
        C = np.full_like(X, np.nan, dtype=float)
        success = np.full_like(X, False, dtype=bool)

        for i, v1 in enumerate(values1):
            for j, v2 in enumerate(values2):
                cali_try = cali['baseline'].copy()
                cali_try[param1] = v1
                cali_try[param2] = v2
                try:
                    ss_try = hh.steady_state(cali_try)
                    for k in range(n_d):
                        DD_all[k][i, j] = ss_try[f'DD_{k}']
                        DD_tilde_all[k][i, j] = ss_try[f'DD_TILDE_{k}']
                    A[i, j] = ss_try['A']
                    C[i, j] = ss_try['C']
                    success[i, j] = True
                    print(f"Success: {param1}={v1:.2f}, {param2}={v2:.2f}")
                except Exception as e:
                    print(f"Fail: {param1}={v1:.2f}, {param2}={v2:.2f} | {e}")
    else:
        DD_all = results['DD_all']
        DD_tilde_all = results['DD_tilde_all']
        A = results['A']
        C = results['C']
        success = results['success']

    # === Interpolation ===
    def interpolate_missing(Z):
        points = np.column_stack((X[~np.isnan(Z)], Y[~np.isnan(Z)]))
        values = Z[~np.isnan(Z)]
        return griddata(points, values, (X, Y), method='linear')

    DD_interp_all = [interpolate_missing(DD) for DD in DD_all]
    DD_tilde_interp_all = [interpolate_missing(DD_tilde) for DD_tilde in DD_tilde_all]
    A_interp = interpolate_missing(A)
    C_interp = interpolate_missing(C)

    # === First figure: DD and DD_TILDE by vintage ===
    ncols = 4
    nrows = int(np.ceil(n_d / ncols))
    #fig1 = plt.figure(figsize=(6 * ncols, 5 * nrows))
    fig1 = plt.figure(figsize=(18,9))


    for k in range(n_d):
        ax = fig1.add_subplot(nrows, ncols, k + 1, projection='3d')

        ax.plot_surface(X, Y, DD_interp_all[k], cmap='cividis', alpha=0.60, edgecolor='none')
        ax.plot_surface(X, Y, DD_tilde_interp_all[k], cmap='viridis', alpha=0.60, edgecolor='none')
        ax.scatter(X[success], Y[success], DD_all[k][success], color='blue', s=10, label='DD')
        ax.scatter(X[success], Y[success], DD_tilde_all[k][success], color='green', s=10, label='DD_TILDE')

        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_zlabel(f'DD / DD_TILDE')
        ax.set_title(f'Durable state {k}')
        if k == 0:
            ax.legend()

    plt.tight_layout()
    plt.show()

    # === Second figure: A and C ===
    #fig2 = plt.figure(figsize=(12, 6))
    fig2 = plt.figure(figsize=(8, 4))

    ax1 = fig2.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, A_interp, cmap='Oranges', alpha=0.7, edgecolor='none')
    ax1.scatter(X[success], Y[success], A[success], color='darkorange', s=10)
    ax1.set_xlabel(param1)
    ax1.set_ylabel(param2)
    ax1.set_zlabel('A')
    ax1.set_title('Assets (A)')

    ax2 = fig2.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, C_interp, cmap='Purples', alpha=0.7, edgecolor='none')
    ax2.scatter(X[success], Y[success], C[success], color='indigo', s=10)
    ax2.set_xlabel(param1)
    ax2.set_ylabel(param2)
    ax2.set_zlabel('C')
    ax2.set_title('Consumption (C)')

    plt.tight_layout()
    plt.show()

    return {
        'X': X, 'Y': Y,
        'DD_all': DD_all,
        'DD_tilde_all': DD_tilde_all,
        'DD_interp_all': DD_interp_all,
        'DD_tilde_interp_all': DD_tilde_interp_all,
        'A': A, 'C': C,
        'A_interp': A_interp,
        'C_interp': C_interp,
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

def plot_linear_irfs(shocks_list, unknowns_td, targets_td, ha, ss, outputs,
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
    show_irfs([irfs], outputs, labels=labels, ylabel=ylabel, T_plot=T, figsize=figsize)

def evaluate_param_changes(param_name, values_list, ha, cali):
    # Get baseline calibration and steady state
    baseline_calib = deepcopy(cali)
    baseline_ss = ha.steady_state(baseline_calib)

    # Variables to report in SS output
    ss_vars = ['goods_mkt', 'asset_mkt', 'Tax', 'r', 'beta', 'G', 'B', 'N', 'Y', 'Z']

    # Store results
    calibration_results = []
    ss_results = []

    # Baseline row
    calibration_results.append(('Baseline', baseline_calib.get(param_name, 'N/A')))
    ss_results.append(('Baseline', [baseline_ss[v] if v in baseline_ss else 'N/A' for v in ss_vars]))

    # Loop over alternative values
    for val in values_list:
        # Create new calibration
        modified_calib = deepcopy(baseline_calib)
        modified_calib[param_name] = val

        # Compute steady state
        ss = ha.steady_state(modified_calib)

        # Store results
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