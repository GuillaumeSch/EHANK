#%% Import packages

# Standard libraries
import inspect
# Numerical computing
import numpy as np
from numba import njit
from scipy.interpolate import interp1d, griddata
# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# Sequence-Jacobian framework
from sequence_jacobian import grids, interpolate
from sequence_jacobian.blocks.stage_block import StageBlock
# Custom utilities
from SSJ_Fun.utils import LogitChoiceDurables, ExogenousMaker, Continuous1D_Durables


#%% Interactive plot
%matplotlib qt

#%% Some useful functions for debugging

#Function to vizualize policy function
def policy_functions(ss, amax=150, amin=0, d_tilde_list=[0], d_list=[0], iz_list=[3], figsize=0.6, models=['baseline']):
    a_grid = ss['baseline'].internals['hh']['a_grid']
    a, da, c, P, V = dict(), dict(), dict(), dict(), dict()

    for i in models:
        a[i] = ss[i].internals['hh']['consav']['a']
        da[i] = a[i] - a_grid
        c[i] = ss[i].internals['hh']['consav']['c']
        P[i] = ss[i].internals['hh']['labsup']['law_of_motion'].P
        V[i] = ss[i].internals['hh']['labsup']['V']

    fig, axes = plt.subplots(1, 3, figsize=(12 * figsize, 4 * figsize))
    ax = axes.flatten()

    # Assign unique colors to each model
    model_colors = {model: color for model, color in zip(models, plt.cm.tab10.colors)}

    # Define line styles for combinations of (d_tilde, d, iz)
    linestyles = ['-', '--', '-.', ':']
    while len(linestyles) < len(d_tilde_list) * len(d_list) * len(iz_list):
        linestyles += linestyles

    for model in models:
        color = model_colors[model]
        combos = [(d_tilde, d, iz) for d_tilde in d_tilde_list for d in d_list for iz in iz_list]
        for idx, (d_tilde, d, iz) in enumerate(combos):
            linestyle = linestyles[idx]
            label = f"{model} ($\\tilde{{d}}$={d_tilde}, d={d}, z={iz})"

            ax[0].plot(
                a_grid[:amax],
                #np.sum(P[model][:, d, iz, :amax] * a[model][:, d, iz, :amax], axis=0),
                a[model][d_tilde, d, iz, :amax],
                label=label,
                linewidth=2,
                color=color,
                linestyle=linestyle
            )

            ax[1].plot(
                a_grid[:amax],
                #np.sum(P[model][:, d, iz, :amax] * c[model][:, d, iz, :amax], axis=0),
                c[model][d_tilde, d, iz, :amax],
                label=label,
                linewidth=2,
                color=color,
                linestyle=linestyle
            )

            ax[2].plot(
                a_grid[amin:amax],
                P[model][1, d, iz, amin:amax],
                label=label,
                linewidth=2,
                color=color,
                linestyle=linestyle
            )

    ax[0].plot(a_grid[:amax], a_grid[:amax], color='gray', linestyle=':')
    ax[0].axhline(0, color='gray', linestyle=':')

    ax[0].set_title(r'Assets ($a^*(\tilde{d},\, d,\, z,\, a^{-})$)')
    ax[1].set_title(r'Consumption ($c^*(\tilde{d},\, d,\, z,\, a^{-})$)')
    ax[2].set_title(r'Discrete choice ($Pr(\tilde{d}^*(d,\, z,\, a^{-})=1$)')

    for k in ax:
        k.set_xlabel('assets')
        k.legend(frameon=False)

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

def analyze_steady_state(param_name, param_values, cali, hh):
    """
    Vary a calibration parameter and plot steady-state outcomes for DD, A, and C.

    Args:
        param_name (str): Name of the parameter in cali['baseline'] to vary.
        param_values (array-like): Grid of values to assign to the parameter.
        cali (dict): Dictionary containing the baseline calibration.
        hh (module/object): Object with a method `steady_state(cali_dict)`.

    Returns:
        dict: Dictionary with raw and interpolated results for DD, A, and C.
    """

    dd_vals = []
    dd2_vals = []
    a_vals = []
    c_vals = []
    success_flags = []

    for val in param_values:
        cali_try = cali['baseline'].copy()
        cali_try[param_name] = val
        try:
            ss_try = hh.steady_state(cali_try)
            dd_vals.append(ss_try['DD'])
            dd2_vals.append(ss_try['DD_2'])
            a_vals.append(ss_try['A'])
            c_vals.append(ss_try['C'])
            success_flags.append(True)
            print(f"SS found for {param_name} = {val:.3f}!")
        except Exception as e:
            print(f"Failed for {param_name} = {val:.3f}: {e}")
            dd_vals.append(np.nan)
            dd2_vals.append(np.nan)
            a_vals.append(np.nan)
            c_vals.append(np.nan)
            success_flags.append(False)

    # Convert to arrays
    param_values = np.array(param_values)
    dd_vals = np.array(dd_vals)
    dd2_vals = np.array(dd2_vals)
    a_vals = np.array(a_vals)
    c_vals = np.array(c_vals)

    # Interpolation
    mask_dd = ~np.isnan(dd_vals)
    mask_dd2 = ~np.isnan(dd2_vals)
    mask_a = ~np.isnan(a_vals)
    mask_c = ~np.isnan(c_vals)

    interp_dd = interp1d(param_values[mask_dd], dd_vals[mask_dd], kind='linear', fill_value="extrapolate")
    interp_dd2 = interp1d(param_values[mask_dd2], dd_vals[mask_dd2], kind='linear', fill_value="extrapolate")
    interp_a = interp1d(param_values[mask_a], a_vals[mask_a], kind='linear', fill_value="extrapolate")
    interp_c = interp1d(param_values[mask_c], c_vals[mask_c], kind='linear', fill_value="extrapolate")

    dd_vals_interp = dd_vals.copy()
    dd2_vals_interp = dd2_vals.copy()
    a_vals_interp = a_vals.copy()
    c_vals_interp = c_vals.copy()

    dd_vals_interp[~mask_dd] = interp_dd(param_values[~mask_dd])
    dd2_vals_interp[~mask_dd2] = interp_dd2(param_values[~mask_dd2])
    a_vals_interp[~mask_a] = interp_a(param_values[~mask_a])
    c_vals_interp[~mask_c] = interp_c(param_values[~mask_c])

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), sharex=True)

    # DD
    # Plot interpolated curves
    axs[0].plot(param_values, dd_vals_interp, '--', color='gray', label='DD1')
    axs[0].plot(param_values, dd2_vals_interp, '--', color='black', label='DD2')
    axs[0].plot(param_values[mask_dd], dd_vals[mask_dd], 'o', color='blue')
    axs[0].plot(param_values[~mask_dd], dd_vals_interp[~mask_dd], 'x', color='blue')
    axs[0].plot(param_values[mask_dd2], dd2_vals[mask_dd2], 's', color='green')
    axs[0].plot(param_values[~mask_dd2], dd2_vals_interp[~mask_dd2], 'x', color='green')
    # Labels and grid
    axs[0].set_ylabel('DD')
    axs[0].set_title(f'SS DD (share of durable owner) vs. {param_name}')
    axs[0].grid(True)
    # Clean legend (remove duplicates)
    handles, labels = axs[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axs[0].legend(unique.values(), unique.keys())

    # A
    axs[1].plot(param_values, a_vals_interp, '--', color='gray', label='Interpolated')
    axs[1].plot(param_values[mask_a], a_vals[mask_a], 'o', color='blue', label='Computed')
    axs[1].plot(param_values[~mask_a], a_vals_interp[~mask_a], 'x', color='red', label='Interpolation')
    axs[1].set_xlabel(f'{param_name}')
    axs[1].set_ylabel('A (Assets)')
    axs[1].set_title(f'SS A vs. {param_name}')
    axs[1].legend()
    axs[1].grid(True)

    # C
    axs[2].plot(param_values, c_vals_interp, '--', color='gray')
    axs[2].plot(param_values[mask_c], c_vals[mask_c], 'o', color='blue')
    axs[2].plot(param_values[~mask_c], c_vals_interp[~mask_c], 'x', color='red')
    axs[2].set_xlabel(f'{param_name}')
    axs[2].set_ylabel('C (Consumption)')
    axs[2].set_title(f'SS C vs. {param_name}')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    return {
        'param_values': param_values,
        'dd_vals': dd_vals,
        'a_vals': a_vals,
        'c_vals': c_vals,
        'dd_vals_interp': dd_vals_interp,
        'a_vals_interp': a_vals_interp,
        'c_vals_interp': c_vals_interp,
        'success_flags': success_flags
    }

def analyze_steady_state_3d(param1, values1, param2, values2, cali, hh, results=None):
    """
    Vary two calibration parameters and plot steady-state outcomes (DD, A, C) in 3D.
    If results are provided, skip computation and use them for plotting.

    Args:
        param1 (str): First parameter name.
        values1 (array-like): Grid for the first parameter.
        param2 (str): Second parameter name.
        values2 (array-like): Grid for the second parameter.
        cali (dict): Calibration dictionary with 'baseline'.
        hh (object): Must have method `steady_state(cali_dict)`.
        results (dict, optional): Previous results object to reuse.

    Returns:
        dict: Grid and results for DD, A, and C.
    """

    values1 = np.array(values1)
    values2 = np.array(values2)
    X, Y = np.meshgrid(values1, values2, indexing='ij')

    if results is None:
        DD = np.full_like(X, np.nan, dtype=float)
        DD2 = np.full_like(X, np.nan, dtype=float)
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
                    DD[i, j] = ss_try['DD']
                    DD2[i, j] = ss_try['DD_2']
                    A[i, j] = ss_try['A']
                    C[i, j] = ss_try['C']
                    success[i, j] = True
                    print(f"Success: {param1}={v1:.2f}, {param2}={v2:.2f}")
                except Exception as e:
                    print(f"Fail: {param1}={v1:.2f}, {param2}={v2:.2f} | {e}")
    else:
        DD = results['DD']
        DD2 = results['DD_2']
        A = results['A']
        C = results['C']
        success = results['success']

    # === Interpolation of missing values ===
    def interpolate_missing(Z):
        points = np.column_stack((X[~np.isnan(Z)], Y[~np.isnan(Z)]))
        values = Z[~np.isnan(Z)]
        return griddata(points, values, (X, Y), method='linear')

    DD_interp = interpolate_missing(DD)
    DD2_interp = interpolate_missing(DD2)
    A_interp = interpolate_missing(A)
    C_interp = interpolate_missing(C)

    # === Plotting ===
    fig = plt.figure()

    for k, (Z, Z_interp, label) in enumerate(zip([DD, DD2, A, C], [DD_interp,DD2_interp, A_interp, C_interp], ['DD', 'DD2', 'A', 'C'])):
        ax = fig.add_subplot(2, 2, k + 1, projection='3d')
        ax.plot_surface(X, Y, Z_interp, cmap='viridis', alpha=0.6, edgecolor='none')
        ax.scatter(X[success], Y[success], Z[success], color='blue', label='Computed', s=10)
        ax.scatter(X[~success], Y[~success], Z_interp[~success], color='red', label='Interpolated', s=10)
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_zlabel(label)
        ax.set_title(f'{label} vs. {param1} and {param2}')
        ax.legend()

    plt.tight_layout()
    plt.show()

    return {
        'X': X, 'Y': Y,
        'DD': DD,'DD_2': DD2, 'A': A, 'C': C,
        'DD_interp': DD_interp,'DD2_interp': DD2_interp, 'A_interp': A_interp, 'C_interp': C_interp,
        'success': success
    }


#%% Stage 1 - Productivity shock (Expected value function given initial state of individual prod. level z_)

#Initialize Stage 1a
#prod_stage = ExogenousMaker(markov_name='z_markov', index=2, name='prod')
prod_stage = ExogenousMaker(markov_name='z_markov', index=1, name='prod')


#Initialize Stage 1b
#depreciation_stage = ExogenousMaker(markov_name='d_markov', index=1, name='durable')
depreciation_stage = ExogenousMaker(markov_name='d_markov', index=0, name='durable')


#%% Stage 2 - Discrete choice (Labor participation)

#Initialize Stage 2
#`value`: name of value function, SSJ has to know which object to apply the logsum formula to
#`backward`: names of other variables that have to be propagated backward, typically this is partial value function needed for EGM in continuous choice stage
#`index`: axis of correspoinding state
#`name`: name of stage
#`taste_shock_scale`: name of $\sigma_\varepsilon$ parameter, needed for all formulas
#`f`: (optional) function that implements additive utility cost on expanded state $(n| n_-, z, a_-)$. This is useful to implement costs that depend on origin as well as destination $(n|n_-)$. Setting some costs to infinity implements constraints on discrete choice (more on this below).

#labsup_stage = LogitChoice(value='V', backward='Va', index=0, name='labsup',
#                           taste_shock_scale='taste_shock')

labsup_stage = LogitChoiceDurables(value='V', backward='Va', index=0, name='labsup',
                           taste_shock_scale='taste_shock')

#%% Stage 3 - Consumption-Savings Continuous Choice

#Discrete Choice - Endogenous Grid point Method. Performs single step of backward iteration.
def dcegm(V, Va, a_grid, disp_inc, durable_exp, y_w, r, beta, eis, gamma, eta):
    """DC-EGM algorithm"""
    # use all FOCs on endogenous grid
    W = beta * V                                                  # end-of-stage vfun
    W = np.stack([W, W], axis=0)                                  # Add first dimension to match the dimensions
    uc_endo = beta * Va                                           # envelope condition
    c_endo = uc_endo** (-eis)                                     # Euler equation
    a_endo = (c_endo[np.newaxis, ...] + a_grid[np.newaxis, np.newaxis, np.newaxis, ...] + durable_exp[..., np.newaxis,np.newaxis] - y_w[np.newaxis, ..., np.newaxis]) / (1 + r)     # budget constraint

    d_bool = np.zeros_like(a_endo)
    d_bool[1,:,:,:] = 1 #Decide to have a car (either keeping the existing car or buying)

    # interpolate with upper envelope, enforce borrowing limit
    V, c, a = upperenv(W, a_endo, disp_inc, a_grid, d_bool, eis, gamma, eta)

    # update Va on exogenous grid
    uc = c ** (-1 / eis)                                          # Euler equation
    uc = make_strictly_decreasing(uc)                             # Correct for the infinite values.
    Va = (1 + r) * uc                                             # envelope condition

    return V, Va, a, c



#Simple wrapper to make it independent of the size of the state space. Temporarily collapse states associated with all other stages into a single axis.
def upperenv(W, a_endo, disp_inc, a_grid, d_bool, *args):
    # collapse (d_tilde, d, z, a) into (b, a)
    shape = W.shape
    W = W.reshape((-1, shape[-1]))
    a_endo = a_endo.reshape((-1, shape[-1]))
    d_bool = d_bool.reshape((-1, shape[-1]))
    disp_inc = disp_inc.reshape((-1, shape[-1]))
    V, c, a = upperenv_vec(W, a_endo, disp_inc, a_grid, d_bool, *args)

    # report on (d_tilde, d, z, a)
    return V.reshape(shape), c.reshape(shape), a.reshape(shape)


#Core upper envelope step:
# Consider every segment of the endogenous grid $(a_{j}^{endo}, a_{j+1}^{endo})$ and find all the exogenous gridpoints $a^{grid}_i$ that fall into that segment.
# Interpolate there to get a candidate solution $a_i$.
# Since the endogenous grid is non-monotonic, the same point $a^{grid}_i$ may be bracketed by another segment $(a_{\tilde j}^{endo}, a_{\tilde j+1}^{endo}).$
# When this happens, we keep the solution that gives higher value.
@njit
def upperenv_vec(W, a_endo, disp_inc, a_grid, d_bool, *args):
    """Interpolate value function and consumption to exogenous grid."""
    n_b, n_a = W.shape
    a = np.zeros_like(W)
    c = np.zeros_like(W)
    V = -np.inf * np.ones_like(W)

    # loop over other states, collapsed into single axis
    for ib in range(n_b):
        #d = min(ib * 2 // n_b, 2 - 1)
        d = d_bool[ib,0]
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
def util(c, d, eis, gamma, eta = 0):
    d = np.array(d)
    if d == 1:
        if eis == 1:
            u = np.log(c) + gamma * d
        else:
            u = c ** (1 - 1 / eis) / (1 - 1 / eis) + gamma * d
    elif d == 0:
        if eis == 1:
            u = np.log(c)
        else:
            u = c ** (1 - 1 / eis) / (1 - 1 / eis)
    else:
        Warning('GSCHW: Problem in the utility function.')
    return u


#Report the aggregate demand for d
def D_demand(c):
    dd = np.zeros_like(c)
    dd_2 = np.zeros_like(c)
    dd[1, ...] = 1
    dd_2[:,1, ...] = 1
    return dd, dd_2
#Initialize Stage 3
consav_stage = Continuous1D_Durables(backward=['V', 'Va'], policy='a', f=dcegm,
                            name='consav', hetoutputs=[D_demand])

# %% Other basic necessary functions
# hh_init: function that constructs the initial guess for backward variables
def hh_init(disp_inc, a_grid, eis, gamma, eta):
    V = util(disp_inc-np.min(disp_inc)+1, 0, eis, gamma, eta) #Avoid strange behaviour due to negative values. Not too important as only for first guess.
    V = (V[0,:,:,:] + V[1,:,:,:])/2 #Get rid of first dimension
    Va = np.empty_like(V)
    Va[..., 1:-1] = (V[..., 2:] - V[..., :-2]) / (a_grid[2:] - a_grid[:-2])
    Va[..., 0] = (V[..., 1] - V[..., 0]) / (a_grid[1] - a_grid[0])
    Va[..., -1] = (V[..., -1] - V[..., -2]) / (a_grid[-1] - a_grid[-2])
    return V, Va

#construct Markov process for productivity, for depreciation of durables and the assets grid
def make_grids(rho_z, sd_z, n_z, min_a, max_a, n_a, dep_pr):
    z_grid, z_dist, z_markov = grids.markov_rouwenhorst(rho_z, sd_z, n_z)
    a_grid = grids.agrid(max_a, n_a, min_a)
    d_grid = np.array([0, 1])
    d_markov = np.array([[1, 0.00],
            [dep_pr, 1-dep_pr]])
    return z_grid, z_dist, z_markov, a_grid, d_grid, d_markov

def disp_inc_f(a_grid, z_grid, r, w, p_d, chi): #Disposible income for consumption and assets after buying the durable good
    p = np.array([0, p_d]) #Vector of prices of durables
    durable_exp = p[:, None] - (1 - chi) * p  # Create matrix of adjustment costs
    np.fill_diagonal(durable_exp, 0)  # set diagonal to 0 (no cost if no switching)
    y_w = z_grid[np.newaxis] * w            # on (1, z)
    disp_inc = (1 + r) * a_grid[np.newaxis, np.newaxis, :] + y_w[..., np.newaxis] - durable_exp[..., np.newaxis, np.newaxis] # on (nd, z, a) #Disposable income for consumption
    return y_w, disp_inc, durable_exp


#%% Assemble the HH block (staged block)
hh = StageBlock([depreciation_stage, prod_stage, labsup_stage, consav_stage], name='hh',
                backward_init=hh_init, hetinputs=[make_grids, disp_inc_f])

print(hh)
print(f"Inputs: {hh.inputs}")
print(f"Outputs: {hh.outputs}")

#%%
# -------------------------------
# --Solving the baseline hh block --
# -------------------------------

#Specify different calibration
cali = dict()
p_d = 0.80
gamma = 1
eta = 0.5 #Useless for now
n_d = 1
dep_pr = 0.25 #Depreciation probability of the durable good
chi = 0.5 #Loss of value of durable if sold.

cali['baseline'] = {'taste_shock': 1E-1, 'vphi': 0.0, 'r': 0.02/4, 'beta': 0.97, 'eis': 0.5,
               'rho_z': 0.95, 'sd_z': 0.5, 'n_z': 7,
               'min_a': 0.0, 'max_a': 200, 'n_a': 200, 'w': 1.0, 'p_d': p_d, 'n_d': n_d, 'gamma' : gamma, 'eta': eta, 'dep_pr': dep_pr, 'chi' : chi}

taste_shock = 1e-5
vphi = 0.0
r = 0.02 / 4
beta = 0.97
eis = 0.5
rho_z = 0.95
sd_z = 0.5
n_z = 7
min_a = 0.0
max_a = 200
n_a = 200
w = 1.0

#%% Only useful for debugging
z_grid, z_dist, z_markov, a_grid, d_grid, d_markov = make_grids(rho_z, sd_z, n_z, min_a, max_a, n_a, dep_pr)
y_w, disp_inc, durable_exp = disp_inc_f(a_grid, z_grid, r, w, p_d, chi)
V, Va = hh_init(disp_inc, a_grid, eis, gamma, eta)
#%% Baseline model

ss = dict()
ss['baseline'] = hh.steady_state(cali['baseline'])
print(ss['baseline']['A'])
print('Proportion of people with a car at the end of the period (choice variable)',ss['baseline']['DD'])
print('Proportion of people with a car at the beginning of the period (state variable)',ss['baseline']['DD_2'])
print('Ratio of DD2/DD: ',ss['baseline']['DD_2'] / ss['baseline']['DD'])
print(ss['baseline']['C'])
#%%
policy_functions(ss, amax=150, d_tilde_list=[0] ,d_list = [0],iz_list=[0,3], figsize=0.8, models = ['baseline'])

#%%
ss['baseline'].internals['hh']['labsup']['law_of_motion'].P.shape

#%% Comparative statics of SS
results = analyze_steady_state('p_d', np.linspace(0.8, 3, 5), cali, hh)

#%%
results = analyze_steady_state('dep_pr', np.linspace(0.05, 0.5, 5), cali, hh)
#%%
results = analyze_steady_state('chi', np.linspace(0.1, 0.5, 5), cali, hh)


#%% Comparative statics of SS - 3d
CS_dep_pr_chi_2 = analyze_steady_state_3d(
    param1='dep_pr',
    values1=np.linspace(0.05, 0.75, 5),
    param2='chi',
    values2=np.linspace(0.1, 0.8, 5),
    cali=cali,
    hh=hh
    )

#%%
#analyze_steady_state_path_3d(param1='dep_pr',values1=np.linspace(0.05, 0.5, 5), param2='chi',values2=np.linspace(0.1, 0.5, 5),cali=cali,hh=hh, results=CS_dep_pr_chi_2) # 0.99 - 0.01
#analyze_steady_state_path_3d(param1='dep_pr',values1=np.linspace(0.05, 0.5, 5), param2='chi',values2=np.linspace(0.1, 0.5, 5),cali=cali,hh=hh, results=CS_dep_pr_chi_1) # 0.95 - 0.05
#analyze_steady_state_path_3d(param1='dep_pr',values1=np.linspace(0.05, 0.5, 5), param2='chi',values2=np.linspace(0.1, 0.5, 5),cali=cali,hh=hh, results=CS_dep_pr_chi_0) # 1.00 - 0.00


#%% Comparative statics
CS_gamma_p_d = analyze_steady_state_3d(
    param1='gamma',
    values1=np.linspace(0.1, 5, 5),
    param2='p_d',
    values2=np.linspace(0.1, 5, 5),
    cali=cali,
    hh=hh
)

# %%
