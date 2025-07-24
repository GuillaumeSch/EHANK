import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_adj_costs(adjustment_cost_func, a_grid, ra, chi0, chi1, chi2,
                                   a_target=None, plot_3d=True):
    """
    Plots adjustment costs as a 3D surface or 2D slice.

    Parameters:
    -----------
    adjustment_cost_func : function
        The function to compute adjustment costs: f(a, a_grid, ra, chi0, chi1, chi2)
    a_grid : array
        Asset grid over which to compute adjustment costs
    ra, chi0, chi1, chi2 : floats
        Parameters for the adjustment cost function
    a_target : float or None
        If provided, plots a 2D slice at the closest a ≈ a_target
    plot_3d : bool
        Whether to display the full 3D surface (ignored if a_target is set)
    """
    # Prepare meshgrid
    A_current, A_target = np.meshgrid(a_grid, a_grid)
    Z = np.zeros_like(A_current)

    # Fill Z with adjustment costs
    for i, a in enumerate(a_grid):
        Z[i, :] = adjustment_cost_func(a, a_grid, ra, chi0, chi1, chi2)

    if a_target is not None:
        # Find closest index to a_target
        idx = np.argmin(np.abs(a_grid - a_target))
        a_closest = a_grid[idx]
        costs_slice = Z[idx, :]

        # Plot 2D slice
        plt.figure(figsize=(8, 5))
        plt.plot(a_grid, costs_slice, label=f'Current a ≈ {a_closest:.2f}')
        plt.xlabel("Target Asset Level (a')")
        plt.ylabel("Adjustment Cost")
        plt.title(f"Adjustment Cost Slice (Current a ≈ {a_closest:.2f})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif plot_3d:
        # Plot full 3D surface
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(A_current, A_target, Z, cmap='viridis')
        ax.set_xlabel('Current Asset Level (a)')
        ax.set_ylabel("Target Asset Level (a')")
        ax.set_zlabel('Adjustment Cost')
        ax.set_title("Adjustment Cost Surface")
        plt.tight_layout()
        plt.show()
