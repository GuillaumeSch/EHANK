"""Plotting helpers for IRF overlays and parameter sweeps."""
import numpy as np
import matplotlib.pyplot as plt

def show_irfs(irfs, outputs, labels=None, titles=None, T_plot=24,
              ylabel='% dev. from ss', figsize=None, ncol=4, save_path=None):
    """Overlay several IRF dicts on a grid of variables."""
    labels = labels or [f'irf {i}' for i in range(len(irfs))]
    titles = titles or outputs
    nrow = int(np.ceil(len(outputs) / ncol))
    figsize = figsize or (4 * ncol, 3.1 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    ax_flat = np.atleast_1d(axes).flatten()
    for ax, k, t in zip(ax_flat, outputs, titles):
        for irf, lab in zip(irfs, labels):
            ax.plot(100 * np.asarray(irf[k])[:T_plot], lw=2, label=lab)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel('quarters', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=8)
    for ax in ax_flat[len(outputs):]:
        ax.axis('off')
    ax_flat[0].legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    return fig

def sweep_parameter(model, ss, param, values, shocks, unknowns_td, targets_td,
                    check=None):
    """Sweep a parameter that does not move the steady state."""
    irfs, labels = [], []
    for v in values:
        s = ss.copy()
        s[param] = v
        irf = model.solve_impulse_linear(s, unknowns_td, targets_td, shocks)
        if check is not None:
            check(irf)
        irfs.append(irf)
        labels.append(rf'${param} = {v}$')
    return irfs, labels

def durable_shares_by_wealth(ss, block='hh_0', truncate_at=5, save_path=None):
    """Share of each durable state at every wealth level."""
    labels = {0: 'BB brown', 1: 'BG (zero mass)', 2: 'GB switching', 3: 'GG green'}
    colors = {0: '#8B4513', 1: '#808080', 2: '#90EE90', 3: '#228B22'}
    D = np.asarray(ss.internals[block]['consav']['D'])
    a_grid = np.asarray(ss.internals[block]['a_grid'])
    A = float(np.sum(D.sum(axis=(0, 1)) * a_grid))
    M = D.sum(axis=1)                       # collapse productivity -> (durable, assets)
    if truncate_at is not None:
        cut = max(1, np.searchsorted(a_grid, truncate_at * A, side='right') - 1)
        M, a_grid = M[:, :cut + 1], a_grid[:cut + 1]
    tot = M.sum(axis=0)
    tot[tot == 0] = np.nan
    shares = M / tot[None, :]
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in range(shares.shape[0]):
        ax.plot(a_grid / A, shares[d], lw=2, color=colors[d], label=labels[d])
    ax.set_xlabel('wealth / average wealth')
    ax.set_ylabel('share choosing durable')
    ax.legend(frameon=False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    return fig
