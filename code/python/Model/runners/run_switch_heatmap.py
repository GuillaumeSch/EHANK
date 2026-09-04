"""Switch-probability heatmap: P(brown -> green) across the (productivity, wealth)
grid at the baseline steady state.

Companion visual to tab_switch_prob_dist (Section calibration): the table reports
the distribution-weighted quantiles of the quarterly switch probability; this figure
shows the full (e, a) surface. Extraction is identical to switch_prob_table in
run_taste_identification.py, so the two are consistent by construction. Levels are
tiny at the calibrated rare margin, so a log color scale is used to reveal the
wealth x productivity gradient the table collapses.

Emits fig_switch_heatmap.png. Runtime: one SS solve (~15-25s); run locally.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from core.model import build_model, run
from core.calibration import make_calibration
from core import household as hh_mod

NUMERAIRE, BOOKING = 'cpi', 'import'
OUT = 'paper/output'


# =============================================================================
# 1  Steady state and the (e, a) switch-probability surface
# =============================================================================
def switch_prob_surface(model):
    """P(choose green | brown incumbent, e, a) for the MEDIAN discount-factor type,
    shape (n_e, n_a). We plot a single type's POLICY FUNCTION rather than the
    mass-weighted pool: pooling mixes three very different decision rules (patient
    households switch far less than impatient ones) whose changing composition across
    the wealth grid produces a spurious non-monotonicity. Each type's rule is
    monotone in wealth and productivity; the distribution across types is documented
    separately in Table~\\ref{tab:switch_prob_dist}."""
    cal = make_calibration(NUMERAIRE, BOOKING)
    e_grid, _, a_grid, _ = make_grids_from_cal(cal)
    ss, _ = run(model, shock_kind='price', policy='none')

    greens = np.where(hh_mod.IS_GREEN > 0)[0]
    browns = np.where(hh_mod.IS_GREEN == 0)[0]
    blocks = sorted(k for k in ss.internals if k.startswith('hh_'))
    betas = {b: float(ss[f"beta_{b.split('_')[1]}"]) for b in blocks}
    median_b = sorted(blocks, key=lambda b: betas[b])[len(blocks) // 2]

    inter = ss.internals[median_b]
    P = inter['durables']['law_of_motion'].P     # (chosen, origin, e, a)
    Dd = inter['durables']['D']
    Psw = P[greens].sum(axis=0)                   # P(green | origin, e, a)
    origin = browns[int(np.argmax([Dd[d].sum() for d in browns]))]

    # population mean wealth for the relative-wealth axis
    Dpop = sum(ss.internals[b]['durables']['D'] for b in blocks)
    mean_a = float((Dpop.sum(axis=(0, 1)) * a_grid).sum() / Dpop.sum())
    return e_grid, a_grid / mean_a, Psw[origin], betas[median_b]


def make_grids_from_cal(cal):
    from core.household import make_grids
    return make_grids(cal['rho_e'], cal['sd_e'], cal['n_e'],
                      cal['min_a'], cal['max_a'], cal['n_a'], cal['delta_g'])


# =============================================================================
# 2  Figure
# =============================================================================
def figure(e_grid, a_grid, P_switch, out_png):
    P = 100.0 * P_switch                              # percent per quarter
    pos = P[np.isfinite(P) & (P > 0)]
    vmin, vmax = max(pos.min(), 1e-3), pos.max()

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    mesh = ax.pcolormesh(a_grid, e_grid, np.clip(P, vmin, None),
                         norm=LogNorm(vmin=vmin, vmax=vmax),
                         shading='auto', cmap='viridis')
    ax.set_xscale('symlog', linthresh=0.1)           # wealth grid is right-skewed
    ax.set_xlabel(r'Household wealth relative to mean, $a/\bar a$')
    ax.set_ylabel(r'Idiosyncratic productivity $e$')
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r'$P(\mathrm{brown}\!\to\!\mathrm{green})$, % per quarter')
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


if __name__ == '__main__':
    model = build_model(NUMERAIRE, booking=BOOKING)
    e_grid, a_grid, P_switch, beta_med = switch_prob_surface(model)
    fin = P_switch[np.isfinite(P_switch)]
    print(f"median-type beta={beta_med:.3f}; switch prob (%/q): "
          f"mean={100*fin.mean():.3f} max={100*fin.max():.2f} at (e,a) grid "
          f"{np.unravel_index(np.nanargmax(P_switch), P_switch.shape)}")
    path = figure(e_grid, a_grid, P_switch, os.path.join(OUT, f'fig_switch_heatmap.png'))
    print(f"  -> {path}")
