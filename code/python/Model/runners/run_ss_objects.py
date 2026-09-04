"""Steady-state objects for the calibration section (Section 4).

One baseline SS solve, three population objects aggregated over the three
discount-factor types (equal mass, verified D.sum()=1 per type):

  1  Switch-probability CURVES, P(brown->green) vs wealth, one line per productivity
     (median type's policy function). Replaces the heatmap: the surface spans four
     orders of magnitude, which a single log color scale renders ambiguous (the
     top-productivity row looks non-monotone though the data are strictly monotone).
     Lines on a log-y axis show the true structure -- monotone in wealth and in
     productivity, converging at high wealth where the constraint never binds.

  2  Two SS objects that rationalise the policy results:
     (a) green-holding rate by wealth -- the STOCK behind the switch FLOW; selection
         into green rises with wealth (richer households afford the durable).
     (b) MPC by durable state across wealth -- constrained brown households carry the
         high MPCs, which is why a transfer to them stabilises output so effectively
         (Section 6) and why energy-indexed targeting, which tracks the wealth-rising
         energy bill, is regressive (Section 6.5).

Emits fig_switch_curves_import.png, fig_ss_objects_import.pdf. ~1 SS solve.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from core.model import build_model, run
from core.calibration import make_calibration
from core import household as hh_mod
from core.household import make_grids

NUMERAIRE, BOOKING = 'cpi', 'import'
OUT = 'paper/output'


def load_ss(model):
    cal = make_calibration(NUMERAIRE, BOOKING)
    e_grid, _, a_grid, _ = make_grids(cal['rho_e'], cal['sd_e'], cal['n_e'],
                                      cal['min_a'], cal['max_a'], cal['n_a'],
                                      cal['delta_g'])
    ss, _ = run(model, shock_kind='price', policy='none')
    blocks = sorted(k for k in ss.internals if k.startswith('hh_'))
    return ss, blocks, e_grid, a_grid


# =============================================================================
# 1  Switch-probability curves (median type)
# =============================================================================
def switch_curves(ss, blocks, e_grid, a_grid, out_png):
    greens = np.where(hh_mod.IS_GREEN > 0)[0]
    browns = np.where(hh_mod.IS_GREEN == 0)[0]
    betas = {b: float(ss[f"beta_{b.split('_')[1]}"]) for b in blocks}
    med = sorted(blocks, key=lambda b: betas[b])[len(blocks) // 2]
    inter = ss.internals[med]
    P = inter['durables']['law_of_motion'].P
    Dd = inter['durables']['D']
    Psw = P[greens].sum(axis=0)
    origin = browns[int(np.argmax([Dd[d].sum() for d in browns]))]
    S = 100.0 * Psw[origin]                                     # (n_e, n_a) percent

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    cols = plt.cm.viridis(np.linspace(0.1, 0.9, len(e_grid)))
    for ie in range(len(e_grid)):
        ax.plot(a_grid, S[ie], color=cols[ie], lw=1.8, label=f'{e_grid[ie]:.2f}')
    ax.set_xscale('symlog', linthresh=1.0)
    ax.set_yscale('log')
    ax.set_xlabel(r'Household wealth $a$')
    ax.set_ylabel(r'$P(\mathrm{brown}\!\to\!\mathrm{green})$, % per quarter')
    ax.legend(title=r'productivity $e$', fontsize=7, ncol=2)
    ax.grid(True, which='both', alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# =============================================================================
# 2  Green-holding rate and MPC by state, across wealth (population)
# =============================================================================
def ss_objects(ss, blocks, e_grid, a_grid, out_pdf):
    greens = np.where(hh_mod.IS_GREEN > 0)[0]
    browns = np.where(hh_mod.IS_GREEN == 0)[0]
    w = 1.0 / len(blocks)                                       # equal type mass

    D = sum(w * ss.internals[b]['durables']['D'] for b in blocks)   # (4,7,50)
    mean_a = float((D.sum(axis=(0, 1)) * a_grid).sum() / D.sum())
    a_rel = a_grid / mean_a                                     # wealth / mean wealth
    mpc = sum(w * ss.internals[b]['consav']['mpc'] * ss.internals[b]['durables']['D']
              for b in blocks)                                  # mass-weighted mpc

    mass_a = D.sum(axis=(0, 1))                                 # (n_a,)
    green_a = D[greens].sum(axis=(0, 1)) / np.maximum(mass_a, 1e-16)
    mpc_brown = (mpc[browns].sum(axis=(0, 1))
                 / np.maximum(D[browns].sum(axis=(0, 1)), 1e-16))
    mpc_green = (mpc[greens].sum(axis=(0, 1))
                 / np.maximum(D[greens].sum(axis=(0, 1)), 1e-16))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(a_rel, 100 * green_a, color='#2a2', lw=2)
    ax[0].set_xscale('symlog', linthresh=0.1)
    ax[0].set_xlabel(r'Household wealth relative to mean, $a/\bar a$')
    ax[0].set_ylabel(r'green-durable holders, %')
    ax[0].set_title('Green-durable holding rate', fontsize=10)
    ax[0].grid(True, which='both', alpha=0.2)

    ax[1].plot(a_rel, mpc_brown, color='#a33', lw=2, label='brown incumbents')
    ax[1].plot(a_rel, mpc_green, color='#2a2', lw=2, label='green holders')
    ax[1].set_xscale('symlog', linthresh=0.1)
    ax[1].set_xlabel(r'Household wealth relative to mean, $a/\bar a$')
    ax[1].set_ylabel(r'quarterly MPC')
    ax[1].set_title('Quarterly MPC, by durable state', fontsize=10)
    ax[1].legend(fontsize=8)
    ax[1].grid(True, which='both', alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

    # energy expenditure share by wealth (printed; motivates the homotheticity caveat)
    pE_B, pE_G = float(ss['pE_B_P']), float(ss['pE_G_P'])
    energy = sum(w * (pE_B * ss.internals[b]['consav']['cE_b']
                      + pE_G * ss.internals[b]['consav']['cE_g'])
                 * ss.internals[b]['durables']['D'] for b in blocks)
    cnondur = sum(w * ss.internals[b]['consav']['c']
                  * ss.internals[b]['durables']['D'] for b in blocks)
    e_a = energy.sum(axis=(0, 1))
    tot_a = (energy + cnondur).sum(axis=(0, 1))
    share = e_a / np.maximum(tot_a, 1e-16)
    lo = np.average(share[:len(a_grid) // 3], weights=mass_a[:len(a_grid) // 3])
    hi = np.average(share[-len(a_grid) // 3:], weights=mass_a[-len(a_grid) // 3:])
    print(f"energy expenditure share: bottom-third wealth {100*lo:.2f}%, "
          f"top-third {100*hi:.2f}%  (near-flat -> homothetic -> regressive targeting)")
    print(f"green holders: a=0 {100*green_a[0]:.2f}%, a=max {100*green_a[-1]:.2f}%")
    print(f"MPC brown a=0 {mpc_brown[0]:.3f} vs green a=0 {mpc_green[0]:.3f}")


if __name__ == '__main__':
    model = build_model(NUMERAIRE, booking=BOOKING)
    ss, blocks, e_grid, a_grid = load_ss(model)
    switch_curves(ss, blocks, e_grid, a_grid,
                  os.path.join(OUT, f'fig_switch_curves_{BOOKING}.png'))
    ss_objects(ss, blocks, e_grid, a_grid,
               os.path.join(OUT, f'fig_ss_objects_{BOOKING}.pdf'))
    print(f"-> figures written to {OUT}")
