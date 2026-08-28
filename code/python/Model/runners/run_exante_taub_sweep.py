"""Version A -- tau_b sweep of the ex-ante ETS economy under a brown-energy
price shock, for two carbon-revenue recycling schemes.

Extends run_exante_expost.py (fixed TAU_B=0.10, recycle='rebate') to:
  (i)  sweep tau_b over a grid, and
  (ii) compare two uses of the carbon revenue:
         'rebate'        lump-sum Trebate (baseline)
         'green_subsidy' the revenue pays a share s_g of the green switching
                         cost. s_g is solved ENDOGENOUSLY per tau_b so the
                         scheme is exactly budget-neutral (Trebate = 0): all
                         carbon revenue, and only the carbon revenue, funds the
                         switching subsidy.

Self-contained: no plotting/latex_tables dependency. Writes two PNGs and prints
the endogenous s_g* and a budget check per tau_b.
"""
import os
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.model import (build_model, run, solve_ss, ss_unknowns_targets_fixed_psi)
from core.calibration import make_calibration

NUMERAIRE, BOOKING = 'core', 'import'
H = 24
TAU_GRID = [0.10, 0.20, 0.30]     # 0.30 is the safe upper bound (fixed-psi SS
                                  # bracket breaks above ~0.30, handoff item 7)
OUT = 'paper/output'
ROOT_CACHE = 'cache/sg_roots.pkl'       # cached budget-neutral s_g* (see below)

PANELS = [('C', r'Consumption $C$'),
          ('D_GREEN', r'Green share $D_{\mathrm{GREEN}}$'),
          ('pE_B_P', r'Brown price $P^E_B/P$'),
          ('y', r'Output $y$')]


def pc(irf, k, h=H):
    return 100 * np.asarray(irf[k])[:h]


# ---------------------------------------------------------------------------
# Endogenous s_g: solve for the switching-subsidy rate that exhausts carbon
# revenue exactly (Trebate = 0). SS-only, so cheap enough to call inside brentq.
# ---------------------------------------------------------------------------
def ets_ss_only(model, psi_fixed, tb, s_g):
    """Prepared ETS steady state at a GIVEN switching-subsidy rate s_g,
    psi_g held fixed at the no-ETS baseline (D_GREEN floats)."""
    ov = dict(tau_b=tb, tau_g=0.0, s_g_ets=s_g, s_g=s_g)
    calib = make_calibration(NUMERAIRE, booking=BOOKING, ets=True, **ov)
    calib['psi_g'] = psi_fixed
    u, t = ss_unknowns_targets_fixed_psi(BOOKING, ets=True)
    return solve_ss(model, calib, unknowns=u, targets=t, booking=BOOKING)


def s_g_budget_neutral(model, psi_fixed, tb, hi=0.6):
    """s_g* such that Trebate(s_g*) = 0. Trebate = R_carbon - s_g*psi_g*D_SWITCH
    is decreasing in s_g (>0 at s_g=0, <0 once the subsidy over-runs revenue),
    so a sign change is bracketed on [0, hi]."""
    def resid(s_g):
        return float(ets_ss_only(model, psi_fixed, tb, s_g)['Trebate'])
    r_lo = resid(0.0)                       # = R_carbon > 0
    r_hi = resid(hi)
    if r_lo * r_hi > 0:                     # widen if the subsidy never exhausts
        hi = 0.9
        r_hi = resid(hi)
    return brentq(resid, 0.0, hi, xtol=1e-4)


def baseline_psi(model):
    ss, _ = run(model, shock_kind='price', policy='none',
                numeraire=NUMERAIRE, booking=BOOKING)
    return float(ss['psi_g'])


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def solve_scenario(model, tb, recycle, psi_fixed):
    """Full run (SS + IRF). For green_subsidy, s_g is set to the budget-neutral
    root; the returned split confirms Trebate ~ 0."""
    if recycle == 'rebate':
        ss, irf = run(model, shock_kind='price', policy='none', ets=True,
                      ets_kwargs=dict(tau_b=tb, recycle='rebate'),
                      numeraire=NUMERAIRE, booking=BOOKING)
        s_star = 0.0
    else:
        import os as _os, pickle as _pk
        cache = (_pk.load(open(ROOT_CACHE, 'rb'))['roots']
                 if _os.path.exists(ROOT_CACHE) else {})
        s_star = (cache[tb]['s_g'] if tb in cache
                  else s_g_budget_neutral(model, psi_fixed, tb))
        ss, irf = run(model, shock_kind='price', policy='none', ets=True,
                      ets_kwargs=dict(tau_b=tb, recycle='green_subsidy',
                                      s_g_ets=s_star),
                      numeraire=NUMERAIRE, booking=BOOKING)
    R = float(ss['R_carbon']); trebate = float(ss['Trebate'])
    split = dict(DG=float(ss['D_GREEN']), R=R, gsub=R - trebate,
                 trebate=trebate, s_g=s_star)
    return irf, split


def make_figure(model, recycle, irf_lf, irf_cap, psi_fixed, fname):
    ets, split = {}, {}
    for tb in TAU_GRID:
        try:
            ets[tb], split[tb] = solve_scenario(model, tb, recycle, psi_fixed)
        except Exception as e:
            print(f"  [skip] tau_b={tb:.2f} ({recycle}): "
                  f"{type(e).__name__}: {e}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (k, title) in zip(axes, PANELS):
        ax.plot(pc(irf_lf, k),  'k-',   lw=2, label='Laissez-faire')
        ax.plot(pc(irf_cap, k), 'C0--', lw=2, label='Ex-post cap')
        for j, tb in enumerate(TAU_GRID):
            if tb in ets:
                ax.plot(pc(ets[tb], k), lw=2, color=f'C{j+1}',
                        label=rf'ETS $\tau_b={tb:.2f}$')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('quarters', fontsize=8)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7, loc='best')
    tag = ('lump-sum rebate' if recycle == 'rebate'
           else r'budget-neutral green-subsidy recycling ($T^{\mathrm{reb}}=0$)')
    fig.suptitle(rf'Version A: ex-ante ETS ($\tau_b$ sweep) vs ex-post cap '
                 rf'under a brown-energy price shock -- {tag}', y=1.03)
    fig.tight_layout()
    fig.savefig(fname, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return split


def main():
    os.makedirs(OUT, exist_ok=True)
    model = build_model(NUMERAIRE, booking=BOOKING)
    psi_fixed = baseline_psi(model)

    _, irf_lf  = run(model, shock_kind='price', policy='none',
                     numeraire=NUMERAIRE, booking=BOOKING)
    _, irf_cap = run(model, shock_kind='price', policy='subsidy',
                     numeraire=NUMERAIRE, booking=BOOKING)

    for recycle, fname in [('rebate',        f'{OUT}/fig_exante_taub_rebate.png'),
                           ('green_subsidy',  f'{OUT}/fig_exante_taub_greensub_bn.png')]:
        print(f"\n=== recycle = {recycle} ===")
        split = make_figure(model, recycle, irf_lf, irf_cap, psi_fixed, fname)
        print(f"{'tau_b':>6}{'s_g*':>8}{'D_G ss%':>9}{'R_carbon':>11}"
              f"{'->greensub':>12}{'Trebate':>11}")
        for tb in TAU_GRID:
            if tb in split:
                s = split[tb]
                print(f"{tb:6.2f}{s['s_g']:8.4f}{100*s['DG']:9.2f}{s['R']:11.5f}"
                      f"{s['gsub']:12.5f}{s['trebate']:11.2e}")
        print(f"  -> {fname}")


if __name__ == '__main__':
    main()
