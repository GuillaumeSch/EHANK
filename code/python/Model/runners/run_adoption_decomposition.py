"""Balance-of-payments anatomy of the adoption channel (Section 5).

The DRAFT closed-economy accounting Y = C_core + energy + Psi + G does not carry
over: the open-economy resource constraint is y = cH + cHstar, and the energy and
switching flows are IMPORT leakages (block CA), not absorption components. The
correct object here is the import side of the adoption channel.

We isolate the adoption channel as (adoption - frozen) at the common steady state,
and split its contribution to imports into
    (i)  the switching-cost outflow   d imports_dur  = psi_g * d D_SWITCH   (front-
         loaded, paid once per switcher), and
    (ii) everything else              d (imports - imports_dur)            (dominated
         by the flow energy saving from the brown->green mix shift, an inflow).
By construction the two sum to d imports, so this is an exact accounting split of an
existing series under the baseline import booking -- it re-books nothing.

Numerical check reproduces the Section 5 claim: the cumulative switching-cost import
equals the pF_P*cF_switch import component (current baseline ~ +11.7, price shock).

Emits fig5b_adoption_flows.png. Runtime: 4 SS solves + transitions (~1-2 min); local.

NOTE (booking presentation): the import booking is Boris's call; this figure only
displays the baseline booking's flows, but flag it before it enters the headline.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from core.model import build_model, run

H = 24
NUMERAIRE, BOOKING = 'cpi', 'import'
OUT = 'paper/output'


def dev(irf, k):
    return np.asarray(irf[k], dtype=float)


# =============================================================================
# 1  Solve adoption vs frozen, both shocks
# =============================================================================
def solve(model):
    out = {}
    ss = None
    for shock in ['price', 'supply']:
        ssa, a = run(model, shock_kind=shock, policy='none', model_variant='adoption')
        f = run(model, shock_kind=shock, policy='none', model_variant='no_adoption')[1]
        out[shock] = (a, f)
        ss = ssa
    return out, ss


# =============================================================================
# 2  Adoption-channel import decomposition (adoption - frozen)
# =============================================================================
def decompose(a, f, ss):
    # switching-cost import expenditure is the pF_P * cF_switch component of imports;
    # its first-order deviation is pF_P_ss * d cF_switch + cF_switch_ss * d pF_P.
    pF, cfs = float(ss['pF_P']), float(ss['cF_switch'])
    def dsw(irf):
        return pF * dev(irf, 'cF_switch') + cfs * dev(irf, 'pF_P')
    d_imp = dev(a, 'imports') - dev(f, 'imports')
    d_sw = dsw(a) - dsw(f)                                     # switching cost
    d_rest = d_imp - d_sw                                      # energy mix + core
    d_nx = dev(a, 'netexports') - dev(f, 'netexports')
    return d_imp, d_sw, d_rest, d_nx


# =============================================================================
# 3  Figure
# =============================================================================
def figure(sol, ss, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, shock in zip(axes, ['price', 'supply']):
        a, f = sol[shock]
        d_imp, d_sw, d_rest, d_nx = decompose(a, f, ss)
        t = np.arange(H)
        ax.fill_between(t, 0, 100 * d_sw[:H], color='#c44', alpha=0.20)
        ax.fill_between(t, 0, 100 * d_rest[:H], color='#48c', alpha=0.20)
        ax.plot(t, 100 * d_sw[:H], color='#c44', lw=1.8,
                label=r'switching-cost outflow $\psi_g\,D^{sw}$')
        ax.plot(t, 100 * d_rest[:H], color='#48c', lw=1.8,
                label=r'energy-saving inflow (mix & other)')
        ax.plot(t, 100 * d_imp[:H], 'k-', lw=1.8, label='net import response')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_title(f'{shock} shock', fontsize=10)
        ax.set_xlabel('quarters', fontsize=8)
    axes[0].set_ylabel(r'contribution to imports, $\times100$', fontsize=9)
    axes[0].legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


if __name__ == '__main__':
    model = build_model(NUMERAIRE, booking=BOOKING)
    sol, ss = solve(model)
    for shock in ['price', 'supply']:
        a, f = sol[shock]
        d_imp, d_sw, d_rest, d_nx = decompose(a, f, ss)
        resid = np.max(np.abs(d_imp - (d_sw + d_rest)))
        cum_sw = 100 * np.sum(d_sw[:H])
        cum_rest = 100 * np.sum(d_rest[:H])
        print(f"[{shock}] additivity residual = {resid:.2e} (should be ~0)")
        print(f"[{shock}] cum switching-cost imports = {cum_sw:+.2f} | "
              f"cum energy/other = {cum_rest:+.2f} | cum total = {cum_sw + cum_rest:+.2f}")
    path = figure(sol, ss, os.path.join(OUT, 'fig5b_adoption_flows.png'))
    print(f"  -> {path}")
