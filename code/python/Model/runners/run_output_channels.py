"""Deeper output decomposition: exports, domestic-ARS, domestic-adoption (Section 5).

The two-way split y = c_H + c_H^* separates domestic demand from exports. One level
deeper, the domestic component itself splits---exactly, via the frozen-choice
counterfactual---into what it would be with the adoption margin shut (the ARS
real-income/substitution channel) and the adoption channel's increment:

    y = c_H^*                         exports (external / expenditure-switching)
      + c_H^{frozen}                  domestic demand, adoption margin frozen (ARS)
      + (c_H - c_H^{frozen})          domestic demand, adoption-channel increment

The three sum to y by construction. The adoption channel's output effect lands almost
entirely here (cum domestic -8.97 of a -8.79 total; exports +0.18): the switching-cost
import outflow crowds out domestic home-good demand, not exports.

Emits fig_output_channels_import.pdf. ~2 SS solves.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from core.model import build_model, run

H = 24
NUMERAIRE, BOOKING = 'cpi', 'import'
OUT = 'paper/output'


def d(irf, k):
    return 100.0 * np.asarray(irf[k], dtype=float)[:H]


if __name__ == '__main__':
    model = build_model(NUMERAIRE, booking=BOOKING)
    _, a = run(model, shock_kind='price', policy='none', model_variant='adoption')
    _, f = run(model, shock_kind='price', policy='none', model_variant='no_adoption')

    y = d(a, 'y')
    exports = d(a, 'cHstar')
    dom_ars = d(f, 'cH')
    dom_adopt = d(a, 'cH') - d(f, 'cH')
    resid = np.max(np.abs(y - (exports + dom_ars + dom_adopt)))

    t = np.arange(H)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(t, dom_ars, color='#48c', lw=1.8, label=r'domestic demand, frozen-adoption ($c_H^{\rm frozen}$)')
    ax.plot(t, dom_adopt, color='#c44', lw=1.8, label=r'domestic demand, adoption channel')
    ax.plot(t, exports, color='#2a2', lw=1.8, label=r'exports ($c_H^{*}$)')
    ax.plot(t, y, 'k-', lw=2.0, label=r'output $y$')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('quarters', fontsize=9)
    ax.set_ylabel(r'level deviation $\times100$', fontsize=9)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    out = os.path.join(OUT, f'fig_output_channels_{BOOKING}.pdf')
    fig.savefig(out)
    plt.close(fig)

    print(f"additivity residual: {resid:.2e}")
    print(f"cum: y={np.sum(y):.1f} = exports {np.sum(exports):+.1f} "
          f"+ domestic-ARS {np.sum(dom_ars):+.1f} + domestic-adoption {np.sum(dom_adopt):+.1f}")
    print(f"  -> {out}")
