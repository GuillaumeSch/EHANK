"""Direct vs indirect decomposition of the consumption response (Section 5).

The household consumption response to the brown-price shock, split into the channels
through which the shock reaches the budget, using the household consumption Jacobians:

    dC = sum_x  J^{C,x} @ dx ,   x in household inputs,

which is exact to first order (the transition is solved linearly). We run it in the
CPI numeraire (p=1), where the household inputs are unambiguous -- real after-tax
income, the real rate, and the relative energy price -- so the split maps to
economics rather than to a unit of account. The total is numeraire-invariant and
equals the baseline cumulative consumption response.

Grouping:
  direct (energy price)   pE_B_P, pE_B_pretax_P, pE_G_P, pE_P  (the energy bill in
                          the budget; the pre/post-tax pair is collinear at tau_b=0,
                          so only the sum is meaningful)
  indirect: real income   atw_n_num, n   (wages, hours, profits -- general
                          equilibrium)
  indirect: real rate     r_num
  relative import price    pHF_P

Result: the response is almost entirely the indirect real-income channel; the direct
energy-price effect on consumption is negligible, because energy is a small budget
share and the shock propagates through factor income. This is the ARS / Kaplan-Moll-
Violante finding that indirect effects dominate in HANK.

A long Jacobian horizon (T=200) is required for the reconstruction to close: forward-
looking consumption at date t responds to input paths at s>t, so truncating before
the shock decays leaves a residual. Emits fig_direct_indirect_import.pdf. ~1 SS solve
plus one household Jacobian.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from core.model import build_model, run
from core.household import hh_ha_durable

T, H = 200, 24
NUMERAIRE, BOOKING = 'cpi', 'import'
OUT = 'paper/output'

GROUPS = {
    'direct: energy price': ['pE_B_P', 'pE_B_pretax_P', 'pE_G_P', 'pE_P'],
    'indirect: real income': ['atw_n_num', 'n'],
    'indirect: real rate': ['r_num'],
    'relative import price': ['pHF_P'],
}
COLORS = {'direct: energy price': '#c44', 'indirect: real income': '#48c',
          'indirect: real rate': '#2a2', 'relative import price': '#963'}


if __name__ == '__main__':
    model = build_model(NUMERAIRE, booking=BOOKING)
    hh = hh_ha_durable()
    ss, irf = run(model, shock_kind='price', policy='none', numeraire=NUMERAIRE)

    moving = [x for x in hh.inputs
              if x in irf and np.max(np.abs(np.asarray(irf[x])[:T])) > 1e-12]
    J = hh.jacobian(ss, inputs=moving, outputs=['C'], T=T)
    dC = np.asarray(irf['C'], dtype=float)[:T]
    contrib = {x: J['C'][x] @ np.asarray(irf[x], dtype=float)[:T] for x in moving}

    grouped = {g: sum(contrib[x] for x in ks if x in contrib)
               for g, ks in GROUPS.items()}
    recon = sum(grouped.values())
    res = float(np.max(np.abs((recon - dC)[:H])))

    t = np.arange(H)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(t, 100 * dC[:H], 'k-', lw=2.2, label='total $C$')
    for g, v in grouped.items():
        if np.max(np.abs(v[:H])) > 1e-6:
            ax.plot(t, 100 * v[:H], color=COLORS[g], lw=1.8, label=g)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('quarters')
    ax.set_ylabel(r'consumption response, level dev. $\times100$')
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT, f'fig_direct_indirect_{BOOKING}.pdf')
    fig.savefig(out)
    plt.close(fig)

    def cum(v):
        return 100 * float(np.sum(v[:H]))
    print(f'closure max|res|[:24] = {res:.2e}   (dC peak {np.max(np.abs(dC)):.3f})')
    print(f'cum C[:24] = {cum(dC):.1f}')
    for g, v in grouped.items():
        print(f'  {g:24s} {cum(v):8.2f}')
    print(f'  -> {out}')
