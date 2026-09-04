"""The private economics of one switch, at the steady state (Section 4).

For a single household, adopting green is an upfront outlay psi_g against a gradual
operating saving: each quarter the durable survives, energy is bought at PEG instead
of PEB. The figure prices that trade the way the marginal-switcher condition
(eq:margin) does. For the median discount-factor type at median productivity, and
three wealth levels, it plots the cumulative energy saving discounted by survival
and interest,

    S(T) = sum_{t=1}^{T} [ (1-delta_g)/(1+r) ]^t  (PEB - PEG) cE_G(e_med, a),

against the horizontal line psi_g; S(infinity) is the left side of eq:margin. cE_G
is the green incumbent's steady-state energy demand at that grid point, so wealthier
households, who use more energy in levels, recoup the cost faster. The vertical line
marks the expected durable life 1/delta_g: a household whose saving crosses psi_g
inside that life is a natural adopter, one that never crosses stays brown. No shock
anywhere: this is the steady-state investment problem the crisis then perturbs.

Emits fig_adoption_payoff_import.pdf. One SS solve.
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
T = 60
REL_WEALTH = [1.0, 25.0, 40.0]          # a / mean(a) of the three households
COLS = ['#c44', '#48c', '#2a2']


if __name__ == '__main__':
    model = build_model(NUMERAIRE, booking=BOOKING)
    cal = make_calibration(NUMERAIRE, BOOKING)
    e_grid, _, a_grid, _ = make_grids(cal['rho_e'], cal['sd_e'], cal['n_e'],
                                      cal['min_a'], cal['max_a'], cal['n_a'],
                                      cal['delta_g'])
    ss, _ = run(model, shock_kind='price', policy='none')

    blocks = sorted(k for k in ss.internals if k.startswith('hh_'))
    betas = {b: float(ss[f"beta_{b.split('_')[1]}"]) for b in blocks}
    med = sorted(blocks, key=lambda b: betas[b])[len(blocks) // 2]
    inter = ss.internals[med]

    Dpop = sum(ss.internals[b]['durables']['D'] for b in blocks)
    mean_a = float((Dpop.sum(axis=(0, 1)) * a_grid).sum() / Dpop.sum())

    green_inc = int(np.where((hh_mod.IS_GREEN > 0) & (hh_mod.PAYS_SWITCH == 0))[0][0])
    ie = len(e_grid) // 2
    cE_G = inter['consav']['cE_g'][green_inc, ie]           # (n_a,)

    psi_g = float(ss['psi_g'])
    pEB, pEG = float(ss['pE_B_P']), float(ss['pE_G_P'])
    r = float(ss['r_num'])
    delta_g = cal['delta_g']
    disc = (1.0 - delta_g) / (1.0 + r)

    t = np.arange(1, T + 1)
    wgt = np.cumsum(disc ** t)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for rel, col in zip(REL_WEALTH, COLS):
        ia = int(np.argmin(np.abs(a_grid - rel * mean_a)))
        flow = (pEB - pEG) * float(cE_G[ia])
        S = wgt * flow
        ax.plot(t, S, color=col, lw=1.8,
                label=fr'wealth $a = {a_grid[ia] / mean_a:.0f}\,\bar a$')
        if S[-1] >= psi_g:
            tb = int(np.argmax(S >= psi_g)) + 1
            ax.plot(tb, psi_g, 'o', color=col, ms=5, zorder=5)
    ax.axhline(psi_g, color='k', lw=1.2, ls='--',
               label=fr'switching cost $\psi_g = {psi_g:.2f}$')
    ax.axvline(1.0 / delta_g, color='gray', lw=1.0, ls=':',
               label=r'expected durable life $1/\delta_g$')
    ax.set_xlabel('quarters since adoption')
    ax.set_ylabel('discounted cumulative energy saving')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.22, lw=0.5)
    fig.tight_layout()
    out = os.path.join(OUT, f'fig_adoption_payoff_{BOOKING}.pdf')
    fig.savefig(out)
    plt.close(fig)

    for rel in REL_WEALTH:
        ia = int(np.argmin(np.abs(a_grid - rel * mean_a)))
        flow = (pEB - pEG) * float(cE_G[ia])
        Sinf = flow * disc / (1 - disc)
        S60 = float(wgt[-1] * flow)
        cross = int(np.argmax(wgt * flow >= psi_g)) + 1 if S60 >= psi_g else None
        print(f"a/mean={a_grid[ia] / mean_a:5.1f}: flow {flow:.4f}, "
              f"S(inf)={Sinf:.3f}, crossing at q={cross}")
    print(f"  -> {out}")
