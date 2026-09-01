"""Green/adoption subsidy as a crisis instrument: IRFs vs the other policies.

policy='green' layers a transitory switching subsidy s_g (model.shock_green) on
top of the brown-price shock, so the government pays a fraction of psi_g during
the acute crisis (GREEN_SIZE=1.0 = full switching cost at impact, decaying at the
shock's half-life). The steady state is untouched (s_g=0 at the SS), so this is
like-for-like with none/cap/transfer. This isolates what the one-line table entry
in run_summary_table.py cannot: the DYNAMICS of the forced adoption wave and its
output / consumption / fiscal cost.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.model import build_model, run
from core.welfare import cev

NUMERAIRE, BOOKING = 'cpi', 'import'
H, OUT = 24, 'paper/output'

POLICIES = [('none', 'Laissez-faire', 'k-'),
            ('subsidy', 'Price cap', 'C0--'),
            ('transfer', 'Slutsky transfer', 'C2:'),
            ('green', 'Green subsidy', 'C3-')]
PANELS = [('D_GREEN', r'Green share $D^{G}$ (pp)'),
          ('D_SWITCH', r'Switchers $D^{\mathrm{sw}}$ (pp)'),
          ('y', r'Output $y$ (\%)'),
          ('C', r'Consumption $C$ (\%)'),
          ('pE_B_P', r'Brown price $P^E_B/P$ (\%)'),
          ('spending', r'Fiscal spending (\%)')]


def pc(irf, k, h=H):
    return 100 * np.asarray(irf[k])[:h]


def main():
    os.makedirs(OUT, exist_ok=True)
    model = build_model(NUMERAIRE, booking=BOOKING)
    irfs, sss = {}, {}
    for key, _, _ in POLICIES:
        sss[key], irfs[key] = run(model, shock_kind='price', policy=key,
                                  numeraire=NUMERAIRE, booking=BOOKING)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (k, title) in zip(axes.flat, PANELS):
        for key, lab, sty in POLICIES:
            ax.plot(pc(irfs[key], k), sty, lw=2, label=lab)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('quarters', fontsize=8)
        ax.tick_params(labelsize=8)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Green/adoption subsidy vs the other crisis instruments '
                 '(brown-price shock, import booking)')
    fig.tight_layout()
    fpath = os.path.join(OUT, f'fig_green_subsidy_{BOOKING}.pdf')
    fig.savefig(fpath, dpi=140, bbox_inches='tight')
    plt.close(fig)

    print(f"{'policy':<18s}{'peakDG':>9s}{'peakDsw':>9s}{'cumY':>9s}"
          f"{'cumC':>9s}{'fiscal':>9s}{'CEV%':>9s}")
    for key, lab, _ in POLICIES:
        irf = irfs[key]
        m, _ = cev(sss[key], irf)
        print(f"{lab:<18s}{np.max(pc(irf,'D_GREEN')):9.3f}"
              f"{np.max(pc(irf,'D_SWITCH')):9.3f}{np.sum(pc(irf,'y')):9.2f}"
              f"{np.sum(pc(irf,'C')):9.2f}{np.sum(pc(irf,'spending')):9.2f}"
              f"{100*m:9.3f}")
    print(f"[figure] {fpath}")


if __name__ == '__main__':
    main()
