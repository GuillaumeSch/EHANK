"""Domestic vs external decomposition of the output response (y = cH + cHstar)."""
import os
import numpy as np
import matplotlib.pyplot as plt

from core.model import build_model, run

H = 24
NUMERAIRE, BOOKING = 'cpi', 'import'
OUT = 'paper/output'


def dev(irf, k):
    return 100.0 * np.asarray(irf[k], dtype=float)[:H]


if __name__ == '__main__':
    model = build_model(NUMERAIRE, booking=BOOKING)
    labels = [('none', 'No policy'), ('subsidy', 'Price cap'),
              ('transfer', 'Slutsky transfer')]
    irfs = {p: run(model, shock_kind='price', policy=p, model_variant='adoption')[1]
            for p, _ in labels}

    t = np.arange(H)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), sharey=True)
    for ax, (p, title) in zip(axes, labels):
        irf = irfs[p]
        y, cH, cHs = dev(irf, 'y'), dev(irf, 'cH'), dev(irf, 'cHstar')
        ax.fill_between(t, 0, cH, color='#48c', alpha=0.20)
        ax.fill_between(t, 0, cHs, color='#2a2', alpha=0.20)
        ax.plot(t, cH, color='#48c', lw=1.8, label='domestic demand $c_H$')
        ax.plot(t, cHs, color='#2a2', lw=1.8, label='exports $c_H^{*}$')
        ax.plot(t, y, 'k-', lw=1.8, label='output $y$')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('quarters', fontsize=8)
    axes[0].set_ylabel(r'level deviation $\times100$', fontsize=9)
    axes[0].legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    out = os.path.join(OUT, f'fig_output_domestic_external_{BOOKING}.pdf')
    fig.savefig(out)
    plt.close(fig)

    print(f"{'policy':>9s} {'cum y':>8s} {'domestic':>9s} {'export':>8s}")
    for p, _ in labels:
        irf = irfs[p]
        print(f"{p:>9s} {np.sum(dev(irf,'y')):8.1f} {np.sum(dev(irf,'cH')):9.1f} "
              f"{np.sum(dev(irf,'cHstar')):8.2f}")
    print(f"  -> {out}")
