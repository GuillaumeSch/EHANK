"""Accommodative monetary offset of the energy shock (linear superposition)."""
import os
import numpy as np
import matplotlib.pyplot as plt

from core.model import build_model, run

H = 24
NUMERAIRE, BOOKING = 'cpi', 'import'
OUT = 'paper/output'
MP_UNIT = -0.0025          # reference accommodative innovation, scaled below
KEYS = ['y', 'C', 'D_GREEN', 'D_SWITCH', 'rante', 'pi_ann', 'inom_ann']


def A(irf, k):
    return np.asarray(irf[k], dtype=float)


def cum(irf, k):
    return 100 * float(np.sum(A(irf, k)[:H]))


def peak(irf, k):
    return 100 * float(np.max(A(irf, k)[:H]))


if __name__ == '__main__':
    model = build_model(NUMERAIRE, booking=BOOKING)
    base = run(model, shock_kind='price', policy='none', phi_pi=0.0, phi_pie=1.0)[1]
    mp = run(model, shock_kind='monetary', policy='none', shock_kwargs=dict(size=MP_UNIT))[1]

    # linear closed form: choose alpha so cum-y loss is halved (base + alpha*mp)
    cy_base, cy_mp = cum(base, 'y'), cum(mp, 'y')
    alpha = -0.5 * cy_base / cy_mp
    keys = [k for k in KEYS if k in base and k in mp]
    offset = {k: A(base, k) + alpha * A(mp, k) for k in keys}
    bp = abs(alpha * MP_UNIT) * 1e4      # implied innovation, annualised bp

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    panels = [('y', r'Output $y$'), ('C', r'Consumption $C$'),
              ('D_GREEN', r'Green share $D^G$'), ('pi_ann', r'Inflation (ann.)')]
    t = np.arange(H)
    for ax, (k, title) in zip(axes.ravel(), panels):
        ax.plot(t, 100 * A(base, k)[:H], 'k--', lw=1.6, label='baseline (const. real rate)')
        ax.plot(t, 100 * offset[k][:H], color='#2a7', lw=2.0, label='+ accommodation')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('quarters', fontsize=8)
    axes[0, 0].legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'fig_monetary_offset_{BOOKING}.pdf'))
    plt.close(fig)

    off = {k: {'cum': 100 * float(np.sum(offset[k][:H])),
               'peak': 100 * float(np.max(offset[k][:H]))} for k in keys}
    print(f"implied accommodation: alpha={alpha:.2f} x {MP_UNIT} -> {bp:.0f}bp annualised")
    print(f"{'':12s} {'baseline':>10s} {'offset':>10s}")
    print(f"{'cum y':12s} {cy_base:10.1f} {off['y']['cum']:10.1f}")
    print(f"{'cum C':12s} {cum(base,'C'):10.1f} {off['C']['cum']:10.1f}")
    print(f"{'peak D^G':12s} {peak(base,'D_GREEN'):10.2f} {off['D_GREEN']['peak']:10.2f}")
    print(f"{'peak pi_ann':12s} {peak(base,'pi_ann'):10.2f} {off['pi_ann']['peak']:10.2f}")
    print(f"  -> {OUT}/fig_monetary_offset_{BOOKING}.pdf")
