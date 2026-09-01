"""Monetary stance and the adoption wave under the energy shock."""
import os
import numpy as np
import matplotlib.pyplot as plt

from core.model import build_model, run
from tools.latex_tables import write_table

H = 24
NUMERAIRE, BOOKING = 'cpi', 'import'
OUT = 'paper/output'
MP_CUT = -0.0025


def A(irf, k):
    return np.asarray(irf[k], dtype=float)


def stats(irf):
    return dict(
        dG=100 * float(np.max(A(irf, 'D_GREEN')[:H])),
        dsw=100 * float(np.sum(A(irf, 'D_SWITCH')[:H])),
        cy=100 * float(np.sum(A(irf, 'y')[:H])),
        cC=100 * float(np.sum(A(irf, 'C')[:H])),
        cr=100 * float(np.sum(A(irf, 'rante')[:H])),
        pi0=100 * float(A(irf, 'pi_ann')[0]),
    )


def superpose(a, b, keys):
    return {k: A(a, k) + A(b, k) for k in keys}


if __name__ == '__main__':
    model = build_model(NUMERAIRE, booking=BOOKING)
    keys = ['D_GREEN', 'D_SWITCH', 'y', 'C', 'rante', 'pi_ann']

    base = run(model, shock_kind='price', policy='none', phi_pi=0.0, phi_pie=1.0)[1]
    tay = run(model, shock_kind='price', policy='none', phi_pi=1.5, phi_pie=0.0)[1]
    mp_only = run(model, shock_kind='monetary', policy='none',
                  shock_kwargs=dict(size=MP_CUT))[1]
    accom = superpose(base, mp_only, keys)

    rows = []
    for lab, irf in [('Constant real rate (baseline)', base),
                     ('Active Taylor $\\phi_\\pi=1.5$', tay),
                     ('Baseline $+$ 25bp accommodation', accom)]:
        s = stats(irf)
        rows.append([lab, f"{s['dG']:.2f}", f"{s['dsw']:.2f}",
                     f"{s['cy']:.1f}", f"{s['cC']:.1f}", f"{s['cr']:+.2f}"])
    write_table(
        os.path.join(OUT, f'tab_monetary_stance_{BOOKING}.tex'),
        colspec='lrrrrr',
        header=['stance', r'peak $\Delta D^G$', r'$\sum D^{sw}$', r'$\sum y$',
                r'$\sum C$', r'$\sum r$'],
        rows=rows,
        caption=(r'Monetary stance and the adoption wave under the brown-price shock. '
                 r'Peak green-share response and cumulative switching, output, '
                 r'consumption and the ex-ante real rate ($\times100$, $H=24$). A '
                 r'Taylor rule that leans against the energy-driven inflation raises '
                 r'the real rate and modestly shrinks the wave at a large output '
                 r'cost; a deliberate accommodative innovation amplifies it.'),
        label='tab:monetary_stance', midrule_after={0})

    PHIS = [1.25, 1.5, 2.0, 2.5, 3.0]
    sweep = [stats(run(model, shock_kind='price', policy='none',
                       phi_pi=phi, phi_pie=0.0)[1]) for phi in PHIS]
    base_s = stats(base)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].axhline(base_s['dG'], color='gray', ls=':', lw=1.2, label='constant real rate')
    ax[0].plot(PHIS, [s['dG'] for s in sweep], 'o-', color='#c44', lw=1.8)
    ax[0].set_xlabel(r'Taylor inflation coefficient $\phi_\pi$')
    ax[0].set_ylabel(r'peak $\Delta D^G$ (pp)')
    ax[0].set_title('Adoption wave', fontsize=10)
    ax[0].legend(fontsize=8)
    ax[1].axhline(base_s['cy'], color='gray', ls=':', lw=1.2, label='constant real rate')
    ax[1].plot(PHIS, [s['cy'] for s in sweep], 'o-', color='#48c', lw=1.8)
    ax[1].set_xlabel(r'Taylor inflation coefficient $\phi_\pi$')
    ax[1].set_ylabel(r'$\sum y$ ($\times100$)')
    ax[1].set_title('Output', fontsize=10)
    ax[1].legend(fontsize=8)
    fig.suptitle('Stricter inflation targeting during the energy shock: a small '
                 'transition cost, a large output cost')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'fig_monetary_stance_{BOOKING}.pdf'))
    plt.close(fig)

    s_tay = stats(tay)
    dwave, drate = s_tay['dG'] - base_s['dG'], s_tay['cr'] - base_s['cr']
    print(f"M3 sensitivity (baseline -> Taylor 1.5):")
    print(f"  d(peak dG) = {dwave:+.2f} pp  for cum-rate move {drate:+.2f}"
          f"  -> {dwave/drate:+.3f} pp per unit cum-r")
    print(f"  d(cum y)   = {s_tay['cy']-base_s['cy']:+.1f}")
    print(f"  25bp accommodation standalone: dG peak "
          f"{100*float(np.max(A(mp_only,'D_GREEN')[:H])):+.3f} pp, "
          f"cum r {100*float(np.sum(A(mp_only,'rante')[:H])):+.2f}")
    print(f"  sweep peak dG: {[round(s['dG'],2) for s in sweep]}")
    print(f"  sweep cum  y: {[round(s['cy'],1) for s in sweep]}")
