"""QA for the common-steady-state 'no adoption' counterfactual."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.model import (build_model, frozen_model, solve_ss, run, shock_price,
                   td_unknowns_targets, ss_unknowns_targets)
from core.calibration import make_calibration

NUMERAIRE, BOOKING = 'cpi', 'import'
OUT = 'paper/output'
H = 24
H_LONG = 120

RES_KEYS = ['goods_clearing', 'assets_clearing', 'E_clearing', 'nfares',
           'pires', 'r_res', 'tauY_res', 'w_res']

def main():
    os.makedirs(OUT, exist_ok=True)
    M = build_model(NUMERAIRE, booking=BOOKING)
    MF = frozen_model(NUMERAIRE, booking=BOOKING)
    u, t = td_unknowns_targets(BOOKING)
    su, st = ss_unknowns_targets(BOOKING)

    ss = solve_ss(M, make_calibration(NUMERAIRE, booking=BOOKING),
                 unknowns=su, targets=st, booking=BOOKING)

    lines = ['=== SS identity: adoption vs no_adoption (frozen) ===']
    ss_ad, _ = run(M, shock_kind='price', policy='none',
                   model_variant='adoption', numeraire=NUMERAIRE, booking=BOOKING)
    ss_no, _ = run(M, shock_kind='price', policy='none',
                   model_variant='no_adoption', numeraire=NUMERAIRE, booking=BOOKING)
    max_ss_diff = 0.0
    for k in ['psi_g', 'D_GREEN', 'D_SWITCH', 'C', 'A', 'y']:
        d = abs(float(ss_ad[k]) - float(ss_no[k]))
        max_ss_diff = max(max_ss_diff, d)
        lines.append(f'  {k:10s} adoption={float(ss_ad[k]):.10f}  '
                     f'no_adoption={float(ss_no[k]):.10f}  diff={d:.2e}')
    assert max_ss_diff < 1e-9, (
        f'SS identity broken (max diff={max_ss_diff:.2e}); the adoption and '
        f'no-adoption steady states must be identical for the channel to be '
        f'read as full minus frozen.')
    lines.append(f'  PASS: max SS diff = {max_ss_diff:.2e} (< 1e-9)')

    lines.append('')
    lines.append('=== equilibrium residuals, price shock size=1.0 (max |.| over 24q) ===')
    shk = shock_price(size=1.0)
    A_irf = M.solve_impulse_linear(ss, u, t, shk)
    Fz_irf = MF.solve_impulse_linear(ss, u, t, shk)
    for lab, irf in [('full', A_irf), ('frozen', Fz_irf)]:
        row = f'  {lab:8s}'
        for k in RES_KEYS:
            if k in irf:
                row += f'  {k}={np.max(np.abs(np.asarray(irf[k])[:H])):.1e}'
        lines.append(row)

    report = '\n'.join(lines)
    print(report)
    open(f'{OUT}/frozen_qa_residuals.txt', 'w').write(report + '\n')

    # H=24 panel
    def pc(irf, k, h=H):
        return 100 * np.asarray(irf[k])[:h]
    keys = [('y', 'Output $y$'), ('C', 'Consumption $C$'),
            ('D_GREEN', 'Green share $D^{G}$'), ('nx_gdp', 'Net exports/GDP')]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (k, ti) in zip(axes, keys):
        ax.plot(pc(A_irf, k), 'C0-', lw=2, label='full (adoption on)')
        ax.plot(pc(Fz_irf, k), 'C2--', lw=2, label='no adoption (frozen, common SS)')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(ti, fontsize=10); ax.set_xlabel('quarters', fontsize=8)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle('Common-steady-state adoption counterfactual: full vs frozen '
                 '(price shock, size=1.0)')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_frozen_qa.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

    # long-run panel
    shk_long = shock_price(size=1.0, half_life=16, T=300)
    A_long = M.solve_impulse_linear(ss, u, t, shk_long)
    Fz_long = MF.solve_impulse_linear(ss, u, t, shk_long)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (k, ti) in zip(axes, [('y', 'Output $y$'), ('C', 'Consumption $C$')]):
        full = pc(A_long, k, H_LONG); frz = pc(Fz_long, k, H_LONG)
        ax.plot(full, 'C0-', lw=2, label='full')
        ax.plot(frz, 'C2--', lw=2, label='frozen (common SS)')
        ax.axhline(0, color='k', lw=0.5); ax.axvline(H, color='grey', lw=1, ls=':')
        ax.set_xlabel('quarters'); ax.set_title(ti); ax.legend(fontsize=8)
    fig.suptitle('Persistence of the adoption channel: full vs frozen, 30-year horizon')
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig_frozen_qa_longrun.png', dpi=140, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()
