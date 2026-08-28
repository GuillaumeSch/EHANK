"""Route A -- does ex-ante greening pay off as the crisis becomes more persistent?

The one-off result (run_exante_expost.py) is that ex-post capping dominates
ex-ante greening. The natural defence of the ex-ante case is persistence: a
longer-lived price shock should let the greener economy's lower exposure
compound over more periods. This runner tests that directly by sweeping the
shock half-life and re-solving ONLY the impulse response (the steady state does
not move with persistence), via the general-equilibrium Jacobian G computed once
per steady state.

Metrics, all as total CEV vs the common no-ETS baseline SS:
  CEV_LF   laissez-faire crisis
  CEV_cap  ex-post price cap
  CEV_ETS  ex-ante ETS economy (tau_b), no crisis policy
and the decomposition of the ex-ante number into its constant STANDING part and
the crisis-only 'greening dividend' = CEV_ETS_crisis - CEV_LF, which is the
object that would have to GROW with persistence for the ex-ante case to work.

Writes output/fig_persistence_<booking>.pdf and output/tab_persistence_<booking>.tex.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from core.model import build_model, run, shock_price, td_unknowns_targets, T
from core.welfare import cev_total
from tools.latex_tables import write_table

H = 24
TAU_B = 0.10
NUMERAIRE, BOOKING = 'core', 'import'
HALF_LIVES = [4, 8, 12, 16, 24, 32, 48]
OUT = 'paper/output'
CEV_OUTS = ['UTIL_0', 'UTIL_1', 'UTIL_2', 'n']
FIG_OUTS = ['C', 'D_GREEN', 'y']


def _G(model, ss, u, t):
    return model.solve_jacobian(ss, u, t, inputs=['PEstar_shock'],
                                outputs=CEV_OUTS + FIG_OUTS, T=T)


def _irf(G, shk):
    return {o: G[o]['PEstar_shock'] @ shk for o in G}


def main():
    os.makedirs(OUT, exist_ok=True)
    model = build_model(NUMERAIRE, booking=BOOKING)

    # steady states solved once (persistence does not move them)
    ss_base, _ = run(model, shock_kind='price', policy='none',
                     numeraire=NUMERAIRE, booking=BOOKING)
    ss_cap, _ = run(model, shock_kind='price', policy='subsidy',
                    numeraire=NUMERAIRE, booking=BOOKING)
    ss_ets, _ = run(model, shock_kind='price', policy='none', ets=True,
                    ets_kwargs=dict(tau_b=TAU_B, recycle='rebate'),
                    numeraire=NUMERAIRE, booking=BOOKING)

    u0, t0 = td_unknowns_targets(BOOKING, ets=False)
    uE, tE = td_unknowns_targets(BOOKING, ets=True)
    G_base, G_cap, G_ets = (_G(model, ss_base, u0, t0),
                            _G(model, ss_cap, u0, t0),
                            _G(model, ss_ets, uE, tE))

    # standing CEV of the ETS SS (persistence-independent), for the decomposition
    zero = {k: np.zeros(H) for k in CEV_OUTS}
    chi_stand, _ = cev_total(ss_base, ss_ets, zero)

    print(f'\ntau_b={TAU_B}  ETS D_GREEN={float(ss_ets["D_GREEN"]):.4f}  '
          f'standing CEV={100*chi_stand:+.4f}%\n')
    hdr = f'{"half_life":>10s}{"CEV_LF":>10s}{"CEV_cap":>10s}{"CEV_ETS":>10s}' \
          f'{"greenDiv":>10s}{"capProt":>10s}'
    print(hdr); print('-' * len(hdr))

    rows_tab, series = [], {k: [] for k in
                            ('lf', 'cap', 'ets', 'greendiv', 'capprot')}
    irfs_16 = {}
    for hl in HALF_LIVES:
        shk = shock_price(half_life=hl)['PEstar_shock']
        i_lf, i_cap, i_ets = _irf(G_base, shk), _irf(G_cap, shk), _irf(G_ets, shk)
        clf, _ = cev_total(ss_base, ss_base, i_lf)
        ccap, _ = cev_total(ss_base, ss_base, i_cap)
        cets, _ = cev_total(ss_base, ss_ets, i_ets)
        greendiv = (cets - chi_stand) - clf   # crisis-only greening dividend
        capprot = ccap - clf                  # crisis-only cap protection
        print(f'{hl:10d}{100*clf:10.4f}{100*ccap:10.4f}{100*cets:10.4f}'
              f'{100*greendiv:10.4f}{100*capprot:10.4f}')
        rows_tab.append([f'{hl}', f'{100*clf:.3f}', f'{100*ccap:.3f}',
                         f'{100*cets:.3f}', f'{100*greendiv:.4f}',
                         f'{100*capprot:.3f}'])
        for k, v in zip(('lf', 'cap', 'ets', 'greendiv', 'capprot'),
                        (clf, ccap, cets, greendiv, capprot)):
            series[k].append(100 * v)
        if hl == 16:
            irfs_16 = dict(lf=i_lf, cap=i_cap, ets=i_ets)

    # =========================================================================
    # FIGURE: (left) CEV vs persistence; (right) crisis-only protection vs persist.
    # =========================================================================
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
    axL.plot(HALF_LIVES, series['lf'], 'o-', label='Laissez-faire')
    axL.plot(HALF_LIVES, series['cap'], 's-', label='Ex-post cap')
    axL.plot(HALF_LIVES, series['ets'], '^-', label='Ex-ante ETS')
    axL.set_xlabel('shock half-life (quarters)')
    axL.set_ylabel(r'total CEV vs baseline SS (\%)')
    axL.set_title('Welfare by persistence')
    axL.axhline(0, color='k', lw=0.5); axL.legend(fontsize=9)
    axR.plot(HALF_LIVES, series['capprot'], 's-', label='Ex-post cap protection')
    axR.plot(HALF_LIVES, series['greendiv'], '^-', label='Ex-ante greening dividend')
    axR.set_xlabel('shock half-life (quarters)')
    axR.set_ylabel(r'crisis-only CEV gain vs LF (\%)')
    axR.set_title('Crisis protection vs persistence')
    axR.axhline(0, color='k', lw=0.5); axR.legend(fontsize=9)
    fig.tight_layout()
    fpath = os.path.join(OUT, f'fig_persistence_{BOOKING}.pdf')
    fig.savefig(fpath, dpi=140, bbox_inches='tight')
    print(f'\n[figure] {fpath}')

    # =========================================================================
    # LATEX TABLE
    # =========================================================================
    tpath = os.path.join(OUT, f'tab_persistence_{BOOKING}.tex')
    write_table(
        tpath, colspec='rrrrrr',
        header=[r'half-life', r'CEV$_{\mathrm{LF}}$', r'CEV$_{\mathrm{cap}}$',
                r'CEV$_{\mathrm{ETS}}$', r'green div.', r'cap prot.'],
        rows=rows_tab,
        caption=(f'Route A: welfare by shock persistence (import booking, '
                 f'$\\tau_b={TAU_B}$). Total CEV vs the no-ETS baseline SS. '
                 f'The crisis-only greening dividend '
                 f'(CEV$_{{\\mathrm{{ETS}}}}$ minus the standing '
                 f'{100*chi_stand:+.3f}\\% minus CEV$_{{\\mathrm{{LF}}}}$) does '
                 f'not grow with persistence, while the cap protection does.'),
        label=f'tab:persistence_{BOOKING}')
    print(f'[table]  {tpath}')


if __name__ == '__main__':
    main()
