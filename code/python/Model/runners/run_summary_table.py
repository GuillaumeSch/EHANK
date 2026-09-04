"""Summary table for the five additions."""
import os
import numpy as np

from core.model import build_model, run
from core.welfare import cev_total
from tools.latex_tables import write_table

H = 24
TAU_B = 0.10
NUMERAIRE, BOOKING = 'cpi', 'import'
OUT = 'paper/output'

def _c0(irf, k):
    return 100 * float(np.asarray(irf[k])[0])

def _cum(irf, k):
    return 100 * float(np.sum(np.asarray(irf[k])[:H]))

def _peakDG(irf):
    return 100 * float(np.max(np.asarray(irf['D_GREEN'])[:H]))

def main():
    os.makedirs(OUT, exist_ok=True)
    model = build_model(NUMERAIRE, booking=BOOKING)

    # Panel A: price-shock policies
    ss_base, irf_none = run(model, shock_kind='price', policy='none',
                            numeraire=NUMERAIRE, booking=BOOKING)
    runs = {'none': (ss_base, irf_none)}
    for p in ('subsidy', 'transfer', 'transfer_flat', 'green'):
        runs[p] = run(model, shock_kind='price', policy=p,
                      numeraire=NUMERAIRE, booking=BOOKING)
    ss_ets, irf_ets = run(model, shock_kind='price', policy='none', ets=True,
                          ets_kwargs=dict(tau_b=TAU_B, recycle='rebate'),
                          numeraire=NUMERAIRE, booking=BOOKING)
    runs['ets'] = (ss_ets, irf_ets)

    labels = {'none': 'Laissez-faire', 'subsidy': 'Ex-post cap',
              'transfer': 'Slutsky transfer', 'transfer_flat': 'Flat transfer',
              'green': 'Green subsidy', 'ets': f'Ex-ante ETS ($\\tau_b={TAU_B}$)'}
    fisc_key = {'subsidy': 'Subsidy', 'transfer': 'Ttargeted',
                'transfer_flat': 'Ttargeted', 'green': 'Subsidy_green'}

    print(f'\n=== Panel A: price shock (baseline D_GREEN={float(ss_base["D_GREEN"]):.4f}, '
          f'ETS D_GREEN={float(ss_ets["D_GREEN"]):.4f}) ===')
    hdr = f'{"scenario":<22s}{"y(0)":>9s}{"cumY":>9s}{"peakDG":>9s}{"grossF":>10s}{"CEV":>9s}'
    print(hdr); print('-' * len(hdr))
    rowsA = []
    for k in ('none', 'subsidy', 'transfer', 'transfer_flat', 'green', 'ets'):
        ss_k, irf_k = runs[k]
        pre = ss_ets if k == 'ets' else ss_base
        cev, _ = cev_total(ss_base, pre, irf_k)
        y0, cumY, pk = _c0(irf_k, 'y'), _cum(irf_k, 'y'), _peakDG(irf_k)
        if k == 'none':
            gf = 0.0
        elif k == 'ets':
            gf = H * float(ss_ets['R_carbon'])
        else:
            gf = _cum(irf_k, fisc_key[k])
        print(f'{labels[k]:<22s}{y0:9.3f}{cumY:9.2f}{pk:9.3f}{gf:10.2f}{100*cev:9.3f}')
        rowsA.append([labels[k], f'{y0:.2f}', f'{cumY:.1f}',
                      f'\\textbf{{{pk:.2f}}}', f'{gf:.1f}', f'{100*cev:.3f}'])

    # Panel B -> monetary IRF figure (fig_monetary.pdf); reported in Section 4, not as a table panel
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print('\n=== Monetary-policy shock: IRF figure (100bp annualised tightening) ===')
    mirfs = {}
    for rule in ('taylor', 'real_rate'):
        _, mirfs[rule] = run(model, shock_kind='monetary', policy='none',
                             monetary=rule, numeraire=NUMERAIRE, booking=BOOKING)
    def pcm(irf, k, h=H):
        return 100 * np.asarray(irf[k])[:h]
    panels = [('inom_ann', r'Nominal rate $i$ (ann., pp)'),
              ('y', r'Output $y$ (\%)'),
              ('pi_ann', r'Inflation $\pi$ (ann., pp)'),
              ('C', r'Consumption $C$ (\%)')]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, (k, ti) in zip(axes, panels):
        ax.plot(pcm(mirfs['taylor'], k), 'C0-', lw=2, label='Taylor rule')
        ax.plot(pcm(mirfs['real_rate'], k), 'C3--', lw=2, label='Constant real rate (ARS)')
        ax.axhline(0, color='k', lw=0.5); ax.set_title(ti, fontsize=10)
        ax.set_xlabel('quarters', fontsize=8); ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fpath = os.path.join(OUT, f'fig_monetary_{BOOKING}.pdf')
    fig.savefig(fpath, dpi=140, bbox_inches='tight'); plt.close(fig)
    for rule, name in (('taylor', 'Taylor rule'), ('real_rate', 'Constant real rate')):
        im = mirfs[rule]
        print(f'  {name:<22s} i(0)={_c0(im,"inom_ann"):+.2f}  y(0)={_c0(im,"y"):.3f}'
              f'  pi(0)={_c0(im,"pi_ann"):.3f}  C(0)={_c0(im,"C"):.3f}')

    # write Panel A only
    tpath = os.path.join(OUT, f'tab_summary_{BOOKING}.tex')
    write_table(
        tpath, colspec='lrrrrr',
        header=['Scenario', r'$y(0)$\%', r'$\sum y$\%', r'peak $D_G$',
                r'gross fisc.', r'CEV\%'],
        rows=rowsA,
        caption='Price-shock policy responses',
        notes=(f'$y(0)$ is impact output, $\\sum y$ cumulative '
               f'output over $H={H}$, peak $D_G$ the peak green share, gross '
               f'fiscal the total disbursement, and CEV the total '
               f'consumption-equivalent variation relative to the no-ETS baseline '
               f'steady state.'),
        label=f'tab:summary_price_{BOOKING}')

if __name__ == '__main__':
    main()
