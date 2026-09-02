"""ETS vs baseline: IRFs and the cross-sectional consumption split."""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from core.model import build_model, run
from tools.latex_tables import write_table

H = 24
TAU_B = 0.10
NUMERAIRE, BOOKING = 'cpi', 'import'
OUT = 'paper/output'
CACHE = 'cache/cache_ets_xs'
CLEAR_CACHE = False
SHOCKS = ('price', 'supply')

def _cached_run(model, key, **kw):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, key + '.pkl')
    if os.path.exists(f) and not CLEAR_CACHE:
        with open(f, 'rb') as fh:
            return pickle.load(fh)
    ss, irf = run(model, **kw)
    out = ({k: np.asarray(v) for k, v in ss.items() if np.ndim(v) == 0},
           {k: np.asarray(v) for k, v in irf.items()})
    with open(f, 'wb') as fh:
        pickle.dump(out, fh)
    return out

def get_runs(model):
    """(shock, economy, variant) -> (ss, irf)."""
    runs = {}
    for shk in SHOCKS:
        for var in ('adoption', 'no_adoption'):
            tag = 'frozen' if var == 'no_adoption' else 'adoption'
            runs[(shk, 'base', tag)] = _cached_run(
                model, f'{shk}_base_{tag}_{BOOKING}', shock_kind=shk, policy='none',
                model_variant=var, numeraire=NUMERAIRE, booking=BOOKING)
            runs[(shk, 'ets', tag)] = _cached_run(
                model, f'{shk}_ets{int(100*TAU_B)}_{tag}_{BOOKING}', shock_kind=shk,
                policy='none', model_variant=var, ets=True,
                ets_kwargs=dict(tau_b=TAU_B, recycle='rebate'),
                numeraire=NUMERAIRE, booking=BOOKING)
    return runs

def decompose(ss, irf, irf_f):
    """Two-layer split of dC by current technology group."""
    mG = float(ss['D_GREEN']); mS = float(ss['D_SWITCH']); mB = 1.0 - mG
    m = {'BB': mB, 'GB': mS, 'GG': mG - mS}
    CS = float(ss['C_SWITCH'])
    cb = {'BB': float(ss['C_BROWN']) / mB, 'GB': CS / mS,
          'GG': (float(ss['C_GREEN']) - CS) / (mG - mS)}

    def groups(x):
        return {'BB': np.asarray(x['C_BROWN']), 'GB': np.asarray(x['C_SWITCH']),
                'GG': np.asarray(x['C_GREEN']) - np.asarray(x['C_SWITCH'])}
    dCg, dCg_f = groups(irf), groups(irf_f)
    # frozen choice probabilities, not masses; net residual composition drift out at SS levels
    dmG_f, dmS_f = np.asarray(irf_f['D_GREEN']), np.asarray(irf_f['D_SWITCH'])
    dm_f = {'BB': -dmG_f, 'GB': dmS_f, 'GG': dmG_f - dmS_f}
    assert np.max(np.abs(dmG_f)) < 0.05 * np.max(np.abs(np.asarray(irf['D_GREEN'])))
    dcbar_f = {j: (dCg_f[j] - cb[j] * dm_f[j]) / m[j] for j in m}   # per-capita
    migr = {j: dCg[j] - dCg_f[j] for j in m}             # adoption/migration term
    dC, dC_f = np.asarray(irf['C']), np.asarray(irf_f['C'])
    tot = sum(dCg_f.values()) + sum(migr.values())
    assert np.max(np.abs(tot - dC)) < 1e-9, 'decomposition identity fails'
    return dict(m=m, cbar=cb, dcbar_f=dcbar_f, fixed=dCg_f, migr=migr,
                dC=dC, dC_f=dC_f, dmG=np.asarray(irf['D_GREEN']))

def _cum(x):
    return 100 * float(np.sum(np.asarray(x)[:H]))

IRF_VARS = [
    ('y', r'Output $y$'), ('C', r'Consumption $C$'),
    ('CE_B', r'Brown energy $C^B_E$'), ('pi_ann', r'CPI inflation (ann.)'),
    ('D_GREEN', r'Green share $D^G$'), ('D_SWITCH', r'Switchers $D^{sw}$'),
    ('pE_B_P', r'Brown price $P^E_B/P$'), ('nx_gdp', r'Net exports / GDP'),
]

def fig_irf(runs, shk):
    ssb, irfb = runs[(shk, 'base', 'adoption')]
    sse, irfe = runs[(shk, 'ets', 'adoption')]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2))
    for ax, (k, t) in zip(axes.flat, IRF_VARS):
        ax.plot(100 * irfb[k][:H], lw=2, color='k',
                label=rf'baseline ($D^G_{{ss}}$={100*float(ssb["D_GREEN"]):.1f}%)')
        ax.plot(100 * irfe[k][:H], lw=2, ls='--', color='C2',
                label=rf'ETS $\tau_b$={TAU_B} ($D^G_{{ss}}$={100*float(sse["D_GREEN"]):.1f}%)')
        ax.axhline(0, color='grey', lw=0.5)
        ax.set_title(t, fontsize=10); ax.tick_params(labelsize=8)
        ax.set_xlabel('quarters', fontsize=8)
    axes.flat[0].legend(fontsize=8, frameon=False)
    axes.flat[0].set_ylabel('level dev. from own SS, x100', fontsize=8)
    axes.flat[4].set_ylabel('level dev. from own SS, x100', fontsize=8)
    lab = 'brown-price shock' if shk == 'price' else 'brown-supply shock'
    fig.suptitle(f'Same {lab}, baseline vs ex-ante ETS steady state ({BOOKING} booking, no policy)', y=1.0)
    fig.tight_layout()
    f = os.path.join(OUT, f'fig_ets_irf_{shk}_{BOOKING}.pdf')
    fig.savefig(f, dpi=140, bbox_inches='tight')
    return f

def fig_cross_section(runs, shk):
    ssb, irfb = runs[(shk, 'base', 'adoption')]
    sse, irfe = runs[(shk, 'ets', 'adoption')]
    db = decompose(ssb, irfb, runs[(shk, 'base', 'frozen')][1])
    de = decompose(sse, irfe, runs[(shk, 'ets', 'frozen')][1])
    col = {'BB': 'saddlebrown', 'GB': 'orange', 'GG': 'seagreen'}
    lab = {'BB': 'brown incumbents (BB)', 'GB': 'switchers (GB)',
           'GG': 'green incumbents (GG)'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    # per-capita c by group, % of own SS
    ax = axes[0]
    for d, ls, name in ((db, '-', 'baseline'), (de, '--', 'ETS')):
        for j in ('BB', 'GG'):
            ax.plot(100 * d['dcbar_f'][j][:H] / d['cbar'][j], lw=2, ls=ls, color=col[j],
                    label=f'{lab[j]}, {name}')
    ax.axhline(0, color='grey', lw=0.5)
    ax.set_title('Per-capita $c$ by technology, fixed populations (% of own SS)', fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    # contributions to dC, level x100
    for ax, d, name in ((axes[1], db, 'baseline'), (axes[2], de, f'ETS $\\tau_b$={TAU_B}')):
        ax.plot(100 * d['fixed']['BB'][:H], lw=2, color=col['BB'], label='brown, fixed pop.')
        ax.plot(100 * (d['fixed']['GB'] + d['fixed']['GG'])[:H], lw=2, color=col['GG'],
                label='green, fixed pop.')
        ax.plot(100 * sum(d['migr'].values())[:H], lw=2, color='C0', ls='-.',
                label='adoption / migration term')
        ax.plot(100 * d['dC'][:H], lw=2.5, color='k', ls='--', label='total $dC$')
        ax.axhline(0, color='grey', lw=0.5)
        ax.set_title(f'{name}: contributions to $dC$ (x100)', fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    for a in axes:
        a.tick_params(labelsize=8); a.set_xlabel('quarters', fontsize=8)
    sh = 'brown-price shock' if shk == 'price' else 'brown-supply shock'
    fig.suptitle(f'Consumption by technology group, {sh} ({BOOKING} booking, no policy)', y=1.02)
    fig.tight_layout()
    f = os.path.join(OUT, f'fig_cross_section_{shk}_{BOOKING}.pdf')
    fig.savefig(f, dpi=140, bbox_inches='tight')
    return f, db, de

def main():
    os.makedirs(OUT, exist_ok=True)
    model = build_model(NUMERAIRE, booking=BOOKING)
    runs = get_runs(model)

    rows = []
    for shk in SHOCKS:
        f1 = fig_irf(runs, shk)
        f2, db, de = fig_cross_section(runs, shk)
        for name, d, (ss, irf) in (('baseline', db, runs[(shk, 'base', 'adoption')]),
                                   (f'ETS $\\tau_b={TAU_B}$', de, runs[(shk, 'ets', 'adoption')])):
            pk = 100 * float(np.max(irf['D_GREEN'][:H]))
            pcB = _cum(d['dcbar_f']['BB']) / d['cbar']['BB']
            pcG = _cum(d['dcbar_f']['GG']) / d['cbar']['GG']
            fixB, fixG = _cum(d['fixed']['BB']), _cum(d['fixed']['GB'] + d['fixed']['GG'])
            mig = _cum(sum(d['migr'].values()))
            print(f'\n{shk} / {name}: D_GREEN_ss={100*float(ss["D_GREEN"]):.2f}%  '
                  f'cbar BB/GB/GG = {d["cbar"]["BB"]:.3f}/{d["cbar"]["GB"]:.3f}/{d["cbar"]["GG"]:.3f}  '
                  f'peak dD_GREEN={pk:.2f}pp')
            print(f'  cum dC (H={H})               {_cum(d["dC"]):9.2f}   [frozen: {_cum(d["dC_f"]):.2f}]')
            print(f'  brown, fixed pop.            {fixB:9.2f}   per-capita % own SS {pcB:8.2f}')
            print(f'  green, fixed pop.            {fixG:9.2f}   per-capita % own SS {pcG:8.2f}')
            print(f'  adoption/migration term      {mig:9.2f}')
            rows.append([shk, name, f'{100*float(ss["D_GREEN"]):.1f}', f'{_cum(d["dC"]):.1f}',
                         f'{fixB:.1f}', f'{fixG:.1f}', f'{mig:.1f}',
                         f'{pcB:.1f}', f'{pcG:.1f}', f'\\textbf{{{pk:.2f}}}'])

    tpath = os.path.join(OUT, f'tab_cross_section_{BOOKING}.tex')
    write_table(
        tpath, colspec='llrrrrrrrr',
        header=['Shock', 'Economy', r'$D^G_{ss}$\%', r'$\sum dC$',
                'brown (fixed)', 'green (fixed)', 'adoption term',
                r'$\bar c_{B}$\%', r'$\bar c_{G}$\%', r'peak $D^G$'],
        rows=rows,
        caption=(f'Consumption response by technology group over $H={H}$ '
                 f'({BOOKING} booking, no policy). Cumulative level deviations '
                 f'$\\times 100$: $dC$ = brown (fixed population) + green (fixed '
                 f'population) + adoption term, where the fixed-population terms '
                 f'come from the common-steady-state frozen-choice counterfactual '
                 f'and the adoption term is full minus frozen. $\\bar c_j$\\%: '
                 f'cumulative per-capita consumption response of brown / green '
                 f'incumbents in \\% of their own steady state, fixed populations.'),
        label=f'tab:cross_section_{BOOKING}', midrule_after={1})

if __name__ == '__main__':
    main()
