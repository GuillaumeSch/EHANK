"""Run every paper experiment and write figures to output/."""
import os
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt

from core.model import build_model, run

NUMERAIRE = 'cpi'          # 'cpi' = ARS CPI numeraire (only supported numeraire)
BOOKING = 'import'          # 'import' baseline, or 'domestic' (green sector)

# Reuse the tag-keyed cache across runs; set True to force a full rebuild.
CLEAR_CACHE = False

OUT = 'paper/output'
CDIR = f'cache/cache_{NUMERAIRE}_{BOOKING}'
if CLEAR_CACHE:
    shutil.rmtree(CDIR, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(CDIR, exist_ok=True)
H = 24                       # 24-quarter cumulation window (summary table)
CACHE = {}
MODEL = build_model(NUMERAIRE, booking=BOOKING)
from tools import latex_tables as LT

KEEP = ['y', 'C', 'cE', 'CE_B', 'pi', 'pi_ann', 'D_GREEN', 'D_SWITCH', 'pE_P', 'n',
        'spending', 'nx_gdp', 'PEstar', 'E_supply', 'r_ann', 'w', 'piH_ann',
        'B', 'tauY', 'pE_B_P', 'pE_G_P', 'assets_clearing', 'goods_clearing', 'E_clearing', 'nfares']
SSKEEP = ['alpha_E', 'eta_E', 'pE_B_P', 'pE_G_P', 'D_GREEN', 'eis', 'psi_g'] + \
         [f'beta_{i}' for i in range(3)]

def get(tag, **kw):
    """Solve one experiment, caching (ss scalars, irf) to disk."""
    f = f'{CDIR}/{tag}.pkl'
    if tag in CACHE:
        return CACHE[tag]
    if os.path.exists(f):
        CACHE[tag] = pickle.load(open(f, 'rb'))
        return CACHE[tag]
    ss, irf = run(MODEL, numeraire=NUMERAIRE, booking=BOOKING, **kw)
    irf_kept = {k: np.asarray(irf[k])[:60] for k in KEEP if k in irf}
    d = ({k: float(ss[k]) for k in SSKEEP}, irf_kept)
    pickle.dump(d, open(f, 'wb'))
    CACHE[tag] = d
    return d

def pc(irf, k, h=H):
    return 100 * np.asarray(irf[k])[:h]

def panel(ax, series, k, title, ylab=None):
    for spec in series:
        lab, irf, sty = spec[0], spec[1], spec[2]
        color = spec[3] if len(spec) > 3 else None
        ax.plot(pc(irf, k), lw=2, ls=sty, color=color, label=lab)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('quarters', fontsize=8)
    if ylab:
        ax.set_ylabel(ylab, fontsize=8)
    ax.tick_params(labelsize=8)

def figure(fname, series, keys, suptitle, ncol=3):
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.1 * nrow))
    for ax, (k, t) in zip(np.atleast_1d(axes).flat, keys):
        panel(ax, series, k, t)
    for ax in np.atleast_1d(axes).flat[len(keys):]:
        ax.axis('off')
    np.atleast_1d(axes).flat[0].legend(fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{OUT}/{fname}', dpi=140)
    plt.close(fig)

MACRO = [('y', r'Output $y$'), ('C', r'Consumption $C$'), ('CE_B', r'Brown Energy $c^E_B$'),
         ('pE_B_P', r'Brown energy price $P^E_B/P$'),
         ('D_GREEN', r'Green share $D^{G}$'),
         ('spending', r'Fiscal spending $G$'), ('nx_gdp', r'Net exports / GDP'),
         ('PEstar', r'World energy price $P^{E*}$'), ('E_supply', r'World energy supply $E^*$'),
         ]

# Six-variable panel for the baseline and policy figures (no world/fiscal panels).
MACRO_BASE = [('y', r'Output $y$'), ('C', r'Consumption $C$'),
              ('CE_B', r'Brown energy use $c^E_B$'),
              ('pE_B_P', r'Brown energy price $P^E_B/P$'),
              ('D_GREEN', r'Green share $D^{G}$'),
              ('nx_gdp', r'Net exports / GDP')]
MACRO_POL = [('y', r'Output $y$'), ('C', r'Consumption $C$'),
             ('CE_B', r'Brown energy use $c^E_B$'),
             ('pE_B_P', r'Brown energy price $P^E_B/P$'),
             ('D_GREEN', r'Green share $D^{G}$'),
             ('spending', r'Fiscal outlays $X$')]

print("E0  baseline: price vs supply shock")
p_ad = get('price_none_adoption', shock_kind='price', policy='none', model_variant='adoption')[1]
s_ad = get('supply_none_adoption', shock_kind='supply', policy='none', model_variant='adoption')[1]

figure('fig0_shock.png',
       [('Price Shock', p_ad, '-', 'k'), ('Supply Shock', s_ad, '-', '#48c')],
       MACRO_BASE, '')

print("E1  baseline: price vs supply shock, adoption on/off")
p_ad = get('price_none_adoption', shock_kind='price', policy='none', model_variant='adoption')[1]
p_no = get('price_none_no_adoption', shock_kind='price', policy='none', model_variant='no_adoption')[1]
s_ad = get('supply_none_adoption', shock_kind='supply', policy='none', model_variant='adoption')[1]
s_no = get('supply_none_no_adoption', shock_kind='supply', policy='none', model_variant='no_adoption')[1]

figure('fig1_price_shock.png',
       [('adoption open', p_ad, '-', 'k'), ('adoption frozen (common SS)', p_no, '--', '#c44')],
       MACRO_BASE, '')

figure('fig1_supply_shock.png',
       [('adoption open', s_ad, '-', 'k'), ('adoption frozen (common SS)', s_no, '--', '#c44')],
       MACRO_BASE, '')

print("E2/E3  fiscal policy and its interaction with adoption")
pol, pol_ss = {}, {}
for shock in ['price', 'supply']:
    for p in ['none', 'subsidy', 'transfer']:
        for mv in ['adoption', 'no_adoption']:
            _ss, _irf = get(f'{shock}_{p}_{mv}', shock_kind=shock,
                            policy=p, model_variant=mv)
            pol[(shock, p, mv)] = _irf
            pol_ss[(shock, p, mv)] = _ss

for shock, nm in [('price', 'PRICE'), ('supply', 'SUPPLY')]:
    figure(f'fig2_policy_{shock}.png',
           [('no policy', pol[(shock, 'none', 'adoption')], '-', 'k'),
            ('price cap', pol[(shock, 'subsidy', 'adoption')], '--', '#c44'),
            ('Slutsky transfer', pol[(shock, 'transfer', 'adoption')], '-.', '#48c')],
           MACRO_POL, '')

PLABEL = {'none': 'no policy', 'subsidy': 'price cap', 'transfer': 'Slutsky transfer'}
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for j, (shock, nm) in enumerate([('price', 'PRICE'), ('supply', 'SUPPLY')]):
    for i, k in enumerate(['D_GREEN', 'y', 'C']):
        ax = axes[j, i]
        for p, sty in [('none', '-'), ('subsidy', '--'), ('transfer', ':')]:
            ax.plot(pc(pol[(shock, p, 'adoption')], k), lw=2, ls=sty, label=PLABEL[p])
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(f'{nm}: {k}', fontsize=10)
        ax.set_xlabel('quarters', fontsize=8)
axes[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_policy_adoption.png', dpi=140); plt.close(fig)

print("E4  monetary policy")
mon = {m: get(f'supply_mon_{m}', shock_kind='supply', policy='none',
              model_variant='adoption', monetary=m)[1]
       for m in ['real_rate', 'taylor']}
figure('fig4_monetary.png',
       [('constant real rate', mon['real_rate'], '-', 'k'), ('Taylor rule', mon['taylor'], '--', '#c44')],
       MACRO_BASE, '')

print("E5  adoption-channel decomposition")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, (k, t) in zip(axes, [('y', 'Output'), ('C', 'Consumption'),
                             ('nx_gdp', 'Net exports / GDP')]):
    ax.plot(pc(p_ad, k) - pc(p_no, k), lw=2, color='k', label='price shock')
    ax.plot(pc(s_ad, k) - pc(s_no, k), lw=2, color='#48c', label='supply shock')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'{t}: adoption minus no-adoption', fontsize=10)
    ax.set_xlabel('quarters', fontsize=8)
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig5_decomposition.png', dpi=140); plt.close(fig)

def cum(irf, k, h=H):
    return float(np.sum(np.asarray(irf[k])[:h]))

rows = []
for shock in ['price', 'supply']:
    for p in ['none', 'subsidy', 'transfer']:
        a, n_ = pol[(shock, p, 'adoption')], pol[(shock, p, 'no_adoption')]
        rows.append(dict(
            shock=shock, policy=p,
            y0=100 * float(np.asarray(a['y'])[0]),
            y_cum=100 * cum(a, 'y'),
            pi0=100 * float(np.asarray(a['pi_ann'])[0]),
            dG_peak=100 * float(np.max(np.asarray(a['D_GREEN'])[:H])),
            cE_cum=100 * cum(a, 'cE'),
            fiscal=100 * cum(a, 'spending'),
            adopt_y=100 * (cum(a, 'y') - cum(n_, 'y')),
        ))

hdr = (f"{'shock':>7s} {'policy':>9s} {'y(0)%':>8s} {'cum y':>8s} {'pi(0)':>8s} "
       f"{'peak DG':>9s} {'cum cE':>8s} {'fiscal':>8s} {'adopt->y':>9s}")
lines = [hdr, '-' * len(hdr)]
for r in rows:
    lines.append(f"{r['shock']:>7s} {r['policy']:>9s} {r['y0']:8.3f} {r['y_cum']:8.2f} "
                 f"{r['pi0']:8.3f} {r['dG_peak']:9.3f} {r['cE_cum']:8.2f} "
                 f"{r['fiscal']:8.3f} {r['adopt_y']:9.3f}")
table = '\n'.join(lines)
print('\n' + table)
open(f'{OUT}/summary_table_{NUMERAIRE}_{BOOKING}.txt', 'w').write(table + '\n')

tex_rows = [dict(shock=r['shock'], policy=r['policy'], y0=r['y0'], ycum=r['y_cum'],
                 pi0=r['pi0'], dG_peak=r['dG_peak'], fiscal=r['fiscal'],
                 adopt_y=r['adopt_y'])
           for r in rows]
tex_path = LT.summary_table(f'{OUT}/tab_summary_{NUMERAIRE}_{BOOKING}.tex',
                            tex_rows, H, NUMERAIRE, BOOKING,
                            label=('tab:summary' if BOOKING == 'import'
                                  else f'tab:summary_{BOOKING}'))

pickle.dump({k: v for k, v in CACHE.items()}, open(f'{OUT}/irfs_{NUMERAIRE}_{BOOKING}.pkl', 'wb'))
