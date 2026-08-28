#%%
"""Run every experiment in the paper and write figures to output/.

Experiments
  E1  Baseline: price shock vs supply shock, adoption on/off
  E2  Fiscal policy: no policy / price cap / Slutsky transfer  (Bayer, Langot)
  E3  The policy-adoption interaction: does shielding kill the green margin?
  E4  Monetary policy: constant real rate vs Taylor rule       (ARS)
  E5  Adoption-channel decomposition
  E8  CPI measurement gap left by the brown-anchored deflator (Option C)

CACHES ARE TAGGED BY NUMERAIRE. An IRF computed under the CPI numeraire
differs from one computed under the domestic-good numeraire by ~0.03% -- far
too little to notice by eye, far too much to leave mixed in one table. Never
reuse a cache across numeraires; that is what the tag prevents.
"""
import os
import shutil
import pickle
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.model import build_model, run
from core.deflator import true_inflation

NUMERAIRE = 'core'          # 'core' = domestic good (default), 'cpi' = ARS
BOOKING = 'import'          # 'import' baseline, or 'domestic' (green sector)

# The cache is keyed only by tag string, not by model/calibration content
# (see get() below), so it goes stale silently on any change to model.py,
# calibration.py or the blocks. Wipe it on every run rather than relying on
# a manual `rm -rf cache_*` before each launch. Set to False only if you are
# deliberately resuming a crashed run with an unchanged model.
CLEAR_CACHE = True

OUT = 'paper/output'
CDIR = f'cache/cache_{NUMERAIRE}_{BOOKING}'
if CLEAR_CACHE:
    shutil.rmtree(CDIR, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(CDIR, exist_ok=True)
H = 24                       # matches the paper's "cumulative sums over 24
                              # quarters" convention (Table~tab:summary caption)
CACHE = {}
MODEL = build_model(NUMERAIRE, booking=BOOKING)
from tools import latex_tables as LT

# pi, pE_B_P, pE_G_P and D_GREEN are needed by the deflator diagnostic.
KEEP = ['y', 'C', 'cE', 'CE_DUR_B', 'pi', 'pi_ann', 'D_GREEN', 'D_SWITCH', 'pE_P', 'n',
        'spending', 'nx_gdp', 'PEstar', 'E_supply', 'r_ann', 'w', 'piH_ann',
        'B', 'tauY', 'pE_B_P', 'pE_G_P', 'assets_clearing', 'goods_clearing', 'E_clearing', 'nfares']
SSKEEP = ['alpha_E', 'eta_E', 'pE_B_P', 'pE_G_P', 'D_GREEN', 'eis', 'psi_g'] + \
         [f'beta_{i}' for i in range(3)]


def get(tag, **kw):
    """Solve one experiment, caching (steady-state scalars, IRF) to disk so
    the matrix can be run in resumable chunks.

    IRFs are stored as RAW level deviations from steady state (the 100x is
    applied only at report time, in pc()/cum()), matching the Section 4
    convention and tab:dose / tab:signmap.

    model_variant='no_adoption' is the COMMON-STEADY-STATE counterfactual (see
    model.frozen_model / frozen_adoption.py): 'adoption' and 'no_adoption' share
    a bit-identical steady state and differ only in the transition (the
    adoption choice does not respond to the shock under 'no_adoption'). Raw
    level deviations (no per-series normalisation by the steady state) are kept
    throughout for consistency with the paper's other tables (summary vs
    signmap)."""
    f = f'{CDIR}/{tag}.pkl'
    if tag in CACHE:
        return CACHE[tag]
    if os.path.exists(f):
        CACHE[tag] = pickle.load(open(f, 'rb'))
        return CACHE[tag]
    print(f"  solving {tag} ...", flush=True)
    ss, irf = run(MODEL, numeraire=NUMERAIRE, booking=BOOKING, **kw)
    irf_kept = {k: np.asarray(irf[k])[:60] for k in KEEP if k in irf}
    d = ({k: float(ss[k]) for k in SSKEEP}, irf_kept)
    pickle.dump(d, open(f, 'wb'))
    CACHE[tag] = d
    return d


def pc(irf, k, h=H):
    return 100 * np.asarray(irf[k])[:h]


def panel(ax, series, k, title, ylab=None):
    for lab, irf, sty in series:
        ax.plot(pc(irf, k), lw=2, ls=sty, label=lab)
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
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{OUT}/{fname}', dpi=140)
    plt.close(fig)
    print(f"  -> {OUT}/{fname}")


MACRO = [('y', r'Output $y$'), ('C', r'Consumption $C$'), ('CE_DUR_B', r'Brown Energy $c^E_B$'),
         #('pi_ann', r'CPI inflation (ann.)'),
         # ('n', r'Hours $n$'),
         ('pE_B_P', r'Brown energy price $P^E_B/P$'),
         ('D_GREEN', r'Green share $D^{G}$'), 
         #('D_SWITCH', r'Switchers $D^{sw}$'),
         ('spending', r'Fiscal spending $G$'), ('nx_gdp', r'Net exports / GDP'),
         ('PEstar', r'World energy price $P^{E*}$'), ('E_supply', r'World energy supply $E^*$'),
         #('r_ann', r'Interest rate (ann.)'), 
         #('w', r'Wage $w$'),
         #('piH_ann', r'Domestic-good inflation (ann.)'), 
         #('B', r'Bond holdings $B$'),
         #('tauY', r'Tax revenue / GDP $\tau^Y$'),
        #('assets_clearing', r'assets_clearing'), ('goods_clearing', r'goods_clearing'), ('nfares', r'nfares'),
         ]

# =============================================================================
print("E0  baseline: price vs supply shock")
# NOTE: these are the same four experiments E2 needs. They previously carried
# separate cache tags and were therefore solved twice; the tags are now shared.
p_ad = get('price_none_adoption', shock_kind='price', policy='none', model_variant='adoption')[1]
s_ad = get('supply_none_adoption', shock_kind='supply', policy='none', model_variant='adoption')[1]

figure('fig0_shock.png',
       [('Price Shock', p_ad, '-'), ('Supply Shock', s_ad, '-')],
       MACRO, 'E0a. Brown energy PRICE shock vs SUPPLY shock')


# =============================================================================
print("E1  baseline: price vs supply shock, adoption on/off")
# NOTE: these are the same four experiments E2 needs. They previously carried
# separate cache tags and were therefore solved twice; the tags are now shared.
p_ad = get('price_none_adoption', shock_kind='price', policy='none', model_variant='adoption')[1]
p_no = get('price_none_no_adoption', shock_kind='price', policy='none', model_variant='no_adoption')[1]
s_ad = get('supply_none_adoption', shock_kind='supply', policy='none', model_variant='adoption')[1]
s_no = get('supply_none_no_adoption', shock_kind='supply', policy='none', model_variant='no_adoption')[1]

figure('fig1_price_shock.png',
       [('adoption open', p_ad, '-'), ('adoption frozen (common SS)', p_no, '--')],
       MACRO, 'E1a. Brown energy PRICE shock (ARS-style, elastic supply)')

figure('fig1_supply_shock.png',
       [('adoption open', s_ad, '-'), ('adoption frozen (common SS)', s_no, '--')],
       MACRO, 'E1b. Brown energy SUPPLY shock, -10% for 6q (Bayer-style, fixed quantity)')

# =============================================================================
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
           [('no policy', pol[(shock, 'none', 'adoption')], '-'),
            ('price cap', pol[(shock, 'subsidy', 'adoption')], '--'),
            ('Slutsky transfer', pol[(shock, 'transfer', 'adoption')], ':')],
           MACRO, f'E2. Fiscal response to the {nm} shock (adoption margin open)')

# The paper's headline: policy x adoption interaction
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
fig.suptitle('E3. Does shielding consumers shut down the green adoption margin?')
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_policy_adoption.png', dpi=140); plt.close(fig)
print(f"  -> {OUT}/fig3_policy_adoption.png")

# =============================================================================
print("E4  monetary policy")
mon = {m: get(f'supply_mon_{m}', shock_kind='supply', policy='none',
              model_variant='adoption', monetary=m)[1]
       for m in ['real_rate', 'taylor']}
figure('fig4_monetary.png',
       [('constant real rate', mon['real_rate'], '-'), ('Taylor rule', mon['taylor'], '--')],
       MACRO, 'E4. Monetary policy and the supply shock')

# =============================================================================
print("E5  adoption-channel decomposition")
fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
for ax, (k, t) in zip(axes, [('y', 'Output'), ('C', 'Consumption'),
                             ('cE', 'Energy use'), ('nx_gdp', 'Net exports / GDP')]):
    ax.plot(pc(p_ad, k) - pc(p_no, k), lw=2, label='price shock')
    ax.plot(pc(s_ad, k) - pc(s_no, k), lw=2, ls='--', label='supply shock')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'{t}: adoption minus no-adoption', fontsize=10)
    ax.set_xlabel('quarters', fontsize=8)
axes[0].legend(fontsize=8)
fig.suptitle('E5. Contribution of the green adoption margin')
fig.tight_layout(); fig.savefig(f'{OUT}/fig5_decomposition.png', dpi=140); plt.close(fig)
print(f"  -> {OUT}/fig5_decomposition.png")


# =============================================================================
# SUMMARY TABLE
# =============================================================================
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
print(f"  -> {tex_path}")


# =============================================================================
# E8. CPI MEASUREMENT GAP (Option C diagnostic, no DAG feedback)
# =============================================================================
print("\nE8  brown-anchored deflator vs share-weighted deflator")
hdr2 = (f"{'shock':>7s} {'policy':>9s} {'phi_ss':>9s} {'lvl gap %':>10s} "
        f"{'pi gap pp':>10s} {'|pi| pp':>9s} {'ratio':>8s}")
lines2 = [hdr2, '-' * len(hdr2)]
defl = {}
for shock in ['price', 'supply']:
    for p in ['none', 'subsidy', 'transfer']:
        ss_, a = pol_ss[(shock, p, 'adoption')], pol[(shock, p, 'adoption')]
        d = true_inflation(ss_, a)
        defl[(shock, p)] = d
        g = float(np.max(np.abs(d['pi_gap_ann'][:H])))
        m = float(np.max(np.abs(a['pi_ann'][:H])))
        lines2.append(f"{shock:>7s} {p:>9s} {d['phi_ss']:9.6f} "
                      f"{float(np.max(np.abs(d['phi_gap'][:H]))):10.4f} "
                      f"{100 * g:10.4f} {100 * m:9.3f} {g / max(m, 1e-16):8.2%}")
table2 = '\n'.join(lines2)
print('\n' + table2)
open(f'{OUT}/deflator_table_{NUMERAIRE}_{BOOKING}.txt', 'w').write(table2 + '\n')

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, shock in zip(axes, ['price', 'supply']):
    for p, sty in [('none', '-'), ('subsidy', '--'), ('transfer', ':')]:
        ax.plot(100 * defl[(shock, p)]['pi_gap_ann'][:H], lw=2, ls=sty, label=p)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'{shock.upper()}: true minus measured inflation (ann., pp)', fontsize=10)
    ax.set_xlabel('quarters', fontsize=8)
axes[0].legend(fontsize=8)
fig.suptitle(r'E8. Measurement gap from anchoring the CPI on $P^E_B$ alone')
fig.tight_layout(); fig.savefig(f'{OUT}/fig8_deflator_gap.png', dpi=140); plt.close(fig)
print(f"  -> {OUT}/fig8_deflator_gap.png")

pickle.dump({k: v for k, v in CACHE.items()}, open(f'{OUT}/irfs_{NUMERAIRE}_{BOOKING}.pkl', 'wb'))
print(f"\nDone. Figures + tables + irfs_{NUMERAIRE}.pkl in {OUT}/")

# %%
