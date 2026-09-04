"""E6: partial price cap, dose-response in tauE."""
import os
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt

from core.model import (build_model, solve_ss, shock_price, shock_supply,
                         td_unknowns_targets, _calibrate_supply, ss_unknowns_targets,
                         MODELS, ENERGY_CLOSURE)
from core.calibration import make_calibration, set_energy_grids
from core.welfare import cev, cev_table
from core import blocks as B
from tools import latex_tables as LT

NUMERAIRE = 'cpi'
BOOKING = 'import'          # 'import' baseline, or 'domestic' (green sector)

# Wipe the tag-keyed cache on every run; False only to resume a crashed run.
CLEAR_CACHE = False

OUT, H = 'paper/output', 24
CDIR = f'cache/cache_dose_{NUMERAIRE}_{BOOKING}'
if CLEAR_CACHE:
    shutil.rmtree(CDIR, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(CDIR, exist_ok=True)
MODEL = build_model(NUMERAIRE, booking=BOOKING)
UNKNOWNS_TD, TARGETS_TD = td_unknowns_targets(BOOKING)
TAU_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]

def base_ss(variant='adoption', closure='elastic'):
    ov = dict(ENERGY_CLOSURE[closure]); ov.update(MODELS[variant])
    calib = make_calibration(NUMERAIRE, booking=BOOKING, **ov)
    u, t = ss_unknowns_targets(BOOKING)
    if closure == 'inelastic':
        calib['E_supply_shock'] = _calibrate_supply(MODEL, calib, u, t, BOOKING)
    ss = solve_ss(MODEL, calib, unknowns=u, targets=t, booking=BOOKING)
    return ss, calib

def irf_with(ss, shk, tauE=0.0, insE=0.0, calib=None):
    """Impulse response under a policy."""
    if insE == 0.0:
        s = ss.copy()
        s['tauE'], s['insE'] = tauE, 0.0
        irf = MODEL.solve_impulse_linear(s, UNKNOWNS_TD, TARGETS_TD, shk)
        B.test_targets(irf)
        return irf, ss
    c = set_energy_grids(calib, ss)
    c['insE'], c['tauE'] = insE, tauE
    ss2 = solve_ss(MODEL, c, booking=BOOKING)
    irf = MODEL.solve_impulse_linear(ss2, UNKNOWNS_TD, TARGETS_TD, shk)
    B.test_targets(irf)
    return irf, ss2

def cum(x, h=H):
    return 100 * float(np.sum(np.asarray(x)[:h]))

results = {}
for closure, shock_name in [('elastic', 'price'), ('inelastic', 'supply')]:
    print(f"\n=== {shock_name.upper()} shock ===", flush=True)
    ss, calib = base_ss('adoption', closure)
    shk = (shock_price() if shock_name == 'price'
           else shock_supply(ss))

    rows, irfs = [], {}
    for t in TAU_GRID:
        f = f'{CDIR}/{shock_name}_cap{t:.2f}.pkl'
        if os.path.exists(f):
            irf = pickle.load(open(f, 'rb'))
        else:
            irf, _ = irf_with(ss, shk, tauE=t, calib=calib)
            pickle.dump({k: np.asarray(irf[k]) for k in
                         ['y','C','cE','pi_ann','D_GREEN','spending','n','UTIL_0','UTIL_1','UTIL_2']},
                        open(f, 'wb'))
            irf = pickle.load(open(f, 'rb'))
        irfs[f'cap {t:.2f}'] = irf
        m, _ = cev(ss, irf)
        rows.append(dict(tau=t, dG=100 * float(np.max(np.asarray(irf['D_GREEN'])[:H])),
                         y=cum(irf['y']), pi0=100 * float(np.asarray(irf['pi_ann'])[0]),
                         fisc=cum(irf['spending']), cev=100 * m,
                         cE=cum(irf['cE'])))
        print(f"  tauE={t:.2f}  peakDG={rows[-1]['dG']:7.3f}  cumY={rows[-1]['y']:8.2f}"
              f"  fiscal={rows[-1]['fisc']:7.2f}  CEV={rows[-1]['cev']:+.4f}%", flush=True)

    ft = f'{CDIR}/{shock_name}_transfer.pkl'
    if os.path.exists(ft):
        irf_tr, ss_tr = pickle.load(open(ft, 'rb')), ss
    else:
        irf_tr, ss_tr = irf_with(ss, shk, insE=1.0, calib=calib)
        pickle.dump({k: np.asarray(irf_tr[k]) for k in
                     ['y','C','cE','pi_ann','D_GREEN','spending','n','UTIL_0','UTIL_1','UTIL_2']},
                    open(ft, 'wb'))
        irf_tr = pickle.load(open(ft, 'rb'))
    irfs['transfer'] = irf_tr
    m_tr, _ = cev(ss_tr, irf_tr)
    print(f"  transfer   peakDG={100*float(np.max(np.asarray(irf_tr['D_GREEN'])[:H])):7.3f}"
          f"  cumY={cum(irf_tr['y']):8.2f}  fiscal={cum(irf_tr['spending']):7.2f}"
          f"  CEV={100*m_tr:+.4f}%", flush=True)

    tbl, _ = cev_table(ss, irfs)
    print('\n' + tbl)
    results[shock_name] = dict(rows=rows, transfer=dict(
        dG=100 * float(np.max(np.asarray(irf_tr['D_GREEN'])[:H])), y=cum(irf_tr['y']),
        fisc=cum(irf_tr['spending']), cev=100 * m_tr, cE=cum(irf_tr['cE']),
        pi0=100 * float(np.asarray(irf_tr['pi_ann'])[0])), cev_table=tbl)

    if shock_name == 'price':
        # distributional CEV table: no policy, cap 0.5/1.0, transfer
        cev_irfs = {'no policy': irfs['cap 0.00'],
                   r'cap $\tau^E=0.5$': irfs['cap 0.50'],
                   r'cap $\tau^E=1$': irfs['cap 1.00'],
                   'transfer': irfs['transfer']}
        cev_path = LT.cev_table_tex(
            f'{OUT}/tab_cev_{NUMERAIRE}_{BOOKING}.tex', ss, cev_irfs,
            labels=list(cev_irfs), scenario_note='price shock',
            label=('tab:cev' if BOOKING == 'import' else f'tab:cev_{BOOKING}'))

pickle.dump(results, open(f'{OUT}/dose_response.pkl', 'wb'))

tex_path = LT.dose_response_table(
    f'{OUT}/tab_dose_{NUMERAIRE}_{BOOKING}.tex', results, TAU_GRID,
    label=('tab:dose' if BOOKING == 'import' else f'tab:dose_{BOOKING}'))

fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
COLS = [('dG', 'Peak green share (pp)'), ('y', 'Cumulative output'),
        ('cev', 'Welfare CEV (%)'), ('fisc', 'Fiscal cost')]
for j, sh in enumerate(['price', 'supply']):
    R, tr = results[sh]['rows'], results[sh]['transfer']
    tau = [r['tau'] for r in R]
    for i, (k, lab) in enumerate(COLS):
        ax = axes[j, i]
        ax.plot(tau, [r[k] for r in R], 'o-', lw=2, color='#c44', label='partial cap')
        ax.axhline(tr[k], color='#48c', ls='--', lw=2, label='Slutsky transfer')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(lab, fontsize=9)
        ax.tick_params(labelsize=8)
        if j == 1:
            ax.set_xlabel(r'cap intensity $\tau^E$', fontsize=8)
    axes[j, 0].set_ylabel(f'{sh.capitalize()} shock', fontsize=10)
axes[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig6_dose_response.png', dpi=140); plt.close(fig)

lines = [f"{'shock':>7s} {'policy':>14s} {'peak DG':>9s} {'cum y':>8s} "
         f"{'pi(0)':>8s} {'cum cE':>8s} {'fiscal':>8s} {'CEV %':>9s}",
         '-' * 78]
for sh in ['price', 'supply']:
    for r in results[sh]['rows']:
        lines.append(f"{sh:>7s} {'cap tau=%.2f' % r['tau']:>14s} {r['dG']:9.3f} "
                     f"{r['y']:8.2f} {r['pi0']:8.2f} {r['cE']:8.2f} {r['fisc']:8.2f} {r['cev']:+9.4f}")
    tr = results[sh]['transfer']
    lines.append(f"{sh:>7s} {'transfer':>14s} {tr['dG']:9.3f} {tr['y']:8.2f} "
                 f"{tr['pi0']:8.2f} {tr['cE']:8.2f} {tr['fisc']:8.2f} {tr['cev']:+9.4f}")
tbl = '\n'.join(lines)
print('\n' + tbl)
open(f'{OUT}/dose_response_table.txt', 'w').write(tbl + '\n')
