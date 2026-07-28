#%%
"""E6  Partial price cap: dose-response in tauE.
   E7  Welfare: consumption-equivalent variation of each policy.

Both instruments are zero at the steady state, so the steady state is
INVARIANT to tauE and insE (verified to machine precision). We therefore solve
the steady state once per (adoption variant x energy closure) and sweep the
policy by re-solving the impulse response only. This is the `resolve_ss=False`
idea from the user's my_funs.compare_irfs_by_parameter, specialised to the
policy instruments.
"""
import os
import shutil
import pickle
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import (build_model, solve_ss, shock_price, shock_supply,
                         UNKNOWNS_TD, TARGETS_TD, MODELS, MONETARY, ENERGY_CLOSURE)
from calibration import make_calibration, set_energy_grids
from welfare import cev, cev_table
import blocks as B

# See run_experiments.py: caches are tagged by numeraire on purpose.
NUMERAIRE = 'core'

# See run_experiments.py: cache is keyed by tag only, goes stale silently
# on model/calibration changes. Wipe on every run; set False to resume a
# crashed run with an unchanged model.
CLEAR_CACHE = True

OUT, H = 'output', 24
CDIR = f'cache_dose_{NUMERAIRE}'
if CLEAR_CACHE:
    shutil.rmtree(CDIR, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(CDIR, exist_ok=True)
MODEL = build_model(NUMERAIRE)
TAU_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]


def base_ss(variant='adoption', closure='elastic'):
    ov = dict(ENERGY_CLOSURE[closure]); ov.update(MODELS[variant])
    calib = make_calibration(NUMERAIRE, **ov)
    if closure == 'inelastic':
        c0 = dict(calib); c0['E_supply_elasticity'] = np.inf
        calib['E_supply_shock'] = float(solve_ss(MODEL, c0)['cE'])
    ss = solve_ss(MODEL, calib)
    return ss, calib


def irf_with(ss, shk, tauE=0.0, insE=0.0, calib=None):
    """Impulse response under a policy.

    The price cap (tauE) leaves both the steady state AND the household
    hetinputs untouched, so it can be swept by re-solving the impulse response
    alone -- cheap. The Slutsky transfer changes cE_ss_grid, a household
    hetinput that enters the Jacobian, so injecting it into a copied steady
    state leaves a stale Jacobian (assets_clearing then breaks by ~14). It
    needs a genuine re-solve.
    """
    if insE == 0.0:
        s = ss.copy()
        s['tauE'], s['insE'] = tauE, 0.0
        irf = MODEL.solve_impulse_linear(s, UNKNOWNS_TD, TARGETS_TD, shk)
        B.test_targets(irf)
        return irf, ss
    c = set_energy_grids(calib, ss)
    c['insE'], c['tauE'] = insE, tauE
    ss2 = solve_ss(MODEL, c)
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

pickle.dump(results, open(f'{OUT}/dose_response.pkl', 'wb'))

# =============================================================================
# FIGURE: dose-response
# =============================================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for j, sh in enumerate(['price', 'supply']):
    R, tr = results[sh]['rows'], results[sh]['transfer']
    tau = [r['tau'] for r in R]
    for i, (k, lab) in enumerate([('dG', 'peak green share (log pts)'),
                                  ('y', 'cumulative output'),
                                  ('cev', 'welfare CEV (%)'),
                                  ('fisc', 'fiscal cost')]):
        ax = axes[j, i]
        ax.plot(tau, [r[k] for r in R], 'o-', lw=2, color='C1', label='partial cap')
        ax.axhline(tr[k], color='C2', ls=':', lw=2, label='Slutsky transfer')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel(r'cap intensity $\tau^E$', fontsize=8)
        ax.set_title(f'{sh.upper()}: {lab}', fontsize=9)
        ax.tick_params(labelsize=8)
axes[0, 0].legend(fontsize=8)
fig.suptitle(r'E6. Dose-response in the price cap $\tau^E$, against the transfer benchmark')
fig.tight_layout(); fig.savefig(f'{OUT}/fig6_dose_response.png', dpi=140); plt.close(fig)
print(f"\n-> {OUT}/fig6_dose_response.png")

# =============================================================================
# TABLE
# =============================================================================
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
print("\nDone.")

# %%
