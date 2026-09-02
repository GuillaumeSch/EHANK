#%%
"""Phase 2: durable dimension present but channel CLOSED -> must reproduce ARS."""
import numpy as np, runpy, pickle, sys
import sequence_jacobian as sj
from hh_durable import hh_ha_durable

g = runpy.run_path('auclert_ha.py')
calib = dict(g['calibration']); test_targets = g['test_targets']
B = [g[k] for k in ['hh_outputs','foreign_c','revaluation','mon_policy','fiscal','income',
                    'importPrices','importProfits','profitcenters','UIP','eqm_cond','CA',
                    'unions','CESprices','price_levels','piW_to_W','pitop',
                    'revaluation_dom','annualize','IEA']]
dissolve = ['unions','UIP','CA','piW_to_W','pitop']

# --- CHANNEL CLOSED -------------------------------------------------------
calib['delta_g']   = 0.05
calib['psi_g']      = 0.0        # no switching happens, so no cost is incurred
calib['green_block'] = 1e10      # brown -> green choice forbidden: channel CLOSED
calib['taste_shock'] = 1e-3
calib['cE_ss_agg'] = 0.0         # green/brown price gap booked on zero base
calib['pE_g_P']    = calib.get('pE_P', 1.0)
for i in range(3):
    calib[f'cE_ss_grid_{i}'] = np.zeros((calib['n_e'], calib['n_a']))

model = sj.combine([hh_ha_durable()] + B)
ss = model.solve_steady_state(calib, unknowns={'vphi':1,'beta_max':0.984},
                              targets=['piwres','nfares'], dissolve=dissolve, ttol=1e-14)
test_targets(ss)
print("green share at ss = %.3e   switchers = %.3e" % (ss['D_GREEN'], ss['D_SWITCH']))
irf = model.solve_impulse_linear(ss, g['unknowns_td'], g['targets_td'], g['shocks'])
test_targets(irf)

ref = pickle.load(open('ref_auclert.pkl','rb'))
print("\n=== PHASE 2 (durable present, channel closed) vs ARS ===")
worst = 0.0
for k, v in ref['_ss'].items():
    d = abs(ss[k]-v); worst = max(worst, d)
    print(f"  ss  {k:<8s} {ss[k]:12.6f} {v:12.6f} {d:9.1e}")
for k in ['y','C','cE','n','w','pi','r']:
    d = np.max(np.abs(np.asarray(irf[k])[:200] - ref[k])); worst = max(worst, d)
    print(f"  irf {k:<8s} {100*irf[k][0]:12.6f} {100*ref[k][0]:12.6f} {d:9.1e}")
print(f"\nWORST |diff| = {worst:.2e}   ->", "PASS" if worst < 1e-6 else "FAIL")
