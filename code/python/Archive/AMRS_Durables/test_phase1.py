#%%
"""Phase 1 test: StageBlock household must reproduce the ARS HetBlock exactly."""
import numpy as np, runpy, pickle
import sequence_jacobian as sj
from hh_stage import hh_ha_stage

g = runpy.run_path('auclert_ha.py')
calibration = g['calibration']; test_targets = g['test_targets']
B = {k: g[k] for k in ['hh_outputs','foreign_c','revaluation','mon_policy','fiscal','income',
                       'importPrices','importProfits','profitcenters','UIP','eqm_cond','CA',
                       'unions','CESprices','price_levels','piW_to_W','pitop',
                       'revaluation_dom','annualize','IEA']}
dissolve = ['unions','UIP','CA','piW_to_W','pitop']

model = sj.combine([hh_ha_stage()] + list(B.values()))
ss = model.solve_steady_state(calibration, unknowns={'vphi':1,'beta_max':0.984},
                              targets=['piwres','nfares'], dissolve=dissolve, ttol=1e-14)
test_targets(ss)
irf = model.solve_impulse_linear(ss, g['unknowns_td'], g['targets_td'], g['shocks'])
test_targets(irf)

ref = pickle.load(open('ref_auclert.pkl','rb'))
print("=== PHASE 1: StageBlock vs ARS HetBlock ===")
print(f"{'':12s} {'stage':>12s} {'ARS':>12s} {'|diff|':>10s}")
worst = 0.0
for k, v in ref['_ss'].items():
    d = abs(ss[k]-v); worst = max(worst, d)
    print(f"  ss {k:<8s} {ss[k]:12.6f} {v:12.6f} {d:10.1e}")
for k in ['y','C','cE','n','w','pi','r','MPC']:
    if k not in ref: continue
    d = np.max(np.abs(np.asarray(irf[k])[:200] - ref[k])); worst = max(worst, d)
    print(f"  irf {k:<7s} {100*irf[k][0]:12.6f} {100*ref[k][0]:12.6f} {d:10.1e}")
print(f"\nWORST ABSOLUTE DIFFERENCE: {worst:.2e}")
print('PHASE 1', 'PASS' if worst < 1e-6 else 'FAIL')

# %%
