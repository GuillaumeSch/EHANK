#%%
"""Phase 0 reference: Auclert HA baseline IRF to the energy price shock."""
import numpy as np, runpy, pickle
g = runpy.run_path('auclert_ha.py')
model_ha, ss_ha = g['model_ha'], g['ss_ha']
test_targets = g['test_targets']
T, shocks = g['T'], g['shocks']
unknowns_td, targets_td = g['unknowns_td'], g['targets_td']
dissolve = ['unions', 'UIP', 'CA', 'piW_to_W', 'pitop']

test_targets(ss_ha)
irf = model_ha.solve_impulse_linear(ss_ha, unknowns_td, targets_td, shocks)
test_targets(irf)
print("targets OK (ss + irf)")

OUT = ['y', 'gdp', 'C', 'A', 'cE', 'cH', 'cF', 'n', 'w', 'pi', 'piw', 'r', 'rante',
       'PEstar', 'pE_P', 'Q', 'nfa', 'netexports', 'MPC', 'P']
ref = {k: np.asarray(irf[k])[:200] for k in OUT if k in irf}
ref['_ss'] = {k: float(ss_ha[k]) for k in ['C','A','MPC','beta_max','vphi','y','n','r']}
pickle.dump(ref, open('ref_auclert.pkl','wb'))
print("\nAuclert HA baseline, impact responses (% or level dev.):")
for k in ['y','gdp','C','cE','n','w','pi','r','Q','MPC']:
    if k in ref: print(f"  d{k:<8s} {100*ref[k][0]:+8.4f}")
