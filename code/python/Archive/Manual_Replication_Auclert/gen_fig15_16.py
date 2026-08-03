# %%
"""
Reproduce Fig. 15 (output, consumption, gov. debt) and Fig. 16 (household energy price,
wage inflation, CPI inflation) with the single-beta fiscal model.

Two model assemblies (differ by ONE block, to keep every Jacobian non-degenerate):
  - model_tr  (transfers): no subsidy; household faces the market energy price and the
                cost-of-living wedge p_c = 1 is a *calibration constant* (so SSJ attaches
                no empty/degenerate Jacobian to it). Runs nothing / targeted / untargeted.
  - model_sub (subsidy):   subsidy_price makes p_c a live variable (p_c < 1). Runs subsidy.

Government budget: B is a model UNKNOWN and the budget constraint is a TARGET (budget_res=0),
which breaks the debt self-reference cleanly. GE system: [Y, B] / [goods_clearing, budget_res].

Caveat: Section-3 baseline (flexible home prices, no slow pass-through, no domestic energy
endowment); the paper runs these on the Section-4 quantitative model, so magnitudes differ.
"""
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sequence_jacobian import create_model

from hh_block_gov import household, grids
from blocks_gov import (prices, subsidy_price, market_energy_price, fiscal_instruments,
                        tax_rule, wagebill, after_tax_income, monetary, asset_block,
                        energy_agg, goods_market, government, union, nfa_block)

# ---- calibration ----
mu, rss, eis, phi, zeta, mu_w, theta_w = 1.03, 0.01, 1.0, 2.0, 5.0, 1.0, 0.938
alpha_E, alpha_F, eta_E, eta, gamma = 0.04, 0.27, 0.10, 0.51, 0.51
alpha = alpha_E + (1 - alpha_E) * alpha_F
alpha_star, Cstar = alpha, 1.0
beta_w = 1.0 / (1 + rss)
kappa_w = (1 - beta_w * theta_w) * (1 - theta_w) / theta_w
vphi = (1.0 / mu) ** (1 + zeta) / mu_w
CE_ss, psiB = alpha_E, 0.04

hh = household.add_hetinputs([grids])
common = [fiscal_instruments, tax_rule, wagebill, after_tax_income, monetary,
          asset_block, hh, energy_agg, goods_market, government, union, nfa_block]
model_tr = create_model([prices, market_energy_price] + common, name="Fiscal_transfers")
model_sub = create_model([prices, subsidy_price] + common, name="Fiscal_subsidy")
print("Both models assembled OK (no DAG cycle).")

nS, nA = 7, 200
calib = dict(
    mu=mu, rss=rss, eis=eis, phi=phi, zeta=zeta, mu_w=mu_w,
    alpha_E=alpha_E, alpha_F=alpha_F, eta_E=eta_E, eta=eta, gamma=gamma,
    alpha=alpha, alpha_star=alpha_star, Cstar=Cstar,
    kappa_w=kappa_w, beta_w=beta_w, vphi=vphi, CE_ss=CE_ss, psiB=psiB,
    tauE=0.0, insE=0.0, unt_scale=0.0, pEstar=1.0, e=0.0, Y=1.0, B=0.0,
    p_c=1.0,                                   # constant for the transfers model
    rho_e=0.912, sigma_e=0.883, nS=nS, amax=200.0, nA=nA,
    xfer_base=np.zeros((nS, nA)),
)

# steady state (solve beta for nfa=0; B=0 pins budget_res=0). p_c=1 at ss for both models.
ss = model_tr.solve_steady_state(calib, {'beta': (0.95, 0.96)}, {'nfa': 0.0}, solver='brentq')
ss['xfer_base'] = alpha_E * ss.internals['household']['c']
print(f"ss: beta={ss['beta']:.4f} A={ss['A']:.4f} C={ss['C']:.4f} B={ss['B']:.2e} "
      f"p_c={ss['p_c']:.4f} nfa={ss['nfa']:.2e}")

# ---- experiments ----
T = 300
dpEstar = 0.96 ** np.arange(T)
out = ['Y', 'C', 'B', 'pi_w', 'pH', 'p_c', 'pE_hh', 'tauL', 'CE']
U, TG = ['Y', 'B'], ['goods_clearing', 'budget_res']

def run(model, ss_e):
    producible = set(model.outputs) | set(U)          # unknowns are returned too
    avail = [k for k in out if k in producible]
    G = model.solve_jacobian(ss_e, unknowns=U, targets=TG, inputs=['pEstar'],
                             outputs=avail, T=T)
    return {k: (G[k]['pEstar'] @ dpEstar if k in avail else np.zeros(T)) for k in out}

irf = {}
irf['nothing']    = run(model_tr,  ss.copy())
sse = ss.copy(); sse.update({'insE': 0.5});      irf['targeted']   = run(model_tr, sse)
sse = ss.copy(); sse.update({'unt_scale': 0.5}); irf['untargeted'] = run(model_tr, sse)
# subsidy needs the subsidy model; p_c becomes a live variable there
ss_sub = model_sub.solve_steady_state(calib, {'beta': (0.95, 0.96)}, {'nfa': 0.0}, solver='brentq')
ss_sub['xfer_base'] = alpha_E * ss_sub.internals['household']['c']
sse = ss_sub.copy(); sse.update({'tauE': 0.5});  irf['subsidy']    = run(model_sub, sse)

# ---- nominal post-processing: market and household CPI inflation ----
def prepend_diff(x):
    d = np.empty_like(x); d[0] = x[0]; d[1:] = x[1:] - x[:-1]
    return d
for name, r in irf.items():
    dlogW = np.cumsum(r['pi_w'])
    dlogP = dlogW - r['pH']
    dlogP_hh = dlogP + r['p_c']
    r['pi'] = prepend_diff(dlogP)
    r['pi_hh'] = prepend_diff(dlogP_hh)

order = ['nothing', 'subsidy', 'targeted', 'untargeted']
print("\n              minY     minC    peakB   peakpiw  peakpi_hh")
for name in order:
    r = irf[name]
    print(f"  {name:11s} {100*r['Y'].min():6.2f}% {100*r['C'].min():6.2f}% "
          f"{100*r['B'].max():6.2f} {100*4*r['pi_w'].max():7.2f}% {100*4*r['pi_hh'].max():8.2f}%")

# ---- Fig. 15 ----
H = 300
colors = {'nothing': 'k', 'subsidy': 'C0', 'targeted': 'C1', 'untargeted': 'C2'}
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for name in order:
    ax[0].plot(100 * irf[name]['Y'][:H], color=colors[name], label=name)
    ax[1].plot(100 * irf[name]['C'][:H], color=colors[name], label=name)
    ax[2].plot(100 * irf[name]['B'][:H], color=colors[name], label=name)
for a, ttl in zip(ax, [r'Output, $Y$', r'Consumption, $C$', r'Gov. debt, $B$']):
    a.axhline(0, color='gray', lw=0.5); a.set_title(ttl); a.set_xlabel('Quarters')
ax[0].set_ylabel('Percent of s.s. output'); ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig('fig15_fiscal_singlebeta.png', dpi=130)
print("Saved fig15_fiscal_singlebeta.png")

# ---- Fig. 16 ----
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for name in order:
    ax[0].plot(100 * irf[name]['pE_hh'][:H], color=colors[name], label=name)
    ax[1].plot(100 * 4 * irf[name]['pi_w'][:H], color=colors[name], label=name)
    ax[2].plot(100 * 4 * irf[name]['pi_hh'][:H], color=colors[name], label=name)
for a, ttl in zip(ax, [r'Household energy price, $P^{hh}_E$ (%)',
                       r'Wage inflation, $\pi_w$ (ann.)',
                       r'Household CPI inflation, $\pi$ (ann.)']):
    a.axhline(0, color='gray', lw=0.5); a.set_title(ttl); a.set_xlabel('Quarters')
ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig('fig16_fiscal_singlebeta.png', dpi=130)
print("Saved fig16_fiscal_singlebeta.png")

with open('results_fig1516.pkl', 'wb') as f:
    pickle.dump(dict(irf=irf, T=T), f)
print("Saved results_fig1516.pkl")

# %%
