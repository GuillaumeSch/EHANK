# %%
"""
Iteration 1 — durable E-HANK merged into a small open economy.
Goal: steady state + IRF of Y to an exogenous brown-energy world-price shock.
"""
import numpy as np
import sequence_jacobian as sj

from HH_Block import hh
from blocks_soe import (prod, labor_market, exports, goods_market, external,
                        fiscal, nkpc, nkpc_ss, core_inflation, real_rule)

# %%# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
calib = {
    # preferences
    "beta": 0.965, "gamma": 1 / 0.8, "taste_shock": 1e-1,
    "frisch": 1.0, "vphi": 1.0,
    # productivity process
    "rho_e": 0.95, "sd_e": 0.50, "n_e": 5,
    # assets
    "min_a": 0.0, "max_a": 100.0, "n_a": 20,
    # aggregates / prices
    "N": 1.0, "Y": 1.0, "Z": 1.0, "p_core": 1.0, "Div": 0.0, "w": 1.0,
    # durables
    "delta_g": 0.05, "delta_b": 0.0, "psi_g": 1.5,
    # government
    "B": 2.0, "G_ss": 0.1, "kappa_g": 0.10, "Tax": 0.0,
    # energy (imported world prices)
    "p_e_b": 1.0, "p_e_g": 0.8, "tau_b": 0.0, "tau_g": 0.0,
    # consumption aggregator
    "xi": 0.94, "nu": 0.40, "markup_ss": 1.2,
    # monetary (constant real rate = world rate)
    "rss": 0.05 / 4, "ishock": 0.0, "piw": 0.0, "theta_w": 0.75,
    # exports (Auclert values; alpha_star back-out below via two-pass)
    "gamma_x": 0.51, "Cstar": 1.0, "alpha_star": 0.06,
}

# %%# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------
ss_blocks = [hh, fiscal, prod, labor_market, exports, goods_market, external,
             core_inflation, real_rule, nkpc_ss]
model_ss = sj.create_model(ss_blocks, name="SOE_durable_ss")

tr_blocks = [hh, fiscal, prod, labor_market, exports, goods_market, external,
             core_inflation, real_rule, nkpc]
model = sj.create_model(tr_blocks, name="SOE_durable")
print("Models assembled OK.")
print("model unknowns candidates / outputs:", sorted(model.outputs)[:20])

# %%# ---------------------------------------------------------------------------
# Steady state
#   unknowns: Tax, beta, N, psi_g, vphi
#   targets : GBC, nfa=0 (A=B), labor_mkt, D_B=0.95, wnkpc
# ---------------------------------------------------------------------------
unknowns_ss = {"Tax": 0.0, "beta": 0.965, "N": 1.0, "psi_g": 1.5, "vphi": 1.0}
targets_ss = {"GBC": 0.0, "nfa": 0.0, "labor_mkt": 0.0, "D_B": 0.95, "wnkpc": 0.0}

ss = model_ss.solve_steady_state(calib, unknowns_ss, targets_ss, solver="hybr")

# two-pass export scale: close the trade balance at the ss (X = energy imports)
imports_ss = ss["p_e_b"] * ss["C_E_B"] + ss["p_e_g"] * ss["C_E_G"]
alpha_star_new = imports_ss / (ss["p_core"] ** (-ss["gamma_x"]) * ss["Cstar"])
ss = ss.copy(); ss.update({"alpha_star": alpha_star_new})
# recompute the export-dependent identities with the closed trade balance
ss = model_ss.steady_state(ss)

print("\n================ STEADY STATE ================")
print(f"  beta   = {ss['beta']:.4f}")
print(f"  psi_g  = {ss['psi_g']:.4f}")
print(f"  vphi   = {ss['vphi']:.4f}")
print(f"  N      = {ss['N']:.4f}   Y = {ss['Y']:.4f}")
print(f"  A      = {ss['A']:.4f}   B = {ss['B']:.4f}   nfa = {ss['nfa']:.2e}")
print(f"  C      = {ss['C']:.4f}   C_CORE = {ss['C_CORE']:.4f}")
print(f"  C_E_B  = {ss['C_E_B']:.4f}  C_E_G = {ss['C_E_G']:.4f}")
print(f"  D_B    = {100*ss['D_B']:.2f}%  D_G = {100*ss['D_G']:.2f}%")
print(f"  imports= {ss['imports']:.4f}  exports = {ss['exports_val']:.4f}  TB = {ss['TB']:.2e}")
print(f"  X      = {ss['X']:.4f}  alpha_star = {ss['alpha_star']:.4f}")
print(f"  r      = {ss['r']:.4f}")
print(f"  Walras: goods_clearing = {ss['goods_clearing']:.2e}   nfa_res = {ss['nfa_res']:.2e}")


# %%# ---------------------------------------------------------------------------
# IRF: exogenous +10% brown-energy world-price shock (rho = 0.8)
#   transition unknowns: Tax, Y, N, piw
#   transition targets : GBC, goods_clearing, labor_mkt, wnkpc
#   (asset market dropped: open economy, NFA free; real rate fixed at rss)
# ---------------------------------------------------------------------------
# nkpc (transition only) needs theta_w, absent from the SS model's ResultDict
ss = ss.copy()
ss.update({k: calib[k] for k in ['theta_w', 'ishock', 'piw'] if k not in ss})

T = 300
U = ['Tax', 'Y', 'N', 'piw']
TG = ['GBC', 'goods_clearing', 'labor_mkt', 'wnkpc']

G = model.solve_jacobian(ss, unknowns=U, targets=TG, inputs=['p_e_b'], T=T)

rho, size = 0.80, 0.10
dpeb = size * rho ** np.arange(T)

out = ['Y', 'C', 'C_CORE', 'C_E_B', 'C_E_G', 'imports', 'TB', 'nfa', 'X', 'Tax', 'N']
irf = {k: (G[k]['p_e_b'] @ dpeb if (k in G.nesteddict and 'p_e_b' in G[k]) else np.zeros(T))
       for k in out}

print("\n============ IRF to +10% brown-energy price (pct-pt dev, first 8q) ============")
print("  q :   dY      dC     dCcore   dImp     dTB     dNFA")
for q in range(8):
    print(f"  {q} : {100*irf['Y'][q]:7.3f} {100*irf['C'][q]:7.3f} {100*irf['C_CORE'][q]:7.3f} "
          f"{100*irf['imports'][q]:7.3f} {100*irf['TB'][q]:7.3f} {100*irf['nfa'][q]:7.3f}")

print(f"\n  peak dY   = {100*irf['Y'].min():+.3f}%  at q={int(np.argmin(irf['Y']))}")
print(f"  impact dY = {100*irf['Y'][0]:+.3f}%")
print(f"  cum dY(0:20) = {100*irf['Y'][:20].sum():+.3f}")


# %%# ---------------------------------------------------------------------------
# Dynamic Walras check (BoP identity is NOT targeted): should be ~0 all t
# ---------------------------------------------------------------------------
nfa_res_path = (G['nfa_res']['p_e_b'] @ dpeb) if 'nfa_res' in G.nesteddict else np.zeros(T)
gc_path = (G['goods_clearing']['p_e_b'] @ dpeb) if 'goods_clearing' in G.nesteddict else np.zeros(T)
print(f"\n  max |nfa_res| along path  = {np.max(np.abs(nfa_res_path)):.2e}  (BoP Walras check)")
print(f"  max |goods_clearing| path = {np.max(np.abs(gc_path)):.2e}  (targeted -> ~0)")

# adoption margin
for k in ['D_B', 'D_G', 'D_GB']:
    if k in G.nesteddict and 'p_e_b' in G[k]:
        p = G[k]['p_e_b'] @ dpeb
        print(f"  {k}: impact {100*p[0]:+.4f}pp   peak {100*p[np.argmax(np.abs(p))]:+.4f}pp")

# %%# ---------------------------------------------------------------------------
# Figure: IRFs to +10% brown-energy price shock
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = 40
q = np.arange(H)
series = {
    r'Output $Y$':                 100 * irf['Y'][:H],
    r'Consumption $C$':            100 * irf['C'][:H],
    r'Core cons. $C_{\rm core}$':  100 * irf['C_CORE'][:H],
    r'Brown energy $C_E^b$':       100 * irf['C_E_B'][:H],
    r'Trade balance $TB$':         100 * irf['TB'][:H],
    r'Net foreign assets $nfa$':   100 * irf['nfa'][:H],
}
fig, ax = plt.subplots(2, 3, figsize=(13, 7))
for a, (ttl, y) in zip(ax.flat, series.items()):
    a.plot(q, y, lw=2, color='C0')
    a.axhline(0, color='gray', lw=0.6)
    a.set_title(ttl); a.set_xlabel('Quarters')
ax[0, 0].set_ylabel('% dev. from s.s.')
ax[1, 0].set_ylabel('% dev. from s.s.')
fig.suptitle(r'SOE E-HANK: response to a $+10\%$ brown-energy world-price shock ($\rho=0.8$)',
             fontsize=12)
fig.tight_layout()
fig.savefig('irf_soe_brown_energy.png', dpi=130)
print("\nSaved irf_soe_brown_energy.png")
