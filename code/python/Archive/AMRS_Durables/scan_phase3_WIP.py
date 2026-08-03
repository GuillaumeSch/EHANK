#%%
"""Phase 3 scan: green share as a function of psi_g, for two taste-shock scales."""
import math
import matplotlib.pyplot as plt
import numpy as np, runpy, pickle
import sequence_jacobian as sj
from hh_durable import hh_ha_durable
from blocks_dur import eqm_cond_dur, CA_dur, imports_dur_block, energy_gap_block
g = runpy.run_path('auclert_ha.py')
calib = dict(g['calibration']); test_targets = g['test_targets']
B = [g[k] for k in ['hh_outputs','foreign_c','revaluation','mon_policy','fiscal','income',
                    'importPrices','importProfits','profitcenters','UIP',
                    'unions','CESprices','price_levels','piW_to_W','pitop',
                    'revaluation_dom','annualize','IEA']] + [CA_dur]
dissolve = ['unions','UIP','CA_dur','piW_to_W','pitop']
for i in range(3):
    calib[f'cE_ss_grid_{i}'] = np.zeros((calib['n_e'], calib['n_a']))

cE_ref = float(g['ss_ha']['cE'])            # = alpha_E * C at the ARS steady state
print(f"cE_ss_agg (= ss_ref['cE'] * s_dur, s_dur=1) = {cE_ref:.6f}")
pE = float(g['ss_ha']['pE_P'])
calib.update(delta_g=0.05, cE_ss_agg=cE_ref, pE_g_P=0.8*pE, green_block=0.0)
print(f"pE_P = {pE:.4f}   pE_g_P = {0.8*pE:.4f}   PV of saving ~ {0.2*pE*cE_ref/(0.01+0.05):.4f}\n")

ref = pickle.load(open('ref_auclert.pkl','rb'))

model = sj.combine([hh_ha_durable()] + B + [imports_dur_block, energy_gap_block, eqm_cond_dur])
for ts in [1e-2]:
    calib['taste_shock'] = ts
    print(f"--- taste_shock = {ts:g}")
    print(f"{'psi_g':>8s} {'D_GREEN':>10s} {'D_SWITCH':>10s}")
    for psi in [0.05, 0.15, 0.20, 0.25, 0.30]:
        calib['psi_g'] = psi
        try:
            ss = model.solve_steady_state(calib,
                    unknowns={'vphi':1,'beta_max':0.984,'y':float(g['ss_ha']['y'])},
                    targets=['piwres','nfares','goods_clearing'],
                    solver='broyden_custom', dissolve=dissolve, ttol=1e-14)
            test_targets(ss)
            print(f"{psi:8.3f} {ss['D_GREEN']:10.4f} {ss['D_SWITCH']:10.5f}")
        except Exception as e:
            print(f"{psi:8.3f}   FAILED: {type(e).__name__}")

# %%
irf = model.solve_impulse_linear(ss, g['unknowns_td'], g['targets_td'], g['shocks'])
# %%


def show_irfs(irfs_list, variables, labels=None, ylabel=r"PP (dev. from ss)",
              T_plot=50, figsize=(18, 6), save_path=None, titles=None, show=False):
    """
    Plot impulse response functions (IRFs) for multiple variables and scenarios.
    show=False skips plt.show() (e.g. when only exporting to save_path).
    """
    if labels is None or len(irfs_list) != len(labels):
        labels = ["Scenario {}".format(i+1) for i in range(len(irfs_list))]

    n_var = len(variables)
    n_cols = min(3, n_var)
    n_rows = math.ceil(n_var / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, var in enumerate(variables):
        axes[i].axhline(0, color='grey', lw=0.8, ls='--')
        axes[i].grid(alpha=0.3)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        for j, irf in enumerate(irfs_list):
            if var in irf:
                data = np.array(irf[var][:T_plot])
            else:
                data = np.zeros(T_plot)
            axes[i].plot(data, label=labels[j])

        if titles is not None:
            if isinstance(titles, dict) and var in titles:
                axes[i].set_title(titles[var], usetex=True, fontsize=16)
            elif isinstance(titles, list) and i < len(titles):
                axes[i].set_title(titles[i], usetex=True, fontsize=16)
            else:
                axes[i].set_title(var, fontsize=16)
        else:
            axes[i].set_title(var, fontsize=16)

        axes[i].set_xlabel(r"quarter")
        axes[i].set_ylabel(ylabel)
        axes[i].grid(True)
        if i == 0:
            axes[i].legend()

    for k in range(n_var, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)
# %%
show_irfs([irf,ref],['D_GREEN','D_BROWN'],show=True)
# %%
