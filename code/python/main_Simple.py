#%%
from HH_Durables_Block_Simple import hh
from Model_Blocks import fiscal, mkt_clearing, prod, rsrce_cstrt, nkpc, inflation, taylor_rule
import sequence_jacobian as sj
import json
from copy import deepcopy
from Fun.my_funs import *
from sequence_jacobian import drawdag
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

# %%
# === Calibration dictionary ===
baseline_calibration = {

    # -------------------------
    # 1. Preferences
    # -------------------------
    "beta": 0.95,          # Discount factor
    "gamma": 1/0.8,         # Risk aversion (CRRA parameter)
    "taste_shock": 1e-3,    # Idiosyncratic taste shock

    # Labor disutility
    "frisch": 1,            # Frisch elasticity of labor supply
    "vphi": 1,              # Scale of disutility from work

    # -------------------------
    # 2. Productivity (idiosyncratic)
    # -------------------------
    "rho_e": 0.95,          # Persistence of productivity shocks
    "sd_e": 0.50,            # Std. dev. of productivity shocks
    "n_e": 5,               # Number of grid points

    # -------------------------
    # 3. Asset grid
    # -------------------------
    "min_a": 0.0,           # Minimum asset holdings
    "max_a": 100,           # Maximum asset holdings
    "n_a": 20,              # Grid size

    # -------------------------
    # 4. Labor and production
    # -------------------------
    "N": 1,                 # Total labor supply
    "Y": 1,                 # Total economy-wide output

    "Z": 1,                 # Productivity in core goods
    "p_core": 1,            # Price of nondurable consumption good (numéraire)
    "Div": 0,               # Dividends (if firms distribute profit)
    #"w": 1,                 # Wage

    # -------------------------
    # 5. Durable goods (b = brown, g = green)
    # -------------------------
    "delta_g": 0.05,       # Depreciation rate for green durables
    "psi_g": 0.1,          # Adjustment cost for switching to green durable

    # -------------------------
    # 6. Government
    # -------------------------
    "B": 1,                 # Government debt
    "G": 0.1,               # Government spending
    "Tax": 0.358*0,           # Lump-sum tax
    "tau": 0,               # Labor income tax rate

    # -------------------------
    # 7. Energy (brown vs green)
    # -------------------------
    "p_e_b": 1.0,            # Brown energy price (gasoline)
    "p_e_g": 0.8,            # Green energy price (electricity)

    "tau_b": 0.0,            # Carbon tax applied to brown energy
    "tau_g": 0.0,            # Subsidy for green energy

    # -------------------------
    # 8. Prices and substitution structure
    # -------------------------
    "xi": 0.70,             # Share parameter for core goods in consumption
    "nu": 0.4,              # Elasticity of substitution core vs energy
    "markup_ss": 1,         # Steady-state markup

    # -------------------------
    # 9. Financial environment
    # -------------------------
    "r": 0.05 / 2,          # Quarterly interest rate
    
    
    "rss": 0.05 / 4,          
    "phi_pi": 1.5,
    "ishock": 0,
    "piw": 0.0,
    
    # ------------------------- Only for analyzing HH block in isolation -------------------------
    "w": 1,                 # Wage          
}


#TO DELETE IN FINAL VERSION. ONLY FOR DEBUGGING
for k, v in baseline_calibration.items():
    globals()[k] = v
    
#%% Investiage the HH block in isolation
hh_sol = hh.steady_state(baseline_calibration)
# %%
print("Share of Brown at SS:")
print(np.round(hh_sol['D_B']*100, 3),'%')

print("\nShare of Green at SS:")
print(np.round(hh_sol['D_G']*100, 3),'%')

#%%Policy functions
ss_dict = dict()
ss_dict['baseline'] = hh_sol

policy_functions_Simple(ss_dict, ie_list=[2], d_list=[0], d_tilde_list=[0,1], xmax=10, figsize=0.8)

#%%
#%% === Create the model ===
ha = sj.create_model([hh, fiscal, mkt_clearing, prod], name="Simple HA Model")
#ha = sj.create_model([hh_durables, fiscal, mkt_clearing, prod], name="Simple HA Model")
print(ha)
print('It has inputs: ' + str(ha.inputs))
print('It has outputs: ' + str(ha.outputs))

# %% Steady State
unknowns_ss = {'Tax':baseline_calibration['Tax'],'r':baseline_calibration['r']}
targets_ss = {'GBC': 0.0,'asset_mkt': 0.0}

ss = ha.solve_steady_state(baseline_calibration , unknowns_ss, targets_ss, solver='hybr')
print(ss['B'])
print(ss['Tax'])
print(ss['r'])

# %%
#%% Shock to energy prices
titles = [
        r"Brown Energy Price: $p_e_b$",     
        r"Interest Rate: $r$",
        ]
IRFs = plot_linear_irfs(
    shocks_list=['p_e_b'],
    e = {"p_e_b": 0.01},
    rho = {"p_e_b": 0.80},
    unknowns_td=['Tax','r'],
    targets_td=['asset_mkt', "GBC"],
    ha=ha,
    ss=ss,
    outputs=["p_e_b", "r"],
    titles = titles,
    figsize=(18, 12),
)
# %%
