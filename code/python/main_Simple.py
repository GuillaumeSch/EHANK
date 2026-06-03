#%%
from HH_Durables_Block_Simple import hh
from Model_Blocks import fiscal, mkt_clearing, prod, prod_old, rsrce_cstrt, nkpc, inflation, taylor_rule
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
    "beta": 0.965,          # Discount factor
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
    "B": 2,                 # Government debt
    "G": 0.1,               # Government spending
    "Tax": 0,               # Lump-sum tax
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
    "markup_ss": 1.2,       # Steady-state markup

    # -------------------------
    # 9. Financial environment
    # -------------------------
    "r": 0.05 / 4,          # Quarterly interest rate
    
    
    "rss": 0.05 / 4,          
    "phi_pi": 1.5,
    "ishock": 0,
    "piw": 0.0,
    #"kappa_w": 0.01, #Should be (1 - theta_w) * (1 - beta * theta_w)/theta_w, but simple for now
    "theta_w": 0.75, 
    
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

policy_functions_Simple(ss_dict, ie_list=[2], d_list=[1], d_tilde_list=[0,1], xmax=10, figsize=0.8)

#%%
#%% === Create the model ===
ha = sj.create_model([hh, fiscal, mkt_clearing, rsrce_cstrt, prod_old], name="Simple HA Model")
hank = sj.create_model([hh, fiscal, mkt_clearing, rsrce_cstrt, prod_old, nkpc, inflation, taylor_rule], name="HANK Model")

#ha = sj.create_model([hh_durables, fiscal, mkt_clearing, prod], name="Simple HA Model")
print(ha)
print('It has inputs: ' + str(ha.inputs))
print('It has outputs: ' + str(ha.outputs))

# %% Steady State
unknowns_ss = {'Tax':baseline_calibration['Tax'],'beta':0.97,'N':baseline_calibration['N']}
targets_ss = {'GBC': 0.0,'asset_mkt': 0.0, 'labor_mkt': 0.0}

ss = ha.solve_steady_state(baseline_calibration , unknowns_ss, targets_ss, solver='hybr')
print(ss['B'])
print(ss['Tax'])
print(ss['r'])
print(ss['beta'])
print("Share of Brown at SS:")
print(np.round(ss['D_B']*100, 3),'%')
print("\nShare of Green at SS:")
print(np.round(ss['D_G']*100, 3),'%')
#%%
ss_dict = dict()
ss_dict['baseline'] = ss

policy_functions_Simple(ss_dict, ie_list=[4], d_list=[0], d_tilde_list=[0,1], xmax=10, figsize=0.8)


#%%
unknowns_ss_hank = {'Tax':ss['Tax'],'beta':ss['beta'],'vphi':baseline_calibration['vphi'],'N':baseline_calibration['N']}
targets_ss_hank = {'GBC': 0.0,'asset_mkt': 0.0, 'wnkpc': 0.0, 'labor_mkt': 0.0}

ss_hank = hank.solve_steady_state(ss , unknowns_ss_hank, targets_ss_hank, solver='hybr')


#%% IRFS ####
#############$

outputs = ['C', 'C_CORE','C_E','Y', 'w', 'N', 'N_D','D_B', 'D_G', 'G', 'B', 'Tax', 'r','piw','i','rsrce_cstrt', 'AD', 'AD_CORE', 'AD_DURABLES', 'AS']
names_outputs = [
        r"Consumption: $C$",
        r"Core Consumption: $C_{core}$",
        r"Energy Consumption: $C_E$",
        r"Output: $Y$",
        r"Wage: $w$",
        r"Labor Supply: $N^s$",
        r"Labor Demand: $N^d$",
        r"Brown Durable Stock: $D_B$",
        r"Green Durable Stock: $D_G$",
        r"Government Spending: $G$",
        r"Government Debt: $B$",
        r"Lump-Sum Tax: $Tax$",
        r"Interest Rate: $r$",
        r"Wage Inflation: $\pi_w$",
        r"Nominal Interest Rate: $i$",
        ]

#Equilibrium coniditions to close the model.
targets_td=['asset_mkt', 'GBC', 'wnkpc', 'labor_mkt']
#Unknowns to solve for in the transition (those that are not directly shocked)
unknowns_td=['Tax','Y','N', 'w']

#%% Shock to brown energy price
IRFs_p_e_b = plot_linear_irfs(
    shocks_list=['p_e_b'],
    e = {"p_e_b": 0.01},
    rho = {"p_e_b": 0.80},
    unknowns_td=unknowns_td,
    targets_td=targets_td,
    ha=hank,
    ss=ss_hank,
    outputs= outputs,
    titles = names_outputs,
    figsize=(12, 9),
)
#%% Shock to Carbon tax
IRFs_tau_b = plot_linear_irfs(
    shocks_list=['tau_b'],
    e = {"tau_b": 0.01},
    rho = {"tau_b": 0.80}, 
    unknowns_td=unknowns_td,
    targets_td=targets_td,
    ha=hank,
    ss=ss_hank,
    outputs= outputs,
    titles = names_outputs,
    figsize=(12, 9),
)
#%% Shock to nominal interest rate (monetary policy shock)
IRFs_i = plot_linear_irfs(
    shocks_list=['ishock'],
    e = {"ishock": 0.01},
    rho = {"ishock": 0.80},
    unknowns_td=unknowns_td,
    targets_td=targets_td,
    ha=hank,
    ss=ss_hank,
    outputs= outputs,
    titles = names_outputs,
    figsize=(12, 9),
)
#%% Compare IRFs

show_irfs(
        [IRFs_p_e_b,IRFs_tau_b],
        outputs,
        titles = names_outputs,
    )


#%%  

param_grid = {'Z': np.linspace(0.98, 1.02, 5)}

# Track output, wage, and interest rate
outputs = ['Z','D_B', 'D_G','r','Tax','C','Y']
results = comparative_statics_plot(
    ha=hank,
    ss_base=ss_hank,
    param_grid=param_grid,
    unknowns_ss=unknowns_ss,
    targets_ss=targets_ss,
    outputs=outputs,
    plot_deviation=False
)


# %%

shocks que je veux regarder:
- shock to brown energy price (p_e_b)
- shock to carbon tax (tau_b)
- shock to monetary policy (ishock)

Variables d'intérêt:
- Share of brown and green durables (D_B, D_G)
- Consumption (C)
- Output (Y)
- Interest rate (r)
- Government debt (B)

Variables endogènes:

# %%
