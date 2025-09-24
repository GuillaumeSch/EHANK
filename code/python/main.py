#%%
from HH_Durables_Block import hh_durables
from Model_Blocks import fiscal, mkt_clearing, prod, prod_durables, carbon_tax
import sequence_jacobian as sj
import json
from copy import deepcopy
from Fun.my_funs import *
from sequence_jacobian import drawdag
#%matplotlib qt


# %%
# === Calibration dictionary ===
cali = {}
cali["baseline"] = {
    # Preferences and taste shocks
    "taste_shock": 1e-1,       # Idiosyncratic taste shock
    "vphi": 0.0,               # Value function penalty parameter
    "beta": 0.97,              # Discount factor
    "eis": 0.5,                # Elasticity of intertemporal substitution
    "gamma": 1/0.5,            # Relative risk aversion (curvature)
    "omega": 0.78,                # Relative weight of consumption c versus durable goods ˜d in the utility function
    "dbar": 1,                 # Subsistence level of durable goods
    "r": 0.02 / 4,             # Interest rate (quarterly)
    "N": 1,                    # Total labor supply
    # Productivity process
    "rho_e": 0.95,             # Persistence of productivity shocks
    "sd_e": 0.5,               # Std. deviation of productivity shocks
    "n_e": 5,                  # Number of productivity grid points
    # Asset grid
    "min_a": 0.0,              # Minimum asset level
    "max_a": 100,              # Maximum asset level
    "n_a": 20,                 # Number of asset grid points
    # Labor market
    "N_core": 0.6,             # Labor demand for core goods
    #"N_d1": 0.1, "N_d2": 0.1, "N_d3": 0.1, "N_d4": 0.1,                # Labor demand for durable goods
    "N_d": np.array([ 0.1, 0.1, 0.1, 0.1]),                # Labor demand for durable goods
    "tau":0,                   # Labor income tax
    # Durable goods
    #"p_b": 0.80,               # Initial price of brown durable
    #"dep_frac_b": 0.25,        # Depreciation green (Fraction of oldest vintage relative to newest)
    "n_b": 2,                  # Number of brown vintages
    #"p_g": 0.90,               # Initial price of green durable
    #"dep_frac_g": 0.25,        # Depreciation green (Fraction of oldest vintage relative to newest)
    "n_g": 2,                  # Number of green vintages
    #"n_d": 1 + n_b + n_g,      # Total durable states
    "chi": 0.5,                # Resale loss (fraction)
    "gamma_b": 1.0,            # Utility from brown durable
    "dep_util_frac_b": 1,      # Depreciation utility brown (Fraction of oldest vintage relative to newest)
    "gamma_g": 1.2,            # Utility from green durable
    "dep_util_frac_g": 1,      # Depreciation utility green (Fraction of oldest vintage relative to newest)
    "lifetime_b": 60,          # Average lifetime of brown durables (quarters)
    "lifetime_g": 60,          # Average lifetime of green durables (quarters)
    # Firms
    "alpha": 1,                # Share of labor in prod. function
    #"p_core": 1,               # Price of core, non-durable goods
    "Div": 0,                  # Dividends from firms
    "Z_core": 1,               # Core productivity
    "Z_d1": 15, "Z_d2": 60, "Z_d3": 3, "Z_d4": 1, # Durables productivities
    #"Z_d": np.array([15.52920823, 62.11683292,  1.11111111,  4.44444444]),                # 
    #Government
    #"Y" : 1,                  # Output
    "B" : 4,                   # Stock of debt
    "G" : 0.3,                 # Government spendings
    #Energy
    "p_e_b" : 0.333,            # Price of brown energy (gas/petrol)
    "p_e_g" : 0.04,            # Price of green energy (electricity)
    "tau_b" : 0.50,            # Carbon tax (no free. Dependend of T_E)
    "tau_g" : 0,               # Green energy subsidy
    "T_E" : 0.002,           # Energy tax revenues 
    #Prices
    "xi" : 0.97,               # Governs relative share of core good in non-durable consumption basket. To be improved... Goal, Share_core = 95%
    "nu" : 0.4,                # Elasticity of substitution between core and energy consumption
}

#TO DELETE IN FINAL VERSION. ONLY FOR DEBUGGING

#Import the DD calibration (Some parameters are not present in the DD calibration.)
# with open('calibration_ss_DD.json', 'r') as f:
#     cali_DD = json.load(f)

# cali['baseline_DD'] = deepcopy(cali['baseline'])
# for key in cali_DD.keys():
#     if key in cali['baseline_DD']:
#         cali['baseline_DD'][key] = cali_DD[key]

for k, v in cali["baseline"].items():
    globals()[k] = v

#%% === Create the model ===
ha = sj.create_model([hh_durables, fiscal, mkt_clearing, prod, prod_durables, carbon_tax], name="Simple HA Model")
print(ha)
print('It has inputs: ' + str(ha.inputs))
print('It has outputs: ' + str(ha.outputs))

#%%

# unknowns = ['beta', 'N']
# targets = ['asset_mkt','labor_mkt']
# inputs = ['G']

# drawdag(ha, unknowns, targets, inputs)

#%% Not SS
noss_DD = ha.steady_state(cali['baseline'])

noss_DD.toplevel

#%% Not SS
evaluate_param_changes('omega', [0.50, 0.75], ha, cali['baseline'],
                      ss_vars = ['B', 'D_N', 'D_B', 'D_G', 'asset_mkt', 'C', 'A','Tax'])


#%% === Solve the model Steady State ===
unknowns_ss = {'beta':0.985, 'N':1, 'tau_b':0.50}
targets_ss = {'asset_mkt':0.0, 'labor_mkt':0.0, 'T_E_diff': 0.0}

ss_DD = ha.solve_steady_state(cali['baseline'] , unknowns_ss, targets_ss, solver='hybr')

display_ss_durables(ss_DD)
display_calibrated_from_unknowns(ss_DD, unknowns_ss)

check_resource_constraint(ss_DD)


#%% One shot deviation of SS

param_grid = {'omega': np.linspace(0.50, 1, 5)}

# Track output, wage, and interest rate
outputs = ['tau_b','D_N', 'D_B', 'D_BN', 'D_BO', 'D_G', 'D_GN', 'B','beta']
results = comparative_statics_plot(
    ha=ha,
    ss_base=ss_DD,
    param_grid=param_grid,
    unknowns_ss=unknowns_ss,
    targets_ss=targets_ss,
    outputs=outputs,
    plot_deviation=False
)

# outputs_shares = ['D_N', 'D_B', 'D_G']
# #outputs_shares = ['D_N', 'D_BN',  'D_BO', 'D_GN', 'D_GO']


# results = comparative_statics_plot_shares(
#     ha=ha,
#     ss_base=ss_DD,
#     param_grid=param_grid,
#     unknowns_ss=unknowns_ss,
#     targets_ss=targets_ss,
#     outputs=outputs_shares,
#     line_labels=["None", "Brown", "Green"],
#     title="Durable shares under different carbon tax rates",
#     x_label=r"Carbon Tax Rate $\tau_B$",
#     x_values=["Low", " ", "Medium", "  ", "High"],
#     save_path='../../output/figures/CS_T_E.png',
# )

#%%Price of brown energy
param_grid = {'p_e_b': np.linspace(0.2, 0.8, 5)}

results = comparative_statics_plot_shares(
    ha=ha,
    ss_base=ss_DD,
    param_grid=param_grid,
    unknowns_ss=unknowns_ss,
    targets_ss=targets_ss,
    outputs=outputs_shares,
    line_labels=["None", "Brown", "Green"],
    title="Durable shares under different brown energy prices",
    x_label=r"Brown Energy Price $P_B$",
    x_values=["Low", " ", "Medium", "  ", "High"],
    save_path='../../output/figures/CS_p_e_b.png',
)

#%% Price of durable good
param_grid = {'Z_d3': np.linspace(2, 4, 5)}

results = comparative_statics_plot_shares(
    ha=ha,
    ss_base=ss_DD,
    param_grid=param_grid,
    unknowns_ss=unknowns_ss,
    targets_ss=targets_ss,
    outputs=outputs_shares,
    line_labels=["None", "Brown", "Green"],
    title="Durable shares under different green subsidies",
    x_label=r"Green Subsidies on Aquisition Cost",
    x_values=["Low", " ", "Medium", "  ", "High"],
    save_path='../../output/figures/CS_Z_d3.png',
)




#%%
ss = dict()
ss['baseline'] = ss_DD

policy_functions(ss, plots=['disc'], ie_list=[0], d_list=[0], d_tilde_list=[0, 1, 2, 3, 4], xmax=5.2, figsize=0.8, save_path='../../output/figures/Policy_Functions_dtilde.png')




#%%
plot_distribution(ss_DD, lines_dim = 0, 
                 labels = ['$\\tilde{D}$ = None','$\\tilde{D}$ = New Brown','$\\tilde{D}$ = Old Brown','$\\tilde{D}$ = New Green','$\\tilde{D}$ = Old Green'],
                 truncate_at = 6, save_path='../../output/figures/Distribution_Durables_D_tilde.png')
plot_distribution(ss_DD, lines_dim = 1, 
                 labels = ['None','New Brown','Old Brown','New Green','Old Green'],
                 truncate_at = 6, save_path='../../output/figures/Distribution_Durables_D.png')
plot_distribution(ss_DD, lines_dim = 2, 
                 labels = ['Very Low','Low','Middle','High','Very High'],
                 truncate_at = 6, save_path='../../output/figures/Distribution_Prod.png')
plot_distribution(ss_DD,
                 labels = ['Total'],
                 truncate_at = 6, save_path='../../output/figures/Distribution_Total.png')
#%% Check that the resource constraint holds







#%% === Solve the model transition dynmamics and get IRFs === §
titles = [
        r"Carbon tax revenues: $T_B$",  
        r"Carbon tax rate: $\tau_B$",     
        r"Share of non-durable: $D_N$",  
        r"Share of Brown: $D_B$",
        r"Share of Green: $D_G$",     
        r"Total Consumption: $C$",  
        r"Consu. of Brown energy: $C^B$", 
        r"Consu. of Green energy: $C^G$",]
plot_linear_irfs(
    shocks_list=['T_E'],
    e = {"T_E": 0.01},
    rho = {"T_E": 0.80},
    unknowns_td=['r','N','tau_b'],
    targets_td=['asset_mkt',"labor_mkt", "T_E_diff"],
    ha=ha,
    ss=ss_DD,
    #outputs=["tau_b","T_E_ENDO","B", "r", "Z_core","G", "Tax","D_B","D_BO", "D_BN", "D_G","D_GO", "D_GN", "D_N", "goods_mkt", "asset_mkt", "Y_core","C", "C_E", "C_CORE"],
    outputs=["T_E","tau_b", "D_N", "D_B", "D_G", "C", "C_E_B", "C_E_G"],
    titles = titles,
    figsize=(18, 12),
    #save_path='../../output/figures/IRFs_tau_b.png',
)
# %% Durable Variables
titles = [
        r"Carbon tax revenues: $T_B$",  
        r"Share of Brown: $D_B$",
        r"Share of Green: $D_G$",     
        r"Consu. of Brown energy: $C^B$", 
        r"Consu. of Green energy: $C^G$",
        r"Core Consu.: $C^{Core}$",]
plot_linear_irfs(
    shocks_list=['T_E'],
    e = {"T_E": 0.01},
    rho = {"T_E": 0.80},
    unknowns_td=['G','N','tau_b'],
    targets_td=['asset_mkt',"labor_mkt", "T_E_diff"],
    ha=ha,
    ss=ss_DD,
    #outputs=["tau_b","T_E_ENDO","B", "r", "Z_core","G", "Tax","D_B","D_BO", "D_BN", "D_G","D_GO", "D_GN", "D_N", "goods_mkt", "asset_mkt", "Y_core","C", "C_E", "C_CORE"],
    outputs=["T_E_ENDO","D_B", "D_G", "C_E_B", "C_E_G", "C_CORE"],
    titles = titles,
    figsize=(12, 6),
    save_path='../../output/figures/IRFs_T_E.png',
)

# %%
titles = [
        r"Real Interest Rate: $r$",  
        r"Government Debt: $B$",
        r"Government Expenditures: $G$",     
        r"Total Non-Durable Consu.: $C$", 
        r"Lump-Sum Tax: $T$",
        r"Primary Deficit: $G-(T_E + T)$",r"G"]
plot_linear_irfs(
    shocks_list=['T_E'],
    e = {"T_E": 0.01},
    rho = {"T_E": 0.80},
    unknowns_td=['G','N','tau_b'],
    targets_td=['asset_mkt',"labor_mkt", "T_E_diff"],
    ha=ha,
    ss=ss_DD,
    #outputs=["tau_b","T_E_ENDO","B", "r", "Z_core","G", "Tax","D_B","D_BO", "D_BN", "D_G","D_GO", "D_GN", "D_N", "goods_mkt", "asset_mkt", "Y_core","C", "C_E", "C_CORE"],
    outputs=["r","B","G","C", "Tax", "deficit"],
    titles = titles,
    figsize=(12, 6),
    save_path='../../output/figures/IRFs_T_E_macro.png',
)








# === Analyze the steady state ===
#%%
evaluate_param_changes('gamma_g', [1, 2, 3, 4], ha, ss_DD,
                           ss_vars=['N_core', 'labor_mkt','asset_mkt', 'Tax',
                                    'A', 'B', 'N', 'C', 'C_CORE', 'C_E','D_N','D_G','D_B'])


analyze_steady_state('gamma_g', [1, 2, 3, 4, 10], ss_DD, hh_durables, variables= ['A', 'D_N', 'D_G', 'D_B'])

#%%

analyze_steady_state_3d('gamma_g', [1, 2],'gamma_b', [1, 2], ss_DD, hh_durables, variables= ['A', 'D_N', 'D_G', 'D_B'])


#%%
unknowns_ss_2 = {'N':ss_DD['N'],'mu_N_d':ss_DD['mu_N_d'], 'p_b': 0.80, 'gamma_g':1.2}
targets_ss_2 = {'asset_mkt':0,'labor_mkt':0, 'D_B':0.050,'D_G':0.020, }


unknowns_ss_2 = {
    #'N': (0, ss_DD['N'], 1),
    'B': (0, ss_DD['B'], 4),
    'N': (0, ss_DD['N'], 1),
    #'mu_N_d':(0, ss_DD['mu_N_d'], 1),
    'p_b': (0.01, 0.273, 10),
    #'p_g': (0.01, 0.9, 10),
    'gamma_g': (0,1.243,100),
    #'dep_util_frac_b': (0.1,0.99,1)
}

targets_ss_2 = {
    'asset_mkt': 0,
    'labor_mkt': 0.,
    'D_N': 0.80,
    #'D_G': 0.05,
    'D_B':0.01,
}

#ss_DD = ha.solve_steady_state(cali['baseline'] , unknowns_ss, targets_ss, solver='hybr')
ss_DD_2 = ha.solve_steady_state(ss_DD , unknowns_ss_2, targets_ss_2, solver='broyden_custom')

display_ss_durables(ss_DD_2)
display_calibrated_from_unknowns(ss_DD_2, unknowns_ss_2)

#%%
cali_DD_alt = ss_DD.copy()
cali_DD_alt['Z_d1'] = 0.5

unknowns_ss = {'B':(0.80,0.999)}
targets_ss = {'asset_mkt'}

ss_DD_alt = ha.solve_steady_state(cali_DD_alt , unknowns_ss, targets_ss, solver='hybr')
display_ss_durables(ss_DD_alt)
display_calibrated_from_unknowns(ss_DD, ss_DD_alt)

