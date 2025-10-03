#%%
from HH_Durables_Block import hh_durables
from Model_Blocks import fiscal, mkt_clearing, prod, prod_durables, nkpc, inflation, taylor_rule
import sequence_jacobian as sj
import json
from copy import deepcopy
from Fun.my_funs import *
from sequence_jacobian import drawdag
#%matplotlib qt


# %%
# === Calibration dictionary ===
baseline_calibration = {
    # Preferences and taste shocks
    "taste_shock": 1e-1,       # Idiosyncratic taste shock
    "beta": 0.959,              # Discount factor
    "eis": 0.5,                # Elasticity of intertemporal substitution
    "gamma": 1/0.5,            # Relative risk aversion (curvature)
    "omega": 0.78,             # Relative weight of consumption c versus durable goods ˜d in the utility function
    "dbar": 1,                 # Subsistence level of durable goods
    "r": 0.06 / 4,             # Interest rate (quarterly)
    "N": 1,                    # Total labor supply
    "frisch": 1,               # Frisch elasticity
    "vphi": 1,                 # Disutility of work
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
    "N_d": np.array([ 0.1, 0.1, 0.1, 0.1]),                # Labor demand for durable goods
    "tau":0,                   # Labor income tax
    # Durable goods
    "n_b": 2,                  # Number of brown vintages
    "n_g": 2,                  # Number of green vintages
    "chi": 0.5,                # Resale loss (fraction)
    "gamma_b": 1.0,            # Utility from brown durable
    "dep_util_frac_b": 0.7,    # Depreciation utility brown (Fraction of oldest vintage relative to newest)
    "gamma_g": 1.2,            # Utility from green durable
    "dep_util_frac_g": 0.7,    # Depreciation utility green (Fraction of oldest vintage relative to newest)
    "lifetime_b": 60,          # Average lifetime of brown durables (quarters)
    "lifetime_g": 60,          # Average lifetime of green durables (quarters)
    # Firms
    "alpha": 1,                # Share of labor in prod. function
    #"p_core": 1,               # Price of core, non-durable goods
    "Div": 0,                  # Dividends from firms
    "Z_core": 1,               # Core productivity
    "Z_d1": 15, "Z_d2": 60, "Z_d3": 10, "Z_d4": 1, # Durables productivities
    #Government
    "B" : 4,                   # Stock of debt
    "G" : 0.3,                 # Government spendings
    "Tax": 0.358,              # Lump-sum tax
    #Energy
    "p_e_b" : 0.333,           # Price of brown energy (gas/petrol)
    "p_e_g" : 0.04,            # Price of green energy (electricity)
    "eps_b" : 0.40,            # Linear inefficiency of brown car vintages
    "eps_g" : 0.20,            # Linear inefficiency of green car vintages
    "tau_b" : 0.202,            # Carbon tax (no free. Dependend of T_E)
    "tau_g" : 0,               # Green energy subsidy
    #"T_E" : 0.002,             # Energy tax revenues 
    #Prices
    "xi" : 0.97,               # Governs relative share of core good in non-durable consumption basket. To be improved... Goal, Share_core = 95%
    "nu" : 0.4,                # Elasticity of substitution between core and energy consumption
    #NK part
    "piw": 0,
    "markup_ss": 1,
    "phi_pi": 1.5,
    "ishock": 0,
    "rss": 0.06 / 4
    
}

#TO DELETE IN FINAL VERSION. ONLY FOR DEBUGGING
for k, v in baseline_calibration.items():
    globals()[k] = v

#%% === Create the model ===
#ha = sj.create_model([hh_durables, fiscal, mkt_clearing, prod, prod_durables, carbon_tax], name="Simple HA Model")
ha = sj.create_model([hh_durables, fiscal, mkt_clearing, prod, prod_durables, nkpc, inflation, taylor_rule], name="Simple HA Model")
print(ha)
print('It has inputs: ' + str(ha.inputs))
print('It has outputs: ' + str(ha.outputs))

#%% DAG

unknowns = ['G','N','Tax', 'piw']
targets = ['asset_mkt',"labor_mkt", "GBC", "piwres"]
inputs = ['G']
drawdag(ha, unknowns, targets, inputs)

#%% Evaluate the model at the calibration
# ss_baseline = ha.steady_state(baseline_calibration)
# ss_baseline.toplevel

# # #%% Evaluate the model at the calibration with differences in a parameter.
# evaluate_param_changes('xi', [0.963 , 0.965, 0.98 , 0.99], ha, baseline_calibration,
#                       ss_vars = ['B', 'D_N', 'D_B', 'D_G', 'asset_mkt', 'C', 'A','Tax','tau_b','T_E_ENDO'])


#%% === Solve the model Steady State ===
#unknowns_ss = {'beta':0.962, 'N':1, 'Tax':0.358}
#targets_ss = {'asset_mkt':0.0, 'labor_mkt':0.0, 'GBC': 0.0}
t_start = time.perf_counter()

unknowns_ss = {'beta':baseline_calibration['beta'], 'vphi':1, 'N':1, 'Tax':0.358}
targets_ss = {'asset_mkt':0.0, 'piwres':0.0, 'labor_mkt':0.0, 'GBC': 0.0}

ss = ha.solve_steady_state(baseline_calibration , unknowns_ss, targets_ss, solver='hybr')

display_ss_durables(ss)
display_calibrated_from_unknowns(ss, unknowns_ss)
check_resource_constraint(ss)

t_end = time.perf_counter()
print(f"total block runtime (solve + displays): {t_end - t_start:.3f} s")


#%% Efficiency of brown old brown

# param_grid = {'eps_b': np.linspace(0.0, 0.5, 3)}

# # Track output, wage, and interest rate
# outputs = ['tau_b','D_N', 'D_B', 'D_BN', 'D_BO', 'D_G', 'D_GN', 'B','beta']
# results = comparative_statics_plot(
#     ha=ha,
#     ss_base=ss,
#     param_grid=param_grid,
#     unknowns_ss=unknowns_ss,
#     targets_ss=targets_ss,
#     outputs=outputs,
#     plot_deviation=False
# )

#%%

param_grid = {'eps_b': np.linspace(0.0, 0.5, 3)}

outputs_shares = ['D_N', 'D_BN',  'D_BO', 'D_GN', 'D_GO']


results = comparative_statics_plot_shares(
    ha=ha,
    ss_base=ss,
    param_grid=param_grid,
    unknowns_ss=unknowns_ss,
    targets_ss=targets_ss,
    outputs=outputs_shares,
    line_labels=["None","Brown New","Brown Old", "Green New", "Green Old"],
    title="SS Durable shares under different brown inefficiency",
    x_label=r"Brown Inefficiency eps_B",
    #x_values=["Low", " ", "Medium", "  ", "High"],
    save_path='../../output/figures/CS_eps_b.png',
    line_colors = ["#808080","#8B4513","#C4A484","#228B22","#90EE90"]
)

#%%
param_grid = {'omega': np.linspace(0.75, 0.85, 3)}

outputs_shares = ['D_N', 'D_BN',  'D_BO', 'D_GN', 'D_GO']


results = comparative_statics_plot_shares(
    ha=ha,
    ss_base=ss,
    param_grid=param_grid,
    unknowns_ss=unknowns_ss,
    targets_ss=targets_ss,
    outputs=outputs_shares,
    line_labels=["None","Brown New","Brown Old", "Green New", "Green Old"],
    title="SS Durable shares under different omega",
    x_label=r"Non-durable consumption share parameter omega",
    #x_values=["Low", " ", "Medium", "  ", "High"],
    save_path='../../output/figures/CS_omega.png',
    line_colors = ["#808080","#8B4513","#C4A484","#228B22","#90EE90"]
)

#%% Price of durable good
param_grid = {'gamma': np.linspace(1.95, 2.05, 3)}

outputs_shares = ['D_N', 'D_BN',  'D_BO', 'D_GN', 'D_GO']


results = comparative_statics_plot_shares(
    ha=ha,
    ss_base=ss,
    param_grid=param_grid,
    unknowns_ss=unknowns_ss,
    targets_ss=targets_ss,
    outputs=outputs_shares,
    line_labels=["None","Brown New","Brown Old", "Green New", "Green Old"],
    title="SS Durable shares under different gamma",
    x_label=r"Relative risk aversion gamma",
    #x_values=["Low", " ", "Medium", "  ", "High"],
    save_path='../../output/figures/CS_gamma.png',
    line_colors = ["#808080","#8B4513","#C4A484","#228B22","#90EE90"]
)




#%%
ss_dict = dict()
ss_dict['baseline'] = ss

policy_functions(ss_dict, plots=['disc'], ie_list=[0], d_list=[0], d_tilde_list=[0, 1, 2, 3, 4], xmax=5.2, figsize=0.8, save_path='../../output/figures/Policy_Functions_dtilde.png')



#%%
plot_distribution(ss, lines_dim = 0, 
                 labels = ['$\\tilde{D}$ = None','$\\tilde{D}$ = New Brown','$\\tilde{D}$ = Old Brown','$\\tilde{D}$ = New Green','$\\tilde{D}$ = Old Green'],
                 truncate_at = 6, save_path='../../output/figures/Distribution_Durables_D_tilde.png')
plot_distribution(ss, lines_dim = 1, 
                 labels = ['None','New Brown','Old Brown','New Green','Old Green'],
                 truncate_at = 6, save_path='../../output/figures/Distribution_Durables_D.png')
plot_distribution(ss, lines_dim = 2, 
                 labels = ['Very Low','Low','Middle','High','Very High'],
                 truncate_at = 6, save_path='../../output/figures/Distribution_Prod.png')
plot_distribution(ss,
                 labels = ['Total'],
                 truncate_at = 6, save_path='../../output/figures/Distribution_Total.png')




#%% === Solve the model transition dynmamics and get IRFs === §
titles = [
        r"Carbon tax rate: $\tau_B$",     
        r"Carbon tax revenues: $T_B$",  
        r"Lump Sum Tax : $T$",     
        r"Share of no durable holding : $D_N$",  
        r"Share of Brown: $D_B$",
        r"Share of New Brown: $D_BN$",
        r"Share of Old Brown: $D_BO$",
        r"Share of Green: $D_G$",
        r"Share of New Green: $D_GN$",
        r"Share of Old Green: $D_GO$",     
        r"Total Consumption: $C$",  
        r"Consu. of Brown energy: $C^B$", 
        r"Consu. of Green energy: $C^G$",
        r"Wage inflation: $\Pi^w$",
        ]
plot_linear_irfs(
    shocks_list=['tau_b'],
    e = {"tau_b": 0.01},
    rho = {"tau_b": 0.80},
    unknowns_td=['G','N','Tax', 'piw'],
    targets_td=['asset_mkt',"labor_mkt", "GBC", "piwres"],
    ha=ha,
    ss=ss,
    #outputs=["tau_b","T_E_ENDO","B", "r", "Z_core","G", "Tax","D_B","D_BO", "D_BN", "D_G","D_GO", "D_GN", "D_N", "goods_mkt", "asset_mkt", "Y_core","C", "C_E", "C_CORE"],
    outputs=["tau_b", "T_E","Tax", "D_N", "D_B", "D_BN", "D_BO", "D_G", "D_GN", "D_GO", "C", "C_E_B", "C_E_G", "piw", "i", "r"],
    titles = titles,
    figsize=(18, 12),
    save_path='../../output/figures/IRFs_tau_b.png',
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
    ss=ss,
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
    ss=ss,
    #outputs=["tau_b","T_E_ENDO","B", "r", "Z_core","G", "Tax","D_B","D_BO", "D_BN", "D_G","D_GO", "D_GN", "D_N", "goods_mkt", "asset_mkt", "Y_core","C", "C_E", "C_CORE"],
    outputs=["r","B","G","C", "Tax", "deficit"],
    titles = titles,
    figsize=(12, 6),
    save_path='../../output/figures/IRFs_T_E_macro.png',
)








# === Analyze the steady state ===
#%%
evaluate_param_changes('gamma_g', [1, 2, 3, 4], ha, ss,
                           ss_vars=['N_core', 'labor_mkt','asset_mkt', 'Tax',
                                    'A', 'B', 'N', 'C', 'C_CORE', 'C_E','D_N','D_G','D_B'])


analyze_steady_state('gamma_g', [1, 2, 3, 4, 10], ss, hh_durables, variables= ['A', 'D_N', 'D_G', 'D_B'])

#%%

analyze_steady_state_3d('gamma_g', [1, 2],'gamma_b', [1, 2], ss, hh_durables, variables= ['A', 'D_N', 'D_G', 'D_B'])


