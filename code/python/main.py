#%%
from HH_Durables_Block import hh_durables
from Model_Blocks import fiscal, mkt_clearing, prod, prod_durables
import sequence_jacobian as sj
import json
from copy import deepcopy
from Fun.my_funs import *
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
    #"w": 1.0,                  # Wage level
    "N_core": 0.6,             # Labor demand for core goods
    #"N_d1": 0.1, "N_d2": 0.1, "N_d3": 0.1, "N_d4": 0.1,                # Labor demand for durable goods
    "N_d": np.array([ 0.1, 0.1, 0.1, 0.1]),                # Labor demand for durable goods
    #"mu_Z_d": 0.05,            # Fraction that is applied to all labor demand for durables.
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
    #"Tax": 0.5,                # Total tax
    "Z_core": 1,               # Core productivity
    #"Z_d1": 1, "Z_d2": 1, "Z_d3": 1, "Z_d4": 1, # Durables productivities
    "Z_d": np.array([15.52920823, 62.11683292,  1.11111111,  4.44444444]),                # Labor demand for durable goods
    #Government
    #"Y" : 1,                  # Output
    "B" : 4,                   # Stock of debt
    "G" : 0.3,                 # Government spendings
    #Energy
    "p_e_b" : 0.08,             # Price of brown energy (gas/petrol)
    "p_e_g" : 0.04,             # Price of green energy (electricity)
    "tau_b" : 0,               # Carbon tax
    "tau_g" : 0,               # Green energy subsidy
    #Prices
    "xi" : 0.98,                # Governs relative share of core good in non-durable consumption basket. To be improved... Goal, Share_core = 95%
    "nu" : 0.4,                # Elasticity of substitution between core and energy consumption
}

#TO DELETE IN FINAL VERSION. ONLY FOR DEBUGGING

#Import the DD calibration (Some parameters are not present in the DD calibration.)
with open('calibration_ss_DD.json', 'r') as f:
    cali_DD = json.load(f)

cali['baseline_DD'] = deepcopy(cali['baseline'])
for key in cali_DD.keys():
    if key in cali['baseline_DD']:
        cali['baseline_DD'][key] = cali_DD[key]

for k, v in cali["baseline_DD"].items():
    globals()[k] = v

#%% === Create the model ===
ha = sj.create_model([hh_durables, fiscal, mkt_clearing, prod, prod_durables], name="Simple HA Model")
print(ha)
print('It has inputs: ' + str(ha.inputs))
print('It has outputs: ' + str(ha.outputs))


#%% Not SS
#evaluate_param_changes('p_g', [0.1], ha, cali['baseline'],
#                       ss_vars = ['B', 'D_N', 'D_B', 'D_G', 'asset_mkt', 'C', 'A','Tax'])


#%% === Solve the model Steady State ===
unknowns_ss = {'beta':0.946, 'N':1}
targets_ss = {'asset_mkt':0.0, 'labor_mkt':0.0}

#ss_DD = ha.solve_steady_state(cali['baseline'] , unknowns_ss, targets_ss, solver='hybr')
ss_DD = ha.solve_steady_state(cali['baseline'] , unknowns_ss, targets_ss, solver='hybr')

display_ss_durables(ss_DD)
display_calibrated_from_unknowns(ss_DD, unknowns_ss)

check_resource_constraint(ss_DD)


#%% One shot deviation of SS

param_grid = {'gamma_b': np.linspace(0.90, 1.10, 3)}

# Track output, wage, and interest rate
outputs = ['C', 'D_N', 'D_B', 'D_G', 'A']

results = comparative_statics_plot(
    ha=ha,
    ss_base=ss_DD,
    param_grid=param_grid,
    unknowns_ss=unknowns_ss,
    targets_ss=targets_ss,
    outputs=outputs
)



#%% Not SS
evaluate_param_changes('mu_Z_d', [0.05, 0.04, 0.06], ha, ss_DD,
                       ss_vars = ['labor_mkt','B', 'D_N', 'D_B', 'D_BN', 'D_BO', 'D_G', 'D_GN', 'D_GO', 'asset_mkt', 'C', 'A','Tax'])

#%%
ss = dict()
ss['baseline'] = ss_DD

policy_functions(ss, ie_list=[0, 4],  amax=50, figsize=0.8)

#%%
plot_distribution(ss_DD, lines_dim = 0, 
                 labels = ['$\\tilde{D}$ = None','$\\tilde{D}$ = New Brown','$\\tilde{D}$ = Old Brown','$\\tilde{D}$ = New Green','$\\tilde{D}$ = Old Green'],
                 truncate_at = 50)
plot_distribution(ss_DD, lines_dim = 1, 
                 labels = ['$D$ = None','$D$ = New Brown','$D$ = Old Brown','$D$ = New Green','$D$ = Old Green'],
                 truncate_at = 50)
plot_distribution(ss_DD, lines_dim = 2, 
                 labels = ['Prod = Very Low','Prod = Low','Prod = Middle','Prod = High','Prod = Very High'],
                 truncate_at = 50)
#%% Check that the resource constraint holds







#%% === Solve the model transition dynmamics and get IRFs === §
plot_linear_irfs(
    shocks_list=['tau_b'],
    unknowns_td=['G','N'],
    targets_td=['asset_mkt',"labor_mkt"],
    ha=ha,
    ss=ss_DD,
    outputs=["tau_b","N", "G", "Tax","D_BO", "D_BN", "D_GO", "D_GN", "goods_mkt", "asset_mkt"]
)
# %%









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

