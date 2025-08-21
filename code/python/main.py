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
    "n_a": 100,                 # Number of asset grid points
    # Labor market
    #"w": 1.0,                  # Wage level
    "N_core": 0.6,             # Labor demand for core goods
    #"N_d1": 0.1, "N_d2": 0.1, "N_d3": 0.1, "N_d4": 0.1,                # Labor demand for durable goods
    "N_d": np.array([ 0.1, 0.1, 0.1, 0.1]),                # Labor demand for durable goods
    "mu_Z_d": 0.05,            # Fraction that is applied to all labor demand for durables.
    "tau":0,                   # Labor income tax
    # Durable goods
    "p_b": 0.80,               # Initial price of brown durable
    "dep_frac_b": 0.25,        # Depreciation green (Fraction of oldest vintage relative to newest)
    "n_b": 2,                  # Number of brown vintages
    "p_g": 0.90,               # Initial price of green durable
    "dep_frac_g": 0.25,        # Depreciation green (Fraction of oldest vintage relative to newest)
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
    "Z_d": np.array([310.58416461, 1242.33665845, 22.22222222, 88.88888889]),                # Labor demand for durable goods
    #Government
    #"Y" : 1,                  # Output
    "B" : 4,                   # Stock of debt
    "G" : 0.3,                 # Government spendings
    #Energy
    "p_e_b" : 0.0,             # Price of brown energy (gas/petrol)
    "p_e_g" : 0.0,             # Price of green energy (electricity)
    "tau_b" : 0,               # Carbon tax
    "tau_g" : 0,               # Green energy subsidy
    #Prices
    "xi" : 1,                # Governs relative share of core good in non-durable consumption basket. To be improved... Goal, Share_core = 95%
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
evaluate_param_changes('p_g', [0.1], ha, cali['baseline'],
                       ss_vars = ['B', 'D_N', 'D_B', 'D_G', 'asset_mkt', 'C', 'A','Tax'])


#%% === Solve the model Steady State ===
#unknowns_ss = {'N':(0.60,0.8),'mu_N_d':(0,1)}
unknowns_ss = {'beta':(0.80,0.99), 'N':(0.60,0.8)}
targets_ss = {'asset_mkt':0.0,'labor_mkt':0}

#ss_DD = ha.solve_steady_state(cali['baseline'] , unknowns_ss, targets_ss, solver='hybr')
ss_DD = ha.solve_steady_state(cali['baseline'] , unknowns_ss, targets_ss, solver='broyden_custom')

display_ss_durables(ss_DD)
display_calibrated_from_unknowns(ss_DD, unknowns_ss)

#%% === Solve the model Steady State ===
#unknowns_ss = {'N':(0.60,0.8),'mu_N_d':(0,1)}
unknowns_ss = {'beta':0.95}
targets_ss = {'asset_mkt':0.0}

#ss_DD = ha.solve_steady_state(cali['baseline'] , unknowns_ss, targets_ss, solver='hybr')
ss_DD = ha.solve_steady_state(cali['baseline'] , unknowns_ss, targets_ss, solver='hybr')

display_ss_durables(ss_DD)
display_calibrated_from_unknowns(ss_DD, unknowns_ss)

#%% Not SS
evaluate_param_changes('mu_N_d', [0.05, 0.04, 0.06], ha, ss_DD,
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
#%%
#Check that the decomposition is correct.
ss_DD['C_P'] - (ss_DD['C_CORE_P_CORE'] + ss_DD['C_E_P_E'])


D = ss_DD.internals['hh']['consav']['D']
V = ss_DD.internals['hh']['consav']['V']
Va = ss_DD.internals['hh']['consav']['Va']


c = ss_DD.internals['hh']['consav']['c']
c_core = ss_DD.internals['hh']['consav']['c_core']
c_E = ss_DD.internals['hh']['consav']['c_E']

p_core = p_core
p_e = ss_DD.internals['hh']['p_e']
p_d = ss_DD.internals['hh']['p_d']
p_bundle = ss_DD.internals['hh']['p_bundle']





P = ss_DD.internals['hh']['durables']['law_of_motion'].P

P[:,:,:,:]

D[:,0,0,0]


xplus = P.copy()  # Create a copy to avoid modifying the original
for i in range(xplus.shape[2]):  # Loop over 3rd dimension
    for j in range(xplus.shape[3]):  # Loop over 4th dimension
        np.fill_diagonal(xplus[:,:,i,j], 0)

xminus = P.copy()  # Create a copy to avoid modifying the original
for i in range(xminus.shape[2]):  # Loop over 3rd dimension
    for j in range(xminus.shape[3]):  # Loop over 4th dimension
        np.fill_diagonal(xminus[:,:,i,j], 0)

Xplus = np.sum(xplus * D, axis=(1,2,3))

Xminus = np.sum(xminus * D, axis=(0,2,3))



C_core_p_core = np.sum(D * c_core * p_core)
C_E_p_e = np.sum(D * c_E * np.array(p_e)[..., np.newaxis, np.newaxis, np.newaxis])
np.sum(D * c * np.array(p_bundle)[..., np.newaxis, np.newaxis, np.newaxis])





ss_DD['C_CORE_P_CORE'] + C_E_p_e + np.sum(p_d * (Xplus - (1-chi)*Xminus))


ss_DD['w'] * N + ss_DD['r'] * ss_DD['A'] + -ss_DD['Tax']



#% Save the variables
D = ss_DD.internals['hh']['consav']['D']
p_bundle = ss_DD.internals['hh']['p_bundle']
c = ss_DD.internals['hh']['consav']['c']
a_choice = ss_DD.internals['hh']['consav']['a']
a_grid = ss_DD.internals['hh']['a_grid']
adj_matrix = ss_DD.internals['hh']['adj_matrix']
r = ss_DD['r']
N = ss_DD['N']
w = ss_DD['w']
e_grid = ss_DD.internals['hh']['e_grid']
T = ss_DD.internals['hh']['T']
A = ss_DD['A']
Tax = ss_DD['Tax']
p_core = ss_DD['p_core']
c_core = ss_DD.internals['hh']['consav']['c_core']
C_core = np.sum(c_core*D)
c_E = ss_DD.internals['hh']['consav']['c_E']
C_E = np.sum(c_E*D, axis=(1,2,3))
p_E = ss_DD.internals['hh']['p_e']
p_d = ss_DD.internals['hh']['p_d']
Y_core = ss_DD['Y_core']
Y_d = ss_DD['Y_d']
G = ss_DD['G']





#% Individual BC
#LHS
lhs = p_bundle[...,np.newaxis,np.newaxis,np.newaxis] * c \
    + a_choice \
    + adj_matrix[..., np.newaxis,np.newaxis]

#RHS
rhs = ((1 + r) * a_grid)[np.newaxis,np.newaxis,np.newaxis,...] \
    + w * N * e_grid[np.newaxis, np.newaxis, :, np.newaxis] \
    + T[np.newaxis, np.newaxis, :, np.newaxis]
    
diff = lhs - rhs

# Aggregated BC
LHS_0 = np.sum(lhs * D)
RHS_0 = np.sum(rhs * D)


# Get X: Sum over e and a, then remove diagonal for inflows/outflows
S = np.sum(D, axis=(2, 3)) 
X_plus = np.sum(S * (1 - np.eye(S.shape[0])), axis=1)
X_minus = np.sum(S * (1 - np.eye(S.shape[0])), axis=0)

#(1)
LHS = p_core*C_core + np.sum(p_E * C_E) + A + np.sum(X_plus * p_d - X_minus * chi * p_d)
RHS = (1+r)*A + w*N - Tax
DIFF = LHS - RHS
print(DIFF)


#(2) using GBC

LHS_1 = p_core*C_core + np.sum(p_E * C_E) + np.sum(X_plus * p_d - X_minus * chi * p_d) + G
RHS_1 = w*N
print(LHS_1 - RHS_1)


print(w*N)
#1) Using Labor clearing (1)
print(w*(ss_DD['N_core'] + np.sum(ss_DD['N_d'])))

#2) Using Prod function
w * (Y_core / ss_DD['Z_core'] + np.sum(Y_d / (mu_Z_d * np.mean(ss_DD['Z_d']))))

#3) Using pricing equation
w*(p_core * Y_core / w + np.sum(p_d * Y_d / w))



# Aggregate resource constraint
#LHS
LHS_2 = p_core*C_core + np.sum((1+tau_b)**(-1)*p_E * C_E) + np.sum(X_plus * p_d) + G

RHS_2 = p_core * Y_core + np.sum(p_d * (Y_d + (1-chi)*X_minus))

print(LHS_2 - RHS_2)

print(ss_DD['labor_mkt'])






#%% === Solve the model transition dynmamics and get IRFs === §
plot_linear_irfs(
    shocks_list=['tau_b'],
    unknowns_td=['G','N'],
    targets_td=['asset_mkt',"goods_mkt"],
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

