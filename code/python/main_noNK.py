#%%
from HH_Durables_Block import hh_durables
from Model_Blocks import fiscal, mkt_clearing, prod, rsrce_cstrt, nkpc, inflation, taylor_rule
import sequence_jacobian as sj
import json
from copy import deepcopy
from Fun.my_funs import *
from sequence_jacobian import drawdag
#%matplotlib qt
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)


# %%
# === Calibration dictionary ===
baseline_calibration = {

    # -------------------------
    # 1. Preferences
    # -------------------------
    "beta": 0.80,          # Discount factor
    "eis": 0.5,             # Elasticity of intertemporal substitution
    "gamma": 1/2,         # Risk aversion (CRRA parameter)
    "omega": 0.78,          # Weight of nondurable consumption vs durable services
    "taste_shock": 1e-1,    # Idiosyncratic taste shock
    "dbar": 0.005,              # Subsistence level of durable services

    # Labor disutility
    "frisch": 1,            # Frisch elasticity of labor supply
    "vphi": 1,              # Scale of disutility from work

    # -------------------------
    # 2. Productivity (idiosyncratic)
    # -------------------------
    "rho_e": 0.95,          # Persistence of productivity shocks
    "sd_e": 0.5,            # Std. dev. of productivity shocks
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
    "w": 1,                 # Wage

    # -------------------------
    # 5. Durable goods (b = brown, g = green)
    # -------------------------
    "n_b": 2,               # Number of brown vintages
    "n_g": 2,               # Number of green vintages
    "chi": 0.05,             # Resale loss when selling a durable

    # Utility from durables
    "mu_b": 1.0,         # Utility weight of brown durable
    "mu_g": 1.0,         # Utility weight of green durable
    "dep_util_b": 0,     # Depreciation (utility) for oldest brown vintage
    "dep_util_g": 0,     # Same for green vintage
    "mu_mult": 1.0,      # Scale parameter for utility shifter

    # Physical lifetime
    "lifetime_b": 60,       # Brown car durability (quarters)
    "lifetime_g": 60,       # Green car durability (quarters)

    # Durable quantities
    "d0": 0.0, "d1": 0.02, "d2": 0.02*0.85, "d3": 0.025, "d4": 0.025*0.85,
    "d_mult": 100.0,          # Scale parameter for durable quantities

    # -------------------------
    # 6. Government
    # -------------------------
    "B": 4,                 # Government debt
    "G": 0.3,               # Government spending
    "Tax": 0.358,           # Lump-sum tax
    "tau": 0,               # Labor income tax rate

    # -------------------------
    # 7. Energy (brown vs green)
    # -------------------------
    "p_e_n": 0.0,          # Energy price for non-holders (basic energy...)
    "p_e_b": 1.0,          # Brown energy price (gasoline)
    "p_e_g": 1.0,          # Green energy price (electricity)

    "eps_b": 0.40*0,          # Inefficiency of brown vintage
    "eps_g": 0.20*0,          # Inefficiency of green vintage

    "tau_b": 0.202*0,         # Carbon tax applied to brown energy
    "tau_g": 0,             # Subsidy for green energy

    # -------------------------
    # 8. Prices and substitution structure
    # -------------------------
    "xi": 0.90,             # Share parameter for core goods in consumption
    "nu": 0.4,              # Elasticity of substitution core vs energy
    "markup_ss": 1,         # Steady-state markup

    # -------------------------
    # 9. Financial environment
    # -------------------------
    "r": 0.06 / 4,          # Quarterly interest rate
}


#TO DELETE IN FINAL VERSION. ONLY FOR DEBUGGING
for k, v in baseline_calibration.items():
    globals()[k] = v

#%% === Create the model ===
#ha = sj.create_model([hh_durables, fiscal, mkt_clearing, prod, rsrce_cstrt], name="Simple HA Model")
ha = sj.create_model([hh_durables, fiscal, mkt_clearing, prod], name="Simple HA Model")
print(ha)
print('It has inputs: ' + str(ha.inputs))
print('It has outputs: ' + str(ha.outputs))



#%% NOT SS. BUT SOLUTION OF HH PROBLEM.
hh_solution = ha.steady_state(baseline_calibration)


#%%
unknowns_ss = {'B':baseline_calibration['B'],'Tax':baseline_calibration['Tax']}
targets_ss = {'asset_mkt':0.0,'GBC': 0.0}

ss = ha.solve_steady_state(baseline_calibration , unknowns_ss, targets_ss, solver='hybr')
check_resource_constraint(ss)
display_ss_durables(ss)

#%% Effet of tau_b on durables shares
comparative_statics_plot_shares(ha, ss, {"tau_b": np.linspace(0, 1, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_tau_b.png')
#%% Effet of eps_b on durables shares
comparative_statics_plot_shares(ha, ss, {"eps_b": np.linspace(0.2, 0.6, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_eps_b.png')
#%% Effet of d_mult on durables shares
comparative_statics_plot_shares(ha, ss, {"d_mult": np.linspace(1, 2, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_d_mult.png')
#%% Effet of d_1 (quantity of BN) on durables shares
comparative_statics_plot_shares(ha, ss, {"d1": np.linspace(0.02, 0.06, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_d_1.png')
#%% Effet of d_2 (quantity of BO) on durables shares
comparative_statics_plot_shares(ha, ss, {"d2": np.linspace(0.02, 0.06, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_d_2.png')
#%% Effet of d_3 (quantity of GN) on durables shares
comparative_statics_plot_shares(ha, ss, {"d3": np.linspace(0.02, 0.06, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_d_3.png')
#%% Effet of d_4 (quantity of GO) on durables shares
comparative_statics_plot_shares(ha, ss, {"d4": np.linspace(0.02, 0.06, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_d_4.png')
#%% Effet of d_bar on durables shares
comparative_statics_plot_shares(ha, ss, {"dbar": np.linspace(0.001, 0.02, 10)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_dbar.png')
#%% Effet of gamma_mult on durables shares
comparative_statics_plot_shares(ha, ss, {"mu_mult": np.linspace(1, 3, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_gamma_mult.png')
#%% Effet of gamma_b on durables shares
comparative_statics_plot_shares(ha, ss, {"mu_b": np.linspace(1, 3, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_gamma_b.png')
#%% Effet of gamma_g on durables shares
comparative_statics_plot_shares(ha, ss, {"mu_g": np.linspace(1, 3, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_gamma_g.png')
#%% Effet of p_e_b on durables shares
comparative_statics_plot_shares(ha, ss, {"p_e_b": np.linspace(0.01, 0.5, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_p_e_b.png')
#%% Effet of lifetime_b on durables shares
comparative_statics_plot_shares(ha, ss, {"lifetime_b": np.linspace(40, 80, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_lifetime_b.png')
#%% Effet of chi on durables shares
comparative_statics_plot_shares(ha, ss, {"chi": np.linspace(0, 0.8, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_chi.png')
#%% Effet of w on durables shares
comparative_statics_plot_shares(ha, ss, {"w": np.linspace(0.8, 1.05, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_w.png')




#%%
ss_dict = dict()
ss_dict['baseline'] = ss

policy_functions(ss_dict, plots=['disc'], ie_list=[2], d_list=[0], d_tilde_list=[0, 1, 2, 3, 4], xmax=5, figsize=0.8, save_path='../../output/figures/Policy_Functions_dtilde.png')



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

#%%
evaluate_param_changes('d_mult', [100, 0.100], ha, ss,
                      ss_vars = ['asset_mkt', 'rsrce_cstrt', 'D_N', 'D_B', 'D_G'])


#%%
ss_dict = dict()
ss_dict['baseline'] = deepcopy(ss)
ss_dict['alt'] = deepcopy(ss)
ss_dict['alt']['dbar'] = 0.1
ss_dict['alt'] = ha.steady_state(ss_dict['alt'])


ss_dict['alt2'] = deepcopy(ss_dict['alt'])
ss_dict['alt2']['gamma_mult'] = 1.5
ss_dict['alt2'] = ha.steady_state(ss_dict['alt2'])


display_ss_durables(ss_dict['alt2'] )



#policy_functions(ss_dict, xmax=5, d_tilde_list=[0], models=['baseline','alt'])

#plot_distribution(ss)

#%%
comparative_statics_plot_shares(ha, ss, {"d_mult": np.linspace(0.1, 5, 3)}, unknowns_ss, targets_ss, ["D_N", "D_B", "D_G"])


#%%
evaluate_param_changes('d_mult', [0.5, 1.05, 1.50, 2, 5, 10, 20, 150], ha, ss,
                      ss_vars = ['asset_mkt', 'rsrce_cstrt', 'D_N', 'D_B', 'D_G'])

#%%
evaluate_param_changes('d_mult', [1, 2, 3, 10, 20, 50], ha, ss,
                      ss_vars = ['asset_mkt', 'rsrce_cstrt', 'D_N', 'D_B', 'D_G'])


#%%
evaluate_two_param_changes('d_mult', [1, 2],
                           'dbar', [0.1, 0.5, 1.0],
                           ha, ss, ss_vars = ['asset_mkt', 'rsrce_cstrt', 'D_N', 'D_B', 'D_G'])





#%%
IRFs = plot_linear_irfs(
    shocks_list=['tau_b'],
    e = {"tau_b": 0.10},
    rho = {"tau_b": 0.80},
    unknowns_td=['G','B'],
    targets_td=['asset_mkt', "GBC"],
    ha=ha,
    ss=ss,
    #outputs=["tau_b","T_E_ENDO","B", "r", "Z_core","G", "Tax","D_B","D_BO", "D_BN", "D_G","D_GO", "D_GN", "D_N", "goods_mkt", "asset_mkt", "Y_core","C", "C_E", "C_CORE"],
    outputs=["tau_b", "T_E","Tax", "D_N","D_B", "D_BN", "D_BO", "D_G", "D_GN", "D_GO", "C", "C_E_B", "C_E_G", "i", "r", "G", "B"],
    #titles = titles,
    figsize=(18, 12),
    save_path='../../output/figures/IRFs_tau_b.png',
)


#%%

J = hh_durables.jacobian(ss, ['d1'], T=50)

plt.plot(J['D_G']['d1'][:20, 0], label='test')
plt.axhline(0, color='gray', linestyle=':')
plt.legend()
plt.show()


#%%
# #%% Evaluate the model at the calibration with differences in a parameter.
evaluate_param_changes('r', [0.016, 0.02, 0.0], ha, ss_3,
                      ss_vars = ['asset_mkt', 'diff_core','diff_BN', 'diff_GN', 'diff_BO', 'diff_GO', 'labor_mkt'])

#%%
drawdag(ha)
#%% 

ss_baseline = ha.steady_state(baseline_calibration)
ss_baseline.toplevel

check_resource_constraint(ss_baseline)


#%%



#%%


