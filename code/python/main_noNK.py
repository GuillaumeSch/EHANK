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
    "beta": 0.95,          # Discount factor
    "gamma": 1/0.8,         # Risk aversion (CRRA parameter)
    "omega": 0.90,          # Weight of nondurable consumption vs durable services
    "taste_shock": 1e-3,    # Idiosyncratic taste shock
    "dbar": 0.05,              # Subsistence level of durable services

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
    "n_a": 100,              # Grid size

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
    "mu_g": 0.98,         # Relative utility of green durable w.r.t brown durable
    "mu_mult": 3.0,      # Scale parameter for utility shifter

    # Physical lifetime
    "lifetime_new": 16,  # New car durability (quarters)
    "lifetime_old": 32,  # Used car durability (quarters)

    # Durable quantities
    "d0": 0.0, "d1": 0.2,     # Quantity of new brown durable. The other quantities will follow from premium and depreciations.
    "x_g": 0.20,            # Green durable premium
    "delta_A": 0.080,       # Depreciation for old durables

    # -------------------------
    # 6. Government
    # -------------------------
    "B": 4,                 # Government debt
    "G": 0.0,               # Government spending
    "Tax": 0.358,           # Lump-sum tax
    "tau": 0,               # Labor income tax rate

    # -------------------------
    # 7. Energy (brown vs green)
    # -------------------------
    "p_e_n": 1.0,          # Energy price for non-holders (basic energy...)
    "p_e_b": 1.0,          # Brown energy price (gasoline)
    "p_e_g": 1.,          # Green energy price (electricity)

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
    "r": 0.05 / 4,          # Quarterly interest rate
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
print(ss['B'])



#%%-------------Graphs for NBB meeting-------------------
#%% Durables shares at baseline SS
display_ss_durables(ss, save_path='../../output/figures/Durable_Shares_SS.png', show_plot=True)


#%%Effet of tau_b on durables shares
comparative_statics_plot_shares(ha, ss, {"tau_b": np.linspace(0, 0.25, 5)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_tau_b.png', title='Effect of carbon tax on durable shares at SS', x_label='Carbon tax on brown energy')

#%%Effet of dbar on durables shares
comparative_statics_plot_shares(ha, ss, {"dbar": np.linspace(0.01, 0.1, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_dbar.png', title='Effect of dbar on durable shares at SS', x_label='Dbar (subsistence level of durables)')

#%%Policy functions
ss_dict = dict()
ss_dict['baseline'] = ss

policy_functions(ss_dict, plots=['assets', 'da', 'cons', 'disc'], ie_list=[4], d_list=[0], d_tilde_list=[0, 1, 2, 3, 4], xmax=5, figsize=0.8, save_path='../../output/figures/Policy_Functions_dtilde.png')






#%% Effet of tau_b on durables shares
comparative_statics_plot_shares(ha, ss, {"tau_b": np.linspace(0, 1, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_tau_b.png', title='Effect of carbon tax on durables shares at SS', x_label='Carbon tax on brown energy')
#%% Effet of eps_b on durables shares
comparative_statics_plot_shares(ha, ss, {"eps_b": np.linspace(0.2, 1, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_eps_b.png')
#%% Effet of d_1 (quantity of BN) on durables shares
comparative_statics_plot_shares(ha, ss, {"d1": np.linspace(1, 3, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_d_1.png')
#%% Effet of d_bar on durables shares
comparative_statics_plot_shares(ha, ss, {"dbar": np.linspace(0.001, 0.02, 10)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_dbar.png')
#%% Effet of gamma_mult on durables shares
comparative_statics_plot_shares(ha, ss, {"mu_mult": np.linspace(1, 5, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_gamma_mult.png')
#%% Effet of gamma_b on durables shares
comparative_statics_plot_shares(ha, ss, {"mu_b": np.linspace(1, 3, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_gamma_b.png')
#%% Effet of gamma_g on durables shares
comparative_statics_plot_shares(ha, ss, {"mu_g": np.linspace(1, 3, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_gamma_g.png')
#%% Effet of p_e_b on durables shares
comparative_statics_plot_shares(ha, ss, {"p_e_b": np.linspace(0.01, 0.5, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_p_e_b.png')
#%% Effet of lifetime_new on durables shares
comparative_statics_plot_shares(ha, ss, {"lifetime_new": np.linspace(40, 80, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_lifetime_new.png')
#%% Effet of chi on durables shares
comparative_statics_plot_shares(ha, ss, {"chi": np.linspace(0, 0.8, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_chi.png')
#%% Effet of w on durables shares
comparative_statics_plot_shares(ha, ss, {"w": np.linspace(0.8, 1.05, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_w.png')
#%% Effet of p_e_n on durables shares
comparative_statics_plot_shares(ha, ss, {"p_e_n": np.linspace(0, 1, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_p_e_n.png')

#%% Effet of beta on durables shares
comparative_statics_plot_shares(ha, ss, {"beta": np.linspace(0.5, 0.95, 10)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_beta.png')
#%% Effet of gamma on durables shares
comparative_statics_plot_shares(ha, baseline_calibration, {"gamma": np.linspace(1/1, 1/5, 10)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_gamma.png')

#%% Effet of omega on durables shares
comparative_statics_plot_shares(ha, baseline_calibration, {"omega": np.linspace(0.5, 0.99, 5)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_omega.png')

#%% Effet of x_g on durables shares
comparative_statics_plot_shares(ha, baseline_calibration, {"x_g": np.linspace(0.1, 0.5, 5)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_x_g.png')

#%% Effet of mu_g on durables shares
comparative_statics_plot_shares(ha, baseline_calibration, {"mu_g": np.linspace(0.97, 1, 5)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_mu_g.png')



#%% Effet of delta_A on durables shares
comparative_statics_plot_shares(ha, baseline_calibration, {"delta_A": np.linspace(0.070, 0.090, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_delta_A.png')



#%%
ss_dict = dict()
ss_dict['baseline'] = ss

policy_functions(ss_dict, plots=['disc'], ie_list=[4], d_list=[1], d_tilde_list=[0, 1, 2, 3, 4], xmax=5, figsize=0.8, save_path='../../output/figures/Policy_Functions_dtilde.png')

#%%Distribution
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
titles = [
        r"Carbon tax rate: $\tau_B$",     
        r"Carbon tax revenues: $T_E$",  
        r"Lump Sum Tax : $T$",     
        r"Share of no durable holding : $D_N$",  
        r"Share of Brown: $D_B$",
        r"Share of New Brown: $D_{BN}$",
        r"Share of Old Brown: $D_{BO}$",
        r"Share of Green: $D_G$",
        r"Share of New Green: $D_{GN}$",
        r"Share of Old Green: $D_{GO}$",     
        r"Total Consumption: $C$",  
        r"Consu. of Brown energy: $C^B$", 
        r"Consu. of Green energy: $C^G$",
        r"Government Expenditures: $G$",
        r"Government Debt: $B$",
        ]
IRFs = plot_linear_irfs(
    shocks_list=['tau_b'],
    e = {"tau_b": 0.10},
    rho = {"tau_b": 0.80},
    unknowns_td=['Tax','B'],
    targets_td=['asset_mkt', "GBC"],
    ha=ha,
    ss=ss,
    #outputs=["tau_b","T_E_ENDO","B", "r", "Z_core","G", "Tax","D_B","D_BO", "D_BN", "D_G","D_GO", "D_GN", "D_N", "goods_mkt", "asset_mkt", "Y_core","C", "C_E", "C_CORE"],
    outputs=["tau_b", "T_E","Tax", "D_N","D_B", "D_BN", "D_BO", "D_G", "D_GN", "D_GO", "C", "C_E_B", "C_E_G", "G", "B"],
    titles = titles,
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


