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
    "B": 1,                 # Government debt
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
    
    
    "rss": 0.05 / 4,          
    "phi_pi": 1.5,
    "ishock": 0,
    "piw": 0.0,          
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
hh_solution





#%%
unknowns_ss = {'B':baseline_calibration['B'],'Tax':baseline_calibration['Tax']}
targets_ss = {'asset_mkt':0.0,'GBC': 0.0}

ss = ha.solve_steady_state(baseline_calibration , unknowns_ss, targets_ss, solver='hybr')
check_resource_constraint(ss)
display_ss_durables(ss)
print(ss['B'])

#%% This simply allows to satisfy the GBC. Not the equilibrium.
unknowns_ss = {'Tax':baseline_calibration['Tax']}
targets_ss = {'GBC': 0.0}

sol_GBC = ha.solve_steady_state(baseline_calibration , unknowns_ss, targets_ss, solver='hybr')
check_resource_constraint(ss)
display_ss_durables(ss)
print(ss['B'])
print(ss['Tax'])

#%%
comparative_statics_plot(ha, ss, {"G": np.linspace(0.05 / 4, 0.06 / 4, 2)}, unknowns_ss, targets_ss, ["B", "Tax", "G","C","asset_mkt","A"], plot_deviation=False)


#%%
comparative_statics_plot(ha, ss, {"r": np.linspace(0.05 / 4, 0.08 / 4, 4)}, unknowns_ss, targets_ss, ["B", "Tax", "G","C","asset_mkt","A"], plot_deviation=False)

#%%
comparative_statics_plot(ha, ss, {"B": np.linspace(1, 4, 4)}, unknowns_ss, targets_ss, ["B", "Tax", "G","C","asset_mkt","A"], plot_deviation=False)


#%% This is the propor equilibrium
unknowns_ss = {'Tax':sol_GBC['Tax'],'r':sol_GBC['r']}
targets_ss = {'GBC': 0.0,'asset_mkt': 0.0}

ss = ha.solve_steady_state(baseline_calibration , unknowns_ss, targets_ss, solver='hybr')
check_resource_constraint(ss)
display_ss_durables(ss)
print(ss['B'])
print(ss['Tax'])

#%%
comparative_statics_plot(ha, ss, {"tau_b": np.linspace(0, 0.20, 4)}, unknowns_ss, targets_ss, ["B", "Tax", "G","C","r","asset_mkt","A","D_B","D_G","D_N"], plot_deviation=False)


#%%-------------Graphs for NBB meeting-------------------
#%% Durables shares at baseline SS
display_ss_durables(ss, save_path='../../output/figures/Durable_Shares_SS_restr.png', show_plot=True, durables_to_plot="restricted")
display_ss_durables(ss, save_path='../../output/figures/Durable_Shares_SS_full.png', show_plot=True, durables_to_plot="full")



#%%Effet of tau_b on durables shares
comparative_statics_plot_shares(ha, ss, {"tau_b": np.linspace(0, 0.25, 10)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_tau_b_full.png', title='Effect of carbon pricing on durable shares at SS', x_label='Carbon pricing on brown energy')
comparative_statics_plot_shares(ha, ss, {"tau_b": np.linspace(0, 0.25, 10)}, unknowns_ss, targets_ss, ["D_N", "D_B", "D_G"], save_path='../../output/figures/comp_stat_tau_b_restr.png', title='Effect of carbon pricing on durable shares at SS', x_label='Carbon pricing on brown energy')

#%%Effet of dbar on durables shares
comparative_statics_plot_shares(ha, ss, {"dbar": np.linspace(0.01, 0.1, 3)}, unknowns_ss, targets_ss, ["D_N", "D_BN", "D_BO", "D_GN", "D_GO"], save_path='../../output/figures/comp_stat_dbar.png', title='Effect of dbar on durable shares at SS', x_label='Dbar (subsistence level of durables)')

#%%Policy functions
ss_dict = dict()
ss_dict['baseline'] = ss

policy_functions(ss_dict, plots=['disc'], ie_list=[2], d_list=[0], d_tilde_list=[0, 1, 2, 3, 4], xmax=10, vintage_groups={"None": [0], "Brown": [1, 2],"Green": [3, 4]}, figsize=0.8, save_path='../../output/figures/Policy_Functions_dtilde_0_2.png')
policy_functions(ss_dict, plots=['disc'], ie_list=[4], d_list=[0], d_tilde_list=[0, 1, 2, 3, 4], xmax=10, figsize=0.8, save_path='../../output/figures/Policy_Functions_dtilde_0_4.png')


#%% Shares at each wealth level
plot_durable_choice_shares(ss, truncate_at=50, save_path='../../output/figures/Durable_Choice_Shares_by_Wealth.png', title='Durable choice shares by wealth level')

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
titles = [
        r"Carbon pricing: $\tau_B$",     
        r"Carbon tax revenues: $T_E$",  
        r"Lump Sum Tax : $T$",     
        r"Share of no durable holding : $D_N$",
        r"Share of no durable holding (Choice) : $D_{T_N}$",  
        r"Share of Brown: $D_B$",
        r"Share of New Brown: $D_{BN}$",
        r"Share of Old Brown: $D_{BO}$",
        r"Share of Green: $D_G$",
        r"Share of New Green: $D_{GN}$",
        r"Share of Old Green: $D_{GO}$",     
        r"Total Consumption: $C$",  
        r"Consu. of Brown energy: $C^B$", 
        r"Consu. of Green energy: $C^G$",
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
    outputs=["tau_b", "T_E","Tax", "D_N","D_T_N","D_B", "D_BN", "D_BO", "D_G", "D_GN", "D_GO", "C", "C_E_B", "C_E_G", "B", "r"],
    titles = titles,
    figsize=(18, 12),
    save_path='../../output/figures/IRFs_tau_b.png',
)

#%% IRFS Macro variables
titles = [
        r"Carbon Pricing: $\tau_B$",     
        r"Share of Brown: $D_B$",
        r"Share of Green: $D_G$", 
        r"Total Consumption: $C$",  
        r"Consu. of Brown Energy: $C^B$", 
        r"Consu. of Green Energy: $C^G$",
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
    outputs=["tau_b","D_B", "D_G", "C", "C_E_B", "C_E_G"],
    titles = titles,
    figsize=(9, 5),
    save_path='../../output/figures/IRFs_tau_b_macro.png',
)

#%% IRFS Inequality variables




IRFs = plot_linear_irfs(
    shocks_list=['tau_b'],
    e = {"tau_b": 0.10},
    rho = {"tau_b": 0.80},
    unknowns_td=['Tax','B'],
    targets_td=['asset_mkt', "GBC"],
    ha=ha,
    ss=ss,
    #outputs=["tau_b","T_E_ENDO","B", "r", "Z_core","G", "Tax","D_B","D_BO", "D_BN", "D_G","D_GO", "D_GN", "D_N", "goods_mkt", "asset_mkt", "Y_core","C", "C_E", "C_CORE"],
    outputs=["tau_b","C", "C_CORE", "C_0_0_2_10", "C_1_1_2_10", "C_2_2_2_10", "C_3_3_2_10", "C_4_4_2_10", "C_1_1_2_0","C_1_1_2_19", "V_0_2_10", "V_1_2_10", "V_2_2_10", "V_3_2_10", "V_4_2_10"],
    #titles = titles,
    figsize=(18, 12),
    save_path='../../output/figures/IRFs_tau_b_ineq.png',
)

# %%
titles = [
        r"Carbon Pricing: $\tau_B$",     
        r"Total Consumption: $C$",
        r"Total Core Consumption: $C_{core}$", 
        r"$C_{core}(d = N) = \sum c(\tilde{d},d = N,e,a) \cdot g(\tilde{d},d = N,e,a)$",  
        r"$C_{core}(d = BN) = \sum c(\tilde{d},d = BN,e,a) \cdot g(\tilde{d},d = BN,e,a)$",
        r"$C_{core}(d = BO) = \sum c(\tilde{d},d = BO,e,a) \cdot g(\tilde{d},d = BO,e,a)$",
        r"$C_{core}(d = GN) = \sum c(\tilde{d},d = GN,e,a) \cdot g(\tilde{d},d = GN,e,a)$",
        r"$C_{core}(d = GO) = \sum c(\tilde{d},d = GO,e,a) \cdot g(\tilde{d},d = GO,e,a)$",
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
    outputs=["tau_b","C", "C_CORE", "C_0", "C_1", "C_2", "C_3", "C_4"],
    titles = titles,
    figsize=(18, 12),
    save_path='../../output/figures/IRFs_tau_b_ineq.png',
)
# %%







titles = [
        r"Carbon pricing: $\tau_B$",     
        r"Carbon tax revenues: $T_E$",  
        r"Lump Sum Tax : $T$",     
        r"Share of no durable holding : $D_N$",
        r"Share of no durable holding (Choice) : $D_{T_N}$",  
        r"Share of Brown: $D_B$",
        r"Share of New Brown: $D_{BN}$",
        r"Share of Old Brown: $D_{BO}$",
        r"Share of Green: $D_G$",
        r"Share of New Green: $D_{GN}$",
        r"Share of Old Green: $D_{GO}$",     
        r"Total Consumption: $C$",  
        r"Consu. of Brown energy: $C^B$", 
        r"Consu. of Green energy: $C^G$",
        r"Government Debt: $B$",
        ]
IRFs = plot_linear_irfs(
    shocks_list=['tau_b'],
    e = {"tau_b": 0.010},
    rho = {"tau_b": 0.80},
    unknowns_td=['Tax','d0'],
    targets_td=["GBC",'asset_mkt'],
    ha=ha,
    ss=ss,
    #outputs=["tau_b","T_E_ENDO","B", "r", "Z_core","G", "Tax","D_B","D_BO", "D_BN", "D_G","D_GO", "D_GN", "D_N", "goods_mkt", "asset_mkt", "Y_core","C", "C_E", "C_CORE"],
    outputs=["tau_b", "T_E","Tax", "D_N","D_T_N","D_B", "D_BN", "D_BO", "D_G", "D_GN", "D_GO", "C", "C_E_B", "C_E_G", "B", "r","GBC","asset_mkt","N","labor_mkt","w"],
    titles = titles,
    figsize=(18, 12),
    #save_path='../../output/figures/IRFs_tau_b.png',
)
# %%


unknowns__1 = {'N':1,'Tax':ss['Tax'],'r':ss['r']}
targets_1 = {'labor_mkt':0.0,'GBC': 0.0,'asset_mkt': 0.0}

comparative_statics_plot(ha, ss, {"Y": np.linspace(0.99, 1.01, 3)}, unknowns__1, targets_1, ["B","r", "Tax", "G","C","asset_mkt","A","N_Y","Y","N","labor_mkt"], plot_deviation=False)

# %%
IRFs = plot_linear_irfs(
    shocks_list=['tau_b','G'],
    e = {"tau_b": 0.010,"G": 0.001},
    rho = {"tau_b": 0.80,"G": 0.80},
    unknowns_td=['Tax','Y','N'],
    targets_td=["GBC",'asset_mkt','labor_mkt'],
    ha=ha,
    ss=ss,
    #outputs=["tau_b","T_E_ENDO","B", "r", "Z_core","G", "Tax","D_B","D_BO", "D_BN", "D_G","D_GO", "D_GN", "D_N", "goods_mkt", "asset_mkt", "Y_core","C", "C_E", "C_CORE"],
    outputs=["tau_b", "T_E","Tax", "D_N","D_T_N","D_B", "D_BN", "D_BO", "D_G", "D_GN", "D_GO", "C", "C_E_B", "C_E_G", "B", "r","GBC","asset_mkt","N","labor_mkt","w","Y","goods_mkt","G"],
    titles = titles,
    figsize=(18, 12),
    #save_path='../../output/figures/IRFs_tau_b.png',
)
# %%
unknowns = ['Tax','Y','N']
targets = ["GBC",'asset_mkt','labor_mkt']
inputs = ['G']
drawdag(ha, unknowns, targets, inputs)