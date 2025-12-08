#%%
from HH_Durables_Block import hh_durables
from Model_Blocks import fiscal, mkt_clearing, prod, rsrce_cstrt, nkpc, inflation, taylor_rule
import sequence_jacobian as sj
import json
from copy import deepcopy
from Fun.my_funs import *
from sequence_jacobian import drawdag
#%matplotlib qt


# %%
# === Calibration dictionary ===
baseline_calibration = {

    # -------------------------
    # 1. Preferences
    # -------------------------
    "beta": 0.959,          # Discount factor
    "eis": 0.5,             # Elasticity of intertemporal substitution
    "gamma": 1/0.5,         # Risk aversion (CRRA parameter)
    "omega": 0.78,          # Weight of nondurable consumption vs durable services
    "taste_shock": 1e-1,    # Idiosyncratic taste shock
    "dbar": 1,              # Subsistence level of durable services

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

    "alpha": 1,             # Labor share in production
    "Z": 1,                 # Productivity in core goods
    "p_core": 1,            # Price of nondurable consumption good
    "Div": 0,               # Dividends (if firms distribute profit)
    "w": 1,                 # Wage

    # -------------------------
    # 5. Durable goods (b = brown, g = green)
    # -------------------------
    "n_b": 2,               # Number of brown vintages
    "n_g": 2,               # Number of green vintages
    "chi": 0.5,             # Resale loss when selling a durable

    # Utility from durables
    "gamma_b": 1.0,         # Utility weight of brown durable
    "gamma_g": 1.0,         # Utility weight of green durable
    "dep_util_frac_b": 1,   # Depreciation (utility) for oldest brown vintage
    "dep_util_frac_g": 1,   # Same for green vintage
    "gamma_mult": 1.0,      # Scale parameter for utility shifter

    # Physical lifetime
    "lifetime_b": 60,       # Brown car durability (quarters)
    "lifetime_g": 60,       # Green car durability (quarters)

    # Durable quantities
    "d0": 0, "d1": 0.02, "d2": 0.018, "d3": 0.03, "d4": 0.025,
    "d_mult": 1.0,          # Scale parameter for durable quantities

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
    "p_e_b": 0.00,         # Brown energy price (gasoline)
    "p_e_g": 0.00,          # Green energy price (electricity)

    "eps_b": 0.40,          # Inefficiency of brown vintage
    "eps_g": 0.20,          # Inefficiency of green vintage

    "tau_b": 0.202,         # Carbon tax applied to brown energy
    "tau_g": 0,             # Subsidy for green energy

    # -------------------------
    # 8. Prices and substitution structure
    # -------------------------
    "xi": 0.97,             # Share parameter for core goods in consumption
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


#%%Goal of the next section: Find some better calibration for higher kappa

#%% NOT SS. BUT SOLUTION OF HH PROBLEM.
ss_0 = ha.steady_state(baseline_calibration)


#%%
unknowns_ss = {'beta':baseline_calibration['beta'],'Tax':baseline_calibration['Tax']}
targets_ss = {'asset_mkt':0.0,'GBC': 0.0}

ss_0 = ha.solve_steady_state(baseline_calibration , unknowns_ss, targets_ss, solver='hybr')
check_resource_constraint(ss_0)
display_ss_durables(ss_0)

#%%
evaluate_param_changes('d_mult', [100, 0.100], ha, ss_0,
                      ss_vars = ['asset_mkt', 'rsrce_cstrt', 'D_N', 'D_B', 'D_G'])


#%%
ss_dict = dict()
ss_dict['baseline'] = deepcopy(ss_0)
ss_dict['alt'] = deepcopy(ss_0)
ss_dict['alt']['dbar'] = 0.1
ss_dict['alt'] = ha.steady_state(ss_dict['alt'])


ss_dict['alt2'] = deepcopy(ss_dict['alt'])
ss_dict['alt2']['gamma_mult'] = 1.5
ss_dict['alt2'] = ha.steady_state(ss_dict['alt2'])


display_ss_durables(ss_dict['alt2'] )



#policy_functions(ss_dict, xmax=5, d_tilde_list=[0], models=['baseline','alt'])

#plot_distribution(ss_0)

#%%
comparative_statics_plot_shares(ha, ss_0, {"d_mult": np.linspace(0.1, 5, 3)}, unknowns_ss, targets_ss, ["D_N", "D_B", "D_G"])


#%%
evaluate_param_changes('d_mult', [0.5, 1.05, 1.50, 2, 5, 10, 20, 150], ha, ss_0,
                      ss_vars = ['asset_mkt', 'rsrce_cstrt', 'D_N', 'D_B', 'D_G'])

#%%
evaluate_param_changes('d_mult', [1, 2, 3, 10, 20, 50], ha, ss_0,
                      ss_vars = ['asset_mkt', 'rsrce_cstrt', 'D_N', 'D_B', 'D_G'])


#%%
evaluate_two_param_changes('d_mult', [1, 2],
                           'dbar', [0.1, 0.5, 1.0],
                           ha, ss_0, ss_vars = ['asset_mkt', 'rsrce_cstrt', 'D_N', 'D_B', 'D_G'])





#%%
IRFs = plot_linear_irfs(
    shocks_list=['tau_b'],
    e = {"tau_b": 0.01},
    rho = {"tau_b": 0.80},
    unknowns_td=['r','Tax'],
    targets_td=['asset_mkt', "GBC"],
    ha=ha,
    ss=ss_0,
    #outputs=["tau_b","T_E_ENDO","B", "r", "Z_core","G", "Tax","D_B","D_BO", "D_BN", "D_G","D_GO", "D_GN", "D_N", "goods_mkt", "asset_mkt", "Y_core","C", "C_E", "C_CORE"],
    outputs=["tau_b", "T_E","Tax", "D_N","D_B", "D_BN", "D_BO", "D_G", "D_GN", "D_GO", "C", "C_E_B", "C_E_G", "i", "r"],
    #titles = titles,
    figsize=(18, 12),
    save_path='../../output/figures/IRFs_tau_b.png',
)

#%%
IRFs = plot_linear_irfs(
    shocks_list=['d1'],
    e = {"d1": 0.01},
    rho = {"d1": 0.80},
    unknowns_td=['r','Tax'],
    targets_td=['asset_mkt', "GBC"],
    ha=ha, 
    ss=ss_0,
    #outputs=["tau_b","T_E_ENDO","B", "r", "Z_core","G", "Tax","D_B","D_BO", "D_BN", "D_G","D_GO", "D_GN", "D_N", "goods_mkt", "asset_mkt", "Y_core","C", "C_E", "C_CORE"],
    outputs=["d1", "T_E","Tax", "D_N","D_B", "D_BN", "D_BO", "D_G", "D_GN", "D_GO", "C", "C_E_B", "C_E_G", "i", "r"],
    #titles = titles,
    figsize=(18, 12),
    save_path='../../output/figures/IRFs_tau_b.png',
)

#%%

J = hh_durables.jacobian(ss_0, ['d1'], T=50)

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







# Fonction D(dtilde) – tu peux la changer
def D_fun(dtilde):
    return dtilde

# Fonction d'utilité
def u_fun(c, dtilde):
    term = c**omega * (dbar + D_fun(dtilde))**(1 - omega)
    return term**(1 - gamma) / (1 - gamma)

# Grille
c_vals = np.linspace(0.1, 2, 100)
dtilde_vals = np.linspace(0.1, 5, 100)
C, DT = np.meshgrid(c_vals, dtilde_vals)

# Calcul utilité
U = u_fun(C, DT)

# -------- Graphique 3D --------
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(C, DT, U, cmap='viridis', edgecolor='none')
ax.set_xlabel("c")
ax.set_ylabel("d̃")
ax.set_zlabel("u(c, d̃)")
plt.title("Surface de la fonction d'utilité")
plt.tight_layout()
plt.show()

# -------- Contour plot --------
plt.figure(figsize=(8, 6))
cp = plt.contourf(C, DT, U, levels=50, cmap='viridis')
plt.colorbar(cp)
plt.xlabel("c")
plt.ylabel("d̃")
plt.title("Contours de u(c, d̃)")
plt.tight_layout()
plt.show()

# %%


shifters = ss_0.internals['hh']['shifters']
c = ss_0['C']

util(c, [0,1,2,3,4], omega, gamma, dbar, shifters)
# %%
def util(ss):
    shifters = ss.internals['hh']['shifters']
    c = ss_0['C']
    
    def util(c, d, omega, gamma, dbar, shifters):
        d_eff = dbar + shifters[d]
        X = (c ** omega) * (d_eff ** (1 - omega))
        if gamma == 1.0:
            # log utility limit
            u = np.log(X)
        else:
            u = (X ** (1 - gamma)) / (1 - gamma)
        return u
    
    
    utily = util(c, [0,1,2,3,4], omega, gamma, dbar, shifters)
    return utily
# %%
