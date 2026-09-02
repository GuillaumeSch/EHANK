#%%
"""Auclert-Rognlie-Straub (2023) HA model. Extracted verbatim from the replication
notebook; only the two edits noted inline. Reference baseline for the E-HANK merge."""

# ===== nb cell 3 =====
import numpy as np
#import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sequence_jacobian as sj


# ===== nb cell 10 =====
calibration = dict(r = 0.01,            # Real interest rate
                   eis = 1,             # Elasticity of intertemporal substitution
                   eta_E = 0.1,         # Elasticity of Energy with (H, F) bundles (outer nest)
                   frisch = 0.5,        # Elasticity of labor supply
                   alpha_E = 0.04,      # Spending share on imported E energy
                   E_supply_elasticity = np.inf,  # Supply elasticity of imports (SOE: infinite)
                   markup_ss = 1.03,    
                   sd_e = 0.57,
                   rho_e = 0.92,
                   n_e = 7,             # Number of income states
                   n_beta = 3,          # Number of betas
                   beta_spread = 0.06,  # Spread between betas
                   beta_max = 0.95,
                   min_a = 0, max_a = 400, n_a = 150, # Asset grid
                   zetaF = 0, zetaE = 0, zetaEsupply=0,
                   prodE_es=0.1,
                   theta_E = 0.65,
                   theta_F=0.9
                   )

# ===== nb cell 12 =====
calibration['alpha'] = 0.3
chi_target = 0.3

# ===== nb cell 14 =====
# Monetary policy rule
calibration.update({'rho_i': 0, 'phi_pi': 0, 'phi_pie': 1, 'phi_piw': 0}) # Real rate rule

# Fiscal policy
calibration.update({'B': 0, 'psiB': 0.04, 'tauY': 0, 'epsT': 0, 'insE': 0, 'tauE': 0, 'bb': 0}) 

# Real wage rigidity
calibration.update({'w_BG': 5}) # real wage smoothing motive, 0 = none

# International supply
calibration.update({'Gamma_arb': 100}) # intertemporal arbitrage, inf = none

# Other
calibration.update(dict(ghh_prefs=0,  # use GHH preferences? 0 = no
                        w_index=0,    # wage indexation parameter, 0 = no indexation to past CPI inflation, 1 = full indexation
                        scale_w=0,    # whether non-homotheticity scales with wage ( = 1) or is lump sum ( = 0)
                        cbarE=0,      # is this non-homotheticity?
                        eps_dcp=1,    # ad hoc dampening factor dampening gamma?
                        pcX_home=1,   # where are profits booked?
                        wealth_effect=1, # wealth effect in union wedge
                        prodE_share=0, # is energy used in production?
                        lambda_c = 0.25 # share of constrained agents in TA model
                       ))

# ===== nb cell 16 =====
# Openness
calibration['alpha_F'] = (calibration['alpha']-calibration['alpha_E'])/(1-calibration['alpha_E'])
# Spending share on imported F products
calibration['alpha_F_tilde'] = (1-calibration['alpha_E'])*calibration['alpha_F']
# Elasticity of export demand
calibration['gamma'] = (chi_target - (1-calibration['alpha'])*(1-calibration['alpha_F'])*calibration['eta_E']) / ((1-calibration['alpha'])*calibration['alpha_F'] + calibration['alpha'])
# Elasticity of substitution between H and F (inner nest)
calibration['eta'] = calibration['gamma']

# ===== nb cell 18 =====
calibration.update(
        {'Q': 1, 'y': 1, 'ishock': 0, 'Cstar': 1, 'piw': 0, 'pi':0, 'P': 1, 'B': 0, 'nfa': 0, 'PFstar': 1, 'W': 1, 'Z': calibration['markup_ss'],
         'rstar': calibration['r'], 'pH_PHF': 1, 'pHstar': 1, 'pF_PHF': 1, 'pEhh_P': 1, 'pHF_P': 1, 'dividend_X': 0, 'vphi': 1,
         'rante': calibration['r'], 'beta_RA': 1/(1+calibration['r']), 'C': 1, 'A': 1, 'w':1, 
         'alphastar': calibration['alpha'], 'E_supply_shock': calibration['alpha_E']})

calibration.update(dict(
    PEstar_shock = 1,  # shock to dollar price of E
    PEstar = 1,  # dollar price of E
    PFstar = 1,  # dollar price of F
    inom_t = 0,
    union_wedge = 0
))

calibration['cE_ss_grid_0'] = np.zeros((calibration['n_e'],calibration['n_a']))
calibration['cE_ss_grid_1'] = np.zeros((calibration['n_e'],calibration['n_a']))
calibration['cE_ss_grid_2'] = np.zeros((calibration['n_e'],calibration['n_a']))
calibration['cE_ss'] = 0
calibration['c_ss_grid_0'] = np.ones((calibration['n_e'],calibration['n_a']))
calibration['c_ss_grid_1'] = np.ones((calibration['n_e'],calibration['n_a']))
calibration['c_ss_grid_2'] = np.ones((calibration['n_e'],calibration['n_a']))

# ===== nb cell 20 =====
# If you want energy endowment in baseline
markup_original = calibration['markup_ss']
zetaEsupply = 0.33
calibration.update({'zetaEsupply':zetaEsupply,
                    'y': 1-zetaEsupply*calibration['alpha_E'],
                    'markup_ss': calibration['markup_ss'] * (1-zetaEsupply*calibration['alpha_E']),
                    'alphastar': calibration['alpha'] - calibration['alpha_E']*zetaEsupply,
                    'Z': calibration['markup_ss'] * (1-zetaEsupply*calibration['alpha_E'])})

# To achieve piw
calibration.update({'theta_w': 0.938})

# ===== nb cell 23 =====
def hh_init(coh, r, eis):
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return Va

@sj.het(exogenous='Pi', policy='a', backward='Va', backward_init=hh_init)
def hh(Va_p, a_grid, r, beta_g, eis, coh, ghh):
    uc_nextgrid = beta_g * Va_p
    c_nextgrid = uc_nextgrid ** (-eis) + ghh
    a = sj.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    sj.misc.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * (c - ghh) ** (-1 / eis)
    return Va, a, c

def make_grids(rho_e, sd_e, n_e, min_a, max_a, n_a):
    e_grid, _, Pi = sj.grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    a_grid = sj.grids.asset_grid(min_a, max_a, n_a)
    return e_grid, Pi, a_grid

def hh_income(e_grid, atw_n, r, pEhh_P, cbarE, scale_w, markup_ss, a_grid, n, frisch, ghh_prefs, epsT, cE_ss_grid, insE, pE_P, pE_P_ss):
    Tf = - pEhh_P * cbarE * (atw_n * markup_ss) * scale_w - pEhh_P * cbarE * (1 - scale_w)
    Tfiscal = epsT + insE * (pE_P - pE_P_ss) * cE_ss_grid
    coh = (1 + r) * a_grid + atw_n * e_grid[:, np.newaxis] + Tf + Tfiscal
    n_ss = 1
    ghh = ghh_prefs * 1/(1+1/frisch) * (n**(1+1/frisch) - n_ss**(1+1/frisch))
    return coh, ghh

def compute_weighted_mpc(c, a_grid, r, e_grid):
    """Approximate mpc out of wealth, with symmetric differences where possible, exactly setting mpc=1 for constrained agents."""
    mpc = np.empty_like(c)
    post_return = (1 + r) * a_grid
    mpc[:, 1:-1] = (c[:, 2:] - c[:, 0:-2]) / (post_return[2:] - post_return[:-2])
    mpc[:, 0] = (c[:, 1] - c[:, 0]) / (post_return[1] - post_return[0])
    mpc[:, -1] = (c[:, -1] - c[:, -2]) / (post_return[-1] - post_return[-2])
    # mpc[a == a_grid[0]] = 1
    mpc = mpc * e_grid[:, np.newaxis]
    return mpc

def inequality(c, c_ss_grid):
    c2 = c ** 2 
    logc = np.log(c)
    logc2 = np.log(c) ** 2 
    
    diff_c = c - c_ss_grid
    diff_c2 = (c - c_ss_grid) ** 2
    diff_logc = np.log(c) - np.log(c_ss_grid)
    diff_logc2 = (np.log(c) - np.log(c_ss_grid)) ** 2
    return c2, logc, logc2, diff_c, diff_c2, diff_logc, diff_logc2

hh_ha = hh.add_hetinputs([make_grids, hh_income]).add_hetoutputs([compute_weighted_mpc, inequality])
group_vars = ['C','A','MPC','cE_ss_grid','c_ss_grid','C2','LOGC2','LOGC', 'DIFF_C', 'DIFF_C2', 'DIFF_LOGC', 'DIFF_LOGC2']
hh_ha_list = [hh_ha.rename(suffix=f'_{i}').remap({f'{x}': f'{x}_{i}' for x in group_vars}).remap({'beta_g': f'beta_{i}'}) for i in range(3)]

@sj.simple
def group_betas(beta_spread,beta_max,n_beta):
    beta_2 = beta_max
    beta_1 = beta_max - beta_spread/2
    beta_0 = beta_max - beta_spread
    return beta_0, beta_1, beta_2

@sj.simple
def aggregate_groups(C_0,C_1,C_2,A_0,A_1,A_2,MPC_0,MPC_1,MPC_2,beta_0, beta_1, beta_2):
    C = (C_0+C_1+C_2)/3
    A = (A_0+A_1+A_2)/3
    MPC = (MPC_0+MPC_1+MPC_2)/3
    beta = (beta_0+beta_1+beta_2)/3
    return C, A, MPC, beta

hh_ha = sj.create_model(hh_ha_list+[group_betas,aggregate_groups])

# ===== nb cell 30 =====
@sj.simple
def hh_outputs(C, atw_n, pH_PHF, pEhh_P, pF_PHF, pHF_P, eta, eta_E, alpha_E, alpha_F, markup_ss, cbarE, scale_w):
    alpha_F_tilde = (1-alpha_E)*alpha_F
    cH = (1-alpha_E - alpha_F_tilde) * (pH_PHF)**(-eta) * pHF_P**(-eta_E) * C
    cF = alpha_F_tilde * (pF_PHF)**(-eta) * pHF_P**(-eta_E) * C
    cE = cbarE*(atw_n*markup_ss)*scale_w + cbarE*(1-scale_w) + alpha_E * pEhh_P**(-eta_E) * C
    return cH, cF, cE

# ===== nb cell 32 =====
@sj.solved(unknowns={'J': 15., 'j': 15.}, targets=['Jres', 'jres'], solver="broyden_custom")
def income(y, w, Z, pH_P, pE_P, J, j, rante, dividend_X, tauY, pcX_home, markup_ss, prodE_share, prodE_es):
    
    prodE = prodE_share * (pH_P/pE_P) ** prodE_es * y
    
    if prodE_share == 0:
        n = y / Z 
    else:
        n = (1-prodE_share) * (y/Z) * ((markup_ss * w) / (Z * pH_P)) ** (-prodE_es)
        
    btw_n = w * n 
    atw_n = (1-tauY) * btw_n
    atw = atw_n / n
    gdp = Z * n
    
    dividend = (markup_ss - 1) * w * n

    div_tot = dividend
    if pcX_home == 1: div_tot += dividend_X
    Jres = div_tot + J(1) / (1 + rante) - J
    jres = J(1) / (1 + rante) - j
    
    return jres, Jres, atw_n, dividend, gdp, atw, n, btw_n, prodE

# ===== nb cell 33 =====
@sj.simple
def profitcenters(Q, pH_P, cHstar, eps_dcp):
    dividend_X = (Q ** (1 - eps_dcp) * pH_P ** (eps_dcp) - pH_P) * cHstar
    return dividend_X

# ===== nb cell 36 =====
@sj.solved(unknowns={'piF':0.,'PF':1.}, targets=['piF_res','PFres'], solver="broyden_custom")
def foreignPrices(piF,PF,P,Q,PFstar,rante,theta_F):
    PFres = (1+piF) * PF(-1) - PF
    pF_P = PF / P
    beta_F = 1/(1+rante.ss)
    kappa_F = (1 - theta_F) * (1 - beta_F * theta_F) / theta_F
    piF_term = Q * PFstar / pF_P - 1
    piF_res = kappa_F * piF_term + beta_F * piF(1) * (1+piF(1)) - piF * (1+piF) 
    return piF_res, PFres, pF_P

@sj.solved(unknowns={'piE':0.,'PE':1.}, targets=['piE_res','PEres'], solver="broyden_custom")
def energyPrices(piE,PE,P,Q,PEstar,rante,theta_E,tauE):
    PEres = (1+piE) * PE(-1) - PE
    pE_P = PE / P
    beta_E = 1/(1+rante.ss)
    kappa_E = (1 - theta_E) * (1 - beta_E * theta_E) / theta_E
    piE_term = Q * PEstar / pE_P - 1
    piE_res = kappa_E * piE_term + beta_E * piE(1) * (1+piE(1)) - piE * (1+piE)
    
    pE_P_ss = pE_P.ss # Looks like you cant use PE.ss inside a het input(?)
    pEhh_P = (1-tauE) * pE_P + tauE * pE_P.ss
    pEhh = pEhh_P * P
    return piE_res, PEres, pE_P, pE_P_ss, pEhh_P, pEhh

importPrices = sj.combine([foreignPrices,energyPrices])

# ===== nb cell 37 =====
@sj.solved(unknowns={'JF': 0., 'JE': 0}, targets=['JF_res', 'JE_res'], solver="broyden_custom")
def importProfits(JF, JE, pF_P, pE_P, Q, PFstar, PEstar, cF, cE, prodE, rante):
    DF = (pF_P - Q * PFstar) * cF
    JF_res = DF + JF(1)/(1+rante) - JF
    jF = JF(1)/(1+rante)
    
    DE = (pE_P - Q * PEstar) * (cE + prodE)
    JE_res = DE + JE(1)/(1+rante) - JE
    jE = JE(1)/(1+rante)
    return JF_res, jF, JE_res, jE, DF, DE

# ===== nb cell 39 =====
@sj.simple
def revaluation(r, j, J, jF, JF, zetaF, jE, JE, zetaE,j_Esupply,J_Esupply,zetaEsupply):
    r_res = (J + zetaF * JF + zetaE * JE + zetaEsupply * J_Esupply)/(j(-1) + zetaF * jF(-1) + zetaE * jE(-1) + zetaEsupply * j_Esupply(-1)) - 1 - r
    return r_res 

# ===== nb cell 41 =====
@sj.simple
def foreign_c(pHstar, alphastar, gamma, Cstar, eps_dcp):
    cHstar = alphastar * pHstar ** (-gamma * eps_dcp) * Cstar
    return cHstar

# ===== nb cell 43 =====
@sj.solved(unknowns={'Q': (0.1, 2)}, targets=['uip'])
def UIP(Q, rante, rstar):
    uip = Q / Q(+1) * (1 + rante) - (1 + rstar)
    return uip

# ===== nb cell 45 =====
@sj.solved(unknowns={'piw': (-2, 2)}, targets=['piwres'], solver="brentq")
def unions(n,C,atw,w,vphi,w_BG,theta_w,beta,markup_ss,frisch,eis,piw,union_wedge):
    kappa_w = (1 - theta_w) * (1 - beta * theta_w) / theta_w
    psi_nr_inv = kappa_w  / (vphi * (n**(1+(1/frisch))))
    
    BG_term = (w-w.ss)*w * C.ss**(-1/eis) * n.ss / (markup_ss * n * w.ss)
    piwterm = vphi * n ** (1/frisch) - (atw*(C**(-1/eis)))/markup_ss - w_BG * BG_term + union_wedge
    piwres = piw*(1+piw) - beta*piw(1)*(1+piw(1)) - psi_nr_inv * n * piwterm
    
    return piwres, piwterm

@sj.solved(unknowns={'W': (0.5, 2)}, targets=['Wres'], solver="brentq")
def piW_to_W(piw,W):
    Wres = W(-1) * (1 + piw) - W
    return Wres

# ===== nb cell 47 =====
@sj.solved(unknowns={'pHF_P': 1., 'pH_PHF': 1.}, targets=['inner_nest', 'outer_nest'])
def CESprices(Q, eta, alpha_F, gamma, PFstar, PEstar, pHF_P, pH_PHF, eta_E, alpha_E, pEhh_P, pF_P):

    alpha = alpha_E + (1-alpha_E)*alpha_F
    
    pF_PHF = pF_P / pHF_P
    pH_P = pH_PHF * pHF_P
    pHstar = pH_P / Q

    # next define the nests
    if eta_E == 1:
        inner_nest = (pHF_P) ** (1 - alpha_E) * (pEhh_P) ** alpha_E - 1
    else:
        inner_nest = (1 - alpha_E) * (pHF_P) ** (1 - eta_E) + alpha_E * (pEhh_P) ** (1 - eta_E) - 1

    if eta == 1:
        outer_nest = (pH_PHF) ** (1 - alpha_F) * (pF_PHF) ** alpha_F - 1
    else:
        outer_nest = (1-alpha_F) * (pH_PHF)**(1-eta) + alpha_F * (pF_PHF)**(1-eta) - 1

    chi_tilde = (1-alpha) * (alpha_F * eta + (1-alpha_F) * eta_E) + alpha*gamma
    return chi_tilde, pF_PHF, inner_nest, outer_nest, pH_P, pHstar, alpha

# ===== nb cell 48 =====
@sj.simple
def price_levels(piw, W, w, Z, pH_P, Q, P, PE, markup_ss, prodE_es, prodE_share):
    if prodE_es == 1:
        PH = (markup_ss / Z * W) ** (1-prodE_share) * PE ** prodE_share
    else:
        PH = ((1-prodE_share)*(markup_ss / Z * W)**(1-prodE_es) + prodE_share*(PE)**(1-prodE_es)) ** (1/(1-prodE_es))
    
    pires = PH / pH_P - P
    E = P * Q 
    piH = PH / PH(-1) - 1
    w_res = W / P - w
    return PH, pires, E, piH, w_res

# ===== nb cell 49 =====
@sj.simple
def annualize(pi, piw, inom, r, rante):
    pi_ann = (1+pi) ** 4 - 1 
    piw_ann = (1+piw) ** 4 - 1
    inom_ann = (1+inom) ** 4 - 1
    r_ann = (1+r) ** 4 - 1
    rante_ann = (1+rante) ** 4 - 1
    
    return pi_ann, piw_ann, inom_ann, r_ann, rante_ann

# ===== nb cell 50 =====
@sj.solved(unknowns={'P': (0.5, 2)}, targets=['Pres'])
def pitop(P,pi):   
    Pres = P - (1+pi)*P(-1)
    return Pres

# ===== nb cell 52 =====
@sj.solved(unknowns={'inom': (-1, 1)}, targets=['inom_res'])
def mon_policy(ishock, rstar, pi, phi_pi, phi_pie, rho_i, inom, phi_piw, w, P, inom_t):
    W_here = w * P
    piw_here = (W_here/W_here(-1)) - 1
    inom_res = -(1+inom) + rho_i * (1+inom(-1)) + (1-rho_i) * (1 + rstar.ss) * (1 + phi_pi * pi) * (1 + phi_piw * piw_here) * (1 + phi_pie * pi(+1)) + ishock
    rante = (1 + inom)/(1 + pi(+1)) - 1
    inom_t_res = inom - inom_t # if you want to choose ishock to target a particular path for nominal rates
    return rante, inom_res, inom_t_res


# ===== nb cell 54 =====
@sj.solved(unknowns={'B': (-1, 1)}, targets=['B_res'])
def fiscal(B, rante, btw_n, psiB, tauY, epsT, insE, pE_P, cE, tauE, bb):
    Tuntargeted = epsT
    Ttargeted = insE * (pE_P - pE_P.ss) * cE.ss 
    Subsidy = tauE * (pE_P - pE_P.ss) * cE 
    
    spending = Tuntargeted + Ttargeted + Subsidy
    taxation = tauY * btw_n
    
    B_res = (1 + rante(-1)) * B(-1) + spending - taxation - B
    tauY_res = (1-bb)*(psiB * (B(-1) - B.ss) - tauY) + bb*(B-B.ss)
    return B_res, tauY_res, Tuntargeted, Ttargeted, spending, taxation

# ===== nb cell 56 =====
@sj.solved(unknowns={'nfa': (-2, 2)}, targets=['nfares'], solver="brentq")
def CA(nfa, Q, pHstar, cHstar, PFstar, cF, PEstar, cE, prodE, rante, r, A, rdom, Adom,zetaEsupply,E_supply,y):
    exports = Q * (pHstar * cHstar + PEstar * zetaEsupply * E_supply)
    imports = Q * (PFstar * cF + PEstar * (cE+prodE))
    imports_pc = imports / imports.ss
    exports_pc = exports / exports.ss
    
    netexports = Q * (pHstar * cHstar - PFstar * cF - PEstar * (cE+prodE-zetaEsupply*E_supply))
    revaluation_term = (r - rante(-1)) * A(-1) - (rdom - rante(-1)) * Adom(-1)
    nfares = netexports + revaluation_term + (1 + rante(-1)) * nfa(-1) - nfa

    nx_gdp = netexports / y 
    return nfares, netexports, revaluation_term, exports, imports, nx_gdp, imports_pc, exports_pc

# ===== nb cell 57 =====
@sj.simple
def revaluation_dom(j, J, jF, JF, jE, JE, zetaEsupply, j_Esupply, J_Esupply):
    rdom = (J + JF + JE + zetaEsupply*J_Esupply)/(j(-1) + jF(-1) + jE(-1) + zetaEsupply*j_Esupply(-1)) - 1
    Adom = j + jF + jE + zetaEsupply*j_Esupply
    return rdom, Adom

# ===== nb cell 59 =====
@sj.solved(unknowns={'J_Esupply': (0, 100)}, targets=['J_Esupply_res'])
def IEA(J_Esupply,PEstar,P,rstar,Gamma_arb,E_supply_shock, rante,Q, zetaEsupply):
    E_stock = ((PEstar(1)/(1+rstar)) - PEstar) / Gamma_arb
    E_supply = E_supply_shock + (E_stock(-1) - E_stock) 
    
    D_Esupply = Q * (PEstar * E_supply) 
    J_Esupply_res = D_Esupply + J_Esupply(1)/(1+rante) - J_Esupply
    j_Esupply = J_Esupply(1)/(1+rante)
    D_Esupply_H = zetaEsupply * D_Esupply # the dividends that go to home
    return E_supply, D_Esupply, J_Esupply_res, j_Esupply, D_Esupply_H, E_stock

# ===== nb cell 61 =====
@sj.simple
def eqm_cond(y, cH, cHstar, A, gdp, nfa, j, jF, jE, B, cE, prodE, PEstar, PEstar_shock, E_supply_elasticity, E_supply_shock, E_supply, zetaF, zetaE, rante, r, Q, pHstar, PFstar, cF, netexports, E, P, DF, pF_P, DE, pE_P, pH_P, dividend, markup_ss, btw_n, J, JF, JE, zetaEsupply, j_Esupply):
    goods_clearing = cH + cHstar - y
    assets_clearing = A - nfa - j - jF - jE - B - zetaEsupply*j_Esupply
    if E_supply_elasticity == np.inf:
        E_clearing = PEstar - PEstar_shock
    else:
        E_clearing = (cE+prodE) - E_supply
    PEstar_diff = PEstar - PEstar_shock
    gdp_t = gdp - 1
    return goods_clearing, assets_clearing, E_clearing, PEstar_diff, gdp_t

# ===== nb cell 63 =====
def test_targets(dictionary, extra_checks=[], err_tol=1e-5, noisy=False):
    for target in (['uip', 'piwres', 'nfares', 'goods_clearing', 'assets_clearing', 'Pres', 'w_res', 'E_clearing', 'inner_nest', 'outer_nest'] + extra_checks):
        assert np.isclose(np.max(dictionary[target]), 0, atol=err_tol), print(target+' : '+str(np.linalg.norm(dictionary[target])))
        if noisy:
            print(target+' : '+str(np.linalg.norm(dictionary[target])))
    return

# ===== nb cell 65 =====
model_ha = sj.combine([hh_ha, hh_outputs, foreign_c, revaluation, mon_policy, fiscal, income, importPrices, importProfits, profitcenters, UIP, eqm_cond, CA,
                       unions, CESprices, price_levels, piW_to_W, pitop, revaluation_dom, annualize, IEA])

# Temporary fix to allow (guess_min,guess,guess_max): comment out lines 36--39 in steady_state.py. 
ss_ha = model_ha.solve_steady_state(calibration, unknowns={'vphi': 1, 'beta_max': 0.984}, targets=['piwres', 'nfares'], ttol=1e-14, dissolve=['unions', 'UIP', 'CA', 'piW_to_W', 'pitop'])

test_targets(ss_ha) 

# ===== nb cell 67 =====
ss_ha['cE_ss_grid_0'] = ss_ha.internals['hh_0']['c'] * ss_ha['cE'] / ss_ha['C']
ss_ha['cE_ss_grid_1'] = ss_ha.internals['hh_1']['c'] * ss_ha['cE'] / ss_ha['C']
ss_ha['cE_ss_grid_2'] = ss_ha.internals['hh_2']['c'] * ss_ha['cE'] / ss_ha['C']
ss_ha['cE_ss'] = ss_ha['cE']

ss_ha['c_ss_grid_0'] = ss_ha.internals['hh_0']['c']
ss_ha['c_ss_grid_1'] = ss_ha.internals['hh_1']['c']
ss_ha['c_ss_grid_2'] = ss_ha.internals['hh_2']['c']

# ===== nb cell 75 =====
T = 100
unknowns_td = {'y', 'pi', 'r', 'tauY', 'PEstar','w'}
targets_td = {'goods_clearing', 'pires', 'r_res', 'tauY_res', 'E_clearing','w_res'}

# ===== nb cell 79 =====
half_life = 16
shock_size = 1
rho = 2 ** (-1/half_life) 

shocks = {'PEstar_shock': (shock_size*rho**np.arange(T))}

# %%
