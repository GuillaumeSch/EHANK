import sequence_jacobian as sj
import numpy as np

@sj.simple
def fiscal(B, r, G_ss, kappa_g, Tax, T_E):
    G = G_ss - kappa_g * (B - B.ss)
    GBC = Tax - ((1 + r) * B(-1) + G - B - T_E)
    return GBC, G

@sj.simple
def others(D_B):
    D_B_target = D_B - D_B.ss
    return D_B_target

@sj.simple
def mkt_clearing(A, B, N, N_D):
    asset_mkt = A - B
    labor_mkt = N - N_D
    return asset_mkt, labor_mkt

#Compute the resource constraint
@sj.simple
def rsrce_cstrt(C_CORE, Y, G, D_GB, psi_g, p_e_b, p_e_g, C_E_B, C_E_G):
    p_E_vec = np.array([p_e_b, p_e_g])
    C_E_vec = np.array([C_E_B, C_E_G])
    AD_NONDURABLES = C_CORE + np.sum(p_E_vec * C_E_vec)
    AD_DURABLES = (D_GB) * psi_g
    AD = AD_NONDURABLES + AD_DURABLES + G
    AS = Y
    rsrce_cstrt = AD - AS
    return rsrce_cstrt, AD, AD_NONDURABLES, AD_DURABLES, AS

@sj.simple
def rsrce_cstrt_leak(C, C_CORE, C_E, Y, G, D_GB, psi_g, p_e_b, p_e_g, C_E_B, C_E_G, leakage):
    p_E_vec = np.array([p_e_b, p_e_g])
    C_E_vec = np.array([C_E_B, C_E_G])
    AD_NONDURABLES = C_CORE + np.sum(p_E_vec * C_E_vec)
    AD_DURABLES = D_GB * psi_g
    AD = AD_NONDURABLES + AD_DURABLES + G
    AS = Y

    # Leakage total
    #rsrce_cstrt = (AS - AS.ss) - (1 - leakage) * (AD - AD.ss)
    
    # Brown Leakage
    brown_leakage = leakage * p_e_b * (C_E_B - C_E_B.ss)
    rsrce_cstrt = (AS - AS.ss) - (AD - AD.ss) - brown_leakage

    
    return rsrce_cstrt, AD, AD_NONDURABLES, AD_DURABLES, AS

@sj.simple
def rsrce_cstrt_leak_E(C_CORE, Y, G, D_GB, psi_g, p_e_b, p_e_g, C_E_B, C_E_G, leakage_E):
    # Boris's baseline identity is Y - C = p_E*E (balanced trade every period):
    # the energy import bill must be financed by extra domestic production/exports.
    # leakage_E relaxes this: the DEVIATION of the energy bill from its SS level
    # is (partially) excluded from the domestic resource constraint, i.e. financed
    # by a temporary current account imbalance instead of extra domestic output.
    # leakage_E=0 nests the current baseline exactly (no SS re-solve needed).
    p_E_vec = np.array([p_e_b, p_e_g])
    C_E_vec = np.array([C_E_B, C_E_G])
    energy_bill = np.sum(p_E_vec * C_E_vec)

    # .ss must be read off the raw inputs (p_e_b.ss, C_E_B.ss, ...) BEFORE any
    # np.array()/np.sum() packaging — intermediate variables like energy_bill
    # don't carry a .ss attribute, only the function's own input arguments do.
    energy_bill_ss = p_e_b.ss * C_E_B.ss + p_e_g.ss * C_E_G.ss

    AD_NONDURABLES = C_CORE + energy_bill - leakage_E * (energy_bill - energy_bill_ss)
    AD_DURABLES = D_GB * psi_g
    AD = AD_NONDURABLES + AD_DURABLES + G
    AS = Y
    rsrce_cstrt = AD - AS
    return rsrce_cstrt, AD, AS, energy_bill

@sj.simple
def prod(Y, Z):
    N_D = Y / Z
    return N_D

@sj.simple
def nkpc(piw, N, vphi, frisch, markup_ss, gamma, beta_bar, theta_w, w, Y, C, UCE):
    kappa_w = (1 - theta_w) * (1 - beta_bar * theta_w)/(theta_w*(1+vphi*(markup_ss/(markup_ss-1))))
    wnkpc = kappa_w * (vphi * (N)**(1/frisch) - 1/markup_ss * w * UCE) + beta_bar * piw(1) - piw
    return wnkpc

@sj.simple
def nkpc_ss(N, frisch, markup_ss, gamma, w, C, UCE):
    vphi =  1/markup_ss * w * UCE / (N)**(1/frisch)
    wnkpc = vphi * (N)**(1/frisch) - 1/markup_ss * w * UCE
    return wnkpc, vphi

@sj.simple
def core_inflation(piw):
    pi_core = piw
    return pi_core

@sj.simple
def headline_inflation(piw, p_e_b, p_e_g, omega_eb, omega_eg):
    pi_core = piw
    pi_e_b  = (p_e_b - p_e_b(-1)) / p_e_b(-1)
    pi_e_g  = (p_e_g - p_e_g(-1)) / p_e_g(-1)
    pi_headline = pi_core + omega_eb * pi_e_b + omega_eg * pi_e_g
    return pi_headline

# @sj.simple
# def monetary_taylor(pi_core, ishock, rss, phi_pi):
#     i = rss + phi_pi * pi_core + ishock
#     r_ante = i - pi_core(1)
#     return r_ante, i

@sj.solved(unknowns={"i": (-0.2, 0.2)}, targets=["taylor_resid"])
def monetary_taylor_headline(i, pi_headline, pi_core, ishock, rss, phi_pi, rho_i):
    taylor_resid = i - (rho_i * i(-1) + (1 - rho_i) * (rss + phi_pi * pi_headline) + ishock)
    r_ante = i - pi_core(1)
    return r_ante, taylor_resid

@sj.simple
def monetary_real(pi_core, ishock, rss):
    i = rss + pi_core(1) + ishock
    r_ante = i - pi_core(1)
    return r_ante, i

@sj.simple
def ex_post_rate(r_ante):
    r = r_ante(-1)
    return r

# taylor_rule = sj.combine([monetary_taylor,ex_post_rate], name="Taylor_rule")
taylor_rule_headline = sj.combine([monetary_taylor_headline,ex_post_rate], name="Taylor_rule_headline")
real_rule = sj.combine([monetary_real,ex_post_rate], name="Real_rule")

