import sequence_jacobian as sj
import numpy as np

@sj.simple
def fiscal(B, r, G_ss, kappa_g, Tax, T_E):
    G = G_ss - kappa_g * (B - B.ss)
    GBC = Tax - ((1 + r) * B(-1) + G - B - T_E)
    return GBC, G

@sj.simple
def others(D_B, tau_b, p_e_b):
    D_B_target = D_B - D_B.ss
    p_e_b_net = (1+tau_b)*p_e_b
    return D_B_target, p_e_b_net

@sj.simple
def mkt_clearing(A, B, N, N_D, C_E_B, C_E_B_S):
    asset_mkt = A - B
    labor_mkt = N - N_D
    brown_energy_mkt = C_E_B - C_E_B_S
    return asset_mkt, labor_mkt, brown_energy_mkt

#Compute the resource constraint.
@sj.simple
def rsrce_cstrt(C, C_CORE, C_E, Y, G, D_GB, psi_g, p_e_b, p_e_g, C_E_B, C_E_G):
    p_E_vec = np.array([p_e_b, p_e_g])
    C_E_vec = np.array([C_E_B, C_E_G])
    AD_NONDURABLES = C_CORE + np.sum(p_E_vec * C_E_vec)
    AD_DURABLES = (D_GB) * psi_g
    AD = AD_NONDURABLES + AD_DURABLES + G
    AS = Y
    rsrce_cstrt = AD - AS
    return rsrce_cstrt, AD, AD_NONDURABLES, AD_DURABLES, AS

@sj.simple
def prod(Y, Z):
    N_D = Y / Z
    return N_D

@sj.simple
def nkpc(piw, N, vphi, frisch, markup_ss, gamma, beta, theta_w, w, Y, C, UCE):
    kappa_w = (1 - theta_w) * (1 - beta * theta_w)/(theta_w*(1+vphi*(markup_ss/(markup_ss-1))))
    wnkpc = kappa_w * (vphi * (N)**(1/frisch) - 1/markup_ss * w * UCE) + beta * piw(1) - piw
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
def headline_inflation(piw, p_e_b, p_e_g, omega_core, omega_eb, omega_eg):
    """
    Aggregate (headline) consumer price inflation.

    omega_core/eb/eg are FIXED steady-state nominal expenditure shares,
    computed once outside the model (see main.py) and passed in as plain
    calibration constants — NOT recomputed from hh outputs each period.
    This is required to avoid a cyclic dependency (hh -> Taylor_rule ->
    headline_inflation -> hh), since C_CORE/C_E_B/C_E_G are genuine
    outputs of hh, while omega_* only need their steady-state value
    (first-order-exact Laspeyres weights, see derivation in previous notes).
    """
    pi_core = piw
    pi_e_b  = (p_e_b - p_e_b(-1)) / p_e_b(-1)
    pi_e_g  = (p_e_g - p_e_g(-1)) / p_e_g(-1)
    pi_headline = pi_core + omega_eb * pi_e_b + omega_eg * pi_e_g
    return pi_headline

@sj.simple
def monetary_taylor(pi_core, ishock, rss, phi_pi):
    i = rss + phi_pi * pi_core + ishock
    r_ante = i - pi_core(1)
    return r_ante, i

@sj.simple
def monetary_real(pi_core, ishock, rss):
    i = rss + pi_core(1) + ishock
    r_ante = i - pi_core(1)
    return r_ante, i

@sj.simple
def ex_post_rate(r_ante):
    r = r_ante(-1)
    return r

taylor_rule = sj.combine([monetary_taylor,ex_post_rate], name="Taylor_rule")
real_rule = sj.combine([monetary_real,ex_post_rate], name="Real_rule")

