import sequence_jacobian as sj
import numpy as np

@sj.simple
def fiscal(B, r, G, Tax, T_E):
    GBC = Tax - ((1 + r) * B(-1) + G - B - T_E)  # total tax burden
    return GBC

@sj.simple
def mkt_clearing(A, B, N, N_Y):
    asset_mkt = A - B
    labor_mkt = N - N_Y
    return asset_mkt, labor_mkt

#Compute the resource constraint (equivalent of goods market clearing condition)
@sj.simple
def rsrce_cstrt(kappa_d, eps_vec, p_E_vec, p_core, C_CORE, C_E1, C_E2, C_E3, C_E4, G, Y, chi, XPLUS_N, XPLUS_BN, XPLUS_BO, XPLUS_GN, XPLUS_GO, XMINUS_N, XMINUS_BN, XMINUS_BO, XMINUS_GN, XMINUS_GO):
    #Need to construct vector of energy consumption
    C_E_vec = np.array([0, C_E1, C_E2, C_E3, C_E4])
    AD_CORE = p_core*C_CORE + np.sum((1+eps_vec) * p_E_vec * C_E_vec)
    AD_DURABLES = np.sum(kappa_d *([XPLUS_N, XPLUS_BN, XPLUS_BO, XPLUS_GN, XPLUS_GO] - (1-chi)*np.array([XMINUS_N, XMINUS_BN, XMINUS_BO, XMINUS_GN, XMINUS_GO])))
    AD = AD_CORE + AD_DURABLES + G
    AS = Y
    rsrce_cstrt = AD - AS
    return rsrce_cstrt, AD, AD_CORE, AD_DURABLES, AS

@sj.simple
def prod(Y, Z, markup_ss):
    N_Y = Y / Z
    #w = Z_core / markup_ss
    return N_Y

@sj.simple
def nkpc(piw, N, C, vphi, frisch, markup_ss, gamma, beta):
    kappa_w = 0.01 #(1 - theta_w) * (1 - beta * theta_w)/theta_w #to adjust better
    piwres = kappa_w * (vphi * (N)**(1/frisch) - 1/markup_ss * C**(-gamma)) + beta * piw(1) - piw
    return piwres

@sj.simple
def inflation(piw):
    pi = piw
    return pi

@sj.simple
def monetary_taylor(pi, ishock, rss, phi_pi):
    i = rss + phi_pi * pi + ishock
    r_ante = i - pi(1)
    return r_ante, i

@sj.simple
def ex_post_rate(r_ante):
    r = r_ante(-1)
    return r

taylor_rule = sj.combine([monetary_taylor,ex_post_rate], name="Taylor_rule")
