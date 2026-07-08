import sequence_jacobian as sj
import numpy as np

@sj.simple
def fiscal(B, r, G_ss, Tax, T_E):
    G = G_ss - 0.10 * (B - B.ss)
    GBC = Tax - ((1 + r) * B(-1) + G - B - T_E)  # total tax burden
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
def inflation(piw):
    pi = piw
    return pi

@sj.simple
def monetary_taylor(pi, ishock, rss, phi_pi, p_e_b):
    i = rss + phi_pi * pi + ishock + 0.01*(p_e_b-p_e_b.ss)/p_e_b.ss
    r_ante = i - pi(1)
    return r_ante, i

@sj.simple
def monetary_real(pi, ishock, rss):
    i = rss + pi(1) + ishock
    r_ante = i - pi(1)
    return r_ante, i

@sj.simple
def ex_post_rate(r_ante):
    r = r_ante(-1)
    return r

taylor_rule = sj.combine([monetary_taylor,ex_post_rate], name="Taylor_rule")
real_rule = sj.combine([monetary_real,ex_post_rate], name="Real_rule")

