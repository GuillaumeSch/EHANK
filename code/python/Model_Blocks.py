import sequence_jacobian as sj
import numpy as np

@sj.simple
def fiscal(B, r, G, Tax, T_E):
    #Tax = (1 + r) * B(-1) + G - B - T_E  # total tax burden
    GBC = Tax - ((1 + r) * B(-1) + G - B - T_E)  # total tax burden
    return GBC

@sj.simple
def mkt_clearing(A, B, N, N_core, N_d):
    asset_mkt = A - B
    labor_mkt = N - (N_core + np.sum(N_d))
    return asset_mkt, labor_mkt

@sj.simple
def prod(Z_core, N_core, markup_ss):
    Y_core = Z_core * N_core
    w = Z_core / markup_ss
    p_core = w / Z_core
    # Div = Y_core - w * N_core - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2*Y_core
    return Y_core, p_core, w


@sj.simple
def prod_durables(Z_d1, Z_d2, Z_d3, Z_d4, N_d, w):
    Y_d0 = 0.0
    Y_d1 = Z_d1 * N_d[0]
    Y_d2 = Z_d2 * N_d[1]
    Y_d3 = Z_d3 * N_d[2]
    Y_d4 = Z_d4 * N_d[3]

    p_d0 = 0.0
    p_d1 = w / Z_d1
    p_d2 = w / Z_d2
    p_d3 = w / Z_d3
    p_d4 = w / Z_d4

    return Y_d0, Y_d1, Y_d2, Y_d3, Y_d4, p_d0, p_d1, p_d2, p_d3, p_d4


# @sj.simple
# def nkpc_ss(Z_core, mu):
#     w = Z_core / mu 
#     return w 

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
