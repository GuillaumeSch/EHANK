import sequence_jacobian as sj
import numpy as np

@sj.simple
def fiscal(B, r, G, Y_core):
    Tax = (1 + r) * B(-1) + G - B  # total tax burden
    deficit = G - Tax
    return Tax, deficit

@sj.simple
def mkt_clearing(A, B, N, N_core, N_d):
    asset_mkt = A - B
    #labor_mkt = N - N_core - N_d
    labor_mkt = N - (N_core + np.sum(N_d))
    return asset_mkt, labor_mkt

@sj.simple
def prod(Z_core, N_core, alpha):
    Y_core = Z_core * N_core**alpha
    w = N_core**(alpha-1)
    p_core = w / (Z_core * N_core**(alpha-1))
    return Y_core, w, p_core


@sj.simple
def prod_durables(Z_d1, Z_d2, Z_d3, Z_d4, N_d, alpha, w):
    Y_d0 = 0.0
    Y_d1 = Z_d1 * (N_d[0] ** alpha)
    Y_d2 = Z_d2 * (N_d[1] ** alpha)
    Y_d3 = Z_d3 * (N_d[2] ** alpha)
    Y_d4 = Z_d4 * (N_d[3] ** alpha)

    p_d0 = 0.0
    p_d1 = w / (Z_d1 * N_d[0]**(alpha - 1))
    p_d2 = w / (Z_d2 * N_d[1]**(alpha - 1))
    p_d3 = w / (Z_d3 * N_d[2]**(alpha - 1))
    p_d4 = w / (Z_d4 * N_d[3]**(alpha - 1))

    return Y_d0, Y_d1, Y_d2, Y_d3, Y_d4, p_d0, p_d1, p_d2, p_d3, p_d4

# @sj.simple
# def prod_durables(Z_d, N_d, alpha, w):
#     Y_d0 = 0.0
#     Y_d1 = Z_d[0] * (N_d[0] ** alpha)
#     Y_d2 = Z_d[1] * (N_d[1] ** alpha)
#     Y_d3 = Z_d[2] * (N_d[2] ** alpha)
#     Y_d4 = Z_d[3] * (N_d[3] ** alpha)

#     p_d0 = 0.0
#     p_d1 = w / (Z_d[0] * N_d[0]**(alpha - 1))
#     p_d2 = w / (Z_d[1] * N_d[1]**(alpha - 1))
#     p_d3 = w / (Z_d[2] * N_d[2]**(alpha - 1))
#     p_d4 = w / (Z_d[3] * N_d[3]**(alpha - 1))

#     return Y_d0, Y_d1, Y_d2, Y_d3, Y_d4, p_d0, p_d1, p_d2, p_d3, p_d4
