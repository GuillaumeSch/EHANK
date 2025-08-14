import sequence_jacobian as sj
import numpy as np

@sj.simple
def fiscal(B, r, G, Y_core):
    Tax = (1 + r) * B(-1) + G - B  # total tax burden
    deficit = G - Tax
    return Tax, deficit

@sj.simple
def mkt_clearing(A, B, Y_core, C, G, N, N_core, N_d1, N_d2, N_d3, N_d4):
    asset_mkt = A - B
    goods_mkt = Y_core - C - G
    #labor_mkt = N - N_core - N_d
    labor_mkt = N - (N_core + N_d1 + N_d2 + N_d3 + N_d4)
    return asset_mkt, goods_mkt, labor_mkt

@sj.simple
def prod(Z_core, N_core, alpha):
    Y_core = Z_core * N_core**alpha
    w = N_core**(alpha-1)
    return Y_core, w

@sj.simple
def prod_durables(Z_d1, Z_d2, Z_d3, Z_d4, N_d1_b, N_d2_b, N_d3_b, N_d4_b, alpha, mu_N_d):
    N_d1, N_d2, N_d3, N_d4 = mu_N_d * N_d1_b, mu_N_d * N_d2_b, mu_N_d * N_d3_b, mu_N_d * N_d4_b
    #N_d = np.sum(N_d1, N_d2, N_d3, N_d4)
    Y_d1 = Z_d1 * N_d1 **alpha
    Y_d2 = Z_d2 * N_d2 **alpha
    Y_d3 = Z_d3 * N_d3 **alpha
    Y_d4 = Z_d4 * N_d4 **alpha
    Y_d = [Y_d1, Y_d2, Y_d3, Y_d4]
    Z_d = [Z_d1, Z_d2, Z_d3, Z_d4]
    return Y_d1, Y_d2, Y_d3, Y_d4, Y_d, Z_d, N_d1, N_d2, N_d3, N_d4