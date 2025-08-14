import sequence_jacobian as sj
import numpy as np

@sj.simple
def fiscal(B, r, G, Y_core):
    Tax = (1 + r) * B(-1) + G - B  # total tax burden
    Z = Y_core - Tax
    deficit = G - Tax
    return Tax, deficit, Z

@sj.simple
def mkt_clearing(A, B, Y_core, C, G):
    asset_mkt = A - B
    goods_mkt = Y_core - C - G
    return asset_mkt, goods_mkt

@sj.simple
def prod(Z_core, N_core, alpha):
    Y_core = Z_core * N_core**alpha
    w = N_core**(alpha-1)
    return Y_core, w

@sj.simple
def prod_durables(Z_d1, Z_d2, Z_d3, Z_d4, N_d1, N_d2, N_d3, N_d4, alpha):
    Y_d1 = Z_d1 * N_d1 **alpha
    Y_d2 = Z_d2 * N_d1 **alpha
    Y_d3 = Z_d3 * N_d1 **alpha
    Y_d4 = Z_d4 * N_d1 **alpha
    Y_d = [Y_d1, Y_d2, Y_d3, Y_d4]
    Z_d = [Z_d1, Z_d2, Z_d3, Z_d4]
    return Y_d1, Y_d2, Y_d3, Y_d4, Y_d, Z_d