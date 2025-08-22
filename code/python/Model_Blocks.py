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
def prod_durables(Z_d, N_d, alpha, mu_Z_d, w):
    Z_d = np.array(Z_d, dtype=float) #Should be nd - 1
    N_d = np.array(N_d, dtype=float) #Should be nd - 1
    Z_d_scaled = mu_Z_d * Z_d
    # Production function for each vintage
    Y_d = np.concatenate(([0.0], Z_d_scaled * (N_d ** alpha)))
    p_d = np.concatenate(([0.0], w / (Z_d_scaled * N_d**(alpha-1))))
    return Y_d, Z_d_scaled, p_d

# @sj.simple
# def prod_durables(Z_d, N_d, alpha, mu_Z_d, w):
#     Z_d = np.array(Z_d, dtype=float)  # e.g. [Z_b1, Z_b2, Z_g1, Z_g2]
#     N_d = np.array(N_d, dtype=float)  # e.g. [N_b1, N_b2, N_g1, N_g2]
#     Z_d_scaled = mu_Z_d * Z_d
#     # Production for each vintage (add "0" vintage at the front)
#     Y_d = np.concatenate(([0.0], Z_d_scaled * (N_d ** alpha)))
#     p_d = np.concatenate(([0.0], w / (Z_d_scaled * N_d**(alpha-1))))
#     # Explicit outputs (assuming order: [b1, b2, g1, g2])
#     Y_d0, Y_d1, Y_d2, Y_d3, Y_d4 = Y_d
#     Z_d1, Z_d2, Z_d3, Z_d4 = Z_d_scaled
#     p_d0, p_d1, p_d2, p_d3, p_d4 = p_d
#     return (
#         Y_d0, Y_d1, Y_d2, Y_d3, Y_d4,
#         Z_d1, Z_d2, Z_d3, Z_d4,
#         p_d0, p_d1, p_d2, p_d3, p_d4,
#     )