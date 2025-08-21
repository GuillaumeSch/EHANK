import sequence_jacobian as sj
import numpy as np

@sj.simple
def fiscal(B, r, G, Y_core):
    Tax = (1 + r) * B(-1) + G - B  # total tax burden
    deficit = G - Tax
    return Tax, deficit

@sj.simple
def mkt_clearing(A, B, Y_core, C, G, N, N_core, N_d):
    asset_mkt = A - B
    goods_mkt = Y_core - C - G
    #labor_mkt = N - N_core - N_d
    labor_mkt = N - (N_core + np.sum(N_d))
    return asset_mkt, goods_mkt, labor_mkt

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

@sj.simple
def make_flow_durables(D):
    S = np.sum(D, axis=(2, 3)) 
    X_plus = np.sum(S * (1 - np.eye(S.shape[0])), axis=1)
    X_minus = np.sum(S * (1 - np.eye(S.shape[0])), axis=0)
    return X_plus, X_minus