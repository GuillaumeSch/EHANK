"""
Aggregate + government blocks for the fiscal-policy model (Auclert et al. 2023, Sec. 5),
single-beta, WITH a price-facing energy subsidy.

Three programs: energy subsidy (tauE, via the household price wedge p_c), targeted transfers
(insE), untargeted transfers (unt_scale). All deficit-financed; proportional labor tax tauL
responds to lagged debt. Steady-state debt B_ss = 0; all fiscal flows vanish at the ss, and
p_c == 1 when tauE = 0, so the household ss is unchanged.
"""
import sequence_jacobian as sj
from sequence_jacobian import simple, solved


# ---- real block ----
@simple
def prices(pEstar, alpha_E, alpha_F, eta, eta_E, mu):
    pHF = ((1 - alpha_E * pEstar ** (1 - eta_E)) / (1 - alpha_E)) ** (1 / (1 - eta_E))
    pH = ((pHF ** (1 - eta) - alpha_F) / (1 - alpha_F)) ** (1 / (1 - eta))
    w = pH / mu
    return pHF, pH, w


# ---- subsidy price wedge: pE_hh and household cost-of-living index p_c = P_hh/P ----
@simple
def subsidy_price(pEstar, tauE, pHF, alpha_E, eta_E):
    pE_hh = (1 - tauE) * pEstar + tauE                        # subsidized real energy price
    p_c = (alpha_E * pE_hh ** (1 - eta_E)
           + (1 - alpha_E) * pHF ** (1 - eta_E)) ** (1 / (1 - eta_E))
    return pE_hh, p_c


# ---- no-subsidy variant: household faces the market energy price; p_c is a constant (=1),
#      passed via calibration so SSJ attaches no (degenerate, empty) Jacobian to it ----
@simple
def market_energy_price(pEstar):
    pE_hh = pEstar
    return pE_hh


# ---- targeted/untargeted transfer instruments ----
@simple
def fiscal_instruments(pEstar, insE, unt_scale, CE_ss):
    price_gap = pEstar - 1.0
    tE = insE * price_gap                                     # targeted-transfer scale
    T_unt = unt_scale * price_gap * CE_ss                     # untargeted transfer
    return price_gap, tE, T_unt


@simple
def wagebill(pH, Y, mu):
    N = Y
    wN = (1 / mu) * pH * Y
    D = (1 - 1 / mu) * pH * Y
    return N, wN, D


@simple
def after_tax_income(tauL, wN):
    Z = (1 - tauL) * wN
    return Z


@simple
def monetary(rss, e):
    rante = rss + e
    return rante


@solved(unknowns={'j': (0.1, 50.0)}, targets=['jval'])
def asset_block(j, D, rante):
    jval = j - (D(+1) + j(+1)) / (1 + rante)
    r = (D + j) / j(-1) - 1
    return jval, r


# ---- household energy demand at the subsidized price (for budget + reporting) ----
@simple
def energy_agg(pE_hh, p_c, C, alpha_E, eta_E):
    CE = alpha_E * (pE_hh / p_c) ** (-eta_E) * C
    return CE


# ---- goods-market clearing: home demand carries the p_c wedge ----
@simple
def goods_market(pH, pHF, p_c, C, Y, Cstar, alpha, alpha_star, eta, eta_E, gamma):
    CH = (1 - alpha) * (pH / pHF) ** (-eta) * (pHF / p_c) ** (-eta_E) * C
    CstarH = alpha_star * pH ** (-gamma) * Cstar
    goods_clearing = CH + CstarH - Y
    return CH, CstarH, goods_clearing


# ---- government budget / debt ----
# ---- tax rule: reads lagged debt (B is a model unknown, exogenous to the DAG) ----
@simple
def tax_rule(B, psiB):
    tauL = psiB * B(-1)
    return tauL


# ---- government budget constraint as a residual (target); B promoted to unknown ----
@simple
def government(B, rante, tauE, price_gap, insE, T_unt, wN, CE_ss, tauL):
    # first-order exact: (CE - CE_ss)*price_gap is second order, so use CE_ss
    budget_res = (B - ((1 + rante(-1)) * B(-1)
                       + tauE * price_gap * CE_ss             # energy-subsidy cost
                       + insE * price_gap * CE_ss             # targeted-transfer cost
                       + T_unt - tauL * wN))
    return budget_res


# ---- union: after-tax wage NKPC ----
@solved(unknowns={'pi_w': (-0.5, 0.5)}, targets=['wage_res'])
def union(pi_w, N, C, w, tauL, kappa_w, beta_w, mu_w, phi, eis, zeta, vphi):
    gap = vphi * N ** phi * C ** (1 / eis) * mu_w / ((1 - tauL) * w ** (1 + zeta)) - 1
    wage_res = pi_w - (kappa_w * gap + beta_w * pi_w(+1))
    return wage_res


# ---- nominal side: wage, market CPI, household CPI ----
@solved(unknowns={'W': (0.01, 100.0)}, targets=['W_res'])
def nominal_wage(W, pi_w):
    W_res = W / W(-1) - 1 - pi_w
    return W_res


@simple
def cpi(W, w, p_c):
    P = W / w                                                 # market CPI (real wage identity)
    pi = P / P(-1) - 1
    P_hh = p_c * P                                            # household price index
    pi_hh = P_hh / P_hh(-1) - 1                               # household CPI inflation
    return P, pi, P_hh, pi_hh


@simple
def nfa_block(A, j, B):
    nfa = A - j - B
    return nfa
