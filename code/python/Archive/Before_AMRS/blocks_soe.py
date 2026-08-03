"""
Small-open-economy aggregate blocks for the durable E-HANK model (iteration 1).

Design (defaults agreed with GC):
  - Core good = home good, produced by labor (Y = Z*N), numeraire p_core = pH = 1,
    constant real wage w = 1 (as in the closed baseline).
  - Brown and green energy are IMPORTED at exogenous world prices p_e_b, p_e_g.
    The "brown energy supply shock" is an exogenous rise in p_e_b.
  - Minimal SOE closure: bonds-only assets, firm profits rebated lump-sum (Div = 0
    kept as in the closed baseline; equity valuation deferred to iteration A).
  - Home-good market clears: Y = C_CORE + G + X, with X = foreign export demand.
  - Real rate fixed at the world rate rss (constant-real-rate rule). NFA is free;
    the trade balance / BoP identity holds by Walras (reported, not targeted).

Key change vs the closed model: energy spending p_e*C_E is an import (a leakage
abroad), NOT domestic output. Higher energy prices therefore do not add to domestic
Y; the negative wealth effect contracts C_CORE and hence Y. This is the channel we
want to test.
"""
import numpy as np
import sequence_jacobian as sj


# ---- production: home good produced by labor ----
@sj.simple
def prod(Y, Z):
    N_D = Y / Z
    return N_D


@sj.simple
def labor_market(N, N_D):
    labor_mkt = N - N_D
    return labor_mkt


# ---- foreign demand for the home good (exports). p_core = pH numeraire = 1, so X
#      is constant in iteration 1; the CES form is kept so it responds once pH moves.
@sj.simple
def exports(p_core, alpha_star, gamma_x, Cstar):
    X = alpha_star * p_core ** (-gamma_x) * Cstar
    return X


# ---- home-good market clearing (this is the SOE resource constraint) ----
#      Domestic home-good demand: core consumption + government + exports +
#      durable-switching cost (D_GB * psi_g is a real home-good expenditure).
@sj.simple
def goods_market(C_CORE, G, X, D_GB, psi_g, Y):
    AD_durables = D_GB * psi_g
    goods_clearing = C_CORE + G + X + AD_durables - Y
    return goods_clearing, AD_durables


# ---- external accounts: imports = energy bill; trade balance & NFA (reporting) ----
@sj.simple
def external(X, p_core, p_e_b, p_e_g, C_E_B, C_E_G, A, B, r):
    imports = p_e_b * C_E_B + p_e_g * C_E_G
    exports_val = p_core * X
    TB = exports_val - imports
    nfa = A - B
    nfa_res = (A - B) - ((1 + r) * (A(-1) - B(-1)) + TB)   # BoP identity; Walras check
    return imports, exports_val, TB, nfa, nfa_res


# ---- government: constant debt B, lump-sum tax balances the budget ----
@sj.simple
def fiscal(B, r, G_ss, kappa_g, Tax, T_E):
    G = G_ss - kappa_g * (B - B.ss)
    GBC = Tax - ((1 + r) * B(-1) + G - B - T_E)
    return GBC, G


# ---- wage NKPC (transition) and its steady-state calibration of vphi ----
@sj.simple
def nkpc(piw, N, vphi, frisch, markup_ss, beta, theta_w, w, UCE):
    kappa_w = (1 - theta_w) * (1 - beta * theta_w) / (theta_w * (1 + vphi * (markup_ss / (markup_ss - 1))))
    wnkpc = kappa_w * (vphi * N ** (1 / frisch) - 1 / markup_ss * w * UCE) + beta * piw(1) - piw
    return wnkpc


@sj.simple
def nkpc_ss(N, frisch, markup_ss, w, UCE):
    vphi = 1 / markup_ss * w * UCE / N ** (1 / frisch)
    wnkpc = vphi * N ** (1 / frisch) - 1 / markup_ss * w * UCE
    return wnkpc, vphi


# ---- nominal side / monetary policy: constant real rate = world rate rss ----
@sj.simple
def core_inflation(piw):
    pi_core = piw
    return pi_core


@sj.simple
def monetary_real(pi_core, ishock, rss):
    i = rss + pi_core(1) + ishock
    r_ante = i - pi_core(1)
    return r_ante, i


@sj.simple
def ex_post_rate(r_ante):
    r = r_ante(-1)
    return r


real_rule = sj.combine([monetary_real, ex_post_rate], name="Real_rule")
