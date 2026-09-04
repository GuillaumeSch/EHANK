"""Aggregate blocks for E-HANK."""
import numpy as np
import sequence_jacobian as sj


@sj.simple
def hh_outputs(CE_B, CE_G, CHF, CHF_SWITCH, pH_PHF, pF_PHF,
                   pH_PHF_SWITCH, pF_PHF_SWITCH, eta, alpha_F, alpha_F_switch,
                   cbarE, scale_w, atw_n, markup_ss, pHF_SWITCH_P):
    """Household energy/non-energy aggregates and the adoption-bundle CES split.
    alpha_F_switch = 1 gives the pure-import booking (cH_switch = 0); lower
    values route adoption spending onto the home good (into goods clearing)."""
    cE = cbarE * (atw_n * markup_ss) * scale_w + cbarE * (1 - scale_w) + CE_B + CE_G
    cH = (1 - alpha_F) * pH_PHF ** (-eta) * CHF
    cF = alpha_F * pF_PHF ** (-eta) * CHF
    cH_switch = (1 - alpha_F_switch) * pH_PHF_SWITCH ** (-eta) * CHF_SWITCH
    cF_switch = alpha_F_switch * pF_PHF_SWITCH ** (-eta) * CHF_SWITCH
    CHF_SWITCH_exp = CHF_SWITCH * pHF_SWITCH_P
    return cH, cF, cE, cH_switch, cF_switch, CHF_SWITCH_exp


@sj.simple
def energy_gap(pE_B_pretax_P, pE_G_pretax_P, CE_G):
    """Balance-of-payments counterpart of adopters' cheaper energy (pretax)."""
    energy_gap_agg = (pE_B_pretax_P - pE_G_pretax_P) * CE_G
    return energy_gap_agg




# Exactly one of the two blocks below goes in the DAG (numeraire choice).


@sj.simple
def numeraire_cpi(r, atw_n):
    r_num = r
    atw_n_num = atw_n
    return r_num, atw_n_num


@sj.simple
def assets_convert(A, p_num):
    """A is in units of account; assets_clearing and CA are in CPI units."""
    A_cpi = A * p_num
    return A_cpi


@sj.solved(unknowns={'J': 15., 'j': 15.}, targets=['Jres', 'jres'], solver="broyden_custom")
def income(y, w, Z, pH_P, pE_P, J, j, rante, dividend_X, tauY, pcX_home,
           markup_ss, prodE_share, prodE_es, psi_g_bar, pHF_SWITCH_P):
    prodE = prodE_share * (pH_P / pE_P) ** prodE_es * y
    if prodE_share == 0:
        n = y / Z
    else:
        n = (1 - prodE_share) * (y / Z) * ((markup_ss * w) / (Z * pH_P)) ** (-prodE_es)
    btw_n = w * n
    atw_n = (1 - tauY) * btw_n
    atw = atw_n / n
    gdp = Z * n
    dividend = (markup_ss - 1) * w * n
    div_tot = dividend
    if pcX_home == 1:
        div_tot += dividend_X
    Jres = div_tot + J(1) / (1 + rante) - J
    jres = J(1) / (1 + rante) - j
    psi_g = psi_g_bar * pHF_SWITCH_P   # switching cost = real bundle priced at pHF_SWITCH_P
    return jres, Jres, atw_n, dividend, gdp, atw, n, btw_n, prodE, psi_g


@sj.simple
def profitcenters(Q, pH_P, cHstar, eps_dcp):
    dividend_X = (Q ** (1 - eps_dcp) * pH_P ** eps_dcp - pH_P) * cHstar
    return dividend_X


@sj.solved(unknowns={'piF': 0., 'PF': 1.}, targets=['piF_res', 'PFres'], solver="broyden_custom")
def foreignPrices(piF, PF, P, Q, PFstar, rante, theta_F):
    PFres = (1 + piF) * PF(-1) - PF
    pF_P = PF / P
    beta_F = 1 / (1 + rante.ss)
    kappa_F = (1 - theta_F) * (1 - beta_F * theta_F) / theta_F
    piF_term = Q * PFstar / pF_P - 1
    piF_res = kappa_F * piF_term + beta_F * piF(1) * (1 + piF(1)) - piF * (1 + piF)
    return piF_res, PFres, pF_P


@sj.solved(unknowns={'piE': 0., 'PE': 1.}, targets=['piE_res', 'PEres'], solver="broyden_custom")
def energyPrices(piE, PE, P, Q, PEstar, PEGstar, rante, theta_E, tauE, tau_b, tau_g):
    """Sticky retail brown energy price with cap and carbon tax; green energy at
    the exogenous world price Q*PEGstar, plus its own carbon tax tau_g. Both
    household prices are tax-inclusive (pE_B_P, pE_G_P); pretax prices feed the
    resource cost (BoP) and the ETS carbon account."""
    PEres = (1 + piE) * PE(-1) - PE
    pE_P = PE / P
    beta_E = 1 / (1 + rante.ss)
    kappa_E = (1 - theta_E) * (1 - beta_E * theta_E) / theta_E
    piE_term = Q * PEstar / pE_P - 1
    piE_res = kappa_E * piE_term + beta_E * piE(1) * (1 + piE(1)) - piE * (1 + piE)
    pE_P_ss = pE_P.ss
    pE_B_pretax_P = (1 - tauE) * pE_P + tauE * pE_P.ss
    pE_B_P = pE_B_pretax_P * (1 + tau_b)
    pE_B = pE_B_P * P
    pE_G_pretax_P = Q * PEGstar
    pE_G_P = pE_G_pretax_P * (1 + tau_g)
    return piE_res, PEres, pE_P, pE_P_ss, pE_B_P, pE_B, pE_B_pretax_P, pE_G_P, pE_G_pretax_P


importPrices = sj.combine([foreignPrices, energyPrices])


@sj.simple
def importProfits(pF_P, pE_P, Q, PFstar, PEstar, cF, cE, prodE):
    # diagnostic
    DF = (pF_P - Q * PFstar) * cF
    DE = (pE_P - Q * PEstar) * (cE + prodE)
    return DF, DE


@sj.simple
def revaluation(r, j, J, j_Esupply, J_Esupply, zetaEsupply):
    r_res = (J + zetaEsupply * J_Esupply) / (j(-1) + zetaEsupply * j_Esupply(-1)) - 1 - r
    return r_res


@sj.simple
def revaluation_dom(j, J, zetaEsupply, j_Esupply, J_Esupply):
    rdom = (J + zetaEsupply * J_Esupply) / (j(-1) + zetaEsupply * j_Esupply(-1)) - 1
    Adom = j + zetaEsupply * j_Esupply
    return rdom, Adom


@sj.simple
def foreign_c(pHstar, alphastar, gamma, Cstar, eps_dcp):
    cHstar = alphastar * pHstar ** (-gamma * eps_dcp) * Cstar
    return cHstar


@sj.solved(unknowns={'Q': (0.1, 2)}, targets=['uip'])
def UIP(Q, rante, rstar):
    uip = Q / Q(+1) * (1 + rante) - (1 + rstar)
    return uip


@sj.solved(unknowns={'J_Esupply': (0, 100)}, targets=['J_Esupply_res'])
def IEA(J_Esupply, PEstar, P, rstar, Gamma_arb, E_supply_shock, rante, Q, zetaEsupply):
    """World energy supply."""
    E_stock = ((PEstar(1) / (1 + rstar)) - PEstar) / Gamma_arb
    E_supply = E_supply_shock + (E_stock(-1) - E_stock)
    D_Esupply = Q * (PEstar * E_supply)
    J_Esupply_res = D_Esupply + J_Esupply(1) / (1 + rante) - J_Esupply
    j_Esupply = J_Esupply(1) / (1 + rante)
    D_Esupply_H = zetaEsupply * D_Esupply
    return E_supply, D_Esupply, J_Esupply_res, j_Esupply, D_Esupply_H, E_stock


@sj.solved(unknowns={'nfa': (-2, 2)}, targets=['nfares'], solver="brentq")
def CA(nfa, Q, pHstar, cHstar, pF_P, cF, cF_switch, pE_P, cE, prodE, rante, r, A_cpi,
       rdom, Adom, zetaEsupply, PEstar, E_supply, y, energy_gap_agg, PEGstar, CE_B, CE_G):
    """Balance of payments (import booking)."""
    exports = Q * (pHstar * cHstar + PEstar * zetaEsupply * E_supply)
    # imports = pF_P * cF + pF_P * cF_switch + pE_P * (cE + prodE) - energy_gap_agg
    imports = pF_P * cF + pF_P * cF_switch + pE_P*(CE_B + prodE) +Q* PEGstar*CE_G
    imports_pc = imports / imports.ss
    exports_pc = exports / exports.ss
    netexports = exports - imports
    revaluation_term = (r - rante(-1)) * A_cpi(-1) - (rdom - rante(-1)) * Adom(-1)
    nfares = netexports + revaluation_term + (1 + rante(-1)) * nfa(-1) - nfa
    nx_gdp = netexports / y
    return nfares, netexports, revaluation_term, exports, imports, nx_gdp, imports_pc, exports_pc


@sj.solved(unknowns={'piw': (-2, 2)}, targets=['piwres'], solver="brentq")
def unions(n, C, atw, w, vphi, w_BG, theta_w, beta, markup_ss, frisch, eis, piw, union_wedge):
    kappa_w = (1 - theta_w) * (1 - beta * theta_w) / theta_w
    psi_nr_inv = kappa_w / (vphi * (n ** (1 + (1 / frisch))))
    BG_term = (w - w.ss) * w * C.ss ** (-1 / eis) * n.ss / (markup_ss * n * w.ss)
    piwterm = vphi * n ** (1 / frisch) - (atw * (C ** (-1 / eis))) / markup_ss - w_BG * BG_term + union_wedge
    piwres = piw * (1 + piw) - beta * piw(1) * (1 + piw(1)) - psi_nr_inv * n * piwterm
    return piwres, piwterm


@sj.solved(unknowns={'W': (0.5, 2)}, targets=['Wres'], solver="brentq")
def piW_to_W(piw, W):
    Wres = W(-1) * (1 + piw) - W
    return Wres


@sj.simple
def CESprices(Q, eta, alpha_F, gamma, eta_E, alpha_E, pE_B_P, pE_G_P, pF_P, markup_ss, Z, w,
              alpha_F_switch):
    alpha = alpha_E + (1 - alpha_E) * alpha_F
    pH_P = markup_ss / Z * w
    if eta == 1:
        pHF_P = pH_P ** (1 - alpha_F) * pF_P ** alpha_F
    else:
        pHF_P = ((1 - alpha_F) * pH_P ** (1 - eta) + alpha_F * pF_P ** (1 - eta)) ** (1 / (1 - eta))
    pF_PHF = pF_P / pHF_P
    pH_PHF = pH_P / pHF_P
    pHstar = pH_P / Q
    if eta == 1:
        outer_nest = pH_PHF ** (1 - alpha_F) * pF_PHF ** alpha_F - 1
    else:
        outer_nest = (1 - alpha_F) * pH_PHF ** (1 - eta) + alpha_F * pF_PHF ** (1 - eta) - 1
    if eta_E == 1:
        pB_P = pHF_P ** (1 - alpha_E) * pE_B_P ** alpha_E
        pG_P = pHF_P ** (1 - alpha_E) * pE_G_P ** alpha_E
    else:
        pB_P = ((1 - alpha_E) * pHF_P ** (1 - eta_E) + alpha_E * pE_B_P ** (1 - eta_E)) ** (1 / (1 - eta_E))
        pG_P = ((1 - alpha_E) * pHF_P ** (1 - eta_E) + alpha_E * pE_G_P ** (1 - eta_E)) ** (1 / (1 - eta_E))
    # Adoption-bundle price index (home/foreign CES, import share alpha_F_switch).
    if eta == 1:
        pHF_SWITCH_P = pH_P ** (1 - alpha_F_switch) * pF_P ** alpha_F_switch
    else:
        pHF_SWITCH_P = ((1 - alpha_F_switch) * pH_P ** (1 - eta) + alpha_F_switch * pF_P ** (1 - eta)) ** (1 / (1 - eta))
    pF_PHF_SWITCH = pF_P / pHF_SWITCH_P
    pH_PHF_SWITCH = pH_P / pHF_SWITCH_P
    chi_tilde = (1 - alpha) * (alpha_F * eta + (1 - alpha_F) * eta_E) + alpha * gamma
    return chi_tilde, pF_PHF, pH_PHF, pHstar, pHF_P, alpha, outer_nest, pH_P, pB_P, pG_P, pHF_SWITCH_P, pF_PHF_SWITCH, pH_PHF_SWITCH


@sj.simple
def price_levels(piw, W, w, Z, pH_P, Q, P, PE, markup_ss, prodE_es, prodE_share):
    if prodE_es == 1:
        PH = (markup_ss / Z * W) ** (1 - prodE_share) * PE ** prodE_share
    else:
        PH = ((1 - prodE_share) * (markup_ss / Z * W) ** (1 - prodE_es) + prodE_share * PE ** (1 - prodE_es)) ** (1 / (1 - prodE_es))
    E = P * Q
    piH = PH / PH(-1) - 1
    w_res = W / P - w
    return PH, E, piH, w_res


@sj.solved(unknowns={'P': (0.5, 2)}, targets=['Pres'])
def pitop(P, pi):
    Pres = P - (1 + pi) * P(-1)
    return Pres


@sj.solved(unknowns={'inom': (-1, 1)}, targets=['inom_res'])
def mon_policy(ishock, rstar, pi, phi_pi, phi_pie, rho_i, inom, phi_piw, w, P, inom_t):
    W_here = w * P
    piw_here = (W_here / W_here(-1)) - 1
    inom_res = -(1 + inom) + rho_i * (1 + inom(-1)) + (1 - rho_i) * (1 + rstar.ss) * (1 + phi_pi * pi) * (1 + phi_piw * piw_here) * (1 + phi_pie * pi(+1)) + ishock
    rante = (1 + inom) / (1 + pi(+1)) - 1
    inom_t_res = inom - inom_t
    return rante, inom_res, inom_t_res


@sj.solved(unknowns={'B': (-1, 1)}, targets=['B_res'])
def fiscal(B, rante, btw_n, psiB, tauY, epsT, insE, pE_P, cE, CE_B, tauE, bb,
           s_g, psi_g, D_SWITCH, tau_b, tau_g, pE_G_P,
           CE_G, Trebate, pE_B_pretax_P, pE_G_pretax_P):
    """Government budget: energy-crisis instruments plus the ETS carbon account."""
    Tuntargeted = epsT
    Ttargeted = insE * (pE_P - pE_P.ss) * CE_B.ss
    base_S = cE   # all energy is imported at the market price pE_P (see CA); the cap subsidy bridges household payment vs that market cost, so the base is total energy
    Subsidy = tauE * (pE_P - pE_P.ss) * base_S
    Subsidy_green = (s_g - s_g.ss) * psi_g * D_SWITCH
    R_carbon = tau_b * pE_B_pretax_P * CE_B + tau_g * pE_G_pretax_P * CE_G
    green_subsidy_p = s_g.ss * psi_g * D_SWITCH
    Trebate_res = Trebate - (R_carbon - green_subsidy_p)
    spending = Tuntargeted + Ttargeted + Subsidy + Subsidy_green + Trebate + green_subsidy_p
    taxation = tauY * btw_n + R_carbon
    B_res = (1 + rante(-1)) * B(-1) + spending - taxation - B
    tauY_res = (1 - bb) * (psiB * (B(-1) - B.ss) - tauY) + bb * (B - B.ss)
    return B_res, tauY_res, Tuntargeted, Ttargeted, spending, taxation, Subsidy, Subsidy_green, R_carbon, Trebate_res


@sj.simple
def annualize(pi, piw, inom, r, rante, piH):
    pi_ann = (1 + pi) ** 4 - 1
    piw_ann = (1 + piw) ** 4 - 1
    inom_ann = (1 + inom) ** 4 - 1
    r_ann = (1 + r) ** 4 - 1
    rante_ann = (1 + rante) ** 4 - 1
    piH_ann = (1 + piH) ** 4 - 1
    return pi_ann, piw_ann, inom_ann, r_ann, rante_ann, piH_ann


@sj.simple
def eqm_cond(y, cH, cHstar, cH_switch, A_cpi, gdp, nfa, j, B, cE, prodE, PEstar,
             PEstar_shock, E_supply_elasticity, E_supply, zetaEsupply, j_Esupply,
             D_GREEN, D_GREEN_ss_target, CE_B):
    """Market clearing: nfa = A - j - B - zetaEsupply*j_Esupply."""
    goods_clearing = cH + cHstar + cH_switch - y
    assets_clearing = A_cpi - nfa - j - B - zetaEsupply * j_Esupply
    if E_supply_elasticity == np.inf:
        E_clearing = PEstar - PEstar_shock
    else:
        # E_clearing = (cE + prodE) - E_supply
        E_clearing = CE_B  - E_supply 
    PEstar_diff = PEstar - PEstar_shock
    gdp_t = gdp - 1
    D_GREEN_res = D_GREEN - D_GREEN_ss_target
    return goods_clearing, assets_clearing, E_clearing, PEstar_diff, gdp_t, D_GREEN_res


TARGETS = ['uip', 'piwres', 'nfares', 'goods_clearing', 'assets_clearing',
           'Pres', 'w_res', 'E_clearing', 'outer_nest']

# assert TD residuals on the active window only (terminal truncation tail ~1e-4)
TD_WINDOW = 100


def test_targets(d, extra=(), tol=1e-4, window=TD_WINDOW, noisy=False):
    """Assert every equilibrium residual is zero."""
    bad = []
    for t in list(TARGETS) + list(extra):
        arr = np.abs(np.asarray(d[t], dtype=float))
        full = float(arr.max()) if arr.size else 0.0
        if window is not None and arr.ndim >= 1 and arr.shape[0] > window:
            v = float(arr[:window].max())
        else:
            v = full
        if noisy:
            tail = f"  (full-horizon {full:.2e})" if v != full else ""
            print(f"  {t:18s} {v:.2e}{tail}")
        if not np.isclose(v, 0, atol=tol):
            bad.append((t, v))
    if bad:
        raise AssertionError("residuals not zero: "
                             + ", ".join(f"{t}={v:.2e}" for t, v in bad))
    return True


@sj.simple
def reweight_cpi(P_times_C, C):
    pires = P_times_C - C
    return pires
