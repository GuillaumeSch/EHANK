"""Aggregate blocks for E-HANK.

Self-contained: nothing is imported from the original ARS notebook. Blocks
whose equations are unchanged from Auclert, Monnery, Rognlie and Straub (2024)
are marked [ARS]; blocks written for this paper are marked [NEW].

All `*_P` variables produced here are prices relative to the CPI: this is the
base `household.py`'s CES demand system (`durable_shares`) uses throughout,
regardless of the unit of account. The unit of account (numeraire) is a
separate choice, made in the `numeraire_core` / `numeraire_cpi` blocks below:
the domestic good is the default (`numeraire='core'` in model.py), not the
CPI basket. It only enters the household's budget constraint and Euler
equation (`p_rel_num`, `r_num`, `atw_n_num`), never the CES demand ratios
above -- keeping the demand system CPI-relative and the budget numeraire-
relative is a numerical-robustness requirement of the current SSJ version,
not just a convention; see `household.py`'s module docstring. See
ehank_results.tex, Section 2.1, for why the domestic good was adopted for the
numeraire.
"""
import numpy as np
import sequence_jacobian as sj


# =============================================================================
# 1. HOUSEHOLD-SIDE AGGREGATION                                          [NEW]
# =============================================================================
@sj.simple
def hh_outputs_dur(CE_DUR_B, CE_DUR_G, CHF_DUR, pH_PHF, pF_PHF, eta, alpha_F,
                   cbarE, scale_w, atw_n, markup_ss):
    """Replaces ARS's `hh_outputs`.

    ARS compute cE = alpha_E * pE_B_P^(-eta_E) * C, the CES energy demand
    evaluated at the AGGREGATE price index. With durable-type-specific energy
    prices that is wrong by Jensen's inequality. Here cE and the non-energy
    composite come straight from the household block's exact hetoutputs; only
    the inner home/foreign nest, which is common to all households, is applied
    on top.
    """
    cE = cbarE * (atw_n * markup_ss) * scale_w + cbarE * (1 - scale_w) + CE_DUR_B + CE_DUR_G
    cH = (1 - alpha_F) * pH_PHF ** (-eta) * CHF_DUR
    cF = alpha_F * pF_PHF ** (-eta) * CHF_DUR
    return cH, cF, cE


@sj.simple
def green_energy_price(pE_B_P, pE_g_ratio, pass_g):
    """GREEN energy price P_G^E, with explicit pass-through.             [NEW]

    pass_g = 0 : green price fixed in real terms (adopters fully insulated)
    pass_g = 1 : green price moves one-for-one with the brown/aggregate index

    Note this reads P_B^E, the BROWN price households actually pay, which is
    net of any energy subsidy tauE. A price cap therefore compresses the
    brown-green gap and blunts the adoption incentive -- the policy
    interaction this paper is about.
    """
    pE_G_P = pE_g_ratio * pE_B_P.ss * (pE_B_P / pE_B_P.ss) ** pass_g
    return pE_G_P


@sj.simple
def switching_imports(psi_g, D_SWITCH):
    """Resource cost of brown->green switching, booked as an import.     [NEW]

    Kept in its own block: eqm_cond needs nfa (a CA output) while CA needs
    imports_dur, so co-locating them creates a genuine DAG cycle.
    """
    imports_dur = psi_g * D_SWITCH
    return imports_dur


@sj.simple
def energy_gap(pE_B_P, pE_G_P, CE_DUR_G):
    """Balance-of-payments counterpart of adopters' cheaper energy.      [NEW]

    MODELING CHOICE: this books the green energy saving as a real reduction in
    the national import bill, not merely a domestic transfer. Without it the
    windfall accruing to adopters has no external counterpart and asset market
    clearing fails. Documented in docs/model.tex.
    """
    energy_gap_agg = (pE_B_P - pE_G_P) * CE_DUR_G
    return energy_gap_agg


# =============================================================================
# 1bis. UNIT OF ACCOUNT                                                  [NEW]
# =============================================================================
# The household block is numeraire-generic: it reads p_num, r_num, atw_n_num
# and returns A denominated in units of account. Exactly one of the two blocks
# below must be in the DAG.
#
#   numeraire_cpi   ARS convention. p_num is NOT produced here -- it is passed
#                   as the calibration constant p_num = 1. Producing it as a
#                   block output would give it an identically-zero Jacobian
#                   row and trigger SimpleSparse's empty-operator crash.
#   numeraire_core  domestic good is the unit of account, p_num = pH_P.
#                   1+r_num = (1+r)*pH_P(-1)/pH_P: assets carried from t-1 were
#                   priced at pH_P(-1) and are redeemed at pH_P.


@sj.simple
def numeraire_cpi(r, atw_n):
    r_num = r
    atw_n_num = atw_n
    return r_num, atw_n_num


@sj.simple
def numeraire_core(r, atw_n, pH_P):
    p_num = pH_P
    r_num = (1 + r) * pH_P(-1) / pH_P - 1
    atw_n_num = atw_n / pH_P
    return p_num, r_num, atw_n_num


@sj.simple
def assets_convert(A, p_num):
    """A is in units of account; assets_clearing and CA are in CPI units."""
    A_cpi = A * p_num
    return A_cpi


# =============================================================================
# 2. FIRMS, PRICES, INCOME                                               [ARS]
# =============================================================================
@sj.solved(unknowns={'J': 15., 'j': 15.}, targets=['Jres', 'jres'], solver="broyden_custom")
def income(y, w, Z, pH_P, pE_P, J, j, rante, dividend_X, tauY, pcX_home,
           markup_ss, prodE_share, prodE_es):
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
    return jres, Jres, atw_n, dividend, gdp, atw, n, btw_n, prodE


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
def energyPrices(piE, PE, P, Q, PEstar, rante, theta_E, tauE):
    """Sticky retail energy price, plus the SUBSIDY / PRICE CAP.

    Produces the BROWN household energy price P_B^E:

        pE_B_P = (1-tauE)*pE_P + tauE*pE_P.ss

    pE_P is the wholesale/retail market price; pE_B_P is what a brown
    household actually pays. The GREEN price P_G^E comes from
    `green_energy_price`.

    tauE = 0 : households pay the market price.
    tauE = 1 : the household price is pinned at its pre-crisis level -- the
               full price cap of Bayer et al. and the French tariff shield of
               Langot et al.
    """
    PEres = (1 + piE) * PE(-1) - PE
    pE_P = PE / P
    beta_E = 1 / (1 + rante.ss)
    kappa_E = (1 - theta_E) * (1 - beta_E * theta_E) / theta_E
    piE_term = Q * PEstar / pE_P - 1
    piE_res = kappa_E * piE_term + beta_E * piE(1) * (1 + piE(1)) - piE * (1 + piE)
    pE_P_ss = pE_P.ss
    pE_B_P = (1 - tauE) * pE_P + tauE * pE_P.ss
    pE_B = pE_B_P * P
    return piE_res, PEres, pE_P, pE_P_ss, pE_B_P, pE_B


importPrices = sj.combine([foreignPrices, energyPrices])


@sj.solved(unknowns={'JF': 0., 'JE': 0}, targets=['JF_res', 'JE_res'], solver="broyden_custom")
def importProfits(JF, JE, pF_P, pE_P, Q, PFstar, PEstar, cF, cE, prodE, rante):
    DF = (pF_P - Q * PFstar) * cF
    JF_res = DF + JF(1) / (1 + rante) - JF
    jF = JF(1) / (1 + rante)
    DE = (pE_P - Q * PEstar) * (cE + prodE)
    JE_res = DE + JE(1) / (1 + rante) - JE
    jE = JE(1) / (1 + rante)
    return JF_res, jF, JE_res, jE, DF, DE


@sj.simple
def revaluation(r, j, J, jF, JF, zetaF, jE, JE, zetaE, j_Esupply, J_Esupply, zetaEsupply):
    r_res = (J + zetaF * JF + zetaE * JE + zetaEsupply * J_Esupply) / (j(-1) + zetaF * jF(-1) + zetaE * jE(-1) + zetaEsupply * j_Esupply(-1)) - 1 - r
    return r_res


@sj.simple
def revaluation_dom(j, J, jF, JF, jE, JE, zetaEsupply, j_Esupply, J_Esupply):
    rdom = (J + JF + JE + zetaEsupply * J_Esupply) / (j(-1) + jF(-1) + jE(-1) + zetaEsupply * j_Esupply(-1)) - 1
    Adom = j + jF + jE + zetaEsupply * j_Esupply
    return rdom, Adom


# =============================================================================
# 3. OPEN ECONOMY                                                        [ARS]
# =============================================================================
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
    """World energy supply. `E_supply_shock` is the QUANTITY of energy
    available -- the supply-shock instrument when E_supply_elasticity is
    finite."""
    E_stock = ((PEstar(1) / (1 + rstar)) - PEstar) / Gamma_arb
    E_supply = E_supply_shock + (E_stock(-1) - E_stock)
    D_Esupply = Q * (PEstar * E_supply)
    J_Esupply_res = D_Esupply + J_Esupply(1) / (1 + rante) - J_Esupply
    j_Esupply = J_Esupply(1) / (1 + rante)
    D_Esupply_H = zetaEsupply * D_Esupply
    return E_supply, D_Esupply, J_Esupply_res, j_Esupply, D_Esupply_H, E_stock


@sj.solved(unknowns={'nfa': (-2, 2)}, targets=['nfares'], solver="brentq")
def CA(nfa, Q, pHstar, cHstar, PFstar, cF, PEstar, cE, prodE, rante, r, A_cpi,
       rdom, Adom, zetaEsupply, E_supply, y, imports_dur, energy_gap_agg):
    """ARS balance of payments plus the two durable-margin terms."""
    exports = Q * (pHstar * cHstar + PEstar * zetaEsupply * E_supply)
    imports = Q * (PFstar * cF + PEstar * (cE + prodE)) + imports_dur - energy_gap_agg
    imports_pc = imports / imports.ss
    exports_pc = exports / exports.ss
    netexports = Q * (pHstar * cHstar - PFstar * cF - PEstar * (cE + prodE - zetaEsupply * E_supply)) - imports_dur + energy_gap_agg
    revaluation_term = (r - rante(-1)) * A_cpi(-1) - (rdom - rante(-1)) * Adom(-1)
    nfares = netexports + revaluation_term + (1 + rante(-1)) * nfa(-1) - nfa
    nx_gdp = netexports / y
    return nfares, netexports, revaluation_term, exports, imports, nx_gdp, imports_pc, exports_pc


# =============================================================================
# 4. NOMINAL BLOCK, POLICY RULES                                         [ARS]
# =============================================================================
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


@sj.solved(unknowns={'pHF_P': 1., 'pH_PHF': 1.}, targets=['inner_nest', 'outer_nest'])
def CESprices(Q, eta, alpha_F, gamma, PFstar, PEstar, pHF_P, pH_PHF, eta_E, alpha_E, pE_B_P, pF_P):
    """CPI aggregation. DELIBERATE CHOICE (Option C, see docs/model.tex): the
    energy leg of the CPI is anchored on the BROWN price P_B^E alone, not on
    the share-weighted index [(1-D_G)P_B^(1-eta_E) + D_G P_G^(1-eta_E)].

    Rationale. (i) It is what statistical agencies do -- the energy basket is
    not reweighted every quarter for technology adoption. (ii) It keeps
    p_rel(brown) == 1 exactly, on which the phase-2 nesting test rests.
    (iii) Feeding D_GREEN back into CESprices would close a DAG cycle
    (household -> D_GREEN -> CPI -> pE_B_P -> household) requiring the CPI to
    be promoted to a model unknown.

    The resulting measurement gap is quantified ex post, outside the DAG, by
    ehank/deflator.py. Under the domestic-good numeraire the structural anchor
    is p_core == 1, so this choice only affects the MEASURED deflator and the
    monetary rule, not the resource constraint."""
    alpha = alpha_E + (1 - alpha_E) * alpha_F
    pF_PHF = pF_P / pHF_P
    pH_P = pH_PHF * pHF_P
    pHstar = pH_P / Q
    if eta_E == 1:
        inner_nest = pHF_P ** (1 - alpha_E) * pE_B_P ** alpha_E - 1
    else:
        inner_nest = (1 - alpha_E) * pHF_P ** (1 - eta_E) + alpha_E * pE_B_P ** (1 - eta_E) - 1
    if eta == 1:
        outer_nest = pH_PHF ** (1 - alpha_F) * pF_PHF ** alpha_F - 1
    else:
        outer_nest = (1 - alpha_F) * pH_PHF ** (1 - eta) + alpha_F * pF_PHF ** (1 - eta) - 1
    chi_tilde = (1 - alpha) * (alpha_F * eta + (1 - alpha_F) * eta_E) + alpha * gamma
    return chi_tilde, pF_PHF, inner_nest, outer_nest, pH_P, pHstar, alpha


@sj.simple
def price_levels(piw, W, w, Z, pH_P, Q, P, PE, markup_ss, prodE_es, prodE_share):
    if prodE_es == 1:
        PH = (markup_ss / Z * W) ** (1 - prodE_share) * PE ** prodE_share
    else:
        PH = ((1 - prodE_share) * (markup_ss / Z * W) ** (1 - prodE_es) + prodE_share * PE ** (1 - prodE_es)) ** (1 / (1 - prodE_es))
    pires = PH / pH_P - P
    E = P * Q
    piH = PH / PH(-1) - 1
    w_res = W / P - w
    return PH, pires, E, piH, w_res


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
def fiscal(B, rante, btw_n, psiB, tauY, epsT, insE, pE_P, cE, tauE, bb):
    """Government budget. Two energy-crisis instruments:

      Subsidy   = tauE * (pE_P - pE_P.ss) * cE      price cap, cost scales
                                                    with ACTUAL consumption
      Ttargeted = insE * (pE_P - pE_P.ss) * cE.ss   Slutsky transfer, cost
                                                    scales with PRE-CRISIS
                                                    consumption
    Both are debt-financed and repaid through the distortionary labour tax
    tauY via the psiB feedback rule.
    """
    Tuntargeted = epsT
    Ttargeted = insE * (pE_P - pE_P.ss) * cE.ss
    Subsidy = tauE * (pE_P - pE_P.ss) * cE
    spending = Tuntargeted + Ttargeted + Subsidy
    taxation = tauY * btw_n
    B_res = (1 + rante(-1)) * B(-1) + spending - taxation - B
    tauY_res = (1 - bb) * (psiB * (B(-1) - B.ss) - tauY) + bb * (B - B.ss)
    return B_res, tauY_res, Tuntargeted, Ttargeted, spending, taxation, Subsidy


@sj.simple
def annualize(pi, piw, inom, r, rante, piH):
    pi_ann = (1 + pi) ** 4 - 1
    piw_ann = (1 + piw) ** 4 - 1
    inom_ann = (1 + inom) ** 4 - 1
    r_ann = (1 + r) ** 4 - 1
    rante_ann = (1 + rante) ** 4 - 1
    piH_ann = (1 + piH) ** 4 - 1
    return pi_ann, piw_ann, inom_ann, r_ann, rante_ann, piH_ann


# =============================================================================
# 5. MARKET CLEARING                                                     [ARS]
# =============================================================================
@sj.simple
def eqm_cond(y, cH, cHstar, A_cpi, gdp, nfa, j, jF, jE, B, cE, prodE, PEstar,
             PEstar_shock, E_supply_elasticity, E_supply, zetaEsupply, j_Esupply, D_GREEN, D_GREEN_ss_target):
    """Energy market closure switches on E_supply_elasticity:

      inf    -> PEstar is exogenous (price-taking SOE). Shock: PEstar_shock.
      finite -> quantity fixed by IEA, PEstar clears the market.
                Shock: E_supply_shock.
    """
    goods_clearing = cH + cHstar - y
    assets_clearing = A_cpi - nfa - j - jF - jE - B - zetaEsupply * j_Esupply
    if E_supply_elasticity == np.inf:
        E_clearing = PEstar - PEstar_shock
    else:
        E_clearing = (cE + prodE) - E_supply
    PEstar_diff = PEstar - PEstar_shock
    gdp_t = gdp - 1
    D_GREEN_res = D_GREEN - D_GREEN_ss_target
    return goods_clearing, assets_clearing, E_clearing, PEstar_diff, gdp_t, D_GREEN_res


# =============================================================================
# 6. DIAGNOSTICS
# =============================================================================
TARGETS = ['uip', 'piwres', 'nfares', 'goods_clearing', 'assets_clearing',
           'Pres', 'w_res', 'E_clearing', 'inner_nest', 'outer_nest']


def test_targets(d, extra=(), tol=1e-4, noisy=False):
    """Assert every equilibrium residual is zero. solve_steady_state does NOT
    check its own residuals, so this must be called after every solve."""
    bad = []
    for t in list(TARGETS) + list(extra):
        v = float(np.max(np.abs(np.asarray(d[t]))))
        if noisy:
            print(f"  {t:18s} {v:.2e}")
        if not np.isclose(v, 0, atol=tol):
            bad.append((t, v))
    if bad:
        raise AssertionError("residuals not zero: "
                             + ", ".join(f"{t}={v:.2e}" for t, v in bad))
    return True
