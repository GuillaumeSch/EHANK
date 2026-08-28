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
def green_energy_price(pE_B_P, pE_P, pE_g_ratio, pass_g, tau_g):
    """GREEN energy price P_G^E, with explicit pass-through.             [NEW]

    pass_g = 0 : green price fixed in real terms (adopters fully insulated)
    pass_g = 1 : green price moves one-for-one with the brown price

    Anchored on the PRE-tax wholesale SS price pE_P.ss (NOT the carbon-taxed
    brown price pE_B_P.ss), with its own low carbon tax tau_g. This decouples
    the green price from the brown carbon tax tau_b: raising tau_b widens the
    brown-green gap and lifts adoption, instead of taxing both legs equally.
    The effective SS ratio is pE_G_P.ss/pE_B_P.ss = pE_g_ratio*(1+tau_g)/(1+tau_b);
    at tau_b=tau_g=0 it is pE_g_ratio and the block is bit-identical to before.

    The pass-through still references pE_B_P, the (capped) brown price, so a
    price cap compresses the brown-green gap and blunts adoption -- the policy
    interaction this paper is about.
    """
    pE_G_P = pE_g_ratio * pE_P.ss * (1 + tau_g) * (pE_B_P / pE_B_P.ss) ** pass_g
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
def energy_gap(pE_B_pretax_P, pE_G_P, tau_g, CE_DUR_G):
    """Balance-of-payments counterpart of adopters' cheaper energy.      [NEW]

    MODELING CHOICE (booking='import'): this books the green energy saving as a
    real reduction in the national import bill, not merely a domestic transfer.
    Without it the windfall accruing to adopters has no external counterpart and
    asset market clearing fails. Documented in docs/model.tex.

    Uses PRE-carbon-tax prices on both legs: the carbon tax is a domestic wedge
    (rebated in carbon_sector), so it must not enter the import bill. At
    tau_b=tau_g=0 these equal pE_B_P and pE_G_P, leaving baseline/cap untouched.
    """
    pE_G_pretax_P = pE_G_P / (1 + tau_g)
    energy_gap_agg = (pE_B_pretax_P - pE_G_pretax_P) * CE_DUR_G
    return energy_gap_agg


@sj.simple
def green_sector(pE_G_P, CE_DUR_G, psi_g, D_SWITCH, Tgreen):
    """Domestic green sector, booking='domestic' (unifies Tasks 1 and 2). [NEW]

    Green energy (near-zero marginal cost) AND green-durable installation are
    produced DOMESTICALLY at zero profit and rebated to households lump-sum as
    Tgreen. Tgreen is a model unknown; Tgreen_res closes it. The unknown/target
    pair breaks the income->hh->income cycle (Newton solves the fixed point),
    and the zero markup means no phantom dividend, so assets_clearing holds.

    One flow of domestic value added: green energy supply pE_G_P*CE_DUR_G plus
    green installation psi_g*D_SWITCH. Households still pay both privately (the
    green energy price through p_rel, the switching cost through Tswitch); only
    the balance-of-payments counterpart moves from imports to a domestic rebate.
    MC=0 makes the domestic value added an UPPER BOUND; the robust result is the
    sign reversal of the adoption channel relative to import booking.
    """
    Tgreen_res = Tgreen - (pE_G_P * CE_DUR_G + psi_g * D_SWITCH)
    return Tgreen_res




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
def energyPrices(piE, PE, P, Q, PEstar, rante, theta_E, tauE, tau_b):
    """Sticky retail energy price, the SUBSIDY / PRICE CAP, and the CARBON TAX.

    Produces the BROWN household energy price P_B^E:

        pE_B_P = [(1-tauE)*pE_P + tauE*pE_P.ss] * (1 + tau_b)

    pE_P is the wholesale/retail market price; pE_B_P is what a brown household
    actually pays, gross of the carbon tax tau_b. The GREEN price P_G^E comes
    from `green_energy_price` and is anchored on the PRE-tax base so tau_b does
    not leak into it. tau_b = 0 recovers the no-ETS price exactly.

    tauE = 0 : households pay the market price (+ carbon tax).
    tauE = 1 : the pre-carbon-tax household price is pinned at its pre-crisis
               level -- the full price cap of Bayer et al. / Langot et al.
    """
    PEres = (1 + piE) * PE(-1) - PE
    pE_P = PE / P
    beta_E = 1 / (1 + rante.ss)
    kappa_E = (1 - theta_E) * (1 - beta_E * theta_E) / theta_E
    piE_term = Q * PEstar / pE_P - 1
    piE_res = kappa_E * piE_term + beta_E * piE(1) * (1 + piE(1)) - piE * (1 + piE)
    pE_P_ss = pE_P.ss
    # Pre-carbon-tax consumer brown price: what the BoP / import blocks must use,
    # since the carbon tax tau_b is a DOMESTIC wedge (rebated via carbon_sector),
    # not a change in the world price at which energy is imported. Equals pE_B_P
    # exactly when tau_b = 0, so baseline and cap experiments are untouched.
    pE_B_pretax_P = (1 - tauE) * pE_P + tauE * pE_P.ss
    pE_B_P = pE_B_pretax_P * (1 + tau_b)
    pE_B = pE_B_P * P
    return piE_res, PEres, pE_P, pE_P_ss, pE_B_P, pE_B, pE_B_pretax_P


importPrices = sj.combine([foreignPrices, energyPrices])


@sj.simple
def importProfits(pF_P, pE_P, Q, PFstar, PEstar, cF, cE, prodE):
    # Retail importer margins now accrue ABROAD (foreign-owned); kept only as
    # diagnostics. No domestic capitalisation, so no JF/JE/jF/jE.
    DF = (pF_P - Q * PFstar) * cF
    DE = (pE_P - Q * PEstar) * (cE + prodE)
    return DF, DE


@sj.simple
def importProfits_dom(pF_P, pE_P, Q, PFstar, PEstar, cF, CE_DUR_B, prodE):
    # Domestic booking: only brown energy is imported, so the (foreign-owned)
    # energy margin is levied on CE_DUR_B alone. Diagnostics only.
    DF = (pF_P - Q * PFstar) * cF
    DE = (pE_P - Q * PEstar) * (CE_DUR_B + prodE)
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
def CA(nfa, Q, pHstar, cHstar, pF_P, cF, pE_P, cE, prodE, rante, r, A_cpi,
       rdom, Adom, zetaEsupply, PEstar, E_supply, y, imports_dur, energy_gap_agg):
    """ARS balance of payments: imports valued at the RETAIL price (importers
    foreign-owned), the home energy endowment exported at the world price, plus
    the two durable-margin terms."""
    exports = Q * (pHstar * cHstar + PEstar * zetaEsupply * E_supply)
    imports = pF_P * cF + pE_P * (cE + prodE) + imports_dur - energy_gap_agg
    imports_pc = imports / imports.ss
    exports_pc = exports / exports.ss
    netexports = exports - imports
    revaluation_term = (r - rante(-1)) * A_cpi(-1) - (rdom - rante(-1)) * Adom(-1)
    nfares = netexports + revaluation_term + (1 + rante(-1)) * nfa(-1) - nfa
    nx_gdp = netexports / y
    return nfares, netexports, revaluation_term, exports, imports, nx_gdp, imports_pc, exports_pc


@sj.solved(unknowns={'nfa': (-2, 2)}, targets=['nfares'], solver="brentq")
def CA_dom(nfa, Q, pHstar, cHstar, pF_P, cF, pE_P, CE_DUR_B, prodE, rante, r,
           A_cpi, rdom, Adom, zetaEsupply, PEstar, E_supply, y):
    """ARS balance of payments, booking='domestic': only BROWN energy is imported
    (retail-priced); green energy and the switching cost are domestic."""
    exports = Q * (pHstar * cHstar + PEstar * zetaEsupply * E_supply)
    imports = pF_P * cF + pE_P * (CE_DUR_B + prodE)
    imports_pc = imports / imports.ss
    exports_pc = exports / exports.ss
    netexports = exports - imports
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
def CESprices(Q, eta, alpha_F, gamma, PFstar, PEstar, pHF_P, pH_PHF, eta_E, alpha_E, pE_B_pretax_P, pF_P):
    """CPI aggregation. DELIBERATE CHOICE (Option C, see docs/model.tex): the
    energy leg of the CPI is anchored on the BROWN price P_B^E alone, not on
    the share-weighted index [(1-D_G)P_B^(1-eta_E) + D_G P_G^(1-eta_E)].

    The anchor is the PRE-carbon-tax brown price pE_B_pretax_P. At tau_b = 0
    this equals pE_B_P and the block is bit-identical to before. Under a carbon
    tax the CPI excludes the tax wedge, so brown households genuinely bear it
    (p_rel(brown) > 1 in energy_price_bundle) and the carbon revenue has real
    incidence -- anchoring on the TAX-INCLUSIVE price instead would insulate
    brown households at p_rel == 1 and leak the revenue out of asset clearing.

    Rationale for Option C. (i) It is what statistical agencies do -- the energy
    basket is not reweighted every quarter for technology adoption. (ii) It
    keeps p_rel(brown) == 1 exactly at tau_b = 0, on which the phase-2 nesting
    test rests. (iii) Feeding D_GREEN back into CESprices would close a DAG
    cycle (household -> D_GREEN -> CPI -> pE_B_P -> household) requiring the CPI
    to be promoted to a model unknown.

    The resulting measurement gap is quantified ex post, outside the DAG, by
    ehank/deflator.py. Under the domestic-good numeraire the structural anchor
    is p_core == 1, so this choice only affects the MEASURED deflator and the
    monetary rule, not the resource constraint."""
    alpha = alpha_E + (1 - alpha_E) * alpha_F
    pF_PHF = pF_P / pHF_P
    pH_P = pH_PHF * pHF_P
    pHstar = pH_P / Q
    if eta_E == 1:
        inner_nest = pHF_P ** (1 - alpha_E) * pE_B_pretax_P ** alpha_E - 1
    else:
        inner_nest = (1 - alpha_E) * pHF_P ** (1 - eta_E) + alpha_E * pE_B_pretax_P ** (1 - eta_E) - 1
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
def fiscal(B, rante, btw_n, psiB, tauY, epsT, insE, pE_P, cE, CE_DUR_B, tauE, bb,
           subsidy_brown_only, s_g, psi_g, D_SWITCH, tau_b, tau_g, pE_g_ratio,
           CE_DUR_G, Trebate, pE_B_pretax_P):
    """Government budget. Energy-crisis instruments + the ETS carbon account.

      Subsidy   = tauE * (pE_P - pE_P.ss) * base_S     price cap, cost scales
                                                       with ACTUAL consumption
      Ttargeted = insE * (pE_P - pE_P.ss) * CE_DUR_B.ss Slutsky transfer, cost
                                                       scales with PRE-CRISIS
                                                       BROWN consumption

    Only the BROWN price is capped (green households pay the fixed pE_G_P and
    are untouched), so the Slutsky transfer is ALWAYS brown-only: it is a
    lump sum whose incidence is set on the household side (durable-state-
    specific, zero in green states), so it balances under either booking.

    The subsidy works through PRICES and the import system, so its base is
    booking-dependent (subsidy_brown_only, set by make_calibration):
      import   base_S = cE (total). Green energy is imported at the market
               price the cap references, so the green part of the subsidy has
               an import-side counterpart and assets clear.
      domestic base_S = CE_DUR_B (brown only). Green energy is a domestic
               industry with no import content, so a total-energy subsidy
               would over-pay by the green mass and leak into a permanent nfa
               drift; only brown clears.
    Both crisis instruments are debt-financed and repaid through the
    distortionary labour tax tauY via the psiB feedback rule.

    ETS (tau_b, tau_g > 0). The carbon tax is the wedge between what households
    pay for energy (pE_B_P, pE_G_P) and its world resource cost (pE_P). That
    wedge has no other recipient in the flow of funds, so it MUST be booked as
    government revenue here; otherwise it is money the household spends that
    reaches neither firms nor foreigners, and asset-market clearing fails. The
    revenue is recycled in a balanced sub-account so B is untouched by the ETS:
      R_carbon        = tau_b*pE_B_pretax_P*CE_DUR_B
                        + tau_g*(pE_g_ratio*pE_P.ss)*CE_DUR_G
      green_subsidy_p = s_g.ss*psi_g*D_SWITCH   (PERMANENT switch subsidy,
                        recycle='green_subsidy'; 0 for 'rebate')
      Trebate         = R_carbon - green_subsidy_p    (lump-sum residual)
    The brown carbon base is the PRE-tax consumer price pE_B_pretax_P, not the
    world price: under a price cap that base is pinned, so the ETS and the cap
    compose correctly (the household is taxed on what it actually pays, capped).
    Trebate is a model unknown closed by Trebate_res; because spending gains
    (Trebate + green_subsidy_p) = R_carbon and taxation gains R_carbon, the ETS
    nets out of B_res exactly. Only the TRANSITORY switch subsidy (s_g - s_g.ss)
    is debt-financed; the permanent part is carbon-financed here. All carbon
    terms vanish at tau_b=tau_g=0, leaving the no-ETS block bit-identical.
    """
    Tuntargeted = epsT
    Ttargeted = insE * (pE_P - pE_P.ss) * CE_DUR_B.ss
    base_S = CE_DUR_B if subsidy_brown_only else cE
    Subsidy = tauE * (pE_P - pE_P.ss) * base_S
    Subsidy_green = (s_g - s_g.ss) * psi_g * D_SWITCH
    R_carbon = tau_b * pE_B_pretax_P * CE_DUR_B + tau_g * (pE_g_ratio * pE_P.ss) * CE_DUR_G
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


# =============================================================================
# 5. MARKET CLEARING                                                     [ARS]
# =============================================================================
@sj.simple
def eqm_cond(y, cH, cHstar, A_cpi, gdp, nfa, j, B, cE, prodE, PEstar,
             PEstar_shock, E_supply_elasticity, E_supply, zetaEsupply, j_Esupply,
             D_GREEN, D_GREEN_ss_target):
    """nfa = A - j - B - zetaEsupply*j_Esupply (importers no longer held
    domestically). Energy closure switches on E_supply_elasticity as before."""
    goods_clearing = cH + cHstar - y
    assets_clearing = A_cpi - nfa - j - B - zetaEsupply * j_Esupply
    if E_supply_elasticity == np.inf:
        E_clearing = PEstar - PEstar_shock
    else:
        E_clearing = (cE + prodE) - E_supply
    PEstar_diff = PEstar - PEstar_shock
    gdp_t = gdp - 1
    D_GREEN_res = D_GREEN - D_GREEN_ss_target
    return goods_clearing, assets_clearing, E_clearing, PEstar_diff, gdp_t, D_GREEN_res


@sj.simple
def eqm_cond_dom(y, cH, cHstar, A_cpi, gdp, nfa, j, B, CE_DUR_B, prodE, PEstar,
                 PEstar_shock, E_supply_elasticity, E_supply, zetaEsupply, j_Esupply,
                 D_GREEN, D_GREEN_ss_target):
    """Domestic booking: only brown (imported) energy clears against world supply."""
    goods_clearing = cH + cHstar - y
    assets_clearing = A_cpi - nfa - j - B - zetaEsupply * j_Esupply
    if E_supply_elasticity == np.inf:
        E_clearing = PEstar - PEstar_shock
    else:
        E_clearing = (CE_DUR_B + prodE) - E_supply
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
