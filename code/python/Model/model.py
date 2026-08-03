"""Model assembly, steady state, shocks, and experiment runner.

BOOKING TOGGLE (`booking`)
--------------------------
The green adoption margin's balance-of-payments treatment is a modelling
choice with a first-order effect on the adoption channel's OUTPUT sign, so
both are kept selectable and reportable:
    'import'    (baseline) the switching cost psi_g*D_SWITCH is booked as an
                import; green energy is imported at the world brown price and
                the saving (pE_B_P-pE_G_P)*CE_DUR_G is booked as an import
                reduction (energy_gap). All energy clears against world supply.

    'domestic'  green is a domestic industry: green energy (near-zero MC) and
                green-durable installation are produced domestically at zero
                profit and rebated lump-sum (green_sector, unknown Tgreen).
                Only BROWN energy is imported and clears against world supply.
                Unifies the two prior tasks (installer + green energy sector).

Households pay the switching cost and the green energy price in BOTH bookings;
only the BoP counterpart differs. The domestic booking is an UPPER BOUND on the
domestic value added (MC=0); the robust finding is the sign reversal.
"""
import numpy as np
import sequence_jacobian as sj

import blocks as B
from household import hh_ha_durable
from calibration import make_calibration, set_energy_grids

T = 300

# --- transition-path unknowns/targets, by booking ---------------------------
_UNKNOWNS_TD_BASE = {'y', 'pi', 'r', 'tauY', 'PEstar', 'w'}
_TARGETS_TD_BASE = {'goods_clearing', 'pires', 'r_res', 'tauY_res', 'E_clearing', 'w_res'}

# --- steady-state unknowns/targets, by booking ------------------------------
_SS_UNKNOWNS_BASE = {'vphi': 1, 'beta_max': 0.984, 'y': 0.9868, 'psi_g': 0.253}
_SS_TARGETS_BASE = ['piwres', 'nfares', 'goods_clearing', 'D_GREEN_res']


def td_unknowns_targets(booking='import'):
    u, t = set(_UNKNOWNS_TD_BASE), set(_TARGETS_TD_BASE)
    if booking == 'domestic':
        u.add('Tgreen'); t.add('Tgreen_res')
    return u, t


def ss_unknowns_targets(booking='import'):
    u, t = dict(_SS_UNKNOWNS_BASE), list(_SS_TARGETS_BASE)
    if booking == 'domestic':
        u = dict(u); u['Tgreen'] = 0.002
        t = t + ['Tgreen_res']
    return u, t


# Kept for back-compatibility (import-booking defaults).
UNKNOWNS_TD, TARGETS_TD = td_unknowns_targets('import')
SS_UNKNOWNS, SS_TARGETS = ss_unknowns_targets('import')


def dissolve_list(booking='import'):
    ca = 'CA_dom' if booking == 'domestic' else 'CA'
    return ['unions', 'UIP', ca, 'piW_to_W', 'pitop']


# Under the no_adoption counterfactual (green_block huge) D_GREEN is pinned at
# exactly 0 for every psi_g: D_GREEN_res = 0 - D_GREEN_ss_target has zero
# derivative in psi_g, so solving for psi_g against it is infeasible (singular
# Jacobian), not just numerically fragile. psi_g must instead be carried over,
# fixed, from the matching 'adoption' steady state -- same primitives, only
# the discrete choice is blocked.
def ss_unknowns_targets_fixed_psi(booking='import'):
    u, t = ss_unknowns_targets(booking)
    u = {k: v for k, v in u.items() if k != 'psi_g'}
    t = [x for x in t if x != 'D_GREEN_res']
    return u, t


SS_UNKNOWNS_FIXED_PSI, SS_TARGETS_FIXED_PSI = ss_unknowns_targets_fixed_psi('import')


def build_model(numeraire='core', booking='import'):
    """Assemble the full DAG.

    `numeraire` selects the household's unit of account:
        'core'  domestic good, p_num = pH_P   (DEFAULT since Option C)
        'cpi'   ARS convention, p_num = 1 (a calibration constant)

    `booking` selects the green-margin balance-of-payments treatment (see the
    module docstring). The calibration must be built with the SAME numeraire
    AND booking (make_calibration(numeraire, booking=...)): 'cpi' needs p_num
    as a constant, and the import booking needs Tgreen fixed at 0 while the
    domestic booking carries Tgreen as an unknown.
    """
    num = B.numeraire_core if numeraire == 'core' else B.numeraire_cpi
    if booking == 'domestic':
        margin = [B.green_sector]
        ca, imp, eqm = B.CA_dom, B.importProfits_dom, B.eqm_cond_dom
    else:
        margin = [B.switching_imports, B.energy_gap]
        ca, imp, eqm = B.CA, B.importProfits, B.eqm_cond
    return sj.combine([
        hh_ha_durable(),
        num, B.assets_convert,
        B.hh_outputs_dur, B.green_energy_price, *margin,
        B.income, B.profitcenters, B.importPrices, imp,
        B.revaluation, B.revaluation_dom, B.foreign_c, B.UIP, B.IEA, ca,
        B.unions, B.piW_to_W, B.CESprices, B.price_levels, B.pitop,
        B.mon_policy, B.fiscal, B.annualize, eqm,
    ])


def solve_ss(model, calib, verbose=False, unknowns=None, targets=None,
             booking='import'):
    u, t = ss_unknowns_targets(booking)
    ss = model.solve_steady_state(
        calib, unknowns=unknowns or u, targets=targets or t,
        solver='broyden_custom', dissolve=dissolve_list(booking), ttol=1e-14)
    B.test_targets(ss, noisy=verbose)
    return ss


def _energy_demand(ss, booking):
    """SS energy demand that clears against the fixed world supply: total energy
    under 'import' booking, BROWN energy only under 'domestic' (green is a
    domestic industry, not drawn from the world market)."""
    return (float(ss['CE_DUR_B'] + ss['prodE']) if booking == 'domestic'
            else float(ss['cE'] + ss['prodE']))


def _calibrate_supply(model, calib, unknowns, targets, booking, iters=6, tol=1e-10):
    """Fixed-quantity closure: E_supply_shock (the fixed world supply) must equal
    the model's own SS energy demand. The home country's energy rents depend on
    E_supply_shock, so the demand shifts with the guess -- a single shot leaves an
    O(1e-4) E_clearing residual under the domestic booking (where demand is brown
    only ~0.037, off the aE=0.04 endowment normalisation). Iterate to a fixed
    point (import converges in one step; domestic in two or three). Solved under
    the price-taking closure, where E_supply = E_supply_shock at the SS, so the
    fixed point transfers exactly to the inelastic SS."""
    cal = dict(calib); cal['E_supply_elasticity'] = np.inf
    demand = _energy_demand(solve_ss(model, cal, unknowns=unknowns, targets=targets,
                                     booking=booking), booking)
    for _ in range(iters):
        cal['E_supply_shock'] = demand
        new = _energy_demand(solve_ss(model, cal, unknowns=unknowns, targets=targets,
                                      booking=booking), booking)
        if abs(new - demand) < tol:
            demand = new; break
        demand = new
    return demand


# =============================================================================
# SHOCKS
# =============================================================================
def shock_price(size=1.0, half_life=16, T=T):
    """Brown energy PRICE shock (ARS). Requires E_supply_elasticity = inf.
    `size` is the impact log-deviation of the world energy price."""
    rho = 2 ** (-1 / half_life)
    return {'PEstar_shock': size * rho ** np.arange(T)}


def shock_supply(ss, drop=0.1, quarters=6, T=T):
    """Brown energy SUPPLY shock (Bayer et al.). Requires E_supply_elasticity
    finite. Energy availability falls by `drop` for `quarters` periods and
    then returns; the world price clears the market endogenously."""
    path = np.zeros(T)
    path[:quarters] = -drop * float(ss['E_supply_shock'])
    return {'E_supply_shock': path}


# =============================================================================
# EXPERIMENTS
# =============================================================================
#   name -> calibration overrides applied on top of the scenario baseline
POLICIES = {
    'none':     dict(tauE=0.0, insE=0.0),
    'subsidy':  dict(tauE=1.0, insE=0.0),   # full price cap  (Bayer P1, Langot)
    'transfer': dict(tauE=0.0, insE=1.0),   # Slutsky compens. (Bayer P2, Germany)
}

MODELS = {
    'adoption':    dict(green_block=0.0),   # green adoption margin OPEN
    'no_adoption': dict(green_block=1e10),  # margin SHUT: counterfactual
}

MONETARY = {
    'real_rate': dict(phi_pi=0.0, phi_pie=1.0),   # ARS baseline: constant real rate
    'taylor':    dict(phi_pi=1.5, phi_pie=0.0),   # standard Taylor rule
}

ENERGY_CLOSURE = {
    'elastic':   dict(E_supply_elasticity=np.inf),  # price-taking SOE (ARS/Langot)
    'inelastic': dict(E_supply_elasticity=1.0),     # fixed quantity  (Bayer)
}


def run(model, shock_kind='price', policy='none', model_variant='adoption',
        monetary='real_rate', shock_kwargs=None, numeraire='core',
        booking='import', **extra):
    """Solve one experiment end to end. Returns (ss, irf).

    The energy closure is implied by shock_kind: a price shock needs the
    price-taking closure, a supply shock needs the fixed-quantity closure.

    `numeraire` and `booking` must match the ones `model` was built with.
    """
    closure = 'elastic' if shock_kind == 'price' else 'inelastic'
    ov = {}
    ov.update(ENERGY_CLOSURE[closure])
    ov.update(MONETARY[monetary])
    ov.update(MODELS[model_variant])
    ov.update(POLICIES[policy])
    ov.update(extra)

    calib = make_calibration(numeraire, booking=booking, **ov)
    unknowns_td, targets_td = td_unknowns_targets(booking)

    if model_variant == 'no_adoption':
        # psi_g cannot be solved under this variant (see ss_unknowns_targets_
        # fixed_psi above): carry it over, fixed, from the matching 'adoption'
        # steady state -- same primitives, only the discrete choice is blocked.
        adopt_ov = dict(ov); adopt_ov.update(MODELS['adoption'])
        adopt_calib = make_calibration(numeraire, booking=booking, **adopt_ov)
        if closure == 'inelastic':
            au, at = ss_unknowns_targets(booking)
            adopt_calib['E_supply_shock'] = _calibrate_supply(
                model, adopt_calib, au, at, booking)
        ss_adopt = solve_ss(model, adopt_calib, booking=booking)
        calib['psi_g'] = float(ss_adopt['psi_g'])
        unknowns, targets = ss_unknowns_targets_fixed_psi(booking)
    else:
        unknowns, targets = ss_unknowns_targets(booking)

    # Under the fixed-quantity closure, world energy supply must equal the
    # model's own steady-state energy demand (brown only under the domestic
    # booking). The rents depend on E_supply_shock, so this is a fixed point;
    # see _calibrate_supply.
    if closure == 'inelastic':
        calib['E_supply_shock'] = _calibrate_supply(
            model, calib, unknowns, targets, booking)

    ss = solve_ss(model, calib, unknowns=unknowns, targets=targets, booking=booking)

    # The Slutsky transfer is indexed to pre-crisis household energy use, so
    # the grids must be filled from the solved steady state and the model
    # re-solved on them.
    if POLICIES[policy].get('insE', 0.0) != 0.0:
        calib = set_energy_grids(calib, ss)
        ss = solve_ss(model, calib, unknowns=unknowns, targets=targets, booking=booking)

    shk = (shock_price(**(shock_kwargs or {})) if shock_kind == 'price'
           else shock_supply(ss, **(shock_kwargs or {})))
    irf = model.solve_impulse_linear(ss, unknowns_td, targets_td, shk)
    B.test_targets(irf)
    return ss, irf
