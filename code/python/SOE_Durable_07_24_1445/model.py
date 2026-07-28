"""Model assembly, steady state, shocks, and experiment runner."""
import numpy as np
import sequence_jacobian as sj

import blocks as B
from household import hh_ha_durable
from calibration import make_calibration, set_energy_grids

T = 300

# Unknowns/targets of the transition path. PEstar is an unknown and E_clearing
# a target under BOTH energy closures; which one binds is set by
# E_supply_elasticity inside eqm_cond.
UNKNOWNS_TD = {'y', 'pi', 'r', 'tauY', 'PEstar', 'w'}
TARGETS_TD = {'goods_clearing', 'pires', 'r_res', 'tauY_res', 'E_clearing', 'w_res'}

SS_UNKNOWNS = {'vphi': 1, 'beta_max': 0.984, 'y': 0.9868, 'psi_g': 0.253}
SS_TARGETS = ['piwres', 'nfares', 'goods_clearing', 'D_GREEN_res']
DISSOLVE = ['unions', 'UIP', 'CA', 'piW_to_W', 'pitop']

# Under the no_adoption counterfactual (green_block huge) D_GREEN is pinned at
# exactly 0 for every psi_g: D_GREEN_res = 0 - D_GREEN_ss_target has zero
# derivative in psi_g, so solving for psi_g against it is infeasible (singular
# Jacobian), not just numerically fragile. psi_g must instead be carried over,
# fixed, from the matching 'adoption' steady state -- same primitives, only
# the discrete choice is blocked.
SS_UNKNOWNS_FIXED_PSI = {k: v for k, v in SS_UNKNOWNS.items() if k != 'psi_g'}
SS_TARGETS_FIXED_PSI = [t for t in SS_TARGETS if t != 'D_GREEN_res']


def build_model(numeraire='core'):
    """Assemble the full DAG.

    `numeraire` selects the household's unit of account:
        'core'  domestic good, p_num = pH_P   (DEFAULT since Option C)
        'cpi'   ARS convention, p_num = 1 (a calibration constant)

    One model object serves every experiment at a given numeraire; experiments
    differ only through the calibration. The calibration must be built with
    the SAME numeraire (make_calibration(numeraire=...)), because 'cpi' needs
    p_num supplied as a constant and 'core' must NOT have it in the dict.
    """
    num = B.numeraire_core if numeraire == 'core' else B.numeraire_cpi
    return sj.combine([
        hh_ha_durable(),
        num, B.assets_convert,
        B.hh_outputs_dur, B.green_energy_price, B.switching_imports, B.energy_gap,
        B.income, B.profitcenters, B.importPrices, B.importProfits,
        B.revaluation, B.revaluation_dom, B.foreign_c, B.UIP, B.IEA, B.CA,
        B.unions, B.piW_to_W, B.CESprices, B.price_levels, B.pitop,
        B.mon_policy, B.fiscal, B.annualize, B.eqm_cond,
    ])


def solve_ss(model, calib, verbose=False, unknowns=None, targets=None):
    ss = model.solve_steady_state(
        calib, unknowns=unknowns or SS_UNKNOWNS, targets=targets or SS_TARGETS,
        solver='broyden_custom', dissolve=DISSOLVE, ttol=1e-14)
    B.test_targets(ss, noisy=verbose)
    return ss


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
        monetary='real_rate', shock_kwargs=None, numeraire='core', **extra):
    """Solve one experiment end to end. Returns (ss, irf).

    The energy closure is implied by shock_kind: a price shock needs the
    price-taking closure, a supply shock needs the fixed-quantity closure.

    `numeraire` must match the one `model` was built with.
    """
    closure = 'elastic' if shock_kind == 'price' else 'inelastic'
    ov = {}
    ov.update(ENERGY_CLOSURE[closure])
    ov.update(MONETARY[monetary])
    ov.update(MODELS[model_variant])
    ov.update(POLICIES[policy])
    ov.update(extra)

    calib = make_calibration(numeraire, **ov)

    if model_variant == 'no_adoption':
        # psi_g cannot be solved under this variant (see SS_UNKNOWNS_FIXED_PSI
        # above): carry it over, fixed, from the matching 'adoption' steady
        # state -- same primitives, only the discrete choice is blocked.
        adopt_ov = dict(ov); adopt_ov.update(MODELS['adoption'])
        adopt_calib = make_calibration(numeraire, **adopt_ov)
        if closure == 'inelastic':
            c0 = dict(adopt_calib); c0['E_supply_elasticity'] = np.inf
            ss0 = solve_ss(model, c0)
            adopt_calib['E_supply_shock'] = float(ss0['cE'] + ss0['prodE'])
        ss_adopt = solve_ss(model, adopt_calib)
        calib['psi_g'] = float(ss_adopt['psi_g'])
        unknowns, targets = SS_UNKNOWNS_FIXED_PSI, SS_TARGETS_FIXED_PSI
    else:
        unknowns, targets = SS_UNKNOWNS, SS_TARGETS

    # Under the fixed-quantity closure, world energy supply must equal the
    # model's own steady-state energy demand, otherwise E_clearing does not
    # hold at the steady state. Demand comes from the durable block's exact
    # CES hetoutputs (0.0402), not from alpha_E*C (0.0400), so it has to be
    # read off a solved steady state rather than assumed. The steady state
    # itself is identical under both closures (PEstar = 1 either way), so we
    # solve once under the price-taking closure to calibrate the quantity.
    if closure == 'inelastic':
        cal0 = dict(calib)
        cal0['E_supply_elasticity'] = np.inf
        ss0 = solve_ss(model, cal0, unknowns=unknowns, targets=targets)
        calib['E_supply_shock'] = float(ss0['cE'] + ss0['prodE'])

    ss = solve_ss(model, calib, unknowns=unknowns, targets=targets)

    # The Slutsky transfer is indexed to pre-crisis household energy use, so
    # the grids must be filled from the solved steady state and the model
    # re-solved on them.
    if POLICIES[policy].get('insE', 0.0) != 0.0:
        calib = set_energy_grids(calib, ss)
        ss = solve_ss(model, calib, unknowns=unknowns, targets=targets)

    shk = (shock_price(**(shock_kwargs or {})) if shock_kind == 'price'
           else shock_supply(ss, **(shock_kwargs or {})))
    irf = model.solve_impulse_linear(ss, UNKNOWNS_TD, TARGETS_TD, shk)
    B.test_targets(irf)
    return ss, irf
