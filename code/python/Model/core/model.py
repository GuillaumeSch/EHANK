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

from core import blocks as B
from core.household import hh_ha_durable
from core.calibration import make_calibration, set_energy_grids, set_energy_grids_flat
from core.frozen_adoption import build_model_frozen

T = 300

# --- transition-path unknowns/targets, by booking ---------------------------
_UNKNOWNS_TD_BASE = {'y', 'pi', 'r', 'tauY', 'PEstar', 'w'}
_TARGETS_TD_BASE = {'goods_clearing', 'pires', 'r_res', 'tauY_res', 'E_clearing', 'w_res'}

# --- steady-state unknowns/targets, by booking ------------------------------
_SS_UNKNOWNS_BASE = {'vphi': 1, 'beta_max': 0.984, 'y': 0.9868, 'psi_g': 0.253}
_SS_TARGETS_BASE = ['piwres', 'nfares', 'goods_clearing', 'D_GREEN_res']


def td_unknowns_targets(booking='import', ets=False):
    u, t = set(_UNKNOWNS_TD_BASE), set(_TARGETS_TD_BASE)
    if booking == 'domestic':
        u.add('Tgreen'); t.add('Tgreen_res')
    if ets:
        u.add('Trebate'); t.add('Trebate_res')
    return u, t


def ss_unknowns_targets(booking='import', ets=False):
    u, t = dict(_SS_UNKNOWNS_BASE), list(_SS_TARGETS_BASE)
    if booking == 'domestic':
        u = dict(u); u['Tgreen'] = 0.002
        t = t + ['Tgreen_res']
    if ets:
        u = dict(u); u['Trebate'] = 0.0
        t = t + ['Trebate_res']
    return u, t


# Kept for back-compatibility (import-booking defaults).
UNKNOWNS_TD, TARGETS_TD = td_unknowns_targets('import')
SS_UNKNOWNS, SS_TARGETS = ss_unknowns_targets('import')


def dissolve_list(booking='import'):
    ca = 'CA_dom' if booking == 'domestic' else 'CA'
    return ['unions', 'UIP', ca, 'piW_to_W', 'pitop']


# Used only by the ETS 'prepared economy' (psi_g fixed at the no-ETS baseline,
# D_GREEN floats up under the permanent carbon tax -- see run()). NOT used by
# the no_adoption counterfactual any more: that counterfactual now shares the
# adoption economy's steady state exactly (see frozen_model() / MODELS below),
# so psi_g is a genuine target there too and this carryover is unnecessary.
def ss_unknowns_targets_fixed_psi(booking='import', ets=False):
    u, t = ss_unknowns_targets(booking, ets=ets)
    u = {k: v for k, v in u.items() if k != 'psi_g'}
    t = [x for x in t if x != 'D_GREEN_res']
    return u, t


SS_UNKNOWNS_FIXED_PSI, SS_TARGETS_FIXED_PSI = ss_unknowns_targets_fixed_psi('import')


def build_model(numeraire='core', booking='import', ets=False):
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
    # ETS carbon revenue/rebate is folded into the fiscal block (so it enters
    # the asset identity via B); no separate block is needed. `ets` here only
    # documents intent -- the Trebate unknown/target is toggled in
    # ss/td_unknowns_targets, and the carbon terms vanish at tau_b=tau_g=0.
    return sj.combine([
        hh_ha_durable(),
        num, B.assets_convert,
        B.hh_outputs_dur, B.green_energy_price, *margin,
        B.income, B.profitcenters, B.importPrices, imp,
        B.revaluation, B.revaluation_dom, B.foreign_c, B.UIP, B.IEA, ca,
        B.unions, B.piW_to_W, B.CESprices, B.price_levels, B.pitop,
        B.mon_policy, B.fiscal, B.annualize, eqm,
    ])


_FROZEN_CACHE = {}


def frozen_model(numeraire='core', booking='import', ets=False):
    """The 'no adoption' counterfactual DAG, common-steady-state version.

    Identical to build_model(numeraire, booking, ets) except the household's
    discrete green-adoption choice does not respond to shocks (households keep
    switching at their STEADY-STATE rate, so the durable composition is frozen
    at its steady-state shares through the whole transition -- see
    frozen_adoption.py for the mechanism and the bit-identical-SS proof).

    Used only for the TRANSITION. The steady state for this counterfactual is
    solved with build_model(...) as usual (solve_ss(model, ...), NOT this
    function) -- the two DAGs share a bit-identical steady state by
    construction, so there is no separate steady state to solve here. This
    replaces the older 'no_adoption' counterfactual (green_block huge), which
    shut the margin by pinning D_GREEN=0 IN THE STEADY STATE and therefore
    confounded the crisis response with a different starting steady-state
    composition (5% vs 0% green). See frozen_adoption.py and
    ADOPTION_decomposition.md for the comparison.

    Cached per (numeraire, booking, ets); building the DAG is cheap but there
    is no reason to repeat it.
    """
    key = (numeraire, booking, ets)
    if key not in _FROZEN_CACHE:
        _FROZEN_CACHE[key] = build_model_frozen(numeraire, booking=booking, ets=ets)
    return _FROZEN_CACHE[key]


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


def shock_green(size=1.0, half_life=16, T=T):
    """Transitory green/adoption subsidy path (s_g), layered on top of the
    driving energy shock for the 'green' crisis-response policy. s_g = 0 at the
    SS; here it jumps to `size` on impact and decays with the crisis (same
    default half-life as the price shock), so the SS is untouched and the E2/E3
    comparison against none/cap/transfer is like-for-like."""
    rho = 2 ** (-1 / half_life)
    return {'s_g': size * rho ** np.arange(T)}


def shock_mon(size=0.0025, half_life=4, T=T):
    """Monetary policy shock: additive innovation to the gross nominal rate in
    mon_policy's inom_res. `size` is the impact deviation of the QUARTERLY gross
    nominal rate; 0.0025 ~ 100bp annualised at impact. size > 0 is a tightening.
    Standard MP shock under the Taylor rule (monetary='taylor'); under the ARS
    constant-real-rate rule (phi_pie=1) it perturbs the nominal rate with the
    real rate pinned. AR(1) decay with the given half-life."""
    rho = 2 ** (-1 / half_life)
    return {'ishock': size * rho ** np.arange(T)}


# =============================================================================
# EXPERIMENTS
# =============================================================================
#   name -> calibration overrides applied on top of the scenario baseline
POLICIES = {
    'none':          dict(tauE=0.0, insE=0.0),
    'subsidy':       dict(tauE=1.0, insE=0.0),  # full price cap (Bayer P1, Langot)
    'transfer':      dict(tauE=0.0, insE=1.0),  # Slutsky compens. (Bayer P2, Germany)
    'transfer_flat': dict(tauE=0.0, insE=1.0),  # untargeted lump sum, same envelope
    'green':         dict(tauE=0.0, insE=0.0),  # adoption/switch subsidy (s_g path
                                                # layered on the shock; 0 at SS)
}

# Impact size of the transitory green-subsidy path for policy='green'. 1.0 = the
# government pays the full switching cost during the acute crisis.
GREEN_SIZE = 1.0

# 'no_adoption' now means the COMMON-STEADY-STATE counterfactual: identical
# calibration to 'adoption' (no override -- both target the same D_GREEN=5% SS),
# so the steady state solved in run() is bit-identical either way. The only
# difference is which DAG computes the TRANSITION: run() swaps in
# frozen_model() for 'no_adoption', whose adoption choice does not respond to
# the shock. green_block (household.py) is no longer used by this switch; it
# remains available as a primitive for a DIFFERENT, unused counterfactual
# ("the technology does not exist"), which is not what 'no_adoption' means here.
MODELS = {
    'adoption':    dict(),
    'no_adoption': dict(),
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
        booking='import', ets=False, ets_kwargs=None, **extra):
    """Solve one experiment end to end. Returns (ss, irf).

    The energy closure is implied by shock_kind: a price shock needs the
    price-taking closure, a supply shock needs the fixed-quantity closure.

    `numeraire` and `booking` must match the ones `model` was built with.

    ets=True solves the "prepared" ETS steady state (Version A): a permanent
    carbon tax with psi_g held FIXED at its no-ETS value, so D_GREEN floats up.
    ets_kwargs = dict(tau_b=..., tau_g=..., s_g_ets=..., recycle='rebate'|
    'green_subsidy'). The shock and policy are then applied on this greener SS.

    model_variant='no_adoption' is the COMMON-STEADY-STATE counterfactual (see
    frozen_model()): the steady state solved below is IDENTICAL to
    model_variant='adoption' (same calibration, same D_GREEN, same psi_g --
    MODELS['no_adoption'] is a no-op override by construction); only the
    TRANSITION differs, computed on frozen_model(numeraire, booking, ets)
    instead of `model`. `model` itself must always be the ADOPTION dag
    (build_model(...)), regardless of model_variant -- it is used to solve the
    (shared) steady state either way.
    """
    closure = 'inelastic' if shock_kind == 'supply' else 'elastic'
    ov = {}
    ov.update(ENERGY_CLOSURE[closure])
    ov.update(MONETARY[monetary])
    ov.update(MODELS[model_variant])   # no-op for both variants now
    ov.update(POLICIES[policy])
    ov.update(extra)

    ek = dict(ets_kwargs or {})
    recycle = ek.pop('recycle', 'rebate')
    if ets:
        ov.update({k: ek[k] for k in ('tau_b', 'tau_g', 's_g_ets') if k in ek})
        if recycle == 'green_subsidy':
            ov['s_g'] = ek.get('s_g_ets', 0.0)   # permanent, carbon-financed

    calib = make_calibration(numeraire, booking=booking, ets=ets, **ov)
    unknowns_td, targets_td = td_unknowns_targets(booking, ets=ets)

    if ets:
        # Prepared economy: psi_g fixed at the no-ETS baseline, D_GREEN floats.
        base_calib = make_calibration(numeraire, booking=booking, **{
            k: v for k, v in ov.items()
            if k not in ('tau_b', 'tau_g', 's_g_ets', 's_g')})
        ss_base = solve_ss(model, base_calib, booking=booking)
        calib['psi_g'] = float(ss_base['psi_g'])
        unknowns, targets = ss_unknowns_targets_fixed_psi(booking, ets=True)
    else:
        # Same branch for 'adoption' AND 'no_adoption': the steady state is
        # shared by construction (common-SS counterfactual).
        unknowns, targets = ss_unknowns_targets(booking)

    if closure == 'inelastic':
        calib['E_supply_shock'] = _calibrate_supply(
            model, calib, unknowns, targets, booking)

    ss = solve_ss(model, calib, unknowns=unknowns, targets=targets, booking=booking)

    if POLICIES[policy].get('insE', 0.0) != 0.0:
        filler = set_energy_grids_flat if policy == 'transfer_flat' else set_energy_grids
        calib = filler(calib, ss)
        ss = solve_ss(model, calib, unknowns=unknowns, targets=targets, booking=booking)

    if shock_kind == 'price':
        shk = shock_price(**(shock_kwargs or {}))
    elif shock_kind == 'monetary':
        shk = shock_mon(**(shock_kwargs or {}))
    else:
        shk = shock_supply(ss, **(shock_kwargs or {}))
    if policy == 'green':
        hl = (shock_kwargs or {}).get('half_life', 16)
        shk = {**shk, **shock_green(size=GREEN_SIZE, half_life=hl)}

    # Only place model_variant affects the TRANSITION: swap in the frozen-choice
    # DAG for 'no_adoption'. Everything upstream (ss, shk) is shared.
    transition_model = (frozen_model(numeraire, booking, ets)
                        if model_variant == 'no_adoption' else model)
    irf = transition_model.solve_impulse_linear(ss, unknowns_td, targets_td, shk)
    B.test_targets(irf)
    return ss, irf
