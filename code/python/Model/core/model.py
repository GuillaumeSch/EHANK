"""Model assembly, steady state, shocks, and the experiment runner."""
import numpy as np
import sequence_jacobian as sj

from core import blocks as B
from core.household import hh_ha_durable
from core.calibration import make_calibration, set_energy_grids, set_energy_grids_flat
from core.frozen_adoption import build_model_frozen

T = 300

_UNKNOWNS_TD_BASE = {'y', 'pi', 'r', 'tauY', 'PEstar', 'w'}
_TARGETS_TD_BASE = {'goods_clearing', 'pires', 'r_res', 'tauY_res', 'E_clearing', 'w_res'}

_SS_UNKNOWNS_BASE = {'vphi': 1, 'beta_max': 0.984, 'y': 0.9868, 'psi_g_bar': 0.253, 'Z': 1.03}
_SS_TARGETS_BASE = ['piwres', 'nfares', 'goods_clearing', 'D_GREEN_res', 'pires']


def td_unknowns_targets(booking='import', ets=False):
    u, t = set(_UNKNOWNS_TD_BASE), set(_TARGETS_TD_BASE)
    if ets:
        u.add('Trebate'); t.add('Trebate_res')
    return u, t


def ss_unknowns_targets(booking='import', ets=False):
    u, t = dict(_SS_UNKNOWNS_BASE), list(_SS_TARGETS_BASE)
    if ets:
        u = dict(u); u['Trebate'] = 0.0
        t = t + ['Trebate_res']
    return u, t


UNKNOWNS_TD, TARGETS_TD = td_unknowns_targets('import')
SS_UNKNOWNS, SS_TARGETS = ss_unknowns_targets('import')


def dissolve_list(booking='import'):
    return ['unions', 'UIP', 'CA', 'piW_to_W', 'pitop']


def ss_unknowns_targets_fixed_psi(booking='import', ets=False):
    u, t = ss_unknowns_targets(booking, ets=ets)
    u = {k: v for k, v in u.items() if k != 'psi_g_bar'}
    t = [x for x in t if x != 'D_GREEN_res']
    return u, t


SS_UNKNOWNS_FIXED_PSI, SS_TARGETS_FIXED_PSI = ss_unknowns_targets_fixed_psi('import')


def build_model(numeraire='cpi', booking='import', ets=False):
    """Assemble the full DAG."""
    num = B.numeraire_cpi
    margin = [B.energy_gap]
    ca, imp, eqm = B.CA, B.importProfits, B.eqm_cond
    return sj.combine([
        hh_ha_durable(),
        num, B.assets_convert,
        B.hh_outputs, *margin,
        B.income, B.profitcenters, B.importPrices, imp,
        B.revaluation, B.revaluation_dom, B.foreign_c, B.UIP, B.IEA, ca,
        B.unions, B.piW_to_W, B.CESprices, B.price_levels, B.pitop,
        B.mon_policy, B.fiscal, B.annualize, eqm, B.reweight_cpi,
    ])


_FROZEN_CACHE = {}


def frozen_model(numeraire='cpi', booking='import', ets=False):
    """The 'no adoption' counterfactual DAG (common steady state)."""
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


# def _energy_demand(ss, booking):
#     """SS energy demand clearing against fixed world supply."""
#     return float(ss['cE'] + ss['prodE'])

def _energy_demand(ss, booking):
    """SS energy demand clearing against fixed world supply."""
    return float(ss['CE_B'] + ss['prodE'])


_ESUP_CACHE = {}


def _calibrate_supply(model, calib, unknowns, targets, booking, iters=6, tol=1e-10):
    """Fixed-quantity closure: solve E_supply_shock so demand meets supply.
    Memoised per session on the SS-relevant calibration (invariant across policies)."""
    key = (booking, tuple(sorted((k, v) for k, v in calib.items()
           if isinstance(v, (int, float, str, bool))
           and k not in ('E_supply_shock', 'E_supply_elasticity'))))
    if key in _ESUP_CACHE:
        return _ESUP_CACHE[key]
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
    _ESUP_CACHE[key] = demand
    return demand


def shock_price(size=1.0, half_life=16, T=T):
    """Brown energy price shock (requires E_supply_elasticity = inf)."""
    rho = 2 ** (-1 / half_life)
    return {'PEstar_shock': size * rho ** np.arange(T)}


import os as _os
_MATCH_CACHE = _os.path.join(_os.path.dirname(__file__), '..', 'data', 'matched')


def matched_supply_path(model, size=1.0, half_life=16, T=T,
                        numeraire='cpi', booking='import',
                        cache=True, recompute=False):
    """Matched E_supply_shock path: reproduces the price shock in the no-policy
    baseline. Cached to disk (key: numeraire/booking/size/half_life/T)."""
    key = f'supply_matched_{numeraire}_{booking}_s{size}_hl{half_life}_T{T}.npy'
    fpath = _os.path.join(_MATCH_CACHE, key)
    if cache and not recompute and _os.path.exists(fpath):
        return np.load(fpath)
    _, irf = run(model, shock_kind='price', policy='none', numeraire=numeraire,
                 booking=booking,
                 shock_kwargs=dict(size=size, half_life=half_life, T=T))
    path = np.asarray(irf['CE_B'], float) - np.asarray(irf['E_supply'], float)
    if cache:
        _os.makedirs(_MATCH_CACHE, exist_ok=True)
        np.save(fpath, path)
    return path


def shock_supply_matched(model, size=1.0, half_life=16, T=T,
                         numeraire='cpi', booking='import',
                         cache=True, recompute=False):
    """{'E_supply_shock': matched path}. See matched_supply_path."""
    return {'E_supply_shock': matched_supply_path(
        model, size=size, half_life=half_life, T=T, numeraire=numeraire,
        booking=booking, cache=cache, recompute=recompute)}


def shock_green(size=1.0, half_life=16, T=T):
    """Transitory green-subsidy (s_g) path."""
    rho = 2 ** (-1 / half_life)
    return {'s_g': size * rho ** np.arange(T)}


def shock_mon(size=0.0025, half_life=4, T=T):
    """Monetary policy shock: innovation to the gross nominal rate."""
    rho = 2 ** (-1 / half_life)
    return {'ishock': size * rho ** np.arange(T)}


POLICIES = {
    'none':          dict(tauE=0.0, insE=0.0),
    'subsidy':       dict(tauE=.5, insE=0.0),  # full price cap (Bayer P1, Langot)
    'transfer':      dict(tauE=0.0, insE=1.0),  # Slutsky compensation (Bayer P2)
    'transfer_flat': dict(tauE=0.0, insE=1.0),  # untargeted lump sum, same envelope
    'green':         dict(tauE=0.0, insE=0.0),  # adoption subsidy (s_g path layered
                                                # on the shock; 0 at SS)
}

GREEN_SIZE = 1.0

# 'no_adoption' shares the 'adoption' SS; only the transition differs
MODELS = {
    'adoption':    dict(),
    'no_adoption': dict(),
}

MONETARY = {
    'real_rate': dict(phi_pi=0.0, phi_pie=1.0),   # constant real rate (ARS baseline)
    'taylor':    dict(phi_pi=1.5, phi_pie=0.0),   # standard Taylor rule
}

ENERGY_CLOSURE = {
    'elastic':   dict(E_supply_elasticity=np.inf),  # price-taking SOE (ARS/Langot)
    'inelastic': dict(E_supply_elasticity=1.0),     # fixed quantity  (Bayer)
}


def run(model, shock_kind='price', policy='none', model_variant='adoption',
        monetary='real_rate', shock_kwargs=None, numeraire='cpi',
        booking='import', ets=False, ets_kwargs=None, **extra):
    """Solve one experiment end to end; returns (ss, irf)."""
    closure = 'inelastic' if shock_kind == 'supply' else 'elastic'
    ov = {}
    ov.update(ENERGY_CLOSURE[closure])
    ov.update(MONETARY[monetary])
    ov.update(MODELS[model_variant])
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

    adoption_shut = float(ov.get('green_block', 0.0)) > 0.0
    if ets or adoption_shut:
        # psi_g fixed at baseline; D_GREEN floats
        base_calib = make_calibration(numeraire, booking=booking, **{
            k: v for k, v in ov.items()
            if k not in ('tau_b', 'tau_g', 's_g_ets', 's_g', 'green_block')})
        base_calib['E_supply_elasticity'] = np.inf   # closure-invariant here; avoids a spurious SS energy residual
        ss_base = solve_ss(model, base_calib, booking=booking)
        calib['psi_g_bar'] = float(ss_base['psi_g_bar'])
        unknowns, targets = ss_unknowns_targets_fixed_psi(booking, ets=ets)
    else:
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
    else:  # 'supply' = matched supply shock
        shk = shock_supply_matched(model, numeraire=numeraire, booking=booking,
                                   **(shock_kwargs or {}))
    if policy == 'green':
        hl = (shock_kwargs or {}).get('half_life', 16)
        shk = {**shk, **shock_green(size=GREEN_SIZE, half_life=hl)}

    # frozen-choice DAG for 'no_adoption'; ss and shk are shared
    transition_model = (frozen_model(numeraire, booking, ets)
                        if model_variant == 'no_adoption' else model)
    irf = transition_model.solve_impulse_linear(ss, unknowns_td, targets_td, shk)
    B.test_targets(irf)
    return ss, irf