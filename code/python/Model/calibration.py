"""Calibration for E-HANK.

Three layers, kept separate on purpose:
  1. BASE       structural parameters, common to every experiment
  2. DURABLE    the green-adoption block (this paper's addition)
  3. POLICY     experiment switches; all default to "no policy"

Anything an experiment changes goes through `make_calibration(**overrides)`,
so a run is fully described by its overrides.
"""
import numpy as np

# =============================================================================
# 1. BASE (Auclert-Monnery-Rognlie-Straub small open economy)
# =============================================================================
BASE = dict(
    # --- preferences / income risk
    r=0.01, eis=1, frisch=0.5,
    sd_e=0.57, rho_e=0.92, n_e=7,
    n_beta=3, beta_spread=0.06, beta_max=0.95,
    min_a=0, max_a=400, n_a=50,

    # --- consumption nests: energy vs (home, foreign)
    eta_E=0.1,        # energy / non-energy substitution
    alpha_E=0.04,     # steady-state energy expenditure share
    alpha=0.3,        # total import share

    # --- firms / price setting
    markup_ss=1.03, theta_w=0.938, theta_E=0.65, theta_F=0.9,
    w_BG=5,           # real wage rigidity (0 = none)
    prodE_share=0, prodE_es=0.1,   # energy in production (off in baseline)

    # --- energy supply
    # E_supply_elasticity = inf  -> price-taking SOE: PEstar exogenous.
    # E_supply_elasticity finite -> quantity fixed by IEA, PEstar endogenous.
    E_supply_elasticity=np.inf,
    zetaEsupply=0.0,  # home ownership share of energy rents
    Gamma_arb=100,     # intertemporal arbitrage in energy stocks

    # --- portfolio / open economy
    zetaF=0, zetaE=0, eps_dcp=1, pcX_home=1,
    ghh_prefs=0, w_index=0, scale_w=0, cbarE=0, wealth_effect=1,

    # --- monetary policy: real-rate rule (ARS baseline)
    rho_i=0, phi_pi=0, phi_pie=1, phi_piw=0,

    # --- fiscal
    B=0, psiB=0.04, tauY=0, bb=0,
)


# =============================================================================
# 2. DURABLE / GREEN ADOPTION
# =============================================================================
DURABLE = dict(
    delta_g=0.05,      # green durable breakdown rate (quarterly)
    psi_g=0.253,       # switching cost, calibrated to D_GREEN_ss = 0.05
    taste_shock=0.05,  # logit scale on the adoption choice
    pE_g_ratio=0.8,    # steady-state green/brown energy price ratio
    pass_g=0.0,        # pass-through of brown energy price to green price
    green_block=0.0,   # 0 = adoption open; large = adoption shut (counterfactual)
    D_GREEN_ss_target=0.05,
)


# =============================================================================
# 2bis. ETS / CARBON TAX (steady-state, budget-neutral). All zero = no ETS.
# =============================================================================
# Permanent carbon tax on the consumer energy price, with the revenue recycled
# in a balanced sub-account (carbon_sector). Only active when build_model /
# make_calibration are called with ets=True.
#
#   tau_b   carbon tax rate on BROWN energy      (pE_B_P = pE_P*(1+tau_b))
#   tau_g   carbon tax rate on GREEN energy      (~0; green is near-clean)
#   s_g_ets PERMANENT green subsidy rate financed by carbon revenue
#           (recycle='green_subsidy'); 0 for recycle='rebate'
#
# The effective SS brown/green price ratio becomes pE_g_ratio*(1+tau_g)/(1+tau_b),
# so a positive tau_b raises steady-state adoption (D_GREEN floats up; psi_g is
# held fixed at its no-ETS value). At tau_b=tau_g=0 the model is bit-identical
# to the no-ETS baseline.
ETS = dict(tau_b=0.0, tau_g=0.0, s_g_ets=0.0)


# =============================================================================
# 3. POLICY (all zero = laissez-faire)
# =============================================================================
POLICY = dict(
    tauE=0.0,   # energy price cap / subsidy. 1 = household price fully capped
    insE=0.0,   # Slutsky transfer indexed to pre-crisis energy consumption
    epsT=0.0,   # untargeted lump-sum transfer
    s_g=0.0,    # green/adoption subsidy: fraction of the switching cost psi_g
                # paid by the government. 0 at the SS (no distortion); fed as a
                # transition path for the 'green' crisis-response policy.
)


# =============================================================================
# STEADY-STATE NORMALISATIONS AND INITIAL GUESSES
# =============================================================================
def _derived(c):
    """Openness parameters and steady-state normalisations implied by BASE."""
    c['alpha_F'] = (c['alpha'] - c['alpha_E']) / (1 - c['alpha_E'])
    c['alpha_F_tilde'] = (1 - c['alpha_E']) * c['alpha_F']
    chi_target = 0.3
    c['gamma'] = ((chi_target - (1 - c['alpha']) * (1 - c['alpha_F']) * c['eta_E'])
                  / ((1 - c['alpha']) * c['alpha_F'] + c['alpha']))
    c['eta'] = c['gamma']

    # Home owns a share zetaEsupply of the energy endowment: rescale output,
    # the markup and the foreign demand shifter consistently.
    z, aE = c['zetaEsupply'], c['alpha_E']
    c['y'] = 1 - z * aE
    c['markup_ss'] = c['markup_ss'] * (1 - z * aE)
    c['alphastar'] = c['alpha'] - aE * z
    c['Z'] = c['markup_ss']
    c['E_supply_shock'] = aE

    c.update({'Q': 1, 'ishock': 0, 'Cstar': 1, 'piw': 0, 'pi': 0, 'P': 1,
              'nfa': 0, 'PFstar': 1, 'W': 1, 'rstar': c['r'],
              'pH_PHF': 1, 'pHstar': 1, 'pF_PHF': 1, 'pE_B_P': 1, 'pHF_P': 1,
              'dividend_X': 0, 'vphi': 1, 'rante': c['r'],
              'beta_RA': 1 / (1 + c['r']), 'C': 1, 'A': 1, 'w': 1,
              'PEstar_shock': 1, 'PEstar': 1, 'inom_t': 0, 'union_wedge': 0,
              'Tgreen': 0.0, 'Trebate': 0.0})

    # Per-household steady-state BROWN energy grid (n_e, n_a), collapsed over
    # the durable axis. Zeros switch the Slutsky transfer OFF; `set_energy_grids`
    # fills it from a solved steady state when the transfer experiment needs it.
    for i in range(3):
        c[f'cE_ss_grid_{i}'] = np.zeros((c['n_e'], c['n_a']))
    return c


def make_calibration(numeraire='core', booking='import', ets=False, **overrides):
    """Build a calibration. `numeraire`, `booking` and `ets` must match build_model's.

    Under 'cpi' the price of the unit of account is a CONSTANT p_num = 1 and
    is supplied here rather than produced by a block: a block returning a
    literal 1 would have an identically-zero Jacobian row and hit SSJ's
    SimpleSparse empty-operator crash. Under 'core' p_num = pH_P is a genuine
    block output and must be absent from the calibration.

    Tgreen (the domestic green-sector rebate) is symmetric: under the 'import'
    booking there is no green sector, so Tgreen is the fixed constant 0 the
    household reads; under 'domestic' it is a genuine block output / model
    unknown produced by green_sector and must be absent from the calibration.

    Trebate (the carbon-revenue lump-sum rebate) is the same: 0 constant when
    ets=False, and a genuine carbon_sector output / model unknown when ets=True.
    """
    c = dict(BASE)
    c.update(DURABLE)
    c.update(ETS)
    c.update(POLICY)
    c.update(overrides)
    c = _derived(c)
    c['subsidy_brown_only'] = 1.0 if booking == 'domestic' else 0.0
    if numeraire == 'cpi':
        c['p_num'] = 1.0
    else:
        c.pop('p_num', None)
    if booking == 'domestic':
        c.pop('Tgreen', None)
    if ets:
        c.pop('Trebate', None)
    return c


def set_energy_grids_flat(calib, ss):
    """Untargeted (flat) lump-sum counterpart of set_energy_grids.

    Fills cE_ss_grid_i with the UNIFORM scalar CE_DUR_B.ss instead of each
    household's own pre-crisis brown energy. The distribution-weighted aggregate
    is identical (mean of the targeted grid equals CE_DUR_B.ss by construction),
    so the government outlay -- and hence assets_clearing -- is unchanged; only
    the INCIDENCE differs. This isolates targeting from envelope: the flat
    transfer pays every household the same lump sum whose total equals what the
    Slutsky transfer costs, letting E-HANK's distributional bite be read off the
    CEV-by-type gap between the two.
    """
    out = dict(calib)
    k = float(ss['CE_DUR_B'])
    for i in range(3):
        out[f'cE_ss_grid_{i}'] = np.full((calib['n_e'], calib['n_a']), k)
    return out


def set_energy_grids(calib, ss):
    """Fill cE_ss_grid_i with each household's steady-state BROWN energy demand,
    collapsed (mass-weighted) over the durable axis to (n_e, n_a).

    Required for the Slutsky transfer (insE > 0): the transfer is a LUMP SUM
    indexed to PRE-CRISIS energy use, as in Bayer et al. Two properties are
    imposed together:

      (i)  BROWN only. Only the brown price rises (pE_G_P fixed at pass_g = 0),
           so green energy must not be in the compensation base; cE_dur_b is
           already zero in the green durable states.
      (ii) Collapsed over the durable axis and then broadcast identically to
           every durable state in hh_income. This is what makes it a genuine
           lump sum: a household that switches brown->green during the crisis
           (state GB) keeps the SAME transfer, so the transfer does NOT tax the
           switching margin. Indexing it to the CURRENT durable state instead
           (zero for green states) would make switching forfeit the transfer
           and would spuriously shut down the adoption channel.

    The mass-weighting is by TOTAL durable mass, so the aggregate of the
    broadcast grid over the stationary distribution equals CE_DUR_B.ss exactly
    (verified) -- matching the government's Ttargeted and keeping assets_clearing
    at machine zero.
    """
    out = dict(calib)
    for i in range(3):
        cE_b = np.asarray(ss.internals[f'hh_{i}']['consav']['cE_dur_b'])  # (4,e,a)
        D = np.asarray(ss.internals[f'hh_{i}']['consav']['D'])
        w = D.sum(axis=0, keepdims=True)
        w = np.where(w > 0, w, 1.0)
        out[f'cE_ss_grid_{i}'] = (cE_b * D).sum(axis=0) / w[0]            # (e,a)
    return out
