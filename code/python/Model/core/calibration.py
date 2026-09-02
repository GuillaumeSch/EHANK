"""Calibration for E-HANK: base parameters, green adoption, ETS, policy."""
import numpy as np

BASE = dict(
    # preferences / income risk
    r=0.01, eis=1, frisch=0.5,
    sd_e=0.57, rho_e=0.92, n_e=7,
    n_beta=3, beta_spread=0.06, beta_max=0.95,
    min_a=0, max_a=400, n_a=50,

    # consumption nests: energy vs (home, foreign)
    eta_E=0.1,        # energy / non-energy substitution
    alpha_E=0.04,     # steady-state energy expenditure share
    alpha=0.3,        # total import share

    # firms / price setting
    markup_ss=1.03, theta_w=0.938, theta_E=0.65, theta_F=0.9,
    w_BG=5,           # real wage rigidity (0 = none)
    prodE_share=0, prodE_es=0.1,   # energy in production (off in baseline)

    # inf -> price-taking SOE (PEstar exog); finite -> fixed quantity (PEstar endog)
    E_supply_elasticity=np.inf,
    zetaEsupply=0.0,  # home ownership share of energy rents
    Gamma_arb=100,    # intertemporal arbitrage in energy stocks

    # portfolio / open economy
    eps_dcp=1, pcX_home=1, ghh_prefs=0, scale_w=0, cbarE=0,

    # monetary policy: real-rate rule
    rho_i=0, phi_pi=0, phi_pie=1, phi_piw=0,

    # fiscal
    B=0, psiB=0.04, tauY=0, bb=0,
)


DURABLE = dict(
    delta_g=0.05,      # green durable breakdown rate (quarterly)
    psi_g_bar=0.253,   # switching-cost bundle quantity; solved to D_GREEN_ss=0.05
    taste_shock=0.05,  # logit scale on the adoption choice
    # Green/brown operating-cost ratio rho = (P_elec*e_BEV)/(P_petrol*e_ICE)
    #   = (0.290 EUR/kWh * 0.21 kWh/km) / (1.62 EUR/L * 0.070 L/km) = 0.54
    # Prices: Eurostat nrg_pc_204 (electricity), EC Weekly Oil Bulletin (petrol).
    # Efficiencies: IEA Global EV Outlook 2026 (BEV 21 kWh/100km); on-road ICE
    # ~7 L/100km (FR SDES 7.1).
    PEGstar=0.54,      # exogenous world price of green energy (green/brown SS ratio)
    green_block=0.0,   # 0 = adoption open; large = adoption shut (counterfactual)
    D_GREEN_ss_target=0.05,
    alpha_F_switch=1.0,  # import share of the adoption-expenditure bundle;
                         # 1 = pure-import booking (baseline), <1 routes part of
                         # adoption spending onto domestic output
)


# ETS/carbon tax: tau_b, tau_g brown/green tax; s_g_ets green subsidy (recycle='green_subsidy')
ETS = dict(tau_b=0.0, tau_g=0.0, s_g_ets=0.0)


POLICY = dict(
    tauE=0.0,   # energy price cap / subsidy. 1 = household brown price fully capped
    insE=0.0,   # Slutsky transfer indexed to pre-crisis energy consumption
    epsT=0.0,   # untargeted lump-sum transfer
    s_g=0.0,    # green subsidy: fraction of psi_g paid by the government. 0 at
                # the SS; fed as a transition path for the 'green' policy.
)


def _derived(c):
    """Openness parameters and steady-state normalisations implied by BASE."""
    c['alpha_F'] = (c['alpha'] - c['alpha_E']) / (1 - c['alpha_E'])
    c['alpha_F_tilde'] = (1 - c['alpha_E']) * c['alpha_F']
    chi_target = 0.3
    c['gamma'] = ((chi_target - (1 - c['alpha']) * (1 - c['alpha_F']) * c['eta_E'])
                  / ((1 - c['alpha']) * c['alpha_F'] + c['alpha']))
    c['eta'] = c['gamma']

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

    # SS brown-energy grid (e,a); zeros switch the Slutsky transfer off.
    for i in range(3):
        c[f'cE_ss_grid_{i}'] = np.zeros((c['n_e'], c['n_a']))
    return c


def make_calibration(numeraire='cpi', booking='import', ets=False, **overrides):
    """Build a calibration; numeraire, booking, ets must match build_model."""
    c = dict(BASE)
    c.update(DURABLE)
    c.update(ETS)
    c.update(POLICY)
    c.update(overrides)
    c = _derived(c)
    c['p_num'] = 1.0
    if ets:
        c.pop('Trebate', None)
    return c


def set_energy_grids_flat(calib, ss):
    """Flat (untargeted) counterpart of set_energy_grids."""
    out = dict(calib)
    k = float(ss['CE_B'])
    for i in range(3):
        out[f'cE_ss_grid_{i}'] = np.full((calib['n_e'], calib['n_a']), k)
    return out


def set_energy_grids(calib, ss):
    """Fill cE_ss_grid_i with each household's SS brown energy demand."""
    out = dict(calib)
    for i in range(3):
        cE_b = np.asarray(ss.internals[f'hh_{i}']['consav']['cE_b'])  # (4,e,a)
        D = np.asarray(ss.internals[f'hh_{i}']['consav']['D'])
        w = D.sum(axis=0, keepdims=True)
        w = np.where(w > 0, w, 1.0)
        out[f'cE_ss_grid_{i}'] = (cE_b * D).sum(axis=0) / w[0]            # (e,a)
    return out
