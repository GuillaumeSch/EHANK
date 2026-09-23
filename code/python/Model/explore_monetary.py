#%%
import sys; sys.path.insert(0, '.')
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from core.model import build_model, run


#%% ---------------------------------------------------------------- 1. config
NUM, BOOK, H = 'cpi', 'import', 21
SAVE = False
RESULTS_PATH = 'explore_monetary_results.pkl'

FIGDIR = 'figures'
os.makedirs(FIGDIR, exist_ok=True)
def _fp(name):
    return os.path.join(FIGDIR, name)


ECONOMIES = {
    'baseline': dict(),
    #'ETS':      dict(ets=True, ets_kwargs=dict(tau_b=0.10, recycle='rebate')),
    #'brown':    dict(green_block=20.0),
}
SHOCKS = {
    'price':  dict(shock_kind='price'),
    #'supply': dict(shock_kind='supply'),
}
VARIANTS = ['adoption']            # , 'no_adoption'
FISCAL   = ['none']                # , 'subsidy', 'transfer', 'transfer_flat'

# Monetary rules overlaid within a cell. Baseline = constant real rate (ARS);
# Taylor ladder = increasing phi_pi (phi_pie stays 0 via the 'taylor' preset).
MON_RULES = {
    'Constant real rate':      dict(monetary='real_rate'),
    r'Taylor $\phi_\pi=1.25$': dict(monetary='taylor', phi_pi=1.25),
    r'Taylor $\phi_\pi=1.5$':  dict(monetary='taylor', phi_pi=1.5),
    r'Taylor $\phi_\pi=2$':    dict(monetary='taylor', phi_pi=2.0),
    r'Taylor $\phi_\pi=3$':    dict(monetary='taylor', phi_pi=3.0),
}

# Panels.
VARS_MON = [
    ('y_pc',          r'Output $Y$',                    '% dev. from SS'),
    ('C_pc',          r'Consumption $C$',               '% dev. from SS'),
    ('pi_ann_pp',     r'Inflation $\pi$ (annualized)',  'p.p dev. from SS'),
    ('rante_ann_pp',  r'Real rate $r$ (annualized)',    'p.p dev. from SS'),
    ('inom_ann_pp',   r'Nominal rate $i$ (annualized)', 'p.p dev. from SS'),
    ('D_GREEN_share', 'Green technology users',         'p.p dev. from SS'),
]

# Line styling: baseline in black, Taylor ladder on a sequential blue ramp.
BASE_STYLE  = dict(color='k', linestyle='-', linewidth=2.8, alpha=1.0)
TAYLOR_CMAP = plt.cm.Blues

model = build_model(NUM, booking=BOOK)


#%% ---------------------------------------------------------------- 2. run
def _augment(irf):
    """Annualized rates in p.p (pi_ann_pp already provided by the model)."""
    irf['rante_ann_pp'] = 100 * np.asarray(irf['rante_ann'])
    irf['inom_ann_pp']  = 100 * np.asarray(irf['inom_ann'])
    return irf


def run_all(model, economies=ECONOMIES, shocks=SHOCKS, variants=VARIANTS,
            fiscal=FISCAL, rules=MON_RULES, verbose=True):
    """Solve every (pol, econ, sname, variant, mon) cell; dict keyed by that
    tuple -> irf. Mirrors explore_irfs.run_all with monetary as an extra axis."""
    results = {}
    results_ss = {}
    for pol in fiscal:
        for econ, ekw in economies.items():
            for sname, shk in shocks.items():
                for variant in variants:
                    for mon, mkw in rules.items():
                        try:
                            ss, irf = run(model, policy=pol, model_variant=variant,
                                         **shk, **ekw, **mkw)
                            results[(pol, econ, sname, variant, mon)] = _augment(irf)
                            results_ss[(pol, econ, sname, variant, mon)] = ss
                            if verbose:
                                print(f'PASS {pol:8s} {econ:9s} {sname:7s} {variant:11s} {mon}')
                        except Exception as e:
                            if verbose:
                                print(f'FAIL {pol:8s} {econ:9s} {sname:7s} {variant:11s} {mon}: '
                                      f'{type(e).__name__}: {e}')
    return results, results_ss


def run_accommodation(model, shock_kind='price', econ='baseline', policy='none',
                      variant='adoption', monetary='real_rate',
                      mon_shock=dict(size=-0.0025, half_life=4)):
    """Energy shock alone vs the same shock with a simultaneous accommodative
    rate cut. Each is one impulse run; the combined case feeds both shocks into
    a single solve via run(..., mon_overlay=...). ishock < 0 is a rate cut."""
    ekw = ECONOMIES[econ] if isinstance(econ, str) else (econ or {})
    _, base = run(model, shock_kind=shock_kind, policy=policy,
                  model_variant=variant, monetary=monetary, **ekw)
    _, both = run(model, shock_kind=shock_kind, policy=policy,
                  model_variant=variant, monetary=monetary,
                  mon_overlay=mon_shock, **ekw)
    return _augment(base), _augment(both)


def save_results(results, path=RESULTS_PATH):
    with open(path, 'wb') as f:
        pickle.dump(results, f)


def load_results(path=RESULTS_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


results, results_ss = run_all(model)
if SAVE:
    save_results(results)


#%% ---------------------------------------------------------------- 3. plot
def plot_irfs(scenarios, variables=VARS_MON, len_irf=H, ni=3, nj=2,
              figsize=(7.5, 8.5), legend_ax_idx=0, legend_loc='best',
              save_path=None):
    """scenarios: list of (irf, label, style_kwargs)."""
    titlesize, labelsize, legendsize, axislabelsize = 11, 9, 8, 9
    fig, axes = plt.subplots(ni, nj, figsize=figsize, sharex=True)
    axes = axes.flatten()
    for ax, (var, title, ylabel) in zip(axes, variables):
        for irf, label, style in scenarios:
            y = np.asarray(irf[var])[:len_irf] if var in irf else np.zeros(len_irf)
            ax.plot(y, label=label, **style)
        ax.axhline(0, color='black', linewidth=1., alpha=0.6, zorder=0)
        ax.set_title(title, fontsize=titlesize)
        ax.tick_params(labelsize=labelsize)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xlabel('quarter', fontsize=axislabelsize)
        ax.set_ylabel(ylabel, fontsize=axislabelsize)
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.tick_params(labelbottom=True)
    for ax in axes[len(variables):]:
        ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[legend_ax_idx].legend(handles, labels, loc=legend_loc,
                               fontsize=legendsize, frameon=True, framealpha=0.85)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(_fp(save_path), bbox_inches='tight')
    return fig, axes


def rule_scenarios(results, pol, econ, sname, variant, rules=MON_RULES):
    """(irf, label, style) per monetary rule for one cell: baseline black,
    Taylor ladder on the blue ramp (darker = more aggressive)."""
    labels = list(rules)
    shades = np.linspace(0.45, 0.95, len(labels) - 1)
    scen = [(results[(pol, econ, sname, variant, labels[0])], labels[0], BASE_STYLE)]
    for lab, sh in zip(labels[1:], shades):
        scen.append((results[(pol, econ, sname, variant, lab)], lab,
                     dict(color=TAYLOR_CMAP(sh), linestyle='-', linewidth=2.4, alpha=1.0)))
    return scen


#%% Figure M1: monetary rules compared, for one selected cell
CELL = ('none', 'baseline', 'price', 'adoption')   # (fiscal, economy, shock, variant)
fig_rules, _ = plot_irfs(rule_scenarios(results, *CELL), legend_ax_idx=2,
                         save_path='irf_monetary_rules_price.pdf')
plt.show()


#%% Figure M2: energy shock with vs without accommodation, for one selected cell
irf_base, irf_both = run_accommodation(
    model, shock_kind='price', econ='baseline', policy='none',
    variant='adoption', monetary='real_rate', mon_shock=dict(size=-0.0025, half_life=4))

scenarios_accom = [
    (irf_base, 'Energy shock',                 dict(color='tab:blue', linestyle='-', linewidth=2.6, alpha=1.0)),
    (irf_both, 'Energy shock + accommodation', dict(color='tab:red',  linestyle='-', linewidth=2.6, alpha=1.0)),
]
fig_accom, _ = plot_irfs(scenarios_accom, legend_ax_idx=2,
                         save_path='irf_monetary_accommodation_price.pdf')
plt.show()

# %%
