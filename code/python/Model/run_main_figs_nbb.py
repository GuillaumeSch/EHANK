#%%
import sys; sys.path.insert(0, '.')
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from core.model import build_model, run, matched_supply_path


#%%
NUM, BOOK, H = 'cpi', 'import', 24
SAVE = True
RESULTS_PATH = 'explore_results.pkl'
RESULTS_SS_PATH = 'explore_results_ss.pkl'

# All figures are written into FIGDIR (created on import if missing).
FIGDIR = 'figures'
os.makedirs(FIGDIR, exist_ok=True)
def _fp(name):
    return os.path.join(FIGDIR, name)

ECONOMIES = {                     # colour
    'baseline': dict(),
    # 'ETS':      dict(ets=True, ets_kwargs=dict(tau_b=0.10, recycle='rebate')),
    # 'ETS_green_subsidy':      dict(ets=True, ets_kwargs=dict(tau_b=0.10, recycle='green_subsidy')),
    # 'ETS_green_subsidy_v2':  dict(ets=True, ets_kwargs=dict(tau_b=0.10, recycle='green_subsidy_dynamics')),
    'brown':    dict(green_block=20.0),
    # 'baseline_high_pe': dict(PEstar=1.1, PEstar_shock=1.1),
}
SHOCKS = {
    'price':  dict(shock_kind='price'),
    'supply': dict(shock_kind='supply'),
}
VARIANTS = [
    'adoption',
    'no_adoption'
    ]

FISCAL = [
    'none',
    'subsidy',
    'transfer',
    'transfer_flat',
    # "green"
    ]

model = build_model(NUM, booking=BOOK)

#%%
def run_all(model, economies=ECONOMIES, shocks=SHOCKS, variants=VARIANTS,
            fiscal=FISCAL, numeraire=NUM, booking=BOOK, verbose=True):
    """Solve every (fiscal, economy, shock, variant) cell; return a results dict
    keyed by (pol, econ, sname, variant) -> irf."""
    results = {}
    results_ss = {}
    for pol in fiscal:
        for econ, ekw in economies.items():
            for sname, shk in shocks.items():
                for variant in variants:
                    try:
                        ss, irf = run(model, numeraire=numeraire, booking=booking,
                                     model_variant=variant, policy=pol,
                                     **ekw, **shk)
                        results[(pol, econ, sname, variant)] = irf
                        results_ss[(pol, econ, sname, variant)] = ss
                        if verbose:
                            print(f'PASS {pol:14s} {econ:9s} {sname:7s} {variant}')
                    except Exception as e:
                        if verbose:
                            print(f'FAIL {pol:14s} {econ:9s} {sname:7s} {variant}: '
                                  f'{type(e).__name__}: {e}')
    return results, results_ss

def plot_irfs(scenarios, variables=None, len_irf=21, ni=3, nj=2,
              figsize=(7.5, 8.5), legend_loc='best', legend_ax_idx=0,
              save_path=None, ylims=None):
    """Plot IRFs across scenarios in a grid of subplots.

    scenarios : list of (irf_dict, label, style_kwargs).
    variables : list of (key, title, ylabel); defaults to the no-policy 6-var set.
    ylims     : optional {key: (lo, hi)} to force identical vertical scales
                across figures (used to share the y-axis between the baseline
                and fossil-economy fiscal panels).
    """
    from matplotlib.ticker import MaxNLocator, MultipleLocator
    if variables is None:
        variables = [
            ('pi_ann_pp',     r'Inflation $\pi$ (annualized)',        'p.p dev. from SS'),
            ('y_pc',          r'Output $Y$',                          '% dev. from SS'),
            ('CE_G_pc',       r'Green energy consumption $C_{Eg}$',   '% dev. from SS'),
            ('D_GREEN_share', 'Green technology users',               'p.p dev. from SS'),
            ('CE_B_pc',       r'Fossil energy consumption $C_{Eb}$',  '% dev. from SS'),
            ('PEstar_pc',     r'World fossil energy price $P^*_{Eb}$', '% dev. from SS'),
        ]

    titlesize, labelsize, legendsize, axislabelsize = 11, 9, 8, 9
    fig, axes = plt.subplots(ni, nj, figsize=figsize, sharex=True)
    axes = axes.flatten()

    for ax, (var, title, ylabel) in zip(axes, variables):
        for data, label, style in scenarios:
            ax.plot(data[var][:len_irf], label=label, **style)
        ax.axhline(0, color='black', linewidth=1., alpha=0.6, zorder=0)
        ax.set_title(title, fontsize=titlesize)
        ax.tick_params(labelsize=labelsize)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xlabel('quarter', fontsize=axislabelsize)
        ax.set_ylabel(ylabel, fontsize=axislabelsize)
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.tick_params(labelbottom=True)   # show x numbers on every panel, not just the bottom row
        if ylims and var in ylims:
            ax.set_ylim(*ylims[var])

    for ax in axes[len(variables):]:      # blank unused panels (fossil econ has 5 vars)
        ax.axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    axes[legend_ax_idx].legend(handles, labels, loc=legend_loc, fontsize=legendsize,
                               frameon=True, framealpha=0.85)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(_fp(save_path), bbox_inches='tight')
    return fig, axes


def shared_ylims(scenario_lists, variables, len_irf=21, pad=0.06):
    """Common (lo, hi) per variable across several scenario lists, so paired
    figures (baseline vs fossil economy) use identical vertical scales."""
    import numpy as _np
    lims = {}
    for var, _, _ in variables:
        vals = []
        for scen in scenario_lists:
            for data, _, _ in scen:
                if var in data:
                    vals.append(_np.asarray(data[var])[:len_irf])
        if vals:
            allv = _np.concatenate(vals)
            lo, hi = float(_np.nanmin(allv)), float(_np.nanmax(allv))
            lo, hi = min(lo, 0.0), max(hi, 0.0)   # always keep 0 in view
            m = (hi - lo) * pad or 0.1
            lims[var] = (lo - m, hi + m)
    return lims




#%% RUN MODEL 
##################################################################
##################################################################
results, results_ss = run_all(model)
##################################################################
##################################################################

# %%  Store results

# No policy, price shock
irf_base_nopol_price = results[('none', 'baseline', 'price', 'adoption')]
irf_frozen_nopol_price = results[('none', 'baseline', 'price', 'no_adoption')] 
irf_brown_nopol_price = results[('none', 'brown', 'price', 'no_adoption')] 

# No policy, supply shock
irf_base_nopol_supply = results[('none', 'baseline', 'supply', 'adoption')]
irf_frozen_nopol_supply = results[('none', 'baseline', 'supply', 'no_adoption')] 
irf_brown_nopol_supply = results[('none', 'brown', 'supply', 'adoption')] 


# Price subsidy, price shock 
irf_base_subsidy_price = results[('subsidy', 'baseline', 'price', 'adoption')]
irf_frozen_subsidy_price = results[('subsidy', 'baseline', 'price', 'no_adoption')] 
irf_brown_subsidy_price = results[('subsidy', 'brown', 'price', 'no_adoption')] 

# Price subsidy, supply shock 
irf_base_subsidy_supply = results[('subsidy', 'baseline', 'supply', 'adoption')]
irf_frozen_subsidy_supply = results[('subsidy', 'baseline', 'supply', 'no_adoption')] 
irf_brown_subsidy_supply = results[('subsidy', 'brown', 'supply', 'adoption')] 


# Transfer, price shock 
irf_base_transfer_price = results[('transfer', 'baseline', 'price', 'adoption')]
irf_frozen_transfer_price = results[('transfer', 'baseline', 'price', 'no_adoption')] 
irf_brown_transfer_price = results[('transfer', 'brown', 'price', 'no_adoption')] 

#  Transfer, supply shock 
irf_base_transfer_supply = results[('transfer', 'baseline', 'supply', 'adoption')]
irf_frozen_transfer_supply = results[('transfer', 'baseline', 'supply', 'no_adoption')] 
irf_brown_transfer_supply = results[('transfer', 'brown', 'supply', 'adoption')] 

# Untargeted transfer, price shock

irf_base_untargeted_price = results[('transfer_flat', 'baseline', 'price', 'adoption')]
irf_frozen_untargeted_price = results[('transfer_flat', 'baseline', 'price', 'no_adoption')] 
irf_brown_untargeted_price = results[('transfer_flat', 'brown', 'price', 'no_adoption')] 

#  Untageted transfer, supply shock 
irf_base_untargeted_supply = results[('transfer_flat', 'baseline', 'supply', 'adoption')]
irf_frozen_untargeted_supply = results[('transfer_flat', 'baseline', 'supply', 'no_adoption')] 
irf_brown_untargeted_supply = results[('transfer_flat', 'brown', 'supply', 'adoption')] 


# pct series divide by zero in brown case, so use levels instead
irf_brown_nopol_supply['CE_G_pc'] = irf_brown_nopol_supply['CE_G'] 
irf_brown_nopol_price['CE_G_pc'] = irf_brown_nopol_supply['CE_G']
irf_brown_subsidy_supply['CE_G_pc'] = irf_brown_subsidy_supply['CE_G'] 
irf_brown_subsidy_price['CE_G_pc'] = irf_brown_subsidy_supply['CE_G']
irf_brown_transfer_supply['CE_G_pc'] = irf_brown_transfer_supply['CE_G'] 
irf_brown_transfer_price['CE_G_pc'] = irf_brown_transfer_supply['CE_G']
irf_brown_untargeted_supply['CE_G_pc'] = irf_brown_untargeted_supply['CE_G'] 
irf_brown_untargeted_price['CE_G_pc'] = irf_brown_untargeted_supply['CE_G']


VARS_NOPOL = [
    ('pi_ann_pp',     r'Inflation $\pi$ (annualized)',        'p.p dev. from SS'),
    ('y_pc',          r'Output $Y$',                          '% dev. from SS'),
    ('CE_G_pc',       r'Green energy consumption $C_{Eg}$',   '% dev. from SS'),
    ('D_GREEN_share', 'Green technology users',               'p.p dev. from SS'),
    ('CE_B_pc',       r'Fossil energy consumption $C_{Eb}$',  '% dev. from SS'),
    ('PEstar_pc',     r'World fossil energy price $P^*_{Eb}$', '% dev. from SS'),
]

# 6-var layout (baseline, keeps green-user panel) and 5-var layout (fossil econ).
VARS_FP6 = [
    ('y_pc',          r'Output $Y$',                          '% dev. from SS'),
    ('C_pc',          r'Consumption $C$',                     '% dev. from SS'),
    ('pi_ann_pp',     r'Inflation $\pi$ (annualized)',        'p.p dev. from SS'),
    ('B_yss',         'Gov. debt $B$',                        '% of SS output'),
    ('PE_B_pc',       r'Domestic fossil energy price $P_{Eb}$', '% dev. from SS'),
    ('D_GREEN_share', 'Green technology users',               'p.p dev. from SS'),
]
VARS_FP5 = VARS_FP6[:-1]   # fossil economy: adoption is off, drop the green-user panel



#%% Figure 1a
color = 'tab:blue'

scenarios1_price = [
    (irf_base_nopol_price,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_frozen_nopol_price, 'Constant adoption',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_brown_nopol_price,  'Dirty economy',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]

scenarios1_supply = [
    (irf_base_nopol_supply,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_frozen_nopol_supply, 'Constant adoption',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_brown_nopol_supply,  'Dirty economy',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]


ylims_nopol = shared_ylims([scenarios1_price, scenarios1_supply], VARS_NOPOL)
irf_no_policy_price, axes = plot_irfs(scenarios1_price, variables=VARS_NOPOL, ylims=ylims_nopol)
irf_no_policy_price.savefig(_fp('irf_no_policy_price.pdf'))
plt.show()

#%% Figure 1b

irf_no_policy_supply, axes = plot_irfs(scenarios1_supply, variables=VARS_NOPOL, ylims=ylims_nopol)
irf_no_policy_supply.savefig(_fp('irf_no_policy_supply.pdf'))
plt.show()


# %% Figure 2a
scenarios2_price = [
    (irf_base_subsidy_price,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_frozen_subsidy_price, 'Constant adoption',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_brown_subsidy_price,  'Dirty economy',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]

irf_subsidy_price, axes = plot_irfs(scenarios2_price)
irf_subsidy_price.savefig(_fp('irf_subsidy_price.pdf'))
plt.show()


# %% Figure 2b
scenarios2_supply = [
    (irf_base_subsidy_supply,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_frozen_subsidy_supply, 'Constant adoption',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_brown_subsidy_supply,  'Dirty economy',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]

irf_subsidy_supply, axes = plot_irfs(scenarios2_supply)
irf_subsidy_supply.savefig(_fp('irf_subsidy_supply.pdf'))
plt.show()

# %% Figure 3a
scenarios3_price = [
    (irf_base_transfer_price,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_frozen_transfer_price, 'Constant adoption',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_brown_transfer_price,  'Dirty economy',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]

irf_transfer_price, axes = plot_irfs(scenarios3_price)
irf_transfer_price.savefig(_fp('irf_transfer_price.pdf'))
plt.show()

# %% Figure 3b
scenarios3_supply = [
    (irf_base_transfer_supply,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_frozen_transfer_supply, 'Constant adoption',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_brown_transfer_supply,  'Dirty economy',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]

irf_transfer_supply, axes = plot_irfs(scenarios3_supply)
irf_transfer_supply.savefig(_fp('irf_transfer_supply.pdf'))
plt.show()

# %% Fiscal-policy comparison figures (Fig. 4 price / Fig. 5 supply).
# Baseline and fossil-economy panels share a common vertical scale per variable.
color = 'tab:blue'
FISCAL_STYLE = [
    ('No policy',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    ('Energy subsidy',      dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    ('Targeted transfer',   dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
    ('Untargeted transfer', dict(color=color, linestyle='-.', alpha=0.6, linewidth=2.6)),
]



def _fp_scenarios(nopol, subsidy, transfer, untargeted):
    irfs = [nopol, subsidy, transfer, untargeted]
    return [(irf, lab, sty) for irf, (lab, sty) in zip(irfs, FISCAL_STYLE)]


# -------- price shock --------
scenarios_fp_base_price = _fp_scenarios(
    irf_base_nopol_price, irf_base_subsidy_price, irf_base_transfer_price, irf_base_untargeted_price)
scenarios_fp_brown_price = _fp_scenarios(
    irf_brown_nopol_price, irf_brown_subsidy_price, irf_brown_transfer_price, irf_brown_untargeted_price)

ylims_price = shared_ylims([scenarios_fp_base_price, scenarios_fp_brown_price], VARS_FP6)

irf_fp_base_price, _ = plot_irfs(scenarios_fp_base_price, variables=VARS_FP6,
                                 legend_ax_idx=1, ylims=ylims_price, save_path='irf_fp_base_price.pdf')
irf_fp_brown_price, _ = plot_irfs(scenarios_fp_brown_price, variables=VARS_FP5,
                                  legend_ax_idx=1, ylims=ylims_price, save_path='irf_fp_brown_price.pdf')

# -------- supply shock --------
scenarios_fp_base_supply = _fp_scenarios(
    irf_base_nopol_supply, irf_base_subsidy_supply, irf_base_transfer_supply, irf_base_untargeted_supply)
scenarios_fp_brown_supply = _fp_scenarios(
    irf_brown_nopol_supply, irf_brown_subsidy_supply, irf_brown_transfer_supply, irf_brown_untargeted_supply)

ylims_supply = shared_ylims([scenarios_fp_base_supply, scenarios_fp_brown_supply], VARS_FP6)

irf_fp_base_supply, _ = plot_irfs(scenarios_fp_base_supply, variables=VARS_FP6,
                                  legend_ax_idx=1, ylims=ylims_supply, save_path='irf_fp_base_supply.pdf')
irf_fp_brown_supply, _ = plot_irfs(scenarios_fp_brown_supply, variables=VARS_FP5,
                                   legend_ax_idx=1, ylims=ylims_supply, save_path='irf_fp_brown_supply.pdf')
#%%
