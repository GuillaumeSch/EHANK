#%%
import sys; sys.path.insert(0, '.')
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

ECONOMIES = {                     # colour
    'baseline': dict(),
    #'ETS':      dict(ets=True, ets_kwargs=dict(tau_b=0.10, recycle='rebate')),
    'brown':    dict(green_block=20.0),
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
    'transfer_flat'
    ]

VAR_LS     = {'adoption': '-', 'no_adoption': '--'}   # solid = adoption on, dashed = adoption off
SHOCK_LW   = {'price': 2.0, 'supply': 1.1}            # used only when economies and shocks both vary
DEFAULT_LW = 2.0
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
ECON_LIST = list(ECONOMIES)

OUTPUTS = {
    'y':       r'Output $y$',
    'C':       r'Consumption $C$',
    'D_GREEN': r'Green share $D^G$',
    'CHF_SWITCH_exp': r'Adoption expenditures',
    'pi_ann':  r'Inflation (ann.)',
    'piw_ann':  r'Wage inflation (ann.)',
    'w': r'Real wage',
    'PEstar': r'Market price of brown energy (in USD) $P^*_{Eb}$',
    # 'E_supply': r'Energy supply',
    'E_supply_shock': r'Supply shock, exog. ($E^{sup}_{shock}$)',
    'pE_B_P':  r'Brown price $P^E_B/P$',
    'CE_B': r'Brown energy consumption ($C_{Eb}$)',
    'CE_G': r'Green energy consumption ($C_{Eg}$)',
    'nx_gdp':  r'Net exports / GDP',
    'exports': r'Exports (level)',
    'imports': r'Imports (level)',
    'nfa': r'NFA',
    'pB_P': r'Rel. price of cons. basket, brown users ($p^B$)',
    'pG_P': r'Rel. price of cons. basket, green users ($p^G$)',
    'C_BROWN': r'Total cons., brown users',
    'C_BROWN_PC': r'Per capita cons., brown users',
    'C_GREEN': r'Total cons., green users',
    'C_GREEN_PC': r'Per capita cons., green users',
    'LAB_INC_GREEN': r'Avg. labour income (green users)',
    'LAB_INC_BROWN': r'Avg. labour income (brown users)',
    'r': r'Real int. rate ($r$)',
    }

model = build_model(NUM, booking=BOOK)


#%%  recompute the matched supply-shock path (run after changing the model)
# matched_supply_path(model, numeraire=NUM, booking=BOOK, recompute=True)


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


def save_results(results, path=RESULTS_PATH):
    with open(path, 'wb') as f:
        pickle.dump(results, f)


def load_results(path=RESULTS_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


#%%
def legend_handles(econ_list, shocks, variants, present):
    """One entry per plotted line: exact colour/linestyle/width, label from the
    dimensions that vary."""
    multi_e, multi_s, multi_v = len(econ_list) > 1, len(shocks) > 1, len(variants) > 1
    multi_es = multi_e and multi_s
    shk_list = list(shocks)
    h = []
    for ci, econ in enumerate(econ_list):
        for sname in shk_list:
            for variant in variants:
                if (econ, sname, variant) not in present:
                    continue
                if multi_e:
                    color = COLORS[ci % len(COLORS)]
                elif multi_s:
                    color = COLORS[shk_list.index(sname) % len(COLORS)]
                else:
                    color = COLORS[0]
                lw = SHOCK_LW[sname] if multi_es else DEFAULT_LW
                label = ', '.join(x for x, flag in
                                  [(econ, multi_e), (sname, multi_s), (variant, multi_v)]
                                  if flag) or econ
                h.append(Line2D([], [], color=color, ls=VAR_LS[variant], lw=lw, label=label))
    return h


def plot_grid(results, pol, outputs=OUTPUTS, H=H, economies=None, shocks=None,
              variants=None, ncol=4, save=SAVE, savepath=None):
    """Grid of IRFs for one fiscal policy, read from a stored results dict."""
    econ_list = list(economies) if economies is not None else ECON_LIST
    shk_list  = list(shocks)    if shocks    is not None else list(SHOCKS)
    var_list  = list(variants)  if variants  is not None else list(VARIANTS)
    multi_e, multi_s = len(econ_list) > 1, len(shk_list) > 1
    shock_idx = {s: i for i, s in enumerate(shk_list)}
    present = {(e, s, v) for e in econ_list for s in shk_list for v in var_list
               if results.get((pol, e, s, v)) is not None}

    nrow = int(np.ceil(len(outputs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, (k, lab) in zip(axes.flat, outputs.items()):
        for ci, econ in enumerate(econ_list):
            for sname in shk_list:
                for variant in var_list:
                    irf = results.get((pol, econ, sname, variant))
                    if irf is None:
                        continue
                    y = np.asarray(irf[k])[:H] if k in irf else np.zeros(H)
                    if multi_e:
                        color = COLORS[ci % len(COLORS)]
                    elif multi_s:
                        color = COLORS[shock_idx[sname] % len(COLORS)]
                    else:
                        color = COLORS[0]
                    lw = SHOCK_LW[sname] if (multi_e and multi_s) else DEFAULT_LW
                    ax.plot(100 * y, color=color, ls=VAR_LS[variant], lw=lw)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel('quarters', fontsize=8)
    for ax in list(axes.flat)[len(outputs):]:
        ax.axis('off')
    axes.flat[0].legend(handles=legend_handles(econ_list, shk_list, var_list, present), fontsize=8)
    fig.suptitle(pol, fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(savepath or f'irf_{pol}.pdf')
    return fig


#%%
results, results_ss = run_all(model)
if SAVE:
    save_results(results)
    save_results(results_ss, path=RESULTS_SS_PATH)


#%%
# results = load_results()
# results_ss = load_results(path=RESULTS_SS_PATH))
for pol in FISCAL:
    plot_grid(results, pol)
plt.show()

# %%
ECONOMIES = {                     # colour
    'baseline': dict(),
    #'ETS':      dict(ets=True, ets_kwargs=dict(tau_b=0.10, recycle='rebate')),
    'brown':    dict(green_block=20.0),
}
SHOCKS = {
    'price':  dict(shock_kind='price'),
    # 'supply': dict(shock_kind='supply'),
}
VARIANTS = [
    'adoption',
    'no_adoption'
    ]

FISCAL = [
    'none',
    'subsidy',
    #'transfer',
    #'transfer_flat'
    ]

plot_grid(results, 'none')
plt.show()


#%% Steady state results
Pswitch_ss = results_ss[('none', 'baseline', 'price', 'adoption')].internals['hh_0']['durables']['law_of_motion'].P

a_grid = results_ss[('none', 'baseline', 'price', 'adoption')].internals['hh_1']['a_grid']

fig, ax = plt.subplots()

ax.plot(a_grid, Pswitch_ss[2,0,0,:].T*100,
        color='#08519c', linestyle='-', linewidth=2.6, label='Low productivity')
ax.plot(a_grid, Pswitch_ss[2,0,3,:].T*100,
        color='#4292c6', linestyle='--', linewidth=2.6, label='Medium productivity')
ax.plot(a_grid, Pswitch_ss[2,0,-1,:].T*100,
        color='#9ecae1', linestyle=':', linewidth=2.6, label='High productivity')

ax.set_xlabel('Individual assets')
ax.set_ylabel('Probability of adjusting (%)')
ax.set_title('Clean adoption probability')
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

ax.legend()

plt.show()
fig.savefig( f'ss_adj_probs.pdf')

# dir(results_ss[('none', 'baseline', 'price', 'adoption')])
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


#%% Plot helper
def plot_irfs(scenarios, variables=None, len_irf=21, ni=3, nj=2,
              figsize=(7.5, 8.5), legend_loc='upper right', legend_ax_idx=0,
              save_path=None):
    """
    Plot IRFs across scenarios in a grid of subplots.

    Parameters
    ----------
    scenarios : list of (dict, str, dict)
        Each tuple is (irf_dict, label, plot_style_kwargs), e.g.
        (irf_base_nopol_price, 'Baseline', dict(linestyle='-', alpha=1.0, linewidth=1.6))
    variables : list of (str, str, str)
        Each tuple is (key, title, ylabel). Defaults to the standard 6-variable set.
    len_irf : int
        Number of periods to plot.
    ni, nj : int
        Grid dimensions.
    figsize : tuple
        Figure size in inches.
    legend_loc : str
        Legend location within the chosen subplot.
    legend_ax_idx : int
        Index (into flattened axes) of the subplot that holds the legend.
    save_path : str or None
        If given, saves the figure to this path (e.g. 'irf_comparison.pdf').

    Returns
    -------
    fig, axes
    """
    if variables is None:
        variables = [
            ('pi_ann_pp',   'Inflation $\pi$ (annualised)',            'p.p dev. from SS'),
            ('y_pc',        'Output $Y$',                              '% dev. from SS'),
            ('CE_G_pc',     'Green energy consumption $C_{Eg}$',       '% dev. from SS'),
            ('D_GREEN_share', 'Green technology users',                'p.p dev, from SS'),
            ('CE_B_pc',     'Brown energy consumption $C_{Eb}$',       '% dev. from SS'),
            ('PEstar_pc',   r'Brown energy price $P^*_{Eb}$',          '% dev. from SS'),
        ]

    titlesize = 11
    labelsize = 9
    legendsize = 8
    axislabelsize = 9

    fig, axes = plt.subplots(ni, nj, figsize=figsize, sharex=True)
    axes = axes.flatten()

    for ax, (var, title, ylabel) in zip(axes, variables):
        for data, label, style in scenarios:
            ax.plot(data[var][:len_irf], label=label, **style)
        ax.axhline(0, color='black', linewidth=1., alpha=0.6, zorder=0)
        ax.set_title(title, fontsize=titlesize)
        ax.tick_params(labelsize=labelsize)
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.set_xlabel('quarter', fontsize=axislabelsize)
        ax.set_ylabel(ylabel, fontsize=axislabelsize)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[legend_ax_idx].legend(
        handles, labels,
        loc=legend_loc,
        fontsize=legendsize,
        frameon=True,
        framealpha=0.8,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

    return fig, axes



#%% Figure 1a
color = 'tab:blue'

scenarios1_price = [
    (irf_base_nopol_price,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_frozen_nopol_price, 'Constant adoption',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_brown_nopol_price,  'Brown',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]

irf_no_policy_price, axes = plot_irfs(scenarios1_price)
irf_no_policy_price.savefig( f'irf_no_policy_price.pdf')
plt.show()

#%% Figure 1b 

scenarios1_supply = [
    (irf_base_nopol_supply,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_frozen_nopol_supply, 'Constant adoption',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_brown_nopol_supply,  'Brown',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]

irf_no_policy_supply, axes = plot_irfs(scenarios1_supply)
irf_no_policy_supply.savefig( f'irf_no_policy_supply.pdf')
plt.show()


# %% Figure 2a
scenarios2_price = [
    (irf_base_subsidy_price,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_frozen_subsidy_price, 'Constant adoption',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_brown_subsidy_price,  'Brown',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]

irf_subsidy_price, axes = plot_irfs(scenarios2_price)
irf_subsidy_price.savefig( f'irf_subsidy_price.pdf')
plt.show()


# %% Figure 2b
scenarios2_supply = [
    (irf_base_subsidy_supply,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_frozen_subsidy_supply, 'Constant adoption',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_brown_subsidy_supply,  'Brown',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]

irf_subsidy_supply, axes = plot_irfs(scenarios2_supply)
irf_subsidy_supply.savefig( f'irf_subsidy_supply.pdf')
plt.show()

# %% Figure 3a
scenarios3_price = [
    (irf_base_transfer_price,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_frozen_transfer_price, 'Constant adoption',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_brown_transfer_price,  'Brown',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]

irf_transfer_price, axes = plot_irfs(scenarios3_price)
irf_transfer_price.savefig( f'irf_transfer_price.pdf')
plt.show()

# %% Figure 3b
scenarios3_supply = [
    (irf_base_transfer_supply,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_frozen_transfer_supply, 'Constant adoption',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_brown_transfer_supply,  'Brown',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]

irf_transfer_supply, axes = plot_irfs(scenarios3_supply)
irf_transfer_supply.savefig( f'irf_transfer_supply.pdf')
plt.show()

# %% Comparison of fiscal policies -- Price shock -- Adoption
color = 'tab:blue'

scenarios_fp_base_price = [
    (irf_base_nopol_price,   'No policy',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_base_subsidy_price, 'Energy subsidy',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_base_transfer_price,  'Targeted transfer',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
    (irf_base_untargeted_price,  'Untargeted transfer',              dict(color=color, linestyle='-.',  alpha=0.6, linewidth=2.6)),
]

vars_ = [  ('y_pc',        'Output $Y$',                              '% dev.  from SS'),
            ('C_pc',        'Consumption $C$',                              '% dev.  from SS'),
            ('pi_ann_pp',   'Inflation $\pi$ (annualised)',            'p.p dev. from SS'),
            ('B_yss', 'Gov. debt',                '% of SS output'),
            ('PE_B_pc',     'Brown energy price $P_{Eb}$',       '% dev. from SS'),
            ('D_GREEN_share', 'Green technology users',                'p.p dev, from SS'),
        ]

irf_fp_base_price, axes = plot_irfs(scenarios_fp_base_price, variables=vars_, legend_ax_idx=1)
irf_fp_base_price.savefig( f'irf_fp_base_price.pdf')

# %% Comparison of fiscal policies -- Price shock --  Brown economy 
scenarios_fp_brown_price = [
    (irf_brown_nopol_price,   'No policy',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_brown_subsidy_price, 'Energy subsidy',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_brown_transfer_price,  'Targeted transfer',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
    (irf_brown_untargeted_price,  'Untargeted transfer',              dict(color=color, linestyle='-.',  alpha=0.6, linewidth=2.6)),
]

vars_ = [  ('y_pc',        'Output $Y$',                              '% dev.  from SS'),
            ('C_pc',        'Consumption $C$',                              '% dev.  from SS'),
            ('pi_ann_pp',   'Inflation $\pi$ (annualised)',            'p.p dev. from SS'),
            ('B_yss', 'Gov. debt',                '% of SS output'),
            ('PE_B_pc',     'Brown energy price $P_{Eb}$',       '% dev. from SS'),
        ]

irf_fp_brown_price, axes = plot_irfs(scenarios_fp_brown_price, variables=vars_, legend_ax_idx=1)
irf_fp_brown_price.savefig( f'irf_fp_brown_price.pdf')

# %% Comparison of fiscal policies -- Supply shock -- Adoption
scenarios_fp_base_supply = [
    (irf_base_nopol_supply,   'No policy',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_base_subsidy_supply, 'Energy subsidy',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_base_transfer_supply,  'Targeted transfer',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
    (irf_base_untargeted_supply,  'Untargeted transfer',              dict(color=color, linestyle='-.',  alpha=0.6, linewidth=2.6)),
]

vars_ = [  ('y_pc',        'Output $Y$',                              '% dev.  from SS'),
            ('C_pc',        'Consumption $C$',                              '% dev.  from SS'),
            ('pi_ann_pp',   'Inflation $\pi$ (annualised)',            'p.p dev. from SS'),
            ('B_yss', 'Gov. debt',                '% of SS output'),
            ('PE_B_pc',     'Brown energy price $P_{Eb}$',       '% dev. from SS'),
            ('D_GREEN_share', 'Green technology users',                'p.p dev, from SS'),
        ]

irf_fp_base_supply, axes = plot_irfs(scenarios_fp_base_supply, variables=vars_, legend_ax_idx=1)
irf_fp_base_supply.savefig( f'irf_fp_base_supply.pdf')

# %% Comparison of fiscal policies -- Supply shock --  Brown economy 
scenarios_fp_brown_supply = [
    (irf_brown_nopol_supply,   'No policy',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_brown_subsidy_supply, 'Energy subsidy',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_brown_transfer_supply,  'Targeted transfer',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
    (irf_brown_untargeted_supply,  'Untargeted transfer',              dict(color=color, linestyle='-.',  alpha=0.6, linewidth=2.6)),
]

vars_ = [  ('y_pc',        'Output $Y$',                              '% dev.  from SS'),
            ('C_pc',        'Consumption $C$',                              '% dev.  from SS'),
            ('pi_ann_pp',   'Inflation $\pi$ (annualised)',            'p.p dev. from SS'),
            ('B_yss', 'Gov. debt',                '% of SS output'),
            ('PE_B_pc',     'Brown energy price $P_{Eb}$',       '% dev. from SS'),
        ]

irf_fp_brown_supply, axes = plot_irfs(scenarios_fp_brown_supply, variables=vars_, legend_ax_idx=1)
irf_fp_brown_supply.savefig( f'irf_fp_brown_supply.pdf')
# %%
