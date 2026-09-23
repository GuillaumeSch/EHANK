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
    'ETS':      dict(ets=True, ets_kwargs=dict(tau_b=0.10, recycle='rebate')),
    'ETS_green_subsidy':      dict(ets=True, ets_kwargs=dict(tau_b=0.10, recycle='green_subsidy')),
    'ETS_green_subsidy_v2':  dict(ets=True, ets_kwargs=dict(tau_b=0.10, recycle='green_subsidy_dynamics')),
    #'brown':    dict(green_block=20.0),
    'baseline_high_pe': dict(PEstar=1.1, PEstar_shock=1.1),
}
SHOCKS = {
    'price':  dict(shock_kind='price'),
    'supply': dict(shock_kind='supply'),
}
VARIANTS = [
    'adoption',
    #'no_adoption'
    ]

FISCAL = [
    'none',
    #'subsidy',
    #'transfer',
    #'transfer_flat',
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


# Plot helper
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


#%% Run models 
results, results_ss = run_all(model)

# %% ETS policies 

ss_ets_rebated = results_ss[('none', 'ETS', 'price', 'adoption')]
ss_ets_subsidy = results_ss[('none', 'ETS_green_subsidy', 'price', 'adoption')]
ss_ets_subsidy2 = results_ss[('none', 'ETS_green_subsidy_v2', 'price', 'adoption')]
ss_baseline = results_ss[('none', 'baseline', 'price', 'adoption')]
ss_high_pe = results_ss[('none', 'baseline_high_pe', 'price', 'adoption')]

irf_ets_rebated_price = results[('none', 'ETS', 'price', 'adoption')] 
irf_ets_subsidy_price = results[('none', 'ETS_green_subsidy', 'price', 'adoption')] 
irf_ets_subsidy2_price = results[('none', 'ETS_green_subsidy_v2', 'price', 'adoption')] 
irf_baseline_price = results[('none', 'baseline', 'price', 'adoption')] 
irf_high_pe_price = results[('none', 'baseline_high_pe', 'price', 'adoption')] 

irf_ets_rebated_supply = results[('none', 'ETS', 'supply', 'adoption')] 
irf_ets_subsidy_supply = results[('none', 'ETS_green_subsidy', 'supply', 'adoption')] 
irf_ets_subsidy2_supply = results[('none', 'ETS_green_subsidy_v2', 'supply', 'adoption')] 
irf_baseline_supply = results[('none', 'baseline', 'supply', 'adoption')] 
irf_high_pe_supply = results[('none', 'baseline_high_pe', 'supply', 'adoption')] 

#%% Add new variables

irf_ets_rebated_price['D_GREEN_share_level'] = irf_ets_rebated_price['D_GREEN_share'] + ss_ets_rebated['D_GREEN_share']
irf_ets_rebated_price['CE_G_level'] = irf_ets_rebated_price['CE_G'] + ss_ets_rebated['CE_G']
irf_ets_rebated_price['CE_B_level'] = irf_ets_rebated_price['CE_B'] + ss_ets_rebated['CE_B']

irf_ets_subsidy_price['D_GREEN_share_level'] = irf_ets_subsidy_price['D_GREEN_share'] + ss_ets_subsidy['D_GREEN_share']
irf_ets_subsidy_price['CE_G_level'] = irf_ets_subsidy_price['CE_G'] + ss_ets_subsidy['CE_G']
irf_ets_subsidy_price['CE_B_level'] = irf_ets_subsidy_price['CE_B'] + ss_ets_subsidy['CE_B']

irf_ets_subsidy2_price['D_GREEN_share_level'] = irf_ets_subsidy2_price['D_GREEN_share'] + ss_ets_subsidy2['D_GREEN_share']
irf_ets_subsidy2_price['CE_G_level'] = irf_ets_subsidy2_price['CE_G'] + ss_ets_subsidy2['CE_G']
irf_ets_subsidy2_price['CE_B_level'] = irf_ets_subsidy2_price['CE_B'] + ss_ets_subsidy2['CE_B']

irf_baseline_price['D_GREEN_share_level'] = irf_baseline_price['D_GREEN_share'] + ss_baseline['D_GREEN_share']
irf_baseline_price['CE_G_level'] = irf_baseline_price['CE_G'] + ss_baseline['CE_G']
irf_baseline_price['CE_B_level'] = irf_baseline_price['CE_B'] + ss_baseline['CE_B']

irf_high_pe_price['D_GREEN_share_level'] = irf_high_pe_price['D_GREEN_share'] + ss_high_pe['D_GREEN_share']
irf_high_pe_price['CE_G_level'] = irf_high_pe_price['CE_G'] + ss_high_pe['CE_G']
irf_high_pe_price['CE_B_level'] = irf_high_pe_price['CE_B'] + ss_high_pe['CE_B']


irf_baseline_price['R_carbon_level'] = (irf_baseline_price['R_carbon'] + ss_baseline['R_carbon']) / ss_baseline['y'] *100 
irf_high_pe_price['R_carbon_level'] = (irf_baseline_price['R_carbon'] + ss_baseline['R_carbon']) / ss_high_pe['y'] *100 

irf_ets_rebated_price['R_carbon_level'] = (irf_ets_rebated_price['R_carbon'] + ss_ets_rebated['R_carbon']) / ss_ets_rebated['y'] *100 
irf_ets_subsidy_price['R_carbon_level'] = (irf_ets_subsidy_price['R_carbon'] + ss_ets_subsidy['R_carbon'])/ ss_ets_subsidy['y'] *100 
irf_ets_subsidy2_price['R_carbon_level'] = (irf_ets_subsidy2_price['R_carbon'] + ss_ets_subsidy2['R_carbon'])/ ss_ets_subsidy2['y'] *100 



irf_baseline_price['s_g_level']= irf_baseline_price['R_carbon']*0 + ss_baseline['s_g']
irf_high_pe_price['s_g_level']= irf_high_pe_price['R_carbon']*0 + ss_high_pe['s_g']

irf_ets_rebated_price['s_g_level']= irf_ets_rebated_price['R_carbon']*0 + ss_ets_rebated['s_g']
irf_ets_subsidy_price['s_g_level'] = irf_ets_subsidy_price['s_g'] + ss_ets_subsidy['s_g']
irf_ets_subsidy2_price['s_g_level'] = irf_ets_subsidy2_price['s_g'] + ss_ets_subsidy2['s_g']


irf_baseline_price['D_SWITCH_level']= irf_baseline_price['D_SWITCH_share'] + ss_baseline['D_SWITCH_share']
irf_high_pe_price['D_SWITCH_level']= irf_high_pe_price['D_SWITCH_share'] + ss_high_pe['D_SWITCH_share']

irf_ets_rebated_price['D_SWITCH_level']= irf_ets_rebated_price['D_SWITCH_share'] + ss_ets_rebated['D_SWITCH_share']
irf_ets_subsidy_price['D_SWITCH_level'] = irf_ets_subsidy_price['D_SWITCH_share'] + ss_ets_subsidy['D_SWITCH_share']
irf_ets_subsidy2_price['D_SWITCH_level'] = irf_ets_subsidy2_price['D_SWITCH_share'] + ss_ets_subsidy2['D_SWITCH_share']



# vars for plots
VARS_NOPOL = [
    ('pi_ann_pp',     r'Inflation $\pi$ (annualized)',        'p.p dev. from SS'),
    ('y_pc',          r'Output $Y$',                          '% dev. from SS'),
    ('CE_G_pc',       r'Green energy consumption $C_{Eg}$',   '% dev. from SS'),
    ('D_GREEN_share', 'Green technology users',               'p.p dev. from SS'),
    ('CE_B_pc',       r'Fossil energy consumption $C_{Eb}$',  '% dev. from SS'),
    ('PEstar_pc',     r'World fossil energy price $P^*_{Eb}$', '% dev. from SS'),
]

VARS_ETS = [
    ('pi_ann_pp',     r'Inflation $\pi$ (annualized)',        'p.p dev. from SS'),
    ('y_pc',          r'Output $Y$',                          '% dev. from SS'),
    ('CE_G_level',       r'Green energy consumption $C_{Eg}$',   'level'),
    ('CE_B_level',       r'Fossil energy consumption $C_{Eb}$',  'level'),
    ('D_GREEN_share_level', 'Green technology users',               'share of households (%)'),
    ('D_SWITCH_level', 'New adopters',               'share of households (%)'),
    ('R_carbon_level',          r'Cabon tax revenues',                          '% of SS. output'),
    ('PEstar_pc',     r'World fossil energy price $P^*_{Eb}$', '% dev. from SS'),
]

VARS_NOPOL_LEVEL = [
    ('pi_ann_pp',     r'Inflation $\pi$ (annualized)',        'p.p dev. from SS'),
    ('y_pc',          r'Output $Y$',                          '% dev. from SS'),
    ('CE_G_level',       r'Green energy consumption $C_{Eg}$',   'level'),
    ('D_GREEN_share_level', 'Green technology users',               'share of households (%)'),
    ('CE_B_level',       r'Fossil energy consumption $C_{Eb}$',  'level'),
    ('PEstar_pc',     r'World fossil energy price $P^*_{Eb}$', '% dev. from SS'),
]

VARS_C_PER_CAP = [
    ('C_BROWN_PC_pc',     r'Fossil energy users',        r'Per-capita consumption, % dev. from SS'),
    ('C_GREEN_PC_pc', r'Green energy users',              r'Per-capita consumption, % dev. from SS'),
]

#%% Figure: ETS (with own SS)
color = 'tab:blue'

scenarios1_ets_price = [
    (irf_baseline_price,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_ets_rebated_price, 'ETS, lump-sum recycling',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_ets_subsidy_price,  'ETS, green-subsidy recycling',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]

scenarios1_ets_supply = [
    (irf_baseline_supply,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_ets_rebated_supply, 'ETS, lump-sum recycling',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_ets_subsidy_supply,  'ETS, green-subsidy recycling',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]

# Price shock 
ylims = shared_ylims([scenarios1_ets_price], VARS_NOPOL_LEVEL)
irf_ets_price_01, axes = plot_irfs(scenarios1_ets_price, variables=VARS_NOPOL_LEVEL, ylims=ylims)
irf_ets_price_01.savefig(_fp('irf_ets_price_01.pdf'))
plt.show()

# Price shock full set of vers
ylims = shared_ylims([scenarios1_ets_price], VARS_ETS)
irf_ets_price_new_01, axes = plot_irfs(scenarios1_ets_price, variables=VARS_ETS, ylims=ylims, ni=4)
irf_ets_price_new_01.savefig(_fp('irf_ets_price_new_01.pdf'))
plt.show()

# Supply shock 
ylims = shared_ylims([scenarios1_ets_supply], VARS_NOPOL)
irf_ets_supply_01, axes = plot_irfs(scenarios1_ets_supply, variables=VARS_NOPOL, ylims=ylims)
irf_ets_supply_01.savefig(_fp('irf_ets_supply_01.pdf'))
plt.show()


# %% Figure: ETS with similar SS 
color = 'tab:blue'

scenarios2_ets_price = [
    (irf_high_pe_price,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_ets_rebated_price, 'ETS, lump-sum recycling',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_ets_subsidy2_price,  'ETS, green-subsidy recycling',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]

scenarios2_ets_supply = [
    (irf_high_pe_supply,   'Baseline',           dict(color=color, linestyle='-',  alpha=1.0, linewidth=2.6)),
    (irf_ets_rebated_supply, 'ETS, lump-sum recycling',  dict(color=color, linestyle='--', alpha=0.9, linewidth=2.6)),
    (irf_ets_subsidy2_supply,  'ETS, green-subsidy recycling',              dict(color=color, linestyle=':',  alpha=0.8, linewidth=2.6)),
]

ylims = shared_ylims([scenarios2_ets_price], VARS_NOPOL_LEVEL)

irf_ets_price_02, axes = plot_irfs(scenarios2_ets_price, variables=VARS_NOPOL_LEVEL, ylims=ylims)
irf_ets_price_02.savefig(_fp('irf_ets_price_02.pdf'))
plt.show()

# Price shock full set of vers
ylims = shared_ylims([scenarios2_ets_price], VARS_ETS)
irf_ets_price_new_02, axes = plot_irfs(scenarios2_ets_price, variables=VARS_ETS, ylims=ylims, ni=4)
irf_ets_price_new_02.savefig(_fp('irf_ets_price_new_02.pdf'))
plt.show()


irf_ets_supply_02, axes = plot_irfs(scenarios2_ets_supply, variables=VARS_NOPOL, ylims=ylims)
irf_ets_supply_02.savefig(_fp('irf_ets_supply_02.pdf'))
plt.show()



#%%
ylims = shared_ylims([scenarios1_ets_price], VARS_C_PER_CAP)

irf_ets_cons_01, axes = plot_irfs(scenarios1_ets_price, variables=VARS_C_PER_CAP, ylims=ylims, ni=1, nj=2, figsize=(11, 4.4))
irf_ets_cons_01.tight_layout()
irf_ets_cons_01.savefig(_fp('irf_ets_cons_01.pdf'), bbox_inches='tight')
plt.show()

ylims = shared_ylims([scenarios2_ets_price], VARS_C_PER_CAP)
irf_ets_cons_02, axes = plot_irfs(scenarios2_ets_price, variables=VARS_C_PER_CAP, ylims=ylims, ni=1, nj=2, figsize=(11, 4.4))
irf_ets_cons_02.tight_layout()
irf_ets_cons_02.savefig(_fp('irf_ets_cons_02.pdf'), bbox_inches='tight')
plt.show()
# %%
ni=1; nj=2; TT=21
plt.figure()
plt.subplot(ni, nj, 1)
plt.plot(irf_ets_rebated_price['R_carbon'][:TT] + ss_ets_rebated['R_carbon'], label="Lump-sum rebate")
plt.plot(irf_ets_subsidy_price['R_carbon'][:TT] + ss_ets_subsidy['R_carbon'], label="Green subsidy" )
plt.plot(irf_ets_subsidy2_price['R_carbon'][:TT] + ss_ets_subsidy2['R_carbon'], label="Green subsidy (Rebate in SS)" )
plt.title("Carbon tax revenues")
plt.ylabel("Tax revenues (level)")

plt.subplot(ni, nj, 2)
plt.plot(irf_ets_rebated_price['R_carbon'][:TT]*0 + ss_ets_rebated['s_g'], label="Lump-sum rebate")
plt.plot(irf_ets_subsidy_price['s_g'][:TT] + ss_ets_subsidy['s_g'], label="Green subsidy" )
plt.plot(irf_ets_subsidy2_price['s_g'][:TT] + ss_ets_subsidy2['s_g'], label="Green subsidy (Rebate in SS)" )
plt.title("Subsidy rate")
plt.ylabel("$s_g$")
plt.legend(loc="upper right")

plt.show()

# %%
