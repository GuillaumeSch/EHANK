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
TAU_B = 0.10
VARSET = 'restricted'                 # 'restricted' (VARS_NOPOL, plot_irfs) or 'extended' (OUTPUTS, plot_grid)
SAVE = True
RESULTS_PATH = 'bve_results.pkl'
RESULTS_SS_PATH = 'bve_results_ss.pkl'

FIGDIR = 'figures'
os.makedirs(FIGDIR, exist_ok=True)
def _fp(name):
    return os.path.join(FIGDIR, name)

ECONOMIES = {                     # colour
    'baseline': dict(),
    'ETS':      dict(ets=True, ets_kwargs=dict(tau_b=TAU_B, recycle='rebate')),
}
SHOCKS = {
    'price':  dict(shock_kind='price'),
    'supply': dict(shock_kind='supply'),
}
VARIANTS = [
    'adoption',
    ]

FISCAL = [
    'none',
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
        fig.savefig(_fp(savepath or f'irf_{pol}.pdf'))
    return fig


def plot_irfs(scenarios, variables=None, len_irf=21, ni=3, nj=2,
              figsize=(7.5, 8.5), legend_loc='best', legend_ax_idx=0,
              save_path=None, ylims=None):
    """Plot IRFs across scenarios in a grid of subplots.

    scenarios : list of (irf_dict, label, style_kwargs).
    variables : list of (key, title, ylabel); defaults to the no-policy 6-var set.
    ylims     : optional {key: (lo, hi)} to force identical vertical scales.
    """
    from matplotlib.ticker import MaxNLocator, MultipleLocator
    if variables is None:
        variables = VARS_NOPOL

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
        ax.tick_params(labelbottom=True)
        if ylims and var in ylims:
            ax.set_ylim(*ylims[var])

    for ax in axes[len(variables):]:
        ax.axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    axes[legend_ax_idx].legend(handles, labels, loc=legend_loc, fontsize=legendsize,
                               frameon=True, framealpha=0.85)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(_fp(save_path), bbox_inches='tight')
    return fig, axes


def shared_ylims(scenario_lists, variables, len_irf=21, pad=0.06):
    """Common (lo, hi) per variable across several scenario lists."""
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
            lo, hi = min(lo, 0.0), max(hi, 0.0)
            m = (hi - lo) * pad or 0.1
            lims[var] = (lo - m, hi + m)
    return lims


VARS_NOPOL = [
    ('pi_ann_pp',     r'Inflation $\pi$ (annualized)',        'p.p dev. from SS'),
    ('y_pc',          r'Output $Y$',                          '% dev. from SS'),
    ('CE_G_pc',       r'Green energy consumption $C_{Eg}$',   '% dev. from SS'),
    ('D_GREEN_share', 'Green technology users',               'p.p dev. from SS'),
    ('CE_B_pc',       r'Fossil energy consumption $C_{Eb}$',  '% dev. from SS'),
    ('PEstar_pc',     r'World fossil energy price $P^*_{Eb}$', '% dev. from SS'),
]

ECON_STYLE = {
    'baseline': dict(color=COLORS[0], linestyle='-',  linewidth=2.6),
    'ETS':      dict(color=COLORS[2], linestyle='--', linewidth=2.6),
}


def econ_label(econ, sname):
    dg = 100 * float(np.asarray(results_ss[('none', econ, sname, 'adoption')]['D_GREEN']))
    if econ == 'baseline':
        return rf'baseline ($D^G_{{ss}}$={dg:.1f}%)'
    return rf'ETS $\tau_b$={TAU_B} ($D^G_{{ss}}$={dg:.1f}%)'


def scenarios_for(sname):
    return [(results[('none', econ, sname, 'adoption')], econ_label(econ, sname),
             ECON_STYLE[econ]) for econ in ECONOMIES]


#%%
if os.path.exists(RESULTS_PATH) and os.path.exists(RESULTS_SS_PATH):
    results, results_ss = load_results(), load_results(path=RESULTS_SS_PATH)
else:
    results, results_ss = run_all(model)
if SAVE and not (os.path.exists(RESULTS_PATH) and os.path.exists(RESULTS_SS_PATH)):
    save_results(results)
    save_results(results_ss, path=RESULTS_SS_PATH)


#%%
if VARSET == 'restricted':
    ylims = shared_ylims([scenarios_for(s) for s in SHOCKS], VARS_NOPOL)
    for sname in SHOCKS:
        fig, _ = plot_irfs(scenarios_for(sname), variables=VARS_NOPOL, ylims=ylims,
                           save_path=f'irf_baseline_vs_ets_{sname}_restricted.pdf')
        plt.show()
else:
    for sname in SHOCKS:
        plot_grid(results, 'none', shocks=[sname],
                  savepath=f'irf_baseline_vs_ets_{sname}_extended.pdf')
        plt.show()
