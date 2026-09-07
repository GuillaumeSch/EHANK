#%%
import sys; sys.path.insert(0, '.')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from core.model import build_model, run, matched_supply_path


#%%
NUM, BOOK, H = 'cpi', 'import', 24
SAVE = False
RESULTS_PATH = 'explore_results.pkl'

ECONOMIES = {                     # colour
    'baseline': dict(),
    #'ETS':      dict(ets=True, ets_kwargs=dict(tau_b=0.10, recycle='rebate')),
    #'brown':    dict(green_block=20.0),
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
    #'subsidy',
    #'transfer',
    #'transfer_flat'
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
    'E_supply': r'Energy supply',
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
#matched_supply_path(model, numeraire=NUM, booking=BOOK, recompute=True)


#%%
def run_all(model, economies=ECONOMIES, shocks=SHOCKS, variants=VARIANTS,
            fiscal=FISCAL, numeraire=NUM, booking=BOOK, verbose=True):
    """Solve every (fiscal, economy, shock, variant) cell; return a results dict
    keyed by (pol, econ, sname, variant) -> irf."""
    results = {}
    for pol in fiscal:
        for econ, ekw in economies.items():
            for sname, shk in shocks.items():
                for variant in variants:
                    try:
                        _, irf = run(model, numeraire=numeraire, booking=booking,
                                     model_variant=variant, policy=pol,
                                     **ekw, **shk)
                        results[(pol, econ, sname, variant)] = irf
                        if verbose:
                            print(f'PASS {pol:14s} {econ:9s} {sname:7s} {variant}')
                    except Exception as e:
                        if verbose:
                            print(f'FAIL {pol:14s} {econ:9s} {sname:7s} {variant}: '
                                  f'{type(e).__name__}: {e}')
    return results


def save_results(results, path=RESULTS_PATH):
    with open(path, 'wb') as f:
        pickle.dump(results, f)


def load_results(path=RESULTS_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)


#%%
def legend_handles(econ_list, shocks, variants):
    """Legend entries only for dimensions that vary."""
    multi_e, multi_s, multi_v = len(econ_list) > 1, len(shocks) > 1, len(variants) > 1
    h = []
    if multi_e:
        h += [Line2D([], [], color=COLORS[i % len(COLORS)], lw=2, label=e)
              for i, e in enumerate(econ_list)]
    elif multi_s:
        h += [Line2D([], [], color=COLORS[i % len(COLORS)], lw=2, label=s)
              for i, s in enumerate(shocks)]
    if multi_e and multi_s:
        h += [Line2D([], [], color='k', lw=SHOCK_LW[s], label=s) for s in shocks]
    if multi_v:
        h += [Line2D([], [], color='k', ls=VAR_LS[v], lw=2, label=v) for v in variants]
    if not h:
        h = [Line2D([], [], color=COLORS[0], ls=VAR_LS[variants[0]], lw=2, label=econ_list[0])]
    return h


def plot_grid(results, pol, outputs=OUTPUTS, H=H, economies=None, shocks=None,
              variants=None, ncol=4, save=SAVE, savepath=None):
    """Grid of IRFs for one fiscal policy, read from a stored results dict."""
    econ_list = list(economies) if economies is not None else ECON_LIST
    shk_list  = list(shocks)    if shocks    is not None else list(SHOCKS)
    var_list  = list(variants)  if variants  is not None else list(VARIANTS)
    multi_e, multi_s = len(econ_list) > 1, len(shk_list) > 1
    shock_idx = {s: i for i, s in enumerate(shk_list)}

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
    axes.flat[0].legend(handles=legend_handles(econ_list, shk_list, var_list), fontsize=8)
    fig.suptitle(pol, fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(savepath or f'irf_{pol}.pdf')
    return fig


#%%
results = run_all(model)
if SAVE:
    save_results(results)


#%%
# results = load_results()
for pol in FISCAL:
    plot_grid(results, pol)
plt.show()

# %%
