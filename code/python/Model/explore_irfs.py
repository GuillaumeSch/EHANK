#%%
import sys; sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from core.model import build_model, run


#%%
NUM, BOOK, H = 'cpi', 'import', 24
SAVE = False           

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
    # 'no_adoption'
    ]

FISCAL = [                        
    'none',
    'subsidy',
    #'transfer',
    #'transfer_flat'
    ]

SHOCK_LS = {'price': '-', 'supply': '--'}
VAR_LW   = {'adoption': 2.0, 'no_adoption': 1.1}
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


def legend_handles():
    """Legend entries only for dimensions that vary."""
    multi_e, multi_s, multi_v = len(ECON_LIST) > 1, len(SHOCKS) > 1, len(VARIANTS) > 1
    base = 'k' if multi_e else COLORS[0]
    h = []
    if multi_e:
        h += [Line2D([], [], color=COLORS[i % len(COLORS)], lw=2, label=e)
              for i, e in enumerate(ECON_LIST)]
    if multi_s:
        h += [Line2D([], [], color=base, ls=SHOCK_LS[s], lw=2, label=s) for s in SHOCKS]
    if multi_v:
        h += [Line2D([], [], color=base, lw=VAR_LW[v], label=v) for v in VARIANTS]
    if not h:   # nothing varies
        h = [Line2D([], [], color=COLORS[0], ls=SHOCK_LS[list(SHOCKS)[0]],
                    lw=VAR_LW[VARIANTS[0]], label=ECON_LIST[0])]
    return h

#%%

for pol in FISCAL:
    series = {}
    for econ, ekw in ECONOMIES.items():
        for sname, shk in SHOCKS.items():
            for variant in VARIANTS:
                try:
                    _, irf = run(model, numeraire=NUM, booking=BOOK,
                                 model_variant=variant, policy=pol, **ekw, **shk)
                    series[(econ, sname, variant)] = irf
                    print(f'PASS {pol:14s} {econ:9s} {sname:7s} {variant}')
                except Exception as e:
                    print(f'FAIL {pol:14s} {econ:9s} {sname:7s} {variant}: {type(e).__name__}: {e}')
    ncol = 4
    nrow = int(np.ceil(len(OUTPUTS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, (k, lab) in zip(axes.flat, OUTPUTS.items()):
        for ci, econ in enumerate(ECON_LIST):
            for sname in SHOCKS:
                for variant in VARIANTS:
                    irf = series.get((econ, sname, variant))
                    if irf is None:
                        continue
                    y = np.asarray(irf[k])[:H] if k in irf else np.zeros(H)
                    ax.plot(100 * y,
                            color=COLORS[ci % len(COLORS)],
                            ls=SHOCK_LS[sname], lw=VAR_LW[variant])
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel('quarters', fontsize=8)
    for ax in list(axes.flat)[len(OUTPUTS):]:
        ax.axis('off')
    axes.flat[0].legend(handles=legend_handles(), fontsize=8)
    fig.suptitle(pol, fontsize=12)
    fig.tight_layout()
    if SAVE:
        fig.savefig(f'irf_{pol}.pdf')

plt.show()

# %%   