#%%
import sys; sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt
from core.model import build_model, run


#%%
NUM, BOOK, H = 'cpi', 'import', 24
SAVE = True           # True -> also write irf_<version>.pdf
TAX = dict(ets=False, ets_kwargs=dict(tau_b=0.0, recycle='rebate')) 

SCENARIOS = {
    'price_notax':  dict(shock_kind='price',  ets=False),
    #'price_tax':    dict(shock_kind='price',  **TAX),
    'supply_notax': dict(shock_kind='supply', ets=False),
    #'supply_tax':   dict(shock_kind='supply', **TAX),
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

# pB and pG inflation 
# 

OUTPUTS = {
    'y':       r'Output $y$',
    'C':       r'Consumption $C$',
    'D_GREEN': r'Green share $D^G$',
    'CHF_SWITCH_exp': r'Adoption expenditures',
    'pi_ann':  r'Inflation (ann.)',
    'PEstar': r'Market price of brown energy (in USD) $P^*_{Eb}$', 
    # 'E_supply': r'Energy supply', 
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

LS = {'adoption': '-', 'no_adoption': '--'}
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
model = build_model(NUM, booking=BOOK)

#%%

for name, scn in SCENARIOS.items():
    series = {}
    for pol in FISCAL:
        for variant in VARIANTS:
            try:
                _, irf = run(model, numeraire=NUM, booking=BOOK,
                             model_variant=variant, policy=pol, **scn)
                series[(pol, variant)] = irf
                print(f'PASS {name:13s} {pol:14s} {variant}')
            except Exception as e:
                print(f'FAIL {name:13s} {pol:14s} {variant}: {type(e).__name__}: {e}')
    ncol = 4
    nrow = int(np.ceil(len(OUTPUTS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, (k, lab) in zip(axes.flat, OUTPUTS.items()):
        for ci, pol in enumerate(FISCAL):
            for variant in VARIANTS:
                irf = series.get((pol, variant))
                if irf is None:
                    continue
                label = pol if len(VARIANTS) == 1 else f'{pol} / {variant}'
                ax.plot(100 * np.asarray(irf[k])[:H], lw=2,
                        color=COLORS[ci % len(COLORS)], ls=LS.get(variant, '-'),
                        label=label)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel('quarters', fontsize=8)
    for ax in list(axes.flat)[len(OUTPUTS):]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(name, fontsize=12)
    fig.tight_layout()
    if SAVE:
        fig.savefig(f'irf_{name}.pdf')

plt.show()

# %%
