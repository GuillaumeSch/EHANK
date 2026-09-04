"""App A robustness: adoption-spending import content (alpha_F_switch) sweep.

Replaces the old import-vs-domestic booking comparison. alpha_F_switch is the
import share of the adoption-expenditure bundle: 1.0 is the pure-import baseline
(spending leaves the country, most contractionary), lower values route adoption
spending onto the home good and cushion the output contraction. This is the
continuous version of the old binary booking robustness.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.model import build_model, run

# ---- 1. setup ----------------------------------------------------------------
NUMERAIRE = 'cpi'
AFS_GRID = [1.0, 0.5, 0.0]   # 3 points: import / mixed / domestic endpoints
SHOCKS = ['price', 'supply']
H = 24
OUT = 'paper/output'
os.makedirs(OUT, exist_ok=True)

cum = lambda irf, k: 100 * float(np.sum(np.asarray(irf[k])[:H]))
peak = lambda irf, k: 100 * float(np.asarray(irf[k])[:H][np.argmax(np.abs(np.asarray(irf[k])[:H]))])
y0 = lambda irf: 100 * float(np.asarray(irf['y'])[0])

model = build_model(NUMERAIRE, booking='import')

# ---- 2. sweep: full economy vs frozen-choice counterfactual ------------------
rows, paths = [], {}
for afs in AFS_GRID:
    for shock in SHOCKS:
        _, ad = run(model, shock_kind=shock, policy='none', model_variant='adoption',
                    numeraire=NUMERAIRE, booking='import', alpha_F_switch=afs)
        _, no = run(model, shock_kind=shock, policy='none', model_variant='no_adoption',
                    numeraire=NUMERAIRE, booking='import', alpha_F_switch=afs)
        adopt_y = cum(ad, 'y') - cum(no, 'y')   # adoption channel: full - frozen (common SS)
        print(f"  done afs={afs:.2f} {shock:>6s}: peak y={peak(ad,'y'):.3f}%, adopt->y={adopt_y:.2f}", flush=True)
        rows.append(dict(afs=afs, shock=shock, y0=y0(ad), ypeak=peak(ad, 'y'),
                         dG=peak(ad, 'D_GREEN'), nxgdp=peak(ad, 'nx_gdp'),
                         cEcum=cum(ad, 'cE'), adopt_y=adopt_y))
        paths[(afs, shock)] = np.asarray(ad['y'])[:H]

# ---- 3. table ----------------------------------------------------------------
hdr = (f"{'aFswitch':>9s} {'shock':>7s} {'y(0)%':>8s} {'peak y%':>9s} "
       f"{'peak DG':>9s} {'peak nx/gdp':>12s} {'cum cE':>9s} {'adopt->y':>10s}")
lines = [hdr, '-' * len(hdr)]
for r in rows:
    lines.append(f"{r['afs']:9.2f} {r['shock']:>7s} {r['y0']:8.3f} {r['ypeak']:9.3f} "
                 f"{r['dG']:9.3f} {r['nxgdp']:12.3f} {r['cEcum']:9.2f} {r['adopt_y']:10.2f}")
tbl = '\n'.join(lines)
print(tbl)
open(f'{OUT}/afs_sweep.txt', 'w').write(tbl + '\n')

# ---- 4. figure: output IRF across the import-content grid --------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, shock in zip(axes, SHOCKS):
    for afs in AFS_GRID:
        ax.plot(100 * paths[(afs, shock)], label=rf'$\alpha_F^{{switch}}={afs:.2f}$')
    ax.axhline(0, lw=0.6, color='k')
    ax.set_title(f'{shock} shock')
    ax.set_xlabel('quarters')
axes[0].set_ylabel(r'output $y$, % dev.')
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_afs_sweep.png', dpi=150)

# ---- 5. LaTeX table ----------------------------------------------------------
tex = [r'\begin{tabular}{lrrrrr}', r'\toprule',
       r'$\alpha_F^{switch}$ & shock & peak $y$ (\%) & peak $\Delta D^G$ (pp) '
       r'& peak $nx/gdp$ (\%) & adoption$\to y$ \\', r'\midrule']
for r in rows:
    tex.append(f"{r['afs']:.2f} & {r['shock']} & {r['ypeak']:.3f} & {r['dG']:.3f} "
               f"& {r['nxgdp']:.3f} & {r['adopt_y']:.2f} \\\\")
tex += [r'\bottomrule', r'\end{tabular}']
open(f'{OUT}/tab_afs_sweep.tex', 'w').write('\n'.join(tex) + '\n')
print('ALLDONE', flush=True)
