#%%
"""E9  Is the adoption margin still in the linear region at the baseline shock?

Motivation. The baseline price shock is size = 1.0, i.e. the world energy
price doubles on impact. The LINEARISED response of D_GREEN peaks at about
+0.092 in levels, from a steady state of 0.05: the green share would go from
5% to ~14%, a factor of ~2.8. D_GREEN is a share bounded in [0,1] and the logit
is sharp (taste_shock = 0.05), so this is a first-order approximation evaluated
far from the point of approximation -- and it is the paper's central channel.

Two cheap checks:

  (A) SCALING. Solve the linear IRF at several shock sizes and normalise by
      size. In a linear model the normalised paths coincide exactly, so this
      alone proves nothing -- it is run only to fix the reference. The
      informative object is (B).

  (B) NONLINEAR. Solve the full nonlinear transition at each size and compare
      D_GREEN against size * (linear response at size 1). The ratio
      nonlinear/linear as a function of size is the curvature of the adoption
      margin.

MEASURED RESULT (this machine, SSJ 1.0.0). The margin is strongly CONVEX and
the ratio is ABOVE 1, rising with size: NL/L = 1.16, 1.33, 1.67 at sizes
0.125, 0.25, 0.50 -- so linearisation UNDERSTATES adoption, increasingly with
size. At the baseline size = 1.0 the nonlinear solve does NOT converge (>30
backward iterations). The headline linear numbers are therefore first-order in
a region where the nonlinear problem is not even well-behaved; the baseline
shock should be reduced (0.25-0.5 still doubles-ish the price at its peak and
converges) or the headline experiments solved nonlinearly.

Runtime warning: solve_impulse_nonlinear is much slower than the linear
solve. Start with SIZES = [0.25, 1.0] if time is short.
"""
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.model import (build_model, solve_ss, shock_price,
                         UNKNOWNS_TD, TARGETS_TD)
from core.calibration import make_calibration

NUMERAIRE = 'cpi'
SIZES = [0.125, 0.25, 0.5, 1.0]
H = 24
OUT = 'paper/output'
os.makedirs(OUT, exist_ok=True)

MODEL = build_model(NUMERAIRE)
ss = solve_ss(MODEL, make_calibration(NUMERAIRE))
KEYS = ['y', 'C', 'cE', 'D_GREEN', 'pi_ann']

# --- reference linear response, per unit of shock ---------------------------
lin1 = MODEL.solve_impulse_linear(ss, UNKNOWNS_TD, TARGETS_TD, shock_price(size=1.0))
lin1 = {k: np.asarray(lin1[k])[:H] for k in KEYS}

rows = []
nl_paths = {}
converged_sizes = []
for sz in SIZES:
    print(f"  nonlinear, size = {sz} ...", flush=True)
    try:
        nl = MODEL.solve_impulse_nonlinear(ss, UNKNOWNS_TD, TARGETS_TD,
                                           shock_price(size=sz), verbose=False)
    except ValueError as e:
        print(f"  size={sz}: {e} -- skipped")
        continue
    nl = {k: np.asarray(nl[k])[:H] for k in KEYS}
    nl_paths[sz] = nl
    converged_sizes.append(sz)
    r = {}
    for k in KEYS:
        lin = sz * lin1[k]
        pk = int(np.argmax(np.abs(lin)))
        r[k] = (float(lin[pk]), float(nl[k][pk]),
                float(nl[k][pk] / lin[pk]) if lin[pk] != 0 else np.nan)
    rows.append((sz, r))
SIZES = converged_sizes

# --------------------------------------------------------------------- table
hdr = f"{'size':>6s} " + " ".join(f"{k + ' NL/L':>14s}" for k in KEYS)
lines = [hdr, '-' * len(hdr)]
for sz, r in rows:
    lines.append(f"{sz:6.3f} " + " ".join(f"{r[k][2]:14.4f}" for k in KEYS))
lines.append("")
lines.append("peak values at the LINEAR peak date (linear -> nonlinear):")
for sz, r in rows:
    lines.append(f"  size={sz:5.3f}  " +
                 "  ".join(f"{k}: {r[k][0]:+.4f} -> {r[k][1]:+.4f}" for k in KEYS))
tbl = '\n'.join(lines)
print('\n' + tbl)
open(f'{OUT}/linearity_table.txt', 'w').write(tbl + '\n')

# -------------------------------------------------------------------- figure
fig, axes = plt.subplots(1, len(KEYS), figsize=(3.6 * len(KEYS), 3.4))
for ax, k in zip(np.atleast_1d(axes).flat, KEYS):
    for sz in SIZES:
        ax.plot(100 * nl_paths[sz][k] / sz, lw=1.8, label=f'NL, size={sz}')
    ax.plot(100 * lin1[k], 'k--', lw=2, label='linear')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f'{k}, per unit of shock', fontsize=10)
    ax.set_xlabel('quarters', fontsize=8)
    ax.tick_params(labelsize=8)
np.atleast_1d(axes).flat[0].legend(fontsize=7)
fig.suptitle(r'E9. Curvature of the adoption margin: nonlinear vs linear, normalised by shock size')
fig.tight_layout(); fig.savefig(f'{OUT}/fig9_linearity.png', dpi=140); plt.close(fig)
print(f"  -> {OUT}/fig9_linearity.png")

pickle.dump({'linear_unit': lin1, 'nonlinear': nl_paths, 'sizes': SIZES},
            open(f'{OUT}/linearity.pkl', 'wb'))
print("\nDone.")

# %%
