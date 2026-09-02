"""Adoption's cumulative-output contribution vs shock half-life (diagnostic)."""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from core.model import (build_model, frozen_model, solve_ss, run, shock_price,
                   td_unknowns_targets, ss_unknowns_targets)
from core.calibration import make_calibration

BOOKING = 'import'
NUMERAIRE = 'cpi'
OUT = 'paper/output'
CDIR = 'cache_signmap'
os.makedirs(OUT, exist_ok=True)
# Wipe the cache on every run.
import shutil
shutil.rmtree(CDIR, ignore_errors=True)
os.makedirs(CDIR, exist_ok=True)

H = 24
HALF_LIVES = [2, 4, 6, 8, 11, 16, 24, 32, 48]

MODEL = build_model(NUMERAIRE, booking=BOOKING)
MODEL_FROZEN = frozen_model(NUMERAIRE, booking=BOOKING)
utd, ttd = td_unknowns_targets(BOOKING)

def cum_y(irf, h=H):
    return 100 * float(np.sum(np.asarray(irf['y'])[:h]))

# steady state, solved once (full and frozen models share it)
u_ss, t_ss = ss_unknowns_targets(BOOKING)
ss = solve_ss(MODEL, make_calibration(NUMERAIRE, booking=BOOKING),
              unknowns=u_ss, targets=t_ss, booking=BOOKING)

# half-life sweep: re-solve the linear IRF only; Delta_24 = cum_y(full) - cum_y(frozen)
delta = []
for hl in HALF_LIVES:
    f = f'{CDIR}/delta_hl{hl}.pkl'
    if os.path.exists(f):
        d = pickle.load(open(f, 'rb'))
    else:
        shk = shock_price(half_life=hl)
        irf_ad = MODEL.solve_impulse_linear(ss, utd, ttd, shk)
        irf_no = MODEL_FROZEN.solve_impulse_linear(ss, utd, ttd, shk)
        d = cum_y(irf_ad) - cum_y(irf_no)
        pickle.dump(d, open(f, 'wb'))
    delta.append(d)
    print(f"  half_life={hl:4d}  Delta_{H}={d:+.2f}", flush=True)
delta = np.array(delta)

# supply-shock marker (own inelastic-closure steady state)
fs = f'{CDIR}/delta_supply.pkl'
if os.path.exists(fs):
    delta_supply = pickle.load(open(fs, 'rb'))
else:
    _, irf_s_ad = run(MODEL, shock_kind='supply', policy='none',
                      model_variant='adoption', booking=BOOKING)
    _, irf_s_no = run(MODEL, shock_kind='supply', policy='none',
                      model_variant='no_adoption', booking=BOOKING)
    delta_supply = cum_y(irf_s_ad) - cum_y(irf_s_no)
    pickle.dump(delta_supply, open(fs, 'wb'))
print(f"  supply shock  Delta_{H}={delta_supply:+.2f}")

# zero crossing (linear interpolation)
sign_change = np.where(np.diff(np.sign(delta)) != 0)[0]
crossing = None
if len(sign_change):
    i = sign_change[0]
    x0, x1 = HALF_LIVES[i], HALF_LIVES[i + 1]
    y0, y1 = delta[i], delta[i + 1]
    crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
    print(f"  zero crossing at half_life ~= {crossing:.1f} quarters")
else:
    print("  (no sign change: adoption channel is single-signed across the grid, "
          "as expected under the common-steady-state counterfactual)")

# figure
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(HALF_LIVES, delta, 'o-', color='C1', lw=2, label='price shock')
ax.axhline(0, color='k', lw=0.7)
ax.axhline(delta_supply, color='C0', ls='--', lw=1.8, label='supply shock (marked)')
if crossing is not None:
    ax.axvline(crossing, color='gray', ls=':', lw=1)
ax.set_xlabel('brown-price-shock half-life (quarters)')
ax.set_ylabel(rf'$\Delta_{{{H}}}$ (adoption contribution to cum. output)')
ax.set_title('Adoption channel sign map (import booking)')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_signmap.png', dpi=140)
plt.close(fig)
