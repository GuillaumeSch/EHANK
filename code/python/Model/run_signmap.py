#%%
"""Route A sign-map: adoption's cumulative-output contribution (Delta_24) as a
function of the brown-price-shock half-life, with the supply-shock value
marked for comparison. Produces output/fig_signmap.png
(Figure~\\ref{fig:signmap} in ehank_results.tex, Section~\\ref{sec:signmap}).

Booking: 'import'. The Route A characterisation (persistence and closure
comparative statics) is stated for the import booking; the domestic-booking
comparison is a separate result (Table~\\ref{tab:signmap}, run_booking_compare.py).

EFFICIENCY. The steady state does not depend on the shock's half-life (only
the shock PATH does), so it is solved ONCE per adoption variant and the
half-life sweep re-solves only the linear impulse response -- the
`resolve_ss=False` idiom used throughout the codebase (see plotting.py).
"""
import os
import pickle
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import (build_model, solve_ss, run, shock_price,
                   td_unknowns_targets, ss_unknowns_targets,
                   ss_unknowns_targets_fixed_psi, MODELS, ENERGY_CLOSURE)
from calibration import make_calibration

BOOKING = 'import'
NUMERAIRE = 'core'
OUT = 'output'
CDIR = 'cache_signmap'
os.makedirs(OUT, exist_ok=True)
os.makedirs(CDIR, exist_ok=True)

H = 24
HALF_LIVES = [2, 4, 6, 8, 11, 16, 24, 32, 48]

MODEL = build_model(NUMERAIRE, booking=BOOKING)
utd, ttd = td_unknowns_targets(BOOKING)


def cum_y(irf, h=H):
    return 100 * float(np.sum(np.asarray(irf['y'])[:h]))


# --- steady states, solved ONCE (price shock's own closure: elastic) --------
ov = dict(ENERGY_CLOSURE['elastic'])
calib_ad = make_calibration(NUMERAIRE, booking=BOOKING, **{**ov, **MODELS['adoption']})
u_ad, t_ad = ss_unknowns_targets(BOOKING)
ss_ad = solve_ss(MODEL, calib_ad, unknowns=u_ad, targets=t_ad, booking=BOOKING)

# no_adoption carries psi_g over from the adoption steady state (see
# model.run's no_adoption handling: D_GREEN is pinned at 0, so psi_g cannot be
# solved against it -- same primitives, only the discrete choice is blocked).
calib_no = make_calibration(NUMERAIRE, booking=BOOKING, **{**ov, **MODELS['no_adoption']})
calib_no['psi_g'] = float(ss_ad['psi_g'])
u_no, t_no = ss_unknowns_targets_fixed_psi(BOOKING)
ss_no = solve_ss(MODEL, calib_no, unknowns=u_no, targets=t_no, booking=BOOKING)

# --- half-life sweep: re-solve the LINEAR IRF only (SS is shock-invariant) --
delta = []
for hl in HALF_LIVES:
    f = f'{CDIR}/delta_hl{hl}.pkl'
    if os.path.exists(f):
        d = pickle.load(open(f, 'rb'))
    else:
        shk = shock_price(half_life=hl)
        irf_ad = MODEL.solve_impulse_linear(ss_ad, utd, ttd, shk)
        irf_no = MODEL.solve_impulse_linear(ss_no, utd, ttd, shk)
        d = cum_y(irf_ad) - cum_y(irf_no)
        pickle.dump(d, open(f, 'wb'))
    delta.append(d)
    print(f"  half_life={hl:4d}  Delta_{H}={d:+.2f}", flush=True)
delta = np.array(delta)

# --- supply-shock marker (needs its own inelastic-closure steady state) -----
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

# --- zero crossing (linear interpolation between bracketing grid points) ----
sign_change = np.where(np.diff(np.sign(delta)) != 0)[0]
crossing = None
if len(sign_change):
    i = sign_change[0]
    x0, x1 = HALF_LIVES[i], HALF_LIVES[i + 1]
    y0, y1 = delta[i], delta[i + 1]
    crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
    print(f"  zero crossing at half_life ~= {crossing:.1f} quarters")
else:
    print("  WARNING: no sign change found in the swept half-life grid")

# --- figure (colours match the caption: price=orange/C1, supply=blue/C0) ----
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
print(f"-> {OUT}/fig_signmap.png")

# %%
