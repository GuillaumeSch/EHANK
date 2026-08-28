"""Cross-booking summary: the adoption channel's output contribution under the
import vs domestic balance-of-payments booking, for the price and supply shocks.

Uses the shared run() machinery, so the 'no_adoption' counterfactual is the
COMMON-STEADY-STATE (frozen-choice) construction (model.frozen_model): full and
frozen share a bit-identical steady state and differ only in the transition, and
the adoption channel is the clean difference (full - frozen). This script does
NOT reimplement the counterfactual, so it needed no logic change for the frozen
migration -- but its on-disk cache PREDATES that migration, so it is wiped by
default (CLEAR_CACHE) to avoid resuming from stale, mixed-SS pickles.

Headline real-activity metric is cumulative output over H=24 (cum y), matching
Section 4 and tab:dose; the adoption channel is (full - frozen) cum y.
"""
import os
import pickle
import shutil
import numpy as np
from core.model import build_model, run
from tools import latex_tables as LT

NUMERAIRE = 'core'
BOOKING_LIST = ['import', 'domestic']
H = 24
OUT, CDIR = 'paper/output', 'cache/cache_bk'

# The cache is keyed by tag only and goes stale silently on any model change
# (it predates the frozen migration). Wipe on every run; set False only to
# resume a crashed run with an unchanged model.
CLEAR_CACHE = True
if CLEAR_CACHE:
    shutil.rmtree(CDIR, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(CDIR, exist_ok=True)

cum = lambda irf, k: 100 * float(np.sum(np.asarray(irf[k])[:H]))
peak = lambda irf, k: 100 * float(np.max(np.asarray(irf[k])[:H]))
y0 = lambda irf: 100 * float(np.asarray(irf['y'])[0])
KEEP = ['y', 'D_GREEN', 'D_SWITCH', 'cE', 'spending']


def cached(booking, shock, policy, mv, M):
    f = f'{CDIR}/{booking}_{shock}_{policy}_{mv}.pkl'
    if os.path.exists(f):
        return pickle.load(open(f, 'rb'))
    irf = run(M, shock_kind=shock, policy=policy, model_variant=mv,
              booking=booking, numeraire=NUMERAIRE)[1]
    d = {k: np.asarray(irf[k]) for k in KEEP}
    pickle.dump(d, open(f, 'wb'))
    print(f'  done {booking} {shock} {policy} {mv}', flush=True)
    return d


rows = []
for booking in BOOKING_LIST:
    M = build_model(NUMERAIRE, booking=booking)
    for shock in ['price', 'supply']:
        ad = cached(booking, shock, 'none', 'adoption', M)
        no = cached(booking, shock, 'none', 'no_adoption', M)
        adopt_y = cum(ad, 'y') - cum(no, 'y')          # full - frozen (common SS)
        for p in ['none', 'subsidy', 'transfer']:
            irf = ad if p == 'none' else cached(booking, shock, p, 'adoption', M)
            rows.append(dict(booking=booking, shock=shock, policy=p,
                y0=y0(irf), ycum=cum(irf, 'y'), dG=peak(irf, 'D_GREEN'),
                dSw=peak(irf, 'D_SWITCH'), cEcum=cum(irf, 'cE'),
                fisc=cum(irf, 'spending'),
                adopt_y=(adopt_y if p == 'none' else np.nan)))

hdr = (f"{'booking':>9s} {'shock':>7s} {'policy':>9s} {'y(0)%':>8s} {'cum y':>9s} "
       f"{'peak DG':>9s} {'cum cE':>9s} {'fiscal':>9s} {'adopt->y':>10s}")
lines = [hdr, '-' * len(hdr)]
for r in rows:
    ay = '' if np.isnan(r['adopt_y']) else f"{r['adopt_y']:10.2f}"
    lines.append(f"{r['booking']:>9s} {r['shock']:>7s} {r['policy']:>9s} "
                 f"{r['y0']:8.3f} {r['ycum']:9.2f} {r['dG']:9.3f} "
                 f"{r['cEcum']:9.2f} {r['fisc']:9.2f} {ay:>10s}")
tbl = '\n'.join(lines)
print(tbl)
open(f'{OUT}/booking_compare.txt', 'w').write(tbl + '\n')

signmap = {(r['booking'], r['shock']): r['adopt_y']
           for r in rows if r['policy'] == 'none'}
tex_path = LT.booking_signmap_table(f'{OUT}/tab_signmap_booking.tex', signmap, H)
print(f"  -> {tex_path}")
print('ALLDONE', flush=True)
