"""Cross-booking summary. Caches each solve so it can resume."""
import os, pickle
import numpy as np
from model import build_model, run
import latex_tables as LT

os.makedirs('output', exist_ok=True)
os.makedirs('cache_bk', exist_ok=True)
H = 24  # matches the paper's "cumulative sums over 24 quarters" convention;
        # cache stores full-length series, so this needs no cache wipe.
cum = lambda irf, k: 100 * float(np.sum(np.asarray(irf[k])[:H]))
peak = lambda irf, k: 100 * float(np.max(np.asarray(irf[k])[:H]))
y0 = lambda irf: 100 * float(np.asarray(irf['y'])[0])
KEEP = ['y', 'D_GREEN', 'cE', 'spending']

def cached(booking, shock, policy, mv, M):
    f = f'cache_bk/{booking}_{shock}_{policy}_{mv}.pkl'
    if os.path.exists(f):
        return pickle.load(open(f, 'rb'))
    irf = run(M, shock_kind=shock, policy=policy, model_variant=mv, booking=booking)[1]
    d = {k: np.asarray(irf[k]) for k in KEEP}
    pickle.dump(d, open(f, 'wb'))
    print(f'  done {booking} {shock} {policy} {mv}', flush=True)
    return d

rows = []
for booking in ['import', 'domestic']:
    M = build_model('core', booking=booking)
    for shock in ['price', 'supply']:
        ad = cached(booking, shock, 'none', 'adoption', M)
        no = cached(booking, shock, 'none', 'no_adoption', M)
        adopt_y = cum(ad, 'y') - cum(no, 'y')
        for p in ['none', 'subsidy', 'transfer']:
            irf = ad if p == 'none' else cached(booking, shock, p, 'adoption', M)
            rows.append(dict(booking=booking, shock=shock, policy=p,
                y0=y0(irf), ycum=cum(irf, 'y'), dG=peak(irf, 'D_GREEN'),
                cEcum=cum(irf, 'cE'), fisc=cum(irf, 'spending'),
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
open('output/booking_compare.txt', 'w').write(tbl + '\n')

signmap = {(r['booking'], r['shock']): r['adopt_y'] for r in rows if r['policy'] == 'none'}
tex_path = LT.booking_signmap_table('output/tab_signmap_booking.tex', signmap, H)
print(f"  -> {tex_path}")
print('ALLDONE', flush=True)
