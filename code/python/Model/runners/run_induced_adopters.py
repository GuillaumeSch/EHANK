"""Who transitions because of the shock: the induced-adopter field (Section 5).

The shock recruits switchers who would not have switched at the pre-crisis steady
state. We characterise them with a comparative static: the switching-probability
policy function of the median discount-factor type at a steady state where the brown
energy price is permanently 10% higher (tau_b=0.10) minus the baseline field. The
difference dP(a,e) > 0 is the extra switching the higher brown price induces, located
in the wealth-productivity plane.

This is a comparative-static proxy for the transition object: it uses a permanent
price change rather than the transitory crisis path, so it characterises the
DIRECTION and LOCATION of induced switching rather than its exact transitional
magnitude. The full-transition version would read the disaggregated choice
probabilities along the impulse response; the comparative static is the tractable
first pass and is signed and located identically.

Finding: induced switching rises with productivity and, at high productivity, the
recruitment broadens to more moderate wealth -- the shock pulls in high-productivity
households who at the baseline sat just below the switching threshold.

Emits fig_induced_adopters_import.pdf. ~2 SS solves.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from core.model import build_model, run
from core.calibration import make_calibration
from core.household import make_grids
from core import household as hh_mod

NUMERAIRE, BOOKING = 'cpi', 'import'
OUT = 'paper/output'


def median_field(model, ets):
    kw = dict(ets=True, ets_kwargs=dict(tau_b=0.10, recycle='rebate')) if ets else {}
    ss, _ = run(model, shock_kind='price', policy='none', model_variant='adoption', **kw)
    greens = np.where(hh_mod.IS_GREEN > 0)[0]
    browns = np.where(hh_mod.IS_GREEN == 0)[0]
    blocks = sorted(k for k in ss.internals if k.startswith('hh_'))
    betas = {b: float(ss[f"beta_{b.split('_')[1]}"]) for b in blocks}
    med = sorted(blocks, key=lambda b: betas[b])[len(blocks) // 2]
    it = ss.internals[med]
    P = it['durables']['law_of_motion'].P
    Dd = it['durables']['D']
    Psw = P[greens].sum(axis=0)
    origin = browns[int(np.argmax([Dd[d].sum() for d in browns]))]
    Dpop = sum(ss.internals[b]['durables']['D'] for b in blocks)
    a_grid = it['a_grid']
    mean_a = float((Dpop.sum(axis=(0, 1)) * a_grid).sum() / Dpop.sum())
    return Psw[origin], Dd[origin], mean_a


if __name__ == '__main__':
    model = build_model(NUMERAIRE, booking=BOOKING)
    cal = make_calibration(NUMERAIRE, BOOKING)
    e_grid, _, a_grid, _ = make_grids(cal['rho_e'], cal['sd_e'], cal['n_e'],
                                      cal['min_a'], cal['max_a'], cal['n_a'], cal['delta_g'])
    P0, D0, mean_a = median_field(model, False)
    P1, _, _ = median_field(model, True)
    dP = 100.0 * (P1 - P0)                      # induced switching, pp per quarter
    a_rel = a_grid / mean_a                     # wealth relative to the mean

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    im = ax.pcolormesh(np.arange(cal['n_a']), np.arange(cal['n_e']), dP,
                       cmap='BuGn', shading='nearest')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'induced $\Delta P(\mathrm{switch})$, pp per quarter')
    xt = np.linspace(0, cal['n_a'] - 1, 6).astype(int)
    yt = np.arange(cal['n_e'])
    ax.set_xticks(xt); ax.set_xticklabels([f'{a_rel[i]:.1f}' for i in xt])
    ax.set_yticks(yt); ax.set_yticklabels([f'{e:.2f}' for e in e_grid])
    ax.set_xlabel(r'household wealth relative to mean, $a/\bar a$')
    ax.set_ylabel(r'idiosyncratic productivity $e$')
    fig.tight_layout()
    out = os.path.join(OUT, f'fig_induced_adopters_{BOOKING}.pdf')
    fig.savefig(out)
    plt.close(fig)

    mw = dP * D0
    ce, ca = np.unravel_index(np.argmax(mw), mw.shape)
    print(f"induced switching: mean {dP.mean():.2f}pp, max {dP.max():.2f}pp")
    print(f"mass-weighted peak at e={e_grid[ce]:.2f}, a={a_grid[ca]:.1f}")
    print(f"  -> {out}")
