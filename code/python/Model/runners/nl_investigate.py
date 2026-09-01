"""Deep dive: where does the model's nonlinearity live, and is it a bug?

Design:
  A. size sweep, adoption open vs frozen  -> localize NL to D_GREEN vs macro
  B. taste_shock sweep at fixed size      -> test the logit-tail mechanism
Each (experiment, point) is checkpointed so the run is resumable.

'no_adoption' (variant='no_adoption') is the COMMON-STEADY-STATE counterfactual
(frozen_model(): same SS as adoption, D_GREEN does not respond to the shock in
the transition). It replaces the earlier version that shut the margin in the
steady state itself (green_block huge, D_GREEN_ss=0%), so cached results from
before this change (nl_size.pkl / nl_taste.pkl under the OLD methodology) are
NOT comparable and must be regenerated -- hence the '_frozen' cache tag below.

AUDIT FIX (maxit): the top-level solve_impulse_nonlinear(maxit=...) sets the
outer Newton count only. The inner GE Newton of the energyPrices solved block
reads the class-level default Block.solve_impulse_nonlinear_options=dict(maxit=30),
which the outer kwarg does NOT reach. Lifting that default (below) gives the
mid-range solves headroom and removes a latent cliff. It does NOT buy the
baseline shock size: eps=1.0 converges only from a fresh model, not inside a
grid loop that has already solved smaller sizes on the same instance (state
carried in M between solves), so the default grid stops at 0.75 -- which is
exactly where the paper's nonlinearity claims live. eps=1.0 is left out and
documented as fragile rather than forced.
"""
import os, pickle, sys, time
import numpy as np
from sequence_jacobian.blocks.block import Block
from core.model import build_model, run, shock_price, td_unknowns_targets, frozen_model

# --- AUDIT FIX: lift the inner-block backward-iteration cap (default 30). ---
INNER_MAXIT = 100
Block.solve_impulse_nonlinear_options = dict(tol=1e-8, maxit=INNER_MAXIT, verbose=False)

M = build_model('cpi', booking='import')
U, T = td_unknowns_targets('import')
SERIES = ['y', 'C', 'cE', 'D_GREEN', 'D_SWITCH']
H = 24


def metrics(lin, nl):
    out = {}
    for k in SERIES:
        L = np.asarray(lin[k]); N = np.asarray(nl[k])
        Li, Ni = 100 * float(L[0]), 100 * float(N[0])
        Lp = 100 * float(np.max(np.abs(L[:H]))); Np = 100 * float(np.max(np.abs(N[:H])))
        out[k] = dict(Li=Li, Ni=Ni, Lp=Lp, Np=Np)
    return out


def _ratio(m, key, floor=1e-3):
    """NL/L peak ratio, guarded. Returns nan when the linear peak is ~0.

    The frozen ('no_adoption') D_GREEN barely moves, so its linear peak is
    numerically zero and the raw ratio explodes to O(100). Reporting that as a
    'nonlinearity' is meaningless; guard it.
    """
    Lp = abs(m[key]['Lp'])
    return m[key]['Np'] / Lp if Lp > floor else float('nan')


def solve_pair(size, variant='adoption', taste=None, maxit=200):
    ov = {}
    if taste is not None:
        ov['taste_shock'] = taste
    ss, lin = run(M, shock_kind='price', policy='none', model_variant=variant,
                  shock_kwargs=dict(size=size), **ov)
    shk = shock_price(size=size)
    # Nonlinear solve must use the SAME dag as the linear one above: the
    # frozen-choice DAG for 'no_adoption' (common SS, but the household internals
    # -- hence the nonlinear residual functions -- differ from the adoption dag).
    nl_model = frozen_model('cpi', 'import') if variant == 'no_adoption' else M
    t0 = time.time()
    # SSJ raises on non-convergence, so a returned path is converged by
    # construction. We still record the outer maxit and wall time for provenance.
    nl = nl_model.solve_impulse_nonlinear(ss, U, T, shk, maxit=maxit, tol=1e-8,
                                          verbose=False)
    dt = time.time() - t0
    return dict(m=metrics(lin, nl), psi_g=float(ss['psi_g']),
                DG_ss=float(ss['D_GREEN']), dt=dt, maxit_inner=INNER_MAXIT, ok=True)


def load(f):
    return pickle.load(open(f, 'rb')) if os.path.exists(f) else {}


def run_grid(cache, grid, **fixed):
    res = load(cache)
    for key, kw in grid:
        if key in res and res[key].get('ok'):
            print(f'skip {key}', flush=True); continue
        try:
            res[key] = solve_pair(**kw, **fixed)
            yr = _ratio(res[key]['m'], 'y')
            dgr = _ratio(res[key]['m'], 'D_GREEN')
            print(f'done {key}  ({res[key]["dt"]:.0f}s)  '
                  f'y NL/L={yr:.3f}  DG NL/L={dgr:.3f}', flush=True)
        except Exception as e:
            res[key] = dict(ok=False, err=f'{type(e).__name__}: {e}')
            print(f'FAIL {key}: {res[key]["err"]}', flush=True)
        pickle.dump(res, open(cache, 'wb'))
    return res


# Cache filenames carry a '_frozen' tag: results cached under the OLD
# methodology (nl_size.pkl / nl_taste.pkl, if left over from before this
# change) are NOT comparable and must not be silently reused.
SIZE_CACHE = 'cache/nl_size_frozen.pkl'
TASTE_CACHE = 'cache/nl_taste_frozen.pkl'

# Sizes: the run grid and build_fig9 read the same list, so the figure never
# silently drops points. Capped at 0.75 (the convergent range and the paper's
# stated range); eps=1.0 is fragile in a warm grid loop -- see module docstring.
SIZES = [0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75]
NO_SIZES = [0.125, 0.25, 0.5, 0.75]
TASTES = [0.02, 0.05, 0.10, 0.20, 0.40]


def build_fig9(out='paper/output/fig9_nonlinearity.png'):
    """Assemble the E9 nonlinearity figure from the checkpointed sweeps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rs = load(SIZE_CACHE); rt = load(TASTE_CACHE)

    def ok(d, k):
        return k in d and d[k].get('ok')

    ad_sz = [s for s in SIZES if ok(rs, f'ad_{s}')]
    ad_y = [_ratio(rs[f'ad_{s}']['m'], 'y') for s in ad_sz]
    ad_dg = [_ratio(rs[f'ad_{s}']['m'], 'D_GREEN') for s in ad_sz]
    no_sz = [s for s in NO_SIZES if ok(rs, f'no_{s}')]
    no_y = [_ratio(rs[f'no_{s}']['m'], 'y') for s in no_sz]
    tv = [ts for ts in TASTES if ok(rt, f'ts_{ts}')]
    t_dg = [_ratio(rt[f'ts_{ts}']['m'], 'D_GREEN') for ts in tv]
    t_y = [_ratio(rt[f'ts_{ts}']['m'], 'y') for ts in tv]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.axhline(1, color='k', lw=0.6, ls=':')
    a1.plot(ad_sz, ad_dg, 'o-', color='C1', lw=2, label=r'$D^{G}$ (adoption on)')
    a1.plot(ad_sz, ad_y, 's-', color='C0', lw=2, label=r'output $y$ (adoption on)')
    a1.plot(no_sz, no_y, '^--', color='C3', lw=2,
            label=r'output $y$ (adoption frozen, common SS)')
    a1.set_xlabel('shock size (impact log-dev of world energy price)')
    a1.set_ylabel('nonlinear / linear (peak)')
    a1.set_title('Nonlinearity vs shock size'); a1.legend(fontsize=8)
    a2.axhline(1, color='k', lw=0.6, ls=':')
    a2.plot(tv, t_dg, 'o-', color='C1', lw=2, label=r'$D^{G}$')
    a2.plot(tv, t_y, 's-', color='C0', lw=2, label=r'output $y$')
    a2.axvline(0.05, color='grey', lw=1, ls='--')
    a2.set_xlabel(r'logit taste scale $\sigma_\varepsilon$')
    a2.set_ylabel('nonlinear / linear (peak)')
    a2.set_title(r'Nonlinearity vs logit scale (size=0.5)'); a2.legend(fontsize=8)
    fig.suptitle('E9. The nonlinearity lives in the discrete adoption margin', y=1.02)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'  -> {out}')


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'size'
    if which == 'size':
        grid = ([(f'ad_{s}', dict(size=s, variant='adoption')) for s in SIZES] +
                [(f'no_{s}', dict(size=s, variant='no_adoption')) for s in NO_SIZES])
        run_grid(SIZE_CACHE, grid)
    elif which == 'taste':
        grid = [(f'ts_{ts}', dict(size=0.5, variant='adoption', taste=ts))
                for ts in TASTES]
        run_grid(TASTE_CACHE, grid)
    elif which == 'fig':
        build_fig9()
