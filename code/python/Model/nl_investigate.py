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
"""
import os, pickle, time, numpy as np
from model import build_model, run, shock_price, td_unknowns_targets, frozen_model

M = build_model('core', booking='import')
U, T = td_unknowns_targets('import')
SERIES = ['y', 'C', 'cE', 'D_GREEN', 'D_SWITCH']
H = 24


def metrics(lin, nl):
    out = {}
    for k in SERIES:
        L = np.asarray(lin[k]); N = np.asarray(nl[k])
        Li, Ni = 100*float(L[0]), 100*float(N[0])
        Lp = 100*float(np.max(np.abs(L[:H]))); Np = 100*float(np.max(np.abs(N[:H])))
        out[k] = dict(Li=Li, Ni=Ni, Lp=Lp, Np=Np)
    return out


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
    nl_model = frozen_model('core', 'import') if variant == 'no_adoption' else M
    t0 = time.time()
    nl = nl_model.solve_impulse_nonlinear(ss, U, T, shk, maxit=maxit, tol=1e-8,
                                          verbose=False)
    dt = time.time() - t0
    return dict(m=metrics(lin, nl), psi_g=float(ss['psi_g']),
                DG_ss=float(ss['D_GREEN']), dt=dt, ok=True)


def load(f):
    return pickle.load(open(f, 'rb')) if os.path.exists(f) else {}


def run_grid(cache, grid, **fixed):
    res = load(cache)
    for key, kw in grid:
        if key in res:
            print(f'skip {key}', flush=True); continue
        try:
            res[key] = solve_pair(**kw, **fixed)
            print(f'done {key}  ({res[key]["dt"]:.0f}s)  '
                  f'y NL/L={res[key]["m"]["y"]["Np"]/max(abs(res[key]["m"]["y"]["Lp"]),1e-9):.3f}  '
                  f'DG NL/L={res[key]["m"]["D_GREEN"]["Np"]/max(abs(res[key]["m"]["D_GREEN"]["Lp"]),1e-9):.3f}',
                  flush=True)
        except Exception as e:
            res[key] = dict(ok=False, err=f'{type(e).__name__}: {e}')
            print(f'FAIL {key}: {res[key]["err"]}', flush=True)
        pickle.dump(res, open(cache, 'wb'))
    return res


# Cache filenames carry a '_frozen' tag: results cached under the OLD
# methodology (nl_size.pkl / nl_taste.pkl, if left over from before this
# change) are NOT comparable and must not be silently reused.
SIZE_CACHE = 'nl_size_frozen.pkl'
TASTE_CACHE = 'nl_taste_frozen.pkl'

if __name__ == '__main__':
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else 'size'
    if which == 'size':
        grid = ([(f'ad_{s}', dict(size=s, variant='adoption'))
                 for s in [0.125, 0.25, 0.5, 0.75, 1.0]] +
                [(f'no_{s}', dict(size=s, variant='no_adoption'))
                 for s in [0.125, 0.25, 0.5, 0.75, 1.0]])
        run_grid(SIZE_CACHE, grid)
    elif which == 'taste':
        grid = [(f'ts_{ts}', dict(size=0.5, variant='adoption', taste=ts))
                for ts in [0.02, 0.05, 0.10, 0.20, 0.40]]
        run_grid(TASTE_CACHE, grid)


def build_fig9(out='output/fig9_nonlinearity.png'):
    """Assemble the E9 nonlinearity figure from the checkpointed sweeps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rs = load(SIZE_CACHE); rt = load(TASTE_CACHE)
    def pkr(v, s):
        m = v['m'][s]; return m['Np'] / m['Lp'] if abs(m['Lp']) > 1e-6 else float('nan')
    sizes = [0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75]
    ad_y = [pkr(rs[f'ad_{s}'], 'y') for s in sizes if f'ad_{s}' in rs]
    ad_dg = [pkr(rs[f'ad_{s}'], 'D_GREEN') for s in sizes if f'ad_{s}' in rs]
    sz_ad = [s for s in sizes if f'ad_{s}' in rs]
    no_sizes = [0.125, 0.25, 0.5, 0.75]
    no_y = [pkr(rs[f'no_{s}'], 'y') for s in no_sizes if f'no_{s}' in rs]
    sz_no = [s for s in no_sizes if f'no_{s}' in rs]
    tastes = [0.02, 0.05, 0.10, 0.20, 0.40]
    t_dg = [pkr(rt[f'ts_{ts}'], 'D_GREEN') for ts in tastes if f'ts_{ts}' in rt]
    t_y = [pkr(rt[f'ts_{ts}'], 'y') for ts in tastes if f'ts_{ts}' in rt]
    tv = [ts for ts in tastes if f'ts_{ts}' in rt]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.axhline(1, color='k', lw=0.6, ls=':')
    a1.plot(sz_ad, ad_dg, 'o-', color='C1', lw=2, label=r'$D^{G}$ (adoption on)')
    a1.plot(sz_ad, ad_y, 's-', color='C0', lw=2, label=r'output $y$ (adoption on)')
    a1.plot(sz_no, no_y, '^--', color='C3', lw=2,
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


if __name__ == '__main__' and len(__import__('sys').argv) > 1 and __import__('sys').argv[1] == 'fig':
    build_fig9()
