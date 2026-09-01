"""Nonlinear robustness of the welfare (CEV) numbers.

The CEVs in the paper are FIRST-ORDER in the shock size: `UTIL_i` along the
linear impulse response is d/d(eps) of aggregate flow utility (see
paper/notes/welfare_note.pdf). This runner recomputes each scenario's welfare on the
NONLINEAR perfect-foresight transition (solve_impulse_nonlinear: Newton on the
aggregate unknowns, household block re-solved exactly), where `UTIL_i` is the
exact aggregate of u(c) over the exact distribution, and compares:

    * CEV level, per scenario and per beta type:  chi_NL / chi_L
    * the policy ORDERING (transfer > flat > cap > LF, ETS ~ LF)

at reduced shock sizes SIZES (the nonlinear solver does not converge at the
headline size 1.0; App. B). Both economies of the ex-ante comparison
(baseline and ETS steady state) are covered.

Nonlinear ImpulseDicts from SSJ are DEVIATIONS from the steady state, so they
feed welfare.cev_total exactly like the linear ones. Cached per (scenario,
size) in cache_nl_cev/; delete to recompute. Runtime: one nonlinear solve is
minutes, so run this in the background (python -u run_nl_cev.py > nl_cev.txt).
"""
import os
import sys
import time
import pickle
import numpy as np

from core.model import (build_model, run, frozen_model, shock_price,
                   td_unknowns_targets)
from core.welfare import cev_total
from tools.latex_tables import write_table

# Inner sj.solved blocks (the retail-energy NKPC in particular) converge only
# linearly under the cap and exceed SSJ's class-level default maxit=30; per-block
# `options` do not reach them, so raise the class default in place (instances
# share the dict).
from sequence_jacobian.blocks.block import Block as _SJBlock
_SJBlock.solve_impulse_nonlinear_options['maxit'] = 500

H = 24
TAU_B = 0.10
NUMERAIRE, BOOKING = 'cpi', 'import'
OUT = 'paper/output'
CACHE = 'cache/cache_nl_cev'
SIZES = [0.25, 0.5, 0.75]
MAXIT, TOL = 200, 1e-8

# (label, policy, ets)
SCENARIOS = [
    ('Laissez-faire',    'none',          False),
    ('Ex-post cap',      'subsidy',       False),
    ('Slutsky transfer', 'transfer',      False),
    ('Flat transfer',    'transfer_flat', False),
    ('Ex-ante ETS',      'none',          True),
]


def _cache(key):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, key + '.pkl')


def _slim(d):
    return {k: np.asarray(v) for k, v in d.items()}


def solve_scenario(model, policy, ets, size):
    """Returns (ss, lin, nl) for one scenario at one shock size. ss is the
    scenario's own pre-crisis steady state; the CEV is later taken against the
    common baseline SS via welfare.cev_total."""
    key = f'{policy}_{"ets" if ets else "base"}_s{size}_{BOOKING}'
    f = _cache(key)
    if os.path.exists(f):
        with open(f, 'rb') as fh:
            return pickle.load(fh)
    kw = dict(shock_kind='price', policy=policy, shock_kwargs=dict(size=size),
              numeraire=NUMERAIRE, booking=BOOKING)
    if ets:
        kw.update(ets=True, ets_kwargs=dict(tau_b=TAU_B, recycle='rebate'))
    ss, lin = run(model, **kw)
    U, Tg = td_unknowns_targets(BOOKING, ets=ets)
    t0 = time.time()
    try:
        nl = model.solve_impulse_nonlinear(ss, U, Tg, shock_price(size=size),
                                           maxit=MAXIT, tol=TOL, verbose=False)
        nl = _slim(nl); ok = True
    except ValueError as e:               # no convergence
        print(f'  [{key}] nonlinear solve failed: {e}', flush=True)
        nl, ok = None, False
    print(f'  [{key}] nonlinear solve {"ok" if ok else "FAILED"} '
          f'in {time.time()-t0:.0f}s', flush=True)
    out = ({k: np.asarray(v) for k, v in ss.items() if np.ndim(v) == 0},
           _slim(lin), nl)
    with open(f, 'wb') as fh:
        pickle.dump(out, fh)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    model = build_model(NUMERAIRE, booking=BOOKING)
    sizes = SIZES if len(sys.argv) < 2 else [float(s) for s in sys.argv[1:]]

    # common reference: baseline SS (size-independent)
    ss_base = solve_scenario(model, 'none', False, sizes[0])[0]

    rows, results = [], {}
    for size in sizes:
        print(f'\n=== shock size {size} ===', flush=True)
        for lbl, policy, ets in SCENARIOS:
            ss, lin, nl = solve_scenario(model, policy, ets, size)
            mL, byL = cev_total(ss_base, ss, lin)
            if nl is None:
                results[(size, lbl)] = (mL, byL, np.nan, byL * np.nan)
                rows.append([f'{size}', lbl, f'{100*mL:.4f}', 'n.c.', '--', '--', '--', '--'])
                continue
            mN, byN = cev_total(ss_base, ss, nl)
            results[(size, lbl)] = (mL, byL, mN, byN)
            pkL = 100 * float(np.max(lin['D_GREEN'][:H]))
            pkN = 100 * float(np.max(nl['D_GREEN'][:H]))
            print(f'{lbl:<18s} CEV_L {100*mL:8.4f}  CEV_NL {100*mN:8.4f}  '
                  f'ratio {mN/mL:6.3f}   by type NL/L '
                  + ' '.join(f'{n/l:5.3f}' for n, l in zip(byN, byL))
                  + f'   peakDG L/NL {pkL:.2f}/{pkN:.2f}', flush=True)
            rows.append([f'{size}', lbl, f'{100*mL:.4f}', f'{100*mN:.4f}',
                         f'{mN/mL:.3f}'] + [f'{n/l:.3f}' for n, l in zip(byN, byL)])

        # ordering check
        lab = [s[0] for s in SCENARIOS]
        oL = sorted(lab, key=lambda l: -results[(size, l)][0])
        oN = sorted([l for l in lab if not np.isnan(results[(size, l)][2])],
                    key=lambda l: -results[(size, l)][2])
        print(f'ordering (best first)  linear:    {oL}')
        print(f'                       nonlinear: {oN}   '
              f'{"SAME" if oL[:len(oN)] == oN else "DIFFERS"}', flush=True)
        # the two fragile gaps
        for a, b in (('Ex-ante ETS', 'Laissez-faire'), ('Slutsky transfer', 'Ex-post cap'),
                     ('Flat transfer', 'Slutsky transfer')):
            gL = 100 * (results[(size, a)][0] - results[(size, b)][0])
            gN = 100 * (results[(size, a)][2] - results[(size, b)][2])
            print(f'  gap {a} - {b}: linear {gL:+.4f} pp   nonlinear {gN:+.4f} pp')

    write_table(
        os.path.join(OUT, f'tab_nl_cev_{BOOKING}.tex'), colspec='llrrrrrr',
        header=['size', 'Scenario', 'CEV\\% (linear)', 'CEV\\% (nonlinear)',
                'NL/L', 'impat.', 'middle', 'patient'],
        rows=rows,
        caption=('Welfare on the linear vs the nonlinear perfect-foresight '
                 'transition, brown-price shock at reduced sizes (import booking). '
                 'CEV is the total consumption-equivalent variation relative to the '
                 'no-tax baseline steady state; NL/L is the ratio of nonlinear to '
                 'linear CEV, overall and by discount-factor type. The headline '
                 'shock (size 1.0) does not converge nonlinearly.'),
        label=f'tab:nl_cev_{BOOKING}',
        midrule_after={len(SCENARIOS) * i - 1 for i in range(1, len(sizes))})
    with open(os.path.join(OUT, f'nl_cev_{BOOKING}.pkl'), 'wb') as fh:
        pickle.dump(results, fh)
    print(f'\n[table] {OUT}/tab_nl_cev_{BOOKING}.tex')


if __name__ == '__main__':
    main()
