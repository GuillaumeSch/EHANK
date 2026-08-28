"""Non-regression suite for E-HANK.

Check 1 (`test_targets`) lives in blocks.py because it is needed inside the
solver loop. Checks 2-5 are here.

    2. envelope        Va == dV/da, converging at O(h^2)
    3. monotonicity    a'(a) non-decreasing  (silent prerequisite of interpolate_y)
    4. stock-flow      D_SWITCH == delta_g * D_GREEN
    5. phase 2         channel shut => durable machinery is inert (ARS nesting)

Every check returns (passed, value) so a run can be logged rather than only
asserted.
"""
import numpy as np

import blocks as B


# =============================================================================
# 2. ENVELOPE CONDITION
# =============================================================================
def check_envelope(ss, block='hh_0', trim=2):
    """Compare the analytic Va against a central difference of V along a.

    V and Va are both indexed on the state entering `consav`, i.e. on incoming
    assets, so dV/da is the plain derivative along the last axis. The asset
    grid is log-spaced, hence the non-uniform central difference.
    """
    I = ss.internals[block]['consav']
    V, Va = np.asarray(I['V']), np.asarray(I['Va'])
    a = np.asarray(ss.internals[block]['a_grid'])
    hm, hp = a[1:-1] - a[:-2], a[2:] - a[1:-1]
    dV = (hm ** 2 * V[..., 2:] - hp ** 2 * V[..., :-2]
          - (hm ** 2 - hp ** 2) * V[..., 1:-1]) / (hm * hp * (hm + hp))
    sl = slice(trim, -trim) if trim else slice(None)
    num, ana = dV[..., sl], Va[..., 1:-1][..., sl]
    rel = np.abs(num - ana) / np.maximum(np.abs(ana), 1e-12)
    return float(np.max(rel)), float(np.median(rel))


def check_envelope_order(solve, make, n_a_list=(150, 300, 600), block='hh_0'):
    """Refine the grid and report the error ratio. O(h^2) => ratio ~ 4.

    `make` must be a calibration FACTORY, not a dict: cE_ss_grid is shaped
    (n_e, n_a) and has to be rebuilt when the grid is refined.
    """
    errs = []
    for n_a in n_a_list:
        _, med = check_envelope(solve(make(n_a=n_a)), block=block)
        errs.append(med)
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    return errs, ratios


# =============================================================================
# 3. MONOTONICITY OF THE SAVINGS POLICY
# =============================================================================
def check_monotonicity(ss, blocks=('hh_0', 'hh_1', 'hh_2')):
    """interpolate_y assumes the ENDOGENOUS grid is increasing; a violation is
    silent and produces garbage rather than an error."""
    worst = 0.0
    for b in blocks:
        a = np.asarray(ss.internals[b]['consav']['a'])
        d = np.diff(a, axis=-1)
        worst = min(worst, float(np.min(d)))
    return worst >= -1e-12, worst


# =============================================================================
# 4. STOCK-FLOW CONSISTENCY
# =============================================================================
def check_stock_flow(ss):
    res = float(ss['D_SWITCH']) - float(ss['delta_g']) * float(ss['D_GREEN'])
    return abs(res) < 1e-8, res


# =============================================================================
# 5. PHASE 2: CHANNEL SHUT => DURABLE BLOCK INERT
# =============================================================================
PHASE2_KEYS = ['y', 'C', 'A', 'cE', 'cH', 'cF', 'n', 'w', 'pH_P', 'nfa', 'vphi',
               'beta_max', 'MPC']


def check_phase2(solve, make, probes=({'pE_g_ratio': 0.8}, {'pE_g_ratio': 0.5},
                                      {'psi_g': 0.253}, {'psi_g': 1.0}),
                  unknowns_shut=None, targets_shut=None):
    """With `green_block` large nobody ever holds a green durable, so the
    steady state must be numerically independent of every durable parameter.

    This is the self-contained form of the ARS nesting test: it does not need
    the original notebook, only invariance of the ARS-side aggregates.

    Under `green_block=1e10`, D_GREEN is pinned at exactly 0 for every psi_g,
    so D_GREEN_res has zero derivative in psi_g and solving for psi_g against
    it is a singular Jacobian, not just numerically fragile (see model.py's
    `no_adoption` handling). psi_g must be dropped from the unknowns/targets
    here; `unknowns_shut`/`targets_shut` default to model.py's
    SS_UNKNOWNS_FIXED_PSI / SS_TARGETS_FIXED_PSI when not supplied.
    """
    if unknowns_shut is None or targets_shut is None:
        import model as M
        unknowns_shut = unknowns_shut or M.SS_UNKNOWNS_FIXED_PSI
        targets_shut = targets_shut or M.SS_TARGETS_FIXED_PSI
    ref, out = None, []
    for p in probes:
        cal = make(green_block=1e10, **p)
        ss = solve(cal, unknowns=unknowns_shut, targets=targets_shut)
        v = np.array([float(ss[k]) for k in PHASE2_KEYS])
        if ref is None:
            ref = v
        out.append(float(np.max(np.abs(v - ref))))
    return max(out) < 1e-7, max(out)


# =============================================================================
# DRIVER
# =============================================================================
def run_all(ss, solve=None, make=None, irf=None, label='', full=False):
    print(f"--- non-regression {label}")
    B.test_targets(ss)
    print("  1. test_targets(ss)          PASS")
    if irf is not None:
        B.test_targets(irf)
        print("  1'. test_targets(irf)        PASS")
    mx, md = check_envelope(ss)
    print(f"  2. envelope Va=dV/da         max {mx:.2e}  median {md:.2e}")
    ok, w = check_monotonicity(ss)
    print(f"  3. monotonic a'              {'PASS' if ok else 'FAIL'}  min diff {w:.2e}")
    ok, res = check_stock_flow(ss)
    print(f"  4. D_SWITCH = delta_g*D_GREEN {'PASS' if ok else 'FAIL'}  resid {res:.2e}")
    if full and solve is not None:
        ok, d = check_phase2(solve, make)
        print(f"  5. phase 2 (channel shut)    {'PASS' if ok else 'FAIL'}  max dev {d:.2e}")
        errs, ratios = check_envelope_order(solve, make)
        print(f"  2'. envelope order           errs {['%.2e' % e for e in errs]}"
              f"  ratios {['%.2f' % r for r in ratios]}")
