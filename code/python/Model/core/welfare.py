"""Consumption-equivalent variation.

Utilitarian welfare for discount-factor type i is

    W_i = sum_t beta_i^t E[u(c_it) - v(n_t)]

where u(c_it) is the flow-utility hetoutput UTIL_i aggregated by SSJ over the
distribution at t, and v(n_t) = vphi * n_t^(1+1/frisch) / (1+1/frisch) is the
economy-wide labor disutility implied by the union's wage-setting FOC
(blocks.unions). Under the baseline calibration (ghh_prefs=0) preferences are
separable and v(n) never enters the household's own problem or UTIL_i, but it
is a genuine felicity cost shared by every household under indivisible labor
-- a scenario that raises employment is not a free lunch. v(n) is computed
here ex post from the cached IRF (no DAG change or re-solve is required):

Because the distribution evolves from its date-0 value, W_i is exactly the
expected discounted utility of a household drawn at random before the shock
-- the right object for a utilitarian planner.

With log utility a permanent proportional consumption change chi satisfies
    dW = log(1+chi)/(1-beta)   =>   chi = exp((1-beta) dW) - 1.
With CRRA(1/eis), u(c)=c^(1-1/eis)/(1-1/eis) scales by (1+chi)^(1-1/eis) under a
permanent proportional change chi, so [(1+chi)^(1-1/eis)-1] u_ss/(1-beta) = dW,
    chi = [1 + dW (1-beta) / u_ss] ** (1/(1-1/eis)) - 1,
with u_ss = UTIL_i^ss the SS felicity level (the normalisation 1/(1-1/eis) is
already inside u_ss, so it must NOT reappear in the bracket). Reduces to the log
expression as eis -> 1.

chi < 0 means the scenario is worse than the steady state: households would
pay |chi| of permanent consumption to avoid it.
"""
import numpy as np

N_BETA = 3


def _labor_disutility_dev(ss, irf):
    """First-order deviation of v(n) = vphi*n^(1+1/frisch)/(1+1/frisch).

    v'(n_ss) = vphi * n_ss^(1/frisch), so dv_t = v'(n_ss) * dn_t. Common to
    every household (indivisible labor, aggregate n), so it enters every
    type's welfare identically.
    """
    vphi, frisch, n_ss = float(ss['vphi']), float(ss['frisch']), float(ss['n'])
    dn = np.asarray(irf['n'])
    return vphi * n_ss ** (1 / frisch) * dn


def cev_total(base_ss, pre_ss, irf, eis=None, T=None):
    """Total CEV of a full scenario relative to a COMMON reference base_ss.

    A scenario = sit permanently at pre_ss (its pre-crisis steady state), then
    experience the transition in irf. Its welfare relative to base_ss is

        dW_i = (UTIL_i^pre - UTIL_i^base - dv_lvl_i)/(1-beta_i)
               + sum_t beta_i^t (UTIL_i^irf_t - dv_t)

    The first term is the STANDING gap between pre_ss and base_ss (0 when the
    scenario starts from base_ss, e.g. the ex-post cap); the second is the
    crisis transition (identical to `cev`'s integrand). dv is the shared labour
    disutility v(n): its standing level difference dv_lvl and its transition
    deviation dv_t both enter, so a scenario that runs the economy hotter in
    steady state or in the crisis is not counted as a free lunch.

    Returns (chi_mean, chi_by_type) relative to base_ss. Use this -- not `cev`
    -- to compare the ex-ante ETS (pre_ss = ETS SS) against the ex-post cap
    (pre_ss = base_ss) on one axis.
    """
    eis = float(base_ss['eis']) if eis is None else eis
    vphi = float(base_ss['vphi']); frisch = float(base_ss['frisch'])
    n_base = float(base_ss['n'])
    dv_lvl = vphi * n_base ** (1 / frisch) * (float(pre_ss['n']) - n_base)
    dv = _labor_disutility_dev(pre_ss, irf)
    chis = []
    for i in range(N_BETA):
        b = float(base_ss[f'beta_{i}'])
        u_base = float(base_ss[f'UTIL_{i}'])
        standing = (float(pre_ss[f'UTIL_{i}']) - u_base - dv_lvl) / (1 - b)
        du = np.asarray(irf[f'UTIL_{i}']) - dv
        T_ = len(du) if T is None else min(T, len(du))
        dW = standing + float(np.sum(b ** np.arange(T_) * du[:T_]))
        if eis == 1:
            chis.append(np.exp((1 - b) * dW) - 1)
        else:
            p = 1 - 1 / eis
            chis.append((1 + dW * (1 - b) / u_base) ** (1 / p) - 1)
    chis = np.array(chis)
    return float(chis.mean()), chis


def cev(ss, irf, eis=None, T=None):
    """CEV of the scenario in `irf`, per beta type and averaged.

    Returns (chi_mean, chi_by_type), both in units of permanent consumption
    (multiply by 100 for percent).
    """
    eis = float(ss['eis']) if eis is None else eis
    dv = _labor_disutility_dev(ss, irf)
    chis = []
    for i in range(N_BETA):
        b = float(ss[f'beta_{i}'])
        du = np.asarray(irf[f'UTIL_{i}']) - dv
        T_ = len(du) if T is None else min(T, len(du))
        dW = float(np.sum(b ** np.arange(T_) * du[:T_]))
        if eis == 1:
            chis.append(np.exp((1 - b) * dW) - 1)
        else:
            u_ss = float(ss[f'UTIL_{i}'])
            p = 1 - 1 / eis
            chis.append((1 + dW * (1 - b) / u_ss) ** (1 / p) - 1)
    chis = np.array(chis)
    return float(chis.mean()), chis


def cev_table(ss, irfs, labels=None):
    """Formatted CEV comparison across scenarios. `irfs` is a dict."""
    labels = labels or list(irfs)
    lines = [f"{'scenario':<22s} {'CEV mean':>10s} {'impatient':>10s} "
             f"{'middle':>10s} {'patient':>10s}",
             '-' * 66]
    out = {}
    for k in labels:
        m, byt = cev(ss, irfs[k])
        out[k] = (m, byt)
        lines.append(f"{k:<22s} {100*m:10.4f} " + " ".join(f"{100*x:10.4f}" for x in byt))
    return '\n'.join(lines), out
