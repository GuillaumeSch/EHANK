"""Consumption-equivalent variation.

Headline welfare is the aggregate CEV chi_star: the common permanent consumption
supplement, granted to every type in the reference steady state, that reproduces the
equal-weighted (utilitarian) sum of type-level welfare changes. This aggregates in
common consumption units, unlike a mean of per-type CEV shares (each of which is a
share of a different type's consumption); see Davila and Schaab (2025). Per-type
chi_i are returned alongside for the distributional report.
"""
import numpy as np

N_BETA = 3


def _labor_disutility_dev(ss, irf):
    """First-order deviation of labour disutility v(n)."""
    vphi, frisch, n_ss = float(ss['vphi']), float(ss['frisch']), float(ss['n'])
    dn = np.asarray(irf['n'])
    return vphi * n_ss ** (1 / frisch) * dn


def _chi_from_dW(dW, b, u_ss, eis):
    """Per-type CEV from a type's welfare deviation dW."""
    if eis == 1:
        return np.exp((1 - b) * dW) - 1
    p = 1 - 1 / eis
    return (1 + dW * (1 - b) / u_ss) ** (1 / p) - 1


def _chi_star(dWs, betas, u_sss, eis):
    """Aggregate CEV in common consumption units (equal type weights)."""
    dW = float(np.sum(dWs))
    if eis == 1:
        denom = sum(1.0 / (1 - b) for b in betas)
        return np.exp(dW / denom) - 1
    p = 1 - 1 / eis
    denom = sum(u / (1 - b) for u, b in zip(u_sss, betas))
    return (1 + dW / denom) ** (1 / p) - 1


def cev_total(base_ss, pre_ss, irf, eis=None, T=None):
    """Total CEV of a scenario relative to a common reference steady state.

    Returns (chi_star, chis): the aggregate CEV and the per-type CEVs.
    """
    eis = float(base_ss['eis']) if eis is None else eis
    vphi = float(base_ss['vphi']); frisch = float(base_ss['frisch'])
    n_base = float(base_ss['n'])
    dv_lvl = vphi * n_base ** (1 / frisch) * (float(pre_ss['n']) - n_base)
    dv = _labor_disutility_dev(pre_ss, irf)
    dWs, u_sss, betas, chis = [], [], [], []
    for i in range(N_BETA):
        b = float(base_ss[f'beta_{i}'])
        u_base = float(base_ss[f'UTIL_{i}'])
        standing = (float(pre_ss[f'UTIL_{i}']) - u_base - dv_lvl) / (1 - b)
        du = np.asarray(irf[f'UTIL_{i}']) - dv
        T_ = len(du) if T is None else min(T, len(du))
        dW = standing + float(np.sum(b ** np.arange(T_) * du[:T_]))
        dWs.append(dW); u_sss.append(u_base); betas.append(b)
        chis.append(_chi_from_dW(dW, b, u_base, eis))
    chi_star = _chi_star(dWs, betas, u_sss, eis)
    return float(chi_star), np.array(chis)


def cev(ss, irf, eis=None, T=None):
    """CEV relative to the scenario's own steady state.

    Returns (chi_star, chis): the aggregate CEV and the per-type CEVs.
    """
    eis = float(ss['eis']) if eis is None else eis
    dv = _labor_disutility_dev(ss, irf)
    dWs, u_sss, betas, chis = [], [], [], []
    for i in range(N_BETA):
        b = float(ss[f'beta_{i}'])
        u_ss = float(ss[f'UTIL_{i}'])
        du = np.asarray(irf[f'UTIL_{i}']) - dv
        T_ = len(du) if T is None else min(T, len(du))
        dW = float(np.sum(b ** np.arange(T_) * du[:T_]))
        dWs.append(dW); u_sss.append(u_ss); betas.append(b)
        chis.append(_chi_from_dW(dW, b, u_ss, eis))
    chi_star = _chi_star(dWs, betas, u_sss, eis)
    return float(chi_star), np.array(chis)


def cev_table(ss, irfs, labels=None):
    """Formatted CEV comparison across scenarios. `irfs` is a dict."""
    labels = labels or list(irfs)
    lines = [f"{'scenario':<22s} {'CEV*':>10s} {'impatient':>10s} "
             f"{'middle':>10s} {'patient':>10s}",
             '-' * 66]
    out = {}
    for k in labels:
        m, byt = cev(ss, irfs[k])
        out[k] = (m, byt)
        lines.append(f"{k:<22s} {100*m:10.4f} " + " ".join(f"{100*x:10.4f}" for x in byt))
    return '\n'.join(lines), out
