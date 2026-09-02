"""Consumption-equivalent variation."""
import numpy as np

N_BETA = 3


def _labor_disutility_dev(ss, irf):
    """First-order deviation of labour disutility v(n)."""
    vphi, frisch, n_ss = float(ss['vphi']), float(ss['frisch']), float(ss['n'])
    dn = np.asarray(irf['n'])
    return vphi * n_ss ** (1 / frisch) * dn


def cev_total(base_ss, pre_ss, irf, eis=None, T=None):
    """Total CEV of a scenario relative to a common reference steady state."""
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
    """CEV per beta type and averaged."""
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
