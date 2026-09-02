"""Share-weighted energy deflator, computed OUTSIDE the DAG.

The model's CPI anchors its energy leg on the BROWN price alone (see
CESprices, Option C). The economically correct index reweights brown and green
by the adoption share:

    P_E,true^(1-eta_E) = (1-D_G) * P_B^(1-eta_E) + D_G * P_G^(1-eta_E)

Because CESprices enforces, at EVERY date,

    alpha_E * pE_B_P^(1-eta_E) + (1-alpha_E) * pHF_P^(1-eta_E) = 1,

the ratio of the true CPI to the model's CPI has a closed form that does not
involve pHF_P at all:

    phi^(1-eta_E) = 1 + alpha_E * D_G * (pE_G_P^(1-eta_E) - pE_B_P^(1-eta_E))

phi < 1 whenever green energy is cheaper: the model's CPI OVERSTATES the cost
of living, by construction, and the overstatement grows with adoption.

This is a pure post-processing step: it reads a solved steady state and a
solved impulse response, closes no loop, and therefore introduces no DAG
cycle. Nothing here feeds back into the monetary rule -- that is precisely
the approximation being measured.

Levels are reconstructed as ss + deviation, so the computation is EXACT
(non-linearised) given the linearised paths of its three arguments.
"""
import numpy as np


def _lvl(ss, irf, k):
    return float(ss[k]) + np.asarray(irf[k])


def phi_path(ss, irf):
    """True CPI / model CPI, and its steady-state value."""
    aE, eE = float(ss['alpha_E']), float(ss['eta_E'])
    pB, pG, DG = _lvl(ss, irf, 'pE_B_P'), _lvl(ss, irf, 'pE_G_P'), _lvl(ss, irf, 'D_GREEN')
    pB_ss, pG_ss, DG_ss = float(ss['pE_B_P']), float(ss['pE_G_P']), float(ss['D_GREEN'])

    def f(pb, pg, dg):
        if eE == 1:
            return (pg / pb) ** (aE * dg)
        return (1 + aE * dg * (pg ** (1 - eE) - pb ** (1 - eE))) ** (1 / (1 - eE))

    return f(pB, pG, DG), f(pB_ss, pG_ss, DG_ss)


def true_inflation(ss, irf):
    """Return a dict of deviations, all comparable to the model's own IRFs.

        phi_gap      100 * log(phi_t / phi_ss)      price-LEVEL measurement gap, %
        pi_true      quarterly true inflation, deviation from ss
        pi_true_ann  annualised, deviation from ss
        pi_gap_ann   pi_true_ann - pi_ann           what the Taylor rule misses
    """
    phi, phi_ss = phi_path(ss, irf)
    pi = np.asarray(irf['pi'])                      # pi_ss = 0
    lphi = np.log(phi)
    lphi_lag = np.concatenate(([np.log(phi_ss)], lphi[:-1]))
    pi_true = (1 + pi) * np.exp(lphi - lphi_lag) - 1
    pi_true_ann = (1 + pi_true) ** 4 - 1
    return {'phi_gap': 100 * (lphi - np.log(phi_ss)),
            'pi_true': pi_true,
            'pi_true_ann': pi_true_ann,
            'pi_gap_ann': pi_true_ann - np.asarray(irf['pi_ann']),
            'phi_ss': phi_ss}


def report(ss, irf, label='', h=24):
    d = true_inflation(ss, irf)
    pa = np.asarray(irf['pi_ann'])[:h]
    print(f"  {label:<28s} phi_ss={d['phi_ss']:.6f}"
          f"  max|level gap|={np.max(np.abs(d['phi_gap'][:h])):.4f}%"
          f"  max|pi gap, ann|={100*np.max(np.abs(d['pi_gap_ann'][:h])):.4f}pp"
          f"  (max|pi_ann|={100*np.max(np.abs(pa)):.2f}pp,"
          f" ratio={np.max(np.abs(d['pi_gap_ann'][:h]))/max(np.max(np.abs(pa)),1e-16):.2%})")
    return d
