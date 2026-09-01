"""Identification of the logit taste scale sigma_eps.

psi_g and sigma_eps are not separately identified by the single steady-state
target D_GREEN_ss = 5%: for any sigma_eps, psi_g re-solves to hit the same 5%
share, so the steady state is invariant along the (sigma_eps, psi_g) locus. What
the locus does NOT leave invariant is the RESPONSIVENESS of the margin, which a
second moment---an adoption elasticity---pins down.

This runner traces the locus and reports, at each sigma_eps (with psi_g
recalibrated to 5%):
  * the steady-state adoption elasticity: the response of the green share to a
    permanent improvement in the green/brown operating-cost ratio (pE_g_ratio),
    computed at FIXED psi_g via the common-SS 'fixed-psi' configuration
    (ss_unknowns_targets_fixed_psi). This is the object to confront with the
    micro estimates of Beresteanu--Li (2011) and Muehlegger--Rapson (2022).
  * the crisis response: peak dD_GREEN to the baseline brown-price shock.
  * the aggregate output loss (Sum y over H): near-invariant, which is why the
    paper's macro conclusions and policy ordering do not turn on sigma_eps.

Emits tab_taste_identification_<booking>.tex, fig_taste_identification.png.
Runtime: one SS solve is ~15-25s, so this is a few minutes; run locally.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from core.model import (build_model, run, solve_ss,
                   SS_UNKNOWNS_FIXED_PSI, SS_TARGETS_FIXED_PSI)
from core.calibration import make_calibration
from tools.latex_tables import write_table

H = 24
NUMERAIRE, BOOKING = 'cpi', 'import'
OUT = 'paper/output'
SIGMAS = [0.02, 0.035, 0.05, 0.07, 0.10, 0.15, 0.20]
# Dollar mapping for the subsidy moment: one model unit = quarterly household
# consumption. Stated as an explicit, adjustable assumption in the paper.
QUARTERLY_C_USD = 20_000
BASE_RATIO = 0.80          # pE_g_ratio at calibration
DRATIO = 0.01              # perturbation for the SS elasticity
SAVING0 = 1.0 - BASE_RATIO  # baseline green operating-cost advantage


def _flow_fixed(model, sigma, psi):
    cal = make_calibration(NUMERAIRE, BOOKING, taste_shock=sigma, psi_g=psi,
                           pE_g_ratio=BASE_RATIO)
    ss = solve_ss(model, cal, unknowns=SS_UNKNOWNS_FIXED_PSI,
                  targets=SS_TARGETS_FIXED_PSI)
    return float(ss['D_SWITCH']) / (1 - float(ss['D_GREEN']))


def _DG_fixed(model, sigma, ratio, psi):
    cal = make_calibration(NUMERAIRE, BOOKING, taste_shock=sigma, psi_g=psi,
                           pE_g_ratio=ratio)
    ss = solve_ss(model, cal, unknowns=SS_UNKNOWNS_FIXED_PSI,
                  targets=SS_TARGETS_FIXED_PSI)
    return float(ss['D_GREEN'])


def sweep(model):
    rows = []
    for sig in SIGMAS:
        ss, irf = run(model, shock_kind='price', policy='none', taste_shock=sig)
        psi = float(ss['psi_g'])
        dgss = float(ss['D_GREEN'])
        # SS adoption elasticity: d ln(D_GREEN) / d ln(saving) at fixed psi
        d0 = _DG_fixed(model, sig, BASE_RATIO, psi)
        dm = _DG_fixed(model, sig, BASE_RATIO - DRATIO, psi)
        elast = np.log(dm / d0) / np.log((SAVING0 + DRATIO) / SAVING0)
        # upfront-cost margin: semi-elasticity of the switching flow to a $1,000
        # cut in psi_g (the moment matched to the vehicle-subsidy literature)
        f0 = _flow_fixed(model, sig, psi)
        f1 = _flow_fixed(model, sig, psi * 0.99)
        per1000 = 100 * (np.exp(np.log(f1 / f0) / (0.01 * psi * QUARTERLY_C_USD) * 1000) - 1)
        prem_usd = psi * QUARTERLY_C_USD
        peak = 100 * float(np.max(irf['D_GREEN'][:H]))
        sy = 100 * float(np.sum(irf['y'][:H]))
        rows.append(dict(sigma=sig, psi=psi, dgss=100 * dgss, elast=elast,
                         per1000=per1000, prem=prem_usd, peak=peak, sy=sy))
        print(f"sigma={sig:.3f}  psi_g={psi:.4f}  D_G_ss={100*dgss:.3f}%  "
              f"elast={elast:.3f}  peakDG={peak:.2f}  Sy={sy:.1f}", flush=True)
    return rows


def make_figure(rows, out):
    sig = [r['sigma'] for r in rows]
    el = [r['elast'] for r in rows]
    pk = [r['peak'] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(sig, el, 'o-', color='C0', lw=2)
    ax[0].axvline(0.05, color='grey', ls='--', lw=1)
    ax[0].set_xlabel(r'logit taste scale $\sigma_\varepsilon$')
    ax[0].set_ylabel(r'SS adoption elasticity (share vs cost gap)')
    ax[0].set_title('Identifying moment')
    ax[1].plot(sig, pk, 's-', color='C1', lw=2)
    ax[1].axvline(0.05, color='grey', ls='--', lw=1)
    ax[1].set_xlabel(r'logit taste scale $\sigma_\varepsilon$')
    ax[1].set_ylabel(r'peak $\Delta D^{G}$ (pp), price shock')
    ax[1].set_title('Crisis adoption magnitude')
    for a in ax:
        a.tick_params(labelsize=9)
    fig.suptitle(r'The steady state is invariant along the $(\sigma_\varepsilon,\psi_g)$ '
                 r'locus; the adoption response is not', y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)


def switch_prob_table(model, out_tex):
    """Distribution-weighted quantiles of the quarterly switch probability among
    brown incumbents, by discount-factor type. Documents the rare-margin regime
    of Appendix app:taste: essentially all brown mass sits in the exponential
    tail of the logit, so the semi-elasticity of the switching flow equals
    1/sigma_eps household by household and cannot be attenuated by heterogeneity."""
    import household as hh_mod
    ss, _ = run(model, shock_kind='price', policy='none')
    greens = np.where(hh_mod.IS_GREEN > 0)[0]
    browns = np.where(hh_mod.IS_GREEN == 0)[0]

    blocks = sorted(k for k in ss.internals if k.startswith('hh_'))
    betas = {b: float(ss[f"beta_{b.split('_')[1]}"]) for b in blocks}
    order = sorted(blocks, key=lambda b: betas[b])
    labels = ['impatient', 'middle', 'patient'][:len(order)]

    def wquant(p, w, q):
        idx = np.argsort(p); cw = np.cumsum(w[idx])
        return float(p[idx][np.searchsorted(cw, q * cw[-1])])

    rows, pool_p, pool_w = [], [], []
    for lab, b in zip(labels, order):
        inter = ss.internals[b]
        P = inter['durables']['law_of_motion'].P
        Dd = inter['durables']['D']
        Psw = P[greens].sum(axis=0)
        p = np.concatenate([Psw[d].ravel() for d in browns])
        w = np.concatenate([Dd[d].ravel() for d in browns])
        mass = w.sum()
        rows.append([lab, f'{betas[b]:.3f}', f'{100*mass:.1f}',
                     f'{100*float((w/mass*p).sum()):.2f}',
                     f'{100*wquant(p, w, 0.50):.2f}', f'{100*wquant(p, w, 0.95):.2f}',
                     f'{100*wquant(p, w, 0.999):.2f}', f'{100*float(p[w>0].max()):.1f}'])
        pool_p.append(p); pool_w.append(w / len(order))   # types have weight 1/3
    p, w = np.concatenate(pool_p), np.concatenate(pool_w)
    mass = w.sum()
    rows.append([r'\textbf{all}', '--', f'{100*mass:.1f}',
                 f'{100*float((w/mass*p).sum()):.2f}',
                 f'{100*wquant(p, w, 0.50):.2f}', f'{100*wquant(p, w, 0.95):.2f}',
                 f'{100*wquant(p, w, 0.999):.2f}', f'{100*float(p[w>0].max()):.1f}'])

    write_table(
        out_tex, colspec='llrrrrrr',
        header=['type', r'$\beta$', r'mass\%', 'mean', 'p50', 'p95', 'p99.9', 'max'],
        rows=rows,
        caption=(r'Quarterly switch probabilities among brown incumbents '
                 r'(distribution-weighted, baseline steady state). Mean, median '
                 r'and upper quantiles of $P(\mathrm{switch})$ in \%, by '
                 r'discount-factor type and pooled. The mean equals the '
                 r'steady-state flow identity $\delta_g \bar D^G/(1-\bar D^G)'
                 r'=0.26\%$; virtually no mass approaches interior probabilities, '
                 r'so the brown population sits in the exponential tail of the '
                 r'logit (Appendix~\ref{app:taste}).'),
        label='tab:switch_prob_dist', midrule_after={len(order) - 1})
    print(f'[table] {out_tex}')


def main():
    os.makedirs(OUT, exist_ok=True)
    model = build_model(NUMERAIRE, booking=BOOKING)
    switch_prob_table(model, os.path.join(OUT, f'tab_switch_prob_dist_{BOOKING}.tex'))
    rows = sweep(model)
    trows = [[f"{r['sigma']:.3f}", f"{r['psi']:.3f}",
              f"{r['prem']/1000:.1f}", f"{r['elast']:.2f}", f"{r['per1000']:.0f}",
              f"{r['peak']:.2f}", f"{r['sy']:.1f}"] for r in rows]
    write_table(
        os.path.join(OUT, f'tab_taste_identification_{BOOKING}.tex'),
        colspec='rrrrrrr',
        header=[r'$\sigma_\varepsilon$', r'$\psi_g$', r'premium \$000',
                r'cost-gap elast.', r'\%/\$1{,}000', r'peak $\Delta D^G$',
                r'$\sum y$'],
        rows=trows,
        caption=(r'Identification of the logit taste scale. For each '
                 r'$\sigma_\varepsilon$, $\psi_g$ is recalibrated to hold '
                 r'$D^G_{ss}=5\%$ (steady state invariant). Columns: the implied '
                 r'green premium in dollars (one model unit = quarterly household '
                 r'consumption of \$20{,}000); the steady-state elasticity of the '
                 r'green share to the operating-cost gap at fixed $\psi_g$; the '
                 r'semi-elasticity of the switching flow to a \$1{,}000 cut in '
                 r'$\psi_g$ (matched to \citealp{GallagherMuehlegger2011,Chandra2010,'
                 r'MuehleggerRapson2022}); the peak crisis adoption response; and '
                 r'the cumulative output loss, which is flat while the adoption '
                 r'responsiveness varies by an order of magnitude.'),
        label=f'tab:taste_identification')
    make_figure(rows, os.path.join(OUT, 'fig_taste_identification.png'))
    print(f"\n[table] {OUT}/tab_taste_identification_{BOOKING}.tex")
    print(f"[figure] {OUT}/fig_taste_identification.png")


if __name__ == '__main__':
    main()
