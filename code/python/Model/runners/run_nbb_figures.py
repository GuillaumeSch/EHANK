"""Steady-state adoption panel and consumption-inequality figures.

Produces:
  fig_ss_adoption.pdf         (a) SS adoption prob. by wealth/productivity,
                                 (b) D_GREEN vs psi_g, (c) D_GREEN vs carbon tax
  fig_adoption_dynamics.pdf   green-adoption probability by wealth, steady state vs
                                 quarter 0, all productivity groups
  fig_cons_variance.pdf       variance of log consumption, both shocks, four
                                 fiscal responses
  fig_cons_percapita.pdf      per-capita consumption of green vs brown users, four
                                 fiscal responses

Run:  python -m runners.run_nbb_figures
Needs the household 'inequality' hetoutput (core/household.py), which exposes
LOGC_i, LOGC2_i per discount-factor group.

Inequality measure: within-minus-between variance of log consumption across the
three discount-factor groups,
  V(t) = mean_g[ LOGC2_g(t) - LOGC_g(t)^2 ] - var_g[ LOGC_g(t) ],
on level paths (steady state + linear impulse), plotted as its deviation from the
steady state, times 100.
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

from core.model import (build_model, run, solve_ss,
                        ss_unknowns_targets_fixed_psi)
from core.calibration import make_calibration

NUM, BOOK, H = 'cpi', 'import', 21

POLS = ('none', 'subsidy', 'transfer', 'transfer_flat')
# Single-hue blue + line styles, consistent with the fiscal IRF figures.
POL_COLOR = {'none': 'tab:blue', 'subsidy': 'tab:blue',
             'transfer': 'tab:blue', 'transfer_flat': 'tab:blue'}
POL_LS    = {'none': '-', 'subsidy': '--', 'transfer': ':', 'transfer_flat': '-.'}
POL_ALPHA = {'none': 1.0, 'subsidy': 0.9, 'transfer': 0.8, 'transfer_flat': 0.6}
POL_LABEL = {'none': 'No policy', 'subsidy': 'Energy subsidy',
             'transfer': 'Targeted transfer', 'transfer_flat': 'Untargeted transfer'}
LW = 2.6

plt.rcParams.update({
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 11, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'legend.frameon': True, 'legend.framealpha': 0.85, 'figure.dpi': 120,
})


# =============================================================== steady state
def ss_probability_panel(ax):
    ss = run(build_model(NUM, booking=BOOK), shock_kind='price', policy='none',
             model_variant='adoption')[0]
    P = ss.internals['hh_0']['durables']['law_of_motion'].P   # (choice, from, e, a)
    a = ss.internals['hh_1']['a_grid']
    for e, sty, lab in [(0, dict(color='#08519c', ls='-'), 'Low productivity'),
                        (3, dict(color='#4292c6', ls='--'), 'Median productivity'),
                        (-1, dict(color='#9ecae1', ls=':'), 'High productivity')]:
        ax.plot(a, 100 * P[2, 0, e, :], lw=2.4, **sty, label=lab)
    ax.set_xlabel('Individual assets'); ax.set_ylabel('Prob. of adopting green (%)')
    ax.set_title('(a) Adoption probability, by wealth'); ax.legend(loc='best')


def _dgreen(model, u, t, **ov):
    ss = solve_ss(model, make_calibration(NUM, booking=BOOK, **ov),
                  unknowns=u, targets=t, booking=BOOK)
    return float(ss['D_GREEN'])


def psi_panel(ax, n=25):
    m = build_model(NUM, booking=BOOK)
    u, t = ss_unknowns_targets_fixed_psi(BOOK)
    psi0 = float(solve_ss(m, make_calibration(NUM, booking=BOOK), booking=BOOK)['psi_g_bar'])
    grid = np.linspace(0.45 * psi0, 1.55 * psi0, n)
    dg = np.array([_dgreen(m, u, t, psi_g_bar=float(p)) for p in grid])
    ax.plot(grid, 100 * dg, color='#0072B2', lw=2.4)
    ax.plot([psi0], [5.0], 'o', color='#D55E00', ms=6, zorder=5,
            markeredgecolor='white', markeredgewidth=0.8, label='Baseline')
    ax.set_xlabel(r'Durable size $\overline{d}_g$'); ax.set_ylabel('SS green share (%)')
    ax.set_title('(b) Adoption vs durable size'); ax.legend(loc='best')


def carbon_panel(ax, tb_max=0.35, n=15):
    m0 = build_model(NUM, booking=BOOK, ets=False)
    psi0 = float(solve_ss(m0, make_calibration(NUM, booking=BOOK), booking=BOOK)['psi_g_bar'])
    m = build_model(NUM, booking=BOOK, ets=True)
    u, t = ss_unknowns_targets_fixed_psi(BOOK, ets=True)
    grid = np.linspace(0.0, tb_max, n)
    dg = []
    for tb in grid:
        try:
            dg.append(_dgreen(m, u, t, ets=True, tau_b=float(tb), psi_g_bar=psi0))
        except Exception:
            dg.append(np.nan)   # bracketing solvers fail past tau_b ~ 0.4
    dg = np.array(dg)
    ax.plot(100 * grid, 100 * dg, color='#0072B2', lw=2.4)
    ax.plot([0.0], [100 * dg[0]], 'o', color='#D55E00', ms=6, zorder=5,
            markeredgecolor='white', markeredgewidth=0.8, label='Baseline')
    ax.set_xlabel(r'Steady-state carbon price $\tau^b_{ss}$ (%)'); ax.set_ylabel('SS green share (%)')
    ax.set_title('(c) Adoption vs carbon price'); ax.legend(loc='best')


def fig_steady_state():
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ss_probability_panel(ax[0]); psi_panel(ax[1]); carbon_panel(ax[2])
    fig.tight_layout(); fig.savefig('fig_ss_adoption.pdf', bbox_inches='tight')
    return fig


def adoption_prob_paths(irf, ss, quarters, eps=1e-4):
    """First-order (linear) response of the green-adoption choice probability
    P[GB<-BB](e, a) at each date: the directional derivative of the logit choice
    along the GE aggregate paths, P_ss + (P(eps*paths) - P_ss)/eps. Consistent
    with the linear GE. The stage block does not expose these internals through
    impulse, so we capture the law-of-motion path from backward_nonlinear."""
    from core import household as _H
    blk = (_H.hh_one.rename(suffix='_0')
           .remap({x: f'{x}_0' for x in _H.GROUP_VARS}).remap({'beta_g': 'beta_0'}))
    moving = [k for k in blk.inputs if k in irf.keys()
              and np.max(np.abs(np.asarray(irf[k]))) > 1e-12]
    cap = {}
    orig = blk.backward_nonlinear
    def _wrap(s, i):
        rp, lp = orig(s, i); cap['lom'] = lp; return rp, lp
    blk.backward_nonlinear = _wrap
    blk.impulse_nonlinear(ss, {k: eps * np.asarray(irf[k]) for k in moving})
    blk.backward_nonlinear = orig
    dur = 2   # stage order: dep, prod, durables, consav
    Pss = np.asarray(ss.internals['hh_0']['durables']['law_of_motion'].P)
    return {t: Pss + (np.asarray(cap['lom'][t][dur].P) - Pss) / eps for t in quarters}


def fig_adoption_dynamics():
    m = build_model(NUM, booking=BOOK)
    ss, irf = run(m, shock_kind='price', policy='none', model_variant='adoption')
    Pss = np.asarray(ss.internals['hh_0']['durables']['law_of_motion'].P)
    a = np.asarray(ss.internals['hh_1']['a_grid'])
    P0 = adoption_prob_paths(irf, ss, (0,))[0]   # linear response at impact
    padopt = lambda P, e: 100 * P[2, 0, e, :]    # choice GB, from BB
    groups = [(0, '#9ecae1', 'Low'), (3, '#4292c6', 'Median'), (-1, '#08519c', 'High')]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for e, c, _ in groups:
        ax.plot(a, padopt(Pss, e), color=c, ls='--', lw=2.2)   # before (steady state)
        ax.plot(a, padopt(P0, e), color=c, ls='-', lw=2.2)     # after (quarter 0)
    ax.set_xlim(0, 50); ax.set_xlabel('Individual assets')
    ax.set_ylabel('Probability of adjusting (%)')
    ax.set_title('Clean adoption probability after the price shock')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    col_h = [Line2D([], [], color=c, lw=2.2, label=f'{lab} productivity') for _, c, lab in groups]
    sty_h = [Line2D([], [], color='0.35', lw=2.2, ls='--', label='Steady state'),
             Line2D([], [], color='0.35', lw=2.2, ls='-', label='Quarter 0')]
    leg1 = ax.legend(handles=col_h, loc='upper left', fontsize=8); ax.add_artist(leg1)
    ax.legend(handles=sty_h, loc='lower right', fontsize=8)
    fig.tight_layout(); fig.savefig('fig_adoption_dynamics.pdf', bbox_inches='tight')
    return fig


# =============================================================== inequality
def build_irfs(**extra):
    """IRFs for the four fiscal responses, both shocks. Pass extra overrides
    to run (e.g. green_block=20.0 for the brown, adoption-off economy)."""
    m = build_model(NUM, booking=BOOK)
    return {(shock, pol): run(m, shock_kind=shock, policy=pol,
                              model_variant='adoption', **extra)
            for shock in ('price', 'supply') for pol in POLS}


def varlogc_dev(ss, irf):
    """Within-minus-between variance of log consumption, deviation from SS."""
    L = [float(ss[f'LOGC_{i}']) + np.asarray(irf[f'LOGC_{i}']) for i in range(3)]
    L2 = [float(ss[f'LOGC2_{i}']) + np.asarray(irf[f'LOGC2_{i}']) for i in range(3)]
    V = np.mean([L2[i] - L[i] ** 2 for i in range(3)], axis=0) - np.var(L, axis=0)
    Vss = (np.mean([float(ss[f'LOGC2_{i}']) - float(ss[f'LOGC_{i}']) ** 2 for i in range(3)])
           - np.var([float(ss[f'LOGC_{i}']) for i in range(3)]))
    return 100 * (V - Vss)


def fig_variance(irfs, fname='fig_cons_variance.pdf', suptitle='Var. of log consumption'):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (shock, title) in zip(axes, [('price', '(a) Price shock'),
                                         ('supply', '(b) Supply shock')]):
        for pol in POLS:
            ss, irf = irfs[(shock, pol)]
            ax.plot(varlogc_dev(ss, irf)[:H], color=POL_COLOR[pol], ls=POL_LS[pol],
                    alpha=POL_ALPHA[pol], lw=LW, label=POL_LABEL[pol])
        ax.axhline(0, color='k', lw=0.6, zorder=0); ax.set_xlabel('quarter')
        ax.set_title(title); ax.set_ylabel('Percent')
        ax.xaxis.set_major_locator(MultipleLocator(4))
    axes[0].legend(loc='lower right')
    fig.suptitle(suptitle)
    fig.tight_layout(); fig.savefig(fname, bbox_inches='tight')
    return fig


def _pct(irf, key, ss):
    return 100 * np.asarray(irf[key]) / float(ss[key])


def fig_percapita(irfs, shock='price'):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (key, title) in zip(axes, [('C_GREEN_PC', '(a) Green users'),
                                       ('C_BROWN_PC', '(b) Fossil users')]):
        for pol in POLS:
            ss, irf = irfs[(shock, pol)]
            ax.plot(_pct(irf, key, ss)[:H], color=POL_COLOR[pol], ls=POL_LS[pol],
                    alpha=POL_ALPHA[pol], lw=LW, label=POL_LABEL[pol])
        ax.axhline(0, color='k', lw=0.6, zorder=0); ax.set_xlabel('quarter'); ax.set_title(title)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.xaxis.set_major_locator(MultipleLocator(4))
    axes[0].set_ylabel('Per-capita consumption, % dev. from SS')
    axes[0].legend(loc='upper right')
    fig.suptitle('Per-capita consumption response, by durable technology')
    fig.tight_layout(); fig.savefig('fig_cons_percapita.pdf', bbox_inches='tight')
    return fig


if __name__ == '__main__':
    fig_steady_state()
    fig_adoption_dynamics()
    irfs = build_irfs()
    fig_variance(irfs)
    fig_percapita(irfs, shock='price')
    print('done')
