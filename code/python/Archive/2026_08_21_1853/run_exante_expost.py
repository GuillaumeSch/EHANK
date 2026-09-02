"""Version A -- ex-ante greening vs ex-post capping (steady-state comparison).

The policy question: to shield households from a transitory brown-energy price
crisis, is it better to (EA) prepare EX ANTE with a permanent carbon tax that
leaves the economy greener and less exposed, or (EP) respond EX POST with an
energy price cap during the crisis?

Both economies are hit by the SAME price shock. Welfare is the total CEV of the
full scenario relative to a COMMON reference -- the no-ETS baseline steady state
-- so the standing cost/benefit of the permanent ETS and the crisis transition
are on one axis (welfare.cev_total). We report gross fiscal disbursement
alongside welfare: the cap's crisis outlay vs the ETS's standing carbon revenue,
so the ex-ante economy is not scored as a free lunch.

Claim on CONSUMPTION / WELFARE (CEV by discount-factor type), not output: the
adoption channel's output contribution is booking-dependent (import vs domestic)
whereas the consumption/CEV ranking is not.

Numbers are printed and also written to output/tab_exante_expost_*.tex and
output/fig_exante_expost_*.pdf.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from model import build_model, run, solve_ss, ss_unknowns_targets_fixed_psi
from calibration import make_calibration
from welfare import cev_total
from latex_tables import write_table
import blocks as B

H = 24                      # welfare / disbursement horizon (matches .tex captions)
TAU_B = 0.10                # headline permanent carbon tax for the EA economy
NUMERAIRE, BOOKING = 'core', 'import'
OUT = 'output'


# =============================================================================
# 1. STEADY STATES
# =============================================================================
def solve_ets_ss(model, psi_fixed, tau_b, tau_g=0.0, s_g_ets=0.0,
                 recycle='rebate'):
    """Prepared ETS steady state: psi_g fixed at the no-ETS baseline, D_GREEN
    floats up under the permanent carbon tax."""
    ov = dict(tau_b=tau_b, tau_g=tau_g, s_g_ets=s_g_ets)
    if recycle == 'green_subsidy':
        ov['s_g'] = s_g_ets
    calib = make_calibration(NUMERAIRE, booking=BOOKING, ets=True, **ov)
    calib['psi_g'] = psi_fixed
    u, t = ss_unknowns_targets_fixed_psi(BOOKING, ets=True)
    ss = solve_ss(model, calib, unknowns=u, targets=t, booking=BOOKING)
    return ss


# =============================================================================
# 2. STANDING WELFARE: the welfare-optimal permanent carbon tax
# =============================================================================
def standing_sweep(model, ss_base, psi_fixed, taus):
    """CEV of each ETS steady state vs the baseline SS, with NO crisis (a zero
    impulse). Reveals whether -- and where -- a permanent carbon tax raises
    steady-state welfare in this externality-free model (green energy is
    structurally cheaper, so the logit friction leaves green under-adopted)."""
    zero = {k: np.zeros(H) for k in ('UTIL_0', 'UTIL_1', 'UTIL_2', 'n')}
    rows = []
    for tb in taus:
        ss = solve_ets_ss(model, psi_fixed, tb)
        chi, _ = cev_total(ss_base, ss, zero)
        rows.append((tb, float(ss['D_GREEN']), float(ss['R_carbon']), 100 * chi))
    return rows


# =============================================================================
# 3. CRISIS COMPARISON
# =============================================================================
def gross_disbursement(irf, key):
    return float(np.sum(np.asarray(irf[key])[:H]))


def main():
    os.makedirs(OUT, exist_ok=True)
    model = build_model(NUMERAIRE, booking=BOOKING)

    # --- baseline (common reference) and the fixed psi_g it pins ---
    ss_base, irf_lf = run(model, shock_kind='price', policy='none',
                          numeraire=NUMERAIRE, booking=BOOKING)
    psi_fixed = float(ss_base['psi_g'])

    # --- ex-post responses on the baseline economy ---
    _, irf_cap = run(model, shock_kind='price', policy='subsidy',
                     numeraire=NUMERAIRE, booking=BOOKING)
    _, irf_tr = run(model, shock_kind='price', policy='transfer',
                    numeraire=NUMERAIRE, booking=BOOKING)

    # --- ex-ante prepared (ETS) economy, no crisis policy, and ETS + cap ---
    ss_ets, irf_ea = run(model, shock_kind='price', policy='none', ets=True,
                         ets_kwargs=dict(tau_b=TAU_B, recycle='rebate'),
                         numeraire=NUMERAIRE, booking=BOOKING)
    _, irf_ea_cap = run(model, shock_kind='price', policy='subsidy', ets=True,
                        ets_kwargs=dict(tau_b=TAU_B, recycle='rebate'),
                        numeraire=NUMERAIRE, booking=BOOKING)

    # --- welfare (total CEV vs the common baseline SS) + gross disbursement ---
    scen = [
        ('Laissez-faire',        ss_base, irf_lf,     'none'),
        ('Ex-post cap',          ss_base, irf_cap,    'Subsidy'),
        ('Ex-post transfer',     ss_base, irf_tr,     'Ttargeted'),
        ('Ex-ante ETS',          ss_ets,  irf_ea,     'R_carbon'),
        ('Ex-ante ETS + cap',    ss_ets,  irf_ea_cap, 'both'),
    ]
    print(f'\nBaseline D_GREEN = {float(ss_base["D_GREEN"]):.4f}   '
          f'ETS D_GREEN = {float(ss_ets["D_GREEN"]):.4f}   '
          f'(tau_b = {TAU_B}, psi_g fixed = {psi_fixed:.6f})\n')
    hdr = f'{"scenario":<20s}{"CEV%":>9s}{"impat":>8s}{"mid":>8s}{"pat":>8s}' \
          f'{"peakDG":>9s}{"grossFisc":>11s}'
    print(hdr); print('-' * len(hdr))
    table_rows = []
    for lbl, pre, irf, disb in scen:
        m, byt = cev_total(ss_base, pre, irf)
        pk = 100 * float(np.max(np.asarray(irf['D_GREEN'])[:H]))
        if disb == 'none':
            g = 0.0
        elif disb == 'both':
            g = gross_disbursement(irf, 'Subsidy') + H * float(ss_ets['R_carbon'])
        elif disb == 'R_carbon':
            g = H * float(ss_ets['R_carbon'])        # standing revenue over H
        else:
            g = gross_disbursement(irf, disb)
        print(f'{lbl:<20s}{100*m:9.4f}{100*byt[0]:8.4f}{100*byt[1]:8.4f}'
              f'{100*byt[2]:8.4f}{pk:9.3f}{100*g:11.3f}')
        table_rows.append([lbl, f'{100*m:.3f}', f'{100*byt[0]:.3f}',
                           f'{100*byt[1]:.3f}', f'{100*byt[2]:.3f}',
                           f'\\textbf{{{pk:.2f}}}', f'{100*g:.2f}'])

    # --- standing welfare sweep: welfare-optimal permanent carbon tax ---
    taus = [0.0, 0.05, 0.10, 0.20, 0.30]
    print('\nStanding welfare (ETS SS vs baseline, permanent, no crisis):')
    print(f'{"tau_b":>7s}{"D_GREEN":>10s}{"R_carbon":>10s}{"standCEV%":>11s}')
    sweep = standing_sweep(model, ss_base, psi_fixed, taus)
    for tb, dg, rc, chi in sweep:
        print(f'{tb:7.2f}{dg:10.4f}{rc:10.5f}{chi:11.4f}')

    # --- figure: welfare-optimal permanent carbon tax (hump) + induced greening
    tb_arr = np.array([r[0] for r in sweep])
    dg_arr = np.array([r[1] for r in sweep])
    chi_arr = np.array([r[3] for r in sweep])
    tb_star = tb_arr[int(np.argmax(chi_arr))]
    figc, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(100 * tb_arr, chi_arr, 'o-')
    a1.axhline(0, color='k', lw=0.5)
    a1.axvline(100 * tb_star, color='C3', ls='--',
               label=rf'$\tau_b^*\approx{100*tb_star:.0f}\%$')
    a1.set_xlabel(r'permanent carbon tax $\tau_b$ (\%)')
    a1.set_ylabel(r'standing CEV vs baseline (\%)')
    a1.set_title('Welfare-optimal carbon tax'); a1.legend(fontsize=9)
    a2.plot(100 * tb_arr, 100 * dg_arr, 's-')
    a2.set_xlabel(r'permanent carbon tax $\tau_b$ (\%)')
    a2.set_ylabel(r'steady-state green share $D_{\mathrm{GREEN}}$ (\%)')
    a2.set_title('Induced greening')
    figc.tight_layout()
    fcpath = os.path.join(OUT, f'fig_carbon_optimal_{BOOKING}.pdf')
    figc.savefig(fcpath, dpi=140, bbox_inches='tight')
    print(f'[figure] {fcpath}   (tau_b* = {100*tb_star:.0f}%)')

    # =========================================================================
    # 4. FIGURE: crisis dynamics, LF vs ex-post cap vs ex-ante ETS
    # =========================================================================
    from plotting import show_irfs
    fig = show_irfs(
        [irf_lf, irf_cap, irf_ea],
        outputs=['C', 'D_GREEN', 'pE_B_P', 'y'],
        labels=['Laissez-faire', 'Ex-post cap', 'Ex-ante ETS'],
        titles=[r'Consumption $C$', r'Green share $D_{\mathrm{GREEN}}$',
                r'Brown price $P^E_B$', r'Output $y$'],
        T_plot=H, ncol=4)
    fig.suptitle(r'Version A: ex-ante greening vs ex-post capping', y=1.03)
    fpath = os.path.join(OUT, f'fig_exante_expost_{BOOKING}.pdf')
    fig.savefig(fpath, dpi=140, bbox_inches='tight')
    print(f'\n[figure] {fpath}')

    # =========================================================================
    # 5. LATEX TABLE
    # =========================================================================
    tpath = os.path.join(OUT, f'tab_exante_expost_{BOOKING}.tex')
    write_table(
        tpath,
        colspec='lrrrrrr',
        header=['Scenario', 'CEV\\%', 'impat.', 'middle', 'patient',
                'peak $D_G$', 'gross fisc.'],
        rows=table_rows,
        caption=(f'Ex-ante greening vs ex-post capping under a brown-energy '
                 f'price shock (import booking, $\\tau_b={TAU_B}$). CEV is the '
                 f'total consumption-equivalent variation relative to the '
                 f'no-ETS steady state, by discount-factor type; gross fiscal '
                 f'is the cap outlay (ex-post) or the standing carbon revenue '
                 f'over $H={H}$ (ex-ante).'),
        label=f'tab:exante_expost_{BOOKING}')
    print(f'[table]  {tpath}')


if __name__ == '__main__':
    main()
