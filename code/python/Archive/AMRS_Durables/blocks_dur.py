"""Aggregate blocks modified for the durable margin.

DECISION (this session): the resource cost of switching brown -> green is booked
as an IMPORT (option 2), not domestic demand for the home good (option 1).

Why option 1 fails here: ARS have markup_ss = 1.03 with capitalised profits
(block `income`, dividend = (markup_ss-1)*w*n, priced into J/j). Booking
AD_durables as extra demand for the home good makes `y` rise to meet it, which
mechanically creates a new dividend flow that is NOT owned by any household ->
`assets_clearing` breaks by the capitalised value of that marginal dividend
(0.5-0.7 in the scan, roughly psi_g*D_SWITCH capitalised at 1/r). This is a
structural difference from `blocks_soe.py`, where Div=0 by construction and the
same closure (`AD_durables` in `goods_clearing`) is exactly right -- see that
module's docstring. Do not port that closure here without re-deriving it.

OPTION 2 (chosen): the green durable (EV, heat pump, ...) is an imported capital
good. The switching expenditure leaves the country directly, exactly like the
brown/green energy bill in `external`. It does NOT enter `goods_clearing` (no
new domestic dividend created) and DOES enter the trade balance / nfa. This is
also the more defensible empirical story (Europe imports EVs and batteries) and
gives the model an extra channel worth a sentence in the paper: adoption widens
the external deficit before it starts saving on the energy bill.

    imports_dur = psi_g * D_SWITCH
    goods_clearing UNCHANGED from ARS (y does not respond to imports_dur)
    nfa loses imports_dur each period, on top of the ordinary BoP identity

TO REVISIT for an "option 1" variant (kept here for the record, not implemented):
booking AD_durables as domestic demand requires either (a) Div=0 as in
blocks_soe.py, so no capitalised dividend is created, or (b) an explicit rule
for who receives the marginal dividend from the extra unit of y. Neither is
implemented in this ARS-side file.

STATUS (this session): imports_dur is now wired into the BoP/nfa identity via
`CA_dur`, a full rewrite of ARS's `CA` block that subtracts imports_dur from
netexports. eqm_cond_dur's signature was trimmed of ~20 unused leftover
arguments inherited from ARS's `eqm_cond` (dead params from the notebook
copy-paste, never used in the body). One of them, `netexports`, was NOT just
dead weight: keeping it would have created a DAG cycle once `CA_dur` depends
on `imports_dur` (an eqm_cond_dur output) while `eqm_cond_dur` formally
depends on `netexports` (a CA_dur output) -- sj.simple treats a declared
argument as a graph edge regardless of whether the body uses it. Same failure
mode as the omega-weights cycle in the beta-heterogeneity work; same fix
(drop the spurious edge).

Units check for CA_dur: psi_g * D_SWITCH is booked in hh_durable.py's budget
constraint in the same real numeraire as `coh`/C (see that module's Tdur).
netexports in ARS's CA is already expressed in that same home-real numeraire
(Q converts the foreign-currency legs). So `netexports - imports_dur` is a
same-units subtraction, no additional Q scaling needed.
"""
import numpy as np
import sequence_jacobian as sj


@sj.simple
def imports_dur_block(psi_g, D_SWITCH):
    """Isolated on purpose: imports_dur must NOT sit in the same block as
    assets_clearing (see eqm_cond_dur below). eqm_cond_dur needs nfa, which is
    a CA_dur output; if imports_dur were bundled there too, CA_dur (needs
    imports_dur) and eqm_cond_dur (needs nfa) would form a genuine 2-cycle --
    not a dead-argument artifact this time, an actual simultaneity introduced
    by co-locating two economically unrelated equations in one function."""
    imports_dur = psi_g * D_SWITCH
    return imports_dur


@sj.simple
def energy_gap_block(pEhh_P, pE_g_P, cE_ss_agg, D_GREEN):
    """BoP counterpart of the household-side energy-price-gap transfer in
    hh_durable.py's Tdur (`-(pE_d - pEhh_P) * cE_ss_agg`). Green-type
    households receive +(pEhh_P - pE_g_P) * cE_ss_agg through Tdur; with no
    matching change to the country's physical energy import bill (cE stays
    the aggregate ARS value, independent of D_GREEN in this reduced-form
    setup), that windfall is a domestic transfer manufactured from nothing.
    CONFIRMED by isolating it from psi_g/D_SWITCH: it alone breaks
    assets_clearing by ~0.5-0.7 in the scan, same order as the whole residual.

    FIX (proposed, not yet validated against model.tex): book it as a real
    saving on the energy import bill, i.e. increase netexports by
    energy_gap_agg. Economic reading: the green durable really does let its
    owner buy the same cE_ss_agg energy service at a lower world price, so
    the country's actual energy import bill is
        pEhh_P * cE_ss_agg * (1-D_GREEN) + pE_g_P * cE_ss_agg * D_GREEN
    which is exactly (PEstar-based ARS bill) - energy_gap_agg.
    THIS IS A MODELING CHOICE, not just an accounting patch -- flag to
    Guillaume before trusting IRFs: it asserts the green energy saving is a
    real resource saving for the country, not merely a domestic transfer.
    """
    energy_gap_agg = (pEhh_P - pE_g_P) * cE_ss_agg * D_GREEN
    return energy_gap_agg


@sj.simple
def eqm_cond_dur(y, cH, cHstar, A, gdp, nfa, j, jF, jE, B, cE, prodE, PEstar,
                 PEstar_shock, E_supply_elasticity, E_supply, zetaEsupply, j_Esupply):
    goods_clearing = cH + cHstar - y
    assets_clearing = A - nfa - j - jF - jE - B - zetaEsupply * j_Esupply
    if E_supply_elasticity == np.inf:
        E_clearing = PEstar - PEstar_shock
    else:
        E_clearing = (cE + prodE) - E_supply
    PEstar_diff = PEstar - PEstar_shock
    gdp_t = gdp - 1
    return goods_clearing, assets_clearing, E_clearing, PEstar_diff, gdp_t


@sj.solved(unknowns={'nfa': (-2, 2)}, targets=['nfares'], solver="brentq")
def CA_dur(nfa, Q, pHstar, cHstar, PFstar, cF, PEstar, cE, prodE, rante, r, A,
          rdom, Adom, zetaEsupply, E_supply, y, imports_dur, energy_gap_agg):
    """Full rewrite of ARS's CA block (auclert_ha.py): two durable-margin
    terms added on top of the ordinary tradables/energy BoP legs.
      - imports_dur: switching cost psi_g*D_SWITCH, an extra outflow (option 2).
      - energy_gap_agg: cheaper energy for green adopters, an extra inflow
        (reduced import bill) -- see energy_gap_block docstring.
    Everything else is verbatim ARS."""
    exports = Q * (pHstar * cHstar + PEstar * zetaEsupply * E_supply)
    imports = Q * (PFstar * cF + PEstar * (cE + prodE)) + imports_dur - energy_gap_agg
    imports_pc = imports / imports.ss
    exports_pc = exports / exports.ss

    netexports = Q * (pHstar * cHstar - PFstar * cF
                      - PEstar * (cE + prodE - zetaEsupply * E_supply)) - imports_dur + energy_gap_agg
    revaluation_term = (r - rante(-1)) * A(-1) - (rdom - rante(-1)) * Adom(-1)
    nfares = netexports + revaluation_term + (1 + rante(-1)) * nfa(-1) - nfa

    nx_gdp = netexports / y
    return nfares, netexports, revaluation_term, exports, imports, nx_gdp, imports_pc, exports_pc


@sj.simple
def green_target(D_GREEN, D_GREEN_ss_target):
    """Calibration residual: pins psi_g to hit an observed green share."""
    green_res = D_GREEN - D_GREEN_ss_target
    return green_res
