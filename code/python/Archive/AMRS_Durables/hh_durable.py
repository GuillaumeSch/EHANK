"""E-HANK household with a brown/green durable, built to nest ARS exactly.

DURABLE STATE (4 points) = the PAIR (holding entering the period, holding chosen
last period), which is what the switching cost needs and what a plain 2-point
state cannot carry. Index order matches the user's HH_Block convention:

    0 = BB   brown now, brown before
    1 = BG   brown now, green before
    2 = GB   green now, brown before   <- pays the switching cost psi_g
    3 = GG   green now, green before

Stage order (forward): [dep, prod, durables, consav].

BREAKDOWN MARKOV (`d_markov`), applied in the `dep` stage:
    brown is ABSORBING (no depreciation)      -> BB, BG map to BB w.p. 1
    green breaks down to brown at rate delta_g -> GB, GG map to BB w.p. delta_g,
                                                  and stay green otherwise
This is the user's specification: no brown depreciation, green depreciates.

HOW THE DURABLE ENTERS THE BUDGET
ARS households are homothetic and hold a real bond in units of the consumption
basket, so the energy price does NOT appear in their budget constraint at all --
it acts only through the CPI, via the real wage atw_n = W/P. Adding a
type-specific energy price therefore means a type-specific cost of the basket.
Rather than introducing a second numeraire (the problem flagged in the user's
blocks_soe.py), we book the difference as a type-specific income term, exactly
parallel to the ARS non-homotheticity term `Tf`:

    Tdur(d) = -(pE_d(d) - pEhh_P) * cE_ss   -   psi_g * switch_indicator(d)

The first term is the extra (or saved) energy bill of a household whose durable
makes it face pE_d(d) rather than the economy-wide index pEhh_P; it is exact to
first order (see SSJ_SKILL Sec. 7: the product of a level and a first-order gap
is second order, so cE_ss may replace cE_t). The second term is the resource
cost of switching brown -> green, paid in basket units.

NESTING TEST (phase 2): with pE_g = pE_b the first term is identically zero, and
with psi_g large the logit puts zero mass on switching, so every household stays
brown and the durable dimension is payoff-irrelevant. The model must then return
the ARS baseline to solver precision.

ROOT-CAUSE BUG FOUND AND FIXED (this session): the value function returned by
`consav` was computed as `V = u + beta*V`, adding the incoming continuation
value at the CURRENT state's grid index instead of interpolating it at the
CHOSEN a'. That silently assumes a'(a_i) == a_grid[i]. The resulting error
a'(a_i) - a_grid[i] is large (up to +-3.5 on this grid) and, critically,
DIFFERS ACROSS DURABLE STATES: GB's coh is lower by psi_g, so GB dissaves more
relative to the grid (measured mean drift -0.099 for GB vs -0.008 for BB at
psi_g=0.10, a gap of ~psi_g). Since V rises in a, reading V at a_grid[i]
overstated GB's continuation value MORE than BB's, by an amount growing in
psi_g -- so raising the switching cost RAISED adoption. ARS never hit this
because their household has no discrete choice and needs only Va, never V.
Fix: `V = u + beta_g * interpolate_y(a_grid, a, V)`.

Post-fix behaviour (verified): D_GREEN is monotone decreasing in psi_g in both
PE and GE; with zero energy benefit a small psi_g kills adoption; and
psi_g ~ 0.20 hits the D_GREEN_ss_target = 0.05 calibration target in GE, which
sits above the PV of the adoption gain (0.1333) exactly as expected. Adoption
dispersion across beta types is now mild and correctly signed (0.025/0.044/
0.070 at psi_g=0.20) rather than the 0.33/0.68/0.91 spread seen pre-fix.

SUPERSEDED (recorded so it is not re-derived): earlier in this session the high
and non-monotone D_GREEN was wrongly attributed first to a wealth-tail effect
of a level cost, then to a beta-heterogeneity artifact, then to `hh_init`
seeding each durable slot from its own coh. All three were symptoms of the V
bug above. The `hh_init` change in particular was a band-aid and has been
REVERTED: with V computed correctly, the plain per-slot init reproduces the
same results to 5 decimals.

FEASIBILITY GUARD (this session): with psi_g large enough, coh in state GB can
go negative for low-(e,a) households (min_a=0, no borrowing room to smooth a
level cost). util_l masks that state's CHOICE PROBABILITY to ~0 via -1e10,
but consav's backward recursion still evaluates log/power on it unconditionally
for every (d,e,a) before the mask applies -- so consav also floors
consumption net of ghh at 1e-10 before the log/power, purely to keep V finite;
the mask (not this floor) is what keeps the model from ever choosing it. Both
guards are inert at the calibrated psi_g ~ 0.20 (checked: min c = 0.023).
"""
import numpy as np
import sequence_jacobian as sj
from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.blocks.support.stages import ExogenousMaker, Continuous1D, LogitChoice


# =============================================================================
# 1. DURABLE GRID AND BREAKDOWN MARKOV
# =============================================================================
def make_durable_markov(delta_g):
    """Rows = state today, columns = state next period. Order BB, BG, GB, GG."""
    d_markov = np.array([
        [1.0, 0.0, 0.0, 0.0],                  # BB -> brown absorbing
        [1.0, 0.0, 0.0, 0.0],                  # BG -> brown absorbing
        [delta_g, 0.0, 0.0, 1.0 - delta_g],    # GB -> breaks to brown, else green
        [delta_g, 0.0, 0.0, 1.0 - delta_g],    # GG -> breaks to brown, else green
    ])
    return d_markov


# is the household GREEN this period? (states GB, GG)
IS_GREEN = np.array([0.0, 0.0, 1.0, 1.0])
# does the household PAY the switching cost? (state GB = green chosen, brown held)
PAYS_SWITCH = np.array([0.0, 0.0, 1.0, 0.0])


def make_grids(rho_e, sd_e, n_e, min_a, max_a, n_a, delta_g):
    e_grid, _, Pi = sj.grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    a_grid = sj.grids.asset_grid(min_a, max_a, n_a)
    d_markov = make_durable_markov(delta_g)
    return e_grid, Pi, a_grid, d_markov


def hh_income(e_grid, atw_n, r, pEhh_P, pE_g_P, cbarE, scale_w, markup_ss, a_grid,
              n, frisch, ghh_prefs, epsT, cE_ss_grid, insE, pE_P, pE_P_ss,
              psi_g, cE_ss_agg):
    """ARS income, plus the two durable terms. Leading axis is the durable state."""
    Tf = - pEhh_P * cbarE * (atw_n * markup_ss) * scale_w - pEhh_P * cbarE * (1 - scale_w)
    Tfiscal = epsT + insE * (pE_P - pE_P_ss) * cE_ss_grid

    # type-specific energy price gap, first-order exact (see module docstring)
    pE_d = pEhh_P + IS_GREEN * (pE_g_P - pEhh_P)
    Tdur = -(pE_d - pEhh_P) * cE_ss_agg - psi_g * PAYS_SWITCH

    coh = ((1 + r) * a_grid + atw_n * e_grid[:, np.newaxis] + Tf + Tfiscal)
    coh = coh[np.newaxis, ...] + Tdur[:, np.newaxis, np.newaxis]

    n_ss = 1
    ghh = ghh_prefs * 1 / (1 + 1 / frisch) * (n ** (1 + 1 / frisch) - n_ss ** (1 + 1 / frisch))
    return coh, ghh


def hh_init(coh, r, eis):
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    V = Va / (1 - 0.98)
    return Va, V


# =============================================================================
# 2. STAGES
# =============================================================================
dep_stage = ExogenousMaker(markov_name='d_markov', index=0, name='dep')
prod_stage = ExogenousMaker(markov_name='Pi', index=1, name='prod')


def util_l(V, green_block, coh):
    """Flow payoff of choosing durable d given the durable held, on (d | d_).

    -1e10 forbids a transition. Brown -> green IS allowed (state GB); its
    resource cost psi_g is charged in the budget, not here -- EXCEPT when
    that charge would leave coh non-positive for a given (e,a). min_a=0 in
    this calibration (no borrowing), so there is no room to smooth a level
    cost psi_g against low cash-on-hand: without this mask, consav's EGM
    hits log(c<=0) for exactly those low-coh states once psi_g is large
    enough to matter (confirmed: breaks at psi_g=0.5 in the phase-3 scan).
    ghh (labor disutility offset) is state-independent and =0 in the ARS
    calibration (ghh_prefs=0); if that's ever turned on, subtract it from
    coh here too before testing the sign.
    """
    # `green_block` shuts the adoption margin: 0 = free choice (cost billed in the
    # budget via psi_g), large = brown->green forbidden, which is the phase-2 closure.
    gb = -green_block
    # coh[2] = cash-on-hand in state GB, i.e. net of psi_g already (see hh_income).
    infeasible = coh[2] <= 0.0
    gb_masked = np.where(infeasible, -1e10, gb)

    flow_u = np.full((4, 4) + V.shape[1:], -1e10)
    flow_u[0, 0] = 0.
    flow_u[0, 1] = 0.
    flow_u[2, 0] = gb_masked
    flow_u[2, 1] = gb_masked
    flow_u[3, 2] = 0.
    flow_u[3, 3] = 0.
    return flow_u


durables_stage = LogitChoice(value='V', backward='Va', index=0, name='durables',
                             taste_shock_scale='taste_shock', f=util_l)


def consav(V, Va, a_grid, r, beta_g, eis, coh, ghh):
    """EGM step. Identical algebra to ARS, plus the value function the logit needs."""
    uc_nextgrid = beta_g * Va
    c_nextgrid = uc_nextgrid ** (-eis) + ghh
    a = sj.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    np.maximum(a, a_grid[0], out=a)   # setmin() is hardcoded 2-D; state is 3-D here
    c = coh - a
    # Floor consumption net of ghh away from 0: coh can go negative for state
    # GB at low (e,a) once psi_g is large (min_a=0 leaves no borrowing room to
    # smooth a level cost). util_l's -1e10 mask zeroes that state's CHOICE
    # PROBABILITY, but this backward step still evaluates it unconditionally
    # for every (d,e,a), so log/power on a non-positive argument must be
    # guarded here to keep V finite; the mask (not this floor) is what
    # actually keeps the model from choosing it.
    c_safe = np.maximum(c - ghh, 1e-10)
    Va = (1 + r) * c_safe ** (-1 / eis)
    if eis == 1:
        u = np.log(c_safe)
    else:
        u = c_safe ** (1 - 1 / eis) / (1 - 1 / eis)
    # Continuation value must be read at the CHOSEN a', not at the current
    # state's grid index. `u` is indexed by today's assets a_i; the incoming V
    # is indexed by tomorrow's asset grid. Writing `V = u + beta*V` silently
    # assumes a'(a_i) == a_grid[i], which is false, and the resulting error
    # a'(a_i) - a_grid[i] differs ACROSS DURABLE STATES (GB dissaves more,
    # by ~psi_g). Since V rises in a, that overstates GB's continuation more
    # than BB's, by an amount growing in psi_g -- which made adoption rise
    # with the switching cost. ARS never hit this because their household has
    # no discrete choice and so never needs V, only Va.
    V = u + beta_g * sj.interpolate.interpolate_y(a_grid, a, V)
    return Va, V, a, c


def compute_weighted_mpc(c, a_grid, r, e_grid):
    mpc = np.empty_like(c)
    post_return = (1 + r) * a_grid
    mpc[..., 1:-1] = (c[..., 2:] - c[..., :-2]) / (post_return[2:] - post_return[:-2])
    mpc[..., 0] = (c[..., 1] - c[..., 0]) / (post_return[1] - post_return[0])
    mpc[..., -1] = (c[..., -1] - c[..., -2]) / (post_return[-1] - post_return[-2])
    mpc = mpc * e_grid[:, np.newaxis]
    return mpc


def durable_shares(c):
    """Population shares by durable status, and the switching flow."""
    d_green = np.zeros_like(c) + IS_GREEN[:, np.newaxis, np.newaxis]
    d_switch = np.zeros_like(c) + PAYS_SWITCH[:, np.newaxis, np.newaxis]
    return d_green, d_switch


consav_stage = Continuous1D(backward=['V', 'Va'], policy='a', f=consav, name='consav',
                            hetoutputs=[compute_weighted_mpc, durable_shares])


# =============================================================================
# 3. ASSEMBLY (beta heterogeneity via the ARS rename/remap idiom)
# =============================================================================
hh_one = StageBlock([dep_stage, prod_stage, durables_stage, consav_stage], name='hh',
                    backward_init=hh_init,
                    hetinputs=[make_grids, hh_income])

GROUP_VARS = ['C', 'A', 'MPC', 'cE_ss_grid', 'D_GREEN', 'D_SWITCH']


@sj.simple
def group_betas(beta_spread, beta_max):
    beta_2 = beta_max
    beta_1 = beta_max - beta_spread / 2
    beta_0 = beta_max - beta_spread
    return beta_0, beta_1, beta_2


@sj.simple
def aggregate_groups(C_0, C_1, C_2, A_0, A_1, A_2, MPC_0, MPC_1, MPC_2,
                     D_GREEN_0, D_GREEN_1, D_GREEN_2, D_SWITCH_0, D_SWITCH_1, D_SWITCH_2,
                     beta_0, beta_1, beta_2):
    C = (C_0 + C_1 + C_2) / 3
    A = (A_0 + A_1 + A_2) / 3
    MPC = (MPC_0 + MPC_1 + MPC_2) / 3
    D_GREEN = (D_GREEN_0 + D_GREEN_1 + D_GREEN_2) / 3
    D_SWITCH = (D_SWITCH_0 + D_SWITCH_1 + D_SWITCH_2) / 3
    beta = (beta_0 + beta_1 + beta_2) / 3
    return C, A, MPC, D_GREEN, D_SWITCH, beta


def hh_ha_durable(n_beta=3):
    hh_list = [hh_one.rename(suffix=f'_{i}')
                     .remap({x: f'{x}_{i}' for x in GROUP_VARS})
                     .remap({'beta_g': f'beta_{i}'})
               for i in range(n_beta)]
    return sj.create_model(hh_list + [group_betas, aggregate_groups], name='hh_ha_durable')
