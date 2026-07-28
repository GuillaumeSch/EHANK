# E-HANK — agent instructions

Open-economy HANK with a brown/green durable adoption margin (logit via
StageBlock), SSJ 1.0.0. Thesis paper, deadline 1 Oct 2026.

## How to talk to me
- Converse in FRENCH. Write all code and comments in ENGLISH.
- Senior macro audience: skip basics, go to mechanisms and assumptions.
- Be concise and concrete. If a result is weak, say so plainly.
- NEVER invent an academic reference. Verify or omit.
- Flag inconsistencies and risk zones proactively, unprompted.
- Prefer complete function rewrites over patches.

## Environment
- Activate the venv before running anything: `source .venv/bin/activate`
- SSJ 1.0.0 and numba required.
- Solve a steady state:
  `python -c "from model import build_model, solve_ss; from calibration import make_calibration; solve_ss(build_model('core'), make_calibration('core'))"`

## Non-regression — run after ANY change to the household or blocks
1. Envelope: Va == dV/da converges at O(h^2)
2. Savings policy monotone (needed by interpolate_y)
3. Stock-flow: D_SWITCH == delta_g * D_GREEN
4. Sensitivity to initial conditions
5. Phase-2 nesting (p_rel(brown) == 1 exactly)

## Settled decisions — DO NOT reopen
- Numeraire = domestic good ('core' default).
- CPI anchored on brown price only (Option C); gap quantified ex post.
- Switch cost booked as an import; psi_g denominated in CPI.
- hh_outputs_dur replaces hh_outputs (Jensen-correct aggregation).
- LogitChoiceDurable / StageBlockDurables in utils.py are DEAD code.
- psi_g / D_GREEN are NOT wired as SS unknown/target; calibration is external.

## Known traps
- solve_steady_state never validates its own residuals in SSJ 1.0.0.
- internals['durables']['V'] is the stage INPUT, not output.
- The analytic elasticity in calibration_moments.py overstates the true
  (partial-FD) value; use a partial finite-difference on the household block
  (all inputs frozen, only psi_g moved, baseline distribution) as the moment.

## Calibration status (as of last session)
- Adoption-price elasticity has a floor ~ -4 on the D_GREEN=0.05 locus:
  targets of -2.5 or -3.3 are unattainable at plausible sigma.
- Recommended: sigma (taste_shock) = 0.12, psi_g ~= 1.02, eps_true ~= -4.4.
- Changing taste_shock alone does NOT recalibrate psi_g; rerun the external
  bisection to hold D_GREEN = 0.05.

## Reference docs — read these before non-trivial work
- README.md            package layout, run order, numeraire, open points
- SSJ_SKILL.md         SSJ-specific gotchas (verified)
- SSJ_REFERENCE_1_.md  SSJ reference notes (verified)
