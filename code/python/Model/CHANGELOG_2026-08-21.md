# E-HANK — reconciliation to current state (2026-08-21)

The uploaded archive was a **7 August snapshot**. It was missing the work from
the 18 August session ("Analyse des chocs énergétiques et welfare"), which had
never been re-integrated locally. This changelog records exactly what was
reconstructed and re-applied, and how each piece was verified.

Everything was reproduced by execution in SSJ 1.0.0. Nothing was retyped from a
draft; every number below came from a fresh solve.

## Re-integrated from the 18 August session

### `household.py` — cross-sectional consumption hetoutputs
Added to `durable_shares`: `c_green`, `c_brown`, `c_switch` (state GB), aggregated
by SSJ to `C_GREEN`, `C_BROWN`, `C_SWITCH`, threaded through `GROUP_VARS` and
`aggregate_groups`.
Verified: steady state unchanged (psi_g = 0.53804, y(0) = -2.75, peak D_G =
8.87pp), identity `C = C_BROWN + C_GREEN` holds to 6.1e-12 on the IRF,
`assets_clearing = -6.8e-7`. The frozen model inherits the hetoutputs.

### `run_ets_cross_section.py` — NEW (registered in `main.py`, group `exante`)
ETS-vs-baseline IRF overlay + consumption decomposition by technology group,
both economies, price and supply shocks, full and frozen variants, cache
`cache_ets_xs/`. Emits `fig_ets_irf_{price,supply}_import.pdf`,
`fig_cross_section_{price,supply}_import.pdf`, `tab_cross_section_import.tex`.
The decomposition uses the frozen counterfactual for fixed-population per-capita
responses; the adoption/migration term (full minus frozen) is reported
separately, and the residual frozen-mass drift (~1% of the full response,
because frozen fixes choice probabilities, not masses) is netted at SS levels.
Verified: reproduces the documented headline numbers exactly — price shock,
baseline ΣdC −105.6 (brown −88.7 / green −3.7 / adoption −13.2), peak ΔD_G 8.87;
ETS ΣdC −106.5 (−78.7 / −4.9 / −22.8), peak ΔD_G 14.98; supply peaks 1.91 / 3.86.

### `run_nl_cev.py` — NEW (registered in `main.py`, group `nonlin`)
Welfare (CEV) on the nonlinear perfect-foresight transition vs the first-order
linear CEV, scenarios LF / cap / Slutsky / flat / ETS at sizes 0.25 / 0.5 / 0.75,
cache `cache_nl_cev/`. Emits `tab_nl_cev_import.tex`, `nl_cev_import.pkl`. Lifts
the inner-block `maxit` to 500 (see the SSJ gotcha below). One nonlinear solve is
~20s; the full sweep is minutes and is meant to run locally in the background
(`python -u run_nl_cev.py > nl_cev.txt`).
Verified: LF at size 0.25 reproduces CEV_L = -0.4713, CEV_NL = -0.4626,
NL/L = 0.981 (matches the documented table).

### `main.py`
Registered both new runners: `run_ets_cross_section.py` in `exante`,
`run_nl_cev.py` in `nonlin`.

### `SSJ_Skills/SSJ_SKILL.md`
Appended the inner-`maxit=30` gotcha for `solve_impulse_nonlinear`.

## From the 21 August audit (this session)

### `nl_investigate.py` — replaced with the audited version
- Lifts the inner-block backward-iteration cap (the same `maxit=30` limit).
- Aligns the sweep grid and the figure grid (the old code solved `[0.125..1.0]`
  but the figure read `[0.03125..0.75]`, silently dropping the two smallest
  points). Both now read one `SIZES` list, capped at 0.75.
- Guards the frozen-variant D_GREEN NL/L ratio (division by a ~0 linear peak
  otherwise prints 145–255).
- eps=1.0 documented as fragile (converges only from a fresh model, not in a warm
  grid loop) rather than forced; the paper's claims live over [0.03125, 0.75].
Verified: the App B numbers reproduce — D_GREEN NL/L 1.16 at eps=0.125, 2.04 at
eps=0.75; aggregate output y NL/L in [1.00, 1.01].

## SSJ gotcha (also in SSJ_SKILL.md)

`solve_impulse_nonlinear` inner solved blocks default to `maxit=30` at the class
level; the outer `maxit=` kwarg and per-block `options=` do not reach them. Raise
`Block.solve_impulse_nonlinear_options['maxit']` in place before solving. Even
then the cap does not converge at eps=0.75 and nothing converges at eps=1.0.

### `welfare_note.tex` — §4 rewrite reconstructed and recompiled
Section 4 "What we will do" replaced by "The nonlinear check": the nonlinear-CEV
table (all three sizes), the four findings (levels overstated 1–2%→7–26%;
ordering flat > Slutsky > cap > ETS > LF survives; ex-ante vs LF a few bp, crisis
dividend "zero to first order"; impatient-under-flat NL/L 1.16→1.89), and the
precedents paragraph. The two upstream sentences (§3 and the intro) were updated
to match. Recompiled to `welfare_note.pdf` (4 pages). Updated in place under
`Manuscript versions/08_07_1937/` and copied to the project root.

### Outputs regenerated (shipped in `Model/output/`)
`tab_nl_cev_import.tex` (freshly computed, matches the note table; cap does not
converge at eps=0.75, shown as n.c.), `fig_ets_irf_{price,supply}_import.pdf`,
`fig_cross_section_{price,supply}_import.pdf`, `tab_cross_section_import.tex`,
`fig9_nonlinearity.png`.

## Not included / still Overleaf-side

- The **current main manuscript** is not in this tree. The newest `.tex` here
  (`Manuscript versions/08_07_1937/ehank_paper_restructured.tex`) is the 7 August
  restructured paper with 4 `\NEEDSRERUN` flags in App B — NOT the ~10 August
  draft (16 flags, updated §6). The live paper lives in Overleaf; the
  `Manuscript versions/` folder is kept as historical snapshots only.
- Session compute caches (`cache_ets_xs/`, `cache_nl_cev/`) were removed; the
  figures and tables they back are already regenerated and shipped in
  `output/`. Re-running `python main.py exante` / `python main.py nonlin` will
  recompute them from scratch (the nonlinear CEV sweep is ~10 min).

## Manuscript integration (2026-08-21, second pass)

`Manuscript versions/08_07_1937/ehank_paper_restructured.tex` updated in place
(the uploaded copy was byte-identical to the 7 August version and carried none of
the 18 August work). Recompiles clean: 34 pages, 0 undefined references.

- **App B.1** — the four `\NEEDSRERUN` flags cleared. Paragraph 1 rewritten:
  1.16/2.04 kept (reproduced), the "margin shut / concave 0.92" claim dropped
  (it rested on the deprecated zero-SS construction; the frozen counterfactual
  gives y NL/L ~ 1.19, the opposite sign), reframed as "freezing widens the
  NL/L wedge". eps=1.0 fragility footnoted. Paragraph 2 (logit curvature) kept.
- **App B.2 (new)** — "First-order welfare and the nonlinear check": the CEV
  first-order caveat + `\input{tab_nl_cev_import.tex}` + the three findings
  (levels overstated 1–26%, ordering survives, distribution reinforced).
- **§6.2** — ETS-vs-baseline IRF overlay (`fig_ets_irf`) and the cross-sectional
  consumption decomposition (`fig_cross_section` + `\input{tab_cross_section_import.tex}`)
  inserted, with the selection/fixed-population framing.
- **§6.4** — the crisis-only greening dividend restated as "zero to first order".
  **Mechanism corrected** and flagged in-source (`% [REVISION ...]`): the old text
  claimed the prepared economy adopts LESS in the crisis; the ETS cross-section
  shows it adopts MORE (peak dD_G 15.0 vs 8.9pp), so the dividend result is about
  welfare value, not switching volume. The persistence figure caption and the §6
  takeaway were made consistent. **This mechanism change touches a central claim
  and needs Boris's sign-off.**

Still to do against this manuscript: fix Table 8 (App. A.1) booking numbers
(flagged in the 18 August handoff as still carrying July values), and coordinate
the booking convention with Boris before regenerating headline tables.

## Taste-scale identification (2026-08-21, third pass)

New material addressing the sigma_eps under-identification (the #1 credibility item).

- **`run_taste_identification.py`** (NEW, registered in `main.py` group `taste_id`):
  sweeps sigma_eps, recalibrates psi_g to D_G=5%, computes the steady-state
  adoption elasticity (green share vs the operating-cost gap, at fixed psi_g via
  the fixed-psi SS config) and the crisis peak dD_G. Emits
  `tab_taste_identification_import.tex`, `fig_taste_identification.png`.
- **Paper, new §3.1 "Identifying the taste scale"** (+1 page, 35 total): the
  (sigma_eps, psi_g) locus leaves the steady state invariant (D_G and D_sw fixed
  as sigma ranges 0.02-0.20, psi_g 0.32-1.53) but the adoption elasticity varies
  2.8 -> 0.36 and the crisis peak dD_G 19.9 -> 2.0pp, while Sum y is flat
  (-59 to -62): magnitudes are conditional on sigma_eps, the ordering and macro
  are not. Baseline sigma_eps=0.05 => elasticity 1.3.
- **§Limitations item 1** upgraded to point at §3.1 (from "it's a limitation" to
  "characterised; ordering invariant; elasticity moment pins the magnitude").

All model numbers verified in-session (SS invariance and the elasticity computed
by finite difference on the fixed-psi steady state; crisis peaks from the price
shock).

### OPEN (needs the papers' numbers -- not fabricated)
- The empirical elasticity pin is set up but not closed: the paper names
  Beresteanu and Li (2011, IER) and Muehlegger and Rapson (2022, JPubE) as the
  moments to match, and states the baseline implies elasticity 1.3, but does NOT
  assert what those papers estimate. Their adoption-elasticity values are needed
  to state whether sigma_eps=0.05 is consistent or should move.
- Those two papers are named in-text but are NOT yet in the bibliography
  (only ARS2024, Bayer2026, DCEGM2017, Langot2026 have \bibitem). Proper entries
  must be added with verified citation details.

## Taste-scale appendix (2026-08-21, fourth pass)

- **Paper, new Appendix E "The logit taste scale: smoothing device and structural
  elasticity"** (36 pages total, compiles clean, 0 undefined refs): the dual role
  (logit smoothing required by the sequence-space method vs structural elasticity,
  dP_i = P_i(dV_i - dV)/sigma_eps); the two regimes (interior-probability margins
  where sigma->0 is a deterministic cutoff and the scale is legitimately numerical,
  vs rare margins where the logit tail is exponential and the flow semi-elasticity
  is identically 1/sigma_eps for every household); verification that our margin is
  rare (flow identity delta_g D_G/(1-D_G) = 0.26%/quarter; switch-probability
  distribution table); the three consequences and where each is handled (§3.1,
  App B.1, ordering invariance).
- **`run_taste_identification.py` extended** with `switch_prob_table()`: computes
  the distribution-weighted quantiles of quarterly P(switch) among brown
  incumbents by beta type from the household internals, auto-generates
  `tab_switch_prob_dist_import.tex` (pooled mean 0.26% = flow identity; p99.9
  = 8.5%, max 26%). Runs as part of the `taste_id` group.
- **§3.1** pointer sentence added; **App B.1** cross-ref updated to point at
  §3.1 + Appendix E.
- **Bibliography**: Rust (1987), Econometrica 55(5), 999-1033, added -- citation
  verified against the Econometric Society's own record before insertion.

### Still open (next step: proper calibration of sigma_eps)
- Fetch and verify the adoption-elasticity estimates from Beresteanu and Li
  (2011, IER) and Muehlegger and Rapson (2022, JPubE); add their bibitems; map
  the estimates into the model's elasticity to pin sigma_eps (baseline 0.05
  implies 1.3); if the pinned sigma_eps is materially below 0.05, the App B.1
  nonlinearity caveat gains weight.

## Repository reorganisation (2026-08-21, fifth pass)

Flat Model/ split into core/ (model), tools/ (table+plot utilities), runners/
(all experiments), cache/ (all pickles), paper/ (live modular manuscript with
its own output/). All imports rewritten to package form (`from core.model
import ...`); runners now invoked as modules (`python -m runners.<name>`) by
main.py; all output paths point to paper/output/, all caches to cache/.

The manuscript was split losslessly: paper/main.tex + sections/00-09 +
appendix/A-E + bibliography.tex compile to a PDF whose extracted text is
byte-identical to the monolith (36 pages, 0 undefined refs, same byte size).
`Manuscript versions/` is frozen as history; the live paper is paper/.

Verified after the move: all modules compile; SS solves (psi_g=0.53804,
assets_clearing=-6.8e-7); `main.py list` intact incl. taste_id;
`python -m runners.nl_investigate fig` runs end-to-end reading cache/ and
writing paper/output/. README.md added at Model root.

## Calibration of sigma_eps (2026-08-21, sixth pass)

Calibration designed from scratch against verified sources (all fetched and read
in-session, none quoted from memory):

- **Moment**: semi-elasticity of adoption to a $1,000 upfront incentive
  (quasi-experimental vehicle literature). Model side computed on the fixed-psi
  SS: a 1% psi_g cut moves the switching flow by 5.1%->3.8% across sigma
  (nearly flat), but psi_g* itself rises 0.32->1.53 along the locus, so the
  per-$1,000 semi-elasticity falls 80% -> 12% (sigma 0.02 -> 0.20) under the
  stated mapping (1 model unit = quarterly household consumption = $20,000).
- **Verified anchors**: Gallagher-Muehlegger (2011, JEEM 61(1) 1-15) and
  Chandra-Gulati-Kandlikar (2010, JEEM 60(2) 78-93): $1,000 incentive -> +31-38%
  hybrid adoption (early adopters). Muehlegger-Rapson (2022, JPubE 216,
  published version): EV demand elasticity -2.1 (mass market, low/middle
  income) ~ 8%/$1,000 at their $26,000 mean price. Beresteanu-Li (2011, IER
  52(1) 161-182) cited for the gasoline-price channel (no number quoted).
- **Result**: at D_G=5% the model's switchers are early adopters (2x average
  consumption), so the early-adopter range is the target: sigma_eps in
  [0.05, 0.07], implied green premium $11k-$14k (heat-pump/EV range; sigma>=0.10
  implies premia >= $18k, counterfactual). Baseline 0.05 = elastic end of the
  pinned interval; MR's mass-market estimate is the responsiveness floor,
  spanned by the sensitivity table.
- **Implemented**: runner extended (psi_g-margin semi-elasticity + QUARTERLY_C_USD
  mapping constant), tab_taste_identification extended (premium and %/$1,000
  columns), section 3.1 rewritten with the full confrontation, four verified
  bibitems added (closes the "named but not in bibliography" debt). Paper: 36
  pages, 0 undefined references.

## Calibration documentation + publication-scope pass (2026-08-23, seventh pass)

- **New appendix subsection "From micro estimates to the taste scale: the
  procedure"** (now D.1 after renumbering): the four-step procedure (locus,
  fixed-psi configuration, operating-cost margin, upfront-cost margin with the
  dollar conversion formula), the conversion of each verified empirical anchor,
  the population-matching argument, the premium cross-check, the explicit
  sensitivity of the pin to the dollar mapping (C=$16k -> [0.07,0.10]; $20k ->
  [0.05,0.07]; $24k -> [0.04,0.06], all inside the sensitivity grid), the stated
  limits (external validity; upfront vs operating margin), and the reproduction
  entry point (python main.py taste_id).
- **Working-paper artifacts removed from the compiled paper**: the dead
  \NEEDSRERUN macro and its comment block (0 usages remained); Appendix D
  "Summaries of the three reference papers" (explicitly internal material)
  detached from main.tex -- the .tex file stays in paper/appendix/ for the
  coauthors; section 7's opener "This section is written for the coauthors"
  replaced by a neutral framing and its dangling reference to the summaries
  removed. Appendices renumber automatically (taste appendix: E -> D).
- Paper: 35 pages, 0 undefined references, compiles clean.

## Publication pass 1 + referee report + handoff (2026-08-23, eighth pass)

- Paper restructured to journal standard (JME/AEJ:Macro target): new 150-word
  abstract + keywords + JEL (E21,E32,F41,Q43,Q58); new introduction opening on
  the EU policy stakes (all facts verified: Bruegel >EUR600bn shields, DOI
  10.64153/LQHK8283; EHPA heat-pump cycle 2022 +38% / 2023 -5% / 2024 -21%
  with "subsidised gas" among named causes / 2025 rebound; ETS2 start 2028;
  Social Climate Fund >=EUR86.7bn, <=37.5% direct income support); literature
  moved to Section 2; new Section 8 "Implications for European policy" (shield
  incomes not prices / pay the SCF flat / defend ETS2 as externality pricing);
  conclusion absorbs the limitations in prose (label sec:limits preserved);
  calibration bridge sentence (generic importer vs EU narrative); 5 verified
  institutional bibitems added (Bruegel2023, ECETS2, EHPA2023/2024/2025).
  36 pages, 0 undefined references. Old 08_limitations.tex kept on disk,
  detached from main.tex.
- Referee pass written as a real JME report: paper/notes/referee_report_2026-08-23.md
  (majors M1-M6, minors m1-m6, recommendation major revision). Pass 2 = first
  task of the next session.
- HANDOFF_2026-08-23.md at Model root: full state, verified facts with sources,
  calibration result, remaining work in priority order, execution reminders.
