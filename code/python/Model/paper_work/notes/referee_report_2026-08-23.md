# Referee report — "Energy Shocks and the Green Adoption Channel in a Small Open Economy"
## (internal pass, written as a JME referee would; input for revision pass 2)

**Summary.** The paper adds a discrete brown/green durable adoption margin to the
small open economy HANK of Auclert et al. and shows (i) a retail price cap kills
the crisis adoption wave a transfer preserves, (ii) flat transfers dominate
energy-indexed ones, (iii) ex-ante carbon taxation provides no first-order crisis
insurance. The question is timely, the mechanism is clean, the identification of
the taste scale is unusually careful, and the EU framing lands. My concerns are
about the empirical grounding, calibration coherence, and the robustness of two
signed claims.

### Major points

**M1. The motivating evidence is cited, not shown.** The heat-pump boom–bust
(EHPA) is the paper's empirical hook, yet there is no exhibit: no figure of
sales against the electricity/gas price ratio, no table. At minimum, construct
one figure from EHPA sales and Eurostat energy prices (nrg_pc_202/204),
2019–2025, and show the co-movement the introduction narrates. Better: use the
country panel to *estimate* the adoption semi-elasticity and compare it with the
U.S. vehicle anchors used to pin sigma_eps. As written, the calibration imports
a U.S. vehicle elasticity into a European heat-pump narrative; the bridge
sentence in Section 4 acknowledges this, but a referee will want the EU number,
or at least the figure.

**M2. Which economy is this?** The aggregate calibration is ARS's stylized small
open economy; the narrative is the EU; the dollar mapping uses U.S. household
consumption. This is internally consistent as a "generic energy importer," but
the paper oscillates between "we speak to EU policy" (Sections 1, 8) and "the
calibration is deliberately generic" (Section 4). Choose: either calibrate the
energy share, green share, and premium to euro-area objects (my preference,
given Section 8), or tone the EU claims into "illustrative mapping."

**M3. The ex-ante section leans on a corrected mechanism that needs an exhibit.**
The claim that the prepared economy adopts *more* in volume but gains less in
welfare (Sections 1 and 7) is subtle and central. It currently rests on prose.
Add the decomposition that proves it: crisis switching volume and the welfare
value of the marginal switch, baseline vs ETS, in one table or figure. (I
understand from the source comments this awaits a coauthor's sign-off; it is
also the referee's first question.)

**M4. First-order welfare where basis points are quoted.** Appendix B.2 is the
right robustness, but Section 7 quotes CEV differences of a few basis points
(the 2bp "barely insures" comparison) in the main text without the first-order
caveat attached at the point of use. One sentence and a pointer at first
mention; otherwise a referee will do it for you.

**M5. rho_g = 0 is an upper bound stated, not quantified.** The conclusion
flags full insulation of adopters as an upper bound on the channel. Run one
robustness with partial pass-through (e.g., rho_g = 0.3–0.5 calibrated to
electricity–gas co-movement) and report how the headline peak (9pp) and the
cap/transfer gap move. Cheap, and it converts a caveat into a result.

**M6. Booking convention.** Appendix A is honest that the adoption channel's
output sign flips with the booking of green resource flows. Given the paper's
output-decomposition rhetoric ("the cost is the extensive margin itself"),
report the domestic-booking headline numbers in the main text alongside the
import-booking baseline, not only in the appendix, and state once which
booking the EU narrative corresponds to.

### Minor points

m1. Abstract: "9 percentage points" appears without the level (5% steady-state
share); add "from a 5% base" for scale.
m2. Section 2 (related literature) still reads as three mini-referee reports;
compress by a third and add the empirical adoption literature block (currently
only in Section 4).
m3. JEL codes: consider H23 (externalities; redistribution) given Section 8.
m4. Confirm every number in Tables 1–2 regenerates from the released pipeline
("python main.py all"); state the replication entry point in a footnote.
m5. The introduction's paragraph on conditional results is excellent practice;
move the booking sentence into it from paragraph 7 so both caveats sit together.
m6. Figures: several diagnostic-style titles (e.g., "E9.") survive from the
working paper; retitle for publication.

**Recommendation.** Major revision. The mechanism and the policy mapping are
publishable at a top field journal or better; M1–M3 determine whether the
empirical grounding matches the framing.
