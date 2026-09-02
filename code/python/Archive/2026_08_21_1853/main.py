#%%
r"""E-HANK -- single entry point.

Regenerates every figure and table the paper \input's / \includegraphics's,
by running each experiment runner in the paper's order. Each runner is a
self-contained script that writes to output/; main.py only sequences them and
reports what is missing.

    python main.py            # run the full pipeline in paper order
    python main.py baseline   # one group only (see GROUPS)
    python main.py list       # show groups and the artifacts each produces

Design: runners are executed as subprocesses, not imported, because several of
them run work at module top level and set their own NUMERAIRE/BOOKING/CLEAR_CACHE
globals. Sequencing them as processes keeps those namespaces isolated.
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')

# group -> list of (script, argv). Order within 'all' is the paper's order.
GROUPS = {
    # QA: asserts the 'no_adoption' counterfactual shares a bit-identical
    # steady state with 'adoption' (common-SS property), and plots full vs
    # frozen. Runs first: if this fails, every adoption-channel number
    # downstream is unreliable, and it's cheap (a handful of solves).
    'qa':          [('run_frozen_qa.py', [])],
    # Section 4 experiments: baseline IRFs, policy figures, summary table,
    # deflator (E8). Produces fig0..fig5, fig8, tab_summary_core_import,
    # deflator table.
    'baseline':    [('run_experiments.py', [])],
    # Sign-map of the adoption channel across shock persistence (diagnostic,
    # appendix material at most under the frozen construction).
    'signmap':     [('run_signmap.py', [])],
    # Dose-response of the cap + distributional CEV (E6/E7).
    # fig6, tab_dose_core_import, tab_cev_core_import.
    'welfare':     [('run_dose_response.py', [])],
    # Section 5 (welfare table + monetary figure) and Section 6 (ex-ante/ex-post,
    # green subsidy). run_summary_table -> tab_summary_import + fig_monetary;
    # run_exante_expost -> tab_exante_expost + fig_carbon_optimal + crisis fig;
    # run_green_subsidy -> fig_green_subsidy (Section 6.3).
    'exante':      [('run_summary_table.py', []),
                    ('run_exante_expost.py', []),
                    ('run_green_subsidy.py', []),
                    # ETS vs baseline IRF overlay + consumption by technology
                    # group (fig_ets_irf_*, fig_cross_section_*, tab_cross_section).
                    ('run_ets_cross_section.py', [])],
    # Persistence sweep (Route A). tab_persistence_import + fig_persistence.
    'persistence': [('run_persistence.py', [])],
    # Carbon-tau_b sweep: rebate vs budget-neutral green-subsidy recycling
    # (Appendix A.2, sec:recycle).
    'taub':        [('run_exante_taub_sweep.py', [])],
    # Appendix A.1: cross-booking adoption-channel sign map.
    # tab_signmap_booking.tex + booking_compare.txt.
    'booking':     [('run_booking_compare.py', [])],
    # Nonlinearity diagnosis (E9): size sweep, taste_shock sweep, figure.
    'nonlin':      [('nl_investigate.py', ['size']),
                    ('nl_investigate.py', ['taste']),
                    ('nl_investigate.py', ['fig']),
                    # welfare on the nonlinear transition (tab_nl_cev_import.tex)
                    ('run_nl_cev.py', [])],
}
 
ORDER = ['qa', 'baseline', 'signmap', 'welfare', 'exante', 'persistence',
         'taub', 'booking', 'nonlin']


def run_script(script, argv):
    path = os.path.join(HERE, script)
    if not os.path.exists(path):
        print(f"  [MISSING] {script} not found -- skipping "
              f"(its figures/tables will be absent).")
        return False
    print(f"  >>> python {script} {' '.join(argv)}", flush=True)
    r = subprocess.run([sys.executable, path, *argv], cwd=HERE)
    if r.returncode != 0:
        print(f"  [FAIL] {script} exited {r.returncode}")
        return False
    return True


def run_group(name):
    print(f"\n=== {name} ===", flush=True)
    ok = True
    for script, argv in GROUPS[name]:
        ok = run_script(script, argv) and ok
    return ok


def main():
    os.makedirs(OUT, exist_ok=True)
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if arg == 'list':
        for g in ORDER:
            arts = ', '.join(s for s, _ in GROUPS[g])
            print(f"{g:12s} -> {arts}")
        return
    groups = ORDER if arg == 'all' else [arg]
    for g in groups:
        if g not in GROUPS:
            print(f"unknown group '{g}'. choose from: {', '.join(ORDER)} | all | list")
            return
        run_group(g)
    print("\nDone. All figures/tables are in output/.")


if __name__ == '__main__':
    main()

# %%
