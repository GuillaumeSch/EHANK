r"""E-HANK pipeline entry point. Runs each experiment runner in paper order.

    python main.py            # full pipeline
    python main.py baseline   # one group (see GROUPS)
    python main.py list       # groups and their artifacts
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'paper', 'output')

# group -> [(script, argv)]; 'all' runs them in the paper's order
GROUPS = {
    # QA: assert 'adoption' and 'no_adoption' share a bit-identical steady state; runs first
    'qa':          [('run_frozen_qa.py', [])],
    # Section 4: baseline IRFs, policy figures, and the summary table.
    'baseline':    [('run_experiments.py', [])],
    # adoption-channel sign-map across persistence (diagnostic)
    'signmap':     [('run_signmap.py', [])],
    # Dose-response of the cap and the distributional CEV table.
    'welfare':     [('run_dose_response.py', [])],
    # taste-scale identification (tab_taste_identification, fig_taste_identification)
    'taste_id':    [('run_taste_identification.py', [])],
    # Sections 5-6: welfare table, monetary figure, ex-ante/ex-post, green subsidy
    'exante':      [('run_summary_table.py', []),
                    ('run_exante_expost.py', []),
                    ('run_green_subsidy.py', []),
                    # ETS vs baseline IRF overlay + consumption by technology group
                    ('run_ets_cross_section.py', [])],
    # Persistence sweep (Route A). tab_persistence_import + fig_persistence.
    'persistence': [('run_persistence.py', [])],
    # carbon-tau_b sweep: rebate vs green-subsidy recycling (Appendix A.2)
    'taub':        [('run_exante_taub_sweep.py', [])],
    # Appendix A.1: cross-booking adoption-channel sign map
    'booking':     [('run_booking_compare.py', [])],
    # Nonlinearity diagnosis (E9): size sweep, taste_shock sweep, figure.
    'nonlin':      [('nl_investigate.py', ['size']),
                    ('nl_investigate.py', ['taste']),
                    ('nl_investigate.py', ['fig']),
                    # welfare on the nonlinear transition
                    ('run_nl_cev.py', [])],
}

ORDER = ['qa', 'baseline', 'signmap', 'welfare', 'taste_id', 'exante',
         'persistence', 'taub', 'booking', 'nonlin']


def run_script(script, argv):
    path = os.path.join(HERE, 'runners', script)
    if not os.path.exists(path):
        print(f"  [MISSING] {script} not found -- skipping "
              f"(its figures/tables will be absent).")
        return False
    print(f"  >>> python {script} {' '.join(argv)}", flush=True)
    r = subprocess.run([sys.executable, '-m', f"runners.{script[:-3]}", *argv], cwd=HERE)
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
    print("\nDone. All figures/tables are in paper/output/.")


if __name__ == '__main__':
    main()

