# E-HANK — Model directory

Reorganised 2026-08-21. One rule: **everything runs from this directory** (`Model/`).

## Layout

    main.py            orchestrator: `python main.py <group>` or `python main.py list`
    core/              the model itself -- household.py (StageBlock + logit adoption),
                       blocks.py, model.py (build_model / run / solve_ss / frozen_model),
                       calibration.py, welfare.py (CEV), frozen_adoption.py, deflator.py
    tools/             latex_tables.py (write_table), plotting.py
    runners/           one experiment = one file (run_*.py, nl_investigate.py, tests.py);
                       invoked as modules: `python -m runners.run_persistence`
    cache/             all pickled checkpoints (delete a file/dir to force recomputation)
    paper/             the LIVE manuscript, self-contained:
                         main.tex          master file (preamble + \input of parts)
                         sections/         one .tex per section (00_frontmatter .. 09_conclusion)
                         appendix/         one .tex per appendix (A_booking .. E_taste)
                         bibliography.tex  thebibliography
                         notes/            standalone co-author memos (welfare_note.*), compiled on their own
                         output/           ALL figures and tables -- written directly by
                                           the runners, \input'd / \includegraphics'd by
                                           the .tex with the relative path `output/...`
    Manuscript versions/   historical snapshots only. The live paper is paper/.

## How things flow

1. `python main.py all` (or a group: qa, baseline, signmap, welfare, taste_id, exante,
   persistence, taub, booking, nonlin) runs the runners in the paper's order.
2. Every runner writes its figures (`fig.savefig`) and tables (`tools.latex_tables.
   write_table`) straight into `paper/output/`. No number is ever typed by hand.
3. Compile the paper from `paper/`: `pdflatex main.tex` (twice). For Overleaf,
   upload the `paper/` folder wholesale -- the relative `output/...` paths work as is.

## Conventions

- Runners are subprocesses (isolated NUMERAIRE/BOOKING/CLEAR_CACHE globals); always
  launch them from `Model/` (paths `paper/output/`, `cache/` are root-relative).
- Core imports are absolute (`from core.model import ...`); works because module-mode
  invocation puts `Model/` on sys.path.
- Adding a runner: put it in `runners/`, write to `paper/output/`, cache under
  `cache/`, register it in the relevant group(s) in `main.py` -- an unregistered
  runner is silently never regenerated.
- `python main.py <group>` after any model change; check `assets_clearing` (~1e-6
  band) and `E_clearing` before trusting numbers.
