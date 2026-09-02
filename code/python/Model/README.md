# E-HANK

Replication code and manuscript for a heterogeneous-agent New Keynesian small
open economy with a discrete brown/green durable adoption margin, used to study
household responses to energy-price shocks and carbon pricing.

## Requirements

    python >= 3.10
    sequence_jacobian, numba, numpy, scipy, matplotlib

    pip install sequence_jacobian numba

The manuscript compiles with pdflatex (needs `lmodern` and `natbib`).

## Layout

    main.py            pipeline driver: `python main.py <group>` or `python main.py list`
    run_baseline.py    minimal end-to-end solve (steady state + one price-shock IRF)
    core/              model: household.py (StageBlock + logit adoption), blocks.py,
                       model.py (build_model / run / solve_ss / frozen_model),
                       calibration.py, welfare.py (CEV), frozen_adoption.py
    tools/             latex_tables.py, plotting.py
    runners/           one experiment per file (run_*.py, nl_investigate.py, tests.py),
                       invoked as modules, e.g. `python -m runners.run_persistence`
    cache/             pickled checkpoints (delete a file to force recomputation)
    paper/             manuscript: main.tex, sections/, appendix/, bibliography.tex,
                       output/ (figures and tables written by the runners)

## Running

Everything runs from this directory (`Model/`).

    python main.py all        # every group, in the paper's order
    python main.py baseline    # a single group
    python main.py list        # groups and the artifacts each produces

Each runner writes its figures (`fig.savefig`) and tables
(`tools.latex_tables`) into `paper/output/`; no reported number is typed by
hand. Compile the paper from `paper/` with `pdflatex main.tex` (run twice, with
`bibtex main` in between); the relative `output/...` paths also work when the
`paper/` folder is uploaded to Overleaf as is.

## Conventions

- Runners are launched from `Model/` (root-relative paths `paper/output/`,
  `cache/`); each sets its own numeraire/booking globals, so `main.py` runs them
  as subprocesses.
- Core imports are absolute (`from core.model import ...`); module-mode
  invocation puts `Model/` on `sys.path`.
- The verified configuration is `numeraire='cpi'`, `booking='import'`.
- After a model change, run the relevant group and check `assets_clearing`
  (~1e-6 band) and `E_clearing` before trusting the numbers.
