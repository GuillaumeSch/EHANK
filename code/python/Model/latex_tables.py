"""Write paper-style LaTeX table fragments to output/tab_*.tex, \\input-able
directly from ehank_results.tex. Keeps the .tex file's hand-written tables in
sync with the code that produces the numbers, instead of manual copy-paste.

USAGE IN THE MAIN .tex
-----------------------
Replace a hand-written `table` environment with:

    \\input{output/tab_summary_core_import.tex}

Every writer here emits a COMPLETE `table` environment (booktabs style, same
as the paper's existing tables), so no wrapper is needed at the call site.

CONVENTIONS
-----------
- Bold the peak-D_GREEN column (the paper's convention for the adoption-margin
  headline number).
- Numbers are pre-formatted strings, not raw floats: every writer takes
  already-rounded values so the table content in output/*.tex matches exactly
  what's printed to stdout / the .txt logs, with no silent rounding drift.
- `\\label{}` keys are stable across bookings/numeraires (tag suffix in the
  filename, not the label), so \\ref{} in prose does not need to change when
  the input file changes.
"""
import os


def _escape(s):
    return str(s).replace('_', r'\_').replace('%', r'\%')


def write_table(path, colspec, header, rows, caption, label,
                 small=True, centering=True, midrule_after=None):
    """Generic booktabs table writer.

    header: list of column header strings (LaTeX-ready, e.g. r'$y(0)$\\%').
    rows:   list of lists of cell strings (LaTeX-ready). Use '' for blank.
    midrule_after: set of 0-indexed row numbers after which to insert a
        \\midrule (e.g. to separate shocks), or None.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    lines = [r'\begin{table}[H]' + (r'\centering' if centering else '')]
    if small:
        lines.append(r'\small')
    lines.append(r'\begin{tabular}{' + colspec + '}')
    lines.append(r'\toprule')
    lines.append(' & '.join(header) + r'\\')
    lines.append(r'\midrule')
    for i, row in enumerate(rows):
        lines.append(' & '.join(row) + r'\\')
        if midrule_after and i in midrule_after and i != len(rows) - 1:
            lines.append(r'\midrule')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\caption{' + caption + '}')
    if label:
        lines.append(r'\label{' + label + '}')
    lines.append(r'\end{table}')
    open(path, 'w').write('\n'.join(lines) + '\n')
    return path


# =============================================================================
# PAPER-SPECIFIC BUILDERS
# =============================================================================
def summary_table(path, rows, H, numeraire, booking, label='tab:summary'):
    """Main E2/E3 fiscal-policy summary table (run_experiments.py).

    rows: list of dicts with keys
        shock, policy, y0, ycum, pi0, dG_peak, fiscal, adopt_y
      (adopt_y only meaningful for policy=='none'; pass '' otherwise upstream).
    """
    header = ['Shock', 'Policy', r'$y(0)$\%', r'$\sum y$', r'$\pi(0)$ ann.',
              r'peak $D^{G}$ pp', 'fiscal cost', r'adoption $\to \sum y$']
    body, mid = [], set()
    shocks = []
    for r in rows:
        if r['shock'] not in shocks:
            if shocks:
                mid.add(len(body) - 1)
            shocks.append(r['shock'])
        body.append([
            r['shock'] if r['policy'] == 'none' else '',
            r['policy'],
            f"${r['y0']:+.2f}$", f"${r['ycum']:+.1f}$", f"${r['pi0']:+.2f}$",
            r'\textbf{' + f"{r['dG_peak']:.2f}" + '}',
            f"{r['fiscal']:.2f}", f"${r['adopt_y']:+.2f}$",
        ])
    booking_note = (' (domestic booking: green is a domestic industry, only '
                    'brown energy imported)' if booking == 'domestic' else '')
    caption = (f"Cumulative sums over {H} quarters{booking_note}. The last "
              r"column is the contribution of the adoption margin: the "
              r"difference between the model with the margin open and the "
              r"same model with it shut.")
    return write_table(path, 'llrrrrrr', header, body, caption, label,
                       midrule_after=mid)


def booking_signmap_table(path, rows, H, label='tab:signmap'):
    """Route A cross-booking sign-map (run_booking_compare.py): adoption's
    output contribution by booking x shock, 'none' policy only.

    rows: dict {(booking, shock): adopt_y_value}
    """
    header = [f'$\\Delta_{{{H}}}$, adoption output contribution',
              'import booking', 'domestic booking']
    body = [
        ['price shock', f"${rows[('import','price')]:+.2f}$",
         f"${rows[('domestic','price')]:+.2f}$"],
        ['supply shock', f"${rows[('import','supply')]:+.2f}$",
         f"${rows[('domestic','supply')]:+.2f}$"],
    ]
    caption = (r'Adoption channel contribution to cumulative output '
              f'($\\Delta_{{{H}}}$), import vs.\\ domestic booking '
              r'(Section~\ref{sec:bop}). \emph{None} policy.')
    return write_table(path, 'lcc', header, body, caption, label, small=True)


def dose_response_table(path, rows_by_shock, tau_grid, label='tab:dose'):
    """E6/E7 dose-response table (run_dose_response.py).

    rows_by_shock: {shock: {'rows': [dict(tau=..,dG=..,y=..,pi0=..,cE=..,fisc=..,cev=..)],
                            'transfer': dict(same keys, no tau)}}
    """
    header = ['Shock', 'Policy', 'peak $D^{G}$ pp', r'$\sum y$', r'$\pi(0)$',
              r'$\sum c_E$', 'fiscal', 'CEV \\%']
    body, mid = [], set()
    for shock, d in rows_by_shock.items():
        for r in d['rows']:
            body.append([shock if r['tau'] == tau_grid[0] else '',
                        rf"cap $\tau^E={r['tau']:.2f}$",
                        f"{r['dG']:.3f}", f"{r['y']:.2f}", f"{r['pi0']:.2f}",
                        f"{r['cE']:.2f}", f"{r['fisc']:.2f}", f"{r['cev']:+.4f}"])
        tr = d['transfer']
        body.append(['', 'transfer', f"{tr['dG']:.3f}", f"{tr['y']:.2f}",
                    f"{tr['pi0']:.2f}", f"{tr['cE']:.2f}", f"{tr['fisc']:.2f}",
                    f"{tr['cev']:+.4f}"])
        mid.add(len(body) - 1)
    caption = (r'Dose-response in the price-cap intensity $\tau^E$, against the '
              r'Slutsky transfer benchmark (full compensation, $\iota=1$).')
    return write_table(path, 'llrrrrrr', header, body, caption, label,
                       midrule_after=mid)


def cev_table_tex(path, ss, irfs, labels=None, scenario_note='price shock',
                  label='tab:cev'):
    """Distributional CEV table, from welfare.cev_table's underlying numbers."""
    from welfare import cev
    labels = labels or list(irfs)
    header = ['Scenario (' + scenario_note + ')', 'CEV mean', 'impatient',
              'middle', 'patient']
    body = []
    for k in labels:
        m, byt = cev(ss, irfs[k])
        body.append([k] + [f"${100*x:+.2f}$" for x in [m, *byt]])
    caption = (r'CEV by discount-factor type (\%). Impatient types hold little '
              r'wealth and have high MPCs. Private household welfare only.')
    return write_table(path, 'lrrrr', header, body, caption, label)
