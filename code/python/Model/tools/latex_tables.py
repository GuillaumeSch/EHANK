"""LaTeX table fragments written to output/tab_*.tex."""
import os

def _escape(s):
    return str(s).replace('_', r'\_').replace('%', r'\%')

def write_table(path, colspec, header, rows, caption, label,
                 small=True, centering=True, midrule_after=None, notes=None,
                 fit_width=False):
    """Generic booktabs table writer.

    caption -> concise title placed ABOVE the tabular via \\figtitle.
    notes   -> factual details placed BELOW via \\fignotes (small, left-aligned).
    If notes is None the title alone is emitted (back-compatible).
    fit_width -> wrap the tabular in \\resizebox{\\textwidth}{!}{...} so a
    many-column table is scaled down to the text width.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    lines = [r'\begin{table}[H]' + (r'\centering' if centering else '')]
    lines.append(r'\figtitle{' + caption + '}')
    if small:
        lines.append(r'\small')
    tab = [r'\begin{tabular}{' + colspec + '}', r'\toprule',
           ' & '.join(header) + r'\\', r'\midrule']
    for i, row in enumerate(rows):
        tab.append(' & '.join(row) + r'\\')
        if midrule_after and i in midrule_after and i != len(rows) - 1:
            tab.append(r'\midrule')
    tab.append(r'\bottomrule')
    tab.append(r'\end{tabular}')
    if fit_width:
        lines.append(r'\resizebox{\textwidth}{!}{%')
        lines.extend(tab)
        lines.append(r'}')
    else:
        lines.extend(tab)
    if label:
        lines.append(r'\label{' + label + '}')
    if notes:
        lines.append(r'\fignotes{' + notes + '}')
    lines.append(r'\end{table}')
    open(path, 'w').write('\n'.join(lines) + '\n')
    return path

def summary_table(path, rows, H, numeraire, booking, label='tab:summary'):
    """Main fiscal-policy summary table."""
    header = ['Shock', 'Policy', r'$y(0)$\%', r'$\sum y$', r'$\pi(0)$ ann.',
              r'peak $D^{G}$ pp', 'fiscal cost', r'adoption $\to \sum y$']
    body, mid = [], set()
    shocks = []
    PLABEL = {'none': 'no policy', 'subsidy': 'price cap',  # cap, not a subsidy: labelled 'price cap' everywhere
              'transfer': 'Slutsky transfer', 'transfer_flat': 'flat transfer'}
    for r in rows:
        if r['shock'] not in shocks:
            if shocks:
                mid.add(len(body) - 1)
            shocks.append(r['shock'])
        body.append([
            r['shock'] if r['policy'] == 'none' else '',
            PLABEL.get(r['policy'], r['policy']),
            f"${r['y0']:+.2f}$", f"${r['ycum']:+.1f}$", f"${r['pi0']:+.2f}$",
            r'\textbf{' + f"{r['dG_peak']:.2f}" + '}',
            f"{r['fiscal']:.2f}", f"${r['adopt_y']:+.2f}$",
        ])
    booking_note = (' Domestic booking: green is a domestic industry, only '
                    'brown energy imported.' if booking == 'domestic' else '')
    title = 'Fiscal policy: summary'
    notes = (f"Cumulative sums over {H} quarters.{booking_note} The last "
             r"column is the contribution of the adoption margin, the "
             r"difference between the full model and a common-steady-state "
             r"counterfactual in which the adoption choice does not respond "
             r"to the shock (Section~\ref{sec:model}).")
    return write_table(path, 'llrrrrrr', header, body, title, label,
                       midrule_after=mid, notes=notes)

def booking_signmap_table(path, rows, H, label='tab:signmap'):
    """Cross-booking sign-map table."""
    header = [f'$\\Delta_{{{H}}}$, adoption output contribution',
              'import booking', 'domestic booking']
    body = [
        ['price shock', f"${rows[('import','price')]:+.2f}$",
         f"${rows[('domestic','price')]:+.2f}$"],
        ['supply shock', f"${rows[('import','supply')]:+.2f}$",
         f"${rows[('domestic','supply')]:+.2f}$"],
    ]
    title = 'Adoption channel output contribution, by booking'
    notes = (f'Cumulative adoption contribution $\\Delta_{{{H}}}$, import against '
             r'domestic booking (Section~\ref{sec:bop}), measured against the '
             r'common-steady-state counterfactual of Section~\ref{sec:model}. '
             r'No policy.')
    return write_table(path, 'lcc', header, body, title, label, small=True,
                       notes=notes)

def dose_response_table(path, rows_by_shock, tau_grid, label='tab:dose'):
    """Cap-intensity sweep table."""
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
    title = 'Price cap: response across intensity'
    notes = (r'Cap intensity $\tau^E$ swept from $0$ to $1$, against the Slutsky '
             r'transfer benchmark (full compensation, $\iota=1$). CEV is total '
             r'consumption-equivalent variation.')
    return write_table(path, 'llrrrrrr', header, body, title, label,
                       midrule_after=mid, notes=notes)

def cev_table_tex(path, ss, irfs, labels=None, scenario_note='price shock',
                  label='tab:cev'):
    """Distributional CEV table, from welfare.cev_table's underlying numbers."""
    from core.welfare import cev
    labels = labels or list(irfs)
    header = ['Scenario (' + scenario_note + ')', 'CEV mean', 'impatient',
              'middle', 'patient']
    body = []
    for k in labels:
        m, byt = cev(ss, irfs[k])
        body.append([k] + [f"${100*x:+.2f}$" for x in [m, *byt]])
    title = 'CEV by discount-factor type'
    notes = (r'Consumption-equivalent variation (\%), private household welfare '
             r'only. Types are the three permanent discount factors of '
             r'Section~\ref{sec:model}, of equal mass.')
    return write_table(path, 'lrrrr', header, body, title, label, notes=notes)
