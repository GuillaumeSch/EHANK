import sys; sys.path.insert(0, '.')
import numpy as np
from core import model as M
from core.calibration import make_calibration

NUM, BOOK = 'cpi', 'import'
model = M.build_model(numeraire=NUM, booking=BOOK)
calib = make_calibration(numeraire=NUM, booking=BOOK, **M.MONETARY['real_rate'])
diss = M.dissolve_list(BOOK)

u, t = M.ss_unknowns_targets(BOOK)
ss = model.solve_steady_state(calib, unknowns=u, targets=t,
    solver='broyden_custom', dissolve=diss, ttol=1e-11)

u_td, t_td = M.td_unknowns_targets(BOOK)
T = 300; rho = 2 ** (-1 / 16)
shock = {'PEstar_shock': 1.0 * rho ** np.arange(T)}
ss_td = model.steady_state(ss, dissolve=diss)
irf = model.solve_impulse_linear(ss_td, u_td, t_td, shock)

if __name__ == '__main__':
    print(f"SS: Z={float(ss['Z']):.5f}  pires={float(ss['pires']):.1e}  "
          f"assets_clearing={float(ss['assets_clearing']):.1e}  D_GREEN={float(ss['D_GREEN']):.4f}")
    print(f"IRF peaks: y={100*np.max(np.abs(np.asarray(irf['y'])[:24])):.2f}%  "
          f"C={100*np.max(np.abs(np.asarray(irf['C'])[:24])):.2f}%  "
          f"D_GREEN={100*np.max(np.abs(np.asarray(irf['D_GREEN'])[:24])):.2f}pp  "
          f"assets_dyn={np.max(np.abs(np.asarray(irf['assets_clearing']))):.1e}")
