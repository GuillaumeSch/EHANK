"""The two shocks, side by side (Section 5): what is exogenous and what clears.

Price shock (elastic supply, E_supply_elasticity=inf): the WORLD ENERGY PRICE is
the exogenous driver (PEstar_shock), a 100 log-point impulse with a 16-quarter
half-life; the energy QUANTITY clears and falls. Supply shock (finite elasticity,
Bayer et al.): the QUANTITY is the exogenous driver -- availability drops 10% for
6 quarters -- and the world PRICE clears the reduced quantity endogenously.

The figure anchors the distinction: in the price shock we move the price and quantity
responds; in the supply shock we move the quantity and the price responds. It also
reports how large a price spike a 10% quantity cut implies.

Emits fig_shocks_import.pdf. ~2 SS solves.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from core.model import build_model, run

H = 24
NUMERAIRE, BOOKING = 'cpi', 'import'
OUT = 'paper/output'


def pct(irf, ss, k):
    return 100.0 * np.asarray(irf[k], dtype=float)[:H] / float(ss[k])


if __name__ == '__main__':
    model = build_model(NUMERAIRE, booking=BOOKING)
    ssP, price = run(model, shock_kind='price', policy='none', model_variant='adoption')
    ssS, supply = run(model, shock_kind='supply', policy='none', model_variant='adoption')

    t = np.arange(H)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)

    # Panel A: price shock -- exogenous world price (driver), endogenous quantity
    axA = ax[0]
    axA.plot(t, pct(price, ssP, 'PEstar'), color='#c44', lw=2.0,
             label=r'world energy price (exogenous driver)')
    axA.plot(t, pct(price, ssP, 'cE'), color='#48c', lw=2.0,
             label=r'energy quantity $c_E$ (clears)')
    axA.axhline(0, color='k', lw=0.5)
    axA.set_title('Price shock', fontsize=10)
    axA.set_xlabel('quarters', fontsize=8)
    axA.set_ylabel(r'% deviation from own steady state', fontsize=9)
    axA.legend(fontsize=8)

    # Panel B: supply shock -- exogenous quantity cut (driver), stock-buffered
    # realized availability, and the endogenous clearing price
    axB = ax[1]
    axB.plot(t, pct(supply, ssS, 'E_supply_shock'), color='#48c', lw=2.0,
             label=r'availability cut (exogenous driver)')
    axB.plot(t, pct(supply, ssS, 'E_supply'), color='#48c', lw=1.2, ls='--',
             label=r'realized availability (stock-buffered)')
    axB.plot(t, pct(supply, ssS, 'PEstar'), color='#c44', lw=2.0,
             label=r'world energy price (clears)')
    axB.axhline(0, color='k', lw=0.5)
    axB.set_title('Supply shock', fontsize=10)
    axB.set_xlabel('quarters', fontsize=8)
    axB.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(OUT, f'fig_shocks_{BOOKING}.pdf')
    fig.savefig(out)
    plt.close(fig)

    # report magnitudes
    peak_q = pct(supply, ssS, 'E_supply').min()
    peak_p = pct(supply, ssS, 'PEstar').max()
    print(f"price shock: world price peak +{pct(price,ssP,'PEstar').max():.0f}%, "
          f"energy quantity trough {pct(price,ssP,'cE').min():.1f}%")
    print(f"supply shock: availability trough {peak_q:.1f}%, "
          f"endogenous world price peak +{peak_p:.0f}%")
    print(f"  -> a {abs(peak_q):.0f}% quantity cut implies a +{peak_p:.0f}% price spike "
          f"(inelastic demand, eta_E=0.10)")
    print(f"  -> {out}")
