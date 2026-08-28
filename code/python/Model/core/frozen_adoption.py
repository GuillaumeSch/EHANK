"""Common-SS adoption decomposition.

The `no_adoption` variant (green_block large) shuts the margin by pinning
D_GREEN=0 in the STEADY STATE, so an adoption-on vs adoption-off comparison mixes
two things: the missing crisis response AND a different starting steady state
(5% green vs 0% green).

`FrozenLogitChoice` removes that confound. It inherits LogitChoice.backward_step
unchanged -- so the steady state is bit-identical to the adoption-on economy
(same choice probabilities, same D_GREEN=0.05, same psi_g) -- and overrides only
backward_step_shock to zero the choice-probability response dP. In the
transition the durable composition is therefore frozen at its steady-state
values (no crisis-induced switching), while green households keep their
insulation and everything else responds normally. The adoption channel is then
the clean difference (full - frozen) from a COMMON steady state.
"""
import numpy as np
import sequence_jacobian as sj
from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.blocks.support.stages import LogitChoice
from sequence_jacobian.blocks.support.law_of_motion import DiscreteChoice

from core import blocks as B
from core import household as H


class FrozenLogitChoice(LogitChoice):
    """LogitChoice whose choice probabilities do not respond to shocks.

    backward_step (hence the steady state) is inherited unchanged; only the
    perturbation zeroes dP. dEV keeps the envelope term sum(P*dV), so values and
    Va still propagate through the FIXED steady-state choice probabilities."""

    def backward_step_shock(self, ss, shocks, precomputed):
        f, lom = precomputed
        dV = np.swapaxes(shocks[self.value][np.newaxis, ...], 0, self.index + 1)
        if f is not None:
            dflow_u = next(iter(f.diff(shocks).values()))
            dflow_u = np.nan_to_num(dflow_u)
        else:
            dflow_u = np.zeros_like(lom.P)
        dV = dflow_u + dV
        dEV = np.sum(lom.P * dV, axis=0)          # envelope, frozen probabilities
        dlom = DiscreteChoice(np.zeros_like(lom.P), self.index)   # dP = 0
        doutputs = {self.value: dEV}
        for k in self.backward:
            doutputs[k] = dlom.T @ ss[k]          # identically zero
            if k in shocks:
                doutputs[k] += lom.T @ shocks[k]
        return doutputs, dlom


def hh_ha_durable_frozen(n_beta=3):
    durables_frozen = FrozenLogitChoice(
        value='V', backward='Va', index=0, name='durables',
        taste_shock_scale='taste_shock', f=H.util_l)
    hh_one = StageBlock(
        [H.dep_stage, H.prod_stage, durables_frozen, H.consav_stage], name='hh',
        backward_init=H.hh_init,
        hetinputs=[H.make_grids, H.energy_price_bundle, H.hh_income])
    hh_list = [hh_one.rename(suffix=f'_{i}')
                     .remap({x: f'{x}_{i}' for x in H.GROUP_VARS})
                     .remap({'beta_g': f'beta_{i}'})
               for i in range(n_beta)]
    return sj.create_model(hh_list + [H.group_betas, H.aggregate_groups],
                           name='hh_ha_durable_frozen')


def build_model_frozen(numeraire='cpi', booking='import', ets=False):
    """Same DAG as model.build_model, but with the frozen-choice household."""
    num = B.numeraire_core if numeraire == 'core' else B.numeraire_cpi
    if booking == 'domestic':
        margin = [B.green_sector]
        ca, imp, eqm = B.CA_dom, B.importProfits_dom, B.eqm_cond_dom
    else:
        margin = [B.switching_imports, B.energy_gap]
        ca, imp, eqm = B.CA, B.importProfits, B.eqm_cond
    return sj.combine([
        hh_ha_durable_frozen(),
        num, B.assets_convert,
        B.hh_outputs_dur, B.green_energy_price, *margin,
        B.income, B.profitcenters, B.importPrices, imp,
        B.revaluation, B.revaluation_dom, B.foreign_c, B.UIP, B.IEA, ca,
        B.unions, B.piW_to_W, B.CESprices, B.price_levels, B.pitop,
        B.mon_policy, B.fiscal, B.annualize, eqm, B.reweight_cpi,
    ])
