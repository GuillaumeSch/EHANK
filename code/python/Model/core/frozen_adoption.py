"""Common-steady-state 'no adoption' counterfactual."""
import numpy as np
import sequence_jacobian as sj
from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.blocks.support.stages import LogitChoice
from sequence_jacobian.blocks.support.law_of_motion import DiscreteChoice

from core import blocks as B
from core import household as H


class FrozenLogitChoice(LogitChoice):
    """LogitChoice whose choice probabilities do not respond to shocks."""

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
    """Same DAG as model.build_model, with the frozen-choice household."""
    num = B.numeraire_cpi
    margin = [B.energy_gap]
    ca, imp, eqm = B.CA, B.importProfits, B.eqm_cond
    return sj.combine([
        hh_ha_durable_frozen(),
        num, B.assets_convert,
        B.hh_outputs, *margin,
        B.income, B.profitcenters, B.importPrices, imp,
        B.revaluation, B.revaluation_dom, B.foreign_c, B.UIP, B.IEA, ca,
        B.unions, B.piW_to_W, B.CESprices, B.price_levels, B.pitop,
        B.mon_policy, B.fiscal, B.annualize, eqm, B.reweight_cpi,
    ])
