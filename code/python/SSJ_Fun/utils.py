# Standard libraries
import copy
import numpy as np
# Sequence-Jacobian: Stages and Law of Motion
from sequence_jacobian.blocks.support.stages import (
    Continuous1D, ExogenousMaker, LogitChoice, Stage
)
from sequence_jacobian.blocks.support.law_of_motion import LawOfMotion, PolicyLottery1D, DiscreteChoice
from sequence_jacobian.blocks.support import het_compiled
from sequence_jacobian.blocks.stage_block import StageBlock

# Sequence-Jacobian: Utilities
from sequence_jacobian.utilities.misc import make_tuple, logit_choice
from sequence_jacobian.utilities.ordered_set import OrderedSet
from sequence_jacobian.utilities.interpolate import interpolate_coord_robust, interpolate_coord
from sequence_jacobian import utilities as utils
from sequence_jacobian.classes import ImpulseDict
from sequence_jacobian.utilities.multidim import batch_multiply_ith_dimension



def make_d_grid_simple(delta_g=0.01, delta_b=0.00):
    # 0 = brown
    # 1 = green
    d_grid = np.arange(2)
    mapping = {
        0: 'brown',
        1: 'green'
    }
    d_markov = np.array([
        [1-delta_b,    delta_b],
        [delta_g,  1 - delta_g]
    ])
    return d_grid, d_markov, mapping


def make_d_grid(n_b=3, n_g=3, lifetime_new=16, lifetime_old=32):
    """
    Create d_grid and Markov matrix for durable goods with two types (brown/green),
    but depreciation depends only on whether durable is NEW (vintage 1)
    or OLD (vintage 2+).

    Args:
        n_b (int): number of brown vintages
        n_g (int): number of green vintages
        lifetime_new (int): average lifetime (in quarters) of new durables (vintage = 1)
        lifetime_old (int): average lifetime (in quarters) of old durables (vintage >= 2)

    Returns:
        d_grid (np.array): state labels
        d_markov (np.array): Markov transition matrix
        mapping (dict): maps each state index to (type, vintage)
    """

    # Depreciation rates for each vintage category
    delta_new = 1 / lifetime_new        # depreciation prob for vintage = 1
    delta_old = 1 / lifetime_old        # depreciation prob for vintages >= 2

    total_states = 1 + n_b + n_g  # state 0 = no car
    d_grid = np.arange(total_states)
    d_markov = np.zeros((total_states, total_states))

    mapping = {0: ('none', 0)}

    # Brown vintages
    for i in range(n_b):
        s = 1 + i
        mapping[s] = ('brown', i + 1)

    # Green vintages
    for i in range(n_g):
        s = 1 + n_b + i
        mapping[s] = ('green', i + 1)

    # No car is absorbing
    d_markov[0, 0] = 1.0

    def get_delta(vintage):
        """Return depreciation rate depending on whether vintage is new or old."""
        return delta_new if vintage == 1 else delta_old

    # Fill transitions for all brown vintages
    for i in range(n_b):
        s = 1 + i
        vintage = i + 1
        delta = get_delta(vintage)

        if vintage < n_b:
            d_markov[s, s] = 1 - delta
            d_markov[s, s + 1] = delta
        else:
            # last vintage → no car
            d_markov[s, s] = 1 - delta
            d_markov[s, 0] = delta

    # Fill transitions for all green vintages
    for i in range(n_g):
        s = 1 + n_b + i
        vintage = i + 1
        delta = get_delta(vintage)

        if vintage < n_g:
            d_markov[s, s] = 1 - delta
            d_markov[s, s + 1] = delta
        else:
            d_markov[s, s] = 1 - delta
            d_markov[s, 0] = delta

    return d_grid, d_markov, mapping



            
            
class StageBlockDurables(StageBlock):
    def _impulse_nonlinear(self, ssin, inputs, outputs, ss_initial):
        ss = self.extract_ss_dict(ssin)
        if ss_initial is not None:
            #ss_init = self.extract_ss_dict(ss_initial) #GSCHWEGLER MODIFICATION
            #ss[self.stages[0].name]['D'] = ss_init[self.name][self.stages[0].name]['D']
            ss[self.stages[0].name]['D'] = ss_initial.internals[self.name][self.stages[0].name]['D']

        # report_path is dict(stage: {output: TxN-dim array})
        # lom_path is list[t][stage] in chronological order
        report_path, lom_path = self.backward_nonlinear(ss, inputs)
        
        # D_path is dict(stage: TxN-dim array)
        D_path = self.forward_nonlinear(ss, lom_path)

        aggregates = {}
        for stage in self.stages:
            for o in stage.report:
                if self.M_outputs @ o in outputs:
                    aggregates[self.M_outputs @ o] = utils.optimized_routines.fast_aggregate(D_path[stage.name], report_path[stage.name][o])

        return ImpulseDict(aggregates, T=inputs.T) - ssin
    
class DiscreteChoiceDurables(DiscreteChoice):
    def __matmul__(self, X):
        if self.forward:
            return (self.P * X[np.newaxis, ...]).sum(axis=self.i+2)
            #return batch_multiply_ith_dimension(self.P, self.i, X)
        else:
            return batch_multiply_ith_dimension(self.P_T, self.i, X)
    
class LogitChoiceDurable(LogitChoice):
    def backward_step(self, inputs, lawofmotion=False):
        # start with value we're given
        V_next = inputs[self.value]

        # add dimension at beginning to allow for choice, then swap (today's choice determines next stages's state)
        #V = V_next[np.newaxis, ...]
        #V = np.swapaxes(V, 0, self.index+1)
        #GSCHW Modifications. Replace the two previous lines. Allow for a fantome state.
        V = np.repeat(
        np.expand_dims(V_next, axis=self.index+2),
        V_next.shape[self.index+1],
        axis=self.index+2
        )

        # call f if we have it to get flow utility
        if self.f is not None:
            flow_u = self.f(inputs)
            flow_u = next(iter(flow_u.values()))
        else:
            # create phantom state variable, convenient but bit wasteful
            nchoice = V.shape[0]
            flow_u = np.zeros((nchoice,) + V_next.shape)

        V = flow_u + V
        
        # calculate choice probabilities and expected value
        P, EV = logit_choice(V, inputs[self.taste_shock_scale])
        
        # make law of motion, use it to take expectations of everything else
        lom = DiscreteChoiceDurables(P, self.index)

        # take expectations
        outputs = {k: lom.T @ inputs[k] for k in self.backward}
        outputs[self.value] = EV

        if not lawofmotion:
            return outputs
        else:
            return outputs, lom

    def backward_step_shock(self, ss, shocks, precomputed):
            """See 'discrete choice math' note for background. Note that scale is inverse of 'c' in that note."""
            f, lom = precomputed

            # this part parallel to backward_step, just with derivatives...
            dV_next = shocks[self.value]
            # dV = dV_next[np.newaxis, ...]
            # dV = np.swapaxes(dV, 0, self.index+1)
            #GSCHW Modifications. Replace the two previous lines. Allow for a fantome state.
            dV = np.repeat(
            np.expand_dims(dV_next, axis=self.index+2),
            dV_next.shape[self.index+1],
            axis=self.index+2
            )

            if f is not None:
                dflow_u = f.diff(shocks)
                dflow_u = next(iter(dflow_u.values()))
                dflow_u = np.nan_to_num(dflow_u)  # -inf - (-inf) = nan, want zeros
            else:
                dflow_u = np.zeros_like(lom.P)
            
            dV = dflow_u + dV

            # simply take expectations to get shock to expected value function (envelope result)
            dEV = np.sum(lom.P * dV, axis=0)

            # calculate shocks to choice probabilities (note nifty broadcasting of dEV)
            scale = ss[self.taste_shock_scale]
            dP = lom.P * (dV - dEV) / scale
            dlom = DiscreteChoiceDurables(dP, self.index)

            # find shocks to outputs, aggregate everything of interest
            doutputs = {self.value: dEV}
            for k in self.backward:
                doutputs[k] = dlom.T @ ss[k]
                if k in shocks:
                    doutputs[k] += lom.T @ shocks[k]
            
            return doutputs, dlom