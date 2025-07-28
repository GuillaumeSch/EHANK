# Standard libraries
import copy
import numpy as np
# Sequence-Jacobian: Stages and Law of Motion
from sequence_jacobian.blocks.support.stages import (
    Continuous1D, ExogenousMaker, LogitChoice, Stage
)
from sequence_jacobian.blocks.support.law_of_motion import LawOfMotion, PolicyLottery1D
from sequence_jacobian.blocks.support import het_compiled
# Sequence-Jacobian: Utilities
from sequence_jacobian.utilities.misc import make_tuple, logit_choice
from sequence_jacobian.utilities.ordered_set import OrderedSet
from sequence_jacobian.utilities.interpolate import interpolate_coord_robust, interpolate_coord


def make_d_grid(n_b=3, n_g=3, lifetime_b=60, lifetime_g=60):
    """
    Create d_grid and Markov matrix for durable goods with two types: brown and green.

    Args:
        n_b (int): number of brown vintages
        n_g (int): number of green vintages
        lifetime_b (int): average lifetime of brown cars (in quarters)
        lifetime_g (int): average lifetime of green cars (in quarters)

    Returns:
        d_grid (np.array): state labels (0 = no car, 1... = vintages)
        d_markov (np.array): Markov transition matrix
        mapping (dict): maps each state index to a tuple (type, vintage)
    """

    delta_b = n_b / lifetime_b
    delta_g = n_g / lifetime_g

    total_states = 1 + n_b + n_g  # state 0 is "no car"
    d_grid = np.arange(total_states)
    d_markov = np.zeros((total_states, total_states))

    mapping = {0: ('none', 0)}
    # Fill brown states
    for i in range(n_b):
        s = 1 + i
        mapping[s] = ('brown', i + 1)

    # Fill green states
    for i in range(n_g):
        s = 1 + n_b + i
        mapping[s] = ('green', i + 1)

    # No car is absorbing
    d_markov[0, 0] = 1.0

    # Brown transitions
    for i in range(n_b):
        s = 1 + i
        if i < n_b - 1:
            d_markov[s, s] = 1 - delta_b         # stay in vintage
            d_markov[s, s + 1] = delta_b         # move to next vintage
        else:
            d_markov[s, s] = 1 - delta_b         # last brown vintage
            d_markov[s, 0] = delta_b             # depreciate to no car

    # Green transitions
    for i in range(n_g):
        s = 1 + n_b + i
        if i < n_g - 1:
            d_markov[s, s] = 1 - delta_g
            d_markov[s, s + 1] = delta_g
        else:
            d_markov[s, s] = 1 - delta_g
            d_markov[s, 0] = delta_g

    return d_grid, d_markov, mapping



class LogitChoiceDurables(LogitChoice):
    def backward_step(self, inputs, lawofmotion=False):
        # start with value we're given
        V_next = inputs[self.value]

        # add dimension at beginning to allow for choice, then swap (today's choice determines next stages's state)
        #V = V_next
        # V = V_next[np.newaxis, ...]
        # V = np.swapaxes(V, 0, self.index+1)

        # call f if we have it to get flow utility
        if self.f is not None:
            flow_u = self.f(inputs)
            flow_u = next(iter(flow_u.values()))
        #else:
            # create phantom state variable, convenient but bit wasteful
            #nchoice = V.shape[0]
            #flow_u = np.zeros((nchoice,) + V_next.shape)
            #flow_u = np.zeros(V_next.shape)

        #V = flow_u + V SHOULD BE ADDED BACK

        # calculate choice probabilities and expected value
        P, EV = logit_choice(V_next, inputs[self.taste_shock_scale])

        # make law of motion, use it to take expectations of everything else
        lom = DiscreteChoiceDurables(P, self.index)
        #lom_2 = DiscreteChoice(P, self.index)

        # take expectations
        outputs = {k: lom.T @ inputs[k] for k in self.backward}
        outputs[self.value] = EV

        if not lawofmotion:
            return outputs
        else:
            return outputs, lom


class DiscreteChoiceDurables(LawOfMotion):
    def __init__(self, P, i):
        self.P = P                     # choice prob P(d|...s_i...), 0 for unavailable choices
        self.i = i                     # dimension of state space that will be updated

        # cache "transposed" version of this, since we'll always need both!
        self.forward = True
        #self.P_T = P.swapaxes(0, 1+self.i).copy()
        self.P_T = P.swapaxes(0, self.i).copy()

    @property
    def T(self):
        newself = copy.copy(self)
        newself.forward = not self.forward
        return newself

    def __matmul__(self, X):
        if self.forward:
            return batch_multiply_ith_dimension(self.P, self.i, X)
        else:
            return batch_multiply_ith_dimension(self.P_T, self.i, X)

def batch_multiply_ith_dimension(P, i, X):
    """If P is (D, X.shape) array, multiply P and X along ith dimension of X."""
    if len(P.shape) <= len(X.shape):
        P = P.swapaxes(1, 1 + i)
        X = X.swapaxes(0, i)
        Pshape = P.shape
        P = P.reshape((*Pshape[:1], -1))
        X = X.reshape((X.shape[0], -1))
        X = np.einsum('jb,jb->b', P, X)
        X = X.reshape(Pshape[0], *Pshape[2:])
    elif len(P.shape) > len(X.shape):
        P = P.swapaxes(1, 1 + i)
        X = X.swapaxes(0, i)
        Pshape = P.shape
        P = P.reshape((*Pshape[:2], -1))
        X = X.reshape((X.shape[0], -1))
        X = np.einsum('ijb,jb->ijb', P, X)
        X = X.reshape(Pshape)

    return X.swapaxes(0, i)


class LogitChoiceOG(LogitChoice):
    def backward_step(self, inputs, lawofmotion=False):
        # start with value we're given
        V_next = inputs[self.value]

        # add dimension at beginning to allow for choice, then swap (today's choice determines next stages's state)
        #V = V_next
        V = V_next[np.newaxis, ...]
        V = np.swapaxes(V, 0, self.index+1)

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
        lom = DiscreteChoiceOG(P, self.index)

        # take expectations
        outputs = {k: lom.T @ inputs[k] for k in self.backward}
        outputs[self.value] = EV

        if not lawofmotion:
            return outputs
        else:
            return outputs, lom


class DiscreteChoiceOG(LawOfMotion):
    def __init__(self, P, i):
        self.P = P                     # choice prob P(d|...s_i...), 0 for unavailable choices
        self.i = i                     # dimension of state space that will be updated

        # cache "transposed" version of this, since we'll always need both!
        self.forward = True
        self.P_T = P.swapaxes(0, 1+self.i).copy()
        #self.P_T = P.swapaxes(0, self.i).copy()

    @property
    def T(self):
        newself = copy.copy(self)
        newself.forward = not self.forward
        return newself

    def __matmul__(self, X):
        if self.forward:
            return batch_multiply_ith_dimension_OG(self.P, self.i, X)
        else:
            return batch_multiply_ith_dimension_OG(self.P_T, self.i, X)

def batch_multiply_ith_dimension_OG(P, i, X):
    """If P is (D, X.shape) array, multiply P and X along ith dimension of X."""
    # standardize arrays
    P = P.swapaxes(1, 1 + i)
    X = X.swapaxes(0, i)
    Pshape = P.shape
    P = P.reshape((*Pshape[:2], -1))
    X = X.reshape((X.shape[0], -1))

    # P[i, j, ...] @ X[j, ...]
    X = np.einsum('ijb,jb->ib', P, X)

    # original shape and order
    X = X.reshape(Pshape[0], *Pshape[2:])
    return X.swapaxes(0, i)


class Exogenous(Stage):
    """Stage that applies exogenous Markov process along one dimension"""
    def __init__(self, markov_name, index, name, backward, hetoutputs=None):
        # subclass-specific attributes
        self.markov_name = markov_name
        self.index = index

        # attributes needed for any stage
        self.name = name
        self.backward_outputs = backward
        self.report = OrderedSet([])
        self.inputs = backward | [markov_name]

        super().__init__(hetoutputs)

    def __repr__(self):
        return f"<Stage-Exogenous '{self.name}' with Markov matrix '{self.markov_name}'>"

    def backward_step(self, inputs, lawofmotion=False):
        Pi = Markov(inputs[self.markov_name], self.index)
        outputs = {k: Pi @ inputs[k] for k in self.backward_outputs}

        if not lawofmotion:
            return outputs
        else:
            return outputs, Pi.T

    def backward_step_shock(self, ss, shocks, precomputed=None):
        Pi = Markov(ss[self.markov_name], self.index)
        outputs = {k: Pi @ shocks[k] for k in self.backward_outputs if k in shocks}

        if self.markov_name in shocks:
            dPi = Markov(shocks[self.markov_name], self.index)
            for k in self.backward_outputs:
                if k in outputs:
                    outputs[k] += dPi @ ss[k]
                else:
                    outputs[k] = dPi @ ss[k]
            return outputs, dPi.T
        else:
            return outputs, None

class Markov(LawOfMotion):
    def __init__(self, Pi, i):
        self.Pi = Pi
        self.i = i

    @property
    def T(self):
        newself = copy.copy(self)
        newself.Pi = newself.Pi.T
        if isinstance(newself.Pi, np.ndarray):
            # optimizing: copy to get right order in memory
            newself.Pi = newself.Pi.copy()
        return newself

    def __matmul__(self, X):
        return multiply_ith_dimension(self.Pi, self.i, X)

def multiply_ith_dimension(Pi, i, X):
    """If Pi is a matrix, multiply Pi times the ith dimension of X and return"""
    X = X.swapaxes(0, i)
    shape = X.shape
    X = X.reshape((shape[0], -1))

    # iterate forward using Pi
    X = Pi @ X

    # reverse steps
    X = X.reshape((Pi.shape[0], *shape[1:]))
    return X.swapaxes(0, i)

class ExogenousMaker:
    """Call make_stage with backward returned by next stage to get Exogenous stage"""
    def __init__(self, markov_name, index, name=None, hetoutputs=None):
        self.markov_name = markov_name
        self.index = index
        if name is None:
            name = f"exog_{markov_name}"
        self.name = name
        self.hetoutputs = hetoutputs

    def make_stage(self, backward):
        return Exogenous(self.markov_name, self.index, self.name, backward, self.hetoutputs)




class Continuous1D_Durables(Continuous1D):
    def backward_step(self, inputs, lawofmotion=False):
            outputs = self.f(inputs)

            if not lawofmotion:
                return outputs
            else:
                return outputs, lottery_1d_Durables(outputs[self.policy], inputs[self.policy + '_grid'], monotonic=False)

def lottery_1d_Durables(a, a_grid, monotonic=False):
    if not monotonic:
        return PolicyLottery1D_Durables(*interpolate_coord_robust(a_grid, a), a_grid)
    else:
        return PolicyLottery1D_Durables(*interpolate_coord(a_grid, a), a_grid)

class PolicyLottery1D_Durables(PolicyLottery1D):
    def __matmul__(self, X):
        if self.forward:
            return het_compiled.forward_policy_1d(X.reshape(self.flatshape), self.i, self.pi).reshape(self.shape).sum(axis=1)
        else:
            return het_compiled.expectation_policy_1d(X.reshape(self.flatshape), self.i, self.pi).reshape(self.shape)
