from scipy import optimize
from pyDOE import lhs
import numpy as np


class AOlbfgs:
    def __init__(self, acquisition, lb, ub):
        self.acquisition = acquisition
        self.lb = lb
        self.ub = ub
        self.dim = acquisition.model.X.shape[1]

    def opt(self, restarts=1):

        samples = []
        eis = []
        x0 = lhs(self.dim, restarts) * (self.ub - self.lb) + self.lb
        for i in range(restarts):
            res = optimize.fmin_l_bfgs_b(self.acquisition.value_and_gradient,
                                         x0=x0[i],
                                         approx_grad=False,
                                         bounds=[[self.lb[i], self.ub[i]] for i in range(len(self.lb))])
            samples.append(res[0])
            eis.append(res[1])

        return samples[eis.index(min(eis))], min(eis)

