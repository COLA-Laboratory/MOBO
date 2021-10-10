from scipy import optimize
from pyDOE import lhs
import numpy as np


class AOlbfgs:
    def __init__(self, acquisition, lb, ub):
        self.acquisition = acquisition
        self.lb = lb
        self.ub = ub
        self.dim = acquisition.model.X.shape[1]

    def opt(self):
        x0 = lhs(self.dim, 1) * (self.ub - self.lb) + self.lb
        res = optimize.fmin_l_bfgs_b(self.acquisition.value_and_gradient,
                                     x0=x0,
                                     approx_grad=False,
                                     bounds=[[self.lb[i], self.ub[i]] for i in range(len(self.lb))])
        print(res)
        return res[0], res[1]

