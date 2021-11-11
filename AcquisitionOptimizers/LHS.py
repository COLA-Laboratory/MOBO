from pyDOE import lhs
import numpy as np


class LHS:
    def __init__(self, acquisition, lb, ub):
        self.acquisition = acquisition
        self.lb = lb
        self.ub = ub
        self.dim = acquisition.model.X.shape[1]

    def opt(self, n):
        x = lhs(self.dim, n) * (self.ub - self.lb) + self.lb
        ei = np.nan_to_num(self.acquisition.collective_function(x), nan=0.0)

        idx = np.where(ei == np.min(ei))
        return x[idx][0], ei[idx][0]