"""
Vanilla Bayesian Optimization (Minimization) with RBF ARD Kernel.
Created By Phoenix Williams 11/10/2021
"""
from pyDOE import lhs  # for initial dataset
import numpy as np

# Mobo Libraries
from Kernels.JAX.rbf import RBF
from Likelihoods.JAX.chol import Likelihood
from Models.JAX.GPregression import GPregression
from Acquisitions.JAX.EI import AcquisitionEI
from ModelOptimizers.lbfgsb import lbfgsb
from AcquisitionOptimizers.lbfgsb import AOlbfgs


class BO:
    def __init__(self, function, iterations, lb, ub, dim, x_init=None, y_init=None, n_init=10.):

        self.function = function
        self.iterations = iterations
        self.lb = lb
        self.ub = ub
        self.dim = dim

        assert len(x_init) == len(y_init) or x_init == y_init

        if x_init is None:
            self.x_init = lhs(dim, n_init) * (self.ub - self.lb) + self.lb
            self.y_init = np.array([self.function(xi) for xi in self.x_init]).reshape(-1, 1)

    def opt(self, verbose=None):

        if verbose is None:
            verbose = {"verbose": False,
                       "mod": 1}

        x = self.x_init
        y = self.y_init

        for it in range(self.iterations):

            if verbose["verbose"] and it % verbose["mod"] == 0:
                print("iteration %i min value: %.5f" % (it, min(y)))

            # 1. Design and Train Model
            kernel = RBF(x.shape, ARD=True)
            model = GPregression(kernel, x, y)
            likelihood = Likelihood(model)
            # call likelihood.evaluate() -> if you do not wish to train the model
            model_optimizer = lbfgsb(model)
            model_optimizer.opt()

            # 2. Select next sample point
            acquisition = AcquisitionEI(model_optimizer, min(y))
            acquisition_optimizer = AOlbfgs(acquisition, self.lb, self.ub)
            sample, ei_val = acquisition_optimizer.opt(restarts=5)

            # 3. Evaluate Sample and append to the dataset
            s_eval = self.function(sample)
            x = np.concatenate([x, sample])
            y = np.concatenate([y, s_eval], axis=0)

        return x, y
