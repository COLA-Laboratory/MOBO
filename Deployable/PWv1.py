"""
Vanilla Bayesian Optimization (Minimization) with RBF ARD Kernel.
Created By Phoenix Williams 11/10/2021
"""
from pyDOE import lhs  # for initial dataset
import numpy as np

# Mobo Libraries
from Kernels.JAX.Vectzy.rbf import RBF
from Likelihoods.JAX.chol import Likelihood
from Models.JAX.GPregression import GPregression
from ModelOptimizers.lbfgsb import lbfgsb

from Acquisitions.JAX.EI import AcquisitionEI
from AcquisitionOptimizers.lbfgsb import AOlbfgs

from GPy.models import GPRegression
from GPy.kern import RBF as gpymatern


class BO:
    def __init__(self, function, iterations, lb, ub, x_init=None, y_init=None, n_init: int = 10):

        self.function = function
        self.iterations = iterations
        self.lb = lb
        self.ub = ub
        assert len(lb) == len(ub)
        assert type(n_init) == int
        self.dim = len(lb)

        if x_init is None:
            assert y_init is None
            self.x_init = lhs(self.dim, n_init) * (self.ub - self.lb) + self.lb
            self.y_init = np.array([self.function(xi) for xi in self.x_init]).reshape(-1, 1)

        else:
            self.x_init = x_init
            self.y_init = y_init

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
            kernel = RBF(dataset_shape=x.shape, ARD=True)
            model = GPregression(kernel, x, y)
            likelihood = Likelihood(model)
            # call likelihood.evaluate() -> if you do not wish to train the model
            model_optimizer = lbfgsb(model)
            model_optimizer.opt()
            print("MOBO:", model.log_likelihood)

            # 1. Design and Train Model
            gpy_kernel = gpymatern(input_dim=self.dim, ARD=True)
            gpy_model = GPRegression(x, y, kernel=gpy_kernel)
            gpy_model.optimize()
            print("GPY:", gpy_model.log_likelihood())

            # 2. Select next sample point
            acquisition = AcquisitionEI(model, min(y))
            acquisition_optimizer = AOlbfgs(acquisition, self.lb, self.ub)
            sample, ei_val = acquisition_optimizer.opt(restarts=10, verbose=False)

            # 3. Evaluate Sample and append to the dataset
            s_eval = self.function(sample)
            x = np.concatenate([x, np.array([sample])], axis=0)
            y = np.concatenate([y, np.array([[s_eval]])], axis=0)

        print("final min value:", min(y))
        process = [self.x_init[np.where(self.y_init == min(y))]]
        process.extend(x[len(self.x_init):])

        y_process = [min(self.y_init)]
        y_process.extend(y[len(self.y_init):])

        return process, y_process
