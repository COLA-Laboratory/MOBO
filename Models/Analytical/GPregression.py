"""
Single task Gaussian Process for Regression tasks
"""
import numpy as np


class GPregression:
    def __init__(self, kernel, X, y, noise=1.):
        self.kernel = kernel
        self.variance = np.ones((1,)) * noise * 1.
        self.variance_bounds = [[1e-100, None]]

        self.X = X
        self.y = y

        self.likelihood = None

    @property
    def bounds(self):
        b = self.kernel.bounds
        b["noise"] = self.variance_bounds
        return b

    @property
    def parameters(self):
        p = self.kernel.parameters
        p["noise"] = self.variance
        return p

    def set_parameters(self, params):
        self.variance = params["noise"]
        self.kernel.set_parameters(params)

    @property
    def log_likelihood(self):
        return -self.likelihood.log_likelihood_and_grad(self.parameters)[0]

    @property
    def gradients(self):
        grads = self.likelihood.log_likelihood_and_grad(self.parameters)[1]
        for gi in grads:
            grads[gi] = -grads[gi]

        return grads

    def predict(self, Xnew, full_cov=False):
        if full_cov:
            mu, var = self.likelihood.predict(Xnew)
            return mu, var
        else:
            mu, var = self.likelihood.predict(Xnew)
            return mu, np.diag(var)