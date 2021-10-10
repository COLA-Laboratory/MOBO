"""
GPY acquisitions
"""
import numpy as np
from Acquisitions.JAX.EI import AcquisitionEI
from Kernels.JAX.rbf import RBF
from Models.JAX.GPregression import GPregression
from Likelihoods.JAX.chol import Likelihood
from ModelOptimizers.lbfgsb import lbfgsb
from scipy.special import erfc

import matplotlib.pyplot as plt

from GPy.kern import RBF as RBFg
from GPy.models import GPRegression as gpy


class GPyEI:
    def __init__(self, model, fmin, jitter=0.01):
        self.model = model
        self.jitter = jitter
        self.fmin = fmin

    def predict_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        """
        if X.ndim == 1: X = X[None, :]
        m, v = self.model.predict(X)
        v = np.clip(v, 1e-10, np.inf)
        dmdx, dvdx = self.model.predictive_gradients(X)
        dmdx = dmdx[:, :, 0]
        dsdx = dvdx / (2 * np.sqrt(v))

        return m, np.sqrt(v), dmdx, dsdx

    def get_quantiles(self, acquisition_par, fmin, m, s):
        if isinstance(s, np.ndarray):
            s[s < 1e-10] = 1e-10
        elif s < 1e-10:
            s = 1e-10
        u = (fmin - m - acquisition_par) / s
        phi = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
        Phi = 0.5 * erfc(-u / np.sqrt(2))
        return (phi, Phi, u)

    def _compute_acq_withGradients(self, x):
        """
        Computes the Expected Improvement and its derivative (has a very easy derivative!)
        """
        m, s, dmdx, dsdx = self.predict_withGradients(x)
        phi, Phi, u = self.get_quantiles(self.jitter, self.fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        df_acqu = dsdx * phi - Phi * dmdx
        return f_acqu[0][0], -df_acqu[0]

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost
        """
        m, s = self.model.predict(x)
        fmin = self.fmin
        phi, Phi, u = self.get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        return -f_acqu


def sphere(x):
    return sum(xi**2 for xi in x)


def rastrigin(x):
    sum1 =0
    d = len(x)
    for i in range(d):
        sum1 += (x[i] ** 2 - 10 * np.cos(1 * np.pi * x[i]))
    return 10 * d + sum1


if __name__ == "__main__":
    from jax import value_and_grad, vjp
    import jax.numpy as jnp
    np.random.seed(1)
    dim = 1
    f = sphere
    X = np.random.uniform(-5, 5, (5, dim))
    y = np.array([f(xi) for xi in X]).reshape(-1, 1)

    #plt.plot(X, y, 'o')
    #plt.xlim(-5, 5)
    #plt.show()

    jax_kernel = RBF(dataset_shape=X.shape, ARD=False)
    jax_model = GPregression(jax_kernel, X, y)
    jax_likelihood = Likelihood(jax_model)
    jax_optimier = lbfgsb(jax_model)
    jax_likelihood.evaluate()
    #jax_optimier.opt()
    jax_ei = AcquisitionEI(jax_model, min(y))

    Xtest = np.linspace(-5, 5, 100).reshape(-1, 1)
    Xtest1 = np.random.uniform(-5, 5, (1, dim))

    eis = np.array([jax_ei.value_and_gradient(xi.reshape(-1, 1))[0] for xi in Xtest]).flatten()

    #plt.plot(Xtest.flatten(), eis)

    #print(jax_ei.value_and_gradient(Xtest1))

    gpy_kernel = RBFg(input_dim=dim, ARD=False)
    gpy_model = gpy(X, y, gpy_kernel)
    gpy_model.optimize()

    gpy_ei = GPyEI(gpy_model, min(y))
    eis = np.array([gpy_ei._compute_acq(xi.reshape(-1, 1)) for xi in Xtest]).flatten()

    plt.plot(Xtest.flatten(), eis)
    plt.legend(["jax", "gpy"])
    plt.show()