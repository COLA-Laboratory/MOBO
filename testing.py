"""
Example of how to use the current version of the library.
"""

from Kernels.JAX.rbf import RBF as RBFj
from Models.JAX.GPregression import GPregression as JGP
from Likelihoods.JAX.chol import Likelihood as JL

from ModelOptimizers.lbfgsb import lbfgsb

import numpy as np

log_2_pi = np.log(2. * np.pi)


def sphere(x):
    return sum(xi**2 for xi in x)


def rastrigin(x):
    sum1 =0
    d = len(x)
    for i in range(d):
        sum1 += (x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]))
    return 10 * d + sum1


if __name__ == "__main__":
    import time
    np.random.seed(0)
    dim = 10
    f = rastrigin
    X = np.random.uniform(-5, 5, (100, dim))
    y = np.array([f(xi) for xi in X]).reshape(-1, 1)

    Xtest = np.random.uniform(-5, 5, (100, dim))
    ytest = np.array([f(xi) for xi in Xtest]).reshape(-1, 1)

    jax_kern = RBFj(X.shape, ARD=True)
    jax_model = JGP(jax_kern, X, y)
    jax_likelihood = JL(jax_model)
    jax_optimizer = lbfgsb(jax_model)
    jax_likelihood.evaluate()
    start = time.time()
    jax_optimizer.opt()
    print("jax:", time.time() - start)
    print(jax_model.log_likelihood)

    jax_mean, jax_variance = jax_model.predict(Xtest, full_cov=True)
    print("jax:", np.mean(abs(jax_mean - ytest)))

