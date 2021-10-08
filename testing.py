from Kernels.JAX.rbf import RBF as RBFj
from Models.JAX.GPregression import GPregression as JGP
from Likelihoods.JAX.chol import Likelihood as JL
from Likelihoods.Analytical.gpy import Likelihood

from Kernels.Analytical.rbf import RBF
from Models.Analytical.GPregression import GPregression

from GPy.kern import RBF as RBFg
from GPy.models import GPRegression as GyGP

from ModelOptimizers.lbfgsb import lbfgsb

import numpy as np

log_2_pi = np.log(2. * np.pi)


def sphere(x):
    return sum(xi**2 for xi in x)


if __name__ == "__main__":
    import time
    np.random.seed(0)
    dim = 10
    f = sphere
    X = np.random.uniform(-5, 5, (1000, dim))
    y = np.array([f(xi) for xi in X]).reshape(-1, 1)

    gpy_kern = RBFg(input_dim=dim, ARD=True)
    gpy_model = GyGP(X, y, kernel=gpy_kern)
    start = time.time()
    #gpy_model.optimize()
    print("gpy:", time.time() - start)
    print(gpy_model.log_likelihood())

    #print("gpy:", gpy_model.log_likelihood(), gpy_model.gradient)

    #mobo_kern = RBF(input_dim=dim, ARD=True)
    #mobo_model = GPregression(mobo_kern, X, y)
    #mobo_likelihood = Likelihood(mobo_model)
    #mobo_optimizer = lbfgsb(mobo_model)
    #start = time.time()
    #mobo_optimizer.opt()
    #print("mobo:", time.time() - start)
    #print(mobo_model.log_likelihood)
    #print(mobo_likelihood.objective_and_grad(mobo_model.parameters))

    #print("mobo:", mobo_model.log_likelihood, gpy_model.gradient)
    jax_kern = RBFj(X.shape, ARD=True)
    jax_model = JGP(jax_kern, X, y)
    jax_likelihood = JL(jax_model)
    jax_optimizer = lbfgsb(jax_model)
    start = time.time()
    #jax_optimizer.opt()
    print("jax:", time.time() - start)
    print(jax_model.log_likelihood)

    #print("jax:", jax_model.log_likelihood, jax_model.gradients)
    #print("jax:", jax_model.objective_and_grad)
