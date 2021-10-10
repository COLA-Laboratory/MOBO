from jax import jit, config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial


@jit
def ieuclidean_distance(X, diagonals):
    Xsq = jnp.sum(jnp.square(X), 1)
    r2 = -2. * jnp.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    r2 = r2 * diagonals
    #r2 = jnp.clip(r2, 0, jnp.inf)
    return jnp.sqrt(r2)

@jit
def ieuclidean_distance_s(X, diagonals):
    Xsq = jnp.sum(jnp.square(X), 1)
    r2 = -2. * jnp.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    r2 = r2 * diagonals
    return r2

@jit
def euclidean_distance(X, X2):
    X1sq = jnp.sum(jnp.square(X), 1)
    X2sq = jnp.sum(jnp.square(X2), 1)
    r2 = -2. * jnp.dot(X, X2.T) + (X1sq[:, None] + X2sq[None, :])
    r2 = jnp.clip(r2, 0, jnp.inf)
    return jnp.sqrt(r2)

@jit
def euclidean_distance_s(X , X2):
    X1sq = jnp.sum(jnp.square(X), 1)
    X2sq = jnp.sum(jnp.square(X2), 1)
    r2 = -2. * jnp.dot(X, X2.T) + (X1sq[:, None] + X2sq[None, :])
    r2 = jnp.clip(r2, 0, jnp.inf)
    return r2


class RBF:
    def __init__(self, dataset_shape, lengthscale=1., variance=1., ARD=False):
        dim = dataset_shape[1]
        if ARD:
            self.parameters = {"variance": jnp.ones((1,)) * variance * 1.,
                               "lengthscale": jnp.ones((dim,)) * lengthscale * 1.}
            self.function = self.ARDf
            self.cov = self.ARD

        else:
            self.parameters = {"variance": jnp.ones((1,)) * variance * 1.,
                                "lengthscale": jnp.ones((1,)) * lengthscale * 1.}
            self.function = self.NARDf
            self.cov = self.NARD

        self.diagonal = np.ones((dataset_shape[0], dataset_shape[0]))
        np.fill_diagonal(self.diagonal, 0.)

    @partial(jit, static_argnums=(0,))
    def NARDf(self, X, params):
        r = ieuclidean_distance(X, self.diagonal) / params["lengthscale"]
        return params["variance"] * jnp.exp(-0.5 * r ** 2)

    @partial(jit, static_argnums=(0,))
    def ARDf(self, X, params):
        r = ieuclidean_distance_s(X / params["lengthscale"], self.diagonal)
        return params["variance"] * jnp.exp(-0.5 * r)

    @partial(jit, static_argnums=(0,))
    def NARD(self, X, X2):
        r = euclidean_distance_s(X, X2) / self.parameters["lengthscale"]
        return self.parameters["variance"] * jnp.exp(-0.5 * r)

    @partial(jit, static_argnums=(0,))
    def ARD(self, X, X2):
        r = euclidean_distance_s(X / self.parameters["lengthscale"], X2 / self.parameters["lengthscale"])
        return self.parameters["variance"] * jnp.exp(-0.5 * r)

    def set_parameters(self, params):
        self.parameters["lengthscale"] = params["lengthscale"]
        self.parameters["variance"] = params["variance"]