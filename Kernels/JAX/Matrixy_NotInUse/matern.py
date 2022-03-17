from jax import jit, config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
import numpy as np
from Kernels.Kernel import VanillaKernel


@jit
def ieuclidean_distance(X, diagonals):
    Xsq = jnp.sum(jnp.square(X), 1)
    r2 = -2. * jnp.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    r2 = r2 * diagonals
    r2 = jnp.clip(r2, 0, jnp.inf)
    #return jnp.sqrt(r2)
    return r2


@jit
def euclidean_distance(X, X2):
    X1sq = jnp.sum(jnp.square(X), 1)
    X2sq = jnp.sum(jnp.square(X2), 1)
    r2 = -2. * jnp.dot(X, X2.T) + (X1sq[:, None] + X2sq[None, :])
    r2 = jnp.clip(r2, 0, jnp.inf)
    #return jnp.sqrt(r2)
    return r2


class MATERN32(VanillaKernel):
    def __init__(self, dataset_shape, lengthscale=1., variance=1., ARD=False):
        dim = dataset_shape[1]
        self.id = 0
        if ARD:
            self.parameters = {"variance"+str(self.id): jnp.ones((1,)) * variance * 1.,
                               "lengthscale"+str(self.id): jnp.ones((dim,)) * lengthscale * 1.}
            self.function = self.ARDf
            self.cov = self.ARD

        else:
            self.parameters = {"variance"+str(self.id): jnp.ones((1,)) * variance * 1.,
                                "lengthscale"+str(self.id): jnp.ones((1,)) * lengthscale * 1.}
            self.function = self.NARDf
            self.cov = self.NARD

        self.diagonal = np.ones((dataset_shape[0], dataset_shape[0]))
        np.fill_diagonal(self.diagonal, 0.)

    @partial(jit, static_argnums=(0,))
    def NARDf(self, X, params):
        r = ieuclidean_distance(X, self.diagonal) / params["lengthscale"+str(self.id)]
        return params["variance"+str(self.id)] * (1. + jnp.sqrt(3.) * r) * jnp.exp(-jnp.sqrt(3.) * r)

    @partial(jit, static_argnums=(0,))
    def ARDf(self, X, params):
        r = ieuclidean_distance(X / params["lengthscale"+str(self.id)], self.diagonal)
        return params["variance"+str(self.id)] * (1. + jnp.sqrt(3.) * r) * jnp.exp(-jnp.sqrt(3.) * r)

    @partial(jit, static_argnums=(0,))
    def NARD(self, X, X2):
        r = euclidean_distance(X, X2) / self.parameters["lengthscale"+str(self.id)]
        return self.parameters["variance"+str(self.id)] * (1. + jnp.sqrt(3.) * r) * jnp.exp(-jnp.sqrt(3.) * r)

    @partial(jit, static_argnums=(0,))
    def ARD(self, X, X2):
        r = euclidean_distance(X / self.parameters["lengthscale"+str(self.id)], X2 / self.parameters["lengthscale"])
        return self.parameters["variance"+str(self.id)] * (1. + jnp.sqrt(3.) * r) * jnp.exp(-jnp.sqrt(3.) * r)

    def set_parameters(self, params):
        self.parameters["lengthscale"+str(self.id)] = params["lengthscale"+str(self.id)]
        self.parameters["variance"+str(self.id)] = params["variance"+str(self.id)]

    def change_id(self, new_id):
        self.parameters["variance"+str(new_id)] = self.parameters.pop("variance"+str(self.id))
        self.parameters["lengthscale" + str(new_id)] = self.parameters.pop("lengthscale" + str(self.id))
        self.id = new_id


class MATERN52(VanillaKernel):
    def __init__(self, dataset_shape, lengthscale=1., variance=1., ARD=False):
        dim = dataset_shape[1]
        self.id = 0
        if ARD:
            self.parameters = {"variance"+str(self.id): jnp.ones((1,)) * variance * 1.,
                               "lengthscale"+str(self.id): jnp.ones((dim,)) * lengthscale * 1.}
            self.function = self.ARDf
            self.cov = self.ARD

        else:
            self.parameters = {"variance"+str(self.id): jnp.ones((1,)) * variance * 1.,
                                "lengthscale"+str(self.id): jnp.ones((1,)) * lengthscale * 1.}
            self.function = self.NARDf
            self.cov = self.NARD

        self.diagonal = np.ones((dataset_shape[0], dataset_shape[0]))
        np.fill_diagonal(self.diagonal, 0.)

    @partial(jit, static_argnums=(0,))
    def NARDf(self, X, params):
        r = ieuclidean_distance(X, self.diagonal) / params["lengthscale"+str(self.id)]
        return params["variance"+str(self.id)]*(1+jnp.sqrt(5.)*r+5./3*r**2)*jnp.exp(-jnp.sqrt(5.)*r)

    @partial(jit, static_argnums=(0,))
    def ARDf(self, X, params):
        r = ieuclidean_distance(X / params["lengthscale"+str(self.id)], self.diagonal)
        return params["variance"+str(self.id)]*(1+jnp.sqrt(5.)*r+5./3*r**2)*jnp.exp(-jnp.sqrt(5.)*r)

    @partial(jit, static_argnums=(0,))
    def NARD(self, X, X2):
        r = euclidean_distance(X, X2) / self.parameters["lengthscale"+str(self.id)]
        return self.parameters["variance"+str(self.id)]*(1+jnp.sqrt(5.)*r+5./3*r**2)*jnp.exp(-jnp.sqrt(5.)*r)

    @partial(jit, static_argnums=(0,))
    def ARD(self, X, X2):
        r = euclidean_distance(X / self.parameters["lengthscale"+str(self.id)], X2 / self.parameters["lengthscale"])
        return self.parameters["variance"+str(self.id)]*(1+jnp.sqrt(5.)*r+5./3*r**2)*jnp.exp(-jnp.sqrt(5.)*r)

    def set_parameters(self, params):
        self.parameters["lengthscale"+str(self.id)] = params["lengthscale"+str(self.id)]
        self.parameters["variance"+str(self.id)] = params["variance"+str(self.id)]

    def change_id(self, new_id):
        self.parameters["variance"+str(new_id)] = self.parameters.pop("variance"+str(self.id))
        self.parameters["lengthscale" + str(new_id)] = self.parameters.pop("lengthscale" + str(self.id))
        self.id = new_id