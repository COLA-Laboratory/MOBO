from functools import partial
from jax import jit, vmap, config
import jax.numpy as jnp
from typing import Dict
from Kernels.Kernel import VanillaKernel
config.update("jax_enable_x64", True)


@jit
def seuclidean_distance(x, y) -> float:
    return jnp.sum((x - y) ** 2)


@jit
def euclidean_distance(x, y) -> float:
    return jnp.sqrt(seuclidean_distance(x, y))


@jit
def NARDf32(params, x, y):
    r = euclidean_distance(x, y) / params["lengthscale"][0]
    return params["variance"][0] * (1. + jnp.sqrt(3.) * r) * jnp.exp(-jnp.sqrt(3.) * r)

@jit
def ARDf32(params, x, y):
    r = seuclidean_distance(x / params["lengthscale"], y / params["lengthscale"])
    return params["variance"][0] * (1. + jnp.sqrt(3.) * r) * jnp.exp(-jnp.sqrt(3.) * r)


@jit
def NARDf52(params, x, y):
    r = euclidean_distance(x, y) / params["lengthscale"][0]
    return params["variance"][0]*(1+jnp.sqrt(5.)*r+5./3*r**2)*jnp.exp(-jnp.sqrt(5.)*r)

@jit
def ARDf52(params, x, y):
    r = seuclidean_distance(x / params["lengthscale"], y / params["lengthscale"])
    return params["variance"][0]*(1+jnp.sqrt(5.)*r+5./3*r**2)*jnp.exp(-jnp.sqrt(5.)*r)


class MATERN32(VanillaKernel):
    def __init__(self, dataset_shape, lengthscale=1., variance=1., ARD=False):
        self.dataset_shape = dataset_shape
        if ARD:
            self.parameters = {"variance": jnp.ones((1,)) * variance * 1.,
                               "lengthscale": jnp.ones((dataset_shape[1],)) * lengthscale * 1.}

            self.rbf_kernel = ARDf32
        else:
            self.parameters = {"variance": jnp.ones((1,)) * variance * 1.,
                               "lengthscale": jnp.ones((1,)) * lengthscale * 1.}
            self.rbf_kernel = NARDf32

    @partial(jit, static_argnums=(0, ))
    def function(self, X, params):
        return self.gram_matrix(params, X, X)

    @partial(jit, static_argnums=(0, ))
    def cov(self, X, X2):
        return self.gram_matrix(self.parameters, X, X2)

    @partial(jit, static_argnums=(0, ))
    def gram_matrix(self, params: Dict, x, y) -> jnp.ndarray:
        """
        Computes the covariance matrix.

        Given a function 'callable' with parameters 'params' we use jax.vmap to calculate
        the covariance matrix as the function is applied to each of the points.
        """

        mapx1 = vmap(lambda x, y: self.rbf_kernel(params, x, y), in_axes=(0, None), out_axes=0)
        mapx2 = vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)

        return mapx2(x, y)

    def set_parameters(self, params):
        self.parameters["lengthscale"] = params["lengthscale"]
        self.parameters["variance"] = params["variance"]


class MATERN52(VanillaKernel):
    def __init__(self, dataset_shape, lengthscale=1., variance=1., ARD=False):
        self.dataset_shape = dataset_shape
        if ARD:
            self.parameters = {"variance": jnp.ones((1,)) * variance * 1.,
                               "lengthscale": jnp.ones((dataset_shape[1],)) * lengthscale * 1.}

            self.rbf_kernel = ARDf52
        else:
            self.parameters = {"variance": jnp.ones((1,)) * variance * 1.,
                               "lengthscale": jnp.ones((1,)) * lengthscale * 1.}
            self.rbf_kernel = NARDf52

    @partial(jit, static_argnums=(0, ))
    def function(self, X, params):
        return self.gram_matrix(params, X, X)

    @partial(jit, static_argnums=(0, ))
    def cov(self, X, X2):
        return self.gram_matrix(self.parameters, X, X2)

    @partial(jit, static_argnums=(0, ))
    def gram_matrix(self, params: Dict, x, y) -> jnp.ndarray:
        """
        Computes the covariance matrix.

        Given a function 'callable' with parameters 'params' we use jax.vmap to calculate
        the covariance matrix as the function is applied to each of the points.
        """

        mapx1 = vmap(lambda x, y: self.rbf_kernel(params, x, y), in_axes=(0, None), out_axes=0)
        mapx2 = vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)

        return mapx2(x, y)

    def set_parameters(self, params):
        self.parameters["lengthscale"] = params["lengthscale"]
        self.parameters["variance"] = params["variance"]
