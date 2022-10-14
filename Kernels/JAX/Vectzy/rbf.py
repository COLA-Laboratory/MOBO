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


@partial(jit, static_argnums=(3,))
def NARDf(params, x, y, id):
    r = euclidean_distance(x, y) / params["lengthscale"+str(id)][0]
    return params["variance"+str(id)][0] * jnp.exp(-0.5 * r ** 2)


@partial(jit, static_argnums=(3,))
def ARDf(params, x, y, id):
    r = seuclidean_distance(x / params["lengthscale"+str(id)], y / params["lengthscale"+str(id)])
    return params["variance"+str(id)][0] * jnp.exp(-0.5 * r)


class RBF(VanillaKernel):
    def __init__(self, dataset_shape, lengthscale=1., variance=1., ARD=False, active_dims=None):
        self.dataset_shape = dataset_shape
        self.id = 0
        if active_dims is not None:
            l = len(active_dims)
            self.active_dims = jnp.arange(l)
        else:
            l = dataset_shape[1]
            self.active_dims = active_dims
        if ARD:
            self.parameters = {"variance"+str(self.id): jnp.ones((1,)) * variance * 1.,
                               "lengthscale"+str(self.id): jnp.ones((l,)) * lengthscale * 1.}

            self.rbf_kernel = ARDf
        else:
            self.parameters = {"variance"+str(self.id): jnp.ones((1,)) * variance * 1.,
                               "lengthscale"+str(self.id): jnp.ones((1,)) * lengthscale * 1.}
            self.rbf_kernel = NARDf

    def change_id(self, new_id):
        self.parameters["variance"+str(new_id)] = self.parameters.pop("variance"+str(self.id))
        self.parameters["lengthscale" + str(new_id)] = self.parameters.pop("lengthscale" + str(self.id))
        self.id = new_id

    @partial(jit, static_argnums=(0, ))
    def function(self, X, params):
        return self.gram_matrix(params, X[:, self.active_dims], X[:, self.active_dims])

    @partial(jit, static_argnums=(0, ))
    def cov(self, X, X2):
        return self.gram_matrix(self.parameters, X[:, self.active_dims], X2[:, self.active_dims])

    @partial(jit, static_argnums=(0, ))
    def gram_matrix(self, params: Dict, x, y) -> jnp.ndarray:
        """
        Computes the covariance matrix.

        Given a function 'callable' with parameters 'params' we use jax.vmap to calculate
        the covariance matrix as the function is applied to each of the points.
        """
        mapx1 = vmap(lambda x, y: self.rbf_kernel(params, x, y, self.id), in_axes=(0, None), out_axes=0)
        mapx2 = vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)

        return mapx2(x, y)

    def set_parameters(self, params):
       self.parameters["lengthscale"+str(self.id)] = params["lengthscale"+str(self.id)]
       self.parameters["variance"+str(self.id)] = params["variance"+str(self.id)]
