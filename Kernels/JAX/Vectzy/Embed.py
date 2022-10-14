from functools import partial
from jax import jit, vmap, config
import jax.numpy as jnp
from typing import Dict
from Kernels.Kernel import VanillaKernel

config.update("jax_enable_x64", True)
import numpy as np
from jax import random



@jit
def linear_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum(x * y)


@partial(jit, static_argnums=(2, 3, ))
def predict(params, x, layer1_shape, id):

    w1 = params['w1%d' % id].reshape(layer1_shape)
    b1 = params['b1%d' % id]

    z1 = jnp.dot(w1, x) + b1
    return z1


class Linear(VanillaKernel):
    def __init__(self, dataset_shape, embed_dim, active_dims=None):
        self.dataset_shape = dataset_shape
        self.input_dims = dataset_shape[1]
        self.id = 0
        self.embed_dim = embed_dim
        self.layer1_shape = (embed_dim, self.input_dims)

        self.parameters = {
            'w1%d' % self.id: jnp.ones((self.layer1_shape[0]*self.layer1_shape[1],)),
            'b1%d' % self.id: jnp.ones((embed_dim,)),
        }

        self.mapx1 = vmap(lambda x, y: linear_kernel(x, y), in_axes=(0, None), out_axes=0)
        self.mapx2 = vmap(lambda x, y: self. mapx1(x, y), in_axes=(None, 0), out_axes=1)

        if active_dims is None:
            self.active_dims = jnp.arange(dataset_shape[1])
        else:
            self.active_dims = active_dims

    def change_id(self, new_id):
        self.parameters["w1" + str(new_id)] = self.parameters.pop("w1" + str(self.id))
        self.parameters["b1" + str(new_id)] = self.parameters.pop("b1" + str(self.id))
        self.id = new_id

    @partial(jit, static_argnums=(0,))
    def function(self, X, params):
        return self.gram_matrix(params, X[:, self.active_dims], X[:, self.active_dims])

    @partial(jit, static_argnums=(0,))
    def cov(self, X, X2):
        return self.gram_matrix(self.parameters, X[:, self.active_dims], X2[:, self.active_dims])

    @partial(jit, static_argnums=(0,))
    def gram_matrix(self, params: Dict, x, y) -> jnp.ndarray:
        """
        Computes the covariance matrix.

        Given a function 'callable' with parameters 'params' we use jax.vmap to calculate
        the covariance matrix as the function is applied to each of the points.
        """
        embed = vmap(lambda x: predict(params, x, self.layer1_shape, self.id), in_axes=(0,))
        z1 = embed(x)
        z2 = embed(y)
        return self.mapx2(z1, z2)

    def set_parameters(self, params):
        self.parameters["w1" + str(self.id)] = params["w1" + str(self.id)]
        self.parameters["b1" + str(self.id)] = params["b1" + str(self.id)]
