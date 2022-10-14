from functools import partial
from jax import jit, vmap, config
import jax.numpy as jnp
from typing import Dict
from Kernels.Kernel import VanillaKernel

config.update("jax_enable_x64", True)
import numpy as np
from jax import random


@jit
def relu(x):
    return jnp.maximum(0, x)


@jit
def linear_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear kernel
    .. math:: k_i = \sum_i^N x_i-y_i
    Parameters
    ----------
    params : None
        kept for compatibility
    x : jax.numpy.ndarray
        the inputs
    y : jax.numpy.ndarray
        the inputs
    Returns
    -------
    kernel_mat : jax.numpy.ndarray
        the kernel matrix (n_samples, n_samples)
    """
    return np.sum(x * y)


@partial(jit, static_argnums=(2, 3, 4, 5, 6, ))
def predict(params, x, layer1_shape, layer2_shape, n_hidden, input_dims, id):

    w1 = params['w1%d' % id].reshape(layer1_shape)
    b1 = params['b1%d' % id]#.reshape((n_hidden,))
    w2 = params['w2%d' % id].reshape(layer2_shape)
    b2 = params['b2%d' % id]#.reshape((input_dims,))

    z1 = jnp.dot(w1, x) + b1
    a1 = relu(z1)  # .block_until_ready()
    z2 = jnp.dot(w2, a1) + b2
    return z2


class SingleLayer(VanillaKernel):
    def __init__(self, dataset_shape, n_hidden, active_dims=None):
        self.dataset_shape = dataset_shape
        self.input_dims = dataset_shape[1]
        self.id = 0
        self.n_hidden = n_hidden
        self.layer1_shape = (n_hidden, self.input_dims)
        self.layer2_shape = (self.input_dims, n_hidden)

        key = random.PRNGKey(0)

        self.parameters = {
            'w1%d' % self.id: jnp.ones((self.layer1_shape[0]*self.layer1_shape[1],)),
            'b1%d' % self.id: jnp.ones((self.n_hidden,)),
            'w2%d' % self.id: jnp.ones((self.layer2_shape[0]*self.layer2_shape[1],)),
            'b2%d' % self.id: jnp.ones((self.input_dims,)),
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
        self.parameters["w2" + str(new_id)] = self.parameters.pop("w2" + str(self.id))
        self.parameters["b2" + str(new_id)] = self.parameters.pop("b2" + str(self.id))
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
        embed = vmap(lambda x: predict(params, x, self.layer1_shape, self.layer2_shape,
                                       self.n_hidden, self.input_dims, self.id), in_axes=(0,))
        z1 = embed(x)
        z2 = embed(y)
        return self.mapx2(z1, z2)

    def set_parameters(self, params):
        self.parameters["w1" + str(self.id)] = params["w1" + str(self.id)]
        self.parameters["w2" + str(self.id)] = params["w2" + str(self.id)]
        self.parameters["b1" + str(self.id)] = params["b1" + str(self.id)]
        self.parameters["b2" + str(self.id)] = params["b2" + str(self.id)]
