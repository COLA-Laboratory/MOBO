from functools import partial
from jax import jit, vmap, config
import jax.numpy as jnp
from typing import Dict
from Kernels.Kernel import VanillaKernel
config.update("jax_enable_x64", True)


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

@jit
def predict(params, x):
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']

    z1 = jnp.dot(w1, x) + b1
    a1 = jit_ReLU(z1)  # .block_until_ready()
    z2 = jnp.dot(w2, x) + b2
    return z2


class SingleLayer(VanillaKernel):
    def __init__(self, dataset_shape, n_hidden, active_dims=None):
        self.dataset_shape = dataset_shape
        self.input_dims = dataset_shape[1]
        self.id = 0
        self.n_hidden = n_hidden
        self.layer1_shape = (n_hidden, input_dims)
        self.layer2_shape = (output_dims, n_hidden)

        w1 = scale * random.normal(w_key, layer1_shape)
        b1 = scale * random.normal(b_key, (n_hidden,))

        w2 = scale * random.normal(w_key, layer2_shape)
        b2 = scale * random.normal(b_key, (output_dims,))

        self.parameters = {
            'w1%d'%d: w1,
            'b1%d'%d: b1,
            'w2%d'%d: w2,
            'b2%d'%d: b2,
        }

        self.mapx1 = vmap(lambda x, y: linear_kernel(x, y), in_axes=(0, None), out_axes=0)
        self.mapx2 = vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)

        if active_dims is None:
            self.active_dims = jnp.arange(dataset_shape[1])
        else:
            self.active_dims = active_dims

    def change_id(self, new_id):
        self.parameters["w1"+str(new_id)] = self.parameters.pop("w1"+str(self.id))
        self.parameters["b1" + str(new_id)] = self.parameters.pop("b1" + str(self.id))
        self.parameters["w2" + str(new_id)] = self.parameters.pop("w2" + str(self.id))
        self.parameters["b2" + str(new_id)] = self.parameters.pop("b2" + str(self.id))
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
        embed = vmap(lambda x: predict(params, x), in_axes=(0,))
        z = embed(X)
        return self.mapx2(z, z)

    def set_parameters(self, params):
       self.parameters["lengthscale"+str(self.id)] = params["lengthscale"+str(self.id)]
       self.parameters["variance"+str(self.id)] = params["variance"+str(self.id)]
