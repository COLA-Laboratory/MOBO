from functools import partial
from jax import jit, vmap, config
import jax.numpy as jnp
from typing import Dict
from Kernels.Kernel import VanillaKernel
config.update("jax_enable_x64", True)

@jit
def KumaraswamyCDF(params, x):
    return 1 - (1 - x ** params["b"]) ** params["b"]


class KumaraswamyKernel(VanillaKernel):
    """
    Input warping function is the Kumaraswamy CDF: 1 - (1 - x^d)^b
    CDF outputs are bounded to [0, 1]
     -> please ensure your data is also bounded within this range...until
     I suss out a method of not requiring such a normalization
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.parameters = kernel.parameters
        self.parameters["a"] = 1. * jnp.ones((1,))
        self.parameters["b"] = 1. * jnp.ones((1,))
        self.warping = KumaraswamyCDF

    @partial(jit, static_argnums=(0,))
    def function(self, X, params):
        warped = self.warping(params, X)
        return self.kernel.function(warped, params)

    @partial(jit, static_argnums=(0,))
    def cov(self, X, X2):
        return self.kernel.cov(
            self.warping(self.parameters, X),
            self.warping(self.parameters, X2)
        )

    def set_parameters(self, params):
        self.parameters = params


