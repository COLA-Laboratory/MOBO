from functools import partial
from jax import jit, config
import jax.numpy as jnp
from Kernels.Kernel import VanillaKernel
from Embeddings.Matricies import SimpleGaussian, NormalizedGaussian
config.update("jax_enable_x64", True)

@jit
def embedding(params, X):
    return jnp.matmul(X, params["embedding"])


class Learnt(VanillaKernel):
    """
    Sets the D x de matrix as a kernel parameters and learns the by log likelihood optimization.
    """

    def __init__(self, kernel, effective_dim, original_dim):
        self.kernel = kernel
        self.parameters = kernel.parameters
        self.parameters["embedding"] = SimpleGaussian(original_dim, effective_dim)

        self.warping = embedding

    @partial(jit, static_argnums=(0,))
    def function(self, X, params):
        warped = self.warping(params, X)
        return self.kernel.function(warped, params)

    @partial(jit, static_argnums=(0,))
    def cov(self, X, X2):
        X = self.warping(self.parameters, X)
        X2 = self.warping(self.parameters, X2)
        return self.kernel.cov(X, X2)

    def set_parameters(self, params):
        self.parameters = params
        self.kernel.set_parameters(params)


