from Kernels.Kernel import VanillaKernel
from Kernels.MultiTask.Coregionalize import Coregionalize
from functools import partial
from jax import jit


def LCM(input_dim, num_outputs, kernels_list, W_ranks):
    K = ICM(input_dim, num_outputs, kernels_list[0], W_ranks[0])
    for i in range(len(kernels_list[1:])):
        K += ICM(input_dim, num_outputs, kernels_list[i], W_ranks[i])

    return K


class ICM(VanillaKernel):
    def __init__(self, input_dim, num_outputs, kernel, W_rank=1):
        self.num_outputs = num_outputs
        self.kernel = kernel
        self.icm = Coregionalize(num_outputs, rank=W_rank)
        self.input_dim = input_dim

    @partial(jit, static_argnums=(0,))
    def function(self, X, params):
        """
        X is of the format [[data, index]]
        """
        kx = self.kernel.function(X[:, :self.input_dim], params)
        kx *= self.icm.function(X[:, self.input_dim, None], params)
        return kx

    @partial(jit, static_argnums=(0,))
    def cov(self, X, X2):
        kx = self.kernel.cov(X[:, :self.input_dim], X2[:, :self.input_dim])
        kx *= self.icm.cov(X[:, self.input_dim, None], X2[:, self.input_dim, None])
        return kx

    def change_id(self, new_id):
        self.icm.change_id(new_id)
        self.kernel.change_id(new_id)

    def set_parameters(self, params):
        self.icm.set_parameters(params)
        self.kernel.set_parameters(params)

    @property
    def parameters(self):
        p = self.kernel.parameters
        p.update(self.icm.parameters)
        return p
