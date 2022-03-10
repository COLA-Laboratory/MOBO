from Kernels.Kernel import VanillaKernel
from Kernels.MultiTask.Coregionalize import Coregionalize


class ICM(VanillaKernel):
    def __init__(self, input_dim, num_outputs, kernel, W_rank=1):
        self.num_outputs = num_outputs
        self.kernel = kernel
        self.icm = Coregionalize(num_outputs, rank=W_rank)
        self.input_dim = input_dim

    def function(self, X, params):
        """
        X is of the format [[data, index]]
        """
        kx = self.kernel.function(X[:, :self.input_dim], params)
        kx *= self.icm.function(X[:, self.input_dim, None], params)
        return kx

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
