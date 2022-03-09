from copy import deepcopy
from functools import reduce
import jax.numpy as jnp
from functools import partial
from jax import jit


class VanillaKernel:

    def function(self, X, params):
        pass

    def cov(self, X, X2):
        pass

    def __copy__(self):
        return deepcopy(self)

    def __mul__(self, other):
        return Product([self, other])

    def set_parameters(self, params):
        pass

    def change_id(self, new_id):
        pass


class Product(VanillaKernel):
    def __init__(self, kernels):
        _newkerns = []
        k = 0
        for kern in kernels:
            if isinstance(kern, Product):
                for part in kern.parts:
                    part.change_id(k)
                    _newkerns.append(part)
                    k += 1

            else:
                kern.change_id(k)
                _newkerns.append(kern)
                k += 1

        self.parts = _newkerns

    @partial(jit, static_argnums=(0,))
    def function(self, X, params):
        r = reduce(jnp.multiply, (p.function(X, params) for p in self.parts))
        return r

    @property
    def parameters(self):
        p = {}
        for kern in self.parts:
            p.update(kern.parameters)

        return p

    def set_parameters(self, params):
        for kern in self.parts:
            kern.set_parameters(params)
