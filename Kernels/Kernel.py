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

    def __add__(self, other):
        return Sum([self, other])

    def set_parameters(self, params):
        pass

    def change_id(self, new_id):
        pass


class Embedding(VanillaKernel):
    def __init__(self, embedding, kernel):
        self.embedding = embedding
        self.kernel = kernel

    @partial(jit, static_argnums=(0,))
    def function(self, X, params):
        r = self.kernel.function(self.embedding.function(X, params), params)
        return r

    @property
    def parameters(self):
        p = {}
        p.update(self.kernel.parameters)
        p.update(self.embedding.parameters)

        return p

    def set_parameters(self, params):
        self.kernel.set_parameters(params)
        self.embedding.set_parameters(params)

    def cov(self, X, X2):
        return self.kernel.cov(self.embedding.cov(X, X2))


class Sum(VanillaKernel):
    def __init__(self, kernels):
        _newkerns = []
        k = 0
        for kern in kernels:
            if isinstance(kern, Sum):
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
        r = sum(p.function(X, params) for p in self.parts)
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

    def cov(self, X, X2):
        return sum(p.cov(X, X2) for p in self.parts)


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

    def cov(self, X, X2):
        return reduce(jnp.multiply, (p.cov(X, X2) for p in self.parts))
