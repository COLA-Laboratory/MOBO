"""
Negative Expected Improvement (for minimization, take negative for maximization)
"""
import jax.numpy as jnp
from jax import jit
from jax import value_and_grad
from functools import partial
from jax.scipy.special import erfc
import numpy as np


class AcquisitionEI:
    def __init__(self, jax_model, fmin, jitter=0.001):
        self.model = jax_model
        self.jitter = jitter
        self.fmin = fmin

    def value_and_gradient(self, x):
        x = x.reshape(1, len(x))
        v, g = value_and_grad(self.function, argnums=0)(x)
        #return jnp.nan_to_num(v, nan=0.), np.array([jnp.nan_to_num(gi) for gi in g[0]])
        return v, np.array(g[0])

    @partial(jit, static_argnums=(0,))
    def function(self, x):
        mean, var = self.model.predict(x)

        s = jnp.sqrt(var)
        u = (self.fmin - mean - self.jitter) / s
        phi = jnp.exp(-0.5 * u ** 2) / jnp.sqrt(2 * jnp.pi)
        Phi = 0.5 * erfc(-u / jnp.sqrt(2))

        return -jnp.sum(s * (u * Phi + phi))