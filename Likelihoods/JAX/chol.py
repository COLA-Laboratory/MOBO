"""
This likelihood implementation uses cholesky decomposition
"""
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular, cholesky
from jax import jit, value_and_grad, config
import numpy as np
from functools import partial
from jax.lax import stop_gradient
config.update("jax_enable_x64", True)

log_2_pi = jnp.log(2. * jnp.pi)

_log_lim_val = np.log(np.finfo(np.float64).max)
_exp_lim_val = np.finfo(np.float64).max
_lim_val = 36.0
epsilon = np.finfo(np.float64).resolution


class Likelihood:
    def __init__(self, model):
        self.model = model
        self.N = self.model.X.shape[0]

        self.model.likelihood = self
        self.value = None

        self.L = None
        self.alpha = None

    def gradfactor(self, f, df):
        return df * jnp.where(f > _lim_val, 1., - jnp.expm1(-f))

    def fi(self, x):
        return jnp.where(x > _lim_val, x, jnp.log1p(jnp.exp(jnp.clip(x, -_log_lim_val, _lim_val))))  # + epsilon
        # raises overflow warning: return np.where(x>_lim_val, x, np.log(1. + np.exp(x)))

    def finv(self, f):
        return np.where(f > _lim_val, f, np.log(np.expm1(f)))

    def evaluate(self):
        K = self.model.kernel.function(self.model.X,
                                       self.model.parameters)\
            + jnp.eye(self.N) * (self.model.parameters["noise"] + 1e-8)

        self.L = cholesky(K, lower=True)
        self.alpha = solve_triangular(self.L.T, solve_triangular(self.L, self.model.y, lower=True))

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, params):
        self.model.set_parameters(params)
        kx = self.model.kernel.function(self.model.X, params) + jnp.eye(self.N) * (params["noise"] + 1e-8)
        L = cholesky(kx, lower=True)

        alpha = solve_triangular(L.T, solve_triangular(L, self.model.y, lower=True))
        W_logdet = 2. * jnp.sum(jnp.log(jnp.diag(L)))
        log_marginal = 0.5 * (
                -self.model.y.size * log_2_pi - self.model.y.shape[1] * W_logdet - jnp.sum(alpha * self.model.y))

        return log_marginal

    def log_likelihood_and_grad(self, params):
        return value_and_grad(self.log_likelihood, argnums=0)(params)

    def objective(self, params):
        return -1. * self.log_likelihood(params)

    def objective_and_grad(self, params):

        for pi in params:
            params[pi] = self.fi(params[pi])

        value, gradients = value_and_grad(self.objective, argnums=0)(params)

        for gi in gradients:
            gradients[gi] = jnp.nan_to_num(self.gradfactor(params[gi], gradients[gi]))

        if jnp.isnan(value): value = np.inf
        return value, gradients

    @partial(jit, static_argnums=(0,))
    def predict(self, Xnew):
        Kx = self.model.kernel.cov(self.model.X, Xnew)
        mu = jnp.dot(Kx.T, self.alpha)

        Kxx = self.model.kernel.cov(Xnew, Xnew)

        tmp = solve_triangular(self.L, Kx, lower=True)

        var = Kxx - jnp.dot(tmp.T, tmp) + jnp.eye(Xnew.shape[0]) * self.model.variance

        return mu, var

