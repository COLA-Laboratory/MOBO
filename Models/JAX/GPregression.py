import numpy as np
from jax.lax import stop_gradient
import jax.numpy as jnp
import jax


class GPregression:
    def __init__(self, kernel, X, y, noise=1.):
        self.kernel = kernel
        self.variance = np.ones((1,)) * noise * 1.

        self.X = X
        self.y = y

        self.likelihood = None

    @property
    def parameters(self):
        p = self.kernel.parameters
        p["noise"] = self.variance
        return p

    @property
    def log_likelihood(self):
        return self.likelihood.value

    @property
    def gradients(self):
        return self.likelihood.log_likelihood_and_grad(self.parameters)[1]

    @property
    def objective_and_grad(self):
        return self.likelihood.objective_and_grad(self.parameters)

    def set_parameters(self, params):
        self.variance = params["noise"]
        self.kernel.set_parameters(params)

    def predict(self, Xnew, full_cov=False):
        with jax.checking_leaks():
            if full_cov:
                mu, var = self.likelihood.predict(Xnew)
                return mu, var
            else:
                mu, var = self.likelihood.predict(Xnew)
                return mu, jnp.diag(var)
