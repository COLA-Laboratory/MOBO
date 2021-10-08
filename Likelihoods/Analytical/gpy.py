"""
This likelihood implementation is based on the GPy packaage implementation.
"""
from util.linalg import diag_add, pdinv, dpotrs
import numpy as np
from util.linalg import dtrtrs, tdot
log_2_pi = np.log(2. * np.pi)

_log_lim_val = np.log(np.finfo(np.float64).max)
_exp_lim_val = np.finfo(np.float64).max
_lim_val = 36.0
epsilon = np.finfo(np.float64).resolution


class Likelihood:
    def __init__(self, model):
        self.model = model
        self.model.likelihood = self

        self.alpha = None
        self.LW = None

    def gradfactor(self, f, df):
        return df * np.where(f > _lim_val, 1., - np.expm1(-f))

    def fi(self, x):
        return np.where(x > _lim_val, x, np.log1p(np.exp(np.clip(x, -_log_lim_val, _lim_val))))  # + epsilon
        # raises overflow warning: return np.where(x>_lim_val, x, np.log(1. + np.exp(x)))

    def finv(self, f):
        return np.where(f > _lim_val, f, np.log(np.expm1(f)))

    def evaluate(self):
        K = self.model.kernel.K(self.model.X)
        Ky = K.copy()
        diag_add(Ky, self.model.variance+1e-8)
        _, LW, _, _ = pdinv(Ky)
        self.LW = LW
        self.alpha, _ = dpotrs(LW, self.model.y, lower=1)

    def exact_inference_gradients(self, dL_dKdiag):
        return dL_dKdiag.sum()

    def log_likelihood_and_grad(self, params):
        self.model.kernel.set_parameters(params)
        K = self.model.kernel.K(self.model.X)
        Ky = K.copy()
        diag_add(Ky, params["noise"] + 1e-8)
        Wi, LW, LWi, W_logdet = pdinv(Ky)

        alpha, _ = dpotrs(LW, self.model.y, lower=1)

        log_marginal = 0.5 * (
                    -self.model.y.size * log_2_pi - self.model.y.shape[1] * W_logdet - np.sum(alpha * self.model.y))

        dL_dK = 0.5 * (tdot(alpha) - self.model.y.shape[1] * Wi)
        dL_dthetaL = self.exact_inference_gradients(np.diag(dL_dK))

        gradients = self.model.kernel.gradients(dL_dK, self.model.X)
        gradients["noise"] = np.asarray([-1. * dL_dthetaL])

        return -log_marginal, gradients

    def objective_and_grad(self, params):
        for pi in params:
            params[pi] = self.fi(params[pi])

        self.model.kernel.set_parameters(params)
        K = self.model.kernel.K(self.model.X)
        Ky = K.copy()
        diag_add(Ky, params["noise"]+1e-8)
        Wi, LW, LWi, W_logdet = pdinv(Ky)

        alpha, _ = dpotrs(LW, self.model.y, lower=1)

        log_marginal = 0.5*(-self.model.y.size * log_2_pi - self.model.y.shape[1] * W_logdet - np.sum(alpha * self.model.y))

        dL_dK = 0.5 * (tdot(alpha) - self.model.y.shape[1] * Wi)
        dL_dthetaL = self.exact_inference_gradients(np.diag(dL_dK))

        gradients = self.model.kernel.gradients(dL_dK, self.model.X)
        gradients["noise"] = [-1. * dL_dthetaL]

        for gi in gradients:
            gradients[gi] = self.gradfactor(params[gi], gradients[gi])

        return -log_marginal, gradients

    def predict(self, Xnew):
        Kx = self.model.kernel.K(self.model.X, Xnew)
        mu = np.dot(Kx.T, self.alpha)

        Kxx = self.model.kernel.K(Xnew)
        tmp = dtrtrs(self.LW, Kx)[0]
        var = Kxx - tdot(tmp.T)
        return mu, var
