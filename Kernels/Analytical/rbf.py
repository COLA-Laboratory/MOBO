import numpy as np
from util.linalg import tdot, diag_view


class RBF:
    def __init__(self, input_dim, lengthscale=1., variance=1., ARD=False):
        self.input_dim = input_dim
        if ARD:
            self.parameters = {"variance": np.ones((1,)) * variance * 1.,
                               "lengthscale": np.ones((input_dim,)) * lengthscale * 1.}

        else:
            self.parameters = {"variance": np.ones((1,)) * variance * 1.,
                               "lengthscale": np.ones((1,)) * lengthscale * 1.}

        self.ARD = ARD

    def set_parameters(self, params):
        self.parameters["lengthscale"] = params["lengthscale"]
        self.parameters["variance"] = params["variance"]

    def K(self, X, X2=None):
        r = self._scaled_dist(X, X2)
        return self.K_of_r(r)

    def K_of_r(self, r):
        return self.parameters["variance"] * np.exp(-0.5 * r ** 2)

    def _scaled_dist(self, X, X2=None):
        if self.ARD:
            if X2 is not None:
                X2 = X2 / self.parameters["lengthscale"]
            return self._unscaled_dist(X / self.parameters["lengthscale"])

        else:
            return self._unscaled_dist(X, X2) / self.parameters["lengthscale"]

    def dK_dr(self, r):
        return -r * self.K_of_r(r)

    def dK_dr_via_X(self, X, X2):
        return self.dK_dr(self._scaled_dist(X, X2))

    def _unscaled_dist(self, X, X2=None):
        if X2 is None:
            Xsq = np.sum(np.square(X), 1)
            r2 = -2. * tdot(X) + (Xsq[:, None] + Xsq[None, :])
            diag_view(r2)[:, ] = 0.  # force diagnoal to be zero: sometime numerically a little negative
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)
        else:
            X1sq = np.sum(np.square(X), 1)
            X2sq = np.sum(np.square(X2), 1)
            r2 = -2. * np.dot(X, X2.T) + (X1sq[:, None] + X2sq[None, :])
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)

    def _inv_dist(self, X, X2=None):
        """
        Compute the elementwise inverse of the distance matrix, expecpt on the
        diagonal, where we return zero (the distance on the diagonal is zero).
        This term appears in derviatives.
        """
        dist = self._scaled_dist(X, X2).copy()
        return 1. / np.where(dist != 0., dist, np.inf)

    def _lengthscale_grads_pure(self, tmp, X, X2):
        return -np.array([np.sum(tmp * np.square(X[:,q:q+1] - X2[:,q:q+1].T))
                          for q in range(self.input_dim)])/self.parameters["lengthscale"]**3

    def gradients(self, dL_dK, X, X2=None):
        dK_dv = -np.sum(self.K(X, X2) * dL_dK) / self.parameters["variance"]

        dL_dr = self.dK_dr_via_X(X, X2) * dL_dK

        if self.ARD:
            tmp = dL_dr * self._inv_dist(X, X2)
            if X2 is None: X2 = X
            dK_dl = self._lengthscale_grads_pure(tmp, X, X2)

        else:
            r = self._scaled_dist(X, X2)
            dK_dl = -np.sum(-dL_dr * r) / self.parameters["lengthscale"]

        return {"variance": dK_dv, "lengthscale": dK_dl}
