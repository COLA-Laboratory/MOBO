import numpy as np
import jax
import jax.numpy as jnp


def build_XY(input_list, output_list=None, index=None):
    num_outputs = len(input_list)
    if output_list is not None:
        assert num_outputs == len(output_list)
        Y = np.vstack(output_list)
    else:
        Y = None

    if index is not None:
        assert len(index) == num_outputs
        I = np.hstack([np.repeat(j, _x.shape[0]) for _x, j in zip(input_list, index)])
    else:
        I = np.hstack([np.repeat(j, _x.shape[0]) for _x, j in zip(input_list, range(num_outputs))])

    X = np.vstack(input_list)
    X = np.hstack([X, I[:, None]])

    return X, Y, I[:, None]  # slices


class GPCoregionalizedRegression:

    def __init__(self, input_dim, X_list, Y_list, kernel, noise=1.):
        self.kernel = kernel
        self.variance = np.ones((1,)) * noise * 1.
        self.X, self.y, self.I = build_XY(X_list, Y_list)

        self.likelihood = None
        self.input_dim = input_dim

    @property
    def parameters(self):
        p = self.kernel.parameters
        p["noise"] = self.variance
        return p

    def set_parameters(self, params):
        self.variance = params["noise"]
        self.kernel.set_parameters(params)

    @property
    def log_likelihood(self):
        return self.likelihood.value

    def predict(self, Xnew_list, index=None, full_cov=False):
        """
        Predicts inputs Xnew_list, if you want predictions of all tasks leave index=False,
        else set index to the integer value of the task, i.e. task 0 = 1.
        """
        index = np.ones(len(Xnew_list)) * index
        Xnew, _, _ = build_XY(Xnew_list, index=index)
        #print(Xnew)
        with jax.checking_leaks():
            if full_cov:
                mu, var = self.likelihood.predict(Xnew)
                return mu, var
            else:
                mu, var = self.likelihood.predict(Xnew)
                return mu, jnp.diag(var)

