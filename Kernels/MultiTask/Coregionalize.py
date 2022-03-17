from Kernels.Kernel import VanillaKernel
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import jit


class Dynamic(VanillaKernel):
    """
    Covariance function for intrinsic/linear coregionalization models.
    """
    def __init__(self, output_dim, rank=1, time=1.):
        e = []
        for i in range(output_dim):
            e_ = []
            for j in range(output_dim):
                e_.append(jnp.abs((i - j)))

            e.append(e_)

        self.E = jnp.asarray(e)

        self.output_dim = output_dim
        self.rank = rank
        self.id = 0
        if self.rank>output_dim:
            print("Warning: Unusual choice of rank, it should normally be less than the output_dim.")
        W = 0.5 * np.abs(np.random.randn(self.output_dim*self.rank,)) / np.sqrt(self.rank)
        W = jnp.array(W)
        #W = 0.5 * jnp.ones(self.output_dim*self.rank)

        kappa = 0.5 * jnp.ones(self.output_dim)

        self.parameters = {"W"+str(self.id): W,
                           "kappa"+str(self.id): kappa,
                           "time"+str(self.id): jnp.ones((1,)) * time}

    def change_id(self, new_id):
        self.parameters["W"+str(new_id)] = self.parameters.pop("W"+str(self.id))
        self.parameters["kappa"+str(new_id)] = self.parameters.pop("kappa"+str(self.id))
        self.parameters["time"+str(new_id)] = self.parameters.pop("time"+str(self.id))

        self.id = new_id

    @partial(jit, static_argnums=(0,))
    def function(self, X, params):
        """
        X is a list of task indices!
        """
        index = jnp.asarray(X, dtype=jnp.int_)
        W = jnp.reshape(params["W"+str(self.id)], (self.output_dim, self.rank))
        B = jnp.dot(W, W.T)
        B += jnp.diag(self.parameters["kappa"+str(self.id)])

        time = (1. - params["time"+str(self.id)]) ** self.E
        return B[index, index.T] * time[index, index.T]

    @partial(jit, static_argnums=(0,))
    def cov(self, X, X2):
        index = jnp.asarray(X, dtype=jnp.int_)
        index2 = jnp.asarray(X2, dtype=jnp.int_)

        W = jnp.reshape(self.parameters["W" + str(self.id)], (self.output_dim, self.rank))
        B = jnp.dot(W, W.T)
        B += jnp.diag(self.parameters["kappa" + str(self.id)])

        time = (1. - self.parameters["time"+str(self.id)]) ** self.E

        return B[index, index2.T] * time[index, index2.T]

    def set_parameters(self, params):
        self.parameters["W"+str(self.id)] = params["W"+str(self.id)]
        self.parameters["kappa"+str(self.id)] = params["kappa"+str(self.id)]
        self.parameters["time"+str(self.id)] = params["time"+str(self.id)]


class Coregionalize(VanillaKernel):
    """
    Covariance function for intrinsic/linear coregionalization models.
    """
    def __init__(self, output_dim, rank=1):
        self.output_dim = output_dim
        self.rank = rank
        self.id = 0
        if self.rank>output_dim:
            print("Warning: Unusual choice of rank, it should normally be less than the output_dim.")
        W = 0.5 * np.abs(np.random.randn(self.output_dim*self.rank,)) / np.sqrt(self.rank)
        W = jnp.array(W)
        #W = 0.5 * jnp.ones(self.output_dim*self.rank)

        kappa = 0.5 * jnp.ones(self.output_dim)

        self.parameters = {"W"+str(self.id): W,
                           "kappa"+str(self.id): kappa}

    def change_id(self, new_id):
        self.parameters["W"+str(new_id)] = self.parameters.pop("W"+str(self.id))
        self.parameters["kappa"+str(new_id)] = self.parameters.pop("kappa"+str(self.id))

        self.id = new_id

    @partial(jit, static_argnums=(0,))
    def function(self, X, params):
        """
        X is a list of task indices!
        """
        index = jnp.asarray(X, dtype=jnp.int_)
        W = jnp.reshape(params["W"+str(self.id)], (self.output_dim, self.rank))
        B = jnp.dot(W, W.T)
        B += jnp.diag(self.parameters["kappa"+str(self.id)])

        return B[index, index.T]

    @partial(jit, static_argnums=(0,))
    def cov(self, X, X2):
        index = jnp.asarray(X, dtype=jnp.int_)
        index2 = jnp.asarray(X2, dtype=jnp.int_)

        W = jnp.reshape(self.parameters["W" + str(self.id)], (self.output_dim, self.rank))
        B = jnp.dot(W, W.T)
        B += jnp.diag(self.parameters["kappa" + str(self.id)])

        return B[index, index2.T]

    def set_parameters(self, params):
        self.parameters["W"+str(self.id)] = params["W"+str(self.id)]
        self.parameters["kappa"+str(self.id)] = params["kappa"+str(self.id)]
