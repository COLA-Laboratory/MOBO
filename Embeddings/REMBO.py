"""
Note: All embeddings assume bounds of [-1, 1]

Original: Bayesian Optimization in High Dimensions via Random Embeddings - 2013
PSI: A warped kernel improving robustness in Bayesian optimization
     via random embeddings - 2015
LS: On the choice of the low-dimensional domain for global optimization via random embeddings - 2018
RS: Re-Examining Linear Embeddings for High-Dimensional Bayesian Optimization
"""

from Embeddings.Matricies import SimpleGaussian, NormalizedGaussian
import numpy as np
from cvxopt import matrix, solvers


class Random:
    def __init__(self, effective_dim, original_dim, normal=False):
        self.effective_dim = effective_dim
        self.original_dim = original_dim
        if normal:
            self.projection = NormalizedGaussian(effective_dim, original_dim)

        else:
            self.projection = SimpleGaussian(effective_dim, original_dim)

    def project(self, low_dim_vector):
        pass

    def evaluate(self, low_dim_vector):
        pass


class Original(Random):
    def __init__(self, effective_dim, original_dim, normal=False):
        super(Original, self).__init__(effective_dim, original_dim, normal)

    def project(self, low_dim_vector):
        return np.matmul(low_dim_vector, self.projection.A)

    def evaluate(self, low_dim_vector):
        return np.clip(self.project(low_dim_vector), -1, 1)


class PSI(Random):
    def __init__(self, effective_dim, original_dim, normal=False):
        super(PSI, self).__init__(effective_dim, original_dim, normal)

        A = self.projection.A
        org_bp_matrix = np.matmul(np.matmul(A, np.linalg.inv(np.matmul(A.T, A))), A)
        self.bp_matrix = np.transpose(org_bp_matrix)

    def project(self, low_dim_vector):
        return np.matmul(low_dim_vector, self.projection.A)

    def evaluate(self, low_dim_vector):
        x = self.project(low_dim_vector)
        invalid_idxs = np.where((x < -1) | (x > 1))
        if len(invalid_idxs) == 0:
            return x

        else:
            d = self.original_dim
            z = np.matmul(x, self.bp_matrix)
            z_prime = z / max(np.absolute(z))
            return z_prime + np.linalg.norm(x - z_prime, d) * (z_prime / np.linalg.norm(z_prime, d))


class LS(Random):
    def __init__(self, effective_dim, original_dim, normal=False):
        super(LS, self).__init__(effective_dim, original_dim, normal)

        A = self.projection.A
        org_bp_matrix = np.matmul(np.matmul(A, np.linalg.inv(np.matmul(A.T, A))), A)
        self.bp_matrix = np.transpose(org_bp_matrix)

    def ls(self, y):
        """
        This method projects solutions into higher dimensional space to a point x that minimizes
        ||x-B.Ty|| s.t Bx = y
        """

        map = np.dot(self.projection.A.T, y)
        A = matrix(self.projection.A)
        b = matrix(y)
        P = matrix(2 * np.eye(self.original_dim))
        q = matrix(-2 * map)

        sol = solvers.qp(P, q, A=A, b=b)

        res = np.reshape(np.array(sol['x']), (self.original_dim,))
        return res

    def project(self, low_dim_vector):
        return np.matmul(low_dim_vector, self.projection.A)

    def evaluate(self, low_dim_vector):
        x = self.ls(low_dim_vector)
        invalid_idxs = np.where((x < -1) | (x > 1))
        if len(invalid_idxs) == 0:
            return x

        else:
            d = len(x)
            z = np.matmul(x, self.bp_matrix)
            z_prime = z / max(np.absolute(z))
            return z_prime + np.linalg.norm(x - z_prime, d) * (z_prime / np.linalg.norm(z_prime, d))


class RS(Random):
    def __init__(self, effective_dim, original_dim):
        super(RS, self).__init__(effective_dim, original_dim, True)

    def project(self, low_dim_vector):
        return np.matmul(low_dim_vector, self.projection.A)

    def evaluate(self, low_dim_vector):
        """
        We first project up to the ambient space and then project down to the true subspace.
        This method projects up to the original space, carries out the re-formatting and projects
        back into the subspace.
        """
        return np.matmul(self.projection.Ainv, np.clip(self.project(low_dim_vector), -1, 1))