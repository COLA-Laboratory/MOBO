"""
Projection Matricies
"""

import numpy as np


class SimpleGaussian:
    def __init__(self, effective_dim, main_dim):
        self.A = np.random.normal(0, 1, [effective_dim, main_dim])
        self.Ainv = np.linalg.pinv(self.A)

    def evaluate(self):
        return self.A

    def inverse(self):
        return np.linalg.pinv(self.A)


class NormalizedGaussian:
    def __init__(self, effective_dim, main_dim):
        A = np.random.normal(0, 1, [effective_dim, main_dim])

        effective_dim = len(A)
        main_dim = len(A[0])
        new_matrix = np.zeros([effective_dim, main_dim])
        for i in range(effective_dim):
            norm = np.linalg.norm(A[i])
            new_matrix[i] = A[i] / norm

        self.A = new_matrix
        self.Ainv = np.linalg.pinv(self.A)
