"""
A Framework for Bayesian Optimization in Embedded Subspaces - 2019
"""
import numpy as np


class CountSketch:
    def __init__(self, effective_dim, main_dim):
        self.h = np.random.choice(range(effective_dim), main_dim)
        self.sign = np.random.choice([-1, 1], main_dim)
        self.effective_dim = effective_dim
        self.main_dim = main_dim

    def evaluate(self, low_dim_vectors):
        n = len(low_dim_vectors)
        high_dim_vectors = np.empty(n, self.main_dim)
        for i in range(self.main_dim):
            high_dim_vectors[:, i] = self.sign[i] * low_dim_vectors[:, self.h[i]]

        for i in range(n):
            high_dim_vectors[i] = np.clip(high_dim_vectors[i], -1, 1)

        return high_dim_vectors