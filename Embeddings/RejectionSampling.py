import numpy as np


class EmbeddingRejectionSampling:

    def __init__(self, embedding: np.array, bound: list):
        """
        Rejection sampling within the embedding space subject to bounds.
        Recursively reduced the bounds within the embedding until all sample
        points satisfy the bound constraints
        """

        self.embedding = embedding
        self.bound = bound

    def sample(self, n=10):
        """
        Samples the embedding space until all projections satisfy the original
        space bounds.
        """

        finished = False
        b = 1e+6
        dimension = self.embedding.shape[0]
        samples = np.random.uniform(0, 1, (int(1e+7), dimension))
        while not finished:
            X_b = 2 * b * samples - b

            # Project up to the original space
            X = np.matmul(X_b, self.embedding)

            idx = np.where((X >= self.bound[0]).all(axis=1) &
                           (X <= self.bound[1]).all(axis=1))[0]

            if len(idx) >= n:
                finished = True

            else:
                b = b / 2.0  # Constrict the space

        return X_b[idx][:n]
