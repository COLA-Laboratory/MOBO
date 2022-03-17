"""
Example of how to use the current version of the library.
"""

import numpy as np


if __name__ == "__main__":
    a = np.random.uniform(-5, 5, (5, 5))
    print(a[:, [1, 2]])
    print(a)
