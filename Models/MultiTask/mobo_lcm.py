import numpy as np
import matplotlib.pyplot as plt

from functools import reduce

from Kernels.JAX.Matrixy.rbf import RBF
from Models.JAX.GPregression import GPregression
from ModelOptimizers.lbfgsb import lbfgsb
from Likelihoods.JAX.chol import Likelihood

from Kernels.MultiTask.CMS import LCM
from Models.MultiTask.GPCoregionalizedRegression import GPCoregionalizedRegression

import GPy


def sphere1(x):
    return sum(xi**2 for xi in x)


def sphere2(x):
    return sum(xi for xi in x)


if __name__ == "__main__":

    np.random.seed(0)

    data1 = np.random.uniform(-5, 5, size=(8, 1))
    data2 = np.random.uniform(-5, 5, size=(3, 1))

    y1 = np.array([sphere1(di) for di in data1]).reshape((-1, 1))
    y2 = np.array([sphere2(di) for di in data2]).reshape((-1, 1))

    input_kern1 = RBF(dataset_shape=(len(data1)+len(data2), data1.shape[1]), ARD=True)
    input_kern2 = RBF(dataset_shape=(len(data1)+len(data2), data1.shape[1]), ARD=True)

    #kern = ICM(input_dim=data1.shape[1], num_outputs=2, kernel=input_kern, W_rank=2)
    kern = LCM(input_dim=data1.shape[1], num_outputs=2, kernels_list=[input_kern1, input_kern2], W_ranks=[1, 1])
    gp = GPCoregionalizedRegression(5, [data1, data2], [y1, y2], kernel=kern)
    likelihood = Likelihood(gp)
    optimizer = lbfgsb(gp)
    optimizer.opt()
    print(gp.log_likelihood)
    print(gp.parameters)
    likelihood.evaluate()

    test_data1 = np.arange(-5, 5, 0.1).reshape((-1, 1))

    y1_test = np.array([sphere1(di) for di in test_data1]).reshape((-1, 1))
    y2_test = np.array([sphere2(di) for di in test_data1]).reshape((-1, 1))

    mu, _ = gp.predict(Xnew_list=[test_data1], index=0)

    plt.plot(test_data1, mu)
    plt.plot(test_data1, y1_test)
    plt.legend(["mu", "true"])
    plt.plot(data1, y1, 'o')
    plt.title("MOBO")
    plt.show()

    mu, _ = gp.predict(Xnew_list=[test_data1], index=1)

    plt.plot(test_data1, mu)
    plt.plot(test_data1, y2_test)
    plt.legend(["mu", "true"])
    plt.plot(data2, y2, 'o')
    plt.title("MOBO")
    plt.show()
