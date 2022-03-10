import numpy as np
import matplotlib.pyplot as plt

from functools import reduce

from Kernels.JAX.Matrixy.rbf import RBF
from Models.JAX.GPregression import GPregression
from ModelOptimizers.lbfgsb import lbfgsb
from Likelihoods.JAX.chol import Likelihood

from Kernels.MultiTask.ICM import ICM
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

    input_kern = RBF(dataset_shape=(len(data1)+len(data2), data1.shape[1]))
    kern = ICM(input_dim=data1.shape[1], num_outputs=2, kernel=input_kern, W_rank=2)
    gp = GPCoregionalizedRegression(5, [data1, data2], [y1, y2], kernel=kern)
    likelihood = Likelihood(gp)
    optimizer = lbfgsb(gp)
    optimizer.opt()
    print(gp.log_likelihood)
    likelihood.evaluate()

    test_data1 = np.arange(-5, 5, 0.1).reshape((-1, 1))

    y1_test = np.array([sphere1(di) for di in test_data1]).reshape((-1, 1))
    y2_test = np.array([sphere2(di) for di in test_data1]).reshape((-1, 1))

    mu, _ = gp.predict(Xnew_list=[test_data1], index=0)

    plt.plot(test_data1, mu)
    plt.plot(test_data1, y1_test)
    plt.legend(["mu", "true"])
    plt.plot(data1, y1, 'o')
    plt.show()

    mu, _ = gp.predict(Xnew_list=[test_data1], index=1)

    plt.plot(test_data1, mu)
    plt.plot(test_data1, y2_test)
    plt.legend(["mu", "true"])
    plt.plot(data2, y2, 'o')
    plt.show()

    k = GPy.kern.RBF(input_dim=data1.shape[1])
    icm = GPy.util.multioutput.ICM(input_dim=data1.shape[1], num_outputs=2, kernel=k, W_rank=2)
    m = GPy.models.GPCoregionalizedRegression([data1, data2], [y1, y2], kernel=icm)

    m.optimize()
    print(m.log_likelihood())

    test_data1 = np.arange(-5, 5, 0.1).reshape((-1, 1))

    y1_test = np.array([sphere1(di) for di in test_data1]).reshape((-1, 1))
    y2_test = np.array([sphere2(di) for di in test_data1]).reshape((-1, 1))

    test = np.hstack([test_data1, 0 * np.ones_like(test_data1)])

    n = {'output_index': np.asarray([[0] for _ in range(len(test_data1))]).astype(int)}
    mu, var = m.predict(test, Y_metadata=n)

    plt.plot(test_data1, mu)
    plt.plot(test_data1, y1_test)
    plt.legend(["mu", "true"])
    plt.plot(data1, y1, 'o')
    plt.show()

    test = np.hstack([test_data1, 1 * np.ones_like(test_data1)])

    n = {'output_index': np.asarray([[1] for _ in range(len(test_data1))]).astype(int)}
    mu, var = m.predict(test, Y_metadata=n)

    plt.plot(test_data1, mu)
    plt.plot(test_data1, y2_test)
    plt.legend(["mu", "true"])
    plt.plot(data2, y2, 'o')
    plt.show()
