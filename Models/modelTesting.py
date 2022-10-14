from GPy.kern import RBF as RBFg
from GPy.models import GPRegression
import numpy as np
from Kernels.Analytical.rbf import RBF
from Models.Analytical.GPregression import GPregression as moboGP
from Likelihoods.Analytical.gpy import Likelihood
from ModelOptimizers.lbfgsb import lbfgsb


def sphere(x):
    return sum(xi**2 for xi in x)


def rosebbrock(x):
    sum1 = 0
    for i in range(0, len(x)-1):
        sum1 += (100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2)
    return sum1


if __name__ == "__main__":
    np.random.seed(1)
    dim = 10
    f = sphere
    X = np.random.uniform(-5, 5, (10, dim))
    y = np.array([f(xi) for xi in X]).reshape(-1, 1)

    gpy_kern = RBFg(input_dim=dim, ARD=False)
    gpy_model = GPRegression(X, y, kernel=gpy_kern)
    gpy_model.optimize()

    mobo_kern = RBF(input_dim=dim, ARD=False)
    mobo_model = moboGP(mobo_kern, X, y)
    mobo_likelihood = Likelihood(mobo_model)
    mobo_likelihood.evaluate()

    optimizer = lbfgsb(mobo_model)
    optimizer.opt()

    print("gpy:", gpy_model.log_likelihood())
    print("mobo:", mobo_model.log_likelihood)

    Xtest = np.random.uniform(-5, 5, (1000, dim))
    ytest = np.array([f(xi) for xi in Xtest]).reshape(-1, 1)

    gpy_prediction, gpy_cov = gpy_model.predict(Xtest)
    mobo_prediction, mobo_cov = mobo_model.predict(Xtest)

    print("gpy:", np.mean(abs(ytest - gpy_prediction)))
    print("mobo:", np.mean(abs(ytest - mobo_prediction)))



