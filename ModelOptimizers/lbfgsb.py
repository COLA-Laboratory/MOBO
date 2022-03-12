from scipy import optimize
import numpy as np


class lbfgsb:
    def __init__(self, model, maxfun=1000):
        self.lengths = [len(model.parameters[pi]) for pi in model.parameters]
        self.x0 = model.likelihood.finv(np.concatenate([model.parameters[pi] for pi in model.parameters]))
        self.param_words = list(model.parameters.keys())
        self.model = model
        self.function = model.likelihood.objective_and_grad
        self.maxfun = maxfun

    def to_dict(self, x):
        c_idx = 0
        p = {}
        for i in range(len(self.lengths)):
            p[self.param_words[i]] = x[c_idx: c_idx+self.lengths[i]]
            c_idx+=self.lengths[i]

        return p

    def value_and_gradient(self, x):
        p = self.to_dict(x)
        value, grad = self.function(p)
        #print(grad)
        return value, np.concatenate([grad[pi] for pi in self.param_words])

    def opt(self):
        res = optimize.fmin_l_bfgs_b(self.value_and_gradient, x0=self.x0, maxfun=self.maxfun, maxiter=self.maxfun)
        self.model.set_parameters(self.to_dict(self.model.likelihood.fi(res[0])))
        self.model.likelihood.evaluate()
        self.model.likelihood.value = -res[1]
