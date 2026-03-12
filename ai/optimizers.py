"""
Optimizers - pure Python.
SGD, SGD+Momentum, Adam
"""
import math


class SGD:
    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = {}

    def step(self, params):
        """params: list of (param_matrix, grad_matrix, name)"""
        for param, grad, name in params:
            if name not in self._velocity:
                from .matrix import Matrix
                self._velocity[name] = Matrix.zeros(param.rows, param.cols)

            v = self._velocity[name]
            for i in range(param.rows):
                for j in range(param.cols):
                    g = grad.data[i][j]
                    if self.weight_decay > 0:
                        g += self.weight_decay * param.data[i][j]
                    v.data[i][j] = self.momentum * v.data[i][j] - self.lr * g
                    param.data[i][j] += v.data[i][j]


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = {}
        self._v = {}
        self._t = {}

    def step(self, params):
        for param, grad, name in params:
            if name not in self._m:
                from .matrix import Matrix
                self._m[name] = Matrix.zeros(param.rows, param.cols)
                self._v[name] = Matrix.zeros(param.rows, param.cols)
                self._t[name] = 0

            self._t[name] += 1
            t = self._t[name]
            m = self._m[name]
            v = self._v[name]
            b1, b2 = self.beta1, self.beta2
            lr_t = self.lr * math.sqrt(1 - b2 ** t) / (1 - b1 ** t)

            for i in range(param.rows):
                for j in range(param.cols):
                    g = grad.data[i][j]
                    if self.weight_decay > 0:
                        g += self.weight_decay * param.data[i][j]
                    m.data[i][j] = b1 * m.data[i][j] + (1 - b1) * g
                    v.data[i][j] = b2 * v.data[i][j] + (1 - b2) * g * g
                    param.data[i][j] -= lr_t * m.data[i][j] / (math.sqrt(v.data[i][j]) + self.eps)


OPTIMIZERS = {
    "sgd":  SGD,
    "adam": Adam,
}
