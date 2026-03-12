"""
Activation functions - pure Python, no external libs.
Each returns (output_matrix, cache) where cache is used for backprop.
"""
import math
from .matrix import Matrix


def relu(x: Matrix):
    out = x.apply(lambda v: max(0.0, v))
    return out, x  # cache = input

def relu_backward(dout: Matrix, cache: Matrix):
    dx = dout.hadamard(cache.apply(lambda v: 1.0 if v > 0 else 0.0))
    return dx


def sigmoid(x: Matrix):
    def _sig(v):
        if v >= 0:
            return 1.0 / (1.0 + math.exp(-v))
        e = math.exp(v)
        return e / (1.0 + e)
    out = x.apply(_sig)
    return out, out  # cache = sigmoid output

def sigmoid_backward(dout: Matrix, cache: Matrix):
    # d/dx sigmoid = sigmoid*(1-sigmoid)
    s = cache
    one_minus = s.apply(lambda v: 1.0 - v)
    dx = dout.hadamard(s).hadamard(one_minus)
    return dx


def tanh(x: Matrix):
    out = x.apply(math.tanh)
    return out, out

def tanh_backward(dout: Matrix, cache: Matrix):
    dx = dout.hadamard(cache.apply(lambda v: 1.0 - v * v))
    return dx


def softmax(x: Matrix):
    """Row-wise softmax."""
    result = Matrix(x.rows, x.cols)
    for i in range(x.rows):
        row = x.data[i]
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        s = sum(exps)
        result.data[i] = [e / s for e in exps]
    return result, result

def softmax_backward(dout: Matrix, cache: Matrix):
    """Simplified: combined with cross-entropy -> dout is already correct gradient."""
    return dout


def linear(x: Matrix):
    return x, x

def linear_backward(dout: Matrix, cache: Matrix):
    return dout


ACTIVATIONS = {
    "relu":    (relu,    relu_backward),
    "sigmoid": (sigmoid, sigmoid_backward),
    "tanh":    (tanh,    tanh_backward),
    "softmax": (softmax, softmax_backward),
    "linear":  (linear,  linear_backward),
}
