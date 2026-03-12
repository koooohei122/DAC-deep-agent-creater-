"""
Loss functions - pure Python.
"""
import math
from .matrix import Matrix


def mse(pred: Matrix, target: Matrix):
    """Mean Squared Error."""
    diff = pred.sub(target)
    sq = diff.hadamard(diff)
    loss = sq.sum() / (pred.rows * pred.cols)
    grad = diff.scale(2.0 / (pred.rows * pred.cols))
    return loss, grad


def mae(pred: Matrix, target: Matrix):
    """Mean Absolute Error."""
    n = pred.rows * pred.cols
    loss = 0.0
    grad = Matrix(pred.rows, pred.cols)
    for i in range(pred.rows):
        for j in range(pred.cols):
            d = pred.data[i][j] - target.data[i][j]
            loss += abs(d)
            grad.data[i][j] = (1.0 if d > 0 else -1.0) / n
    return loss / n, grad


def binary_cross_entropy(pred: Matrix, target: Matrix):
    """Binary cross-entropy (sigmoid output assumed)."""
    eps = 1e-9
    loss = 0.0
    grad = Matrix(pred.rows, pred.cols)
    n = pred.rows * pred.cols
    for i in range(pred.rows):
        for j in range(pred.cols):
            p = max(eps, min(1 - eps, pred.data[i][j]))
            t = target.data[i][j]
            loss += -(t * math.log(p) + (1 - t) * math.log(1 - p))
            grad.data[i][j] = (-t / p + (1 - t) / (1 - p)) / n
    return loss / n, grad


def cross_entropy(pred: Matrix, target: Matrix):
    """
    Categorical cross-entropy.
    pred: softmax probabilities (N, C)
    target: one-hot labels (N, C)
    """
    eps = 1e-9
    loss = 0.0
    grad = Matrix(pred.rows, pred.cols)
    for i in range(pred.rows):
        for j in range(pred.cols):
            p = max(eps, pred.data[i][j])
            t = target.data[i][j]
            loss -= t * math.log(p)
            # Combined softmax+cross-entropy gradient = pred - target
            grad.data[i][j] = (pred.data[i][j] - t) / pred.rows
    return loss / pred.rows, grad


LOSSES = {
    "mse":    mse,
    "mae":    mae,
    "bce":    binary_cross_entropy,
    "cross_entropy": cross_entropy,
}
