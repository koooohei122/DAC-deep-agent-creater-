"""
Multi-layer Perceptron - pure Python.
Supports arbitrary layer sizes, activations, batch training,
forward/backward pass, serialization.
"""
import json
import math
import random
from .matrix import Matrix
from .activations import ACTIVATIONS
from .losses import LOSSES
from .optimizers import OPTIMIZERS


class DenseLayer:
    def __init__(self, in_size, out_size, activation="relu", name=None):
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        self.name = name or f"dense_{in_size}_{out_size}"
        self._init_params()
        self._cache = {}

    def _init_params(self):
        self.W = Matrix.random(self.in_size, self.out_size)
        self.b = Matrix.zeros(1, self.out_size)
        self.dW = Matrix.zeros(self.in_size, self.out_size)
        self.db = Matrix.zeros(1, self.out_size)

    # ------------------------------------------------------------------ #
    def forward(self, x: Matrix) -> Matrix:
        """x: (batch, in_size) -> (batch, out_size)"""
        z = x.mul(self.W).add(self.b)          # (N, out)
        act_fn, _ = ACTIVATIONS[self.activation]
        out, act_cache = act_fn(z)
        self._cache = {"x": x, "z": z, "act_cache": act_cache}
        return out

    def backward(self, dout: Matrix) -> Matrix:
        """dout: gradient w.r.t. layer output -> returns gradient w.r.t. input."""
        _, act_bwd = ACTIVATIONS[self.activation]
        dz = act_bwd(dout, self._cache["act_cache"])

        x = self._cache["x"]
        N = x.rows

        # dW = X^T · dZ  (in, out)
        self.dW = x.T().mul(dz).scale(1.0 / N)
        # db = mean over batch  (1, out)
        self.db = dz.sum_rows().scale(1.0 / N)
        # dX = dZ · W^T  (N, in)
        dx = dz.mul(self.W.T())
        return dx

    def params(self):
        return [
            (self.W, self.dW, self.name + "/W"),
            (self.b, self.db, self.name + "/b"),
        ]

    # ------------------------------------------------------------------ #
    def to_dict(self):
        return {
            "in_size":    self.in_size,
            "out_size":   self.out_size,
            "activation": self.activation,
            "name":       self.name,
            "W":          self.W.to_dict(),
            "b":          self.b.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        layer = cls(d["in_size"], d["out_size"], d["activation"], d["name"])
        layer.W = Matrix.from_dict(d["W"])
        layer.b = Matrix.from_dict(d["b"])
        return layer


# ======================================================================== #

class NeuralNetwork:
    """
    Simple MLP with:
     - arbitrary depth
     - configurable activations per layer
     - mini-batch training
     - loss history tracking for live preview
    """

    def __init__(self, config: dict):
        """
        config = {
            "layers": [
                {"in": 4, "out": 16, "activation": "relu"},
                {"in": 16, "out": 8, "activation": "relu"},
                {"in": 8,  "out": 1, "activation": "sigmoid"},
            ],
            "loss":      "mse",          # mse | bce | cross_entropy
            "optimizer": "adam",         # sgd | adam
            "lr":        0.001,
            "batch_size": 32,
            "epochs":    100,
        }
        """
        self.config = config
        self.layers = []
        for i, lc in enumerate(config["layers"]):
            act = lc.get("activation", "relu")
            layer = DenseLayer(lc["in"], lc["out"], act, f"layer_{i}")
            self.layers.append(layer)

        opt_cls = OPTIMIZERS[config.get("optimizer", "adam")]
        lr = config.get("lr", 0.001)
        if config.get("optimizer", "adam") == "adam":
            self.optimizer = opt_cls(lr=lr)
        else:
            self.optimizer = opt_cls(lr=lr, momentum=config.get("momentum", 0.9))

        self.loss_fn = LOSSES[config.get("loss", "mse")]
        self.history = {"loss": [], "val_loss": [], "accuracy": []}
        self._is_training = False
        self._stop_flag = False

    # ------------------------------------------------------------------ #
    def forward(self, x: Matrix) -> Matrix:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad: Matrix):
        dout = grad
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def _update(self):
        params = []
        for layer in self.layers:
            params.extend(layer.params())
        self.optimizer.step(params)

    # ------------------------------------------------------------------ #
    def _make_batches(self, X, y, batch_size):
        n = X.rows
        indices = list(range(n))
        random.shuffle(indices)
        batches = []
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            xb = Matrix.from_2d([X.data[i] for i in idx])
            yb = Matrix.from_2d([y.data[i] for i in idx])
            batches.append((xb, yb))
        return batches

    def train(self, X: Matrix, y: Matrix,
              X_val=None, y_val=None,
              callback=None):
        """
        Train the network.
        callback(epoch, loss, val_loss, accuracy) called each epoch.
        """
        self._is_training = True
        self._stop_flag = False
        epochs = self.config.get("epochs", 100)
        batch_size = self.config.get("batch_size", 32)

        for epoch in range(1, epochs + 1):
            if self._stop_flag:
                break

            batches = self._make_batches(X, y, batch_size)
            epoch_loss = 0.0
            for xb, yb in batches:
                pred = self.forward(xb)
                loss_val, grad = self.loss_fn(pred, yb)
                epoch_loss += loss_val
                self.backward(grad)
                self._update()

            epoch_loss /= len(batches)
            self.history["loss"].append(epoch_loss)

            val_loss = None
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                vl, _ = self.loss_fn(val_pred, y_val)
                val_loss = vl
                self.history["val_loss"].append(vl)

            acc = self._accuracy(X, y)
            self.history["accuracy"].append(acc)

            if callback:
                callback(epoch, epoch_loss, val_loss, acc)

        self._is_training = False

    def stop(self):
        self._stop_flag = True

    def _accuracy(self, X: Matrix, y: Matrix) -> float:
        pred = self.forward(X)
        correct = 0
        last_act = self.layers[-1].activation if self.layers else "linear"
        for i in range(pred.rows):
            if last_act == "softmax":
                p_class = pred.data[i].index(max(pred.data[i]))
                t_class = y.data[i].index(max(y.data[i]))
                if p_class == t_class:
                    correct += 1
            elif last_act in ("sigmoid", "bce"):
                p = 1 if pred.data[i][0] >= 0.5 else 0
                t = round(y.data[i][0])
                if p == t:
                    correct += 1
            else:
                # regression: within 10% tolerance
                p = pred.data[i][0]
                t = y.data[i][0]
                rng = abs(t) * 0.1 + 1e-6
                if abs(p - t) <= rng:
                    correct += 1
        return correct / max(pred.rows, 1)

    def predict(self, X: Matrix) -> Matrix:
        return self.forward(X)

    # ------------------------------------------------------------------ #
    def to_dict(self):
        return {
            "config": self.config,
            "layers": [l.to_dict() for l in self.layers],
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d):
        nn = cls(d["config"])
        nn.layers = [DenseLayer.from_dict(ld) for ld in d["layers"]]
        nn.history = d.get("history", {"loss": [], "val_loss": [], "accuracy": []})
        return nn

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def summary(self) -> list:
        rows = []
        total = 0
        for i, l in enumerate(self.layers):
            params = l.in_size * l.out_size + l.out_size
            total += params
            rows.append({
                "layer": i + 1,
                "name": l.name,
                "in": l.in_size,
                "out": l.out_size,
                "activation": l.activation,
                "params": params,
            })
        return {"layers": rows, "total_params": total}
