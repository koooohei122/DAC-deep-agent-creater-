"""
Pure Python matrix operations - no external libraries.
Implements: creation, multiplication, transpose, element-wise ops, etc.
"""
import math
import random


class Matrix:
    """2D matrix with pure Python operations."""

    def __init__(self, rows, cols, data=None):
        self.rows = rows
        self.cols = cols
        if data is not None:
            self.data = [list(row) for row in data]
        else:
            self.data = [[0.0] * cols for _ in range(rows)]

    # ------------------------------------------------------------------ #
    # Factory methods
    # ------------------------------------------------------------------ #
    @classmethod
    def zeros(cls, rows, cols):
        return cls(rows, cols)

    @classmethod
    def ones(cls, rows, cols):
        m = cls(rows, cols)
        m.data = [[1.0] * cols for _ in range(rows)]
        return m

    @classmethod
    def from_list(cls, lst):
        """Create column vector from a flat list."""
        m = cls(len(lst), 1)
        for i, v in enumerate(lst):
            m.data[i][0] = float(v)
        return m

    @classmethod
    def from_2d(cls, lst2d):
        rows = len(lst2d)
        cols = len(lst2d[0])
        m = cls(rows, cols)
        m.data = [[float(v) for v in row] for row in lst2d]
        return m

    @classmethod
    def random(cls, rows, cols, scale=0.1):
        """Xavier / small random init."""
        m = cls(rows, cols)
        limit = math.sqrt(6.0 / (rows + cols))
        m.data = [[random.uniform(-limit, limit) * scale / 0.1
                   for _ in range(cols)] for _ in range(rows)]
        return m

    @classmethod
    def identity(cls, n):
        m = cls(n, n)
        for i in range(n):
            m.data[i][i] = 1.0
        return m

    # ------------------------------------------------------------------ #
    # Basic operations
    # ------------------------------------------------------------------ #
    def copy(self):
        return Matrix.from_2d(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, val):
        self.data[idx] = val

    def get(self, r, c):
        return self.data[r][c]

    def set(self, r, c, v):
        self.data[r][c] = float(v)

    # ------------------------------------------------------------------ #
    # Arithmetic
    # ------------------------------------------------------------------ #
    def add(self, other):
        """Element-wise addition (supports broadcasting over rows)."""
        if isinstance(other, Matrix):
            if self.rows == other.rows and self.cols == other.cols:
                result = Matrix(self.rows, self.cols)
                for i in range(self.rows):
                    for j in range(self.cols):
                        result.data[i][j] = self.data[i][j] + other.data[i][j]
                return result
            # broadcast: other is (1, cols)
            if other.rows == 1 and self.cols == other.cols:
                result = Matrix(self.rows, self.cols)
                for i in range(self.rows):
                    for j in range(self.cols):
                        result.data[i][j] = self.data[i][j] + other.data[0][j]
                return result
            raise ValueError(f"Shape mismatch add: {self.shape()} vs {other.shape()}")
        # scalar
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other
        return result

    def sub(self, other):
        if isinstance(other, Matrix):
            if self.rows == other.rows and self.cols == other.cols:
                result = Matrix(self.rows, self.cols)
                for i in range(self.rows):
                    for j in range(self.cols):
                        result.data[i][j] = self.data[i][j] - other.data[i][j]
                return result
            raise ValueError(f"Shape mismatch sub: {self.shape()} vs {other.shape()}")
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] - other
        return result

    def mul(self, other):
        """Matrix multiplication."""
        if self.cols != other.rows:
            raise ValueError(f"Shape mismatch mul: {self.shape()} vs {other.shape()}")
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for k in range(self.cols):
                if self.data[i][k] == 0.0:
                    continue
                for j in range(other.cols):
                    result.data[i][j] += self.data[i][k] * other.data[k][j]
        return result

    def hadamard(self, other):
        """Element-wise multiplication."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"Shape mismatch hadamard: {self.shape()} vs {other.shape()}")
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] * other.data[i][j]
        return result

    def scale(self, s):
        """Scalar multiplication."""
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] * s
        return result

    def divide(self, s):
        return self.scale(1.0 / s)

    # ------------------------------------------------------------------ #
    # Shape / transform
    # ------------------------------------------------------------------ #
    def T(self):
        """Transpose."""
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result

    def shape(self):
        return (self.rows, self.cols)

    def flatten(self):
        out = []
        for row in self.data:
            out.extend(row)
        return out

    def sum_rows(self):
        """Sum over rows -> (1, cols)."""
        result = Matrix(1, self.cols)
        for j in range(self.cols):
            s = 0.0
            for i in range(self.rows):
                s += self.data[i][j]
            result.data[0][j] = s
        return result

    def mean_rows(self):
        return self.sum_rows().scale(1.0 / self.rows)

    # ------------------------------------------------------------------ #
    # Apply function element-wise
    # ------------------------------------------------------------------ #
    def apply(self, fn):
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = fn(self.data[i][j])
        return result

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def max_val(self):
        m = float('-inf')
        for row in self.data:
            for v in row:
                if v > m:
                    m = v
        return m

    def min_val(self):
        m = float('inf')
        for row in self.data:
            for v in row:
                if v < m:
                    m = v
        return m

    def sum(self):
        s = 0.0
        for row in self.data:
            for v in row:
                s += v
        return s

    def mean(self):
        return self.sum() / (self.rows * self.cols)

    def std(self):
        mu = self.mean()
        var = 0.0
        n = self.rows * self.cols
        for row in self.data:
            for v in row:
                var += (v - mu) ** 2
        return math.sqrt(var / n)

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self):
        return {"rows": self.rows, "cols": self.cols, "data": self.data}

    @classmethod
    def from_dict(cls, d):
        return cls.from_2d(d["data"])

    def __repr__(self):
        return f"Matrix({self.rows}x{self.cols})"
