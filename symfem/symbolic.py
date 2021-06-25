"""Symbolic tools."""

import sympy


class Polynomial:
    def __init__(self, xpow=0, ypow=0, zpow=0, negative=False):
        self._x = xpow
        self._y = ypow
        self._z = zpow
        self._negative = negative

    def to_sympy(self):
        _x = [sympy.Symbol("x"), sympy.Symbol("y"), sympy.Symbol("z")]
        if self._negative:
            return _x[0] ** self._x * _x[1] ** self._y * _x[2] ** self._z
        else:
            return -_x[0] ** self._x * _x[1] ** self._y * _x[2] ** self._z

    def __eq__(self, other):
        if isinstance(other, Polynomial):
            return self._x == other._x and self._y == other._y and self._z == other._z and self._negative == other._negative
        return self.to_sympy() == other

    def __mul__(self, other):
        if isinstance(other, Polynomial):
            return Polynomial(self._x + other._x, self._y + other._y, self._z + other._z, self._negative)
        return self.to_sympy() * other

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Polynomial):
            return Polynomial(self._x - other._x, self._y - other._y, self._z - other._z, self._negative)
        return self.to_sympy() / other

    def __rtruediv__(self, other):
        if isinstance(other, Polynomial):
            return Polynomial(other._x - self._x, other._y - self._y, other._z - self._z, self._negative)
        return other / self.to_sympy()

    def __pow__(self, power):
        return Polynomial(self._x * power, self._y * power, self._z * power, self._negative)

    def __add__(self, other):
        return self.to_sympy() + other

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.to_sympy() - other

    def __rsub__(self, other):
        return other - self.to_sympy()

    def __neg__(self):
        return Polynomial(self._x, self._y, self._z, not self._negative)

    def __getattr__(self, attr):
        return getattr(self.to_sympy(), attr)

x = [Polynomial(xpow=1), Polynomial(ypow=1), Polynomial(zpow=1)]

t = [sympy.Symbol("t0"), sympy.Symbol("t1"), sympy.Symbol("t2")]

zero = 0
one = 1

_dummy = [sympy.Symbol("symbolicpyDUMMYx"), sympy.Symbol("symbolicpyDUMMYy"),
          sympy.Symbol("symbolicpyDUMMYz")]


def subs(f, vars, values):
    """Substitute values into a sympy expression."""

    if isinstance(f, Polynomial):
        return subs(f.to_sympy(), vars, values)
    for i, j in enumerate(vars):
        if isinstance(j, Polynomial):
            return subs(f, vars[:i] + [j.to_sympy()] + vars[i + 1: ], values)
    for i, j in enumerate(values):
        if isinstance(j, Polynomial):
            return subs(f, vars, values[:i] + [j.to_sympy()] + values[i + 1: ])

    if isinstance(f, PiecewiseFunction):
        return f.evaluate(values)
    try:
        return tuple(subs(f_j, vars, values) for f_j in f)
    except TypeError:
        pass
    if isinstance(vars, sympy.Symbol):
        return (zero + f).subs(vars, values)

    if isinstance(f, int):
        return f

    if len(values) == 1:
        return f.subs(vars[0], values[0])
    if len(values) == 2:
        return f.subs(vars[0], _dummy[0]).subs(vars[1], _dummy[1]).subs(
            _dummy[0], values[0]).subs(_dummy[1], values[1])
    if len(values) == 3:
        return f.subs(vars[0], _dummy[0]).subs(vars[1], _dummy[1]).subs(
            vars[2], _dummy[2]).subs(_dummy[0], values[0]).subs(
                _dummy[1], values[1]).subs(_dummy[2], values[2])


def sym_sum(ls):
    """Symbolically computes the sum of a list."""
    out = zero
    for i in ls:
        out += i
    return out


class PiecewiseFunction:
    """A function defined piecewise on a collection of triangles."""

    def __init__(self, pieces):
        self.pieces = pieces

    def get_piece(self, point):
        """Get the piece of the function defined at the given point."""
        from .vectors import point_in_triangle
        for tri, value in self.pieces:
            if point_in_triangle(point[:2], tri):
                return value

        raise NotImplementedError("Evaluation of piecewise functions outside domain not supported.")

    def evaluate(self, values):
        """Evaluate a function."""
        try:
            return subs(self.get_piece(values), x, values)
        except TypeError:
            return PiecewiseFunction([(i, subs(j, x, values)) for i, j in self.pieces])

    def diff(self, variable):
        """Differentiate the function."""
        return PiecewiseFunction([(i, j.diff(variable)) for i, j in self.pieces])

    def __rmul__(self, other):
        """Multiply the function by a scalar."""
        return PiecewiseFunction([(i, other * j) for i, j in self.pieces])

    def __mul__(self, other):
        """Multiply the function by a scalar."""
        return self.__rmul__(other)

    def __radd__(self, other):
        """Add another piecewise function or a scalar."""
        return self.__add__(other)

    def __add__(self, other):
        """Add another piecewise function or a scalar."""
        if isinstance(other, PiecewiseFunction):
            for i, j in zip(self.pieces, other.pieces):
                assert i[0] == j[0]
            return PiecewiseFunction(
                [(i[0], i[1] + j[1]) for i, j in zip(self.pieces, other.pieces)])

        return PiecewiseFunction([(i, other + j) for i, j in self.pieces])
