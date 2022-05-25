"""Symbolic tools."""

import sympy


def to_sympy(i):
    """Convert to a sympy expression."""
    from .basis_function import BasisFunction
    if isinstance(i, list):
        return [to_sympy(j) for j in i]
    if isinstance(i, tuple):
        return tuple(to_sympy(j) for j in i)

    if isinstance(i, int):
        return sympy.Integer(i)

    if isinstance(i, BasisFunction):
        return i.get_function()

    return i


def to_float(i):
    """Convert to a float."""
    if isinstance(i, list):
        return [to_float(j) for j in i]
    if isinstance(i, tuple):
        return tuple(to_float(j) for j in i)

    return float(i)


x = [sympy.Symbol("x"), sympy.Symbol("y"), sympy.Symbol("z")]
t = [sympy.Symbol("t0"), sympy.Symbol("t1"), sympy.Symbol("t2")]
_dummy = [sympy.Symbol("symbolicpyDUMMYx"), sympy.Symbol("symbolicpyDUMMYy"),
          sympy.Symbol("symbolicpyDUMMYz")]


def subs(f, vars, values):
    """Substitute values into a sympy expression."""
    if isinstance(f, PiecewiseFunction):
        return f.evaluate(values)
    try:
        return tuple(subs(f_j, vars, values) for f_j in f)
    except TypeError:
        pass
    if isinstance(vars, sympy.Symbol):
        return to_sympy(f).subs(vars, values)

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
    out = to_sympy(0)
    for i in ls:
        out += i
    return out


def sym_product(ls):
    """Symbolically computes the sum of a list."""
    out = to_sympy(1)
    for i in ls:
        out *= i
    return out


class PiecewiseFunction:
    """A function defined piecewise on a collection of triangles."""

    def __init__(self, pieces):
        self.pieces = pieces

    def get_piece(self, point):
        """Get the piece of the function defined at the given point."""
        if len(self.pieces[0][0]) == 3:
            from .vectors import point_in_triangle
            for tri, value in self.pieces:
                if point_in_triangle(point[:2], tri):
                    return value
        if len(self.pieces[0][0]) == 4:
            from .vectors import point_in_tetrahedron
            for tet, value in self.pieces:
                if point_in_tetrahedron(point, tet):
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
        from .calculus import diff
        return PiecewiseFunction([(i, diff(j, variable)) for i, j in self.pieces])

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

    def _iter_list(self):
        """Make am iterable list."""
        from .basis_function import BasisFunction
        for p in self.pieces:
            assert isinstance(p[1], (list, tuple)) or (
                isinstance(p[1], BasisFunction) and
                isinstance(p[1].get_function(), (list, tuple)))
        return [PiecewiseFunction([(j[0], j[1][i])
                                   for j in self.pieces])
                for i in range(len(self.pieces[0][1]))]

    def __iter__(self):
        """Get iterable."""
        return self._iter_list().__iter__()


def symequal(a, b):
    """Check if two symbolic numbers or vectors are equal."""
    if isinstance(a, (list, tuple)):
        for i, j in zip(a, b):
            if not symequal(i, j):
                return False
        return True
    return sympy.expand(sympy.simplify(a)) == sympy.expand(sympy.simplify(b))
