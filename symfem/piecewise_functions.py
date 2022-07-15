"""Piecewise basis function classes."""

from __future__ import annotations
import sympy
import typing
from abc import ABC, abstractmethod
from .symbols import x
from .functions import AnyFunction, ScalarFunction, _to_sympy_format


class PiecewiseFunction(AnyFunction):
    """A piecewise function."""

    def __init__(self, pieces: typing.List[
        typing.Tuple[typing.List[typing.Tuple[int, ...]], AnyFunction]
    ], tdim: int):
        super().__init__()
        self._pieces = pieces
        self.tdim = tdim

    def get_piece(self, point: PointType):
        if self.tdim == 2:
            from .vectors import point_in_triangle, point_in_quadrilateral
            for cell, value in self._pieces:
                if len(cell) == 3:
                    if point_in_triangle(point[:2], cell):
                        return value
                elif len(cell) == 4:
                    if point_in_quadrilateral(point[:2], cell):
                        return value
                else:
                    raise ValueError("Unsupported cell")
        if self.tdim == 3:
            from .vectors import point_in_tetrahedron
            for cell, value in self._pieces:
                if len(cell) == 4:
                    if point_in_tetrahedron(point, cell):
                        return value
                else:
                    raise ValueError("Unsupported cell")

        raise NotImplementedError("Evaluation of piecewise functions outside domain not supported.")

    def __eq__(self, other: typing.Any) -> bool:
        """Check if two functions are equal."""
        if not isinstance(other, PiecewiseFunction):
            return False

        if self.tdim != other.tdim:
            return False
        for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
            assert shape0 == shape1
            if f0 != f1:
                return False
        return True

    @property
    def pieces(self):
        """Get the pieces of the function."""
        return self._pieces


class PiecewiseScalarFunction(PiecewiseFunction):
    """A piecewise scalar-valued function."""

    def __init__(self, pieces: typing.List[
        typing.Tuple[typing.List[typing.Tuple[int, ...]], typing.Union[ScalarFunction, sympy.core.expr.Expr, int]]
    ], tdim: int):
        super().__init__([(shape, (
            f if isinstance(f, ScalarFunction
        ) else ScalarFunction(f))) for (shape, f) in pieces], tdim)

    def __add__(self, other: typing.Any) -> PiecewiseScalarFunction:
        """Add."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f0 + f1))
            return PiecewiseScalarFunction(new_pieces, self.tdim)
        if isinstance(other, ScalarFunction):
            return PiecewiseScalarFunction([(shape, f + other._f) for shape, f in self._pieces], self.tdim)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return PiecewiseScalarFunction([(shape, f + other) for shape, f in self._pieces], self.tdim)
        return NotImplemented

    def __radd__(self, other: typing.Any) -> PiecewiseScalarFunction:
        """Add."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f1 + f0))
            return PiecewiseScalarFunction(new_pieces, self.tdim)
        if isinstance(other, ScalarFunction):
            return PiecewiseScalarFunction([(shape, other._f + f) for shape, f in self._pieces], self.tdim)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return PiecewiseScalarFunction([(shape, other + f) for shape, f in self._pieces], self.tdim)
        return NotImplemented

    def __sub__(self, other: typing.Any) -> PiecewiseScalarFunction:
        """Subtract."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f0 - f1))
            return PiecewiseScalarFunction(new_pieces, self.tdim)
        if isinstance(other, ScalarFunction):
            return PiecewiseScalarFunction([(shape, f - other._f) for shape, f in self._pieces], self.tdim)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return PiecewiseScalarFunction([(shape, f - other) for shape, f in self._pieces], self.tdim)
        return NotImplemented

    def __rsub__(self, other: typing.Any) -> PiecewiseScalarFunction:
        """Subtract."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f1 - f0))
            return PiecewiseScalarFunction(new_pieces, self.tdim)
        if isinstance(other, ScalarFunction):
            return PiecewiseScalarFunction([(shape, other._f - f) for shape, f in self._pieces], self.tdim)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return PiecewiseScalarFunction([(shape, other - f) for shape, f in self._pieces], self.tdim)
        return NotImplemented

    def __truediv__(self, other: typing.Any) -> PiecewiseScalarFunction:
        """Divide."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f0 / f1))
            return PiecewiseScalarFunction(new_pieces, self.tdim)
        if isinstance(other, ScalarFunction):
            return PiecewiseScalarFunction([(shape, f / other._f) for shape, f in self._pieces], self.tdim)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return PiecewiseScalarFunction([(shape, f / other) for shape, f in self._pieces], self.tdim)
        return NotImplemented

    def __rtruediv__(self, other: typing.Any) -> PiecewiseScalarFunction:
        """Divide."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f1 / f0))
            return PiecewiseScalarFunction(new_pieces, self.tdim)
        if isinstance(other, ScalarFunction):
            return PiecewiseScalarFunction([(shape, other._f / f) for shape, f in self._pieces], self.tdim)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return PiecewiseScalarFunction([(shape, other / f) for shape, f in self._pieces], self.tdim)
        return NotImplemented

    def __mul__(self, other: typing.Any) -> PiecewiseScalarFunction:
        """Multiply."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f0 * f1))
            return PiecewiseScalarFunction(new_pieces, self.tdim)
        if isinstance(other, ScalarFunction):
            return PiecewiseScalarFunction([(shape, f * other._f) for shape, f in self._pieces], self.tdim)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return PiecewiseScalarFunction([(shape, f * other) for shape, f in self._pieces], self.tdim)
        return NotImplemented

    def __rmul__(self, other: typing.Any) -> PiecewiseScalarFunction:
        """Multiply."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f1 * f0))
            return PiecewiseScalarFunction(new_pieces, self.tdim)
        if isinstance(other, ScalarFunction):
            return PiecewiseScalarFunction([(shape, other._f * f) for shape, f in self._pieces], self.tdim)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return PiecewiseScalarFunction([(shape, other * f) for shape, f in self._pieces], self.tdim)
        return NotImplemented

    def __neg__(self) -> PiecewiseScalarFunction:
        """Negate."""
        return PiecewiseScalarFunction([(shape, -f) for shape, f in self._pieces], self.tdim)

    def as_sympy(self) -> SympyFormat:
        """Convert to a sympy expression."""
        return tuple((tuple(shape), f) for shape, f in self._pieces)

    def subs(self, vars: AxisVariables, values: ValuesToSubstitute) -> typing.Union(PiecewiseScalarFunction, ScalarFunction):
        """Substitute values into the function."""
        if isinstance(values, (tuple, list)) and len(values) == self.tdim:
            for i in values:
                if not _to_sympy_format(i).is_constant():
                    break
            else:
                return self.get_piece(values).subs(vars, values)

        return PiecewiseScalarFunction([(shape, f.subs(vars, values)) for shape, f in self._pieces], self.tdim)

    def diff(self, variable: sympy.core.symbol.Symbol) -> PiecewiseScalarFunction:
        """Differentiate the function."""
        return PiecewiseScalarFunction([(shape, f.diff(variable)) for shape, f in self._pieces], self.tdim)

    def directional_derivative(self, direction: PointType):
        """Compute a directional derivative."""
        out = 0
        for i, j in zip(x, direction):
            out += j * self.diff(i)
        return out

    def jacobian_component(self, component: typing.Tuple[int, int]) -> PiecewiseScalarFunction:
        """Compute a component of the jacobian."""
        return self.diff(x[component[0]]).diff(x[component[1]])

    def jacobian(self, dim: int) -> MatrixFunction:
        """Compute the jacobian."""
        raise NotImplementedError()

    def dot(self, other: AnyFunction) -> PiecewiseScalarFunction:
        """Compute the dot product with another function."""
        if isinstance(other, ScalarFunction):
            return self * other
        raise NotImplementedError()

    def div(self):
        """Compute the div of the function."""
        raise ValueError("Cannot compute the div of a scalar-valued function.")

    def grad(self, dim: int) -> VectorFunction:
        """Compute the grad of the function."""
        return VectorFunction([self.diff(x[i]) for i in range(dim)])

    def curl(self):
        """Compute the curl of the function."""
        raise ValueError("Cannot compute the curl of a scalar-valued function.")

    def integrate(self, *limits: typing.Tuple[sympy.core.symbol.Symbol, typing.Union[int, sympy.core.expr.Expr], typing.Union[int, sympy.core.expr.Expr]]):
        """Compute the integral of the function."""
        return self._f.integrate(*limits)


class PiecewiseVectorFunction(AnyFunction):
    """A piecewise vector-valued function."""

    pass


class PiecewiseMatrixFunction(AnyFunction):
    """A piecewise matrix-valued function."""

    pass
