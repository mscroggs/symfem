"""Piecewise basis function classes."""

from __future__ import annotations
import sympy
import typing
from .functions import (AnyFunction, _to_sympy_format, AxisVariables, ValuesToSubstitute,
                        SympyFormat, FunctionInput, parse_function_input)
from .geometry import PointType


class PiecewiseFunction(AnyFunction):
    """A piecewise function."""

    _pieces: typing.List[AnyFunction]

    def __init__(self, pieces: typing.List[
        typing.Tuple[typing.List[typing.Tuple[int, ...]], FunctionInput]
    ], tdim: int):
        self._pieces = [(shape, parse_function_input(f)) for shape, f in pieces]
        self.tdim = tdim

        assert len(self._pieces) > 0
        if self._pieces[0][1].is_scalar:
            for _, f in self._pieces:
                assert f.is_scalar
            super().__init__(scalar=True)
        elif self._pieces[0][1].is_vector:
            for _, f in self._pieces:
                assert f.is_vector
            super().__init__(vector=True)
        else:
            for _, f in self._pieces:
                assert f.is_matrix
            super().__init__(matrix=True)

    def as_sympy(self) -> SympyFormat:
        """Convert to a sympy expression."""
        return {tuple(shape): f for shape, f in self._pieces}

    def as_tex(self) -> str:
        """Convert to a TeX expression."""
        out = "\\begin{cases}\n"
        joiner = ""
        for shape, f in self._pieces:
            out += joiner
            joiner = "\\\\"
            out += f.as_latex.replace("\\frac", "\\tfrac")
            out += "&\\text{in }\\operatorname{"
            if self.tdim == 2:
                if len(shape) == 3:
                    out += "Triangle"
                elif len(shape) == 4:
                    out += "Quadrilateral"
                else:
                    raise ValueError("Unsupported shape")
            elif self.tdim == 3:
                if len(shape) == 4:
                    out += "Tetrahedron"
                else:
                    raise ValueError("Unsupported shape")
            else:
                raise ValueError("Unsupported shape")
            out += "}"
            out += f"({shape})"
        out += "\\end{cases}"
        return out

    def get_piece(self, point: PointType) -> AnyFunction:
        """Get a pieces of the function."""
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
    def pieces(self) -> typing.List[
        typing.Tuple[typing.List[typing.Tuple[int, ...]], AnyFunction]
    ]:
        """Get the pieces of the function."""
        return self._pieces

    def __add__(self, other: typing.Any) -> PiecewiseFunction:
        """Add."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f0 + f1))
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction(
            [(shape, f + other) for shape, f in self._pieces], self.tdim)

    def __radd__(self, other: typing.Any) -> PiecewiseFunction:
        """Add."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f1 + f0))
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction(
            [(shape, other + f) for shape, f in self._pieces], self.tdim)

    def __sub__(self, other: typing.Any) -> PiecewiseFunction:
        """Subtract."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f0 - f1))
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction(
            [(shape, f - other) for shape, f in self._pieces], self.tdim)

    def __rsub__(self, other: typing.Any) -> PiecewiseFunction:
        """Subtract."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f1 - f0))
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction([
            (shape, other - f) for shape, f in self._pieces], self.tdim)

    def __truediv__(self, other: typing.Any) -> PiecewiseFunction:
        """Divide."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f0 / f1))
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction([
            (shape, f / other) for shape, f in self._pieces], self.tdim)

    def __rtruediv__(self, other: typing.Any) -> PiecewiseFunction:
        """Divide."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f1 / f0))
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction([
            (shape, other / f) for shape, f in self._pieces], self.tdim)

    def __mul__(self, other: typing.Any) -> PiecewiseFunction:
        """Multiply."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f0 * f1))
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction([
            (shape, f * other) for shape, f in self._pieces], self.tdim)

    def __rmul__(self, other: typing.Any) -> PiecewiseFunction:
        """Multiply."""
        if isinstance(other, PiecewiseFunction):
            new_pieces = []
            for (shape0, f0), (shape1, f1) in zip(self._pieces, other._pieces):
                assert shape0 == shape1
                new_pieces.append((shape0, f1 * f0))
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction([
            (shape, other * f) for shape, f in self._pieces], self.tdim)

    def __neg__(self) -> PiecewiseFunction:
        """Negate."""
        return PiecewiseFunction([(shape, -f) for shape, f in self._pieces], self.tdim)

    def subs(
        self, vars: AxisVariables, values: ValuesToSubstitute
    ) -> PiecewiseFunction:
        """Substitute values into the function."""
        if isinstance(values, (tuple, list)) and len(values) == self.tdim:
            for i in values:
                if not _to_sympy_format(i).is_constant():
                    break
            else:
                return self.get_piece(values).subs(vars, values)

        return PiecewiseFunction([
            (shape, f.subs(vars, values)) for shape, f in self._pieces], self.tdim)

    def diff(self, variable: sympy.core.symbol.Symbol) -> PiecewiseFunction:
        """Differentiate the function."""
        return PiecewiseFunction([
            (shape, f.diff(variable)) for shape, f in self._pieces], self.tdim)

    def directional_derivative(self, direction: PointType) -> PiecewiseFunction:
        """Compute a directional derivative."""
        return PiecewiseFunction([
            (shape, f.directional_derivative(direction)) for shape, f in self._pieces], self.tdim)

    def jacobian_component(self, component: typing.Tuple[int, int]) -> PiecewiseFunction:
        """Compute a component of the jacobian."""
        return PiecewiseFunction([
            (shape, f.jacobian_component(component)) for shape, f in self._pieces], self.tdim)

    def jacobian(self, dim: int) -> PiecewiseFunction:
        """Compute the jacobian."""
        return PiecewiseFunction([
            (shape, f.jacobian(dim)) for shape, f in self._pieces], self.tdim)

    def dot(self, other: AnyFunction) -> PiecewiseFunction:
        """Compute the dot product with another function."""
        return PiecewiseFunction([
            (shape, f.dot(other)) for shape, f in self._pieces], self.tdim)

    def div(self) -> PiecewiseFunction:
        """Compute the div of the function."""
        return PiecewiseFunction([
            (shape, f.div()) for shape, f in self._pieces], self.tdim)

    def grad(self, dim: int) -> PiecewiseFunction:
        """Compute the grad of the function."""
        return PiecewiseFunction([
            (shape, f.grad(dim)) for shape, f in self._pieces], self.tdim)

    def curl(self) -> PiecewiseFunction:
        """Compute the curl of the function."""
        return PiecewiseFunction([
            (shape, f.curl()) for shape, f in self._pieces], self.tdim)

    def integrate(
        self, *limits: typing.Tuple[
            sympy.core.symbol.Symbol, typing.Union[int, sympy.core.expr.Expr],
            typing.Union[int, sympy.core.expr.Expr]]
    ):
        """Compute the integral of the function."""
        raise NotImplementedError()
