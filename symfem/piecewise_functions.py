"""Piecewise basis function classes."""

from __future__ import annotations
import sympy
import typing
from .functions import (AnyFunction, _to_sympy_format, ValuesToSubstitute,
                        SympyFormat, FunctionInput, parse_function_input, VectorFunction)
from .geometry import (PointType, SetOfPoints, point_in_triangle, point_in_quadrilateral,
                       point_in_tetrahedron, SetOfPointsInput, parse_set_of_points_input)
from .references import Reference
from .symbols import x, t, AxisVariables, AxisVariablesNotSingle


class PiecewiseFunction(AnyFunction):
    """A piecewise function."""

    _pieces: typing.Dict[SetOfPoints, AnyFunction]

    def __init__(
        self, pieces: typing.Dict[SetOfPointsInput, FunctionInput], tdim: int
    ):
        self._pieces = {parse_set_of_points_input(shape): parse_function_input(f)
                        for shape, f in pieces.items()}
        self.tdim = tdim

        assert len(self._pieces) > 0
        self.first_piece = list(self._pieces.values())[0]
        if self.first_piece.is_scalar:
            for f in self._pieces.values():
                assert f.is_scalar
            super().__init__(scalar=True)
        elif self.first_piece.is_vector:
            for f in self._pieces.values():
                assert f.is_vector
            super().__init__(vector=True)
        else:
            for f in self._pieces.values():
                assert f.is_matrix
            super().__init__(matrix=True)

    def __len__(self):
        """Get the length of the vector."""
        if not self.is_vector:
            raise TypeError(f"object of type '{self.__class__.__name__}' has no len()")
        return len(self.first_piece)

    def as_sympy(self) -> SympyFormat:
        """Convert to a sympy expression."""
        for f in self._pieces.values():
            if f != self.first_piece:
                break
        else:
            return self.first_piece.as_sympy()
        out = {}
        for shape, f in self._pieces.items():
            fs = f.as_sympy()
            assert isinstance(fs, (sympy.core.expr.Expr, tuple,
                                   sympy.matrices.dense.MutableDenseMatrix))
            out[tuple(shape)] = fs
        return out

    def as_tex(self) -> str:
        """Convert to a TeX expression."""
        out = "\\begin{cases}\n"
        joiner = ""
        for shape, f in self._pieces.items():
            out += joiner
            joiner = "\\\\"
            out += f.as_tex().replace("\\frac", "\\tfrac")
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
            for cell, value in self._pieces.items():
                if len(cell) == 3:
                    if point_in_triangle(point[:2], cell):
                        return value
                elif len(cell) == 4:
                    if point_in_quadrilateral(point[:2], cell):
                        return value
                else:
                    raise ValueError("Unsupported cell")
        if self.tdim == 3:
            for cell, value in self._pieces.items():
                if len(cell) == 4:
                    if point_in_tetrahedron(point, cell):
                        return value
                else:
                    raise ValueError("Unsupported cell")

        raise NotImplementedError("Evaluation of piecewise functions outside domain not supported.")

    def __getitem__(self, key) -> PiecewiseFunction:
        """Get a component or slice of the function."""
        return PiecewiseFunction(
            {shape: f.__getitem__(key) for shape, f in self._pieces.items()}, self.tdim)

    def __eq__(self, other: typing.Any) -> bool:
        """Check if two functions are equal."""
        if not isinstance(other, PiecewiseFunction):
            return False

        if self.tdim != other.tdim:
            return False
        for (shape0, f0), (shape1, f1) in zip(self._pieces.items(), other._pieces.items()):
            assert shape0 == shape1
            if f0 != f1:
                return False
        return True

    @property
    def pieces(self) -> typing.Dict[SetOfPoints, AnyFunction]:
        """Get the pieces of the function."""
        return self._pieces

    def __add__(self, other: typing.Any) -> PiecewiseFunction:
        """Add."""
        if isinstance(other, PiecewiseFunction):
            new_pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
            for (shape0, f0), (shape1, f1) in zip(self._pieces.items(), other._pieces.items()):
                assert shape0 == shape1
                new_pieces[shape0] = f0 + f1
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction(
            {shape: f + other for shape, f in self._pieces.items()}, self.tdim)

    def __radd__(self, other: typing.Any) -> PiecewiseFunction:
        """Add."""
        if isinstance(other, PiecewiseFunction):
            new_pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
            for (shape0, f0), (shape1, f1) in zip(self._pieces.items(), other._pieces.items()):
                assert shape0 == shape1
                new_pieces[shape0] = f1 + f0
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction(
            {shape: other + f for shape, f in self._pieces.items()}, self.tdim)

    def __sub__(self, other: typing.Any) -> PiecewiseFunction:
        """Subtract."""
        if isinstance(other, PiecewiseFunction):
            new_pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
            for (shape0, f0), (shape1, f1) in zip(self._pieces.items(), other._pieces.items()):
                assert shape0 == shape1
                new_pieces[shape0] = f0 - f1
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction(
            {shape: f - other for shape, f in self._pieces.items()}, self.tdim)

    def __rsub__(self, other: typing.Any) -> PiecewiseFunction:
        """Subtract."""
        if isinstance(other, PiecewiseFunction):
            new_pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
            for (shape0, f0), (shape1, f1) in zip(self._pieces.items(), other._pieces.items()):
                assert shape0 == shape1
                new_pieces[shape0] = f1 - f0
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction(
            {shape: other - f for shape, f in self._pieces.items()}, self.tdim)

    def __truediv__(self, other: typing.Any) -> PiecewiseFunction:
        """Divide."""
        if isinstance(other, PiecewiseFunction):
            new_pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
            for (shape0, f0), (shape1, f1) in zip(self._pieces.items(), other._pieces.items()):
                assert shape0 == shape1
                new_pieces[shape0] = f0 / f1
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction(
            {shape: f / other for shape, f in self._pieces.items()}, self.tdim)

    def __rtruediv__(self, other: typing.Any) -> PiecewiseFunction:
        """Divide."""
        if isinstance(other, PiecewiseFunction):
            new_pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
            for (shape0, f0), (shape1, f1) in zip(self._pieces.items(), other._pieces.items()):
                assert shape0 == shape1
                new_pieces[shape0] = f1 / f0
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction(
            {shape: other / f for shape, f in self._pieces.items()}, self.tdim)

    def __mul__(self, other: typing.Any) -> PiecewiseFunction:
        """Multiply."""
        if isinstance(other, PiecewiseFunction):
            new_pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
            for (shape0, f0), (shape1, f1) in zip(self._pieces.items(), other._pieces.items()):
                assert shape0 == shape1
                new_pieces[shape0] = f0 * f1
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction(
            {shape: f * other for shape, f in self._pieces.items()}, self.tdim)

    def __rmul__(self, other: typing.Any) -> PiecewiseFunction:
        """Multiply."""
        if isinstance(other, PiecewiseFunction):
            new_pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
            for (shape0, f0), (shape1, f1) in zip(self._pieces.items(), other._pieces.items()):
                assert shape0 == shape1
                new_pieces[shape0] = f1 * f0
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction(
            {shape: other * f for shape, f in self._pieces.items()}, self.tdim)

    def __matmul__(self, other: typing.Any) -> PiecewiseFunction:
        """Multiply."""
        if isinstance(other, PiecewiseFunction):
            new_pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
            for (shape0, f0), (shape1, f1) in zip(self._pieces.items(), other._pieces.items()):
                assert shape0 == shape1
                new_pieces[shape0] = f0 @ f1
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction(
            {shape: f @ other for shape, f in self._pieces.items()}, self.tdim)

    def __rmatmul__(self, other: typing.Any) -> PiecewiseFunction:
        """Multiply."""
        if isinstance(other, PiecewiseFunction):
            new_pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
            for (shape0, f0), (shape1, f1) in zip(self._pieces.items(), other._pieces.items()):
                assert shape0 == shape1
                new_pieces[shape0] = f1 @ f0
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction(
            {shape: other @ f for shape, f in self._pieces.items()}, self.tdim)

    def __pow__(self, other: typing.Any) -> PiecewiseFunction:
        """Raise to a power."""
        if isinstance(other, PiecewiseFunction):
            new_pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
            for (shape0, f0), (shape1, f1) in zip(self._pieces.items(), other._pieces.items()):
                assert shape0 == shape1
                new_pieces[shape0] = f0 ** f1
            return PiecewiseFunction(new_pieces, self.tdim)
        return PiecewiseFunction(
            {shape: f ** other for shape, f in self._pieces.items()}, self.tdim)

    def __neg__(self) -> PiecewiseFunction:
        """Negate."""
        return PiecewiseFunction({shape: -f for shape, f in self._pieces.items()}, self.tdim)

    def subs(
        self, vars: AxisVariables, values: ValuesToSubstitute
    ) -> PiecewiseFunction:
        """Substitute values into the function."""
        if isinstance(values, AnyFunction):
            values = values.as_sympy()

        if isinstance(values, (tuple, list)) and len(values) == self.tdim:
            for i in values:
                i_s = _to_sympy_format(i)
                assert isinstance(i_s, sympy.core.expr.Expr)
                if not i_s.is_constant():
                    break
            else:
                return self.get_piece(tuple(values)).subs(vars, values)

        return PiecewiseFunction(
            {shape: f.subs(vars, values) for shape, f in self._pieces.items()}, self.tdim)

    def diff(self, variable: sympy.core.symbol.Symbol) -> PiecewiseFunction:
        """Differentiate the function."""
        return PiecewiseFunction(
            {shape: f.diff(variable) for shape, f in self._pieces.items()}, self.tdim)

    def directional_derivative(self, direction: PointType) -> PiecewiseFunction:
        """Compute a directional derivative."""
        return PiecewiseFunction(
            {shape: f.directional_derivative(direction)
             for shape, f in self._pieces.items()}, self.tdim)

    def jacobian_component(self, component: typing.Tuple[int, int]) -> PiecewiseFunction:
        """Compute a component of the jacobian."""
        return PiecewiseFunction(
            {shape: f.jacobian_component(component) for shape, f in self._pieces.items()},
            self.tdim)

    def jacobian(self, dim: int) -> PiecewiseFunction:
        """Compute the jacobian."""
        return PiecewiseFunction(
            {shape: f.jacobian(dim) for shape, f in self._pieces.items()}, self.tdim)

    def dot(self, other_in: FunctionInput) -> PiecewiseFunction:
        """Compute the dot product with another function."""
        return PiecewiseFunction(
            {shape: f.dot(other_in) for shape, f in self._pieces.items()}, self.tdim)

    def cross(self, other_in: FunctionInput) -> PiecewiseFunction:
        """Compute the cross product with another function."""
        return PiecewiseFunction(
            {shape: f.cross(other_in) for shape, f in self._pieces.items()}, self.tdim)

    def div(self) -> PiecewiseFunction:
        """Compute the div of the function."""
        return PiecewiseFunction(
            {shape: f.div() for shape, f in self._pieces.items()}, self.tdim)

    def grad(self, dim: int) -> PiecewiseFunction:
        """Compute the grad of the function."""
        return PiecewiseFunction(
            {shape: f.grad(dim) for shape, f in self._pieces.items()}, self.tdim)

    def curl(self) -> PiecewiseFunction:
        """Compute the curl of the function."""
        return PiecewiseFunction(
            {shape: f.curl() for shape, f in self._pieces.items()}, self.tdim)

    def norm(self) -> PiecewiseFunction:
        """Compute the norm of the function."""
        return PiecewiseFunction(
            {shape: f.norm() for shape, f in self._pieces.items()}, self.tdim)

    def integral(self, domain: Reference, vars: AxisVariablesNotSingle = t) -> AnyFunction:
        """Compute the integral of the function."""
        # TODO: Add check that the domain is a subset of one piece
        # TODO: Add integral over multiple pieces
        p = self.get_piece(domain.midpoint())
        return p.integral(domain, vars)

    def det(self) -> PiecewiseFunction:
        """Compute the determinant."""
        if not self.is_matrix:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'det'")
        return PiecewiseFunction(
            {shape: f.det() for shape, f in self._pieces.items()}, self.tdim)

    def transpose(self) -> PiecewiseFunction:
        """Compute the transpose."""
        if not self.is_matrix:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'transpose'")
        return PiecewiseFunction(
            {shape: f.transpose() for shape, f in self._pieces.items()}, self.tdim)

    def map_pieces(self, fwd_map: PointType):
        """Map the function's pieces."""
        new_pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
        for shape, f in self._pieces.items():
            nshape = []
            for v in shape:
                pt = VectorFunction(fwd_map).subs(x, v).as_sympy()
                assert isinstance(pt, tuple)
                nshape.append(pt)
            new_pieces[tuple(nshape)] = f
        self._pieces = {parse_set_of_points_input(shape): parse_function_input(f)
                        for shape, f in new_pieces.items()}

    @property
    def shape(self) -> typing.Tuple[int, ...]:
        """Get the value shape of the function."""
        return self.first_piece.shape

    def plot_values(
        self, reference: Reference, img: typing.Any,
        scale: sympy.core.expr.Expr = sympy.Integer(1), n: int = 6
    ):
        """Plot the function's values."""
        from .plotting import Picture
        from .create import create_reference
        assert isinstance(img, Picture)

        for shape, f in self._pieces.items():
            if self.tdim == 2:
                if len(shape) == 3:
                    ref = create_reference("triangle", shape)
                elif len(shape) == 4:
                    ref = create_reference("quadrilateral", shape)
                else:
                    raise ValueError("Unsupported cell type")
            elif self.tdim == 3:
                if len(shape) == 4:
                    ref = create_reference("tetrahedron", shape)
                else:
                    raise ValueError("Unsupported cell type")
            else:
                raise ValueError("Unsupported tdim")
            f.plot_values(ref, img, scale, n // 2)
