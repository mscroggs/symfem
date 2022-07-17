"""Basis function classes."""

from __future__ import annotations
import sympy
import typing
from abc import ABC, abstractmethod
from .symbols import x
from .geometry import PointType

SingleSympyFormat = typing.Union[
    sympy.core.expr.Expr,
    typing.Tuple[sympy.core.expr.Expr, ...],
    sympy.matrices.dense.MutableDenseMatrix
]
SympyFormat = typing.Union[
    SingleSympyFormat,
    typing.Dict[typing.Tuple[typing.Tuple[int, ...], ...], SingleSympyFormat]
]
AxisVariables = typing.Union[
    typing.Tuple[sympy.core.symbol.Symbol, ...],
    typing.List[sympy.core.symbol.Symbol],
    sympy.core.symbol.Symbol,
    ]
ValuesToSubstitute = typing.Union[
    typing.Tuple[typing.Any, ...],
    typing.List[typing.Any],
    typing.Any]


def _to_sympy_format(item: typing.Any) -> SympyFormat:
    """Convert to Sympy format used by these functions."""
    if isinstance(item, AnyFunction):
        return item.as_sympy()

    if isinstance(item, int):
        return sympy.Integer(item)
    if isinstance(item, sympy.core.expr.Expr):
        return item

    if isinstance(item, (list, tuple)):
        if isinstance(item[0], (list, tuple)):
            return sympy.Matrix(item)
        out = []
        for i in item:
            ii = _to_sympy_format(i)
            assert isinstance(ii, sympy.core.expr.Expr)
            out.append(ii)
        return tuple(out)

    if isinstance(item, sympy.matrices.dense.MutableDenseMatrix):
        return item

    raise NotImplementedError()


def _check_equal(first: SympyFormat, second: SympyFormat) -> bool:
    """Check if two items are equal."""
    if isinstance(first, sympy.core.expr.Expr) and isinstance(second, sympy.core.expr.Expr):
        return (first - second).expand().simplify() == 0

    if isinstance(first, tuple) and isinstance(second, tuple):
        if len(first) != len(second):
            return False
        for i, j in zip(first, second):
            if not _check_equal(i, j):
                return False
        return True

    if isinstance(
        first, sympy.matrices.dense.MutableDenseMatrix
    ) and isinstance(
        second, sympy.matrices.dense.MutableDenseMatrix
    ):
        if first.rows != second.rows:
            return False
        if first.cols != second.cols:
            return False
        for i in range(first.rows):
            for j in range(first.cols):
                if not _check_equal(first[i, j], second[i, j]):
                    return False
        return True

    return False


class AnyFunction(ABC):
    """A function."""

    def __init__(self, scalar=False, vector=False, matrix=False):
        assert len([i for i in [scalar, vector, matrix] if i]) == 1
        self.is_scalar = scalar
        self.is_vector = vector
        self.is_matrix = matrix

    @abstractmethod
    def __add__(self, other: typing.Any):
        """Add."""
        pass

    @abstractmethod
    def __radd__(self, other: typing.Any):
        """Add."""
        pass

    @abstractmethod
    def __sub__(self, other: typing.Any):
        """Subtract."""
        pass

    @abstractmethod
    def __rsub__(self, other: typing.Any):
        """Subtract."""
        pass

    @abstractmethod
    def __neg__(self):
        """Negate."""
        pass

    @abstractmethod
    def __truediv__(self, other: typing.Any):
        """Divide."""
        pass

    @abstractmethod
    def __rtruediv__(self, other: typing.Any):
        """Divide."""
        pass

    @abstractmethod
    def __mul__(self, other: typing.Any):
        """Multiply."""
        pass

    @abstractmethod
    def __rmul__(self, other: typing.Any):
        """Multiply."""
        pass

    @abstractmethod
    def as_sympy(self) -> SympyFormat:
        """Convert to a sympy expression."""
        pass

    @abstractmethod
    def as_tex(self) -> str:
        """Convert to a TeX expression."""
        pass

    @abstractmethod
    def subs(self, vars: AxisVariables, values: ValuesToSubstitute):
        """Substitute values into the function."""
        pass

    @abstractmethod
    def diff(self, variable: sympy.core.symbol.Symbol):
        """Differentiate the function."""
        pass

    @abstractmethod
    def directional_derivative(self, direction: PointType):
        """Compute a directional derivative."""
        pass

    @abstractmethod
    def jacobian_component(self, direction: typing.Tuple[int, int]):
        """Compute a component of the jacobian."""
        pass

    @abstractmethod
    def jacobian(self, dim: int):
        """Compute the jacobian."""
        pass

    @abstractmethod
    def dot(self, other: AnyFunction):
        """Compute the dot product with another function."""
        pass

    @abstractmethod
    def div(self):
        """Compute the div of the function."""
        pass

    @abstractmethod
    def grad(self, dim: int):
        """Compute the grad of the function."""
        pass

    @abstractmethod
    def curl(self):
        """Compute the curl of the function."""
        pass

    @abstractmethod
    def integrate(
        self, *limits: typing.Tuple[
            sympy.core.symbol.Symbol, typing.Union[int, sympy.core.expr.Expr],
            typing.Union[int, sympy.core.expr.Expr]]
    ):
        """Compute the integral of the function."""
        pass

    def __repr__(self) -> str:
        """Representation."""
        return self.as_sympy().__repr__()

    def __eq__(self, other: typing.Any) -> bool:
        """Check if two functions are equal."""
        return _check_equal(_to_sympy_format(self), _to_sympy_format(other))


class ScalarFunction(AnyFunction):
    """A scalar-valued function."""

    _f: sympy.core.expr.Expr

    def __init__(self, f: typing.Union[int, sympy.core.expr.Expr]):
        super().__init__(scalar=True)
        if isinstance(f, int):
            self._f = sympy.Integer(f)
        else:
            self._f = f
        assert isinstance(self._f, sympy.core.expr.Expr)

    def __add__(self, other: typing.Any) -> ScalarFunction:
        """Add."""
        if isinstance(other, ScalarFunction):
            return ScalarFunction(self._f + other._f)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return ScalarFunction(self._f + other)
        return NotImplemented

    def __radd__(self, other: typing.Any) -> ScalarFunction:
        """Add."""
        if isinstance(other, ScalarFunction):
            return ScalarFunction(other._f + self._f)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return ScalarFunction(other + self._f)
        return NotImplemented

    def __sub__(self, other: typing.Any) -> ScalarFunction:
        """Subtract."""
        if isinstance(other, ScalarFunction):
            return ScalarFunction(self._f - other._f)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return ScalarFunction(self._f - other)
        return NotImplemented

    def __rsub__(self, other: typing.Any) -> ScalarFunction:
        """Subtract."""
        if isinstance(other, ScalarFunction):
            return ScalarFunction(other._f - self._f)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return ScalarFunction(other - self._f)
        return NotImplemented

    def __truediv__(self, other: typing.Any) -> ScalarFunction:
        """Divide."""
        if isinstance(other, ScalarFunction):
            return ScalarFunction(self._f / other._f)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return ScalarFunction(self._f / other)
        return NotImplemented

    def __rtruediv__(self, other: typing.Any) -> ScalarFunction:
        """Divide."""
        if isinstance(other, ScalarFunction):
            return ScalarFunction(other._f / self._f)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return ScalarFunction(other / self._f)
        return NotImplemented

    def __mul__(self, other: typing.Any) -> ScalarFunction:
        """Multiply."""
        if isinstance(other, ScalarFunction):
            return ScalarFunction(self._f * other._f)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return ScalarFunction(self._f * other)
        return NotImplemented

    def __rmul__(self, other: typing.Any) -> ScalarFunction:
        """Multiply."""
        if isinstance(other, ScalarFunction):
            return ScalarFunction(other._f * self._f)
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return ScalarFunction(other * self._f)
        return NotImplemented

    def __neg__(self) -> ScalarFunction:
        """Negate."""
        return ScalarFunction(-self._f)

    def as_sympy(self) -> SympyFormat:
        """Convert to a sympy expression."""
        return self._f

    def as_tex(self) -> str:
        """Convert to a TeX expression."""
        out = sympy.latex(sympy.simplify(sympy.expand(self._f)))
        out = out.replace("\\left[", "\\left(")
        out = out.replace("\\right]", "\\right)")
        return out

    def subs(self, vars: AxisVariables, values: ValuesToSubstitute) -> ScalarFunction:
        """Substitute values into the function."""
        subbed = self._f
        if isinstance(vars, (list, tuple)):
            assert isinstance(values, (list, tuple))
            dummy = [sympy.Symbol(f"DUMMY_{i}") for i, _ in enumerate(values)]
            for i, j in zip(vars, dummy):
                subbed = subbed.subs(i, j)
            for i, j in zip(dummy, values):
                subbed = subbed.subs(i, j)
        else:
            assert not isinstance(values, (list, tuple))
            subbed = subbed.subs(vars, values)
        return ScalarFunction(subbed)

    def diff(self, variable: sympy.core.symbol.Symbol) -> ScalarFunction:
        """Differentiate the function."""
        return ScalarFunction(self._f.diff(variable))

    def directional_derivative(self, direction: PointType) -> ScalarFunction:
        """Compute a directional derivative."""
        out = ScalarFunction(0)
        for i, j in zip(x, direction):
            out += j * self.diff(i)
        return out

    def jacobian_component(self, component: typing.Tuple[int, int]) -> ScalarFunction:
        """Compute a component of the jacobian."""
        return self.diff(x[component[0]]).diff(x[component[1]])

    def jacobian(self, dim: int) -> MatrixFunction:
        """Compute the jacobian."""
        return MatrixFunction([
            [self.jacobian_component((i, j)).as_sympy() for j in range(dim)]
            for i in range(dim)])

    def dot(self, other: AnyFunction) -> ScalarFunction:
        """Compute the dot product with another function."""
        if isinstance(other, ScalarFunction):
            return self * other

        if isinstance(other, AnyFunction) and other.is_scalar:
            return other.dot(self)

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

    def integrate(
        self, *limits: typing.Tuple[
            sympy.core.symbol.Symbol, typing.Union[int, sympy.core.expr.Expr],
            typing.Union[int, sympy.core.expr.Expr]]
    ) -> ScalarFunction:
        """Compute the integral of the function."""
        return ScalarFunction(self._f.integrate(*limits))


class VectorFunction(AnyFunction):
    """A vector-valued function."""

    _vec: tuple[ScalarFunction, ...]

    def __init__(self, vec: typing.Union[
        typing.Tuple[typing.Union[ScalarFunction, int, sympy.core.expr.Expr], ...],
        typing.List[typing.Union[ScalarFunction, int, sympy.core.expr.Expr]]
    ]):
        super().__init__(vector=True)
        self._vec = tuple(i if isinstance(i, ScalarFunction) else ScalarFunction(i) for i in vec)

    def __len__(self):
        """Get the length of the vector."""
        return len(self._vec)

    def __getitem__(self, key) -> typing.Union[ScalarFunction, VectorFunction]:
        """Get a component or slice of the vector."""
        fs = self._vec[key]
        if isinstance(fs, ScalarFunction):
            return fs
        return VectorFunction(fs)

    def __add__(self, other: typing.Any) -> VectorFunction:
        """Add."""
        if isinstance(other, VectorFunction):
            assert len(self._vec) == len(other._vec)
            return VectorFunction(tuple(i._f + j._f for i, j in zip(self._vec, other._vec)))
        if isinstance(other, (list, tuple)):
            return VectorFunction(tuple(i._f + j for i, j in zip(self._vec, other)))
        return NotImplemented

    def __radd__(self, other: typing.Any) -> VectorFunction:
        """Add."""
        if isinstance(other, VectorFunction):
            assert len(self._vec) == len(other._vec)
            return VectorFunction(tuple(i._f + j._f for i, j in zip(other._vec, self._vec)))
        if isinstance(other, (list, tuple)):
            return VectorFunction(tuple(i + j._f for i, j in zip(other, self._vec)))
        return NotImplemented

    def __sub__(self, other: typing.Any) -> VectorFunction:
        """Subtract."""
        if isinstance(other, VectorFunction):
            assert len(self._vec) == len(other._vec)
            return VectorFunction(tuple(i._f - j._f for i, j in zip(self._vec, other._vec)))
        if isinstance(other, (list, tuple)):
            return VectorFunction(tuple(i._f - j for i, j in zip(self._vec, other)))
        return NotImplemented

    def __rsub__(self, other: typing.Any) -> VectorFunction:
        """Subtract."""
        if isinstance(other, VectorFunction):
            assert len(self._vec) == len(other._vec)
            return VectorFunction(tuple(i._f - j._f for i, j in zip(other._vec, self._vec)))
        if isinstance(other, (list, tuple)):
            return VectorFunction(tuple(i - j._f for i, j in zip(other, self._vec)))
        return NotImplemented

    def __neg__(self) -> VectorFunction:
        """Negate."""
        return VectorFunction(tuple(-i._f for i in self._vec))

    def __truediv__(self, other: typing.Any) -> VectorFunction:
        """Divide."""
        if isinstance(other, ScalarFunction):
            return VectorFunction(tuple(i._f / other._f for i in self._vec))
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return VectorFunction(tuple(i._f / other for i in self._vec))
        return NotImplemented

    def __rtruediv__(self, other: typing.Any) -> VectorFunction:
        """Divide."""
        return NotImplemented

    def __mul__(self, other: typing.Any) -> VectorFunction:
        """Multiply."""
        if isinstance(other, ScalarFunction):
            return VectorFunction(tuple(i._f * other._f for i in self._vec))
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return VectorFunction(tuple(i._f * other for i in self._vec))
        return NotImplemented

    def __rmul__(self, other: typing.Any) -> VectorFunction:
        """Multiply."""
        if isinstance(other, ScalarFunction):
            return VectorFunction(tuple(other._f * i._f for i in self._vec))
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return VectorFunction(tuple(other * i._f for i in self._vec))
        if isinstance(other, MatrixFunction):
            assert other.shape[1] == len(self)
            return VectorFunction([other.row(i).dot(self) for i in range(other.shape[0])])
        return NotImplemented

    def as_sympy(self) -> SympyFormat:
        """Convert to a sympy expression."""
        return tuple(i._f for i in self._vec)

    def as_tex(self) -> str:
        """Convert to a TeX expression."""
        return "\\left(\\begin{array}{c}" + "\\\\".join([
            "\\displaystyle " + i.as_tex() for i in self._vec
        ]) + "\\end{array}\\right)"

    def subs(self, vars: AxisVariables, values: ValuesToSubstitute) -> VectorFunction:
        """Substitute values into the function."""
        subbed = tuple(i.subs(vars, values) for i in self._vec)
        return VectorFunction(subbed)

    def diff(self, variable: sympy.core.symbol.Symbol) -> VectorFunction:
        """Differentiate the function."""
        return VectorFunction([i.diff(variable) for i in self._vec])

    def directional_derivative(self, direction: PointType):
        """Compute a directional derivative."""
        raise NotImplementedError()

    def jacobian_component(self, direction: typing.Tuple[int, int]):
        """Compute a component of the jacobian."""
        raise NotImplementedError()

    def jacobian(self, dim: int) -> MatrixFunction:
        """Compute the jacobian."""
        raise NotImplementedError()

    def dot(self, other: AnyFunction) -> ScalarFunction:
        """Compute the dot product with another function."""
        if isinstance(other, VectorFunction):
            assert len(self._vec) == len(other._vec)
            out = 0
            for i, j in zip(self._vec, other._vec):
                out += i._f * j._f
            return ScalarFunction(out)

        if isinstance(other, AnyFunction) and other.is_vector:
            return other.dot(self)

        # TODO: remove
        if isinstance(other, tuple):
            assert len(self._vec) == len(other)
            out = 0
            for i, j in zip(self._vec, other):
                out += i._f * j
            return ScalarFunction(out)

        raise NotImplementedError()

    def div(self) -> ScalarFunction:
        """Compute the div of the function."""
        out = ScalarFunction(0)
        for i, j in zip(self._vec, x):
            out += i.diff(j)
        return out

    def grad(self):
        """Compute the grad of the function."""
        raise ValueError("Cannot compute the grad of a vector-valued function.")

    def curl(self) -> VectorFunction:
        """Compute the curl of the function."""
        assert len(self._vec) == 3
        return VectorFunction([
            self._vec[2].diff(x[1]) - self._vec[1].diff(x[2]),
            self._vec[0].diff(x[2]) - self._vec[2].diff(x[0]),
            self._vec[1].diff(x[0]) - self._vec[0].diff(x[1])
        ])

    def integrate(
        self, *limits: typing.Tuple[
            sympy.core.symbol.Symbol, typing.Union[int, sympy.core.expr.Expr],
            typing.Union[int, sympy.core.expr.Expr]]
    ):
        """Compute the integral of the function."""
        raise NotImplementedError()


class MatrixFunction(AnyFunction):
    """A matrix-valued function."""

    _mat: typing.Tuple[typing.Tuple[ScalarFunction, ...], ...]
    shape: typing.Tuple[int, int]

    def __init__(self, mat: typing.Union[
        typing.Tuple[typing.Tuple[typing.Union[ScalarFunction, int, sympy.core.expr.Expr],
                                  ...], ...],
        typing.Tuple[typing.List[typing.Union[ScalarFunction, int, sympy.core.expr.Expr]], ...],
        typing.List[typing.Tuple[typing.Union[ScalarFunction, int, sympy.core.expr.Expr], ...]],
        typing.List[typing.List[typing.Union[ScalarFunction, int, sympy.core.expr.Expr]]],
        sympy.matrices.dense.MutableDenseMatrix
    ]):
        super().__init__(matrix=True)
        if isinstance(mat, sympy.matrices.dense.MutableDenseMatrix):
            mat = tuple(tuple(mat[i, j] for j in range(mat.cols)) for i in range(mat.rows))
        assert isinstance(mat, (list, tuple))
        self._mat = tuple(tuple(j if isinstance(j, ScalarFunction) else ScalarFunction(j)
                                for j in i) for i in mat)
        self.shape = (len(self._mat), 0 if len(self._mat) == 0 else len(self._mat[0]))
        for i in self._mat:
            assert len(i) == self.shape[0]

    def row(self, n: int) -> VectorFunction:
        """Get a row of the matrix."""
        return VectorFunction([self._mat[n][i] for i in range(self.shape[1])])

    def col(self, n: int) -> VectorFunction:
        """Get a colunm of the matrix."""
        return VectorFunction([self._mat[i][n] for i in range(self.shape[0])])

    def __add__(self, other: typing.Any) -> MatrixFunction:
        """Add."""
        if isinstance(other, MatrixFunction):
            assert self.shape == other.shape
            return MatrixFunction(tuple(tuple(
                ii._f + jj._f for ii, jj in zip(i, j)) for i, j in zip(self._mat, other._mat)))
        if isinstance(other, (list, tuple)):
            data = []
            for i, j in zip(self._mat, other):
                assert isinstance(j, (list, tuple)) and len(j) == self.shape[1]
                data.append(tuple(ii._f + jj for ii, jj in zip(i, j)))
            return MatrixFunction(data)
        if isinstance(other, sympy.matrices.dense.MutableDenseMatrix):
            assert other.rows == self.shape[0]
            assert other.cols == self.shape[1]
            return MatrixFunction([[self._mat[i][j]._f + other[i, j]
                                    for j in range(self.shape[1])] for i in range(self.shape[0])])
        return NotImplemented

    def __radd__(self, other: typing.Any) -> MatrixFunction:
        """Add."""
        if isinstance(other, MatrixFunction):
            assert self.shape == other.shape
            return MatrixFunction(tuple(tuple(
                ii._f + jj._f for ii, jj in zip(i, j)) for i, j in zip(other._mat, self._mat)))
        if isinstance(other, (list, tuple)):
            data = []
            for i, j in zip(other, self._mat):
                assert isinstance(i, (list, tuple)) and len(i) == self.shape[1]
                data.append(tuple(ii + jj._f for ii, jj in zip(i, j)))
            return MatrixFunction(data)
        if isinstance(other, sympy.matrices.dense.MutableDenseMatrix):
            assert other.rows == self.shape[0]
            assert other.cols == self.shape[1]
            return MatrixFunction([[other[i, j] + self._mat[i][j]._f
                                    for j in range(self.shape[1])] for i in range(self.shape[0])])
        return NotImplemented

    def __sub__(self, other: typing.Any) -> MatrixFunction:
        """Subtract."""
        if isinstance(other, MatrixFunction):
            assert self.shape == other.shape
            return MatrixFunction(tuple(tuple(
                ii._f - jj._f for ii, jj in zip(i, j)) for i, j in zip(self._mat, other._mat)))
        if isinstance(other, (list, tuple)):
            data = []
            for i, j in zip(self._mat, other):
                assert isinstance(j, (list, tuple)) and len(j) == self.shape[1]
                data.append(tuple(ii._f - jj for ii, jj in zip(i, j)))
            return MatrixFunction(data)
        if isinstance(other, sympy.matrices.dense.MutableDenseMatrix):
            assert other.rows == self.shape[0]
            assert other.cols == self.shape[1]
            return MatrixFunction([[self._mat[i][j]._f - other[i, j]
                                    for j in range(self.shape[1])] for i in range(self.shape[0])])
        return NotImplemented

    def __rsub__(self, other: typing.Any) -> MatrixFunction:
        """Subtract."""
        if isinstance(other, MatrixFunction):
            assert self.shape == other.shape
            return MatrixFunction(tuple(tuple(
                ii._f - jj._f for ii, jj in zip(i, j)) for i, j in zip(other._mat, self._mat)))
        if isinstance(other, (list, tuple)):
            data = []
            for i, j in zip(other, self._mat):
                assert isinstance(i, (list, tuple)) and len(i) == self.shape[1]
                data.append(tuple(ii - jj._f for ii, jj in zip(i, j)))
            return MatrixFunction(data)
        if isinstance(other, sympy.matrices.dense.MutableDenseMatrix):
            assert other.rows == self.shape[0]
            assert other.cols == self.shape[1]
            return MatrixFunction([[other[i, j] - self._mat[i][j]._f
                                    for j in range(self.shape[1])] for i in range(self.shape[0])])
        return NotImplemented

    def __neg__(self) -> MatrixFunction:
        """Negate."""
        return MatrixFunction(tuple(tuple(-j._f for j in i) for i in self._mat))

    def __truediv__(self, other: typing.Any) -> MatrixFunction:
        """Divide."""
        if isinstance(other, ScalarFunction):
            return MatrixFunction(tuple(tuple(j._f / other._f for j in i) for i in self._mat))
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return MatrixFunction(tuple(tuple(j._f / other for j in i) for i in self._mat))
        return NotImplemented

    def __rtruediv__(self, other: typing.Any) -> MatrixFunction:
        """Divide."""
        return NotImplemented

    def __mul__(self, other: typing.Any) -> MatrixFunction:
        """Multiply."""
        if isinstance(other, ScalarFunction):
            return MatrixFunction(tuple(tuple(j._f * other._f for j in i) for i in self._mat))
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return MatrixFunction(tuple(tuple(j._f * other for j in i) for i in self._mat))
        return NotImplemented

    def __rmul__(self, other: typing.Any) -> MatrixFunction:
        """Multiply."""
        if isinstance(other, ScalarFunction):
            return MatrixFunction(tuple(tuple(other._f * j._f for j in i) for i in self._mat))
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return MatrixFunction(tuple(tuple(other * j._f for j in i) for i in self._mat))
        return NotImplemented

    def as_sympy(self) -> SympyFormat:
        """Convert to a sympy expression."""
        return sympy.Matrix([[j._f for j in i] for i in self._mat])

    def as_tex(self) -> str:
        """Convert to a TeX expression."""
        out = "\\left(\\begin{array}{"
        out += "c" * self.shape[1]
        out += "}"
        out += "\\\\".join([
            "&".join(["\\displaystyle" + j.as_tex() for j in i])
            for i in self._mat
        ])
        out += "\\end{array}\\right)"
        return out

    def subs(self, vars: AxisVariables, values: ValuesToSubstitute) -> MatrixFunction:
        """Substitute values into the function."""
        subbed = tuple(tuple(j.subs(vars, values) for j in i) for i in self._mat)
        return MatrixFunction(subbed)

    def diff(self, variable: sympy.core.symbol.Symbol) -> MatrixFunction:
        """Differentiate the function."""
        return MatrixFunction([
            [self._mat[i][j].diff(variable) for j in range(self.shape[1])]
            for i in range(self.shape[0])])

    def directional_derivative(self, direction: PointType):
        """Compute a directional derivative."""
        raise NotImplementedError()

    def jacobian_component(self, direction: typing.Tuple[int, int]):
        """Compute a component of the jacobian."""
        raise NotImplementedError()

    def jacobian(self, dim: int):
        """Compute the jacobian."""
        raise NotImplementedError()

    def dot(self, other: AnyFunction) -> ScalarFunction:
        """Compute the dot product with another function."""
        if isinstance(other, MatrixFunction):
            assert self.shape == other.shape
            out = ScalarFunction(0)
            for i in range(self.shape[0]):
                for j in range(self.shape[0]):
                    out += self._mat[i][j] * other._mat[i][j]
            return out

        if isinstance(other, AnyFunction) and other.is_matrix:
            return other.dot(self)

        raise NotImplementedError()

    def div(self):
        """Compute the div of the function."""
        raise ValueError("Cannot compute the div of a matrix-valued function.")

    def grad(self):
        """Compute the grad of the function."""
        raise ValueError("Cannot compute the grad of a matrix-valued function.")

    def curl(self):
        """Compute the curl of the function."""
        raise ValueError("Cannot compute the curl of a matrix-valued function.")

    def integrate(
        self, *limits: typing.Tuple[
            sympy.core.symbol.Symbol, typing.Union[int, sympy.core.expr.Expr],
            typing.Union[int, sympy.core.expr.Expr]]
    ):
        """Compute the integral of the function."""
        raise NotImplementedError()


FunctionInput = typing.Union[
    AnyFunction,
    sympy.core.expr.Expr, int,
    typing.Tuple[typing.Union[sympy.core.expr.Expr, int], ...],
    typing.List[typing.Union[sympy.core.expr.Expr, int]],
    typing.Tuple[typing.Tuple[typing.Union[sympy.core.expr.Expr, int], ...], ...],
    typing.Tuple[typing.List[typing.Union[sympy.core.expr.Expr, int]], ...],
    typing.List[typing.Tuple[typing.Union[sympy.core.expr.Expr, int], ...]],
    typing.List[typing.List[typing.Union[sympy.core.expr.Expr, int]]],
    sympy.matrices.dense.MutableDenseMatrix]


def parse_function_input(f: FunctionInput) -> AnyFunction:
    """Parse a function."""
    if isinstance(f, AnyFunction):
        return f
    if isinstance(f, (sympy.core.expr.Expr, int)):
        return ScalarFunction(f)
    if isinstance(f, (list, tuple)):
        if isinstance(f[0], (list, tuple)):
            return MatrixFunction(f)
        else:
            return VectorFunction(f)
    raise ValueError(f"Could not parse input function: {f}")


def parse_function_list_input(
    functions: typing.Union[typing.List[FunctionInput], typing.Tuple[FunctionInput, ...]]
) -> typing.List[AnyFunction]:
    """Parse a list of functions."""
    return [parse_function_input(f) for f in functions]
