"""Basis function classes."""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod

import sympy

from .geometry import PointType
from .references import Reference
from .symbols import AxisVariables, AxisVariablesNotSingle, t, x

SingleSympyFormat = typing.Union[
    sympy.core.expr.Expr,
    typing.Tuple[sympy.core.expr.Expr, ...],
    sympy.matrices.dense.MutableDenseMatrix
]
SympyFormat = typing.Union[
    SingleSympyFormat,
    typing.Dict[typing.Tuple[typing.Tuple[sympy.core.expr.Expr, ...], ...], SingleSympyFormat]
]
_ValuesToSubstitute = typing.Union[
    typing.Tuple[typing.Any, ...],
    typing.List[typing.Any],
    typing.Any]


def _to_sympy_format(item: typing.Any) -> SympyFormat:
    """Convert to Sympy format used by these functions.

    Args:
        item: The input item

    Returns:
        The item in Sympy format expected by functions
    """
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
    """Check if two items are equal.

    Args:
        first: The first item
        second: The second item

    Returns:
        Are the two items equal?
    """
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
        for ii in range(first.rows):
            for jj in range(first.cols):
                if not _check_equal(first[ii, jj], second[ii, jj]):
                    return False
        return True

    return False


class AnyFunction(ABC):
    """A function."""

    def __init__(self, scalar: bool = False, vector: bool = False, matrix: bool = False):
        """Create a function.

        Args:
            scalar: Is the function scalar-valued?
            vector: Is the function vector-valued?
            matrix: Is the function matrix-valued?
        """
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
    def __matmul__(self, other: typing.Any):
        """Multiply."""
        pass

    @abstractmethod
    def __rmatmul__(self, other: typing.Any):
        """Multiply."""
        pass

    @abstractmethod
    def __pow__(self, other: typing.Any):
        """Raise to a power."""
        pass

    @abstractmethod
    def as_sympy(self) -> SympyFormat:
        """Convert to a Sympy expression.

        Returns:
            A Sympy expression
        """
        pass

    @abstractmethod
    def as_tex(self) -> str:
        """Convert to a TeX expression.

        Returns:
            A TeX string
        """
        pass

    @abstractmethod
    def subs(self, vars: AxisVariables, values: typing.Union[AnyFunction, _ValuesToSubstitute]):
        """Substitute values into the function.

        Args:
            vars: The variables to substitute out
            values: The value to substitute in

        Returns:
            The substituted function
        """
        pass

    @abstractmethod
    def diff(self, variable: sympy.core.symbol.Symbol):
        """Differentiate the function.

        Args:
            variable: The variable to differentiate with respect to

        Returns:
            The differentiated function
        """
        pass

    @abstractmethod
    def directional_derivative(self, direction: PointType):
        """Compute a directional derivative.

        Args:
            direction: The diection

        Returns:
            The directional differentiate
        """
        pass

    @abstractmethod
    def jacobian_component(self, component: typing.Tuple[int, int]):
        """Compute a component of the jacobian.

        Args:
            component: The component

        Returns:
            The component of the jacobian
        """
        pass

    @abstractmethod
    def jacobian(self, dim: int):
        """Compute the jacobian.

        Args:
            dim: The topological dimension of the cell

        Returns:
            The jacobian
        """
        pass

    @abstractmethod
    def dot(self, other_in: FunctionInput):
        """Compute the dot product with another function.

        Args:
            other_in: The function to multiply with

        Returns:
            The product
        """
        pass

    @abstractmethod
    def cross(self, other_in: FunctionInput):
        """Compute the cross product with another function.

        Args:
            other_in: The function to multiply with

        Returns:
            The cross product
        """
        pass

    @abstractmethod
    def div(self):
        """Compute the div of the function.

        Returns:
            The divergence
        """
        pass

    @abstractmethod
    def grad(self, dim: int):
        """Compute the grad of the function.

        Returns:
            The gradient
        """
        pass

    @abstractmethod
    def curl(self):
        """Compute the curl of the function.

        Returns:
            The curl
        """
        pass

    @abstractmethod
    def norm(self):
        """Compute the norm of the function.

        Returns:
            The norm
        """
        pass

    @abstractmethod
    def integral(self, domain: Reference, vars: AxisVariablesNotSingle = t):
        """Compute the integral of the function.

        Args:
            domain: The domain of the integral
            vars: The variables to integrate with respect to

        Returns:
            The integral
        """
        pass

    @abstractmethod
    def with_floats(self) -> AnyFunction:
        """Return a version the function with floats as coefficients.

        Returns:
            A version the function with floats as coefficients
        """
        pass

    def integrate(self, *limits: typing.Tuple[
        sympy.core.symbol.Symbol,
        typing.Union[int, sympy.core.expr.Expr],
        typing.Union[int, sympy.core.expr.Expr]
    ]):
        """Integrate the function.

        Args:
            limits: The variables and limits

        Returns:
            The integral
        """
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'integrate'")

    def det(self):
        """Compute the determinant.

        Returns:
            The deteminant
        """
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'det'")

    def transpose(self):
        """Compute the transpose.

        Returns:
            The transpose
        """
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'transpose'")

    @property
    def shape(self) -> typing.Tuple[int, ...]:
        """Get the value shape of the function.

        Returns:
            The value shape
        """
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'shape'")

    def plot(
        self, reference: Reference, filename: typing.Union[str, typing.List[str]],
        dof_point: typing.Optional[PointType] = None,
        dof_direction: typing.Optional[PointType] = None,
        dof_entity: typing.Optional[typing.Tuple[int, int]] = None,
        dof_n: typing.Optional[int] = None, value_scale: sympy.core.expr.Expr = sympy.Integer(1),
        plot_options: typing.Dict[str, typing.Any] = {}, **kwargs: typing.Any
    ):
        """Plot the function.

        Args:
            reference: The reference cell
            filename: The file name
            dof_point: The DOF point
            dof_direction: The direction of the DOF
            dof_entity: The entity the DOF is associated with
            dof_n: The number of the DOF
            value_scale: The scale factor for the function values
            plot_options: Options for the plot
            kwargs: Keyword arguments
        """
        from .plotting import Picture, colors

        extra: typing.Tuple[int, ...] = tuple()
        if self.is_scalar:
            extra = (0, )

        img = Picture(**kwargs)

        if dof_entity is not None and dof_entity[0] > 1:
            sub_e = reference.sub_entity(*dof_entity)
            img.add_fill([i + extra for i in sub_e.clockwise_vertices], colors.BLUE, 0.5)

        for ze in reference.z_ordered_entities_extra_dim():
            for dim, entity in ze:
                if dim == 1:
                    c = colors.GRAY
                    if dof_entity == (1, entity):
                        c = colors.BLUE
                    img.add_line(
                        reference.vertices[reference.edges[entity][0]] + extra,
                        reference.vertices[reference.edges[entity][1]] + extra, c)

                if dim == reference.tdim:
                    self.plot_values(reference, img, value_scale)

            if (dim, entity) in ze:
                if dof_direction is not None:
                    assert dof_point is not None and dof_n is not None
                    img.add_dof_arrow(dof_point + extra, dof_direction + extra, dof_n,
                                      colors.PURPLE, bold=False)
                elif dof_point is not None:
                    assert dof_n is not None
                    img.add_dof_marker(dof_point + extra, dof_n, colors.PURPLE, bold=False)

        img.save(filename, plot_options=plot_options)

    def plot_values(
        self, reference: Reference, img: typing.Any,
        value_scale: sympy.core.expr.Expr = sympy.Integer(1), n: int = 6
    ):
        """Plot the function's values.

        Args:
            reference: The reference cell
            img: The image to plot on
            value_scale: The scale factor for the function values
            n: The number of points per side for plotting
        """
        raise ValueError(f"Cannot plot function of type '{self.__class__.__name__}'")

    def __len__(self):
        """Compute the determinant."""
        raise TypeError(f"object of type '{self.__class__.__name__}' has no len()")

    def __getitem__(self, key) -> AnyFunction:
        """Get a component or slice of the function."""
        raise ValueError(f"'{self.__class__.__name__}' object is not subscriptable")

    def _sympy_(self) -> SympyFormat:
        """Convert to Sympy format."""
        return self.as_sympy()

    def __float__(self) -> float:
        """Convert to a float."""
        if self.is_scalar:
            a_s = self.as_sympy()
            assert isinstance(a_s, sympy.core.expr.Expr)
            return float(a_s)
        raise TypeError("Cannot convert function to a float.")

    def __lt__(self, other: typing.Any) -> bool:
        """Check inequality."""
        return self.as_sympy() < other

    def __le__(self, other: typing.Any) -> bool:
        """Check inequality."""
        return self.as_sympy() <= other

    def __gt__(self, other: typing.Any) -> bool:
        """Check inequality."""
        return self.as_sympy() > other

    def __ge__(self, other: typing.Any) -> bool:
        """Check inequality."""
        return self.as_sympy() >= other

    def __repr__(self) -> str:
        """Representation."""
        return self.as_sympy().__repr__()

    def __eq__(self, other: typing.Any) -> bool:
        """Check if two functions are equal."""
        return _check_equal(_to_sympy_format(self), _to_sympy_format(other))

    def __ne__(self, other: typing.Any) -> bool:
        """Check if two functions are not equal."""
        return not self.__eq__(other)


ValuesToSubstitute = typing.Union[AnyFunction, _ValuesToSubstitute]


class ScalarFunction(AnyFunction):
    """A scalar-valued function."""

    _f: sympy.core.expr.Expr

    def __init__(self, f: typing.Union[int, sympy.core.expr.Expr]):
        """Create a scalar-valued function.

        Args:
            f: The sympy representation of the function.
        """
        super().__init__(scalar=True)
        if isinstance(f, int):
            self._f = sympy.Integer(f)
        else:
            self._f = f
        assert isinstance(self._f, sympy.core.expr.Expr)
        self._plot_beziers: typing.Dict[typing.Tuple[Reference, int], typing.List[
            typing.Tuple[PointType, PointType, PointType, PointType]]] = {}

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

    def __matmul__(self, other: typing.Any):
        """Multiply."""
        return NotImplemented

    def __rmatmul__(self, other: typing.Any):
        """Multiply."""
        return NotImplemented

    def __pow__(self, other: typing.Any) -> ScalarFunction:
        """Raise to a power."""
        return ScalarFunction(self._f ** other)

    def __neg__(self) -> ScalarFunction:
        """Negate."""
        return ScalarFunction(-self._f)

    def as_sympy(self) -> SympyFormat:
        """Convert to a sympy expression.

        Returns:
            A Sympy expression
        """
        return self._f

    def as_tex(self) -> str:
        """Convert to a TeX expression.

        Returns:
            A TeX string
        """
        out = sympy.latex(sympy.simplify(sympy.expand(self._f)))
        out = out.replace("\\left[", "\\left(")
        out = out.replace("\\right]", "\\right)")
        return out

    def subs(self, vars: AxisVariables, values: ValuesToSubstitute) -> ScalarFunction:
        """Substitute values into the function.

        Args:
            vars: The variables to substitute out
            values: The value to substitute in

        Returns:
            The substituted function
        """
        subbed = self._f
        if isinstance(values, AnyFunction):
            values = values.as_sympy()
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
        """Differentiate the function.

        Args:
            variable: The variable to differentiate with respect to

        Returns:
            The differentiated function
        """
        return ScalarFunction(self._f.diff(variable))

    def directional_derivative(self, direction: PointType) -> ScalarFunction:
        """Compute a directional derivative.

        Args:
            direction: The diection

        Returns:
            The directional derivatve
        """
        out = ScalarFunction(0)
        for i, j in zip(x, direction):
            out += j * self.diff(i)
        return out

    def jacobian_component(self, component: typing.Tuple[int, int]) -> ScalarFunction:
        """Compute a component of the jacobian.

        Args:
            component: The component

        Returns:
            The component of the jacobian
        """
        return self.diff(x[component[0]]).diff(x[component[1]])

    def jacobian(self, dim: int) -> MatrixFunction:
        """Compute the jacobian.

        Args:
            dim: The topological dimension of the cell

        Returns:
            The jacobian
        """
        return MatrixFunction(tuple(
            tuple(self.jacobian_component((i, j)) for j in range(dim))
            for i in range(dim)))

    def dot(self, other_in: FunctionInput) -> ScalarFunction:
        """Compute the dot product with another function.

        Args:
            other_in: The function to multiply with

        Returns:
            The product
        """
        other = parse_function_input(other_in)
        if isinstance(other, ScalarFunction):
            return self * other

        if isinstance(other, AnyFunction) and other.is_scalar:
            return other.dot(self)

        raise NotImplementedError()

    def cross(self, other_in: FunctionInput):
        """Compute the cross product with another function.

        Args:
            other_in: The function to multiply with

        Returns:
            The cross product
        """
        raise ValueError("Cannot cross a scalar-valued function.")

    def div(self):
        """Compute the div of the function.

        Returns:
            The divergence
        """
        raise ValueError("Cannot compute the div of a scalar-valued function.")

    def grad(self, dim: int) -> VectorFunction:
        """Compute the grad of the function.

        Returns:
            The gradient
        """
        return VectorFunction([self.diff(x[i]) for i in range(dim)])

    def curl(self):
        """Compute the curl of the function.

        Returns:
            The curl
        """
        raise ValueError("Cannot compute the curl of a scalar-valued function.")

    def norm(self) -> ScalarFunction:
        """Compute the norm of the function.

        Returns:
            The norm
        """
        return ScalarFunction(abs(self._f))

    def integral(self, domain: Reference, vars: AxisVariablesNotSingle = t) -> ScalarFunction:
        """Compute the integral of the function.

        Args:
            domain: The domain of the integral
            vars: The variables to integrate with respect to

        Returns:
            The integral
        """
        limits = domain.integration_limits(vars)

        point = VectorFunction(domain.origin)
        for ti, a in zip(t, domain.axes):
            point += ti * VectorFunction(a)
        out = self._f.subs(x, point)

        if len(limits[0]) == 2:
            for i in limits:
                assert len(i) == 2
                out = out.subs(*i)
            return out

        out *= domain.jacobian()
        return ScalarFunction(out.integrate(*limits))

    def integrate(self, *limits: typing.Tuple[
        sympy.core.symbol.Symbol,
        typing.Union[int, sympy.core.expr.Expr],
        typing.Union[int, sympy.core.expr.Expr]
    ]):
        """Integrate the function.

        Args:
            limits: The variables and limits

        Returns:
            The integral
        """
        return ScalarFunction(self._f.integrate(*limits))

    def plot_values(
        self, reference: Reference, img: typing.Any,
        value_scale: sympy.core.expr.Expr = sympy.Integer(1), n: int = 6
    ):
        """Plot the function's values.

        Args:
            reference: The reference cell
            img: The image to plot on
            value_scale: The scale factor for the function values
            n: The number of points per side for plotting
        """
        from .plotting import Picture, colors
        assert isinstance(img, Picture)

        if (reference, n) not in self._plot_beziers:
            self._plot_beziers[(reference, n)] = []
            pts, pairs = reference.make_lattice_with_lines_float(n)

            value_scale *= sympy.Rational(5, 8)
            value_scale = sympy.Float(float(value_scale))

            deriv = self.grad(reference.tdim)
            evals = []
            for p in pts:
                value = self.subs(x, p).as_sympy()
                assert isinstance(value, sympy.core.expr.Expr)
                value = sympy.Float(float(value))
                value *= value_scale
                evals.append(value)

            for i, j in pairs:
                pi = VectorFunction(pts[i])
                pj = VectorFunction(pts[j])
                d_pi = (2 * pi + pj) / 3
                d_pj = (2 * pj + pi) / 3
                di = deriv.subs(x, pi).dot(d_pi - pts[i]).as_sympy()
                dj = deriv.subs(x, pj).dot(d_pj - pts[j]).as_sympy()
                assert isinstance(di, sympy.core.expr.Expr)
                assert isinstance(dj, sympy.core.expr.Expr)
                self._plot_beziers[(reference, n)].append((
                    tuple(pi) + (evals[i], ), tuple(d_pi) + (evals[i] + di * value_scale, ),
                    tuple(d_pj) + (evals[j] + dj * value_scale, ), tuple(pj) + (evals[j], )))

        for s, m1, m2, e in self._plot_beziers[(reference, n)]:
            img.add_bezier(s, m1, m2, e, colors.ORANGE)

    def with_floats(self) -> AnyFunction:
        """Return a version the function with floats as coefficients.

        Returns:
            A version the function with floats as coefficients
        """
        out = sympy.Float(0.0)
        for term, co in self._f.as_coefficients_dict().items():
            out += float(co) * term
        return ScalarFunction(out)


class VectorFunction(AnyFunction):
    """A vector-valued function."""

    _vec: tuple[ScalarFunction, ...]

    def __init__(self, vec: typing.Union[
        typing.Tuple[typing.Union[AnyFunction, int, sympy.core.expr.Expr], ...],
        typing.List[typing.Union[AnyFunction, int, sympy.core.expr.Expr]]
    ]):
        """Create a vector-valued function.

        Args:
            vec: The sympy representation of the function.
        """
        from .basis_functions import BasisFunction
        super().__init__(vector=True)
        vec_l = []
        for i in vec:
            if isinstance(i, AnyFunction):
                if isinstance(i, BasisFunction):
                    i = i.get_function()
                assert isinstance(i, ScalarFunction)
                vec_l.append(i)
            else:
                vec_l.append(ScalarFunction(i))
        self._vec = tuple(vec_l)
        for i in self._vec:
            assert i.is_scalar

        self._plot_arrows: typing.Dict[typing.Tuple[Reference, int], typing.List[
            typing.Tuple[typing.Tuple[sympy.core.expr.Expr, ...], VectorFunction, float]]] = {}

    def __len__(self):
        """Get the length of the vector."""
        return len(self._vec)

    @property
    def shape(self) -> typing.Tuple[int, ...]:
        """Get the value shape of the function.

        Returns:
            The value shape
        """
        return (len(self), )

    def __getitem__(self, key) -> typing.Union[ScalarFunction, VectorFunction]:
        """Get a component or slice of the function."""
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
        return NotImplemented

    def __matmul__(self, other: typing.Any) -> VectorFunction:
        """Multiply."""
        if isinstance(other, MatrixFunction):
            assert other.shape[1] == len(self)
            return VectorFunction([other.col(i).dot(self) for i in range(other.shape[0])])
        return NotImplemented

    def __rmatmul__(self, other: typing.Any) -> VectorFunction:
        """Multiply."""
        if isinstance(other, MatrixFunction):
            assert other.shape[1] == len(self)
            return VectorFunction([other.row(i).dot(self) for i in range(other.shape[0])])
        return NotImplemented

    def __pow__(self, other: typing.Any) -> VectorFunction:
        """Raise to a power."""
        if isinstance(other, ScalarFunction):
            return VectorFunction(tuple(i._f ** other._f for i in self._vec))
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return VectorFunction(tuple(i._f ** other for i in self._vec))
        return NotImplemented

    def as_sympy(self) -> SympyFormat:
        """Convert to a sympy expression.

        Returns:
            A Sympy expression
        """
        return tuple(i._f for i in self._vec)

    def as_tex(self) -> str:
        """Convert to a TeX expression.

        Returns:
            A TeX string
        """
        return "\\left(\\begin{array}{c}" + "\\\\".join([
            "\\displaystyle " + i.as_tex() for i in self._vec
        ]) + "\\end{array}\\right)"

    def subs(self, vars: AxisVariables, values: ValuesToSubstitute) -> VectorFunction:
        """Substitute values into the function.

        Args:
            vars: The variables to substitute out
            values: The value to substitute in

        Returns:
            The substituted function
        """
        subbed = tuple(i.subs(vars, values) for i in self._vec)
        return VectorFunction(subbed)

    def diff(self, variable: sympy.core.symbol.Symbol) -> VectorFunction:
        """Differentiate the function.

        Args:
            variable: The variable to differentiate with respect to

        Returns:
            The differentiated function
        """
        return VectorFunction([i.diff(variable) for i in self._vec])

    def directional_derivative(self, direction: PointType):
        """Compute a directional derivative.

        Args:
            direction: The diection

        Returns:
            The directional derivatve
        """
        raise NotImplementedError()

    def jacobian_component(self, component: typing.Tuple[int, int]):
        """Compute a component of the jacobian.

        Args:
            component: The component

        Returns:
            The component of the jacobian
        """
        raise NotImplementedError()

    def jacobian(self, dim: int) -> MatrixFunction:
        """Compute the jacobian.

        Args:
            dim: The topological dimension of the cell

        Returns:
            The jacobian
        """
        raise NotImplementedError()

    def dot(self, other_in: FunctionInput) -> ScalarFunction:
        """Compute the dot product with another function.

        Args:
            other_in: The function to multiply with

        Returns:
            The product
        """
        other = parse_function_input(other_in)
        if isinstance(other, VectorFunction):
            assert len(self._vec) == len(other._vec)
            out = 0
            for i, j in zip(self._vec, other._vec):
                out += i._f * j._f
            return ScalarFunction(out)

        if isinstance(other, AnyFunction) and other.is_vector:
            return other.dot(self)

        raise NotImplementedError()

    def cross(self, other_in: FunctionInput) -> typing.Union[VectorFunction, ScalarFunction]:
        """Compute the cross product with another function.

        Args:
            other_in: The function to multiply with

        Returns:
            The cross product
        """
        other = parse_function_input(other_in)
        assert other.is_vector and len(self) == len(other)
        if len(self) == 2:
            return self[0] * other[1] - self[1] * other[0]
        else:
            assert len(self) == 3
            return VectorFunction([self[1] * other[2] - self[2] * other[1],
                                   self[2] * other[0] - self[0] * other[2],
                                   self[0] * other[1] - self[1] * other[0]])

    def div(self) -> ScalarFunction:
        """Compute the div of the function.

        Returns:
            The divergence
        """
        out = ScalarFunction(0)
        for i, j in zip(self._vec, x):
            out += i.diff(j)
        return out

    def grad(self):
        """Compute the grad of the function.

        Returns:
            The gradient
        """
        raise ValueError("Cannot compute the grad of a vector-valued function.")

    def curl(self) -> VectorFunction:
        """Compute the curl of the function.

        Returns:
            The curl
        """
        assert len(self._vec) == 3
        return VectorFunction([
            self._vec[2].diff(x[1]) - self._vec[1].diff(x[2]),
            self._vec[0].diff(x[2]) - self._vec[2].diff(x[0]),
            self._vec[1].diff(x[0]) - self._vec[0].diff(x[1])
        ])

    def norm(self) -> ScalarFunction:
        """Compute the norm of the function.

        Returns:
            The norm
        """
        a = sympy.Integer(0)
        for i in self._vec:
            a += i._f ** 2
        return ScalarFunction(sympy.sqrt(a))

    def integral(self, domain: Reference, vars: AxisVariablesNotSingle = t):
        """Compute the integral of the function.

        Args:
            domain: The domain of the integral
            vars: The variables to integrate with respect to

        Returns:
            The integral
        """
        raise NotImplementedError()

    def __iter__(self):
        """Get iterable."""
        self.iter_n = 0
        return self

    def __next__(self):
        """Get next item."""
        if self.iter_n < len(self._vec):
            self.iter_n += 1
            return self._vec[self.iter_n - 1]
        else:
            raise StopIteration

    def plot_values(
        self, reference: Reference, img: typing.Any,
        value_scale: sympy.core.expr.Expr = sympy.Integer(1), n: int = 6
    ):
        """Plot the function's values.

        Args:
            reference: The reference cell
            img: The image to plot on
            value_scale: The scale factor for the function values
            n: The number of points per side for plotting
        """
        from .plotting import Picture, colors
        assert isinstance(img, Picture)

        if (reference, n) not in self._plot_arrows:
            self._plot_arrows[(reference, n)] = []
            pts = reference.make_lattice_float(n)
            value_scale /= 4

            for p in pts:
                value_s = self.subs(x, p).as_sympy()
                assert isinstance(value_s, tuple)
                value = VectorFunction(tuple(sympy.Float(float(v)) for v in value_s))
                value *= value_scale
                size = float(value.norm() * 40)
                self._plot_arrows[(reference, n)].append((p, VectorFunction(p) + value, size))
        for p, q, size in self._plot_arrows[(reference, n)]:
            img.add_arrow(p, q, colors.ORANGE, size)

    def with_floats(self) -> AnyFunction:
        """Return a version the function with floats as coefficients.

        Returns:
            A version the function with floats as coefficients
        """
        return VectorFunction(tuple(f.with_floats() for f in self._vec))


class MatrixFunction(AnyFunction):
    """A matrix-valued function."""

    _mat: typing.Tuple[typing.Tuple[ScalarFunction, ...], ...]

    def __init__(self, mat: typing.Union[
        typing.Tuple[typing.Tuple[typing.Union[AnyFunction, int, sympy.core.expr.Expr],
                                  ...], ...],
        typing.Tuple[typing.List[typing.Union[AnyFunction, int, sympy.core.expr.Expr]], ...],
        typing.List[typing.Tuple[typing.Union[AnyFunction, int, sympy.core.expr.Expr], ...]],
        typing.List[typing.List[typing.Union[AnyFunction, int, sympy.core.expr.Expr]]],
        sympy.matrices.dense.MutableDenseMatrix
    ]):
        """Create a matrix-valued function.

        Args:
            mat: The sympy representation of the function.
        """
        from .basis_functions import BasisFunction
        super().__init__(matrix=True)
        if isinstance(mat, sympy.matrices.dense.MutableDenseMatrix):
            mat = tuple(tuple(mat[i, j] for j in range(mat.cols)) for i in range(mat.rows))
        assert isinstance(mat, (list, tuple))
        mat_l = []
        for i in mat:
            row = []
            for j in i:
                if isinstance(j, AnyFunction):
                    if isinstance(j, BasisFunction):
                        j = j.get_function()
                    assert isinstance(j, ScalarFunction)
                    row.append(j)
                else:
                    row.append(ScalarFunction(j))
            mat_l.append(tuple(row))

        self._mat = tuple(mat_l)
        self._shape = (len(self._mat), 0 if len(self._mat) == 0 else len(self._mat[0]))
        for i in self._mat:
            assert len(i) == self._shape[1]
            for j in i:
                assert j.is_scalar

    @property
    def shape(self) -> typing.Tuple[int, ...]:
        """Get the value shape of the function.

        Returns:
            The value shape
        """
        return self._shape

    def __getitem__(self, key) -> typing.Union[ScalarFunction, VectorFunction]:
        """Get a component or slice of the function."""
        if isinstance(key, tuple):
            assert len(key) == 2
            return self._mat[key[0]][key[1]]
        return self.row(key)

    def row(self, n: int) -> VectorFunction:
        """Get a row of the matrix.

        Args:
            n: The row number

        Returns:
            The row of the matrix
        """
        return VectorFunction([self._mat[n][i] for i in range(self.shape[1])])

    def col(self, n: int) -> VectorFunction:
        """Get a colunm of the matrix.

        Args:
            n: The column number

        Returns:
            The column of the matrix
        """
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

    def __matmul__(self, other: typing.Any) -> MatrixFunction:
        """Multiply."""
        if isinstance(other, MatrixFunction):
            assert other.shape[0] == self.shape[1]
            return MatrixFunction(tuple(
                tuple(self.row(i).dot(other.col(j)) for j in range(other.shape[1]))
                for i in range(self.shape[0])))
        return NotImplemented

    def __rmatmul__(self, other: typing.Any) -> MatrixFunction:
        """Multiply."""
        if isinstance(other, MatrixFunction):
            assert self.shape[0] == other.shape[1]
            return MatrixFunction(tuple(
                tuple(other.row(i).dot(self.col(j)) for j in range(self.shape[1]))
                for i in range(other.shape[0])))
        return NotImplemented

    def __pow__(self, other: typing.Any) -> MatrixFunction:
        """Raise to a power."""
        if isinstance(other, ScalarFunction):
            return MatrixFunction(tuple(tuple(j._f ** other._f for j in i) for i in self._mat))
        if isinstance(other, (int, sympy.core.expr.Expr)):
            return MatrixFunction(tuple(tuple(j._f ** other for j in i) for i in self._mat))
        return NotImplemented

    def as_sympy(self) -> SympyFormat:
        """Convert to a sympy expression.

        Returns:
            A Sympy expression
        """
        return sympy.Matrix([[j._f for j in i] for i in self._mat])

    def as_tex(self) -> str:
        """Convert to a TeX expression.

        Returns:
            A TeX string
        """
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
        """Substitute values into the function.

        Args:
            vars: The variables to substitute out
            values: The value to substitute in

        Returns:
            The substituted function
        """
        subbed = tuple(tuple(j.subs(vars, values) for j in i) for i in self._mat)
        return MatrixFunction(subbed)

    def diff(self, variable: sympy.core.symbol.Symbol) -> MatrixFunction:
        """Differentiate the function.

        Args:
            variable: The variable to differentiate with respect to

        Returns:
            The differentiated function
        """
        return MatrixFunction(tuple(
            tuple(self._mat[i][j].diff(variable) for j in range(self.shape[1]))
            for i in range(self.shape[0])))

    def directional_derivative(self, direction: PointType):
        """Compute a directional derivative.

        Args:
            direction: The diection

        Returns:
            The directional derivatve
        """
        raise NotImplementedError()

    def jacobian_component(self, component: typing.Tuple[int, int]):
        """Compute a component of the jacobian.

        Args:
            component: The component

        Returns:
            The component of the jacobian
        """
        raise NotImplementedError()

    def jacobian(self, dim: int):
        """Compute the jacobian.

        Args:
            dim: The topological dimension of the cell

        Returns:
            The jacobian
        """
        raise NotImplementedError()

    def dot(self, other_in: FunctionInput) -> ScalarFunction:
        """Compute the dot product with another function.

        Args:
            other_in: The function to multiply with

        Returns:
            The product
        """
        other = parse_function_input(other_in)
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

    def cross(self, other_in: FunctionInput):
        """Compute the cross product with another function.

        Args:
            other_in: The function to multiply with

        Returns:
            The cross product
        """
        raise ValueError("Cannot cross a matrix-valued function.")

    def div(self):
        """Compute the div of the function.

        Returns:
            The divergence
        """
        raise ValueError("Cannot compute the div of a matrix-valued function.")

    def grad(self):
        """Compute the grad of the function.

        Returns:
            The gradient
        """
        raise ValueError("Cannot compute the grad of a matrix-valued function.")

    def curl(self):
        """Compute the curl of the function.

        Returns:
            The curl
        """
        raise ValueError("Cannot compute the curl of a matrix-valued function.")

    def norm(self) -> ScalarFunction:
        """Compute the norm of the function.

        Returns:
            The norm
        """
        raise NotImplementedError()

    def integral(self, domain: Reference, vars: AxisVariablesNotSingle = t):
        """Compute the integral of the function.

        Args:
            domain: The domain of the integral
            vars: The variables to integrate with respect to

        Returns:
            The integral
        """
        raise NotImplementedError()

    def det(self) -> ScalarFunction:
        """Compute the determinant.

        Returns:
            The deteminant
        """
        if self.shape[0] == self.shape[1]:
            mat = self.as_sympy()
            assert isinstance(mat, sympy.matrices.dense.MutableDenseMatrix)
            return ScalarFunction(mat.det())
        if self.shape[0] == 3 and self.shape[1] == 2:
            return self.col(0).cross(self.col(1)).norm()
        raise ValueError(f"Cannot find determinant of {self.shape[0]}x{self.shape[1]} matrix.")

    def transpose(self) -> MatrixFunction:
        """Compute the transpose.

        Returns:
            The transpose
        """
        mat = self.as_sympy()
        assert isinstance(mat, sympy.matrices.dense.MutableDenseMatrix)
        return MatrixFunction(mat.transpose())

    def with_floats(self) -> AnyFunction:
        """Return a version the function with floats as coefficients.

        Returns:
            A version the function with floats as coefficients
        """
        return MatrixFunction(tuple(tuple(f.with_floats() for f in row) for row in self._mat))


FunctionInput = typing.Union[
    AnyFunction,
    sympy.core.expr.Expr, int,
    typing.Tuple[typing.Union[sympy.core.expr.Expr, int, AnyFunction], ...],
    typing.List[typing.Union[sympy.core.expr.Expr, int, AnyFunction]],
    typing.Tuple[typing.Tuple[typing.Union[sympy.core.expr.Expr, int, AnyFunction], ...], ...],
    typing.Tuple[typing.List[typing.Union[sympy.core.expr.Expr, int, AnyFunction]], ...],
    typing.List[typing.Tuple[typing.Union[sympy.core.expr.Expr, int, AnyFunction], ...]],
    typing.List[typing.List[typing.Union[sympy.core.expr.Expr, int, AnyFunction]]],
    sympy.matrices.dense.MutableDenseMatrix]


def parse_function_input(f: FunctionInput) -> AnyFunction:
    """Parse a function.

    Args:
        f: A function

    Returns:
        The function as a Symfem function
    """
    if isinstance(f, AnyFunction):
        return f
    if isinstance(f, (sympy.core.expr.Expr, int)):
        return ScalarFunction(f)
    if isinstance(f, (list, tuple)):
        if isinstance(f[0], (list, tuple)):
            mat = []
            for i in f:
                assert isinstance(i, (tuple, list))
                mat.append(tuple(i))
            return MatrixFunction(tuple(mat))
        else:
            vec = []
            for i in f:
                assert not isinstance(i, (tuple, list))
                vec.append(i)
            return VectorFunction(tuple(vec))
    raise ValueError(f"Could not parse input function: {f}")


def parse_function_list_input(
    functions: typing.Union[typing.List[FunctionInput], typing.Tuple[FunctionInput, ...]]
) -> typing.List[AnyFunction]:
    """Parse a list of functions.

    Args:
        functions: The functions

    Returns:
        The functions as Symfem functions
    """
    return [parse_function_input(f) for f in functions]
