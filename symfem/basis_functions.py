"""Abstract basis function classes and functions."""

from __future__ import annotations

import typing
from abc import abstractmethod

import sympy

from .functions import AnyFunction, FunctionInput, ScalarFunction, SympyFormat, ValuesToSubstitute
from .geometry import PointType
from .references import Reference
from .symbols import AxisVariables, AxisVariablesNotSingle, t


class BasisFunction(AnyFunction):
    """A basis function of a finite element.

    This basis function can be used before the element's basis functions have been computed. When
    the explicit basis function is needed, only then will it be computed.
    """

    def __init__(self, scalar=False, vector=False, matrix=False):
        """Create a basis function.

        Args:
            scalar: Is the function a scalar?
            vector: Is the function a vector?
            matrix: Is the function a matrix?
        """
        super().__init__(scalar=scalar, vector=vector, matrix=matrix)

    @abstractmethod
    def get_function(self) -> AnyFunction:
        """Get the actual basis function.

        Returns:
            The basis function
        """
        pass

    def __add__(self, other: typing.Any) -> AnyFunction:
        """Add."""
        return self.get_function().__add__(other)

    def __radd__(self, other: typing.Any) -> AnyFunction:
        """Add."""
        return self.get_function().__radd__(other)

    def __sub__(self, other: typing.Any) -> AnyFunction:
        """Subtract."""
        return self.get_function().__sub__(other)

    def __rsub__(self, other: typing.Any) -> AnyFunction:
        """Subtract."""
        return self.get_function().__rsub__(other)

    def __neg__(self) -> AnyFunction:
        """Negate."""
        return self.get_function().__neg__()

    def __truediv__(self, other: typing.Any) -> AnyFunction:
        """Divide."""
        return self.get_function().__truediv__(other)

    def __rtruediv__(self, other: typing.Any) -> AnyFunction:
        """Divide."""
        return self.get_function().__rtruediv__(other)

    def __mul__(self, other: typing.Any) -> AnyFunction:
        """Multiply."""
        return self.get_function().__mul__(other)

    def __rmul__(self, other: typing.Any) -> AnyFunction:
        """Multiply."""
        return self.get_function().__rmul__(other)

    def __matmul__(self, other: typing.Any) -> AnyFunction:
        """Multiply."""
        return self.get_function().__matmul__(other)

    def __rmatmul__(self, other: typing.Any) -> AnyFunction:
        """Multiply."""
        return self.get_function().__rmatmul__(other)

    def __pow__(self, other: typing.Any) -> AnyFunction:
        """Raise to a power."""
        return self.get_function().__pow__(other)

    def as_sympy(self) -> SympyFormat:
        """Convert to a sympy expression."""
        return self.get_function().as_sympy()

    def as_tex(self) -> str:
        """Convert to a TeX expression."""
        return self.get_function().as_tex()

    def diff(self, variable: sympy.core.symbol.Symbol) -> AnyFunction:
        """Differentiate the function."""
        return self.get_function().diff(variable)

    def directional_derivative(self, direction: PointType) -> AnyFunction:
        """Compute a directional derivative."""
        return self.get_function().directional_derivative(direction)

    def jacobian_component(self, component: typing.Tuple[int, int]) -> AnyFunction:
        """Compute a component of the jacobian."""
        return self.get_function().jacobian_component(component)

    def jacobian(self, dim: int) -> AnyFunction:
        """Compute the jacobian."""
        return self.get_function().jacobian(dim)

    def dot(self, other_in: FunctionInput) -> AnyFunction:
        """Compute the dot product with another function."""
        return self.get_function().dot(other_in)

    def cross(self, other_in: FunctionInput) -> AnyFunction:
        """Compute the cross product with another function."""
        return self.get_function().cross(other_in)

    def div(self) -> AnyFunction:
        """Compute the div of the function."""
        return self.get_function().div()

    def grad(self, dim: int) -> AnyFunction:
        """Compute the grad of the function."""
        return self.get_function().grad(dim)

    def curl(self) -> AnyFunction:
        """Compute the curl of the function."""
        return self.get_function().curl()

    def norm(self) -> ScalarFunction:
        """Compute the norm of the function."""
        raise self.get_function().norm()

    def integral(self, domain: Reference, vars: AxisVariablesNotSingle = t) -> AnyFunction:
        """Compute the integral of the function."""
        return self.get_function().integral(domain, vars)

    def subs(self, vars: AxisVariables, values: ValuesToSubstitute) -> BasisFunction:
        """Substitute values into the function."""
        return SubbedBasisFunction(self, vars, values)

    def __getitem__(self, key) -> AnyFunction:
        """Forward all other function calls to symbolic function."""
        return self.get_function().__getitem__(key)

    def __len__(self) -> int:
        """Get length."""
        return self.get_function().__len__()

    def det(self) -> ScalarFunction:
        """Compute the determinant."""
        if not self.is_matrix:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'det'")
        return self.get_function().det()

    def transpose(self) -> ScalarFunction:
        """Compute the transpose."""
        if not self.is_matrix:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'transpose'")
        return self.get_function().transpose()

    def with_floats(self) -> AnyFunction:
        """Return a version the function with floats as coefficients."""
        return self.get_function().with_floats()


class SubbedBasisFunction(BasisFunction):
    """A basis function following a substitution."""

    def __init__(
        self, f: BasisFunction, vars: AxisVariables, values: ValuesToSubstitute
    ):
        """Create a basis function following a substitution.

        Args:
            f: The basis function
            vars: The variables that have been substituted
            values: The values to substitute
        """
        super().__init__(scalar=f.is_scalar, vector=f.is_vector, matrix=f.is_matrix)
        self.f = f
        self._vars = vars
        self._values = values

    def get_function(self) -> AnyFunction:
        """Return the symbolic function."""
        return self.f.get_function().subs(self._vars, self._values)

    @property
    def shape(self) -> typing.Tuple[int, ...]:
        """Get the value shape of the function."""
        return self.f.get_function().shape

    def plot_values(
        self, reference: Reference, img: typing.Any,
        value_scale: sympy.core.expr.Expr = sympy.Integer(1), n: int = 6
    ):
        """Plot the function's values."""
        self.f.plot_values(reference, img, value_scale, n)
