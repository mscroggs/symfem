"""Abstract basis function classes and functions."""

from abc import ABC, abstractmethod
import sympy
import typing
from .functions import AnyFunction, ScalarFunction


class BasisFunction(ABC):
    """A basis function."""

    @abstractmethod
    def get_function(self) -> AnyFunction:
        """Return the symbolic function."""
        pass

    def subs(
        self, pre: sympy.core.symbol.Symbol, post: sympy.core.expr.Expr
    ) -> typing.Any:
        # ) -> BasisFunction:
        """Substitute a value into the function."""
        return SubbedBasisFunction(self, pre, post)

    def __getattr__(self, attr: str) -> typing.Any:
        """Forward all other function calls to symbolic function."""
        return getattr(self.get_function(), attr)

    def __getitem__(self, key: typing.Any) -> ScalarFunction:
        """Forward all other function calls to symbolic function."""
        f = self.get_function()
        assert isinstance(f, tuple)
        return f[key]

    def __mul__(self, other: ScalarFunction) -> ScalarFunction:
        """Multiply."""
        f = self.get_function()
        assert not isinstance(f, tuple)
        return f * other

    def __rmul__(self, other: ScalarFunction) -> ScalarFunction:
        """Multiply."""
        return self.__mul__(other)

    def __truediv__(self, other: ScalarFunction) -> ScalarFunction:
        """Divide."""
        f = self.get_function()
        assert not isinstance(f, tuple)
        if isinstance(f, int):
            f = sympy.Integer(f)
        if isinstance(other, int):
            other = sympy.Integer(other)
        assert isinstance(f, sympy.core.expr.Expr)
        assert isinstance(other, sympy.core.expr.Expr)
        return f / other

    def __rtruediv__(self, other: ScalarFunction) -> ScalarFunction:
        """Divide."""
        f = self.get_function()
        assert not isinstance(f, tuple)
        if isinstance(f, int):
            f = sympy.Integer(f)
        if isinstance(other, int):
            other = sympy.Integer(other)
        assert isinstance(f, sympy.core.expr.Expr)
        assert isinstance(other, sympy.core.expr.Expr)
        return other / f

    def __add__(self, other: ScalarFunction) -> ScalarFunction:
        """Add."""
        f = self.get_function()
        assert not isinstance(f, tuple)
        return f + other

    def __sub__(self, other: ScalarFunction) -> ScalarFunction:
        """Subtract."""
        f = self.get_function()
        assert not isinstance(f, tuple)
        assert not isinstance(f, PiecewiseFunction)
        return f - other

    def __iter__(self) -> typing.Iterable:
        """Return an iterable."""
        f = self.get_function()
        assert isinstance(f, tuple)
        return iter(f)

    def __neg__(self) -> ScalarFunction:
        """Negate."""
        f = self.get_function()
        assert not isinstance(f, tuple)
        assert not isinstance(f, PiecewiseFunction)
        return -f

    def __len__(self) -> int:
        """Get length."""
        f = self.get_function()
        assert isinstance(f, tuple)
        return len(f)


class SubbedBasisFunction(BasisFunction):
    """A basis function following a substitution."""

    def __init__(
        self, f: BasisFunction, sub_pre: sympy.core.symbol.Symbol, sub_post: ScalarFunction
    ):
        self.f = f
        self.sub_pre = sub_pre
        self.sub_post = sub_post

    def get_function(self) -> AnyFunction:
        """Return the symbolic function."""
        return subs(self.f.get_function(), self.sub_pre, self.sub_post)
