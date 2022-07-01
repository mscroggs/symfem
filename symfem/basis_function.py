"""Abstract basis function classes and functions."""

from abc import ABC, abstractmethod
import sympy
import typing
from .symbolic import subs


class BasisFunction(ABC):
    """A basis function."""

    @abstractmethod
    def get_function(self) -> sympy.core.expr.Expr:
        """Return the symbolic function."""
        pass

    def subs(
        self, pre: sympy.core.symbol.Symbol, post: sympy.core.expr.Expr
    ) -> sympy.core.expr.Expr:
        """Substitute a value into the function."""
        return SubbedBasisFunction(self, pre, post)

    def __getattr__(self, attr: str) -> typing.Any:
        """Forward all other function calls to symbolic function."""
        return getattr(self.get_function(), attr)

    def __getitem__(self, key: typing.Any) -> sympy.core.expr.Expr:
        """Forward all other function calls to symbolic function."""
        return self.get_function()[key]

    def __mul__(self, other: sympy.core.expr.Expr) -> sympy.core.expr.Expr:
        """Multiply."""
        return self.get_function() * other

    def __rmul__(self, other: sympy.core.expr.Expr) -> sympy.core.expr.Expr:
        """Multiply."""
        return self.__mul__(other)

    def __truediv__(self, other: sympy.core.expr.Expr) -> sympy.core.expr.Expr:
        """Divide."""
        return self.get_function() / other

    def __rtruediv__(self, other: sympy.core.expr.Expr) -> sympy.core.expr.Expr:
        """Divide."""
        return other / self.get_function()

    def __add__(self, other: sympy.core.expr.Expr) -> sympy.core.expr.Expr:
        """Add."""
        return self.get_function() + other

    def __sub__(self, other: sympy.core.expr.Expr) -> sympy.core.expr.Expr:
        """Subtract."""
        return self.get_function() - other

    def __iter__(self) -> typing.Iterable:
        """Return an iterable."""
        return iter(self.get_function())

    def __neg__(self) -> sympy.core.expr.Expr:
        """Negate."""
        return -self.get_function()

    def __len__(self) -> int:
        """Get length."""
        return len(self.get_function())


class SubbedBasisFunction(BasisFunction):
    """A basis function following a substitution."""

    def __init__(
        self, f: BasisFunction, sub_pre: sympy.core.symbol.Symbol, sub_post: sympy.core.expr.Expr
    ):
        self.f = f
        self.sub_pre = sub_pre
        self.sub_post = sub_post

    def get_function(self) -> sympy.core.expr.Expr:
        """Return the symbolic function."""
        return subs(self.f.get_function(), self.sub_pre, self.sub_post)
