"""Abstract basis function classes and functions."""

from .symbolic import subs


class BasisFunction:
    """A basis function."""

    def get_function(self):
        """Return the symbolic function."""
        raise NotImplementedError()

    def subs(self, pre, post):
        """Substitute a value into the function."""
        return SubbedBasisFunction(self, pre, post)

    def __getattr__(self, attr):
        """Forward all other function calls to symbolic function."""
        return getattr(self.get_function(), attr)

    def __rmul__(self, other):
        """Multiply."""
        return self.get_function() * other

    def __lmul__(self, other):
        """Multiply."""
        return self.__rmul__(other)

    def __truediv__(self, other):
        """Divide."""
        return self.get_function() / other

    def __add__(self, other):
        """Add."""
        return self.get_function() + other

    def __sub__(self, other):
        """Subtract."""
        return self.get_function() - other

    def __iter__(self):
        """Return an iterable."""
        return iter(self.get_function())

    def __neg__(self):
        """Negate."""
        return -self.get_function()


class ElementBasisFunction(BasisFunction):
    """A basis function of a finite element."""

    def __init__(self, element, n):
        self.element = element
        self.n = n

    def get_function(self):
        """Return the symbolic function."""
        return self.element.get_basis_functions()[self.n]


class SubbedBasisFunction(BasisFunction):
    """A basis function following a substitution."""

    def __init__(self, f, sub_pre, sub_post):
        self.f = f
        self.sub_pre = sub_pre
        self.sub_post = sub_post

    def get_function(self):
        """Return the symbolic function."""
        return subs(self.f.get_function(), self.sub_pre, self.sub_post)
