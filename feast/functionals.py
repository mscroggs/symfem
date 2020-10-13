"""Functionals used to define the dual sets."""
from .symbolic import subs, x, t
from .vectors import vdot


class BaseFunctional:
    """A functional."""

    def eval(self, fun):
        """Apply to the functional to a function."""
        raise NotImplementedError


class PointEvaluation(BaseFunctional):
    """A point evaluation."""

    def __init__(self, point):
        self.point = point

    def eval(self, function):
        """Apply to the functional to a function."""
        return subs(function, x, self.point)


class DotPointEvaluation(BaseFunctional):
    """A point evaluation in a given direction."""

    def __init__(self, point, vector):
        self.point = point
        self.vector = vector

    def eval(self, function):
        """Apply to the functional to a function."""
        return subs(vdot(function, self.vector), x, self.point)


class IntegralMoment(BaseFunctional):
    """An integral moment."""

    def __init__(self, reference, f):
        self.reference = reference
        self.f = subs(f, x, t)
        if isinstance(self.f, tuple):
            self.f = tuple(
                o + sum(self.reference.axes[j][i] * c for j, c in enumerate(self.f))
                for i, o in enumerate(self.reference.origin)
            )

    def eval(self, function):
        """Apply to the functional to a function."""
        point = [i for i in self.reference.origin]
        for i, a in enumerate(zip(*self.reference.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        integrand = self.dot(subs(function, x, point))
        return self.reference.integral(integrand)

    def dot(self, function):
        """Dot a function with the moment function."""
        return vdot(function, self.f)


class VecIntegralMoment(IntegralMoment):
    """An integral moment applied to a component of a vector."""

    def __init__(self, reference, f, dot_with):
        super().__init__(reference, f)
        self.dot_with = dot_with

    def dot(self, function):
        """Dot a function with the moment function."""
        return vdot(function, self.dot_with) * self.f


class TangentIntegralMoment(VecIntegralMoment):
    """An integral moment in the tangential direction."""

    def __init__(self, reference, f):
        super().__init__(reference, f, reference.tangent())


class NormalIntegralMoment(VecIntegralMoment):
    """An integral moment in the normal direction."""

    def __init__(self, reference, f):
        super().__init__(reference, f, reference.normal())
