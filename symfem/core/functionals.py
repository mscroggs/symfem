"""Functionals used to define the dual sets."""
from .symbolic import subs, x, t
from .vectors import vdot
from .calculus import derivative, jacobian_component


class BaseFunctional:
    """A functional."""

    def __init__(self, entity=(None, None)):
        self.entity = entity

    def eval(self, fun):
        """Apply to the functional to a function."""
        raise NotImplementedError

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return tuple(None for i in range(self.reference.gdim))

    def dof_direction(self):
        """Get the direction of the DOF."""
        return None

    def entity_dim(self):
        """Get the dimension of the entitiy this DOF is associated with."""
        return self.entity[0]

    name = None


class PointEvaluation(BaseFunctional):
    """A point evaluation."""

    def __init__(self, point, entity=(None, None)):
        super().__init__(entity)
        self.point = point

    def eval(self, function):
        """Apply to the functional to a function."""
        return subs(function, x, self.point)

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return self.point

    name = "Point evaluation"


class PointDirectionalDerivativeEvaluation(BaseFunctional):
    """A point evaluation of a derivative in a fixed direction."""

    def __init__(self, point, direction, entity=(None, None)):
        super().__init__(entity)
        self.point = point
        self.dir = direction

    def eval(self, function):
        """Apply to the functional to a function."""
        return subs(derivative(function, self.dir), x, self.point)

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return self.point

    def dof_direction(self):
        """Get the direction of the DOF."""
        return self.dir

    name = "Point evaluation of directional derivative"


class PointNormalDerivativeEvaluation(PointDirectionalDerivativeEvaluation):
    """A point evaluation of a normal derivative."""

    def __init__(self, point, edge, entity=(None, None)):
        super().__init__(point, edge.normal(), entity=entity)
        self.reference = edge

    name = "Point evaluation of normal derivative"


class PointComponentSecondDerivativeEvaluation(BaseFunctional):
    """A point evaluation of a component of a second derivative."""

    def __init__(self, point, component, entity=(None, None)):
        super().__init__(entity)
        self.point = point
        self.component = component

    def eval(self, function):
        """Apply to the functional to a function."""
        return subs(jacobian_component(function, self.component), x, self.point)

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return self.point

    name = "Point evaluation of Jacobian component"


class PointInnerProduct(BaseFunctional):
    """An evaluation of an inner product at a point."""

    def __init__(self, point, vec, entity=(None, None)):
        super().__init__(entity)
        self.point = point
        self.vec = vec

    def eval(self, function):
        """Apply to the functional to a function."""
        v = subs(function, x, self.point)
        tdim = len(self.vec)
        return vdot(self.vec,
                    tuple(vdot(v[tdim * i: tdim * (i + 1)], self.vec)
                          for i in range(0, tdim)))

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return self.point

    def dof_direction(self):
        """Get the location of the DOF in the cell."""
        return self.vec

    name = "Point inner product"


class DotPointEvaluation(BaseFunctional):
    """A point evaluation in a given direction."""

    def __init__(self, point, vector, entity=(None, None)):
        super().__init__(entity)
        self.point = point
        self.vector = vector

    def eval(self, function):
        """Apply to the functional to a function."""
        return subs(vdot(function, self.vector), x, self.point)

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return self.point

    def dof_direction(self):
        """Get the direction of the DOF."""
        return self.vector

    name = "Dot point evaluation"


class IntegralMoment(BaseFunctional):
    """An integral moment."""

    def __init__(self, reference, f, dof, entity=(None, None)):
        super().__init__(entity)
        self.reference = reference
        self.dof = dof
        self.f = subs(f, x, t)
        if isinstance(self.f, tuple):
            # TODO: is this one of the mappings?
            self.f = tuple(
                sum(self.reference.axes[j][i] * c / self.reference.jacobian()
                    for j, c in enumerate(self.f))
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

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        p = self.dof.dof_point()
        return tuple(
            o + sum(self.reference.axes[j][i] * c for j, c in enumerate(p))
            for i, o in enumerate(self.reference.origin)
        )

    def dof_direction(self):
        """Get the direction of the DOF."""
        p = self.dof.dof_direction()
        if p is None:
            return None
        return tuple(
            sum(self.reference.axes[j][i] * c for j, c in enumerate(p))
            for i in range(self.reference.gdim)
        )

    name = "Integral moment"


class VecIntegralMoment(IntegralMoment):
    """An integral moment applied to a component of a vector."""

    def __init__(self, reference, f, dot_with, dof, entity=(None, None)):
        super().__init__(reference, f, dof, entity=entity)
        self.dot_with = dot_with

    def dot(self, function):
        """Dot a function with the moment function."""
        return vdot(function, self.dot_with) * self.f

    def dof_direction(self):
        """Get the direction of the DOF."""
        return self.dot_with

    name = "Vector integral moment"


class TangentIntegralMoment(VecIntegralMoment):
    """An integral moment in the tangential direction."""

    def __init__(self, reference, f, dof, entity=(None, None)):
        super().__init__(reference, f, reference.tangent(), dof, entity=entity)

    name = "Tangential integral moment"


class NormalIntegralMoment(VecIntegralMoment):
    """An integral moment in the normal direction."""

    def __init__(self, reference, f, dof, entity=(None, None)):
        super().__init__(reference, f, reference.normal(), dof, entity=entity)

    name = "Normal integral moment"
