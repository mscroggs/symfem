"""Functionals used to define the dual sets."""
import sympy
import numpy
from abc import ABC, abstractmethod
from .symbolic import subs, x, t, PiecewiseFunction, sym_sum, to_sympy, to_float
from .vectors import vdot
from .calculus import derivative, jacobian_component, grad, diff, div
from .basis_function import BasisFunction
from . import mappings


def _to_tex(f, tfrac=False):
    """Concert an expresson to tex."""
    if isinstance(f, (list, tuple)):
        return "\\left(\\begin{array}{c}" + "\\\\".join(
            ["\\displaystyle " + _to_tex(i) for i in f]) + "\\end{array}\\right)"
    if isinstance(f, PiecewiseFunction):
        out = "\\begin{cases}\n"
        joiner = ""
        for points, func in f.pieces:
            out += joiner
            joiner = "\\\\"
            out += _to_tex(func, True)
            out += f"&\\text{{in }}\\operatorname{{Triangle}}({points})"
        out += "\\end{cases}"
        return out
    out = sympy.latex(sympy.simplify(sympy.expand(f)))
    out = out.replace("\\left[", "\\left(")
    out = out.replace("\\right]", "\\right)")

    if tfrac:
        return out.replace("\\frac", "\\tfrac")
    else:
        return out


def _nth(n):
    """Add th, st or nd to a number."""
    if n % 10 == 1 and n != 11:
        return f"{n}st"
    if n % 10 == 2 and n != 12:
        return f"{n}nd"
    return f"{n}th"


def _get_entity(reference, dof):
    """Get the entity the dof is associated with."""
    if dof.entity_dim() == reference.tdim:
        return "R"
    else:
        return f"{'vefc'[dof.entity_dim()]}_{{{_get_entity_number(reference, dof)}}}"


def _get_entity_definition(reference, dof):
    """Get the definition of the entity the dof is associated with."""
    if dof.entity_dim() == reference.tdim:
        return "\\(R\\) is the reference element"
    else:
        return (f"\\({_get_entity(reference, dof)}\\) is the "
                f"{_nth(_get_entity_number(reference, dof))} "
                f"{['vertex', 'edge', 'face', 'volume'][reference.tdim]}")


def _get_entity_number(reference, dof):
    """Get the number of the entity the dof is associated with."""
    if dof.entity_dim() == 1:
        entities = reference.edges
    elif dof.entity_dim() == 2:
        entities = reference.faces
    elif dof.entity_dim() == 3:
        entities = reference.volumes
    return str(entities.index(
        tuple(reference.vertices.index(i) for i in dof.reference.vertices)))


class BaseFunctional(ABC):
    """A functional."""

    def __init__(self, entity=(None, None), mapping="identity"):
        self.entity = entity
        self.mapping = mapping

    def entity_dim(self):
        """Get the dimension of the entitiy this DOF is associated with."""
        return self.entity[0]

    def perform_mapping(self, fs, map, inverse_map, tdim):
        """Map functions to a cell."""
        return [getattr(mappings, self.mapping)(f, map, inverse_map, tdim) for f in fs]

    @abstractmethod
    def dof_point(self):
        """Get the location of the DOF in the cell."""
        pass

    @abstractmethod
    def dof_direction(self):
        """Get the direction of the DOF."""
        pass

    @abstractmethod
    def eval(self, fun, symbolic=True):
        """Apply to the functional to a function."""
        pass

    @abstractmethod
    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        pass

    get_points_and_weights = None
    name = None


class PointEvaluation(BaseFunctional):
    """A point evaluation."""

    def __init__(self, point, entity=(None, None), mapping="identity"):
        super().__init__(entity, mapping)
        self.point = point

    def eval(self, function, symbolic=True):
        """Apply the functional to a function."""
        value = subs(function, x, self.point)
        if symbolic:
            return value
        else:
            return to_float(value)

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return self.point

    def dof_direction(self):
        """Get the direction of the DOF."""
        return None

    def get_points_and_weights(self, max_order=None):
        """Get points and weights that can be used to numerically evaluate functional."""
        return numpy.array([self.point]), numpy.array([1])

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        return f"v\\mapsto v({','.join([_to_tex(i, True) for i in self.point])})", []

    name = "Point evaluation"


class WeightedPointEvaluation(BaseFunctional):
    """A point evaluation."""

    def __init__(self, point, weight, entity=(None, None), mapping="identity"):
        super().__init__(entity, mapping)
        self.point = point
        self.weight = weight

    def eval(self, function, symbolic=True):
        """Apply the functional to a function."""
        value = subs(function, x, self.point) * self.weight
        if symbolic:
            return value
        else:
            return to_float(value)

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return self.point

    def dof_direction(self):
        """Get the direction of the DOF."""
        return None

    def get_points_and_weights(self, max_order=None):
        """Get points and weights that can be used to numerically evaluate functional."""
        return numpy.array([self.point]), numpy.array([self.weight])

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        return (f"v\\mapsto {_to_tex(self.weight)} "
                f"v({','.join([_to_tex(i, True) for i in self.point])})"), []

    name = "Weighted point evaluation"


class DerivativePointEvaluation(BaseFunctional):
    """A point evaluation of a given derivative."""

    def __init__(self, point, derivative, entity=(None, None), mapping=None):
        super().__init__(entity, mapping)
        self.point = point
        self.derivative = derivative

    def eval(self, function, symbolic=True):
        """Apply the functional to a function."""
        for i, j in zip(x, self.derivative):
            for k in range(j):
                function = diff(function, i)
        value = subs(function, x, self.point)
        if symbolic:
            return value
        else:
            return to_float(value)

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return self.point

    def dof_direction(self):
        """Get the direction of the DOF."""
        return None

    def perform_mapping(self, fs, map, inverse_map, tdim):
        """Map functions to a cell."""
        if self.mapping is not None:
            return super().perform_mapping(fs, map, inverse_map, tdim)
        out = []
        J = sympy.Matrix([[diff(map[i], x[j]) for j in range(tdim)] for i in range(tdim)])
        for dofs in zip(*[fs[i::tdim] for i in range(tdim)]):
            for i in range(tdim):
                out.append(sym_sum(a * b for a, b in zip(dofs, J.row(i))))
        return [subs(b, x, inverse_map) for b in out]

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        if len(self.point) == 1:
            desc = "v\\mapsto "
            desc += f"v'({','.join([_to_tex(i, True) for i in self.point])})"
            return desc, []
        desc = "v\\mapsto"
        desc += "\\frac{\\partial"
        if sum(self.derivative) > 1:
            desc += f"^{{{sum(self.derivative)}}}"
        desc += "}{"
        for v, i in zip("xyz", self.derivative):
            if i > 0:
                desc += f"\\partial {v}"
                if i > 1:
                    desc += f"^{{{i}}}"
        desc += "}"
        desc += f"v({','.join([_to_tex(i, True) for i in self.point])})"
        return desc, []

    name = "Point derivative evaluation"


class PointDirectionalDerivativeEvaluation(BaseFunctional):
    """A point evaluation of a derivative in a fixed direction."""

    def __init__(self, point, direction, entity=(None, None), mapping="identity"):
        super().__init__(entity, mapping)
        self.point = point
        self.dir = direction

    def eval(self, function, symbolic=True):
        """Apply the functional to a function."""
        if isinstance(function, PiecewiseFunction):
            function = function.get_piece(self.point)
        value = subs(derivative(function, self.dir), x, self.point)
        if symbolic:
            return value
        else:
            return to_float(value)

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return self.point

    def dof_direction(self):
        """Get the direction of the DOF."""
        return self.dir

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        if len(self.point) == 1:
            desc = "v\\mapsto "
            desc += f"v'({','.join([_to_tex(i, True) for i in self.point])})"
            return desc, []
        desc = "v\\mapsto"
        desc += f"\\nabla{{v}}({','.join([_to_tex(i, True) for i in self.point])})"
        desc += "\\cdot\\left(\\begin{array}{c}"
        desc += "\\\\".join([_to_tex(i) for i in self.dir])
        desc += "\\end{array}\\right)"
        return desc, []

    name = "Point evaluation of directional derivative"


class PointNormalDerivativeEvaluation(PointDirectionalDerivativeEvaluation):
    """A point evaluation of a normal derivative."""

    def __init__(self, point, edge, entity=(None, None), mapping="identity"):
        super().__init__(point, edge.normal(), entity=entity, mapping=mapping)
        self.reference = edge

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        desc = "v\\mapsto"
        desc += "\\nabla{v}(" + ",".join([_to_tex(i, True) for i in self.dof_point()]) + ")"
        entity_n = _get_entity_number(self.reference, self)
        desc += "\\cdot\\hat{\\boldsymbol{n}}" + f"_{{{entity_n}}}"
        return desc, [
            "\\(\\hat{\\boldsymbol{n}}" + f"_{{{entity_n}}}\\) is the normal to facet {entity_n}"
        ]

    name = "Point evaluation of normal derivative"


class PointComponentSecondDerivativeEvaluation(BaseFunctional):
    """A point evaluation of a component of a second derivative."""

    def __init__(self, point, component, entity=(None, None), mapping="identity"):
        super().__init__(entity, mapping)
        self.point = point
        self.component = component

    def eval(self, function, symbolic=True):
        """Apply the functional to a function."""
        value = subs(jacobian_component(function, self.component), x, self.point)
        if symbolic:
            return value
        else:
            return to_float(value)

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return self.point

    def dof_direction(self):
        """Get the direction of the DOF."""
        return None

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        desc = "v\\mapsto"
        desc += "\\frac{\\partial^2v}{"
        for c in self.component:
            desc += "\\partial " + "xyz"[c]
        desc += "}(" + ",".join([_to_tex(i, True) for i in self.dof_point()]) + ")"
        return desc, []

    name = "Point evaluation of Jacobian component"


class PointInnerProduct(BaseFunctional):
    """An evaluation of an inner product at a point."""

    def __init__(self, point, lvec, rvec, entity=(None, None), mapping="identity"):
        super().__init__(entity, mapping)
        self.point = point
        self.lvec = lvec
        self.rvec = rvec

    def eval(self, function, symbolic=True):
        """Apply the functional to a function."""
        v = subs(function, x, self.point)
        tdim = len(self.lvec)
        assert len(function) == tdim ** 2
        value = vdot(self.lvec,
                     tuple(vdot(v[tdim * i: tdim * (i + 1)], self.rvec)
                           for i in range(0, tdim)))
        if symbolic:
            return value
        else:
            return to_float(value)

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return self.point

    def dof_direction(self):
        """Get the direction of the DOF."""
        if self.rvec != self.lvec:
            return None
        return self.lvec

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        desc = "\\mathbf{V}\\mapsto"
        desc += "\\left(\\begin{array}{c}"
        desc += "\\\\".join([_to_tex(i) for i in self.lvec])
        desc += "\\end{array}\\right)^{\\text{t}}"
        desc += "\\mathbf{V}(" + ",".join([_to_tex(i, True) for i in self.dof_point()]) + ")"
        desc += "\\left(\\begin{array}{c}"
        desc += "\\\\".join([_to_tex(i) for i in self.rvec])
        desc += "\\end{array}\\right)"
        return desc, []

    name = "Point inner product"


class DotPointEvaluation(BaseFunctional):
    """A point evaluation in a given direction."""

    def __init__(self, point, vector, entity=(None, None), mapping="identity"):
        super().__init__(entity, mapping)
        self.point = point
        self.vector = vector

    def eval(self, function, symbolic=True):
        """Apply the functional to a function."""
        value = vdot(subs(function, x, self.point), subs(self.vector, x, self.point))
        if symbolic:
            return value
        else:
            return to_float(value)

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return self.point

    def dof_direction(self):
        """Get the direction of the DOF."""
        return self.vector

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        desc = "\\boldsymbol{v}\\mapsto"
        desc += "\\boldsymbol{v}(" + ",".join([_to_tex(i, True) for i in self.dof_point()]) + ")"
        if isinstance(self.vector, (tuple, list)):
            desc += "\\cdot\\left(\\begin{array}{c}"
            desc += "\\\\".join([_to_tex(i) for i in self.vector])
            desc += "\\end{array}\\right)"
        elif self.vector != 1:
            desc += f"\\cdot{_to_tex(self.vector)}"
        return desc, []

    name = "Dot point evaluation"


class IntegralAgainst(BaseFunctional):
    """An integral against a function."""

    def __init__(self, reference, f, entity=(None, None), mapping="identity"):
        super().__init__(entity, mapping)
        self.reference = reference

        if isinstance(f, BasisFunction):
            f = f.get_function()
        f = subs(f, x, t)

        if isinstance(f, tuple):
            if len(f) == self.reference.tdim:
                self.f = mappings.contravariant(
                    f, reference.get_map_to_self(), reference.get_inverse_map_to_self(),
                    reference.tdim)
            else:
                assert len(f) == self.reference.tdim ** 2
                self.f = mappings.double_contravariant(
                    f, reference.get_map_to_self(), reference.get_inverse_map_to_self(),
                    reference.tdim)
        else:
            self.f = f

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return tuple(sympy.Rational(sum(i), len(i)) for i in zip(*self.reference.vertices))

    def dof_direction(self):
        """Get the direction of the DOF."""
        return None

    def eval(self, function, symbolic=True):
        """Apply the functional to a function."""
        point = [i for i in self.reference.origin]
        for i, a in enumerate(zip(*self.reference.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        integrand = self.dot(subs(function, x, point))
        value = self.reference.integral(integrand)
        if symbolic:
            return value
        else:
            return to_float(value)

    def dot(self, function):
        """Dot a function with the moment function."""
        return vdot(function, self.f)

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = _get_entity(self.reference, self)
        entity_def = _get_entity_definition(self.reference, self)
        if isinstance(self.f, tuple):
            desc = "\\mathbf{v}\\mapsto"
            desc += f"\\displaystyle\\int_{{{entity}}}"
            desc += _to_tex(self.f, True) + "\\cdot"
            desc += "\\mathbf{v}"
        else:
            desc = "\\mathbf{v}\\mapsto"
            desc += f"\\displaystyle\\int_{{{entity}}}"
            if self.f != 1:
                desc += "(" + _to_tex(self.f, True) + ")"
            desc += "v"
        return desc, [entity_def]

    name = "Integral against"


class IntegralOfDivergenceAgainst(BaseFunctional):
    """An integral of the divergence against a function."""

    def __init__(self, reference, f, entity=(None, None), mapping="identity"):
        super().__init__(entity, mapping)
        self.reference = reference

        if isinstance(f, BasisFunction):
            f = f.get_function()
        self.f = subs(f, x, t)

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return tuple(sympy.Rational(sum(i), len(i)) for i in zip(*self.reference.vertices))

    def dof_direction(self):
        """Get the direction of the DOF."""
        return None

    def eval(self, function, symbolic=True):
        """Apply the functional to a function."""
        point = [i for i in self.reference.origin]
        for i, a in enumerate(zip(*self.reference.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        integrand = self.dot(subs(div(function), x, point))
        value = self.reference.integral(integrand)
        if symbolic:
            return value
        else:
            return to_float(value)

    def dot(self, function):
        """Dot a function with the moment function."""
        return function * self.f

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = _get_entity(self.reference, self)
        entity_def = _get_entity_definition(self.reference, self)
        desc = "\\mathbf{v}\\mapsto"
        desc += f"\\displaystyle\\int_{{{entity}}}"
        if self.f != 1:
            desc += "(" + _to_tex(self.f, True) + ")"
        desc += "\\nabla\\cdot\\mathbf{v}"
        return desc, [entity_def]

    name = "Integral of divergence against"


class IntegralOfDirectionalMultiderivative(BaseFunctional):
    """An integral of a directional derivative of a scalar function."""

    def __init__(self, reference, directions, orders, scale=1, entity=(None, None),
                 mapping="identity"):
        super().__init__(entity, mapping)
        self.reference = reference
        self.directions = directions
        self.orders = orders
        self.scale = scale

    def dof_point(self):
        """Get the location of the DOF in the cell."""
        return tuple(sympy.Rational(sum(i), len(i)) for i in zip(*self.reference.vertices))

    def dof_direction(self):
        """Get the direction of the DOF."""
        return None

    def eval(self, function, symbolic=True):
        """Apply the functional to a function."""
        for dir, o in zip(self.directions, self.orders):
            for i in range(o):
                function = sum(d * diff(function, x[j]) for j, d in enumerate(dir))
        point = [i for i in self.reference.origin]
        for i, a in enumerate(zip(*self.reference.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        integrand = self.scale * subs(function, x, point)
        value = self.reference.integral(integrand)
        if symbolic:
            return value
        else:
            return to_float(value)

    def perform_mapping(self, fs, map, inverse_map, tdim):
        """Map functions to a cell."""
        if sum(self.orders) > 0:
            raise NotImplementedError("Mapping high order derivatives not implemented")
        return super().perform_mapping(fs, map, inverse_map, tdim)

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = _get_entity(self.reference, self)
        entity_def = _get_entity_definition(self.reference, self)
        desc = "\\mathbf{V}\\mapsto"
        desc += "\\displaystyle"
        if self.scale != 1:
            desc += _to_tex(self.scale)
        desc += f"\\int_{{{entity}}}"
        for order, dir in zip(self.orders, self.directions):
            if order > 0:
                desc += "\\frac{\\partial"
                if order > 1:
                    desc += f"^{{{order}}}"
                desc += "}{"
                desc += "\\partial" + _to_tex(dir)
                if order > 1:
                    desc += f"^{{{order}}}"
                desc += "}"
        desc += "v"
        return desc, [entity_def]

    name = "Integral of a directional derivative"


class IntegralMoment(BaseFunctional):
    """An integral moment."""

    def __init__(self, reference, f, dof, entity=(None, None), mapping="identity"):
        super().__init__(entity, mapping)
        self.reference = reference
        self.dof = dof

        if isinstance(f, BasisFunction):
            f = f.get_function()
        f = subs(f, x, t)

        if isinstance(f, tuple):
            if len(f) == self.reference.tdim:
                self.f = mappings.contravariant(
                    f, reference.get_map_to_self(), reference.get_inverse_map_to_self(),
                    reference.tdim)
            else:
                assert len(f) == self.reference.tdim ** 2
                self.f = mappings.double_contravariant(
                    f, reference.get_map_to_self(), reference.get_inverse_map_to_self(),
                    reference.tdim)
        else:
            self.f = f

    def eval(self, function, symbolic=True):
        """Apply the functional to a function."""
        point = [i for i in self.reference.origin]
        for i, a in enumerate(zip(*self.reference.axes)):
            for j, k in zip(a, t):
                point[i] += j * k

        integrand = self.dot(subs(function, x, point))
        if isinstance(integrand, PiecewiseFunction):
            integrand = integrand.get_piece(self.reference.midpoint())
        value = self.reference.integral(to_sympy(integrand))
        if symbolic:
            return value
        else:
            return to_float(value)

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

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = _get_entity(self.reference, self)
        entity_def = _get_entity_definition(self.reference, self)
        try:
            self.f[0]
            if len(self.f) == self.reference.tdim:
                desc = "\\boldsymbol{v}\\mapsto"
                desc += f"\\displaystyle\\int_{{{entity}}}"
                desc += "\\boldsymbol{v}\\cdot"
                desc += "\\left(\\begin{array}{c}"
                desc += "\\\\".join([_to_tex(i) for i in self.f])
                desc += "\\end{array}\\right)"
            else:
                assert len(self.f) == self.reference.tdim ** 2
                desc = "\\mathbf{V}\\mapsto"
                desc += f"\\displaystyle\\int_{{{entity}}}"
                desc += "\\mathbf{V}:"
                desc += "\\left(\\begin{array}{" + "c" * self.reference.tdim + "}"
                desc += "\\\\".join(["&".join(
                    [_to_tex(self.f[i]) for i in range(self.reference.tdim * row,
                                                       self.reference.tdim * (row + 1))]
                ) for row in range(self.reference.tdim)])
                desc += "\\end{array}\\right)"
        except:  # noqa: E722
            desc = "v\\mapsto"
            desc += f"\\displaystyle\\int_{{{entity}}}"
            if self.f != 1:
                desc += "(" + _to_tex(self.f) + ")"
            desc += "v"
        return desc, [entity_def]

    name = "Integral moment"


class VecIntegralMoment(IntegralMoment):
    """An integral moment applied to a component of a vector."""

    def __init__(self, reference, f, dot_with, dof, entity=(None, None), mapping="identity"):
        super().__init__(reference, f, dof, entity=entity, mapping=mapping)
        self.dot_with = dot_with

    def dot(self, function):
        """Dot a function with the moment function."""
        return vdot(function, self.dot_with) * self.f

    def dof_direction(self):
        """Get the direction of the DOF."""
        return self.dot_with

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = _get_entity(self.reference, self)
        entity_def = _get_entity_definition(self.reference, self)
        desc = "\\boldsymbol{v}\\mapsto"
        desc += f"\\displaystyle\\int_{{{entity}}}"
        if self.f != 1:
            desc += "(" + _to_tex(self.f, True) + ")"
        desc += "\\boldsymbol{v}\\cdot"
        desc += _to_tex(self.dot_with)
        return desc, [entity_def]

    name = "Vector integral moment"


class DerivativeIntegralMoment(IntegralMoment):
    """An integral moment of the derivative of a scalar function."""

    def __init__(self, reference, f, dot_with, dof, entity=(None, None), mapping="identity"):
        super().__init__(reference, f, dof, entity=entity, mapping=mapping)
        self.dot_with = dot_with

    def dot(self, function):
        """Dot a function with the moment function."""
        return vdot(function, self.dot_with) * self.f

    def dof_direction(self):
        """Get the direction of the DOF."""
        return self.dot_with

    def eval(self, function, symbolic=True):
        """Apply the functional to a function."""
        point = [i for i in self.reference.origin]
        for i, a in enumerate(zip(*self.reference.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        integrand = self.dot(subs(grad(function, self.reference.gdim), x, point))
        value = self.reference.integral(integrand)
        if symbolic:
            return value
        else:
            return to_float(value)

    name = "Derivative integral moment"


class DivergenceIntegralMoment(IntegralMoment):
    """An integral moment of the divergence of a vector function."""

    def __init__(self, reference, f, dof, entity=(None, None), mapping="identity"):
        super().__init__(reference, f, dof, entity=entity, mapping=mapping)

    def eval(self, function, symbolic=True):
        """Apply the functional to a function."""
        point = [i for i in self.reference.origin]
        for i, a in enumerate(zip(*self.reference.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        integrand = self.dot(subs(div(function), x, point))
        value = self.reference.integral(integrand)
        if symbolic:
            return value
        else:
            return to_float(value)

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = _get_entity(self.reference, self)
        entity_def = _get_entity_definition(self.reference, self)
        desc = "\\boldsymbol{v}\\mapsto"
        desc += f"\\displaystyle\\int_{{{entity}}}"
        if self.f != 1:
            desc += "(" + _to_tex(self.f, True) + ")"
        desc += "\\nabla\\cdot\\boldsymbol{v}"
        return desc, [entity_def]

    name = "Integral moment of divergence"


class TangentIntegralMoment(VecIntegralMoment):
    """An integral moment in the tangential direction."""

    def __init__(self, reference, f, dof, entity=(None, None), mapping="covariant"):
        super().__init__(reference, f, reference.tangent(), dof, entity=entity, mapping=mapping)

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = _get_entity(self.reference, self)
        entity_n = _get_entity_number(self.reference, self)
        entity_def = _get_entity_definition(self.reference, self)
        desc = "\\boldsymbol{v}\\mapsto"
        desc += f"\\displaystyle\\int_{{{entity}}}"
        desc += "\\boldsymbol{v}\\cdot"
        if self.f != 1:
            desc += "(" + _to_tex(self.f) + ")"
        desc += "\\hat{\\boldsymbol{t}}" + f"_{{{entity_n}}}"
        return desc, [
            entity_def,
            f"\\(\\hat{{\\boldsymbol{{t}}}}_{{{entity_n}}}\\) is the tangent to edge {entity_n}"
        ]

    name = "Tangential integral moment"


class NormalIntegralMoment(VecIntegralMoment):
    """An integral moment in the normal direction."""

    def __init__(self, reference, f, dof, entity=(None, None), mapping="contravariant"):
        super().__init__(reference, f, reference.normal(), dof, entity=entity, mapping=mapping)

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = _get_entity(self.reference, self)
        entity_n = _get_entity_number(self.reference, self)
        entity_def = _get_entity_definition(self.reference, self)
        desc = "\\boldsymbol{v}\\mapsto"
        desc += f"\\displaystyle\\int_{{{entity}}}"
        desc += "\\boldsymbol{v}\\cdot"
        if self.f != 1:
            desc += "(" + _to_tex(self.f, True) + ")"
        desc += "\\hat{\\boldsymbol{n}}" + f"_{{{entity_n}}}"
        return desc, [
            entity_def,
            f"\\(\\hat{{\\boldsymbol{{n}}}}_{{{entity_n}}}\\) is the normal to facet {entity_n}"
        ]

    name = "Normal integral moment"


class NormalDerivativeIntegralMoment(DerivativeIntegralMoment):
    """An integral moment in the normal direction."""

    def __init__(self, reference, f, dof, entity=(None, None), mapping="identity"):
        super().__init__(reference, f, reference.normal(), dof, entity=entity, mapping=mapping)

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = _get_entity(self.reference, self)
        entity_n = _get_entity_number(self.reference, self)
        entity_def = _get_entity_definition(self.reference, self)
        desc = "v\\mapsto"
        desc += f"\\displaystyle\\int_{{{entity}}}"
        if self.f != 1:
            desc += "(" + _to_tex(self.f, True) + ")"
        desc += "\\frac{\\partial v}"
        desc += "{\\partial\\hat{\\boldsymbol{n}}" + f"_{{{entity_n}}}" + "}"
        return desc, [
            entity_def,
            f"\\(\\hat{{\\boldsymbol{{n}}}}_{{{entity_n}}}\\) is the normal to facet {entity_n}"
        ]

    name = "Normal derivative integral moment"


class InnerProductIntegralMoment(IntegralMoment):
    """An integral moment of the inner product with a vector."""

    def __init__(self, reference, f, inner_with_left, inner_with_right, dof,
                 entity=(None, None), mapping="identity"):
        super().__init__(reference, f, dof, entity=entity, mapping=mapping)
        self.inner_with_left = inner_with_left
        self.inner_with_right = inner_with_right

    def dot(self, function):
        """Take the inner product of a function with the moment direction."""
        tdim = len(self.inner_with_left)
        return vdot(self.inner_with_left,
                    tuple(vdot(function[tdim * i: tdim * (i + 1)], self.inner_with_right)
                          for i in range(0, tdim))) * self.f * self.reference.jacobian()

    def dof_direction(self):
        """Get the direction of the DOF."""
        if self.inner_with_left != self.inner_with_right:
            return None
        return self.inner_with_left

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = _get_entity(self.reference, self)
        entity_def = _get_entity_definition(self.reference, self)
        desc = "\\boldsymbol{V}\\mapsto"
        desc += f"\\displaystyle\\int_{{{entity}}}"
        if self.f != 1:
            desc += "(" + _to_tex(self.f, True) + ")"
        desc += _to_tex(self.inner_with_left) + "^{\\text{t}}"
        desc += "\\boldsymbol{V}"
        desc += _to_tex(self.inner_with_right)
        return desc, [entity_def]

    name = "Inner product integral moment"


class NormalInnerProductIntegralMoment(InnerProductIntegralMoment):
    """An integral moment of the inner product with the normal direction."""

    def __init__(self, reference, f, dof, entity=(None, None), mapping="double_contravariant"):
        super().__init__(reference, f, reference.normal(), reference.normal(), dof, entity=entity,
                         mapping=mapping)

    def get_tex(self):
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = _get_entity(self.reference, self)
        entity_n = _get_entity_number(self.reference, self)
        entity_def = _get_entity_definition(self.reference, self)
        desc = "\\mathbf{V}\\mapsto"
        desc += f"\\displaystyle\\int_{{{entity}}}"
        if self.f != 1:
            desc += "(" + _to_tex(self.f, True) + ")"
        desc += f"|{{{entity}}}|"
        desc += "\\hat{\\boldsymbol{n}}^{\\text{t}}" + f"_{{{entity_n}}}"
        desc += "\\mathbf{V}"
        desc += "\\hat{\\boldsymbol{n}}" + f"_{{{entity_n}}}"
        return desc, [
            entity_def,
            f"\\(\\hat{{\\boldsymbol{{n}}}}_{{{entity_n}}}\\) is the normal to facet {entity_n}"
        ]

    name = "Normal inner product integral moment"
