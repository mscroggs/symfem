"""Functionals used to define the dual sets."""

import typing
import sympy
import numpy
from abc import ABC, abstractmethod
from .symbolic import (subs, x, t, PiecewiseFunction, sym_sum,
                       ListOfAnyFunctions, PointType, AnyFunction, ScalarValue,
                       ListOfScalarFunctions,
                       ScalarFunction, VectorFunction, SetOfPoints)
from .vectors import vdot
from .calculus import derivative, jacobian_component, grad, diff, div
from .basis_function import BasisFunction
from .references import Reference, Interval
from . import mappings

ScalarValueOrFloat = typing.Union[ScalarValue, float]


def _to_tex(f: typing.Any, tfrac: bool = False) -> str:
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


def _nth(n: int) -> str:
    """Add th, st or nd to a number."""
    if n % 10 == 1 and n != 11:
        return f"{n}st"
    if n % 10 == 2 and n != 12:
        return f"{n}nd"
    return f"{n}th"


class BaseFunctional(ABC):
    """A functional."""

    def __init__(self, reference: Reference, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None]):
        self.reference = reference
        self.entity = entity
        self.mapping = mapping

    def entity_dim(self) -> int:
        """Get the dimension of the entity this DOF is associated with."""
        return self.entity[0]

    def entity_number(self) -> int:
        """Get the number of the entity this DOF is associated with."""
        return self.entity[1]

    def perform_mapping(
        self, fs: ListOfAnyFunctions, map: PointType, inverse_map: PointType, tdim: int
    ) -> ListOfAnyFunctions:
        """Map functions to a cell."""
        assert self.mapping is not None
        return [getattr(mappings, self.mapping)(f, map, inverse_map, tdim) for f in fs]

    def entity_tex(self) -> str:
        """Get the entity the dof is associated with."""
        if self.entity[0] == self.reference.tdim:
            return "R"
        else:
            return f"{'vefc'[self.entity[0]]}_{{{self.entity[1]}}}"

    def entity_definition(self) -> str:
        """Get the definition of the entity the dof is associated with."""
        if self.entity[0] == self.reference.tdim:
            return "\\(R\\) is the reference element"
        else:
            desc = f"\\({self.entity_tex()}\\) is the {_nth(self.entity[1])} "
            desc += ['vertex', 'edge', 'face', 'volume'][self.entity[0]]
            return desc

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF."""
        return None

    @abstractmethod
    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell."""
        pass

    @abstractmethod
    def eval(self, function: AnyFunction, symbolic: bool = True) -> ScalarValueOrFloat:
        """Apply to the functional to a function."""
        pass

    @abstractmethod
    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved."""
        pass

    def get_points_and_weights(self, max_order: int = None) -> typing.Union[
        typing.Tuple[numpy.typing.NDArray, numpy.typing.NDArray], None
    ]:
        """Get points and weights that can be used to numerically evaluate functional."""
        return None

    name = "Base functional"


class PointEvaluation(BaseFunctional):
    """A point evaluation."""

    def __init__(self, reference: Reference, point: PointType, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, entity, mapping)
        self.point = point

    def eval(self, function: AnyFunction, symbolic: bool = True) -> ScalarValueOrFloat:
        """Apply the functional to a function."""
        if isinstance(function, PiecewiseFunction):
            function = function.get_piece(self.point)
        assert isinstance(function, (int, sympy.core.expr.Expr))
        value = subs(function, x, self.point)
        assert isinstance(value, (int, sympy.core.expr.Expr))
        if symbolic:
            return value
        else:
            return float(value)

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell."""
        return self.point

    def get_points_and_weights(
        self, max_order: int = None
    ) -> typing.Tuple[numpy.typing.NDArray, numpy.typing.NDArray]:
        """Get points and weights that can be used to numerically evaluate functional."""
        return numpy.array([self.point]), numpy.array([1])

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved."""
        return f"v\\mapsto v({','.join([_to_tex(i, True) for i in self.point])})", []

    name = "Point evaluation"


class WeightedPointEvaluation(BaseFunctional):
    """A point evaluation."""

    def __init__(self, reference: Reference, point: PointType, weight: ScalarValue,
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, entity, mapping)
        self.point = point
        self.weight = weight

    def eval(self, function: AnyFunction, symbolic: bool = True) -> ScalarValueOrFloat:
        """Apply the functional to a function."""
        if isinstance(function, PiecewiseFunction):
            function = function.get_piece(self.point)
        assert isinstance(function, (int, sympy.core.expr.Expr))
        value = subs(function, x, self.point) * self.weight
        assert isinstance(value, (int, sympy.core.expr.Expr))
        if symbolic:
            return value
        else:
            return float(value)

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell."""
        return self.point

    def get_points_and_weights(
        self, max_order: int = None
    ) -> typing.Tuple[numpy.typing.NDArray, numpy.typing.NDArray]:
        """Get points and weights that can be used to numerically evaluate functional."""
        return numpy.array([self.point]), numpy.array([self.weight])

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved."""
        return (f"v\\mapsto {_to_tex(self.weight)} "
                f"v({','.join([_to_tex(i, True) for i in self.point])})"), []

    name = "Weighted point evaluation"


class DerivativePointEvaluation(BaseFunctional):
    """A point evaluation of a given derivative."""

    def __init__(self, reference: Reference, point: PointType, derivative: typing.Tuple[int, ...],
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = None):
        super().__init__(reference, entity, mapping)
        self.point = point
        self.derivative = derivative

    def eval(self, function: AnyFunction, symbolic: bool = True) -> ScalarValueOrFloat:
        """Apply the functional to a function."""
        if isinstance(function, PiecewiseFunction):
            function = function.get_piece(self.point)
        assert isinstance(function, (int, sympy.core.expr.Expr))
        for i, j in zip(x, self.derivative):
            for k in range(j):
                function = diff(function, i)
        value = subs(function, x, self.point)
        assert isinstance(value, (int, sympy.core.expr.Expr))
        if symbolic:
            return value
        else:
            return float(value)

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell."""
        return self.point

    def perform_mapping(
        self, fs: ListOfAnyFunctions, map: PointType, inverse_map: PointType, tdim: int
    ) -> ListOfAnyFunctions:
        """Map functions to a cell."""
        if self.mapping is not None:
            return super().perform_mapping(fs, map, inverse_map, tdim)
        out = []
        J = sympy.Matrix([[diff(map[i], x[j]) for j in range(tdim)] for i in range(tdim)])
        for dofs in zip(*[fs[i::tdim] for i in range(tdim)]):
            for i in range(tdim):
                out.append(sym_sum(a * b for a, b in zip(dofs, J.row(i))))

        out2: ListOfScalarFunctions = []
        for b in out:
            item = subs(b, x, inverse_map)
            assert isinstance(item, (int, sympy.core.expr.Expr))
            out2.append(item)
        return out2

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
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

    def __init__(self, reference: Reference, point: PointType, direction: PointType,
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, entity, mapping)
        self.point = point
        self.dir = direction

    def eval(self, function: AnyFunction, symbolic: bool = True) -> ScalarValueOrFloat:
        """Apply the functional to a function."""
        if isinstance(function, PiecewiseFunction):
            function = function.get_piece(self.point)
        assert isinstance(function, (int, sympy.core.expr.Expr))
        value = subs(derivative(function, self.dir), x, self.point)
        assert isinstance(value, (int, sympy.core.expr.Expr))
        if symbolic:
            return value
        else:
            return float(value)

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell."""
        return self.point

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF."""
        return self.dir

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
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

    def __init__(self, reference: Reference, point: PointType, edge: Interval,
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, point, edge.normal(), entity=entity, mapping=mapping)
        self.reference = edge

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved."""
        desc = "v\\mapsto"
        desc += "\\nabla{v}(" + ",".join([_to_tex(i, True) for i in self.dof_point()]) + ")"
        entity_n = self.entity_number()
        desc += "\\cdot\\hat{\\boldsymbol{n}}" + f"_{{{entity_n}}}"
        return desc, [
            "\\(\\hat{\\boldsymbol{n}}" + f"_{{{entity_n}}}\\) is the normal to facet {entity_n}"
        ]

    name = "Point evaluation of normal derivative"


class PointComponentSecondDerivativeEvaluation(BaseFunctional):
    """A point evaluation of a component of a second derivative."""

    def __init__(self, reference: Reference, point: PointType, component: typing.Tuple[int, int],
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, entity, mapping)
        self.point = point
        self.component = component

    def eval(self, function: AnyFunction, symbolic: bool = True) -> ScalarValueOrFloat:
        """Apply the functional to a function."""
        if isinstance(function, PiecewiseFunction):
            function = function.get_piece(self.point)
        assert isinstance(function, (int, sympy.core.expr.Expr))
        value = subs(jacobian_component(function, self.component), x, self.point)
        assert isinstance(value, (int, sympy.core.expr.Expr))
        if symbolic:
            return value
        else:
            return float(value)

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell."""
        return self.point

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
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

    def __init__(self, reference: Reference, point: PointType, lvec: PointType, rvec: PointType,
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, entity, mapping)
        self.point = point
        self.lvec = lvec
        self.rvec = rvec

    def eval(self, function: AnyFunction, symbolic: bool = True) -> ScalarValueOrFloat:
        """Apply the functional to a function."""
        if isinstance(function, PiecewiseFunction):
            function = function.get_piece(self.point)
        v = subs(function, x, self.point)
        if isinstance(function, sympy.Matrix):
            function = tuple(function[i, j]
                             for i in range(function.rows) for j in range(function.cols))
        assert isinstance(function, tuple)
        if isinstance(v, sympy.Matrix):
            v = tuple(v[i, j] for i in range(v.rows) for j in range(v.cols))
        assert isinstance(v, tuple)
        tdim = len(self.lvec)
        assert len(function) == tdim ** 2
        value = vdot(self.lvec,
                     tuple(vdot(v[tdim * i: tdim * (i + 1)], self.rvec)
                           for i in range(0, tdim)))
        assert isinstance(value, (int, sympy.core.expr.Expr))
        if symbolic:
            return value
        else:
            return float(value)

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell."""
        return self.point

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF."""
        if self.rvec != self.lvec:
            return None
        return self.lvec

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
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

    def __init__(self, reference: Reference, point: PointType, vector: PointType,
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, entity, mapping)
        self.point = point
        self.vector = vector

    def eval(self, function: AnyFunction, symbolic: bool = True) -> ScalarValueOrFloat:
        """Apply the functional to a function."""
        if isinstance(function, PiecewiseFunction):
            function = function.get_piece(self.point)
        v1 = subs(function, x, self.point)
        v2 = subs(self.vector, x, self.point)
        if isinstance(v2, (int, sympy.core.expr.Expr)):
            assert isinstance(v1, (int, sympy.core.expr.Expr))
            value = v1 * v2
        elif isinstance(v2, tuple):
            if isinstance(v1, sympy.Matrix):
                v1t = tuple(v1[i, j] for i in range(v1.rows) for j in range(v1.cols))
                value = vdot(v1t, v2)
            else:
                assert isinstance(v1, tuple)
                value = vdot(v1, v2)
        else:
            assert isinstance(v1, sympy.Matrix)
            assert isinstance(v2, sympy.Matrix)
            value = 0
            for i in range(v1.rows):
                for j in range(v2.cols):
                    value += v1[i, j] * v2[i, j]
        if symbolic:
            return value
        else:
            return float(value)

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell."""
        return self.point

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF."""
        return self.vector

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
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

    def __init__(self, reference: Reference, integral_domain: Reference,
                 f: typing.Union[AnyFunction, BasisFunction],
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, entity, mapping)
        self.integral_domain = integral_domain

        if isinstance(f, BasisFunction):
            f = f.get_function()
        f = subs(f, x, tuple(t))

        self.f: AnyFunction = 0

        if isinstance(f, tuple):
            if len(f) == self.integral_domain.tdim:
                self.f = mappings.contravariant(
                    f, integral_domain.get_map_to_self(), integral_domain.get_inverse_map_to_self(),
                    integral_domain.tdim)
            else:
                assert len(f) == self.integral_domain.tdim ** 2
                self.f = mappings.double_contravariant(
                    f, integral_domain.get_map_to_self(), integral_domain.get_inverse_map_to_self(),
                    integral_domain.tdim)
        else:
            self.f = f

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell."""
        return tuple(sympy.Rational(sum(i), len(i)) for i in zip(*self.integral_domain.vertices))

    def eval(self, function: AnyFunction, symbolic: bool = True) -> ScalarValueOrFloat:
        """Apply the functional to a function."""
        point = [i for i in self.integral_domain.origin]
        for i, a in enumerate(zip(*self.integral_domain.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        v1 = subs(function, x, point)
        integrand = self.dot(v1)
        value = self.integral_domain.integral(integrand)
        assert isinstance(value, (int, sympy.core.expr.Expr))
        if symbolic:
            return value
        else:
            return float(value)

    def dot(self, function: AnyFunction) -> ScalarValue:
        """Dot a function with the moment function."""
        if isinstance(self.f, tuple):
            if isinstance(function, sympy.Matrix):
                function = tuple(function[i, j]
                                 for i in range(function.rows) for j in range(function.cols))
            assert isinstance(function, tuple)
            return vdot(function, self.f)
        assert isinstance(self.f, (int, sympy.core.expr.Expr))
        assert isinstance(function, (int, sympy.core.expr.Expr))
        return function * self.f

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = self.entity_tex()
        entity_def = self.entity_definition()
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

    def __init__(self, reference: Reference, integral_domain: Reference,
                 f: typing.Union[BasisFunction, AnyFunction],
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, entity, mapping)
        self.integral_domain = integral_domain

        if isinstance(f, BasisFunction):
            f = f.get_function()
        f = subs(f, x, tuple(t))
        assert isinstance(f, (int, sympy.core.expr.Expr))
        self.f: ScalarFunction = f

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell."""
        return tuple(sympy.Rational(sum(i), len(i)) for i in zip(*self.integral_domain.vertices))

    def eval(self, function: AnyFunction, symbolic: bool = True) -> ScalarValueOrFloat:
        """Apply the functional to a function."""
        point = [i for i in self.integral_domain.origin]
        for i, a in enumerate(zip(*self.integral_domain.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        assert isinstance(function, tuple)
        v1 = subs(div(function), x, point)
        assert isinstance(v1, (int, sympy.core.expr.Expr))
        integrand = self.dot(v1)
        value = self.integral_domain.integral(integrand)
        assert isinstance(value, (int, sympy.core.expr.Expr))
        if symbolic:
            return value
        else:
            return float(value)

    def dot(self, function: ScalarFunction) -> ScalarValue:
        """Dot a function with the moment function."""
        assert isinstance(function, (int, sympy.core.expr.Expr))
        return function * self.f

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = self.entity_tex()
        entity_def = self.entity_definition()
        desc = "\\mathbf{v}\\mapsto"
        desc += f"\\displaystyle\\int_{{{entity}}}"
        if self.f != 1:
            desc += "(" + _to_tex(self.f, True) + ")"
        desc += "\\nabla\\cdot\\mathbf{v}"
        return desc, [entity_def]

    name = "Integral of divergence against"


class IntegralOfDirectionalMultiderivative(BaseFunctional):
    """An integral of a directional derivative of a scalar function."""

    def __init__(self, reference: Reference, integral_domain: Reference, directions: SetOfPoints,
                 orders: typing.Tuple[int], entity: typing.Tuple[int, int], scale: int = 1,
                 mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, entity, mapping)
        self.integral_domain = integral_domain
        self.directions = directions
        self.orders = orders
        self.scale = scale

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell."""
        return tuple(sympy.Rational(sum(i), len(i)) for i in zip(*self.integral_domain.vertices))

    def eval(self, function: AnyFunction, symbolic: bool = True) -> ScalarValueOrFloat:
        """Apply the functional to a function."""
        for dir, o in zip(self.directions, self.orders):
            for i in range(o):
                assert isinstance(function, (int, sympy.core.expr.Expr))
                function = sum(d * diff(function, x[j]) for j, d in enumerate(dir))
        point = [i for i in self.integral_domain.origin]
        for i, a in enumerate(zip(*self.integral_domain.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        integrand = self.scale * subs(function, x, point)
        assert isinstance(integrand, (int, sympy.core.expr.Expr))
        value = self.integral_domain.integral(integrand)
        assert isinstance(value, (int, sympy.core.expr.Expr))
        if symbolic:
            return value
        else:
            return float(value)

    def perform_mapping(
        self, fs: ListOfAnyFunctions, map: PointType, inverse_map: PointType, tdim: int
    ) -> ListOfAnyFunctions:
        """Map functions to a cell."""
        if sum(self.orders) > 0:
            raise NotImplementedError("Mapping high order derivatives not implemented")
        return super().perform_mapping(fs, map, inverse_map, tdim)

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = self.entity_tex()
        entity_def = self.entity_definition()
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

    def __init__(self, reference: Reference, integral_domain: Reference,
                 f: typing.Union[BasisFunction, AnyFunction], dof: BaseFunctional,
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, entity, mapping)
        self.integral_domain = integral_domain
        self.dof = dof

        if isinstance(f, BasisFunction):
            f = f.get_function()
        f = subs(f, x, tuple(t))

        self.f: AnyFunction = 0

        if isinstance(f, tuple):
            if len(f) == self.integral_domain.tdim:
                self.f = mappings.contravariant(
                    f, integral_domain.get_map_to_self(), integral_domain.get_inverse_map_to_self(),
                    integral_domain.tdim)
            else:
                assert len(f) == self.integral_domain.tdim ** 2
                self.f = mappings.double_contravariant(
                    f, integral_domain.get_map_to_self(), integral_domain.get_inverse_map_to_self(),
                    integral_domain.tdim)
        else:
            self.f = f

    def eval(self, function: AnyFunction, symbolic: bool = True) -> ScalarValueOrFloat:
        """Apply the functional to a function."""
        point = [i for i in self.integral_domain.origin]
        for i, a in enumerate(zip(*self.integral_domain.axes)):
            for j, k in zip(a, t):
                point[i] += j * k

        v1 = subs(function, x, point)
        integrand: AnyFunction = self.dot(v1)
        if isinstance(integrand, PiecewiseFunction):
            integrand = integrand.get_piece(self.integral_domain.midpoint())
        assert isinstance(integrand, (int, sympy.core.expr.Expr))
        value = self.integral_domain.integral(integrand)
        assert isinstance(value, (int, sympy.core.expr.Expr))
        if symbolic:
            return value
        else:
            return float(value)

    def dot(self, function: AnyFunction) -> ScalarValue:
        """Dot a function with the moment function."""
        if isinstance(function, tuple):
            assert isinstance(self.f, tuple)
            return vdot(function, self.f)

        if isinstance(function, sympy.Matrix):
            result = 0
            for i in range(function.rows):
                for j in range(function.cols):
                    if isinstance(self.f, sympy.Matrix):
                        result += function[i, j] * self.f[i, j]
                    else:
                        assert isinstance(self.f, tuple)
                        result += function[i, j] * self.f[i * function.cols + j]
            return result

        assert isinstance(self.f, (int, sympy.core.expr.Expr))
        assert isinstance(function, (int, sympy.core.expr.Expr))
        return function * self.f

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell."""
        p = self.dof.dof_point()
        return tuple(
            o + sum(self.integral_domain.axes[j][i] * c for j, c in enumerate(p))
            for i, o in enumerate(self.integral_domain.origin)
        )

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF."""
        p = self.dof.dof_direction()
        if p is None:
            return None
        return tuple(
            sum(self.integral_domain.axes[j][i] * c for j, c in enumerate(p))
            for i in range(self.integral_domain.gdim)
        )

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = self.entity_tex()
        entity_def = self.entity_definition()
        if isinstance(self.f, tuple):
            if len(self.f) in [self.reference.tdim, self.integral_domain.tdim]:
                desc = "\\boldsymbol{v}\\mapsto"
                desc += f"\\displaystyle\\int_{{{entity}}}"
                desc += "\\boldsymbol{v}\\cdot"
                desc += "\\left(\\begin{array}{c}"
                desc += "\\\\".join([_to_tex(i) for i in self.f])
                desc += "\\end{array}\\right)"
            else:
                if len(self.f) == self.integral_domain.tdim ** 2:
                    size = self.integral_domain.tdim
                elif len(self.f) == self.reference.tdim ** 2:
                    size = self.reference.tdim
                else:
                    raise NotImplementedError()
                desc = "\\mathbf{V}\\mapsto"
                desc += f"\\displaystyle\\int_{{{entity}}}"
                desc += "\\mathbf{V}:"
                desc += "\\left(\\begin{array}{" + "c" * size + "}"
                desc += "\\\\".join(["&".join(
                    [_to_tex(self.f[i]) for i in range(size * row, size * (row + 1))]
                ) for row in range(size)])
                desc += "\\end{array}\\right)"
        else:
            desc = "v\\mapsto"
            desc += f"\\displaystyle\\int_{{{entity}}}"
            if self.f != 1:
                desc += "(" + _to_tex(self.f) + ")"
            desc += "v"
        return desc, [entity_def]

    name = "Integral moment"


class VecIntegralMoment(IntegralMoment):
    """An integral moment applied to a component of a vector."""

    def __init__(self, reference: Reference, integral_domain: Reference, f: ScalarFunction,
                 dot_with: VectorFunction, dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, integral_domain, f, dof, entity=entity, mapping=mapping)
        self.dot_with = dot_with

    def dot(self, function: AnyFunction) -> ScalarValue:
        """Dot a function with the moment function."""
        assert isinstance(self.f, (int, sympy.core.expr.Expr))

        if isinstance(function, sympy.Matrix):
            result = 0
            for i in range(function.rows):
                for j in range(function.cols):
                    result += function[i, j] * self.dot_with[i * function.cols + j]
            return result * self.f

        if isinstance(function, PiecewiseFunction):
            function = tuple(function._iter_list())

        assert isinstance(function, tuple)
        return vdot(function, self.dot_with) * self.f

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF."""
        return self.dot_with

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = self.entity_tex()
        entity_def = self.entity_definition()
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

    def __init__(self, reference: Reference, integral_domain: Reference, f: ScalarFunction,
                 dot_with: VectorFunction, dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, integral_domain, f, dof, entity=entity, mapping=mapping)
        self.dot_with = dot_with

    def dot(self, function: AnyFunction) -> ScalarValue:
        """Dot a function with the moment function."""
        assert isinstance(function, tuple)
        assert isinstance(self.f, (int, sympy.core.expr.Expr))
        return vdot(function, self.dot_with) * self.f

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF."""
        return self.dot_with

    def eval(self, function: AnyFunction, symbolic: bool = True) -> ScalarValueOrFloat:
        """Apply the functional to a function."""
        point = [i for i in self.integral_domain.origin]
        for i, a in enumerate(zip(*self.integral_domain.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        assert isinstance(function, (int, sympy.core.expr.Expr))
        integrand = self.dot(subs(grad(function, self.integral_domain.gdim), x, point))
        value = self.integral_domain.integral(integrand)
        assert isinstance(value, (int, sympy.core.expr.Expr))
        if symbolic:
            return value
        else:
            return float(value)

    name = "Derivative integral moment"


class DivergenceIntegralMoment(IntegralMoment):
    """An integral moment of the divergence of a vector function."""

    def __init__(self, reference: Reference, integral_domain: Reference, f: ScalarFunction,
                 dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, integral_domain, f, dof, entity=entity, mapping=mapping)

    def eval(self, function: AnyFunction, symbolic: bool = True) -> ScalarValueOrFloat:
        """Apply the functional to a function."""
        point = [i for i in self.integral_domain.origin]
        for i, a in enumerate(zip(*self.integral_domain.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        assert isinstance(function, tuple)
        integrand = self.dot(subs(div(function), x, point))
        value = self.integral_domain.integral(integrand)
        assert isinstance(value, (int, sympy.core.expr.Expr))
        if symbolic:
            return value
        else:
            return float(value)

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = self.entity_tex()
        entity_def = self.entity_definition()
        desc = "\\boldsymbol{v}\\mapsto"
        desc += f"\\displaystyle\\int_{{{entity}}}"
        if self.f != 1:
            desc += "(" + _to_tex(self.f, True) + ")"
        desc += "\\nabla\\cdot\\boldsymbol{v}"
        return desc, [entity_def]

    name = "Integral moment of divergence"


class TangentIntegralMoment(VecIntegralMoment):
    """An integral moment in the tangential direction."""

    def __init__(self, reference: Reference, integral_domain: Reference, f: ScalarFunction,
                 dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "covariant"):
        super().__init__(reference, integral_domain, f, integral_domain.tangent(), dof,
                         entity=entity, mapping=mapping)

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = self.entity_tex()
        entity_n = self.entity_number()
        entity_def = self.entity_definition()
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

    def __init__(self, reference: Reference, integral_domain: Reference, f: ScalarFunction,
                 dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "contravariant"):
        super().__init__(reference, integral_domain, f, integral_domain.normal(), dof,
                         entity=entity, mapping=mapping)

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = self.entity_tex()
        entity_n = self.entity_number()
        entity_def = self.entity_definition()
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

    def __init__(self, reference: Reference, integral_domain: Reference, f: ScalarFunction,
                 dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, integral_domain, f, integral_domain.normal(), dof,
                         entity=entity, mapping=mapping)

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = self.entity_tex()
        entity_n = self.entity_number()
        entity_def = self.entity_definition()
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

    def __init__(self, reference: Reference, integral_domain: Reference, f: ScalarFunction,
                 inner_with_left: VectorFunction, inner_with_right: VectorFunction,
                 dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "identity"):
        super().__init__(reference, integral_domain, f, dof, entity=entity, mapping=mapping)
        self.inner_with_left = inner_with_left
        self.inner_with_right = inner_with_right

    def dot(self, function: AnyFunction) -> ScalarValue:
        """Take the inner product of a function with the moment direction."""
        if isinstance(function, sympy.Matrix):
            function = tuple(function[i, j]
                             for i in range(function.rows) for j in range(function.cols))
        assert isinstance(function, tuple)
        tdim = len(self.inner_with_left)
        assert isinstance(self.f, (int, sympy.core.expr.Expr))
        return vdot(self.inner_with_left,
                    tuple(vdot(function[tdim * i: tdim * (i + 1)], self.inner_with_right)
                          for i in range(0, tdim))) * self.f * self.integral_domain.jacobian()

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF."""
        if self.inner_with_left != self.inner_with_right:
            return None
        return self.inner_with_left

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = self.entity_tex()
        entity_def = self.entity_definition()
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

    def __init__(self, reference: Reference, integral_domain: Reference, f: ScalarFunction,
                 dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "double_contravariant"):
        super().__init__(reference, integral_domain, f, integral_domain.normal(),
                         integral_domain.normal(), dof, entity=entity, mapping=mapping)

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved."""
        entity = self.entity_tex()
        entity_n = self.entity_number()
        entity_def = self.entity_definition()
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
