"""Functionals used to define the dual sets."""

import typing
from abc import ABC, abstractmethod

import sympy

from . import mappings
from .functions import (AnyFunction, FunctionInput, ScalarFunction, VectorFunction,
                        parse_function_input)
from .geometry import PointType, SetOfPoints
from .piecewise_functions import PiecewiseFunction
from .references import Interval, NonDefaultReferenceError, Reference
from .symbols import t, x

ScalarValueOrFloat = typing.Union[sympy.core.expr.Expr, float]


def _to_tex(f: FunctionInput, tfrac: bool = False) -> str:
    r"""Convert an expresson to TeX.

    Args:
        f: the function
        tfrac: Should \\tfrac be used in the place of \\frac?

    Returns:
        The function as TeX
    """
    out = parse_function_input(f).as_tex()

    if tfrac:
        return out.replace("\\frac", "\\tfrac")
    else:
        return out


def _nth(n: int) -> str:
    """Add th, st or nd to a number.

    Args:
        n: The number

    Returns:
        The string (n)th, (n)st, (n)rd or (n)nd
    """
    if n % 10 == 1 and n != 11:
        return f"{n}st"
    if n % 10 == 2 and n != 12:
        return f"{n}nd"
    if n % 10 == 3 and n != 12:
        return f"{n}rd"
    return f"{n}th"


class BaseFunctional(ABC):
    """A functional."""

    def __init__(self, reference: Reference, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None]):
        """Create the functional.

        Args:
            reference: The reference cell
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        self.reference = reference
        self.entity = entity
        self.mapping = mapping

    def entity_dim(self) -> int:
        """Get the dimension of the entity this DOF is associated with.

        Returns:
            The dimension of the entity this DOF is associated with
        """
        return self.entity[0]

    def entity_number(self) -> int:
        """Get the number of the entity this DOF is associated with.

        Returns:
            The number of the entity this DOF is associated with
        """
        return self.entity[1]

    def perform_mapping(
        self, fs: typing.List[AnyFunction], map: PointType, inverse_map: PointType
    ) -> typing.List[AnyFunction]:
        """Map functions to a cell.

        Args:
            fs: functions
            map: Map from the reference cell to a physical cell
            inverse_map: Map to the reference cell from a physical cell

        Returns:
            Mapped functions
        """
        assert self.mapping is not None
        return [getattr(mappings, self.mapping)(f, map, inverse_map) for f in fs]

    def entity_tex(self) -> str:
        """Get the entity the DOF is associated with in TeX format.

        Returns:
            TeX representation of entity
        """
        if self.entity[0] == self.reference.tdim:
            return "R"
        else:
            return f"{'vefc'[self.entity[0]]}_{{{self.entity[1]}}}"

    def entity_definition(self) -> str:
        """Get the definition of the entity the DOF is associated with.

        Returns:
            The definition
        """
        if self.entity[0] == self.reference.tdim:
            return "\\(R\\) is the reference element"
        else:
            desc = f"\\({self.entity_tex()}\\) is the {_nth(self.entity[1])} "
            desc += ['vertex', 'edge', 'face', 'volume'][self.entity[0]]
            return desc

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF.

        Returns:
            The direction
        """
        return None

    def eval(self, function: AnyFunction, symbolic: bool = True) -> typing.Union[
        ScalarFunction, float
    ]:
        """Apply to the functional to a function.

        Args:
            function: The function
            symbolic: Should it be applied symbolically?

        Returns:
            The value of the functional for the function
        """
        value = self.eval_symbolic(function)
        if symbolic:
            return value
        else:
            return float(value)

    @abstractmethod
    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell.

        Returns:
            The point
        """
        pass

    def adjusted_dof_point(self) -> PointType:
        """Get the adjusted position of the DOF in the cell for plotting.

        Returns:
            The point
        """
        point = self.dof_point()
        if self.entity[0] == 0:
            return point

        midpoint = self.reference.sub_entity(*self.entity).midpoint()
        return tuple(
            m + sympy.Rational(7, 10) * (p - m)
            for m, p in zip(midpoint, point))

    def eval_symbolic(self, function: AnyFunction) -> ScalarFunction:
        """Symbolically apply the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        e = self._eval_symbolic(function)
        assert isinstance(e, ScalarFunction)
        es = e.as_sympy()
        assert isinstance(es, sympy.core.expr.Expr)
        assert es.is_constant()
        return e

    @abstractmethod
    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply to the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        pass

    @abstractmethod
    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
        pass

    name = "Base functional"


class PointEvaluation(BaseFunctional):
    """A point evaluation."""

    def __init__(self, reference: Reference, point_in: FunctionInput,
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        """Create the functional.

        Args:
            reference: The reference cell
            point_in: The point
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        super().__init__(reference, entity, mapping)
        self.point = parse_function_input(point_in)
        assert self.point.is_vector

    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply to the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        return function.subs(x, self.point)

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell.

        Returns:
            The point
        """
        pt = self.point.as_sympy()
        assert isinstance(pt, tuple)
        return pt

    def adjusted_dof_point(self) -> PointType:
        """Get the adjusted position of the DOF in the cell for plotting.

        Returns:
            The point
        """
        return self.dof_point()

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
        assert isinstance(self.point, VectorFunction)
        return f"v\\mapsto v({','.join([_to_tex(i, True) for i in self.point])})", []

    name = "Point evaluation"


class WeightedPointEvaluation(BaseFunctional):
    """A point evaluation."""

    def __init__(self, reference: Reference, point_in: FunctionInput, weight: sympy.core.expr.Expr,
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        """Create the functional.

        Args:
            reference: The reference cell
            point_in: The point
            weight: The weight
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        super().__init__(reference, entity, mapping)
        self.point = parse_function_input(point_in)
        assert self.point.is_vector
        self.weight = weight

    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply to the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        return function.subs(x, self.point) * self.weight

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell.

        Returns:
            The point
        """
        pt = self.point.as_sympy()
        assert isinstance(pt, tuple)
        return pt

    def adjusted_dof_point(self) -> PointType:
        """Get the adjusted position of the DOF in the cell for plotting.

        Returns:
            The point
        """
        return self.dof_point()

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
        assert isinstance(self.point, VectorFunction)
        return (f"v\\mapsto {_to_tex(self.weight)} "
                f"v({','.join([_to_tex(i, True) for i in self.point])})"), []

    name = "Weighted point evaluation"


class DerivativePointEvaluation(BaseFunctional):
    """A point evaluation of a given derivative."""

    def __init__(self, reference: Reference, point_in: FunctionInput,
                 derivative: typing.Tuple[int, ...],
                 entity: typing.Tuple[int, int], mapping: typing.Optional[str] = None):
        """Create the functional.

        Args:
            reference: The reference cell
            point_in: The point
            derivative: The order(s) of the derivative
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        super().__init__(reference, entity, mapping)
        self.point = parse_function_input(point_in)
        assert self.point.is_vector
        self.derivative = derivative

    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply to the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        for i, j in zip(x, self.derivative):
            for k in range(j):
                function = function.diff(i)
        return function.subs(x, self.point)

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell.

        Returns:
            The point
        """
        pt = self.point.as_sympy()
        assert isinstance(pt, tuple)
        return pt

    def adjusted_dof_point(self) -> PointType:
        """Get the adjusted position of the DOF in the cell for plotting.

        Returns:
            The point
        """
        return self.dof_point()

    def perform_mapping(
        self, fs: typing.List[AnyFunction], map: PointType, inverse_map: PointType
    ) -> typing.List[AnyFunction]:
        """Map functions to a cell.

        Args:
            fs: functions
            map: Map from the reference cell to a physical cell
            inverse_map: Map to the reference cell from a physical cell

        Returns:
            Mapped functions
        """
        raise mappings.MappingNotImplemented()

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
        assert isinstance(self.point, VectorFunction)
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

    def __init__(self, reference: Reference, point_in: FunctionInput, direction_in: FunctionInput,
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        """Create the functional.

        Args:
            reference: The reference cell
            point_in: The points
            direction_in: The diection
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        super().__init__(reference, entity, mapping)
        self.point = parse_function_input(point_in)
        assert self.point.is_vector
        self.dir = parse_function_input(direction_in)

    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply to the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        d = self.dir.as_sympy()
        assert isinstance(d, tuple)
        return function.directional_derivative(d).subs(x, self.dof_point())

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell.

        Returns:
            The point
        """
        pt = self.point.as_sympy()
        assert isinstance(pt, tuple)
        return pt

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF.

        Returns:
            The direction
        """
        d = self.dir.as_sympy()
        assert isinstance(d, tuple)
        return d

    def adjusted_dof_point(self) -> PointType:
        """Get the adjusted position of the DOF in the cell for plotting.

        Returns:
            The point
        """
        return self.dof_point()

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
        assert isinstance(self.point, VectorFunction)
        assert isinstance(self.dir, VectorFunction)
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

    def __init__(self, reference: Reference, point_in: FunctionInput, edge: Reference,
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        """Create the functional.

        Args:
            reference: The reference cell
            point_in: The point
            edge: The edge of the cell
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        assert isinstance(edge, Interval)
        super().__init__(reference, point_in, edge.normal(), entity=entity, mapping=mapping)
        self.reference = edge

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
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

    def __init__(self, reference: Reference, point_in: FunctionInput,
                 component: typing.Tuple[int, int],
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        """Create the functional.

        Args:
            reference: The reference cell
            point_in: The point
            component: The compenent of the derivative
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        super().__init__(reference, entity, mapping)
        self.point = parse_function_input(point_in)
        assert self.point.is_vector
        self.component = component

    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply to the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        return function.jacobian_component(self.component).subs(x, self.point)

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell.

        Returns:
            The point
        """
        pt = self.point.as_sympy()
        assert isinstance(pt, tuple)
        return pt

    def adjusted_dof_point(self) -> PointType:
        """Get the adjusted position of the DOF in the cell for plotting.

        Returns:
            The point
        """
        return self.dof_point()

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
        desc = "v\\mapsto"
        desc += "\\frac{\\partial^2v}{"
        for c in self.component:
            desc += "\\partial " + "xyz"[c]
        desc += "}(" + ",".join([_to_tex(i, True) for i in self.dof_point()]) + ")"
        return desc, []

    name = "Point evaluation of Jacobian component"


class PointInnerProduct(BaseFunctional):
    """An evaluation of an inner product at a point."""

    def __init__(self, reference: Reference, point_in: FunctionInput, lvec: FunctionInput,
                 rvec: FunctionInput,
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        """Create the functional.

        Args:
            reference: The reference cell
            point_in: The points
            lvec: The vector to multiply on the left
            rvec: The vector to multiply on the right
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        super().__init__(reference, entity, mapping)
        self.point = parse_function_input(point_in)
        assert self.point.is_vector
        self.lvec = parse_function_input(lvec)
        self.rvec = parse_function_input(rvec)
        assert self.lvec.is_vector
        assert self.rvec.is_vector

    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply to the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        v = function.subs(x, self.point)

        return self.lvec.dot(v @ self.rvec)

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell.

        Returns:
            The point
        """
        pt = self.point.as_sympy()
        assert isinstance(pt, tuple)
        return pt

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF.

        Returns:
            The direction
        """
        if self.rvec != self.lvec:
            return None
        lv = self.lvec.as_sympy()
        assert isinstance(lv, tuple)
        return lv

    def adjusted_dof_point(self) -> PointType:
        """Get the adjusted position of the DOF in the cell for plotting.

        Returns:
            The point
        """
        return self.dof_point()

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
        assert isinstance(self.lvec, VectorFunction)
        assert isinstance(self.rvec, VectorFunction)
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

    def __init__(self, reference: Reference, point_in: FunctionInput, vector_in: FunctionInput,
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        """Create the functional.

        Args:
            reference: The reference cell
            point_in: The points
            vector_in: The vector to dot with
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        super().__init__(reference, entity, mapping)
        self.point = parse_function_input(point_in)
        assert self.point.is_vector
        self.vector = parse_function_input(vector_in)

    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply to the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        v1 = function.subs(x, self.point)
        v2 = self.vector.subs(x, self.point)
        return v1.dot(v2)

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell.

        Returns:
            The point
        """
        pt = self.point.as_sympy()
        assert isinstance(pt, tuple)
        return pt

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF.

        Returns:
            The direction
        """
        v = self.vector.as_sympy()
        if isinstance(v, tuple):
            return v
        return None

    def adjusted_dof_point(self) -> PointType:
        """Get the adjusted position of the DOF in the cell for plotting.

        Returns:
            The point
        """
        return self.dof_point()

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
        desc = "\\boldsymbol{v}\\mapsto"
        desc += "\\boldsymbol{v}(" + ",".join([_to_tex(i, True) for i in self.dof_point()]) + ")"
        if self.vector.is_vector:
            desc += "\\cdot\\left(\\begin{array}{c}"
            assert hasattr(self.vector, "__iter__")
            desc += "\\\\".join([_to_tex(i) for i in self.vector])
            desc += "\\end{array}\\right)"
        elif self.vector != 1:
            desc += f"\\cdot{_to_tex(self.vector)}"
        return desc, []

    name = "Dot point evaluation"


class PointDivergenceEvaluation(BaseFunctional):
    """A point evaluation of the divergence."""

    def __init__(self, reference: Reference, point_in: FunctionInput,
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        """Create the functional.

        Args:
            reference: The reference cell
            point_in: The points
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        super().__init__(reference, entity, mapping)
        self.point = parse_function_input(point_in)
        assert self.point.is_vector

    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply to the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        out = ScalarFunction(0)
        if isinstance(function, PiecewiseFunction):
            pt = self.point.as_sympy()
            assert isinstance(pt, tuple)
            function = function.get_piece(pt)
        assert isinstance(function, VectorFunction)
        for f, i in zip(function, x):
            out += f.diff(i)
        return out.subs(x, self.point)

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell.

        Returns:
            The point
        """
        pt = self.point.as_sympy()
        assert isinstance(pt, tuple)
        return pt

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF.

        Returns:
            The direction
        """
        return None

    def adjusted_dof_point(self) -> PointType:
        """Get the adjusted position of the DOF in the cell for plotting.

        Returns:
            The point
        """
        return self.dof_point()

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
        desc = "\\boldsymbol{v}\\mapsto"
        desc += "\\nablaa\\cdot\\boldsymbol{v}"
        desc += "(" + ",".join([_to_tex(i, True) for i in self.dof_point()]) + ")"
        return desc, []

    name = "Point evaluation of divergence"


class IntegralAgainst(BaseFunctional):
    """An integral against a function."""

    f: AnyFunction

    def __init__(self, reference: Reference, f_in: FunctionInput,
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        """Create the functional.

        Args:
            reference: The reference cell
            f_in: The function to multiply with
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        super().__init__(reference, entity, mapping)
        self.integral_domain = reference.sub_entity(*entity)

        f = parse_function_input(f_in) / reference.jacobian()
        f = f.subs(x, t)

        if f.is_vector:
            assert len(f) == self.integral_domain.tdim
            self.f = mappings.contravariant(
                f, self.integral_domain.get_map_to_self(),
                self.integral_domain.get_inverse_map_to_self())
        elif f.is_matrix:
            assert f.shape[0] == self.integral_domain.tdim
            assert f.shape[1] == self.integral_domain.tdim
            self.f = mappings.double_contravariant(
                f, self.integral_domain.get_map_to_self(),
                self.integral_domain.get_inverse_map_to_self())
        else:
            self.f = f

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell.

        Returns:
            The point
        """
        return tuple(sympy.Rational(sum(i), len(i)) for i in zip(*self.integral_domain.vertices))

    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply to the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        point = [i for i in self.integral_domain.origin]
        for i, a in enumerate(zip(*self.integral_domain.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        v1 = function.subs(x, point)
        integrand = self.dot(v1)
        return integrand.integral(self.integral_domain)

    def dot(self, function: AnyFunction) -> ScalarFunction:
        """Dot a function with the moment function.

        Args:
            function: The function

        Returns:
            The product of the function and the moment function
        """
        return function.dot(self.f)

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
        entity = self.entity_tex()
        entity_def = self.entity_definition()
        if self.f.is_vector:
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

    def __init__(self, reference: Reference, f_in: FunctionInput,
                 entity: typing.Tuple[int, int], mapping: typing.Union[str, None] = "identity"):
        """Create the functional.

        Args:
            reference: The reference cell
            f_in: The function to multiply with
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        super().__init__(reference, entity, mapping)
        self.integral_domain = reference.sub_entity(*entity)

        f = parse_function_input(f_in)
        self.f = f.subs(x, t)

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell.

        Returns:
            The point
        """
        return tuple(sympy.Rational(sum(i), len(i)) for i in zip(*self.integral_domain.vertices))

    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply to the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        point = [i for i in self.integral_domain.origin]
        for i, a in enumerate(zip(*self.integral_domain.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        v1 = function.div().subs(x, point)
        integrand = self.dot(v1)
        return integrand.integral(self.integral_domain)

    def dot(self, function: ScalarFunction) -> ScalarFunction:
        """Dot a function with the moment function.

        Args:
            function: The function

        Returns:
            The product of the function and the moment function
        """
        return function * self.f

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
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

    def __init__(self, reference: Reference, directions: SetOfPoints,
                 orders: typing.Tuple[int, ...], entity: typing.Tuple[int, int], scale: int = 1,
                 mapping: typing.Union[str, None] = "identity"):
        """Create the functional.

        Args:
            reference: The reference cell
            integral_domain: The domain of the integral
            directions: The diections of the derivatives
            orders: The orders of the derivatives
            entity: The entity the functional is associated with
            scale: The scale factor
            mapping: The type of mappping from the reference cell to a physical cell
        """
        super().__init__(reference, entity, mapping)
        self.integral_domain = reference.sub_entity(*entity)
        self.directions = directions
        self.orders = orders
        self.scale = scale

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell.

        Returns:
            The point
        """
        return tuple(sympy.Rational(sum(i), len(i)) for i in zip(*self.integral_domain.vertices))

    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply to the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        for dir, o in zip(self.directions, self.orders):
            for i in range(o):
                function = function.grad(len(dir)).dot(VectorFunction(dir))
        point = [i for i in self.integral_domain.origin]
        for i, a in enumerate(zip(*self.integral_domain.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        integrand = ScalarFunction(self.scale) * function.subs(x, point)
        return integrand.integral(self.integral_domain)

    def perform_mapping(
        self, fs: typing.List[AnyFunction], map: PointType, inverse_map: PointType
    ) -> typing.List[AnyFunction]:
        """Map functions to a cell.

        Args:
            fs: functions
            map: Map from the reference cell to a physical cell
            inverse_map: Map to the reference cell from a physical cell

        Returns:
            Mapped functions
        """
        raise mappings.MappingNotImplemented()

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
        entity = self.entity_tex()
        entity_def = self.entity_definition()
        desc = "\\mathbf{V}\\mapsto"
        desc += "\\displaystyle "
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

    f: AnyFunction

    def __init__(self, reference: Reference,
                 f_in: FunctionInput, dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "identity", map_function: bool = True):
        """Create the functional.

        Args:
            reference: The reference cell
            integral_domain: The domain of the integral
            f_in: The function to multiply with
            dof: The DOF in a moment space that the function is associated with
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
            map_function: Should the function be mapped?
        """
        super().__init__(reference, entity, mapping)
        self.integral_domain = reference.sub_entity(*entity)
        self.dof = dof

        id_def = reference.default_reference().sub_entity(*entity)

        f = parse_function_input(f_in)
        self.f = f.subs(x, t)

        # Map from reference entity to entity
        if id_def.default_reference() != id_def:
            if self.f.is_vector and len(self.f) != reference.gdim:
                self.f = mappings.contravariant(
                    self.f, id_def.get_map_to_self(),
                    id_def.get_inverse_map_to_self())
            elif self.f.is_matrix:
                assert self.f.shape[0] == id_def.tdim
                assert self.f.shape[1] == id_def.tdim
                self.f = mappings.double_contravariant(
                    self.f, id_def.get_map_to_self(),
                    id_def.get_inverse_map_to_self())

        # Map from default reference to reference
        if map_function and reference != reference.default_reference():
            if mapping is None or not hasattr(mappings, f"{mapping}_inverse_transpose"):
                raise NonDefaultReferenceError()
            mf = getattr(mappings, f"{mapping}_inverse_transpose")
            self.f = mf(self.f, reference.get_map_to_self(),
                        reference.get_inverse_map_to_self(),
                        substitute=False)
            self.f *= id_def.volume() / self.integral_domain.volume()

    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply to the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        point = [i for i in self.integral_domain.origin]
        for i, a in enumerate(zip(*self.integral_domain.axes)):
            for j, k in zip(a, t):
                point[i] += j * k

        v1 = function.subs(x, point)
        integrand = self.dot(v1)
        return integrand.integral(self.integral_domain)

    def dot(self, function: AnyFunction) -> ScalarFunction:
        """Dot a function with the moment function.

        Args:
            function: The function

        Returns:
            The product of the function and the moment function
        """
        return function.dot(self.f)

    def dof_point(self) -> PointType:
        """Get the location of the DOF in the cell.

        Returns:
            The point
        """
        p = self.dof.dof_point()
        return tuple(
            o + sum(self.integral_domain.axes[j][i] * c for j, c in enumerate(p))
            for i, o in enumerate(self.integral_domain.origin)
        )

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF.

        Returns:
            The direction
        """
        p = self.dof.dof_direction()
        if p is None:
            if self.f.is_vector:
                p = (self.f.subs(t, self.dof_point())).as_sympy()
                assert isinstance(p, tuple)
                return p
            return None
        vp = VectorFunction(p)
        out = []
        for i in range(self.integral_domain.gdim):
            entry = vp.dot(VectorFunction([a[i] for a in self.integral_domain.axes])).as_sympy()
            assert isinstance(entry, sympy.core.expr.Expr)
            out.append(entry)
        return tuple(out)

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
        entity = self.entity_tex()
        entity_def = self.entity_definition()
        if self.f.is_vector:
            if len(self.f) in [self.reference.tdim, self.integral_domain.tdim]:
                desc = "\\boldsymbol{v}\\mapsto"
                desc += f"\\displaystyle\\int_{{{entity}}}"
                desc += "\\boldsymbol{v}\\cdot"
                desc += "\\left(\\begin{array}{c}"
                assert hasattr(self.f, "__iter__")
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


class DerivativeIntegralMoment(IntegralMoment):
    """An integral moment of the derivative of a scalar function."""

    def __init__(self, reference: Reference, f: FunctionInput,
                 dot_with_in: FunctionInput, dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "identity"):
        """Create the functional.

        Args:
            reference: The reference cell
            f: The function to multiply with
            dot_with_in: The vector to take the dot product with
            dof: The DOF in a moment space that the function is associated with
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        super().__init__(reference, f, dof, entity=entity, mapping=mapping)
        self.dot_with = parse_function_input(dot_with_in)

    def dot(self, function: AnyFunction) -> ScalarFunction:
        """Dot a function with the moment function.

        Args:
            function: The function

        Returns:
            The product of the function and the moment function
        """
        assert function.is_vector
        assert self.f.is_scalar
        return function.dot(self.dot_with) * self.f

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF.

        Returns:
            The direction
        """
        dw = self.dot_with.as_sympy()
        assert isinstance(dw, tuple)
        return dw

    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply to the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        point = [i for i in self.integral_domain.origin]
        for i, a in enumerate(zip(*self.integral_domain.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        integrand = self.dot(function.grad(self.integral_domain.gdim).subs(x, point))
        value = integrand.integral(self.integral_domain)
        return value

    name = "Derivative integral moment"


class DivergenceIntegralMoment(IntegralMoment):
    """An integral moment of the divergence of a vector function."""

    def __init__(self, reference: Reference, f_in: FunctionInput,
                 dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "identity"):
        """Create the functional.

        Args:
            reference: The reference cell
            f_in: The function to multiply with
            dof: The DOF in a moment space that the function is associated with
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        f = parse_function_input(f_in)
        assert f.is_scalar
        super().__init__(reference, f, dof, entity=entity, mapping=mapping)

    def _eval_symbolic(self, function: AnyFunction) -> AnyFunction:
        """Apply to the functional to a function.

        Args:
            function: The function

        Returns:
            The value of the functional for the function
        """
        point = [i for i in self.integral_domain.origin]
        for i, a in enumerate(zip(*self.integral_domain.axes)):
            for j, k in zip(a, t):
                point[i] += j * k
        assert function.is_vector
        integrand = self.dot(function.div().subs(x, point))
        return integrand.integral(self.integral_domain)

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
        entity = self.entity_tex()
        entity_def = self.entity_definition()
        desc = "\\boldsymbol{v}\\mapsto"
        desc += f"\\displaystyle\\int_{{{entity}}}"
        if self.f != 1:
            desc += "(" + _to_tex(self.f, True) + ")"
        desc += "\\nabla\\cdot\\boldsymbol{v}"
        return desc, [entity_def]

    name = "Integral moment of divergence"


class TangentIntegralMoment(IntegralMoment):
    """An integral moment in the tangential direction."""

    def __init__(self, reference: Reference, f_in: FunctionInput,
                 dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "covariant"):
        """Create the functional.

        Args:
            reference: The reference cell
            f_in: The function to multiply with
            dof: The DOF in a moment space that the function is associated with
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        self._scalar_f = parse_function_input(f_in).subs(x, t)
        integral_domain = reference.sub_entity(*entity)
        assert self._scalar_f.is_scalar
        super().__init__(
            reference, tuple(self._scalar_f * i for i in integral_domain.tangent()), dof,
            entity=entity, mapping=mapping, map_function=False)

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF.

        Returns:
            The direction
        """
        return self.integral_domain.tangent()

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
        entity = self.entity_tex()
        entity_n = self.entity_number()
        entity_def = self.entity_definition()
        desc = "\\boldsymbol{v}\\mapsto"
        desc += f"\\displaystyle\\int_{{{entity}}}"
        desc += "\\boldsymbol{v}\\cdot"
        if self.f != 1:
            desc += "(" + _to_tex(self._scalar_f) + ")"
        desc += "\\hat{\\boldsymbol{t}}" + f"_{{{entity_n}}}"
        return desc, [
            entity_def,
            f"\\(\\hat{{\\boldsymbol{{t}}}}_{{{entity_n}}}\\) is the tangent to edge {entity_n}"
        ]

    name = "Tangential integral moment"


class NormalIntegralMoment(IntegralMoment):
    """An integral moment in the normal direction."""

    def __init__(self, reference: Reference, f_in: FunctionInput,
                 dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "contravariant"):
        """Create the functional.

        Args:
            reference: The reference cell
            f_in: The function to multiply with
            dof: The DOF in a moment space that the function is associated with
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        self._scalar_f = parse_function_input(f_in).subs(x, t)
        assert self._scalar_f.is_scalar
        integral_domain = reference.sub_entity(*entity)
        super().__init__(
            reference, tuple(self._scalar_f * i for i in integral_domain.normal()), dof,
            entity=entity, mapping=mapping, map_function=False)

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF.

        Returns:
            The direction
        """
        return self.integral_domain.normal()

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
        entity = self.entity_tex()
        entity_n = self.entity_number()
        entity_def = self.entity_definition()
        desc = "\\boldsymbol{v}\\mapsto"
        desc += f"\\displaystyle\\int_{{{entity}}}"
        desc += "\\boldsymbol{v}\\cdot"
        if self.f != 1:
            desc += "(" + _to_tex(self._scalar_f, True) + ")"
        desc += "\\hat{\\boldsymbol{n}}" + f"_{{{entity_n}}}"
        return desc, [
            entity_def,
            f"\\(\\hat{{\\boldsymbol{{n}}}}_{{{entity_n}}}\\) is the normal to facet {entity_n}"
        ]

    name = "Normal integral moment"


class NormalDerivativeIntegralMoment(DerivativeIntegralMoment):
    """An integral moment in the normal direction."""

    def __init__(self, reference: Reference, f_in: FunctionInput,
                 dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "identity"):
        """Create the functional.

        Args:
            reference: The reference cell
            f_in: The function to multiply with
            dof: The DOF in a moment space that the function is associated with
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        f = parse_function_input(f_in)
        assert f.is_scalar
        integral_domain = reference.sub_entity(*entity)
        super().__init__(reference, f, integral_domain.normal(), dof,
                         entity=entity, mapping=mapping)

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
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

    def __init__(self, reference: Reference, f_in: FunctionInput,
                 inner_with_left_in: FunctionInput, inner_with_right_in: FunctionInput,
                 dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "identity"):
        """Create the functional.

        Args:
            reference: The reference cell
            f_in: The function to multiply with
            dof: The DOF in a moment space that the function is associated with
            inner_with_left_in: The vector to multiply on the left
            inner_with_right_in: The vector to multiply on the right
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        f = parse_function_input(f_in)
        inner_with_left = parse_function_input(inner_with_left_in)
        inner_with_right = parse_function_input(inner_with_right_in)
        assert f.is_scalar
        assert inner_with_left.is_vector
        assert inner_with_right.is_vector

        super().__init__(reference, f, dof, entity=entity, mapping=mapping)
        self.inner_with_left = inner_with_left
        self.inner_with_right = inner_with_right

    def dot(self, function: AnyFunction) -> ScalarFunction:
        """Take the inner product of a function with the moment direction.

        Args:
            function: The function

        Returns:
            The inner product of the function and the moment direction
        """
        assert function.is_matrix
        return self.inner_with_left.dot(
            function @ self.inner_with_right) * self.f * self.integral_domain.jacobian()

    def dof_direction(self) -> typing.Union[PointType, None]:
        """Get the direction of the DOF.

        Returns:
            The direction
        """
        if self.inner_with_left != self.inner_with_right:
            return None
        il = self.inner_with_left.as_sympy()
        assert isinstance(il, tuple)
        return il

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
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

    def __init__(self, reference: Reference, f_in: FunctionInput,
                 dof: BaseFunctional, entity: typing.Tuple[int, int],
                 mapping: typing.Union[str, None] = "double_contravariant"):
        """Create the functional.

        Args:
            reference: The reference cell
            f_in: The function to multiply with
            dof: The DOF in a moment space that the function is associated with
            entity: The entity the functional is associated with
            mapping: The type of mappping from the reference cell to a physical cell
        """
        f = parse_function_input(f_in)
        assert f.is_scalar
        integral_domain = reference.sub_entity(*entity)
        super().__init__(reference, f, integral_domain.normal(),
                         integral_domain.normal(), dof, entity=entity, mapping=mapping)

    def get_tex(self) -> typing.Tuple[str, typing.List[str]]:
        """Get a representation of the functional as TeX, and list of terms involved.

        Returns:
            Representation of the functional as TeX, and list of terms involved
        """
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


# Types
ListOfFunctionals = typing.List[BaseFunctional]
