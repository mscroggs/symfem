"""Reference elements."""

from __future__ import annotations
import typing
import sympy
from abc import ABC, abstractmethod
from .geometry import (PointType, PointTypeInput, SetOfPoints, SetOfPointsInput,
                       parse_set_of_points_input, parse_point_input)
from .symbols import t, x, AxisVariablesNotSingle


def _vsub(v: PointTypeInput, w: PointTypeInput) -> PointType:
    """Subtract."""
    return tuple(i - j for i, j in zip(parse_point_input(v), parse_point_input(w)))


def _vadd(v: PointTypeInput, w: PointTypeInput) -> PointType:
    """Add."""
    return tuple(i + j for i, j in zip(parse_point_input(v), parse_point_input(w)))


def _vdot(v: PointTypeInput, w: PointTypeInput) -> sympy.core.expr.Expr:
    """Compute dot product."""
    out = sympy.Integer(0)
    for i, j in zip(parse_point_input(v), parse_point_input(w)):
        out += i * j
    return out


def _vcross(v_in: PointTypeInput, w_in: PointTypeInput) -> PointType:
    """Compute cross product."""
    v = parse_point_input(v_in)
    w = parse_point_input(w_in)
    assert len(v) == len(w) == 3
    return (v[1] * w[2] - v[2] * w[1], v[2] * w[0] - v[0] * w[2], v[0] * w[1] - v[1] * w[0])


def _vnorm(v_in: PointTypeInput) -> sympy.core.expr.Expr:
    """Find the norm of a vector."""
    v = parse_point_input(v_in)
    return sympy.sqrt(_vdot(v, v))


def _vnormalise(v_in: PointTypeInput) -> PointType:
    """Normalise a vector."""
    v = parse_point_input(v_in)
    n = _vnorm(v)
    return tuple(i / n for i in v)


class Reference(ABC):
    """A reference cell on which a finite element can be defined."""

    def __init__(
        self, tdim: int, name: str, origin: PointTypeInput, axes: SetOfPointsInput,
        reference_vertices: SetOfPointsInput, vertices: SetOfPointsInput,
        edges: typing.Tuple[typing.Tuple[int, int], ...],
        faces: typing.Tuple[typing.Tuple[int, ...], ...],
        volumes: typing.Tuple[typing.Tuple[int, ...], ...],
        sub_entity_types: typing.List[typing.Union[typing.List[str], str, None]],
        simplex: bool = False, tp: bool = False
    ):
        self.tdim = tdim
        self.origin: PointType = parse_point_input(origin)
        self.gdim = len(self.origin)
        self.name = name
        self.axes: SetOfPoints = parse_set_of_points_input(axes)
        self.reference_vertices: SetOfPoints = parse_set_of_points_input(reference_vertices)
        self.vertices: SetOfPoints = parse_set_of_points_input(vertices)
        self.edges = edges
        self.faces = faces
        self.volumes = volumes
        self.sub_entity_types = sub_entity_types
        self.simplex = simplex
        self.tp = tp
        self._inverse_map_to_self: typing.Union[PointType, None] = None
        self._map_to_self: typing.Union[PointType, None] = None

    @abstractmethod
    def default_reference(self) -> Reference:
        """Get the default reference for this cell type."""
        pass

    def z_ordered_entities(self) -> typing.List[typing.List[typing.Tuple[int, int]]]:
        """Get the subentities of the cell in back-to-front plotting order."""
        return [[(i, j) for i in range(self.tdim, -1, -1) for j in range(self.sub_entity_count(i))]]

    def get_point(self, reference_coords: PointType) -> typing.Tuple[sympy.core.expr.Expr, ...]:
        """Get a point in the reference from reference coordinates."""
        assert len(reference_coords) == len(self.axes)
        pt = [i for i in self.origin]
        for a, b in zip(self.axes, reference_coords):
            for i, ai in enumerate(a):
                pt[i] += ai * b
        return tuple(pt)

    @abstractmethod
    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> typing.List[typing.Union[
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr, sympy.core.expr.Expr],
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr],
    ]]:
        """Get the limits for an integral over this reference."""
        pass

    @abstractmethod
    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        pass

    @abstractmethod
    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        pass

    def get_map_to_self(self) -> PointType:
        """Get the map from the canonical reference to this reference."""
        if self._map_to_self is None:
            self._map_to_self = self._compute_map_to_self()
        assert self._map_to_self is not None
        return self._map_to_self

    @abstractmethod
    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        pass

    def get_inverse_map_to_self(self) -> PointType:
        """Get the inverse map from the canonical reference to this reference."""
        if self._inverse_map_to_self is None:
            self._inverse_map_to_self = self._compute_inverse_map_to_self()
        assert self._inverse_map_to_self is not None
        return self._inverse_map_to_self

    @abstractmethod
    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        pass

    @abstractmethod
    def volume(self) -> sympy.core.expr.Expr:
        """Calculate the volume."""
        pass

    def midpoint(self) -> PointType:
        """Calculate the midpoint."""
        return tuple(sum(i) * sympy.Integer(1) / len(i) for i in zip(*self.vertices))

    def jacobian(self) -> sympy.core.expr.Expr:
        """Calculate the jacobian."""
        from .functions import VectorFunction
        assert len(self.axes) == self.tdim
        vaxes = [VectorFunction(a) for a in self.axes]
        if self.tdim == 1:
            out = vaxes[0].norm().as_sympy()
        elif self.tdim == 2:
            out = vaxes[0].cross(vaxes[1]).norm().as_sympy()
        elif self.tdim == 3:
            out = vaxes[0].cross(vaxes[1]).dot(vaxes[2]).norm().as_sympy()
        else:
            raise ValueError(f"Unsupported tdim: {self.tdim}")
        assert isinstance(out, sympy.core.expr.Expr)
        return out

    def scaled_axes(self) -> SetOfPoints:
        """Return the unit axes of the reference."""
        return tuple(_vnormalise(a) for a in self.axes)

    def tangent(self) -> PointType:
        """Calculate the tangent to the element."""
        if self.tdim == 1:
            norm = sympy.sqrt(sum(i ** 2 for i in self.axes[0]))
            return _vnormalise(tuple(i / norm for i in self.axes[0]))

        raise RuntimeError

    def normal(self) -> PointType:
        """Calculate the normal to the element."""
        if self.tdim == 1:
            if self.gdim == 2:
                return _vnormalise((-self.axes[0][1], self.axes[0][0]))
        if self.tdim == 2:
            if self.gdim == 3:
                crossed = _vcross(self.axes[0], self.axes[1])
                assert isinstance(crossed, tuple)
                return _vnormalise(crossed)
        raise RuntimeError

    def sub_entities(
        self, dim: int = None, codim: int = None
    ) -> typing.Tuple[typing.Tuple[int, ...], ...]:
        """Get the sub entities of a given dimension."""
        if dim is None:
            assert codim is not None
            dim = self.tdim - codim
        if dim == 0:
            return tuple((i, ) for i, _ in enumerate(self.vertices))
        if dim == 1:
            return self.edges
        if dim == 2:
            return self.faces
        if dim == 3:
            return self.volumes
        raise ValueError(f"Unsupported dimension: {dim}")

    def sub_entity_count(self, dim: int) -> int:
        """Get the number of sub entities of a given dimension."""
        return len(self.sub_entities(dim))

    def sub_entity(self, dim: int, n: int, reference_vertices: bool = False) -> typing.Any:
        # def sub_entity(self, dim: int, n: int, reference_vertices: bool = False) -> Reference:
        """Get the sub entity of a given dimension and number."""
        from symfem import create_reference

        entity_type = self.sub_entity_types[dim]
        if not isinstance(entity_type, str):
            assert isinstance(entity_type, list)
            entity_type = entity_type[n]

        if reference_vertices:
            return create_reference(
                entity_type, tuple(self.reference_vertices[i] for i in self.sub_entities(dim)[n]))
        else:
            if self.tdim == dim:
                return self
            else:
                return create_reference(
                    entity_type, tuple(self.vertices[i] for i in self.sub_entities(dim)[n]))

    def at_vertex(self, point: PointType) -> bool:
        """Check if a point is a vertex of the reference."""
        for v in self.vertices:
            if v == tuple(point):
                return True
        return False

    def on_edge(self, point_in: PointType) -> bool:
        """Check if a point is on an edge of the reference."""
        from .functions import VectorFunction
        point = VectorFunction(point_in)
        for e in self.edges:
            v0 = VectorFunction(self.vertices[e[0]])
            v1 = VectorFunction(self.vertices[e[1]])
            crossed = (v0 - point).cross(v1 - point).norm()
            if crossed == 0:
                return True
        return False

    def on_face(self, point_in: PointType) -> bool:
        """Check if a point is on a face of the reference."""
        from .functions import VectorFunction
        point = VectorFunction(point_in)
        for f in self.faces:
            v0 = VectorFunction(self.vertices[f[0]])
            v1 = VectorFunction(self.vertices[f[1]])
            v2 = VectorFunction(self.vertices[f[2]])
            crossed = (v0 - point).cross(v1 - point)
            if crossed.dot(v2 - point):
                return True
        return False

    @abstractmethod
    def contains(self, point: PointType) -> bool:
        """Check is a point is contained in the reference."""
        pass


class Point(Reference):
    """A point."""

    def __init__(
        self, vertices: SetOfPointsInput = (tuple(), )
    ):
        assert len(vertices) == 1
        super().__init__(
            tdim=0,
            name="point",
            origin=vertices[0],
            axes=tuple(),
            reference_vertices=(tuple(), ),
            vertices=vertices,
            edges=tuple(),
            faces=tuple(),
            volumes=tuple(),
            sub_entity_types=["point", None, None, None],
            simplex=True, tp=True)

    def default_reference(self) -> typing.Any:
        """Get the default reference for this cell type."""
        return Point(self.reference_vertices)

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> typing.List[typing.Union[
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr, sympy.core.expr.Expr],
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr],
    ]]:
        """Get the limits for an integral over this reference."""
        return list(zip(vars, self.vertices[0]))

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return parse_point_input(vertices[0])

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        return self.vertices[0]

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        return self.vertices[0]

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        return self.reference_vertices[0]

    def volume(self) -> sympy.core.expr.Expr:
        """Calculate the volume."""
        return sympy.Integer(0)

    def contains(self, point: PointType) -> bool:
        """Check is a point is contained in the reference."""
        if self.tdim == 0:
            return point in self.vertices
        raise NotImplementedError()


class Interval(Reference):
    """An interval."""

    def __init__(
        self, vertices: SetOfPointsInput = ((0,), (1,))
    ):
        assert len(vertices) == 2
        super().__init__(
            tdim=1,
            name="interval",
            origin=vertices[0],
            axes=(_vsub(vertices[1], vertices[0]),),
            reference_vertices=((0,), (1,)),
            vertices=vertices,
            edges=((0, 1),),
            faces=tuple(),
            volumes=tuple(),
            sub_entity_types=["point", "interval", None, None],
            simplex=True, tp=True)

    def default_reference(self) -> typing.Any:
        """Get the default reference for this cell type."""
        return Interval(self.reference_vertices)

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> typing.List[typing.Union[
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr, sympy.core.expr.Expr],
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr],
    ]]:
        """Get the limits for an integral over this reference."""
        return [(vars[0], sympy.Integer(0), sympy.Integer(1))]

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(v0 + (v1 - v0) * x[0] for v0, v1 in zip(*vertices))

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        vertices = parse_set_of_points_input(vertices_in)
        p = _vsub(tuple(x), vertices[0])
        v = _vsub(vertices[1], vertices[0])
        return (_vdot(p, v) * sympy.Integer(1) / _vdot(v, v), )

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        return tuple(v0 + (v1 - v0) * x[0] for v0, v1 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        p = _vsub(tuple(x), self.vertices[0])
        v = _vsub(self.vertices[1], self.vertices[0])
        return (_vdot(p, v) * sympy.Integer(1) / _vdot(v, v), )

    def volume(self) -> sympy.core.expr.Expr:
        """Calculate the volume."""
        return self.jacobian()

    def contains(self, point: PointType) -> bool:
        """Check is a point is contained in the reference."""
        if self.vertices != self.reference_vertices:
            raise NotImplementedError()
        return 0 <= point[0] <= 1


class Triangle(Reference):
    """A triangle."""

    def __init__(
        self, vertices: SetOfPointsInput = ((0, 0), (1, 0), (0, 1))
    ):
        assert len(vertices) == 3
        super().__init__(
            tdim=2,
            name="triangle",
            origin=vertices[0],
            axes=(_vsub(vertices[1], vertices[0]), _vsub(vertices[2], vertices[0])),
            reference_vertices=((0, 0), (1, 0), (0, 1)),
            vertices=vertices,
            edges=((1, 2), (0, 2), (0, 1)),
            faces=((0, 1, 2),),
            volumes=tuple(),
            sub_entity_types=["point", "interval", "triangle", None],
            simplex=True)

    def default_reference(self) -> typing.Any:
        """Get the default reference for this cell type."""
        return Triangle(self.reference_vertices)

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> typing.List[typing.Union[
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr, sympy.core.expr.Expr],
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr],
    ]]:
        """Get the limits for an integral over this reference."""
        return [(vars[1], sympy.Integer(0), 1 - vars[0]),
                (vars[0], sympy.Integer(0), sympy.Integer(1))]

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(v0 + (v1 - v0) * x[0] + (v2 - v0) * x[1] for v0, v1, v2 in zip(*vertices))

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        vertices = parse_set_of_points_input(vertices_in)
        assert len(vertices[0]) == 2
        p = _vsub(tuple(x), vertices[0])
        v1 = _vsub(vertices[1], vertices[0])
        v2 = _vsub(vertices[2], vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0]],
                            [v1[1], v2[1]]]).inv()
        return (_vdot(mat.row(0), p), _vdot(mat.row(1), p))

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        return tuple(v0 + (v1 - v0) * x[0] + (v2 - v0) * x[1]
                     for v0, v1, v2 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        if len(self.vertices[0]) == 2:
            p = _vsub(tuple(x), self.vertices[0])
            v1 = _vsub(self.vertices[1], self.vertices[0])
            v2 = _vsub(self.vertices[2], self.vertices[0])
            mat = sympy.Matrix([[v1[0], v2[0]],
                                [v1[1], v2[1]]]).inv()
            return (_vdot(mat.row(0), p), _vdot(mat.row(1), p))

        return tuple(
            _vdot(_vsub(tuple(x), self.origin), a) * sympy.Integer(1) / _vnorm(a) for a in self.axes
        )

    def volume(self) -> sympy.core.expr.Expr:
        """Calculate the volume."""
        return sympy.Rational(1, 2) * self.jacobian()

    def contains(self, point: PointType) -> bool:
        """Check is a point is contained in the reference."""
        if self.vertices != self.reference_vertices:
            raise NotImplementedError()
        return 0 <= point[0] and 0 <= point[1] and sum(point) <= 1


class Tetrahedron(Reference):
    """A tetrahedron."""

    def __init__(
        self, vertices: SetOfPointsInput = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    ):
        assert len(vertices) == 4
        super().__init__(
            tdim=3,
            name="tetrahedron",
            origin=vertices[0],
            axes=(
                _vsub(vertices[1], vertices[0]),
                _vsub(vertices[2], vertices[0]),
                _vsub(vertices[3], vertices[0]),
            ),
            reference_vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            vertices=vertices,
            edges=((2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1)),
            faces=((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)),
            volumes=((0, 1, 2, 3),),
            sub_entity_types=["point", "interval", "triangle", "tetrahedron"],
            simplex=True)

    def z_ordered_entities(self) -> typing.List[typing.List[typing.Tuple[int, int]]]:
        """Get the subentities of the cell in back-to-front plotting order."""
        return [
            [(2, 0), (2, 1), (2, 3), (1, 0), (1, 2), (1, 4), (0, 2)],
            [(3, 0)],
            [(2, 2), (1, 1), (1, 3), (1, 5), (0, 0), (0, 1), (0, 3)]
        ]

    def default_reference(self) -> typing.Any:
        """Get the default reference for this cell type."""
        return Tetrahedron(self.reference_vertices)

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> typing.List[typing.Union[
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr, sympy.core.expr.Expr],
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr],
    ]]:
        """Get the limits for an integral over this reference."""
        return [(vars[0], sympy.Integer(0), 1 - vars[1] - vars[2]),
                (vars[1], sympy.Integer(0), 1 - vars[2]),
                (vars[2], sympy.Integer(0), sympy.Integer(1))]

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(v0 + (v1 - v0) * x[0] + (v2 - v0) * x[1] + (v3 - v0) * x[2]
                     for v0, v1, v2, v3 in zip(*vertices))

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        vertices = parse_set_of_points_input(vertices_in)
        assert len(vertices[0]) == 3
        p = _vsub(tuple(x), vertices[0])
        v1 = _vsub(vertices[1], vertices[0])
        v2 = _vsub(vertices[2], vertices[0])
        v3 = _vsub(vertices[3], vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return (_vdot(mat.row(0), p), _vdot(mat.row(1), p), _vdot(mat.row(2), p))

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        return tuple(v0 + (v1 - v0) * x[0] + (v2 - v0) * x[1] + (v3 - v0) * x[2]
                     for v0, v1, v2, v3 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        p = _vsub(tuple(x), self.vertices[0])
        v1 = _vsub(self.vertices[1], self.vertices[0])
        v2 = _vsub(self.vertices[2], self.vertices[0])
        v3 = _vsub(self.vertices[3], self.vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return (_vdot(mat.row(0), p), _vdot(mat.row(1), p), _vdot(mat.row(2), p))

    def volume(self) -> sympy.core.expr.Expr:
        """Calculate the volume."""
        return sympy.Rational(1, 6) * self.jacobian()

    def contains(self, point: PointType) -> bool:
        """Check is a point is contained in the reference."""
        if self.vertices != self.reference_vertices:
            raise NotImplementedError()
        return 0 <= point[0] and 0 <= point[1] and 0 <= point[2] and sum(point) <= 1


class Quadrilateral(Reference):
    """A quadrilateral."""

    def __init__(
        self, vertices: SetOfPointsInput = ((0, 0), (1, 0), (0, 1), (1, 1))
    ):
        assert len(vertices) == 4
        super().__init__(
            tdim=2,
            name="quadrilateral",
            origin=vertices[0],
            axes=(_vsub(vertices[1], vertices[0]), _vsub(vertices[2], vertices[0])),
            reference_vertices=((0, 0), (1, 0), (0, 1), (1, 1)),
            vertices=vertices,
            edges=((0, 1), (0, 2), (1, 3), (2, 3)),
            faces=((0, 1, 2, 3),),
            volumes=tuple(),
            sub_entity_types=["point", "interval", "quadrilateral", None],
            tp=True)

    def default_reference(self) -> typing.Any:
        """Get the default reference for this cell type."""
        return Quadrilateral(self.reference_vertices)

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> typing.List[typing.Union[
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr, sympy.core.expr.Expr],
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr],
    ]]:
        """Get the limits for an integral over this reference."""
        return [(vars[1], sympy.Integer(0), sympy.Integer(1)),
                (vars[0], sympy.Integer(0), sympy.Integer(1))]

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1) + x[1] * ((1 - x[0]) * v2 + x[0] * v3)
            for v0, v1, v2, v3 in zip(*vertices))

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        vertices = parse_set_of_points_input(vertices_in)
        assert _vadd(vertices[0], vertices[3]) == _vadd(vertices[1], vertices[2])
        p = _vsub(tuple(x), vertices[0])
        v1 = _vsub(vertices[1], vertices[0])
        v2 = _vsub(vertices[2], vertices[0])

        if len(self.vertices[0]) == 2:
            mat = sympy.Matrix([[v1[0], v2[0]],
                                [v1[1], v2[1]]]).inv()
        elif len(self.vertices[0]) == 3:
            v3 = _vcross(v1, v2)
            mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                                [v1[1], v2[1], v3[1]],
                                [v1[2], v2[2], v3[2]]]).inv()
        else:
            raise RuntimeError("Cannot get inverse map.")

        return (_vdot(mat.row(0), p), _vdot(mat.row(1), p))

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        return tuple(
            (1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1) + x[1] * ((1 - x[0]) * v2 + x[0] * v3)
            for v0, v1, v2, v3 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        assert _vadd(
            self.vertices[0], self.vertices[3]) == _vadd(self.vertices[1], self.vertices[2])
        p = _vsub(tuple(x), self.vertices[0])
        v1 = _vsub(self.vertices[1], self.vertices[0])
        v2 = _vsub(self.vertices[2], self.vertices[0])

        if len(self.vertices[0]) == 2:
            mat = sympy.Matrix([[v1[0], v2[0]],
                                [v1[1], v2[1]]]).inv()
        elif len(self.vertices[0]) == 3:
            v3 = _vcross(v1, v2)
            mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                                [v1[1], v2[1], v3[1]],
                                [v1[2], v2[2], v3[2]]]).inv()
        else:
            raise RuntimeError("Cannot get inverse map.")

        return (_vdot(mat.row(0), p), _vdot(mat.row(1), p))

    def volume(self) -> sympy.core.expr.Expr:
        """Calculate the volume."""
        return self.jacobian()

    def contains(self, point: PointType) -> bool:
        """Check is a point is contained in the reference."""
        if self.vertices != self.reference_vertices:
            raise NotImplementedError()
        return 0 <= point[0] <= 1 and 0 <= point[1] <= 1


class Hexahedron(Reference):
    """A hexahedron."""

    def __init__(
        self, vertices: SetOfPointsInput = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                                            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))
    ):
        assert len(vertices) == 8
        super().__init__(
            tdim=3,
            name="hexahedron",
            origin=vertices[0],
            axes=(
                _vsub(vertices[1], vertices[0]),
                _vsub(vertices[2], vertices[0]),
                _vsub(vertices[4], vertices[0]),
            ),
            reference_vertices=(
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)),
            vertices=vertices,
            edges=(
                (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)),
            faces=(
                (0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6),
                (1, 3, 5, 7), (2, 3, 6, 7), (4, 5, 6, 7)),
            volumes=((0, 1, 2, 3, 4, 5, 6, 7),),
            sub_entity_types=["point", "interval", "quadrilateral", "hexahedron"],
            tp=True)

    def z_ordered_entities(self) -> typing.List[typing.List[typing.Tuple[int, int]]]:
        """Get the subentities of the cell in back-to-front plotting order."""
        return [
            [(2, 4), (2, 0), (1, 5), (2, 2), (1, 6), (1, 1), (0, 2)],
            [(3, 0)],
            [(2, 3), (1, 3), (1, 7), (0, 3)],
            [(2, 1), (1, 0), (1, 2), (1, 4), (0, 0), (0, 1)],
            [(2, 5), (1, 8), (1, 9), (1, 10), (1, 11), (0, 4), (0, 5), (0, 6), (0, 7)]
        ]

    def default_reference(self) -> typing.Any:
        """Get the default reference for this cell type."""
        return Hexahedron(self.reference_vertices)

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> typing.List[typing.Union[
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr, sympy.core.expr.Expr],
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr],
    ]]:
        """Get the limits for an integral over this reference."""
        return [(vars[2], sympy.Integer(0), sympy.Integer(1)),
                (vars[1], sympy.Integer(0), sympy.Integer(1)),
                (vars[0], sympy.Integer(0), sympy.Integer(1))]

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[2]) * ((1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1)
                          + x[1] * ((1 - x[0]) * v2 + x[0] * v3))
            + x[2] * ((1 - x[1]) * ((1 - x[0]) * v4 + x[0] * v5)
                      + x[1] * ((1 - x[0]) * v6 + x[0] * v7))
            for v0, v1, v2, v3, v4, v5, v6, v7 in zip(*vertices))

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        vertices = parse_set_of_points_input(vertices_in)
        assert len(vertices[0]) == 3
        for a, b, c, d in self.faces:
            assert _vadd(vertices[a], vertices[d]) == _vadd(vertices[b], vertices[c])
        p = _vsub(tuple(x), vertices[0])
        v1 = _vsub(vertices[1], vertices[0])
        v2 = _vsub(vertices[2], vertices[0])
        v3 = _vsub(vertices[4], vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return tuple(_vdot(mat.row(i), p) for i in range(mat.rows))

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        return tuple(
            (1 - x[2]) * ((1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1)
                          + x[1] * ((1 - x[0]) * v2 + x[0] * v3))
            + x[2] * ((1 - x[1]) * ((1 - x[0]) * v4 + x[0] * v5)
                      + x[1] * ((1 - x[0]) * v6 + x[0] * v7))
            for v0, v1, v2, v3, v4, v5, v6, v7 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        assert len(self.vertices[0]) == 3
        for a, b, c, d in self.faces:
            assert _vadd(
                self.vertices[a], self.vertices[d]) == _vadd(self.vertices[b], self.vertices[c])
        p = _vsub(tuple(x), self.vertices[0])
        v1 = _vsub(self.vertices[1], self.vertices[0])
        v2 = _vsub(self.vertices[2], self.vertices[0])
        v3 = _vsub(self.vertices[4], self.vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return tuple(_vdot(mat.row(i), p) for i in range(mat.rows))

    def volume(self) -> sympy.core.expr.Expr:
        """Calculate the volume."""
        return self.jacobian()

    def contains(self, point: PointType) -> bool:
        """Check is a point is contained in the reference."""
        if self.vertices != self.reference_vertices:
            raise NotImplementedError()
        return 0 <= point[0] <= 1 and 0 <= point[1] <= 1 and 0 <= point[2] <= 1


class Prism(Reference):
    """A (triangular) prism."""

    def __init__(
        self, vertices: SetOfPointsInput = ((0, 0, 0), (1, 0, 0), (0, 1, 0),
                                            (0, 0, 1), (1, 0, 1), (0, 1, 1))
    ):
        assert len(vertices) == 6
        super().__init__(
            tdim=3,
            name="prism",
            origin=vertices[0],
            axes=(
                _vsub(vertices[1], vertices[0]),
                _vsub(vertices[2], vertices[0]),
                _vsub(vertices[3], vertices[0]),
            ),
            reference_vertices=(
                (0, 0, 0), (1, 0, 0), (0, 1, 0),
                (0, 0, 1), (1, 0, 1), (0, 1, 1)),
            vertices=vertices,
            edges=(
                (0, 1), (0, 2), (0, 3), (1, 2), (1, 4),
                (2, 5), (3, 4), (3, 5), (4, 5)),
            faces=(
                (0, 1, 2), (0, 1, 3, 4), (0, 2, 3, 5),
                (1, 2, 4, 5), (3, 4, 5)),
            volumes=((0, 1, 2, 3, 4, 5),),
            sub_entity_types=[
                "point", "interval",
                ["triangle", "quadrilateral", "quadrilateral", "quadrilateral", "triangle"],
                "prism"],
            tp=True)

    def z_ordered_entities(self) -> typing.List[typing.List[typing.Tuple[int, int]]]:
        """Get the subentities of the cell in back-to-front plotting order."""
        return [
            [(2, 0), (2, 3), (1, 3)],
            [(2, 2), (1, 1), (1, 5), (0, 2)],
            [(3, 0)],
            [(2, 1), (1, 0), (1, 2), (1, 4), (0, 0), (0, 1)],
            [(2, 4), (1, 6), (1, 7), (1, 8), (0, 3), (0, 4), (0, 5)]
        ]

    def default_reference(self) -> typing.Any:
        """Get the default reference for this cell type."""
        return Prism(self.reference_vertices)

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> typing.List[typing.Union[
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr, sympy.core.expr.Expr],
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr],
    ]]:
        """Get the limits for an integral over this reference."""
        return [(vars[2], sympy.Integer(0), sympy.Integer(1)),
                (vars[1], sympy.Integer(0), sympy.Integer(1) - vars[0]),
                (vars[0], sympy.Integer(0), sympy.Integer(1))]

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[2]) * (v0 + x[0] * (v1 - v0) + x[1] * (v2 - v0))
            + x[2] * (v3 + x[0] * (v4 - v3) + x[1] * (v5 - v3))
            for v0, v1, v2, v3, v4, v5 in zip(*vertices))

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        vertices = parse_set_of_points_input(vertices_in)
        assert len(vertices[0]) == 3
        for a, b, c, d in self.faces[1:4]:
            assert _vadd(vertices[a], vertices[d]) == _vadd(vertices[b], vertices[c])
        p = _vsub(tuple(x), vertices[0])
        v1 = _vsub(vertices[1], vertices[0])
        v2 = _vsub(vertices[2], vertices[0])
        v3 = _vsub(vertices[3], vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return tuple(_vdot(mat.row(i), p) for i in range(mat.rows))

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        return tuple(
            (1 - x[2]) * (v0 + x[0] * (v1 - v0) + x[1] * (v2 - v0))
            + x[2] * (v3 + x[0] * (v4 - v3) + x[1] * (v5 - v3))
            for v0, v1, v2, v3, v4, v5 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        assert len(self.vertices[0]) == 3
        for a, b, c, d in self.faces[1:4]:
            assert _vadd(
                self.vertices[a], self.vertices[d]) == _vadd(self.vertices[b], self.vertices[c])
        p = _vsub(tuple(x), self.vertices[0])
        v1 = _vsub(self.vertices[1], self.vertices[0])
        v2 = _vsub(self.vertices[2], self.vertices[0])
        v3 = _vsub(self.vertices[3], self.vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return tuple(_vdot(mat.row(i), p) for i in range(mat.rows))

    def volume(self) -> sympy.core.expr.Expr:
        """Calculate the volume."""
        return sympy.Rational(1, 2) * self.jacobian()

    def contains(self, point: PointType) -> bool:
        """Check is a point is contained in the reference."""
        if self.vertices != self.reference_vertices:
            raise NotImplementedError()
        return (point[0] >= 0 and point[1] >= 0 and point[2] >= 0
                and point[2] <= 1 and point[0] + point[1] <= 1)


class Pyramid(Reference):
    """A (square-based) pyramid."""

    def __init__(
        self, vertices: SetOfPointsInput = ((0, 0, 0), (1, 0, 0), (0, 1, 0),
                                            (1, 1, 0), (0, 0, 1))
    ):
        assert len(vertices) == 5
        super().__init__(
            tdim=3,
            name="pyramid",
            origin=vertices[0],
            axes=(
                _vsub(vertices[1], vertices[0]),
                _vsub(vertices[2], vertices[0]),
                _vsub(vertices[4], vertices[0]),
            ),
            reference_vertices=(
                (0, 0, 0), (1, 0, 0), (0, 1, 0),
                (1, 1, 0), (0, 0, 1)),
            vertices=vertices,
            edges=(
                (0, 1), (0, 2), (0, 4), (1, 3),
                (1, 4), (2, 3), (2, 4), (3, 4)),
            faces=(
                (0, 1, 2, 3), (0, 1, 4), (0, 2, 4),
                (1, 3, 4), (2, 3, 4)),
            volumes=((0, 1, 2, 3, 4),),
            sub_entity_types=[
                "point", "interval",
                ["quadrilateral", "triangle", "triangle", "triangle", "triangle"],
                "pyramid"],
            tp=True)

    def z_ordered_entities(self) -> typing.List[typing.List[typing.Tuple[int, int]]]:
        """Get the subentities of the cell in back-to-front plotting order."""
        return [
            [(2, 0), (2, 4), (1, 5)],
            [(2, 2), (1, 1), (1, 6), (0, 2)],
            [(3, 0)],
            [(2, 3), (1, 3), (1, 7), (0, 3)],
            [(2, 1), (1, 0), (1, 2), (1, 4), (0, 0), (0, 1), (0, 4)]
        ]

    def default_reference(self) -> typing.Any:
        """Get the default reference for this cell type."""
        return Pyramid(self.reference_vertices)

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> typing.List[typing.Union[
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr, sympy.core.expr.Expr],
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr],
    ]]:
        """Get the limits for an integral over this reference."""
        return [(vars[0], sympy.Integer(0), 1 - vars[2]),
                (vars[1], sympy.Integer(0), 1 - vars[2]),
                (vars[2], sympy.Integer(0), sympy.Integer(1))]

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[2]) * (
                (1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1)
                + x[1] * ((1 - x[0]) * v2 + x[0] * v3)
            ) + x[2] * v4
            for v0, v1, v2, v3, v4 in zip(*vertices))

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        vertices = parse_set_of_points_input(vertices_in)
        assert len(vertices[0]) == 3
        for a, b, c, d in self.faces[:1]:
            assert _vadd(vertices[a], vertices[d]) == _vadd(vertices[b], vertices[c])
        p = _vsub(tuple(x), vertices[0])
        v1 = _vsub(vertices[1], vertices[0])
        v2 = _vsub(vertices[2], vertices[0])
        v3 = _vsub(vertices[4], vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return tuple(_vdot(mat.row(i), p) for i in range(mat.rows))

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        return tuple(
            (1 - x[2]) * (
                (1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1)
                + x[1] * ((1 - x[0]) * v2 + x[0] * v3)
            ) + x[2] * v4
            for v0, v1, v2, v3, v4 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        assert len(self.vertices[0]) == 3
        for a, b, c, d in self.faces[:1]:
            assert _vadd(
                self.vertices[a], self.vertices[d]) == _vadd(self.vertices[b], self.vertices[c])
        p = _vsub(tuple(x), self.vertices[0])
        v1 = _vsub(self.vertices[1], self.vertices[0])
        v2 = _vsub(self.vertices[2], self.vertices[0])
        v3 = _vsub(self.vertices[4], self.vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return tuple(_vdot(mat.row(i), p) for i in range(mat.rows))

    def volume(self) -> sympy.core.expr.Expr:
        """Calculate the volume."""
        return sympy.Rational(1, 3) * self.jacobian()

    def contains(self, point: PointType) -> bool:
        """Check is a point is contained in the reference."""
        if self.vertices != self.reference_vertices:
            raise NotImplementedError()
        return (point[0] >= 0 and point[1] >= 0 and point[2] >= 0
                and point[0] + point[2] <= 1 and point[1] + point[2] <= 1)


class DualPolygon(Reference):
    """A polygon on a barycentric dual grid."""

    reference_origin: PointType

    def __init__(
        self, number_of_triangles: int,
        vertices: SetOfPointsInput = None
    ):
        self.number_of_triangles = number_of_triangles
        self.reference_origin = (sympy.Integer(0), sympy.Integer(0))
        reference_vertices = []
        for tri in range(number_of_triangles):
            angle = sympy.pi * 2 * tri / number_of_triangles
            next_angle = sympy.pi * 2 * (tri + 1) / number_of_triangles

            reference_vertices.append((sympy.cos(angle), sympy.sin(angle)))
            reference_vertices.append(
                ((sympy.cos(next_angle) + sympy.cos(angle)) / 2,
                 (sympy.sin(next_angle) + sympy.sin(angle)) / 2))

        origin: PointType = self.reference_origin
        if vertices is None:
            vertices = tuple(reference_vertices)
        else:
            assert len(vertices) == 1 + len(reference_vertices)
            origin = parse_point_input(vertices[0])
            vertices = vertices[1:]

        super().__init__(
            tdim=2,
            name="dual polygon",
            axes=tuple(),
            origin=origin,
            vertices=vertices,
            reference_vertices=tuple(reference_vertices),
            edges=tuple((i, (i + 1) % (2 * number_of_triangles))
                        for i in range(2 * number_of_triangles)),
            faces=(tuple(range(2 * number_of_triangles)), ),
            volumes=tuple(),
            sub_entity_types=["point", "interval", f"dual polygon({number_of_triangles})", None],

        )

    def contains(self, point: PointType) -> bool:
        """Check is a point is contained in the reference."""
        raise NotImplementedError()

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> typing.List[typing.Union[
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr, sympy.core.expr.Expr],
        typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr],
    ]]:
        """Get the limits for an integral over this reference."""
        raise NotImplementedError()

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        raise NotImplementedError()

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        raise NotImplementedError()

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        raise NotImplementedError()

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        raise NotImplementedError()

    def volume(self) -> sympy.core.expr.Expr:
        """Calculate the volume."""
        raise NotImplementedError()

    def default_reference(self) -> Reference:
        """Get the default reference for this cell type."""
        raise NotImplementedError()
