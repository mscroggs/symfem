"""Reference elements."""

import typing
import sympy
from .symbolic import (
    t, x, subs, sym_sum, SetOfPoints, PointType, ScalarFunction, ScalarValue,
    SetOfPointsInput)
from .vectors import vsub, vnorm, vdot, vcross, vnormalise, vadd


class Reference:
    """A reference cell on which a finite element can be defined."""

    def __init__(
        self, tdim: int, name: str, origin: PointType, axes: SetOfPoints,
        reference_vertices: SetOfPoints, vertices: SetOfPoints,
        edges: typing.Tuple[typing.Tuple[int, int], ...],
        faces: typing.Tuple[typing.Tuple[int, ...], ...],
        volumes: typing.Tuple[typing.Tuple[int, ...], ...],
        sub_entity_types: typing.List[typing.Union[typing.List[str], str, None]],
        simplex: bool = False, tp: bool = False
    ):
        self.tdim = tdim
        self.gdim = len(origin)
        self.name = name
        self.origin = origin
        self.axes = axes
        self.reference_vertices = reference_vertices
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.volumes = volumes
        self.sub_entity_types = sub_entity_types
        self.simplex = simplex
        self.tp = tp
        self._inverse_map_to_self: typing.Union[PointType, None] = None
        self._map_to_self: typing.Union[PointType, None] = None

    def default_reference(self) -> typing.Any:
        # def default_reference(self) -> Reference:
        """Get the default reference for this cell type."""
        raise NotImplementedError()

    def z_ordered_entities(self) -> typing.List[typing.List[typing.Tuple[int, int]]]:
        """Get the subentities of the cell in back-to-front plotting order."""
        return [[(i, j) for i in range(self.tdim, -1, -1) for j in range(self.sub_entity_count(i))]]

    def get_point(self, reference_coords: SetOfPoints) -> typing.Tuple[sympy.core.expr.Expr, ...]:
        """Get a point in the reference from reference coordinates."""
        assert len(reference_coords) == len(self.axes)
        return tuple(o + sym_sum(a[i] * b for a, b in zip(self.axes, reference_coords))
                     for i, o in enumerate(self.origin))

    def integral(
        self, f: ScalarFunction, vars: typing.List[sympy.core.symbol.Symbol] = t
    ) -> sympy.core.expr.Expr:
        """Calculate the integral over the element."""
        raise NotImplementedError()

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        raise NotImplementedError()

    def get_inverse_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        raise NotImplementedError()

    def get_map_to_self(self) -> PointType:
        """Get the map from the canonical reference to this reference."""
        if self._map_to_self is None:
            self._map_to_self = self._compute_map_to_self()
        assert self._map_to_self is not None
        return self._map_to_self

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        raise NotImplementedError()

    def get_inverse_map_to_self(self) -> PointType:
        """Get the inverse map from the canonical reference to this reference."""
        if self._inverse_map_to_self is None:
            self._inverse_map_to_self = self._compute_inverse_map_to_self()
        assert self._inverse_map_to_self is not None
        return self._inverse_map_to_self

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        raise NotImplementedError()

    def volume(self) -> ScalarValue:
        """Calculate the volume."""
        raise NotImplementedError()

    def midpoint(self) -> PointType:
        """Calculate the midpoint."""
        return tuple(sum(i) * sympy.Integer(1) / len(i) for i in zip(*self.vertices))

    def jacobian(self) -> sympy.core.expr.Expr:
        """Calculate the jacobian."""
        assert len(self.axes) == self.tdim
        if self.tdim == 1:
            return vnorm(self.axes[0])
        if self.tdim == 2:
            return vnorm(vcross(self.axes[0], self.axes[1]))
        if self.tdim == 3:
            return vnorm(vdot(vcross(self.axes[0], self.axes[1]), self.axes[2]))
        raise ValueError(f"Unsupported tdim: {self.tdim}")

    def scaled_axes(self) -> SetOfPoints:
        """Return the unit axes of the reference."""
        return tuple(vnormalise(a) for a in self.axes)

    def tangent(self) -> PointType:
        """Calculate the tangent to the element."""
        if self.tdim == 1:
            norm = sympy.sqrt(sum(i ** 2 for i in self.axes[0]))
            return vnormalise(tuple(i / norm for i in self.axes[0]))

        raise RuntimeError

    def normal(self) -> PointType:
        """Calculate the normal to the element."""
        if self.tdim == 1:
            if self.gdim == 2:
                return vnormalise((-self.axes[0][1], self.axes[0][0]))
        if self.tdim == 2:
            if self.gdim == 3:
                return vnormalise(vcross(self.axes[0], self.axes[1]))
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

    def on_edge(self, point: PointType) -> bool:
        """Check if a point is on an edge of the reference."""
        for e in self.edges:
            v0 = self.vertices[e[0]]
            v1 = self.vertices[e[1]]
            if vnorm(vcross(vsub(v0, point), vsub(v1, point))) == 0:
                return True
        return False

    def on_face(self, point: PointType) -> bool:
        """Check if a point is on a face of the reference."""
        for f in self.faces:
            v0 = self.vertices[f[0]]
            v1 = self.vertices[f[1]]
            v2 = self.vertices[f[2]]
            if vdot(vcross(vsub(v0, point), vsub(v1, point)), vsub(v2, point)):
                return True
        return False

    def contains(self, point: PointType) -> bool:
        """Check is a point is contained in the reference."""
        raise NotImplementedError()


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
            vertices=tuple(vertices),
            edges=tuple(),
            faces=tuple(),
            volumes=tuple(),
            sub_entity_types=["point", None, None, None],
            simplex=True, tp=True)

    def default_reference(self) -> typing.Any:
        """Get the default reference for this cell type."""
        return Point(self.reference_vertices)

    def integral(
        self, f: ScalarFunction, vars: typing.List[sympy.core.symbol.Symbol] = t
    ) -> sympy.core.expr.Expr:
        """Calculate the integral over the element."""
        return subs(f, vars, self.vertices[0])

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return vertices[0]

    def get_inverse_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        return self.vertices[0]

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        return self.vertices[0]

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        return self.reference_vertices[0]

    def volume(self) -> ScalarValue:
        """Calculate the volume."""
        return 0

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
            axes=(vsub(vertices[1], vertices[0]),),
            reference_vertices=((0,), (1,)),
            vertices=tuple(vertices),
            edges=((0, 1),),
            faces=tuple(),
            volumes=tuple(),
            sub_entity_types=["point", "interval", None, None],
            simplex=True, tp=True)

    def default_reference(self) -> typing.Any:
        """Get the default reference for this cell type."""
        return Interval(self.reference_vertices)

    def integral(
        self, f: ScalarFunction, vars: typing.List[sympy.core.symbol.Symbol] = t
    ) -> sympy.core.expr.Expr:
        """Calculate the integral over the element."""
        return (f * self.jacobian()).integrate((vars[0], 0, 1))

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(v0 + (v1 - v0) * x[0] for v0, v1 in zip(*vertices))

    def get_inverse_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        p = vsub(x, vertices[0])
        v = vsub(vertices[1], vertices[0])
        return (vdot(p, v) / vdot(v, v), )

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        return tuple(v0 + (v1 - v0) * x[0] for v0, v1 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        p = vsub(x, self.vertices[0])
        v = vsub(self.vertices[1], self.vertices[0])
        return (vdot(p, v) / vdot(v, v), )

    def volume(self) -> ScalarValue:
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
            axes=(vsub(vertices[1], vertices[0]), vsub(vertices[2], vertices[0])),
            reference_vertices=((0, 0), (1, 0), (0, 1)),
            vertices=tuple(vertices),
            edges=((1, 2), (0, 2), (0, 1)),
            faces=((0, 1, 2),),
            volumes=tuple(),
            sub_entity_types=["point", "interval", "triangle", None],
            simplex=True)

    def default_reference(self) -> typing.Any:
        """Get the default reference for this cell type."""
        return Triangle(self.reference_vertices)

    def integral(
        self, f: ScalarFunction, vars: typing.List[sympy.core.symbol.Symbol] = t
    ) -> sympy.core.expr.Expr:
        """Calculate the integral over the element."""
        return (f * self.jacobian()).integrate(
            (vars[1], 0, 1 - vars[0]), (vars[0], 0, 1))

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(v0 + (v1 - v0) * x[0] + (v2 - v0) * x[1] for v0, v1, v2 in zip(*vertices))

    def get_inverse_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        assert len(vertices[0]) == 2
        p = vsub(x, vertices[0])
        v1 = vsub(vertices[1], vertices[0])
        v2 = vsub(vertices[2], vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0]],
                            [v1[1], v2[1]]]).inv()
        return (vdot(mat.row(0), p), vdot(mat.row(1), p))

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        return tuple(v0 + (v1 - v0) * x[0] + (v2 - v0) * x[1]
                     for v0, v1, v2 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        if len(self.vertices[0]) == 2:
            p = vsub(x, self.vertices[0])
            v1 = vsub(self.vertices[1], self.vertices[0])
            v2 = vsub(self.vertices[2], self.vertices[0])
            mat = sympy.Matrix([[v1[0], v2[0]],
                                [v1[1], v2[1]]]).inv()
            return (vdot(mat.row(0), p), vdot(mat.row(1), p))

        return tuple(
            vdot(vsub(x, self.origin), a) / vnorm(a) for a in self.axes
        )

    def volume(self) -> ScalarValue:
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
                vsub(vertices[1], vertices[0]),
                vsub(vertices[2], vertices[0]),
                vsub(vertices[3], vertices[0]),
            ),
            reference_vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            vertices=tuple(vertices),
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

    def integral(
        self, f: ScalarFunction, vars: typing.List[sympy.core.symbol.Symbol] = t
    ) -> sympy.core.expr.Expr:
        """Calculate the integral over the element."""
        return (f * self.jacobian()).integrate(
            (vars[0], 0, 1 - vars[1] - vars[2]), (vars[1], 0, 1 - vars[2]), (vars[2], 0, 1))

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(v0 + (v1 - v0) * x[0] + (v2 - v0) * x[1] + (v3 - v0) * x[2]
                     for v0, v1, v2, v3 in zip(*vertices))

    def get_inverse_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        assert len(vertices[0]) == 3
        p = vsub(x, vertices[0])
        v1 = vsub(vertices[1], vertices[0])
        v2 = vsub(vertices[2], vertices[0])
        v3 = vsub(vertices[3], vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return (vdot(mat.row(0), p), vdot(mat.row(1), p), vdot(mat.row(2), p))

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        return tuple(v0 + (v1 - v0) * x[0] + (v2 - v0) * x[1] + (v3 - v0) * x[2]
                     for v0, v1, v2, v3 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        p = vsub(x, self.vertices[0])
        v1 = vsub(self.vertices[1], self.vertices[0])
        v2 = vsub(self.vertices[2], self.vertices[0])
        v3 = vsub(self.vertices[3], self.vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return (vdot(mat.row(0), p), vdot(mat.row(1), p), vdot(mat.row(2), p))

    def volume(self) -> ScalarValue:
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
            axes=(vsub(vertices[1], vertices[0]), vsub(vertices[2], vertices[0])),
            reference_vertices=((0, 0), (1, 0), (0, 1), (1, 1)),
            vertices=tuple(vertices),
            edges=((0, 1), (0, 2), (1, 3), (2, 3)),
            faces=((0, 1, 2, 3),),
            volumes=tuple(),
            sub_entity_types=["point", "interval", "quadrilateral", None],
            tp=True)

    def default_reference(self) -> typing.Any:
        """Get the default reference for this cell type."""
        return Quadrilateral(self.reference_vertices)

    def integral(
        self, f: ScalarFunction, vars: typing.List[sympy.core.symbol.Symbol] = t
    ) -> sympy.core.expr.Expr:
        """Calculate the integral over the element."""
        return (f * self.jacobian()).integrate((vars[1], 0, 1), (vars[0], 0, 1))

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1) + x[1] * ((1 - x[0]) * v2 + x[0] * v3)
            for v0, v1, v2, v3 in zip(*vertices))

    def get_inverse_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        assert vadd(vertices[0], vertices[3]) == vadd(vertices[1], vertices[2])
        p = vsub(x, vertices[0])
        v1 = vsub(vertices[1], vertices[0])
        v2 = vsub(vertices[2], vertices[0])

        if len(self.vertices[0]) == 2:
            mat = sympy.Matrix([[v1[0], v2[0]],
                                [v1[1], v2[1]]]).inv()
        elif len(self.vertices[0]) == 3:
            v3 = vcross(v1, v2)
            mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                                [v1[1], v2[1], v3[1]],
                                [v1[2], v2[2], v3[2]]]).inv()
        else:
            raise RuntimeError("Cannot get inverse map.")

        return (vdot(mat.row(0), p), vdot(mat.row(1), p))

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference."""
        return tuple(
            (1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1) + x[1] * ((1 - x[0]) * v2 + x[0] * v3)
            for v0, v1, v2, v3 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference."""
        assert vadd(self.vertices[0], self.vertices[3]) == vadd(self.vertices[1],
                                                                self.vertices[2])
        p = vsub(x, self.vertices[0])
        v1 = vsub(self.vertices[1], self.vertices[0])
        v2 = vsub(self.vertices[2], self.vertices[0])

        if len(self.vertices[0]) == 2:
            mat = sympy.Matrix([[v1[0], v2[0]],
                                [v1[1], v2[1]]]).inv()
        elif len(self.vertices[0]) == 3:
            v3 = vcross(v1, v2)
            mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                                [v1[1], v2[1], v3[1]],
                                [v1[2], v2[2], v3[2]]]).inv()
        else:
            raise RuntimeError("Cannot get inverse map.")

        return (vdot(mat.row(0), p), vdot(mat.row(1), p))

    def volume(self) -> ScalarValue:
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
                vsub(vertices[1], vertices[0]),
                vsub(vertices[2], vertices[0]),
                vsub(vertices[4], vertices[0]),
            ),
            reference_vertices=(
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)),
            vertices=tuple(vertices),
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

    def integral(
        self, f: ScalarFunction, vars: typing.List[sympy.core.symbol.Symbol] = t
    ) -> sympy.core.expr.Expr:
        """Calculate the integral over the element."""
        return (f * self.jacobian()).integrate(
            (vars[2], 0, 1), (vars[1], 0, 1), (vars[0], 0, 1))

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[2]) * ((1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1)
                          + x[1] * ((1 - x[0]) * v2 + x[0] * v3))
            + x[2] * ((1 - x[1]) * ((1 - x[0]) * v4 + x[0] * v5)
                      + x[1] * ((1 - x[0]) * v6 + x[0] * v7))
            for v0, v1, v2, v3, v4, v5, v6, v7 in zip(*vertices))

    def get_inverse_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        assert len(vertices[0]) == 3
        for a, b, c, d in self.faces:
            assert vadd(vertices[a], vertices[d]) == vadd(vertices[b], vertices[c])
        p = vsub(x, vertices[0])
        v1 = vsub(vertices[1], vertices[0])
        v2 = vsub(vertices[2], vertices[0])
        v3 = vsub(vertices[4], vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return tuple(vdot(mat.row(i), p) for i in range(mat.rows))

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
            assert vadd(self.vertices[a], self.vertices[d]) == vadd(self.vertices[b],
                                                                    self.vertices[c])
        p = vsub(x, self.vertices[0])
        v1 = vsub(self.vertices[1], self.vertices[0])
        v2 = vsub(self.vertices[2], self.vertices[0])
        v3 = vsub(self.vertices[4], self.vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return tuple(vdot(mat.row(i), p) for i in range(mat.rows))

    def volume(self) -> ScalarValue:
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
                vsub(vertices[1], vertices[0]),
                vsub(vertices[2], vertices[0]),
                vsub(vertices[3], vertices[0]),
            ),
            reference_vertices=(
                (0, 0, 0), (1, 0, 0), (0, 1, 0),
                (0, 0, 1), (1, 0, 1), (0, 1, 1)),
            vertices=tuple(vertices),
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

    def integral(
        self, f: ScalarFunction, vars: typing.List[sympy.core.symbol.Symbol] = t
    ) -> sympy.core.expr.Expr:
        """Calculate the integral over the element."""
        return(f * self.jacobian()).integrate(
            (vars[2], 0, 1), (vars[1], 0, 1 - vars[0]), (vars[0], 0, 1))

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[2]) * (v0 + x[0] * (v1 - v0) + x[1] * (v2 - v0))
            + x[2] * (v3 + x[0] * (v4 - v3) + x[1] * (v5 - v3))
            for v0, v1, v2, v3, v4, v5 in zip(*vertices))

    def get_inverse_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        assert len(vertices[0]) == 3
        for a, b, c, d in self.faces[1:4]:
            assert vadd(vertices[a], vertices[d]) == vadd(vertices[b], vertices[c])
        p = vsub(x, vertices[0])
        v1 = vsub(vertices[1], vertices[0])
        v2 = vsub(vertices[2], vertices[0])
        v3 = vsub(vertices[3], vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return tuple(vdot(mat.row(i), p) for i in range(mat.rows))

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
            assert vadd(self.vertices[a], self.vertices[d]) == vadd(self.vertices[b],
                                                                    self.vertices[c])
        p = vsub(x, self.vertices[0])
        v1 = vsub(self.vertices[1], self.vertices[0])
        v2 = vsub(self.vertices[2], self.vertices[0])
        v3 = vsub(self.vertices[3], self.vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return tuple(vdot(mat.row(i), p) for i in range(mat.rows))

    def volume(self) -> ScalarValue:
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
                vsub(vertices[1], vertices[0]),
                vsub(vertices[2], vertices[0]),
                vsub(vertices[4], vertices[0]),
            ),
            reference_vertices=(
                (0, 0, 0), (1, 0, 0), (0, 1, 0),
                (1, 1, 0), (0, 0, 1)),
            vertices=tuple(vertices),
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

    def integral(
        self, f: ScalarFunction, vars: typing.List[sympy.core.symbol.Symbol] = t
    ) -> sympy.core.expr.Expr:
        """Calculate the integral over the element."""
        return (f * self.jacobian()).integrate(
            (vars[0], 0, 1 - vars[2]), (vars[1], 0, 1 - vars[2]), (vars[2], 0, 1))

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[2]) * (
                (1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1)
                + x[1] * ((1 - x[0]) * v2 + x[0] * v3)
            ) + x[2] * v4
            for v0, v1, v2, v3, v4 in zip(*vertices))

    def get_inverse_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        assert len(vertices[0]) == 3
        for a, b, c, d in self.faces[:1]:
            assert vadd(vertices[a], vertices[d]) == vadd(vertices[b], vertices[c])
        p = vsub(x, vertices[0])
        v1 = vsub(vertices[1], vertices[0])
        v2 = vsub(vertices[2], vertices[0])
        v3 = vsub(vertices[4], vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return tuple(vdot(mat.row(i), p) for i in range(mat.rows))

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
            assert vadd(self.vertices[a], self.vertices[d]) == vadd(self.vertices[b],
                                                                    self.vertices[c])
        p = vsub(x, self.vertices[0])
        v1 = vsub(self.vertices[1], self.vertices[0])
        v2 = vsub(self.vertices[2], self.vertices[0])
        v3 = vsub(self.vertices[4], self.vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return tuple(vdot(mat.row(i), p) for i in range(mat.rows))

    def volume(self) -> ScalarValue:
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

    def __init__(
        self, number_of_triangles: int,
        vertices: SetOfPointsInput = None
    ):
        self.number_of_triangles = number_of_triangles
        self.reference_origin = (0, 0)
        reference_vertices = []
        for tri in range(number_of_triangles):
            angle = sympy.pi * 2 * tri / number_of_triangles
            next_angle = sympy.pi * 2 * (tri + 1) / number_of_triangles

            reference_vertices.append((sympy.cos(angle), sympy.sin(angle)))
            reference_vertices.append(
                ((sympy.cos(next_angle) + sympy.cos(angle)) / 2,
                 (sympy.sin(next_angle) + sympy.sin(angle)) / 2))

        if vertices is None:
            origin: PointType = self.reference_origin
            vertices = tuple(reference_vertices)
        else:
            assert len(vertices) == 1 + len(reference_vertices)
            origin = vertices[0]
            vertices = vertices[1:]

        super().__init__(
            tdim=2,
            name="dual polygon",
            axes=tuple(),
            origin=origin,
            vertices=tuple(vertices),
            reference_vertices=tuple(reference_vertices),
            edges=tuple((i, (i + 1) % (2 * number_of_triangles))
                        for i in range(2 * number_of_triangles)),
            faces=(tuple(range(2 * number_of_triangles)), ),
            volumes=tuple(),
            sub_entity_types=["point", "interval", f"dual polygon({number_of_triangles})", None],

        )
