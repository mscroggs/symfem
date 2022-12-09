"""Reference elements."""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod

import sympy

from .geometry import (PointType, PointTypeInput, SetOfPoints, SetOfPointsInput, parse_point_input,
                       parse_set_of_points_input)
from .symbols import AxisVariablesNotSingle, t, x

LatticeWithLines = typing.Tuple[SetOfPoints, typing.List[typing.Tuple[int, int]]]
IntLimits = typing.List[typing.Union[
    typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr, sympy.core.expr.Expr],
    typing.Tuple[sympy.core.symbol.Symbol, sympy.core.expr.Expr]]]


def _vsub(v: PointTypeInput, w: PointTypeInput) -> PointType:
    """Subtract.

    Args:
        v: A vector
        w: A vector

    Returns:
        The vector v - w
    """
    return tuple(i - j for i, j in zip(parse_point_input(v), parse_point_input(w)))


def _vadd(v: PointTypeInput, w: PointTypeInput) -> PointType:
    """Add.

    Args:
        v: A vector
        w: A vector

    Returns:
        The vector v + w
    """
    return tuple(i + j for i, j in zip(parse_point_input(v), parse_point_input(w)))


def _vdot(v: PointTypeInput, w: PointTypeInput) -> sympy.core.expr.Expr:
    """Compute the dot product.

    Args:
        v: A vector
        w: A vector

    Returns:
        The scalar v.w
    """
    out = sympy.Integer(0)
    for i, j in zip(parse_point_input(v), parse_point_input(w)):
        out += i * j
    return out


def _vcross(v_in: PointTypeInput, w_in: PointTypeInput) -> PointType:
    """Compute the cross product.

    Args:
        v: A vector
        w: A vector

    Returns:
        The vector v x w
    """
    v = parse_point_input(v_in)
    w = parse_point_input(w_in)
    assert len(v) == len(w) == 3
    return (v[1] * w[2] - v[2] * w[1], v[2] * w[0] - v[0] * w[2], v[0] * w[1] - v[1] * w[0])


def _vnorm(v_in: PointTypeInput) -> sympy.core.expr.Expr:
    """Find the norm of a vector.

    Args:
        v_in: A vector

    Returns:
        The norm of v_in
    """
    v = parse_point_input(v_in)
    return sympy.sqrt(_vdot(v, v))


def _vnormalise(v_in: PointTypeInput) -> PointType:
    """Normalise a vector.

    Args:
        v_in: A vector

    Returns:
        A unit vector pointing in the same direction as v_in
    """
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
        """Create a reference cell.

        Args:
            tdim: The topological dimension of the cell
            name: The name of the cell
            origin: The coordinates of the origin
            axes: Vectors representing the axes of the cell
            reference_vertices: The vertices of the default version of this cell
            vertices: The vertices of this cell
            edges: Pairs of vertex numbers that form the edges of the cell
            faces: Tuples of vertex numbers that form the faces of the cell
            volumes: Tuples of vertex numbers that form the volumes of the cell
            sub_entity_types: The cell types of each sub-entity of the cell
            simplex: Is the cell a simplex (interval/triangle/tetrahedron)?
            tp: Is the cell a tensor product (interval/quadrilateral/hexahedron)?
        """
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

    @property
    def clockwise_vertices(self) -> SetOfPoints:
        """Get list of vertices in clockwise order.

        Returns:
            A list of vertices
        """
        return self.vertices

    @abstractmethod
    def default_reference(self) -> Reference:
        """Get the default reference for this cell type.

        Returns:
            The default reference
        """
        pass

    @abstractmethod
    def make_lattice(self, n: int) -> SetOfPoints:
        """Make a lattice of points.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points offset from the edge of the cell
        """
        pass

    @abstractmethod
    def make_lattice_with_lines(self, n: int) -> LatticeWithLines:
        """Make a lattice of points, and a list of lines connecting them.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points including the edges of the cell
            Pairs of point numbers that make a mesh of lines across the cell
        """
        pass

    def make_lattice_float(self, n: int) -> SetOfPoints:
        """Make a lattice of points.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points offset from the edge of the cell
        """
        return tuple(tuple(sympy.Float(float(i)) for i in p) for p in self.make_lattice(n))

    def make_lattice_with_lines_float(self, n: int) -> LatticeWithLines:
        """Make a lattice of points, and a list of lines connecting them.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points including the edges of the cell
            Pairs of point numbers that make a mesh of lines across the cell
        """
        pts, pairs = self.make_lattice_with_lines(n)
        return tuple(tuple(sympy.Float(float(i)) for i in p) for p in pts), pairs

    def z_ordered_entities(self) -> typing.List[typing.List[typing.Tuple[int, int]]]:
        """Get the subentities of the cell in back-to-front plotting order.

        Returns:
            List of lists of subentity dimensions and numbers
        """
        return [[(i, j) for i in range(self.tdim, -1, -1) for j in range(self.sub_entity_count(i))]]

    def z_ordered_entities_extra_dim(self) -> typing.List[typing.List[typing.Tuple[int, int]]]:
        """Get the subentities in back-to-front plotting order when using an extra dimension.

        Returns:
            List of lists of subentity dimensions and numbers
        """
        return self.z_ordered_entities()

    def get_point(self, reference_coords: PointType) -> typing.Tuple[sympy.core.expr.Expr, ...]:
        """Get a point in the reference from reference coordinates.

        Args:
            reference_coords: The reference coordinates

        Returns:
            A point in the cell
        """
        assert len(reference_coords) == len(self.axes)
        pt = [i for i in self.origin]
        for a, b in zip(self.axes, reference_coords):
            for i, ai in enumerate(a):
                pt[i] += ai * b
        return tuple(pt)

    @abstractmethod
    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> IntLimits:
        """Get the limits for an integral over this reference.

        Args:
            vars: The variables to use for each direction

        Returns:
            Integration limits that can be passed into sympy.integrate
        """
        pass

    @abstractmethod
    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell.

        Args:
            vertices: The vertices of a call

        Returns:
            The map
        """
        pass

    @abstractmethod
    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference.

        Args:
            vertices_in: The vertices of a cell

        Returns:
            The inverse map
        """
        pass

    def get_map_to_self(self) -> PointType:
        """Get the map from the canonical reference to this reference.

        Returns:
            The map
        """
        if self._map_to_self is None:
            self._map_to_self = self._compute_map_to_self()
        assert self._map_to_self is not None
        return self._map_to_self

    @abstractmethod
    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference.

        Returns:
            The map
        """
        pass

    def get_inverse_map_to_self(self) -> PointType:
        """Get the inverse map from the canonical reference to this reference.

        Returns:
            The map
        """
        if self._inverse_map_to_self is None:
            self._inverse_map_to_self = self._compute_inverse_map_to_self()
        assert self._inverse_map_to_self is not None
        return self._inverse_map_to_self

    @abstractmethod
    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference.

        Returns:
            The map
        """
        pass

    @abstractmethod
    def volume(self) -> sympy.core.expr.Expr:
        """Calculate the volume.

        Returns:
            The volume of the cell
        """
        pass

    def midpoint(self) -> PointType:
        """Calculate the midpoint.

        Returns:
            The midpoint of the cell
        """
        return tuple(sum(i) * sympy.Integer(1) / len(i) for i in zip(*self.vertices))

    def jacobian(self) -> sympy.core.expr.Expr:
        """Calculate the Jacobian.

        Returns:
            The Jacobian
        """
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
        """Return the unit axes of the reference.

        Returns:
            The axes
        """
        return tuple(_vnormalise(a) for a in self.axes)

    def tangent(self) -> PointType:
        """Calculate the tangent to the element.

        Returns:
            The tangent
        """
        if self.tdim == 1:
            norm = sympy.sqrt(sum(i ** 2 for i in self.axes[0]))
            return _vnormalise(tuple(i / norm for i in self.axes[0]))

        raise RuntimeError

    def normal(self) -> PointType:
        """Calculate the normal to the element.

        Returns:
            The normal
        """
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
        self, dim: typing.Optional[int] = None, codim: typing.Optional[int] = None
    ) -> typing.Tuple[typing.Tuple[int, ...], ...]:
        """Get the sub-entities of a given dimension.

        Args:
            dim: The dimension of the sub-entity
            codim: The co-dimension of the sub-entity

        Returns:
            A tuple of tuples of vertex numbers
        """
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

    def sub_entity_count(
        self, dim: typing.Optional[int] = None, codim: typing.Optional[int] = None
    ) -> int:
        """Get the number of sub-entities of a given dimension.

        Args:
            dim: the dimension of the sub-entity
            codim: the codimension of the sub-entity

        Returns:
            The number of sub-entities
        """
        return len(self.sub_entities(dim, codim))

    def sub_entity(self, dim: int, n: int, reference_vertices: bool = False) -> typing.Any:
        # def sub_entity(self, dim: int, n: int, reference_vertices: bool = False) -> Reference:
        """Get the sub-entity of a given dimension and number.

        Args:
            dim: the dimension of the sub-entity
            n: The sub-entity number
            reference_vertices: Should the reference vertices be used?

        Returns:
            The sub-entity
        """
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
        """Check if a point is a vertex of the reference.

        Args:
            point: The point

        Returns:
            Is the point a vertex?
        """
        for v in self.vertices:
            if v == tuple(point):
                return True
        return False

    def on_edge(self, point_in: PointType) -> bool:
        """Check if a point is on an edge of the reference.

        Args:
            point_in: The point

        Returns:
            Is the point on an edge?
        """
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
        """Check if a point is on a face of the reference.

        Args:
            point_in: The point

        Returns:
            Is the point on a face?
        """
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
        """Check if a point is contained in the reference.

        Args:
            point: The point

        Returns:
            Is the point contained in the reference?
        """
        pass

    def plot_entity_diagrams(
        self, filename: typing.Union[str, typing.List[str]],
        plot_options: typing.Dict[str, typing.Any] = {}, **kwargs: typing.Any
    ):
        """Plot diagrams showing the entity numbering of the reference."""
        from .plotting import Picture, colors

        img = Picture(**kwargs)

        if self.tdim == 1:
            offset_unit: typing.Tuple[typing.Union[
                sympy.core.expr.Expr, int], ...] = (sympy.Rational(4, 3), )
            img.add_arrow((-sympy.Rational(1, 2), 0), (-sympy.Rational(1, 3), 0))
            img.add_math((-sympy.Rational(8, 25), 0), "x", anchor="west")
        elif self.tdim == 2:
            if self.name.startswith("dual polygon"):
                offset_unit = (sympy.Rational(9, 4), 0)
                rt52 = sympy.sqrt(3) / 4
                img.add_arrow((-sympy.Rational(3, 4), -rt52), (-sympy.Rational(7, 12), -rt52))
                img.add_arrow((-sympy.Rational(3, 4), -rt52),
                              (-sympy.Rational(3, 4), sympy.Rational(1, 6) - rt52))
                img.add_math((-sympy.Rational(171, 300), -rt52), "x", anchor="west")
                img.add_math((-sympy.Rational(3, 4), sympy.Rational(9, 50) - rt52), "y",
                             anchor="south")
            else:
                offset_unit = (sympy.Rational(4, 3), 0)
                img.add_arrow((-sympy.Rational(1, 2), 0), (-sympy.Rational(1, 3), 0))
                img.add_arrow((-sympy.Rational(1, 2), 0),
                              (-sympy.Rational(1, 2), sympy.Rational(1, 6)))
                img.add_math((-sympy.Rational(8, 25), 0), "x", anchor="west")
                img.add_math((-sympy.Rational(1, 2), sympy.Rational(9, 50)), "y", anchor="south")
        elif self.tdim == 3:
            if self.name == "pyramid":
                offset_unit = (sympy.Rational(17, 12), sympy.Rational(17, 30), 0)
            elif self.name == "hexahedron":
                offset_unit = (sympy.Rational(3, 2), sympy.Rational(9, 15), 0)
            else:
                offset_unit = (1, sympy.Rational(2, 5), 0)
            img.add_arrow((-sympy.Rational(3, 8), -sympy.Rational(3, 20), 0),
                          (-sympy.Rational(5, 24), -sympy.Rational(3, 20), 0))
            img.add_arrow((-sympy.Rational(3, 8), -sympy.Rational(3, 20), 0),
                          (-sympy.Rational(3, 8), sympy.Rational(1, 60), 0))
            img.add_arrow((-sympy.Rational(3, 8), -sympy.Rational(3, 20), 0),
                          (-sympy.Rational(3, 8), -sympy.Rational(3, 20), sympy.Rational(1, 6)))
            img.add_math((-sympy.Rational(39, 200), -sympy.Rational(3, 20), 0), "x", anchor="west")
            img.add_math((-sympy.Rational(3, 8), sympy.Rational(1, 60), 0), "y",
                         anchor="south west")
            img.add_math((-sympy.Rational(3, 8), -sympy.Rational(3, 20), sympy.Rational(9, 50)),
                         "z", anchor="south")
        else:
            raise ValueError("Unsupported tdim")

        def offset(point, cd):
            return tuple((p + cd * a) / 2 for p, a in zip(point, offset_unit))

        for current_dim in range(self.tdim + 1):
            for entities in self.z_ordered_entities():
                for dim, n in entities:
                    if dim == 1:
                        start, end = [offset(self.vertices[i], current_dim) for i in self.edges[n]]
                        img.add_line(start, end)

                    if dim == current_dim:
                        ref = self.sub_entity(dim, n)
                        img.add_ncircle(offset(ref.midpoint(), current_dim), n, colors.entity(dim))

        img.save(filename, plot_options)


class Point(Reference):
    """A point."""

    def __init__(self, vertices: SetOfPointsInput = ((), )):
        """Create a point.

        Args:
            vertices: The vertices of the point.
        """
        assert len(vertices) == 1
        super().__init__(
            tdim=0,
            name="point",
            origin=vertices[0],
            axes=(),
            reference_vertices=((), ),
            vertices=vertices,
            edges=(),
            faces=(),
            volumes=(),
            sub_entity_types=["point", None, None, None],
            simplex=True, tp=True)

    def default_reference(self) -> Reference:
        """Get the default reference for this cell type.

        Returns:
            The default reference
        """
        return Point(self.reference_vertices)

    def make_lattice(self, n: int) -> SetOfPoints:
        """Make a lattice of points.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points offset from the edge of the cell
        """
        raise NotImplementedError()

    def make_lattice_with_lines(self, n: int) -> LatticeWithLines:
        """Make a lattice of points, and a list of lines connecting them.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points including the edges of the cell
            Pairs of point numbers that make a mesh of lines across the cell
        """
        raise NotImplementedError()

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> IntLimits:
        """Get the limits for an integral over this reference.

        Args:
            vars: The variables to use for each direction

        Returns:
            Integration limits that can be passed into sympy.integrate
        """
        return list(zip(vars, self.vertices[0]))

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell.

        Args:
            vertices: The vertices of a call

        Returns:
            The map
        """
        assert self.vertices == self.reference_vertices
        return parse_point_input(vertices[0])

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference.

        Args:
            vertices_in: The vertices of a cell

        Returns:
            The inverse map
        """
        assert self.vertices == self.reference_vertices
        return self.vertices[0]

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference.

        Returns:
            The map
        """
        return self.vertices[0]

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference.

        Returns:
            The map
        """
        return self.reference_vertices[0]

    def volume(self) -> sympy.core.expr.Expr:
        """Calculate the volume.

        Returns:
            The volume of the cell
        """
        return sympy.Integer(0)

    def contains(self, point: PointType) -> bool:
        """Check if a point is contained in the reference.

        Args:
            point: The point

        Returns:
            Is the point contained in the reference?
        """
        if self.tdim == 0:
            return point in self.vertices
        raise NotImplementedError()


class Interval(Reference):
    """An interval."""

    def __init__(self, vertices: SetOfPointsInput = ((0,), (1,))):
        """Create an interval.

        Args:
            vertices: The vertices of the interval.
        """
        assert len(vertices) == 2
        super().__init__(
            tdim=1,
            name="interval",
            origin=vertices[0],
            axes=(_vsub(vertices[1], vertices[0]),),
            reference_vertices=((0,), (1,)),
            vertices=vertices,
            edges=((0, 1),),
            faces=(),
            volumes=(),
            sub_entity_types=["point", "interval", None, None],
            simplex=True, tp=True)

    def default_reference(self) -> Reference:
        """Get the default reference for this cell type.

        Returns:
            The default reference
        """
        return Interval(self.reference_vertices)

    def make_lattice(self, n: int) -> SetOfPoints:
        """Make a lattice of points.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points offset from the edge of the cell
        """
        assert self.vertices == self.reference_vertices
        return tuple((sympy.Rational(2 * i + 1, 2 * (n + 1)), ) for i in range(n))

    def make_lattice_with_lines(self, n: int) -> LatticeWithLines:
        """Make a lattice of points, and a list of lines connecting them.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points including the edges of the cell
            Pairs of point numbers that make a mesh of lines across the cell
        """
        assert self.vertices == self.reference_vertices
        pts = tuple((sympy.Rational(i, n - 1), ) for i in range(n))
        pairs = [(i, i + 1) for i in range(n - 1)]
        return pts, pairs

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> IntLimits:
        """Get the limits for an integral over this reference.

        Args:
            vars: The variables to use for each direction

        Returns:
            Integration limits that can be passed into sympy.integrate
        """
        return [(vars[0], sympy.Integer(0), sympy.Integer(1))]

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell.

        Args:
            vertices: The vertices of a call

        Returns:
            The map
        """
        assert self.vertices == self.reference_vertices
        return tuple(v0 + (v1 - v0) * x[0] for v0, v1 in zip(*vertices))

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference.

        Args:
            vertices_in: The vertices of a cell

        Returns:
            The inverse map
        """
        assert self.vertices == self.reference_vertices
        vertices = parse_set_of_points_input(vertices_in)
        p = _vsub(tuple(x), vertices[0])
        v = _vsub(vertices[1], vertices[0])
        return (_vdot(p, v) * sympy.Integer(1) / _vdot(v, v), )

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference.

        Returns:
            The map
        """
        return tuple(v0 + (v1 - v0) * x[0] for v0, v1 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference.

        Returns:
            The map
        """
        p = _vsub(tuple(x), self.vertices[0])
        v = _vsub(self.vertices[1], self.vertices[0])
        return (_vdot(p, v) * sympy.Integer(1) / _vdot(v, v), )

    def volume(self) -> sympy.core.expr.Expr:
        """Calculate the volume.

        Returns:
            The volume of the cell
        """
        return self.jacobian()

    def contains(self, point: PointType) -> bool:
        """Check if a point is contained in the reference.

        Args:
            point: The point

        Returns:
            Is the point contained in the reference?
        """
        if self.vertices != self.reference_vertices:
            raise NotImplementedError()
        return 0 <= point[0] <= 1


class Triangle(Reference):
    """A triangle."""

    def __init__(self, vertices: SetOfPointsInput = ((0, 0), (1, 0), (0, 1))):
        """Create a triangle.

        Args:
            vertices: The vertices of the triangle.
        """
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
            volumes=(),
            sub_entity_types=["point", "interval", "triangle", None],
            simplex=True)

    def default_reference(self) -> Reference:
        """Get the default reference for this cell type.

        Returns:
            The default reference
        """
        return Triangle(self.reference_vertices)

    def make_lattice(self, n: int) -> SetOfPoints:
        """Make a lattice of points.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points offset from the edge of the cell
        """
        return tuple(tuple(
            o + ((2 * i + 1) * a0 + (2 * j + 1) * a1) / 2 / (n + 1)
            for o, a0, a1 in zip(self.origin, *self.axes)
        ) for i in range(n) for j in range(n - i))

    def make_lattice_with_lines(self, n: int) -> LatticeWithLines:
        """Make a lattice of points, and a list of lines connecting them.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points including the edges of the cell
            Pairs of point numbers that make a mesh of lines across the cell
        """
        pts = tuple(tuple(
            o + (i * a0 + j * a1) / (n - 1)
            for o, a0, a1 in zip(self.origin, *self.axes)
        ) for i in range(n) for j in range(n - i))
        pairs = []
        s = 0
        for j in range(n-1, 0, -1):
            pairs += [(i, i + 1) for i in range(s, s + j)]
            s += j + 1
        for k in range(n + 1):
            s = k
            for i in range(n, k, -1):
                if i != k + 1:
                    pairs += [(s, s + i)]
                if k != 0:
                    pairs += [(s, s + i - 1)]
                s += i
        return pts, pairs

    def z_ordered_entities_extra_dim(self) -> typing.List[typing.List[typing.Tuple[int, int]]]:
        """Get the subentities in back-to-front plotting order when using an extra dimension.

        Returns:
            List of lists of subentity dimensions and numbers
        """
        return [[(1, 0), (1, 1)], [(0, 2)], [(2, 0), (1, 2), (0, 0), (0, 1)]]

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> IntLimits:
        """Get the limits for an integral over this reference.

        Args:
            vars: The variables to use for each direction

        Returns:
            Integration limits that can be passed into sympy.integrate
        """
        return [(vars[1], sympy.Integer(0), 1 - vars[0]),
                (vars[0], sympy.Integer(0), sympy.Integer(1))]

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell.

        Args:
            vertices: The vertices of a call

        Returns:
            The map
        """
        assert self.vertices == self.reference_vertices
        return tuple(v0 + (v1 - v0) * x[0] + (v2 - v0) * x[1] for v0, v1, v2 in zip(*vertices))

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference.

        Args:
            vertices_in: The vertices of a cell

        Returns:
            The inverse map
        """
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
        """Compute the map from the canonical reference to this reference.

        Returns:
            The map
        """
        return tuple(v0 + (v1 - v0) * x[0] + (v2 - v0) * x[1]
                     for v0, v1, v2 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference.

        Returns:
            The map
        """
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
        """Calculate the volume.

        Returns:
            The volume of the cell
        """
        return sympy.Rational(1, 2) * self.jacobian()

    def contains(self, point: PointType) -> bool:
        """Check if a point is contained in the reference.

        Args:
            point: The point

        Returns:
            Is the point contained in the reference?
        """
        if self.vertices != self.reference_vertices:
            raise NotImplementedError()
        return 0 <= point[0] and 0 <= point[1] and sum(point) <= 1


class Tetrahedron(Reference):
    """A tetrahedron."""

    def __init__(self, vertices: SetOfPointsInput = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))):
        """Create a tetrahedron.

        Args:
            vertices: The vertices of the tetrahedron.
        """
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

    @property
    def clockwise_vertices(self) -> SetOfPoints:
        """Get list of vertices in clockwise order.

        Returns:
            A list of vertices
        """
        return (self.vertices[0], self.vertices[1], self.vertices[3])

    def z_ordered_entities(self) -> typing.List[typing.List[typing.Tuple[int, int]]]:
        """Get the subentities of the cell in back-to-front plotting order.

        Returns:
            List of lists of subentity dimensions and numbers
        """
        return [
            [(2, 0), (2, 1), (2, 3), (1, 0), (1, 2), (1, 4), (0, 2)],
            [(3, 0)],
            [(2, 2), (1, 1), (1, 3), (1, 5), (0, 0), (0, 1), (0, 3)]
        ]

    def default_reference(self) -> Reference:
        """Get the default reference for this cell type.

        Returns:
            The default reference
        """
        return Tetrahedron(self.reference_vertices)

    def make_lattice(self, n: int) -> SetOfPoints:
        """Make a lattice of points.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points offset from the edge of the cell
        """
        return tuple(tuple(
            o + ((2 * i + 1) * a0 + (2 * j + 1) * a1 + (2 * k + 1) * a2) / 2 / (n + 1)
            for o, a0, a1, a2 in zip(self.origin, *self.axes)
        ) for i in range(n) for j in range(n - i) for k in range(n - i - j))

    def make_lattice_with_lines(self, n: int) -> LatticeWithLines:
        """Make a lattice of points, and a list of lines connecting them.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points including the edges of the cell
            Pairs of point numbers that make a mesh of lines across the cell
        """
        raise NotImplementedError()

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> IntLimits:
        """Get the limits for an integral over this reference.

        Args:
            vars: The variables to use for each direction

        Returns:
            Integration limits that can be passed into sympy.integrate
        """
        return [(vars[0], sympy.Integer(0), 1 - vars[1] - vars[2]),
                (vars[1], sympy.Integer(0), 1 - vars[2]),
                (vars[2], sympy.Integer(0), sympy.Integer(1))]

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell.

        Args:
            vertices: The vertices of a call

        Returns:
            The map
        """
        assert self.vertices == self.reference_vertices
        return tuple(v0 + (v1 - v0) * x[0] + (v2 - v0) * x[1] + (v3 - v0) * x[2]
                     for v0, v1, v2, v3 in zip(*vertices))

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference.

        Args:
            vertices_in: The vertices of a cell

        Returns:
            The inverse map
        """
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
        """Compute the map from the canonical reference to this reference.

        Returns:
            The map
        """
        return tuple(v0 + (v1 - v0) * x[0] + (v2 - v0) * x[1] + (v3 - v0) * x[2]
                     for v0, v1, v2, v3 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference.

        Returns:
            The map
        """
        p = _vsub(tuple(x), self.vertices[0])
        v1 = _vsub(self.vertices[1], self.vertices[0])
        v2 = _vsub(self.vertices[2], self.vertices[0])
        v3 = _vsub(self.vertices[3], self.vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0], v3[0]],
                            [v1[1], v2[1], v3[1]],
                            [v1[2], v2[2], v3[2]]]).inv()
        return (_vdot(mat.row(0), p), _vdot(mat.row(1), p), _vdot(mat.row(2), p))

    def volume(self) -> sympy.core.expr.Expr:
        """Calculate the volume.

        Returns:
            The volume of the cell
        """
        return sympy.Rational(1, 6) * self.jacobian()

    def contains(self, point: PointType) -> bool:
        """Check if a point is contained in the reference.

        Args:
            point: The point

        Returns:
            Is the point contained in the reference?
        """
        if self.vertices != self.reference_vertices:
            raise NotImplementedError()
        return 0 <= point[0] and 0 <= point[1] and 0 <= point[2] and sum(point) <= 1


class Quadrilateral(Reference):
    """A quadrilateral."""

    def __init__(self, vertices: SetOfPointsInput = ((0, 0), (1, 0), (0, 1), (1, 1))):
        """Create a quadrilateral.

        Args:
            vertices: The vertices of the quadrilateral.
        """
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
            volumes=(),
            sub_entity_types=["point", "interval", "quadrilateral", None],
            tp=True)

    @property
    def clockwise_vertices(self) -> SetOfPoints:
        """Get list of vertices in clockwise order.

        Returns:
            A list of vertices
        """
        return (self.vertices[0], self.vertices[1], self.vertices[3], self.vertices[2])

    def default_reference(self) -> Reference:
        """Get the default reference for this cell type.

        Returns:
            The default reference
        """
        return Quadrilateral(self.reference_vertices)

    def make_lattice(self, n: int) -> SetOfPoints:
        """Make a lattice of points.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points offset from the edge of the cell
        """
        return tuple(tuple(
            o + ((2 * i + 1) * a0 + (2 * j + 1) * a1) / 2 / (n + 1)
            for o, a0, a1 in zip(self.origin, *self.axes)
        ) for i in range(n + 1) for j in range(n + 1))

    def make_lattice_with_lines(self, n: int) -> LatticeWithLines:
        """Make a lattice of points, and a list of lines connecting them.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points including the edges of the cell
            Pairs of point numbers that make a mesh of lines across the cell
        """
        pts = tuple(tuple(
            o + (i * a0 + j * a1) / (n - 1)
            for o, a0, a1 in zip(self.origin, *self.axes)
        ) for i in range(n) for j in range(n))
        pairs = []
        for i in range(n):
            for j in range(n):
                node = i * n + j
                if j != n - 1:
                    pairs += [(node, node + 1)]
                if i != n - 1:
                    pairs += [(node, node + n)]
                    if j != 0:
                        pairs += [(node, node + n - 1)]
        return pts, pairs

    def z_ordered_entities_extra_dim(self) -> typing.List[typing.List[typing.Tuple[int, int]]]:
        """Get the subentities in back-to-front plotting order when using an extra dimension.

        Returns:
            List of lists of subentity dimensions and numbers
        """
        return [[(1, 3), (1, 1), (0, 2)], [(2, 0)], [(1, 2), (0, 3), (1, 0), (0, 0), (0, 1)]]

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> IntLimits:
        """Get the limits for an integral over this reference.

        Args:
            vars: The variables to use for each direction

        Returns:
            Integration limits that can be passed into sympy.integrate
        """
        return [(vars[1], sympy.Integer(0), sympy.Integer(1)),
                (vars[0], sympy.Integer(0), sympy.Integer(1))]

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell.

        Args:
            vertices: The vertices of a call

        Returns:
            The map
        """
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1) + x[1] * ((1 - x[0]) * v2 + x[0] * v3)
            for v0, v1, v2, v3 in zip(*vertices))

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference.

        Args:
            vertices_in: The vertices of a cell

        Returns:
            The inverse map
        """
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
        """Compute the map from the canonical reference to this reference.

        Returns:
            The map
        """
        return tuple(
            (1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1) + x[1] * ((1 - x[0]) * v2 + x[0] * v3)
            for v0, v1, v2, v3 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference.

        Returns:
            The map
        """
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
        """Calculate the volume.

        Returns:
            The volume of the cell
        """
        return self.jacobian()

    def contains(self, point: PointType) -> bool:
        """Check if a point is contained in the reference.

        Args:
            point: The point

        Returns:
            Is the point contained in the reference?
        """
        if self.vertices != self.reference_vertices:
            raise NotImplementedError()
        return 0 <= point[0] <= 1 and 0 <= point[1] <= 1


class Hexahedron(Reference):
    """A hexahedron."""

    def __init__(self, vertices: SetOfPointsInput = (
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))
    ):
        """Create a hexahedron.

        Args:
            vertices: The vertices of the hexahedron.
        """
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

    @property
    def clockwise_vertices(self) -> SetOfPoints:
        """Get list of vertices in clockwise order.

        Returns:
            A list of vertices
        """
        return (self.vertices[0], self.vertices[1], self.vertices[3], self.vertices[7],
                self.vertices[6], self.vertices[4])

    def z_ordered_entities(self) -> typing.List[typing.List[typing.Tuple[int, int]]]:
        """Get the subentities of the cell in back-to-front plotting order.

        Returns:
            List of lists of subentity dimensions and numbers
        """
        return [
            [(2, 4), (2, 0), (1, 5), (2, 2), (1, 6), (1, 1), (0, 2)],
            [(3, 0)],
            [(2, 3), (1, 3), (1, 7), (0, 3)],
            [(2, 1), (1, 0), (1, 2), (1, 4), (0, 0), (0, 1)],
            [(2, 5), (1, 8), (1, 9), (1, 10), (1, 11), (0, 4), (0, 5), (0, 6), (0, 7)]
        ]

    def default_reference(self) -> Reference:
        """Get the default reference for this cell type.

        Returns:
            The default reference
        """
        return Hexahedron(self.reference_vertices)

    def make_lattice(self, n: int) -> SetOfPoints:
        """Make a lattice of points.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points offset from the edge of the cell
        """
        assert self.vertices == self.reference_vertices
        return tuple((
            sympy.Rational(2 * i + 1, 2 * (n + 1)),
            sympy.Rational(2 * j + 1, 2 * (n + 1)),
            sympy.Rational(2 * k + 1, 2 * (n + 1))
        ) for i in range(n + 1) for j in range(n + 1) for k in range(n + 1))

    def make_lattice_with_lines(self, n: int) -> LatticeWithLines:
        """Make a lattice of points, and a list of lines connecting them.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points including the edges of the cell
            Pairs of point numbers that make a mesh of lines across the cell
        """
        raise NotImplementedError()

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> IntLimits:
        """Get the limits for an integral over this reference.

        Args:
            vars: The variables to use for each direction

        Returns:
            Integration limits that can be passed into sympy.integrate
        """
        return [(vars[2], sympy.Integer(0), sympy.Integer(1)),
                (vars[1], sympy.Integer(0), sympy.Integer(1)),
                (vars[0], sympy.Integer(0), sympy.Integer(1))]

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell.

        Args:
            vertices: The vertices of a call

        Returns:
            The map
        """
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[2]) * ((1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1)
                          + x[1] * ((1 - x[0]) * v2 + x[0] * v3))
            + x[2] * ((1 - x[1]) * ((1 - x[0]) * v4 + x[0] * v5)
                      + x[1] * ((1 - x[0]) * v6 + x[0] * v7))
            for v0, v1, v2, v3, v4, v5, v6, v7 in zip(*vertices))

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference.

        Args:
            vertices_in: The vertices of a cell

        Returns:
            The inverse map
        """
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
        """Compute the map from the canonical reference to this reference.

        Returns:
            The map
        """
        return tuple(
            (1 - x[2]) * ((1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1)
                          + x[1] * ((1 - x[0]) * v2 + x[0] * v3))
            + x[2] * ((1 - x[1]) * ((1 - x[0]) * v4 + x[0] * v5)
                      + x[1] * ((1 - x[0]) * v6 + x[0] * v7))
            for v0, v1, v2, v3, v4, v5, v6, v7 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference.

        Returns:
            The map
        """
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
        """Calculate the volume.

        Returns:
            The volume of the cell
        """
        return self.jacobian()

    def contains(self, point: PointType) -> bool:
        """Check if a point is contained in the reference.

        Args:
            point: The point

        Returns:
            Is the point contained in the reference?
        """
        if self.vertices != self.reference_vertices:
            raise NotImplementedError()
        return 0 <= point[0] <= 1 and 0 <= point[1] <= 1 and 0 <= point[2] <= 1


class Prism(Reference):
    """A (triangular) prism."""

    def __init__(self, vertices: SetOfPointsInput = (
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1))
    ):
        """Create a prism.

        Args:
            vertices: The vertices of the prism.
        """
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

    @property
    def clockwise_vertices(self) -> SetOfPoints:
        """Get list of vertices in clockwise order.

        Returns:
            A list of vertices
        """
        return (self.vertices[0], self.vertices[1], self.vertices[4], self.vertices[5],
                self.vertices[3])

    def z_ordered_entities(self) -> typing.List[typing.List[typing.Tuple[int, int]]]:
        """Get the subentities of the cell in back-to-front plotting order.

        Returns:
            List of lists of subentity dimensions and numbers
        """
        return [
            [(2, 0), (2, 3), (1, 3)],
            [(2, 2), (1, 1), (1, 5), (0, 2)],
            [(3, 0)],
            [(2, 1), (1, 0), (1, 2), (1, 4), (0, 0), (0, 1)],
            [(2, 4), (1, 6), (1, 7), (1, 8), (0, 3), (0, 4), (0, 5)]
        ]

    def default_reference(self) -> Reference:
        """Get the default reference for this cell type.

        Returns:
            The default reference
        """
        return Prism(self.reference_vertices)

    def make_lattice(self, n: int) -> SetOfPoints:
        """Make a lattice of points.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points offset from the edge of the cell
        """
        assert self.vertices == self.reference_vertices
        return tuple((
            sympy.Rational(2 * i + 1, 2 * (n + 1)),
            sympy.Rational(2 * j + 1, 2 * (n + 1)),
            sympy.Rational(2 * k + 1, 2 * (n + 1))
        ) for i in range(n + 1) for j in range(n + 1 - i) for k in range(n + 1))

    def make_lattice_with_lines(self, n: int) -> LatticeWithLines:
        """Make a lattice of points, and a list of lines connecting them.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points including the edges of the cell
            Pairs of point numbers that make a mesh of lines across the cell
        """
        raise NotImplementedError()

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> IntLimits:
        """Get the limits for an integral over this reference.

        Args:
            vars: The variables to use for each direction

        Returns:
            Integration limits that can be passed into sympy.integrate
        """
        return [(vars[2], sympy.Integer(0), sympy.Integer(1)),
                (vars[1], sympy.Integer(0), sympy.Integer(1) - vars[0]),
                (vars[0], sympy.Integer(0), sympy.Integer(1))]

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell.

        Args:
            vertices: The vertices of a call

        Returns:
            The map
        """
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[2]) * (v0 + x[0] * (v1 - v0) + x[1] * (v2 - v0))
            + x[2] * (v3 + x[0] * (v4 - v3) + x[1] * (v5 - v3))
            for v0, v1, v2, v3, v4, v5 in zip(*vertices))

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference.

        Args:
            vertices_in: The vertices of a cell

        Returns:
            The inverse map
        """
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
        """Compute the map from the canonical reference to this reference.

        Returns:
            The map
        """
        return tuple(
            (1 - x[2]) * (v0 + x[0] * (v1 - v0) + x[1] * (v2 - v0))
            + x[2] * (v3 + x[0] * (v4 - v3) + x[1] * (v5 - v3))
            for v0, v1, v2, v3, v4, v5 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference.

        Returns:
            The map
        """
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
        """Calculate the volume.

        Returns:
            The volume of the cell
        """
        return sympy.Rational(1, 2) * self.jacobian()

    def contains(self, point: PointType) -> bool:
        """Check if a point is contained in the reference.

        Args:
            point: The point

        Returns:
            Is the point contained in the reference?
        """
        if self.vertices != self.reference_vertices:
            raise NotImplementedError()
        return (point[0] >= 0 and point[1] >= 0 and point[2] >= 0
                and point[2] <= 1 and point[0] + point[1] <= 1)


class Pyramid(Reference):
    """A (square-based) pyramid."""

    def __init__(self, vertices: SetOfPointsInput = (
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1))
    ):
        """Create a pyramid.

        Args:
            vertices: The vertices of the pyramid.
        """
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

    @property
    def clockwise_vertices(self) -> SetOfPoints:
        """Get list of vertices in clockwise order.

        Returns:
            A list of vertices
        """
        return (self.vertices[0], self.vertices[1], self.vertices[3], self.vertices[4])

    def z_ordered_entities(self) -> typing.List[typing.List[typing.Tuple[int, int]]]:
        """Get the subentities of the cell in back-to-front plotting order.

        Returns:
            List of lists of subentity dimensions and numbers
        """
        return [
            [(2, 0), (2, 4), (1, 5)],
            [(2, 2), (1, 1), (1, 6), (0, 2)],
            [(3, 0)],
            [(2, 3), (1, 3), (1, 7), (0, 3)],
            [(2, 1), (1, 0), (1, 2), (1, 4), (0, 0), (0, 1), (0, 4)]
        ]

    def default_reference(self) -> Reference:
        """Get the default reference for this cell type.

        Returns:
            The default reference
        """
        return Pyramid(self.reference_vertices)

    def make_lattice(self, n: int) -> SetOfPoints:
        """Make a lattice of points.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points offset from the edge of the cell
        """
        raise NotImplementedError()

    def make_lattice_with_lines(self, n: int) -> LatticeWithLines:
        """Make a lattice of points, and a list of lines connecting them.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points including the edges of the cell
            Pairs of point numbers that make a mesh of lines across the cell
        """
        raise NotImplementedError()

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> IntLimits:
        """Get the limits for an integral over this reference.

        Args:
            vars: The variables to use for each direction

        Returns:
            Integration limits that can be passed into sympy.integrate
        """
        return [(vars[0], sympy.Integer(0), 1 - vars[2]),
                (vars[1], sympy.Integer(0), 1 - vars[2]),
                (vars[2], sympy.Integer(0), sympy.Integer(1))]

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell.

        Args:
            vertices: The vertices of a call

        Returns:
            The map
        """
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[2]) * (
                (1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1)
                + x[1] * ((1 - x[0]) * v2 + x[0] * v3)
            ) + x[2] * v4
            for v0, v1, v2, v3, v4 in zip(*vertices))

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference.

        Args:
            vertices_in: The vertices of a cell

        Returns:
            The inverse map
        """
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
        """Compute the map from the canonical reference to this reference.

        Returns:
            The map
        """
        return tuple(
            (1 - x[2]) * (
                (1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1)
                + x[1] * ((1 - x[0]) * v2 + x[0] * v3)
            ) + x[2] * v4
            for v0, v1, v2, v3, v4 in zip(*self.vertices))

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference.

        Returns:
            The map
        """
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
        """Calculate the volume.

        Returns:
            The volume of the cell
        """
        return sympy.Rational(1, 3) * self.jacobian()

    def contains(self, point: PointType) -> bool:
        """Check if a point is contained in the reference.

        Args:
            point: The point

        Returns:
            Is the point contained in the reference?
        """
        if self.vertices != self.reference_vertices:
            raise NotImplementedError()
        return (point[0] >= 0 and point[1] >= 0 and point[2] >= 0
                and point[0] + point[2] <= 1 and point[1] + point[2] <= 1)


class DualPolygon(Reference):
    """A polygon on a barycentric dual grid."""

    reference_origin: PointType

    def __init__(
        self, number_of_triangles: int, vertices: typing.Optional[SetOfPointsInput] = None
    ):
        """Create a dual polygon.

        Args:
            number_of_triangles: The number of triangles that make up the dual polygon
            vertices: The vertices of the dual polygon.
        """
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
            axes=(),
            origin=origin,
            vertices=vertices,
            reference_vertices=tuple(reference_vertices),
            edges=tuple((i, (i + 1) % (2 * number_of_triangles))
                        for i in range(2 * number_of_triangles)),
            faces=(tuple(range(2 * number_of_triangles)), ),
            volumes=(),
            sub_entity_types=["point", "interval", f"dual polygon({number_of_triangles})", None],

        )

    def contains(self, point: PointType) -> bool:
        """Check if a point is contained in the reference.

        Args:
            point: The point

        Returns:
            Is the point contained in the reference?
        """
        raise NotImplementedError()

    def integration_limits(self, vars: AxisVariablesNotSingle = t) -> IntLimits:
        """Get the limits for an integral over this reference.

        Args:
            vars: The variables to use for each direction

        Returns:
            Integration limits that can be passed into sympy.integrate
        """
        raise NotImplementedError()

    def get_map_to(self, vertices: SetOfPointsInput) -> PointType:
        """Get the map from the reference to a cell.

        Args:
            vertices: The vertices of a call

        Returns:
            The map
        """
        raise NotImplementedError()

    def get_inverse_map_to(self, vertices_in: SetOfPointsInput) -> PointType:
        """Get the inverse map from a cell to the reference.

        Args:
            vertices_in: The vertices of a cell

        Returns:
            The inverse map
        """
        raise NotImplementedError()

    def _compute_map_to_self(self) -> PointType:
        """Compute the map from the canonical reference to this reference.

        Returns:
            The map
        """
        raise NotImplementedError()

    def _compute_inverse_map_to_self(self) -> PointType:
        """Compute the inverse map from the canonical reference to this reference.

        Returns:
            The map
        """
        raise NotImplementedError()

    def volume(self) -> sympy.core.expr.Expr:
        """Calculate the volume.

        Returns:
            The volume of the cell
        """
        raise NotImplementedError()

    def default_reference(self) -> Reference:
        """Get the default reference for this cell type.

        Returns:
            The default reference
        """
        raise NotImplementedError()

    def make_lattice(self, n: int) -> SetOfPoints:
        """Make a lattice of points.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points offset from the edge of the cell
        """
        assert self.vertices == self.reference_vertices
        from .create import create_reference
        lattice: SetOfPoints = ()
        for v1, v2 in zip(self.vertices, self.vertices[1:] + (self.vertices[0], )):
            ref = create_reference("triangle", (self.origin, v1, v2))
            lattice += ref.make_lattice(n // 2)
        return lattice

    def make_lattice_with_lines(self, n: int) -> LatticeWithLines:
        """Make a lattice of points, and a list of lines connecting them.

        Args:
            n: The number of points along each edge

        Returns:
            A lattice of points including the edges of the cell
            Pairs of point numbers that make a mesh of lines across the cell
        """
        raise NotImplementedError()
