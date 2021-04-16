"""Reference elements."""

import sympy
from .symbolic import t, x
from .vectors import vsub, vnorm, vdot, vcross, vnormalise, vadd


class Reference:
    """A reference element."""

    def __init__(self, simplex=False, tp=False):
        self.gdim = len(self.origin)
        self.simplex = simplex
        self.tp = tp

    def integral(self, f):
        """Calculate the integral over the element."""
        raise NotImplementedError

    def get_map_to(self, vertices):
        """Get the map from the reference to a cell."""
        raise NotImplementedError

    def get_inverse_map_to(self, vertices):
        """Get the inverse map from a cell to the reference."""
        raise NotImplementedError

    def jacobian(self):
        """Calculate the jacobian."""
        assert len(self.axes) == self.tdim
        if self.tdim == 1:
            return vnorm(self.axes[0])
        if self.tdim == 2:
            return vnorm(vcross(self.axes[0], self.axes[1]))
        if self.tdim == 3:
            return vnorm(vdot(vcross(self.axes[0], self.axes[1]), self.axes[2]))

    def scaled_axes(self):
        """Return the unit axes of the reference."""
        return [vnormalise(a) for a in self.axes]

    def tangent(self):
        """Calculate the tangent to the element."""
        if self.tdim == 1:
            norm = sympy.sqrt(sum(i ** 2 for i in self.axes[0]))
            return vnormalise(tuple(i / norm for i in self.axes[0]))

        raise RuntimeError

    def normal(self):
        """Calculate the normal to the element."""
        if self.tdim == 1:
            if self.gdim == 2:
                return vnormalise((-self.axes[0][1], self.axes[0][0]))
        if self.tdim == 2:
            if self.gdim == 3:
                return vnormalise(vcross(self.axes[0], self.axes[1]))
        raise RuntimeError

    def sub_entities(self, dim=None, codim=None):
        """Get the sub entities of a given dimension."""
        if dim is None:
            dim = self.tdim - codim
        if dim == 0:
            return self.vertices
        if dim == 1:
            return self.edges
        if dim == 2:
            return self.faces
        if dim == 3:
            return self.volumes

    def sub_entity_count(self, dim):
        """Get the number of sub entities of a given dimension."""
        return len(self.sub_entities(dim))

    def sub_entity(self, dim, n):
        """Get the sub entity of a given dimension and number."""
        from symfem import create_reference

        entity_type = self.sub_entity_types[dim]
        if not isinstance(entity_type, str):
            entity_type = entity_type[n]

        return create_reference(
            entity_type, [self.vertices[i] for i in self.sub_entities(dim)[n]])

    def at_vertex(self, point):
        """Check if a point is a vertex of the reference."""
        for v in self.vertices:
            if v == tuple(point):
                return True
        return False

    def on_edge(self, point):
        """Check if a point is on an edge of the reference."""
        for e in self.edges:
            v0 = self.vertices[e[0]]
            v1 = self.vertices[e[1]]
            if vnorm(vcross(vsub(v0, point), vsub(v1, point))) == 0:
                return True
        return False

    def on_face(self, point):
        """Check if a point is on a face of the reference."""
        for f in self.faces:
            v0 = self.vertices[f[0]]
            v1 = self.vertices[f[1]]
            v2 = self.vertices[f[2]]
            if vdot(vcross(vsub(v0, point), vsub(v1, point)), vsub(v2, point)):
                return True
        return False


class Interval(Reference):
    """An interval."""

    def __init__(self, vertices=((0,), (1,))):
        self.tdim = 1
        self.name = "interval"
        self.origin = vertices[0]
        self.axes = (vsub(vertices[1], vertices[0]),)
        self.reference_vertices = ((0,), (1,))
        self.vertices = tuple(vertices)
        self.edges = ((0, 1),)
        self.faces = tuple()
        self.volumes = tuple()
        self.sub_entity_types = ["point", "interval", None, None]
        super().__init__(simplex=True, tp=True)

    def integral(self, f):
        """Calculate the integral over the element."""
        return (f * self.jacobian()).integrate((t[0], 0, 1))

    def get_map_to(self, vertices):
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(v0 + (v1 - v0) * x[0] for v0, v1 in zip(*vertices))

    def get_inverse_map_to(self, vertices):
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        p = vsub(x, vertices[0])
        v = vsub(vertices[1], vertices[0])
        return (vdot(p, v) / vdot(v, v), )


class Triangle(Reference):
    """A triangle."""

    def __init__(self, vertices=((0, 0), (1, 0), (0, 1))):
        self.tdim = 2
        self.name = "triangle"
        self.origin = vertices[0]
        self.axes = (vsub(vertices[1], vertices[0]), vsub(vertices[2], vertices[0]))
        self.reference_vertices = ((0, 0), (1, 0), (0, 1))
        self.vertices = tuple(vertices)
        self.edges = ((1, 2), (0, 2), (0, 1))
        self.faces = ((0, 1, 2),)
        self.volumes = tuple()
        self.sub_entity_types = ["point", "interval", "triangle", None]
        super().__init__(simplex=True)

    def integral(self, f):
        """Calculate the integral over the element."""
        return (
            (f * self.jacobian()).integrate((t[1], 0, 1 - t[0])).integrate((t[0], 0, 1))
        )

    def get_map_to(self, vertices):
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(v0 + (v1 - v0) * x[0] + (v2 - v0) * x[1] for v0, v1, v2 in zip(*vertices))

    def get_inverse_map_to(self, vertices):
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        assert len(vertices[0]) == 2
        p = vsub(x, vertices[0])
        v1 = vsub(vertices[1], vertices[0])
        v2 = vsub(vertices[2], vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0]],
                            [v1[1], v2[1]]]).inv()
        return (vdot(mat.row(0), p), vdot(mat.row(1), p))


class Tetrahedron(Reference):
    """A tetrahedron."""

    def __init__(self, vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))):
        self.tdim = 3
        self.name = "tetrahedron"
        self.origin = vertices[0]
        self.axes = (
            vsub(vertices[1], vertices[0]),
            vsub(vertices[2], vertices[0]),
            vsub(vertices[3], vertices[0]),
        )
        self.reference_vertices = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        self.vertices = tuple(vertices)
        self.edges = ((2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1))
        self.faces = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
        self.volumes = ((0, 1, 2, 3),)
        self.sub_entity_types = ["point", "interval", "triangle", "tetrahedron"]
        super().__init__(simplex=True)

    def integral(self, f):
        """Calculate the integral over the element."""
        return (
            (f * self.jacobian())
            .integrate((t[2], 0, 1 - t[0] - t[1]))
            .integrate((t[1], 0, 1 - t[0]))
            .integrate((t[0], 0, 1))
        )

    def get_map_to(self, vertices):
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(v0 + (v1 - v0) * x[0] + (v2 - v0) * x[1] + (v3 - v0) * x[2]
                     for v0, v1, v2, v3 in zip(*vertices))

    def get_inverse_map_to(self, vertices):
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


class Quadrilateral(Reference):
    """A quadrilateral."""

    def __init__(self, vertices=((0, 0), (1, 0), (0, 1), (1, 1))):
        self.tdim = 2
        self.name = "quadrilateral"
        self.origin = vertices[0]
        self.axes = (vsub(vertices[1], vertices[0]), vsub(vertices[2], vertices[0]))
        self.reference_vertices = ((0, 0), (1, 0), (0, 1), (1, 1))
        self.vertices = tuple(vertices)
        self.edges = ((0, 1), (0, 2), (1, 3), (2, 3))
        self.faces = ((0, 1, 2, 3),)
        self.volumes = tuple()
        self.sub_entity_types = ["point", "interval", "quadrilateral", None]
        super().__init__(tp=True)

    def integral(self, f):
        """Calculate the integral over the element."""
        return (f * self.jacobian()).integrate((t[1], 0, 1)).integrate((t[0], 0, 1))

    def get_map_to(self, vertices):
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1) + x[1] * ((1 - x[0]) * v2 + x[0] * v3)
            for v0, v1, v2, v3 in zip(*vertices))

    def get_inverse_map_to(self, vertices):
        """Get the inverse map from a cell to the reference."""
        assert self.vertices == self.reference_vertices
        assert len(vertices[0]) == 2
        assert vadd(vertices[0], vertices[3]) == vadd(vertices[1], vertices[2])
        p = vsub(x, vertices[0])
        v1 = vsub(vertices[1], vertices[0])
        v2 = vsub(vertices[2], vertices[0])
        mat = sympy.Matrix([[v1[0], v2[0]],
                            [v1[1], v2[1]]]).inv()
        return (vdot(mat.row(0), p), vdot(mat.row(1), p))


class Hexahedron(Reference):
    """A hexahedron."""

    def __init__(self, vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                                 (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))):
        self.tdim = 3
        self.name = "hexahedron"
        self.origin = vertices[0]
        self.axes = (
            vsub(vertices[1], vertices[0]),
            vsub(vertices[2], vertices[0]),
            vsub(vertices[4], vertices[0]),
        )
        self.reference_vertices = (
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))
        self.vertices = tuple(vertices)
        self.edges = (
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
        self.faces = (
            (0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6),
            (1, 3, 5, 7), (2, 3, 6, 7), (4, 5, 6, 7))
        self.volumes = ((0, 1, 2, 3, 4, 5, 6, 7),)
        self.sub_entity_types = ["point", "interval", "quadrilateral", "hexahedron"]
        super().__init__(tp=True)

    def integral(self, f):
        """Calculate the integral over the element."""
        return (
            (f * self.jacobian())
            .integrate((t[2], 0, 1))
            .integrate((t[1], 0, 1))
            .integrate((t[0], 0, 1))
        )

    def get_map_to(self, vertices):
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[2]) * ((1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1)
                          + x[1] * ((1 - x[0]) * v2 + x[0] * v3))
            + x[2] * ((1 - x[1]) * ((1 - x[0]) * v4 + x[0] * v5)
                      + x[1] * ((1 - x[0]) * v6 + x[0] * v7))
            for v0, v1, v2, v3, v4, v5, v6, v7 in zip(*vertices))

    def get_inverse_map_to(self, vertices):
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


class Prism(Reference):
    """A (triangular) prism."""

    def __init__(self, vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0),
                                 (0, 0, 1), (1, 0, 1), (0, 1, 1))):
        self.tdim = 3
        self.name = "prism"
        self.origin = vertices[0]
        self.axes = (
            vsub(vertices[1], vertices[0]),
            vsub(vertices[2], vertices[0]),
            vsub(vertices[3], vertices[0]),
        )
        self.reference_vertices = (
            (0, 0, 0), (1, 0, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1))
        self.vertices = tuple(vertices)
        self.edges = (
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 4),
            (2, 5), (3, 4), (3, 5), (4, 5))
        self.faces = (
            (0, 1, 2), (0, 1, 3, 4), (0, 2, 3, 5),
            (1, 2, 4, 5), (3, 4, 5))
        self.volumes = ((0, 1, 2, 3, 4, 5),)
        self.sub_entity_types = [
            "point", "interval",
            ["triangle", "quadrilateral", "quadrilateral", "quadrilateral", "triangle"],
            "prism"]
        super().__init__(tp=True)

    def integral(self, f):
        """Calculate the integral over the element."""
        return (
            (f * self.jacobian())
            .integrate((t[2], 0, 1))
            .integrate((t[1], 0, 1 - t[0]))
            .integrate((t[0], 0, 1))
        )

    def get_map_to(self, vertices):
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[2]) * (v0 + x[0] * (v1 - v0) + x[1] * (v2 - v0))
            + x[2] * (v3 + x[0] * (v4 - v3) + x[1] * (v5 - v3))
            for v0, v1, v2, v3, v4, v5 in zip(*vertices))

    def get_inverse_map_to(self, vertices):
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


class Pyramid(Reference):
    """A (square-based) pyramid."""

    def __init__(self, vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0),
                                 (1, 1, 0), (0, 0, 1))):
        self.tdim = 3
        self.name = "pyramid"
        self.origin = vertices[0]
        self.axes = (
            vsub(vertices[1], vertices[0]),
            vsub(vertices[2], vertices[0]),
            vsub(vertices[4], vertices[0]),
        )
        self.reference_vertices = (
            (0, 0, 0), (1, 0, 0), (0, 1, 0),
            (1, 1, 0), (0, 0, 1))
        self.vertices = tuple(vertices)
        self.edges = (
            (0, 1), (0, 2), (0, 4), (1, 3),
            (1, 4), (2, 3), (2, 4), (3, 4))
        self.faces = (
            (0, 1, 2, 3), (0, 1, 4), (0, 2, 4),
            (1, 3, 4), (2, 3, 4))
        self.volumes = ((0, 1, 2, 3, 4),)
        self.sub_entity_types = [
            "point", "interval",
            ["quadrilateral", "triangle", "triangle", "triangle", "triangle"],
            "pyramid"]
        super().__init__(tp=True)

    def integral(self, f):
        """Calculate the integral over the element."""
        return (
            (f * self.jacobian())
            .integrate((t[2], 0, 1))
            .integrate((t[1], 0, 1 - t[0]))
            .integrate((t[0], 0, 1))
        )

    def get_map_to(self, vertices):
        """Get the map from the reference to a cell."""
        assert self.vertices == self.reference_vertices
        return tuple(
            (1 - x[2]) * (
                (1 - x[1]) * ((1 - x[0]) * v0 + x[0] * v1)
                + x[1] * ((1 - x[0]) * v2 + x[0] * v3)
            ) + x[2] * v4
            for v0, v1, v2, v3, v4 in zip(*vertices))

    def get_inverse_map_to(self, vertices):
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


class DualPolygon(Reference):
    """A polygon on a barycentric dual grid."""

    def __init__(self, number_of_triangles, vertices=None):
        self.tdim = 2
        self.name = "dual polygon"
        self.number_of_triangles = number_of_triangles
        self.reference_origin = (0, 0)
        self.reference_vertices = []
        for tri in range(number_of_triangles):
            angle = sympy.pi * 2 * tri / number_of_triangles
            next_angle = sympy.pi * 2 * (tri + 1) / number_of_triangles

            self.reference_vertices.append((sympy.cos(angle), sympy.sin(angle)))
            self.reference_vertices.append(
                ((sympy.cos(next_angle) + sympy.cos(angle)) / 2,
                 (sympy.sin(next_angle) + sympy.sin(angle)) / 2))

        if vertices is None:
            self.origin = self.reference_origin
            self.vertices = self.reference_vertices
        else:
            assert len(vertices) == 1 + len(self.reference_vertices)
            self.origin = vertices[0]
            self.vertices = vertices[1:]

        self.edges = tuple((i, (i + 1) % (2 * number_of_triangles))
                           for i in range(2 * number_of_triangles))
        self.faces = (tuple(range(2 * number_of_triangles)), )
        self.volumes = tuple()
        self.sub_entity_types = ["point", "interval", f"dual polygon({number_of_triangles})", None]

        super().__init__()
