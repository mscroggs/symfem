"""Reference elements."""
import sympy
from .symbolic import t
from .vectors import vsub, vnorm, vdot, vcross, vnormalise


class Reference:
    """A reference element."""

    def __init__(self, simplex=False, tp=False):
        self.gdim = len(self.origin)
        self.simplex = simplex
        self.tp = tp

    def integral(self, f):
        """Calculate the integral over the element."""
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

    def sub_entities(self, dim):
        """Get the sub entities of a given dimension."""
        if dim == 0:
            return self.vertices
        if dim == 1:
            return self.edges
        if dim == 2:
            return self.faces
        if dim == 3:
            return self.volumes

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
        self.vertices = vertices
        self.edges = ((0, 1),)
        self.faces = tuple()
        self.volumes = tuple()
        self.sub_entity_types = ["point", "interval", None, None]
        super().__init__(simplex=True, tp=True)

    def integral(self, f):
        """Calculate the integral over the element."""
        return (f * self.jacobian()).integrate((t[0], 0, 1))


class Triangle(Reference):
    """A triangle."""

    def __init__(self, vertices=((0, 0), (1, 0), (0, 1))):
        self.tdim = 2
        self.name = "triangle"
        self.origin = vertices[0]
        self.axes = (vsub(vertices[1], vertices[0]), vsub(vertices[2], vertices[0]))
        self.reference_vertices = ((0, 0), (1, 0), (0, 1))
        self.vertices = vertices
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
        self.vertices = vertices
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


class Quadrilateral(Reference):
    """A quadrilateral."""

    def __init__(self, vertices=((0, 0), (1, 0), (0, 1), (1, 1))):
        self.tdim = 2
        self.name = "quadrilateral"
        self.origin = vertices[0]
        self.axes = (vsub(vertices[1], vertices[0]), vsub(vertices[2], vertices[0]))
        self.reference_vertices = ((0, 0), (1, 0), (0, 1), (1, 1))
        self.vertices = vertices
        self.edges = ((0, 1), (0, 2), (1, 3), (2, 3))
        self.faces = ((0, 1, 2, 3),)
        self.volumes = tuple()
        self.sub_entity_types = ["point", "interval", "quadrilateral", None]
        super().__init__(tp=True)

    def integral(self, f):
        """Calculate the integral over the element."""
        return (f * self.jacobian()).integrate((t[1], 0, 1)).integrate((t[0], 0, 1))


class Hexahedron(Reference):
    """A hexahedron."""

    def __init__(
        self,
        vertices=(
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ),
    ):
        self.tdim = 3
        self.name = "hexahedron"
        self.origin = vertices[0]
        self.axes = (
            vsub(vertices[1], vertices[0]),
            vsub(vertices[2], vertices[0]),
            vsub(vertices[4], vertices[0]),
        )
        self.reference_vertices = (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        )
        self.vertices = vertices
        self.edges = (
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        )
        self.faces = (
            (0, 1, 2, 3),
            (0, 1, 4, 5),
            (0, 2, 4, 6),
            (1, 3, 5, 7),
            (2, 3, 6, 7),
            (4, 5, 6, 7),
        )
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
