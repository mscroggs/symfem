"""Functions to handle vectors."""

import sympy
import numpy
from .symbolic import PointType, ScalarValue, SetOfPoints, PointTypeInput, parse_point_input


def vsub(v: PointTypeInput, w: PointTypeInput) -> PointType:
    """Subtract a vector from another."""
    return tuple(i - j for i, j in zip(parse_point_input(v), parse_point_input(w)))


def vadd(v: PointTypeInput, w: PointTypeInput) -> PointType:
    """Add two vectors."""
    return tuple(i + j for i, j in zip(parse_point_input(v), parse_point_input(w)))


def vdiv(v: PointTypeInput, a: ScalarValue) -> PointType:
    """Divide a vector by a scalar."""
    return tuple(i / a for i in parse_point_input(v))


def vnorm(v: PointTypeInput) -> ScalarValue:
    """Find the norm of a vector."""
    return sympy.sqrt(sum(a ** 2 for a in parse_point_input(v)))


def vdot(v: PointTypeInput, w: PointTypeInput) -> ScalarValue:
    """Find the dot product of two vectors."""
    return sum(a * b for a, b in zip(parse_point_input(v), parse_point_input(w)))


def vcross(v: PointTypeInput, w: PointTypeInput) -> PointType:
    """Find the cross product of two vectors."""
    v2 = parse_point_input(v)
    w2 = parse_point_input(w)
    if len(v2) == 2:
        return _vcross2d(v2, w2)
    else:
        assert len(v2) == 3
        return _vcross3d(v2, w2)


def _vcross2d(v: PointType, w: PointType) -> ScalarValue:
    """Find the cross product of two 2D vectors."""
    return v[0] * w[1] - v[1] * w[0]


def _vcross3d(v: PointType, w: PointType) -> PointType:
    """Find the cross product of two 3D vectors."""
    return (
        v[1] * w[2] - v[2] * w[1],
        v[2] * w[0] - v[0] * w[2],
        v[0] * w[1] - v[1] * w[0],
    )


def vnormalise(v: PointTypeInput) -> PointType:
    """Normalise a vector."""
    v2 = parse_point_input(v)
    return vdiv(v2, vnorm(v2))


def point_in_triangle(point: PointTypeInput, triangle: SetOfPoints) -> bool:
    """Check if a point is inside a triangle."""
    v0 = vsub(triangle[2], triangle[0])
    v1 = vsub(triangle[1], triangle[0])
    v2 = vsub(parse_point_input(point), triangle[0])

    dot00 = vdot(v0, v0)
    dot01 = vdot(v0, v1)
    dot02 = vdot(v0, v2)
    dot11 = vdot(v1, v1)
    dot12 = vdot(v1, v2)

    det = (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) / det
    v = (dot00 * dot12 - dot01 * dot02) / det

    if numpy.isclose(float(u), 0):
        u = 0
    if numpy.isclose(float(v), 0):
        v = 0
    if u >= 0 and v >= 0 and numpy.isclose(float(u + v), 1):
        return True

    return u >= 0 and v >= 0 and u + v <= 1


def point_in_quadrilateral(point: PointTypeInput, quad: SetOfPoints) -> bool:
    """Check if a point is inside a quadrilateral."""
    point2 = parse_point_input(point)

    e0 = vsub(quad[1], quad[0])
    e1 = vsub(quad[0], quad[2])
    e2 = vsub(quad[3], quad[1])
    e3 = vsub(quad[2], quad[3])

    n0 = (-e0[1], e0[0])
    n1 = (-e1[1], e1[0])
    n2 = (-e2[1], e2[0])
    n3 = (-e3[1], e3[0])

    d0 = vdot(n0, vsub(point2, quad[0]))
    d1 = vdot(n1, vsub(point2, quad[2]))
    d2 = vdot(n2, vsub(point2, quad[1]))
    d3 = vdot(n3, vsub(point2, quad[3]))

    if numpy.isclose(float(d0), 0):
        d0 = 0
    if numpy.isclose(float(d1), 0):
        d1 = 0
    if numpy.isclose(float(d2), 0):
        d2 = 0
    if numpy.isclose(float(d3), 0):
        d3 = 0

    return d0 >= 0 and d1 >= 0 and d2 >= 0 and d3 >= 0


def point_in_tetrahedron(point: PointTypeInput, tetrahedron: SetOfPoints) -> bool:
    """Check if a point is inside a tetrahedron."""
    v0 = vsub(tetrahedron[3], tetrahedron[0])
    v1 = vsub(tetrahedron[2], tetrahedron[0])
    v2 = vsub(tetrahedron[1], tetrahedron[0])
    v3 = vsub(parse_point_input(point), tetrahedron[0])

    dot00 = vdot(v0, v0)
    dot01 = vdot(v0, v1)
    dot02 = vdot(v0, v2)
    dot03 = vdot(v0, v3)
    dot11 = vdot(v1, v1)
    dot12 = vdot(v1, v2)
    dot13 = vdot(v1, v3)
    dot22 = vdot(v2, v2)
    dot23 = vdot(v2, v3)

    det = dot00 * (dot11 * dot22 - dot12 * dot12)
    det += dot01 * (dot02 * dot12 - dot01 * dot22)
    det += dot02 * (dot01 * dot12 - dot11 * dot02)

    u = (dot11 * dot22 - dot12 * dot12) * dot03
    u += (dot02 * dot12 - dot01 * dot22) * dot13
    u += (dot01 * dot12 - dot02 * dot11) * dot23
    u /= det
    v = (dot12 * dot02 - dot01 * dot22) * dot03
    v += (dot00 * dot22 - dot02 * dot02) * dot13
    v += (dot02 * dot01 - dot00 * dot12) * dot23
    v /= det
    w = (dot01 * dot12 - dot11 * dot02) * dot03
    w += (dot01 * dot02 - dot00 * dot12) * dot13
    w += (dot00 * dot11 - dot01 * dot01) * dot23
    w /= det

    if numpy.isclose(float(u), 0):
        u = 0
    if numpy.isclose(float(v), 0):
        v = 0
    if numpy.isclose(float(w), 0):
        w = 0
    if u >= 0 and v >= 0 and w >= 0 and numpy.isclose(float(u + v + w), 1):
        return True

    return u >= 0 and v >= 0 and w >= 0 and u + v + w <= 1
