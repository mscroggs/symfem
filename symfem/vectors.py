"""Functions to handle vectors."""

import sympy
import numpy
from .symbolic import PointType, ScalarValue, SetOfPoints, PointTypeInput


def vsub(v: PointTypeInput, w: PointTypeInput) -> PointType:
    """Subtract a vector from another."""
    if isinstance(v, (tuple, list)):
        return tuple(i - j for i, j in zip(v, w))
    return v - w


def vadd(v: PointTypeInput, w: PointTypeInput) -> PointType:
    """Add two vectors."""
    try:
        return tuple(i + j for i, j in zip(v, w))
    except TypeError:
        return v + w


def vdiv(v: PointTypeInput, a: ScalarValue) -> PointType:
    """Divide a vector by a scalar."""
    try:
        return tuple(i / a for i in v)
    except TypeError:
        return v / a


def vnorm(v: PointTypeInput) -> ScalarValue:
    """Find the norm of a vector."""
    try:
        return sympy.sqrt(sum(a ** 2 for a in v))
    except TypeError:
        return abs(v)


def vdot(v: PointTypeInput, w: PointTypeInput) -> ScalarValue:
    """Find the dot product of two vectors."""
    try:
        return sum(a * b for a, b in zip(v, w))
    except TypeError:
        return v * w


def vcross(v: PointTypeInput, w: PointTypeInput) -> PointType:
    """Find the cross product of two vectors."""
    if len(v) == 2:
        return _vcross2d(v, w)
    else:
        assert len(v) == 3
        return _vcross3d(v, w)


def _vcross2d(v: PointTypeInput, w: PointTypeInput) -> ScalarValue:
    """Find the cross product of two 2D vectors."""
    return v[0] * w[1] - v[1] * w[0]


def _vcross3d(v: PointTypeInput, w: PointTypeInput) -> PointType:
    """Find the cross product of two 3D vectors."""
    return (
        v[1] * w[2] - v[2] * w[1],
        v[2] * w[0] - v[0] * w[2],
        v[0] * w[1] - v[1] * w[0],
    )


def vnormalise(v: PointTypeInput) -> PointType:
    """Normalise a vector."""
    return vdiv(v, vnorm(v))


def point_in_triangle(point: PointTypeInput, triangle: SetOfPoints) -> bool:
    """Check if a point is inside a triangle."""
    v0 = vsub(triangle[2], triangle[0])
    v1 = vsub(triangle[1], triangle[0])
    v2 = vsub(point, triangle[0])

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


def point_in_tetrahedron(point: PointTypeInput, tetrahedron: SetOfPoints) -> bool:
    """Check if a point is inside a tetrahedron."""
    v0 = vsub(tetrahedron[3], tetrahedron[0])
    v1 = vsub(tetrahedron[2], tetrahedron[0])
    v2 = vsub(tetrahedron[1], tetrahedron[0])
    v3 = vsub(point, tetrahedron[0])

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
