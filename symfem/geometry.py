"""Geometry."""

import typing
import sympy

PointType = typing.Tuple[sympy.core.expr.Expr, ...]
SetOfPoints = typing.Tuple[PointType, ...]
PointTypeInput = typing.Union[
    typing.Tuple[typing.Union[sympy.core.expr.Expr, int], ...],
    typing.List[typing.Union[sympy.core.expr.Expr, int]],
    sympy.matrices.dense.MutableDenseMatrix]
SetOfPointsInput = typing.Union[
    typing.Tuple[PointTypeInput, ...],
    typing.List[PointTypeInput]]


def parse_set_of_points_input(points: SetOfPointsInput) -> SetOfPoints:
    """Convert an input set of points to the correct format."""
    return tuple(parse_point_input(p) for p in points)


def parse_point_input(point: PointTypeInput) -> PointType:
    """Convert an input point to the correct format."""
    if isinstance(point, sympy.Matrix):
        assert point.rows == 1 or point.cols == 1
        if point.rows == 1:
            return tuple(point[0, i] for i in range(point.cols))
        else:
            return tuple(point[i, 0] for i in range(point.rows))
    return tuple(point)


def _vsub(v: PointType, w: PointType) -> PointType:
    """Subtract."""
    return tuple(i - j for i, j in zip(v, w))


def _vdot(v: PointType, w: PointType) -> sympy.core.expr.Expr:
    """Compute dot product."""
    out = sympy.Integer(0)
    for i, j in zip(v, w):
        out += i * j
    return out


def point_in_triangle(point: PointType, triangle: SetOfPoints) -> bool:
    """Check if a point is inside a triangle."""
    v0 = _vsub(triangle[2], triangle[0])
    v1 = _vsub(triangle[1], triangle[0])
    v2 = _vsub(point, triangle[0])

    dot00 = _vdot(v0, v0)
    dot01 = _vdot(v0, v1)
    dot02 = _vdot(v0, v2)
    dot11 = _vdot(v1, v1)
    dot12 = _vdot(v1, v2)

    det = (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) / det
    v = (dot00 * dot12 - dot01 * dot02) / det

    return u >= 0 and v >= 0 and u + v <= 1


def point_in_quadrilateral(point: PointType, quad: SetOfPoints) -> bool:
    """Check if a point is inside a quadrilateral."""
    e0 = _vsub(quad[1], quad[0])
    e1 = _vsub(quad[0], quad[2])
    e2 = _vsub(quad[3], quad[1])
    e3 = _vsub(quad[2], quad[3])

    n0 = (-e0[1], e0[0])
    n1 = (-e1[1], e1[0])
    n2 = (-e2[1], e2[0])
    n3 = (-e3[1], e3[0])

    d0 = _vdot(n0, _vsub(point, quad[0]))
    d1 = _vdot(n1, _vsub(point, quad[2]))
    d2 = _vdot(n2, _vsub(point, quad[1]))
    d3 = _vdot(n3, _vsub(point, quad[3]))

    return d0 >= 0 and d1 >= 0 and d2 >= 0 and d3 >= 0


def point_in_tetrahedron(point: PointType, tetrahedron: SetOfPoints) -> bool:
    """Check if a point is inside a tetrahedron."""
    v0 = _vsub(tetrahedron[3], tetrahedron[0])
    v1 = _vsub(tetrahedron[2], tetrahedron[0])
    v2 = _vsub(tetrahedron[1], tetrahedron[0])
    v3 = _vsub(point, tetrahedron[0])

    dot00 = _vdot(v0, v0)
    dot01 = _vdot(v0, v1)
    dot02 = _vdot(v0, v2)
    dot03 = _vdot(v0, v3)
    dot11 = _vdot(v1, v1)
    dot12 = _vdot(v1, v2)
    dot13 = _vdot(v1, v3)
    dot22 = _vdot(v2, v2)
    dot23 = _vdot(v2, v3)

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

    return u >= 0 and v >= 0 and w >= 0 and u + v + w <= 1
