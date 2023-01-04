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


def _is_close(a: sympy.core.expr.Expr, b: int) -> bool:
    """Check if a Sympy expression is close to an int.

    Args:
        a: A Sympy expression
        b: An integer

    Returns:
        Is the sympy expression close to the integer?
    """
    return abs(a - b) < 1e-8


def parse_set_of_points_input(points: SetOfPointsInput) -> SetOfPoints:
    """Convert an input set of points to the correct format.

    Args:
        points: A set of points in some input format

    Returns:
        A set of points
    """
    return tuple(parse_point_input(p) for p in points)


def parse_point_input(point: PointTypeInput) -> PointType:
    """Convert an input point to the correct format.

    Args:
        point: A point in some input fotmat

    Returns:
        A point
    """
    if isinstance(point, sympy.Matrix):
        assert point.rows == 1 or point.cols == 1
        if point.rows == 1:
            return tuple(point[0, i] for i in range(point.cols))
        else:
            return tuple(point[i, 0] for i in range(point.rows))
    return tuple(sympy.S(i) for i in point)


def _vsub(v: PointType, w: PointType) -> PointType:
    """Subtract.

    Args:
        v: A vector
        w: A vector

    Returns:
        The vector v - w
    """
    return tuple(i - j for i, j in zip(v, w))


def _vdot(v: PointType, w: PointType) -> sympy.core.expr.Expr:
    """Compute dot product.

    Args:
        v: A vector
        w: A vector

    Returns:
        The dot product of v and w
    """
    out = sympy.Integer(0)
    for i, j in zip(v, w):
        out += i * j
    return out


def point_in_triangle(point: PointType, triangle: SetOfPoints) -> bool:
    """Check if a point is inside a triangle.

    Args:
        point: The point
        traingle: The vertices of the triangle

    Returns:
        Is the point inside the triangle?
    """
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

    uv = u + v

    if isinstance(u, sympy.Float) and _is_close(u, 0):
        u = sympy.Integer(0)
    if isinstance(v, sympy.Float) and _is_close(v, 0):
        v = sympy.Integer(0)
    if isinstance(uv, sympy.Float) and _is_close(uv, 1):
        uv = sympy.Integer(1)

    return u >= 0 and v >= 0 and uv <= 1


def point_in_quadrilateral(point: PointType, quad: SetOfPoints) -> bool:
    """Check if a point is inside a quadrilateral.

    Args:
        point: The point
        traingle: The vertices of the quadrilateral

    Returns:
        Is the point inside the quadrilateral?
    """
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

    if isinstance(d0, sympy.Float) and _is_close(d0, 0):
        d0 = sympy.Integer(0)
    if isinstance(d1, sympy.Float) and _is_close(d1, 0):
        d1 = sympy.Integer(0)
    if isinstance(d2, sympy.Float) and _is_close(d2, 0):
        d2 = sympy.Integer(0)
    if isinstance(d3, sympy.Float) and _is_close(d3, 0):
        d3 = sympy.Integer(0)

    return d0 >= 0 and d1 >= 0 and d2 >= 0 and d3 >= 0


def point_in_tetrahedron(point: PointType, tetrahedron: SetOfPoints) -> bool:
    """Check if a point is inside a tetrahedron.

    Args:
        point: The point
        traingle: The vertices of the tetrahedron

    Returns:
        Is the point inside the tetrahedron?
    """
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

    uvw = u + v + w

    if isinstance(u, sympy.Float) and _is_close(u, 0):
        u = sympy.Integer(0)
    if isinstance(v, sympy.Float) and _is_close(v, 0):
        v = sympy.Integer(0)
    if isinstance(w, sympy.Float) and _is_close(w, 0):
        w = sympy.Integer(0)
    if isinstance(uvw, sympy.Float) and _is_close(uvw, 1):
        uvw = sympy.Integer(1)

    return u >= 0 and v >= 0 and w >= 0 and uvw <= 1
