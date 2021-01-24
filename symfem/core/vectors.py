"""Functions to handle vectors."""

import sympy


def vsub(v, w):
    """Subtract a vector from another."""
    try:
        return tuple(i - j for i, j in zip(v, w))
    except TypeError:
        return v - w


def vadd(v, w):
    """Add two vectors."""
    try:
        return tuple(i + j for i, j in zip(v, w))
    except TypeError:
        return v + w


def vdiv(v, a):
    """Divide a vector by a scalar."""
    try:
        return tuple(i / a for i in v)
    except TypeError:
        return v / a


def vnorm(v):
    """Find the norm of a vector."""
    try:
        return sympy.sqrt(sum(a ** 2 for a in v))
    except TypeError:
        return abs(v)


def vdot(v, w):
    """Find the dot product of two vectors."""
    try:
        return sum(a * b for a, b in zip(v, w))
    except TypeError:
        return v * w


def vcross(v, w):
    """Find the cross product of two vectors."""
    if len(v) == 2:
        return _vcross2d(v, w)
    else:
        assert len(v) == 3
        return _vcross3d(v, w)


def _vcross2d(v, w):
    """Find the cross product of two 2D vectors."""
    return v[0] * w[1] - v[1] * w[0]


def _vcross3d(v, w):
    """Find the cross product of two 3D vectors."""
    return (
        v[1] * w[2] - v[2] * w[1],
        v[2] * w[0] - v[0] * w[2],
        v[0] * w[1] - v[1] * w[0],
    )


def vnormalise(v):
    """Normalise a vector."""
    return vdiv(v, vnorm(v))


def point_in_triangle(point, triangle):
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

    return u >= 0 and v >= 0 and u + v < 1
