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
