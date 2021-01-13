"""Functions to map functions between cells."""

import sympy
from .symbolic import subs, x
from .vectors import vdot


def default(f, map, inverse_map, tdim):
    """Map functions."""
    g = subs(f, x, inverse_map)
    return g


def covariant(f, map, inverse_map, tdim):
    """Map H(curl) functions."""
    g = subs(f, x, inverse_map)
    J = sympy.Matrix([[map[i].diff(x[j]) for j in range(tdim)] for i in range(tdim)])
    Jinv = J.inv().transpose()
    return tuple(vdot(Jinv.row(i), g) for i in range(Jinv.rows))


def contravariant(f, map, inverse_map, tdim):
    """Map H(div) functions."""
    g = subs(f, x, inverse_map)
    J = sympy.Matrix([[map[i].diff(x[j]) for j in range(tdim)] for i in range(tdim)])
    J /= J.det()
    return tuple(vdot(J.row(i), g) for i in range(J.rows))


def double_covariant(f, map, inverse_map, tdim):
    """Map H(curl) functions."""
    raise NotImplementedError()
