"""Functions to map functions between cells."""

import sympy
from .symbolic import subs, x
from .vectors import vdot, vcross, vnorm
from .calculus import diff


def _det(M):
    """Find the determinant."""
    if M.rows == M.cols:
        return M.det()
    if M.rows == 3 and M.cols == 2:
        return vnorm(vcross(M.col(0), M.col(1)))
    raise ValueError(f"Cannot find determinant of {M.rows}x{M.cols} matrix.")


def identity(f, map, inverse_map, tdim):
    """Map functions."""
    g = subs(f, x, inverse_map)
    return g


def covariant(f, map, inverse_map, tdim):
    """Map H(curl) functions."""
    g = subs(f, x, inverse_map)
    j_inv = sympy.Matrix([[diff(i, x[j]) for j in range(len(map))]
                          for i in inverse_map]).transpose()
    return tuple(vdot(j_inv.row(i), g) for i in range(j_inv.rows))


def contravariant(f, map, inverse_map, tdim):
    """Map H(div) functions."""
    g = subs(f, x, inverse_map)
    jacobian = sympy.Matrix([[diff(i, x[j]) for j in range(tdim)] for i in map])
    jacobian /= _det(jacobian)
    return tuple(vdot(jacobian.row(i), g) for i in range(jacobian.rows))


def double_covariant(f, map, inverse_map, tdim):
    """Map matrix functions."""
    g = subs(f, x, inverse_map)
    j_inv = sympy.Matrix([[diff(i, x[j]) for j in range(len(map))]
                          for i in inverse_map]).transpose()

    g_mat = sympy.Matrix([g[i * tdim: (i + 1) * tdim] for i in range(tdim)])
    out = j_inv * g_mat * j_inv.transpose()
    return tuple(out[i] for i in range(out.rows * out.cols))


def double_contravariant(f, map, inverse_map, tdim):
    """Map matrix functions."""
    g = subs(f, x, inverse_map)
    jacobian = sympy.Matrix([[diff(i, x[j]) for j in range(tdim)] for i in map])
    jacobian /= _det(jacobian)

    g_mat = sympy.Matrix([g[i * tdim: (i + 1) * tdim] for i in range(tdim)])
    out = jacobian * g_mat * jacobian.transpose()
    return tuple(out[i] for i in range(out.rows * out.cols))
