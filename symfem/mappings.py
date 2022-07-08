"""Functions to map functions between cells."""

import sympy
from .symbolic import subs, x, MatrixFunction, ScalarFunction, VectorFunction, PointType
from .vectors import vdot, vcross, vnorm
from .calculus import diff


def _det(M: MatrixFunction) -> ScalarFunction:
    """Find the determinant."""
    if M.rows == M.cols:
        return M.det()
    if M.rows == 3 and M.cols == 2:
        crossed = vcross(M.col(0), M.col(1))
        assert isinstance(crossed, tuple)
        return vnorm(crossed)
    raise ValueError(f"Cannot find determinant of {M.rows}x{M.cols} matrix.")


def identity(
    f: ScalarFunction, map: PointType, inverse_map: PointType, tdim: int
) -> ScalarFunction:
    """Map functions."""
    g = subs(f, x, inverse_map)
    assert isinstance(g, (int, sympy.core.expr.Expr))
    return g


def covariant(
    f: VectorFunction, map: PointType, inverse_map: PointType, tdim: int
) -> VectorFunction:
    """Map H(curl) functions."""
    g = subs(f, x, inverse_map)
    assert isinstance(g, tuple)
    j_inv = sympy.Matrix([[diff(i, x[j]) for j in range(len(map))]
                          for i in inverse_map]).transpose()
    return tuple(vdot(j_inv.row(i), g) for i in range(j_inv.rows))


def contravariant(
    f: VectorFunction, map: PointType, inverse_map: PointType, tdim: int
) -> VectorFunction:
    """Map H(div) functions."""
    g = subs(f, x, inverse_map)
    assert isinstance(g, tuple)
    jacobian = sympy.Matrix([[diff(i, x[j]) for j in range(tdim)] for i in map])
    jacobian /= _det(jacobian)
    return tuple(vdot(jacobian.row(i), g) for i in range(jacobian.rows))


def double_covariant(
    f: VectorFunction, map: PointType, inverse_map: PointType, tdim: int
) -> VectorFunction:
    """Map matrix functions."""
    g = subs(f, x, inverse_map)
    assert isinstance(g, tuple)
    j_inv = sympy.Matrix([[diff(i, x[j]) for j in range(len(map))]
                          for i in inverse_map]).transpose()
    g_mat = sympy.Matrix([g[i * tdim: (i + 1) * tdim] for i in range(tdim)])
    out = j_inv * g_mat * j_inv.transpose()
    return tuple(out[i] for i in range(out.rows * out.cols))


def double_contravariant(
    f: VectorFunction, map: PointType, inverse_map: PointType, tdim: int
) -> VectorFunction:
    """Map matrix functions."""
    g = subs(f, x, inverse_map)
    assert isinstance(g, tuple)
    jacobian = sympy.Matrix([[diff(i, x[j]) for j in range(tdim)] for i in map])
    jacobian /= _det(jacobian)

    g_mat = sympy.Matrix([g[i * tdim: (i + 1) * tdim] for i in range(tdim)])
    out = jacobian * g_mat * jacobian.transpose()
    return tuple(out[i] for i in range(out.rows * out.cols))
