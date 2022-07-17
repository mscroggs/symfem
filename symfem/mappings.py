"""Functions to map functions between cells."""

import sympy
from .symbols import x
from .functions import (ScalarFunction, MatrixFunction, VectorFunction, AnyFunction,
                        FunctionInput, parse_function_input)
from .vectors import vdot, vcross3d, vnorm

PointType = None


def _det(M: MatrixFunction) -> ScalarFunction:
    """Find the determinant."""
    if M.rows == M.cols:
        return M.det()
    if M.rows == 3 and M.cols == 2:
        crossed = vcross3d(M.col(0), M.col(1))
        assert isinstance(crossed, tuple)
        return vnorm(crossed)
    raise ValueError(f"Cannot find determinant of {M.rows}x{M.cols} matrix.")


def identity(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int
) -> AnyFunction:
    """Map functions."""
    f = parse_function_input(f_in)
    g = f.subs(x, inverse_map)
    return g


def covariant(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int
) -> VectorFunction:
    """Map H(curl) functions."""
    f = parse_function_input(f_in)
    assert f.is_vector
    g = f.subs(x, inverse_map)
    assert g.is_vector
    j_inv = sympy.Matrix([[i.diff(x[j]) for j in range(len(map))]
                          for i in inverse_map]).transpose()
    return VectorFunction([vdot(j_inv.row(i), g.as_sympy()) for i in range(j_inv.rows)])


def contravariant(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int
) -> VectorFunction:
    """Map H(div) functions."""
    f = parse_function_input(f_in)
    assert f.is_vector
    g = f.subs(x, inverse_map)
    assert g.is_vector
    jacobian = sympy.Matrix([[i.diff(x[j]) for j in range(tdim)] for i in map])
    jacobian /= _det(jacobian)
    return VectorFunction([vdot(jacobian.row(i), g) for i in range(jacobian.rows)])


def double_covariant(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int
) -> MatrixFunction:
    """Map matrix functions."""
    f = parse_function_input(f_in)
    assert f.is_matrix
    g = f.subs(x, inverse_map).as_sympy()
    j_inv = sympy.Matrix([[i.diff(x[j]) for j in range(len(map))]
                          for i in inverse_map]).transpose()
    out = j_inv * g * j_inv.transpose()
    return MatrixFunction(out)


def double_contravariant(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int
) -> MatrixFunction:
    """Map matrix functions."""
    f = parse_function_input(f_in)
    assert f.is_matrix
    g = f.subs(x, inverse_map).as_sympy()
    jacobian = sympy.Matrix([[i.diff(x[j]) for j in range(tdim)] for i in map])
    jacobian /= _det(jacobian)

    out = jacobian * g * jacobian.transpose()
    return MatrixFunction(out)
