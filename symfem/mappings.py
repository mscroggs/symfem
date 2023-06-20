"""Functions to map functions between cells."""

import sympy

from .functions import AnyFunction, FunctionInput, MatrixFunction, parse_function_input
from .geometry import PointType
from .symbols import x


class MappingNotImplemented(NotImplementedError):
    """Exception thrown when a mapping is not implemented for an element."""


def identity(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int,
    substitute: bool = True,
) -> AnyFunction:
    """Map functions.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        tdim: The topological dimension of the cell

    Returns:
        The mapped function
    """
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)
    return f


def l2(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int,
    substitute: bool = True,
) -> AnyFunction:
    """Map functions, scaling by the determinant of the jacobian.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        tdim: The topological dimension of the cell

    Returns:
        The mapped function
    """
    jdet = MatrixFunction([[i.diff(x[j]) for j in range(tdim)] for i in map]).det().as_sympy()
    assert isinstance(jdet, sympy.core.expr.Expr)
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)
    return f / abs(jdet)


def covariant(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int,
    substitute: bool = True,
) -> AnyFunction:
    """Map H(curl) functions.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        tdim: The topological dimension of the cell

    Returns:
        The mapped function
    """
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)

    assert f.is_vector

    j_inv = MatrixFunction([[i.diff(x[j]) for i in inverse_map] for j in range(tdim)])
    return j_inv @ f


def contravariant(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int,
    substitute: bool = True
) -> AnyFunction:
    """Map H(div) functions.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        tdim: The topological dimension of the cell

    Returns:
        The mapped function
    """
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)

    assert f.is_vector

    jacobian = MatrixFunction([[i.diff(x[j]) for j in range(tdim)] for i in map])
    jacobian /= jacobian.det()
    return jacobian @ f


def double_covariant(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int,
    substitute: bool = True,
) -> MatrixFunction:
    """Map matrix functions.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        tdim: The topological dimension of the cell

    Returns:
        The mapped function
    """
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)
    assert f.is_matrix

    j_inv = MatrixFunction([[i.diff(x[j]) for i in inverse_map] for j in range(tdim)])
    return j_inv @ f @ j_inv.transpose()


def double_contravariant(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int,
    substitute: bool = True,
) -> MatrixFunction:
    """Map matrix functions.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        tdim: The topological dimension of the cell

    Returns:
        The mapped function
    """
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)
    assert f.is_matrix

    jacobian = MatrixFunction([[i.diff(x[j]) for j in range(tdim)] for i in map])
    jacobian /= jacobian.det()
    return jacobian @ f @ jacobian.transpose()


def identity_inverse_transpose(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int,
    substitute: bool = True,
) -> AnyFunction:
    """Inverse transpose of identity().

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        tdim: The topological dimension of the cell

    Returns:
        The mapped function
    """
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)
    return f


def l2_inverse_transpose(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int,
    substitute: bool = True,
) -> AnyFunction:
    """Inverse transpose of l2().

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        tdim: The topological dimension of the cell

    Returns:
        The mapped function
    """
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)
    jdet = MatrixFunction([[i.diff(x[j]) for j in range(tdim)] for i in map]).det().as_sympy()
    assert isinstance(jdet, sympy.core.expr.Expr)
    return f * abs(jdet)


def covariant_inverse_transpose(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int,
    substitute: bool = True,
) -> AnyFunction:
    """Inverse transpose of covariant().

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        tdim: The topological dimension of the cell

    Returns:
        The mapped function
    """
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)

    assert f.is_vector

    jacobian = MatrixFunction([[i.diff(x[j]) for j in range(tdim)] for i in map])
    return jacobian @ f


def contravariant_inverse_transpose(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int,
    substitute: bool = True,
) -> AnyFunction:
    """Inverse transpose of contravariant().

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        tdim: The topological dimension of the cell

    Returns:
        The mapped function
    """
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)

    assert f.is_vector

    j_inv = MatrixFunction([[i.diff(x[j]) for i in inverse_map] for j in range(tdim)])
    j_inv /= j_inv.det()
    return j_inv @ f
