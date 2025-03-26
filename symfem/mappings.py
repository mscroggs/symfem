"""Functions to map functions between cells."""

from __future__ import annotations

from typing import Callable

import sympy

from symfem.functions import Function, FunctionInput, MatrixFunction, parse_function_input
from symfem.geometry import PointType
from symfem.symbols import x

__all__ = [
    "MappingNotImplemented",
    "identity",
    "identity_inverse",
    "identity_inverse_transpose",
    "l2",
    "l2_inverse_transpose",
    "l2_inverse",
    "covariant",
    "covariant_inverse",
    "covariant_inverse_transpose",
    "contravariant",
    "contravariant_inverse",
    "contravariant_inverse_transpose",
    "double_covariant",
    "double_covariant_inverse",
    "double_contravariant",
    "double_contravariant_inverse",
    "co_contravariant",
    "co_contravariant_inverse",
    "get_mapping",
]


class MappingNotImplemented(NotImplementedError):
    """Exception thrown when a mapping is not implemented for an element."""


def identity(
    f_in: FunctionInput,
    map: PointType,
    inverse_map: PointType,
    substitute: bool = True,
) -> Function:
    """Map functions.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)
    return f


def l2(
    f_in: FunctionInput,
    map: PointType,
    inverse_map: PointType,
    substitute: bool = True,
) -> Function:
    """Map functions, scaling by the determinant of the jacobian.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    tdim = len(inverse_map)
    jdet = MatrixFunction([[i.diff(x[j]) for j in range(tdim)] for i in map]).det().as_sympy()
    assert isinstance(jdet, sympy.core.expr.Expr)
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)
    return f / abs(jdet)


def covariant(
    f_in: FunctionInput,
    map: PointType,
    inverse_map: PointType,
    substitute: bool = True,
) -> Function:
    """Map H(curl) functions.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    tdim = len(inverse_map)
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)

    assert f.is_vector

    j_inv_t = MatrixFunction([[i.diff(x[j]) for i in inverse_map] for j in range(tdim)])
    return j_inv_t @ f


def contravariant(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, substitute: bool = True
) -> Function:
    """Map H(div) functions.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    tdim = len(inverse_map)
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)

    assert f.is_vector

    jacobian = MatrixFunction([[i.diff(x[j]) for j in range(tdim)] for i in map])
    jacobian /= jacobian.det()
    return jacobian @ f


def double_covariant(
    f_in: FunctionInput,
    map: PointType,
    inverse_map: PointType,
    substitute: bool = True,
) -> MatrixFunction:
    """Map matrix functions.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    tdim = len(inverse_map)
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)
    assert f.is_matrix

    j_inv_t = MatrixFunction([[i.diff(x[j]) for i in inverse_map] for j in range(tdim)])
    return j_inv_t @ f @ j_inv_t.transpose()


def double_contravariant(
    f_in: FunctionInput,
    map: PointType,
    inverse_map: PointType,
    substitute: bool = True,
) -> MatrixFunction:
    """Map matrix functions.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    tdim = len(inverse_map)
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)
    assert f.is_matrix

    jacobian = MatrixFunction([[i.diff(x[j]) for j in range(tdim)] for i in map])
    jacobian /= jacobian.det()
    return jacobian @ f @ jacobian.transpose()


def co_contravariant(
    f_in: FunctionInput,
    map: PointType,
    inverse_map: PointType,
    substitute: bool = True,
) -> MatrixFunction:
    """Map matrix functions.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """

    tdim = len(inverse_map)
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)
    assert f.is_matrix

    j_inv_t = MatrixFunction([[i.diff(x[j]) for i in inverse_map] for j in range(tdim)])
    jacobian = MatrixFunction([[i.diff(x[j]) for j in range(tdim)] for i in map])
    jacobian /= jacobian.det()
    return j_inv_t @ f @ jacobian.transpose()


def identity_inverse_transpose(
    f_in: FunctionInput,
    map: PointType,
    inverse_map: PointType,
    substitute: bool = True,
) -> Function:
    """Inverse transpose of identity().

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)
    return f


def l2_inverse_transpose(
    f_in: FunctionInput,
    map: PointType,
    inverse_map: PointType,
    substitute: bool = True,
) -> Function:
    """Inverse transpose of l2().

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    tdim = len(inverse_map)
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)
    jdet = MatrixFunction([[i.diff(x[j]) for j in range(tdim)] for i in map]).det().as_sympy()
    assert isinstance(jdet, sympy.core.expr.Expr)
    return f * abs(jdet)


def covariant_inverse_transpose(
    f_in: FunctionInput,
    map: PointType,
    inverse_map: PointType,
    substitute: bool = True,
) -> Function:
    """Inverse transpose of covariant().

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    tdim = len(inverse_map)
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)

    assert f.is_vector

    jacobian = MatrixFunction([[i.diff(x[j]) for j in range(tdim)] for i in map])
    return jacobian @ f


def contravariant_inverse_transpose(
    f_in: FunctionInput,
    map: PointType,
    inverse_map: PointType,
    substitute: bool = True,
) -> Function:
    """Inverse transpose of contravariant().

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    tdim = len(inverse_map)
    f = parse_function_input(f_in)
    if substitute:
        f = f.subs(x, inverse_map)

    assert f.is_vector

    j_inv_t = MatrixFunction([[i.diff(x[j]) for i in inverse_map] for j in range(tdim)])
    j_inv_t /= j_inv_t.det()
    return j_inv_t @ f


def identity_inverse(
    f_in: FunctionInput,
    map: PointType,
    inverse_map: PointType,
    substitute: bool = True,
) -> Function:
    """Inverse of identity().

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    return identity(f_in, inverse_map, map, substitute)


def l2_inverse(
    f_in: FunctionInput,
    map: PointType,
    inverse_map: PointType,
    substitute: bool = True,
) -> Function:
    """Inverse of l2().

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    return l2(f_in, inverse_map, map, substitute)


def covariant_inverse(
    f_in: FunctionInput,
    map: PointType,
    inverse_map: PointType,
    substitute: bool = True,
) -> Function:
    """Inverse of covariant().

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    return covariant(f_in, inverse_map, map, substitute)


def contravariant_inverse(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, substitute: bool = True
) -> Function:
    """Inverse of contravariant().

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    return contravariant(f_in, inverse_map, map, substitute)


def double_covariant_inverse(
    f_in: FunctionInput,
    map: PointType,
    inverse_map: PointType,
    substitute: bool = True,
) -> Function:
    """Inverse of double_covariant().

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    return double_covariant(f_in, inverse_map, map, substitute)


def double_contravariant_inverse(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, substitute: bool = True
) -> Function:
    """Inverse of double_contravariant().

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    return double_contravariant(f_in, inverse_map, map, substitute)


def co_contravariant_inverse(
    f_in: FunctionInput,
    map: PointType,
    inverse_map: PointType,
    substitute: bool = True,
) -> MatrixFunction:
    """Map matrix functions.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        substitute: Should the inverse map be substituted in?

    Returns:
        The mapped function
    """
    return co_contravariant(f_in, inverse_map, map, substitute)


def get_mapping(
    mapname: str, inverse: bool = False, transpose: bool = False
) -> Callable[[FunctionInput, PointType, PointType, bool], Function]:
    """Get a mapping.

    Args:
        mapname: The name of the mapping
        inverse: Should the map be inverted
        transpose: Should the map be transposed

    Returns:
        A function that performs the mapping
    """
    function_name = mapname
    if inverse:
        function_name += "_inverse"
    if transpose:
        function_name += "_transpose"

    if function_name not in globals():
        raise ValueError(f"Invalid map name: {function_name}")

    return globals()[function_name]
