"""Functions to map functions between cells."""

from .functions import (AnyFunction, FunctionInput, MatrixFunction, VectorFunction,
                        parse_function_input)
from .geometry import PointType
from .symbols import x


def identity(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int
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
    return parse_function_input(f_in).subs(x, inverse_map)


def covariant(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int
) -> VectorFunction:
    """Map H(curl) functions.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        tdim: The topological dimension of the cell

    Returns:
        The mapped function
    """
    f = parse_function_input(f_in).subs(x, inverse_map)
    assert f.is_vector

    j_inv = MatrixFunction([[i.diff(x[j]) for i in inverse_map] for j in range(tdim)])
    return j_inv @ f


def contravariant(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int
) -> VectorFunction:
    """Map H(div) functions.

    Args:
        f_in: The function
        map: The map from the reference cell to the physical cell
        inverse_map: The map to the reference cell from the physical cell
        tdim: The topological dimension of the cell

    Returns:
        The mapped function
    """
    f = parse_function_input(f_in).subs(x, inverse_map)
    assert f.is_vector

    jacobian = MatrixFunction([[i.diff(x[j]) for j in range(tdim)] for i in map])
    jacobian /= jacobian.det()
    return jacobian @ f


def double_covariant(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int
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
    f = parse_function_input(f_in).subs(x, inverse_map)
    assert f.is_matrix

    j_inv = MatrixFunction([[i.diff(x[j]) for i in inverse_map] for j in range(tdim)])
    return j_inv @ f @ j_inv.transpose()


def double_contravariant(
    f_in: FunctionInput, map: PointType, inverse_map: PointType, tdim: int
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
    f = parse_function_input(f_in).subs(x, inverse_map)
    assert f.is_matrix

    jacobian = MatrixFunction([[i.diff(x[j]) for j in range(tdim)] for i in map])
    jacobian /= jacobian.det()
    return jacobian @ f @ jacobian.transpose()
