"""Dual polynomials."""

from ..functions import ScalarFunction, VectorFunction
from ..symbols import AxisVariablesNotSingle, x, t
from .legendre import orthonormal_basis
import typing
import sympy


def l2_dual(cell: str, poly: typing.List[ScalarFunction]) -> typing.List[ScalarFunction]:
    """Compute the L2 dual of a set of polynomials.

    Args:
        cell: The cell type
        poly: The set of polynomial

    Returns:
        The L2 dual polynomials
    """
    from ..create import create_reference
    reference = create_reference(cell)

    matrix = sympy.Matrix([[(p * q).integrate(*reference.integration_limits(x)) for q in poly] for p in poly])
    minv = matrix.inv("LU")

    return [sum(j * p for j, p in zip(minv.row(i), poly)) for i in range(minv.rows)]
    return poly
