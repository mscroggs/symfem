"""Dual polynomials."""

import typing

import sympy

from ..functions import ScalarFunction


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

    matrix = sympy.Matrix([[(p * q).integral(reference) for q in poly] for p in poly])
    minv = matrix.inv("LU")

    out = []
    for i in range(minv.rows):
        f = ScalarFunction(0)
        for j, p in zip(minv.row(i), poly):
            f += j * p
        out.append(f)

    return out
