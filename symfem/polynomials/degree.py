"""Functions to compute the degree of polynomials."""

import typing
import sympy
from symfem.references import Reference
from symfem.functions import ScalarFunction
from symfem.symbols import x


def simplex_degree(polynomial: ScalarFunction, vars: typing.Tuple[sympy.Symbol, ...] = x) -> int:
    """Get the degree of a polynomial on a simplex cell.

    Args:
        polynomial: The polynomial

    Returns:
        The degree of the polynomial on a simplex cell
    """
    p = polynomial.as_sympy().as_poly()
    if p is None:
        return 0
    else:
        return p.degree()


def tp_degree(polynomial: ScalarFunction, vars: typing.Tuple[sympy.Symbol, ...] = x) -> int:
    """Get the degree of a polynomial on a tensor product cell.

    Args:
        polynomial: The polynomial

    Returns:
        The degree of the polynomial on a tensor product cell
    """
    return max(
        0 if p is None else p.degree()
        for p in [
            polynomial.as_sympy().subs(vars[i], 1).subs(vars[j], 1).as_poly()
            for i, j in [(0, 1), (0, 2), (1, 2)]
        ]
    )


def degree(
    reference: Reference, polynomial: ScalarFunction, vars: typing.Tuple[sympy.Symbol, ...] = x
) -> int:
    """Get the degree of a polynomial on a reference cell.

    Args:
        reference: The reference cell
        polynomial: The polynomial

    Returns:
        The degree of the polynomial on the reference cell
    """
    if reference.name in ["interval", "triangle", "tetrahedron"]:
        return simplex_degree(polynomial, vars)
    if reference.name in ["quadrilataral", "hexahedron"]:
        return simplex_degree(polynomial, vars)

    raise ValueError(f"Unsupported cell: {reference.name}")
