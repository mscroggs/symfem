"""Functions to compute the degree of polynomials."""

import typing
import sympy
from symfem.references import Reference
from symfem.functions import ScalarFunction, Function
from symfem.basis_functions import SubbedBasisFunction
from symfem.symbols import x


def simplex_degree(polynomial: ScalarFunction, vars: typing.Tuple[sympy.Symbol, ...] = x) -> int:
    """Get the degree of a polynomial on a simplex cell.

    Args:
        polynomial: The polynomial

    Returns:
        The degree of the polynomial on a simplex cell
    """
    p = polynomial.subs(x[1:], [x[0], x[0]]).as_sympy().as_poly()
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
        simplex_degree(polynomial.subs([vars[i], vars[j]], [1, 1]))
        for i, j in [(0, 1), (0, 2), (1, 2)]
    )


def prism_degree(polynomial: ScalarFunction, vars: typing.Tuple[sympy.Symbol, ...] = x) -> int:
    """Get the degree of a polynomial on a prism.

    Args:
        polynomial: The polynomial

    Returns:
        The degree of the polynomial on a prism
    """
    return max(
        simplex_degree(polynomial.subs([vars[0]], [1])),
        simplex_degree(polynomial.subs([vars[1]], [1])),
    )


def pyramid_degree(polynomial: ScalarFunction, vars: typing.Tuple[sympy.Symbol, ...] = x) -> int:
    """Get the degree of a polynomial on a pyramid.

    Args:
        polynomial: The polynomial

    Returns:
        The degree of the polynomial on a pyramid
    """
    return tp_degree(polynomial.subs([x[0], x[1]], [x[0] * (1 - x[2]), x[1] * (1 - x[2])]))


def degree(
    reference: Reference, polynomial: Function, vars: typing.Tuple[sympy.Symbol, ...] = x
) -> int:
    """Get the degree of a polynomial on a reference cell.

    Args:
        reference: The reference cell
        polynomial: The polynomial

    Returns:
        The degree of the polynomial on the reference cell
    """
    if polynomial.is_vector:
        return max(degree(reference, component, vars) for component in polynomial)
    if polynomial.is_matrix:
        return max(
            degree(reference, polynomial[i][j], vars)
            for i in range(polynomial.shape[0])
            for j in range(polynomial.shape[1])
        )

    if isinstance(polynomial, SubbedBasisFunction):
        return degree(reference, polynomial.get_function(), vars)

    if not isinstance(polynomial, ScalarFunction):
        raise NotImplementedError(f"Unsupported polynomial type: {type(polynomial)}")

    if reference.name in ["interval", "triangle", "tetrahedron"]:
        return simplex_degree(polynomial, vars)
    if reference.name in ["quadrilateral", "hexahedron"]:
        return tp_degree(polynomial, vars)
    if reference.name == "prism":
        return prism_degree(polynomial, vars)
    if reference.name == "pyramid":
        return pyramid_degree(polynomial, vars)
    raise ValueError(f"Unsupported cell: {reference.name}")
