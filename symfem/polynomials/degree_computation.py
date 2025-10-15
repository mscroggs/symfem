"""Functions to compute the degree of polynomials."""

import typing
import sympy
from symfem.references import Reference
from symfem.functions import ScalarFunction, Function
from symfem.basis_functions import SubbedBasisFunction
from symfem.symbols import x


def monomial_degree(term: sympy.core.expr.Expr, vars: typing.Tuple[sympy.Symbol, ...] = x) -> int:
    """Get the degree of a sympy monomial."""
    return term.as_poly(vars).degree()


def simplex_degree(
    polynomial: sympy.core.expr.Expr, vars: typing.Tuple[sympy.Symbol, ...] = x
) -> int:
    """Get the degree of a polynomial on a simplex cell.

    Args:
        polynomial: The polynomial

    Returns:
        The degree of the polynomial on a simplex cell
    """
    return max(monomial_degree(term, vars) for term in polynomial.expand().as_coefficients_dict())


def tp_degree(polynomial: sympy.core.expr.Expr, vars: typing.Tuple[sympy.Symbol, ...] = x) -> int:
    """Get the degree of a polynomial on a tensor product cell.

    Args:
        polynomial: The polynomial

    Returns:
        The degree of the polynomial on a tensor product cell
    """
    return max(monomial_degree(term, vars) for term in polynomial.expand().as_coefficients_dict())


def prism_degree(
    polynomial: sympy.core.expr.Expr, vars: typing.Tuple[sympy.Symbol, ...] = x
) -> int:
    """Get the degree of a polynomial on a prism.

    Args:
        polynomial: The polynomial

    Returns:
        The degree of the polynomial on a prism
    """
    return tp_degree(polynomial.subs(vars[0], vars[1]), vars)


def pyramid_degree(
    polynomial: sympy.core.expr.Expr, vars: typing.Tuple[sympy.Symbol, ...] = x
) -> int:
    """Get the degree of a polynomial on a pyramid.

    Args:
        polynomial: The polynomial

    Returns:
        The degree of the polynomial on a pyramid
    """
    return tp_degree(
        polynomial.subs(vars[0], vars[0] * (1 - vars[2])).subs(vars[1], vars[1] * (1 - vars[2])), vars
    )


def degree(
    reference: Reference, polynomial: Function, vars: typing.Tuple[sympy.Symbol, ...] = x
) -> int:
    """Get the Lagrange degree of a polynomial on a reference cell.

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

    sym_poly = polynomial.as_sympy()
    assert isinstance(sym_poly, sympy.core.expr.Expr)

    if reference.name in ["interval", "triangle", "tetrahedron"]:
        return simplex_degree(sym_poly, vars)
    if reference.name in ["quadrilateral", "hexahedron"]:
        return tp_degree(sym_poly, vars)
    if reference.name == "prism":
        return prism_degree(sym_poly, vars)
    if reference.name == "pyramid":
        return pyramid_degree(sym_poly, vars)
    raise ValueError(f"Unsupported cell: {reference.name}")
