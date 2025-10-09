"""Orthogonal (Legendre) polynomials."""

import typing

import sympy

from symfem.functions import ScalarFunction
from symfem.symbols import AxisVariablesNotSingle, x
from symfem.polynomials.jacobi import jacobi_polynomial

__all__: typing.List[str] = []


def orthogonal_basis_interval(
    order: int, variables: AxisVariablesNotSingle = [x[0]]
) -> typing.List[ScalarFunction]:
    """Create a basis of orthogonal polynomials.

    Args:
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    assert len(variables) == 1

    return [jacobi_polynomial(i, 0, 0, 2 * variables[0] - 1) for i in range(order + 1)]


def orthogonal_basis_triangle(
    order: int, variables: AxisVariablesNotSingle = [x[0], x[1]]
) -> typing.List[ScalarFunction]:
    """Create a basis of orthogonal polynomials.

    Args:
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    assert len(variables) == 2

    return [
        jacobi_polynomial(p, 0, 0, 2 * variables[0] / (1 - variables[1]) - 1)
        * (1 - variables[1]) ** p
        * jacobi_polynomial(q, 2 * p + 1, 0, 2 * variables[1] - 1)
        for p in range(order + 1)
        for q in range(order - p + 1)
    ]


def orthogonal_basis_quadrilateral(
    order: int, variables: AxisVariablesNotSingle = [x[0], x[1]]
) -> typing.List[ScalarFunction]:
    """Create a basis of orthogonal polynomials.

    Args:
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    assert len(variables) == 2
    return [
        a * b
        for a in orthogonal_basis_interval(order, [variables[0]])
        for b in orthogonal_basis_interval(order, [variables[1]])
    ]


def orthogonal_basis_tetrahedron(
    order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[ScalarFunction]:
    """Create a basis of orthogonal polynomials.

    Args:
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    assert len(variables) == 3

    return [
        jacobi_polynomial(p, 0, 0, 2 * variables[0] / (1 - variables[1] - variables[2]) - 1)
        * (1 - variables[1] - variables[2]) ** p
        * jacobi_polynomial(q, 2 * p + 1, 0, 2 * variables[1] / (1 - variables[2]) - 1)
        * (1 - variables[2]) ** q
        * jacobi_polynomial(r, 2 * (p + q + 1), 0, 2 * variables[2] - 1)
        for p in range(order + 1)
        for q in range(order - p + 1)
        for r in range(order - p - q + 1)
    ]


def orthogonal_basis_hexahedron(
    order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[ScalarFunction]:
    """Create a basis of orthogonal polynomials.

    Args:
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    assert len(variables) == 3
    return [
        a * b * c
        for a in orthogonal_basis_interval(order, [variables[0]])
        for b in orthogonal_basis_interval(order, [variables[1]])
        for c in orthogonal_basis_interval(order, [variables[2]])
    ]


def orthogonal_basis_prism(
    order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[ScalarFunction]:
    """Create a basis of orthogonal polynomials.

    Args:
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    assert len(variables) == 3

    return [
        a * b
        for a in orthogonal_basis_triangle(order, [variables[0], variables[1]])
        for b in orthogonal_basis_interval(order, [variables[2]])
    ]


def orthogonal_basis_lagrange_pyramid(
    order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[ScalarFunction]:
    """Create a basis of orthogonal rationomials.

    Args:
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    assert len(variables) == 3

    return [
        jacobi_polynomial(p, 0, 0, (2 * variables[0] + variables[2] - 1) / (1 - variables[2]))
        * jacobi_polynomial(q, 0, 0, (2 * variables[1] + variables[2] - 1) / (1 - variables[2]))
        * (1 - variables[2]) ** max(p, q)
        * jacobi_polynomial(r, 2 * max(p, q) + 2, 0, 2 * variables[2] - 1)
        for r in range(order + 1)
        for p in range(order + 1 - r)
        for q in range(order + 1 - r)
    ]


def orthogonal_basis_full_pyramid(
    order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[ScalarFunction]:
    """Create a basis of orthogonal rationomials.

    Args:
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    assert len(variables) == 3

    return [
        jacobi_polynomial(p, 0, 0, (2 * variables[0] + variables[2] - 1) / (1 - variables[2]))
        * jacobi_polynomial(q, 0, 0, (2 * variables[1] + variables[2] - 1) / (1 - variables[2]))
        * jacobi_polynomial(r, 2, 0, 2 * variables[2] - 1)
        for p in range(order + 1)
        for q in range(order + 1)
        for r in range(order + 1)
    ]


def orthogonal_basis(
    cell: str,
    order: int,
    variables: typing.Optional[AxisVariablesNotSingle] = None,
    ptype: str = "Lagrange",
) -> typing.List[ScalarFunction]:
    """Create a basis of orthogonal polynomials.

    Args:
        cell: The cell type
        order: The maximum polynomial degree
        variables: The variables to use
        ptype: The type of the polynomial set (pyramids only)

    Returns:
        A set of orthogonal polynomials
    """
    args: typing.List[typing.Any] = [order]
    if variables is not None:
        args.append(variables)

    if ptype != "Lagrange" and cell != "pyramid":
        raise ValueError("Cannot use ptype input for non-pyramid cells")

    if cell == "interval":
        return orthogonal_basis_interval(*args)
    if cell == "triangle":
        return orthogonal_basis_triangle(*args)
    if cell == "quadrilateral":
        return orthogonal_basis_quadrilateral(*args)
    if cell == "hexahedron":
        return orthogonal_basis_hexahedron(*args)
    if cell == "tetrahedron":
        return orthogonal_basis_tetrahedron(*args)
    if cell == "prism":
        return orthogonal_basis_prism(*args)
    if cell == "pyramid":
        if ptype == "Lagrange":
            return orthogonal_basis_lagrange_pyramid(*args)
        if ptype == "full":
            return orthogonal_basis_full_pyramid(*args)

    raise ValueError(f"Unsupported cell type: {cell}")


def orthonormal_basis(
    cell: str,
    order: int,
    variables: typing.Optional[AxisVariablesNotSingle] = None,
    ptype: str = "Lagrange",
) -> typing.List[ScalarFunction]:
    """Create a basis of orthonormal polynomials.

    Args:
        cell: The cell type
        order: The maximum polynomial degree
        variables: The variables to use
        ptype: The type of the polynomial set (pyramids only)

    Returns:
        A set of orthonormal polynomials
    """
    from symfem.create import create_reference

    if ptype != "Lagrange" and cell != "pyramid":
        raise ValueError("Cannot use ptype input for non-pyramid cells")

    poly = orthogonal_basis(cell, order, variables, ptype)
    ref = create_reference(cell)
    if variables is None:
        variables = x
    norms = [sympy.sqrt((f**2).integral(ref, dummy_vars=variables)) for f in poly]
    for i, n in enumerate(norms):
        poly[i] /= n
    return poly
