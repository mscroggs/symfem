"""Lobatto polynomials."""

import typing

from ..functions import ScalarFunction
from ..symbols import x
from .dual import l2_dual
from .legendre import orthonormal_basis


def lobatto_basis_interval(order: int) -> typing.List[ScalarFunction]:
    """Get Lobatto polynomials on an interval.

    Args:
        order: The maximum polynomial degree

    Returns:
        Lobatto polynomials
    """
    legendre = orthonormal_basis("interval", order - 1, 0)[0]
    out = [ScalarFunction(1)]
    for f in legendre:
        out.append(f.integrate((x[0], 0, x[0])))
    return out


def lobatto_dual_basis_interval(order: int) -> typing.List[ScalarFunction]:
    """Get L2 dual of Lobatto polynomials on an interval.

    Args:
        order: The maximum polynomial degree

    Returns:
        Dual Lobatto polynomials
    """
    return l2_dual("interval", lobatto_basis_interval(order))


def lobatto_basis(
    cell: str, order: int, include_endpoints: bool = True
) -> typing.List[ScalarFunction]:
    """Get Lobatto polynomials.

    Args:
        cell: The cell type
        order: The maximum polynomial degree
        include_endpoint: should polynomials that are non-zero on the boundary be included?

    Returns:
        Lobatto polynomials
    """
    if cell == "interval":
        if include_endpoints:
            return lobatto_basis_interval(order)
        else:
            return lobatto_basis_interval(order)[2:]
    if cell == "quadrilateral":
        interval = lobatto_basis("interval", order, include_endpoints)
        return [i * j.subs(x[0], x[1]) for i in interval for j in interval]
    if cell == "hexahedron":
        interval = lobatto_basis("interval", order, include_endpoints)
        return [i * j.subs(x[0], x[1]) * k.subs(x[0], x[2])
                for i in interval for j in interval for k in interval]
    raise NotImplementedError(f"Lobatto polynomials not implemented for cell \"{cell}\"")


def lobatto_dual_basis(
    cell: str, order: int, include_endpoints: bool = True
) -> typing.List[ScalarFunction]:
    """Get L2 dual of Lobatto polynomials.

    Args:
        cell: The cell type
        order: The maximum polynomial degree
        include_endpoint: should polynomials that are non-zero on the boundary be included?

    Returns:
        Lobatto polynomials
    """
    if cell == "interval":
        if include_endpoints:
            return lobatto_dual_basis_interval(order)
        else:
            return lobatto_dual_basis_interval(order)[2:]
    if cell == "quadrilateral":
        interval = lobatto_dual_basis("interval", order, include_endpoints)
        return [i * j.subs(x[0], x[1]) for i in interval for j in interval]
    if cell == "hexahedron":
        interval = lobatto_dual_basis("interval", order, include_endpoints)
        return [i * j.subs(x[0], x[1]) * k.subs(x[0], x[2])
                for i in interval for j in interval for k in interval]
    raise NotImplementedError(f"Lobatto polynomials not implemented for cell \"{cell}\"")
