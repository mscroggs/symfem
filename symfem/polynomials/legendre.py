"""Orthogonal (Legendre) polynomials."""

import typing

import sympy

from ..functions import ScalarFunction
from ..symbols import AxisVariablesNotSingle, x


def _jrc(a, n) -> typing.Tuple[
    sympy.core.expr.Expr,
    sympy.core.expr.Expr,
    sympy.core.expr.Expr,
]:
    """Get the Jacobi recurrence relation coefficients.

    Args:
        a: The parameter a
        n: The parameter n

    Returns:
        The Jacobi coefficients
    """
    return (
        sympy.Rational((a + 2 * n + 1) * (a + 2 * n + 2), 2 * (n + 1) * (a + n + 1)),
        sympy.Rational(a * a * (a + 2 * n + 1), 2 * (n + 1) * (a + n + 1) * (a + 2 * n)),
        sympy.Rational(n * (a + n) * (a + 2 * n + 2), (n + 1) * (a + n + 1) * (a + 2 * n))
    )


def orthogonal_basis_interval(
    order: int, derivs: int, variables: AxisVariablesNotSingle = [x[0]]
) -> typing.List[typing.List[ScalarFunction]]:
    """Create a basis of orthogonal polynomials.

    Args:
        order: The maximum polynomial degree
        derivs: The number of derivatives to include
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    assert len(variables) == 1

    poly = [[ScalarFunction(1 if d == 0 else 0) for i in range(order + 1)]
            for d in range(derivs + 1)]
    for dx in range(derivs + 1):
        for i in range(1, order + 1):
            poly[dx][i] = poly[dx][i - 1] * (2 * variables[0] - 1) * (2 * i - 1) / i
            if dx > 0:
                poly[dx][i] += poly[dx - 1][i - 1] * 2 * dx * (2 * i - 1) / i
            if i > 1:
                poly[dx][i] -= (i - 1) * poly[dx][i - 2] / i
    return poly


def orthogonal_basis_triangle(
    order: int, derivs: int, variables: AxisVariablesNotSingle = [x[0], x[1]]
) -> typing.List[typing.List[ScalarFunction]]:
    """Create a basis of orthogonal polynomials.

    Args:
        order: The maximum polynomial degree
        derivs: The number of derivatives to include
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    assert len(variables) == 2

    def index(p: int, q: int) -> int:
        """Get the index."""
        return (p + q + 1) * (p + q) // 2 + q

    d_index = index

    poly = [[ScalarFunction(0) for i in range((order + 1) * (order + 2) // 2)]
            for d in range((derivs + 1) * (derivs + 2) // 2)]

    for dx in range(derivs + 1):
        for dy in range(derivs + 1 - dx):

            for p in range(order + 1):
                if p == 0:
                    poly[d_index(dx, dy)][index(0, p)] = ScalarFunction(1 if dx == dy == 0 else 0)
                else:
                    pinv = sympy.Rational(1, p)

                    poly[d_index(dx, dy)][index(0, p)] = (
                        poly[d_index(dx, dy)][index(0, p - 1)]
                        * (2 * variables[0] + variables[1] - 1) * (2 - pinv)
                    )
                    if dy > 0:
                        poly[d_index(dx, dy)][index(0, p)] += (
                            poly[d_index(dx, dy - 1)][index(0, p - 1)] * dy * (2 - pinv)
                        )
                    if dx > 0:
                        poly[d_index(dx, dy)][index(0, p)] += (
                            2 * dx * poly[d_index(dx - 1, dy)][index(0, p - 1)] * (2 - pinv)
                        )
                    if p > 1:
                        poly[d_index(dx, dy)][index(0, p)] -= (
                            poly[d_index(dx, dy)][index(0, p - 2)]
                            * (1 - variables[1]) ** 2 * (1 - pinv)
                        )
                        if dy > 0:
                            poly[d_index(dx, dy)][index(0, p)] += (
                                2 * dy * (1 - variables[1])
                                * poly[d_index(dx, dy - 1)][index(0, p - 2)] * (1 - pinv)
                            )
                        if dy > 1:
                            poly[d_index(dx, dy)][index(0, p)] -= (
                                dy * (dy - 1) * poly[d_index(dx, dy - 2)][index(0, p - 2)]
                                * (1 - pinv)
                            )

                for q in range(1, order - p + 1):
                    a, b, c = _jrc(2 * p + 1, q - 1)
                    poly[d_index(dx, dy)][index(q, p)] = (
                        poly[d_index(dx, dy)][index(q - 1, p)] * ((2 * variables[1] - 1) * a + b)
                    )
                    if q > 1:
                        poly[d_index(dx, dy)][index(q, p)] -= (
                            poly[d_index(dx, dy)][index(q - 2, p)] * c
                        )
                    if dy > 0:
                        poly[d_index(dx, dy)][index(q, p)] += (
                            poly[d_index(dx, dy - 1)][index(q - 1, p)] * 2 * dy * a
                        )
    return poly


def orthogonal_basis_quadrilateral(
    order: int, derivs: int, variables: AxisVariablesNotSingle = [x[0], x[1]]
) -> typing.List[typing.List[ScalarFunction]]:
    """Create a basis of orthogonal polynomials.

    Args:
        order: The maximum polynomial degree
        derivs: The number of derivatives to include
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    assert len(variables) == 2

    def d_index(p: int, q: int) -> int:
        """Get the derivative index."""
        return (p + q + 1) * (p + q) // 2 + q

    p0 = orthogonal_basis_interval(order, derivs, [variables[0]])
    p1 = orthogonal_basis_interval(order, derivs, [variables[1]])
    poly = [[ScalarFunction(0) for i in range((order + 1) ** 2)]
            for d in range((derivs + 1) * (derivs + 2) // 2)]
    for i in range(derivs + 1):
        for j in range(derivs + 1 - i):
            poly[d_index(i, j)] = [a * b for a in p0[i] for b in p1[j]]
    return poly


def orthogonal_basis_tetrahedron(
    order: int, derivs: int, variables: AxisVariablesNotSingle = x
) -> typing.List[typing.List[ScalarFunction]]:
    """Create a basis of orthogonal polynomials.

    Args:
        order: The maximum polynomial degree
        derivs: The number of derivatives to include
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    assert len(variables) == 3

    def index(p: int, q: int, r: int) -> int:
        """Get the index."""
        return (p + q + r) * (p + q + r + 1) * (p + q + r + 2) // 6 + (q + r) * (q + r + 1) // 2 + r

    d_index = index

    poly = [[ScalarFunction(0) for i in range((order + 1) * (order + 2) * (order + 3) // 6)]
            for d in range((derivs + 1) * (derivs + 2) * (derivs + 3) // 6)]
    for dx in range(derivs + 1):
        for dy in range(derivs + 1 - dx):
            for dz in range(derivs + 1 - dx - dy):

                for p in range(order + 1):
                    if p == 0:
                        poly[d_index(dx, dy, dz)][index(0, 0, p)] = ScalarFunction(
                            1 if dx == dy == dz == 0 else 0)
                    if p > 0:
                        invp = sympy.Rational(1, p)
                        poly[d_index(dx, dy, dz)][index(0, 0, p)] = (
                            poly[d_index(dx, dy, dz)][index(0, 0, p - 1)]
                            * (2 * variables[0] + variables[1] + variables[2] - 1) * (2 - invp)
                        )
                        if dx > 0:
                            poly[d_index(dx, dy, dz)][index(0, 0, p)] += (
                                poly[d_index(dx - 1, dy, dz)][index(0, 0, p - 1)]
                                * 2 * dx * (2 - invp)
                            )
                        if dy > 0:
                            poly[d_index(dx, dy, dz)][index(0, 0, p)] += (
                                poly[d_index(dx, dy - 1, dz)][index(0, 0, p - 1)] * dy * (2 - invp)
                            )
                        if dz > 0:
                            poly[d_index(dx, dy, dz)][index(0, 0, p)] += (
                                poly[d_index(dx, dy, dz - 1)][index(0, 0, p - 1)] * dz * (2 - invp)
                            )

                    if p > 1:
                        poly[d_index(dx, dy, dz)][index(0, 0, p)] -= (
                            poly[d_index(dx, dy, dz)][index(0, 0, p - 2)]
                            * (variables[1] + variables[2] - 1) ** 2 * (1 - invp)
                        )
                        if dy > 0:
                            poly[d_index(dx, dy, dz)][index(0, 0, p)] -= (
                                poly[d_index(dx, dy - 1, dz)][index(0, 0, p - 2)]
                                * 2 * (variables[1] + variables[2] - 1) * dy * (1 - invp)
                            )
                        if dy > 1:
                            poly[d_index(dx, dy, dz)][index(0, 0, p)] -= (
                                poly[d_index(dx, dy - 2, dz)][index(0, 0, p - 2)]
                                * dy * (dy - 1) * (1 - invp)
                            )
                        if dz > 0:
                            poly[d_index(dx, dy, dz)][index(0, 0, p)] -= (
                                poly[d_index(dx, dy, dz - 1)][index(0, 0, p - 2)]
                                * 2 * (variables[1] + variables[2] - 1) * dz * (1 - invp)
                            )
                        if dz > 1:
                            poly[d_index(dx, dy, dz)][index(0, 0, p)] -= (
                                poly[d_index(dx, dy, dz - 2)][index(0, 0, p - 2)]
                                * dz * (dz - 1) * (1 - invp)
                            )
                        if dz > 0 and dy > 0:
                            poly[d_index(dx, dy, dz)][index(0, 0, p)] -= (
                                poly[d_index(dx, dy - 1, dz - 1)][index(0, 0, p - 2)]
                                * 2 * dz * dy * (1 - invp)
                            )

                    for q in range(order - p + 1):
                        if q > 0:
                            a, b, c = _jrc(2 * p + 1, q - 1)
                            poly[d_index(dx, dy, dz)][index(0, q, p)] = (
                                poly[d_index(dx, dy, dz)][index(0, q - 1, p)]
                                * ((2 * variables[1] + variables[2] - 1) * a
                                   + (1 - variables[2]) * b)
                            )
                            if dy > 0:
                                poly[d_index(dx, dy, dz)][index(0, q, p)] += (
                                    poly[d_index(dx, dy - 1, dz)][index(0, q - 1, p)]
                                    * 2 * a * dy
                                )
                            if dz > 0:
                                poly[d_index(dx, dy, dz)][index(0, q, p)] += (
                                    poly[d_index(dx, dy, dz - 1)][index(0, q - 1, p)]
                                    * (a - b) * dz
                                )
                            if q > 1:
                                poly[d_index(dx, dy, dz)][index(0, q, p)] -= (
                                    poly[d_index(dx, dy, dz)][index(0, q - 2, p)]
                                    * (1 - variables[2]) ** 2 * c
                                )
                                if dz > 0:
                                    poly[d_index(dx, dy, dz)][index(0, q, p)] += (
                                        poly[d_index(dx, dy, dz - 1)][index(0, q - 2, p)]
                                        * 2 * (1 - variables[2]) * dz * c
                                    )
                                if dz > 1:
                                    poly[d_index(dx, dy, dz)][index(0, q, p)] -= (
                                        poly[d_index(dx, dy, dz - 2)][index(0, q - 2, p)]
                                        * dz * (dz - 1) * c
                                    )

                        for r in range(1, order - p - q + 1):
                            a, b, c = _jrc(2 * p + 2 * q + 2, r - 1)
                            poly[d_index(dx, dy, dz)][index(r, q, p)] = (
                                poly[d_index(dx, dy, dz)][index(r - 1, q, p)]
                                * ((variables[2] * 2 - 1) * a + b)
                            )
                            if dz > 0:
                                poly[d_index(dx, dy, dz)][index(r, q, p)] += (
                                    poly[d_index(dx, dy, dz - 1)][index(r - 1, q, p)] * 2 * a * dz
                                )
                            if r > 1:
                                poly[d_index(dx, dy, dz)][index(r, q, p)] -= (
                                    poly[d_index(dx, dy, dz)][index(r - 2, q, p)] * c
                                )

    return poly


def orthogonal_basis_hexahedron(
    order: int, derivs: int, variables: AxisVariablesNotSingle = x
) -> typing.List[typing.List[ScalarFunction]]:
    """Create a basis of orthogonal polynomials.

    Args:
        order: The maximum polynomial degree
        derivs: The number of derivatives to include
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    if variables is None:
        variables = x
    assert len(variables) == 3

    def d_index(p: int, q: int, r: int) -> int:
        """Get the derivative index."""
        return (p + q + r) * (p + q + r + 1) * (p + q + r + 2) // 6 + (q + r) * (q + r + 1) // 2 + r

    p0 = orthogonal_basis_interval(order, derivs, [variables[0]])
    p1 = orthogonal_basis_interval(order, derivs, [variables[1]])
    p2 = orthogonal_basis_interval(order, derivs, [variables[2]])
    poly = [[ScalarFunction(0) for i in range((order + 1) ** 3)]
            for d in range((derivs + 1) * (derivs + 2) * (derivs + 3) // 6)]
    for i in range(derivs + 1):
        for j in range(derivs + 1 - i):
            for k in range(derivs + 1 - i - j):
                poly[d_index(i, j, k)] = [a * b * c for a in p0[i] for b in p1[j] for c in p2[k]]
    return poly


def orthogonal_basis_prism(
    order: int, derivs: int, variables: AxisVariablesNotSingle = x
) -> typing.List[typing.List[ScalarFunction]]:
    """Create a basis of orthogonal polynomials.

    Args:
        order: The maximum polynomial degree
        derivs: The number of derivatives to include
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    if variables is None:
        variables = x
    assert len(variables) == 3

    def d_index_tri(p: int, q: int) -> int:
        """Get the derivative index for a triangle."""
        return (p + q + 1) * (p + q) // 2 + q

    def d_index(p: int, q: int, r: int) -> int:
        """Get the derivative index."""
        return (p + q + r) * (p + q + r + 1) * (p + q + r + 2) // 6 + (q + r) * (q + r + 1) // 2 + r

    p01 = orthogonal_basis_triangle(order, derivs, [variables[0], variables[1]])
    p2 = orthogonal_basis_interval(order, derivs, [variables[2]])
    poly = [[ScalarFunction(0) for i in range((order + 1) * (order + 1) * (order + 2) // 2)]
            for d in range((derivs + 1) * (derivs + 2) * (derivs + 3) // 6)]
    for i in range(derivs + 1):
        for j in range(derivs + 1 - i):
            for k in range(derivs + 1 - i - j):
                poly[d_index(i, j, k)] = [a * b for a in p01[d_index_tri(i, j)] for b in p2[k]]
    return poly


def orthogonal_basis_pyramid(
    order: int, derivs: int, variables: AxisVariablesNotSingle = x
) -> typing.List[typing.List[ScalarFunction]]:
    """Create a basis of orthogonal polynomials.

    Args:
        order: The maximum polynomial degree
        derivs: The number of derivatives to include
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    assert len(variables) == 3

    def index(i: int, j: int, k: int) -> int:
        """Get the index."""
        out = k + j * (order + 1) + i * (order + 1) * (order + 2) // 2 - i * (i ** 2 + 5) // 6
        if i > j:
            out -= i * (j - 1)
        else:
            out -= j * (j - 1) // 2 + i * (i - 1) // 2
        return out

    def d_index(p: int, q: int, r: int) -> int:
        """Get the derivative index."""
        return (p + q + r) * (p + q + r + 1) * (p + q + r + 2) // 6 + (q + r) * (q + r + 1) // 2 + r

    def combinations(n: int, k: int) -> int:
        """Get the number of combinations."""
        out = 1
        for i in range(n, n - k, -1):
            out *= i
        return out

    poly = [[ScalarFunction(0) for i in range((2 * order + 3) * (order + 2) * (order + 1) // 6)]
            for d in range((derivs + 1) * (derivs + 2) * (derivs + 3) // 6)]
    for dx in range(derivs + 1):
        for dy in range(derivs + 1 - dx):
            for dz in range(derivs + 1 - dx - dy):
                poly[d_index(dx, dy, dz)][0] = ScalarFunction(1 if dx == dy == dz == 0 else 0)

                for i in range(order + 1):
                    if i > 0:
                        poly[d_index(dx, dy, dz)][index(i, 0, 0)] = (
                            poly[d_index(dx, dy, dz)][index(i - 1, 0, 0)]
                            * (2 * variables[0] + variables[2] - 1) * (2 * i - 1) / i
                        )
                        if dx > 0:
                            poly[d_index(dx, dy, dz)][index(i, 0, 0)] += (
                                poly[d_index(dx - 1, dy, dz)][index(i - 1, 0, 0)]
                                * 2 * dx * (2 * i - 1) / i
                            )
                        if dz > 0:
                            poly[d_index(dx, dy, dz)][index(i, 0, 0)] += (
                                poly[d_index(dx, dy, dz - 1)][index(i - 1, 0, 0)]
                                * dz * (2 * i - 1) / i
                            )
                    if i > 1:
                        poly[d_index(dx, dy, dz)][index(i, 0, 0)] -= (
                            poly[d_index(dx, dy, dz)][index(i - 2, 0, 0)]
                            * (1 - variables[2]) ** 2 * (i - 1) / i
                        )
                        if dz > 0:
                            poly[d_index(dx, dy, dz)][index(i, 0, 0)] += (
                                poly[d_index(dx, dy, dz - 1)][index(i - 2, 0, 0)]
                                * 2 * (1 - variables[2]) * dz * (i - 1) / i
                            )
                        if dz > 1:
                            poly[d_index(dx, dy, dz)][index(i, 0, 0)] -= (
                                poly[d_index(dx, dy, dz - 2)][index(i - 2, 0, 0)]
                                * dz * (dz - 1) * (i - 1) / i
                            )

                    for j in range(order + 1):
                        if j > 0:
                            if i >= j:
                                poly[d_index(dx, dy, dz)][index(i, j, 0)] = (
                                    poly[d_index(dx, dy, dz)][index(i, j - 1, 0)]
                                    * (2 * variables[1] / (1 - variables[2]) - 1)
                                    * (2 * j - 1) / j
                                )
                                if dy > 0:
                                    poly[d_index(dx, dy, dz)][index(i, j, 0)] += (
                                        poly[d_index(dx, dy - 1, dz)][index(i, j - 1, 0)]
                                        * 2 * dy / (1 - variables[2])
                                        * (2 * j - 1) / j
                                    )
                                for di in range(1, dz + 1):
                                    poly[d_index(dx, dy, dz)][index(i, j, 0)] += (
                                        poly[d_index(dx, dy, dz - di)][index(i, j - 1, 0)]
                                        * combinations(dz, di)
                                        * variables[1] / (1 - variables[2]) ** (di + 1)
                                        * 2 * (2 * j - 1) / j
                                    )
                                    if dy > 0:
                                        poly[d_index(dx, dy, dz)][index(i, j, 0)] += (
                                            poly[d_index(dx, dy - 1, dz - di)][index(i, j - 1, 0)]
                                            * combinations(dz, di)
                                            * dy / (1 - variables[2]) ** (di + 1)
                                            * 2 * (2 * j - 1) / j
                                        )
                            else:
                                poly[d_index(dx, dy, dz)][index(i, j, 0)] = (
                                    poly[d_index(dx, dy, dz)][index(i, j - 1, 0)]
                                    * (2 * variables[1] + variables[2] - 1)
                                    * (2 * j - 1) / j
                                )
                                if dy > 0:
                                    poly[d_index(dx, dy, dz)][index(i, j, 0)] += (
                                        poly[d_index(dx, dy - 1, dz)][index(i, j - 1, 0)]
                                        * 2 * dy * (2 * j - 1) / j
                                    )
                                if dz > 0:
                                    poly[d_index(dx, dy, dz)][index(i, j, 0)] += (
                                        poly[d_index(dx, dy, dz - 1)][index(i, j - 1, 0)]
                                        * dz * (2 * j - 1) / j
                                    )
                        if j > 1:
                            if i >= j:
                                poly[d_index(dx, dy, dz)][index(i, j, 0)] -= (
                                    poly[d_index(dx, dy, dz)][index(i, j - 2, 0)]
                                    * (j - 1) / j
                                )
                            elif i + 1 == j:
                                poly[d_index(dx, dy, dz)][index(i, j, 0)] -= (
                                    poly[d_index(dx, dy, dz)][index(i, j - 2, 0)]
                                    * (1 - variables[2]) * (j - 1) / j
                                )
                                if dz > 0:
                                    poly[d_index(dx, dy, dz)][index(i, j, 0)] += (
                                        poly[d_index(dx, dy, dz - 1)][index(i, j - 2, 0)]
                                        * dz * (j - 1) / j
                                    )
                            else:
                                poly[d_index(dx, dy, dz)][index(i, j, 0)] -= (
                                    poly[d_index(dx, dy, dz)][index(i, j - 2, 0)]
                                    * (1 - variables[2]) ** 2 * (j - 1) / j
                                )
                                if dz > 0:
                                    poly[d_index(dx, dy, dz)][index(i, j, 0)] += (
                                        poly[d_index(dx, dy, dz - 1)][index(i, j - 2, 0)]
                                        * 2 * dz * (1 - variables[2]) * (j - 1) / j
                                    )
                                if dz > 1:
                                    poly[d_index(dx, dy, dz)][index(i, j, 0)] -= (
                                        poly[d_index(dx, dy, dz - 2)][index(i, j - 2, 0)]
                                        * dz * (dz - 1) * (j - 1) / j
                                    )

                        for k in range(1, order + 1 - max(i, j)):
                            a, b, c = _jrc(2 * max(i, j) + 2, k - 1)
                            poly[d_index(dx, dy, dz)][index(i, j, k)] = (
                                poly[d_index(dx, dy, dz)][index(i, j, k - 1)]
                                * a * (2 * variables[2] - 1)
                            )
                            if dz > 0:
                                poly[d_index(dx, dy, dz)][index(i, j, k)] += (
                                    poly[d_index(dx, dy, dz - 1)][index(i, j, k - 1)]
                                    * a * 2 * dz
                                )
                            poly[d_index(dx, dy, dz)][index(i, j, k)] += (
                                b * poly[d_index(dx, dy, dz)][index(i, j, k - 1)]
                            )
                            if k > 1:
                                poly[d_index(dx, dy, dz)][index(i, j, k)] -= (
                                    c * poly[d_index(dx, dy, dz)][index(i, j, k - 2)]
                                )

    return poly


def orthogonal_basis(
    cell: str, order: int, derivs: int, variables: typing.Optional[AxisVariablesNotSingle] = None
) -> typing.List[typing.List[ScalarFunction]]:
    """Create a basis of orthogonal polynomials.

    Args:
        cell: The cell type
        order: The maximum polynomial degree
        derivs: The number of derivatives to include
        variables: The variables to use

    Returns:
        A set of orthogonal polynomials
    """
    args: typing.List[typing.Any] = [order, derivs]
    if variables is not None:
        args.append(variables)

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
        return orthogonal_basis_pyramid(*args)

    raise ValueError(f"Unsupported cell type: {cell}")


def orthonormal_basis(
    cell: str, order: int, derivs: int, variables: typing.Optional[AxisVariablesNotSingle] = None
) -> typing.List[typing.List[ScalarFunction]]:
    """Create a basis of orthonormal polynomials.

    Args:
        cell: The cell type
        order: The maximum polynomial degree
        derivs: The number of derivatives to include
        variables: The variables to use

    Returns:
        A set of orthonormal polynomials
    """
    from ..create import create_reference

    poly = orthogonal_basis(cell, order, derivs, variables)
    ref = create_reference(cell)
    if variables is None:
        variables = x
    norms = [sympy.sqrt((f ** 2).integral(ref, dummy_vars=variables)) for f in poly[0]]
    for i, n in enumerate(norms):
        for j in range(len(poly)):
            poly[j][i] /= n
    return poly
