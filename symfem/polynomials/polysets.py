"""Polynomial sets."""

import typing
from itertools import product

from ..functions import ScalarFunction, VectorFunction
from ..symbols import AxisVariablesNotSingle, x


def polynomial_set_1d(
    dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[ScalarFunction]:
    """One dimensional polynomial set.

    Args:
        dim: The number of variables
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    if dim == 1:
        return [ScalarFunction(variables[0] ** i) for i in range(order + 1)]
    if dim == 2:
        return [
            ScalarFunction(variables[0] ** i * variables[1] ** j)
            for j in range(order + 1)
            for i in range(order + 1 - j)
        ]
    if dim == 3:
        return [
            ScalarFunction(variables[0] ** i * variables[1] ** j * variables[2] ** k)
            for k in range(order + 1)
            for j in range(order + 1 - k)
            for i in range(order + 1 - k - j)
        ]
    raise ValueError(f"Unsupported dimension: {dim}")


def polynomial_set_vector(
    domain_dim: int, range_dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[VectorFunction]:
    """Polynomial set.

    Args:
        domain_dim: The number of variables
        range_dim: The dimension of the output vector
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    set1d = polynomial_set_1d(domain_dim, order, variables)
    return [
        VectorFunction([p if i == j else 0 for j in range(range_dim)])
        for p in set1d
        for i in range(range_dim)
    ]


def Hdiv_polynomials(
    domain_dim: int, range_dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[VectorFunction]:
    """Hdiv conforming polynomial set.

    Args:
        domain_dim: The number of variables
        range_dim: The dimension of the output vector
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    assert domain_dim == range_dim
    if domain_dim == 2:
        return [VectorFunction((
            variables[0] * variables[0] ** (order - 1 - j) * variables[1] ** j,
            variables[1] * variables[0] ** (order - 1 - j) * variables[1] ** j,
        )) for j in range(order)]
    if domain_dim == 3:
        basis: typing.List[VectorFunction] = []
        for j in range(order):
            for k in range(order - j):
                p = variables[0] ** (order - 1 - j - k) * variables[1] ** j * variables[2] ** k
                basis.append(VectorFunction((variables[0] * p, variables[1] * p, variables[2] * p)))
        return basis

    raise ValueError(f"Unsupported dimension: {domain_dim}")


def Hcurl_polynomials(
    domain_dim: int, range_dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[VectorFunction]:
    """Hcurl conforming polynomial set.

    Args:
        domain_dim: The number of variables
        range_dim: The dimension of the output vector
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    assert domain_dim == range_dim
    if domain_dim == 2:
        return [VectorFunction((
            variables[0] ** (order - 1 - j) * variables[1] ** (j + 1),
            -variables[0] ** (order - j) * variables[1] ** j,
        )) for j in range(order)]
    if domain_dim == 3:
        poly: typing.List[VectorFunction] = []
        poly += [VectorFunction((
            variables[0] ** (m - 1) * variables[1] ** n * variables[2] ** (order - m - n + 1),
            0, -variables[0] ** m * variables[1] ** n * variables[2] ** (order - m - n)
        )) for n in range(order) for m in range(1, order + 1 - n)]
        poly += [VectorFunction((
            0, variables[0] ** m * variables[1] ** (n - 1) * variables[2] ** (order - m - n + 1),
            -variables[0] ** m * variables[1] ** n * variables[2] ** (order - m - n)
        )) for m in range(order) for n in range(1, order + 1 - m)]
        poly += [VectorFunction((
            variables[0] ** (order - n) * variables[1] ** n,
            -variables[0] ** (order + 1 - n) * variables[1] ** (n - 1), 0
        )) for n in range(1, order + 1)]
        return poly
    raise ValueError(f"Unsupported dimension: {domain_dim}")


def quolynomial_set_1d(
    dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[ScalarFunction]:
    """One dimensional quolynomial set.

    Args:
        dim: The number of variables
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    basis = []
    for j in product(range(order + 1), repeat=dim):
        poly = ScalarFunction(1)
        for a, b in zip(variables, j):
            poly *= a ** b
        basis.append(poly)
    return basis


def quolynomial_set_vector(
    domain_dim: int, range_dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[VectorFunction]:
    """Quolynomial set.

    Args:
        domain_dim: The number of variables
        range_dim: The dimension of the output vector
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    set1d = quolynomial_set_1d(domain_dim, order, variables)
    return [
        VectorFunction([p if i == j else 0 for j in range(range_dim)])
        for p in set1d
        for i in range(range_dim)
    ]


def Hdiv_quolynomials(
    domain_dim: int, range_dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[VectorFunction]:
    """Hdiv conforming quolynomial set.

    Args:
        domain_dim: The number of variables
        range_dim: The dimension of the output vector
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    assert domain_dim == range_dim
    basis: typing.List[VectorFunction] = []
    for d in range(domain_dim):
        for j in product(range(order), repeat=domain_dim - 1):
            poly = 1
            for a, b in zip(variables, j[:d] + (order,) + j[d:]):
                poly *= a ** b
            basis.append(VectorFunction([poly if i == d else 0 for i in range(domain_dim)]))
    return basis


def Hcurl_quolynomials(
    domain_dim: int, range_dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[VectorFunction]:
    """Hcurl conforming quolynomial set.

    Args:
        domain_dim: The number of variables
        range_dim: The dimension of the output vector
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    assert domain_dim == range_dim
    basis: typing.List[VectorFunction] = []
    for d in range(domain_dim):
        for j in product(
            *[range(order) if i == d else range(order + 1) for i in range(domain_dim)]
        ):
            if order not in j:
                continue
            poly = 1
            for a, b in zip(variables, j):
                poly *= a ** b
            basis.append(VectorFunction([poly if i == d else 0 for i in range(domain_dim)]))
    return basis


def serendipity_indices(
    total: int, linear: int, dim: int, done: typing.Optional[typing.List[int]] = None
) -> typing.List[typing.List[int]]:
    """Get the set indices for a serendipity polynomial set.

    Args:
        dim: The number of variables
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    if done is None:
        done = []
    if len(done) == dim:
        if done.count(1) >= linear:
            return [done]
        return []
    if len(done) == dim - 1:
        return serendipity_indices(total, linear, dim, done=done + [total - sum(done)])
    out = []
    for i in range(total - sum(done) + 1):
        out += serendipity_indices(total, linear, dim, done + [i])
    return out


def serendipity_set_1d(
    dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[ScalarFunction]:
    """One dimensional serendipity set.

    Args:
        dim: The number of variables
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    basis: typing.List[ScalarFunction] = []
    for s in range(order + 1, order + dim + 1):
        for i in serendipity_indices(s, s - order, dim):
            p = 1
            for j, k in zip(variables, i):
                p *= j ** k
            basis.append(ScalarFunction(p))
    return basis


def serendipity_set_vector(
    domain_dim: int, range_dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[VectorFunction]:
    """Serendipity set.

    Args:
        domain_dim: The number of variables
        range_dim: The dimension of the output vector
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    set1d = serendipity_set_1d(domain_dim, order, variables)
    return [
        VectorFunction([p if i == j else 0 for j in range(range_dim)])
        for p in set1d
        for i in range(range_dim)
    ]


def Hdiv_serendipity(
    domain_dim: int, range_dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[VectorFunction]:
    """Hdiv conforming serendipity set.

    Args:
        domain_dim: The number of variables
        range_dim: The dimension of the output vector
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    assert domain_dim == range_dim
    if domain_dim == 2:
        return [
            VectorFunction((variables[0] ** (order + 1),
                            (order + 1) * variables[0] ** order * variables[1])),
            VectorFunction(((order + 1) * variables[0] * variables[1] ** order,
                            variables[1] ** (order + 1))),
        ]
    if domain_dim == 3:
        a = []
        if order == 0:
            a.append(VectorFunction((0, variables[0] * variables[2], -variables[0] * variables[1])))
            a.append(VectorFunction((variables[1] * variables[2], 0, -variables[0] * variables[1])))
        else:
            for i in range(order + 1):
                p = variables[1] ** i * variables[2] ** (order - i)
                a.append(VectorFunction((0, variables[0] * variables[2] * p,
                                         -variables[0] * variables[1] * p)))

                p = variables[0] ** i * variables[2] ** (order - i)
                a.append(VectorFunction((variables[1] * variables[2] * p, 0,
                                         -variables[0] * variables[1] * p)))

                p = variables[0] ** i * variables[1] ** (order - i)
                a.append(VectorFunction((variables[1] * variables[2] * p,
                                         -variables[0] * variables[2] * p, 0)))

        return [i.curl() for i in a]

    raise NotImplementedError()


def Hcurl_serendipity(
    domain_dim: int, range_dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[VectorFunction]:
    """Hcurl conforming serendipity set.

    Args:
        domain_dim: The number of variables
        range_dim: The dimension of the output vector
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    assert domain_dim == range_dim
    if domain_dim == 2:
        return [
            VectorFunction(((order + 1) * variables[0] ** order * variables[1],
                            -variables[0] ** (order + 1))),
            VectorFunction((variables[1] ** (order + 1),
                            (order + 1) * variables[0] * variables[1] ** order)),
        ]
    if domain_dim == 3:
        out: typing.List[VectorFunction] = []
        if order == 1:
            out += [
                VectorFunction((0, variables[0] * variables[2], -variables[0] * variables[1])),
                VectorFunction((variables[1] * variables[2], 0, -variables[0] * variables[1])),
            ]
        else:
            for i in range(order):
                p = variables[0] ** i * variables[2] ** (order - 1 - i)
                out.append(VectorFunction((variables[1] * variables[2] * p, 0,
                                           -variables[0] * variables[1] * p)))

                p = variables[1] ** i * variables[2] ** (order - 1 - i)
                out.append(VectorFunction((0, variables[0] * variables[2] * p,
                                           -variables[0] * variables[1] * p)))

                p = variables[0] ** i * variables[1] ** (order - 1 - i)
                out.append(VectorFunction((variables[1] * variables[2] * p,
                                           -variables[0] * variables[2] * p, 0)))

        for p in serendipity_set_1d(domain_dim, order + 1):
            out.append(VectorFunction(tuple(p.diff(i) for i in variables)))
        return out

    raise NotImplementedError()


def prism_polynomial_set_1d(
    dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[ScalarFunction]:
    """One dimensional polynomial set.

    Args:
        dim: The number of variables
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    assert dim == 3
    return [
        ScalarFunction(variables[0] ** i * variables[1] ** j * variables[2] ** k)
        for k in range(order + 1)
        for j in range(order + 1)
        for i in range(order + 1 - j)
    ]


def prism_polynomial_set_vector(
    domain_dim: int, range_dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[VectorFunction]:
    """Polynomial set for a prism.

    Args:
        domain_dim: The number of variables
        range_dim: The dimension of the output vector
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    set1d = prism_polynomial_set_1d(domain_dim, order, variables)
    return [
        VectorFunction([p if i == j else 0 for j in range(range_dim)])
        for p in set1d
        for i in range(range_dim)
    ]


def pyramid_polynomial_set_1d(
    dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[ScalarFunction]:
    """One dimensional polynomial set.

    Args:
        dim: The number of variables
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    assert dim == 3
    if order == 0:
        return [ScalarFunction(1)]

    poly = polynomial_set_1d(3, order)
    for d in range(order):
        for i in range(d + 1):
            for j in range(d + 1 - i):
                p = variables[0] ** i * variables[1] ** j * variables[2] ** (d - i - j)
                p *= (variables[0] * variables[1] / (1 - variables[2])) ** (order - d)
                poly.append(ScalarFunction(p))

    return poly


def pyramid_polynomial_set_vector(
    domain_dim: int, range_dim: int, order: int, variables: AxisVariablesNotSingle = x
) -> typing.List[VectorFunction]:
    """Polynomial set for a pyramid.

    Args:
        domain_dim: The number of variables
        range_dim: The dimension of the output vector
        order: The maximum polynomial degree
        variables: The variables to use

    Returns:
        A set of polynomials
    """
    set1d = pyramid_polynomial_set_1d(domain_dim, order, variables)
    return [
        VectorFunction([p if i == j else 0 for j in range(range_dim)])
        for p in set1d
        for i in range(range_dim)
    ]
