"""Polynomial sets."""
from .symbolic import x, zero, one
from itertools import product


def polynomial_set_1d(dim, order, vars=x):
    """One dimensional polynomial set."""
    if dim == 1:
        return [vars[0] ** i for i in range(order + 1)]
    if dim == 2:
        return [
            vars[0] ** i * vars[1] ** j
            for j in range(order + 1)
            for i in range(order + 1 - j)
        ]
    if dim == 3:
        return [
            vars[0] ** i * vars[1] ** j * vars[2] ** k
            for k in range(order + 1)
            for j in range(order + 1 - k)
            for i in range(order + 1 - k - j)
        ]


def polynomial_set(domain_dim, range_dim, order):
    """Polynomial set."""
    if range_dim == 1:
        return polynomial_set_1d(domain_dim, order)
    set1d = polynomial_set_1d(domain_dim, order)
    return [
        tuple(p if i == j else zero for j in range(range_dim))
        for p in set1d
        for i in range(range_dim)
    ]


def Hdiv_polynomials(domain_dim, range_dim, order):
    """Hdiv conforming polynomial set."""
    assert domain_dim == range_dim
    if domain_dim == 2:
        return [
            (
                x[0] * x[0] ** (order - 1 - j) * x[1] ** j,
                x[1] * x[0] ** (order - 1 - j) * x[1] ** j,
            )
            for j in range(order)
        ]
    if domain_dim == 3:
        basis = []
        for j in range(order):
            for k in range(order - j):
                p = x[0] ** (order - 1 - j - k) * x[1] ** j * x[2] ** k
                basis.append((x[0] * p, x[1] * p, x[2] * p))
        return basis


def Hcurl_polynomials(domain_dim, range_dim, order):
    """Hcurl conforming polynomial set."""
    assert domain_dim == range_dim
    if domain_dim == 2:
        return [
            (
                x[0] ** (order - 1 - j) * x[1] ** (j + 1),
                -x[0] ** (order - j) * x[1] ** j,
            )
            for j in range(order)
        ]
    if domain_dim == 3:
        poly = []
        poly += [[x[0] ** (m - 1) * x[1] ** n * x[2] ** (order - m - n + 1),
                  zero,
                  -x[0] ** m * x[1] ** n * x[2] ** (order - m - n)]
                 for n in range(order) for m in range(1, order + 1 - n)]
        poly += [[zero,
                  x[0] ** m * x[1] ** (n - 1) * x[2] ** (order - m - n + 1),
                  -x[0] ** m * x[1] ** n * x[2] ** (order - m - n)]
                 for m in range(order) for n in range(1, order + 1 - m)]
        poly += [[x[0] ** (order - n) * x[1] ** n,
                  -x[0] ** (order + 1 - n) * x[1] ** (n - 1),
                  zero]
                 for n in range(1, order + 1)]
        return poly


def quolynomial_set_1d(dim, order):
    """One dimensional quolynomial set."""
    basis = []
    for j in product(range(order + 1), repeat=dim):
        poly = 1
        for a, b in zip(x, j):
            poly *= a ** b
        basis.append(poly)
    return basis


def quolynomial_set(domain_dim, range_dim, order):
    """Quolynomial set."""
    if range_dim == 1:
        return quolynomial_set_1d(domain_dim, order)
    set1d = quolynomial_set_1d(domain_dim, order)
    return [
        tuple(p if i == j else zero for j in range(range_dim))
        for p in set1d
        for i in range(range_dim)
    ]


def Hdiv_quolynomials(domain_dim, range_dim, order):
    """Hdiv conforming quolynomial set."""
    assert domain_dim == range_dim
    basis = []
    for d in range(domain_dim):
        for j in product(range(order), repeat=domain_dim - 1):
            poly = one
            for a, b in zip(x, j[:d] + (order,) + j[d:]):
                poly *= a ** b
            basis.append(tuple(poly if i == d else zero for i in range(domain_dim)))
    return basis


def Hcurl_quolynomials(domain_dim, range_dim, order):
    """Hcurl conforming quolynomial set."""
    assert domain_dim == range_dim
    basis = []
    for d in range(domain_dim):
        for j in product(
            *[range(order) if i == d else range(order + 1) for i in range(domain_dim)]
        ):
            if order not in j:
                continue
            poly = one
            for a, b in zip(x, j):
                poly *= a ** b
            basis.append(tuple(poly if i == d else zero for i in range(domain_dim)))
    return basis


def serendipity_indices(total, linear, dim, done=[]):
    """Get the set indices for a serendipity polynomial set."""
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


def serendipity_set_1d(dim, order):
    """One dimensional serendipity set."""
    basis = []
    for s in range(order + 1, order + dim + 1):
        for i in serendipity_indices(s, s - order, dim):
            p = one
            for j, k in zip(x, i):
                p *= j ** k
            basis.append(p)
    return basis


def serendipity_set(domain_dim, range_dim, order):
    """Serendipity set."""
    if range_dim == 1:
        return serendipity_set_1d(domain_dim, order)
    set1d = serendipity_set_1d(domain_dim, order)
    return [
        tuple(p if i == j else zero for j in range(range_dim))
        for p in set1d
        for i in range(range_dim)
    ]


def Hdiv_serendipity(domain_dim, range_dim, order):
    """Hdiv conforming serendipity set."""
    assert domain_dim == range_dim
    if domain_dim == 2:
        return [
            (x[0] ** (order + 1), (order + 1) * x[0] ** order * x[1]),
            ((order + 1) * x[0] * x[1] ** order, x[1] ** (order + 1)),
        ]
    if domain_dim == 3:
        a = []
        if order == 0:
            a.append((zero, x[0] * x[2], -x[0] * x[1]))
            a.append((x[1] * x[2], zero, -x[0] * x[1]))
        else:
            for i in range(order + 1):
                p = x[1] ** i * x[2] ** (order - i)
                a.append((zero, x[0] * x[2] * p, -x[0] * x[1] * p))

                p = x[0] ** i * x[2] ** (order - i)
                a.append((x[1] * x[2] * p, zero, -x[0] * x[1] * p))

                p = x[0] ** i * x[1] ** (order - i)
                a.append((x[1] * x[2] * p, -x[0] * x[2] * p, zero))

        return [
            (
                r.diff(x[1]) - q.diff(x[2]),
                p.diff(x[2]) - r.diff(x[0]),
                q.diff(x[0]) - p.diff(x[1]),
            )
            for p, q, r in a
        ]

    raise NotImplementedError()


def Hcurl_serendipity(domain_dim, range_dim, order):
    """Hcurl conforming serendipity set."""
    assert domain_dim == range_dim
    if domain_dim == 2:
        return [
            ((order + 1) * x[0] ** order * x[1], -x[0] ** (order + 1)),
            (x[1] ** (order + 1), -(order + 1) * x[0] * x[1] ** order),
        ]
    if domain_dim == 3:
        out = []
        if order == 1:
            out += [
                (zero, x[0] * x[2], -x[0] * x[1]),
                (x[1] * x[2], zero, -x[0] * x[1]),
            ]
        else:
            for i in range(order):
                p = x[0] ** i * x[2] ** (order - 1 - i)
                out.append((x[1] * x[2] * p, zero, -x[0] * x[1] * p))

                p = x[1] ** i * x[2] ** (order - 1 - i)
                out.append((zero, x[0] * x[2] * p, -x[0] * x[1] * p))

                p = x[0] ** i * x[1] ** (order - 1 - i)
                out.append((x[1] * x[2] * p, -x[0] * x[2] * p, zero))

        for p in serendipity_set(domain_dim, 1, order + 1):
            out.append(tuple(p.diff(i) for i in x))
        return out

    raise NotImplementedError()


def prism_polynomial_set_1d(dim, order):
    """One dimensional polynomial set."""
    assert dim == 3
    return [
        x[0] ** i * x[1] ** j * x[2] ** k
        for k in range(order + 1)
        for j in range(order + 1)
        for i in range(order + 1 - j)
    ]


def prism_polynomial_set(domain_dim, range_dim, order):
    """Polynomial set for a prism."""
    if range_dim == 1:
        return prism_polynomial_set_1d(domain_dim, order)
    set1d = prism_polynomial_set_1d(domain_dim, order)
    return [
        tuple(p if i == j else zero for j in range(range_dim))
        for p in set1d
        for i in range(range_dim)
    ]


def pyramid_polynomial_set_1d(dim, order):
    """One dimensional polynomial set."""
    assert dim == 3
    if order == 0:
        return [one]

    poly = polynomial_set_1d(3, order)

    poly = [x[0] ** a * x[1] ** b * x[2] ** c / (1 - x[2]) ** (a + b + c - order)
            for c in range(order)
            for a in range(order + 1 - c) for b in range(order + 1 - c)]
    poly.append(x[2] ** order)

    return poly


def pyramid_polynomial_set(domain_dim, range_dim, order):
    """Polynomial set for a pyramid."""
    if range_dim == 1:
        return pyramid_polynomial_set_1d(domain_dim, order)
    set1d = pyramid_polynomial_set_1d(domain_dim, order)
    return [
        tuple(p if i == j else zero for j in range(range_dim))
        for p in set1d
        for i in range(range_dim)
    ]
