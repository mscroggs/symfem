"""Polynomial sets."""
import sympy
from .symbolic import x
from .calculus import curl, diff
from itertools import product


def polynomial_set_1d(dim, order, variables=x):
    """One dimensional polynomial set."""
    if dim == 1:
        return [variables[0] ** i for i in range(order + 1)]
    if dim == 2:
        return [
            variables[0] ** i * variables[1] ** j
            for j in range(order + 1)
            for i in range(order + 1 - j)
        ]
    if dim == 3:
        return [
            variables[0] ** i * variables[1] ** j * variables[2] ** k
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
        tuple(p if i == j else 0 for j in range(range_dim))
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
                  0,
                  -x[0] ** m * x[1] ** n * x[2] ** (order - m - n)]
                 for n in range(order) for m in range(1, order + 1 - n)]
        poly += [[0,
                  x[0] ** m * x[1] ** (n - 1) * x[2] ** (order - m - n + 1),
                  -x[0] ** m * x[1] ** n * x[2] ** (order - m - n)]
                 for m in range(order) for n in range(1, order + 1 - m)]
        poly += [[x[0] ** (order - n) * x[1] ** n,
                  -x[0] ** (order + 1 - n) * x[1] ** (n - 1),
                  0]
                 for n in range(1, order + 1)]
        return poly


def quolynomial_set_1d(dim, order):
    """One dimensional quolynomial set."""
    basis = []
    for j in product(range(order + 1), repeat=dim):
        poly = sympy.Integer(1)
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
        tuple(p if i == j else 0 for j in range(range_dim))
        for p in set1d
        for i in range(range_dim)
    ]


def Hdiv_quolynomials(domain_dim, range_dim, order):
    """Hdiv conforming quolynomial set."""
    assert domain_dim == range_dim
    basis = []
    for d in range(domain_dim):
        for j in product(range(order), repeat=domain_dim - 1):
            poly = 1
            for a, b in zip(x, j[:d] + (order,) + j[d:]):
                poly *= a ** b
            basis.append(tuple(poly if i == d else 0 for i in range(domain_dim)))
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
            poly = 1
            for a, b in zip(x, j):
                poly *= a ** b
            basis.append(tuple(poly if i == d else 0 for i in range(domain_dim)))
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
            p = 1
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
        tuple(p if i == j else 0 for j in range(range_dim))
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
            a.append((0, x[0] * x[2], -x[0] * x[1]))
            a.append((x[1] * x[2], 0, -x[0] * x[1]))
        else:
            for i in range(order + 1):
                p = x[1] ** i * x[2] ** (order - i)
                a.append((0, x[0] * x[2] * p, -x[0] * x[1] * p))

                p = x[0] ** i * x[2] ** (order - i)
                a.append((x[1] * x[2] * p, 0, -x[0] * x[1] * p))

                p = x[0] ** i * x[1] ** (order - i)
                a.append((x[1] * x[2] * p, -x[0] * x[2] * p, 0))

        return [curl(i) for i in a]

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
                (0, x[0] * x[2], -x[0] * x[1]),
                (x[1] * x[2], 0, -x[0] * x[1]),
            ]
        else:
            for i in range(order):
                p = x[0] ** i * x[2] ** (order - 1 - i)
                out.append((x[1] * x[2] * p, 0, -x[0] * x[1] * p))

                p = x[1] ** i * x[2] ** (order - 1 - i)
                out.append((0, x[0] * x[2] * p, -x[0] * x[1] * p))

                p = x[0] ** i * x[1] ** (order - 1 - i)
                out.append((x[1] * x[2] * p, -x[0] * x[2] * p, 0))

        for p in serendipity_set(domain_dim, 1, order + 1):
            out.append(tuple(diff(p, i) for i in x))
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
        tuple(p if i == j else 0 for j in range(range_dim))
        for p in set1d
        for i in range(range_dim)
    ]


def pyramid_polynomial_set_1d(dim, order):
    """One dimensional polynomial set."""
    assert dim == 3
    if order == 0:
        return [sympy.Integer(1)]

    poly = polynomial_set_1d(3, order)
    for d in range(order):
        for i in range(d + 1):
            for j in range(d + 1 - i):
                p = x[0] ** i * x[1] ** j * x[2] ** (d - i - j)
                p *= (x[0] * x[1] / (1 - x[2])) ** (order - d)
                poly.append(p)

    return poly


def pyramid_polynomial_set(domain_dim, range_dim, order):
    """Polynomial set for a pyramid."""
    if range_dim == 1:
        return pyramid_polynomial_set_1d(domain_dim, order)
    set1d = pyramid_polynomial_set_1d(domain_dim, order)
    return [
        tuple(p if i == j else 0 for j in range(range_dim))
        for p in set1d
        for i in range(range_dim)
    ]


def _jrc(a, n):
    """Get the Jacobi recurrence relation coefficients."""
    return (
        sympy.Rational((a + 2 * n + 1) * (a + 2 * n + 2), 2 * (n + 1) * (a + n + 1)),
        sympy.Rational(a * a * (a + 2 * n + 1), 2 * (n + 1) * (a + n + 1) * (a + 2 * n)),
        sympy.Rational(n * (a + n) * (a + 2 * n + 2), (n + 1) * (a + n + 1) * (a + 2 * n))
    )


def orthogonal_basis_interval(order, derivs, variables=None):
    """Create a basis of orthogonal polynomials."""
    if variables is None:
        variables = [x[0]]
    assert len(variables) == 1

    poly = [[None for i in range(order + 1)] for d in range(derivs + 1)]
    for dx in range(derivs + 1):
        poly[dx][0] = sympy.Integer(1 if dx == 0 else 0)
        for i in range(1, order + 1):
            poly[dx][i] = poly[dx][i - 1] * (2 * variables[0] - 1) * (2 * i - 1) / i
            if dx > 0:
                poly[dx][i] += poly[dx - 1][i - 1] * 2 * dx * (2 * i - 1) / i
            if i > 1:
                poly[dx][i] -= (i - 1) * poly[dx][i - 2] / i
    return poly


def orthogonal_basis_triangle(order, derivs, variables=None):
    """Create a basis of orthogonal polynomials."""
    if variables is None:
        variables = [x[0], x[1]]
    assert len(variables) == 2

    def index(p, q):
        return (p + q + 1) * (p + q) // 2 + q

    d_index = index

    poly = [[None for i in range((order + 1) * (order + 2) // 2)]
            for d in range((derivs + 1) * (derivs + 2) // 2)]

    for dx in range(derivs + 1):
        for dy in range(derivs + 1 - dx):

            for p in range(order + 1):
                if p == 0:
                    poly[d_index(dx, dy)][index(0, p)] = sympy.Integer(1 if dx == dy == 0 else 0)
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


def orthogonal_basis_quadrilateral(order, derivs, variables=None):
    """Create a basis of orthogonal polynomials."""
    if variables is None:
        variables = [x[0], x[1]]
    assert len(variables) == 2

    def d_index(p, q):
        return (p + q + 1) * (p + q) // 2 + q

    p0 = orthogonal_basis_interval(order, derivs, [variables[0]])
    p1 = orthogonal_basis_interval(order, derivs, [variables[1]])
    poly = [None for d in range((derivs + 1) * (derivs + 2) // 2)]
    for i in range(derivs + 1):
        for j in range(derivs + 1 - i):
            poly[d_index(i, j)] = [a * b for a in p0[i] for b in p1[j]]
    return poly


def orthogonal_basis_tetrahedron(order, derivs, variables=None):
    """Create a basis of orthogonal polynomials."""
    if variables is None:
        variables = x
    assert len(variables) == 3

    def index(p, q, r):
        return (p + q + r) * (p + q + r + 1) * (p + q + r + 2) // 6 + (q + r) * (q + r + 1) // 2 + r

    d_index = index

    poly = [[None for i in range((order + 1) * (order + 2) * (order + 3) // 6)]
            for d in range((derivs + 1) * (derivs + 2) * (derivs + 3) // 6)]
    for dx in range(derivs + 1):
        for dy in range(derivs + 1 - dx):
            for dz in range(derivs + 1 - dx - dy):

                for p in range(order + 1):
                    if p == 0:
                        poly[d_index(dx, dy, dz)][index(0, 0, p)] = sympy.Integer(
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


def orthogonal_basis_hexahedron(order, derivs, variables=None):
    """Create a basis of orthogonal polynomials."""
    if variables is None:
        variables = x
    assert len(variables) == 3

    def d_index(p, q, r):
        return (p + q + r) * (p + q + r + 1) * (p + q + r + 2) // 6 + (q + r) * (q + r + 1) // 2 + r

    p0 = orthogonal_basis_interval(order, derivs, [variables[0]])
    p1 = orthogonal_basis_interval(order, derivs, [variables[1]])
    p2 = orthogonal_basis_interval(order, derivs, [variables[2]])
    poly = [None for d in range((derivs + 1) * (derivs + 2) * (derivs + 3) // 6)]
    for i in range(derivs + 1):
        for j in range(derivs + 1 - i):
            for k in range(derivs + 1 - i - j):
                poly[d_index(i, j, k)] = [a * b * c for a in p0[i] for b in p1[j] for c in p2[k]]
    return poly


def orthogonal_basis_prism(order, derivs, variables=None):
    """Create a basis of orthogonal polynomials."""
    if variables is None:
        variables = x
    assert len(variables) == 3

    def d_index_tri(p, q):
        return (p + q + 1) * (p + q) // 2 + q

    def d_index(p, q, r):
        return (p + q + r) * (p + q + r + 1) * (p + q + r + 2) // 6 + (q + r) * (q + r + 1) // 2 + r

    p01 = orthogonal_basis_triangle(order, derivs, [variables[0], variables[1]])
    p2 = orthogonal_basis_interval(order, derivs, [variables[2]])
    poly = [None for d in range((derivs + 1) * (derivs + 2) * (derivs + 3) // 6)]
    for i in range(derivs + 1):
        for j in range(derivs + 1 - i):
            for k in range(derivs + 1 - i - j):
                poly[d_index(i, j, k)] = [a * b for a in p01[d_index_tri(i, j)] for b in p2[k]]
    return poly


def orthogonal_basis_pyramid(order, derivs, variables=None):
    """Create a basis of orthogonal polynomials."""
    if variables is None:
        variables = x
    assert len(variables) == 3

    def index(i, j, k):
        out = k + j * (order + 1) + i * (order + 1) * (order + 2) // 2 - i * (i ** 2 + 5) // 6
        if i > j:
            out -= i * (j - 1)
        else:
            out -= j * (j - 1) // 2 + i * (i - 1) // 2
        return out

    def d_index(p, q, r):
        return (p + q + r) * (p + q + r + 1) * (p + q + r + 2) // 6 + (q + r) * (q + r + 1) // 2 + r

    def combinations(n, k):
        out = 1
        for i in range(n, n - k, -1):
            out *= i
        return out

    poly = [[None for i in range((2 * order + 3) * (order + 2) * (order + 1) // 6)]
            for d in range((derivs + 1) * (derivs + 2) * (derivs + 3) // 6)]
    for dx in range(derivs + 1):
        for dy in range(derivs + 1 - dx):
            for dz in range(derivs + 1 - dx - dy):
                poly[d_index(dx, dy, dz)][0] = sympy.Integer(1 if dx == dy == dz == 0 else 0)

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


def orthogonal_basis(cell, order, derivs, variables=None):
    """Create a basis of orthogonal polynomials."""
    if cell == "interval":
        return orthogonal_basis_interval(order, derivs, variables)
    if cell == "triangle":
        return orthogonal_basis_triangle(order, derivs, variables)
    if cell == "quadrilateral":
        return orthogonal_basis_quadrilateral(order, derivs, variables)
    if cell == "hexahedron":
        return orthogonal_basis_hexahedron(order, derivs, variables)
    if cell == "tetrahedron":
        return orthogonal_basis_tetrahedron(order, derivs, variables)
    if cell == "prism":
        return orthogonal_basis_prism(order, derivs, variables)
    if cell == "pyramid":
        return orthogonal_basis_pyramid(order, derivs, variables)
