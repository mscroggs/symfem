"""Polynomial sets."""
from .symbolic import x, Monomial
import sympy
import numpy


def get_index(refname, indices, max_order):
    """Get the index of a monomial."""
    if refname == "interval":
        return indices[0]
    if refname == "triangle":
        return sum(indices[:2]) * (sum(indices[:2]) + 1) // 2 + indices[0]
    if refname == "quadrilateral":
        return indices[1] * (max_order + 1) + indices[0]
#    if refname == "tetrahedron":
#        return (sum(indices) * (sum(indices) + 1) * (sum(indices) + 2) // 6
#                + sum(indices[:2]) * (sum(indices[:2]) + 1) // 2 + indices[0])
    if refname == "hexahedron":
        return indices[2] * (max_order + 1) ** 2 + indices[1] * (max_order + 1) + indices[0]


def num_polynomials(refname, max_order):
    """Get the number of polynomials."""
    if refname == "interval":
        return max_order + 1
    if refname == "triangle":
        return (max_order + 1) * (max_order + 2) // 2
    if refname == "quadrilateral":
        return (max_order + 1) ** 2
#    if refname == "tetrahedron":
#        return (max_order + 1) * (max_order + 2) * (max_order + 3) // 6
    if refname == "hexahedron":
        return (max_order + 1) ** 3


def get_max_order(basis, refname):
    if refname in ["interval", "triangle", "tetrahedron"]:
        return max(sum(i.indices) for i in basis)
    if refname in ["quadrilateral", "hexahedron"]:
        return max(max(i.indices) for i in basis)


def get_min_order(basis, refname):
    if refname in ["interval", "triangle", "tetrahedron"]:
        return min(sum(i.indices) for i in basis)
    if refname in ["quadrilateral", "hexahedron"]:
        return min(max(i.indices) for i in basis)


def _evaluate_legendre_interval(points, max_order):
    legendre = numpy.empty((len(points), num_polynomials("interval", max_order)))
    legendre[:, 0] = 1
    if max_order > 0:
        legendre[:, 1] = 2 * points[:, 0] - 1
    for n in range(1, max_order):
        legendre[:, n + 1] = (2 * n + 1) * legendre[:, 1] * legendre[:, n]
        legendre[:, n + 1] -= n * legendre[:, n - 1]
        legendre[:, n + 1] /= n + 1
    return legendre


def _evaluate_legendre_triangle(points, max_order):
    legendre = numpy.empty((len(points), num_polynomials("triangle", max_order)))
    legendre[:, 0] = 1

    def ind(a, b):
        return get_index("triangle", (a, b), max_order)

    for p in range(1, max_order + 1):
        a = (2 * p - 1) / p
        legendre[:, ind(p, 0)] = (points[:, 0] * 2 + points[:, 1] - 1) * legendre[:, ind(p - 1, 0)] * a

        if p > 1:
            # y^2 terms
            legendre[:, ind(p, 0)] -= (1 - points[:, 1]) ** 2 * legendre[:, ind(p - 2, 0)] * (a - 1)

    for p in range(max_order):
        legendre[:, ind(p, 1)] = legendre[:, ind(p, 0)] * ((points[:, 1] * 2 - 1) * (1.5 + p) + 0.5 + p)
        for q in range(1, max_order - p):
            a1 = (p + q + 1) * (2 * p + 2 * q + 3) / (q + 1) / (2 * p + q + 2)
            a2 = (2 * p + 1) ** 2 * (p + q + 1) / (q + 1) / (2 * p + q + 2) / (2 * p + 2 * q + 1)
            a3 = q * (2 * p + q + 1) * (2 * p + 2 * q + 3) / (q + 1) / (2 * p + q + 2) / (2 * p + 2 * q + 1)

            legendre[:, ind(p, q + 1)] = legendre[:, ind(p, q)] * ((points[:, 1] * 2 - 1) * a1 + a2)
            legendre[:, ind(p, q + 1)] -= legendre[:, ind(p, q - 1)] * a3

    return legendre


def evaluate_legendre_basis(points, basis, reference):
    """Evaluate the Legendre basis spanning the same set as the given basis."""
    for i in basis:
        if not isinstance(i, Monomial):
            return None
    max_order = get_max_order(basis, reference.name)
    min_order = get_min_order(basis, reference.name)

    if min_order is None or max_order is None or min_order < 0:
            return None

    num_p = num_polynomials(reference.name, max_order)
    if num_p is None:
        return None

    if reference.name == "interval":
        legendre = _evaluate_legendre_interval(points, max_order)

    elif reference.name == "quadrilateral":
        ldims = [_evaluate_legendre_interval(points[:, i:i+1], max_order) for i in range(2)]
        legendre = numpy.empty((len(points), num_p))
        for i in range(max_order + 1):
            for j in range(max_order + 1):
                index = get_index(reference.name, (i, j, 0), max_order)
                legendre[:, index] = ldims[0][:, i] * ldims[1][:, j]

    elif reference.name == "hexahedron":
        ldims = [_evaluate_legendre_interval(points[:, i:i+1], max_order) for i in range(3)]
        legendre = numpy.empty((len(points), num_p))
        for i in range(max_order + 1):
            for j in range(max_order + 1):
                for k in range(max_order + 1):
                    index = get_index(reference.name, (i, j, k), max_order)
                    legendre[:, index] = ldims[0][:, i] * ldims[1][:, j] * ldims[2][:, k]

    elif reference.name == "triangle":
        legendre = _evaluate_legendre_triangle(points, max_order)

    else:
        ldims = [numpy.empty((len(points), num_polynomials("interval", max_order)))
                 for i in range(reference.tdim)]
        for i in range(reference.tdim):
            ldims[i][:, 0] = 1
            if max_order > 0:
                ldims[i][:, 1] = 2 * points[:, i] - 1
            for n in range(1, max_order):
                ldims[i][:, n + 1] = (2 * n + 1) * ldims[i][:, 1] * ldims[i][:, n]
                ldims[i][:, n + 1] -= n * ldims[i][:, n - 1]
                ldims[i][:, n + 1] /= n + 1

        if reference.tdim == 1:
            legendre = ldims[0]
        if reference.tdim == 2:
            legendre = numpy.empty((len(points), num_p))
            for o in range(max_order + 1):
                for i in range(o + 1):
                    index = get_index(reference.name, (i, o - i, 0), max_order)
                    legendre[:, index] = ldims[0][:, i] * ldims[1][:, o - i]

    polys = numpy.empty((len(points), len(basis)))
    for i, b in enumerate(basis):
        polys[:, i] = legendre[:, get_index(reference.name, b.indices, max_order)]

    return polys


def _get_legendre_interval(variable, max_order):
    legendre = [1]
    if max_order > 0:
        legendre.append(2 * variable - 1)
    for n in range(1, max_order):
        f = ((2 * n + 1) * legendre[1] * legendre[-1] - n * legendre[-2]) / (n + 1)
        legendre.append(f.expand())
    return legendre


def _get_legendre_triangle(variables, max_order):
    legendre = [0 for i in range(num_polynomials("triangle", max_order))]
    legendre[0] = 1
    ind = lambda a, b: get_index("triangle", (a, b), max_order)

    for p in range(1, max_order + 1):
        a = sympy.Rational(2 * p - 1, p)
        legendre[ind(p, 0)] = (variables[0] * 2 + variables[1] - 1) * legendre[ind(p - 1, 0)] * a

        if p > 1:
            # y^2 terms
            legendre[ind(p, 0)] -= (1 - variables[1]) ** 2 * legendre[ind(p - 2, 0)] * (a - 1)
        legendre[ind(p, 0)] = legendre[ind(p, 0)].expand()

    for p in range(max_order):
        legendre[ind(p, 1)] = legendre[ind(p, 0)] * ((variables[1] * 2 - 1) * (3 + 2 * p) / 2 + sympy.Rational(1, 2) + p)
        legendre[ind(p, 1)] = legendre[ind(p, 1)].expand()
        for q in range(1, max_order - p):
            a1 = sympy.Rational((p + q + 1) * (2 * p + 2 * q + 3), (q + 1) * (2 * p + q + 2))
            a2 = sympy.Rational((2 * p + 1) ** 2 * (p + q + 1), (q + 1) * (2 * p + q + 2) * (2 * p + 2 * q + 1))
            a3 = sympy.Rational(q * (2 * p + q + 1) * (2 * p + 2 * q + 3), (q + 1) * (2 * p + q + 2) * (2 * p + 2 * q + 1))

            legendre[ind(p, q + 1)] = legendre[ind(p, q)] * ((variables[1] * 2 - 1) * a1 + a2)
            legendre[ind(p, q + 1)] -= legendre[ind(p, q - 1)] * a3
            legendre[ind(p, q + 1)] = legendre[ind(p, q + 1)].expand()

    return legendre


def get_legendre_basis(basis, reference):
    """Get the symbolic Legendre basis spanning the same set as the given basis."""
    for i, j in enumerate(basis):
        if isinstance(j, int):
            basis[i] = Monomial()
        elif not isinstance(j, Monomial):
            return None

    max_order = get_max_order(basis, reference.name)
    min_order = get_min_order(basis, reference.name)

    if min_order is None or max_order is None or min_order < 0:
        return None

    num_p = num_polynomials(reference.name, max_order)
    if num_p is None:
        return None

    if reference.name == "interval":
        legendre = _get_legendre_interval(x[0], max_order)

    elif reference.name == "quadrilateral":
        ldims = [_get_legendre_interval(x[i], max_order) for i in range(2)]
        legendre = []
        for j in range(max_order + 1):
            for i in range(max_order + 1):
                assert len(legendre) == get_index(reference.name, (i, j, 0), max_order)
                legendre.append(ldims[0][i] * ldims[1][j])

    elif reference.name == "hexahedron":
        ldims = [_get_legendre_interval(x[i], max_order) for i in range(3)]
        legendre = []
        for k in range(max_order + 1):
            for j in range(max_order + 1):
                for i in range(max_order + 1):
                    assert len(legendre) == get_index(reference.name, (i, j, k), max_order)
                    legendre.append(ldims[0][i] * ldims[1][j] * ldims[2][k])

    elif reference.name == "triangle":
        legendre = _get_legendre_triangle(x, max_order)

    else:
        ldims = [[] for i in range(reference.tdim)]
        for i in range(reference.tdim):
            ldims[i].append(1)
            if max_order > 0:
                ldims[i].append(2 * x[i] - 1)
            for n in range(1, max_order):
                f = ((2 * n + 1) * ldims[i][1] * ldims[i][-1] - n * ldims[i][-2]) / (n + 1)
                ldims[i].append(f.expand())

        if reference.tdim == 1:
            legendre = ldims[0]
        if reference.tdim == 2:
            legendre = []
            for o in range(max_order + 1):
                for i in range(o + 1):
                    legendre.append(ldims[0][i] * ldims[1][o - i])

    polys = []
    for i, b in enumerate(basis):
        polys.append(legendre[get_index(reference.name, b.indices, max_order)])

    return polys
