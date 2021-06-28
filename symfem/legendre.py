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
    """Get the maximum order of a basis."""
    if refname in ["interval", "triangle", "tetrahedron"]:
        return max(sum(i.indices) for i in basis)
    if refname in ["quadrilateral", "hexahedron"]:
        return max(max(i.indices) for i in basis)


def get_min_order(basis, refname):
    """Get the minimum order of a basis."""
    if refname in ["interval", "triangle", "tetrahedron"]:
        return min(sum(i.indices) for i in basis)
    if refname in ["quadrilateral", "hexahedron"]:
        return min(max(i.indices) for i in basis)


def _legendre_interval(max_order, pts, leg, set_leg, divide):
    """Compute Legendre polynomials on an interval."""
    set_leg(0, 1)
    if max_order > 0:
        set_leg(1, 2 * pts(0) - 1)
    for n in range(1, max_order):
        set_leg(n + 1, ((2 * n + 1) * leg(1) * leg(n) - n * leg(n - 1)) / (n + 1))


def _legendre_triangle(max_order, pts, leg, set_leg, divide):
    """Compute Legendre polynomials on a triangle."""

    def ind(a, b):
        return get_index("triangle", (a, b), max_order)

    set_leg(0, 1)

    for p in range(1, max_order + 1):
        a = divide(2 * p - 1, p)
        if p > 1:
            set_leg(ind(p, 0),
                    (pts(0) * 2 + pts(1) - 1) * leg(ind(p - 1, 0)) * a
                    - (1 - pts(1)) ** 2 * leg(ind(p - 2, 0)) * (a - 1))
        else:
            set_leg(ind(p, 0), (pts(0) * 2 + pts(1) - 1) * leg(ind(p - 1, 0)) * a)

    for p in range(max_order):
        set_leg(ind(p, 1),
                leg(ind(p, 0)) * ((pts(1) * 2 - 1) * (divide(3, 2) + p) + divide(1, 2) + p))
        for q in range(1, max_order - p):
            a1 = divide((p + q + 1) * (2 * p + 2 * q + 3),
                        (q + 1) * (2 * p + q + 2))
            a2 = divide((2 * p + 1) ** 2 * (p + q + 1),
                        (q + 1) * (2 * p + q + 2) * (2 * p + 2 * q + 1))
            a3 = divide(q * (2 * p + q + 1) * (2 * p + 2 * q + 3),
                        (q + 1) * (2 * p + q + 2) * (2 * p + 2 * q + 1))

            set_leg(ind(p, q + 1),
                    leg(ind(p, q)) * ((pts(1) * 2 - 1) * a1 + a2)
                    - leg(ind(p, q - 1)) * a3)


def _evaluate_legendre_interval(points, max_order):
    """Evaluate the Legendre polynomials (non-symbolically) at the given points in an interval."""
    legendre = numpy.empty((len(points), num_polynomials("interval", max_order)))

    def pts(i):
        return points[:, i]

    def leg(i):
        return legendre[:, i]

    def set_leg(i, value):
        legendre[:, i] = value

    def divide(a, b):
        return a / b

    _legendre_interval(max_order, pts, leg, set_leg, divide)
    return legendre


def _evaluate_legendre_triangle(points, max_order):
    """Evaluate the Legendre polynomials (non-symbolically) at the given points in a triangle."""
    legendre = numpy.empty((len(points), num_polynomials("triangle", max_order)))

    def pts(i):
        return points[:, i]

    def leg(i):
        return legendre[:, i]

    def set_leg(i, value):
        legendre[:, i] = value

    def divide(a, b):
        return a / b

    _legendre_triangle(max_order, pts, leg, set_leg, divide)

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
    """Get the Legendre polynomials for an interval."""
    legendre = [0 for i in range(num_polynomials("interval", max_order))]

    def pts(i):
        assert i == 0
        return variable

    def leg(i):
        return legendre[i]

    def set_leg(i, value):
        legendre[i] = value

    def divide(a, b):
        return sympy.Rational(a, b)

    _legendre_interval(max_order, pts, leg, set_leg, divide)

    return legendre


def _get_legendre_triangle(variables, max_order):
    """Get the Legendre polynomials for a triangle."""
    legendre = [0 for i in range(num_polynomials("triangle", max_order))]

    def pts(i):
        return variables[i]

    def leg(i):
        return legendre[i]

    def set_leg(i, value):
        legendre[i] = value

    def divide(a, b):
        return sympy.Rational(a, b)

    _legendre_triangle(max_order, pts, leg, set_leg, divide)

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
