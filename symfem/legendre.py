"""Polynomial sets."""
from .symbolic import x, Monomial
import sympy
import numpy


def get_index(refname, indices, max_order):
    """Get the index of a monomial."""
    if refname == "interval":
        return indices[0]
    if refname == "triangle":
        return sum(indices) * (sum(indices) + 1) // 2 + indices[1]
    if refname == "quadrilateral":
        return indices[1] * (max_order + 1) + indices[0]
    if refname == "tetrahedron":
        return (sum(indices) * (sum(indices) + 1) * (sum(indices) + 2) // 6
                + sum(indices[1:]) * (sum(indices[1:]) + 1) // 2 + indices[2])
    if refname == "hexahedron":
        return indices[2] * (max_order + 1) ** 2 + indices[1] * (max_order + 1) + indices[0]
    if refname == "prism":
        return ((max_order + 2) * (max_order + 1) * indices[2] // 2
                + sum(indices[:2]) * (sum(indices[:2]) + 1) // 2 + indices[1])


def num_polynomials(refname, max_order):
    """Get the number of polynomials."""
    if refname == "interval":
        return max_order + 1
    if refname == "triangle":
        return (max_order + 1) * (max_order + 2) // 2
    if refname == "quadrilateral":
        return (max_order + 1) ** 2
    if refname == "tetrahedron":
        return (max_order + 1) * (max_order + 2) * (max_order + 3) // 6
    if refname == "hexahedron":
        return (max_order + 1) ** 3
    if refname == "prism":
        return (max_order + 1) ** 2 * (max_order + 2) // 2


def get_max_order(basis, refname):
    """Get the maximum order of a basis."""
    if refname in ["interval", "triangle", "tetrahedron"]:
        return max(sum(i.indices) for i in basis)
    if refname in ["quadrilateral", "hexahedron"]:
        return max(max(i.indices) for i in basis)
    if refname == "prism":
        return max(max(sum(i.indices[:2]), i.indices[2]) for i in basis)


def get_min_order(basis, refname):
    """Get the minimum order of a basis."""
    if refname in ["interval", "triangle", "tetrahedron"]:
        return min(sum(i.indices) for i in basis)
    if refname in ["quadrilateral", "hexahedron"]:
        return min(max(i.indices) for i in basis)
    if refname == "prism":
        return min(max(sum(i.indices[:2]), i.indices[2]) for i in basis)


def _jrc(a, n, divide):
    """Coefficients in Jacobi polynomial recurrence relation."""
    return (
        divide((a + 2 * n + 1) * (a + 2 * n + 2), 2 * (n + 1) * (a + n + 1)),
        divide(a * a * (a + 2 * n + 1), 2 * (n + 1) * (a + n + 1) * (a + 2 * n)),
        divide(n * (a + n) * (a + 2 * n + 2), (n + 1) * (a + n + 1) * (a + 2 * n))
    )


def _legendre_interval(max_order, pts, leg, set_leg, divide):
    """Compute Legendre polynomials on an interval."""
    set_leg(0, 1)
    if max_order > 0:
        set_leg(1, 2 * pts(0) - 1)
    for n in range(1, max_order):
        set_leg(n + 1, ((2 * n + 1) * leg(1) * leg(n) - n * leg(n - 1)) / (n + 1))


def _legendre_quadrilateral(max_order, pts, leg, set_leg, divide):
    """Compute Legendre polynomials on a quadrilateral."""

    def ind(a, b):
        return get_index("quadrilateral", (a, b), max_order)

    def pts1(i):
        return pts(1 + i)

    def leg1(i):
        return leg(ind(0, i))

    def set_leg1(i, value):
        set_leg(ind(0, i), value)

    _legendre_interval(max_order, pts, leg, set_leg, divide)
    _legendre_interval(max_order, pts1, leg1, set_leg1, divide)

    for i in range(1, max_order + 1):
        for j in range(1, max_order + 1):
            set_leg(ind(i, j), leg(ind(i, 0)) * leg(ind(0, j)))


def _legendre_hexahedron(max_order, pts, leg, set_leg, divide):
    """Compute Legendre polynomials on a hexahedron."""

    def ind(a, b, c):
        return get_index("hexahedron", (a, b, c), max_order)

    def pts1(i):
        return pts(1 + i)

    def leg1(i):
        return leg(ind(0, i, 0))

    def set_leg1(i, value):
        set_leg(ind(0, i, 0), value)

    def pts2(i):
        return pts(2 + i)

    def leg2(i):
        return leg(ind(0, 0, i))

    def set_leg2(i, value):
        set_leg(ind(0, 0, i), value)

    _legendre_interval(max_order, pts, leg, set_leg, divide)
    _legendre_interval(max_order, pts1, leg1, set_leg1, divide)
    _legendre_interval(max_order, pts2, leg2, set_leg2, divide)

    for i in range(max_order + 1):
        for j in range(max_order + 1):
            for k in range(max_order + 1):
                set_leg(ind(i, j, k), leg(ind(i, 0, 0)) * leg(ind(0, j, 0)) * leg(ind(0, 0, k)))


def _legendre_prism(max_order, pts, leg, set_leg, divide):
    """Compute Legendre polynomials on a prism."""

    def ind(a, b, c):
        return get_index("prism", (a, b, c), max_order)

    def pts2(i):
        return pts(2 + i)

    def leg2(i):
        return leg(ind(0, 0, i))

    def set_leg2(i, value):
        set_leg(ind(0, 0, i), value)

    _legendre_triangle(max_order, pts, leg, set_leg, divide)
    _legendre_interval(max_order, pts2, leg2, set_leg2, divide)

    for i in range(max_order + 1):
        for j in range(max_order - i + 1):
            for k in range(max_order + 1):
                set_leg(ind(i, j, k), leg(ind(i, j, 0)) * leg(ind(0, 0, k)))


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
            a1, a2, a3 = _jrc(2 * p + 1, q, divide)
            set_leg(ind(p, q + 1),
                    leg(ind(p, q)) * ((pts(1) * 2 - 1) * a1 + a2) - leg(ind(p, q - 1)) * a3)


def _legendre_tetrahedron(max_order, pts, leg, set_leg, divide):
    """Compute Legendre polynomials on a tetrahedron."""

    def ind(a, b, c):
        return get_index("tetrahedron", (a, b, c), max_order)

    set_leg(0, 1)

    for p in range(1, max_order + 1):
        a = divide(2 * p - 1, p)
        if p > 1:
            set_leg(ind(p, 0, 0),
                    (pts(0) * 2 + pts(1) + pts(2) - 1) * leg(ind(p - 1, 0, 0)) * a
                    - (pts(1) + pts(2) - 1) ** 2 * leg(ind(p - 2, 0, 0)) * (a - 1))
        else:
            set_leg(ind(p, 0, 0),
                    (pts(0) * 2 + pts(1) + pts(2) - 1) * leg(ind(p - 1, 0, 0)) * a)

    for p in range(max_order):
        set_leg(ind(p, 1, 0),
                leg(ind(p, 0, 0)) * (pts(1) * (3 + 2 * p) + pts(2) - 1))

        for q in range(1, max_order - p):
            a1, a2, a3 = _jrc(2 * p + 1, q, divide)
            set_leg(ind(p, q + 1, 0),
                    leg(ind(p, q, 0)) * (pts(1) * 2 * a1 + (pts(2) - 1) * (a1 - a2))
                    - leg(ind(p, q - 1, 0)) * (1 - pts(2)) ** 2 * a3)

    for p in range(max_order):
        for q in range(max_order - p):
            set_leg(ind(p, q, 1),
                    leg(ind(p, q, 0)) * ((1 + p + q) + (pts(2) * 2 - 1) * (2 + p + q)))

    for p in range(max_order - 1):
        for q in range(max_order - p - 1):
            for r in range(1, max_order - p - q):
                a1, a2, a3 = _jrc(2 * p + 2 * q + 2, r, divide)
                set_leg(ind(p, q, r + 1),
                        leg(ind(p, q, r)) * ((pts(2) * 2 - 1) * a1 + a2)
                        - leg(ind(p, q, r - 1)) * a3)


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

    legendre = numpy.empty((len(points), num_polynomials(reference.name, max_order)))

    def pts(i):
        return points[:, i]

    def leg(i):
        return legendre[:, i]

    def set_leg(i, value):
        legendre[:, i] = value

    def divide(a, b):
        return a / b

    if reference.name == "interval":
        _legendre_interval(max_order, pts, leg, set_leg, divide)
    elif reference.name == "triangle":
        _legendre_triangle(max_order, pts, leg, set_leg, divide)
    elif reference.name == "tetrahedron":
        _legendre_tetrahedron(max_order, pts, leg, set_leg, divide)
    elif reference.name == "quadrilateral":
        _legendre_quadrilateral(max_order, pts, leg, set_leg, divide)
    elif reference.name == "hexahedron":
        _legendre_hexahedron(max_order, pts, leg, set_leg, divide)
    elif reference.name == "prism":
        _legendre_prism(max_order, pts, leg, set_leg, divide)

    if len(basis) == legendre.shape[1]:
        return legendre

    polys = numpy.empty((len(points), len(basis)))
    for i, b in enumerate(basis):
        polys[:, i] = legendre[:, get_index(reference.name, b.indices, max_order)]

    return polys


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

    legendre = [0 for i in range(num_polynomials(reference.name, max_order))]

    def pts(i):
        return x[i]

    def leg(i):
        return legendre[i]

    def set_leg(i, value):
        legendre[i] = value

    def divide(a, b):
        return sympy.Rational(a, b)

    if reference.name == "interval":
        _legendre_interval(max_order, pts, leg, set_leg, divide)
    elif reference.name == "triangle":
        _legendre_triangle(max_order, pts, leg, set_leg, divide)
    elif reference.name == "tetrahedron":
        _legendre_tetrahedron(max_order, pts, leg, set_leg, divide)
    elif reference.name == "quadrilateral":
        _legendre_quadrilateral(max_order, pts, leg, set_leg, divide)
    elif reference.name == "hexahedron":
        _legendre_hexahedron(max_order, pts, leg, set_leg, divide)
    elif reference.name == "prism":
        _legendre_prism(max_order, pts, leg, set_leg, divide)

    if len(basis) == len(legendre):
        return legendre

    polys = []
    for i, b in enumerate(basis):
        polys.append(legendre[get_index(reference.name, b.indices, max_order)])

    return polys
