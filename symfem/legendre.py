"""Polynomial sets."""
from .symbolic import x, Monomial
import numpy


def get_index(reference, indices):
    """Get the index of a monomial."""
    if reference.name == "interval":
        return indices[0]


def num_polynomials(reference, max_order):
    """Get the number of polynomials for a reference."""
    if reference.name == "interval":
        return max_order + 1


def evaluate_legendre_basis(points, basis, reference):
    """Evaluate the Legendre basis spanning the same set as the given basis."""
    for i in basis:
        if not isinstance(i, Monomial):
            return None
    max_order = max(i.order for i in basis)

    if reference.name == "interval":
        legendre = numpy.empty((len(points), num_polynomials(reference, max_order)))
        legendre[:, 0] = 1
        if max_order > 0:
            legendre[:, 1] = 2 * points[:, 0] - 1
        for n in range(1, max_order):
            legendre[:, n + 1] = (2 * n + 1) * legendre[:, 1] * legendre[:, n]
            legendre[:, n + 1] -= n * legendre[:, n - 1]
            legendre[:, n + 1] /= n + 1

        polys = numpy.empty((len(points), len(basis)))
        for i, b in enumerate(basis):
            polys[:, i] = legendre[:, get_index(reference, b.indices)]

        return polys


def get_legendre_basis(basis, reference):
    """Get the symbolic Legendre basis spanning the same set as the given basis."""
    for i in basis:
        if not isinstance(i, Monomial):
            return None
    max_order = max(i.order for i in basis)

    if reference.name == "interval":
        polys = []
        polys.append(1)
        if max_order > 0:
            polys.append(2 * x[0] - 1)
        for n in range(1, max_order):
            polys.append(((2 * n + 1) * polys[1] * polys[-1] - n * polys[-2]) / (n + 1))

        return polys
