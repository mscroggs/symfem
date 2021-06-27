"""Polynomial sets."""
from .symbolic import x, Monomial
import numpy


def get_index(tdim, indices):
    """Get the index of a monomial."""
    if tdim == 1:
        return indices[0]
    if tdim == 2:
        return sum(indices[:2]) * (sum(indices[:2]) + 1) // 2 + indices[0]
    if tdim == 3:
        return (sum(indices) * (sum(indices) + 1) * (sum(indices) + 2) // 6
                + sum(indices[:2]) * (sum(indices[:2]) + 1) // 2 + indices[0])


def num_polynomials(tdim, max_order):
    """Get the number of polynomials."""
    if tdim == 1:
        return max_order + 1
    if tdim == 2:
        return (max_order + 1) * (max_order + 2) // 2


def evaluate_legendre_basis(points, basis, reference):
    """Evaluate the Legendre basis spanning the same set as the given basis."""
    for i in basis:
        if not isinstance(i, Monomial):
            return None
    max_order = max(i.order for i in basis)
    min_order = min(i.order for i in basis)

    if min_order < 0:
        return None

    num_p = num_polynomials(reference.tdim, max_order)
    if num_p is None:
        return None

    ldims = [numpy.empty((len(points), num_polynomials(1, max_order)))
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
                index = get_index(reference.tdim, (i, o - i, 0))
                legendre[:, index] = ldims[0][:, i] * ldims[1][:, o - i]

    polys = numpy.empty((len(points), len(basis)))
    for i, b in enumerate(basis):
        polys[:, i] = legendre[:, get_index(reference.tdim, b.indices)]

    return polys


def get_legendre_basis(basis, reference):
    """Get the symbolic Legendre basis spanning the same set as the given basis."""
    for i, j in enumerate(basis):
        if isinstance(j, int):
            basis[i] = Monomial()
        elif not isinstance(j, Monomial):
            print("-->", j, type(j))
            return None

    max_order = max(i.order for i in basis)
    min_order = min(i.order for i in basis)

    if min_order < 0:
        return None

    num_p = num_polynomials(reference.tdim, max_order)
    if num_p is None:
        return None

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
        polys.append(legendre[get_index(reference.tdim, b.indices)])

    return polys
