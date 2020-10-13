"""Elements on tensor product cells."""

import sympy
from itertools import product
from .finite_element import FiniteElement, make_integral_moment_dofs
from .polynomials import quolynomial_set, Hdiv_quolynomials, Hcurl_quolynomials
from .functionals import (
    PointEvaluation,
    DotPointEvaluation,
    IntegralMoment,
    TangentIntegralMoment,
    NormalIntegralMoment,
)


class Q(FiniteElement):
    """A Q element."""

    def __init__(self, reference, order):
        if order == 0:
            dofs = [
                PointEvaluation(
                    tuple(sympy.Rational(1, 2) for i in range(reference.tdim))
                )
            ]
        else:
            dofs = []
            for i in product(range(order + 1), repeat=reference.tdim):
                dofs.append(PointEvaluation(tuple(sympy.Rational(j, order) for j in i)))

        super().__init__(
            quolynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )

    names = ["Q"]


class VectorQ(FiniteElement):
    """A vector Q element."""

    def __init__(self, reference, order):
        if reference.name == "interval":
            directions = [(1,)]
        elif reference.name == "quadrilateral":
            directions = [(1, 0), (0, 1)]
        elif reference.name == "hexahedron":
            directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        scalar_space = Q(reference, order)
        dofs = []
        for p in scalar_space.dofs:
            for d in directions:
                dofs.append(DotPointEvaluation(p.point, d))

        super().__init__(
            quolynomial_set(reference.tdim, reference.tdim, order),
            dofs,
            reference.tdim,
            reference.tdim,
        )

    names = ["vector Q", "vQ"]


class Nedelec(FiniteElement):
    """Nedelec Hcurl finite element."""

    def __init__(self, reference, order):
        poly = quolynomial_set(reference.tdim, reference.tdim, order - 1)
        poly += Hcurl_quolynomials(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, Q, order - 1, 0),
            faces=(IntegralMoment, RaviartThomas, order - 1, 1),
            volumes=(IntegralMoment, RaviartThomas, order - 1, 1),
        )

        super().__init__(poly, dofs, reference.tdim, reference.tdim)

    names = ["Nedelec", "NCE", "RTCE", "Qcurl"]


class RaviartThomas(FiniteElement):
    """Raviart-Thomas Hdiv finite element."""

    def __init__(self, reference, order):
        poly = quolynomial_set(reference.tdim, reference.tdim, order - 1)
        poly += Hdiv_quolynomials(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, Q, order - 1, 0),
            cells=(IntegralMoment, Nedelec, order - 1, 1),
        )

        super().__init__(poly, dofs, reference.tdim, reference.tdim)

    names = ["Raviart-Thomas", "NCF", "RTCF", "Qdiv"]
