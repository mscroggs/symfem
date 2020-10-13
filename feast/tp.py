"""Elements on tensor product cells."""

import sympy
from itertools import product
from .symbolic import one, zero
from .simplex import Lagrange, VectorLagrange
from .finite_element import FiniteElement, make_integral_moment_dofs
from .polynomials import (
    quolynomial_set,
    Hdiv_quolynomials,
    Hcurl_quolynomials,
    serendipity_set,
    polynomial_set,
    Hdiv_serendipity,
    Hcurl_serendipity,
)
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
        scalar_space = Q(reference, order)
        dofs = []
        directions = [
            tuple(one if i == j else zero for j in range(reference.tdim))
            for i in range(reference.tdim)
        ]
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


class Serendipity(FiniteElement):
    """A serendipity element."""

    def __init__(self, reference, order):
        poly = polynomial_set(reference.tdim, 1, order)
        poly += serendipity_set(reference.tdim, 1, order)

        dofs = []
        for p in reference.vertices:
            dofs.append(PointEvaluation(p))
        dofs += make_integral_moment_dofs(
            reference,
            edges=(IntegralMoment, Lagrange, order - 2, 0),
            faces=(IntegralMoment, Lagrange, order - 4, 0),
            volumes=(IntegralMoment, Lagrange, order - 6, 0),
        )

        super().__init__(poly, dofs, reference.tdim, 1)

    names = ["serendipity", "S"]


class SerendipityCurl(FiniteElement):
    """A serendipity Hcurl element."""

    def __init__(self, reference, order):
        poly = polynomial_set(reference.tdim, reference.tdim, order)
        poly += Hcurl_serendipity(reference.tdim, reference.tdim, order)

        dofs = []
        dofs += make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, Lagrange, order, 0),
            faces=(IntegralMoment, VectorLagrange, order - 2, 0),
            volumes=(IntegralMoment, VectorLagrange, order - 4, 0),
        )

        super().__init__(poly, dofs, reference.tdim, reference.tdim)

    names = ["serendipity Hcurl", "Scurl", "BDMCE", "AAE"]


class SerendipityDiv(FiniteElement):
    """A serendipity Hdiv element."""

    def __init__(self, reference, order):
        poly = polynomial_set(reference.tdim, reference.tdim, order)
        poly += Hdiv_serendipity(reference.tdim, reference.tdim, order)

        dofs = []
        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, VectorLagrange, order, 0),
            cells=(IntegralMoment, VectorLagrange, order - 2, 0),
        )

        super().__init__(poly, dofs, reference.tdim, reference.tdim)

    names = ["serendipity Hdiv", "Sdiv", "BDMCF", "AAF"]
