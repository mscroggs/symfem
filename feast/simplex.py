"""Finite elements on simplices."""

import sympy
from itertools import product
from .finite_element import FiniteElement, make_integral_moment_dofs
from .polynomials import polynomial_set, Hcurl_polynomials, Hdiv_polynomials
from .functionals import (
    PointEvaluation,
    DotPointEvaluation,
    TangentIntegralMoment,
    NormalIntegralMoment,
    IntegralMoment,
)


class Lagrange(FiniteElement):
    """Lagrange finite element."""

    def __init__(self, reference, order):
        if order == 0:
            dofs = [
                PointEvaluation(
                    tuple(
                        sympy.Rational(1, reference.tdim + 1)
                        for i in range(reference.tdim)
                    )
                )
            ]
        else:
            dofs = []
            for i in product(range(order + 1), repeat=reference.tdim):
                if sum(i) <= order:
                    dofs.append(
                        PointEvaluation(tuple(sympy.Rational(j, order) for j in i))
                    )

        super().__init__(
            polynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )

    names = ["Lagrange", "P"]


class VectorLagrange(FiniteElement):
    """Vector Lagrange finite element."""

    def __init__(self, reference, order):
        if reference.name == "interval":
            directions = [(1,)]
        elif reference.name == "triangle":
            directions = [(1, 0), (0, 1)]
        elif reference.name == "tetrahedron":
            directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        scalar_space = Lagrange(reference, order)
        dofs = []
        for p in scalar_space.dofs:
            for d in directions:
                dofs.append(DotPointEvaluation(p.point, d))

        super().__init__(
            polynomial_set(reference.tdim, reference.tdim, order),
            dofs,
            reference.tdim,
            reference.tdim,
        )

    names = ["vector Lagrange", "vP"]


class NedelecFirstKind(FiniteElement):
    """Nedelec first kind Hcurl finite element."""

    def __init__(self, reference, order):
        poly = polynomial_set(reference.tdim, reference.tdim, order - 1)
        poly += Hcurl_polynomials(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, Lagrange, order - 1, 0),
            faces=(IntegralMoment, VectorLagrange, order - 2, 0),
            volumes=(IntegralMoment, VectorLagrange, order - 3, 0),
        )

        super().__init__(poly, dofs, reference.tdim, reference.tdim)

    names = ["Nedelec", "Nedelec1", "N1curl"]


class RaviartThomas(FiniteElement):
    """Raviart-Thomas Hdiv finite element."""

    def __init__(self, reference, order):
        poly = polynomial_set(reference.tdim, reference.tdim, order - 1)
        poly += Hdiv_polynomials(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, Lagrange, order - 1, 0),
            cells=(IntegralMoment, VectorLagrange, order - 2, 0)
        )

        super().__init__(poly, dofs, reference.tdim, reference.tdim)

    names = ["Raviart-Thomas", "RT", "N1div"]
