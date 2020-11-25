"""Elements on tensor product cells."""

import sympy
from itertools import product
from .symbolic import one, zero
from .simplex import DiscontinuousLagrange, VectorDiscontinuousLagrange
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
                    tuple(sympy.Rational(1, 2) for i in range(reference.tdim)),
                    entity_dim=reference.tdim)]
        else:
            dofs = []
            for v in reference.reference_vertices:
                dofs.append(PointEvaluation(v, entity_dim=0))
            for edim in range(1, 4):
                for vs in reference.sub_entities(edim):
                    entity = reference.sub_entity_types[edim](
                        vertices=tuple(reference.reference_vertices[i] for i in vs)
                    )
                    for i in product(range(1, order), repeat=edim):
                        dofs.append(
                            PointEvaluation(
                                tuple(o + sum(sympy.Rational(a[j] * b, order)
                                              for a, b in zip(entity.axes, i))
                                      for j, o in enumerate(entity.origin)),
                                entity_dim=edim))

        super().__init__(
            reference,
            quolynomial_set(reference.tdim, 1, order),
            dofs,
            reference.tdim,
            1)

    names = ["Q"]
    min_order = 0


class DiscontinuousQ(FiniteElement):
    """A dQ element."""

    def __init__(self, reference, order):
        if order == 0:
            dofs = [
                PointEvaluation(
                    tuple(sympy.Rational(1, 2) for i in range(reference.tdim)),
                    entity_dim=reference.tdim)]
        else:
            dofs = []
            for i in product(range(order + 1), repeat=reference.tdim):
                dofs.append(PointEvaluation(tuple(sympy.Rational(j, order) for j in i),
                                            entity_dim=reference.tdim))

        super().__init__(
            reference,
            quolynomial_set(reference.tdim, 1, order),
            dofs,
            reference.tdim,
            1)

    names = ["dQ"]
    min_order = 0


class VectorQ(FiniteElement):
    """A vector Q element."""

    def __init__(self, reference, order):
        scalar_space = Q(reference, order)
        dofs = []
        if reference.tdim == 1:
            directions = [1]
        else:
            directions = [
                tuple(one if i == j else zero for j in range(reference.tdim))
                for i in range(reference.tdim)
            ]
        for p in scalar_space.dofs:
            for d in directions:
                dofs.append(DotPointEvaluation(p.point, d, entity_dim=p.entity_dim()))

        super().__init__(
            reference,
            quolynomial_set(reference.tdim, reference.tdim, order),
            dofs,
            reference.tdim,
            reference.tdim,
        )

    names = ["vector Q", "vQ"]
    min_order = 0


class Nedelec(FiniteElement):
    """Nedelec Hcurl finite element."""

    def __init__(self, reference, order):
        poly = quolynomial_set(reference.tdim, reference.tdim, order - 1)
        poly += Hcurl_quolynomials(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, DiscontinuousQ, order - 1),
            faces=(IntegralMoment, RaviartThomas, order - 1),
            volumes=(IntegralMoment, RaviartThomas, order - 1),
        )

        super().__init__(reference, poly, dofs, reference.tdim, reference.tdim)

    names = ["Nedelec", "NCE", "RTCE", "Qcurl"]
    min_order = 1


class RaviartThomas(FiniteElement):
    """Raviart-Thomas Hdiv finite element."""

    def __init__(self, reference, order):
        poly = quolynomial_set(reference.tdim, reference.tdim, order - 1)
        poly += Hdiv_quolynomials(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousQ, order - 1),
            cells=(IntegralMoment, Nedelec, order - 1),
        )

        super().__init__(reference, poly, dofs, reference.tdim, reference.tdim)

    names = ["Raviart-Thomas", "NCF", "RTCF", "Qdiv"]
    min_order = 1


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
            edges=(IntegralMoment, DiscontinuousLagrange, order - 2),
            faces=(IntegralMoment, DiscontinuousLagrange, order - 4),
            volumes=(IntegralMoment, DiscontinuousLagrange, order - 6),
        )

        super().__init__(reference, poly, dofs, reference.tdim, 1)

    names = ["serendipity", "S"]
    min_order = 1


class SerendipityCurl(FiniteElement):
    """A serendipity Hcurl element."""

    def __init__(self, reference, order):
        poly = polynomial_set(reference.tdim, reference.tdim, order)
        poly += Hcurl_serendipity(reference.tdim, reference.tdim, order)

        dofs = []
        dofs += make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, DiscontinuousLagrange, order),
            faces=(IntegralMoment, VectorDiscontinuousLagrange, order - 2),
            volumes=(IntegralMoment, VectorDiscontinuousLagrange, order - 4),
        )

        super().__init__(reference, poly, dofs, reference.tdim, reference.tdim)

    names = ["serendipity Hcurl", "Scurl", "BDMCE", "AAE"]
    min_order = 1


class SerendipityDiv(FiniteElement):
    """A serendipity Hdiv element."""

    def __init__(self, reference, order):
        poly = polynomial_set(reference.tdim, reference.tdim, order)
        poly += Hdiv_serendipity(reference.tdim, reference.tdim, order)

        dofs = []
        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousLagrange, order),
            cells=(IntegralMoment, VectorDiscontinuousLagrange, order - 2),
        )

        super().__init__(reference, poly, dofs, reference.tdim, reference.tdim)

    names = ["serendipity Hdiv", "Sdiv", "BDMCF", "AAF"]
    min_order = 1
