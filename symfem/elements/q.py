"""Q elements on tensor product cells."""

import sympy
from itertools import product
from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..polynomials import quolynomial_set, Hdiv_quolynomials, Hcurl_quolynomials
from ..quadrature import get_quadrature
from ..functionals import (PointEvaluation, DotPointEvaluation, IntegralMoment,
                           TangentIntegralMoment, NormalIntegralMoment)


class Q(CiarletElement):
    """A Q element."""

    def __init__(self, reference, order, variant="equispaced"):
        from symfem import create_reference

        if order == 0:
            dofs = [
                PointEvaluation(
                    tuple(sympy.Rational(1, 2) for i in range(reference.tdim)),
                    entity=(reference.tdim, 0))]
        else:
            points, _ = get_quadrature(variant, order + 1)

            dofs = []
            for v_n, v in enumerate(reference.reference_vertices):
                dofs.append(PointEvaluation(v, entity=(0, v_n)))
            for edim in range(1, 4):
                for e_n, vs in enumerate(reference.sub_entities(edim)):
                    entity = create_reference(
                        reference.sub_entity_types[edim],
                        vertices=tuple(reference.reference_vertices[i] for i in vs))
                    for i in product(range(1, order), repeat=edim):
                        dofs.append(
                            PointEvaluation(
                                tuple(o + sum(a[j] * points[b]
                                              for a, b in zip(entity.axes, i[::-1]))
                                      for j, o in enumerate(entity.origin)),
                                entity=(edim, e_n)))

        super().__init__(
            reference, order,
            quolynomial_set(reference.tdim, 1, order),
            dofs,
            reference.tdim,
            1)

    names = ["Q", "Lagrange", "P"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "C0"


class DiscontinuousQ(CiarletElement):
    """A dQ element."""

    def __init__(self, reference, order, variant="equispaced"):
        if order == 0:
            dofs = [
                PointEvaluation(
                    tuple(sympy.Rational(1, 2) for i in range(reference.tdim)),
                    entity=(reference.tdim, 0))]
        else:
            points, _ = get_quadrature(variant, order + 1)

            dofs = []
            for i in product(range(order + 1), repeat=reference.tdim):
                dofs.append(PointEvaluation(tuple(points[j] for j in i[::-1]),
                                            entity=(reference.tdim, 0)))

        super().__init__(
            reference, order,
            quolynomial_set(reference.tdim, 1, order),
            dofs,
            reference.tdim,
            1)

    names = ["dQ"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "L2"


class VectorQ(CiarletElement):
    """A vector Q element."""

    def __init__(self, reference, order, variant="equispaced"):
        scalar_space = Q(reference, order, variant)
        dofs = []
        if reference.tdim == 1:
            directions = [1]
        else:
            directions = [
                tuple(1 if i == j else 0 for j in range(reference.tdim))
                for i in range(reference.tdim)
            ]
        for p in scalar_space.dofs:
            for d in directions:
                dofs.append(DotPointEvaluation(p.point, d, entity=p.entity))

        super().__init__(
            reference, order,
            quolynomial_set(reference.tdim, reference.tdim, order),
            dofs,
            reference.tdim,
            reference.tdim,
        )

    names = ["vector Q", "vQ"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "C0"


class Nedelec(CiarletElement):
    """Nedelec Hcurl finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        poly = quolynomial_set(reference.tdim, reference.tdim, order - 1)
        poly += Hcurl_quolynomials(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, DiscontinuousQ, order - 1, {"variant": variant}),
            faces=(IntegralMoment, RaviartThomas, order - 1, "covariant", {"variant": variant}),
            volumes=(IntegralMoment, RaviartThomas, order - 1, "covariant", {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["NCE", "RTCE", "Qcurl", "Nedelec", "Ncurl"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(curl)"


class RaviartThomas(CiarletElement):
    """Raviart-Thomas Hdiv finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        poly = quolynomial_set(reference.tdim, reference.tdim, order - 1)
        poly += Hdiv_quolynomials(reference.tdim, reference.tdim, order)

        dofs = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DiscontinuousQ, order - 1, {"variant": variant}),
            cells=(IntegralMoment, Nedelec, order - 1, "contravariant", {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["NCF", "RTCF", "Qdiv"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(div)"
