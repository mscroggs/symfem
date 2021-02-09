"""Gauss-Lobatto-Legendre elements on simplices."""

from itertools import product
from ..core.finite_element import FiniteElement
from ..core.polynomials import quolynomial_set
from ..core.functionals import PointEvaluation
from ..core.quadrature import gll_points


class GaussLobattoLegendre(FiniteElement):
    """Gauss-Lobatto-Legendre finite element."""

    def __init__(self, reference, order):
        from symfem import create_reference
        gll = gll_points(order + 1)
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
                            tuple(o + sum(a[j] * gll[b]
                                          for a, b in zip(entity.axes, i))
                                  for j, o in enumerate(entity.origin)),
                            entity=(edim, e_n)))

        poly = quolynomial_set(reference.tdim, 1, order)
        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Gauss-Lobatto-Legendre", "GLL"]
    references = ["interval", "quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "C0"
    mapping = "identity"
