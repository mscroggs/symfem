"""Crouzeix-Raviart elements on simplices.

This element's definition appears in https://doi.org/10.1051/m2an/197307R300331
(Crouzeix, Raviart, 1973)
"""

from itertools import product
from ..finite_element import CiarletElement
from ..polynomials import polynomial_set
from ..functionals import PointEvaluation
from ..quadrature import get_quadrature


class CrouzeixRaviart(CiarletElement):
    """Crouzeix-Raviart finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        from symfem import create_reference
        assert reference.name in ["triangle", "tetrahedron"]

        if order > 1:
            assert reference.name == "triangle"

        points, _ = get_quadrature(variant, order + reference.tdim)

        dofs = []

        for e_n, vs in enumerate(reference.sub_entities(reference.tdim - 1)):
            entity = create_reference(
                reference.sub_entity_types[reference.tdim - 1],
                vertices=tuple(reference.reference_vertices[i] for i in vs))
            for i in product(range(1, order + 1), repeat=reference.tdim - 1):
                if sum(i) < order + reference.tdim - 1:
                    dofs.append(
                        PointEvaluation(
                            tuple(o + sum(a[j] * points[b]
                                          for a, b in zip(entity.axes, i))
                                  for j, o in enumerate(entity.origin)),
                            entity=(reference.tdim - 1, e_n)))

        points, _ = get_quadrature(variant, order + reference.tdim - 1)
        for i in product(range(1, order), repeat=reference.tdim):
            if sum(i) < order:
                dofs.append(
                    PointEvaluation(
                        tuple(o + sum(a[j] * points[b]
                                      for a, b in zip(reference.axes, i))
                              for j, o in enumerate(reference.origin)),
                        entity=(reference.tdim, 0)))

        poly = polynomial_set(reference.tdim, 1, order)
        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Crouzeix-Raviart", "CR", "Crouzeix-Falk", "CF"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    max_order = {"tetrahedron": 1}
    continuity = "L2"
