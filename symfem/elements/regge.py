"""Regge elements on simplices.

This element's definition appears in https://doi.org/10.1007/BF02733251
(Regge, 1961) and https://doi.org/10.1007/s00211-011-0394-z
(Christiansen, 2011)
"""

import sympy
from itertools import product
from ..finite_element import CiarletElement
from ..polynomials import polynomial_set
from ..functionals import PointInnerProduct


class Regge(CiarletElement):
    """A Regge element."""

    def __init__(self, reference, order):
        from symfem import create_reference
        assert reference.name in ["triangle", "tetrahedron"]
        if reference.tdim == 2:
            poly = [(p[0], p[1], p[1], p[2])
                    for p in polynomial_set(reference.tdim, 3, order)]
        if reference.tdim == 3:
            poly = [(p[0], p[1], p[3], p[1], p[2], p[4], p[3], p[4], p[5])
                    for p in polynomial_set(reference.tdim, 6, order)]

        dofs = []
        for edim in range(1, 4):
            for e_n, vs in enumerate(reference.sub_entities(edim)):
                entity = create_reference(
                    reference.sub_entity_types[edim],
                    vertices=tuple(reference.reference_vertices[i] for i in vs))
                for i in product(range(1, order + 2), repeat=edim):
                    if sum(i) < order + 2:
                        for edge in entity.edges[::-1]:
                            tangent = [b - a for a, b in zip(entity.vertices[edge[0]],
                                                             entity.vertices[edge[1]])]
                            dofs.append(PointInnerProduct(
                                tuple(o + sum(sympy.Rational(a[j] * b, order + 2)
                                              for a, b in zip(entity.axes, i[::-1]))
                                      for j, o in enumerate(entity.origin)),
                                tangent, tangent, entity=(edim, e_n), mapping="double_covariant"))

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim ** 2,
                         (reference.tdim, reference.tdim))

    names = ["Regge"]
    references = ["triangle", "tetrahedron"]
    min_order = 0
    continuity = "inner H(curl)"
