"""Regge elements on simplices."""

import sympy
from itertools import product
from ..core.finite_element import FiniteElement
from ..core.polynomials import polynomial_set
from ..core.functionals import PointInnerProduct


class Regge(FiniteElement):
    """A Regge element."""

    def __init__(self, reference, order):
        assert reference.name in ["triangle", "tetrahedron"]
        if reference.tdim == 2:
            poly = [(p[0], p[1], p[1], p[2])
                    for p in polynomial_set(reference.tdim, 3, order)]
        if reference.tdim == 3:
            poly = [(p[0], p[1], p[3], p[1], p[2], p[4], p[3], p[4], p[5])
                    for p in polynomial_set(reference.tdim, 6, order)]

        dofs = []
        for edim in range(1, 4):
            for vs in reference.sub_entities(edim):
                entity = reference.sub_entity_types[edim](
                    vertices=tuple(reference.reference_vertices[i] for i in vs)
                )
                for i in product(range(1, order + 2), repeat=edim):
                    if sum(i) < order + 2:
                        for edge in entity.edges[::-1]:
                            tangent = [b - a for a, b in zip(entity.vertices[edge[0]],
                                                             entity.vertices[edge[1]])]
                            dofs.append(PointInnerProduct(
                                tuple(o + sum(sympy.Rational(a[j] * b, order + 2)
                                              for a, b in zip(entity.axes, i[::-1]))
                                      for j, o in enumerate(entity.origin)),
                                tangent, entity_dim=edim))

        super().__init__(reference, poly, dofs, reference.tdim, reference.tdim ** 2)

    names = ["Regge"]
    min_order = 0
