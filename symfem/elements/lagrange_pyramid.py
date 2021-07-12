"""Lagrange elements on a pyramid.

This element's definition appears in https://doi.org/10.1007/s10915-009-9334-9
(Bergot, Cohen, Durufle, 2010)
"""

import sympy
from itertools import product
from ..finite_element import CiarletElement
from ..polynomials import pyramid_polynomial_set
from ..functionals import PointEvaluation
from ..quadrature import get_quadrature


class Lagrange(CiarletElement):
    """Lagrange finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        if order == 0:
            dofs = [
                PointEvaluation(
                    tuple(
                        sympy.Rational(1, reference.tdim + 1)
                        for i in range(reference.tdim)
                    ),
                    entity=(reference.tdim, 0)
                )
            ]
        else:
            points, _ = get_quadrature(variant, order + 1)

            dofs = []
            # Vertices
            for v_n, v in enumerate(reference.reference_vertices):
                dofs.append(PointEvaluation(v, entity=(0, v_n)))
            # Edges
            for e_n in range(reference.sub_entity_count(1)):
                entity = reference.sub_entity(1, e_n)
                for i in range(1, order):
                    dofs.append(
                        PointEvaluation(
                            tuple(o + entity.axes[0][j] * points[i]
                                  for j, o in enumerate(entity.origin)),
                            entity=(1, e_n)))
            # Faces
            for e_n in range(reference.sub_entity_count(2)):
                entity = reference.sub_entity(2, e_n)
                for i in product(range(1, order), repeat=2):
                    if len(entity.vertices) == 4 or sum(i) < order:
                        dofs.append(
                            PointEvaluation(
                                tuple(o + sum(a[j] * points[b]
                                              for a, b in zip(entity.axes, i[::-1]))
                                      for j, o in enumerate(entity.origin)),
                                entity=(2, e_n)))

            # Interior
            for i in product(range(1, order), repeat=3):
                if max(i[0], i[1]) + i[2] < order:
                    dofs.append(
                        PointEvaluation(
                            tuple(o + sum(a[j] * points[b]
                                          for a, b in zip(reference.axes, i))
                                  for j, o in enumerate(reference.origin)),
                            entity=(3, 0)))

        poly = pyramid_polynomial_set(reference.tdim, 1, order)

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Lagrange", "P"]
    references = ["pyramid"]
    min_order = 0
    continuity = "C0"
