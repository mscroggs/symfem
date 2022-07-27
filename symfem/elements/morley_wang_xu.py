"""Morley-Wang-Xu elements on simplices.

This element's definition appears in https://doi.org/10.1090/S0025-5718-2012-02611-1
(Wang, Xu, 2013)
"""

import typing

from ..finite_element import CiarletElement
from ..functionals import (IntegralAgainst, IntegralOfDirectionalMultiderivative, ListOfFunctionals,
                           PointEvaluation)
from ..functions import FunctionInput
from ..polynomials import polynomial_set_1d
from ..references import Reference


class MorleyWangXu(CiarletElement):
    """Morley-Wang-Xu finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        assert order <= reference.tdim
        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, order)

        dofs: ListOfFunctionals = []

        if order == 1:
            if reference.tdim == 1:
                for v_n, v in enumerate(reference.vertices):
                    dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))
            else:
                dim = reference.tdim - 1
                for facet_n in range(reference.sub_entity_count(dim)):
                    facet = reference.sub_entity(dim, facet_n)
                    dofs.append(IntegralAgainst(reference, facet, 1 / facet.jacobian(),
                                                entity=(dim, facet_n)))
        elif order == 2:
            if reference.tdim == 2:
                for v_n, v in enumerate(reference.vertices):
                    dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))
            else:
                dim = reference.tdim - 2
                for ridge_n in range(reference.sub_entity_count(dim)):
                    ridge = reference.sub_entity(dim, ridge_n)
                    dofs.append(IntegralAgainst(reference, ridge, 1 / ridge.jacobian(),
                                                entity=(dim, ridge_n)))
            dim = reference.tdim - 1
            for facet_n in range(reference.sub_entity_count(dim)):
                facet = reference.sub_entity(dim, facet_n)
                dofs.append(IntegralOfDirectionalMultiderivative(
                    reference, facet, (facet.normal(), ), (1, ), (dim, facet_n),
                    scale=1 / facet.jacobian()))
        else:
            assert order == reference.tdim == 3
            for v_n, v in enumerate(reference.vertices):
                dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))
            for e_n, vs in enumerate(reference.sub_entities(1)):
                subentity = reference.sub_entity(1, e_n)
                volume = subentity.jacobian()
                normals = []
                for f_n, f_vs in enumerate(reference.sub_entities(2)):
                    if vs[0] in f_vs and vs[1] in f_vs:
                        face = reference.sub_entity(2, f_n)
                        normals.append(face.normal())
                for orders in [(1, 0), (0, 1)]:
                    dofs.append(IntegralOfDirectionalMultiderivative(
                        reference, subentity, tuple(normals), orders, (1, e_n),
                        scale=1 / volume))
            for f_n, vs in enumerate(reference.sub_entities(2)):
                subentity = reference.sub_entity(2, f_n)
                volume = subentity.jacobian()
                dofs.append(IntegralOfDirectionalMultiderivative(
                    reference, subentity, (subentity.normal(), ), (2, ), (2, f_n),
                    scale=1 / volume))

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Morley-Wang-Xu", "MWX"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = 1
    max_order = {"interval": 1, "triangle": 2, "tetrahedron": 3}
    continuity = "C{order}"
