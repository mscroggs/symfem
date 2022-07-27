"""Lagrange elements on a pyramid.

This element's definition appears in https://doi.org/10.1007/s10915-009-9334-9
(Bergot, Cohen, Durufle, 2010)
"""

import typing
from itertools import product

import sympy

from ..finite_element import CiarletElement
from ..functionals import ListOfFunctionals, PointEvaluation
from ..functions import FunctionInput
from ..polynomials import pyramid_polynomial_set_1d
from ..quadrature import get_quadrature
from ..references import Reference


class Lagrange(CiarletElement):
    """Lagrange finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        dofs: ListOfFunctionals = []
        if order == 0:
            dofs = [
                PointEvaluation(
                    reference, tuple(
                        sympy.Rational(1, reference.tdim + 1)
                        for i in range(reference.tdim)
                    ),
                    entity=(reference.tdim, 0)
                )
            ]
        else:
            points, _ = get_quadrature(variant, order + 1)

            # Vertices
            for v_n, v in enumerate(reference.vertices):
                dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))
            # Edges
            for e_n in range(reference.sub_entity_count(1)):
                entity = reference.sub_entity(1, e_n)
                for i in range(1, order):
                    dofs.append(
                        PointEvaluation(
                            reference, tuple(o + entity.axes[0][j] * points[i]
                                             for j, o in enumerate(entity.origin)),
                            entity=(1, e_n)))
            # Faces
            for e_n in range(reference.sub_entity_count(2)):
                entity = reference.sub_entity(2, e_n)
                for ii in product(range(1, order), repeat=2):
                    if len(entity.vertices) == 4 or sum(ii) < order:
                        dofs.append(
                            PointEvaluation(
                                reference, tuple(o + sum(a[j] * points[b]
                                                         for a, b in zip(entity.axes, ii[::-1]))
                                                 for j, o in enumerate(entity.origin)),
                                entity=(2, e_n)))

            # Interior
            for ii in product(range(1, order), repeat=3):
                if max(ii[0], ii[1]) + ii[2] < order:
                    dofs.append(
                        PointEvaluation(
                            reference, tuple(o + sum(a[j] * points[b]
                                                     for a, b in zip(reference.axes, ii))
                                             for j, o in enumerate(reference.origin)),
                            entity=(3, 0)))

        poly: typing.List[FunctionInput] = []
        poly += pyramid_polynomial_set_1d(reference.tdim, order)

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["Lagrange", "P"]
    references = ["pyramid"]
    min_order = 0
    continuity = "C0"
