"""Crouzeix-Raviart elements on simplices.

This element's definition appears in https://doi.org/10.1051/m2an/197307R300331
(Crouzeix, Raviart, 1973)
"""

import typing
from itertools import product

from symfem.finite_element import CiarletElement
from symfem.functionals import ListOfFunctionals, PointEvaluation
from symfem.functions import FunctionInput
from symfem.polynomials import polynomial_set_1d
from symfem.quadrature import get_quadrature
from symfem.references import Reference

__all__ = ["CrouzeixRaviart"]


class CrouzeixRaviart(CiarletElement):
    """Crouzeix-Raviart finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        assert reference.name in ["triangle", "tetrahedron"]

        if order > 1:
            assert reference.name == "triangle"

        points, _ = get_quadrature(variant, order + reference.tdim)

        dofs: ListOfFunctionals = []

        for e_n in range(reference.sub_entity_count(reference.tdim - 1)):
            entity = reference.sub_entity(reference.tdim - 1, e_n)
            for i in product(range(1, order + 1), repeat=reference.tdim - 1):
                if sum(i) < order + reference.tdim - 1:
                    dofs.append(
                        PointEvaluation(
                            reference,
                            tuple(
                                o + sum(a[j] * points[b] for a, b in zip(entity.axes, i))
                                for j, o in enumerate(entity.origin)
                            ),
                            entity=(reference.tdim - 1, e_n),
                        )
                    )

        points, _ = get_quadrature(variant, order + reference.tdim - 1)
        for i in product(range(1, order), repeat=reference.tdim):
            if sum(i) < order:
                dofs.append(
                    PointEvaluation(
                        reference,
                        tuple(
                            o + sum(a[j] * points[b] for a, b in zip(reference.axes, i))
                            for j, o in enumerate(reference.origin)
                        ),
                        entity=(reference.tdim, 0),
                    )
                )

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, order)

        self.variant = variant
        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        return self.order

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order

    names = ["Crouzeix-Raviart", "CR", "Crouzeix-Falk", "CF"]
    references = ["triangle", "tetrahedron"]
    min_order = 1
    max_order = {"tetrahedron": 1}
    continuity = "L2"
    value_type = "scalar"
    last_updated = "2023.05"
