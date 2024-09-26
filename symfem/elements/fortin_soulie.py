"""Fortin-Soulie elements on a triangle.

This element's definition appears in https://doi.org/10.1002/nme.1620190405
(Fortin, Soulie, 1973)
"""

import typing

import sympy

from symfem.finite_element import CiarletElement
from symfem.functionals import ListOfFunctionals, PointEvaluation
from symfem.functions import FunctionInput
from symfem.polynomials import polynomial_set_1d
from symfem.references import NonDefaultReferenceError, Reference

__all__ = ["FortinSoulie"]


class FortinSoulie(CiarletElement):
    """Fortin-Soulie finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        assert reference.name == "triangle"
        assert order == 2
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        third = sympy.Rational(1, 3)
        two_thirds = sympy.Rational(2, 3)
        dofs: ListOfFunctionals = [
            PointEvaluation(reference, (two_thirds, third), entity=(1, 0)),
            PointEvaluation(reference, (third, two_thirds), entity=(1, 0)),
            PointEvaluation(reference, (0, third), entity=(1, 1)),
            PointEvaluation(reference, (0, two_thirds), entity=(1, 1)),
            PointEvaluation(reference, (sympy.Rational(1, 2), 0), entity=(1, 2)),
            PointEvaluation(reference, (third, third), entity=(2, 0)),
        ]

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, order)
        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

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

    names = ["Fortin-Soulie", "FS"]
    references = ["triangle"]
    min_order = 2
    max_order = 2
    continuity = "L2"
    value_type = "scalar"
    last_updated = "2023.05"
