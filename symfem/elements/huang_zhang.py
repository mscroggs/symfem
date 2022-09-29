"""Fortin-Soulie elements on a triangle.

This element's definition appears in https://doi.org/10.1002/nme.1620190405
(Fortin, Soulie, 1973)
"""

import typing

import sympy

from ..finite_element import CiarletElement
from ..functionals import ListOfFunctionals, PointEvaluation
from ..functions import FunctionInput
from ..polynomials import polynomial_set_1d_ZH
from ..references import Reference


class FortinSoulie(CiarletElement):
    """Fortin-Soulie finite element."""

    def __init__(self, reference: Reference, order: int, variant: str):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        assert reference.name == "quadrilateral"

        assert order == 2
        
        half = sympy.Rational(1, 2)
        if variant == "X":
            dofs: ListOfFunctionals = [
                PointEvaluation(reference, (0,0), entity=(0, 0)),
                PointEvaluation(reference, (0,1), entity=(0, 1)),
                PointEvaluation(reference, (1,1), entity=(0, 2)),
                PointEvaluation(reference, (1, 0), entity=(0, 3)),
                PointEvaluation(reference, (half,1), entity=(1, 1)),
                PointEvaluation(reference, (half,0), entity=(1, 3))
            ]

            poly: typing.List[FunctionInput] = []
            poly += polynomial_set_1d_ZH(reference.tdim,1)
        elif variant == "Y":
            dofs: ListOfFunctionals = [
                PointEvaluation(reference, (0,0), entity=(0, 0)),
                PointEvaluation(reference, (0,1), entity=(0, 1)),
                PointEvaluation(reference, (1,1), entity=(0, 2)),
                PointEvaluation(reference, (1, 0), entity=(0, 3)),
                PointEvaluation(reference, (0,1+half), entity=(1, 0)),
                PointEvaluation(reference, (1,1+half), entity=(1, 2))
            ]

            poly: typing.List[FunctionInput] = []
            poly += polynomial_set_1d_ZH(reference.tdim00)
        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Huang-Zhang", "HZ"]
    references = ["quadrilateral"]
    min_order = 2
    max_order = 2
    continuity = "C0"
