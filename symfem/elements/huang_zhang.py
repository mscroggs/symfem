"""Huang-Zhang elements on a triangle.

This element's definition appears in https://doi.org/10.1007/s11464-011-0094-0
(Huang, Zhang, 2011)
"""

import typing

import sympy

from ..finite_element import CiarletElement
from ..functionals import ListOfFunctionals, DotPointEvaluation
from ..functions import FunctionInput
from ..references import Reference
from ..symbols import x
from ..functions import VectorFunction


class HuangZhang(CiarletElement):
    """Huang-Zhang finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        assert reference.name == "quadrilateral"
        assert order == 1

        half = sympy.Rational(1, 2)
        dofs: ListOfFunctionals = []
        poly: typing.List[FunctionInput] = []
        poly += [
            VectorFunction([x[0] ** i * x[1] ** j, 0])
            for j in range(2)
            for i in range(3)
        ]
        poly += [
            VectorFunction([0, x[0] ** i * x[1] ** j])
            for j in range(3)
            for i in range(2)
        ]

        dofs += [
            DotPointEvaluation(reference, (0, 0), (1, 0), entity=(0, 0)),
            DotPointEvaluation(reference, (0, 1), (1, 0), entity=(0, 1)),
            DotPointEvaluation(reference, (1, 1), (1, 0), entity=(0, 2)),
            DotPointEvaluation(reference, (1, 0), (1, 0), entity=(0, 3)),
            DotPointEvaluation(reference, (half, 1), (1, 0), entity=(1, 1)),
            DotPointEvaluation(reference, (half, 0), (1, 0), entity=(1, 3))
        ]

        dofs += [
            DotPointEvaluation(reference, (0, 0), (0, 1), entity=(0, 0)),
            DotPointEvaluation(reference, (0, 1), (0, 1), entity=(0, 1)),
            DotPointEvaluation(reference, (1, 1), (0, 1), entity=(0, 2)),
            DotPointEvaluation(reference, (1, 0), (0, 1), entity=(0, 3)),
            DotPointEvaluation(reference, (0, half), (0, 1), entity=(1, 0)),
            DotPointEvaluation(reference, (1, half), (0, 1), entity=(1, 2))
        ]

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)

    names = ["Huang-Zhang", "HZ"]
    references = ["quadrilateral"]
    min_order = 1
    max_order = 1
    continuity = "H(div)"
