"""Huang-Zhang elements on a triangle.

This element's definition appears in https://doi.org/10.1007/s11464-011-0094-0 
(Huang, Zhang, 2011)
"""

import typing

import sympy

from ..finite_element import CiarletElement
from ..functionals import ListOfFunctionals, PointEvaluation
from ..functions import FunctionInput
from ..references import Reference
from ..symbols import AxisVariablesNotSingle, x
from ..functions import ScalarFunction, VectorFunction

def polynomial_set_1d_ZH(
         dim: int, variant: int, variables: AxisVariablesNotSingle = x
) -> typing.List[ScalarFunction]:
    """Huang-Zhang ."""
    order = 2;
    if dim == 1:
        raise RuntimeError('The Haung-Zhang polynomial space is not implemented for dimension {}'.format(dim))
    if dim == 2:
        if variant == 0:
            return [
                ScalarFunction(variables[0] ** i * variables[1] ** j)
                for j in range(order + 1)
                for i in range(order)
            ]
        elif variant == 1:
            return [
                ScalarFunction(variables[0] ** i * variables[1] ** j)
                for j in range(order)
                for i in range(order+1)
            ]

    if dim == 3:
        raise RuntimeError('The Haung-Zhang polynomial space is not implemented for dimension {}'.format(dim))

class HuangZhang(CiarletElement):
    """Huang-Zhang finite element."""

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
            poly += polynomial_set_1d_ZH(reference.tdim,0)
        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Huang-Zhang", "HZ"]
    references = ["quadrilateral"]
    min_order = 2
    max_order = 2
    continuity = "C0"
