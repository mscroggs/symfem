"""P1 macro elements.

This element's definition appears in https://doi.org/10.1007/s00211-018-0970-6
(Christiansen, Hu, 2018)
"""

import typing

import sympy

from ..finite_element import CiarletElement
from ..functionals import IntegralAgainst, ListOfFunctionals, PointEvaluation
from ..functions import FunctionInput
from ..geometry import SetOfPoints
from ..piecewise_functions import PiecewiseFunction
from ..references import NonDefaultReferenceError, Reference
from ..symbols import x


class P1Macro(CiarletElement):
    """P1 macro finite element on a triangle."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        third = sympy.Rational(1, 3)
        zero = sympy.Integer(0)
        one = sympy.Integer(1)
        tris: typing.List[SetOfPoints] = [
            ((zero, zero), (one, zero), (third, third)),
            ((one, zero), (zero, one), (third, third)),
            ((zero, one), (zero, zero), (third, third)),
        ]
        poly: typing.List[FunctionInput] = [
            PiecewiseFunction({q: 1 for q in tris}, 2),
            PiecewiseFunction({q: x[0] for q in tris}, 2),
            PiecewiseFunction({q: x[1] for q in tris}, 2),
            PiecewiseFunction({
                tris[0]: 3 * x[1],
                tris[1]: 3 * (1 - x[0] - x[1]),
                tris[2]: 3 * x[0],
            }, 2)]

        dofs: ListOfFunctionals = []
        for v_n, v in enumerate(reference.reference_vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))
        dofs.append(IntegralAgainst(reference, reference, 1, entity=(2, 0)))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )

    names = ["P1 macro"]
    references = ["triangle"]
    min_order = 1
    max_order = 1
    continuity = "C0"
    last_updated = "2023.05"
