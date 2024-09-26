"""P1 macro elements.

This element's definition appears in https://doi.org/10.1007/s00211-018-0970-6
(Christiansen, Hu, 2018)
"""

import typing

import sympy

from symfem.finite_element import CiarletElement
from symfem.functionals import IntegralAgainst, ListOfFunctionals, PointEvaluation
from symfem.functions import FunctionInput
from symfem.geometry import SetOfPoints
from symfem.piecewise_functions import PiecewiseFunction
from symfem.references import Reference
from symfem.symbols import x

__all__ = ["P1Macro"]


class P1Macro(CiarletElement):
    """P1 macro finite element on a triangle."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        third = sympy.Rational(1, 3)
        zero = sympy.Integer(0)
        one = sympy.Integer(1)
        tris: typing.List[SetOfPoints] = [
            ((zero, zero), (one, zero), (third, third)),
            ((one, zero), (zero, one), (third, third)),
            ((zero, one), (zero, zero), (third, third)),
        ]
        tris = [tuple(reference.get_point(p) for p in t) for t in tris]
        invmap = reference.get_inverse_map_to_self()
        poly: typing.List[FunctionInput] = [
            PiecewiseFunction({q: 1 for q in tris}, 2),
            PiecewiseFunction({q: x[0] for q in tris}, 2),
            PiecewiseFunction({q: x[1] for q in tris}, 2),
            PiecewiseFunction(
                {
                    tris[0]: 3 * invmap[1],
                    tris[1]: 3 * (1 - invmap[0] - invmap[1]),
                    tris[2]: 3 * invmap[0],
                },
                2,
            ),
        ]

        dofs: ListOfFunctionals = []
        for v_n, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))
        dofs.append(IntegralAgainst(reference, 1, entity=(2, 0)))

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    @property
    def lagrange_subdegree(self) -> int:
        raise NotImplementedError()

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        raise NotImplementedError()

    @property
    def polynomial_subdegree(self) -> int:
        raise NotImplementedError()

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        raise NotImplementedError()

    names = ["P1 macro"]
    references = ["triangle"]
    min_order = 1
    max_order = 1
    continuity = "C0"
    value_type = "scalar macro"
    last_updated = "2023.06"
