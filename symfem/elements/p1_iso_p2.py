"""P1-iso-P2 elements.

This element's definition appears in https://doi.org/10.1007/BF01399555
(Bercovier, Pironneau, 1979)
"""

import typing

import sympy

from ..finite_element import CiarletElement
from ..functionals import ListOfFunctionals, PointEvaluation
from ..functions import FunctionInput
from ..geometry import SetOfPoints
from ..piecewise_functions import PiecewiseFunction
from ..references import Reference


class P1IsoP2Interval(CiarletElement):
    """P1-iso-P2 finite element on an interval."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        zero = reference.get_point((sympy.Integer(0), ))
        half = reference.get_point((sympy.Rational(1, 2), ))
        one = reference.get_point((sympy.Integer(1), ))

        x = reference.get_inverse_map_to_self()[0]
        poly: typing.List[FunctionInput] = [
            PiecewiseFunction({(zero, half): 1 - 2 * x, (half, one): 0}, 1),
            PiecewiseFunction({(zero, half): 2 * x, (half, one): 2 - 2 * x}, 1),
            PiecewiseFunction({(zero, half): 0, (half, one): 2 * x - 1}, 1),
        ]

        dofs: ListOfFunctionals = []
        for v_n, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))
        entity = reference.sub_entity(1, 0)
        dofs.append(PointEvaluation(reference, entity.midpoint(), entity=(1, 0)))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )

    names = ["P1-iso-P2", "P2-iso-P1", "iso-P2 P1"]
    references = ["interval"]
    min_order = 1
    max_order = 1
    continuity = "C0"
    last_updated = "2023.08"


class P1IsoP2Tri(CiarletElement):
    """P1-iso-P2 finite element on a triangle."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        half = sympy.Rational(1, 2)
        zero = sympy.Integer(0)
        one = sympy.Integer(1)
        tris: typing.List[SetOfPoints] = [
            ((zero, zero), (half, zero), (zero, half)),
            ((one, zero), (half, half), (half, zero)),
            ((zero, one), (zero, half), (half, half)),
            ((zero, half), (half, half), (half, zero)),
        ]
        poly: typing.List[FunctionInput] = []
        invmap = reference.get_inverse_map_to_self()
        x = invmap[0]
        y = invmap[1]
        c = 1 - x - y
        for pieces in [
            {0: 2 * c - 1},
            {1: 2 * x - 1},
            {2: 2 * y - 1},
            {0: 2 * x, 1: 2 * c, 3: 1 - 2 * y},
            {0: 2 * y, 2: 2 * c, 3: 1 - 2 * x},
            {1: 2 * y, 2: 2 * x, 3: 1 - 2 * c},
        ]:
            poly.append(PiecewiseFunction({
                tuple(
                    reference.get_point(pt) for pt in q
                ): pieces[i] if i in pieces else 0
                for i, q in enumerate(tris)
            }, 2))

        dofs: ListOfFunctionals = []
        for v_n, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))
        for e_n in range(3):
            entity = reference.sub_entity(1, e_n)
            dofs.append(PointEvaluation(reference, entity.midpoint(), entity=(1, e_n)))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )

    names = ["P1-iso-P2", "P2-iso-P1", "iso-P2 P1"]
    references = ["triangle"]
    min_order = 1
    max_order = 1
    continuity = "C0"
    last_updated = "2023.06"


class P1IsoP2Quad(CiarletElement):
    """P1-iso-P2 finite element on a quadrilateral."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        half = sympy.Rational(1, 2)
        zero = sympy.Integer(0)
        one = sympy.Integer(1)
        quads: typing.List[SetOfPoints] = [
            ((zero, zero), (half, zero), (zero, half), (half, half)),
            ((half, zero), (one, zero), (half, half), (one, half)),
            ((zero, half), (half, half), (zero, one), (half, one)),
            ((half, half), (one, half), (half, one), (one, one)),
        ]
        poly: typing.List[FunctionInput] = []
        invmap = reference.get_inverse_map_to_self()
        x = invmap[0]
        y = invmap[1]
        for pieces in [
            {0: (1 - 2 * x) * (1 - 2 * y)},
            {1: (2 * x - 1) * (1 - 2 * y)},
            {2: (1 - 2 * x) * (2 * y - 1)},
            {3: (2 * x - 1) * (2 * y - 1)},
            {0: 2 * x * (1 - 2 * y), 1: 2 * (1 - x) * (1 - 2 * y)},
            {0: 2 * (1 - 2 * x) * y, 2: 2 * (1 - 2 * x) * (1 - y)},
            {1: 2 * (2 * x - 1) * y, 3: 2 * (2 * x - 1) * (1 - y)},
            {2: 2 * x * (2 * y - 1), 3: 2 * (1 - x) * (2 * y - 1)},
            {0: 4 * x * y, 1: 4 * (1 - x) * y, 2: 4 * x * (1 - y), 3: 4 * (1 - x) * (1 - y)},
        ]:
            poly.append(PiecewiseFunction({
                tuple(
                    reference.get_point(pt) for pt in q
                ): pieces[i] if i in pieces else 0
                for i, q in enumerate(quads)}, 2))

        dofs: ListOfFunctionals = []
        for v_n, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))
        for e_n in range(4):
            entity = reference.sub_entity(1, e_n)
            dofs.append(PointEvaluation(reference, entity.midpoint(), entity=(1, e_n)))
        dofs.append(PointEvaluation(reference, reference.midpoint(), entity=(2, 0)))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )

    names = ["P1-iso-P2", "P2-iso-P1", "iso-P2 P1"]
    references = ["quadrilateral"]
    min_order = 1
    max_order = 1
    continuity = "C0"
    last_updated = "2023.06"
