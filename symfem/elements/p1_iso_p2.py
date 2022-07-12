"""P1-iso-P2 elements.

This element's definition appears in https://doi.org/10.1007/BF01399555
(Bercovier, Pironneau, 1979)
"""

import sympy
from ..references import Reference
from ..functionals import ListOfFunctionals
from ..finite_element import CiarletElement
from ..functionals import PointEvaluation
from ..symbolic import PiecewiseFunction
from ..symbolic import x as x_variables


class P1IsoP2Tri(CiarletElement):
    """P1-iso-P2 finite element on a triangle."""

    def __init__(self, reference: Reference, order: int):
        half = sympy.Rational(1, 2)
        tris = [
            ((0, 0), (half, 0), (0, half)),
            ((1, 0), (half, half), (half, 0)),
            ((0, 1), (0, half), (half, half)),
            ((0, half), (half, half), (half, 0)),
        ]
        poly = []
        x = x_variables[0]
        y = x_variables[1]
        c = 1 - x - y
        for pieces in [
            {0: 2 * c - 1},
            {1: 2 * x - 1},
            {2: 2 * y - 1},
            {0: 2 * x, 1: 2 * c, 3: 1 - 2 * y},
            {0: 2 * y, 2: 2 * c, 3: 1 - 2 * x},
            {1: 2 * y, 2: 2 * x, 3: 1 - 2 * c},
        ]:
            poly.append(PiecewiseFunction([
                (t, pieces[i]) if i in pieces else (t, 0) for i, t in enumerate(tris)], "triangle"))

        dofs: ListOfFunctionals = []
        for v_n, v in enumerate(reference.reference_vertices):
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


class P1IsoP2Quad(CiarletElement):
    """P1-iso-P2 finite element on a quadrilateral."""

    def __init__(self, reference: Reference, order: int):
        half = sympy.Rational(1, 2)
        quads = [
            ((0, 0), (half, 0), (0, half), (half, half)),
            ((half, 0), (1, 0), (half, half), (1, half)),
            ((0, half), (half, half), (0, 1), (half, 1)),
            ((half, half), (1, half), (half, 1), (1, 1)),
        ]
        poly = []
        x = x_variables[0]
        y = x_variables[1]
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
            poly.append(PiecewiseFunction([
                (q, pieces[i]) if i in pieces else (q, 0) for i, q in enumerate(quads)],
                "quadrilateral"))

        dofs: ListOfFunctionals = []
        for v_n, v in enumerate(reference.reference_vertices):
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
