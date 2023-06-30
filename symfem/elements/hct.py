"""Hsieh-Clough-Tocher elements on simplices.

This element's definition appears in https://doi.org/10.2307/2006147
(Ciarlet, 1978)
"""

import typing

import sympy

from ..finite_element import CiarletElement
from ..functionals import (DerivativePointEvaluation, ListOfFunctionals, PointEvaluation,
                           PointNormalDerivativeEvaluation)
from ..functions import FunctionInput, ScalarFunction
from ..piecewise_functions import PiecewiseFunction
from ..references import NonDefaultReferenceError, Reference
from ..symbols import x


class HsiehCloughTocher(CiarletElement):
    """Hsieh-Clough-Tocher finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        assert order == 3
        assert reference.name == "triangle"
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        dofs: ListOfFunctionals = []
        for v_n, vs in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, vs, entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, vs, (1, 0), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, vs, (0, 1), entity=(0, v_n)))
        for e_n in range(reference.sub_entity_count(1)):
            sub_ref = reference.sub_entity(1, e_n)
            dofs.append(PointNormalDerivativeEvaluation(
                reference, sub_ref.midpoint(), sub_ref, entity=(1, e_n)))

        mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))

        subs = [
            (reference.vertices[0], reference.vertices[1], mid),
            (reference.vertices[1], reference.vertices[2], mid),
            (reference.vertices[2], reference.vertices[0], mid)]

        piece_list = [tuple(ScalarFunction(p) for _ in range(3))
                      for p in [1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2,
                                x[0]**3, x[0]**2*x[1], x[0]*x[1]**2, x[1]**3]]
        piece_list.append((
            ScalarFunction(-23*x[0]**3 + 24*x[0]**2*x[1] - 12*x[0]*x[1]**2 + 36*x[1]**2),
            ScalarFunction(
                -28*x[0]**3 + 12*x[0]**2*x[1] + 9*x[0]**2 - 3*x[0] + 32*x[1]**3 + 12*x[1] - 1),
            ScalarFunction(-15*x[0]**2 - 33*x[0]*x[1]**2 + 30*x[0]*x[1] + 22*x[1]**3 + 21*x[1]**2)))
        piece_list.append((
            ScalarFunction(
                22*x[0]**3 - 21*x[0]**2*x[1] - 12*x[0]*x[1]**2 + 30*x[0]*x[1] - 24*x[1]**2),
            ScalarFunction(
                32*x[0]**3 + 12*x[0]**2*x[1] - 21*x[0]**2 + 12*x[0] - 28*x[1]**3 - 3*x[1] - 1),
            ScalarFunction(15*x[0]**2 + 12*x[0]*x[1]**2 - 23*x[1]**3 - 9*x[1]**2)))

        poly: typing.List[FunctionInput] = []
        poly += [
            PiecewiseFunction({i: j for i, j in zip(subs, p)}, 2)
            for p in piece_list]
        poly = reference.map_polyset_from_default(poly)

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )

    names = ["Hsieh-Clough-Tocher", "Clough-Tocher", "HCT", "CT"]
    references = ["triangle"]
    min_order = 3
    max_order = 3
    # continuity = "C1"
    continuity = "C0"
    last_updated = "2023.06"
