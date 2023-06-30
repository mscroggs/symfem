"""Alfeld-Sorokina element on a triangle.

This element's definition appears in https://doi.org/10.1007/s10543-015-0557-x
(Alfeld, Sorokina, 2015)
"""

import typing

import sympy

from ..finite_element import CiarletElement
from ..functionals import DotPointEvaluation, ListOfFunctionals, PointDivergenceEvaluation
from ..functions import FunctionInput, VectorFunction
from ..piecewise_functions import PiecewiseFunction
from ..references import NonDefaultReferenceError, Reference
from ..symbols import x


class AlfeldSorokina(CiarletElement):
    """Alfeld-Sorokina finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        assert order == 2
        assert reference.name == "triangle"
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        dofs: ListOfFunctionals = []
        for v_n, vs in enumerate(reference.vertices):
            dofs.append(DotPointEvaluation(reference, vs, (1, 0), entity=(0, v_n)))
            dofs.append(DotPointEvaluation(reference, vs, (0, 1), entity=(0, v_n)))
            dofs.append(PointDivergenceEvaluation(reference, vs, entity=(0, v_n)))

        for e_n in range(reference.sub_entity_count(1)):
            sub_ref = reference.sub_entity(1, e_n)
            dofs.append(DotPointEvaluation(reference, sub_ref.midpoint(), (1, 0), entity=(1, e_n)))
            dofs.append(DotPointEvaluation(reference, sub_ref.midpoint(), (0, 1), entity=(1, e_n)))

        mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))

        subs = [
            (reference.vertices[0], reference.vertices[1], mid),
            (reference.vertices[1], reference.vertices[2], mid),
            (reference.vertices[2], reference.vertices[0], mid)]

        piece_list = [tuple(VectorFunction((p, 0)) for _ in range(3))
                      for p in [1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2]]
        piece_list += [tuple(VectorFunction((0, p)) for _ in range(3))
                       for p in [1, x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2]]

        piece_list.append((
            VectorFunction((4*x[0] + 12*x[1]**2 - 4*x[1], 6*x[0]*x[1] - 4*x[1])),
            VectorFunction((3*x[0]**2 + 2*x[0] + 4*x[1] - 1, -3*x[0]**2 + 4*x[0] - 2*x[1] - 1)),
            VectorFunction((3*x[0]**2 + 6*x[0]*x[1] + 3*x[1]**2, 9*x[0]**2 - 4*x[0] - 3*x[1]**2))))
        piece_list.append((
            VectorFunction((144*x[0]*x[1] + 10*x[0] + 12*x[1]**2 - 10*x[1],
                            -30*x[0]*x[1] + 36*x[1]**2 - 10*x[1])),
            VectorFunction((-69*x[0]**2 + 104*x[0] + 46*x[1] - 25,
                            24*x[0]**2 - 26*x[0] + 4*x[1] + 2)),
            VectorFunction((39*x[0]**2 + 96*x[0]*x[1] + 21*x[1]**2,
                            -10*x[0] + 6*x[1]**2))))
        piece_list.append((
            VectorFunction((-14*x[0] + 12*x[1]**2 + 14*x[1],
                            42*x[0]*x[1] - 108*x[1]**2 + 14*x[1])),
            VectorFunction((3*x[0]**2 - 16*x[0] + 22*x[1] - 1,
                            24*x[0]**2 + 144*x[0]*x[1] - 50*x[0] - 92*x[1] + 26)),
            VectorFunction((-33*x[0]**2 + 24*x[0]*x[1] + 21*x[1]**2,
                            14*x[0] - 66*x[1]**2))))

        poly: typing.List[FunctionInput] = []
        poly += [
            PiecewiseFunction({i: j for i, j in zip(subs, p)}, 2)
            for p in piece_list]

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )

    names = ["Alfeld-Sorokina", "AS"]
    references = ["triangle"]
    min_order = 2
    max_order = 2
    continuity = "C0"
    last_updated = "2023.05"
