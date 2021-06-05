"""Hsieh-Clough-Tocher elements on simplices.

This element's definition appears in https://doi.org/10.2307/2006147
(Ciarlet, 1978)
"""

import sympy
from ..core.finite_element import CiarletElement
from ..core.functionals import (PointEvaluation, PointNormalDerivativeEvaluation,
                                DerivativePointEvaluation)
from ..core.symbolic import sym_sum, PiecewiseFunction, zero
from .hermite import Hermite


class HsiehCloughTocher(CiarletElement):
    """Hsieh-Clough-Tocher finite element."""

    def __init__(self, reference, order, variant):
        from symfem import create_reference
        assert order == 3
        assert reference.name == "triangle"
        dofs = []
        for v_n, vs in enumerate(reference.sub_entities(0)):
            dofs.append(PointEvaluation(vs, entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(vs, (1, 0), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(vs, (0, 1), entity=(0, v_n)))
        for e_n, vs in enumerate(reference.sub_entities(1)):
            sub_ref = create_reference(
                reference.sub_entity_types[1],
                vertices=[reference.reference_vertices[v] for v in vs])
            midpoint = tuple(sym_sum(i) / len(i)
                             for i in zip(*[reference.vertices[i] for i in vs]))
            dofs.append(
                PointNormalDerivativeEvaluation(midpoint, sub_ref, entity=(1, e_n)))

        mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))

        subs = [
            (reference.vertices[0], reference.vertices[1], mid),
            (reference.vertices[1], reference.vertices[2], mid),
            (reference.vertices[2], reference.vertices[0], mid)]

        refs = [create_reference("triangle", vs) for vs in subs]

        bases = [Hermite(ref, 3, variant).get_basis_functions() for ref in refs]

        piece_list = []
        piece_list.append((bases[0][0], zero, bases[2][3]))
        piece_list.append((bases[0][1], zero, bases[2][4]))
        piece_list.append((bases[0][2], zero, bases[2][5]))
        piece_list.append((bases[0][3], bases[1][0], zero))
        piece_list.append((bases[0][4], bases[1][1], zero))
        piece_list.append((bases[0][5], bases[1][2], zero))
        piece_list.append((bases[0][6], bases[1][6], bases[2][6]))
        piece_list.append((bases[0][7], bases[1][7], bases[2][7]))
        piece_list.append((bases[0][8], bases[1][8], bases[2][8]))
        piece_list.append((zero, bases[1][3], bases[2][0]))
        piece_list.append((zero, bases[1][4], bases[2][1]))
        piece_list.append((zero, bases[1][5], bases[2][2]))
        # TODO: are these right to remove??
        # piece_list.append((bases[0][9], zero, zero))
        # piece_list.append((zero, bases[1][9], zero))
        # piece_list.append((zero, zero, bases[2][9]))

        poly = [
            PiecewiseFunction(list(zip(subs, p)))
            for p in piece_list
        ]

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )

    names = ["Hsieh-Clough-Tocher", "Clough-Tocher", "HCT", "CT"]
    references = ["triangle"]
    min_order = 3
    max_order = 3
    continuity = "C1"
