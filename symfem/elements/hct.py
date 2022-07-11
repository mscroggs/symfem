"""Hsieh-Clough-Tocher elements on simplices.

This element's definition appears in https://doi.org/10.2307/2006147
(Ciarlet, 1978)
"""

import sympy
import typing
from ..references import Reference
from ..functionals import ListOfFunctionals
from ..finite_element import CiarletElement, ElementBasisFunction
from ..functionals import (PointEvaluation, PointNormalDerivativeEvaluation,
                           DerivativePointEvaluation)
from ..symbolic import sym_sum, PiecewiseFunction, ListOfVectorFunctions
from ..basis_function import BasisFunction
from .hermite import Hermite


class HsiehCloughTocher(CiarletElement):
    """Hsieh-Clough-Tocher finite element."""

    def __init__(self, reference: Reference, order: int):
        from symfem import create_reference
        assert order == 3
        assert reference.name == "triangle"
        dofs: ListOfFunctionals = []
        for v_n, vs in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, vs, entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, vs, (1, 0), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, vs, (0, 1), entity=(0, v_n)))
        for e_n, vs in enumerate(reference.sub_entities(1)):
            assert isinstance(reference.sub_entity_types[1], str)
            sub_ref = create_reference(
                reference.sub_entity_types[1],
                vertices=tuple(reference.vertices[v] for v in vs))
            midpoint = tuple(sym_sum(i) / len(i)
                             for i in zip(*[reference.vertices[i] for i in vs]))
            dofs.append(
                PointNormalDerivativeEvaluation(reference, midpoint, sub_ref, entity=(1, e_n)))

        mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))

        subs = [
            (reference.vertices[0], reference.vertices[1], mid),
            (reference.vertices[1], reference.vertices[2], mid),
            (reference.vertices[2], reference.vertices[0], mid)]

        refs = [create_reference("triangle", vs) for vs in subs]

        hermite_spaces = [Hermite(ref, 3) for ref in refs]

        piece_list: typing.List[typing.Tuple[typing.Union[int, BasisFunction], ...]] = []
        piece_list.append((hermite_spaces[0].get_basis_function(0), 0,
                           hermite_spaces[2].get_basis_function(3)))
        piece_list.append((hermite_spaces[0].get_basis_function(1), 0,
                           hermite_spaces[2].get_basis_function(4)))
        piece_list.append((hermite_spaces[0].get_basis_function(2), 0,
                           hermite_spaces[2].get_basis_function(5)))
        piece_list.append((hermite_spaces[0].get_basis_function(3),
                           hermite_spaces[1].get_basis_function(0), 0))
        piece_list.append((hermite_spaces[0].get_basis_function(4),
                           hermite_spaces[1].get_basis_function(1), 0))
        piece_list.append((hermite_spaces[0].get_basis_function(5),
                           hermite_spaces[1].get_basis_function(2), 0))
        piece_list.append((hermite_spaces[0].get_basis_function(6),
                           hermite_spaces[1].get_basis_function(6),
                           hermite_spaces[2].get_basis_function(6)))
        piece_list.append((hermite_spaces[0].get_basis_function(7),
                           hermite_spaces[1].get_basis_function(7),
                           hermite_spaces[2].get_basis_function(7)))
        piece_list.append((hermite_spaces[0].get_basis_function(8),
                           hermite_spaces[1].get_basis_function(8),
                           hermite_spaces[2].get_basis_function(8)))
        piece_list.append((0, hermite_spaces[1].get_basis_function(3),
                           hermite_spaces[2].get_basis_function(0)))
        piece_list.append((0, hermite_spaces[1].get_basis_function(4),
                           hermite_spaces[2].get_basis_function(1)))
        piece_list.append((0, hermite_spaces[1].get_basis_function(5),
                           hermite_spaces[2].get_basis_function(2)))

        # TODO: are these right to remove??
        # piece_list.append((hermite_spaces[0].get_basis_function(9), 0, 0))
        # piece_list.append((0, hermite_spaces[1].get_basis_function(9), 0))
        # piece_list.append((0, 0, hermite_spaces[2].get_basis_function(9)))

        piece_list2: ListOfVectorFunctions = []
        for i in piece_list:
            t = []
            for j in i:
                if isinstance(j, ElementBasisFunction):
                    j_f = j.get_function()
                    assert isinstance(j_f, (int, sympy.core.expr.Expr))
                    t.append(j_f)
                else:
                    assert isinstance(j, int)
                    t.append(j)
            piece_list2.append(tuple(t))

        poly = [
            PiecewiseFunction(list(zip(subs, p)), "triangle")
            for p in piece_list2
        ]

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )

    names = ["Hsieh-Clough-Tocher", "Clough-Tocher", "HCT", "CT"]
    references = ["triangle"]
    min_order = 3
    max_order = 3
    continuity = "C1"
