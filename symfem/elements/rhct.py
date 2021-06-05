"""Reduced Hsieh-Clough-Tocher elements on simplices.

This element's definition appears in https://doi.org/10.2307/2006147
(Ciarlet, 1978)
"""

import sympy
from ..core.finite_element import CiarletElement
from ..core.functionals import PointEvaluation, DerivativePointEvaluation
from ..core.polynomials import polynomial_set
from ..core.symbolic import PiecewiseFunction, zero, x, one


class P1Hermite(CiarletElement):
    """P1Hermite finite element."""

    def __init__(self, reference, order, variant, poly):
        assert order == 3
        dofs = []
        for v_n, vs in enumerate(reference.vertices):
            dofs.append(PointEvaluation(vs, entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(vs, (1, 0), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(vs, (0, 1), entity=(0, v_n)))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )

    names = []
    references = ["triangle"]
    min_order = 3
    max_order = 3
    continuity = "C0"


class ReducedHsiehCloughTocher(CiarletElement):
    """Reduced Hsieh-Clough-Tocher finite element."""

    def __init__(self, reference, order, variant):
        from symfem import create_reference
        assert order == 3
        assert reference.name == "triangle"
        dofs = []
        for v_n, vs in enumerate(reference.sub_entities(0)):
            dofs.append(PointEvaluation(vs, entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(vs, (1, 0), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(vs, (0, 1), entity=(0, v_n)))

        mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))

        subs = [
            (reference.vertices[0], reference.vertices[1], mid),
            (reference.vertices[1], reference.vertices[2], mid),
            (reference.vertices[2], reference.vertices[0], mid)]

        refs = [create_reference("triangle", vs) for vs in subs]

        polys = [
            polynomial_set(reference.tdim, 1, order),
            [],
            polynomial_set(reference.tdim, 1, order),
        ]
        polys[0].remove(x[0] ** 2 * x[1])
        polys[1] = [one, x[0], x[0] ** 2, x[1], x[0] * x[1], x[1] ** 2,
                    x[0] * x[1] ** 2 - x[0] ** 2 * x[1],
                    x[0] ** 3 - x[1] ** 3, x[0] ** 3 + 3 * x[0] * x[1] ** 2]
        polys[2].remove(x[0] * x[1] ** 2)

        bases = [P1Hermite(r, 3, variant, p).get_basis_functions()
                 for r, p in zip(refs, polys)]

        piece_list = []
        piece_list.append((bases[0][0], zero, bases[2][3]))
        piece_list.append((bases[0][1], zero, bases[2][4]))
        piece_list.append((bases[0][2], zero, bases[2][5]))
        piece_list.append((bases[0][3], bases[1][0], zero))
        piece_list.append((bases[0][4], bases[1][1], zero))
        piece_list.append((bases[0][5], bases[1][2], zero))
        # TODO: are these right to remove??
        # piece_list.append((bases[0][6], bases[1][6], bases[2][6]))
        # piece_list.append((bases[0][7], bases[1][7], bases[2][7]))
        # piece_list.append((bases[0][8], bases[1][8], bases[2][8]))
        piece_list.append((zero, bases[1][3], bases[2][0]))
        piece_list.append((zero, bases[1][4], bases[2][1]))
        piece_list.append((zero, bases[1][5], bases[2][2]))

        poly = [
            PiecewiseFunction(list(zip(subs, p)))
            for p in piece_list
        ]

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )

    names = ["reduced Hsieh-Clough-Tocher", "rHCT"]
    references = ["triangle"]
    min_order = 3
    max_order = 3
    continuity = "C1"
