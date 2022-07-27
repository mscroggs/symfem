"""Reduced Hsieh-Clough-Tocher elements on simplices.

This element's definition appears in https://doi.org/10.2307/2006147
(Ciarlet, 1978)
"""

import typing

import sympy

from ..finite_element import CiarletElement
from ..functionals import DerivativePointEvaluation, ListOfFunctionals, PointEvaluation
from ..functions import FunctionInput
from ..piecewise_functions import PiecewiseFunction
from ..references import Reference
from ..symbols import x


class P1Hermite(CiarletElement):
    """P1Hermite finite element."""

    def __init__(self, reference: Reference, order: int, poly: typing.List[FunctionInput]):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            poly: The polynomial basis
        """
        assert order == 3
        dofs: ListOfFunctionals = []
        for v_n, vs in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, vs, entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, vs, (1, 0), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, vs, (0, 1), entity=(0, v_n)))

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"poly": self._basis}

    names: typing.List[str] = []
    references = ["triangle"]
    min_order = 3
    max_order = 3
    continuity = "C0"


class ReducedHsiehCloughTocher(CiarletElement):
    """Reduced Hsieh-Clough-Tocher finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        from symfem import create_reference
        assert order == 3
        assert reference.name == "triangle"
        dofs: ListOfFunctionals = []
        for v_n, vs in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, vs, entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, vs, (1, 0), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, vs, (0, 1), entity=(0, v_n)))

        mid = tuple(sympy.Rational(sum(i), len(i)) for i in zip(*reference.vertices))

        subs = [
            (reference.vertices[0], reference.vertices[1], mid),
            (reference.vertices[1], reference.vertices[2], mid),
            (reference.vertices[2], reference.vertices[0], mid)]

        refs = [create_reference("triangle", vs) for vs in subs]

        polys: typing.List[typing.List[FunctionInput]] = [[], [], []]
        for i in range(order + 1):
            for j in range(order + 1 - i):
                if i != 2 or j != 1:
                    polys[0].append(x[0] ** i * x[1] ** j)
        polys[1] += [1, x[0], x[0] ** 2, x[1], x[0] * x[1], x[1] ** 2,
                     x[0] * x[1] ** 2 - x[0] ** 2 * x[1],
                     x[0] ** 3 - x[1] ** 3, x[0] ** 3 + 3 * x[0] * x[1] ** 2]
        for i in range(order + 1):
            for j in range(order + 1 - i):
                if i != 1 or j != 2:
                    polys[2].append(x[0] ** i * x[1] ** j)

        bases = []
        for r, p in zip(refs, polys):
            bf = []
            for f in P1Hermite(r, 3, p).get_basis_functions():
                bf.append(f)
            bases.append(bf)

        piece_list: typing.List[typing.Tuple[FunctionInput, ...]] = []
        piece_list.append((bases[0][0], 0, bases[2][3]))
        piece_list.append((bases[0][1], 0, bases[2][4]))
        piece_list.append((bases[0][2], 0, bases[2][5]))
        piece_list.append((bases[0][3], bases[1][0], 0))
        piece_list.append((bases[0][4], bases[1][1], 0))
        piece_list.append((bases[0][5], bases[1][2], 0))
        # TODO: are these right to remove??
        # piece_list.append((bases[0][6], bases[1][6], bases[2][6]))
        # piece_list.append((bases[0][7], bases[1][7], bases[2][7]))
        # piece_list.append((bases[0][8], bases[1][8], bases[2][8]))
        piece_list.append((0, bases[1][3], bases[2][0]))
        piece_list.append((0, bases[1][4], bases[2][1]))
        piece_list.append((0, bases[1][5], bases[2][2]))

        poly: typing.List[FunctionInput] = []
        poly += [
            PiecewiseFunction({i: j for i, j in zip(subs, p)}, 2)
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
