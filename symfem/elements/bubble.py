"""Bubble elements on simplices.

This element's definition appears in https://doi.org/10.1007/978-3-642-23099-8_3
(Kirby, Logg, Rognes, Terrel, 2012)
"""

import sympy
import typing
from ..references import Reference
from ..functionals import ListOfFunctionals
from itertools import product
from ..finite_element import CiarletElement
from ..polynomials import polynomial_set_1d, quolynomial_set_1d
from ..functionals import PointEvaluation, DotPointEvaluation
from ..symbolic import ListOfScalarFunctions, ListOfVectorFunctions
from .lagrange import Lagrange


class Bubble(CiarletElement):
    """Bubble finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        from .. import create_element
        p1 = create_element(reference.name, "Lagrange", 1)
        bubble = 1
        for f in p1.get_basis_functions():
            assert isinstance(f, (int, sympy.core.expr.Expr))
            bubble *= f
        # TODO: variants
        if reference.name == "interval":
            poly = [bubble * p for p in polynomial_set_1d(reference.tdim, order - 2)]
        elif reference.name == "triangle":
            poly = [bubble * p
                    for p in polynomial_set_1d(reference.tdim, order - 3)]
        elif reference.name == "tetrahedron":
            poly = [bubble * p
                    for p in polynomial_set_1d(reference.tdim, order - 4)]
        elif reference.name == "quadrilateral":
            poly = [bubble * p
                    for p in quolynomial_set_1d(reference.tdim, order - 2)]
        else:
            assert reference.name == "hexahedron"
            poly = [bubble * p
                    for p in quolynomial_set_1d(reference.tdim, order - 2)]

        dofs: ListOfFunctionals = []
        if reference.name in ["interval", "triangle", "tetrahedron"]:
            def func(i): return sum(i)
        else:
            def func(i): return max(i)
        for i in product(range(1, order), repeat=reference.tdim):
            if func(i) < order:
                point = tuple(sympy.Rational(j, order) for j in i)
                dofs.append(PointEvaluation(reference, point, entity=(reference.tdim, 0)))

        self.variant = variant

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["bubble"]
    references = ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"]
    min_order = {"interval": 2, "triangle": 3, "tetrahedron": 4,
                 "quadrilateral": 2, "hexahedron": 2}
    continuity = "C0"


class BubbleEnrichedLagrange(CiarletElement):
    """Bubble enriched Lagrange element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        lagrange = Lagrange(reference, order, variant)
        bubble = Bubble(reference, order + 2, variant)

        poly: ListOfScalarFunctions = []
        for e in [lagrange, bubble]:
            for p in e._basis:
                assert isinstance(p, (int, sympy.core.expr.Expr))
                poly.append(p)

        self.variant = variant

        super().__init__(
            reference, order, poly,
            lagrange.dofs + bubble.dofs, reference.tdim, 1
        )

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["bubble enriched Lagrange"]
    references = ["triangle"]
    min_order = 1
    continuity = "C0"


class BubbleEnrichedVectorLagrange(CiarletElement):
    """Bubble enriched Lagrange element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        lagrange = Lagrange(reference, order, variant)
        bubble = Bubble(reference, order + 2, variant)

        poly: ListOfVectorFunctions = []
        for e in [lagrange, bubble]:
            for p in e._basis:
                assert isinstance(p, (int, sympy.core.expr.Expr))
                poly.append((p, 0))
                poly.append((0, p))

        dofs: ListOfFunctionals = [DotPointEvaluation(reference, d.dof_point(), v, entity=d.entity)
                                   for d in lagrange.dofs + bubble.dofs
                                   for v in [(1, 0), (0, 1)]]

        self.variant = variant

        super().__init__(reference, order, poly, dofs, reference.tdim, 2)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["bubble enriched vector Lagrange"]
    references = ["triangle"]
    min_order = 1
    continuity = "C0"
