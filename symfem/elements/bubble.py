"""Bubble elements on simplices.

This element's definition appears in https://doi.org/10.1007/978-3-642-23099-8_3
(Kirby, Logg, Rognes, Terrel, 2012)
"""

import typing
from itertools import product

import sympy

from ..finite_element import CiarletElement
from ..functionals import DotPointEvaluation, ListOfFunctionals, PointEvaluation
from ..functions import FunctionInput
from ..polynomials import polynomial_set_1d, quolynomial_set_1d
from ..references import Reference
from .lagrange import Lagrange


class Bubble(CiarletElement):
    """Bubble finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        from .. import create_element
        p1 = create_element(reference.name, "Lagrange", 1)
        bubble = 1
        for f in p1.get_basis_functions():
            bubble *= f

        poly: typing.List[FunctionInput] = []

        # TODO: variants
        if reference.name == "interval":
            poly += [bubble * p for p in polynomial_set_1d(reference.tdim, order - 2)]
        elif reference.name == "triangle":
            poly += [bubble * p for p in polynomial_set_1d(reference.tdim, order - 3)]
        elif reference.name == "tetrahedron":
            poly += [bubble * p for p in polynomial_set_1d(reference.tdim, order - 4)]
        elif reference.name == "quadrilateral":
            poly += [bubble * p for p in quolynomial_set_1d(reference.tdim, order - 2)]
        elif reference.name == "hexahedron":
            poly += [bubble * p for p in quolynomial_set_1d(reference.tdim, order - 2)]
        else:
            raise ValueError(f"Unsupported reference: {reference.name}")

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
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["bubble"]
    references = ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"]
    min_order = {"interval": 2, "triangle": 3, "tetrahedron": 4,
                 "quadrilateral": 2, "hexahedron": 2}
    continuity = "C0"


class BubbleEnrichedLagrange(CiarletElement):
    """Bubble enriched Lagrange element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        lagrange = Lagrange(reference, order, variant)
        bubble = Bubble(reference, order + 2, variant)

        poly: typing.List[FunctionInput] = []
        for e in [lagrange, bubble]:
            for p in e._basis:
                poly.append(p)

        self.variant = variant

        super().__init__(
            reference, order, poly,
            lagrange.dofs + bubble.dofs, reference.tdim, 1
        )

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["bubble enriched Lagrange"]
    references = ["triangle"]
    min_order = 1
    continuity = "C0"


class BubbleEnrichedVectorLagrange(CiarletElement):
    """Bubble enriched Lagrange element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        lagrange = Lagrange(reference, order, variant)
        bubble = Bubble(reference, order + 2, variant)

        poly: typing.List[FunctionInput] = []
        for e in [lagrange, bubble]:
            for p in e._basis:
                poly.append((p, 0))
                poly.append((0, p))

        dofs: ListOfFunctionals = [DotPointEvaluation(reference, d.dof_point(), v, entity=d.entity)
                                   for d in lagrange.dofs + bubble.dofs
                                   for v in [(1, 0), (0, 1)]]

        self.variant = variant

        super().__init__(reference, order, poly, dofs, reference.tdim, 2)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["bubble enriched vector Lagrange"]
    references = ["triangle"]
    min_order = 1
    continuity = "C0"
