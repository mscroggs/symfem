"""Bubble elements on simplices.

This element's definition appears in https://doi.org/10.1007/978-3-642-23099-8_3
(Kirby, Logg, Rognes, Terrel, 2012)
"""

import typing
from itertools import product

import sympy

from symfem.elements.lagrange import Lagrange
from symfem.finite_element import CiarletElement
from symfem.functionals import DotPointEvaluation, ListOfFunctionals, PointEvaluation
from symfem.functions import FunctionInput
from symfem.references import Reference

__all__ = ["Bubble", "BubbleEnrichedLagrange", "BubbleEnrichedVectorLagrange"]


class Bubble(CiarletElement):
    """Bubble finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        from symfem import create_element

        p1 = create_element(reference.name, "Lagrange", 1, vertices=reference.vertices)
        basis = p1.get_basis_functions()
        bubble = 1
        if reference.name in "quadrilateral":
            bubble *= basis[0]
            bubble *= basis[3]
        elif reference.name in "hexahedron":
            bubble *= basis[0]
            bubble *= basis[7]
        else:
            for f in basis:
                bubble *= f

        if reference.name in ["interval", "quadrilateral", "hexahedron"]:
            o = order - 2
        elif reference.name == "triangle":
            o = order - 3
        elif reference.name == "tetrahedron":
            o = order - 4
        else:
            raise ValueError(f"Unsupported reference: {reference.name}")

        pn = create_element(
            reference.name, "Lagrange", o, vertices=reference.vertices, variant=variant
        )

        poly = [bubble * p for p in pn.get_basis_functions()]

        dofs: ListOfFunctionals = []
        if reference.name in ["interval", "triangle", "tetrahedron"]:

            def func(i):
                return sum(i)
        else:

            def func(i):
                return max(i)

        for i in product(range(1, order), repeat=reference.tdim):
            if func(i) < order:
                point = reference.get_point(tuple(sympy.Rational(j, order) for j in i[::-1]))
                dofs.append(PointEvaluation(reference, point, entity=(reference.tdim, 0)))

        self.variant = variant

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        return -1

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order

    @property
    def polynomial_subdegree(self) -> int:
        return -1

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        if self.reference.name in ["quadrilateral", "hexahedron"]:
            return self.order * self.reference.tdim
        return self.order

    names = ["bubble"]
    references = ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"]
    min_order = {
        "interval": 2,
        "triangle": 3,
        "tetrahedron": 4,
        "quadrilateral": 2,
        "hexahedron": 2,
    }
    continuity = "C0"
    value_type = "scalar"
    last_updated = "2024.09"


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

        super().__init__(reference, order, poly, lagrange.dofs + bubble.dofs, reference.tdim, 1)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        return self.order

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order + 2

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order + 2

    names = ["bubble enriched Lagrange"]
    references = ["triangle"]
    min_order = 1
    continuity = "C0"
    value_type = "scalar"
    last_updated = "2024.09"


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

        dofs: ListOfFunctionals = [
            DotPointEvaluation(reference, d.dof_point(), v, entity=d.entity)
            for d in lagrange.dofs + bubble.dofs
            for v in [(1, 0), (0, 1)]
        ]

        self.variant = variant

        super().__init__(reference, order, poly, dofs, reference.tdim, 2)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        return self.order

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order + 2

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order + 2

    names = ["bubble enriched vector Lagrange"]
    references = ["triangle"]
    min_order = 1
    continuity = "C0"
    value_type = "vector"
    last_updated = "2024.09"
