"""DPC elements on tensor product cells."""

import typing
from itertools import product

import sympy

from ..finite_element import CiarletElement
from ..functionals import DotPointEvaluation, IntegralAgainst, ListOfFunctionals, PointEvaluation
from ..functions import FunctionInput
from ..polynomials import polynomial_set_1d, polynomial_set_vector
from ..references import NonDefaultReferenceError, Reference
from .lagrange import Lagrange


class DPC(CiarletElement):
    """A dPc element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        dofs: ListOfFunctionals = []
        if reference.name == "interval":
            for d in Lagrange(reference, order, variant).dofs:
                if isinstance(d, PointEvaluation):
                    dofs.append(PointEvaluation(reference, d.point, entity=(reference.tdim, 0)))
                elif isinstance(d, IntegralAgainst):
                    dofs.append(IntegralAgainst(
                        reference, d.f * reference.jacobian(), entity=(reference.tdim, 0)))
        else:
            if order == 0:
                points = [reference.get_point(tuple(
                    sympy.Rational(1, 2) for _ in range(reference.tdim)))]
            else:
                points = [
                    reference.get_point(tuple(sympy.Rational(j, order) for j in i[::-1]))
                    for i in product(range(order + 1), repeat=reference.tdim)
                    if sum(i) <= order
                ]

            dofs = [
                PointEvaluation(reference, d, entity=(reference.tdim, 0)) for d in points]

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, order)
        poly = reference.map_polyset_from_default(poly)

        super().__init__(
            reference, order, poly, dofs, reference.tdim, 1
        )
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["dPc"]
    references = ["interval", "quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "L2"
    last_updated = "2023.07.1"


class VectorDPC(CiarletElement):
    """Vector dPc finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        scalar_space = DPC(reference, order, variant)
        dofs: ListOfFunctionals = []
        if reference.tdim == 1:
            directions: typing.List[typing.Tuple[int, ...]] = [(1, )]
        else:
            directions = [
                tuple(1 if i == j else 0 for j in range(reference.tdim))
                for i in range(reference.tdim)
            ]
        for p in scalar_space.dofs:
            for d in directions:
                dofs.append(DotPointEvaluation(reference, p.dof_point(), d, entity=p.entity))

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_vector(reference.tdim, reference.tdim, order)

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["vector dPc"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "L2"
    last_updated = "2023.07"
