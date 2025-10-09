"""Rannacher-Turek elements on tensor product cells.

This element's definition appears in https://doi.org/10.1002/num.1690080202
(Rannacher, Turek, 1992)
"""

import typing

from symfem.finite_element import CiarletElement
from symfem.functionals import ListOfFunctionals, PointEvaluation
from symfem.functions import FunctionInput
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import x

__all__ = ["RannacherTurek"]


class RannacherTurek(CiarletElement):
    """Rannacher-Turek finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        assert order == 1
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        dofs: ListOfFunctionals = []
        for e_n, vs in enumerate(reference.sub_entities(reference.tdim - 1)):
            pt = reference.sub_entity(reference.tdim - 1, e_n).midpoint()
            dofs.append(PointEvaluation(reference, pt, entity=(reference.tdim - 1, e_n)))

        poly: typing.List[FunctionInput] = []
        if reference.name == "quadrilateral":
            poly += [1, x[0], x[1], x[0] ** 2 - x[1] ** 2]
        else:
            poly += [1, x[0], x[1], x[2], x[0] ** 2 - x[1] ** 2, x[1] ** 2 - x[2] ** 2]

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {}

    @property
    def lagrange_subdegree(self) -> int:
        return 0

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return 2

    @property
    def polynomial_subdegree(self) -> int:
        return 1

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return 2

    names = ["Rannacher-Turek"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    max_order = 1
    continuity = "L2"
    value_type = "scalar"
    last_updated = "2023.05"
