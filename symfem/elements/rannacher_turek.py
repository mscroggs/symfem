"""Rannacher-Turek elements on tensor product cells.

This element's definition appears in https://doi.org/10.1002/num.1690080202
(Rannacher, Turek, 1992)
"""

import typing

from ..finite_element import CiarletElement
from ..functionals import ListOfFunctionals, PointEvaluation
from ..functions import FunctionInput
from ..references import Reference
from ..symbols import x


class RannacherTurek(CiarletElement):
    """Rannacher-Turek finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        assert order == 1

        dofs: ListOfFunctionals = []
        for e_n, vs in enumerate(reference.sub_entities(reference.tdim - 1)):
            pt = reference.sub_entity(reference.tdim - 1, e_n).midpoint()
            dofs.append(PointEvaluation(reference, pt, entity=(reference.tdim - 1, e_n)))

        poly: typing.List[FunctionInput] = []
        if reference.name == "quadrilateral":
            poly += [1, x[0], x[1], x[0]**2 - x[1]**2]
        else:
            poly += [1, x[0], x[1], x[2], x[0]**2 - x[1]**2, x[1]**2 - x[2]**2]

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {}

    names = ["Rannacher-Turek"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    max_order = 1
    continuity = "L2"
