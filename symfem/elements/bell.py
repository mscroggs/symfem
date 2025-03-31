"""Bell elements on triangle.

This element's definition is given in https://doi.org/10.1002/nme.1620010108 (Bell, 1969)
"""

import typing

from symfem.finite_element import CiarletElement
from symfem.functionals import DerivativePointEvaluation, ListOfFunctionals, PointEvaluation
from symfem.functions import FunctionInput
from symfem.polynomials import polynomial_set_1d
from symfem.references import Reference
from symfem.symbols import x

__all__ = ["Bell"]


class Bell(CiarletElement):
    """Bell finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        assert reference.name == "triangle"
        dofs: ListOfFunctionals = []
        for v_n, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, v, (1, 0), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, v, (0, 1), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, v, (2, 0), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, v, (1, 1), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, v, (0, 2), entity=(0, v_n)))
        self.variant = variant

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, 4)
        poly.append(x[0] ** 5 - x[1] ** 5)
        poly.append(x[0] ** 3 * x[1] ** 2 - x[0] ** 2 * x[1] ** 3)
        poly.append(5 * x[0] ** 2 * x[1] ** 3 - x[0] ** 5)

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        return 4

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return 5

    @property
    def polynomial_subdegree(self) -> int:
        return 4

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return 5

    names = ["Bell"]
    references = ["triangle"]
    min_order = 4
    max_order = 4
    continuity = "C1"
    value_type = "scalar"
    last_updated = "2025.03"
    _max_continuity_test_order = 3
