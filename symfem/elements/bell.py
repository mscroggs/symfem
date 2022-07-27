"""Bell elements on triangle.

This element's definition is given in https://doi.org/10.1002/nme.1620010108 (Bell, 1969)
"""

import typing

from ..finite_element import CiarletElement
from ..functionals import (DerivativePointEvaluation, ListOfFunctionals,
                           NormalDerivativeIntegralMoment, PointEvaluation)
from ..functions import FunctionInput
from ..moments import make_integral_moment_dofs
from ..polynomials import polynomial_set_1d
from ..references import Reference
from .lagrange import Lagrange


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
        assert order == 5
        dofs: ListOfFunctionals = []
        for v_n, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, v, (1, 0), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, v, (0, 1), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, v, (2, 0), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, v, (1, 1), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(reference, v, (0, 2), entity=(0, v_n)))
        dofs += make_integral_moment_dofs(
            reference,
            edges=(NormalDerivativeIntegralMoment, Lagrange, 0, {"variant": variant}),
        )
        self.variant = variant

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, order)

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["Bell"]
    references = ["triangle"]
    min_order = 5
    max_order = 5
    continuity = "C1"
