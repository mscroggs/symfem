"""Bell elements on triangle.

This element's definition is given in https://doi.org/10.1002/nme.1620010108 (Bell, 1969)
"""

from ..finite_element import CiarletElement
from ..moments import make_integral_moment_dofs
from ..polynomials import polynomial_set
from ..functionals import (PointEvaluation, NormalDerivativeIntegralMoment,
                           DerivativePointEvaluation)
from .lagrange import Lagrange


class Bell(CiarletElement):
    """Bell finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        assert reference.name == "triangle"
        assert order == 5
        dofs = []
        for v_n, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(v, entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(v, (1, 0), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(v, (0, 1), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(v, (2, 0), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(v, (1, 1), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(v, (0, 2), entity=(0, v_n)))
        dofs += make_integral_moment_dofs(
            reference,
            edges=(NormalDerivativeIntegralMoment, Lagrange, 0, {"variant": variant}),
        )
        self.variant = variant

        super().__init__(
            reference, order, polynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )

    def init_kwargs(self):
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["Bell"]
    references = ["triangle"]
    min_order = 5
    max_order = 5
    continuity = "C1"
