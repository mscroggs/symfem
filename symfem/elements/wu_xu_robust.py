"""Robust Wu-Xu H3-nonconforming triangle (https://doi.org/10.1090/mcom/3361)."""

from symfem.elements.wu_xu import derivatives
from symfem.finite_element import CiarletElement
from symfem.functionals import (
    DerivativePointEvaluation,
    IntegralOfDirectionalMultiderivative,
    ListOfFunctionals,
    PointEvaluation,
)
from symfem.polynomials import polynomial_set_1d
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import x

__all__ = ["WuXuRobust"]


class WuXuRobust(CiarletElement):
    """The robust 15-dimensional Wu-Xu triangle."""

    def __init__(self, reference: Reference, order: int):
        """Create the element, indexed by polynomial subdegree three."""
        assert reference.name == "triangle" and order == 3
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()
        bubble = x[0] * x[1] * (1 - x[0] - x[1])
        linear = polynomial_set_1d(2, 1)
        poly = polynomial_set_1d(2, 3) + [bubble * p for p in linear[1:]]
        poly += [bubble**2 * p for p in linear]
        dofs: ListOfFunctionals = []
        for i, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, i)))
            for alpha in derivatives(2, 1):
                dofs.append(DerivativePointEvaluation(reference, v, alpha, entity=(0, i)))
        for i in range(3):
            edge = reference.sub_entity(1, i)
            for degree in (1, 2):
                dofs.append(
                    IntegralOfDirectionalMultiderivative(
                        reference,
                        (edge.normal(),) * degree,
                        (1,) * degree,
                        (1, i),
                        scale=1 / edge.jacobian(),
                    )
                )
        super().__init__(reference, order, poly, dofs, 2, 1)

    @property
    def polynomial_subdegree(self) -> int:
        return 3

    @property
    def polynomial_superdegree(self) -> int:
        return 7

    @property
    def lagrange_subdegree(self) -> int:
        return 3

    @property
    def lagrange_superdegree(self) -> int:
        return 7

    names = ["robust Wu-Xu", "Wu-Xu robust"]
    references = ["triangle"]
    min_order = 3
    max_order = 3
    continuity = "C0"
    value_type = "scalar"
    last_updated = "2026.11"
