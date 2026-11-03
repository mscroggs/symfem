"""Triangular C2 elements and their common degrees of freedom."""

from symfem.elements.wu_xu import derivatives
from symfem.finite_element import CiarletElement
from symfem.functionals import (
    DerivativePointEvaluation,
    IntegralAgainst,
    IntegralOfDirectionalMultiderivative,
    ListOfFunctionals,
    PointEvaluation,
)
from symfem.functions import FunctionInput
from symfem.polynomials import polynomial_set_1d
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import x

__all__ = ["BrambleZlamalC2"]


def c2_dofs(reference: Reference, order: int, vertex_order: int) -> ListOfFunctionals:
    """Construct vertex jets and edge and cell moments for a C2 triangle."""
    dofs: ListOfFunctionals = []
    for i, v in enumerate(reference.vertices):
        dofs.append(PointEvaluation(reference, v, entity=(0, i)))
        for degree in range(1, vertex_order + 1):
            for alpha in derivatives(2, degree):
                dofs.append(DerivativePointEvaluation(reference, v, alpha, entity=(0, i)))
    for i in range(3):
        edge = reference.sub_entity(1, i)
        for degree in range(3):
            # Endpoint jets determine 2*(vertex_order-degree+1) conditions.
            for j in range(order - 2 * vertex_order + degree - 1):
                dof = IntegralOfDirectionalMultiderivative(
                    reference,
                    (edge.normal(),) * degree,
                    (1,) * degree,
                    (1, i),
                    scale=1 / edge.jacobian(),
                    weight=x[0] ** j,
                )
                dofs.append(dof)
    q = order - 3 * (vertex_order // 2 + 1)
    if q >= 0:
        for p in polynomial_set_1d(2, q):
            dofs.append(IntegralAgainst(reference, p, (2, 0)))
    return dofs


class BrambleZlamalC2(CiarletElement):
    """The complete-polynomial Bramble-Zlamal C2 triangle."""

    def __init__(self, reference: Reference, order: int):
        """Create the degree-nine or higher C2 triangle with C4 vertex jets."""
        assert reference.name == "triangle" and order >= 9
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()
        poly: list[FunctionInput] = list(polynomial_set_1d(2, order))
        dofs = c2_dofs(reference, order, 4)
        super().__init__(reference, order, poly, dofs, 2, 1)

    def get_dual_matrix(self, inverse=False, caching=True):
        """Return the dual matrix or its exact inverse."""
        matrix = super().get_dual_matrix(False, caching)
        if inverse:
            return matrix.to_DM(extension=True).inv().to_Matrix()
        return matrix

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> int:
        return self.order

    @property
    def lagrange_subdegree(self) -> int:
        return self.order

    @property
    def lagrange_superdegree(self) -> int:
        return self.order

    names = ["Bramble-Zlamal C2"]
    references = ["triangle"]
    min_order = 9
    continuity = "C2"
    value_type = "scalar"
    last_updated = "2026.11"
