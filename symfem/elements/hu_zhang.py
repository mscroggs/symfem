"""Hu-Zhang symmetric stress triangle (https://arxiv.org/abs/1406.7457)."""

import sympy

from symfem.elements.lagrange import Lagrange
from symfem.finite_element import CiarletElement
from symfem.functionals import (
    InnerProductIntegralMoment,
    IntegralMoment,
    ListOfFunctionals,
    PointInnerProduct,
)
from symfem.functions import FunctionInput
from symfem.polynomials import polynomial_set_vector
from symfem.references import NonDefaultReferenceError, Reference

__all__ = ["HuZhang"]


class HuZhang(CiarletElement):
    """Symmetric polynomial stresses with continuous normal traction."""

    def __init__(self, reference: Reference, order: int, variant: str = "integral"):
        """Create the Hu-Zhang element with point or integral degrees of freedom."""
        assert reference.name == "triangle" and order >= 3
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()
        if variant not in ("point", "integral"):
            raise ValueError(f"Unknown Hu-Zhang variant: {variant}")
        self.variant = variant
        poly: list[FunctionInput] = [
            ((p[0], p[1]), (p[1], p[2])) for p in polynomial_set_vector(2, 3, order)
        ]
        directions = [(1, 0), (0, 1)]
        components = [(a, b) for a in range(2) for b in range(a, 2)]
        dofs: ListOfFunctionals = []
        for i, v in enumerate(reference.vertices):
            for a, b in components:
                dofs.append(
                    PointInnerProduct(
                        reference,
                        v,
                        directions[a],
                        directions[b],
                        (0, i),
                        mapping="double_contravariant",
                    )
                )
        for i in range(3):
            edge = reference.sub_entity(1, i)
            if variant == "point":
                for j in range(1, order):
                    v = tuple(
                        a + sympy.Rational(j, order) * b for a, b in zip(edge.origin, edge.axes[0])
                    )
                    for direction in (edge.normal(), edge.tangent()):
                        dofs.append(
                            PointInnerProduct(
                                reference,
                                v,
                                edge.normal(),
                                direction,
                                (1, i),
                                mapping="double_contravariant",
                            )
                        )
            else:
                moment = Lagrange(edge.default_reference(), order - 2)
                for p, d in zip(moment.get_basis_functions(), moment.dofs):
                    for direction in (edge.normal(), edge.tangent()):
                        dofs.append(
                            InnerProductIntegralMoment(
                                reference,
                                p,
                                edge.normal(),
                                direction,
                                d,
                                (1, i),
                                mapping="double_contravariant",
                            )
                        )
        if variant == "point":
            for a in range(1, order):
                for b in range(1, order + 1 - a):
                    v = (sympy.Rational(a, order + 1), sympy.Rational(b, order + 1))
                    for i, j in components:
                        dofs.append(
                            PointInnerProduct(
                                reference,
                                v,
                                directions[i],
                                directions[j],
                                (2, 0),
                                mapping="double_contravariant",
                            )
                        )
        else:
            moment = Lagrange(reference, order - 2)
            for p, d in zip(moment.get_basis_functions(), moment.dofs):
                for i, j in components:
                    matrix = tuple(
                        tuple(p if (a, b) == (i, j) else 0 for b in range(2)) for a in range(2)
                    )
                    dofs.append(
                        IntegralMoment(reference, matrix, d, (2, 0), mapping="double_contravariant")
                    )
        super().__init__(reference, order, poly, dofs, 2, 4, (2, 2))

    def init_kwargs(self) -> dict[str, str]:
        """Return the variant."""
        return {"variant": self.variant}

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

    names = ["Hu-Zhang"]
    references = ["triangle"]
    min_order = 3
    continuity = "H(div)"
    value_type = "symmetric matrix"
    last_updated = "2026.11"
