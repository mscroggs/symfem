"""Bell elements on triangle.

This element's definition is given in https://doi.org/10.1002/nme.1620010108 (Bell, 1969)
"""

from ..core.finite_element import FiniteElement
from ..core.polynomials import polynomial_set
from ..core.functionals import (PointEvaluation, PointNormalDerivativeEvaluation,
                                DerivativePointEvaluation)


class Bell(FiniteElement):
    """Bell finite element."""

    def __init__(self, reference, order, variant):
        assert reference.name == "triangle"
        assert order == 5
        from symfem import create_reference
        dofs = []
        for v_n, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(v, entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(v, (1, 0), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(v, (0, 1), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(v, (2, 0), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(v, (1, 1), entity=(0, v_n)))
            dofs.append(DerivativePointEvaluation(v, (0, 2), entity=(0, v_n)))
        for e_n, e in enumerate(reference.edges):
            sub_ref = create_reference(
                reference.sub_entity_types[1],
                vertices=tuple(reference.vertices[i] for i in e))
            dofs.append(PointNormalDerivativeEvaluation(
                tuple((i + j) / 2 for i, j in zip(reference.vertices[e[0]],
                                                  reference.vertices[e[1]])),
                sub_ref, (1, e_n)))

        super().__init__(
            reference, order, polynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )

    names = ["Bell"]
    references = ["triangle"]
    min_order = 5
    max_order = 5
    continuity = "C1"
    mapping = "identity"
