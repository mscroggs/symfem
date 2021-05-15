"""Morley elements on simplices.

This element's definition appears in https://doi.org/10.1017/S0001925900004546
(Morley, 1968)
"""

from ..core.finite_element import CiarletElement
from ..core.polynomials import polynomial_set
from ..core.functionals import PointEvaluation, PointNormalDerivativeEvaluation
from ..core.symbolic import sym_sum


class Morley(CiarletElement):
    """Morley finite element."""

    def __init__(self, reference, order, variant):
        from symfem import create_reference
        assert order == 2
        assert reference.name == "triangle"
        dofs = []
        for v_n, vs in enumerate(reference.sub_entities(0)):
            dofs.append(PointEvaluation(vs, entity=(0, v_n)))
        for e_n, vs in enumerate(reference.sub_entities(1)):
            sub_ref = create_reference(
                reference.sub_entity_types[1],
                vertices=[reference.reference_vertices[v] for v in vs])
            midpoint = tuple(sym_sum(i) / len(i)
                             for i in zip(*[reference.vertices[i] for i in vs]))
            dofs.append(
                PointNormalDerivativeEvaluation(midpoint, sub_ref, entity=(1, e_n)))

        super().__init__(
            reference, order, polynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )

    names = ["Morley"]
    references = ["triangle"]
    min_order = 2
    max_order = 2
    mapping = "identity"
    continuity = "L2"
