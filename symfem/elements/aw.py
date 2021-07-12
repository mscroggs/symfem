"""Arnold-Winther elements on simplices.

This element's definition appears in https://doi.org/10.1007/s002110100348
(Arnold, Winther, 2002)
"""

import sympy
from ..finite_element import CiarletElement
from ..polynomials import polynomial_set
from ..functionals import (PointInnerProduct, InnerProductIntegralMoment,
                           VecIntegralMoment, IntegralMoment)
from ..symbolic import x
from ..calculus import diff
from .lagrange import DiscontinuousLagrange


class ArnoldWinther(CiarletElement):
    """An Arnold-Winther element."""

    def __init__(self, reference, order, variant="equispaced"):
        from symfem import create_reference
        assert reference.name == "triangle"
        poly = [(p[0], p[1], p[1], p[2])
                for p in polynomial_set(reference.tdim, 3, order - 1)]
        poly += [((order - k + 1) * (order - k + 2) * x[0] ** k * x[1] ** (order - k),
                  -k * (order - k + 2) * x[0] ** (k - 1) * x[1] ** (order - k + 1),
                  -k * (order - k + 2) * x[0] ** (k - 1) * x[1] ** (order - k + 1),
                  -k * (k - 1) * x[0] ** (k - 2) * x[1] ** (order - k + 2))
                 for k in range(order + 1)]
        poly += [(0, x[0] ** order, x[0] ** order, -order * x[0] ** (order - 1) * x[1]),
                 (0, 0, 0, x[0] ** order)]

        dofs = []
        for v_n, v in enumerate(reference.vertices):
            for d in [[(1, 0), (1, 0)],
                      [(1, 0), (0, 1)],
                      [(0, 1), (0, 1)]]:
                dofs.append(PointInnerProduct(v, d[0], d[1], entity=(0, v_n),
                                              mapping="double_contravariant"))
        for e_n, edge in enumerate(reference.edges):
            sub_ref = create_reference(
                reference.sub_entity_types[1],
                vertices=tuple(reference.vertices[i] for i in edge))
            sub_e = DiscontinuousLagrange(sub_ref, order - 2, variant)
            for dof_n, dof in enumerate(sub_e.dofs):
                p = sub_e.get_basis_function(dof_n)
                for component in [sub_ref.normal(), sub_ref.tangent()]:
                    dofs.append(
                        InnerProductIntegralMoment(sub_ref, p, component, sub_ref.normal(), dof,
                                                   entity=(1, e_n), mapping="double_contravariant"))
        sub_e = DiscontinuousLagrange(reference, order - 3, variant)
        for dof_n, dof in enumerate(sub_e.dofs):
            p = sub_e.get_basis_function(dof_n)
            for component in [(1, 0, 0, 0), (0, 1, 0, 0),
                              (0, 0, 0, 1)]:
                dofs.append(VecIntegralMoment(reference, p, component, dof, entity=(2, 0)))
        sub_e = DiscontinuousLagrange(reference, order - 4, variant)
        for p, dof in zip(sub_e.get_basis_functions(), sub_e.dofs):
            if sympy.Poly(p, x[:2]).degree() != order - 4:
                continue
            f = p * x[0] ** 2 * x[1] ** 2 * (1 - x[0] - x[1]) ** 2
            J = tuple(diff(f, x[i], x[j]) for i in range(2) for j in range(2))
            dofs.append(IntegralMoment(reference, J, dof, entity=(2, 0)))

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim ** 2,
                         (reference.tdim, reference.tdim))

    names = ["Arnold-Winther", "AW"]
    references = ["triangle"]
    min_order = 3
    continuity = "integral inner H(div)"
