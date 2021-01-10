"""Mardal-Tai-Winther elements on simplices."""

from ..core.finite_element import FiniteElement, make_integral_moment_dofs
from ..core.polynomials import polynomial_set
from ..core.symbolic import x, zero, one
from ..core.calculus import curl
from ..core.functionals import NormalIntegralMoment, TangentIntegralMoment, VecIntegralMoment
from .lagrange import DiscontinuousLagrange


class MardalTaiWinther(FiniteElement):
    """Mardal-Tai-Winther Hdiv finite element."""

    def __init__(self, reference, order):
        from symfem import create_reference
        assert order == 3

        dofs = make_integral_moment_dofs(
            reference, facets=(NormalIntegralMoment, DiscontinuousLagrange, 1))

        if reference.name == "triangle":
            poly = [(one, zero), (x[0], zero), (x[1], zero),
                    (zero, one), (zero, x[0]), (zero, x[1]),
                    (5 * x[0] ** 2 + 16 * x[0] * x[1], -10 * x[0] * x[1] - 8 * x[1] ** 2),
                    (-x[1] * x[0] ** 2 / 2 - 9 * x[1] ** 2 * x[0] / 7 + 170 * 3 * x[1] ** 2 * x[0] / (27 * 28) - 170 * 7 * x[0] ** 3 / (20 * 27 * 28),  # noqa: E501
                     3 * x[1] ** 3 / 7 + x[0] * x[1] ** 2 / 2 - 170 * x[1] ** 3 / (27 * 28) + 170 * 21 * x[0] ** 2 * x[1] / (20 * 27 * 28)),  # noqa: E501
                    (-x[1] * x[0] ** 2 / 2 - 9 * x[1] ** 2 * x[0] / 7 - 17 * x[0] ** 2 / (28 * 4),
                     3 * x[1] ** 3 / 7 + x[0] * x[1] ** 2 / 2 + 2 * 17 * x[0] * x[1] / (28 * 4))]
            dofs += make_integral_moment_dofs(
                reference, facets=(TangentIntegralMoment, DiscontinuousLagrange, 0),
                facet_tangents=3)
        else:
            assert reference.name == "tetrahedron"

            poly = polynomial_set(reference.tdim, reference.tdim, 1)
            for p in polynomial_set(reference.tdim, reference.tdim, 1):
                poly.append(curl(tuple(i * x[0] * x[1] * x[2] * (1 - x[0] - x[1] - x[2])
                                       for i in p)))

            sub_type = reference.sub_entity_types[2]
            for i, vs in enumerate(reference.sub_entities(2)):
                sub_ref = create_reference(
                    sub_type,
                    vertices=[reference.reference_vertices[v] for v in vs])
                sub_element = DiscontinuousLagrange(sub_ref, 0)
                for f, d in zip(sub_element.get_basis_functions(), sub_element.dofs):
                    for e in sub_ref.edges:
                        edge_vec = tuple(j - k for j, k in zip(sub_ref.vertices[e[0]],
                                                               sub_ref.vertices[e[1]]))
                        dofs.append(VecIntegralMoment(sub_ref, f, edge_vec, d))

        super().__init__(reference, poly, dofs, reference.tdim, reference.tdim)

    names = ["Mardal-Tai-Winther", "MTW"]
    references = ["triangle", "tetrahedron"]
    min_order = 3
    max_order = 3
