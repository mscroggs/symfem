"""Checks for Walkington's tetrahedral macroelement."""

import pytest
import sympy

import symfem
from symfem.symbols import x


def test_walkington_unisolvence():
    element = symfem.create_element("tetrahedron", "Walkington", 4)
    assert element.space_dim == 45
    assert [len(element.entity_dofs(0, i)) for i in range(4)] == [10] * 4
    assert [len(element.entity_dofs(1, i)) for i in range(6)] == [0] * 6
    assert [len(element.entity_dofs(2, i)) for i in range(4)] == [1] * 4
    assert len(element.entity_dofs(3, 0)) == 1
    assert element.get_dual_matrix().rank() == 45
    assert element.polynomial_subdegree == 4
    assert element.polynomial_superdegree == 5


@pytest.mark.parametrize("cell, degree", [("triangle", 4), ("tetrahedron", 3), ("tetrahedron", 6)])
def test_walkington_invalid(cell, degree):
    with pytest.raises(ValueError):
        symfem.create_element(cell, "Walkington", degree)


def test_walkington_constraints():
    element = symfem.create_element("tetrahedron", "Walkington", 4)
    reference = element.reference
    mid = reference.midpoint()
    s, t = sympy.symbols("s t")
    for p in element.get_polynomial_basis()[35:]:
        pieces = list(p._pieces.items())
        # Adjacent pieces agree in value and gradient on their shared face.
        for i, (cell, value) in enumerate(pieces):
            for other, other_value in pieces[i + 1 :]:
                shared = [v for v in cell if v in other and v != mid]
                point = tuple(
                    mid[k] + s * (shared[0][k] - mid[k]) + t * (shared[1][k] - mid[k])
                    for k in range(3)
                )
                delta = value.as_sympy() - other_value.as_sympy()
                for q in [delta] + [sympy.diff(delta, xi) for xi in x]:
                    assert sympy.expand(q.subs(dict(zip(x, point)), simultaneous=True)) == 0
            # Cubic exterior normal derivative.
            face_vertices = [v for v in cell if v != mid]
            a, b, c = map(sympy.Matrix, face_vertices)
            normal = (b - a).cross(c - a)
            point = a + s * (b - a) + t * (c - a)
            derivative = sum(normal[k] * sympy.diff(value.as_sympy(), x[k]) for k in range(3))
            trace = derivative.subs(dict(zip(x, point)), simultaneous=True)
            assert sympy.Poly(trace, s, t).total_degree() <= 3
