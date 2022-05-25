import pytest
import sympy
from symfem.polynomials import Hdiv_polynomials, Hcurl_polynomials, orthogonal_basis
from symfem.vectors import vdot
from symfem.symbolic import x, t, subs
from symfem.calculus import div
from symfem import create_reference, create_element
from random import choice


@pytest.mark.parametrize("reference", ["triangle", "tetrahedron"])
@pytest.mark.parametrize("order", range(1, 4))
def test_Hdiv_space(reference, order):
    ref = create_reference(reference)
    polynomials = Hdiv_polynomials(ref.tdim, ref.tdim, order)
    for p in polynomials:
        for i, j in zip(x, p):
            assert j / i == p[0] / x[0]


@pytest.mark.parametrize("reference", ["triangle", "tetrahedron"])
@pytest.mark.parametrize("order", range(1, 4))
def test_Hcurl_space(reference, order):
    ref = create_reference(reference)
    polynomials = Hcurl_polynomials(ref.tdim, ref.tdim, order)
    for p in polynomials:
        assert vdot(p, x) == 0


@pytest.mark.parametrize("reference", ["triangle"])
def test_MTW_space(reference):
    e = create_element(reference, "MTW", 3)
    polynomials = e.get_polynomial_basis()
    for p in polynomials:
        assert div(p).is_real
        for vs in e.reference.sub_entities(1):
            sub_ref = create_reference(e.reference.sub_entity_types[1],
                                       [e.reference.vertices[i] for i in vs])
            p_edge = subs(p, x, [i + t[0] * j
                                 for i, j in zip(sub_ref.origin, sub_ref.axes[0])])
            poly = vdot(p_edge, sub_ref.normal()).expand().simplify()
            assert poly.is_real or sympy.Poly(poly).degree() <= 1


@pytest.mark.parametrize("reference", ["triangle", "quadrilateral",
                                       "tetrahedron", "hexahedron"])
@pytest.mark.parametrize("order", range(1, 5))
def test_BDFM_space(reference, order):
    e = create_element(reference, "BDFM", order)
    polynomials = e.get_polynomial_basis()
    tdim = e.reference.tdim
    for p in polynomials:
        for vs in e.reference.sub_entities(tdim - 1):
            sub_ref = create_reference(e.reference.sub_entity_types[tdim - 1],
                                       [e.reference.vertices[i] for i in vs])
            if tdim == 2:
                p_edge = subs(p, x, [i + t[0] * j
                                     for i, j in zip(sub_ref.origin, sub_ref.axes[0])])
            else:
                p_edge = subs(p, x, [i + t[0] * j + t[1] * k
                                     for i, j, k in zip(sub_ref.origin, sub_ref.axes[0],
                                                        sub_ref.axes[1])])
            poly = vdot(p_edge, sub_ref.normal()).expand().simplify()
            assert poly.is_real or sympy.Poly(poly).degree() <= order - 1


@pytest.mark.parametrize("reference", [
    "interval", "triangle", "quadrilateral",
    "tetrahedron", "hexahedron", "prism",
    "pyramid"])
@pytest.mark.parametrize("order", range(5))
def test_orthogonal_polynomials(reference, order):
    polynomials = orthogonal_basis(reference, order, 0)[0]
    ref = create_reference(reference)
    if len(polynomials) <= 5:
        for i, p in enumerate(polynomials):
            for q in polynomials[:i]:
                assert ref.integral(p * q, x) == 0
    else:
        for _ in range(15):
            p = choice(polynomials)
            q = choice(polynomials)
            if p != q:
                assert ref.integral(p * q, x) == 0


@pytest.mark.parametrize("reference", [
    "interval",
    "triangle", "quadrilateral",
    "tetrahedron",
    "hexahedron", "prism",
    "pyramid"
])
@pytest.mark.parametrize("order", range(5))
def test_orthogonal_polynomial_derivatives(reference, order):
    polynomials = orthogonal_basis(reference, order, 2)
    ref = create_reference(reference)

    if ref.tdim == 1:
        first_d = [(1, 0)]
        second_d = [(2, 0, 0)]
    elif ref.tdim == 2:
        first_d = [(1, 0), (2, 1)]
        second_d = [(3, 0, 0), (4, 0, 1), (5, 1, 1)]
    elif ref.tdim == 3:
        first_d = [(1, 0), (2, 1), (3, 2)]
        second_d = [(4, 0, 0), (5, 0, 1), (6, 0, 2), (7, 1, 1), (8, 1, 2), (9, 2, 2)]
    else:
        raise NotImplementedError()

    for i, j in first_d:
        for p, q in zip(polynomials[0], polynomials[i]):
            assert (p.diff(x[j]) - q).simplify() == 0
    for i, j, k in second_d:
        for p, q in zip(polynomials[0], polynomials[i]):
            assert (p.diff(x[j]).diff(x[k]) - q).simplify() == 0
