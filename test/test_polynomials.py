import pytest
import sympy
from symfem.core.polynomials import Hdiv_polynomials, Hcurl_polynomials
from symfem.core.vectors import vdot
from symfem.core.symbolic import x, t, subs
from symfem.core.calculus import div
from symfem import create_reference, create_element


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
            p_edge = subs(p, x, [(1 - t[0]) * i + t[0] * j
                                 for i, j in zip(sub_ref.origin, sub_ref.axes[0])])
            poly = vdot(p_edge, sub_ref.normal())
            assert poly.is_real or sympy.Poly(poly).degree() <= 1
