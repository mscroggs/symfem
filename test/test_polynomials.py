"""Test polynomials."""

from random import choice

import pytest
import sympy

from symfem import create_element, create_reference
from symfem.functions import VectorFunction
from symfem.polynomials import (
    Hcurl_polynomials,
    Hdiv_polynomials,
    jacobi_polynomial,
    l2_dual,
    orthogonal_basis,
    orthonormal_basis,
)
from symfem.symbols import t, x
from symfem.utils import allequal


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
        assert p.dot(VectorFunction(x[: ref.tdim])) == 0


@pytest.mark.parametrize("reference", ["triangle"])
def test_MTW_space(reference):
    e = create_element(reference, "MTW", 1)
    polynomials = e.get_polynomial_basis()
    for p in polynomials:
        assert p.div().as_sympy().is_real
        for vs in e.reference.sub_entities(1):
            sub_ref = create_reference(
                e.reference.sub_entity_types[1], [e.reference.vertices[i] for i in vs]
            )
            p_edge = p.subs(x, [i + t[0] * j for i, j in zip(sub_ref.origin, sub_ref.axes[0])])
            poly = p_edge.dot(sub_ref.normal()).as_sympy().expand().simplify()
            assert poly.is_real or sympy.Poly(poly, x).degree() <= 1


@pytest.mark.parametrize("reference", ["triangle", "quadrilateral", "tetrahedron", "hexahedron"])
@pytest.mark.parametrize("order", range(1, 5))
def test_BDFM_space(reference, order):
    e = create_element(reference, "BDFM", order)
    polynomials = e.get_polynomial_basis()
    tdim = e.reference.tdim
    for p in polynomials:
        for vs in e.reference.sub_entities(tdim - 1):
            sub_ref = create_reference(
                e.reference.sub_entity_types[tdim - 1], [e.reference.vertices[i] for i in vs]
            )
            if tdim == 2:
                p_edge = p.subs(x, [i + t[0] * j for i, j in zip(sub_ref.origin, sub_ref.axes[0])])
            else:
                p_edge = p.subs(
                    x,
                    [
                        i + t[0] * j + t[1] * k
                        for i, j, k in zip(sub_ref.origin, sub_ref.axes[0], sub_ref.axes[1])
                    ],
                )
            poly = p_edge.dot(sub_ref.normal()).as_sympy().expand().simplify()
            assert poly.is_real or sympy.Poly(poly, x).degree() <= order - 1


@pytest.mark.parametrize(
    "reference",
    ["interval", "triangle", "quadrilateral", "tetrahedron", "hexahedron", "prism", "pyramid"],
)
@pytest.mark.parametrize("order", range(5))
def test_orthogonal_polynomials(reference, order, speed):
    if speed == "fast" and order > 2:
        pytest.skip()

    polynomials = orthogonal_basis(reference, order)
    ref = create_reference(reference)
    if len(polynomials) <= 5:
        for i, p in enumerate(polynomials):
            for q in polynomials[:i]:
                assert (p * q).integral(ref, x) == 0
    else:
        for _ in range(15):
            p = choice(polynomials)
            q = choice(polynomials)
            if p != q:
                assert (p * q).integral(ref, x) == 0


@pytest.mark.parametrize(
    "reference",
    ["interval", "triangle", "quadrilateral", "tetrahedron", "hexahedron", "prism", "pyramid"],
)
@pytest.mark.parametrize("order", range(3))
def test_orthonormal_polynomials(reference, order, speed):
    if speed == "fast" and order > 2:
        pytest.skip()

    polynomials = orthonormal_basis(reference, order)
    ref = create_reference(reference)
    for p in polynomials:
        assert (p**2).integral(ref, x) == 1


@pytest.mark.parametrize(
    "reference, order",
    [
        (r, o)
        for r, m in [
            ("interval", 6),
            ("triangle", 3),
            ("quadrilateral", 3),
            ("tetrahedron", 2),
            ("hexahedron", 2),
            ("prism", 2),
            ("pyramid", 2),
        ]
        for o in range(m)
    ],
)
def test_dual(reference, order):
    ref = create_reference(reference)
    poly = create_element(reference, "P", order).get_polynomial_basis()
    dual = l2_dual(reference, poly)

    for i, p in enumerate(poly):
        for j, d in enumerate(dual):
            assert (p * d).integrate(*ref.integration_limits(x)) == (1 if i == j else 0)


@pytest.mark.parametrize("degree", range(10))
def test_jacobi(degree):
    p = [0, 1]
    for n in range(1, degree + 1):
        # n*p_n = (2n-1)*p_{n-1} - (n-1)*P_{n-2}
        p = [p[1], ((2 * n - 1) * x[0] * p[1] - (n - 1) * p[0]) / n]

    assert allequal(jacobi_polynomial(degree, 0, 0), p[1])
