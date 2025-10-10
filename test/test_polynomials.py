"""Test polynomials."""

from random import shuffle, randrange

import pytest
import sympy

from symfem import create_element, create_reference
from symfem.functions import VectorFunction, ScalarFunction
from symfem.polynomials import (
    Hcurl_polynomials,
    Hdiv_polynomials,
    jacobi_polynomial,
    monic_jacobi_polynomial,
    l2_dual,
    orthogonal_basis,
    orthonormal_basis,
    degree,
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
@pytest.mark.parametrize("order", range(4))
def test_orthogonal_polynomials(reference, order, speed):
    if speed == "fast" and order > 2:
        pytest.skip()

    polynomials = orthogonal_basis(reference, order)
    ref = create_reference(reference)
    shuffle(polynomials)
    for p in polynomials[:5]:
        for q in polynomials[-5:]:
            if p != q:
                assert (p * q).integral(ref, x) == 0


@pytest.mark.parametrize("order", range(2))
def test_orthogonal_polynomials_pyramid_full(order, speed):
    if speed == "fast" and order > 1:
        pytest.skip()

    polynomials = orthogonal_basis("pyramid", order, ptype="full")
    ref = create_reference("pyramid")
    for i, p in enumerate(polynomials):
        for q in polynomials[:i]:
            if p != q:
                assert (p * q).integral(ref, x) == 0

    polynomials = orthonormal_basis("pyramid", order, ptype="full")
    for p in polynomials:
        assert (p**2).integral(ref, x) == 1


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


@pytest.mark.parametrize("degree", range(10))
def test_monic_jacobi(degree):
    p = [sympy.Integer(0), sympy.Integer(1)]
    for n in range(1, degree + 1):
        # n*p_n = (2n-1)*p_{n-1} - (n-1)*P_{n-2}
        p = [p[1], ((2 * n - 1) * x[0] * p[1] - (n - 1) * p[0]) / n]

    coeff = p[1]
    for i in range(degree):
        coeff = coeff.diff(x[0]) / (degree - i)

    assert allequal(monic_jacobi_polynomial(degree, 0, 0), p[1] / coeff)


@pytest.mark.parametrize("monic", [True, False])
@pytest.mark.parametrize("a", range(5))
@pytest.mark.parametrize("b", range(5))
def test_jacobi_orthogonal(a, b, monic):
    if monic:
        poly = [monic_jacobi_polynomial(degree, a, b) for degree in range(5)]
    else:
        if b != 0:
            pytest.xfail("Non-monic Jacobi polynomials only implemented for b=0")
        poly = [jacobi_polynomial(degree, a, b) for degree in range(10)]

    for i, p in enumerate(poly):
        for q in poly[:i]:
            assert ((1 - x[0]) ** a * (1 + x[0]) ** b * p * q).integrate((x[0], -1, 1)) == 0


@pytest.mark.parametrize(("poly", "ref", "expected_degree"),
    [
        (x[0], "interval", 1),
        (x[0], "triangle", 1),
        (x[0], "quadrilateral", 1),
        (x[0], "tetrahedron", 1),
        (x[0], "hexahedron", 1),
        (x[0] * x[1], "triangle", 2),
        (x[0] * x[1], "quadrilateral", 1),
        (x[0] * x[1], "tetrahedron", 2),
        (x[0] * x[1], "hexahedron", 1),
        (x[0] * x[1] * (x[0] + x[1]), "triangle", 3),
        (x[0] * x[1] * (x[0] + x[1]), "quadrilateral", 2),
        (x[0] * x[1] * (x[0] + x[1]), "tetrahedron", 3),
        (x[0] * x[1] * (x[0] + x[1]), "hexahedron", 2),
    ]
)
def test_degree(poly, ref, expected_degree):
    assert degree(create_reference(ref), ScalarFunction(poly)) == expected_degree


@pytest.mark.parametrize(("ref", "dfunc"), [
    ("tetrahedron", lambda p, q, r: p + q + r),
    ("hexahedron", max),
    ("prism", lambda p, q, r: max(p, q) + r),
])
def test_random_degrees(ref, dfunc):
    cell = create_reference(ref)
    for _ in range(20):
        p = randrange(20)
        q = randrange(20)
        r = randrange(20)
        poly = ScalarFunction(x[0] ** p * x[1] ** q * x[2] ** r)

        assert degree(cell, poly) == dfunc(p, q, r)


def test_random_degrees_pyramid():
    pyr = create_reference("pyramid")
    for _ in range(20):
        p = randrange(20)
        q = randrange(20)
        r = randrange(20)
        poly = ScalarFunction((x[0] / (1 - x[2])) ** p * (x[1] / (1 - x[2])) ** q * x[2] ** r)

        assert degree(pyr, poly) == max(p, q, r)
