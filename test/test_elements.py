"""Test every element."""

import pytest
import sympy

import symfem
from symfem import create_element
from symfem.functions import VectorFunction
from symfem.symbols import x
from symfem.utils import allequal

from .utils import test_elements


def test_all_tested():
    for e in symfem.create._elementlist:
        for r in e.references:
            if r == "dual polygon":
                continue
            for n in e.names:
                if n in test_elements[r]:
                    break
            else:
                raise ValueError(f"{e.names[0]} on a {r} is not tested")


def test_caching():
    for e in symfem.create._elementlist:
        assert e.cache


@pytest.mark.parametrize(
    "ref, element, order", [("triangle", "Hermite", 4), ("tetrahedron", "Crouzeix-Raviart", 2)]
)
def test_too_high_order(ref, element, order):
    with pytest.raises(ValueError):
        symfem.create_element(ref, element, order)


@pytest.mark.parametrize(
    "ref, element, order", [("triangle", "Hermite", 2), ("tetrahedron", "bubble", 3)]
)
def test_too_low_order(ref, element, order):
    with pytest.raises(ValueError):
        symfem.create_element(ref, element, order)


@pytest.mark.parametrize(
    ("cell_type", "element_type", "order", "kwargs"),
    [
        [reference, element, order, kwargs]
        for reference, i in test_elements.items()
        for element, j in i.items()
        for kwargs, k in j
        for order in k
    ],
)
def test_element(elements_to_test, cells_to_test, cell_type, element_type, order, kwargs, speed):
    """Run tests for each element."""
    if elements_to_test != "ALL" and element_type not in elements_to_test:
        pytest.skip()
    if cells_to_test != "ALL" and cell_type not in cells_to_test:
        pytest.skip()
    if speed == "fast":
        if order > 2:
            pytest.skip()
        if order == 2 and cell_type in ["tetrahedron", "hexahedron", "prism", "pyramid"]:
            pytest.skip()

    space = create_element(cell_type, element_type, order, **kwargs)
    space.test()


@pytest.mark.parametrize("n_tri", [3, 4, 6, 8])
@pytest.mark.parametrize("order", range(2))
def test_dual_elements(elements_to_test, cells_to_test, n_tri, order):
    if elements_to_test != "ALL" and "dual" not in elements_to_test:
        pytest.skip()
    if cells_to_test != "ALL" and "dual polygon" not in cells_to_test:
        pytest.skip()

    space = create_element(f"dual polygon({n_tri})", "dual", order)
    sub_e = create_element("triangle", space.fine_space, space.order)
    for f, coeff_list in zip(space.get_basis_functions(), space.dual_coefficients):
        for piece, coeffs in zip(f.pieces.items(), coeff_list):
            fwd_map = VectorFunction(sub_e.reference.get_map_to(piece[0]))
            for dof, value in zip(sub_e.dofs, coeffs):
                point = fwd_map.subs(x, dof.point)
                assert allequal(value, piece[1].subs(x, point))


@pytest.mark.parametrize("n_tri", [3, 4])
@pytest.mark.parametrize("element_type", ["BC", "RBC"])
def test_bc_elements(elements_to_test, cells_to_test, n_tri, element_type):
    if elements_to_test != "ALL" and element_type not in elements_to_test:
        pytest.skip()
    if cells_to_test != "ALL" and "dual polygon" not in cells_to_test:
        pytest.skip()

    create_element(f"dual polygon({n_tri})", element_type, 0)


@pytest.mark.parametrize(
    ("cell_type", "element_type", "order", "kwargs"),
    [
        [reference, element, order, kwargs]
        for reference, i in test_elements.items()
        for element, j in i.items()
        for kwargs, k in j
        for order in k
    ],
)
def test_degree(elements_to_test, cells_to_test, cell_type, element_type, order, kwargs, speed):
    """Check that polynomial subdegree is the canonical degree of every element where possible."""
    if elements_to_test != "ALL" and element_type not in elements_to_test:
        pytest.skip()
    if cells_to_test != "ALL" and cell_type not in cells_to_test:
        pytest.skip()
    if speed == "fast":
        if order > 2:
            pytest.skip()
        if order == 2 and cell_type in ["tetrahedron", "hexahedron", "prism", "pyramid"]:
            pytest.skip()

    if element_type in ["bubble", "transition"]:
        pytest.xfail("Polynomial subdegree not used for this element")

    element = create_element(cell_type, element_type, order, **kwargs)
    try:
        poly_sub = element.polynomial_subdegree
    except NotImplementedError:
        pytest.xfail("Polynomial subdegree not implemented for this element")

    assert element.order == poly_sub


@pytest.fixture(
    scope="module",
    ids=lambda case: f"{case[1]}-{case[2]}-{case[3]}",
    params=[
        ("triangle", "robust Wu-Xu", 3, {}, 15, [3, 2, 0]),
        ("triangle", "Hu-Zhang", 3, {"variant": "point"}, 30, [3, 4, 9]),
        ("triangle", "Hu-Zhang", 3, {"variant": "integral"}, 30, [3, 4, 9]),
        ("triangle", "Hu-Zhang", 4, {"variant": "point"}, 45, [3, 6, 18]),
        ("triangle", "Bramble-Zlamal C2", 9, {}, 55, [15, 3, 1]),
        ("triangle", "Bramble-Zlamal C2", 10, {}, 66, [15, 6, 3]),
        ("triangle", "Alfeld C2", 5, {}, 27, [6, 3, 0]),
        ("triangle", "Alfeld C2", 6, {}, 37, [6, 6, 1]),
        ("tetrahedron", "Hu-Zhang-Zhang", 0, {}, 18, [3, 1, 0, 0]),
        ("tetrahedron", "Hu-Zhang-Zhang", 1, {}, 42, [3, 5, 0, 0]),
    ],
)
def element_case(request):
    cell, name, order, kwargs, dimension, counts = request.param
    return symfem.create_element(cell, name, order, **kwargs), dimension, counts


def test_dimension_and_unisolvence(element_case):
    element, dimension, counts = element_case
    assert element.space_dim == dimension
    for d, count in enumerate(counts):
        assert all(
            len(element.entity_dofs(d, i)) == count
            for i in range(element.reference.sub_entity_count(d))
        )
    assert element.get_dual_matrix().to_DM(extension=True).rank() == dimension


@pytest.mark.parametrize(
    "cell,name,degree",
    [
        ("triangle", "robust Wu-Xu", 2),
        ("tetrahedron", "robust Wu-Xu", 3),
        ("triangle", "Hu-Zhang", 2),
        ("quadrilateral", "Hu-Zhang", 3),
        ("triangle", "Bramble-Zlamal C2", 8),
        ("tetrahedron", "Bramble-Zlamal C2", 9),
        ("triangle", "Alfeld C2", 4),
        ("tetrahedron", "Alfeld C2", 5),
        ("triangle", "Hu-Zhang-Zhang", 0),
        ("tetrahedron", "Hu-Zhang-Zhang", 2),
    ],
)
def test_invalid(cell, name, degree):
    with pytest.raises(ValueError):
        symfem.create_element(cell, name, degree)


def test_hu_zhang_name_is_distinct():
    assert symfem.create_element("quadrilateral", "HZ", 1).range_shape == (2,)
    assert symfem.create_element("triangle", "Hu-Zhang", 3).range_shape == (2, 2)


def test_robust_wu_xu_edge_traces():
    element = symfem.create_element("triangle", "robust Wu-Xu", 3)
    for p in element.get_polynomial_basis():
        for i in range(3):
            edge = element.reference.sub_entity(1, i)
            t = sympy.Symbol("s")
            point = tuple(a + t * b for a, b in zip(edge.origin, edge.axes[0]))
            trace = p.as_sympy().subs(dict(zip(x, point)), simultaneous=True)
            assert sympy.Poly(trace, t).total_degree() <= 3


@pytest.mark.parametrize("power", [0, 1, 2])
def test_weighted_second_derivative_moment(power):
    from symfem.functionals import IntegralOfDirectionalMultiderivative
    from symfem.functions import ScalarFunction

    reference = symfem.create_reference("triangle")
    edge = reference.sub_entity(1, 2)
    dof = IntegralOfDirectionalMultiderivative(
        reference,
        (edge.normal(), edge.normal()),
        (1, 1),
        (1, 2),
        scale=1 / edge.jacobian(),
        weight=x[0] ** power,
    )
    assert dof.eval_symbolic(ScalarFunction(x[0] ** 2)) == sympy.Rational(1, power + 1)
