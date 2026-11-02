"""Checks for the remaining FIAT element families."""

import pytest
import sympy

import symfem
from symfem.symbols import x


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
