import pytest

import symfem


@pytest.mark.parametrize(
    "cell",
    ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron", "prism", "pyramid"],
)
@pytest.mark.parametrize("order", range(1, 4))
def test_lagrange(cell, order):
    e = symfem.create_element(cell, "P", order)
    assert e.maximum_degree == order


@pytest.mark.parametrize("cell", ["triangle", "tetrahedron"])
@pytest.mark.parametrize("order", range(1, 4))
def test_rt(cell, order):
    e = symfem.create_element(cell, "RT", order)
    assert e.maximum_degree == order


@pytest.mark.parametrize("cell", ["quadrilateral", "hexahedron"])
@pytest.mark.parametrize("order", range(1, 4))
def test_rt_tp(cell, order):
    e = symfem.create_element(cell, "RTCF", order)
    assert e.maximum_degree == order


@pytest.mark.parametrize("cell", ["triangle", "tetrahedron"])
@pytest.mark.parametrize("order", range(1, 4))
def test_regge(cell, order):
    e = symfem.create_element(cell, "Regge", order)
    assert e.maximum_degree == order


@pytest.mark.parametrize("cell", ["quadrilateral", "hexahedron"])
@pytest.mark.parametrize("order", range(1, 4))
def test_regge_tp(cell, order):
    e = symfem.create_element(cell, "Regge", order)
    assert e.maximum_degree == order + 1
