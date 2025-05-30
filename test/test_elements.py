"""Test every element."""

import pytest

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
