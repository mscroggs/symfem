"""Test every element."""

import pytest
import sympy

import symfem
from symfem import create_element
from symfem.finite_element import CiarletElement
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


def _on_sub_entity(reference, dim, n, point):
    """Check if a point lies on a sub-entity of a reference."""
    sub = reference.sub_entity(dim, n)
    if dim == 0:
        return all(sympy.simplify(p - v) == 0 for p, v in zip(point, sub.vertices[0]))
    # Solve point = origin + sum_i s_i axes_i for s (the axes are linearly independent)
    s = sympy.symbols(f"s:{dim}")
    eqs = [
        p - o - sum(si * a[j] for si, a in zip(s, sub.axes))
        for j, (p, o) in enumerate(zip(point, sub.origin))
    ]
    sol = sympy.solve(eqs, s, dict=True)
    if len(sol) != 1 or len(sol[0]) != dim:
        return False
    values = [float(sol[0][si]) for si in s]
    tol = 1e-12
    if sub.name == "triangle":
        return all(v >= -tol for v in values) and sum(values) <= 1 + tol
    return all(-tol <= v <= 1 + tol for v in values)


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
def test_dof_entities(
    elements_to_test, cells_to_test, cell_type, element_type, order, kwargs, speed
):
    """Check that the DOF point of each DOF lies on the sub-entity the DOF is associated with."""
    if elements_to_test != "ALL" and element_type not in elements_to_test:
        pytest.skip()
    if cells_to_test != "ALL" and cell_type not in cells_to_test:
        pytest.skip()
    if speed == "fast":
        if order > 2:
            pytest.skip()
        if order == 2 and cell_type in ["tetrahedron", "hexahedron", "prism", "pyramid"]:
            pytest.skip()

    element = create_element(cell_type, element_type, order, **kwargs)
    if not isinstance(element, CiarletElement):
        pytest.skip("Only Ciarlet elements have DOFs associated with sub-entities")
    for dof in element.dofs:
        dim, n = dof.entity
        if dim < element.reference.tdim:
            assert _on_sub_entity(element.reference, dim, n, dof.dof_point())
