"""Test every element."""

import pytest

import symfem
import sympy
from symfem import create_element
from symfem.functions import VectorFunction, AnyFunction
from symfem.symbols import x
from symfem.utils import allequal
from symfem.polynomials import polynomial_set, polynomial_set_1d

from .utils import test_elements


def make_into_value_type(poly, element):
    if element.value_type == "scalar":
        return poly

    if element.value_type == "vector":
        return [VectorFunction([p if i == j else 0 for j in range(element.range_dim)]) for i in range(element.range_dim) for p in poly]


def polydict(f, element):
    if element.reference.name == "pyramid":
        f *= (1 - x[2]) ** element.lagrange_superdegree

    if element.value_type == "scalar":
        return f.as_sympy().expand().as_coefficients_dict()
    if element.value_type == "vector":
        out = {}
        for i, f_i in enumerate(f.as_sympy()):
            for term, coeff in f_i.expand().as_coefficients_dict().items():
                t = tuple(term if i == j else 0 for j, _ in enumerate(f.as_sympy()))
                assert t not in out
                out[t] = coeff
        return out
    raise ValueError(f"Unupported function type: {element.value_type}")


def run_subdegree_test(element, subdegree, pfunc, pargs):
    try:
        basis_coeff = [polydict(p, element) for p in element.get_polynomial_basis()]
    except NotImplementedError:
        basis_coeff = [polydict(p, element) for p in element.get_basis_functions()]

    if subdegree > -1:
        poly = pfunc(*[subdegree if p == "SUBDEGREE" else p for p in pargs])
        print(poly)
        print(make_into_value_type(poly, element))
        poly_coeff = [polydict(p, element) for p in make_into_value_type(poly, element)]

        monomials = list(set([m for p in poly_coeff + basis_coeff for m in p]))

        mat = sympy.Matrix([[p[m] if m in p else 0 for m in monomials] for p in basis_coeff])
        mat2 = sympy.Matrix([[p[m] if m in p else 0 for m in monomials] for p in poly_coeff + basis_coeff])

        assert mat.rank() == mat2.rank()

    poly = pfunc(*[subdegree + 1 if p == "SUBDEGREE" else p for p in pargs])
    poly_coeff = [polydict(p, element) for p in make_into_value_type(poly, element)]

    monomials = list(set([m for p in poly_coeff + basis_coeff for m in p]))

    mat = sympy.Matrix([[p[m] if m in p else 0 for m in monomials] for p in basis_coeff])
    mat2 = sympy.Matrix([[p[m] if m in p else 0 for m in monomials] for p in poly_coeff + basis_coeff])

    assert mat.rank() != mat2.rank()


def run_superdegree_test(element, superdegree, pfunc, pargs):
    try:
        basis_coeff = [polydict(p, element) for p in element.get_polynomial_basis()]
    except NotImplementedError:
        basis_coeff = [polydict(p, element) for p in element.get_basis_functions()]

    if superdegree is not None:
        poly = pfunc(*[superdegree if p == "SUPERDEGREE" else p for p in pargs])
        poly_coeff = [polydict(p, element) for p in make_into_value_type(poly, element)]

        monomials = list(set([m for p in poly_coeff + basis_coeff for m in p]))

        mat = sympy.Matrix([[p[m] if m in p else 0 for m in monomials] for p in poly_coeff])
        mat2 = sympy.Matrix([[p[m] if m in p else 0 for m in monomials] for p in poly_coeff + basis_coeff])

        assert mat.rank() == mat2.rank()

    poly = pfunc(*[(10 if superdegree is None else superdegree - 1) if p == "SUPERDEGREE" else p for p in pargs])
    poly_coeff = [polydict(p, element) for p in make_into_value_type(poly, element)]

    monomials = list(set([m for p in poly_coeff + basis_coeff for m in p]))

    mat = sympy.Matrix([[p[m] if m in p else 0 for m in monomials] for p in poly_coeff])
    mat2 = sympy.Matrix([[p[m] if m in p else 0 for m in monomials] for p in poly_coeff + basis_coeff])

    assert mat.rank() != mat2.rank()


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
def test_lagrange_subdegree(elements_to_test, cells_to_test, cell_type, element_type, order, kwargs, speed):
    """Check Lagrange subdegrees of each element."""
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
    run_subdegree_test(element, element.lagrange_subdegree, polynomial_set, (cell_type, "SUBDEGREE"))


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
def test_polynomial_subdegree(elements_to_test, cells_to_test, cell_type, element_type, order, kwargs, speed):
    """Check polynomial subdegrees of each element."""
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
    run_subdegree_test(element, element.polynomial_subdegree, polynomial_set_1d, (element.reference.tdim, "SUBDEGREE"))


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
def test_lagrange_superdegree(elements_to_test, cells_to_test, cell_type, element_type, order, kwargs, speed):
    """Check Lagrange superdegrees of each element."""
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
    run_superdegree_test(element, element.lagrange_superdegree, polynomial_set, (cell_type, "SUPERDEGREE"))


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
def test_polynomial_superdegree(elements_to_test, cells_to_test, cell_type, element_type, order, kwargs, speed):
    """Check polynomial superdegrees of each element."""
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
    run_superdegree_test(element, element.polynomial_superdegree, polynomial_set_1d, (element.reference.tdim, "SUPERDEGREE"))

