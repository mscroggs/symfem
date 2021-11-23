import pytest
import symfem
import sympy
from itertools import product
from symfem import create_element
from symfem.symbolic import subs, x, symequal, sym_product
from .utils import test_elements


def make_lattice(cell, N=3):
    if cell == "interval":
        return [[sympy.Rational(i, N)] for i in range(N + 1)]
    if cell == "triangle":
        return [[sympy.Rational(i, N), sympy.Rational(j, N)]
                for i in range(N + 1) for j in range(N + 1 - i)]
    if cell == "tetrahedron":
        return [[sympy.Rational(i, N), sympy.Rational(j, N), sympy.Rational(k, N)]
                for i in range(N + 1) for j in range(N + 1 - i) for k in range(N + 1 - i - j)]
    if cell == "quadrilateral":
        return [[sympy.Rational(i, N), sympy.Rational(j, N)]
                for i in range(N + 1) for j in range(N + 1)]
    if cell == "hexahedron":
        return [[sympy.Rational(i, N), sympy.Rational(j, N), sympy.Rational(k, N)]
                for i in range(N + 1) for j in range(N + 1) for k in range(N + 1)]
    if cell == "prism":
        return [[sympy.Rational(i, N), sympy.Rational(j, N), sympy.Rational(k, N)]
                for i in range(N + 1) for j in range(N + 1 - i) for k in range(N + 1)]
    if cell == "pyramid":
        return [[sympy.Rational(i, N), sympy.Rational(j, N), sympy.Rational(k, N)]
                for i in range(N + 1) for j in range(N + 1) for k in range(N + 1 - max(i, j))]


@pytest.mark.parametrize(
    ("cell_type", "element_type", "order", "kwargs"),
    [[reference, element, order, kwargs]
     for reference, i in test_elements.items() for element, j in i.items()
     for kwargs, k in j for order in k])
def test_element(
    elements_to_test, cells_to_test, cell_type, element_type, order, kwargs,
    speed
):
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

    element = create_element(cell_type, element_type, order, **kwargs)
    try:
        factorisation = element.get_tensor_factorisation()
    except symfem.finite_element.NoTensorProduct:
        pytest.skip("This element does not have a tensor product representation.")

    basis = element.get_basis_functions()

    factorised_basis = [None for i in basis]
    for t_type, factors, perm in factorisation:
        if t_type == "scalar":
            tensor_bases = [[subs(i, x[0], x_i) for i in f.get_basis_functions()]
                            for x_i, f in zip(x, factors)]
            for p, k in zip(perm, product(*tensor_bases)):
                factorised_basis[p] = sym_product(k)
        else:
            raise ValueError(f"Unknown tensor product type: {t_type}")

    for i, j in zip(basis, factorised_basis):
        assert symequal(i, j)
