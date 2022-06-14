import symfem
import pytest


@pytest.mark.parametrize("degree", range(4))
def test_interval(degree):
    b = symfem.create_element("interval", "Bernstein", degree)
    poly = symfem.elements.bernstein.bernstein_polynomials(degree, 1)

    for i, j in zip(b.get_basis_functions(), poly):
        assert (i - j).simplify() == 0


@pytest.mark.parametrize("degree", range(3))
def test_triangle(degree):
    b = symfem.create_element("triangle", "Bernstein", degree)
    poly = symfem.elements.bernstein.bernstein_polynomials(degree, 2)

    for i, j in zip(b.get_basis_functions(), poly):
        assert (i - j).simplify() == 0


@pytest.mark.parametrize("degree", range(2))
def test_tetrahedron(degree):
    b = symfem.create_element("tetrahedron", "Bernstein", degree)
    poly = symfem.elements.bernstein.bernstein_polynomials(degree, 3)

    for i, j in zip(b.get_basis_functions(), poly):
        assert (i - j).simplify() == 0
