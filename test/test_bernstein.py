"""Test Bernstein elements."""

import pytest

import symfem


@pytest.mark.parametrize("celltype, degree", [(c, i) for c, n in [
    ("interval", 4), ("triangle", 3), ("tetrahedron", 2)
] for i in range(n)])
def test_bernstein(celltype, degree):
    b = symfem.create_element(celltype, "Bernstein", degree)
    poly = symfem.elements.bernstein.bernstein_polynomials(degree, b.reference.tdim)

    for f in b.get_basis_functions():
        for p in poly:
            if f == p:
                poly.remove(p)
                break
        else:
            raise ValueError(f"{f} is not a Bernstein polynomial.")

    assert len(poly) == 0
