import pytest

import symfem


@pytest.mark.parametrize("cell, order", [
    (c, o) for c, m in [("interval", 5), ("triangle", 4), ("quadrilateral", 4), ("tetrahedron", 3),
                        ("hexahedron", 3), ("prism", 3), ("pyramid", 3)] for o in range(m)])
def test_legendre(cell, order):
    basis = symfem.polynomials.orthonormal_basis(cell, order, 0)[0]
    e = symfem.create_element(cell, "P", order, variant="legendre")
    assert symfem.utils.allequal(e.get_basis_functions(), basis)


@pytest.mark.parametrize("cell, order", [
    (c, o) for c, m in [("interval", 5), ("quadrilateral", 4), ("hexahedron", 3)]
    for o in range(m)])
def test_lobatto(cell, order):
    basis = symfem.polynomials.lobatto_basis(cell, order, False)
    e = symfem.create_element(cell, "P", order, variant="lobatto")
    assert symfem.utils.allequal(e.get_basis_functions()[-len(basis):], basis)
