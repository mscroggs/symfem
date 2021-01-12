from symfem import create_element
from symfem.core.symbolic import subs, x
import pytest
import numpy as np

elements = {
    "interval": [("P", "Lagrange", range(1, 4)), ("dP", "Discontinuous Lagrange", range(1, 4))],
    "triangle": [("P", "Lagrange", range(1, 4)), ("dP", "Discontinuous Lagrange", range(1, 4)),
                 ("N1curl", "Nedelec 1st kind H(curl)", range(1, 4)),
                 ("N2curl", "Nedelec 2nd kind H(curl)", range(1, 4)),
                 ("N1div", "Raviart-Thomas", range(1, 4)),
                 ("N2div", "Brezzi-Douglas-Marini", range(1, 4)),
                 ("Regge", "Regge", range(0, 4)),
                 ("Crouzeix-Raviart", "Crouzeix-Raviart", [1])],
    "tetrahedron": [("P", "Lagrange", range(1, 4)), ("dP", "Discontinuous Lagrange", range(1, 4)),
                    ("N1curl", "Nedelec 1st kind H(curl)", range(1, 3)),
                    ("N2curl", "Nedelec 2nd kind H(curl)", range(1, 3)),
                    ("N1div", "Raviart-Thomas", range(1, 3)),
                    ("N2div", "Brezzi-Douglas-Marini", range(1, 3)),
                    ("Regge", "Regge", range(0, 3)),
                    ("Crouzeix-Raviart", "Crouzeix-Raviart", [1])],
    "quadrilateral": [("Q", "Lagrange", range(1, 4)),
                      ("dQ", "Discontinuous Lagrange", range(1, 4))],
    "hexahedron": [("Q", "Lagrange", range(1, 4)), ("dQ", "Discontinuous Lagrange", range(1, 4))]
}


def to_float(a):
    try:
        return float(a)
    except:  # noqa: E722
        return [to_float(i) for i in a]


def make_lattice(cell, N=3):
    if cell == "interval":
        return np.array([[i / N] for i in range(N + 1)])
    if cell == "triangle":
        return np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)])
    if cell == "tetrahedron":
        return np.array([[i / N, j / N, k / N]
                         for i in range(N + 1) for j in range(N + 1 - i)
                         for k in range(N + 1 - i - j)])
    if cell == "quadrilateral":
        return np.array([[i / N, j / N] for i in range(N + 1) for j in range(N + 1)])
    if cell == "hexahedron":
        return np.array([[i / N, j / N, k / N]
                         for i in range(N + 1) for j in range(N + 1) for k in range(N + 1)])


@pytest.mark.parametrize(("cell", "symfem_type", "basix_type", "order"),
                         [(cell, a, b, order) for cell, ls in elements.items()
                          for a, b, c in ls for order in c])
def test_against_basix(cell, symfem_type, basix_type, order):
    try:
        import basix
    except ImportError:
        pytest.skip("basix must be installed to run this test.")

    element = create_element(cell, symfem_type, order)
    points = make_lattice(cell)
    space = basix.create_element(basix_type, cell, order)
    result = space.tabulate(0, points)[0]

    if element.range_dim == 1:
        basis = element.get_basis_functions(False)
        sym_result = [[float(subs(b, x, p)) for b in basis] for p in points]
    else:
        basis = element.get_basis_functions(False)
        sym_result = [[float(subs(b, x, p)[j]) for j in range(element.range_dim) for b in basis]
                      for p in points]
    assert np.allclose(result, sym_result)
