from symfem import create_element
from symfem.core.symbolic import subs, x
import pytest

elements = {
    "interval": [("P", "Lagrange", range(1, 4)), ("dP", "Discontinuous Lagrange", range(1, 4)),
                 ("serendipity", "Serendipity", range(1, 5)),
                 ("bubble", "Bubble", range(2, 5)),
                 ("dP", "DPC", range(0, 5))],
    "triangle": [("P", "Lagrange", range(1, 4)), ("dP", "Discontinuous Lagrange", range(1, 4)),
                 ("bubble", "Bubble", range(3, 5)),
                 ("N1curl", "Nedelec 1st kind H(curl)", range(1, 4)),
                 ("N2curl", "Nedelec 2nd kind H(curl)", range(1, 4)),
                 ("N1div", "Raviart-Thomas", range(1, 4)),
                 ("N2div", "Brezzi-Douglas-Marini", range(1, 4)),
                 ("Regge", "Regge", range(0, 4)),
                 ("Crouzeix-Raviart", "Crouzeix-Raviart", [1])],
    "tetrahedron": [("P", "Lagrange", range(1, 4)), ("dP", "Discontinuous Lagrange", range(1, 4)),
                    ("bubble", "Bubble", range(4, 6)),
                    ("N1curl", "Nedelec 1st kind H(curl)", range(1, 4)),
                    ("N2curl", "Nedelec 2nd kind H(curl)", range(1, 3)),
                    ("N1div", "Raviart-Thomas", range(1, 3)),
                    ("N2div", "Brezzi-Douglas-Marini", range(1, 3)),
                    ("Regge", "Regge", range(0, 3)),
                    ("Crouzeix-Raviart", "Crouzeix-Raviart", [1])],
    "quadrilateral": [("Q", "Lagrange", range(1, 4)),
                      ("dQ", "Discontinuous Lagrange", range(1, 4)),
                      ("dP", "DPC", range(0, 4)),
                      ("serendipity", "Serendipity", range(1, 5)),
                      ("Qdiv", "Raviart-Thomas", range(1, 4)),
                      ("Qcurl", "Nedelec 1st kind H(curl)", range(1, 4)),
                      ("Sdiv", "Brezzi-Douglas-Marini", range(1, 4)),
                      ("Scurl", "Nedelec 2nd kind H(curl)", range(1, 4))],
    "hexahedron": [("Q", "Lagrange", range(1, 3)), ("dQ", "Discontinuous Lagrange", range(1, 3)),
                   ("serendipity", "Serendipity", range(1, 5)),
                   ("Qdiv", "Raviart-Thomas", range(1, 3)),
                   ("Qcurl", "Nedelec 1st kind H(curl)", range(1, 3)),
                   ("Sdiv", "Brezzi-Douglas-Marini", range(1, 3)),
                   ("Scurl", "Nedelec 2nd kind H(curl)", range(1, 3))],
    "prism": [("Lagrange", "Lagrange", range(1, 4))]
}


def to_float(a):
    try:
        return float(a)
    except:  # noqa: E722
        return [to_float(i) for i in a]


def make_lattice(cell, N=3):
    import numpy as np
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
    if cell == "prism":
        return np.array([[i / N, j / N, k / N]
                         for i in range(N + 1) for j in range(N + 1 - i) for k in range(N + 1)])
    if cell == "pyramid":
        return np.array([[i / N, j / N, k / N]
                         for i in range(N + 1) for j in range(N + 1)
                         for k in range(N + 1 - max(i, j))])


@pytest.mark.parametrize(("cell", "symfem_type", "basix_type", "order"),
                         [(cell, a, b, order) for cell, ls in elements.items()
                          for a, b, c in ls for order in c])
def test_against_basix(has_basix, elements_to_test, cells_to_test, cell, symfem_type,
                       basix_type, order, speed):
    if elements_to_test != "ALL" and symfem_type not in elements_to_test:
        pytest.skip()
    if cells_to_test != "ALL" and cell not in cells_to_test:
        pytest.skip()
    if speed == "fast" and order > 2:
        pytest.skip()

    if has_basix:
        import basix
        from scipy.linalg import block_diag, solve
        import numpy as np
    else:
        try:
            import basix
            from scipy.linalg import block_diag, solve
            import numpy as np
        except ImportError:
            pytest.skip("Basix, numpy and scipy must be installed to run this test.")

    element = create_element(cell, symfem_type, order)
    points = make_lattice(cell, 2)
    space = basix.create_element(basix_type, cell, order)
    result = space.tabulate(0, points)[0]

    mat = element.get_dual_matrix()
    mat = np.array([[float(j) for j in mat.row(i)] for i in range(mat.rows)])

    if element.range_dim == 1:
        evaluated = np.array([[float(subs(b, x, p)) for p in points] for b in element.basis])
    else:
        evaluated = np.array([[float(subs(b, x, p)[j]) for p in points]
                              for j in range(element.range_dim) for b in element.basis])
        mat = block_diag(*[mat for j in range(element.range_dim)])

    sym_result = solve(mat, evaluated).transpose()
    assert np.allclose(result, sym_result)
