from symfem import create_element
import numpy as np
import pytest

elements = {
    "interval": [("P", "Lagrange", range(1, 4), ["equispaced"]),
                 ("dP", "Discontinuous Lagrange", range(1, 4), []),
                 ("serendipity", "Serendipity", range(1, 5), []),
                 ("bubble", "Bubble", range(2, 5), []),
                 ("dP", "DPC", range(0, 5), [])],
    "triangle": [("P", "Lagrange", range(1, 4), ["equispaced"]),
                 ("dP", "Discontinuous Lagrange", range(1, 4), []),
                 ("bubble", "Bubble", range(3, 5), []),
                 ("N1curl", "Nedelec 1st kind H(curl)", range(1, 4), []),
                 ("N2curl", "Nedelec 2nd kind H(curl)", range(1, 4), []),
                 ("N1div", "Raviart-Thomas", range(1, 4), []),
                 ("N2div", "Brezzi-Douglas-Marini", range(1, 4), []),
                 ("Regge", "Regge", range(0, 4), []),
                 ("Crouzeix-Raviart", "Crouzeix-Raviart", [1], [])],
    "tetrahedron": [("P", "Lagrange", range(1, 4), ["equispaced"]),
                    ("dP", "Discontinuous Lagrange", range(1, 4), []),
                    ("bubble", "Bubble", range(4, 6), []),
                    ("N1curl", "Nedelec 1st kind H(curl)", range(1, 4), []),
                    ("N2curl", "Nedelec 2nd kind H(curl)", range(1, 3), []),
                    ("N1div", "Raviart-Thomas", range(1, 3), []),
                    ("N2div", "Brezzi-Douglas-Marini", range(1, 3), []),
                    ("Regge", "Regge", range(0, 3), []),
                    ("Crouzeix-Raviart", "Crouzeix-Raviart", [1], [])],
    "quadrilateral": [("Q", "Lagrange", range(1, 4), ["equispaced"]),
                      ("dQ", "Discontinuous Lagrange", range(1, 4), []),
                      ("dP", "DPC", range(0, 4), []),
                      ("serendipity", "Serendipity", range(1, 5), []),
                      ("Qdiv", "Raviart-Thomas", range(1, 4), []),
                      ("Qcurl", "Nedelec 1st kind H(curl)", range(1, 4), []),
                      ("Sdiv", "Brezzi-Douglas-Marini", range(1, 4), []),
                      ("Scurl", "Nedelec 2nd kind H(curl)", range(1, 4), [])],
    "hexahedron": [("Q", "Lagrange", range(1, 3), ["equispaced"]),
                   ("dQ", "Discontinuous Lagrange", range(1, 3), []),
                   ("serendipity", "Serendipity", range(1, 5), []),
                   ("Qdiv", "Raviart-Thomas", range(1, 3), []),
                   ("Qcurl", "Nedelec 1st kind H(curl)", range(1, 3), []),
                   ("Sdiv", "Brezzi-Douglas-Marini", range(1, 3), []),
                   ("Scurl", "Nedelec 2nd kind H(curl)", range(1, 3), [])],
    "prism": [("Lagrange", "Lagrange", range(1, 4), ["equispaced"])]
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


@pytest.mark.parametrize(("cell", "symfem_type", "basix_type", "order", "args"),
                         [(cell, a, b, order, args) for cell, ls in elements.items()
                          for a, b, orders, args in ls for order in orders])
def test_against_basix(has_basix, elements_to_test, cells_to_test, cell, symfem_type,
                       basix_type, order, args, speed):
    if elements_to_test != "ALL" and symfem_type not in elements_to_test:
        pytest.skip()
    if cells_to_test != "ALL" and cell not in cells_to_test:
        pytest.skip()
    if speed == "fast" and order > 2:
        pytest.skip()

    if has_basix:
        import basix
    else:
        try:
            import basix
        except ImportError:
            pytest.skip("Basix must be installed to run this test.")

    points = make_lattice(cell, 2)
    parsed_args = []
    for a in args:
        parsed_args.append(basix.lattice.string_to_type(a))
    space = basix.create_element(
        basix.finite_element.string_to_family(basix_type, cell),
        basix.cell.string_to_type(cell), order, *parsed_args)
    result = space.tabulate(0, points)[0]

    element = create_element(cell, symfem_type, order)
    sym_result = element.tabulate_basis(points, "xxyyzz", symbolic=False, use_legendre=True)

    assert np.allclose(result, sym_result)
