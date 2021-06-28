import pytest
import numpy as np
from symfem import legendre, create_element
from symfem.symbolic import x, t, subs, to_float


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


max_orders = {"interval": 10, "triangle": 8, "tetrahedron": 5, "quadrilateral": 8, "hexahedron": 5,
              "prism": 5, "pyramid": 5}


@pytest.mark.parametrize(
    ("cell", "order"),
    [(c, o) for c in ["interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron",
                      "prism", "pyramid"] for o in range(1, max_orders[c])])
def test_legendre(cell, order):
    e = create_element(cell, "Lagrange", order)
    points = make_lattice(cell)

    basis = legendre.get_legendre_basis(e._basis, e.reference)
    values = legendre.evaluate_legendre_basis(points, e._basis, e.reference)

    if basis is None:
        assert values is None
        pytest.skip()

    for b in basis:
        assert b != 0

    values2 = np.array([[to_float(subs(b, x, p)) for b in basis] for p in points])

    assert np.allclose(values, values2)


@pytest.mark.parametrize("cell", ["interval", "triangle", "tetrahedron", "quadrilateral",
                                  "hexahedron", "prism", "pyramid"])
def test_orthogonal(cell):
    if cell == "interval":
        e = create_element(cell, "Lagrange", 5)
    else:
        e = create_element(cell, "Lagrange", 2)
    basis = legendre.get_legendre_basis(e._basis, e.reference)

    if basis is None:
        pytest.skip()

    for i, f in enumerate(basis):
        for g in basis[:i]:
            assert e.reference.integral(subs(f * g, x, t)) == 0
