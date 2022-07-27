"""Test that basis functions agree with Basix."""

import typing

import pytest

from symfem import create_element

elements: typing.Dict[str, typing.List[typing.Tuple[
    str, str, typing.Iterable, typing.List[typing.Tuple[str, typing.Any]]
]]] = {
    "interval": [
        ("P", "Lagrange", range(1, 4), [("LagrangeVariant", "equispaced")]),
        ("serendipity", "Serendipity", range(1, 5), [("LagrangeVariant", "equispaced"),
                                                     ("DPCVariant", "simplex_equispaced")]),
        ("bubble", "Bubble", range(2, 5), []),
        ("dPc", "DPC", range(0, 5), [("LagrangeVariant", "equispaced"), ("bool", True)])],
    "triangle": [
        ("P", "Lagrange", range(1, 4), [("LagrangeVariant", "equispaced")]),
        ("bubble", "Bubble", range(3, 5), []),
        ("N1curl", "Nedelec 1st kind H(curl)", range(1, 4), [("LagrangeVariant", "equispaced")]),
        ("N2curl", "Nedelec 2nd kind H(curl)", range(1, 4), [("LagrangeVariant", "equispaced")]),
        ("N1div", "Raviart-Thomas", range(1, 4), [("LagrangeVariant", "equispaced")]),
        ("N2div", "Brezzi-Douglas-Marini", range(1, 4), [("LagrangeVariant", "equispaced")]),
        # ("Regge", "Regge", range(0, 4), []),
        # ("HHJ", "Hellan-Herrmann-Johnson", range(0, 4), []),
        ("Crouzeix-Raviart", "Crouzeix-Raviart", [1], [])],
    "tetrahedron": [
        ("P", "Lagrange", range(1, 4), [("LagrangeVariant", "equispaced")]),
        ("bubble", "Bubble", range(4, 6), []),
        ("N1curl", "Nedelec 1st kind H(curl)", range(1, 4), [("LagrangeVariant", "equispaced")]),
        ("N2curl", "Nedelec 2nd kind H(curl)", range(1, 3), [("LagrangeVariant", "equispaced")]),
        ("N1div", "Raviart-Thomas", range(1, 3), [("LagrangeVariant", "equispaced")]),
        ("N2div", "Brezzi-Douglas-Marini", range(1, 3), [("LagrangeVariant", "equispaced")]),
        # ("Regge", "Regge", range(0, 3), []),
        ("Crouzeix-Raviart", "Crouzeix-Raviart", [1], [])],
    "quadrilateral": [
        ("Q", "Lagrange", range(1, 4), [("LagrangeVariant", "equispaced")]),
        ("dPc", "DPC", range(0, 4), [("DPCVariant", "simplex_equispaced"), ("bool", True)]),
        ("serendipity", "Serendipity", range(1, 5),
         [("LagrangeVariant", "equispaced"), ("DPCVariant", "simplex_equispaced")]),
        ("Qdiv", "Raviart-Thomas", range(1, 4), [("LagrangeVariant", "equispaced")]),
        ("Qcurl", "Nedelec 1st kind H(curl)", range(1, 4), [("LagrangeVariant", "equispaced")]),
        ("Sdiv", "Brezzi-Douglas-Marini", range(1, 4),
         [("LagrangeVariant", "equispaced"), ("DPCVariant", "simplex_equispaced")]),
        ("Scurl", "Nedelec 2nd kind H(curl)", range(1, 4),
         [("LagrangeVariant", "equispaced"), ("DPCVariant", "simplex_equispaced")])],
    "hexahedron": [
        ("Q", "Lagrange", range(1, 3), [("LagrangeVariant", "equispaced")]),
        ("dPc", "DPC", range(0, 3),
         [("DPCVariant", "simplex_equispaced"), ("bool", True)]),
        ("serendipity", "Serendipity", range(1, 5),
         [("LagrangeVariant", "equispaced"), ("DPCVariant", "simplex_equispaced")]),
        ("Qdiv", "Raviart-Thomas", range(1, 3), [("LagrangeVariant", "equispaced")]),
        ("Qcurl", "Nedelec 1st kind H(curl)", range(1, 3), [("LagrangeVariant", "equispaced")]),
        ("Sdiv", "Brezzi-Douglas-Marini", range(1, 3),
         [("LagrangeVariant", "equispaced"), ("DPCVariant", "simplex_equispaced")]),
        ("Scurl", "Nedelec 2nd kind H(curl)", range(1, 3),
         [("LagrangeVariant", "equispaced"), ("DPCVariant", "simplex_equispaced")])],
    "prism": [
        ("Lagrange", "Lagrange", range(1, 4), [("LagrangeVariant", "equispaced")])]
}


def to_float(a):
    try:
        return float(a)
    except:  # noqa: E722
        return [to_float(i) for i in a]


def to_nparray(a):
    import numpy as np
    try:
        return float(a)
    except:  # noqa: E722
        return np.array([to_float(i) for i in a])


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

    # TODO: Implement faster non-symbolic mode and remove this
    if order > 1:
        pytest.skip()

    if symfem_type in ["Sdiv", "Scurl"]:
        pytest.xfail("Basix elements cannot yet be provided equispaced variant")

    if has_basix:
        import basix
    else:
        try:
            import basix
        except ImportError:
            pytest.skip("Basix must be installed to run this test.")

    import numpy as np

    points = make_lattice(cell, 2)
    parsed_args = []
    for a in args:
        if a[0] == "LagrangeVariant":
            parsed_args.append(basix.variants.string_to_lagrange_variant(a[1]))
        elif a[0] == "DPCVariant":
            parsed_args.append(basix.variants.string_to_dpc_variant(a[1]))
        elif a[0] == "bool":
            parsed_args.append(a[1])
        else:
            raise ValueError(f"Unknown arg type: {a[0]}")
    space = basix.create_element(
        basix.finite_element.string_to_family(basix_type, cell),
        basix.cell.string_to_type(cell), order, *parsed_args)
    result = space.tabulate(0, points)[0]

    element = create_element(cell, symfem_type, order)
    sym_result = to_nparray(element.tabulate_basis(points, "xyz,xyz"))

    if len(result.shape) != len(sym_result.shape):
        sym_result = sym_result.reshape(result.shape)

    assert np.allclose(result, sym_result)
