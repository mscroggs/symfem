import pytest
import sympy
from symfem import create_element, _elementlist
from symfem.core.symbolic import subs, x
from symfem.core.vectors import vsub

# Set the max orders for elements that are slow to get basis functions for
max_orders_getting_basis = {
    "triangle": {
        "Brezzi-Douglas-Marini": 4,
        "Nedelec": 3,
        "Nedelec2": 3,
        "Regge": 3,
        "Raviart-Thomas": 3,
        "vector Lagrange": 4,
        "vector discontinuous Lagrange": 4,
    },
    "tetrahedron": {
        "Brezzi-Douglas-Fortin-Marini": 3,
        "Brezzi-Douglas-Marini": 3,
        "Lagrange": 4,
        "Nedelec": 3,
        "Nedelec2": 3,
        "Raviart-Thomas": 3,
        "Regge": 3,
        "vector Lagrange": 3,
        "vector discontinuous Lagrange": 3,
    },
    "quadrilateral": {
        "Brezzi-Douglas-Fortin-Marini": 3,
        "NCE": 3,
        "NCF": 3,
        "serendipity": 4,
        "serendipity Hdiv": 3,
        "serendipity Hcurl": 3,
    },
    "hexahedron": {
        "Brezzi-Douglas-Fortin-Marini": 3,
        "NCE": 3,
        "NCF": 3,
        "Q": 3,
        "dQ": 3,
        "vector Q": 3,
        "serendipity": 3,
        "serendipity Hdiv": 2,
        "serendipity Hcurl": 2,
        "vector discontinuous Lagrange": 3,
    }
}


def all_symequal(a, b):
    if isinstance(a, (list, tuple)):
        for i, j in zip(a, b):
            if not all_symequal(i, j):
                return False
        return True
    return sympy.expand(sympy.simplify(a)) == sympy.expand(sympy.simplify(b))


def elements(max_order=5, include_dual=True, include_non_dual=True,
             getting_basis=False):
    out = []
    for e in _elementlist:
        for r in e.references:
            if r == "dual polygon" and not include_dual:
                continue
            if r != "dual polygon" and not include_non_dual:
                continue
            if hasattr(e, "min_order"):
                min_o = e.min_order
                if isinstance(min_o, dict):
                    min_o = min_o[r]
            else:
                min_o = 0
            if hasattr(e, "max_order"):
                max_o = e.max_order
                if isinstance(max_o, dict):
                    max_o = max_o[r]
            else:
                max_o = 100
            print(r, e.names[0])
            if r in max_orders_getting_basis and e.names[0] in max_orders_getting_basis[r]:
                max_o = min(max_orders_getting_basis[r][e.names[0]], max_o)

            for order in range(min_o, min(max_order, max_o) + 1):
                if r == "dual polygon":
                    for n_tri in range(3, 7):
                        out.append((f"{r}({n_tri})", e.names[0], order))
                else:
                    out.append((r, e.names[0], order))
    return out


@pytest.mark.parametrize(("cell_type", "element_type", "order"),
                         elements(max_order=5, include_dual=False, getting_basis=True))
def test_element_functionals(elements_to_test, cells_to_test, cell_type, element_type, order):
    if elements_to_test != "ALL" and element_type not in elements_to_test:
        pytest.skip()
    if cells_to_test != "ALL" and cell_type not in cells_to_test:
        pytest.skip()

    space = create_element(cell_type, element_type, order)
    for i, f in enumerate(space.get_basis_functions()):
        for j, d in enumerate(space.dofs):
            if i == j:
                assert d.eval(f) == 1
            else:
                assert d.eval(f) == 0
            assert d.entity_dim() is not None


@pytest.mark.parametrize(("cell_type", "element_type", "order"),
                         elements(max_order=3, include_dual=False, getting_basis=True))
def test_element_continuity(elements_to_test, cells_to_test, cell_type, element_type, order):
    if elements_to_test != "ALL" and element_type not in elements_to_test:
        pytest.skip()
    if cells_to_test != "ALL" and cell_type not in cells_to_test:
        pytest.skip()

    if cell_type == "interval":
        vertices = ((-1, ), (0, ))
        entity_pairs = [[0, (0, 1)]]
    elif cell_type == "triangle":
        vertices = ((-1, 2), (0, 0), (0, 1))
        entity_pairs = [[0, (0, 1)], [0, (2, 2)], [1, (1, 0)]]
    elif cell_type == "tetrahedron":
        vertices = ((-1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 1))
        entity_pairs = [[0, (0, 1)], [0, (2, 2)], [0, (3, 3)],
                        [1, (0, 0)], [1, (3, 1)], [1, (4, 2)],
                        [2, (1, 0)]]
    elif cell_type == "quadrilateral":
        vertices = ((0, 0), (0, 1), (-1, 0), (-1, 1))
        entity_pairs = [[0, (0, 0)], [0, (2, 1)], [1, (1, 0)]]
    elif cell_type == "hexahedron":
        vertices = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                    (-1, 0, 0), (-1, 0, 1), (-1, 1, 0), (-1, 1, 1))
        entity_pairs = [[0, (0, 0)], [0, (2, 2)], [0, (4, 1)], [0, (6, 3)],
                        [1, (1, 1)], [1, (2, 0)], [1, (6, 5)], [1, (9, 3)],
                        [2, (0, 2)]]

    space = create_element(cell_type, element_type, order)
    if space.continuity == "L2":
        return

    for dim, entities in entity_pairs:
        for fi, gi in zip(*[space.entity_dofs(dim, i) for i in entities]):
            basis = space.get_basis_functions()
            f = basis[fi]
            g = space.map_to_cell(basis[gi], vertices)

            f = subs(f, [x[0]], [0])
            g = subs(g, [x[0]], [0])

            if space.continuity == "C0":
                pass
            elif space.continuity == "H(div)":
                f = f[0]
                g = g[0]
            elif space.continuity == "H(curl)":
                f = f[1:]
                g = g[1:]
            elif space.continuity == "inner H(curl)":
                if len(vertices[0]) == 2:
                    f = f[3]
                    g = g[3]
                if len(vertices[0]) == 3:
                    if dim == 1:
                        vs = space.reference.sub_entities(1)[entities[0]]
                        v0 = space.reference.vertices[vs[0]]
                        v1 = space.reference.vertices[vs[1]]
                        tangent = vsub(v1, v0)
                        f = sum(i * f[ni * len(tangent) + nj] * j
                                for ni, i in enumerate(tangent)
                                for nj, j in enumerate(tangent))
                        g = sum(i * g[ni * len(tangent) + nj] * j
                                for ni, i in enumerate(tangent)
                                for nj, j in enumerate(tangent))
                    else:
                        assert dim == 2
                        f = [f[4], f[8]]
                        g = [g[4], g[8]]
            else:
                raise ValueError(f"Unknown continuity: {space.continuity}")

            assert all_symequal(f, g)


@pytest.mark.parametrize("n_tri", [3, 4, 6, 8])
@pytest.mark.parametrize("order", range(2))
def test_dual_elements(n_tri, order):
    space = create_element(f"dual polygon({n_tri})", "dual", order)

    sub_e = create_element("triangle", space.fine_space, space.order)
    for f, coeff_list in zip(space.get_basis_functions(), space.dual_coefficients):
        for piece, coeffs in zip(f.pieces, coeff_list):
            map = sub_e.reference.get_map_to(piece[0])
            for dof, value in zip(sub_e.dofs, coeffs):
                point = subs(map, x, dof.point)
                assert all_symequal(value, subs(piece[1], x, point))
