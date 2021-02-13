import pytest
import sympy
from symfem import create_element
from symfem.core.symbolic import subs, x, PiecewiseFunction
from symfem.core.vectors import vsub
from utils import test_elements, all_symequal


@pytest.mark.parametrize(
    ("cell_type", "element_type", "order", "variant"),
    [[reference, element, order, variant]
     for reference, i in test_elements.items() for element, j in i.items()
     for variant, k in j.items() for order in k])
def test_element_functionals_and_continuity(elements_to_test, cells_to_test,
                                            cell_type, element_type, order, variant):
    if elements_to_test != "ALL" and element_type not in elements_to_test:
        pytest.skip()
    if cells_to_test != "ALL" and cell_type not in cells_to_test:
        pytest.skip()

    # Test functionals
    space = create_element(cell_type, element_type, order, variant)
    for i, f in enumerate(space.get_basis_functions()):
        for j, d in enumerate(space.dofs):
            if i == j:
                assert d.eval(f).expand().simplify() == 1
            else:
                assert d.eval(f).expand().simplify() == 0
            assert d.entity_dim() is not None

    if order > 4:
        return  # For high order, testing continuity is slow

    # Test continuity
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

    if space.continuity == "L2":
        return

    for dim, entities in entity_pairs:
        for fi, gi in zip(*[space.entity_dofs(dim, i) for i in entities]):
            basis = space.get_basis_functions()
            basis2 = space.map_to_cell(vertices)
            f = basis[fi]
            g = basis2[gi]

            if isinstance(f, PiecewiseFunction):
                assert space.reference.tdim == 2
                f = f.get_piece((0, sympy.Rational(1, 2)))
                g = g.get_piece((0, sympy.Rational(1, 2)))

            f = subs(f, [x[0]], [0])
            g = subs(g, [x[0]], [0])

            if space.continuity == "C0":
                pass
            elif space.continuity == "C1":
                f = [f] + [f.diff(i) for i in x[:space.reference.tdim]]
                g = [g] + [g.diff(i) for i in x[:space.reference.tdim]]
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
            elif space.continuity == "inner H(div)":
                f = f[0]
                g = g[0]
            elif space.continuity == "integral inner H(div)":
                f = f[0].integrate((x[1], 0, 1))
                g = g[0].integrate((x[1], 0, 1))
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


@pytest.mark.parametrize("n_tri", [3, 4])
@pytest.mark.parametrize("element_type", ["BC", "RBC"])
def test_bc_elements(n_tri, element_type):
    create_element(f"dual polygon({n_tri})", element_type, 1)
