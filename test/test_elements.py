import pytest
import sympy
import symfem
from symfem import create_element
from symfem.core.finite_element import CiarletElement, DirectElement
from symfem.core.symbolic import subs, x, PiecewiseFunction
from symfem.core.vectors import vsub
from utils import test_elements, all_symequal


def test_all_tested():
    for e in symfem.create._elementlist:
        for r in e.references:
            if r == "dual polygon":
                continue
            for n in e.names:
                if n in test_elements[r]:
                    break
            else:
                raise ValueError(f"{e.names[0]} on a {r} is not tested")


@pytest.mark.parametrize(
    ("cell_type", "element_type", "order", "variant"),
    [[reference, element, order, variant]
     for reference, i in test_elements.items() for element, j in i.items()
     for variant, k in j.items() for order in k])
def test_independence(
    elements_to_test, cells_to_test, cell_type, element_type, order, variant,
    speed
):
    """Test that DirectElements have independent basis functions."""
    if elements_to_test != "ALL" and element_type not in elements_to_test:
        pytest.skip()
    if cells_to_test != "ALL" and cell_type not in cells_to_test:
        pytest.skip()
    if speed == "fast":
        if order > 2:
            pytest.skip()
        if order == 2 and cell_type in ["tetrahedron", "hexahedron", "prism", "pyramid"]:
            pytest.skip()

    space = create_element(cell_type, element_type, order, variant)

    # Only run this test for DirectElements
    if not isinstance(space, DirectElement):
        pytest.skip()

    basis = space.get_basis_functions()
    all_terms = set()

    try:
        basis[0].as_coefficients_dict()
        scalar = True
    except AttributeError:
        scalar = False

    if scalar:
        for f in basis:
            for term in f.as_coefficients_dict():
                all_terms.add(term)
        mat = [[0 for i in all_terms] for j in basis]
        for i, t in enumerate(all_terms):
            for j, f in enumerate(basis):
                fd = f.as_coefficients_dict()
                if t in fd:
                    mat[j][i] = fd[t]
    else:
        for f in basis:
            for fi, fpart in enumerate(f):
                for term in fpart.as_coefficients_dict():
                    all_terms.add((fi, term))
        mat = [[0 for i in all_terms] for j in basis]
        for i, (fi, t) in enumerate(all_terms):
            for j, f in enumerate(basis):
                fd = f[fi].as_coefficients_dict()
                if t in fd:
                    mat[j][i] = fd[t]
    mat = sympy.Matrix(mat)

    assert mat.rank() == mat.rows


@pytest.mark.parametrize(
    ("cell_type", "element_type", "order", "variant"),
    [[reference, element, order, variant]
     for reference, i in test_elements.items() for element, j in i.items()
     for variant, k in j.items() for order in k])
def test_element_functionals_and_continuity(
    elements_to_test, cells_to_test, cell_type, element_type, order, variant,
    speed
):
    if elements_to_test != "ALL" and element_type not in elements_to_test:
        pytest.skip()
    if cells_to_test != "ALL" and cell_type not in cells_to_test:
        pytest.skip()
    if speed == "fast":
        if order > 2:
            pytest.skip()
        if order == 2 and cell_type in ["tetrahedron", "hexahedron", "prism", "pyramid"]:
            pytest.skip()

    # Test functionals
    space = create_element(cell_type, element_type, order, variant)
    if isinstance(space, CiarletElement):
        for i, f in enumerate(space.get_basis_functions()):
            for j, d in enumerate(space.dofs):
                if i == j:
                    assert d.eval(f).expand().simplify() == 1
                else:
                    assert d.eval(f).expand().simplify() == 0
                assert d.entity_dim() is not None
    else:
        space.get_basis_functions()

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
    elif cell_type == "prism":
        vertices = ((-1, 0, 0), (0, 0, 0), (0, 1, 0),
                    (-1, 0, 1), (0, 0, 1), (0, 1, 1))
        entity_pairs = [[0, (0, 1)], [0, (2, 2)], [0, (3, 4)], [0, (5, 5)],
                        [1, (1, 3)], [1, (2, 4)], [1, (6, 6)], [1, (7, 8)],
                        [2, (2, 3)]]
    elif cell_type == "pyramid":
        vertices = ((-1, 0, 0), (0, 0, 0), (-1, 1, 0),
                    (0, 1, 0), (0, 0, 1))
        entity_pairs = [[0, (0, 1)], [0, (2, 3)], [0, (4, 4)],
                        [1, (1, 3)], [1, (2, 4)], [1, (6, 7)],
                        [2, (2, 4)]]

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
