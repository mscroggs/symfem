from feast import feast_element
from feast.symbolic import subs, x
import libtab
import pytest
import numpy as np

elements = {
    "triangle": [("P", "Lagrange", range(1, 4)), ("N1curl", "Nedelec", range(1, 4)),
                 ("N1div", "RaviartThomas", range(1, 4))],
    "tetrahedron": [("P", "Lagrange", range(1, 4)), ("N1curl", "Nedelec", range(1, 3))]
}


def to_float(a):
    try:
        return float(a)
    except:  # noqa: E722
        return [to_float(i) for i in a]


@pytest.mark.parametrize(("cell", "feast_type", "libtab_type", "order"),
                         [(cell, a, b, order) for cell, ls in elements.items() for a, b, c in ls for order in c])
def test_against_libtab(cell, feast_type, libtab_type, order):
    element = feast_element(cell, feast_type, order)
    points = libtab.create_lattice(getattr(libtab.CellType, cell), 3, True)
    space = getattr(libtab, libtab_type)(getattr(libtab.CellType, cell), order)
    result = space.tabulate(0, points)[0]

    if element.range_dim == 1:
        basis = element.get_basis_functions()
        sym_result = [[float(subs(b, x, p)) for b in basis] for p in points]
    else:
        basis = element.get_basis_functions()
        sym_result = [[float(subs(b, x, p)[j]) for j in range(element.range_dim) for b in basis] for p in points]

    assert np.allclose(result, sym_result)
