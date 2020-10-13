import pytest
from feast import feast_element


elements = []
elements += [("interval", e, o) for e in ["P", "Q", "S"] for o in [1, 2, 3]]
elements += [
    ("triangle", e, o)
    for e in ["P", "N1curl", "N1div", "N2curl", "N2div"]
    for o in [1, 2, 3]
]
elements += [
    ("tetrahedron", e, o)
    for e in ["P", "N1curl", "N1div", "N2curl", "N2div"]
    for o in [1, 2]
]
elements += [
    ("quadrilateral", e, o)
    for e in ["Q", "S", "Qcurl", "Qdiv", "Scurl", "Sdiv"]
    for o in [1, 2, 3]
]
elements += [
    ("hexahedron", e, o)
    for e in ["Q", "S", "Qcurl", "Qdiv", "Scurl", "Sdiv"]
    for o in [1, 2]
]


@pytest.mark.parametrize(("cell_type", "element_type", "order"), elements)
def test_elements(cell_type, element_type, order):
    space = feast_element(cell_type, element_type, order)
    for i, f in enumerate(space.get_basis_functions()):
        for j, d in enumerate(space.dofs):
            if i == j:
                assert d.eval(f) == 1
            else:
                assert d.eval(f) == 0
