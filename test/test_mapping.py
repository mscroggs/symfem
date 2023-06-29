import pytest

import symfem
from symfem.mappings import MappingNotImplemented
from symfem.references import NonDefaultReferenceError
from symfem.utils import allequal

from .utils import test_elements


@pytest.mark.parametrize(
    ("cell_type", "element_type", "order", "kwargs"),
    [[reference, element, order, kwargs]
     for reference, i in test_elements.items() for element, j in i.items()
     for kwargs, k in j for order in k])
def test_push_forward(
    elements_to_test, cells_to_test, cell_type, element_type, order, kwargs,
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

    if cell_type == "interval":
        vertices = [(3, ), (1, )]
    elif cell_type == "triangle":
        vertices = [(1, 1), (2, 2), (1, 4)]
    elif cell_type == "quadrilateral":
        vertices = [(1, 1), (2, 2), (1, 3), (2, 4)]
    elif cell_type == "tetrahedron":
        vertices = [(1, 1, 1), (2, 2, 2), (-1, 3, 2), (4, 0, 0)]
    elif cell_type == "hexahedron":
        vertices = [(1, 1, 1), (2, 2, 2), (-1, 3, 2), (0, 4, 3),
                    (4, 0, 0), (5, 1, 1), (2, 2, 1), (3, 3, 2)]
    elif cell_type == "prism":
        vertices = [(1, 1, 1), (2, 2, 1), (1, 4, 2), (0, 1, 1), (1, 2, 1), (0, 4, 2)]
    elif cell_type == "pyramid":
        vertices = [(1, 1, 0), (2, 2, 0), (1, 3, 1), (2, 4, 1), (-1, -1, -1)]

    else:
        raise ValueError(f"Unsupported cell type: {cell_type}")

    try:
        e2 = symfem.create_element(cell_type, element_type, order, vertices=vertices, **kwargs)
    except NonDefaultReferenceError:
        pytest.xfail("Cannot create element on non-default reference.")
    e = symfem.create_element(cell_type, element_type, order, **kwargs)

    try:
        assert allequal(e.map_to_cell(vertices), e2.get_basis_functions())
    except MappingNotImplemented:
        pytest.xfail("Mapping not implemented for this element.")
