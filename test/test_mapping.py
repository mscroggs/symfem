import pytest
import symfem
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
    r = symfem.create_reference(cell_type, vertices=vertices)
    assert r.volume() not in [0, 1]

    e = symfem.create_element(cell_type, element_type, order, **kwargs)
    e2 = symfem.create_element(cell_type, element_type, order, vertices=vertices, **kwargs)

    print(e.map_to_cell(vertices))
    print(e2.get_basis_functions())
    a = e.map_to_cell(vertices)
    b = e2.get_basis_functions()

    assert allequal(e.map_to_cell(vertices), e2.get_basis_functions())
