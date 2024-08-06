import pytest

import symfem
from symfem.mappings import MappingNotImplemented
from symfem.references import NonDefaultReferenceError
from symfem.utils import allequal

from .utils import test_elements


@pytest.mark.parametrize(
    ("cell_type", "element_type", "order", "kwargs"),
    [
        [reference, element, order, kwargs]
        for reference, i in test_elements.items()
        for element, j in i.items()
        for kwargs, k in j
        for order in k
    ],
)
def test_push_forward(
    elements_to_test, cells_to_test, cell_type, element_type, order, kwargs, speed
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
        vertices = [(3,), (1,)]
    elif cell_type == "triangle":
        vertices = [(1, 1), (2, 2), (1, 4)]
    elif cell_type == "quadrilateral":
        vertices = [(1, 1), (2, 2), (1, 3), (2, 4)]
    elif cell_type == "tetrahedron":
        vertices = [(1, 1, 1), (2, 2, 2), (-1, 3, 2), (4, 0, 0)]
    elif cell_type == "hexahedron":
        vertices = [
            (1, 1, 1),
            (2, 2, 2),
            (-1, 3, 2),
            (0, 4, 3),
            (4, 0, 0),
            (5, 1, 1),
            (2, 2, 1),
            (3, 3, 2),
        ]
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


@pytest.mark.parametrize(
    "name, inverse, transpose, mapping",
    [
        ("identity", False, False, symfem.mappings.identity),
        ("l2", False, False, symfem.mappings.l2),
        ("covariant", False, False, symfem.mappings.covariant),
        ("contravariant", False, False, symfem.mappings.contravariant),
        ("double_covariant", False, False, symfem.mappings.double_covariant),
        ("double_contravariant", False, False, symfem.mappings.double_contravariant),
        ("identity", True, True, symfem.mappings.identity_inverse_transpose),
        ("l2", True, True, symfem.mappings.l2_inverse_transpose),
        ("covariant", True, True, symfem.mappings.covariant_inverse_transpose),
        ("contravariant", True, True, symfem.mappings.contravariant_inverse_transpose),
        ("identity", True, False, symfem.mappings.identity_inverse),
        ("l2", True, False, symfem.mappings.l2_inverse),
        ("covariant", True, False, symfem.mappings.covariant_inverse),
        ("contravariant", True, False, symfem.mappings.contravariant_inverse),
        ("double_covariant", True, False, symfem.mappings.double_covariant_inverse),
        ("double_contravariant", True, False, symfem.mappings.double_contravariant_inverse),
    ],
)
def test_get_mapping(name, inverse, transpose, mapping):
    assert symfem.mappings.get_mapping(name, inverse=inverse, transpose=transpose) == mapping
