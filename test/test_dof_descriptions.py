"""Test DOF descriptions."""

import pytest

from symfem import create_element
from symfem.finite_element import CiarletElement

from .utils import test_elements


@pytest.mark.parametrize(
    ("cell_type", "element_type", "order", "kwargs"),
    [[reference, element, order, kwargs]
     for reference, i in test_elements.items() for element, j in i.items()
     for kwargs, k in j for order in k])
def test_element(
    elements_to_test, cells_to_test, cell_type, element_type, order, kwargs,
    speed
):
    """Run tests for each element."""
    if elements_to_test != "ALL" and element_type not in elements_to_test:
        pytest.skip()
    if cells_to_test != "ALL" and cell_type not in cells_to_test:
        pytest.skip()
    if speed == "fast":
        if order > 2:
            pytest.skip()
        if order == 2 and cell_type in ["tetrahedron", "hexahedron", "prism", "pyramid"]:
            pytest.skip()

    element = create_element(cell_type, element_type, order, **kwargs)
    if isinstance(element, CiarletElement):
        for d in element.dofs:
            print(d.get_tex()[0])
