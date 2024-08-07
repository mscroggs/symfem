"""Test every element."""

import pytest

from symfem.basix_interface import create_basix_element
from symfem import create_element
from symfem.symbols import x
from symfem.finite_element import DirectElement, EnrichedElement

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
def test_element(elements_to_test, cells_to_test, cell_type, element_type, order, kwargs, speed):
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

    if order == 0:
        pytest.skip()

    element = create_element(cell_type, element_type, order, **kwargs)
    if isinstance(element, (DirectElement, EnrichedElement)):
        pytest.xfail("Cannot convert this element type to a Basix element")
    basix_element = create_basix_element(element)
    if element.range_shape is None:
        assert basix_element.value_shape == []
    else:
        assert basix_element.value_shape == element.range_shape
