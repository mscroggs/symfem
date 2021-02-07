import pytest
from symfem import create_element


@pytest.mark.parametrize("order", range(3, 7))
def test_create(order):
    create_element("triangle", "Arnold-Winther", order)
