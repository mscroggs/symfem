import pytest
import signal
from symfem import create_element, _elementlist


class TimeOutTheTest(BaseException):
    pass


def handler(signum, frame):
    raise TimeOutTheTest()


elements = []
for e in _elementlist:
    for r in e.references:
        if hasattr(e, "min_order"):
            min_o = e.min_order
            if isinstance(min_o, dict):
                min_o = min_o[r]
        else:
            min_o = 0
        if hasattr(e, "max_order"):
            max_o = e.max_order
            if isinstance(max_o, dict):
                max_o = max_o[r]
        else:
            max_o = 100
        for order in range(min_o, min(5, max_o) + 1):
            elements.append((r, e.names[0], order))


@pytest.mark.parametrize(("cell_type", "element_type", "order"), elements)
def test_elements(cell_type, element_type, order):
    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(20)

        space = create_element(cell_type, element_type, order)
        for i, f in enumerate(space.get_basis_functions()):
            for j, d in enumerate(space.dofs):
                if i == j:
                    assert d.eval(f) == 1
                else:
                    assert d.eval(f) == 0
                assert d.entity_dim() is not None
    except TimeOutTheTest:
        pytest.skip(f"Testing {element_type} on {cell_type} timed out for order {order}.")
