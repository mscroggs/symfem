import pytest
import sympy

import symfem
from symfem.utils import allequal


def test_cache():
    e = symfem.create_element("triangle", "Lagrange", 2)

    m1 = e.get_dual_matrix(caching=False, inverse=True)

    def a(*args, **kwargs):
        raise RuntimeError()

    oldinv = sympy.Matrix.inv
    sympy.Matrix.inv = a

    with pytest.raises(RuntimeError):
        e.get_dual_matrix(caching=False, inverse=True)

    m2 = e.get_dual_matrix(inverse=True)

    sympy.Matrix.inv = oldinv

    assert allequal(m1, m2)
