"""Test Bell elements."""

import symfem
from symfem.symbols import t, x


def test_bell_polyset():
    b = symfem.create_element("triangle", "Bell", 5)
    for p in b.get_polynomial_basis():
        gradp = [p.diff(x[0]), p.diff(x[1])]
        for en in range(b.reference.sub_entity_count(1)):
            edge = b.reference.sub_entity(1, en)
            variables = [o + sum(a[i] * t[0] for a in edge.axes)
                         for i, o in enumerate(edge.origin)]
            n = edge.normal()
            normal_deriv = gradp[0] * n[0] + gradp[1] * n[1]
            normal_deriv = normal_deriv.subs(x, variables)
            assert normal_deriv.diff(t[0]).diff(t[0]).diff(t[0]).diff(t[0]) == 0
