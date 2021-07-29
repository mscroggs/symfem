import pytest
import symfem
from symfem.calculus import div


@pytest.mark.parametrize("order", [1])
def test_guzman_neilan_triangle(order):
    e = symfem.create_element("triangle", "Guzman-Neilan", order)

    for p in e._basis[-3:]:
        for piece in p.pieces:
            float(div(piece[1]).expand())


@pytest.mark.parametrize("order", [1, 2])
def test_guzman_neilan_tetrahedron(order):
    e = symfem.create_element("tetrahedron", "Guzman-Neilan", order)

    for p in e._basis[-4:]:
        for piece in p.pieces:
            float(div(piece[1]).expand())
