import symfem
import pytest
import os

dir = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize("reference", [
    "interval", "triangle", "quadrilateral",
    "tetrahedron", "hexahedron", "prism", "pyramid"
])
def test_z_ordering(reference):
    r = symfem.create_reference(reference)

    z = r.z_ordered_entities()

    assert len(z) == len(set(z))

    for i in range(4):
        assert len([e for e in z if e[0] == i]) == r.sub_entity_count(i)


@pytest.mark.parametrize("reference", [
    "interval", "triangle", "quadrilateral",
    "tetrahedron", "hexahedron", "prism", "pyramid"
])
@pytest.mark.parametrize("degree", [0, 2, 5])
def test_dof_diagrams(reference, degree):
    e = symfem.create_element(reference, "P", degree)
    e.plot_dof_diagram(os.path.join(
        dir, f"test-output-test_dof_diagrams-{reference}-{degree}.svg"))
    e.plot_dof_diagram(os.path.join(
        dir, f"test-output-test_dof_diagrams-{reference}-{degree}.png"))
