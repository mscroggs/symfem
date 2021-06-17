from symfem import references
import pytest


@pytest.mark.parametrize(
    "ReferenceClass",
    [references.Interval, references.Triangle, references.Tetrahedron,
     references.Quadrilateral, references.Hexahedron])
def test_reference(ReferenceClass):
    ref = ReferenceClass()
    assert ref.jacobian() == 1


@pytest.mark.parametrize(
    ("ReferenceClass", "points"),
    [(references.Interval, [(1, 0), (1, 2)]),
     (references.Triangle, [(1, 0), (3, 0), (1, 1)]),
     (references.Tetrahedron, [(1, 0, 1), (2, 0, 1), (1, 1, 1), (1, 0, 3)]),
     (references.Quadrilateral, [(1, 0), (3, 0), (1, 1), (3, 1)]),
     (references.Hexahedron, [(1, 0, 1), (2, 0, 1), (1, 1, 1), (2, 1, 1),
                              (1, 0, 3), (2, 0, 3), (1, 1, 3), (2, 1, 3)])])
def test_jacobian(ReferenceClass, points):
    ref = ReferenceClass(vertices=points)
    assert ref.jacobian() == 2


@pytest.mark.parametrize("n_tri", range(3, 10))
def test_dual_reference(n_tri):
    references.DualPolygon(n_tri)
