"""Test that converted Basix elements define the same functionals on every sub-entity.

Basix (and DOLFINx) assume that the DOF functionals on each sub-entity of the reference
cell are the push-forward of the functionals on the reference sub-entity. If they are
not, two cells that share a sub-entity, but see it as different local sub-entities, do
not agree on the value of the shared DOFs, and the element is not conforming.

The functionals of the converted element are compared with those of the corresponding
native Basix element. On every sub-entity, the two sets of functionals must be related
by the same change of basis.
"""

import pytest

basix = pytest.importorskip("basix")
np = pytest.importorskip("numpy")

from symfem import create_element

# (Symfem name, Basix family, variant, Symfem degree minus Basix degree)
elements = [
    ("Raviart-Thomas", "RT", "legendre", -1),
    ("Nedelec", "N1E", "legendre", -1),
    ("Brezzi-Douglas-Marini", "BDM", "legendre", 0),
    ("Nedelec2", "N2E", "legendre", 0),
    ("Regge", "Regge", "integral", 0),
    ("Hellan-Herrmann-Johnson", "HHJ", None, 0),
]


def _functional_matrix(element, dim: int, entity: int, cell, degree: int):
    """Apply the functionals of a sub-entity to a basis of vector-valued polynomials."""
    pts = element.x[dim][entity]
    M = element.M[dim][entity][:, :, :, 0]  # (DOFs, value components, points)
    P = basix.tabulate_polynomials(basix.PolynomialType.legendre, cell, degree, pts)
    return np.einsum("dcp,jp->dcj", M, P).reshape(M.shape[0], M.shape[1] * P.shape[0])


@pytest.mark.parametrize("cell_name", ["triangle", "tetrahedron"])
@pytest.mark.parametrize(("symfem_name", "basix_name", "variant", "offset"), elements)
@pytest.mark.parametrize("degree", [1, 2])
def test_entity_consistency(cell_name, symfem_name, basix_name, variant, offset, degree):
    """Check that DOFs on all sub-entities are related to Basix's by the same map."""
    from symfem.basix_interface import create_basix_element

    cell = getattr(basix.CellType, cell_name)
    if variant is None:
        native = basix.create_element(getattr(basix.ElementFamily, basix_name), cell, degree)
        symfem_element = create_element(cell_name, symfem_name, degree + offset)
    elif variant == "integral":
        native = basix.create_element(getattr(basix.ElementFamily, basix_name), cell, degree)
        symfem_element = create_element(cell_name, symfem_name, degree + offset, variant=variant)
    else:
        native = basix.create_element(
            getattr(basix.ElementFamily, basix_name),
            cell,
            degree,
            getattr(basix.LagrangeVariant, variant),
        )
        symfem_element = create_element(cell_name, symfem_name, degree + offset, variant=variant)
    converted = create_basix_element(symfem_element)
    tdim = basix.topology(cell).__len__() - 1
    poly_degree = native.embedded_superdegree

    for dim in range(1, tdim):
        maps = []
        for entity in range(len(native.x[dim])):
            D_native = _functional_matrix(native, dim, entity, cell, poly_degree)
            D_converted = _functional_matrix(converted, dim, entity, cell, poly_degree)
            if D_native.shape[0] == 0:
                assert D_converted.shape[0] == 0
                continue
            C = D_converted @ np.linalg.pinv(D_native)
            assert np.allclose(C @ D_native, D_converted), "DOF spans differ"
            maps.append(C)
        for entity, C in enumerate(maps[1:], start=1):
            assert np.allclose(C, maps[0]), (
                f"DOFs on sub-entity ({dim}, {entity}) are scaled differently from those on "
                f"sub-entity ({dim}, 0): ratio {np.linalg.norm(C) / np.linalg.norm(maps[0])}"
            )
