"""Abstract finite element classes and functions."""

import sympy
from .symbolic import x, zero, subs


class FiniteElement:
    """Abstract finite element."""

    def __init__(self, basis, dofs, domain_dim, range_dim):
        assert len(basis) == len(dofs)
        self.basis = basis
        self.dofs = dofs
        self.domain_dim = domain_dim
        self.range_dim = range_dim
        self.space_dim = len(dofs)
        self._basis_functions = None

    def get_basis_functions(self):
        """Get the basis functions of the element."""
        if self._basis_functions is None:
            mat = []
            for b in self.basis:
                row = []
                for d in self.dofs:
                    row.append(d.eval(b))
                mat.append(row)
            minv = sympy.Matrix(mat).inv()
            self._basis_functions = []
            if self.range_dim == 1:
                # Scalar space
                for i, dof in enumerate(self.dofs):
                    b = zero
                    for c, d in zip(minv.row(i), self.basis):
                        b += c * d
                    self._basis_functions.append(b)
            else:
                # Vector space
                for i, dof in enumerate(self.dofs):
                    b = [zero for i in self.basis[0]]
                    for c, d in zip(minv.row(i), self.basis):
                        for j, d_j in enumerate(d):
                            b[j] += c * d_j
                    self._basis_functions.append(b)

        return self._basis_functions

    def tabulate_basis(self, points, order="xyzxyz"):
        """Evaluate the basis functions of the element at the given points."""
        if self.range_dim == 1:
            output = []
            for p in points:
                row = []
                for b in self.get_basis_functions():
                    row.append(subs(b, x, p))
                output.append(row)
            return output

        if order == "xxyyzz":
            output = []
            for p in points:
                row = []
                for d in range(self.range_dim):
                    for b in self.get_basis_functions():
                        row.append(subs(b[d], x, p))
                output.append(row)
            return output
        if order == "xyzxyz":
            output = []
            for p in points:
                row = []
                for d in range(self.range_dim):
                    for b in self.get_basis_functions():
                        row.append(subs(b[d], x, p))
                output.append(row)
            return output
        raise ValueError(f"Unknown order: {order}")


def make_integral_moment_dofs(
    reference,
    vertices=None,
    edges=None,
    faces=None,
    volumes=None,
    cells=None,
    facets=None,
    ridges=None,
    peaks=None,
):
    """Generate DOFs due to integral moments on sub entities.

    Parameters
    ----------
    reference: feast.references.Reference
        The reference cell.
    vertices: tuple
        DOFs on dimension 0 entities.
    edges: tuple
        DOFs on dimension 1 entities.
    faces: tuple
        DOFs on dimension 2 entities.
    volumes: tuple
        DOFs on dimension 3 entities.
    cells: tuple
        DOFs on codimension 0 entities.
    facets: tuple
        DOFs on codimension 1 entities.
    ridges: tuple
        DOFs on codimension 2 entities.
    peaks: tuple
        DOFs on codimension 3 entities.
    """
    dofs = []

    # DOFs per dimension
    for dim, moment_data in enumerate([vertices, edges, faces, volumes]):
        if moment_data is not None and moment_data[2] >= moment_data[3]:
            sub_type = reference.sub_entity_types[dim]
            if sub_type is not None:
                assert dim > 0
                for i, vs in enumerate(reference.sub_entities(dim)):
                    sub_ref = sub_type(vertices=[reference.vertices[v] for v in vs])
                    sub_element = moment_data[1](sub_ref, moment_data[2])
                    for d in sub_element.get_basis_functions():
                        dofs.append(moment_data[0](sub_ref, d))

    # DOFs per codimension
    for codim, moment_data in enumerate([cells, facets, ridges, peaks]):
        dim = reference.tdim - codim
        if moment_data is not None and moment_data[2] >= moment_data[3]:
            sub_type = reference.sub_entity_types[dim]
            if sub_type is not None:
                assert dim > 0
                for i, vs in enumerate(reference.sub_entities(dim)):
                    sub_ref = sub_type(vertices=[reference.vertices[v] for v in vs])
                    sub_element = moment_data[1](sub_ref, moment_data[2])
                    for d in sub_element.get_basis_functions():
                        dofs.append(moment_data[0](sub_ref, d))
    return dofs
