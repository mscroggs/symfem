"""Abstract finite element classes and functions."""

import sympy
from .symbolic import x, zero, subs


class FiniteElement:
    """Abstract finite element."""

    def __init__(self, reference, basis, dofs, domain_dim, range_dim,
                 range_shape=None):
        assert len(basis) == len(dofs)
        self.reference = reference
        self.basis = basis
        self.dofs = dofs
        self.domain_dim = domain_dim
        self.range_dim = range_dim
        self.range_shape = range_shape
        self.space_dim = len(dofs)
        self._basis_functions = None
        self._reshaped_basis_functions = None

    def get_polynomial_basis(self, reshape=True):
        """Get the polynomial basis for the element."""
        if reshape and self.range_shape is not None:
            if len(self.range_shape) != 2:
                raise NotImplementedError
            assert self.range_shape[0] * self.range_shape[1] == self.range_dim
            return [sympy.Matrix(
                [b[i * self.range_shape[1]: (i + 1) * self.range_shape[1]]
                 for i in range(self.range_shape[0])]) for b in self.basis]

        return self.basis

    def get_basis_functions(self, reshape=True):
        """Get the basis functions of the element."""
        if self._basis_functions is None:
            mat = []
            for b in self.basis:
                row = []
                for d in self.dofs:
                    row.append(d.eval(b))
                mat.append(row)
            minv = sympy.Matrix(mat).inv("LU")
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

        if reshape and self.range_shape is not None:
            if len(self.range_shape) != 2:
                raise NotImplementedError
            assert self.range_shape[0] * self.range_shape[1] == self.range_dim
            return [sympy.Matrix(
                [b[i * self.range_shape[1]: (i + 1) * self.range_shape[1]]
                 for i in range(self.range_shape[0])]) for b in self._basis_functions]

        return self._basis_functions

    def tabulate_basis(self, points, order="xyzxyz"):
        """Evaluate the basis functions of the element at the given points."""
        if self.range_dim == 1:
            output = []
            for p in points:
                row = []
                for b in self.get_basis_functions(False):
                    row.append(subs(b, x, p))
                output.append(row)
            return output

        if order == "xxyyzz":
            output = []
            for p in points:
                row = []
                for d in range(self.range_dim):
                    for b in self.get_basis_functions(False):
                        row.append(subs(b[d], x, p))
                output.append(row)
            return output
        if order == "xyzxyz":
            output = []
            for p in points:
                row = []
                for b in self.get_basis_functions(False):
                    for i in subs(b, x, p):
                        row.append(i)
                output.append(row)
            return output
        if order == "xyz,xyz":
            output = []
            for p in points:
                row = []
                for b in self.get_basis_functions(False):
                    row.append(subs(b, x, p))
                output.append(row)
            return output
        raise ValueError(f"Unknown order: {order}")

    @property
    def name(self):
        """Get the name of the element."""
        return self.names[0]

    names = []


def make_integral_moment_dofs(
    reference,
    vertices=None, edges=None, faces=None, volumes=None,
    cells=None, facets=None, ridges=None, peaks=None
):
    """Generate DOFs due to integral moments on sub entities.

    Parameters
    ----------
    reference: symfem.references.Reference
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
    from symfem import create_reference
    dofs = []

    # DOFs per dimension
    for dim, moment_data in enumerate([vertices, edges, faces, volumes]):
        if moment_data is not None:
            IntegralMoment, SubElement, order = moment_data
            if order >= SubElement.min_order:
                sub_type = reference.sub_entity_types[dim]
                if sub_type is not None:
                    assert dim > 0
                    for i, vs in enumerate(reference.sub_entities(dim)):
                        sub_ref = create_reference(
                            sub_type,
                            vertices=[reference.reference_vertices[v] for v in vs])
                        sub_element = SubElement(sub_ref, order)
                        for f, d in zip(
                            sub_element.get_basis_functions(), sub_element.dofs
                        ):
                            dofs.append(IntegralMoment(sub_ref, f, d))

    # DOFs per codimension
    for _dim, moment_data in enumerate([peaks, ridges, facets, cells]):
        dim = reference.tdim - 3 + _dim
        if moment_data is not None:
            IntegralMoment, SubElement, order = moment_data
            if order >= SubElement.min_order:
                sub_type = reference.sub_entity_types[dim]
                if sub_type is not None:
                    assert dim > 0
                    for i, vs in enumerate(reference.sub_entities(dim)):
                        sub_ref = create_reference(
                            sub_type,
                            vertices=[reference.reference_vertices[v] for v in vs])
                        sub_element = SubElement(sub_ref, order)
                        for f, d in zip(
                            sub_element.get_basis_functions(), sub_element.dofs
                        ):
                            dofs.append(IntegralMoment(sub_ref, f, d))

    return dofs
