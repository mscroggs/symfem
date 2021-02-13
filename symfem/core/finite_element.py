"""Abstract finite element classes and functions."""

import sympy
from .symbolic import x, zero, subs, sym_sum, PiecewiseFunction
from . import mappings


class FiniteElement:
    """Abstract finite element."""

    def __init__(self, reference, order, basis, dofs, domain_dim, range_dim,
                 range_shape=None):
        assert len(basis) == len(dofs)
        self.reference = reference
        self.order = order
        self.basis = basis
        self.dofs = dofs
        self.domain_dim = domain_dim
        self.range_dim = range_dim
        self.range_shape = range_shape
        self.space_dim = len(dofs)
        self._basis_functions = None
        self._reshaped_basis_functions = None

    def entity_dofs(self, entity_dim, entity_number):
        """Get the numbers of the DOFs associated with the given entity."""
        return [i for i, j in enumerate(self.dofs) if j.entity == (entity_dim, entity_number)]

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

    def get_dual_matrix(self):
        """Get the dual matrix."""
        mat = []
        for b in self.basis:
            row = []
            for d in self.dofs:
                row.append(d.eval(b))
            mat.append(row)
        return sympy.Matrix(mat)

    def get_basis_functions(self, reshape=True):
        """Get the basis functions of the element."""
        if self._basis_functions is None:
            minv = self.get_dual_matrix().inv("LU")
            self._basis_functions = []
            if self.range_dim == 1:
                # Scalar space
                for i, dof in enumerate(self.dofs):
                    self._basis_functions.append(
                        sym_sum(c * d for c, d in zip(minv.row(i), self.basis)))
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

    def map_to_cell(self, vertices, basis=None):
        """Map the basis onto a cell using the appropriate mapping for the element."""
        if basis is None:
            basis = self.get_basis_functions()
        map = self.reference.get_map_to(vertices)
        if isinstance(basis[0], PiecewiseFunction):
            pieces = [[] for i in basis]
            for i, j in enumerate(basis[0].pieces):
                new_i = [subs(map, x, k) for k in j[0]]
                for k, f in enumerate(self.map_to_cell(vertices, [b.pieces[i][1] for b in basis])):
                    pieces[k].append((new_i, f))
            return [PiecewiseFunction(p) for p in pieces]
        inverse_map = self.reference.get_inverse_map_to(vertices)
        return self.perform_mapping(basis, map, inverse_map)

    def perform_mapping(self, basis, map, inverse_map):
        """Map the basis onto a cell using the appropriate mapping for the element."""
        return [getattr(mappings, self.mapping)(f, map, inverse_map, self.reference.tdim)
                for f in basis]

    @property
    def name(self):
        """Get the name of the element."""
        return self.names[0]

    names = []
