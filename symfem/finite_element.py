"""Abstract finite element classes and functions."""

import sympy
import numpy
from .symbolic import x, zero, subs, sym_sum, PiecewiseFunction
from .basis_function import ElementBasisFunction


class FiniteElement:
    """Abstract finite element."""

    def __init__(self, reference, order, space_dim, domain_dim, range_dim,
                 range_shape=None):
        self.reference = reference
        self.order = order
        self.space_dim = space_dim
        self.domain_dim = domain_dim
        self.range_dim = range_dim
        self.range_shape = range_shape

    def entity_dofs(self, entity_dim, entity_number):
        """Get the numbers of the DOFs associated with the given entity."""
        raise NotImplementedError()

    def get_basis_functions(self, reshape=True, symbolic=True):
        """Get the basis functions of the element."""
        raise NotImplementedError()

    def get_basis_function(self, n):
        """Get a single basis function of the element."""
        return ElementBasisFunction(self, n)

    def tabulate_basis(self, points, order="xyzxyz", symbolic=True):
        """Evaluate the basis functions of the element at the given points."""
        assert symbolic

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
        raise NotImplementedError()

    @property
    def name(self):
        """Get the name of the element."""
        return self.names[0]

    names = []


class CiarletElement(FiniteElement):
    """Finite element defined using the Ciarlet definition."""

    def __init__(self, reference, order, basis, dofs, domain_dim, range_dim,
                 range_shape=None):
        super().__init__(reference, order, len(dofs), domain_dim, range_dim, range_shape)
        assert len(basis) == len(dofs)
        self.basis = basis
        self.dofs = dofs
        self._basis_functions = None
        self._reshaped_basis_functions = None
        self._dual_inv = None

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

    def tabulate_basis(self, points, order="xyzxyz", symbolic=True):
        """Evaluate the basis functions of the element at the given points."""
        if symbolic:
            return super().tabulate_basis(points, order, symbolic)

        assert not symbolic

        if self._dual_inv is None:
            dual_mat = self.get_dual_matrix(symbolic=False)
            self._dual_inv = numpy.linalg.inv(dual_mat)

        if self.range_dim == 1:
            tabulated_polyset = numpy.array([
                [float(subs(f, x, p)) for f in self.basis]
                for p in points])
            return numpy.dot(tabulated_polyset, self._dual_inv.transpose())

        tabulated_polyset = numpy.array([
            [tuple(float(i) for i in subs(f, x, p)) for f in self.basis]
            for p in points])

        results = numpy.array([
            numpy.dot(tabulated_polyset[:, :, i], self._dual_inv.transpose())
            for i in range(tabulated_polyset.shape[2])
        ])
        # results[xyz][point][function]

        if order == "xxyyzz":
            return numpy.array([
                [results[xyz, point, function] for xyz in range(results.shape[0])
                 for function in range(results.shape[2])]
                for point in range(results.shape[1])
            ])
        if order == "xyzxyz":
            return numpy.array([
                [results[xyz, point, function] for function in range(results.shape[2])
                 for xyz in range(results.shape[0])]
                for point in range(results.shape[1])
            ])
        if order == "xyz,xyz":
            return numpy.array([
                [[results[xyz, point, function] for xyz in range(results.shape[0])]
                 for function in range(results.shape[2])]
                for point in range(results.shape[1])
            ])
        raise ValueError(f"Unknown order: {order}")

    def get_dual_matrix(self, symbolic=True):
        """Get the dual matrix."""
        mat = []
        for b in self.basis:
            row = []
            for d in self.dofs:
                row.append(d.eval(b))
            mat.append(row)
        if symbolic:
            return sympy.Matrix(mat)
        else:
            return numpy.array([[float(j) for j in i] for i in mat])

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

        out = [None for f in basis]
        for dim in range(self.reference.tdim + 1):
            for e in range(self.reference.sub_entity_count(dim)):
                entity_dofs = self.entity_dofs(dim, e)
                dofs_by_type = {}
                for d in entity_dofs:
                    dof = self.dofs[d]
                    t = (type(dof), dof.mapping)
                    if t not in dofs_by_type:
                        dofs_by_type[t] = []
                    dofs_by_type[t].append(d)
                for ds in dofs_by_type.values():
                    mapped_dofs = self.dofs[ds[0]].perform_mapping(
                        [basis[d] for d in ds],
                        map, inverse_map, self.reference.tdim)
                    for d_n, d in zip(ds, mapped_dofs):
                        out[d_n] = d

        for i in out:
            assert i is not None
        return out

    names = []


class DirectElement(FiniteElement):
    """Finite element defined directly."""

    def __init__(self, reference, order, basis_functions, basis_entities, domain_dim, range_dim,
                 range_shape=None):
        super().__init__(reference, order, len(basis_functions), domain_dim, range_dim,
                         range_shape)
        self._basis_entities = basis_entities
        self._basis_functions = basis_functions
        self._reshaped_basis_functions = None

    def entity_dofs(self, entity_dim, entity_number):
        """Get the numbers of the DOFs associated with the given entity."""
        return [i for i, j in enumerate(self._basis_entities) if j == (entity_dim, entity_number)]

    def get_basis_functions(self, reshape=True):
        """Get the basis functions of the element."""
        if reshape and self.range_shape is not None:
            if len(self.range_shape) != 2:
                raise NotImplementedError
            assert self.range_shape[0] * self.range_shape[1] == self.range_dim
            return [sympy.Matrix(
                [b[i * self.range_shape[1]: (i + 1) * self.range_shape[1]]
                 for i in range(self.range_shape[0])]) for b in self._basis_functions]

        return self._basis_functions

    names = []
