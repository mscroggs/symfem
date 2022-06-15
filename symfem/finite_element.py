"""Abstract finite element classes and functions."""

import sympy
import warnings
import numpy
import math
from itertools import product
from .symbolic import x, subs, sym_sum, PiecewiseFunction, to_sympy, to_float, symequal, sym_product
from .calculus import diff
from .vectors import vsub
from .basis_function import ElementBasisFunction


class NoTensorProduct(Exception):
    """Error for element without a tensor representation."""

    def __init__(self):
        super().__init__("This element does not have a tensor product representation.")


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

    def get_basis_functions(self, reshape=True, symbolic=True, use_tensor_factorisation=False):
        """Get the basis functions of the element."""
        raise NotImplementedError()

    def get_basis_function(self, n):
        """Get a single basis function of the element."""
        return ElementBasisFunction(self, n)

    def tabulate_basis(self, points, order="xyzxyz", symbolic=True):
        """Evaluate the basis functions of the element at the given points."""
        if not symbolic:
            warnings.warn("Converting from symbolic to float. This may be slow.")
            return numpy.array([to_float(self.tabulate_basis(points, order=order,
                                                             symbolic=True))])

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

    def test(self):
        """Run tests for this element."""
        if self.order <= 4:
            self.test_continuity()

    def test_continuity(self):
        """Test that this element has the correct continuity."""
        continuity = self.continuity
        if "{order}" in continuity:
            continuity = continuity.replace("{order}", f"{self.order}")

        # Test continuity
        if self.reference.name == "interval":
            vertices = ((-1, ), (0, ))
            entity_pairs = [[0, (0, 1)]]
        elif self.reference.name == "triangle":
            vertices = ((-1, 0), (0, 0), (0, 1))
            entity_pairs = [[0, (0, 1)], [0, (2, 2)], [1, (1, 0)]]
        elif self.reference.name == "tetrahedron":
            vertices = ((-1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 1))
            entity_pairs = [[0, (0, 1)], [0, (2, 2)], [0, (3, 3)],
                            [1, (0, 0)], [1, (3, 1)], [1, (4, 2)],
                            [2, (1, 0)]]
        elif self.reference.name == "quadrilateral":
            vertices = ((0, 0), (0, 1), (-1, 0), (-1, 1))
            entity_pairs = [[0, (0, 0)], [0, (2, 1)], [1, (1, 0)]]
        elif self.reference.name == "hexahedron":
            vertices = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                        (-1, 0, 0), (-1, 0, 1), (-1, 1, 0), (-1, 1, 1))
            entity_pairs = [[0, (0, 0)], [0, (2, 2)], [0, (4, 1)], [0, (6, 3)],
                            [1, (1, 1)], [1, (2, 0)], [1, (6, 5)], [1, (9, 3)],
                            [2, (0, 2)]]
        elif self.reference.name == "prism":
            vertices = ((-1, 0, 0), (0, 0, 0), (0, 1, 0),
                        (-1, 0, 1), (0, 0, 1), (0, 1, 1))
            entity_pairs = [[0, (0, 1)], [0, (2, 2)], [0, (3, 4)], [0, (5, 5)],
                            [1, (1, 3)], [1, (2, 4)], [1, (6, 6)], [1, (7, 8)],
                            [2, (2, 3)]]
        elif self.reference.name == "pyramid":
            vertices = ((-1, 0, 0), (0, 0, 0), (-1, 1, 0),
                        (0, 1, 0), (0, 0, 1))
            entity_pairs = [[0, (0, 1)], [0, (2, 3)], [0, (4, 4)],
                            [1, (1, 3)], [1, (2, 4)], [1, (6, 7)],
                            [2, (2, 3)]]

        if continuity == "L2":
            return

        for dim, entities in entity_pairs:
            for fi, gi in zip(*[self.entity_dofs(dim, i) for i in entities]):
                basis = self.get_basis_functions()
                try:
                    basis2 = self.map_to_cell(vertices)
                except NotImplementedError:
                    return "Mapping not implemented for this element."

                f = basis[fi]
                g = basis2[gi]

                def get_piece(f, point):
                    if isinstance(f, PiecewiseFunction):
                        return f.get_piece(point)
                    if isinstance(f, list):
                        return [get_piece(g, point) for g in f]
                    if isinstance(f, tuple):
                        return tuple(get_piece(g, point) for g in f)
                    return f

                if self.reference.tdim == 2:
                    f = get_piece(f, (0, sympy.Rational(1, 2)))
                    g = get_piece(g, (0, sympy.Rational(1, 2)))
                elif self.reference.tdim == 3:
                    f = get_piece(f, (0, sympy.Rational(1, 3), sympy.Rational(1, 3)))
                    g = get_piece(g, (0, sympy.Rational(1, 3), sympy.Rational(1, 3)))

                f = subs(f, [x[0]], [0])
                g = subs(g, [x[0]], [0])

                if continuity[0] == "C":
                    order = int(continuity[1:])
                    deriv_f = [f]
                    deriv_g = [g]
                    f = [f]
                    g = [g]
                    for _ in range(order):
                        deriv_f = [diff(d, i) for d in deriv_f for i in x[:self.reference.tdim]]
                        f += deriv_f
                        deriv_g = [diff(d, i) for d in deriv_g for i in x[:self.reference.tdim]]
                        g += deriv_g
                elif continuity == "H(div)" or continuity == "inner H(div)":
                    f = f[0]
                    g = g[0]
                elif continuity == "H(curl)":
                    f = f[1:]
                    g = g[1:]
                elif continuity == "inner H(curl)":
                    if len(vertices[0]) == 2:
                        f = f[3]
                        g = g[3]
                    if len(vertices[0]) == 3:
                        if dim == 1:
                            vs = self.reference.sub_entities(1)[entities[0]]
                            v0 = self.reference.vertices[vs[0]]
                            v1 = self.reference.vertices[vs[1]]
                            tangent = vsub(v1, v0)
                            f = sum(i * f[ni * len(tangent) + nj] * j
                                    for ni, i in enumerate(tangent)
                                    for nj, j in enumerate(tangent))
                            g = sum(i * g[ni * len(tangent) + nj] * j
                                    for ni, i in enumerate(tangent)
                                    for nj, j in enumerate(tangent))
                        else:
                            assert dim == 2
                            f = [f[4], f[8]]
                            g = [g[4], g[8]]
                elif continuity == "integral inner H(div)":
                    f = f[0].integrate((to_sympy(x[1]), 0, 1))
                    g = g[0].integrate((to_sympy(x[1]), 0, 1))
                else:
                    raise ValueError(f"Unknown continuity: {continuity}")

                assert symequal(f, g)

    def get_tensor_factorisation(self):
        """Get the representation of the element as a tensor product."""
        raise NoTensorProduct()

    def _get_basis_functions_tensor(self):
        """Compute the basis functions using the space's tensor product factorisation."""
        factorisation = self.get_tensor_factorisation()
        basis = {}
        for t_type, factors, perm in factorisation:
            if t_type == "scalar":
                tensor_bases = [[subs(i, x[0], x_i) for i in f.get_basis_functions()]
                                for x_i, f in zip(x, factors)]
                for p, k in zip(perm, product(*tensor_bases)):
                    basis[p] = sym_product(k)
            else:
                raise ValueError(f"Unknown tensor product type: {t_type}")
        return [basis[i] for i in range(len(basis))]

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
        self._basis = basis
        self.dofs = dofs
        self._basis_functions = None
        self._reshaped_basis_functions = None
        self._dual_inv = None

    def entity_dofs(self, entity_dim, entity_number):
        """Get the numbers of the DOFs associated with the given entity."""
        return [i for i, j in enumerate(self.dofs) if j.entity == (entity_dim, entity_number)]

    def get_polynomial_basis(self, reshape=True):
        """Get the symbolic polynomial basis for the element."""
        basis = [to_sympy(i) for i in self._basis]

        if reshape and self.range_shape is not None:
            if len(self.range_shape) != 2:
                raise NotImplementedError
            assert self.range_shape[0] * self.range_shape[1] == self.range_dim
            return [sympy.Matrix(
                [b[i * self.range_shape[1]: (i + 1) * self.range_shape[1]]
                 for i in range(self.range_shape[0])]) for b in basis]
        return basis

    def get_tabulated_polynomial_basis(self, points, symbolic=True):
        """Get the value of the polynomial basis at the given points."""
        if symbolic:
            return [
                [subs(f, x, p) for f in self.get_polynomial_basis()]
                for p in points]
        else:
            return numpy.array([
                [to_float(subs(f, x, p)) for f in self.get_polynomial_basis()]
                for p in points])

    def tabulate_basis(self, points, order="xyzxyz", symbolic=True):
        """Evaluate the basis functions of the element at the given points."""
        if symbolic:
            return super().tabulate_basis(points, order, symbolic=True)

        assert not symbolic

        tabulated_polyset = self.get_tabulated_polynomial_basis(
            points, symbolic=False)

        if self._dual_inv is None:
            dual_mat = self.get_dual_matrix(symbolic=False)
            self._dual_inv = numpy.linalg.inv(dual_mat)
        dual_inv = self._dual_inv

        if self.range_dim == 1:
            return numpy.dot(tabulated_polyset, dual_inv.transpose())

        results = numpy.array([
            numpy.dot(tabulated_polyset[:, :, i], dual_inv.transpose())
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
        if symbolic:
            mat = []
            for b in self.get_polynomial_basis():
                row = []
                for d in self.dofs:
                    row.append(to_sympy(d.eval(to_sympy(b), symbolic=symbolic)))
                mat.append(row)
            return sympy.Matrix(mat)
        else:
            for d in self.dofs:
                if d.get_points_and_weights is None:
                    warnings.warn("Cannot numerically evaluate all the DOFs in this element. "
                                  "Converting symbolic evaluations instead (this may be slow).")
                    mat = self.get_dual_matrix(symbolic=True)
                    return numpy.array(
                        [[float(j) for j in mat.row(i)] for i in range(mat.rows)]
                    )
            mat = numpy.empty((len(self.dofs), len(self.dofs)))
            for i, d in enumerate(self.dofs):
                p, w = d.get_points_and_weights()
                mat[:, i] = numpy.dot(w, self.get_tabulated_polynomial_basis(
                    p, symbolic=False))
            return mat

    def init_kwargs(self):
        """Return the kwargs used to create this element."""
        return {}

    def get_basis_functions(self, reshape=True, symbolic=True, use_tensor_factorisation=False):
        """Get the basis functions of the element."""
        if self._basis_functions is None:
            if use_tensor_factorisation:
                self._basis_functions = self._get_basis_functions_tensor()
            else:
                minv = self.get_dual_matrix().inv("LU")
                self._basis_functions = []
                if self.range_dim == 1:
                    # Scalar space
                    for i, dof in enumerate(self.dofs):
                        self._basis_functions.append(
                            sym_sum(c * d for c, d in zip(
                                minv.row(i),
                                self.get_polynomial_basis())))
                else:
                    # Vector space
                    for i, dof in enumerate(self.dofs):
                        b = [0 for i in range(self.range_dim)]
                        for c, d in zip(minv.row(i), self.get_polynomial_basis()):
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

    def map_to_cell(self, vertices, basis=None, forward_map=None, inverse_map=None):
        """Map the basis onto a cell using the appropriate mapping for the element."""
        if basis is None:
            basis = self.get_basis_functions()
        if forward_map is None:
            forward_map = self.reference.get_map_to(vertices)
        if inverse_map is None:
            inverse_map = self.reference.get_inverse_map_to(vertices)

        if isinstance(basis[0], PiecewiseFunction):
            pieces = [[] for i in basis]
            for i, j in enumerate(basis[0].pieces):
                new_i = [subs(forward_map, x, k) for k in j[0]]
                for k, f in enumerate(self.map_to_cell(vertices, [b.pieces[i][1] for b in basis])):
                    pieces[k].append((new_i, f))
            return [PiecewiseFunction(p) for p in pieces]

        if isinstance(basis[0], (list, tuple)) and isinstance(basis[0][0], PiecewiseFunction):
            for b in basis:
                for f in b:
                    for i, j in enumerate(f.pieces):
                        assert j[0] == basis[0][0].pieces[i][0]
            new_tris = [[subs(forward_map, x, k) for k in p[0]] for p in basis[0][0].pieces]
            output_pieces = []
            for p in range(len(basis[0][0].pieces)):
                piece_basis = []
                for b in basis:
                    piece_basis.append([
                        f.pieces[p][1] for f in b
                    ])
                output_pieces.append(self.map_to_cell(
                    vertices, piece_basis, forward_map, inverse_map))

            return [PiecewiseFunction(list(zip(new_tris, fs)))
                    for fs in zip(*output_pieces)]

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
                        forward_map, inverse_map, self.reference.tdim)
                    for d_n, d in zip(ds, mapped_dofs):
                        out[d_n] = d

        for i in out:
            assert i is not None
        return out

    def test(self):
        """Run tests for this element."""
        super().test()
        self.test_functional_entities()
        self.test_functionals()
        self.test_dof_points()

    def test_dof_points(self):
        """Test that DOF points are valid."""
        for d in self.dofs:
            p = d.dof_point()
            assert len(p) == self.reference.gdim
            for i in p:
                if i is None:
                    break
                else:
                    assert not math.isnan(float(i))
            else:
                assert self.reference.contains(p)

    def test_functional_entities(self):
        """Test that the dof entities are valid and match the references of integrals."""
        for dof in self.dofs:
            dim, entity = dof.entity
            assert entity < self.reference.sub_entity_count(dim)
            if hasattr(dof, "integral_domain"):
                assert dim == dof.integral_domain.tdim

    def test_functionals(self):
        """Test that the functionals are satisfied by the basis functions."""
        for i, f in enumerate(self.get_basis_functions()):
            for j, d in enumerate(self.dofs):
                if i == j:
                    assert d.eval(f).expand().simplify() == 1
                else:
                    assert d.eval(f).expand().simplify() == 0
                assert d.entity_dim() is not None

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

    def get_basis_functions(self, reshape=True, symbolic=True, use_tensor_factorisation=False):
        """Get the basis functions of the element."""
        if use_tensor_factorisation:
            return self._get_basis_functions_tensor()

        if reshape and self.range_shape is not None:
            if len(self.range_shape) != 2:
                raise NotImplementedError
            assert self.range_shape[0] * self.range_shape[1] == self.range_dim
            return [sympy.Matrix(
                [b[i * self.range_shape[1]: (i + 1) * self.range_shape[1]]
                 for i in range(self.range_shape[0])]) for b in self._basis_functions]

        return self._basis_functions

    def test(self):
        """Run tests for this element."""
        super().test()
        self.test_independence()

    def test_independence(self):
        """Test that the basis functions of this element are linearly independent."""
        basis = self.get_basis_functions()
        all_terms = set()

        try:
            basis[0].as_coefficients_dict()
            scalar = True
        except AttributeError:
            scalar = False

        if scalar:
            for f in basis:
                for term in f.as_coefficients_dict():
                    all_terms.add(term)
            mat = [[0 for i in all_terms] for j in basis]
            for i, t in enumerate(all_terms):
                for j, f in enumerate(basis):
                    fd = f.as_coefficients_dict()
                    if t in fd:
                        mat[j][i] = fd[t]
        else:
            for f in basis:
                for fi, fpart in enumerate(f):
                    for term in fpart.as_coefficients_dict():
                        all_terms.add((fi, term))
            mat = [[0 for i in all_terms] for j in basis]
            for i, (fi, t) in enumerate(all_terms):
                for j, f in enumerate(basis):
                    fd = f[fi].as_coefficients_dict()
                    if t in fd:
                        mat[j][i] = fd[t]
        mat = sympy.Matrix(mat)

        assert mat.rank() == mat.rows

    names = []
