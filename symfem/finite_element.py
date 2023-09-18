"""Abstract finite element classes and functions."""

from __future__ import annotations

import math
import typing
from abc import ABC, abstractmethod, abstractproperty
from itertools import product

import sympy

from .basis_functions import BasisFunction
from .caching import load_cached_matrix, save_cached_matrix
from .functionals import ListOfFunctionals
from .functions import (AnyFunction, FunctionInput, ScalarFunction, VectorFunction,
                        parse_function_input)
from .geometry import PointType, SetOfPointsInput, parse_set_of_points_input
from .mappings import MappingNotImplemented
from .piecewise_functions import PiecewiseFunction
from .plotting import Picture, colors
from .references import NonDefaultReferenceError, Reference
from .symbols import x
from .utils import allequal
from .version import version

TabulatedBasis = typing.Union[
    typing.List[typing.Union[sympy.core.expr.Expr, int]],
    typing.List[typing.Tuple[typing.Union[sympy.core.expr.Expr, int], ...]],
    typing.List[typing.Tuple[typing.Tuple[typing.Union[sympy.core.expr.Expr, int], ...], ...]],
    typing.List[sympy.matrices.dense.MutableDenseMatrix],
]


class NoTensorProduct(Exception):
    """Error for element without a tensor representation."""

    def __init__(self):
        """Initialise."""
        super().__init__("This element does not have a tensor product representation.")


class FiniteElement(ABC):
    """Abstract finite element."""

    _float_basis_functions: typing.Union[None, typing.List[AnyFunction]]
    _value_scale: typing.Union[None, sympy.core.expr.Expr]

    def __init__(
        self, reference: Reference, order: int, space_dim: int, domain_dim: int, range_dim: int,
        range_shape: typing.Optional[typing.Tuple[int, ...]] = None
    ):
        """Create a finite element.

        Args:
            reference: The reference cell
            order: The polynomial order
            space_dim: The dimension of the finite element space
            domain_dim: The topological dimension of the reference cell
            range_dim: The dimension/value size of functions in the space
            range_shape: The (value) shape of function in the space
        """
        self.reference = reference
        self.order = order
        self.space_dim = space_dim
        self.domain_dim = domain_dim
        self.range_dim = range_dim
        self.range_shape = range_shape
        self._float_basis_functions = None
        self._value_scale = None

    @abstractmethod
    def dof_plot_positions(self) -> typing.List[PointType]:
        """Get the points to plot each DOF at on a DOF diagram.

        Returns:
            The DOF positions
        """

    @abstractmethod
    def dof_directions(self) -> typing.List[typing.Union[PointType, None]]:
        """Get the direction associated with each DOF.

        Returns:
            The DOF directions
        """

    @abstractmethod
    def dof_entities(self) -> typing.List[typing.Tuple[int, int]]:
        """Get the entities that each DOF is associated with.

        Returns:
            The entities
        """

    def plot_dof_diagram(
        self, filename: typing.Union[str, typing.List[str]],
        plot_options: typing.Dict[str, typing.Any] = {}, **kwargs: typing.Any
    ):
        """Plot a diagram showing the DOFs of the element.

        Args:
            filename: The file name
            plot_options: Options for the plot
            kwargs: Keyword arguments
        """
        img = Picture(**kwargs)

        dof_positions = self.dof_plot_positions()
        dof_directions = self.dof_directions()
        dof_entities = self.dof_entities()

        for entities in self.reference.z_ordered_entities():
            for dim, e in entities:
                if dim == 1:
                    pts = tuple(self.reference.vertices[i] for i in self.reference.edges[e])
                    img.add_line(pts[0], pts[1], colors.BLACK)
                if dim == 2:
                    pts = tuple(self.reference.vertices[i] for i in self.reference.faces[e])
                    if len(pts) == 4:
                        pts = (pts[0], pts[1], pts[3], pts[2])
                    img.add_fill(pts, colors.WHITE, 0.5)

            for dim, e in entities:
                dofs = [(dof_positions[i], dof_directions[i], dof_entities[i], i)
                        for i in self.entity_dofs(dim, e)]
                dofs.sort(key=lambda d: img.z(d[0]))
                for d in dofs:
                    direction = d[1]
                    if direction is not None:
                        shifted = False
                        for d2, p in enumerate(dof_positions):
                            if d != d2 and d[0] == p:
                                shifted = True
                                break
                        img.add_dof_arrow(d[0], direction, d[3],
                                          colors.entity(d[2][0]), shifted)
                    else:
                        img.add_dof_marker(
                            d[0], d[3], colors.entity(d[2][0]))

        img.save(filename, plot_options=plot_options)

    @abstractmethod
    def entity_dofs(self, entity_dim: int, entity_number: int) -> typing.List[int]:
        """Get the numbers of the DOFs associated with the given entity.

        Args:
            entity_dim: The dimension of the entity
            entity_number: The number of the entity

        Returns:
            The numbers of the DOFs associated with the entity
        """

    @abstractmethod
    def get_basis_functions(
        self, use_tensor_factorisation: bool = False
    ) -> typing.List[AnyFunction]:
        """Get the basis functions of the element.

        Args:
            use_tensor_factorisation: Should a tensor factorisation be used?

        Returns:
            The basis functions
        """

    def get_basis_function(self, n: int) -> BasisFunction:
        """Get a single basis function of the element.

        Args:
            n: The number of the basis function

        Returns:
            The basis function
        """
        return ElementBasisFunction(self, n)

    def tabulate_basis(
        self, points_in: SetOfPointsInput, order: str = "xyzxyz",
    ) -> TabulatedBasis:
        """Evaluate the basis functions of the element at the given points.

        Args:
            points_in: The points
            order: The order to return the values

        Returns:
            The tabulated basis functions
        """
        points = parse_set_of_points_input(points_in)
        tabbed = [tuple(b.subs(x, p).as_sympy() for b in self.get_basis_functions())
                  for p in points]
        if self.range_dim == 1:
            return tabbed

        if order == "xyz,xyz":
            return tabbed
        elif order == "xxyyzz":
            output = []
            for row in tabbed:
                output.append(tuple(j for i in zip(*row) for j in i))
            return output
        elif order == "xx,yy,zz":
            output = []
            for row in tabbed:
                output.append(tuple(tuple(j for j in i) for i in zip(*row)))
            return output
        elif order == "xyzxyz":
            output = []
            for row in tabbed:
                output.append(tuple(j for i in row for j in i))
            return output
        else:
            raise ValueError(f"Unknown order: {order}")

    def tabulate_basis_float(self, points_in: SetOfPointsInput) -> TabulatedBasis:
        """Evaluate the basis functions of the element at the given points in xyz,xyz order.

        Args:
            points_in: The points

        Returns:
            The tabulated basis functions
        """
        if self._float_basis_functions is None:
            self._float_basis_functions = [b.with_floats() for b in self.get_basis_functions()]

        assert self._float_basis_functions is not None
        points = parse_set_of_points_input(points_in)
        return [tuple(b.subs(x, p).as_sympy() for b in self._float_basis_functions) for p in points]

    def plot_basis_function(
        self, n: int, filename: typing.Union[str, typing.List[str]],
        cell: typing.Optional[Reference] = None, **kwargs: typing.Any
    ):
        """Plot a diagram showing a basis function.

        Args:
            n: The basis function number
            filename: The file name
            cell: The cell to push the basis function to and plot on
            kwargs: Keyword arguments
        """
        if self._value_scale is None:
            max_v = 0.0
            values = self.tabulate_basis_float(self.reference.make_lattice_float(6))
            for row in values:
                assert isinstance(row, tuple)
                for i in row:
                    max_v = max(max_v, float(parse_function_input(i).norm()))
            self._value_scale = 1 / sympy.Float(max_v)

        assert self._value_scale is not None

        if cell is None:
            f = self.get_basis_functions()[n]
            f.plot(self.reference, filename, None, None, None, n, self._value_scale, **kwargs)
        else:
            f = self.map_to_cell(cell.vertices)[n]
            f.plot(cell, filename, None, None, None, n, self._value_scale, **kwargs)

    @abstractmethod
    def map_to_cell(
        self, vertices_in: SetOfPointsInput,
        basis: typing.Optional[typing.List[AnyFunction]] = None,
        forward_map: typing.Optional[PointType] = None,
        inverse_map: typing.Optional[PointType] = None
    ) -> typing.List[AnyFunction]:
        """Map the basis onto a cell using the appropriate mapping for the element.

        Args:
            vertices_in: The vertices of the cell
            basis: The basis functions
            forward_map: The map from the reference to the cell
            inverse_map: The map to the reference from the cell

        Returns:
            The basis functions mapped to the cell
        """

    @abstractmethod
    def get_polynomial_basis(self) -> typing.List[AnyFunction]:
        """Get the symbolic polynomial basis for the element.

        Returns:
            The polynomial basis
        """

    @abstractproperty
    def maximum_degree(self) -> int:
        """Get the maximum degree of this polynomial set for the element."""

    def test(self):
        """Run tests for this element."""
        if self.order <= self._max_continuity_test_order:
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
                except (MappingNotImplemented, NonDefaultReferenceError):
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

                if self.reference.tdim == 1:
                    f = get_piece(f, (0, ))
                    g = get_piece(g, (0, ))
                elif self.reference.tdim == 2:
                    f = get_piece(f, (0, sympy.Rational(1, 2)))
                    g = get_piece(g, (0, sympy.Rational(1, 2)))
                elif self.reference.tdim == 3:
                    f = get_piece(f, (0, sympy.Rational(1, 3), sympy.Rational(1, 3)))
                    g = get_piece(g, (0, sympy.Rational(1, 3), sympy.Rational(1, 3)))

                if continuity[0] == "C":
                    order = int(continuity[1:])
                    deriv_f = [f]
                    deriv_g = [g]
                    f = [f]
                    g = [g]
                    for _ in range(order):
                        deriv_f = [d.diff(i) for d in deriv_f for i in x[:self.reference.tdim]]
                        f += deriv_f
                        deriv_g = [d.diff(i) for d in deriv_g for i in x[:self.reference.tdim]]
                        g += deriv_g

                    f = [i.subs(x[0], 0) for i in f]
                    g = [i.subs(x[0], 0) for i in g]
                else:
                    f = f.subs(x[0], 0)
                    g = g.subs(x[0], 0)

                if continuity[0] == "C":
                    pass

                elif continuity == "H(div)":
                    f = f[0]
                    g = g[0]
                elif continuity == "inner H(div)":
                    f = f[0, 0]
                    g = g[0, 0]
                elif continuity == "H(curl)":
                    f = f[1:]
                    g = g[1:]
                elif continuity == "inner H(curl)":
                    if len(vertices[0]) == 2:
                        f = f[1, 1]
                        g = g[1, 1]
                    if len(vertices[0]) == 3:
                        if dim == 1:
                            vs = self.reference.sub_entities(1)[entities[0]]
                            v0 = self.reference.vertices[vs[0]]
                            v1 = self.reference.vertices[vs[1]]
                            tangent = VectorFunction(v1) - VectorFunction(v0)
                            f = sum(i * f[ni, nj] * j
                                    for ni, i in enumerate(tangent)
                                    for nj, j in enumerate(tangent))
                            g = sum(i * g[ni, nj] * j
                                    for ni, i in enumerate(tangent)
                                    for nj, j in enumerate(tangent))
                        else:
                            assert dim == 2
                            f = [f[1, 1], f[2, 2]]
                            g = [g[1, 1], g[2, 2]]
                elif continuity == "integral inner H(div)":
                    f = f[0, 0].integrate((x[1], 0, 1))
                    g = g[0, 0].integrate((x[1], 0, 1))
                else:
                    raise ValueError(f"Unknown continuity: {continuity}")

                assert allequal(f, g)

    def get_tensor_factorisation(
        self
    ) -> typing.List[typing.Tuple[str, typing.List[FiniteElement], typing.List[int]]]:
        """Get the representation of the element as a tensor product.

        Returns:
            The tensor factorisation
        """
        raise NoTensorProduct()

    def _get_basis_functions_tensor(self) -> typing.List[AnyFunction]:
        """Compute the basis functions using the space's tensor product factorisation.

        Returns:
            The basis functions
        """
        factorisation = self.get_tensor_factorisation()
        basis = {}
        for t_type, factors, perm in factorisation:
            if t_type == "scalar":
                tensor_bases = [[i.subs(x[0], x_i) for i in f.get_basis_functions()]
                                for x_i, f in zip(x, factors)]
                for p, k in zip(perm, product(*tensor_bases)):
                    basis[p] = ScalarFunction(1)
                    for i in k:
                        basis[p] *= i
            else:
                raise ValueError(f"Unknown tensor product type: {t_type}")
        return [basis[i] for i in range(len(basis))]

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the keyword arguments used to create this element.

        Returns:
            Keyword arguments dictionary
        """
        return {}

    @property
    def name(self) -> str:
        """Get the name of the element.

        Returns:
            The name of the element's family
        """
        return self.names[0]

    names: typing.List[str] = []
    references: typing.List[str] = []
    last_updated = version
    cache = True
    _max_continuity_test_order = 4


class CiarletElement(FiniteElement):
    """Finite element defined using the Ciarlet definition."""

    def __init__(
        self, reference: Reference, order: int, basis: typing.List[FunctionInput],
        dofs: ListOfFunctionals, domain_dim: int, range_dim: int,
        range_shape: typing.Optional[typing.Tuple[int, ...]] = None
    ):
        """Create a Ciarlet element.

        Args:
            reference: The reference cell
            order: The polynomial order
            basis: The polynomial basis
            dofs: The DOF functionals
            domain_dim: The topological dimension of the cell
            range_dim: The dimension of the range
            range_shape: The shape of the range
        """
        super().__init__(reference, order, len(dofs), domain_dim, range_dim, range_shape)
        assert len(basis) == len(dofs)
        self._basis = [parse_function_input(b) for b in basis]
        for b in self._basis:
            assert isinstance(b, AnyFunction)
        self.dofs = dofs
        self._basis_functions: typing.Union[typing.List[AnyFunction], None] = None
        self._maximum_degree = None
        if reference.name == "pyramid":
            self._maximum_degree = order

    def entity_dofs(self, entity_dim: int, entity_number: int) -> typing.List[int]:
        """Get the numbers of the DOFs associated with the given entity.

        Args:
            entity_dim: The dimension of the entity
            entity_number: The number of the entity

        Returns:
            The numbers of the DOFs associated with the entity
        """
        return [i for i, j in enumerate(self.dofs) if j.entity == (entity_dim, entity_number)]

    def dof_plot_positions(self) -> typing.List[PointType]:
        """Get the points to plot each DOF at on a DOF diagram.

        Returns:
            The DOF positions
        """
        return [d.adjusted_dof_point() for d in self.dofs]

    def dof_directions(self) -> typing.List[typing.Union[PointType, None]]:
        """Get the direction associated with each DOF.

        Returns:
            The DOF directions
        """
        return [d.dof_direction() for d in self.dofs]

    def dof_entities(self) -> typing.List[typing.Tuple[int, int]]:
        """Get the entities that each DOF is associated with.

        Returns:
            The entities
        """
        return [d.entity for d in self.dofs]

    def get_polynomial_basis(self) -> typing.List[AnyFunction]:
        """Get the symbolic polynomial basis for the element.

        Returns:
            The polynomial basis
        """
        return self._basis

    @property
    def maximum_degree(self) -> int:
        """Get the maximum degree of this polynomial set for the element."""
        if self._maximum_degree is None:
            self._maximum_degree = max(p.maximum_degree(self.reference) for p in self._basis)
        return self._maximum_degree

    def get_dual_matrix(
        self, inverse=False, caching=True
    ) -> sympy.matrices.dense.MutableDenseMatrix:
        """Get the dual matrix.

        Args:
            inverse: Should the dual matrix be inverted?
            caching: Should the result be cached

        Returns:
            The dual matrix
        """
        if caching and self.cache:
            cid = (f"{self.__class__.__name__} {self.order} {self.reference.vertices} "
                   f"{self.init_kwargs()} {self.last_updated}")
            matrix_type = "dualinv" if inverse else "dual"
            mat = load_cached_matrix(matrix_type, cid, (len(self.dofs), len(self.dofs)))
            if mat is None:
                mat = self.get_dual_matrix(inverse, caching=False)
                save_cached_matrix(matrix_type, cid, mat)
            return mat
        else:
            mat = sympy.Matrix([[d.eval_symbolic(b).as_sympy() for d in self.dofs]
                                for b in self.get_polynomial_basis()])
            if inverse:
                return mat.inv("LU")
            else:
                return mat

    def get_basis_functions(
        self, use_tensor_factorisation: bool = False
    ) -> typing.List[AnyFunction]:
        """Get the basis functions of the element.

        Args:
            use_tensor_factorisation: Should a tensor factorisation be used?

        Returns:
            The basis functions
        """
        if self._basis_functions is None:
            if use_tensor_factorisation:
                self._basis_functions = self._get_basis_functions_tensor()
            else:
                minv = self.get_dual_matrix(inverse=True)

                sfs: typing.List[AnyFunction] = []
                pb = self.get_polynomial_basis()
                for i, dof in enumerate(self.dofs):
                    sf = ScalarFunction(minv[i, 0]) * pb[0]
                    for c, d in zip(minv.row(i)[1:], pb[1:]):
                        sf += ScalarFunction(c) * d
                    sfs.append(sf)

                self._basis_functions = sfs

        assert isinstance(self._basis_functions, list)

        return self._basis_functions

    def plot_basis_function(
        self, n: int, filename: typing.Union[str, typing.List[str]],
        cell: typing.Optional[Reference] = None, **kwargs: typing.Any
    ):
        """Plot a diagram showing a basis function.

        Args:
            n: The basis function number
            filename: The file name
            cell: The cell to push the basis function to and plot on
            kwargs: Keyword arguments
        """
        if cell is not None:
            raise NotImplementedError()

        if self._value_scale is None:
            values = self.tabulate_basis_float(self.reference.make_lattice_float(6))
            max_v = 0.0
            for row in values:
                assert isinstance(row, tuple)
                for i in row:
                    max_v = max(max_v, float(parse_function_input(i).norm()))
            self._value_scale = 1 / sympy.Float(max_v)

        f = self.get_basis_functions()[n]
        d = self.dofs[n]
        assert self._value_scale is not None
        f.plot(self.reference, filename, d.dof_point(), d.dof_direction(), d.entity, n,
               self._value_scale, **kwargs)

    def map_to_cell(
        self, vertices_in: SetOfPointsInput,
        basis: typing.Optional[typing.List[AnyFunction]] = None,
        forward_map: typing.Optional[PointType] = None,
        inverse_map: typing.Optional[PointType] = None
    ) -> typing.List[AnyFunction]:
        """Map the basis onto a cell using the appropriate mapping for the element.

        Args:
            vertices_in: The vertices of the cell
            basis: The basis functions
            forward_map: The map from the reference to the cell
            inverse_map: The map to the reference from the cell

        Returns:
            The basis functions mapped to the cell
        """
        vertices = parse_set_of_points_input(vertices_in)
        if basis is None:
            basis = self.get_basis_functions()
        if forward_map is None:
            forward_map = self.reference.get_map_to(vertices)
        if inverse_map is None:
            inverse_map = self.reference.get_inverse_map_to(vertices)

        try:
            functions: typing.List[AnyFunction] = [ScalarFunction(0) for f in basis]
            for dim in range(self.reference.tdim + 1):
                for e in range(self.reference.sub_entity_count(dim)):
                    entity_dofs = self.entity_dofs(dim, e)
                    dofs_by_type: typing.Dict[
                        typing.Tuple[typing.Type, typing.Union[str, None]], typing.List[int]
                    ] = {}
                    for d in entity_dofs:
                        dof = self.dofs[d]
                        t = (type(dof), dof.mapping)
                        if t not in dofs_by_type:
                            dofs_by_type[t] = []
                        dofs_by_type[t].append(d)
                    for ds in dofs_by_type.values():
                        mapped_dofs = self.dofs[ds[0]].perform_mapping(
                            [basis[d] for d in ds],
                            forward_map, inverse_map)
                        for d_n, mdof in zip(ds, mapped_dofs):
                            functions[d_n] = mdof

            for fun in functions:
                if isinstance(fun, PiecewiseFunction):
                    fun.map_pieces(forward_map)

            return functions
        except MappingNotImplemented:
            element = self.__class__(
                self.reference.__class__(vertices=vertices), self.order, **self.init_kwargs())
            return element.get_basis_functions()

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
                    assert d.eval(f) == 1
                else:
                    assert d.eval(f) == 0
                assert d.entity_dim() is not None


class DirectElement(FiniteElement):
    """Finite element defined directly."""

    _basis_functions: typing.List[AnyFunction]

    def __init__(
        self, reference: Reference, order: int, basis_functions: typing.List[FunctionInput],
        basis_entities: typing.List[typing.Tuple[int, int]],
        domain_dim: int, range_dim: int, range_shape: typing.Optional[typing.Tuple[int, ...]] = None
    ):
        """Create a direct element.

        Args:
            reference: The reference cell
            order: The polynomial order
            basis_functions: The basis functions
            basis_entities: The entitiy each basis function is associated with
            domain_dim: The topological dimension of the cell
            range_dim: The dimension of the range
            range_shape: The shape of the range
        """
        super().__init__(reference, order, len(basis_functions), domain_dim, range_dim,
                         range_shape)
        self._basis_entities = basis_entities
        self._basis_functions = [parse_function_input(f) for f in basis_functions]

    def entity_dofs(self, entity_dim: int, entity_number: int) -> typing.List[int]:
        """Get the numbers of the DOFs associated with the given entity.

        Args:
            entity_dim: The dimension of the entity
            entity_number: The number of the entity

        Returns:
            The numbers of the DOFs associated with the entity
        """
        return [i for i, j in enumerate(self._basis_entities) if j == (entity_dim, entity_number)]

    def dof_plot_positions(self) -> typing.List[PointType]:
        """Get the points to plot each DOF at on a DOF diagram.

        Returns:
            The DOF positions
        """
        positions = []
        for n, (dim, e_n) in enumerate(self._basis_entities):
            ed = self.entity_dofs(dim, e_n)
            entity_n = ed.index(n)
            dof_count = len(ed)
            sub_ref = self.reference.sub_entity(dim, e_n)
            if dim == 0:
                assert entity_n == 0
                positions.append(sub_ref.vertices[0])
            elif dim == 1:
                positions.append(tuple(
                    o + sympy.Rational((entity_n + 1) * a, dof_count + 1)
                    for o, a in zip(sub_ref.origin, *sub_ref.axes)))
            elif dim == 2:
                ne = 1
                while ne * (ne + 1) // 2 < dof_count:
                    ne += 1
                i = 0
                while entity_n >= ne - i:
                    entity_n -= ne - i
                    i += 1
                positions.append(tuple(
                    o + sympy.Rational((entity_n + 1) * a + (i + 1) * b, ne + 1)
                    for o, a, b in zip(sub_ref.origin, *sub_ref.axes)))
            elif dim == 3:
                ne = 1
                while ne * (ne + 1) * (ne + 2) // 6 < n:
                    ne += 1
                i = 0
                while entity_n >= (ne - i) * (ne + 1 - i) // 2:
                    entity_n -= (ne - i) * (ne + 1 - i) // 2
                    i += 1
                j = 0
                while entity_n >= ne - j:
                    entity_n -= ne - j
                    j += 1
                positions.append(tuple(
                    o + sympy.Rational((entity_n + 1) * a + (j + 1) * b + (i + 1) * c, n + 1)
                    for o, a, b, c in zip(sub_ref.origin, *sub_ref.axes)))

        return positions

    def dof_directions(self) -> typing.List[typing.Union[PointType, None]]:
        """Get the direction associated with each DOF.

        Returns:
            The DOF directions
        """
        return [None for d in self._basis_entities]

    def dof_entities(self) -> typing.List[typing.Tuple[int, int]]:
        """Get the entities that each DOF is associated with.

        Returns:
            The entities
        """
        return self._basis_entities

    def get_basis_functions(
        self, use_tensor_factorisation: bool = False
    ) -> typing.List[AnyFunction]:
        """Get the basis functions of the element.

        Args:
            use_tensor_factorisation: Should a tensor factorisation be used?

        Returns:
            The basis functions
        """
        if use_tensor_factorisation:
            return self._get_basis_functions_tensor()

        return self._basis_functions

    def map_to_cell(
        self, vertices_in: SetOfPointsInput,
        basis: typing.Optional[typing.List[AnyFunction]] = None,
        forward_map: typing.Optional[PointType] = None,
        inverse_map: typing.Optional[PointType] = None
    ) -> typing.List[AnyFunction]:
        """Map the basis onto a cell using the appropriate mapping for the element.

        Args:
            vertices_in: The vertices of the cell
            basis: The basis functions
            forward_map: The map from the reference to the cell
            inverse_map: The map to the reference from the cell

        Returns:
            The basis functions mapped to the cell
        """
        raise MappingNotImplemented()
        # TODO: make this work
        # vertices = parse_set_of_points_input(vertices_in)
        # e = self.__class__(self.reference.__class__(vertices=vertices), self.order,
        #                   **self.init_kwargs())
        # return e.get_basis_functions()

    def get_polynomial_basis(self) -> typing.List[AnyFunction]:
        """Get the symbolic polynomial basis for the element.

        Returns:
            The polynomial basis
        """
        raise NotImplementedError()

    @property
    def maximum_degree(self) -> int:
        """Get the maximum degree of this polynomial set for the element."""
        raise NotImplementedError()

    def test(self):
        """Run tests for this element."""
        super().test()
        self.test_independence()

    def test_independence(self):
        """Test that the basis functions of this element are linearly independent."""
        basis = self.get_basis_functions()
        all_terms = set()

        scalar = basis[0].is_scalar

        if scalar:
            for f in basis:
                f_s = f.as_sympy()
                assert isinstance(f_s, sympy.core.expr.Expr)
                for term in f_s.as_coefficients_dict():
                    all_terms.add(term)
            mat = [[0 for i in all_terms] for j in basis]
            for i, t in enumerate(all_terms):
                for j, f in enumerate(basis):
                    f_s = f.as_sympy()
                    assert isinstance(f_s, sympy.core.expr.Expr)
                    fd = f_s.as_coefficients_dict()
                    if t in fd:
                        mat[j][i] = fd[t]
        else:
            for f in basis:
                for fi, fpart in enumerate(f):
                    fpart_s = fpart.as_sympy()
                    assert isinstance(fpart_s, sympy.core.expr.Expr)
                    for term in fpart_s.as_coefficients_dict():
                        all_terms.add((fi, term))
            mat = [[0 for i in all_terms] for j in basis]
            for i, (fi, t) in enumerate(all_terms):
                for j, f in enumerate(basis):
                    ffi_s = f[fi].as_sympy()
                    assert isinstance(ffi_s, sympy.core.expr.Expr)
                    fd = ffi_s.as_coefficients_dict()
                    if t in fd:
                        mat[j][i] = fd[t]
        mat = sympy.Matrix(mat)

        assert mat.rank() == mat.rows


class EnrichedElement(FiniteElement):
    """Finite element defined directly."""

    _basis_functions: typing.Optional[typing.List[AnyFunction]]

    def __init__(
        self, subelements: typing.List[FiniteElement],
    ):
        """Create an enriched element.

        Args:
            subelements: The sub elements
        """
        reference = subelements[0].reference
        order = subelements[0].order
        domain_dim = subelements[0].domain_dim
        range_dim = subelements[0].range_dim
        range_shape = subelements[0].range_shape
        for e in subelements:
            assert e.reference == reference
            assert e.domain_dim == domain_dim
            assert e.range_dim == range_dim
            assert e.range_shape == range_shape
        self._basis_functions = None
        self._subelements = subelements

        super().__init__(reference, order, sum(e.space_dim for e in subelements),
                         domain_dim, range_dim, range_shape)

    def entity_dofs(self, entity_dim: int, entity_number: int) -> typing.List[int]:
        """Get the numbers of the DOFs associated with the given entity.

        Args:
            entity_dim: The dimension of the entity
            entity_number: The number of the entity

        Returns:
            The numbers of the DOFs associated with the entity
        """
        start = 0
        dofs = []
        for e in self._subelements:
            dofs += [start + i for i in e.entity_dofs(entity_dim, entity_number)]
            start += e.space_dim
        return dofs

    def dof_plot_positions(self) -> typing.List[PointType]:
        """Get the points to plot each DOF at on a DOF diagram.

        Returns:
            The DOF positions
        """
        positions = []
        for e in self._subelements:
            positions += e.dof_plot_positions()
        return positions

    def dof_directions(self) -> typing.List[typing.Union[PointType, None]]:
        """Get the direction associated with each DOF.

        Returns:
            The DOF directions
        """
        positions = []
        for e in self._subelements:
            positions += e.dof_directions()
        return positions

    def dof_entities(self) -> typing.List[typing.Tuple[int, int]]:
        """Get the entities that each DOF is associated with.

        Returns:
            The entities
        """
        positions = []
        for e in self._subelements:
            positions += e.dof_entities()
        return positions

    def get_basis_functions(
        self, use_tensor_factorisation: bool = False
    ) -> typing.List[AnyFunction]:
        """Get the basis functions of the element.

        Args:
            use_tensor_factorisation: Should a tensor factorisation be used?

        Returns:
            The basis functions
        """
        if use_tensor_factorisation:
            return self._get_basis_functions_tensor()

        if self._basis_functions is None:
            self._basis_functions = []
            for e in self._subelements:
                self._basis_functions += e.get_basis_functions()

        return self._basis_functions

    def map_to_cell(
        self, vertices_in: SetOfPointsInput,
        basis: typing.Optional[typing.List[AnyFunction]] = None,
        forward_map: typing.Optional[PointType] = None,
        inverse_map: typing.Optional[PointType] = None
    ) -> typing.List[AnyFunction]:
        """Map the basis onto a cell using the appropriate mapping for the element.

        Args:
            vertices_in: The vertices of the cell
            basis: The basis functions
            forward_map: The map from the reference to the cell
            inverse_map: The map to the reference from the cell

        Returns:
            The basis functions mapped to the cell
        """
        out = []
        for e in self._subelements:
            out += e.map_to_cell(vertices_in, basis, forward_map, inverse_map)
        return out

    def get_polynomial_basis(self) -> typing.List[AnyFunction]:
        """Get the symbolic polynomial basis for the element.

        Returns:
            The polynomial basis
        """
        raise NotImplementedError()

    @property
    def maximum_degree(self) -> int:
        """Get the maximum degree of this polynomial set for the element."""
        raise NotImplementedError()

    def test(self):
        """Run tests for this element."""
        super().test()


class ElementBasisFunction(BasisFunction):
    """A basis function of a finite element."""

    def __init__(self, element: FiniteElement, n: int):
        """Create an element basis function.

        Args:
            element: The finite element
            n: The basis function number
        """
        if element.range_dim == 1:
            super().__init__(scalar=True)
        else:
            if element.range_shape is None or len(element.range_shape) == 1:
                super().__init__(vector=True)
            else:
                assert len(element.range_shape) == 2
                super().__init__(matrix=True)
        self.element = element
        self.n = n

    def get_function(self) -> AnyFunction:
        """Get the actual basis function.

        Returns:
            The basis function
        """
        return self.element.get_basis_functions()[self.n]
