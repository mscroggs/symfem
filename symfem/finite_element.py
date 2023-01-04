"""Abstract finite element classes and functions."""

from __future__ import annotations

import math
import typing
from abc import ABC, abstractmethod
from itertools import product

import sympy

from .basis_functions import BasisFunction
from .functionals import ListOfFunctionals
from .functions import (AnyFunction, FunctionInput, ScalarFunction, VectorFunction,
                        parse_function_input)
from .geometry import PointType, SetOfPointsInput, parse_set_of_points_input
from .piecewise_functions import PiecewiseFunction
from .plotting import Picture, colors
from .references import Reference
from .symbols import x
from .utils import allequal

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
        pass

    @abstractmethod
    def entity_dofs(self, entity_dim: int, entity_number: int) -> typing.List[int]:
        """Get the numbers of the DOFs associated with the given entity.

        Args:
            entity_dim: The dimension of the entity
            entity_number: The number of the entity

        Returns:
            The numbers of the DOFs associated with the entity
        """
        pass

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
        pass

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
        self, n: int, filename: typing.Union[str, typing.List[str]], **kwargs: typing.Any
    ):
        """Plot a diagram showing a basis function.

        Args:
            n: The basis function number
            filename: The file name
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

        f = self.get_basis_functions()[n]
        assert self._value_scale is not None
        f.plot(self.reference, filename, None, None, None, n, self._value_scale, **kwargs)

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
        pass

    @abstractmethod
    def get_polynomial_basis(self) -> typing.List[AnyFunction]:
        """Get the symbolic polynomial basis for the element.

        Returns:
            The polynomial basis
        """
        pass

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

                f = f.subs(x[0], 0)
                g = g.subs(x[0], 0)

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

    @property
    def name(self) -> str:
        """Get the name of the element.

        Returns:
            The name of the element's family
        """
        return self.names[0]

    names: typing.List[str] = []
    references: typing.List[str] = []


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

    def entity_dofs(self, entity_dim: int, entity_number: int) -> typing.List[int]:
        """Get the numbers of the DOFs associated with the given entity.

        Args:
            entity_dim: The dimension of the entity
            entity_number: The number of the entity

        Returns:
            The numbers of the DOFs associated with the entity
        """
        return [i for i, j in enumerate(self.dofs) if j.entity == (entity_dim, entity_number)]

    def get_polynomial_basis(self) -> typing.List[AnyFunction]:
        """Get the symbolic polynomial basis for the element.

        Returns:
            The polynomial basis
        """
        return self._basis

    def get_dual_matrix(self) -> sympy.matrices.dense.MutableDenseMatrix:
        """Get the dual matrix.

        Returns:
            The dual matrix
        """
        mat = []
        for b in self.get_polynomial_basis():
            row = []
            for d in self.dofs:
                entry = d.eval_symbolic(b).as_sympy()
                row.append(entry)
            mat.append(row)
        return sympy.Matrix(mat)

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the keyword arguments used to create this element.

        Returns:
            Keyword arguments dictionary
        """
        return {}

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
                m = self.get_dual_matrix()
                assert isinstance(m, sympy.Matrix)
                minv = m.inv("LU")

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
        self, n: int, filename: typing.Union[str, typing.List[str]], **kwargs: typing.Any
    ):
        """Plot a diagram showing a basis function.

        Args:
            n: The basis function number
            filename: The file name
            kwargs: Keyword arguments
        """
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

        dofs_by_subentity: typing.Dict[int, typing.Dict[int, ListOfFunctionals]] = {
            i: {j: [] for j in range(self.reference.sub_entity_count(i))}
            for i in range(self.reference.tdim + 1)}

        for d in self.dofs:
            dofs_by_subentity[d.entity[0]][d.entity[1]].append(d)

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
                dofs = dofs_by_subentity[dim][e]
                dofs.sort(key=lambda d: img.z(d.adjusted_dof_point()))
                for d in dofs:
                    direction = d.dof_direction()
                    if direction is not None:
                        shifted = False
                        for d2 in self.dofs:
                            if d != d2 and d.adjusted_dof_point() == d2.adjusted_dof_point():
                                shifted = True
                                break
                        img.add_dof_arrow(d.adjusted_dof_point(), direction, self.dofs.index(d),
                                          colors.entity(d.entity[0]), shifted)
                    else:
                        img.add_dof_marker(
                            d.adjusted_dof_point(), self.dofs.index(d), colors.entity(d.entity[0]))

        img.save(filename, plot_options=plot_options)

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
                        forward_map, inverse_map, self.reference.tdim)
                    for d_n, mdof in zip(ds, mapped_dofs):
                        functions[d_n] = mdof

        for fun in functions:
            if isinstance(fun, PiecewiseFunction):
                fun.map_pieces(forward_map)

        return functions

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
        """Create a Ciarlet element.

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
        raise NotImplementedError()

    def get_polynomial_basis(self) -> typing.List[AnyFunction]:
        """Get the symbolic polynomial basis for the element.

        Returns:
            The polynomial basis
        """
        raise NotImplementedError()

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

        dofs_by_subentity: typing.Dict[int, typing.Dict[int, typing.List[int]]] = {
            i: {j: [] for j in range(self.reference.sub_entity_count(i))}
            for i in range(self.reference.tdim + 1)}

        for i, e in enumerate(self._basis_entities):
            dofs_by_subentity[e[0]][e[1]].append(i)

        for entities in self.reference.z_ordered_entities():
            for dim, e_n in entities:
                if dim == 1:
                    pts = tuple(self.reference.vertices[i] for i in self.reference.edges[e_n])
                    img.add_line(pts[0], pts[1], colors.BLACK)
                if dim == 2:
                    pts = tuple(self.reference.vertices[i] for i in self.reference.faces[e_n])
                    if len(pts) == 4:
                        pts = (pts[0], pts[1], pts[3], pts[2])
                    img.add_fill(pts, colors.WHITE, 0.5)

            for dim, e_n in entities:
                n = len(dofs_by_subentity[dim][e_n])
                if n > 0:
                    sub_ref = self.reference.sub_entity(dim, e_n)
                    points: typing.List[PointType] = []
                    if dim == 0:
                        assert n == 1
                        points = [sub_ref.vertices[0]]
                    elif dim == 1:
                        points = [tuple(o + sympy.Rational(i * a, n + 1)
                                        for o, a in zip(sub_ref.origin, *sub_ref.axes))
                                  for i in range(1, n + 1)]
                    elif dim == 2:
                        ne = 1
                        while ne * (ne + 1) // 2 < n:
                            ne += 1
                        points = [tuple(o + sympy.Rational(i * a + j * b, n + 1)
                                        for o, a, b in zip(sub_ref.origin, *sub_ref.axes))
                                  for i in range(1, ne + 1) for j in range(1, ne + 1 - i)]
                    elif dim == 3:
                        ne = 1
                        while ne * (ne + 1) * (ne + 2) // 6 < n:
                            ne += 1
                        points = [tuple(o + sympy.Rational(i * a + j * b + k * c, n + 1)
                                        for o, a, b, c in zip(sub_ref.origin, *sub_ref.axes))
                                  for i in range(1, ne + 1) for j in range(1, ne + 1 - i)
                                  for k in range(1, ne + 1 - i - j)]

                    for p, d in zip(points, dofs_by_subentity[dim][e_n]):
                        img.add_dof_marker(p, d, colors.entity(dim))

        img.save(filename, plot_options=plot_options)

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
