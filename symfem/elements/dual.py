"""Dual elements.

These elements' definitions appear in https://doi.org/10.1016/j.crma.2004.12.022
(Buffa, Christiansen, 2005)
"""

import typing

import sympy

from ..finite_element import FiniteElement
from ..functions import AnyFunction, FunctionInput, VectorFunction
from ..geometry import PointType, SetOfPoints, SetOfPointsInput
from ..piecewise_functions import PiecewiseFunction
from ..references import DualPolygon, NonDefaultReferenceError


class DualCiarletElement(FiniteElement):
    """Abstract barycentric finite element."""

    def __init__(
        self, dual_coefficients: typing.List[typing.List[typing.List[
            typing.Union[int, sympy.core.expr.Expr]]]],
        fine_space: str, reference: DualPolygon, order: int,
        dof_entities: typing.List[typing.Tuple[int, int]],
        domain_dim: int, range_dim: int,
        range_shape: typing.Optional[typing.Tuple[int, ...]] = None,
        dof_directions: typing.Optional[SetOfPoints] = None
    ):
        """Create a dual element.

        Args:
            dual_coefficients: the coefficients that define this element in terms of the basis
                               functions of the fine space
            fine_space: the family of the fine space
            reference: The reference element
            order: The polynomia order of the fine space
            dof_entities: The cell entity that each basis function is associated with
            domain_dim: the topological dimension of the domain
            range_dim: the dimension of the range
            range_shape: the shape of the range
            dof_directions: The direction that each basis function is associated with
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        self.dual_coefficients = dual_coefficients
        self.fine_space = fine_space
        super().__init__(reference, order, len(dual_coefficients), domain_dim, range_dim,
                         range_shape=range_shape)
        self._basis_functions: typing.Union[typing.List[AnyFunction], None] = None
        self._dof_entities = dof_entities
        self._dof_directions = dof_directions

    def get_polynomial_basis(
        self, reshape: bool = True
    ) -> typing.List[AnyFunction]:
        """Get the symbolic polynomial basis for the element.

        Returns:
            The polynomial basis
        """
        raise ValueError("Polynomial basis not supported for barycentric dual elements.")

    @property
    def maximum_degree(self) -> int:
        """Get the maximum degree of this polynomial set for the element."""
        raise NotImplementedError()

    def get_dual_matrix(self) -> sympy.matrices.dense.MutableDenseMatrix:
        """Get the dual matrix.

        Returns:
            The dual matrix
        """
        raise ValueError("Dual matrix not supported for barycentric dual elements.")

    def get_basis_functions(
        self, use_tensor_factorisation: bool = False
    ) -> typing.List[AnyFunction]:
        """Get the basis functions of the element.

        Args:
            use_tensor_factorisation: Should a tensor factorisation be used?

        Returns:
            The basis functions
        """
        assert not use_tensor_factorisation

        if self._basis_functions is None:
            from symfem import create_element

            bfs: typing.List[AnyFunction] = []
            sub_e = create_element("triangle", self.fine_space, self.order)
            for coeff_list in self.dual_coefficients:
                v0 = self.reference.origin
                pieces: typing.Dict[SetOfPointsInput, FunctionInput] = {}
                for coeffs, v1, v2 in zip(
                    coeff_list, self.reference.vertices,
                    self.reference.vertices[1:] + self.reference.vertices[:1]
                ):
                    sub_basis = sub_e.map_to_cell((v0, v1, v2))

                    if self.range_dim == 1:
                        sub_fun = sympy.Integer(0)
                        for a, b in zip(coeffs, sub_basis):
                            sub_fun += a * b
                    else:
                        sf_list = []
                        for i in range(self.range_dim):
                            sf_item = sympy.Integer(0)
                            for a, b in zip(coeffs, sub_basis):
                                assert isinstance(b, VectorFunction)
                                sf_item += a * b[i]
                            sf_list.append(sf_item)
                        sub_fun = tuple(sf_list)
                    pieces[(v0, v1, v2)] = sub_fun
                bfs.append(PiecewiseFunction(pieces, 2))
            assert len(bfs) == len(self._dof_entities)
            self._basis_functions = bfs

        assert self._basis_functions is not None
        return self._basis_functions

    def entity_dofs(self, entity_dim: int, entity_number: int) -> typing.List[int]:
        """Get the numbers of the DOFs associated with the given entity.

        Args:
            entity_dim: The dimension of the entity
            entity_number: The number of the entity

        Returns:
            The numbers of the DOFs associated with the entity
        """
        out = []
        for i, e in enumerate(self._dof_entities):
            if e == (entity_dim, entity_number):
                out.append(i)
        return out

    def dof_plot_positions(self) -> typing.List[PointType]:
        """Get the points to plot each DOF at on a DOF diagram.

        Returns:
            The DOF positions
        """
        positions = []
        for d, (dim, e_n) in enumerate(self._dof_entities):
            if dim == 0:
                positions.append(self.reference.vertices[e_n])
            elif dim == 1:
                positions.append(tuple((a + b) / 2 for a, b in zip(
                    self.reference.vertices[self.reference.edges[e_n][0]],
                    self.reference.vertices[self.reference.edges[e_n][1]],
                )))
            elif dim == 2:
                positions.append(self.reference.midpoint())
            else:
                raise ValueError("Unsupported tdim")

        return positions

    def dof_directions(self) -> typing.List[typing.Union[PointType, None]]:
        """Get the direction associated with each DOF.

        Returns:
            The DOF directions
        """
        if self._dof_directions is None:
            return [None for d in self._dof_entities]
        return list(self._dof_directions)

    def dof_entities(self) -> typing.List[typing.Tuple[int, int]]:
        """Get the entities that each DOF is associated with.

        Returns:
            The entities
        """
        return self._dof_entities

    def map_to_cell(
        self, vertices_in: SetOfPointsInput, basis:
        typing.Optional[typing.List[AnyFunction]] = None,
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


class Dual(DualCiarletElement):
    """Barycentric dual finite element."""

    def __init__(self, reference: DualPolygon, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        dual_coefficients: typing.List[typing.List[typing.List[
            typing.Union[int, sympy.core.expr.Expr]]]] = []
        if order == 0:
            dual_coefficients = [
                [[1] for i in range(2 * reference.number_of_triangles)]
            ]
            fine_space = "Lagrange"
            dof_entities = [(2, 0)]
        else:
            dual_coefficients = [
                [[sympy.Rational(1, reference.number_of_triangles), 0, 0]
                 for i in range(2 * reference.number_of_triangles)]
                for j in range(reference.number_of_triangles)
            ]

            for j in range(reference.number_of_triangles):
                dual_coefficients[j][2 * j - 1][2] = 1
                dual_coefficients[j][2 * j][1] = 1
                dual_coefficients[j][2 * j - 2][2] = sympy.Rational(1, 2)
                dual_coefficients[j][2 * j - 1][1] = sympy.Rational(1, 2)
                dual_coefficients[j][2 * j][2] = sympy.Rational(1, 2)
                dual_coefficients[j][2 * j + 1][1] = sympy.Rational(1, 2)

            dof_entities = [(0, i) for i in range(0, len(reference.vertices), 2)]

            fine_space = "Lagrange"

        super().__init__(
            dual_coefficients, fine_space, reference, order, dof_entities, reference.tdim, 1
        )

    names = ["dual polynomial", "dual P", "dual"]
    references = ["dual polygon"]
    min_order = 0
    max_order = 1
    continuity = "C0"
    last_updated = "2023.05"


class BuffaChristiansen(DualCiarletElement):
    """Buffa-Christiansen barycentric dual finite element."""

    def __init__(self, reference: DualPolygon, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        assert order == 1
        dual_coefficients: typing.List[typing.List[typing.List[
            typing.Union[int, sympy.core.expr.Expr]]]] = [
            [[0, 0, 0]
             for i in range(2 * reference.number_of_triangles)]
            for j in range(reference.number_of_triangles)
        ]

        for j in range(reference.number_of_triangles):
            dual_coefficients[j][2 * j][0] = sympy.Rational(-1, 2)
            dual_coefficients[j][2 * j - 1][0] = sympy.Rational(-1, 2)
            N = 2 * reference.number_of_triangles
            for i in range(N - 1):
                dual_coefficients[j][(2 * j + i) % N][2] = sympy.Rational(i + 1 - N // 2, N)
                dual_coefficients[j][(2 * j + i + 1) % N][1] = sympy.Rational(i + 1 - N // 2, N)

        dof_entities = [(0, i) for i in range(0, len(reference.vertices), 2)]
        dof_directions: typing.List[PointType] = []
        for i in range(0, len(reference.vertices), 2):
            dof_directions.append(tuple(
                b - a for a, b in zip(reference.origin, reference.vertices[i])))

        super().__init__(
            dual_coefficients, "RT", reference, order, dof_entities, reference.tdim, 2,
            dof_directions=tuple(dof_directions))

    names = ["Buffa-Christiansen", "BC"]
    references = ["dual polygon"]
    min_order = 1
    max_order = 1
    continuity = "H(div)"
    last_updated = "2023.05"


class RotatedBuffaChristiansen(DualCiarletElement):
    """RotatedBuffa-Christiansen barycentric dual finite element."""

    def __init__(self, reference: DualPolygon, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        assert order == 1
        dual_coefficients: typing.List[typing.List[typing.List[
            typing.Union[int, sympy.core.expr.Expr]]]] = [
            [[0, 0, 0]
             for i in range(2 * reference.number_of_triangles)]
            for j in range(reference.number_of_triangles)
        ]

        for j in range(reference.number_of_triangles):
            dual_coefficients[j][2 * j][0] = sympy.Rational(-1, 2)
            dual_coefficients[j][2 * j - 1][0] = sympy.Rational(-1, 2)
            N = 2 * reference.number_of_triangles
            for i in range(N - 1):
                dual_coefficients[j][(2 * j + i) % N][2] = sympy.Rational(N // 2 - 1 - i, N)
                dual_coefficients[j][(2 * j + i + 1) % N][1] = sympy.Rational(N // 2 - 1 - i, N)

        dof_entities = [(0, i) for i in range(0, len(reference.vertices), 2)]
        dof_directions: typing.List[PointType] = []
        for i in range(0, len(reference.vertices), 2):
            dof_directions.append(tuple(
                b - a for a, b in zip(reference.origin, reference.vertices[i])))

        super().__init__(
            dual_coefficients, "N1curl", reference, order, dof_entities,
            reference.tdim, 2, dof_directions=tuple(dof_directions))

    names = ["rotated Buffa-Christiansen", "RBC"]
    references = ["dual polygon"]
    min_order = 1
    max_order = 1
    continuity = "H(div)"
    last_updated = "2023.05"
