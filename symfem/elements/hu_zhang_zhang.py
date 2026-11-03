"""Hu-Zhang-Zhang H(grad curl) tetrahedra, https://doi.org/10.1137/20M1358700.

The corrected Poincare construction follows Sections 3 and 4, with r=k.
"""

from functools import lru_cache
from itertools import product
from math import factorial

import sympy

from symfem import mappings
from symfem.elements.guzman_neilan import get_sub_cells
from symfem.elements.guzman_neilan import poly as bdm_facet
from symfem.elements.lagrange import Lagrange
from symfem.elements.nedelec import NedelecFirstKind
from symfem.finite_element import CiarletElement
from symfem.functionals import DotPointEvaluation, ListOfFunctionals, TangentIntegralMoment
from symfem.piecewise_functions import PiecewiseFunction
from symfem.polynomials import polynomial_set_1d, polynomial_set_vector
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import t, x

__all__ = ["HuZhangZhang"]


def _average(p, vertices):
    """Integrate a polynomial exactly using simplex monomial averages."""
    d = len(vertices) - 1
    point = {
        x[i]: vertices[0][i] + sum(t[j] * (vertices[j + 1][i] - vertices[0][i]) for j in range(d))
        for i in range(3)
    }
    p = sympy.Poly(sympy.expand(p.subs(point, simultaneous=True)), *t[:d])
    return sum(
        c * factorial(d) * sympy.prod(factorial(a) for a in alpha) / factorial(d + sum(alpha))
        for alpha, c in p.terms()
    )


def _div(v):
    return sum(sympy.diff(v[i], x[i]) for i in range(3)).expand()


def _grad(p):
    return sympy.Matrix([sympy.diff(p, xi) for xi in x])


def _coefficients(p, monomials):
    p = sympy.Poly(p, *x)
    return [p.coeff_monomial(m) for m in monomials]


@lru_cache(None)
def _gn_space(reference, order):
    """Construct Sigma^{k,+} using the modified interior-bubble divergence inverse."""
    pieces = get_sub_cells(reference)
    lambdas = [1 - sum(x), *x]
    hats = [
        4 * lambdas[next(i for i, v in enumerate(reference.vertices) if v not in cell)]
        for cell in pieces
    ]
    m = [p.as_sympy() for p in polynomial_set_1d(3, 2)]
    correction = []
    for j in range(3):
        for p in bdm_facet(reference, j):
            correction_values = []
            for h in hats:
                values = []
                for c in p.as_sympy():
                    values.append(sympy.expand(h ** (3 - j) * c))
                correction_values.append(sympy.Matrix(values))
            correction.append(correction_values)
    correction_columns = []
    for field in correction:
        coefficients = []
        for value in field:
            coefficients.extend(_coefficients(_div(value), m))
        correction_columns.append(sympy.Matrix(coefficients))
    matrix = sympy.Matrix.hstack(*correction_columns)
    pivots = matrix.T.rref()[1]
    assert len(pivots) == 39
    inverse = matrix[list(pivots), :].inv()
    pressure = [p.as_sympy() for p in polynomial_set_1d(3, order - 1)]
    mass_rows = []
    for a in pressure:
        row = []
        for b in pressure:
            row.append(_average(a * b, reference.vertices))
        mass_rows.append(row)
    mass = sympy.Matrix(mass_rows)
    fields = []
    for p in polynomial_set_vector(3, 3, order):
        field = []
        for _ in pieces:
            field.append(sympy.Matrix(p.as_sympy()))
        fields.append(field)
    seeds = []
    for i in range(4):
        face = reference.sub_entity(2, i)
        vertices = reference.sub_entities(2)[i]
        bubble = sympy.prod(lambdas[j] for j in vertices)
        a, b, c = map(sympy.Matrix, face.vertices)
        normal = (b - a).cross(c - a)
        seeds.append([bubble * normal for _ in pieces])
    if order == 2:
        seeds.extend([[sympy.eye(3)[:, i] * h for h in hats] for i in range(3)])
    for seed in seeds:
        divergences = [_div(v) for v in seed]
        moments = []
        for q in pressure:
            moment = sum(_average(div * q, cell) for div, cell in zip(divergences, pieces)) / 4
            moments.append(moment)
        moments = sympy.Matrix(moments)
        projection = sum(a * b for a, b in zip(mass.inv() * moments, pressure))
        rhs = []
        for divergence in divergences:
            rhs.extend(_coefficients(divergence - projection, m))
        rhs = sympy.Matrix(rhs)
        coefficients = inverse * rhs[list(pivots), :]
        assert matrix * coefficients == rhs
        field = []
        for i in range(4):
            values = []
            for c in range(3):
                value = seed[i][c] - sum(
                    a * correction_field[i][c]
                    for a, correction_field in zip(coefficients, correction)
                )
                values.append(sympy.expand(value))
            field.append(sympy.Matrix(values))
        fields.append(field)
    divergence_columns = []
    for field in fields:
        coefficients = _coefficients(_div(field[0]), pressure)
        divergence_columns.append(sympy.Matrix(coefficients))
    divmatrix = sympy.Matrix.hstack(*divergence_columns)
    kernel = divmatrix.nullspace()
    assert len(kernel) == (15 if order == 1 else 33)
    kernel_fields = []
    for coefficients in kernel:
        field = []
        for i in range(4):
            values = []
            for c in range(3):
                value = sum(a * basis_field[i][c] for a, basis_field in zip(coefficients, fields))
                values.append(sympy.expand(value))
            field.append(sympy.Matrix(values))
        kernel_fields.append(field)
    return pieces, kernel_fields


@lru_cache(None)
def _correction_data(reference):
    """Build the global polynomial gradient correction, common to both orders."""
    lambdas = [1 - sum(x), *x]
    bubbles = [
        sympy.prod(lam**a for lam, a in zip(lambdas, alpha))
        for alpha in product(range(6), repeat=4)
        if sum(alpha) == 5 and max(alpha) < 5
    ]
    gradients = [_grad(p) for p in bubbles]
    moments = []
    for d in range(1, 4):
        q = [p.as_sympy() for p in polynomial_set_1d(d, 4 - d)]
        for i in range(reference.sub_entity_count(d)):
            entity = reference.sub_entity(d, i)
            radial = sympy.Matrix(x) - sympy.Matrix(entity.midpoint())
            # Local polynomial coordinates from the inverse affine map.
            axis_vectors = [sympy.Matrix(a) for a in entity.axes]
            axes = sympy.Matrix.hstack(*axis_vectors)
            origin = sympy.Matrix(entity.origin)
            local = (axes.T * axes).inv() * axes.T * (sympy.Matrix(x) - origin)
            for p in q:
                weight = p.subs(dict(zip(x, local)), simultaneous=True)
                moments.append((entity.vertices, radial, weight))
    matrix_rows = []
    for vertices, radial, q in moments:
        row = []
        for gradient in gradients:
            row.append(_average(gradient.dot(radial) * q, vertices))
        matrix_rows.append(row)
    matrix = sympy.Matrix(matrix_rows)
    assert matrix.rows == matrix.cols == 52
    return gradients, moments, matrix.inv()


@lru_cache(None)
def hzz_space(reference, order):
    """Lift divergence-free modified bubbles with the conforming Poincare operator."""
    pieces, fields = _gn_space(reference, order)
    center = sympy.Matrix(reference.midpoint())
    radial = sympy.Matrix(x) - center
    u = sympy.symbols("u")
    lifts = []
    for field in fields:
        lift = []
        for v in field:
            value = v.subs(dict(zip(x, center + u * radial)), simultaneous=True).cross(radial) * u
            lift_values = []
            for p in value:
                terms = sympy.Poly(p.expand(), u).terms()
                lift_values.append(sum(c / (alpha[0] + 1) for alpha, c in terms))
            lift_value = sympy.Matrix(lift_values)
            lift.append(lift_value)
        lifts.append(lift)
    nedelec = NedelecFirstKind(reference, 0)
    nb = [sympy.Matrix(p.as_sympy()) for p in nedelec.get_basis_functions()]
    gradients, moments, inverse = _correction_data(reference)
    corrected = []
    for lift in lifts:
        piece_values = {p: tuple(v) for p, v in zip(pieces, lift)}
        piecewise = PiecewiseFunction(piece_values, 3)
        gamma = sum(
            (nb[i] * d.eval(piecewise).as_sympy() for i, d in enumerate(nedelec.dofs)),
            sympy.zeros(3, 1),
        )
        rhs = []
        for vertices, r, q in moments:
            if len(vertices) == 4:
                pairs = zip(pieces, lift)
                rhs_value = sum(_average((v - gamma).dot(r) * q, p) for p, v in pairs) / 4
            else:
                index = next(i for i, cell in enumerate(pieces) if all(v in cell for v in vertices))
                rhs_value = _average((lift[index] - gamma).dot(r) * q, vertices)
            rhs.append(rhs_value)
        rhs = sympy.Matrix(rhs)
        coefficients = inverse * rhs
        correction = sum((a * g for a, g in zip(coefficients, gradients)), sympy.zeros(3, 1))
        corrected_values = {}
        for p, v in zip(pieces, lift):
            corrected_values[p] = tuple(sympy.expand(c) for c in v - correction)
        corrected_piece = PiecewiseFunction(corrected_values, 3)
        corrected.append(corrected_piece)
    gradients = [_grad(p.as_sympy()) for p in polynomial_set_1d(3, order)[1:]]
    gradient_basis = []
    for gradient in gradients:
        gradient_values = {cell: tuple(gradient) for cell in pieces}
        gradient_basis.append(PiecewiseFunction(gradient_values, 3))
    return gradient_basis + corrected


class CurlPointEvaluation(DotPointEvaluation):
    """A component of curl evaluated at a point."""

    @property
    def nderivs(self) -> int:
        return 1

    def _eval_symbolic(self, function):
        return super()._eval_symbolic(function.curl())

    def get_tex(self):
        desc, definitions = super().get_tex()
        return desc.replace("v", "(\\nabla\\times v)"), definitions

    def perform_mapping(self, fs, map, inverse_map):
        """Require construction on the target cell for curl component data."""
        raise mappings.MappingNotImplemented()

    def discrete(self, poly_degree):
        """Discrete curl data are not implemented."""
        raise NotImplementedError()

    name = "Point evaluation of a curl component"


class HuZhangZhang(CiarletElement):
    """The r=k Hu-Zhang-Zhang H(grad curl) tetrahedron."""

    def __init__(self, reference: Reference, order: int):
        """Create the element, indexed by complete polynomial subdegree 0 or 1."""
        assert reference.name == "tetrahedron" and order in (0, 1)
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()
        k = order + 1
        mapping = "covariant"
        dofs: ListOfFunctionals = []
        for i, v in enumerate(reference.vertices):
            for direction in sympy.eye(3).columnspace():
                dof = CurlPointEvaluation(reference, v, tuple(direction), (0, i), mapping=mapping)
                dofs.append(dof)
        for i in range(6):
            edge = reference.sub_entity(1, i)
            moment = Lagrange(edge.default_reference(), k - 1)
            for p, d in zip(moment.get_basis_functions(), moment.dofs):
                dofs.append(TangentIntegralMoment(reference, p, d, (1, i), mapping=mapping))
            if k == 2:
                for direction in sympy.eye(3).columnspace():
                    dof = CurlPointEvaluation(
                        reference, edge.midpoint(), tuple(direction), (1, i), mapping=mapping
                    )
                    dofs.append(dof)
        basis = hzz_space(reference, k)
        super().__init__(reference, order, basis, dofs, 3, 3)

    def get_dual_matrix(self, inverse=False, caching=True):
        """Return the dual matrix or its exact inverse."""
        matrix = super().get_dual_matrix(False, caching)
        if inverse:
            return matrix.to_DM(extension=True).inv().to_Matrix()
        return matrix

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> int:
        return 4

    @property
    def lagrange_subdegree(self) -> int:
        return self.order

    @property
    def lagrange_superdegree(self) -> None:
        return None

    names = ["Hu-Zhang-Zhang"]
    references = ["tetrahedron"]
    min_order = 0
    max_order = 1
    continuity = "H(curl)"
    value_type = "vector macro"
    last_updated = "2026.11"
