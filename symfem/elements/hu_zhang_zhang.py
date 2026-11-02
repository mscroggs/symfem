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
            correction.append(
                [sympy.Matrix([sympy.expand(h ** (3 - j) * c) for c in p.as_sympy()]) for h in hats]
            )
    matrix = sympy.Matrix.hstack(
        *[
            sympy.Matrix([c for v in field for c in _coefficients(_div(v), m)])
            for field in correction
        ]
    )
    pivots = matrix.T.rref()[1]
    assert len(pivots) == 39
    inverse = matrix[list(pivots), :].inv()
    pressure = [p.as_sympy() for p in polynomial_set_1d(3, order - 1)]
    mass = sympy.Matrix([[_average(a * b, reference.vertices) for b in pressure] for a in pressure])
    fields = [
        [sympy.Matrix(p.as_sympy()) for _ in pieces] for p in polynomial_set_vector(3, 3, order)
    ]
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
        moments = sympy.Matrix(
            [
                sum(_average(div * q, cell) for div, cell in zip(divergences, pieces)) / 4
                for q in pressure
            ]
        )
        projection = sum(a * b for a, b in zip(mass.inv() * moments, pressure))
        rhs = sympy.Matrix([c for div in divergences for c in _coefficients(div - projection, m)])
        coefficients = inverse * rhs[list(pivots), :]
        assert matrix * coefficients == rhs
        fields.append(
            [
                sympy.Matrix(
                    [
                        sympy.expand(
                            seed[i][c]
                            - sum(a * field[i][c] for a, field in zip(coefficients, correction))
                        )
                        for c in range(3)
                    ]
                )
                for i in range(4)
            ]
        )
    divmatrix = sympy.Matrix.hstack(
        *[sympy.Matrix(_coefficients(_div(field[0]), pressure)) for field in fields]
    )
    kernel = divmatrix.nullspace()
    assert len(kernel) == (15 if order == 1 else 33)
    return pieces, [
        [
            sympy.Matrix(
                [
                    sympy.expand(sum(a * field[i][c] for a, field in zip(v, fields)))
                    for c in range(3)
                ]
            )
            for i in range(4)
        ]
        for v in kernel
    ]


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
            axes = sympy.Matrix.hstack(*[sympy.Matrix(a) for a in entity.axes])
            local = (axes.T * axes).inv() * axes.T * (sympy.Matrix(x) - sympy.Matrix(entity.origin))
            for p in q:
                weight = p.subs(dict(zip(x, local)), simultaneous=True)
                moments.append((entity.vertices, radial, weight))
    matrix = sympy.Matrix(
        [
            [_average(g.dot(radial) * q, vertices) for g in gradients]
            for vertices, radial, q in moments
        ]
    )
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
            lift.append(
                sympy.Matrix(
                    [
                        sum(c / (alpha[0] + 1) for alpha, c in sympy.Poly(p.expand(), u).terms())
                        for p in value
                    ]
                )
            )
        lifts.append(lift)
    nedelec = NedelecFirstKind(reference, 0)
    nb = [sympy.Matrix(p.as_sympy()) for p in nedelec.get_basis_functions()]
    gradients, moments, inverse = _correction_data(reference)
    corrected = []
    for lift in lifts:
        piecewise = PiecewiseFunction({p: tuple(v) for p, v in zip(pieces, lift)}, 3)
        gamma = sum(
            (nb[i] * d.eval(piecewise).as_sympy() for i, d in enumerate(nedelec.dofs)),
            sympy.zeros(3, 1),
        )
        rhs = []
        for vertices, r, q in moments:
            if len(vertices) == 4:
                rhs.append(
                    sum(_average((v - gamma).dot(r) * q, p) for p, v in zip(pieces, lift)) / 4
                )
            else:
                index = next(i for i, cell in enumerate(pieces) if all(v in cell for v in vertices))
                rhs.append(_average((lift[index] - gamma).dot(r) * q, vertices))
        coefficients = inverse * sympy.Matrix(rhs)
        correction = sum((a * g for a, g in zip(coefficients, gradients)), sympy.zeros(3, 1))
        corrected.append(
            PiecewiseFunction(
                {p: tuple(sympy.expand(c) for c in v - correction) for p, v in zip(pieces, lift)}, 3
            )
        )
    gradients = [_grad(p.as_sympy()) for p in polynomial_set_1d(3, order)[1:]]
    return [
        PiecewiseFunction({cell: tuple(g) for cell in pieces}, 3) for g in gradients
    ] + corrected


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
        dofs: ListOfFunctionals = []
        for i, v in enumerate(reference.vertices):
            for direction in sympy.eye(3).columnspace():
                dofs.append(
                    CurlPointEvaluation(reference, v, tuple(direction), (0, i), mapping="covariant")
                )
        for i in range(6):
            edge = reference.sub_entity(1, i)
            moment = Lagrange(edge.default_reference(), k - 1)
            for p, d in zip(moment.get_basis_functions(), moment.dofs):
                dofs.append(TangentIntegralMoment(reference, p, d, (1, i), mapping="covariant"))
            if k == 2:
                for direction in sympy.eye(3).columnspace():
                    dofs.append(
                        CurlPointEvaluation(
                            reference,
                            edge.midpoint(),
                            tuple(direction),
                            (1, i),
                            mapping="covariant",
                        )
                    )
        super().__init__(reference, order, hzz_space(reference, k), dofs, 3, 3)

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
