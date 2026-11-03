"""Walkington's C1 tetrahedron (https://doi.org/10.1137/130912013).

The centroid split carries piecewise quintics, C1 across internal faces and
C4 at the centroid, whose exterior normal derivatives are cubic (Notation 2.1).
"""

from functools import lru_cache
from itertools import combinations

import sympy
from sympy.polys.matrices import DomainMatrix

from symfem.finite_element import CiarletElement
from symfem.functionals import (
    DerivativePointEvaluation,
    ListOfFunctionals,
    PointDirectionalDerivativeEvaluation,
    PointEvaluation,
)
from symfem.piecewise_functions import PiecewiseFunction
from symfem.polynomials import polynomial_set_1d
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import x

__all__ = ["Walkington"]


@lru_cache(None)
def _homogeneous_quintic_splines(vertices):
    """Construct the ten homogeneous quintic splines by exact linear algebra.

    The splines are C1 across internal faces, C4 at the centroid, and have
    cubic exterior normal derivatives.
    """
    mid = tuple(sum(v[i] for v in vertices) / sympy.Integer(4) for i in range(3))
    s, t = sympy.symbols("s t")
    monomials = [
        x[0] ** a * x[1] ** b * x[2] ** (5 - a - b) for a in range(6) for b in range(6 - a)
    ]
    derivatives = [[p] + [sympy.diff(p, xi) for xi in x] for p in monomials]
    rows = []

    def add_constraints(expressions, blocks):
        polys = [sympy.Poly(p, s, t) for p in expressions]
        powers = sorted(set().union(*(p.monoms() for p in polys)))
        for power in powers:
            row = [sympy.S.Zero] * 84
            for block, sign in blocks:
                for j, p in enumerate(polys):
                    row[21 * block + j] = sign * p.coeff_monomial(power)
            if any(row):
                rows.append(row)

    # Piece i is the cone over the exterior face opposite vertex i.
    for i, j in combinations(range(4), 2):
        shared = [v for k, v in enumerate(vertices) if k not in (i, j)]
        point = {x[k]: s * (shared[0][k] - mid[k]) + t * (shared[1][k] - mid[k]) for k in range(3)}
        for derivative in range(4):
            expressions = [p[derivative].subs(point, simultaneous=True) for p in derivatives]
            blocks = [(i, 1), (j, -1)]
            add_constraints(expressions, blocks)

    for i in range(4):
        face = [sympy.Matrix(v) for j, v in enumerate(vertices) if j != i]
        a, b = face[1] - face[0], face[2] - face[0]
        normal = a.cross(b)
        point = {x[k]: s * a[k] + t * b[k] for k in range(3)}
        expressions = [
            sum(normal[k] * p[k + 1] for k in range(3)).subs(point, simultaneous=True)
            for p in derivatives
        ]
        add_constraints(expressions, [(i, 1)])

    constraint_matrix = DomainMatrix.from_Matrix(sympy.Matrix(rows))
    kernel = constraint_matrix.to_field().nullspace().to_Matrix()
    assert kernel.rows == 10
    shifted = [
        p.subs({x[k]: x[k] - mid[k] for k in range(3)}, simultaneous=True) for p in monomials
    ]
    pieces = [tuple(v for j, v in enumerate(vertices) if i != j) + (mid,) for i in range(4)]
    basis = []
    for row in kernel.tolist():
        piece_values = {
            pieces[i]: sum(row[21 * i + j] * shifted[j] for j in range(21)) for i in range(4)
        }
        basis.append(PiecewiseFunction(piece_values, 3))
    return basis


class Walkington(CiarletElement):
    """The 45-dimensional Walkington element on the centroid split."""

    def __init__(self, reference: Reference, order: int):
        """Create the quintic element on the default tetrahedron."""
        assert reference.name == "tetrahedron"
        assert order == 4
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()
        dofs: ListOfFunctionals = []
        for i, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, i)))
            for degree in (1, 2):
                for a in range(degree + 1):
                    for b in range(degree - a + 1):
                        dof = DerivativePointEvaluation(
                            reference, v, (a, b, degree - a - b), entity=(0, i)
                        )
                        dofs.append(dof)
        for i in range(reference.sub_entity_count(2)):
            face = reference.sub_entity(2, i)
            dof = PointDirectionalDerivativeEvaluation(
                reference, face.midpoint(), face.normal(), entity=(2, i)
            )
            dofs.append(dof)
        dofs.append(PointEvaluation(reference, reference.midpoint(), entity=(3, 0)))
        poly = list(polynomial_set_1d(3, 4)) + _homogeneous_quintic_splines(reference.vertices)
        super().__init__(reference, order, poly, dofs, 3, 1)

    @property
    def lagrange_subdegree(self) -> int:
        return 4

    @property
    def lagrange_superdegree(self) -> int | None:
        return None

    @property
    def polynomial_subdegree(self) -> int:
        return 4

    @property
    def polynomial_superdegree(self) -> int | None:
        return 5

    names = ["Walkington"]
    references = ["tetrahedron"]
    min_order = 4
    max_order = 4
    continuity = "C1"
    value_type = "scalar macro"
    last_updated = "2026.11"
