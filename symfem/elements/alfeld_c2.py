"""C2 splines on the double Alfeld split (Lai and Schumaker, Section 7.5)."""

from functools import lru_cache
from itertools import combinations

import sympy
from sympy.polys.matrices import DomainMatrix

from symfem.elements.bramble_zlamal_c2 import c2_dofs
from symfem.elements.wu_xu import derivatives
from symfem.finite_element import CiarletElement
from symfem.functions import FunctionInput
from symfem.piecewise_functions import PiecewiseFunction
from symfem.polynomials import polynomial_set_1d
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import x

__all__ = ["AlfeldC2"]


@lru_cache(None)
def alfeld_c2_space(vertices: tuple, order: int) -> list[FunctionInput]:
    """Construct the supersmooth C2 spline space with exact constraints."""

    def midpoint(points):
        return tuple(sum(v[i] for v in points) / sympy.Integer(len(points)) for i in range(2))

    major = midpoint(vertices)
    pieces: list[tuple] = []
    minors = []
    for i, j in combinations(range(3), 2):
        parent = (vertices[i], vertices[j], major)
        minor = midpoint(parent)
        minors.append(minor)
        pieces.extend((parent[a], parent[b], minor) for a, b in combinations(range(3), 2))
    poly = [p.as_sympy() for p in polynomial_set_1d(2, order)]
    n = len(poly)
    rows = []
    s = sympy.Symbol("s")
    derivatives_at = {}
    for d in range(order):
        for alpha in derivatives(2, d):
            derivatives_at[alpha] = [sympy.diff(p, x[0], alpha[0], x[1], alpha[1]) for p in poly]

    def constrain(i, j, point, degree, edge):
        for d in range(degree + 1):
            for alpha in derivatives(2, d):
                values = [
                    p.subs(dict(zip(x, point)), simultaneous=True).expand()
                    for p in derivatives_at[alpha]
                ]
                if edge:
                    values = [sympy.Poly(p, s) for p in values]
                    coefficient_rows = [[p.nth(k) for p in values] for k in range(order - d + 1)]
                else:
                    coefficient_rows = [values]
                for coefficients in coefficient_rows:
                    row = [sympy.S.Zero] * (9 * n)
                    for k, v in enumerate(coefficients):
                        row[i * n + k] = v
                        row[j * n + k] = -v
                    if any(row):
                        rows.append(row)

    for i, j in combinations(range(9), 2):
        shared = tuple(v for v in pieces[i] if v in pieces[j])
        if len(shared) == 2:
            point = tuple(a + s * (b - a) for a, b in zip(*shared))
            outer_edge = major in shared and any(v in vertices for v in shared)
            degree = order - 2 if outer_edge else 2
            constrain(i, j, point, degree, True)
    for point, degree in [(major, order - 2)] + [(p, order - 1) for p in minors]:
        indices = [i for i, piece in enumerate(pieces) if point in piece]
        for j in indices[1:]:
            constrain(indices[0], j, point, degree, False)
    kernel = (
        DomainMatrix.from_Matrix(sympy.Matrix(rows))
        .to_field()
        .nullspace(divide_last=True)
        .to_Matrix()
    )
    expected = 18 + 9 * (order - 4) + (order - 5) * (order - 4) // 2
    assert kernel.rows == expected
    return [
        PiecewiseFunction(
            {
                piece: sum(row[i * n + j] * p for j, p in enumerate(poly))
                for i, piece in enumerate(pieces)
            },
            2,
        )
        for row in kernel.tolist()
    ]


class AlfeldC2(CiarletElement):
    """The C2 triangle on a double centroid split."""

    def __init__(self, reference: Reference, order: int):
        """Create a degree-five or higher Alfeld C2 element."""
        assert reference.name == "triangle" and order >= 5
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()
        super().__init__(
            reference,
            order,
            alfeld_c2_space(reference.vertices, order),
            c2_dofs(reference, order, 2),
            2,
            1,
        )

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
        return self.order

    @property
    def lagrange_subdegree(self) -> int:
        return self.order

    @property
    def lagrange_superdegree(self) -> None:
        return None

    names = ["Alfeld C2"]
    references = ["triangle"]
    min_order = 5
    continuity = "C2"
    value_type = "scalar macro"
    last_updated = "2026.11"
