"""Fortin-Soulie elements on a triangle.

This element's definition appears in https://doi.org/10.1002/nme.1620190405
(Fortin, Soulie, 1983)
"""

import sympy
from sympy.polys.matrices import DomainMatrix

from symfem.finite_element import CiarletElement
from symfem.functionals import ListOfFunctionals, PointEvaluation
from symfem.functions import FunctionInput
from symfem.polynomials import polynomial_set_1d
from symfem.references import NonDefaultReferenceError, Reference

__all__ = ["FortinSoulie"]


class FortinSoulie(CiarletElement):
    """Fortin-Soulie finite element."""

    def __init__(self, reference: Reference, order: int):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
        """
        assert reference.name == "triangle"
        assert order == 2
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        # The element is continuous at the two Gauss-Legendre points of each edge. These six
        # points lie on a conic, so they cannot all be used as DOFs: the midpoint is used
        # instead of the two Gauss-Legendre points on edge 0
        g0 = sympy.Rational(1, 2) - sympy.sqrt(3) / 6
        g1 = sympy.Rational(1, 2) + sympy.sqrt(3) / 6
        dofs: ListOfFunctionals = [
            PointEvaluation(reference, (sympy.Rational(1, 2), 0), entity=(1, 0)),
            PointEvaluation(reference, (0, g0), entity=(1, 1)),
            PointEvaluation(reference, (0, g1), entity=(1, 1)),
            PointEvaluation(reference, (g1, g0), entity=(1, 2)),
            PointEvaluation(reference, (g0, g1), entity=(1, 2)),
            PointEvaluation(reference, (sympy.Rational(1, 3), sympy.Rational(1, 3)), entity=(2, 0)),
        ]

        poly: list[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, order)
        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    def get_dual_matrix(
        self, inverse=False, caching=True
    ) -> sympy.matrices.dense.MutableDenseMatrix:
        """Get the dual matrix.

        The DOF points involve sqrt(3), and inverting the dual matrix with sympy's LU
        decomposition gives very large unsimplified expressions. The inverse is therefore
        computed exactly over the field QQ<sqrt(3)>.

        Args:
            inverse: Should the dual matrix be inverted?
            caching: Should the result be cached

        Returns:
            The dual matrix
        """
        if inverse and not caching:
            mat = super().get_dual_matrix(inverse=False, caching=False)
            return DomainMatrix.from_Matrix(mat, extension=True).inv().to_Matrix()
        return super().get_dual_matrix(inverse, caching)

    @property
    def lagrange_subdegree(self) -> int:
        return self.order

    @property
    def lagrange_superdegree(self) -> int | None:
        return self.order

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> int | None:
        return self.order

    names = ["Fortin-Soulie", "FS"]
    references = ["triangle"]
    min_order = 2
    max_order = 2
    continuity = "L2"
    value_type = "scalar"
    last_updated = "2026.09.1"
