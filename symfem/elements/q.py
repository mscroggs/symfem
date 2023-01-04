"""Q elements on tensor product cells."""

import typing
from itertools import product

import sympy

from ..finite_element import CiarletElement, FiniteElement
from ..functionals import (DotPointEvaluation, IntegralMoment, ListOfFunctionals,
                           NormalIntegralMoment, PointEvaluation, TangentIntegralMoment)
from ..functions import FunctionInput
from ..moments import make_integral_moment_dofs
from ..polynomials import (Hcurl_quolynomials, Hdiv_quolynomials, quolynomial_set_1d,
                           quolynomial_set_vector)
from ..quadrature import get_quadrature
from ..references import Reference


class Q(CiarletElement):
    """A Q element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        dofs: ListOfFunctionals = []
        if order == 0:
            dofs = [PointEvaluation(
                reference, tuple(sympy.Rational(1, 2) for i in range(reference.tdim)),
                entity=(reference.tdim, 0))]
        else:
            points, _ = get_quadrature(variant, order + 1)

            for v_n, v in enumerate(reference.vertices):
                dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))
            for edim in range(1, 4):
                for e_n in range(reference.sub_entity_count(edim)):
                    entity = reference.sub_entity(edim, e_n)
                    for i in product(range(1, order), repeat=edim):
                        dofs.append(
                            PointEvaluation(
                                reference, tuple(o + sum(a[j] * points[b]
                                                         for a, b in zip(entity.axes, i[::-1]))
                                                 for j, o in enumerate(entity.origin)),
                                entity=(edim, e_n)))

        poly: typing.List[FunctionInput] = []
        poly += quolynomial_set_1d(reference.tdim, order)

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)
        self.variant = variant

    def get_tensor_factorisation(
        self
    ) -> typing.List[typing.Tuple[str, typing.List[FiniteElement], typing.List[int]]]:
        """Get the representation of the element as a tensor product.

        Returns:
            The tensor factorisation
        """
        from symfem import create_element
        interval_q = create_element("interval", "Lagrange", self.order)

        if self.order == 0:
            perm = [0]
        elif self.reference.name == "quadrilateral":
            n = self.order - 1
            perm = [0, 2] + [4 + n + i for i in range(n)]
            perm += [1, 3] + [4 + 2 * n + i for i in range(n)]
            for i in range(n):
                perm += [4 + i, 4 + 3 * n + i] + [4 + i + (4 + j) * n for j in range(n)]
        elif self.reference.name == "hexahedron":
            n = self.order - 1
            perm = [0, 4] + [8 + 2 * n + i for i in range(n)]
            perm += [2, 6] + [8 + 6 * n + i for i in range(n)]
            for i in range(n):
                perm += [8 + n + i, 8 + 9 * n + i]
                perm += [8 + 12 * n + 2 * n ** 2 + i + n * j for j in range(n)]
            perm += [1, 5] + [8 + 4 * n + i for i in range(n)]
            perm += [3, 7] + [8 + 7 * n + i for i in range(n)]
            for i in range(n):
                perm += [8 + 3 * n + i, 8 + 10 * n + i]
                perm += [8 + 12 * n + 3 * n ** 2 + i + n * j for j in range(n)]
            for i in range(n):
                perm += [8 + i, 8 + 8 * n + i]
                perm += [8 + 12 * n + n ** 2 + i + n * j for j in range(n)]
                perm += [8 + 5 * n + i, 8 + 11 * n + i]
                perm += [8 + 12 * n + 4 * n ** 2 + i + n * j for j in range(n)]
                for j in range(n):
                    perm += [8 + 12 * n + i + n * j, 8 + 12 * n + 5 * n ** 2 + i + n * j]
                    perm += [8 + 12 * n + 6 * n ** 2 + i + n * j + n ** 2 * k for k in range(n)]

        return [("scalar", [interval_q for i in range(self.reference.tdim)], perm)]

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["Q", "Lagrange", "P"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "C0"


class VectorQ(CiarletElement):
    """A vector Q element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        scalar_space = Q(reference, order, variant)
        dofs: ListOfFunctionals = []
        poly: typing.List[FunctionInput] = []
        if reference.tdim == 1:
            for p in scalar_space.dofs:
                dofs.append(PointEvaluation(reference, p.dof_point(), entity=p.entity))

            poly += quolynomial_set_1d(reference.tdim, order)
        else:
            directions = [
                tuple(1 if i == j else 0 for j in range(reference.tdim))
                for i in range(reference.tdim)
            ]
            for p in scalar_space.dofs:
                for d in directions:
                    dofs.append(DotPointEvaluation(reference, p.dof_point(), d, entity=p.entity))

            poly += quolynomial_set_vector(reference.tdim, reference.tdim, order)

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["vector Q", "vQ"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "C0"


class Nedelec(CiarletElement):
    """Nedelec Hcurl finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        poly: typing.List[FunctionInput] = []
        poly += quolynomial_set_vector(reference.tdim, reference.tdim, order - 1)
        poly += Hcurl_quolynomials(reference.tdim, reference.tdim, order)

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, Q, order - 1, {"variant": variant}),
            faces=(IntegralMoment, RaviartThomas, order - 1, "covariant", {"variant": variant}),
            volumes=(IntegralMoment, RaviartThomas, order - 1, "covariant", {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["NCE", "RTCE", "Qcurl", "Nedelec", "Ncurl"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(curl)"


class RaviartThomas(CiarletElement):
    """Raviart-Thomas Hdiv finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        poly: typing.List[FunctionInput] = []
        poly += quolynomial_set_vector(reference.tdim, reference.tdim, order - 1)
        poly += Hdiv_quolynomials(reference.tdim, reference.tdim, order)

        dofs: ListOfFunctionals = make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, Q, order - 1, {"variant": variant}),
            cells=(IntegralMoment, Nedelec, order - 1, "contravariant", {"variant": variant}),
        )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["NCF", "RTCF", "Qdiv"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(div)"
