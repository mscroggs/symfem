"""Trimmed serendipity elements on tensor products.

These elements' definitions appear in https://doi.org/10.1137/16M1073352
(Cockburn, Fu, 2017) and https://doi.org/10.1090/mcom/3354
(Gilette, Kloefkorn, 2018)
"""

import typing

from symfem.elements.dpc import DPC, VectorDPC
from symfem.finite_element import CiarletElement
from symfem.functionals import (
    IntegralAgainst,
    IntegralMoment,
    ListOfFunctionals,
    NormalIntegralMoment,
    TangentIntegralMoment,
)
from symfem.functions import FunctionInput, ScalarFunction, VectorFunction
from symfem.moments import make_integral_moment_dofs
from symfem.polynomials import polynomial_set_vector
from symfem.references import NonDefaultReferenceError, Reference
from symfem.symbols import t, x

__all__ = ["TrimmedSerendipityHcurl", "TrimmedSerendipityHdiv"]


class TrimmedSerendipityHcurl(CiarletElement):
    """Trimmed serendipity Hcurl finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_vector(reference.tdim, reference.tdim, order)
        if reference.tdim == 2:
            poly += [
                (x[0] ** j * x[1] ** (order + 1 - j), -(x[0] ** (j + 1)) * x[1] ** (order - j))
                for j in range(order + 1)
            ]
            if order == 0:
                poly += [(x[1], x[0])]
            else:
                poly += [
                    (x[1] ** (order + 1), (order + 1) * x[0] * x[1] ** order),
                    ((order + 1) * x[0] ** order * x[1], x[0] ** (order + 1)),
                ]
        else:
            for i in range(order + 1):
                for j in range(order + 1 - i):
                    for dim in range(3):
                        if i == 0 or dim != 0:
                            p = VectorFunction(tuple(x)).cross(
                                VectorFunction(
                                    [
                                        x[0] ** i * x[1] ** j * x[2] ** (order - i - j)
                                        if d == dim
                                        else 0
                                        for d in range(3)
                                    ]
                                )
                            )
                            poly.append(p)

            if order == 0:
                poly += [ScalarFunction(x[0] * x[1] * x[2] ** (order + 1)).grad(3)]
            else:
                poly += [
                    ScalarFunction(x[0] * x[1] * x[2] ** (order + 1)).grad(3),
                    ScalarFunction(x[0] * x[1] ** (order + 1) * x[2]).grad(3),
                    ScalarFunction(x[0] ** (order + 1) * x[1] * x[2]).grad(3),
                ]
            for i in range(order + 2):
                poly.append(ScalarFunction(x[0] * x[1] ** i * x[2] ** (order + 1 - i)).grad(3))
                if i != 1:
                    poly.append(ScalarFunction(x[1] * x[0] ** i * x[2] ** (order + 1 - i)).grad(3))
                    if order != i:
                        poly.append(
                            ScalarFunction(x[2] * x[0] ** i * x[1] ** (order + 1 - i)).grad(3)
                        )
            for i in range(order + 1):
                p = x[0] * x[1] ** i * x[2] ** (order - i)
                poly.append((0, -x[2] * p, x[1] * p))
                p = x[1] * x[0] ** i * x[2] ** (order - i)
                poly.append((-x[2] * p, 0, x[0] * p))
                if order > 0:
                    p = x[2] * x[0] ** i * x[1] ** (order - i)
                    poly.append((-x[1] * p, x[0] * p, 0))

        dofs: ListOfFunctionals = []
        dofs += make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, DPC, order, {"variant": variant}),
            faces=(IntegralMoment, VectorDPC, order - 2, "contravariant", {"variant": variant}),
            volumes=(IntegralMoment, VectorDPC, order - 4, "contravariant", {"variant": variant}),
        )
        if order >= 1:
            for f_n in range(reference.sub_entity_count(2)):
                for i in range(order + 1):
                    f = ScalarFunction(x[0] ** (order - i) * x[1] ** i).grad(2)
                    f2 = VectorFunction((f[1], -f[0])).subs(x, tuple(t))
                    dofs.append(
                        IntegralAgainst(reference, f2, entity=(2, f_n), mapping="contravariant")
                    )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        return (self.order + self.reference.tdim) // (self.reference.tdim + 1)

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order + 1

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order + self.reference.tdim - 1

    names = ["trimmed serendipity Hcurl", "TScurl"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "H(curl)"
    value_type = "vector"
    last_updated = "2025.03"


class TrimmedSerendipityHdiv(CiarletElement):
    """Trimmed serendipity Hdiv finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        if reference.vertices != reference.reference_vertices:
            raise NonDefaultReferenceError()

        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_vector(reference.tdim, reference.tdim, order)
        if reference.tdim == 2:
            poly += [
                (x[0] ** (j + 1) * x[1] ** (order - j), x[0] ** j * x[1] ** (order + 1 - j))
                for j in range(order + 1)
            ]
            if order == 0:
                poly += [(x[0], -x[1])]
            else:
                poly += [
                    ((order + 1) * x[0] * x[1] ** order, -(x[1] ** (order + 1))),
                    (-(x[0] ** (order + 1)), (order + 1) * x[0] ** order * x[1]),
                ]
        else:
            for i in range(order + 1):
                for j in range(order + 1 - i):
                    p = x[0] ** i * x[1] ** j * x[2] ** (order - i - j)
                    poly.append((x[0] * p, x[1] * p, x[2] * p))
            for i in range(order + 1):
                p = x[0] * x[1] ** i * x[2] ** (order - i)
                poly.append(VectorFunction((0, -x[2] * p, x[1] * p)).curl())
                p = x[1] * x[0] ** i * x[2] ** (order - i)
                poly.append(VectorFunction((-x[2] * p, 0, x[0] * p)).curl())
                if order > 0:
                    p = x[2] * x[0] ** i * x[1] ** (order - i)
                    poly.append(VectorFunction((-x[1] * p, x[0] * p, 0)).curl())

        dofs: ListOfFunctionals = []
        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DPC, order, {"variant": variant}),
            cells=(IntegralMoment, VectorDPC, order - 2, "covariant", {"variant": variant}),
        )
        if order >= 1:
            if reference.tdim == 2:
                fs = [
                    ScalarFunction(x[0] ** (order - i) * x[1] ** i).grad(2)
                    for i in range(order + 1)
                ]
            else:
                fs = [
                    ScalarFunction(x[0] ** (order - i - j) * x[1] ** i * x[2] ** j).grad(3)
                    for i in range(order + 1)
                    for j in range(order + 1 - i)
                ]
            for f in fs:
                f2 = f.subs(x, tuple(t))
                dofs.append(
                    IntegralAgainst(reference, f2, entity=(reference.tdim, 0), mapping="covariant")
                )

        super().__init__(reference, order, poly, dofs, reference.tdim, reference.tdim)
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    @property
    def lagrange_subdegree(self) -> int:
        return (self.order + 2) // (self.reference.tdim + 1)

    @property
    def lagrange_superdegree(self) -> typing.Optional[int]:
        return self.order + 1

    @property
    def polynomial_subdegree(self) -> int:
        return self.order

    @property
    def polynomial_superdegree(self) -> typing.Optional[int]:
        return self.order + 1

    names = ["trimmed serendipity Hdiv", "TSdiv"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "H(div)"
    value_type = "vector"
    last_updated = "2025.03"
