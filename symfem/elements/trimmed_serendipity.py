"""Trimmed serendipity elements on tensor products.

These elements' definitions appear in https://doi.org/10.1137/16M1073352
(Cockburn, Fu, 2017) and https://doi.org/10.1090/mcom/3354
(Gilette, Kloefkorn, 2018)
"""

import typing

from ..finite_element import CiarletElement
from ..functionals import (IntegralAgainst, IntegralMoment, ListOfFunctionals, NormalIntegralMoment,
                           TangentIntegralMoment)
from ..functions import FunctionInput, ScalarFunction, VectorFunction
from ..moments import make_integral_moment_dofs
from ..polynomials import polynomial_set_vector
from ..references import Reference
from ..symbols import t, x
from .dpc import DPC, VectorDPC


class TrimmedSerendipityHcurl(CiarletElement):
    """Trimmed serendipity Hcurl finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_vector(reference.tdim, reference.tdim, order - 1)
        if reference.tdim == 2:
            poly += [
                (x[0] ** j * x[1] ** (order - j), -x[0] ** (j + 1) * x[1] ** (order - 1 - j))
                for j in range(order)]
            if order == 1:
                poly += [(x[1], x[0])]
            else:
                poly += [(x[1] ** order, order * x[0] * x[1] ** (order - 1)),
                         (order * x[0] ** (order - 1) * x[1], x[0] ** order)]
        else:
            for i in range(order):
                for j in range(order - i):
                    for dim in range(3):
                        if i == 0 or dim != 0:
                            p = VectorFunction(tuple(x)).cross(VectorFunction([
                                x[0] ** i * x[1] ** j * x[2] ** (order - 1 - i - j)
                                if d == dim else 0 for d in range(3)]))
                            poly.append(p)

            if order == 1:
                poly += [ScalarFunction(x[0] * x[1] * x[2] ** order).grad(3)]
            else:
                poly += [ScalarFunction(x[0] * x[1] * x[2] ** order).grad(3),
                         ScalarFunction(x[0] * x[1] ** order * x[2]).grad(3),
                         ScalarFunction(x[0] ** order * x[1] * x[2]).grad(3)]
            for i in range(order + 1):
                poly.append(ScalarFunction(x[0] * x[1] ** i * x[2] ** (order - i)).grad(3))
                if i != 1:
                    poly.append(
                        ScalarFunction(x[1] * x[0] ** i * x[2] ** (order - i)).grad(3))
                    if order - i != 1:
                        poly.append(
                            ScalarFunction(x[2] * x[0] ** i * x[1] ** (order - i)).grad(3))
            for i in range(order):
                p = x[0] * x[1] ** i * x[2] ** (order - 1 - i)
                poly.append((0, -x[2] * p, x[1] * p))
                p = x[1] * x[0] ** i * x[2] ** (order - 1 - i)
                poly.append((-x[2] * p, 0, x[0] * p))
                if order > 1:
                    p = x[2] * x[0] ** i * x[1] ** (order - 1 - i)
                    poly.append((-x[1] * p, x[0] * p, 0))

        dofs: ListOfFunctionals = []
        dofs += make_integral_moment_dofs(
            reference,
            edges=(TangentIntegralMoment, DPC, order - 1, {"variant": variant}),
            faces=(IntegralMoment, VectorDPC, order - 3, "contravariant",
                   {"variant": variant}),
            volumes=(IntegralMoment, VectorDPC, order - 5, "contravariant",
                     {"variant": variant}),
        )
        if order >= 2:
            for f_n in range(reference.sub_entity_count(2)):
                face = reference.sub_entity(2, f_n)
                for i in range(order):
                    f = ScalarFunction(x[0] ** (order - 1 - i) * x[1] ** i).grad(2)
                    f2 = VectorFunction((f[1], -f[0])).subs(x, tuple(t))
                    dofs.append(IntegralAgainst(
                        reference, face, f2, entity=(2, f_n), mapping="contravariant"))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, reference.tdim
        )
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["trimmed serendipity Hcurl", "TScurl"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(curl)"


class TrimmedSerendipityHdiv(CiarletElement):
    """Trimmed serendipity Hdiv finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        """Create the element.

        Args:
            reference: The reference element
            order: The polynomial order
            variant: The variant of the element
        """
        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_vector(reference.tdim, reference.tdim, order - 1)
        if reference.tdim == 2:
            poly += [
                (x[0] ** (j + 1) * x[1] ** (order - 1 - j), x[0] ** j * x[1] ** (order - j))
                for j in range(order)]
            if order == 1:
                poly += [(x[0], -x[1])]
            else:
                poly += [(order * x[0] * x[1] ** (order - 1), -x[1] ** order),
                         (-x[0] ** order, order * x[0] ** (order - 1) * x[1])]
        else:
            for i in range(order):
                for j in range(order - i):
                    p = x[0] ** i * x[1] ** j * x[2] ** (order - 1 - i - j)
                    poly.append((x[0] * p, x[1] * p, x[2] * p))
            for i in range(order):
                p = x[0] * x[1] ** i * x[2] ** (order - 1 - i)
                poly.append(
                    VectorFunction((0, -x[2] * p, x[1] * p)).curl())
                p = x[1] * x[0] ** i * x[2] ** (order - 1 - i)
                poly.append(
                    VectorFunction((-x[2] * p, 0, x[0] * p)).curl())
                if order > 1:
                    p = x[2] * x[0] ** i * x[1] ** (order - 1 - i)
                    poly.append(
                        VectorFunction((-x[1] * p, x[0] * p, 0)).curl())

        dofs: ListOfFunctionals = []
        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DPC, order - 1, {"variant": variant}),
            cells=(IntegralMoment, VectorDPC, order - 3, "covariant",
                   {"variant": variant}),
        )
        if order >= 2:
            if reference.tdim == 2:
                fs = [ScalarFunction(x[0] ** (order - 1 - i) * x[1] ** i).grad(2)
                      for i in range(order)]
            else:
                fs = [ScalarFunction(x[0] ** (order - 1 - i - j) * x[1] ** i * x[2] ** j).grad(3)
                      for i in range(order) for j in range(order - i)]
            for f in fs:
                f2 = f.subs(x, tuple(t))
                dofs.append(IntegralAgainst(reference, reference, f2, entity=(reference.tdim, 0),
                                            mapping="covariant"))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, reference.tdim
        )
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element.

        Returns:
            Keyword argument dictionary
        """
        return {"variant": self.variant}

    names = ["trimmed serendipity Hdiv", "TSdiv"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(div)"
