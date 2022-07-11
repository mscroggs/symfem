"""Trimmed serendipity elements on tensor products.

These elements' definitions appear in https://doi.org/10.1137/16M1073352
(Cockburn, Fu, 2017) and https://doi.org/10.1090/mcom/3354
(Gilette, Kloefkorn, 2018)
"""

import typing
from ..references import Reference
from ..functionals import ListOfFunctionals
from ..finite_element import CiarletElement
from ..polynomials import polynomial_set_vector
from ..functionals import (IntegralMoment, TangentIntegralMoment, IntegralAgainst,
                           NormalIntegralMoment)
from ..symbolic import x, t, subs
from ..calculus import grad, curl
from ..vectors import vcross
from ..moments import make_integral_moment_dofs
from .dpc import DPC, VectorDPC


class TrimmedSerendipityHcurl(CiarletElement):
    """Trimmed serendipity Hcurl finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        poly = polynomial_set_vector(reference.tdim, reference.tdim, order - 1)
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
                            p = vcross(tuple(x), tuple(
                                x[0] ** i * x[1] ** j * x[2] ** (order - 1 - i - j)
                                if d == dim else 0
                                for d in range(3)))
                            assert isinstance(p, tuple)
                            poly.append(p)

            if order == 1:
                poly += [grad(x[0] * x[1] * x[2] ** order, 3)]
            else:
                poly += [grad(x[0] * x[1] * x[2] ** order, 3),
                         grad(x[0] * x[1] ** order * x[2], 3),
                         grad(x[0] ** order * x[1] * x[2], 3)]
            poly += [grad(x[0] * x[1] ** i * x[2] ** (order - i), 3) for i in range(order + 1)]
            poly += [grad(x[1] * x[0] ** i * x[2] ** (order - i), 3) for i in range(order + 1)
                     if i != 1]
            poly += [grad(x[2] * x[0] ** i * x[1] ** (order - i), 3) for i in range(order + 1)
                     if i != 1 and order - i != 1]
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
                    f = grad(x[0] ** (order - 1 - i) * x[1] ** i, 2)
                    f2 = subs((f[1], -f[0]), x, tuple(t))
                    dofs.append(IntegralAgainst(
                        reference, face, f2, entity=(2, f_n), mapping="contravariant"))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, reference.tdim
        )
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["trimmed serendipity Hcurl", "TScurl"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(curl)"


class TrimmedSerendipityHdiv(CiarletElement):
    """Trimmed serendipity Hdiv finite element."""

    def __init__(self, reference: Reference, order: int, variant: str = "equispaced"):
        poly = polynomial_set_vector(reference.tdim, reference.tdim, order - 1)
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
                poly.append(curl((0, -x[2] * p, x[1] * p)))
                p = x[1] * x[0] ** i * x[2] ** (order - 1 - i)
                poly.append(curl((-x[2] * p, 0, x[0] * p)))
                if order > 1:
                    p = x[2] * x[0] ** i * x[1] ** (order - 1 - i)
                    poly.append(curl((-x[1] * p, x[0] * p, 0)))

        dofs: ListOfFunctionals = []
        dofs += make_integral_moment_dofs(
            reference,
            facets=(NormalIntegralMoment, DPC, order - 1, {"variant": variant}),
            cells=(IntegralMoment, VectorDPC, order - 3, "covariant",
                   {"variant": variant}),
        )
        if order >= 2:
            if reference.tdim == 2:
                fs = [grad(x[0] ** (order - 1 - i) * x[1] ** i, 2)
                      for i in range(order)]
            else:
                fs = [grad(x[0] ** (order - 1 - i - j) * x[1] ** i * x[2] ** j, 3)
                      for i in range(order) for j in range(order - i)]
            for f in fs:
                f2 = subs(f, x, tuple(t))
                dofs.append(IntegralAgainst(reference, reference, f2, entity=(reference.tdim, 0),
                                            mapping="covariant"))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, reference.tdim
        )
        self.variant = variant

    def init_kwargs(self) -> typing.Dict[str, typing.Any]:
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["trimmed serendipity Hdiv", "TSdiv"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(div)"
