"""Trimmed serendipity elements on tensor products.

These elements' definitions appear in https://doi.org/10.1137/16M1073352
(Cockburn, Fu, 2017) and https://doi.org/10.1090/mcom/3354
(Gilette, Kloefkorn, 2018)
"""

from ..finite_element import CiarletElement
from ..polynomials import polynomial_set
from ..functionals import IntegralMoment, TangentIntegralMoment, IntegralAgainst
from ..quadrature import get_quadrature
from ..symbolic import x, t, subs
from ..calculus import grad
from ..vectors import vcross
from ..moments import make_integral_moment_dofs
from .dpc import DPC, VectorDPC


class TrimmedSerendipityHcurl(CiarletElement):
    """Trimmed serendipity Hcurl finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        poly = polynomial_set(reference.tdim, reference.tdim, order - 1)
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
            poly += [
                vcross(x, [x[0] ** i * x[1] ** j * x[2] ** (order - 1 - i - j)
                           if d == dim else 0 for d in range(3)])
                for i in range(order) for j in range(order - i) for dim in range(3)
                if i == 0 or dim != 0]

            if order == 1:
                poly += [grad(x[0] * x[1] * x[2] ** order, 3)]
            else:
                poly += [grad(x[0] * x[1] * x[2] ** order, 3),
                         grad(x[0] * x[1] ** order * x[2], 3),
                         grad(x[0] ** order * x[1] * x[2], 3)]
            # TODO
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

        points, _ = get_quadrature(variant, order)

        dofs = []
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
                    f = subs([f[1], -f[0]], x, t)
                    dofs.append(IntegralAgainst(face, f, entity=(2, 0)))

        super().__init__(
            reference, order, poly, dofs, reference.tdim, reference.tdim
        )
        self.variant = variant

    def init_kwargs(self):
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["trimmed serendipity Hcurl", "TScurl"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 1
    continuity = "H(curl)"
