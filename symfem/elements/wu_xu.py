"""Wu-Xu elements on simplices.

This element's definition appears in https://doi.org/10.1090/mcom/3361
(Wu, Xu, 2019)
"""

import typing
from ..finite_element import CiarletElement
from ..functionals import (PointEvaluation, DerivativePointEvaluation,
                           IntegralOfDirectionalMultiderivative, ListOfFunctionals)
from ..functions import FunctionInput
from ..polynomials import polynomial_set_1d
from ..references import Reference
from ..symbols import x


def derivatives(dim: int, order: int) -> typing.List[typing.Tuple[int, ...]]:
    """Return all the orders of a multidimensional derivative."""
    if dim == 1:
        return [(order, )]

    out = []
    for i in range(order + 1):
        out += [(i, ) + j for j in derivatives(dim - 1, order - i)]
    return out


class WuXu(CiarletElement):
    """Wu-Xu finite element."""

    def __init__(self, reference: Reference, order: int):
        assert order == reference.tdim + 1
        poly: typing.List[FunctionInput] = []
        poly += polynomial_set_1d(reference.tdim, order)

        if reference.name == "interval":
            bubble = x[0] * (1 - x[0])
        elif reference.name == "triangle":
            bubble = x[0] * x[1] * (1 - x[0] - x[1])
        elif reference.name == "tetrahedron":
            bubble = x[0] * x[1] * x[2] * (1 - x[0] - x[1] - x[2])

        poly += [bubble * i for i in polynomial_set_1d(reference.tdim, 1)[1:]]

        dofs: ListOfFunctionals = []
        for v_n, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(reference, v, entity=(0, v_n)))
            for i in range(reference.tdim):
                dofs.append(DerivativePointEvaluation(
                    reference, v, tuple(1 if i == j else 0 for j in range(reference.tdim)),
                    entity=(0, v_n)))
        for codim in range(1, reference.tdim):
            dim = reference.tdim - codim
            for e_n, vs in enumerate(reference.sub_entities(codim=codim)):
                subentity = reference.sub_entity(dim, e_n)
                volume = subentity.jacobian()
                normals = []
                if codim == 1:
                    normals = [subentity.normal()]
                elif codim == 2 and reference.tdim == 3:
                    for f_n, f_vs in enumerate(reference.sub_entities(2)):
                        if vs[0] in f_vs and vs[1] in f_vs:
                            face = reference.sub_entity(2, f_n)
                            normals.append(face.normal())
                else:
                    raise NotImplementedError
                for orders in derivatives(len(normals), len(normals)):
                    dofs.append(IntegralOfDirectionalMultiderivative(
                        reference, subentity, tuple(normals), orders, (dim, e_n),
                        scale=1 / volume))

        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["Wu-Xu"]
    references = ["interval", "triangle", "tetrahedron"]
    min_order = {"interval": 2, "triangle": 3, "tetrahedron": 4}
    max_order = {"interval": 2, "triangle": 3, "tetrahedron": 4}
    continuity = "C{order}"
