"""Conforming Crouzeix-Raviart elements on simplices.

This element's definition appears in https://doi.org/10.1051/m2an/197307R300331
(Crouzeix, Raviart, 1973)
"""

import sympy
from ..finite_element import CiarletElement
from ..polynomials import polynomial_set
from ..functionals import PointEvaluation
from ..symbolic import x


class ConformingCrouzeixRaviart(CiarletElement):
    """Conforming Crouzeix-Raviart finite element."""

    def __init__(self, reference, order, variant):
        assert reference.name in ["triangle", "tetrahedron"]

        poly = polynomial_set(reference.tdim, 1, order)

        if reference.name == "triangle":
            poly += [
                x[0] ** i * x[1] ** (order - i) * (x[0] + x[1])
                for i in range(1, order)
            ]
        if reference.name == "tetrahedron":
            poly += [
                x[0] ** i * x[1] ** j * x[2] ** (order - i) * (x[0] + x[1] + x[2])
                for i in range(1, order) for j in range(1, order + 1 - i)
            ]

        dofs = []
        for i, v in enumerate(reference.vertices):
            dofs.append(PointEvaluation(v, entity=(0, i)))
        if order >= 2:
            for i, edge in enumerate(reference.edges):
                for p in range(1, order):
                    v = tuple(o + sympy.Rational(p * a, order) for o, a in zip(
                        reference.vertices[edge[0]], reference.vertices[edge[1]]))
                    dofs.append(PointEvaluation(v, entity=(1, i)))
            if reference.tdim == 2:
                for i in range(1, order):
                    for j in range(1, order + 1 - i):
                        point = (
                            sympy.Rational(3 * i - 1, 3 * order),
                            sympy.Rational(3 * j - 1, 3 * order)
                        )
                        dofs.append(PointEvaluation(point, entity=(2, 0)))

            if reference.tdim == 3:
                if order >= 3:
                    for fn in range(reference.sub_entity_count(2)):
                        face = reference.sub_entity(2, fn)
                        for i in range(1, order - 1):
                            for j in range(1, order - i):
                                point = tuple(
                                    o + a1 * sympy.Rational(i, order - 1)
                                    + a2 * sympy.Rational(j, order - 1)
                                    for o, a1, a2 in zip(face.origin, *face.axes)
                                )
                                dofs.append(PointEvaluation(point, entity=(2, fn)))

                for i in range(1, order):
                    for j in range(1, order + 1 - i):
                        for k in range(1, order + 2 - i - j):
                            print(i, j, k)
                            dofs.append(PointEvaluation(
                                (sympy.Rational(4 * i - 1, 4 * order),
                                 sympy.Rational(4 * j - 1, 4 * order),
                                 sympy.Rational(4 * k - 1, 4 * order)),
                                entity=(3, 0)))

        print(len(poly), len(dofs))
        super().__init__(reference, order, poly, dofs, reference.tdim, 1)

    names = ["conforming Crouzeix-Raviart", "conforming CR"]
    references = ["triangle"]
    min_order = 1
    max_order = {"tetrahedron": 1}
    continuity = "L2"
