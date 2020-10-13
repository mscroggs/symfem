import sympy
from itertools import product
from .finite_element import FiniteElement
from .polynomials import qolynomial_set
from .functionals import PointEvaluation, DotPointEvaluation


class Q(FiniteElement):
    def __init__(self, reference, order):
        if order == 0:
            dofs = [PointEvaluation(tuple(sympy.Rational(1, 2) for i in range(reference.tdim)))]
        else:
            dofs = []
            for i in product(range(order + 1), repeat=reference.tdim):
                dofs.append(PointEvaluation(tuple(sympy.Rational(j, order) for j in i)))

        super().__init__(qolynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1)


class VectorQ(FiniteElement):
    def __init__(self, reference, order):
        if reference.name == "interval":
            directions = [(1, )]
        elif reference.name == "quadrilateral":
            directions = [(1, 0), (0, 1)]
        elif reference.name == "hexahedron":
            directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        scalar_space = Q(reference, order)
        dofs = []
        for p in scalar_space.dofs:
            for d in directions:
                dofs.append(DotPointEvaluation(p.point, d))

        super().__init__(qolynomial_set(reference.tdim, reference.tdim, order), dofs, reference.tdim, reference.tdim)
