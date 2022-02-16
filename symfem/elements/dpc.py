"""DPC elements on tensor product cells."""

from ..finite_element import CiarletElement
from ..polynomials import polynomial_set
from ..functionals import PointEvaluation, DotPointEvaluation
from .lagrange import Lagrange


class DPC(CiarletElement):
    """A dPc element."""

    def __init__(self, reference, order, variant="equispaced"):
        from symfem import create_reference

        if reference.name == "interval":
            lag = Lagrange(reference, order, variant)
        elif reference.name == "quadrilateral":
            lag = Lagrange(create_reference("triangle"), order, variant)
        elif reference.name == "hexahedron":
            lag = Lagrange(create_reference("tetrahedron"), order, variant)

        dofs = [
            PointEvaluation(d.dof_point(), entity=(reference.tdim, 0))
            for d in lag.dofs]

        super().__init__(
            reference, order, polynomial_set(reference.tdim, 1, order), dofs, reference.tdim, 1
        )
        self.variant = variant

    def init_kwargs(self):
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["dPc"]
    references = ["interval", "quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "L2"


class VectorDPC(CiarletElement):
    """Vector dPc finite element."""

    def __init__(self, reference, order, variant="equispaced"):
        scalar_space = DPC(reference, order, variant)
        dofs = []
        if reference.tdim == 1:
            directions = [1]
        else:
            directions = [
                tuple(1 if i == j else 0 for j in range(reference.tdim))
                for i in range(reference.tdim)
            ]
        for p in scalar_space.dofs:
            for d in directions:
                dofs.append(DotPointEvaluation(p.point, d, entity=p.entity))

        super().__init__(
            reference, order,
            polynomial_set(reference.tdim, reference.tdim, order),
            dofs,
            reference.tdim,
            reference.tdim,
        )
        self.variant = variant

    def init_kwargs(self):
        """Return the kwargs used to create this element."""
        return {"variant": self.variant}

    names = ["vector dPc"]
    references = ["quadrilateral", "hexahedron"]
    min_order = 0
    continuity = "L2"
