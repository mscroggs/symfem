import sympy
from .symbolic import x, zero, subs
from .functionals import PointEvaluation, DotPointEvaluation


class FiniteElement:
    def __init__(self, basis, dofs, domain_dim, range_dim):
        assert len(basis) == len(dofs)
        self.basis = basis
        self.dofs = dofs
        self.domain_dim = domain_dim
        self.range_dim = range_dim
        self.space_dim = len(dofs)
        self._basis_functions = None

    def get_basis_functions(self):
        if self._basis_functions is None:
            mat = []
            for b in self.basis:
                row = []
                for d in self.dofs:
                    row.append(d.eval(b))
                mat.append(row)
            minv = sympy.Matrix(mat).inv()
            self._basis_functions = []
            if self.range_dim == 1:
                # Scalar space
                for i, dof in enumerate(self.dofs):
                    b = zero
                    for c, d in zip(minv.row(i), self.basis):
                        b += c * d
                    self._basis_functions.append(b)
            else:
                # Vector space
                for i, dof in enumerate(self.dofs):
                    b = [zero for i in self.basis[0]]
                    for c, d in zip(minv.row(i), self.basis):
                        for j, d_j in enumerate(d):
                            b[j] += c * d_j
                    self._basis_functions.append(b)

        return self._basis_functions

    def tabulate_basis(self, points, order="xyzxyz"):
        if self.range_dim == 1:
            output = []
            for p in points:
                row = []
                for b in self.get_basis_functions():
                    row.append(subs(b, x, p))
                output.append(row)
            return output

        if order == "xxyyzz":
            output = []
            for p in points:
                row = []
                for d in range(self.range_dim):
                    for b in self.get_basis_functions():
                        row.append(subs(b[d], x, p))
                output.append(row)
            return output
        if order == "xyzxyz":
            output = []
            for p in points:
                row = []
                for d in range(self.range_dim):
                    for b in self.get_basis_functions():
                        row.append(subs(b[d], x, p))
                output.append(row)
            return output
        raise ValueError(f"Unknown order: {order}")


def make_integral_moment_dofs(reference, vertices=None, edges=None, faces=None, volumes=None):
    dofs = []
    if vertices is not None:
        if vertices[0] == PointEvaluation:
            for vertex_i, v in enumerate(reference.vertices):
                dofs.append(vertices[0](v))
        elif vertices[0] == DotPointEvaluation:
            for vertex_i, v in enumerate(reference.vertices):
                for p in vertices[1]:
                    dofs.append(vertices[0](v, p))

    for dim, moment_data in zip([1, 2, 3], [edges, faces, volumes]):
        if moment_data is not None and moment_data[2] >= moment_data[3]:
            sub_type = reference.sub_entity_types[dim]
            if sub_type is not None:
                for i, vs in enumerate(reference.sub_entities(dim)):
                    sub_ref = sub_type(vertices=[reference.vertices[v] for v in vs])
                    sub_element = moment_data[1](sub_ref, moment_data[2])
                    for d in sub_element.get_basis_functions():
                        dofs.append(moment_data[0](sub_ref, d))
    return dofs
