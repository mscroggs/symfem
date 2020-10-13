"""FEAST: Finite Element Automatic Symbolic Tabulator."""
from . import simplex, tp, references


def feast_element(cell_type, element_type, order):
    """Make a finite element.

    Parameters
    ----------
    cell_type : str
        The reference cell type.
        Supported values: interval, triangle, quadrilateral, tetrahedron, hexahedron
    element_type: str
        The type of the element.
        Supported values (simplex):
            Lagrange, vector Lagrange, Nedelec, Raviart-Thomas
        Supported values (tensor product):
            Q, vector Q
    order: int
        The order of the element.
    """
    if cell_type == "interval":
        reference = references.Interval()
    elif cell_type == "triangle":
        reference = references.Triangle()
    elif cell_type == "tetrahedron":
        reference = references.Tetrahedron()
    elif cell_type == "quadrilateral":
        reference = references.Quadrilateral()
    elif cell_type == "hexahedron":
        reference = references.Hexahedron()
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")

    if reference.simplex:
        if element_type == "Lagrange":
            return simplex.Lagrange(reference, order)
        if element_type == "vector Lagrange":
            return simplex.VectorLagrange(reference, order)
        if element_type == "Nedelec":
            return simplex.NedelecFirstKind(reference, order)
        if element_type == "Raviart-Thomas":
            return simplex.RaviartThomas(reference, order)

    if reference.tp:
        if element_type == "Q":
            return tp.Q(reference, order)
        if element_type == "vector Q":
            return tp.VectorQ(reference, order)

    raise ValueError(f"Unsupported element type: {element_type}")
