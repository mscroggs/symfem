"""FEAST: Finite Element Automatic Symbolic Tabulator."""
from . import simplex, tp, references
from .finite_element import FiniteElement
import os as _os

_folder = _os.path.dirname(_os.path.realpath(__file__))

if _os.path.isfile(_os.path.join(_folder, "../VERSION")):
    # If running from folder
    _v_folder = os.path.join(ave_folder, "..")
else:
    # If running from installation
    _v_folder = _folder

with open(os.path.join(_v_folder, "VERSION")) as f:
    __version__ = f.read().strip()

version_tuple = tuple(int(i) for i in version.split("."))

_elementlist = {}

for cell_class, module in [("simplex", simplex), ("tp", tp)]:
    _elementlist[cell_class] = {}
    for class_name in dir(module):
        element = getattr(module, class_name)
        if (
            isinstance(element, type)
            and issubclass(element, FiniteElement)
            and element != FiniteElement
        ):
            for n in element.names:
                _elementlist[cell_class][n] = element


def feast_element(cell_type, element_type, order):
    """Make a finite element.

    Parameters
    ----------
    cell_type : str
        The reference cell type.
        Supported values: interval, triangle, quadrilateral, tetrahedron, hexahedron
    element_type: str
        The type of the element.
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
        if element_type in _elementlist["simplex"]:
            return _elementlist["simplex"][element_type](reference, order)

    if reference.tp:
        if element_type in _elementlist["tp"]:
            return _elementlist["tp"][element_type](reference, order)

    raise ValueError(f"Unsupported element type: {element_type}")
