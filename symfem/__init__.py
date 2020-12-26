"""symfem: Symbolic Finite Element Method Definitions."""
import os as _os
from . import simplex as _simplex
from . import tp as _tp
from . import references as _references
from .finite_element import FiniteElement as _FiniteElement

_folder = _os.path.dirname(_os.path.realpath(__file__))

if _os.path.isfile(_os.path.join(_folder, "../VERSION")):
    # If running from folder
    _v_folder = _os.path.join(_folder, "..")
else:
    # If running from installation
    _v_folder = _folder

with open(_os.path.join(_v_folder, "VERSION")) as _f:
    __version__ = _f.read().strip()

_elementlist = {}

for _cell_class, _module in [("simplex", _simplex), ("tp", _tp)]:
    _elementlist[_cell_class] = {}
    for _class_name in dir(_module):
        _element = getattr(_module, _class_name)
        if (
            isinstance(_element, type)
            and issubclass(_element, _FiniteElement)
            and _element != _FiniteElement
        ):
            for _n in _element.names:
                _elementlist[_cell_class][_n] = _element


def create_element(cell_type, element_type, order):
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
        reference = _references.Interval()
    elif cell_type == "triangle":
        reference = _references.Triangle()
    elif cell_type == "tetrahedron":
        reference = _references.Tetrahedron()
    elif cell_type == "quadrilateral":
        reference = _references.Quadrilateral()
    elif cell_type == "hexahedron":
        reference = _references.Hexahedron()
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")

    if reference.simplex:
        if element_type in _elementlist["simplex"]:
            return _elementlist["simplex"][element_type](reference, order)

    if reference.tp:
        if element_type in _elementlist["tp"]:
            return _elementlist["tp"][element_type](reference, order)

    raise ValueError(f"Unsupported element type: {element_type}")
