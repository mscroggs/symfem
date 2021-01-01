"""symfem: Symbolic Finite Element Method Definitions."""
import os as _os
import importlib as _il
from .core import references as _references
from .core.finite_element import FiniteElement as _FiniteElement

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


for _file in _os.listdir(_os.path.join(_folder, "elements")):
    if _file.endswith(".py") and "__init__" not in _file:
        _fname = _file[:-3]
        _module = _il.import_module(f"symfem.elements.{_fname}")

        for _class_name in dir(_module):
            _element = getattr(_module, _class_name)
            if (
                isinstance(_element, type)
                and issubclass(_element, _FiniteElement)
                and _element != _FiniteElement
            ):
                for _n in _element.names:
                    if _n in _elementlist:
                        print(_n)
                        assert _element == _elementlist[_n]
                    _elementlist[_n] = _element


def create_reference(cell_type):
    """Make a reference cell.

    Parameters
    ----------
    cell_type : str
        The reference cell type.
        Supported values: interval, triangle, quadrilateral, tetrahedron, hexahedron
    """
    if cell_type == "interval":
        return _references.Interval()
    elif cell_type == "triangle":
        return _references.Triangle()
    elif cell_type == "tetrahedron":
        return _references.Tetrahedron()
    elif cell_type == "quadrilateral":
        return _references.Quadrilateral()
    elif cell_type == "hexahedron":
        return _references.Hexahedron()
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")


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
    reference = create_reference(cell_type)

    if element_type in _elementlist:
        return _elementlist[element_type](reference, order)

    raise ValueError(f"Unsupported element type: {element_type}")
