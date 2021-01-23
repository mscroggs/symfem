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

_elementmap = {}
_elementlist = []

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
                if _element not in _elementlist:
                    _elementlist.append(_element)
                for _n in _element.names:
                    if _n in _elementmap:
                        assert _element == _elementmap[_n]
                    _elementmap[_n] = _element


def create_reference(cell_type, vertices=None):
    """Make a reference cell.

    Parameters
    ----------
    cell_type : str
        The reference cell type.
        Supported values: interval, triangle, quadrilateral, tetrahedron, hexahedron
    vertices : list
        The vertices of the reference.
    """
    if cell_type == "interval":
        if vertices is not None:
            return _references.Interval(vertices)
        return _references.Interval()
    elif cell_type == "triangle":
        if vertices is not None:
            return _references.Triangle(vertices)
        return _references.Triangle()
    elif cell_type == "tetrahedron":
        if vertices is not None:
            return _references.Tetrahedron(vertices)
        return _references.Tetrahedron()
    elif cell_type == "quadrilateral":
        if vertices is not None:
            return _references.Quadrilateral(vertices)
        return _references.Quadrilateral()
    elif cell_type == "hexahedron":
        if vertices is not None:
            return _references.Hexahedron(vertices)
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
    element_type : str
        The type of the element.
        Supported values: Crouzeix-Raviart, CR, discontinuous Lagrange, dP, DP,
        serendipity, S, serendipity Hcurl, Scurl, BDMCE, AAE, serendipity Hdiv, Sdiv,
        BDMCF, AAF, vector discontinuous Lagrange, vdP, vDP, Lagrange, P,
        vector Lagrange, vP, Regge, Nedelec, Nedelec1, N1curl, Nedelec2, N2curl,
        Raviart-Thomas, RT, N1div, dQ, NCE, RTCE, Qcurl, Q, NCF, RTCF, Qdiv, vector Q,
        vQ, Brezzi-Douglas-Marini, BDM, N2div, Morley, Hermite, Mardal-Tai-Winther,
        MTW, Argyris, bubble
    order : int
        The order of the element.
    """
    reference = create_reference(cell_type)

    if element_type in _elementmap:
        assert cell_type in _elementmap[element_type].references
        return _elementmap[element_type](reference, order)

    raise ValueError(f"Unsupported element type: {element_type}")
