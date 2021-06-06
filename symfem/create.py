"""Create elements and references."""
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
    version = _f.read().strip()

_elementmap = {}
_elementlist = []


def add_element(ElementClass):
    """Add an element to Symfem.

    Parameters
    ----------
    ElementClass : type
        The class defining the element.
    """
    global _elementlist
    global _elementmap
    if not isinstance(ElementClass, type):
        raise TypeError("Element must be defined by a class.")
    if not issubclass(ElementClass, _FiniteElement):
        raise TypeError("Element must inherit from the FiniteElement class.")
    if ElementClass == _FiniteElement:
        raise TypeError("Cannot add the FiniteElement class itself.")
    if len(ElementClass.names) == 0:
        raise TypeError("An element with no names cannot be added")

    if ElementClass not in _elementlist:
        _elementlist.append(ElementClass)
    for _n in ElementClass.names:
        for _r in ElementClass.references:
            if _n not in _elementmap:
                _elementmap[_n] = {}
            if _r in _elementmap[_n]:
                assert ElementClass == _elementmap[_n][_r]
            _elementmap[_n][_r] = ElementClass


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
                and len(_element.names) > 0
            ):
                add_element(_element)


def create_reference(cell_type, vertices=None):
    """Make a reference cell.

    Parameters
    ----------
    cell_type : str
        The reference cell type.

        Supported values:
        interval, triangle, quadrilateral, tetrahedron, hexahedron,
        prism, pyramid, dual polygon(number_of_triangles)
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
    elif cell_type == "prism":
        if vertices is not None:
            return _references.Prism(vertices)
        return _references.Prism()
    elif cell_type == "pyramid":
        if vertices is not None:
            return _references.Pyramid(vertices)
        return _references.Pyramid()
    elif cell_type.startswith("dual polygon"):
        n_tri = int(cell_type.split("(")[1].split(")")[0])
        if vertices is not None:
            return _references.DualPolygon(n_tri, vertices)
        return _references.DualPolygon(n_tri)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")


def create_element(cell_type, element_type, order, variant="equispaced"):
    """Make a finite element.

    Parameters
    ----------
    cell_type : str
        The reference cell type.

        Supported values:
        interval, triangle, quadrilateral, tetrahedron, hexahedron,
        prism, pyramid, dual polygon(number_of_triangles)
    element_type : str
        The type of the element.

        Supported values:
        Lagrange, P,
        discontinuous Lagrange, dP, DP,
        vector Lagrange, vP,
        vector discontinuous Lagrange, vdP, vDP,
        matrix discontinuous Lagrange,
        symmetric matrix discontinuous Lagrange,
        Crouzeix-Raviart, CR, Crouzeix-Falk, CF,
        serendipity, S,
        serendipity Hcurl, Scurl, BDMCE, AAE,
        serendipity Hdiv, Sdiv, BDMCF, AAF,
        direct serendipity,
        Regge,
        Nedelec, Nedelec1, N1curl,
        Nedelec2, N2curl,
        Raviart-Thomas, RT, N1div,
        Brezzi-Douglas-Marini, BDM, N2div,
        Q,
        dQ,
        vector Q, vQ,
        NCE, RTCE, Qcurl,
        NCF, RTCF, Qdiv,
        Morley,
        Morley-Wang-Xu, MWX,
        Hermite,
        Mardal-Tai-Winther, MTW,
        Argyris,
        bubble,
        dual,
        Buffa-Christiansen, BC,
        rotated Buffa-Christiansen, RBC,
        Brezzi-Douglas-Fortin-Marini, BDFM,
        Brezzi-Douglas-Duran-Fortin, BDDF,
        Hellan-Herrmann-Johnson, HHJ,
        Arnold-Winther, AW,
        Bell,
        Kong-Mulder-Veldhuizen, KMV,
        Bernstein, Bernstein-Bezier,
        Hsieh-Clough-Tocher, Clough-Tocher, HCT, CT,
        reduced Hsieh-Clough-Tocher, rHCT,
        Taylor, discontinuous Taylor,
        bubble enriched Lagrange,
        bubble enriched vector Lagrange,
        Bogner-Fox-Schmit, BFS,
        Fortin-Soulie, FS,
        Bernardi-Raugel,
        Wu-Xu
    order : int
        The order of the element.
    variant : str
        The arrangement type of the points used the define the space.

        Supported values:
        equispaced, lobatto, radau, legendre
    """
    reference = create_reference(cell_type)

    if element_type in _elementmap:
        assert reference.name in _elementmap[element_type]
        return _elementmap[element_type][reference.name](reference, order, variant=variant)

    raise ValueError(f"Unsupported element type: {element_type}")
