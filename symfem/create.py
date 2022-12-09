"""Create elements and references."""

import importlib as _il
import os as _os
import typing as _typing

from . import references as _references
from .finite_element import FiniteElement as _FiniteElement
from .geometry import SetOfPointsInput as _SetOfPointsInput
from .geometry import parse_set_of_points_input as _parse_set_of_points_input

_folder = _os.path.dirname(_os.path.realpath(__file__))

_elementmap: _typing.Dict[str, _typing.Dict[str, _typing.Type]] = {}
_elementlist: _typing.List[_typing.Type] = []


def add_element(element_class: _typing.Type):
    """Add an element to Symfem.

    Args:
        element_class: The class defining the element.
    """
    global _elementlist
    global _elementmap
    if not isinstance(element_class, type):
        raise TypeError("Element must be defined by a class.")
    if not issubclass(element_class, _FiniteElement):
        raise TypeError("Element must inherit from the FiniteElement class.")
    if element_class == _FiniteElement:
        raise TypeError("Cannot add the FiniteElement class itself.")
    if len(element_class.names) == 0:
        raise TypeError("An element with no names cannot be added")

    if element_class not in _elementlist:
        _elementlist.append(element_class)
    for _n in element_class.names:
        for _r in element_class.references:
            if _n not in _elementmap:
                _elementmap[_n] = {}
            if _r in _elementmap[_n]:
                assert element_class == _elementmap[_n][_r]
            _elementmap[_n][_r] = element_class


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


def create_reference(
    cell_type: str, vertices: _typing.Optional[_SetOfPointsInput] = None
) -> _references.Reference:
    """Make a reference cell.

    Args:
        cell_type: The reference cell type.
                   Supported values: point, interval, triangle, quadrilateral, tetrahedron,
                   hexahedron, prism, pyramid, dual polygon(number_of_triangles)
        vertices: The vertices of the reference.
    """
    args = []
    if vertices is not None:
        args.append(_parse_set_of_points_input(vertices))

    if cell_type == "point":
        return _references.Point(*args)
    if cell_type == "interval":
        return _references.Interval(*args)
    elif cell_type == "triangle":
        return _references.Triangle(*args)
    elif cell_type == "tetrahedron":
        return _references.Tetrahedron(*args)
    elif cell_type == "quadrilateral":
        return _references.Quadrilateral(*args)
    elif cell_type == "hexahedron":
        return _references.Hexahedron(*args)
    elif cell_type == "prism":
        return _references.Prism(*args)
    elif cell_type == "pyramid":
        return _references.Pyramid(*args)
    elif cell_type.startswith("dual polygon"):
        n_tri = int(cell_type.split("(")[1].split(")")[0])
        return _references.DualPolygon(n_tri, *args)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")


def create_element(
    cell_type: str, element_type: str, order: int, **kwargs: _typing.Any
) -> _FiniteElement:
    """Make a finite element.

    Args:
        cell_type: The reference cell type.
                   Supported values: point, interval, triangle, quadrilateral, tetrahedron,
                   hexahedron, prism, pyramid, dual polygon(number_of_triangles)
        element_type: The type of the element.
                      Supported values:
                      Lagrange, P,
                      vector Lagrange, vP,
                      matrix Lagrange,
                      symmetric matrix Lagrange,
                      dPc,
                      vector dPc,
                      Crouzeix-Raviart, CR, Crouzeix-Falk, CF,
                      conforming Crouzeix-Raviart, conforming CR,
                      serendipity, S,
                      serendipity Hcurl, Scurl, BDMCE, AAE,
                      serendipity Hdiv, Sdiv, BDMCF, AAF,
                      direct serendipity,
                      Regge,
                      Nedelec, Nedelec1, N1curl, Ncurl,
                      Nedelec2, N2curl,
                      Raviart-Thomas, RT, N1div,
                      Brezzi-Douglas-Marini, BDM, N2div,
                      Q,
                      vector Q, vQ,
                      NCE, RTCE, Qcurl,
                      NCF, RTCF, Qdiv,
                      Morley,
                      Morley-Wang-Xu, MWX,
                      Hermite,
                      Mardal-Tai-Winther, MTW,
                      Argyris,
                      bubble,
                      dual polynomial, dual P, dual,
                      Buffa-Christiansen, BC,
                      rotated Buffa-Christiansen, RBC,
                      Brezzi-Douglas-Fortin-Marini, BDFM,
                      Brezzi-Douglas-Duran-Fortin, BDDF,
                      Hellan-Herrmann-Johnson, HHJ,
                      Arnold-Winther, AW, conforming Arnold-Winther,
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
                      Wu-Xu,
                      transition,
                      Guzman-Neilan,
                      nonconforming Arnold-Winther, nonconforming AW,
                      TScurl, trimmed serendipity Hcurl,
                      TSdiv, trimmed serendipity Hdiv,
                      TNT, tiniest tensor,
                      TNTcurl, tiniest tensor Hcurl,
                      TNTdiv, tiniest tensor Hdiv,
                      Arnold-Boffi-Falk, ABF,
                      Arbogast-Correa, AC, AC full, Arbogast-Correa full,
                      Rannacher-Turek,
                      P1-iso-P2, P2-iso-P1, iso-P2 P1,
                      Huang-Zhang, HZ
        order: The order of the element.
    """
    reference = create_reference(cell_type)

    if element_type in _elementmap:
        if reference.name not in _elementmap[element_type]:
            raise ValueError(f"{element_type} element cannot be created on a {reference.name}.")
        element_class = _elementmap[element_type][reference.name]
        if not _order_is_allowed(element_class, reference.name, order):
            raise ValueError(f"Order {order} {element_type} element cannot be created.")
        return element_class(reference, order, **kwargs)

    raise ValueError(f"Unsupported element type: {element_type}")


def _order_is_allowed(
    element_class: _typing.Type, ref: str, order: int
) -> bool:
    """Check that an order is valid for an element.

    Args:
        element_class: The element class
        ref: The reference cell
        order: The polynomial order
    """
    if hasattr(element_class, "min_order"):
        if isinstance(element_class.min_order, dict):
            if ref in element_class.min_order:
                if order < element_class.min_order[ref]:
                    return False
        elif order < element_class.min_order:
            return False
    if hasattr(element_class, "max_order"):
        if isinstance(element_class.max_order, dict):
            if ref in element_class.max_order:
                if order > element_class.max_order[ref]:
                    return False
        elif order > element_class.max_order:
            return False
    return True
