"""Functions to create integral moments."""

import typing

from .functionals import BaseFunctional
from .references import Reference

MomentType = typing.Tuple[typing.Type, typing.Type, int, typing.Union[str, None],
                          typing.Dict[str, typing.Any]]
SingleMomentTypeInput = typing.Union[
    MomentType,
    typing.Tuple[typing.Type, typing.Type, int, str],
    typing.Tuple[typing.Type, typing.Type, int, typing.Dict[str, typing.Any]],
    typing.Tuple[typing.Type, typing.Type, int],
]
MomentTypeInput = typing.Union[
    SingleMomentTypeInput,
    typing.Dict[str, SingleMomentTypeInput]
]


def _extract_moment_data(moment_data: MomentTypeInput, sub_type: str) -> MomentType:
    """Get the information for a moment.

    Args:
        moment_data: The moment data
        sub_type: The subentity type

    Returns:
        The moment type, finite elment, order, mapping, and keyword arguments for the moment
    """
    if isinstance(moment_data, dict):
        return _extract_moment_data(moment_data[sub_type], sub_type)

    mapping: typing.Union[str, None] = None
    if isinstance(moment_data[-1], dict):
        kwargs = moment_data[-1]
        if isinstance(moment_data[-2], str):
            mapping = moment_data[-2]
    else:
        kwargs = {}
        if isinstance(moment_data[-1], str):
            mapping = moment_data[-1]

    assert isinstance(moment_data[0], type)
    assert isinstance(moment_data[1], type)
    assert isinstance(moment_data[2], int)

    return moment_data[0], moment_data[1], moment_data[2], mapping, kwargs


def make_integral_moment_dofs(
    reference: Reference,
    vertices: typing.Optional[MomentTypeInput] = None,
    edges: typing.Optional[MomentTypeInput] = None,
    faces: typing.Optional[MomentTypeInput] = None,
    volumes: typing.Optional[MomentTypeInput] = None,
    cells: typing.Optional[MomentTypeInput] = None,
    facets: typing.Optional[MomentTypeInput] = None,
    ridges: typing.Optional[MomentTypeInput] = None,
    peaks: typing.Optional[MomentTypeInput] = None
) -> typing.List[BaseFunctional]:
    """Generate DOFs due to integral moments on sub entities.

    Args:
        reference: The reference cell.
        vertices: DOFs on dimension 0 entities.
        edges: DOFs on dimension 1 entities.
        faces: DOFs on dimension 2 entities.
        volumes: DOFs on dimension 3 entities.
        cells: DOFs on codimension 0 entities.
        facets: DOFs on codimension 1 entities.
        ridges: DOFs on codimension 2 entities.
        peaks: DOFs on codimension 3 entities.

    Returns:
        A list of DOFs for the element
    """
    dofs = []

    # DOFs per dimension
    for dim, moment_data in [
        (0, vertices), (1, edges), (2, faces), (3, volumes),
        (reference.tdim - 3, peaks), (reference.tdim - 2, ridges), (reference.tdim - 1, facets),
        (reference.tdim, cells),
    ]:
        if moment_data is not None:
            sub_type = reference.sub_entity_types[dim]
            if sub_type is not None:
                assert dim > 0
                for i, vs in enumerate(reference.sub_entities(dim)):
                    sub_ref = reference.sub_entity(dim, i, True)
                    IntegralMoment, SubElement, order, mapping, kwargs = _extract_moment_data(
                        moment_data, sub_ref.name)
                    if order >= SubElement.min_order:
                        sub_element = SubElement(sub_ref.default_reference(), order, **kwargs)
                        for dn, d in enumerate(sub_element.dofs):
                            f = sub_element.get_basis_function(dn)
                            kwargs = {}
                            if mapping is not None:
                                kwargs["mapping"] = mapping
                            dofs.append(IntegralMoment(
                                reference, sub_ref, f, d, (dim, i), **kwargs))
    return dofs
