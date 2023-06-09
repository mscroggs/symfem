"""Functions to create integral moments."""

import typing

from .functionals import BaseFunctional
from .references import Reference
from . import mappings

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
        if moment_data is None:
            continue
        sub_type = reference.sub_entity_types[dim]
        if sub_type is None:
            continue
        assert dim > 0
        for i, vs in enumerate(reference.sub_entities(dim)):
            sub_ref = reference.sub_entity(dim, i, False)
            sub_ref_def = reference.default_reference().sub_entity(dim, i, False)
            IntegralMoment, SubElement, order, mapping, kwargs = _extract_moment_data(
                moment_data, sub_ref.name)
            m_kwargs = {}
            if mapping is not None:
                m_kwargs["mapping"] = mapping
            if order < SubElement.min_order:
                continue
            sub_element = SubElement(sub_ref.default_reference(), order, **kwargs)
            for dn, d in enumerate(sub_element.dofs):
                f = sub_element.get_basis_function(dn)
                if reference.vertices != reference.default_reference().vertices:
                    m = IntegralMoment.default_mapping if mapping is None else mapping
                    if m is None or not hasattr(mappings, f"{m}_inverse_transpose"):
                        raise ValueError(
                            "Cannot create this element on a non-default reference.")
                    #m = "identity"
                    mf = getattr(mappings, f"{m}_inverse_transpose")
                    # print("fbefore", f)
                    # print(sub_ref.vertices)
                    f = mf(f, sub_ref.get_map_to_self(), sub_ref.get_inverse_map_to_self(),
                           sub_ref.tdim, substitute=False)
                    if sub_ref.tdim > 1:
                        f *= sub_ref_def.volume() / sub_ref.volume()
                    #
                    # F = (1,1) + x(1, 1) + y(0, 3)
                    # J = ( 1 0 )
                    #     ( 1 3 )
                    # detJ = 3
                    #
                    # FF : (a, b) -> 1/detJ J (a, b)
                    #      (a, b) -> 1/3 * (a, a+3b)
                    #
                    # J^-1 = (  1   0   )
                    #        ( -1/3 1/3 )
                    # J^-T = ( 1 -1/3 )
                    #        ( 0  1/3 )
                    #
                    # FFit : (c, d) -> detJ J^-T (c, d)
                    #        (c, d) -> (3c - d, d)
                    #
                    # CLAIM: (a, b) . (c, d) = FF(a,b) . FFit(c,d)
                    #   FF(a,b) . FFit(c,d) = 1/3 * (a, a+3b) . (3c - d, d)
                    #                       = 1/3 * (3ac - ad + ad + 3bd)
                    #                       = 1/3 * (3ac + 3bd)
                    #                       = ac + bd
                    # print("fafter", f)
                    # print()
                dofs.append(IntegralMoment(
                    reference, sub_ref, f, d, (dim, i), **m_kwargs))
    return dofs
