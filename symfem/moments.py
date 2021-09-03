"""Functions to create integral moments."""


def extract_moment_data(moment_data, sub_type):
    """Get the information for a moment."""
    if isinstance(moment_data, dict):
        return extract_moment_data(moment_data[sub_type], sub_type)

    if isinstance(moment_data[-1], dict):
        kwargs = moment_data[-1]
        moment_data = moment_data[:-1]
    else:
        kwargs = {}

    if len(moment_data) == 3:
        return moment_data + (None, kwargs)
    else:
        return moment_data + (kwargs, )


def make_integral_moment_dofs(
    reference,
    vertices=None, edges=None, faces=None, volumes=None,
    cells=None, facets=None, ridges=None, peaks=None
):
    """Generate DOFs due to integral moments on sub entities.

    Parameters
    ----------
    reference: symfem.references.Reference
        The reference cell.
    vertices: tuple
        DOFs on dimension 0 entities.
    edges: tuple
        DOFs on dimension 1 entities.
    faces: tuple
        DOFs on dimension 2 entities.
    volumes: tuple
        DOFs on dimension 3 entities.
    cells: tuple
        DOFs on codimension 0 entities.
    facets: tuple
        DOFs on codimension 1 entities.
    ridges: tuple
        DOFs on codimension 2 entities.
    peaks: tuple
        DOFs on codimension 3 entities.
    """
    dofs = []

    # DOFs per dimension
    for dim, moment_data in enumerate([vertices, edges, faces, volumes]):
        if moment_data is not None:
            sub_type = reference.sub_entity_types[dim]
            if sub_type is not None:
                assert dim > 0
                for i, vs in enumerate(reference.sub_entities(dim)):
                    sub_ref = reference.sub_entity(dim, i, True)
                    IntegralMoment, SubElement, order, mapping, kwargs = extract_moment_data(
                        moment_data, sub_ref.name)
                    if order >= SubElement.min_order:
                        sub_element = SubElement(sub_ref.default_reference(), order, **kwargs)
                        for dn, d in enumerate(sub_element.dofs):
                            f = sub_element.get_basis_function(dn)
                            if mapping is None:
                                dofs.append(IntegralMoment(sub_ref, f, d, entity=(dim, i)))
                            else:
                                dofs.append(IntegralMoment(sub_ref, f, d, entity=(dim, i),
                                                           mapping=mapping))

    # DOFs per codimension
    for _dim, moment_data in enumerate([peaks, ridges, facets, cells]):
        dim = reference.tdim - 3 + _dim
        if moment_data is not None:
            sub_type = reference.sub_entity_types[dim]
            if sub_type is not None:
                assert dim > 0
                for i, vs in enumerate(reference.sub_entities(dim)):
                    sub_ref = reference.sub_entity(dim, i, True)
                    IntegralMoment, SubElement, order, mapping, kwargs = extract_moment_data(
                        moment_data, sub_ref.name)
                    if order >= SubElement.min_order:
                        sub_element = SubElement(sub_ref.default_reference(), order, **kwargs)
                        for dn, d in enumerate(sub_element.dofs):
                            f = sub_element.get_basis_function(dn)
                            if mapping is None:
                                dofs.append(IntegralMoment(sub_ref, f, d, entity=(dim, i)))
                            else:
                                dofs.append(IntegralMoment(sub_ref, f, d, entity=(dim, i),
                                                           mapping=mapping))

    return dofs
