"""Functions to create integral moments."""


def make_integral_moment_dofs(
    reference,
    vertices=None, edges=None, faces=None, volumes=None,
    cells=None, facets=None, ridges=None, peaks=None,
    variant="equispaced"
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
    from symfem import create_reference
    dofs = []

    # DOFs per dimension
    for dim, moment_data in enumerate([vertices, edges, faces, volumes]):
        if moment_data is not None:
            if len(moment_data) == 3:
                IntegralMoment, SubElement, order = moment_data
                mapping = None
            else:
                IntegralMoment, SubElement, order, mapping = moment_data
            if order >= SubElement.min_order:
                sub_type = reference.sub_entity_types[dim]
                if sub_type is not None:
                    assert dim > 0
                    for i, vs in enumerate(reference.sub_entities(dim)):
                        sub_ref = create_reference(
                            sub_type,
                            vertices=[reference.reference_vertices[v] for v in vs])
                        sub_element = SubElement(sub_ref, order, variant=variant)
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
            if len(moment_data) == 3:
                IntegralMoment, SubElement, order = moment_data
                mapping = None
            else:
                IntegralMoment, SubElement, order, mapping = moment_data
            if order >= SubElement.min_order:
                sub_type = reference.sub_entity_types[dim]
                if sub_type is not None:
                    assert dim > 0
                    for i, vs in enumerate(reference.sub_entities(dim)):
                        sub_ref = create_reference(
                            sub_type,
                            vertices=[reference.reference_vertices[v] for v in vs])
                        sub_element = SubElement(sub_ref, order, variant=variant)
                        for dn, d in enumerate(sub_element.dofs):
                            f = sub_element.get_basis_function(dn)
                            if mapping is None:
                                dofs.append(IntegralMoment(sub_ref, f, d, entity=(dim, i)))
                            else:
                                dofs.append(IntegralMoment(sub_ref, f, d, entity=(dim, i),
                                                           mapping=mapping))

    return dofs
