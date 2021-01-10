import symfem


def test_available_references():
    r_supported = symfem.create_reference.__doc__
    r_supported = r_supported.split("cell_type : str\n")[1]
    r_supported = r_supported.split("vertices : list\n")[0]
    r_supported = r_supported.split("Supported values:")[1]
    r_supported = set([i.strip() for i in r_supported.split(",")])

    e_supported = symfem.create_element.__doc__
    e_supported = e_supported.split("cell_type : str\n")[1]
    e_supported = e_supported.split("element_type : str\n")[0]
    e_supported = e_supported.split("Supported values:")[1]
    e_supported = set([i.strip() for i in e_supported.split(",")])

    assert r_supported == e_supported


def test_available_elements():
    supported = symfem.create_element.__doc__
    supported = supported.split("element_type : str\n")[1]
    supported = supported.split("order : int\n")[0]
    supported = supported.split("Supported values:")[1]
    supported = set([i.strip() for i in supported.split(",")])

    assert set(symfem._elementmap.keys()) == supported
