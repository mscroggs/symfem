####################################################
Symfem: a symbolic finite element definition library
####################################################

Welcome to the Symfem documention.

*****************
Installing Symfem
*****************

Symfem can be installed from `GitHub repo <https://github.com/mscroggs/symfem>`_ by running::

    git clone https://github.com/mscroggs/symfem.git
    cd symfem
    python3 setup.py install

Alternatively, the latest release can be installed from PyPI by running::

    pip3 install symfem

Using Symfem
============

Finite elements
---------------
Finite elements can be created in symfem using the :func:`symfem.create_element` function.
For example, some elements are created in the following snippet:

.. code-block:: python

    import symfem

    lagrange = symfem.create_element("triangle", "Lagrange", 1)
    rt = symfem.create_element("tetrahedron", "Raviart-Thomas", 2)
    nedelec = symfem.create_element("triangle", "N2curl", 1)
    qcurl = symfem.create_element("quadrilateral", "Qcurl", 2)

`create_element` will create a :class:`symfem.core.finite_element.FiniteElement` object.
From this object, the polynomial basis of the element can be obtained:

.. code-block:: python

    import symfem

    lagrange = symfem.create_element("triangle", "Lagrange", 1)
    print(lagrange.get_polynomial_basis())

::

    [1, x, y]

Each item in the polynomial basis will be a `Sympy <https://www.sympy.org>`_ symbolic expression.

The functionals that define the DOFs of the finite element space can be obtained with the following
snippet.

.. code-block:: python

    import symfem

    lagrange = symfem.create_element("triangle", "Lagrange", 1)
    print(lagrange.dofs)

::

    [<symfem.core.functionals.PointEvaluation object at 0x{ADDRESS}>, <symfem.core.functionals.PointEvaluation object at 0x{ADDRESS}>, <symfem.core.functionals.PointEvaluation object at 0x{ADDRESS}>]

Each functional will be a functional defined in :mod:`symfem.core.functionals`.

The basis functions spanning the finite element space can be obtained, or tabulated
at a set of points:

.. code-block:: python

    import symfem

    lagrange = symfem.create_element("triangle", "Lagrange", 1)
    print(lagrange.get_basis_functions())

    points = [[0, 0], [0.5, 0], [1, 0], [0.25, 0.25]]
    print(lagrange.tabulate_basis(points))

::

    [-x - y + 1, x, y]
    [[1, 0, 0], [0.500000000000000, 0.500000000000000, 0], [0, 1, 0], [0.500000000000000, 0.250000000000000, 0.250000000000000]]

Reference elements
------------------
Reference elements can be obtained from a :class:`symfem.core.finite_element.FiniteElement`:

.. code-block:: python

    import symfem

    lagrange = symfem.create_element("triangle", "Lagrange", 1)
    reference = lagrange.reference

Alternatively, reference elements can be created using the :func:`symfem.create_reference` function.
For example:

.. code-block:: python

    import symfem

    triangle = symfem.create_reference("triangle")
    interval = symfem.create_reference("interval")
    tetrahedron = symfem.create_reference("tetrahedron")
    triangle2 = symfem.create_reference("triangle", ((0, 0, 0), (0, 1, 0), (1, 0, 1)))

In the final example, the vertices of the reference have been provided, so a reference
with these three vertices will be created.

Various information about the reference can be accessed. The reference element's subentities
can be obtained:

.. code-block:: python

    import symfem

    triangle = symfem.create_reference("triangle")
    print(triangle.vertices)
    print(triangle.edges)

::

    ((0, 0), (1, 0), (0, 1))
    ((1, 2), (0, 2), (0, 1))

The origin and axes of the element can be obtained:

.. code-block:: python

    import symfem

    triangle = symfem.create_reference("triangle")
    print(triangle.origin)
    print(triangle.axes)

::

    (0, 0)
    ((1, 0), (0, 1))

The topological and geometric dimensions of the element can be obtained:

.. code-block:: python

    import symfem

    triangle = symfem.create_reference("triangle")
    print(triangle.tdim)
    print(triangle.gdim)

    triangle2 = symfem.create_reference("triangle", ((0, 0, 0), (0, 1, 0), (1, 0, 1)))
    print(triangle2.tdim)
    print(triangle2.gdim)

::

    2
    2
    2
    3

The reference types of the subentities can be obtained. This can be used to create a reference
representing a subentity:

.. code-block:: python

    import symfem

    triangle = symfem.create_reference("triangle")
    print(triangle.sub_entity_types)

    vertices = []
    for i in triangle.edges[0]:
        vertices.append(triangle.vertices[i])
    edge0 = symfem.create_reference(
        triangle.sub_entity_types[1], vertices)
    print(edge0.vertices)

::

    ['point', 'interval', 'triangle', None]
    ((1, 0), (0, 1))

Documentation index
===================
.. toctree::
   :titlesonly:

   docs/index


