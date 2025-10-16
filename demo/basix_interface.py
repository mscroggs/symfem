"""Demo showing how Symfem can be used to create Basix elements.

There are many elements that are included in Symfem but not in Basix, so they cannot immediately be used to solve problems in FEniCSx.

This demo shows how Symfem can be used to create Basix custom elements, and so allows for almost any Symfem element to be used with FEniCSx.
"""

import sympy

import symfem

# As Basix is an optional dependency of Symfem, the basix interface must be imported separately
import symfem.basix_interface

# Create a BDFM element. These are not currently implemented in Basix
element = symfem.create_element("triangle", "BDFM", 2)

# A Basix element can be created from the Symfem BDFM using the create_basix_element function.
basix_element = symfem.basix_interface.create_basix_element(element)

# If you want to use the Basix element
basix_element = symfem.basix_interface.create_basix_element(element, ufl=True)
