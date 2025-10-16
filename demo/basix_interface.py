"""Demo showing how Symfem can be used to create Basix elements.

There are many elements that are included in Symfem but not in Basix, so they
cannot immediately be used to solve problems in FEniCSx.

This demo shows how Symfem can be used to create Basix custom elements, and so
allows for almost any Symfem element to be used with FEniCSx.
"""

import symfem
import numpy as np

# As Basix is an optional dependency of Symfem, the basix interface must be
# imported separately
import symfem.basix_interface

# Create a BDFM element. These are not currently implemented in Basix
element = symfem.create_element("triangle", "BDFM", 2)

# A Basix element can be created from the Symfem BDFM using the
# create_basix_element function.
basix_element = symfem.basix_interface.create_basix_element(element)

# If you want to use the Basix element with DOLFINx, you will most likely
# want to create a basix.ufl element. You can do this by setting the ufl
# optional argument to True.
basix_ufl_element = symfem.basix_interface.create_basix_element(element, ufl=True)

# The create_basix_element function can also take an optional dtype argument
basix_f32_element = symfem.basix_interface.create_basix_element(element, dtype=np.float32)

# Creating Symfem element can be slow (as some symbolic computing is often
# done during the creation), so in many cases it may not be desirable to
# create the Symfem element every time you run your FEniCSx script.
#
# The function generate_basix_element_code can be used to generate code
# that creates a custom element that you can paste in your FEniCSx script.
code = symfem.basix_interface.generate_basix_element_code(element)

# This code can then be printed or written to a file.
print(code)
with open("_bdfm_element.py", "w") as f:
    f.write(code)

# Similar to the create_basix_element function, generate_basix_element_code
# can take optional dtype and ufl arguments. It can additionally take a
# variable_name argument to set the name of the variable used for the element
# in the generated code.
with open("_bdfm_element2.py", "w") as f:
    f.write(
        symfem.basix_interface.generate_basix_element_code(element, variable_name="bdfm_element")
    )

# If you are using the DOLFINx C++ interface, you may want to create a Basix
# custom element via Basix's C++ interface. You can generate code for this
# by setting the language argument to "c++"
with open("_bdfm_element.cpp", "w") as f:
    f.write(symfem.basix_interface.generate_basix_element_code(element, language="c++"))
