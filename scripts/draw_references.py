"""Draw reference cells for the README."""

import os
import sys

import symfem

TESTING = "test" in sys.argv

if TESTING:
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../_temp")
else:
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../img")

for shape in ["interval", "triangle", "tetrahedron",
              "quadrilateral", "hexahedron", "prism", "pyramid",
              "dual polygon"]:
    if shape == "dual polygon":
        ref = symfem.create_reference("dual polygon(6)")
    else:
        ref = symfem.create_reference(shape)

    filename = shape.replace(" ", "_") + "_numbering"
    ref.plot_entity_diagrams(
        f"{folder}/{filename}.png", {"png_width": 230 + 200 * ref.tdim})
