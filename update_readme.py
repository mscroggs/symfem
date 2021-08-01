"""
This script updated the list of available elements in README.md.
"""

import symfem

cells = ["interval", "triangle", "quadrilateral", "tetrahedron",
         "hexahedron", "prism", "pyramid", "dual polygon"]
elementlist = {i: [] for i in cells}

for e in symfem.create._elementlist:
    for r in e.references:
        elementlist[r].append(e.names[0])

for j in elementlist.values():
    j.sort(key=lambda x: x.lower())

with open("README.md") as f:
    pre = f.read().split("## List of supported elements")[0]

with open("README.md", "w") as f:
    f.write(pre)
    f.write("## List of supported elements\n")
    for cell in cells:
        f.write(f"### {cell[0].upper() + cell[1:]}\n")
        f.write("\n".join([f"- {i}" for i in elementlist[cell]]))
        f.write("\n\n")
