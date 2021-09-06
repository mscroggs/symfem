"""
This script updated the list of available elements in README.md.
"""

import symfem

cells = ["interval", "triangle", "quadrilateral", "tetrahedron",
         "hexahedron", "prism", "pyramid", "dual polygon"]
elementlist = {i: [] for i in cells}

for e in symfem.create._elementlist:
    name = e.names[0]
    if len(e.names) > 1:
        name += " (alternative names: " + ", ".join(e.names[1:]) + ")"
    for r in e.references:
        elementlist[r].append(name)

for j in elementlist.values():
    j.sort(key=lambda x: x.lower())

with open("README.md") as f:
    pre = f.read().split("## Reference cells")[0]

with open("README.md", "w") as f:
    f.write(pre)
    f.write("## Reference cells\n\n")
    for cell in cells:
        f.write(f"### {cell[0].upper()}{cell[1:]}\n")
        if cell == "dual polygon":
            f.write(f"The reference {cell} (hexagon) has vertices ")
            ref = symfem.create_reference("dual polygon(6)")
        else:
            f.write(f"The reference {cell} has vertices ")
            ref = symfem.create_reference(cell)
        str_v = [f"{v}" for v in ref.vertices]
        if len(ref.vertices) <= 2:
            f.write(" and ".join(str_v))
        else:
            f.write(", and ".join([", ".join(str_v[:-1]), str_v[-1]]))
        f.write(". Its sub-entities are numbered as follows.\n\n")
        f.write(f"![The numbering of a reference {cell}]"
                f"(img/{cell.replace(' ', '_')}_numbering.png)\n\n")
    f.write("## List of supported elements\n")
    for cell in cells:
        f.write(f"### {cell[0].upper() + cell[1:]}\n")
        f.write("\n".join([f"- {i}" for i in elementlist[cell]]))
        f.write("\n\n")
