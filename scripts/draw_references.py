import os
import symfem

for shape in ["interval", "triangle", "tetrahedron",
              "quadrilateral", "hexahedron", "prism", "pyramid",
              "dual polygon"]:
    if shape == "dual polygon":
        ref = symfem.create_reference("dual polygon(6)")
    else:
        ref = symfem.create_reference(shape)

    yadd = 0
    width = 100
    if shape == "dual polygon":
        yadd = 60
        width = 160

        def to_x(p):
            return float(85 + 85 * p[0])

        def to_y(p):
            return float(35 - 85 * p[1])

    elif ref.gdim == 1:
        yadd = -100

        def to_x(p):
            return float(100 * p[0])

        def to_y(p):
            return float(120)

    elif ref.gdim == 2:
        def to_x(p):
            return float(100 * p[0])

        def to_y(p):
            return float(120 - 100 * p[1])

    elif ref.gdim == 3:
        if shape == "hexahedron":
            yadd = 40
            width = 140
        elif shape == "pyramid":
            width = 140
        elif shape == "prism":
            yadd = 40

        def to_x(p):
            return float(100 * p[0] + 30 * p[1])

        def to_y(p):
            return float(120 - 100 * p[2] - 40 * p[1])

    svg = ""

    if ref.gdim == 1:
        svg += (f"<line x1='20' y1='{120 + yadd}' x2='50' y2='{120 + yadd}' />\n"
                f"<line x1='40' y1='{115 + yadd}' x2='50' y2='{120 + yadd}' />\n"
                f"<line x1='40' y1='{125 + yadd}' x2='50' y2='{120 + yadd}' />\n"
                f"<text x='60' y='{120 + yadd}' fill='#000000' dy='.3em'"
                " style='font-family:\"Libertinus Serif Semibold Italic\";font-size:20px'>x</text>"
                "\n")
    elif ref.gdim == 2:
        svg += (f"<line x1='20' y1='{120 + yadd}' x2='20' y2='{90 + yadd}' />\n"
                f"<line x1='15' y1='{100 + yadd}' x2='20' y2='{90 + yadd}' />\n"
                f"<line x1='25' y1='{100 + yadd}' x2='20' y2='{90 + yadd}' />\n"
                f"<line x1='20' y1='{120 + yadd}' x2='50' y2='{120 + yadd}' />\n"
                f"<line x1='40' y1='{115 + yadd}' x2='50' y2='{120 + yadd}' />\n"
                f"<line x1='40' y1='{125 + yadd}' x2='50' y2='{120 + yadd}' />\n"
                f"<text x='60' y='{120 + yadd}' fill='#000000' dy='.3em'"
                " style='font-family:\"Libertinus Serif Semibold Italic\";font-size:20px'>x</text>"
                "\n"
                f"<text x='20' y='{75 + yadd}' fill='#000000' dy='.3em'"
                " style='font-family:\"Libertinus Serif Semibold Italic\";font-size:20px'>y</text>"
                "\n")
    elif ref.gdim == 3:
        svg += (f"<line x1='20' y1='{120 + yadd}' x2='20' y2='{90 + yadd}' />\n"
                f"<line x1='15' y1='{100 + yadd}' x2='20' y2='{90 + yadd}' />\n"
                f"<line x1='25' y1='{100 + yadd}' x2='20' y2='{90 + yadd}' />\n"
                f"<line x1='20' y1='{120 + yadd}' x2='50' y2='{120 + yadd}' />\n"
                f"<line x1='40' y1='{115 + yadd}' x2='50' y2='{120 + yadd}' />\n"
                f"<line x1='40' y1='{125 + yadd}' x2='50' y2='{120 + yadd}' />\n"
                f"<line x1='20' y1='{120 + yadd}' x2='44' y2='{102 + yadd}' />\n"
                f"<line x1='33' y1='{104 + yadd}' x2='44' y2='{102 + yadd}' />\n"
                f"<line x1='39' y1='{112 + yadd}' x2='44' y2='{102 + yadd}' />\n"
                f"<text x='60' y='{120 + yadd}' fill='#000000' dy='.3em'"
                " style='font-family:\"Libertinus Serif Semibold Italic\";font-size:20px'>x</text>"
                "\n"
                f"<text x='52' y='{91 + yadd}' fill='#000000' dy='.3em'"
                " style='font-family:\"Libertinus Serif Semibold Italic\";font-size:20px'>y</text>"
                "\n"
                f"<text x='20' y='{75 + yadd}' fill='#000000' dy='.3em'"
                " style='font-family:\"Libertinus Serif Semibold Italic\";font-size:20px'>z</text>"
                "\n")

    xpos = 100
    for dim in range(ref.gdim + 1):
        es = ref.sub_entities(dim)
        lines = []
        for e in ref.sub_entities(1):
            lines.append(f"<line x1='{xpos + to_x(ref.vertices[e[0]])}'"
                         f" y1='{yadd + to_y(ref.vertices[e[0]])}'"
                         f" x2='{xpos + to_x(ref.vertices[e[1]])}'"
                         f" y2='{yadd + to_y(ref.vertices[e[1]])}' />\n")

        entities = []
        for n_e, e in enumerate(es):
            points = [ref.vertices[i] for i in e]
            mid = ref.sub_entity(dim, n_e).midpoint()
            e_svg = f"<circle cx='{xpos + to_x(mid)}' cy='{yadd + to_y(mid)}' r='12px' />"
            e_svg += f"<text x='{xpos + to_x(mid)}' y='{yadd + to_y(mid)}' dy='.3em'"
            if n_e >= 10:
                e_svg += " style='font-size:10px'"
            e_svg += f">{n_e}</text>\n"
            entities.append(e_svg)

        if shape == "hexahedron" and dim == 1:
            svg += "".join([lines[i] for i in [5, 6, 7, 11]])
            svg += "".join([entities[i] for i in [5, 6, 7, 11]])
            svg += "".join([lines[i] for i in [1, 3, 9, 10]])
            svg += "".join([entities[i] for i in [1, 3, 9, 10]])
            svg += "".join([lines[i] for i in [0, 2, 4, 8]])
            svg += "".join([entities[i] for i in [0, 2, 4, 8]])
        elif shape == "hexahedron" and dim == 2:
            svg += "".join([lines[i] for i in [5, 6, 7, 11]])
            svg += "".join([entities[i] for i in [4]])
            svg += "".join([lines[i] for i in [1, 3, 9, 10]])
            svg += "".join([entities[i] for i in [0, 2, 3, 5]])
            svg += "".join([lines[i] for i in [0, 2, 4, 8]])
            svg += "".join([entities[i] for i in [1]])
        elif shape == "tetrahedron" and dim == 1:
            svg += "".join([lines[i] for i in [0, 2, 4]])
            svg += "".join([entities[i] for i in [0, 2, 4]])
            svg += "".join([lines[i] for i in [1, 3, 5]])
            svg += "".join([entities[i] for i in [1, 3, 5]])
        elif shape == "tetrahedron" and dim == 2:
            svg += "".join([lines[i] for i in [0, 2, 4]])
            svg += "".join([entities[i] for i in [0, 1, 3]])
            svg += "".join([lines[i] for i in [1, 3, 5]])
            svg += "".join([entities[i] for i in [2]])
        elif shape == "prism" and dim == 1:
            svg += lines[5]
            svg += entities[5]
            svg += "".join([lines[i] for i in [1, 2, 7, 8]])
            svg += "".join([entities[i] for i in [1, 2, 7, 8]])
            svg += "".join([lines[i] for i in [0, 3, 4, 6]])
            svg += "".join([entities[i] for i in [0, 3, 4, 6]])
        elif shape == "pyramid" and dim == 1:
            svg += lines[2]
            svg += entities[2]
            svg += "".join([lines[i] for i in [1, 3, 6, 7]])
            svg += "".join([entities[i] for i in [1, 3, 6, 7]])
            svg += "".join([lines[i] for i in [0, 4, 5]])
            svg += "".join([entities[i] for i in [0, 4, 5]])
        elif shape == "pyramid" and dim == 2:
            svg += lines[5]
            svg += entities[3]
            svg += "".join([lines[i] for i in [1, 3, 6, 7]])
            svg += "".join([entities[i] for i in [0, 2, 4]])
            svg += "".join([lines[i] for i in [0, 4, 2]])
            svg += "".join([entities[i] for i in [1]])
        elif shape == "pyramid" and dim == 3:
            svg += "".join([lines[i] for i in [1, 3, 5, 6, 7]])
            svg += entities[0]
            svg += "".join([lines[i] for i in [0, 2, 4]])
        else:
            svg += "\n".join(lines)
            svg += "\n".join(entities)

        xpos += width + 70

    filename = shape.replace(" ", "_") + "_numbering"

    with open(f"../img/{filename}.svg", "w") as f:
        f.write(f"<svg width='{xpos - 50}' height='{140 + yadd}'>\n")
        f.write("<style type=\"text/css\"><![CDATA[\n"
                "  text { text-anchor:middle; font-size:15px; font-family: \"Lato Bold\" }\n"
                "  line { stroke-width: 4px; stroke-linecap: round; stroke: #000000 }\n"
                "  circle { fill: #FFFFFF; stroke: #000000; stroke-width:4px}\n"
                "  rect { fill: #FFFFFF; stroke: #FFFFFF; stroke-width:4px}\n"
                "]]></style>\n")
        f.write(f"<rect x='0' y='0' width='{xpos - 50}' height='{140 + yadd}' />\n")

        f.write(svg)
        f.write("</svg>")
    os.system(f"convert ../img/{filename}.svg ../img/{filename}.png")
