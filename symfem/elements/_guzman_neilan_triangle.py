"""Values for Guzman-Neilan element."""

import sympy

lambda_0 = {
    ((0, 0), (1, 0), (sympy.S("1/3"), sympy.S("1/3"))): sympy.S("3*y"),
    ((0, 0), (0, 1), (sympy.S("1/3"), sympy.S("1/3"))): sympy.S("3*x"),
    ((1, 0), (0, 1), (sympy.S("1/3"), sympy.S("1/3"))): sympy.S("-3*x - 3*y + 3"),
}
bubbles = [
    {
        1: (sympy.S("2/3 - y"), sympy.S("2/3 - x")),
        2: (sympy.S("-1/6"), sympy.S("-1/6")),
    },
    {
        1: (sympy.S("2*x + 2*y - 2/3"), sympy.S("-2*y - 2/3")),
        2: (sympy.S("-1/3"), sympy.S("2/3")),
    },
    {
        1: (sympy.S("2*x + 2/3"), sympy.S("-2*x - 2*y + 2/3")),
        2: (sympy.S("-2/3"), sympy.S("1/3")),
    },
]
