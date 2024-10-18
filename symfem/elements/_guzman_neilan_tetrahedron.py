"""Values for Guzman-Neilan element."""

import sympy

lambda_0 = {
    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (sympy.S("1/4"), sympy.S("1/4"), sympy.S("1/4"))): sympy.S(
        "4*z"
    ),
    ((0, 0, 0), (1, 0, 0), (0, 0, 1), (sympy.S("1/4"), sympy.S("1/4"), sympy.S("1/4"))): sympy.S(
        "4*y"
    ),
    ((0, 0, 0), (0, 1, 0), (0, 0, 1), (sympy.S("1/4"), sympy.S("1/4"), sympy.S("1/4"))): sympy.S(
        "4*x"
    ),
    ((1, 0, 0), (0, 1, 0), (0, 0, 1), (sympy.S("1/4"), sympy.S("1/4"), sympy.S("1/4"))): sympy.S(
        "-4*x - 4*y - 4*z + 4"
    ),
}
bubbles = [
    {
        1: (
            sympy.S("-10*x**2 - 20*x*y - 20*x*z + 16*x + 10*y*z - 3/2"),
            sympy.S("-20*x*y + 10*x*z - 10*y**2 - 20*y*z + 16*y - 3/2"),
            sympy.S("10*x*y - 20*x*z - 20*y*z - 10*z**2 + 16*z - 3/2"),
        ),
        2: (
            sympy.S("-x/2 + 45*y/4 + 45*z/4 - 8"),
            sympy.S("45*x/4 - y/2 + 45*z/4 - 8"),
            sympy.S("45*x/4 + 45*y/4 - z/2 - 8"),
        ),
        3: (sympy.S("5/3"), sympy.S("5/3"), sympy.S("5/3")),
    },
    {
        1: (
            sympy.S("15*x*y + 15*x*z - 6*x - 30*y*z + 3/2"),
            sympy.S("15*y**2 + 45*y*z - 21*y + 3/2"),
            sympy.S("45*y*z + 15*z**2 - 21*z + 3/2"),
        ),
        2: (
            sympy.S("-123*x/4 - 105*y/8 - 105*z/8 + 27/4"),
            sympy.S("129*y/8 - 165*z/8 + 69/8"),
            sympy.S("-165*y/8 + 129*z/8 + 69/8"),
        ),
        3: (sympy.S("5"), sympy.S("-5"), sympy.S("-5")),
    },
    {
        1: (
            sympy.S("-15*x**2 - 45*x*z + 21*x - 3/2"),
            sympy.S("-15*x*y + 30*x*z - 15*y*z + 6*y - 3/2"),
            sympy.S("-45*x*z - 15*z**2 + 21*z - 3/2"),
        ),
        2: (
            sympy.S("-129*x/8 + 165*z/8 - 69/8"),
            sympy.S("105*x/8 + 123*y/4 + 105*z/8 - 27/4"),
            sympy.S("165*x/8 - 129*z/8 - 69/8"),
        ),
        3: (sympy.S("5"), sympy.S("-5"), sympy.S("5")),
    },
    {
        1: (
            sympy.S("15*x**2 + 45*x*y - 21*x + 3/2"),
            sympy.S("45*x*y + 15*y**2 - 21*y + 3/2"),
            sympy.S("-30*x*y + 15*x*z + 15*y*z - 6*z + 3/2"),
        ),
        2: (
            sympy.S("129*x/8 - 165*y/8 + 69/8"),
            sympy.S("-165*x/8 + 129*y/8 + 69/8"),
            sympy.S("-105*x/8 - 105*y/8 - 123*z/4 + 27/4"),
        ),
        3: (sympy.S("-5"), sympy.S("-5"), sympy.S("5")),
    },
]
