"""Values for Guzman-Neilan element."""

import sympy

bubbles = [
    {
        ((0, 0), (1, 0), (sympy.S('1/3'), sympy.S('1/3'))): (
            sympy.S('-3*x*y+9*y**2/2-2*y'),
            sympy.S('3*y**2/2-2*y')),
        ((0, 0), (0, 1), (sympy.S('1/3'), sympy.S('1/3'))): (
            sympy.S('3*x**2/2-2*x'),
            sympy.S('9*x**2/2-3*x*y-2*x')),
        ((1, 0), (0, 1), (sympy.S('1/3'), sympy.S('1/3'))): (
            sympy.S('3*x**2/2-3*x*y-x-3*y**2/2+2*y-1/2'),
            sympy.S('-3*x**2/2-3*x*y+2*x+3*y**2/2-y-1/2')),
    },
    {
        ((0, 0), (1, 0), (sympy.S('1/3'), sympy.S('1/3'))): (
            sympy.S('3*y**2-4*y'),
            sympy.S('2*y')),
        ((0, 0), (0, 1), (sympy.S('1/3'), sympy.S('1/3'))): (
            sympy.S('-3*x**2+2*x+6*y**2-6*y'),
            sympy.S('-6*x**2+6*x*y+2*x')),
        ((1, 0), (0, 1), (sympy.S('1/3'), sympy.S('1/3'))): (
            sympy.S('9*x**2+24*x*y-14*x+15*y**2-20*y+5'),
            sympy.S('-6*x**2-18*x*y+10*x-12*y**2+16*y-4')),
    },
    {
        ((0, 0), (1, 0), (sympy.S('1/3'), sympy.S('1/3'))): (
            sympy.S('-6*x*y+6*y**2-2*y'),
            sympy.S('-6*x**2+6*x+3*y**2-2*y')),
        ((0, 0), (0, 1), (sympy.S('1/3'), sympy.S('1/3'))): (
            sympy.S('-2*x'),
            sympy.S('-3*x**2+4*x')),
        ((1, 0), (0, 1), (sympy.S('1/3'), sympy.S('1/3'))): (
            sympy.S('12*x**2+18*x*y-16*x+6*y**2-10*y+4'),
            sympy.S('-15*x**2-24*x*y+20*x-9*y**2+14*y-5')),
    },
]
