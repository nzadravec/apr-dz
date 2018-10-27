
from math import sqrt
import numpy as np
from matrix import Matrix

# Golden ratio algorithm
# f - target function
# a, b - initial boundaries of unimodal interval
# e - precision
# 
def goldenRatio(f, a, b, eps, trace=False):
    k =  0.5 * (sqrt(5) - 1)

    if trace:
        print("  a  | f(a) |   c  | f(c) |   d  | f(d) |   b  | f(b)")
    
    c = b - k * (b - a)
    d = a + k * (b - a)
    fc = f(c)
    fd = f(d)
    while (b - a) > eps:
        if fc < fd:
            b = d
            d = c
            c = b - k * (b - a)
            fd = fc
            fc = f(c)

        else:
            a = c
            c = d
            d = a + k * (b - a)
            fc = fd
            fd = f(d)

        if trace:
            print("%4.2f | %4.2f | %4.2f | %4.2f | %4.2f | %4.2f | %4.2f | %4.2f" %
                  (a, f(a), c, f(c), d, f(d), b, f(b)))
        
    return (a + b)/2

# The procedure for seeking a unimodal interval
# point - the starting point of the search
# h - search shift
# f - target function
# return - unimodal interval [l, r]
def unimodal(f, h, point):
    
    l = point - h; r = point + h
    m = point
    step = 1

    fm = f(point)
    fl = f(l)
    fr = f(r)

    if fm < fr and fm < fl:
        return l, r
    elif fm > fr:
        while fm > fr:
            l = m
            m = r
            fm = fr
            step *= 2
            r = point + h * step
            fr = f(r)
    else:
        while fm > fl:
            r = m
            m = l
            fm = fl
            step *= 2
            l = point - h * step
            fl = f(l)

    return l, r

def search(f, point, eps = 1e-6, trace=False):
    a, b = unimodal(f, 1, point)
    return goldenRatio(f, a, b, eps, trace)
