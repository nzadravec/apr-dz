from math import sin, sqrt
import numpy as np

class Func:

    def __init__(self, f):
        self.f = f
        self.count = 0

    def valueAt(self, x):
        self.count+=1
        return self.f(x)

    def reset(self):
        self.count = 0

def f1(x):
    x1, x2 = x[0], x[1]
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

def gf1(x):
    x1, x2 = x[0], x[1]
    grad = list()
    grad.append(-400 * x1 * (x2 - x1**2) - 2 * (1 - x1))
    grad.append(200 * (x2 - x1**2))
    return grad
	
def hf1(x):
    x1, x2 = x[0], x[1]
    hess = list()
    hess.append([1200 * x1**2 - 400 * x2 + 2, -400 * x1])
    hess.append([-400 * x1, 200])
    return hess
	
def f1g1(x):
    x1, x2 = x[0], x[1]
    return x2-x1
    
def f1g2(x):
    x1 = x[0]
    return 2-x1

def f2(x):
    x1, x2 = x[0], x[1]
    return (x1 - 4)**2 + 4 * (x2 - 2)**2

def gf2(x):
    x1, x2 = x[0], x[1]
    grad = list()
    grad.append(2 * x1 - 8)
    grad.append(8 * x2 - 16)
    return grad
	
def hf2(x):
    x1, x2 = x[0], x[1]
    hess = list()
    hess.append([2, 0])
    hess.append([0, 8])
    return hess

def f3(x):
    x1, x2 = x[0], x[1]
    return (x1 - 2)**2 + (x2 + 3)**2

def gf3(x):
    x1, x2 = x[0], x[1]
    grad = list()
    grad.append(2 * x1 - 4)
    grad.append(2 * x2 + 6)
    return grad
	
def hf3(x):
    return [[2, 0], [0, 2]]

def f4(x):
    x1, x2 = x[0], x[1]
    return (x1 - 3)**2 + (x2)**2

def gf4(x):
    x1, x2 = x[0], x[1]
    return [2 * x1 - 6, 2 * x2]

def hf4(x):
    return [[2, 0], [0, 2]]
    
def f4g1(x):
    x1, x2 = x[0], x[1]
    return 3 - x1 - x2
    
def f4g2(x):
    x1, x2 = x[0], x[1]
    return 3 + 1.5 * x1 - x2
    
def f4h1(x):
    x2 = x[1]
    return x2 - 1



