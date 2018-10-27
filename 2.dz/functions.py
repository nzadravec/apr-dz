from math import sin, sqrt

class Func:

    def __init__(self, f):
        self.f = f
        self.counter = 0

    def valueAt(self, x):
        self.counter+=1
        return self.f(x)

    def reset(self):
        self.counter = 0

def f1(x):
    x1, x2 = x[0], x[1]
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

def f2(x):
    x1, x2 = x[0], x[1]
    return (x1 - 4)**2 + 4 * (x2 - 2)**2

def f3(x):
    x_type = type(x)
    if x_type == float or x_type == int:
        return (x - 1)**2
    
    value = 0
    for i in range(len(x)):
        value += (x[i] - (i+1))**2
    return value

def f4(x):
    x1, x2 = x[0], x[1]
    return abs((x1 - x2)*(x1 + x2)) + sqrt(x1**2 + x2**2)

def f6(x):
    x_type = type(x)
    
    if x_type == float or x_type == int:
        sumOfSquares = x**2
    else:
        sumOfSquares = 0
        for i in range(len(x)):
            sumOfSquares += (x[i])**2

    numerator = (sin(sqrt(sumOfSquares)))**2-0.5
    denominator = (1+0.001*sumOfSquares)**2

    return 0.5 + numerator / denominator

    
