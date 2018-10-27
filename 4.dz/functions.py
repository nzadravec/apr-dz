from math import sin, cos, sqrt, exp
import numpy as np

class Func:

    def __init__(self, f):
        self.f = f
        self.count = 0

    def valueAt(self, x):
        self.count+=1
        return self.f(x)
        
class NegativeMeanSquareError:
    
    def __init__(self, predict, samples):
        self.predict = predict
        self.samples = samples
    
    def calcFor(self, x):
        squareError = 0
        for s in self.samples:
            squareError += pow(s[-1] - self.predict(s[:-1], x), 2)
        
        return - squareError / len(self.samples)

def predict(inputs, params):
    x, y = inputs
    b0, b1, b2, b3, b4 = params
    return sin(b0 + b1*x) + b2*cos(x*(b3 + y))*(1/(1 + exp(pow(x - b4, 2))))
 
def loadSamplesFrom(fileName):
    file = open("lib/"+fileName, "r")
    lines = file.readlines()
    file.close()
    
    samples = list()
    for line in lines:
        samples.append(np.fromstring(line, dtype=float, sep='\t'))
    
    return samples

def f1(x):
    x1, x2 = x
    return -(100 * (x2 - x1**2)**2 + (1 - x1)**2)

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
    
def f7(x):
    x_type = type(x)
    
    if x_type == float or x_type == int:
        sumOfSquares = x**2
    else:
        sumOfSquares = 0
        for i in range(len(x)):
            sumOfSquares += (x[i])**2
            
    return pow(sumOfSquares, 0.25)*(
        1 + (sin(50*pow(sumOfSquares, 0.1))**2))
