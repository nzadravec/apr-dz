import numpy as np
from matrix import Matrix
from fminbnd import search

def grad_desc(f, grad_f, x0, eps = 1e-6, opt_step = False):

    n = len(x0)
    x = np.array(x0, dtype=float)
    
    minf = None
    count = 0
    while(True):
    
        if count >= 100:
            print('there was a divergence')
            break

        grad = np.array(grad_f(x))
        if np.linalg.norm(grad) < eps:
            break
        
        v = -1 * grad
        l_min = 1
        if opt_step:
            v = v  / np.linalg.norm(v)
            l_min = search(lambda l : f(x +l*v), 1)

        x = x + l_min * v
        
        fx = f(x)
        if minf == None:
            minf = fx
        elif fx < minf:
            minf = fx
            count = 0
        else:
            count+=1

    return x

def newton(f, grad_f, hess_f, x0, eps = 1e-6, opt_step = False):

    n = len(x0)
    x = np.array(x0, dtype=float)

    minf = None
    count = 0
    while(True):
    
        if count >= 100:
            print('there was a divergence')
            break
    
        grad = np.array(grad_f(x))
        hess_inv = np.array(Matrix(n, n, hess_f(x)).inv().data)
        v = -1 * np.dot(hess_inv, grad)
        l_min = 1
        if opt_step:
            v = v  / np.linalg.norm(v)
            l_min = search(lambda l : f(x +l*v), 1)

        if np.linalg.norm(l_min * v) < eps:
            break

        x = x + l_min * v
            
        fx = f(x)
        if minf == None:
            minf = fx
        elif fx < minf:
            minf = fx
            count = 0
        else:
            count+=1
            
    return x




