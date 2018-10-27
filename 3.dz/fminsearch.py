from math import sqrt
from fminbnd import search
import numpy as np


def explore(f, xp, dx):
    n = len(xp)
    x = xp.copy()
    for i in range(n):
        P = f(x)
        x[i] = x[i] + dx
        N = f(x)
        if N > P:
            x[i] -= 2*dx
            N = f(x)
            if N > P:
                x[i] += dx

    return x

def hj(f, x0, dx=0.5, eps=None, trace=False):

    n = len(x0)
    if eps == None:
        eps = np.array([1e-6]*n, dtype=float)
    
    xp = np.array(x0, dtype=float)
    xb = np.array(x0, dtype=float)

    flag = True
    while flag:
        xn = explore(f, xp, dx)

        if trace:
            print(xb, '-' ,xp, '-',xn)
            print("{:4.2f} - {:4.2f} - {:4.2f}".format(f(xb), f(xp), f(xn)))
            
        if f(xn) < f(xb):
            xp = 2*xn - xb
            xb = xn

        else:
            dx = dx / 2
            xp = xb

        if all(dx < eps):
            flag = False

    return xb
        
    

def simplex(f, x0, eps=1e-6, shift=1, alpha=1, beta=0.5, gamma=2, sigma=0.5, trace=False):
    
    reflection = lambda xc, xh : (1 + alpha)*xc - alpha*xh
    expansion = lambda xc, xr : (1 - gamma)*xc + gamma*xr
    contraction = lambda xc, xh : (1 - beta)*xc + beta*xh

    n = len(x0)
    x = np.array([[0]*n]*(n+1), dtype=float)
    x[0] = np.array(x0, dtype=float)
    for i in range(1, n+1):
        x[i] = np.array(x0, dtype=float)
        x[i][i-1] += shift

    fx = np.array([f(x[i]) for i in range(n+1)], dtype=float)

    flag = True
    while flag:

        h = np.argmax(fx)
        l = np.argmin(fx)

        xc = np.array([0]*n, dtype=float)
        for i in range(n+1):
            if i != h:
                xc += x[i]
        xc /= n

        fc = f(xc)

        if trace:
            print(xc, fc)

        xr = reflection(xc, x[h])
        fr = f(xr)
        if fr < fx[l]:
            xe = expansion(xc, xr)
            fe = f(xe)
            if fe < fx[l]:
                x[h] = xe
                fx[h] = fe
            else:
                x[h] = xr
                fx[h] = fr

        else:
            vs = np.array([fx[j] for j in range(n+1) if j!=h], dtype=float)
            if all(fr > vs):
                if fr < fx[h]:
                    x[h] = xr
                    fx[h] = fr
                xk = contraction(xc, x[h])
                fk = f(xk)
                if fk < fx[h]:
                    x[h] = xk
                    fx[h] = fk
                else:
                    for i in range(n+1):
                        if i != l:
                            #x[i] = x[l] + sigma*(x[i] - x[l])
                            x[i] = sigma*(x[l] + x[i])
                            fx[i] = f(x[i])

            else:
                x[h] = xr
                fx[h] = fr

        v = 0
        for i in range(n+1):
            v += (fx[i]-fc)**2

        v /= n
        v = sqrt(v)

        if v <= eps:
            flag = False

    return x[l]

def coordinateSearch(f, x0, eps=None, trace=False):
    n = len(x0)
    if eps == None:
        eps = np.array([1e-6]*n, dtype=float)
    
    x = np.array(x0, dtype=float)
    ei = np.array([0]*n, dtype=float)

    flag = True
    while flag:
        xs = x.copy()
        for i in range(n):
            ei[i] = 1
            l_min = search(lambda l : f(x +l*ei), xs[i], eps[i])
            x = x + l_min*ei
            ei[i] = 0

        if all(abs(x - xs) <= eps):
            flag = False
    
    return x
