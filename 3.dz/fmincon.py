
from random import random
from functools import partial
import numpy as np

def box(f, xd, xg, g, x0, e = None, a = 1.3):
    
    n = len(x0)
    
    if not all([xd[i]<=x0[i]<=xg[i] for i in range(n)]):
        print('x0 does not satisfy explicit restrictions')
        return None
        
    if not all([g[i](x0)>=0 for i in range(len(g))]):
        print('x0 does not satisfy implicit restrictions')
        return None
    
    if e == None:
        e = np.array([1e-6]*n, dtype=float)
    
    xc = np.array(x0, dtype=float)
    x = np.empty([2*n, n], dtype=float)
    
    for t in range(2*n):
        for i in range(n):
            x[t,i] = xd[i] + random()*(xg[i] - xd[i])
            
        count2 = 0
        while not all([g[i](x[t])>=0 for i in range(len(g))]):
            count2 += 1
            if count2 >= 100:
                print('there was a divergence')
                break
            x[t] = (1/2)*(x[t] + xc)
        
        #xc = (xc * (t+1) + x[t]) / (t+2)
        xc = np.empty(n, dtype=float)
        for i in range(t):
            xc += x[i]
        xc /= t+2
        
    fx = np.array([f(x[i]) for i in range(2*n)], dtype=float)
    
    minf = None
    count = 0
    while True:
    
        if count >= 100:
            print('there was a divergence')
            break
    
        h = np.argmax(fx)
        l = np.argmin(fx)
        flag = True
        for i in range(2*n):
            if flag and i != h:
                h2 = i
                flag = False
                
            elif i != h and fx[h2] < fx[i]:
                h2 = i
        
        #xc = (xc * (2*n + 1) - x[h]) / (2*n)
        
        xc = np.empty(n, dtype=float)
        for i in range(2*n):
            if i != h:
                xc += x[i]
        xc /= 2*n
        
        xr = (1 + a)*xc - a*x[h]
        
        for i in range(n):
            if xr[i] < xd[i]:
                xr[i] = xd[i]
            elif xr[i] > xg[i]:
                xr[i] = xg[i]
            
        count2 = 0   
        while not all([g[i](xr)>=0 for i in range(len(g))]):
            count2 += 1
            if count2 >= 100:
                print('there was a divergence')
                break
            xr = (1/2)*(xr + xc)
          
        fr = f(xr)
        if fr > fx[h2]:
        #if fr > fx[h]:
            xr = (1/2)*(xr + xc)
            fr = f(xr)
            
        x[h] = xr
        fx[h] = fr
        
        #print(x[np.argmin(fx)], np.min(fx))
        
        if any([abs(x[h][i]-xc[i])<e[i] for i in range(n)]):
            break;
            
        fx2 = f(xc)
        if minf == None:
            minf = fx2
        elif fx2 < minf:
            minf = fx2
            count = 0
        else:
            count+=1
        
    #return xc
    return x[l]

def transf_problem(f, x0, alg, hs, gs, t = 1, e=None):
    
    n = len(x0)
    
    def G(t, x):
        sum2 = 0
        for i, g in enumerate(gs):
            if g(x)>=0:
                pass
                #sum2 -= (1/t) * np.log(g(x))
            else:
                sum2 -= t[i] * g(x)
                
        return sum2
                
    x = alg(partial(G, [10]*len(gs)), x0)
    #print(x)
    if not all([g(x)>=0 for g in gs]):
        print('x0 does not satisfy inequality restrictions')
        return None
    
    prev_x = np.array(x0, dtype=float)
    
    if e == None:
        e = np.array([1e-6]*n, dtype=float)
        
    def U(t, x):
        sum2 = f(x)
        
        for g in gs:
            if g(x)>=0:
                pass
                #sum2 -= (1/t) * np.log(g(x))
            else:
                sum2 += 1e6
                
        for h in hs:
            sum2 += t * h(x)**2
            
        return sum2
         
    minf = None
    count = 0
    while(True):
    
        if count >= 100:
            print('there was a divergence')
            break
    
        curr_x = alg(partial(U, t), prev_x)
        if all([abs(curr_x[i]-prev_x[i])<e[i] for i in range(n)]):
            break;
        
        fx = U(t, curr_x)
        #print(curr_x, fx)
        t *= 10
        
        if minf == None:
            minf = fx
        elif fx < minf:
            minf = fx
            count = 0
        else:
            count+=1
        
        
    return curr_x
    
