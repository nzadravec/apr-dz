import numpy as np

def runge_kutta_4(x0, A, B, T, t_max, trace=False, step=1):
    def f(x):
        return np.dot(A, x) + B
        
    if trace:
        print("Integration interval (T):", T)
        print("Integration range (t_max):", t_max)
        print()
        print("t", "\t", "x")
        
    timestep = np.linspace(0, t_max, t_max/T+1)
    x_prev = x0
    states = np.empty((len(timestep), len(x0), 1))
    for i, t_step in enumerate(timestep):
        if trace and i % step == 0:
            print("{:.4}".format(t_step), end="")
            print("\t", end="")
            for j in x_prev:
                print(j, end='')
            print()

        m1 = f(x_prev)
        m2 = f(x_prev + (T/2)*m1)
        m3 = f(x_prev + (T/2)*m2)
        m4 = f(x_prev + T*m3)
        x_next = x_prev + (T/6) * (m1 + 2*m2 + 2*m3 + m4)
        states[i] = x_prev
        x_prev = x_next
    
    return states
    
def trapz(x0, A, B, T, t_max, trace=False, step=1):
    n = A.shape[0]
    U = np.eye(n)
    R = np.dot(np.linalg.inv(U - A*(T/2)), U + A*(T/2))
    S = np.dot(np.linalg.inv(U - A*(T/2)) * T, B)
    
    if trace:
        print("Integration interval (T):", T)
        print("Integration range (t_max):", t_max)
        print()
        print("t", "\t", "x")

    timestep = np.linspace(0, t_max, t_max/T+1)
    x_prev = x0
    states = np.empty((len(timestep), len(x0), 1))
    for i, t_step in enumerate(timestep):
        if trace and i % step == 0:
            print("{:.4}".format(t_step), end="")
            print("\t", end="")
            for j in x_prev:
                print(j, end='')
            print()
            
        x_next = np.dot(R, x_prev) + S
        states[i] = x_prev
        x_prev = x_next
    
    return states
