import numpy as np
import torch
import matplotlib.pyplot as plt
#from torchdiffeq import odeint_adjoint as odeint
from scipy.integrate import odeint

# initialization
size = 1000 # number of data points
p0 = 100*np.abs(np.random.randn(size,1)) # define initial values
p_f = np.random.randn(size,1) # initialize array to store values

# define dp/dg
def dpdg(g,h,p):
    partial_g = g**2+h+np.sqrt(p)
    return partial_g

# define dp/dh
def dpdh(g,h,p):
    partial_h = g+h**2+np.sqrt(p)
    return partial_h

# define dg/dt
def dgdt(y,t):
    grad_g = t**2
    return grad_g

# define dh/dt
def dhdt(y,t):
    grad_h = np.sqrt(t)
    return grad_h


# generation starts here
if __name__ == '__main__':
    dt = 1e-3 # define step size of line integral
    g_0 = 0 # we start at (0,0) in the grid, so g(0)=0
    h_0 = 0 # we start at (0,0) in the grid, so h(0)=0

    t = torch.linspace(0., 1., int(1e3)) # define evaluation points
    g = odeint(dgdt, g_0, t) # calculate g(t) values
    h = odeint(dhdt, h_0, t) # calculate h(t) values
    p_current = p0
    plt.plot(g,h)
    plt.show()

    #for num in range(size): # for each point
    #    p_current = p0[num] # initialization
    #    p_next = p0[num] # initialization
    #p_vis = np.random.randn(100000,1)
    #p_vis[0] = p_current
    for iter in range(int(1e3)): # for each random value, integrate from 0 to 1
        t_current = iter*dt # calculate the current time
        dgdt_current = dgdt(0,t_current) # calculate the current dg/dt
        dhdt_current = dhdt(0,t_current) # calculate the current dh/dt
        p_next = p_current + dt*(dpdg(g[iter],h[iter],p_current)*dgdt_current +dpdh(g[iter],h[iter],p_current)*dhdt_current) # calculate the next p(g,h)
        p_current = p_next # save the value
        #p_vis[iter] = p_current
    #plt.plot(p_vis)
    #plt.show()
    #    p_f[num] = p_current
        #print(t_current-t[iter])
        



# save data into a npy file
with open('input.npy', 'wb') as f:
    np.save(f, p0)

with open('output.npy', 'wb') as f:
    np.save(f, p_current)
