import numpy as np
import torch
import matplotlib.pyplot as plt
#from torchdiffeq import odeint_adjoint as odeint
from scipy.integrate import odeint

# initialization
size = int(1e6) # number of data points
p0 = (20.)*np.abs(np.random.randn(size,1)) # define initial values
p_f = np.random.randn(size,1) # initialize array to store values
g_0 = 0 # we start at (0,0) in the grid, so g(0)=0
h_0 = 0 # we start at (0,0) in the grid, so h(0)=0
l_bound = 0. # define lower bound of line integral
u_bound = 1. # define upper bound of line integral
num_eval = 1e3 # define number of evaluations from lower bound to upper bound
dt = (u_bound-l_bound)/num_eval # define step size of line integral

# define dp/dg
def dpdg(g,h,p):
    partial_g = np.sqrt(g)+h+p
    return partial_g

# define dp/dh
def dpdh(g,h,p):
    partial_h = g+np.sqrt(h)+p
    return partial_h

# define dg/dt
def dgdt(y,t):
    grad_g = 1
    return grad_g

# define dh/dt
def dhdt(y,t):
    grad_h = t/2
    return grad_h


# generation starts here
if __name__ == '__main__':
    

    t = torch.linspace(l_bound, u_bound, int(num_eval)) # define evaluation points
    g = odeint(dgdt, g_0, t) # calculate g(t) values
    h = odeint(dhdt, h_0, t) # calculate h(t) values
    p_current = p0
    #plt.plot(g,h)
    #plt.show()

    #for num in range(size): # for each point
    #    p_current = p0[num] # initialization
    #    p_next = p0[num] # initialization
    #p_vis = np.random.randn(100000,1)
    #p_vis[0] = p_current
    for iter in range(int(num_eval)): # for each random value, integrate from 0 to 1
        t_current = iter*dt # calculate the current time
        dgdt_current = dgdt(0,t_current) # calculate the current dg/dt, 0 is place holder
        dhdt_current = dhdt(0,t_current) # calculate the current dh/dt, 0 is place holder
        p_next = p_current + dt*(dpdg(g[iter],h[iter],p_current)*dgdt_current +dpdh(g[iter],h[iter],p_current)*dhdt_current) # calculate the next p(g,h)
        p_current = p_next # save the value
        #p_vis[iter] = p_current
    #plt.plot(p_vis)
    #plt.show()
    #    p_f[num] = p_current
        #print(t_current-t[iter])
        
    print(np.mean(p_current))


# save data into a npy file
with open('input.npy', 'wb') as f:
    np.save(f, p0)

with open('output.npy', 'wb') as f:
    np.save(f, p_current)
