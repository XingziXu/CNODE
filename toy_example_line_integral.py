import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
from scipy.integrate import odeint as odeint_scipy

# define dp/dg
def dpdg(g,h,p):
    g1 = g*g
    h1 = h
    p1 = p.sqrt()
    partial_g = g1+h1+p1
    return partial_g

# define dp/dh
def dpdh(g,h,p):
    g1 = g
    h1 = h*h
    p1 = p.sqrt()
    partial_h = g1+h1+p1
    return partial_h

"""
# define function for line integral evaluation
def eval(g,h,p0):
    p_current =
    for iter in range(int(1e5)): # for each random value, integrate from 0 to 1
        dt = 1e-5
        
        t_current = iter*dt # calculate the current time
        dgdt_current = dgdt(0,t_current) # calculate the current dg/dt
        dhdt_current = dhdt(0,t_current) # calculate the current dh/dt
        p_next = p_current + dt*(dpdg(g[iter],h[iter],p_current)*dgdt_current +dpdh(g[iter],h[iter],p_current)*dhdt_current) # calculate the next p(g,h)
        p_current = p_next
"""
# define neural network for dg/dt
class ODEFunc(nn.Module):# define ode function, this is what we train on

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 40),
            nn.ReLU(inplace=False),
            nn.Linear(40, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(t+torch.Tensor([[0.]]))

# define visualization function
def visualize(true_g, true_h, pred_g, pred_h):# define the visualization process
    
    #if args.viz:
    if True:
#        makedirs('png')# if there is not a png image already, create one
        import matplotlib.pyplot as plt
        ax_traj = plt.subplot()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('g(t)')
        ax_traj.set_ylabel('h(t)')
        ax_traj.plot(true_g,true_h, 'g-') # green is gound truth
        ax_traj.plot(pred_g.view([1000,1]).detach().numpy(), pred_h.view([1000,1]).detach().numpy(), 'b--') # blue is prediction
        plt.show()
        #ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        #ax_traj.set_ylim(-2, 2)
        #ax_traj.legend()

# define dg/dt ground truth
def dgdt(y,t):
    grad_g = t**2
    return grad_g

# define dh/dt ground truth
def dhdt(y,t):
    grad_h = np.sqrt(t)
    return grad_h

if __name__ == '__main__':

    # calculate the ground truth trajectory for visualization
    dt_true = 1e-3 # define step size of line integral
    g_0_true = 0 # we start at (0,0) in the grid, so g(0)=0
    h_0_true = 0 # we start at (0,0) in the grid, so h(0)=0
    t_true = torch.linspace(0., 1., int(1e3)) # define evaluation points
    g_true = odeint_scipy(dgdt, g_0_true, t_true) # calculate g(t) values
    h_true = odeint_scipy(dhdt, h_0_true, t_true) # calculate h(t) values
    
    # load input and output data
    x_temp = np.load('input.npy') # load data
    y_temp = np.load('output.npy') # load data

    X_train = torch.FloatTensor(x_temp) # transform data into tensor format
    Y_train = torch.FloatTensor(y_temp) # transfrom data into tensor format

    g_learn = ODEFunc() # define neural network for x-dimension route
    h_learn = ODEFunc() # define neural network for y-dimension route

    params = list(g_learn.parameters()) + list(h_learn.parameters()) # define parameters for the combined model
    optimizer = optim.RMSprop(params, lr=0.001) # define optimizer

    iteration_num = 2000 # define the number of training iterations
    batch_size = 20 # define batch size
    data_size = np.size(x_temp)

    g_0 = torch.Tensor([[0.]])
    h_0 = torch.Tensor([[0.]])

    #t = torch.linspace(0., 1., int(1e3)) # define evaluation points
    #g = odeint(g_learn, g_0, t).clone() # calculate g(t) values
    #h = odeint(h_learn, h_0, t).clone() # calculate h(t) values

    for itr in range(1, iteration_num + 1):
        optimizer.zero_grad()# zero out the accumulated gradients
        s = torch.from_numpy(np.random.choice(np.arange(data_size, dtype=np.int64), batch_size, replace=False))
        batch_y0  = torch.FloatTensor(np.ones([batch_size,1,1]))
        batch_x = X_train[s]
        batch_y = Y_train[s]
        p_current = batch_x
        for iter in range(int(1e3)): # for each random value, integrate from 0 to 1
            dt = 1e-3
            t_current = iter*dt*torch.ones((1)) # calculate the current time
            # notice input 1 in g_learn in the line below is just a place holder
            dgdt_current = g_learn(1,torch.Tensor([[t_current]])) # calculate the current dg/dt
            # notice input 1 in h_learn in the line below is just a place holder
            dhdt_current = h_learn(1,torch.Tensor([[t_current]])) # calculate the current dh/dt
            g_current = odeint(g_learn, g_0, t_current)
            h_current = odeint(h_learn, h_0, t_current)
            #p_current = p_current + dt*(dpdg(g[iter],h[iter],p_current)+dpdh(g[iter],h[iter],p_current))
            p_current = p_current + dt*(dpdg(g_current, h_current, p_current)*dgdt_current +dpdh(g_current, h_current ,p_current)*dhdt_current) # calculate the next p(g,h)
            #p_current = p_next
        #pred = p_current
        #for dnum in range(len(batch_t)):
        #    pred[dnum] = integrate.quad(total_func, args.l_lim, args.r_lim)
        loss = torch.mean(torch.abs(p_current - batch_y))### need more work here # calculate the loss
        loss.backward(retain_graph=True)# backpropagation
        torch.autograd.set_detect_anomaly(True)
        print(loss)
        optimizer.step()# gradient descent
        if False:
        # calculate the trained integration path for visualization
            t_prediction = torch.linspace(0., 1., int(1e3)) # define evaluation points
            g_prediction = odeint(g_learn, g_0, t_prediction).clone() # calculate g(t) values
            h_prediction = odeint(h_learn, h_0, t_prediction).clone() # calculate h(t) values
            visualize(g_true, h_true, g_prediction, h_prediction)
        