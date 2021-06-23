import os
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
    g1 = g
    h1 = h
    p1 = p
    partial_g = g1+h1+p1
    return partial_g

# define dp/dh
def dpdh(g,h,p):
    g1 = g
    h1 = h
    p1 = p
    partial_h = g1+h1+p1
    return partial_h

# define dg/dt ground truth
def dgdt(y,t):
    grad_g = np.sqrt(t)
    return grad_g

# define dh/dt ground truth
def dhdt(y,t):
    grad_h = np.tanh(t)
    return grad_h
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
class gODEFunc(nn.Module):# define ode function, this is what we train on

    def __init__(self, input_size : int, width : int, output_size : int):
        super(gODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, width),
            nn.ReLU(),
            nn.Linear(width, output_size),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(t+torch.Tensor([[0.]]))

# define neural network for dh/dt
class hODEFunc(nn.Module):# define ode function, this is what we train on

    def __init__(self, input_size : int, width : int, output_size : int):
        super(hODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, width),
            nn.ReLU(),
            nn.Linear(width, output_size),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(t+torch.Tensor([[0.]]))


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


#if args.viz:
if True:    
    makedirs('png')# if there is not a png image already, create one
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(121, frameon=False)
    ax_loss = fig.add_subplot(122, frameon=False)
#    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

# define visualization function
def visualize(true_g, true_h, pred_g, pred_h, loss, num_eva):# define the visualization process
    
    #if args.viz:
    if True:
#        makedirs('png')# if there is not a png image already, create one
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('g(t)')
        ax_traj.set_ylabel('h(t)')
        ax_traj.plot(true_g,true_h, 'g-') # green is gound truth
        ax_traj.plot(pred_g.view([int(num_eva),1]).detach().numpy(), pred_h.view([int(num_eva),1]).detach().numpy(), 'b--') # blue is prediction
        
        #ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        #ax_traj.set_ylim(-2, 2)
        #ax_traj.legend()
        ax_loss.set_title('Losses')
        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.plot(loss,'k-')

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)

if __name__ == '__main__':
    l_bound = 0. # define lower bound of line integral
    u_bound = 1. # define upper bound of line integral
    num_eval = 2. # define number of evaluations from lower bound to upper bound
    # calculate the ground truth trajectory for visualization
    dt_true = (u_bound-l_bound)/num_eval # define step size of line integral
    g_0_true = 0. # we start at (0,0) in the grid, so g(0)=0
    h_0_true = 0. # we start at (0,0) in the grid, so h(0)=0
    t_true = torch.linspace(l_bound, u_bound, int((u_bound-l_bound)/dt_true)) # define evaluation points
    g_true = odeint_scipy(dgdt, g_0_true, t_true) # calculate g(t) values
    h_true = odeint_scipy(dhdt, h_0_true, t_true) # calculate h(t) values
    input_size = 1 # define network input size
    output_size = 1 # define network output size
    width = 64 # define network width
    
    # load input and output data
    x_temp = np.load('input.npy') # load data
    y_temp = np.load('output.npy') # load data

    X_train = torch.FloatTensor(x_temp) # transform data into tensor format
    Y_train = torch.FloatTensor(y_temp) # transfrom data into tensor format

    g_learn = gODEFunc(input_size, width, output_size) # define neural network for x-dimension route
    h_learn = hODEFunc(input_size, width, output_size) # define neural network for y-dimension route

    params_g = list(g_learn.parameters())
    params_h = list(h_learn.parameters()) # define parameters for the combined model
    optimizer_g = optim.RMSprop(params_g, lr=5e-4) # define optimizer
    optimizer_h = optim.RMSprop(params_h, lr=5e-4)

    iteration_num = 2000 # define the number of training iterations
    batch_size = 128 # define batch size
    data_size = np.size(x_temp)

    g_0 = torch.Tensor([[0.]])
    h_0 = torch.Tensor([[0.]])

    #t = torch.linspace(0., 1., int(1e3)) # define evaluation points
    #g = odeint(g_learn, g_0, t).clone() # calculate g(t) values
    #h = odeint(h_learn, h_0, t).clone() # calculate h(t) values

    loss_vis = []

    for itr in range(1, iteration_num + 1):
        optimizer_g.zero_grad()# zero out the accumulated gradients
        optimizer_h.zero_grad()
        s = torch.from_numpy(np.random.choice(np.arange(data_size, dtype=np.int64), batch_size, replace=False))
        batch_y0  = torch.FloatTensor(np.ones([batch_size,1,1]))
        batch_x = X_train[s]
        batch_y = Y_train[s]
        p_current = batch_x
        dt = (u_bound-l_bound)/num_eval
        for iter in range(1,int(num_eval)+1): # for each random value, integrate from 0 to 1
            t_current = iter*dt*torch.ones((1)) # calculate the current time
            t_calc = iter*dt*torch.Tensor([0.,1.])
            # notice input 1 in g_learn in the line below is just a place holder
            dgdt_current = g_learn(1,torch.Tensor([[t_current]])) # calculate the current dg/dt
            # notice input 1 in h_learn in the line below is just a place holder
            dhdt_current = h_learn(1,torch.Tensor([[t_current]])) # calculate the current dh/dt
            g_current = torch.inner(torch.squeeze(odeint(g_learn, g_0, t_calc, method='dopri5')),torch.tensor([0.,1.]))
            h_current = torch.inner(torch.squeeze(odeint(h_learn, h_0, t_calc, method='dopri5')),torch.tensor([0.,1.]))

            #print(g_current)
            
            #p_current = p_current + dt*(dpdg(g[iter],h[iter],p_current)+dpdh(g[iter],h[iter],p_current))
            p_current = p_current + dt*(dpdg(g_current, h_current, p_current)*dgdt_current +dpdh(g_current, h_current ,p_current)*dhdt_current) # calculate the next p(g,h)
            #print(p_current.max())
            
            #p_current = p_next
        #pred = p_current
        #for dnum in range(len(batch_t)):
        #    pred[dnum] = integrate.quad(total_func, args.l_lim, args.r_lim)
        loss_func = torch.mean(torch.abs(p_current - batch_y))### need more work here # calculate the loss
        #loss_constr = torch.square(torch.mean(torch.abs(g_current-g_true.max()))) + torch.square(torch.mean(torch.abs(h_current-h_true.max())))
        loss_func.backward(retain_graph=True)# backpropagation
        #loss_constr.backward(retain_graph=True)
        loss_vis.append(loss_func.item())
        torch.autograd.set_detect_anomaly(True)
        print('Iter {:04d} | Total Loss {:.6f} | Mean Truth Value {:.6f}'.format(itr, loss_func.item(), torch.mean(batch_y)))
        optimizer_g.step()# gradient descent
        optimizer_h.step()
        if True:
        # calculate the trained integration path for visualization
            t_prediction = torch.linspace(l_bound, u_bound, int(num_eval)) # define evaluation points
            g_prediction = odeint(g_learn, g_0, t_prediction).clone() # calculate g(t) values
            h_prediction = odeint(h_learn, h_0, t_prediction).clone() # calculate h(t) values
            visualize(g_true, h_true, g_prediction, h_prediction, loss_vis, num_eval)
