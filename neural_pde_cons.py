import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
# initialize the settings of training
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')# define ode solver
parser.add_argument('--data_size', type=int, default=1000)# define number of data points
parser.add_argument('--batch_time', type=int, default=10)###
parser.add_argument('--batch_size', type=int, default=20)# define batch size
parser.add_argument('--niters', type=int, default=2000)# define number of iterations
parser.add_argument('--test_freq', type=int, default=20)# define how often do we check error
parser.add_argument('--viz', action='store_true')# indicate if we want to visualize the results
parser.add_argument('--gpu', type=int, default=0)# define if we use gpu
parser.add_argument('--adjoint', action='store_true')# indicate if we use adjoint method
args = parser.parse_args()

if args.adjoint: # check if we indicated to use adjoint method
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu') # define if we use cpu or gpu

true_y0 = torch.tensor([[2.]]).to(device)# initialize y0
t = torch.linspace(0., 25., args.data_size).to(device)# define points to evaluate
true_A = torch.tensor([[1]]).to(device)### defining a weight matrix???
#true_A = torch.tensor([[-0.2, 1.0], [-1.0, -0.5]]).to(device)### defining a weight matrix???

class Lambda(nn.Module):

    def forward(self, t, y):
        #print(y)
        #return torch.mm(y**3, true_A)# define Lambda function
        return torch.Tensor([np.sin(t/10)])

with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri8')# solve ode with Lambda the dy/dt, true_y0 the initial condition, t the points to eval


def get_batch():# get a random batch of data
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False)) # generate 20 random indices
    batch_y0 = true_y[s]  # get the values of the twenty random indices 
    batch_t = t[:args.batch_time]  # define the time array
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # for each of the twenty random number, go ahead in time for batch_time length
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


#if args.viz:
if True:    
    makedirs('png')# if there is not a png image already, create one
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot( frameon=False)
    #ax_phase = fig.add_subplot(132, frameon=False)
    #ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, t, odefunc, itr):# define the visualization process
    
    #if args.viz:
    if True:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy(), 'g-') # green is gound truth
#        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:,:,0], 'g-') # green is gound truth
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy(), 'b--') # blue is prediction
#        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:,:,0], 'b--') # blue is prediction
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        #ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):# define ode function, this is what we train on

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y**3)

class OuterForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(4,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,x):
        y_pred = self.stack(x)
        return y_pred

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    x_temp = np.load('input.npy')
    y_temp = np.load('output.npy')

    X_train = torch.FloatTensor(x_temp)
    Y_train = torch.FloatTensor(y_temp)

    ii = 0

    inner = ODEFunc().to(device)# assign function to the computing device
    outer = OuterForward()

    loss_func = nn.MSELoss()
    params = list(inner.parameters()) + list(outer.parameters())
    optimizer = optim.RMSprop(params, lr=0.001)# define the optimizer we use, and indicate what parameters we want to optimize
    end = time.time()# this is the current time

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)

    # define a testing dataset for visualization
    s_test = (17*49)+np.sort(torch.from_numpy(np.random.choice(np.arange(49, dtype=np.int64), args.batch_size, replace=False)))
    batch_y0_test  = torch.FloatTensor(np.ones([args.batch_size,1,1]))
    batch_t_test = X_train[s_test,0]
    #batch_x = X_train[s,1]
    #batch_y = X_train[s,2]
    batch_x_test = X_train[s_test]
    batch_y_test = Y_train[s_test]
    #batch_y0, batch_t, batch_y = get_batch()# get a random batch of data
    # end of defining the dataset for visualization

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()# zero out the accumulated gradients
        s = ((itr-1)*49)+np.sort(torch.from_numpy(np.random.choice(np.arange(49, dtype=np.int64), args.batch_size, replace=False)))
        batch_y0  = torch.FloatTensor(np.ones([args.batch_size,1,1]))
        batch_t = X_train[s,0]
        #batch_x = X_train[s,1]
        #batch_y = X_train[s,2]
        batch_x = X_train[s]
        batch_y = Y_train[s]
        #batch_y0, batch_t, batch_y = get_batch()# get a random batch of data
        pred_p = odeint(inner, batch_y0, batch_t).to(device)# predict y values with our current neural network (the ODEfunc)
        pred_flattened = pred_p[:,1,0]
        pred_model = outer(torch.cat((batch_x,pred_flattened),dim=1))### need more work here

        loss = torch.mean(torch.abs(pred_model - batch_y))### need more work here # calculate the loss
        loss.backward()# backpropagation
        optimizer.step()# gradient descent

        time_meter.update(time.time() - end)# measure how long the training has taken
        loss_meter.update(loss.item())# measure the error
        
        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_p_test = odeint(inner, batch_y0_test, batch_t_test).to(device)# predict y values with our current neural network (the ODEfunc)
                pred_flattened_test = pred_p_test[:,1,0]
                pred_model_test = outer(torch.cat((batch_x_test,pred_flattened_test),dim=1))### need more work here
                #pred_y = odeint(inner, true_y0, t).to(device)
                #loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(batch_y_test, pred_model_test, batch_t_test, outer, ii)
                ii += 1

        end = time.time()# update the time
