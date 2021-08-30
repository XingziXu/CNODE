from __future__ import print_function
import argparse
import numpy
import torch
from torch._C import parse_ir
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchdiffeq._impl.fixed_adams import _dot_product
from torchvision import datasets, transforms  
from torch.optim.lr_scheduler import StepLR
from torchdiffeq import odeint_adjoint as odeint_adjoint
from torchdiffeq import odeint as odeint
from scipy.integrate import odeint as odeint_scipy
from torch.autograd import Variable
from random import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from math import pi
from torch.distributions import Normal
import numpy as np
from scipy.interpolate import make_interp_spline

class Grad_net(nn.Module): # the Grad_net defines the networks for the path and for the gradients
    def __init__(self, width_path: int, width_grad: int, width_conv2: int):
        super().__init__()
        self.nfe=0 # initialize the number of function evaluations

        self.path = nn.Sequential( # define the network for the integration path
            nn.Linear(2,20),
            nn.Softplus(),
            #nn.LogSigmoid(),
            nn.Linear(20,20),
            nn.Softplus(),
            nn.Linear(20,2)
        )


        self.grad_g = nn.Sequential( # define the network for the gradient on x direction
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
        
        self.grad_h = nn.Sequential( # define the network for the gradient on y direction
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )

    def forward(self, t, x):
        self.nfe+=1 # each time we evaluate the function, the number of evaluations adds one
        t_input = t.expand(x.size(0),1) # resize
        #print(t)
        #t_channel = ((t_input.view(x.size(0),1,1)).expand(x.size(0),1,x.size(2)*x.size(3))).view(x.size(0),1,x.size(2),x.size(3)) # resize
        path_input = torch.cat((t_input, p_i),dim=1) # concatenate the time and the image
        path_input = path_input.view(path_input.size(0),1,1,2)
        g_h_i = self.path(path_input) # calculate the position of the integration path
        g_h_i = g_h_i.view(g_h_i.size(0),2)

        dg_dt = g_h_i[:,0].view(g_h_i[:,0].size(0),1,1,1)
        dh_dt = g_h_i[:,1].view(g_h_i[:,1].size(0),1,1,1)
        
        # dg_dt = g_h_i[:,0].view(g_h_i.size(0),1,1) # resize 
        #dg_dt = dg_dt.expand(dg_dt.size(0),1,x.size(2)*x.size(3)) # resize 
        #dg_dt = dg_dt.view(dg_dt.size(0),1,x.size(2),x.size(3)) # resize 

        #dh_dt = g_h_i[:,1].view(g_h_i.size(0),1,1) # resize 
        #dh_dt = dh_dt.expand(dh_dt.size(0),1,x.size(2)*x.size(3)) # resize 
        #dh_dt = dh_dt.view(dh_dt.size(0),1,x.size(2),x.size(3)) # resize 
        x = x.view(x.size(0),1,1,1)
        dp = torch.mul(self.grad_g(x),dg_dt) + torch.mul(self.grad_g(x),dh_dt)# + torch.mul(self.grad_g(x),di_dt) # calculate the change in p
        dp = dp.view(dp.size(0),1)
        #print(t.item())
        return dp

class Classifier(nn.Module): # define the linear classifier
    def __init__(self, width_conv2: int, width_pool: int):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(1,2)

    def forward(self, x):
        x = self.classifier(x) # generate a 1x10 probability vector based on the flattened image&dimension
        return x

def get_n_params(model): # define a function to measure the number of parameters in a neural network
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def path_g(t,y):
    t_input = t.expand(p_i.size(0),1) # resize
    #print(t)
    #t_channel = ((t_input.view(x.size(0),1,1)).expand(x.size(0),1,x.size(2)*x.size(3))).view(x.size(0),1,x.size(2),x.size(3)) # resize
    path_input = torch.cat((t_input, p_i),dim=1) # concatenate the time and the image
    path_input = path_input.view(path_input.size(0),1,1,2)
    g_h_i = grad_net.path(path_input) # calculate the position of the integration path
    g_h_i = g_h_i.view(g_h_i.size(0),2)
    return g_h_i.squeeze()[0]

def path_h(t,y):
    t_input = t.expand(p_i.size(0),1) # resize
    #print(t)
    #t_channel = ((t_input.view(x.size(0),1,1)).expand(x.size(0),1,x.size(2)*x.size(3))).view(x.size(0),1,x.size(2),x.size(3)) # resize
    path_input = torch.cat((t_input, p_i),dim=1) # concatenate the time and the image
    path_input = path_input.view(path_input.size(0),1,1,2)
    g_h_i = grad_net.path(path_input) # calculate the position of the integration path
    g_h_i = g_h_i.view(g_h_i.size(0),2)
    return g_h_i.squeeze()[1]

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--validation-batch-size', type=int, default=1000, metavar='V',
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--step-size', type=int, default=5, metavar='M',
                        help='how many epochs to we change the learning rate, default is 5')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--adaptive-solver', action='store_true', default=False,
                        help='do we use euler solver or do we use dopri5')
    parser.add_argument('--clipper', action='store_true', default=True,
                        help='do we force the integration path to be monotonically increasing')
    parser.add_argument('--lr-grad', type=float, default=1e-3, metavar='LR',
                        help='learning rate for the gradients (default: 1e-3)')
    parser.add_argument('--lr-path', type=float, default=1e-3, metavar='LR',
                        help='learning rate for the path (default: 1e-3)')
    parser.add_argument('--lr-classifier', type=float, default=1e-3, metavar='LR',
                        help='learning rate for the classifier(default: 1e-3)')
    parser.add_argument('--tol', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--training-frequency', type=int, default=1, metavar='LR',
                        help='how often do we optimize the path network')
    parser.add_argument('--width-grad', type=int, default=64, metavar='LR',
                        help='width of the gradient network')
    parser.add_argument('--width-path', type=int, default=4, metavar='LR',
                        help='width of the path network')
    parser.add_argument('--width-conv2', type=int, default=6, metavar='LR',
                        help='width of the convolution')
    parser.add_argument('--width-pool', type=int, default=8, metavar='LR',
                        help='width of the adaptive average pooling')
    

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available() # check if we have a GPU available

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu") # check if we are using the GPU

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    validation_kwargs = {'batch_size': args.validation_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        validation_kwargs.update(cuda_kwargs)

    global grad_net
    grad_net = Grad_net(width_path=args.width_path, width_grad=args.width_grad, width_conv2=args.width_conv2).to(device) # define grad_net and assign to device
    classifier_net = Classifier(width_conv2=args.width_conv2, width_pool=args.width_pool).to(device) # define classifier network and assign to device
    
    grad_net.load_state_dict(torch.load('C:/Users/xingz/NeuralPDE/grad_net.pt'))
    grad_net.eval()
    classifier_net.load_state_dict(torch.load('C:/Users/xingz/NeuralPDE/classifer_net.pt'))
    classifier_net.eval()
    timesteps=5
    num_points = 10
    hidden = torch.linspace(-2,2,steps=num_points).view((num_points,1))
    t = torch.linspace(0,1,steps=timesteps)
    g = torch.linspace(0,1,steps=timesteps)
    h = torch.linspace(0,1,steps=timesteps)
    dpdt = np.zeros((timesteps, num_points))
    dgdt = np.ones((timesteps, num_points))
    dhdt = np.ones((timesteps, num_points))
    for i in range(len(t)):
        for j in range(len(hidden)):
            # Ensure h_j has shape (1, 1) as this is expected by odefunc
            h_j = hidden[j]
            global p_i
            p_i = h_j.view((1,1))
            t_input = t[i].expand(h_j.size(0),1)
            path_input = torch.cat((t_input, p_i),dim=1) # concatenate the time and the image
            path_input = path_input.view(path_input.size(0),1,1,2)
            g_h_i = grad_net.path(path_input) # calculate the position of the integration path
            g_h_i = g_h_i.view(g_h_i.size(0),2)
            dg_dt = g_h_i[:,0].view(g_h_i[:,0].size(0),1,1,1)
            dh_dt = g_h_i[:,1].view(g_h_i[:,1].size(0),1,1,1)
            dgdt[i, j] = dg_dt.squeeze()
            dhdt[i, j] = dh_dt.squeeze()
            if t[i] ==0:
                g[i] = grad_net.path(torch.cat((torch.Tensor([0.]).squeeze().expand(h_j.size(0),1), p_i),dim=1).view(path_input.size(0),1,1,2)).squeeze()[0]
                h[i] = grad_net.path(torch.cat((torch.Tensor([0.]).squeeze().expand(h_j.size(0),1), p_i),dim=1).view(path_input.size(0),1,1,2)).squeeze()[1]
            else:
                integration_t = torch.Tensor([0.,1.])*t[i]
                g[i] = odeint(path_g, g[0], integration_t, method="euler")[1]
                h[i] = odeint(path_h, h[0], integration_t, method="euler")[1]

            x = h_j.view(h_j.size(0),1,1,1)
            dpdt[i, j] = grad_net.grad_g(x)

    g_grid, p_grid = np.meshgrid(g.detach().numpy(), hidden, indexing='ij')
    plt.quiver(g_grid, p_grid, dgdt, dpdt, width=0.004, alpha=0.6)
    plt.show()

    h_grid, p_grid = np.meshgrid(h.detach().numpy(), hidden, indexing='ij')
    plt.quiver(h_grid, p_grid, dhdt, dpdt, width=0.004, alpha=0.6)
    plt.show()

if __name__ == '__main__':
    main()

