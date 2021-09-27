from __future__ import print_function
import argparse
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms  
from torch.optim.lr_scheduler import StepLR
from torchdiffeq import odeint_adjoint as odeint_adjoint
from torchdiffeq import odeint as odeint
from scipy.integrate import odeint as odeint_scipy
from torch.autograd import Variable
from random import random
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from math import pi
from torch.distributions import Normal
import numpy as np
from scipy.interpolate import make_interp_spline
import random
from mpl_toolkits import mplot3d
dgdt_val = torch.rand(1,requires_grad=True)#torch.Tensor([3])
#dgdt_val.requires_grad = True

def model(t,x):
    print(dgdt_val)
    return dgdt_val

class Func_net(nn.Module): # the Grad_net defines the networks for the path and for the gradients
    def __init__(self):
        super().__init__()
        self.grad = nn.Sequential( # define the network for the gradient on x direction
            nn.Linear(1,4),
            nn.Tanh(),
            #nn.Linear(16,16),
            #nn.Tanh(),
            nn.Linear(4,1)
        )
    def forward(self,x):
        return self.grad(x)

def update(args, model, func_net, optimizer, optimizer_ini, data, target, device):
    optimizer.zero_grad() # the start of updating the path's parameters
    optimizer_ini.zero_grad()
    data.requires_grad = True
    sorted, indices = torch.sort(data[:,1], 0)
    output = torch.empty(data.size(0),1)
    times = data[indices,1]
    dg = torch.zeros(data.size(0),1)
    dg.requires_grad=True
    t = torch.cat((torch.Tensor([0.]),times),0).to(device)
    dg = times*model(1,2)
    target = target[indices]
    output = func_net((data[indices,0]-dg).view(dg.size(0),1)).squeeze()
    #for i,row in enumerate(data):
    #    times = row[1]
    #    dg = torch.Tensor([0]).squeeze()
    #    dg.requires_grad=True
    #    t = torch.cat((torch.Tensor([0.]),times.view(1)),0).to(device) # we look to integrate from t=0 to t=1
    #    #if args.adaptive_solver: # check if we are using the adaptive solver
    #    #    dg = torch.squeeze(odeint_adjoint(model, dg, t,method="dopri5",rtol=args.tol,atol=args.tol)[1]) # solve the neural line integral with an adaptive ode solver
    #        #print("The number of steps taken in this training itr is {}".format(grad_net.nfe)) # print the number of function evaluations we are using
    #    #else:
    #    #    dg = torch.squeeze(odeint(model, dg, t, method="euler")[1]) # solve the neural line integral with the euler's solver
    #    dg = times*model(1,2)
    #    output[i] = torch.cos(row[0]-dg) # classify the transformed images
    loss = torch.norm(output-target.squeeze())#loss(output, target.squeeze())
    loss.backward(retain_graph=True) # backpropagate through the loss
    optimizer.step() # update the path network's parameters
    optimizer_ini.step()
    return loss
    

def evaluate(args, model, func_net, data, target, device):
    data.requires_grad = True
    sorted, indices = torch.sort(data[:,1], 0)
    output = torch.empty(data.size(0),1)
    times = data[indices,1]
    dg = torch.zeros(data.size(0),1)
    dg.requires_grad=True
    t = torch.cat((torch.Tensor([0.]),times),0).to(device)
    dg = times*model(1,2)
    target = target[indices]
    output = func_net((data[indices,0]-dg).view(dg.size(0),1)).squeeze()
    #for i,row in enumerate(data):
    #    times = row[1]
    #    dg = torch.Tensor([0]).squeeze()
    #    dg.requires_grad=True
    #    t = torch.cat((torch.Tensor([0.]),times.view(1)),0).to(device) # we look to integrate from t=0 to t=1
    #    #if args.adaptive_solver: # check if we are using the adaptive solver
    #    #    dg = torch.squeeze(odeint_adjoint(model, dg, t,method="dopri5",rtol=args.tol,atol=args.tol)[1]) # solve the neural line integral with an adaptive ode solver
    #        #print("The number of steps taken in this training itr is {}".format(grad_net.nfe)) # print the number of function evaluations we are using
    #    #else:
    #    #    dg = torch.squeeze(odeint(model, dg, t, method="euler")[1]) # solve the neural line integral with the euler's solver
    #    dg = times*model(1,2)
    #    output[i] = torch.cos(row[0]-dg) # classify the transformed images
    loss = torch.norm(output-target.squeeze())#loss(output, target.squeeze())
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(data[indices,1].detach().numpy(), times.detach().numpy(),output.detach().numpy())
    ax.scatter3D(data[indices,1].detach().numpy(), times.detach().numpy(),target.detach().numpy())
    plt.show()
    return loss

def train(args, model, func_net, device, train_loader, optimizer, optimizer_ini, epoch):
    for batch_idx, (data, target) in enumerate(train_loader): # for each batch
        data, target = data.to(device), target.to(device) # assign data to device
        loss_grad = update(args, model, func_net, optimizer, optimizer_ini, data, target, device) # update gradient networks' weights
        if batch_idx % args.log_interval == 0: # print training loss and training process
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_grad.item()))

def test(args, model, func_net, device, test_loader):
    for batch_idx, (data, target) in enumerate(test_loader): # for each batch
        data, target = data.to(device), target.to(device) # assign data to device
        loss_grad = evaluate(args, model, func_net, data, target, device) # update gradient networks' weights
        if batch_idx % args.log_interval == 0: # print training loss and training process
            print('Test Loss: {:.6f}'.format(loss_grad))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--validation-batch-size', type=int, default=1000, metavar='V',
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
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
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--adaptive-solver', action='store_true', default=False,
                        help='do we use euler solver or do we use dopri5')
    parser.add_argument('--clipper', action='store_true', default=True,
                        help='do we force the integration path to be monotonically increasing')
    parser.add_argument('--lr-grad', type=float, default=1, metavar='LR',
                        help='learning rate for the gradients (default: 1e-3)')
    parser.add_argument('--lr-path', type=float, default=1, metavar='LR',
                        help='learning rate for the path (default: 1e-3)')
    parser.add_argument('--lr-classifier', type=float, default=1, metavar='LR',
                        help='learning rate for the classifier(default: 1e-3)')
    parser.add_argument('--tol', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--training-frequency', type=int, default=1, metavar='LR',
                        help='how often do we optimize the path network')
    parser.add_argument('--width-grad', type=int, default=8, metavar='LR',
                        help='width of the gradient network')
    parser.add_argument('--width-path', type=int, default=8, metavar='LR',
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

    num_pts = 2000
    a = 2*pi
    x = torch.rand(num_pts)*pi
    t_train = torch.rand(num_pts)
    x_t_train = x-a*t_train
    input_data = torch.cat((x.view(num_pts,1),t_train.view(num_pts,1)),1)
    output_data = torch.Tensor(torch.tanh(x_t_train)).view(num_pts,1)
    data_object_train = TensorDataset(input_data,output_data) # create your datset
    train_set = data_object_train

    t_test = torch.rand(num_pts)*3.0
    x_t_test = x-a*t_test
    input_data_test = torch.cat((x.view(num_pts,1),t_test.view(num_pts,1)),1)
    output_data_test = torch.Tensor(torch.tanh(x_t_test)).view(num_pts,1)
    data_object_test = TensorDataset(input_data_test,output_data_test) # create your datset
    test_set = data_object_test
    #train_set, val_set = torch.utils.data.random_split(data_object, [1000, 0])
    
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(test_set,batch_size=num_pts,shuffle=True)

    func_net = Func_net()

    optimizer = optim.Adam([dgdt_val], lr=0.02)
    optimizer_ini = optim.Adam(list(func_net.parameters()), lr = 6e-3, weight_decay=5e-3)

    scheduler_grad = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) # define scheduler for the gradients' network

    print('setup complete')

    accu = 0.0
    for epoch in range(1, args.epochs + 1):
        train(args, model, func_net, device, train_loader, optimizer, optimizer_ini, epoch)
        
        scheduler_grad.step()
    test(args, model, func_net, device, test_loader)
if __name__ == '__main__':
    main()
