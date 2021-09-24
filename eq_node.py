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
dgdt_val = torch.randn(1, requires_grad=True)

def model(t,x):
    #print(dgdt_val)
    return dgdt_val

def update(args, model, optimizer, data, target, device):
    optimizer.zero_grad() # the start of updating the path's parameters
    data.requires_grad = True
    output = torch.empty(data.size(0),1)
    for i,row in enumerate(data):
        times = row[1]
        dg = torch.Tensor([0]).squeeze()
        dg.requires_grad=True
        t = torch.cat((torch.Tensor([0.]),times.view(1)),0).to(device) # we look to integrate from t=0 to t=1
        if args.adaptive_solver: # check if we are using the adaptive solver
            dg = torch.squeeze(odeint_adjoint(model, dg, t,method="dopri5",rtol=args.tol,atol=args.tol)[1]) # solve the neural line integral with an adaptive ode solver
            #print("The number of steps taken in this training itr is {}".format(grad_net.nfe)) # print the number of function evaluations we are using
        else:
            dg = torch.squeeze(odeint(model, dg, t, method="euler")[1]) # solve the neural line integral with the euler's solver
        output[i] = torch.cos(row[0]-dg) # classify the transformed images
    loss = nn.MSELoss()
    loss = loss(output, target.squeeze())
    loss.backward(retain_graph=True) # backpropagate through the loss
    optimizer.step() # update the path network's parameters
    return loss

def evaluate(args, grad_net, data, device):
    data.requires_grad = True
    output = torch.empty(1,1)
    for row in data:
        x0 = row[0]
        times = row[1]
        p = x0 # assign data, initialization
        t = torch.cat((torch.Tensor([0.]),times.view(1)),0).to(device) # we look to integrate from t=0 to t=1
        if args.adaptive_solver: # check if we are using the adaptive solver
            p = torch.squeeze(odeint_adjoint(grad_net, p, t,method="dopri5",rtol=args.tol,atol=args.tol)[1]) # solve the neural line integral with an adaptive ode solver
            print("The number of steps taken in this training itr is {}".format(grad_net.nfe)) # print the number of function evaluations we are using
            grad_net.nfe=0 # reset the number of function of evaluations
        else:
            p = torch.squeeze(odeint(grad_net, p, t, method="euler")[1]) # solve the neural line integral with the euler's solver
            grad_net.nfe=0 # reset the number of function of evaluations
        output = torch.cat((output,p.view(1,1)),dim=0) # classify the transformed images
    return output,p

def train(args, model, device, train_loader, optimizer, epoch):
    for batch_idx, (data, target) in enumerate(train_loader): # for each batch
        data, target = data.to(device), target.to(device) # assign data to device
        loss_grad = update(args, model, optimizer, data, target, device) # update gradient networks' weights
        if batch_idx % args.log_interval == 0: # print training loss and training process
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_grad.item()))

def test(args, grad_net, device, validation_loader):
    grad_net.eval() # set the network on evaluation mode
    test_loss = 0 # initialize test loss
    correct = 0 # initialize the number of correct predictions
    o1 = []
    d1 = []
    for data, target in validation_loader: # for each data batch
        x = np.linspace(0,2*pi,1000)
        data = torch.Tensor(np.cos(x)).view(1000,1)
        global p_i # claim the initial image batch as a global variable
        p_i = data
        output,p = evaluate(args, grad_net, data, device)
    
    a = 2*pi
    x = np.linspace(0,2*pi,1000)
    x_t = x-a
    plt.plot(x_t,output.detach().numpy(),'b')
    plt.plot(x_t,np.cos(x_t),'r')
    plt.show()
    test_loss /= len(validation_loader.dataset) # calculate test loss

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( # print test loss and accuracy
        test_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))
    
    if args.save_model: # check if we are saving the model
        torch.save(grad_net.state_dict(), "grad_net.pt") # save gradients and path model
        print("The current models are saved") # confirm all models are saved
    return 100. * correct / len(validation_loader.dataset), o1

def validation(args, grad_net, device, validation_loader):
    grad_net.eval() # set the network on evaluation mode
    test_loss = 0 # initialize test loss
    correct = 0 # initialize the number of correct predictions
    o1 = []
    d1 = []
    for data, target in validation_loader: # for each data batch
        data, target = data.to(device), target.to(device) # assign data to the device
        global p_i # claim the initial image batch as a global variable
        p_i = data
        output,p = evaluate(args, grad_net, data, device)
        target = target + 1
        target = target /2
        loss = nn.MSELoss()
        loss = loss(output[1:], target.squeeze())
        test_loss += loss  # sum up batch loss
    test_loss /= len(validation_loader.dataset) # calculate test loss

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( # print test loss and accuracy
        test_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))
    
    if args.save_model: # check if we are saving the model
        torch.save(grad_net.state_dict(), "grad_net.pt") # save gradients and path model
        print("The current models are saved") # confirm all models are saved
    return 100. * correct / len(validation_loader.dataset), o1

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--validation-batch-size', type=int, default=1000, metavar='V',
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
    parser.add_argument('--lr-grad', type=float, default=1e-3, metavar='LR',
                        help='learning rate for the gradients (default: 1e-3)')
    parser.add_argument('--lr-path', type=float, default=1e-3, metavar='LR',
                        help='learning rate for the path (default: 1e-3)')
    parser.add_argument('--lr-classifier', type=float, default=1e-3, metavar='LR',
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

    a = 2*pi
    x = torch.linspace(0,pi,3000)
    t = torch.linspace(0.1,1.5,3000)
    x_t = x-a*t
    input_data = torch.cat((x.view(3000,1),t.view(3000,1)),1)
    output_data = torch.Tensor(torch.cos(x_t)).view(3000,1)
    data_object = TensorDataset(input_data,output_data) # create your datset

    train_set, val_set = torch.utils.data.random_split(data_object, [3000, 0])
    
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(train_set,batch_size=1000,shuffle=True)


    optimizer = optim.Adam([dgdt_val], lr=0.2)

    scheduler_grad = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) # define scheduler for the gradients' network

    print('setup complete')

    accu = 0.0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler_grad.step()
if __name__ == '__main__':
    main()
