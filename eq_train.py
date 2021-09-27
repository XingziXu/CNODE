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

class Grad_net(nn.Module): # the Grad_net defines the networks for the path and for the gradients
    def __init__(self, width_path: int, width_grad: int, width_conv2: int):
        super().__init__()
        self.nfe=0 # initialize the number of function evaluations

        self.path = nn.Sequential( # define the network for the integration path
            nn.Linear(1,20),
            #nn.Hardsigmoid(),
            #nn.ELU(),
            nn.Linear(20,20),
            #nn.ELU(),
            nn.Linear(20,1)
        )


        self.grad_g = nn.Sequential( # define the network for the gradient on x direction
            nn.Linear(2,32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32,1)
        )
        
        self.grad_h = nn.Sequential( # define the network for the gradient on y direction
            nn.Linear(2,32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32,1)
        )

    def forward(self, t, x):
        self.nfe+=1 # each time we evaluate the function, the number of evaluations adds one
        t_input = t.expand(x.size(0),1) # resize
        g_h_i = self.path(t_input) # calculate the position of the integration path
        dg_dt = g_h_i[:,0].view(g_h_i[:,0].size(0),1)
        dp = torch.mul(self.grad_g(torch.cat((x,t_input),1).float()),dg_dt) + self.grad_h(torch.cat((x,t_input),1).float())# + torch.mul(self.grad_g(x),di_dt) # calculate the change in p
        return dp

def initialize_grad(m):
    if isinstance(m, nn.Conv2d):
        #nn.init.xavier_normal_(m.weight.data,gain=1.0)
        #torch.nn.init.eye_(m.weight.data)
        nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
        #nn.init.sparse_(m.weight.data,sparsity=0.1)
    if isinstance(m, nn.Linear):
        #nn.init.xavier_normal_(m.weight.data,gain=1.0)
        nn.init.kaiming_normal_(m.weight.data,nonlinearity='relu')
        #nn.init.sparse_(m.weight.data,sparsity=0.1)

def initialize_path(n):
    if isinstance(n, nn.Conv2d):
        nn.init.xavier_uniform_(n.weight.data,gain=0.7)
        #nn.init.orthogonal_(n.weight.data,gain=1.0)
        #nn.init.kaiming_uniform_(n.weight.data,nonlinearity='relu')
    if isinstance(n, nn.Linear):
        nn.init.xavier_uniform_(n.weight.data,gain=0.7)
        #nn.init.kaiming_uniform_(n.weight.data,nonlinearity='relu')
        #nn.init.orthogonal(n.weight.data,gain=1.0)

def initialize_classifier(p):
    #if isinstance(p, nn.Conv2d):
    #    torch.nn.init.normal_(p.weight.data, mean=0.0, std=1.0)
        #torch.nn.init.eye_(m.weight.data)
        #nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
    if isinstance(p, nn.Linear):
        #torch.nn.init.kaiming_uniform_(p.weight.data,nonlinearity='relu')
        torch.nn.init.orthogonal_(p.weight.data,gain=1.0)

def get_n_params(model): # define a function to measure the number of parameters in a neural network
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def update(args, grad_net, optimizer, data, target, device):
    optimizer.zero_grad() # the start of updating the path's parameters
    data.requires_grad = True
    sorted, indices = torch.sort(data[:,1], 0)
    p = data[indices,0].view(data.size(0),1) # assign data, initialization
    target = target[indices].view(data.size(0),1)
    output = torch.empty(1,1)
    #output.requires_grad = True
    t = torch.cat((torch.Tensor([0.]),data[indices,1]),0).to(device) # we look to integrate from t=0 to t=1
    if args.adaptive_solver: # check if we are using the adaptive solver
        p = torch.squeeze(odeint_adjoint(grad_net, p, t,method="dopri5",rtol=args.tol,atol=args.tol)[1]) # solve the neural line integral with an adaptive ode solver
        print("The number of steps taken in this training itr is {}".format(grad_net.nfe)) # print the number of function evaluations we are using
        grad_net.nfe=0 # reset the number of function of evaluations
    else:
        p = torch.squeeze(odeint(grad_net, p, t, method="euler")[1]) # solve the neural line integral with the euler's solver
        grad_net.nfe=0 # reset the number of function of evaluations
    #output = torch.cat((output,p.view(1,1)),dim=0) # classify the transformed images
    #soft_max = nn.Softmax(dim=1) # define a soft max calculator
    #output = soft_max(output) # get the prediction results by getting the most probable ones
    #loss_func = nn.CrossEntropyLoss()
    loss = torch.norm(p-target.squeeze())
    #loss = loss(p.view(p.size(0),1), target)
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
        #p.requires_grad=True # record the computation graph
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

def train(args, grad_net, device, train_loader, optimizer_grad, epoch):
    grad_net.train() # set network on training mode
    for batch_idx, (data, target) in enumerate(train_loader): # for each batch
        global pathdt
        pathdt = torch.Tensor([[ 0.,0.]])
        data, target = data.to(device), target.to(device) # assign data to device
        loss_grad = update(args, grad_net, optimizer_grad, data, target, device) # update gradient networks' weights
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
        #data, target = data.to(device), target.to(device) # assign data to the device
        global p_i # claim the initial image batch as a global variable
        p_i = data
        output,p = evaluate(args, grad_net, data, device)
        #d1 = torch.cat((d1,data),1)
    
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
        #d1 = torch.cat((d1,data),1)
        target = target + 1
        target = target /2
        
        #if o1==[]:
        #    o1 = torch.cat((p.view((p.size(0),1)),target),1)
        #else:
        #    o1 = torch.cat((o1,torch.cat((p.view((p.size(0),1)),target),1)),0)

        loss = nn.MSELoss()
        loss = loss(output[1:], target.squeeze())
        test_loss += loss  # sum up batch loss
        #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        #correct += pred.eq(target.view_as(pred)).sum().item() # sum up the number of correct predictions
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
    parser.add_argument('--lr-grad', type=float, default=5e-3, metavar='LR',
                        help='learning rate for the gradients (default: 1e-3)')
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
    x = torch.linspace(0,pi,1000)
    t_train = torch.linspace(0.1,1.0,1000)
    x_t_train = x-a*t_train
    input_data = torch.cat((x.view(1000,1),t_train.view(1000,1)),1)
    output_data = torch.Tensor(torch.tanh(x_t_train)).view(1000,1)
    data_object_train = TensorDataset(input_data,output_data) # create your datset
    train_set = data_object_train

    t_test = torch.linspace(0.1,1.0,1000)
    x_t_test = x-a*t_test
    input_data_test = torch.cat((x.view(1000,1),t_test.view(1000,1)),1)
    output_data_test = torch.Tensor(torch.tanh(x_t_test)).view(1000,1)
    data_object_test = TensorDataset(input_data_test,output_data_test) # create your datset
    test_set = data_object_test
    #train_set, val_set = torch.utils.data.random_split(data_object, [1000, 0])
    
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(test_set,batch_size=1000,shuffle=True)

    grad_net = Grad_net(width_path=args.width_path, width_grad=args.width_grad, width_conv2=args.width_conv2).to(device) # define grad_net and assign to device

    #grad_net.apply(initialize_grad)
    #grad_net.grad_g.apply(initialize_grad)
    #grad_net.grad_h.apply(initialize_grad)
    #grad_net.path.apply(initialize_path)
    #classifier_net.apply(initialize_classifier)

    optimizer_grad = optim.AdamW(list(grad_net.parameters()), lr=args.lr_grad, weight_decay = 5e-4) # define optimizer on the gradients
    
    print("The number of parameters used is {}".format(get_n_params(grad_net))) # print the number of parameters in our model

    scheduler_grad = StepLR(optimizer_grad, step_size=args.step_size, gamma=args.gamma) # define scheduler for the gradients' network

    print('setup complete')

    accu = 0.0
    #outer = torch.zeros((25,153,3))
    #inner = torch.zeros((25,147,3))
    for epoch in range(1, args.epochs + 1):
        train(args, grad_net, device, train_loader, optimizer_grad, epoch)
        #accu_new, o1 = validation(args, grad_net, device, test_loader)
        #outer[epoch-1,:,:] = o1[o1[:,2]==1.]
        #inner[epoch-1,:,:] = o1[o1[:,2]==0.]
        #if accu_new > accu:
        #    accu = accu_new
        #print('The best accuracy is {:.4f}%\n'.format(accu))
        scheduler_grad.step()
    #test(args, grad_net, device, test_loader)
    a=2
    """for i in range(0,3):
        outer1 = outer[:,i,:]
        inner1 = inner[:,i,:]
        outer1 = outer1.detach().numpy()
        inner1 = inner1.detach().numpy()
        outer_spline = make_interp_spline(outer1[:,0], outer1[:, 1])
        X_ = np.linspace(outer1[:,0].min(), outer1[:,0].max(), 500)
        Y_ = outer_spline(X_)
        plt.plot(X_, Y_)
        inner_spline = make_interp_spline(inner1[:,0], inner1[:, 1])
        X_ = np.linspace(inner1[:,0].min(), inner1[:,0].max(), 500)
        Y_ = inner_spline(X_)
        plt.plot(X_, Y_)
#        plt.plot(outer1[:,0],outer1[:,1],color='r')
#        plt.plot(inner1[:,0],inner1[:,1],color='b')
    plt.show()"""
if __name__ == '__main__':
    main()
