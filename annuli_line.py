from __future__ import print_function
import argparse
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
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def random_point_in_sphere(dim, min_radius, max_radius):
    """Returns a point sampled uniformly at random from a sphere if min_radius
    is 0. Else samples a point approximately uniformly on annulus.

    Parameters
    ----------
    dim : int
        Dimension of sphere

    min_radius : float
        Minimum distance of sampled point from origin.

    max_radius : float
        Maximum distance of sampled point from origin.
    """
    # Sample distance of point from origin
    unif = random()
    distance = (max_radius - min_radius) * (unif ** (1. / dim)) + min_radius
    # Sample direction of point away from origin
    direction = torch.randn(dim)
    unit_direction = direction / torch.norm(direction, 2)
    return distance * unit_direction

class ConcentricSphere(Dataset):
    """Dataset of concentric d-dimensional spheres. Points in the inner sphere
    are mapped to -1, while points in the outer sphere are mapped 1.

    Parameters
    ----------
    dim : int
        Dimension of spheres.

    inner_range : (float, float)
        Minimum and maximum radius of inner sphere. For example if inner_range
        is (1., 2.) then all points in inner sphere will lie a distance of
        between 1.0 and 2.0 from the origin.

    outer_range : (float, float)
        Minimum and maximum radius of outer sphere.

    num_points_inner : int
        Number of points in inner cluster

    num_points_outer : int
        Number of points in outer cluster
    """
    def __init__(self, dim, inner_range, outer_range, num_points_inner,
                 num_points_outer):
        self.dim = dim
        self.inner_range = inner_range
        self.outer_range = outer_range
        self.num_points_inner = num_points_inner
        self.num_points_outer = num_points_outer

        self.data = []
        self.targets = []

        # Generate data for inner sphere
        for _ in range(self.num_points_inner):
            self.data.append(
                random_point_in_sphere(dim, inner_range[0], inner_range[1])
            )
            self.targets.append(torch.Tensor([-1]))

        # Generate data for outer sphere
        for _ in range(self.num_points_outer):
            self.data.append(
                random_point_in_sphere(dim, outer_range[0], outer_range[1])
            )
            self.targets.append(torch.Tensor([1]))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

class Grad_net(nn.Module): # the Grad_net defines the networks for the path and for the gradients
    def __init__(self, width_path: int, width_grad: int, width_conv2: int):
        super().__init__()
        self.nfe=0 # initialize the number of function evaluations

        self.path = nn.Sequential( # define the network for the integration path
            nn.Linear(3,20),
            nn.Softmax(),
            nn.Linear(20,20),
            #nn.Softmax(),
            nn.Linear(20,2)
        )


        self.grad_g = nn.Sequential( # define the network for the gradient on x direction
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,2)
        )
        
        self.grad_h = nn.Sequential( # define the network for the gradient on y direction
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,2)
        )

    def forward(self, t, x):
        self.nfe+=1 # each time we evaluate the function, the number of evaluations adds one

        t_input = t.expand(x.size(0),1) # resize
        #t_channel = ((t_input.view(x.size(0),1,1)).expand(x.size(0),1,x.size(2)*x.size(3))).view(x.size(0),1,x.size(2),x.size(3)) # resize
        path_input = torch.cat((t_input, p_i),dim=1) # concatenate the time and the image
        path_input = path_input.view(path_input.size(0),1,1,3)
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
        x = x.view(x.size(0),1,1,2)
        dp = torch.mul(self.grad_g(x),dg_dt) + torch.mul(self.grad_g(x),dh_dt)# + torch.mul(self.grad_g(x),di_dt) # calculate the change in p
        dp = dp.view(dp.size(0),2)
        #print(t.item())
        return dp

class Classifier(nn.Module): # define the linear classifier
    def __init__(self, width_conv2: int, width_pool: int):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(2,2)

    def forward(self, x):
        x = self.classifier(x) # generate a 1x10 probability vector based on the flattened image&dimension
        return x

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

def update(args, grad_net, classifier_net, optimizer, data, target, device):
    optimizer.zero_grad() # the start of updating the path's parameters
    p = data # assign data, initialization
    p.requires_grad=True # record the computation graph
    t = torch.Tensor([0.,1.]).to(device) # we look to integrate from t=0 to t=1
    t.requires_grad=True # record the computation graph
    if args.adaptive_solver: # check if we are using the adaptive solver
        p = torch.squeeze(odeint_adjoint(grad_net, p, t,method="dopri5",rtol=args.tol,atol=args.tol)[1]) # solve the neural line integral with an adaptive ode solver
        print("The number of steps taken in this training itr is {}".format(grad_net.nfe)) # print the number of function evaluations we are using
        grad_net.nfe=0 # reset the number of function of evaluations
    else:
        p = torch.squeeze(odeint(grad_net, p, t, method="euler")[1]) # solve the neural line integral with the euler's solver
        grad_net.nfe=0 # reset the number of function of evaluations
    output = classifier_net(p) # classify the transformed images
    soft_max = nn.Softmax(dim=1) # define a soft max calculator
    output = soft_max(output) # get the prediction results by getting the most probable ones
    #loss_func = nn.CrossEntropyLoss()
    target = target + 1
    target = target /2
    target = target.to(torch.long)
    target = target.view(target.size(0))
    loss = F.cross_entropy(output, target) # calculate the function loss
    loss.backward(retain_graph=True) # backpropagate through the loss
    optimizer.step() # update the path network's parameters
    return loss

def evaluate(args, grad_net, classifier_net, data, device):
    p = data # assign data, initialization
    p.requires_grad=True # record the computation graph
    t = torch.Tensor([0.,1.]).to(device) # we look to integrate from t=0 to t=1
    t.requires_grad=True # record the computation graph
    if args.adaptive_solver: # check if we are using the adaptive solver
        p = torch.squeeze(odeint_adjoint(grad_net, p, t,method="dopri5",rtol=args.tol,atol=args.tol)[1]) # solve the neural line integral with an adaptive ode solver
        print("The number of steps taken in this testing itr is {}".format(grad_net.nfe)) # print the number of function evaluations we are using
        grad_net.nfe=0 # reset the number of function of evaluations
    else:
        p = torch.squeeze(odeint(grad_net, p, t, method="euler")[1]) # solve the neural line integral with the euler's solver
        grad_net.nfe=0 # reset the number of function of evaluations
    output = classifier_net(p) # classify the transformed images
    soft_max = nn.Softmax(dim=1) # define a soft max calculator
    output = soft_max(output) # get the prediction results by getting the most probable ones
    return output,p

def train(args, grad_net, classifier_net, device, train_loader, optimizer_grad, epoch):
    grad_net.train() # set network on training mode
    classifier_net.train() # set network on training mode
    for batch_idx, (data, target) in enumerate(train_loader): # for each batch
        data, target = data.to(device), target.to(device) # assign data to device
        global p_i # claim the initial image batch as a global variable
        p_i = data
        loss_grad = update(args, grad_net, classifier_net, optimizer_grad, data, target, device) # update gradient networks' weights
        if batch_idx % args.log_interval == 0: # print training loss and training process
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_grad.item()))

def test(args, grad_net, classifier_net, device, test_loader):
    grad_net.eval() # set the network on evaluation mode
    classifier_net.eval() # set the network on evaluation mode
    test_loss = 0 # initialize test loss
    correct = 0 # initialize the number of correct predictions
    for data, target in test_loader: # for each data batch
        data, target = data.to(device), target.to(device) # assign data to the device
        global p_i # claim the initial image batch as a global variable
        p_i = data
        output,p = evaluate(args, grad_net, classifier_net, data, device)
        target = target + 1
        target = target /2
        target = target.to(torch.long)
        target = target.view(target.size(0))
        #output1 = output.detach().numpy()
        #plt.scatter(output1[:,0],output1[:,1])
        #plt.show()
        test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item() # sum up the number of correct predictions

    test_loss /= len(test_loader.dataset) # calculate test loss

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( # print test loss and accuracy
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    if args.save_model: # check if we are saving the model
        torch.save(grad_net.state_dict(), "grad_net.pt") # save gradients and path model
        torch.save(classifier_net.state_dict(), "classifer_net.pt") # save classifier model
        print("The current models are saved") # confirm all models are saved

def validation(args, grad_net, classifier_net, device, validation_loader):
    grad_net.eval() # set the network on evaluation mode
    classifier_net.eval() # set the network on evaluation mode
    test_loss = 0 # initialize test loss
    correct = 0 # initialize the number of correct predictions
    o1 = []
    d1 = []
    for data, target in validation_loader: # for each data batch
        data, target = data.to(device), target.to(device) # assign data to the device
        global p_i # claim the initial image batch as a global variable
        p_i = data
        output,p = evaluate(args, grad_net, classifier_net, data, device)
        #d1 = torch.cat((d1,data),1)
        target = target + 1
        target = target /2
        target = target.to(torch.long)
        
        if o1==[]:
            o1 = torch.cat((p,target),1)
        else:
            o1 = torch.cat((o1,torch.cat((p,target),1)),0)
        target = target.view(target.size(0))
        test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item() # sum up the number of correct predictions

    o1 = o1.detach().numpy()
    outer1 = o1[o1[:,2]==1.]
    inner1 = o1[o1[:,2]==0.]
    plt.scatter(outer1[:,0],outer1[:,1],color='r')
    plt.scatter(inner1[:,0],inner1[:,1])
    plt.show()
    test_loss /= len(validation_loader.dataset) # calculate test loss

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( # print test loss and accuracy
        test_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))
    
    if args.save_model: # check if we are saving the model
        torch.save(grad_net.state_dict(), "grad_net.pt") # save gradients and path model
        torch.save(classifier_net.state_dict(), "classifer_net.pt") # save classifier model
        print("The current models are saved") # confirm all models are saved
    return 100. * correct / len(validation_loader.dataset)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--validation-batch-size', type=int, default=1000, metavar='V',
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
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


    data_object = ConcentricSphere(dim=2,inner_range=[0.0,0.5],outer_range=[1.0,1.5],num_points_inner=500,num_points_outer=1000)

    train_set, val_set = torch.utils.data.random_split(data_object, [1350, 150])
    
    train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(val_set,batch_size=args.batch_size,shuffle=True)

    grad_net = Grad_net(width_path=args.width_path, width_grad=args.width_grad, width_conv2=args.width_conv2).to(device) # define grad_net and assign to device
    classifier_net = Classifier(width_conv2=args.width_conv2, width_pool=args.width_pool).to(device) # define classifier network and assign to device

    #grad_net.apply(initialize_grad)
    #grad_net.grad_g.apply(initialize_grad)
    #grad_net.grad_h.apply(initialize_grad)
    #grad_net.path.apply(initialize_path)
    #classifier_net.apply(initialize_classifier)

    optimizer_grad = optim.AdamW(list(grad_net.parameters())+list(classifier_net.parameters()), lr=args.lr_grad, weight_decay=5e-4) # define optimizer on the gradients
    
    print("The number of parameters used is {}".format(get_n_params(grad_net)+get_n_params(classifier_net))) # print the number of parameters in our model

    scheduler_grad = StepLR(optimizer_grad, step_size=args.step_size, gamma=args.gamma) # define scheduler for the gradients' network

    print('setup complete')

    accu = 0.0
    for epoch in range(1, args.epochs + 1):
        train(args, grad_net, classifier_net, device, train_loader, optimizer_grad, epoch)
        accu_new = validation(args, grad_net, classifier_net, device, test_loader)
        if accu_new > accu:
            accu = accu_new
        print('The best accuracy is {:.4f}%\n'.format(accu))
        scheduler_grad.step()
    test(args, grad_net, classifier_net, device, test_loader)

if __name__ == '__main__':
    main()


"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms  
from torch.optim.lr_scheduler import StepLR
from torchdiffeq import odeint as odeint
from torchdiffeq import odeint_adjoint as odeint_adjoint
from scipy.integrate import odeint as odeint_scipy
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

class LineInt(nn.Module):
    def __init__(self, device, data_dim=2, hidden_dim=16, augment_dim=0,
                 time_dependent=False, non_linearity='relu'):
        super(LineInt, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations
        self.time_dependent = time_dependent

        if time_dependent:
            self.fc1 = nn.Linear(self.input_dim + 1, hidden_dim)
        else:
            self.fc11 = nn.Linear(self.input_dim, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc13 = nn.Linear(hidden_dim, self.input_dim)
        self.fc21 = nn.Linear(self.input_dim, hidden_dim)
        self.fc22 = nn.Linear(hidden_dim, hidden_dim)
        self.fc23 = nn.Linear(hidden_dim, self.input_dim)
        self.path = nn.Sequential( # define the network for the integration path
        nn.Conv2d(2,4, 3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(4,4, 3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(4,2,1),
        nn.Flatten(),
        nn.Linear(2,2),
        nn.ReLU()
        )

        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()

    def forward(self, t, x):

        # Forward pass of model corresponds to one function evaluation, so
        # increment counter
        self.nfe += 1
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(self.device) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, x], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_x)
        else:
            out = self.fc1(x)
        out = self.non_linearity(out)
        out = self.fc2(out)
        out = self.non_linearity(out)
        out = self.fc3(out)
        return out


def main():
    inner_range = 0.5
    outer_range = [1.0,1.5]
    num_pts_inner = 500
    num_pts_outer = 1500

    r_inner = torch.rand(num_pts_inner,1)*inner_range
    theta_inner = torch.rand(num_pts_inner,1) * 2 * np.pi
    inner_pts_x = r_inner*torch.cos(theta_inner)
    inner_pts_y = r_inner*torch.sin(theta_inner)

    r_outer = torch.rand(num_pts_outer,1)*outer_range[1]
    r_outer[r_outer<outer_range[0]]=((r_outer[r_outer<outer_range[0]]+1)/2)+0.5
    theta_outer = torch.rand(num_pts_outer,1) * 2 * np.pi
    outer_pts_x = r_outer*torch.cos(theta_outer)
    outer_pts_y = r_outer*torch.sin(theta_outer)

#    plt.scatter(outer_pts_x,outer_pts_y)

#    plt.scatter(inner_pts_x,inner_pts_y)

#    plt.show()
    
    gnd_truth = torch.cat((torch.zeros(num_pts_outer,1),torch.ones(num_pts_inner,1)),0)

    dataset = torch.cat((torch.cat((torch.cat((inner_pts_x,inner_pts_y),1),torch.cat((outer_pts_x,outer_pts_y),1)),0),gnd_truth),1)

    dataset=dataset[torch.randperm(dataset.size()[0])]

    gnd_truth = dataset[:,2]
    dataset = dataset[:,:2]

    

if __name__ == '__main__':
    main()
"""