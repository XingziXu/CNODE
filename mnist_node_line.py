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
import numpy as np

class Grad_net(nn.Module): # the Grad_net defines the networks for the path and for the gradients
    def __init__(self, width_path: int, width_grad: int, width_conv2: int):
        super().__init__()
        self.nfe=0 # initialize the number of function evaluations
        
        self.conv2 = nn.Conv2d(1,width_conv2,1)

        self.path = nn.Sequential( # define the network for the integration path
        nn.Conv2d(2,width_path, 3, padding=1, bias=False),
        nn.Hardtanh(),
        nn.Conv2d(width_path,width_path, 3, padding=1, bias=False),
        nn.Hardtanh(),
        nn.Conv2d(width_path,3,1),
        nn.Flatten(),
        nn.Linear(2352,2),
        nn.ReLU6()
        )


        self.grad_g = nn.Sequential( # define the network for the gradient on x direction
            nn.InstanceNorm2d(1),
            nn.Conv2d(1,width_grad,1,1,0),
            nn.ReLU(),
            nn.Conv2d(width_grad,width_grad,3,1,1),
            nn.ReLU(),
            nn.InstanceNorm2d(width_grad),
            nn.Conv2d(width_grad,1,1,1,0)
        )
        
        self.grad_h = nn.Sequential( # define the network for the gradient on y direction
            nn.InstanceNorm2d(1),
            nn.Conv2d(1,width_grad,1,1,0),
            nn.ReLU(),
            nn.Conv2d(width_grad,width_grad,3,1,1),
            nn.ReLU(),
            nn.InstanceNorm2d(width_grad),
            nn.Conv2d(width_grad,1,1,1,0)
        )

    def forward(self, t, x):
        self.nfe+=1 # each time we evaluate the function, the number of evaluations adds one

        t_input = t.expand(x.size(0),1) # resize
        t_channel = ((t_input.view(x.size(0),1,1)).expand(x.size(0),1,x.size(2)*x.size(3))).view(x.size(0),1,x.size(2),x.size(3)) # resize
        path_input = torch.cat((t_channel, p_i),dim=1) # concatenate the time and the image
        g_h_i = self.path(path_input) # calculate the position of the integration path

        dg_dt = g_h_i[:,0].view(g_h_i.size(0),1,1) # resize 
        dg_dt = dg_dt.expand(dg_dt.size(0),1,x.size(2)*x.size(3)) # resize 
        dg_dt = dg_dt.view(dg_dt.size(0),1,x.size(2),x.size(3)) # resize 

        dh_dt = g_h_i[:,1].view(g_h_i.size(0),1,1) # resize 
        dh_dt = dh_dt.expand(dh_dt.size(0),1,x.size(2)*x.size(3)) # resize 
        dh_dt = dh_dt.view(dh_dt.size(0),1,x.size(2),x.size(3)) # resize 

        #di_dt = g_h_i[:,2].view(g_h_i.size(0),1,1) # resize 
        #di_dt = di_dt.expand(di_dt.size(0),1,x.size(2)*x.size(3)) # resize 
        #di_dt = di_dt.view(di_dt.size(0),1,x.size(2),x.size(3)) # resize 
        
        dp = torch.mul(self.grad_g(x),dg_dt) + torch.mul(self.grad_g(x),dh_dt)# + torch.mul(self.grad_g(x),di_dt) # calculate the change in p
        #print(t.item())
        return dp

class Classifier(nn.Module): # define the linear classifier
    def __init__(self, width_conv2: int, width_pool: int):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(width_conv2*width_pool*width_pool,10)
        self.pool = nn.AdaptiveAvgPool2d(width_pool)

    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x,1) # flatten the input image&dimension into a vector
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
    output = classifier_net(grad_net.conv2(p.view(p.size(0),1,p.size(1),p.size(2)))) # classify the transformed images
    soft_max = nn.Softmax(dim=1) # define a soft max calculator
    output = soft_max(output) # get the prediction results by getting the most probable ones
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
    output = classifier_net(grad_net.conv2(p.view(p.size(0),1,p.size(1),p.size(2)))) # classify the transformed images
    soft_max = nn.Softmax(dim=1) # define a soft max calculator
    output = soft_max(output) # get the prediction results by getting the most probable ones
    return output

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
    return loss_grad.item()

def test(args, grad_net, classifier_net, device, test_loader):
    grad_net.eval() # set the network on evaluation mode
    classifier_net.eval() # set the network on evaluation mode
    test_loss = 0 # initialize test loss
    correct = 0 # initialize the number of correct predictions
    for data, target in test_loader: # for each data batch
        data, target = data.to(device), target.to(device) # assign data to the device
        global p_i # claim the initial image batch as a global variable
        p_i = data
        output = evaluate(args, grad_net, classifier_net, data, device)
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
    for data, target in validation_loader: # for each data batch
        data, target = data.to(device), target.to(device) # assign data to the device
        global p_i # claim the initial image batch as a global variable
        p_i = data
        output = evaluate(args, grad_net, classifier_net, data, device)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item() # sum up the number of correct predictions

    test_loss /= len(validation_loader.dataset) # calculate test loss

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( # print test loss and accuracy
        test_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))
    
    if args.save_model: # check if we are saving the model
        torch.save(grad_net.state_dict(), "grad_net.pt") # save gradients and path model
        torch.save(classifier_net.state_dict(), "classifer_net.pt") # save classifier model
        print("The current models are saved") # confirm all models are saved
    return test_loss, 100. * correct / len(validation_loader.dataset)

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
    parser.add_argument('--no-cuda', action='store_true', default=False,
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

    transform=transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, download=True,
                       transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

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

    loss_train = []
    loss_test = []
    accu = []
    #outer = torch.zeros((25,153,3))
    #inner = torch.zeros((25,147,3))
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, grad_net, classifier_net, device, train_loader, optimizer_grad, epoch)
        test_loss,accu_new = validation(args, grad_net, classifier_net, device, test_loader)
        #outer[epoch-1,:,:] = o1[o1[:,2]==1.]
        #inner[epoch-1,:,:] = o1[o1[:,2]==0.]
        loss_train.append(train_loss)
        loss_test.append(test_loss)
        accu.append(accu_new)
        #if accu_new > accu:
        #    accu = accu_new
        #print('The best accuracy is {:.4f}%\n'.format(accu))
        scheduler_grad.step()
    #test(args, grad_net, classifier_net, device, test_loader)
    with open('train_loss_mnist_line4.npy', 'wb') as f:
        np.save(f, np.asarray(loss_train))
    with open('test_loss_mnist_line4.npy', 'wb') as f:
        np.save(f, np.asarray(loss_test))
    with open('accuracy_mnist_line4.npy', 'wb') as f:
        np.save(f, np.asarray(accu))

if __name__ == '__main__':
    main()
