from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchdiffeq import odeint_adjoint as odeint
from scipy.integrate import odeint as odeint_scipy
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

class Grad_net(nn.Module):
    def __init__(self, input_size : int, width : int, output_size : int):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_size,width),
            nn.ReLU(),
            nn.GroupNorm(1,width),
           # nn.LayerNorm(width),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.GroupNorm(1,width),
           # nn.LayerNorm(width),
            nn.Linear(width,output_size),
            nn.Tanhshrink()
        )

    def forward(self,x):
        y_pred = self.stack(x)
        return y_pred

#model = Grad_net()

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class ODEFunc(nn.Module):# define ode function, this is what we train on

    def __init__(self, input_size : int, width : int, output_size : int):
        super(ODEFunc, self).__init__()
        self.l1 = nn.Linear(input_size, width)
        self.l2 = nn.Linear(width,width)
        self.l3 = nn.Linear(width, output_size)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, t):
        t = self.l1(t)
        t = F.relu(t)
        t = self.norm1(t)
        t = self.l2(t)
        t = F.relu(t)
        t = self.norm2(t)
        t = self.l3(t)
        t = F.relu(t)
        return t

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3,8,3,1)
        self.conv2 = nn.Conv2d(8,8,3,1)
        self.fc1 = nn.Linear(1568,128)
        self.fc2 = nn.Linear(128,10)
        self.norm1 = nn.BatchNorm2d(8)
        self.norm2 = nn.BatchNorm2d(8)
        self.norm3 = nn.GroupNorm(8,128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.norm2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.norm3(x)
        x = self.fc2(x)
        return x

class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(0, float('inf'))
            module.weight.data = w

def train(args, encoder, path_net, grad_x_net, grad_y_net, device, train_loader, optimizer, epoch):
#    encoder = encoder.to(device)
#    path_net = path_net.to(device)
#    grad_x_net = grad_x_net.to(device)
#    grad_y_net = grad_y_net.to(device)
    encoder.train()
    path_net.train()
    grad_x_net.train()
    grad_y_net.train()
    #print(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        #if batch_idx > 100:
        #    break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        ####### neural path integral starts here #######
        dt = ((args.u_bound-args.l_bound)/args.num_eval)
        p_current = encoder(data)
        p_i = p_current
        p_current = torch.cat((p_current,torch.zeros(p_current.size(0),2).to(device)),dim=1) # augment here
        for iter in range(1,int(args.num_eval)+1): # for each random value, integrate from 0 to 1
            t_data_current = torch.cat((iter*dt*torch.ones((p_current.size(0),1)).to(device),p_i),dim=1) # calculate the current time
            t_data_current = Variable(t_data_current.data, requires_grad=True)
            g_h_current = path_net(t_data_current)
            dg_dt_current = torch.autograd.grad(g_h_current[:,0].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
            dg_dt_current = dg_dt_current.view(dg_dt_current.size(0),1) # calculate the current dg/dt
            dh_dt_current = torch.autograd.grad(g_h_current[:,1].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
            dh_dt_current = dh_dt_current.view(dh_dt_current.size(0),1)
            in_grad = torch.cat((p_current.view(p_current.size()[0], p_current.size()[1]), g_h_current), dim=1)
            p_current = p_current + dt*(grad_x_net(in_grad)*dg_dt_current + grad_y_net(in_grad)*dh_dt_current)
        soft_max = nn.Softmax(dim=1)
        p_current = p_current[:,0:10] # the first ten are features
        ####### neural path integral ends here #######
        p_current = soft_max(p_current)
        loss = F.cross_entropy(p_current, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(args, encoder, path_net, grad_x_net, grad_y_net, device, test_loader):
#    encoder = encoder.to(device)
#    path_net = path_net.to(device)
#    grad_x_net = grad_x_net.to(device)
#    grad_y_net = grad_y_net.to(device)
    encoder.eval()
    path_net.eval()
    grad_x_net.eval()
    grad_y_net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        dt = ((args.u_bound-args.l_bound)/args.num_eval)
        p_current = encoder(data)
        p_i = p_current
        p_current = torch.cat((p_current,torch.zeros(p_current.size(0),2).to(device)),dim=1) # augment here
        for iter in range(1,int(args.num_eval)+1): # for each random value, integrate from 0 to 1
            t_data_current = torch.cat((iter*dt*torch.ones((p_current.size(0),1)).to(device),p_i),dim=1) # calculate the current time
            t_data_current = Variable(t_data_current.data, requires_grad=True)
            g_h_current = path_net(t_data_current)
            dg_dt_current = torch.autograd.grad(g_h_current[:,0].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
            dg_dt_current = dg_dt_current.view(dg_dt_current.size(0),1) # calculate the current dg/dt
            dh_dt_current = torch.autograd.grad(g_h_current[:,1].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
            dh_dt_current = dh_dt_current.view(dh_dt_current.size(0),1)
            in_grad = torch.cat((p_current.view(p_current.size()[0], p_current.size()[1]), g_h_current), dim=1)
            p_current = p_current + dt*(grad_x_net(in_grad)*dg_dt_current + grad_y_net(in_grad)*dh_dt_current)
        soft_max = nn.Softmax(dim=1)
        p_current = p_current[:,0:10] # the first ten are features
        ####### neural path integral ends here #######
        p_current = soft_max(p_current)
        test_loss += F.cross_entropy(p_current, target, reduction='sum').item()  # sum up batch loss
        pred = p_current.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def validation(args, encoder, path_net, grad_x_net, grad_y_net, device, validation_loader):
#    encoder = encoder.to(device)
#    path_net = path_net.to(device)
#    grad_x_net = grad_x_net.to(device)
#    grad_y_net = grad_y_net.to(device)
    encoder.eval()
    path_net.eval()
    grad_x_net.eval()
    grad_y_net.eval()
    test_loss = 0
    correct = 0
    for data, target in validation_loader:
        data, target = data.to(device), target.to(device)
        dt = ((args.u_bound-args.l_bound)/args.num_eval)
        p_current = encoder(data)
        p_i = p_current
        p_current = torch.cat((p_current,torch.zeros(p_current.size(0),2).to(device)),dim=1) # augment here
        for iter in range(1,int(args.num_eval)+1): # for each random value, integrate from 0 to 1
            t_data_current = torch.cat((iter*dt*torch.ones((p_current.size(0),1)).to(device),p_i),dim=1) # calculate the current time
            t_data_current = Variable(t_data_current.data, requires_grad=True)
            g_h_current = path_net(t_data_current)
            dg_dt_current = torch.autograd.grad(g_h_current[:,0].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
            dg_dt_current = dg_dt_current.view(dg_dt_current.size(0),1) # calculate the current dg/dt
            dh_dt_current = torch.autograd.grad(g_h_current[:,1].view(g_h_current.size(0),1), t_data_current, grad_outputs= t_data_current[:,0].view(t_data_current.size(0),1),create_graph=True)[0][:,0]
            dh_dt_current = dh_dt_current.view(dh_dt_current.size(0),1)
            in_grad = torch.cat((p_current.view(p_current.size()[0], p_current.size()[1]), g_h_current), dim=1)
            p_current = p_current + dt*(grad_x_net(in_grad)*dg_dt_current + grad_y_net(in_grad)*dh_dt_current)
        soft_max = nn.Softmax(dim=1)
        p_current = p_current[:,0:10] # the first ten are features
        ####### neural path integral ends here #######
        p_current = soft_max(p_current)
        test_loss += F.cross_entropy(p_current, target, reduction='sum').item()  # sum up batch loss
        pred = p_current.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(validation_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--validation-batch-size', type=int, default=1000, metavar='V',
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--step-size', type=int, default=10, metavar='M',
                        help='how many epochs to we change the learning rate, default is 5')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--l-bound', type=float, default=0., help='Lower bound of line integral t value')
    parser.add_argument('--u-bound', type=float, default=10., help='Upper bound of line integral t value')
    parser.add_argument('--num-eval', type=float, default=1.0, help='Number of evaluations along the line integral')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    validation_kwargs = {'batch_size': args.validation_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 12,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        validation_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.CIFAR10('../data', train=False, download=True,
                       transform=transform)
 
    #dataset4, dataset2 = torch.utils.data.random_split(dataset2, [9990,10])

    #dataset3, dataset1 = torch.utils.data.random_split(dataset1, [10000,40000]) # dataset 1 is training, dataset 2 is testing, dataset 3 is validation

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    #validation_loader = torch.utils.data.DataLoader(dataset3, **validation_kwargs)

    encoder = Encoder().to(device)
    input_size_path = 11
    width_path = 64
    output_size_path = 2
    input_size_grad = 14
    width_grad = 64
    output_size_grad = 12
    clipper = WeightClipper()
    path_net = ODEFunc(input_size_path, width_path, output_size_path).to(device)
    path_net.apply(clipper)
    grad_x_net = Grad_net(input_size_grad, width_grad, output_size_grad).to(device)
    grad_y_net = Grad_net(input_size_grad, width_grad, output_size_grad).to(device)

    a = get_n_params(encoder)
    b = get_n_params(path_net)
    c = get_n_params(grad_x_net)
    d = get_n_params(grad_y_net)
    print(a+b+c+d)

    num_trial = 40
    lr_mat = np.linspace(1e-5, 1e-2, num_trial)
    results = np.zeros((num_trial,args.epochs))

    for lr_index, lr_current in enumerate(lr_mat): 
        optimizer = optim.AdamW(list(encoder.parameters())+list(path_net.parameters())+list(grad_x_net.parameters())+list(grad_y_net.parameters()), lr=lr_current)
        
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        print('setup complete for learning rate', lr_current.item(), ' cycle', lr_index+1, ' of', num_trial)
        for epoch in range(1, args.epochs + 1):
            train(args, encoder, path_net, grad_x_net, grad_y_net, device, train_loader, optimizer, epoch)
            results[lr_index, epoch-1] = test(args, encoder, path_net, grad_x_net, grad_y_net, device, test_loader)
            scheduler.step()
        #test(args, encoder, path_net, grad_x_net, grad_y_net, device, test_loader)

    with open('learning_rates.npy', 'wb') as f:
        np.save(f, lr_mat)

    with open('results.npy', 'wb') as f:
        np.save(f, results)

    #for i in range(0,args.epochs-1):
    #    plt.plot(lr_mat, results[:,i])
    #    plt.legend(str(i+1))
    
    #plt.show()

    if args.save_model:
        torch.save(encoder.state_dict(), "mnist_cnn.pt")
        torch.save(path_net.state_dict(), "path_network.pt")
        torch.save(grad_x_net.state_dict(), "grad_x_net.pt")
        torch.save(grad_y_net.state_dict(), "grad_y_net.pt")


if __name__ == '__main__':
    main()
