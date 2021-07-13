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

class Grad_net(nn.Module):
    def __init__(self):
        super(Grad_net, self).__init__()
        self.conv1 = nn.Conv2d(7,16,3,1,1)
        self.conv2 = nn.Conv2d(16,16,3,1,1)
        self.conv3 = nn.Conv2d(16,3,3,1,1)
        self.norm1 = nn.BatchNorm2d(7)
        self.norm2 = nn.BatchNorm2d(16)
        self.norm3 = nn.BatchNorm2d(16)
        self.norm4 = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.norm4(x)
        return x

#model = Grad_net()

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class Path_net(nn.Module):# define ode function, this is what we train on
    def __init__(self):
        super(Path_net, self).__init__()
        self.conv1 = nn.Conv2d(4,8,3,1,1)
        self.conv2 = nn.Conv2d(8,8,3,1,1)
        self.conv3 = nn.Conv2d(8,2,3,1,1)
        self.norm1 = nn.BatchNorm2d(4)
        self.norm2 = nn.BatchNorm2d(8)
        self.norm3 = nn.BatchNorm2d(8)
        self.norm4 = nn.BatchNorm2d(2)

    def forward(self, x):
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.norm4(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
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

def train(args, path_net, grad_x_net, grad_y_net, classifier_net, device, train_loader, optimizer, epoch):
#    encoder = encoder.to(device)
#    path_net = path_net.to(device)
#    grad_x_net = grad_x_net.to(device)
#    grad_y_net = grad_y_net.to(device)
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
        p_current = data
        p_i = p_current
        for iter in range(1,int(args.num_eval)+1): # for each random value, integrate from 0 to 1
            t_data_current = torch.cat((iter*dt*torch.ones((64,1,p_current.size(2),p_current.size(3))).to(device),p_i),dim=1) # calculate the current time
            t_data_current = Variable(t_data_current.data, requires_grad=True)
            g_h_current = path_net(t_data_current)
            dg_dt_current = torch.autograd.grad(g_h_current[:,0,:,:].view(g_h_current.size(0),1,g_h_current.size(2),g_h_current.size(3)), t_data_current, grad_outputs= t_data_current[:,0,:,:].view(t_data_current.size(0),1,t_data_current.size(2),t_data_current.size(3)),create_graph=True)[0][:,0]
            dg_dt_current = dg_dt_current.view(dg_dt_current.size(0),1,dg_dt_current.size(1),dg_dt_current.size(2)) # calculate the current dg/dt
            dh_dt_current = torch.autograd.grad(g_h_current[:,1,:,:].view(g_h_current.size(0),1,g_h_current.size(2),g_h_current.size(3)), t_data_current, grad_outputs= t_data_current[:,0,:,:].view(t_data_current.size(0),1,t_data_current.size(2),t_data_current.size(3)),create_graph=True)[0][:,0]
            dh_dt_current = dh_dt_current.view(dh_dt_current.size(0),1,dh_dt_current.size(1),dh_dt_current.size(2))
            in_grad = torch.cat((p_current, g_h_current, dg_dt_current, dh_dt_current), dim=1)
            p_current = p_current + dt*(torch.mul(grad_x_net(in_grad),dg_dt_current) + torch.mul(grad_y_net(in_grad),dh_dt_current))
        output = classifier_net(p_current)
        soft_max = nn.Softmax(dim=1)
        ####### neural path integral ends here #######
        output = soft_max(output)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(args, path_net, grad_x_net, grad_y_net, classifier_net, device, test_loader):
#    encoder = encoder.to(device)
#    path_net = path_net.to(device)
#    grad_x_net = grad_x_net.to(device)
#    grad_y_net = grad_y_net.to(device)
    path_net.eval()
    grad_x_net.eval()
    grad_y_net.eval()
    classifier_net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        dt = ((args.u_bound-args.l_bound)/args.num_eval)
        p_current = data
        p_i = p_current
        for iter in range(1,int(args.num_eval)+1): # for each random value, integrate from 0 to 1
            t_data_current = torch.cat((iter*dt*torch.ones((64,1,p_current.size(2),p_current.size(3))).to(device),p_i),dim=1) # calculate the current time
            t_data_current = Variable(t_data_current.data, requires_grad=True)
            g_h_current = path_net(t_data_current)
            dg_dt_current = torch.autograd.grad(g_h_current[:,0,:,:].view(g_h_current.size(0),1,g_h_current.size(2),g_h_current.size(3)), t_data_current, grad_outputs= t_data_current[:,0,:,:].view(t_data_current.size(0),1,t_data_current.size(2),t_data_current.size(3)),create_graph=True)[0][:,0]
            dg_dt_current = dg_dt_current.view(dg_dt_current.size(0),1,dg_dt_current.size(1),dg_dt_current.size(2)) # calculate the current dg/dt
            dh_dt_current = torch.autograd.grad(g_h_current[:,1,:,:].view(g_h_current.size(0),1,g_h_current.size(2),g_h_current.size(3)), t_data_current, grad_outputs= t_data_current[:,0,:,:].view(t_data_current.size(0),1,t_data_current.size(2),t_data_current.size(3)),create_graph=True)[0][:,0]
            dh_dt_current = dh_dt_current.view(dh_dt_current.size(0),1,dh_dt_current.size(1),dh_dt_current.size(2))
            in_grad = torch.cat((p_current, g_h_current, dg_dt_current, dh_dt_current), dim=1)
            p_current = p_current + dt*(torch.mul(grad_x_net(in_grad),dg_dt_current) + torch.mul(grad_y_net(in_grad),dh_dt_current))
        output = classifier_net(p_current)
        soft_max = nn.Softmax(dim=1)
        ####### neural path integral ends here #######
        output = soft_max(output)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
        pred = p_current.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def validation(args, path_net, grad_x_net, grad_y_net, classifier_net, device, validation_loader):
#    encoder = encoder.to(device)
#    path_net = path_net.to(device)
#    grad_x_net = grad_x_net.to(device)
#    grad_y_net = grad_y_net.to(device)
    path_net.eval()
    grad_x_net.eval()
    grad_y_net.eval()
    classifier_net.eval()
    test_loss = 0
    correct = 0
    for data, target in validation_loader:
        data, target = data.to(device), target.to(device)
        dt = ((args.u_bound-args.l_bound)/args.num_eval)
        p_current = data
        p_i = p_current
        for iter in range(1,int(args.num_eval)+1): # for each random value, integrate from 0 to 1
            t_data_current = torch.cat((iter*dt*torch.ones((64,1,p_current.size(2),p_current.size(3))).to(device),p_i),dim=1) # calculate the current time
            t_data_current = Variable(t_data_current.data, requires_grad=True)
            g_h_current = path_net(t_data_current)
            dg_dt_current = torch.autograd.grad(g_h_current[:,0,:,:].view(g_h_current.size(0),1,g_h_current.size(2),g_h_current.size(3)), t_data_current, grad_outputs= t_data_current[:,0,:,:].view(t_data_current.size(0),1,t_data_current.size(2),t_data_current.size(3)),create_graph=True)[0][:,0]
            dg_dt_current = dg_dt_current.view(dg_dt_current.size(0),1,dg_dt_current.size(1),dg_dt_current.size(2)) # calculate the current dg/dt
            dh_dt_current = torch.autograd.grad(g_h_current[:,1,:,:].view(g_h_current.size(0),1,g_h_current.size(2),g_h_current.size(3)), t_data_current, grad_outputs= t_data_current[:,0,:,:].view(t_data_current.size(0),1,t_data_current.size(2),t_data_current.size(3)),create_graph=True)[0][:,0]
            dh_dt_current = dh_dt_current.view(dh_dt_current.size(0),1,dh_dt_current.size(1),dh_dt_current.size(2))
            in_grad = torch.cat((p_current, g_h_current, dg_dt_current, dh_dt_current), dim=1)
            p_current = p_current + dt*(torch.mul(grad_x_net(in_grad),dg_dt_current) + torch.mul(grad_y_net(in_grad),dh_dt_current))
        output = classifier_net(p_current)
        soft_max = nn.Softmax(dim=1)
        ####### neural path integral ends here #######
        output = soft_max(output)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
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
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--gamma', type=float, default=1e-4, metavar='M',
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
    parser.add_argument('--u-bound', type=float, default=1., help='Upper bound of line integral t value')
    parser.add_argument('--num-eval', type=float, default=20.0, help='Number of evaluations along the line integral')


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

    clipper = WeightClipper()
    path_net = Path_net().to(device)
    path_net.apply(clipper)
    grad_x_net = Grad_net().to(device)
    grad_y_net = Grad_net().to(device)
    classifier_net = Classifier().to(device)
    optimizer = optim.AdamW(list(path_net.parameters())+list(grad_x_net.parameters())+list(grad_y_net.parameters())+list(classifier_net.parameters()), lr=args.lr)
    
    a = get_n_params(path_net)
    b = get_n_params(grad_x_net)
    c = get_n_params(grad_y_net)
    d = get_n_params(classifier_net)
    print(a+b+c+d)

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print('setup complete')
    for epoch in range(1, args.epochs + 1):
        train(args, path_net, grad_x_net, grad_y_net, classifier_net, device, train_loader, optimizer, epoch)
        test(args, path_net, grad_x_net, grad_y_net, classifier_net, device, test_loader)
        scheduler.step()
    test(args, path_net, grad_x_net, grad_y_net, classifier_net, device, test_loader)

    if args.save_model:
        torch.save(path_net.state_dict(), "path_network.pt")
        torch.save(grad_x_net.state_dict(), "grad_x_net.pt")
        torch.save(grad_y_net.state_dict(), "grad_y_net.pt")


if __name__ == '__main__':
    main()