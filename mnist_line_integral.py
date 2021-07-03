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

class Grad_net(nn.Module):
    def __init__(self, input_size : int, width : int, output_size : int):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(input_size,width),
            nn.ReLU(),
            nn.GroupNorm(1,32),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.GroupNorm(1,32),
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

        self.net = nn.Sequential(
            nn.Linear(input_size, width),
            nn.ReLU(),
            nn.GroupNorm(2,32),
            nn.Linear(width,width),
            nn.ReLU(),
            nn.GroupNorm(2,32),
            nn.Linear(width, output_size),
            nn.ReLU()
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        return self.net((t+torch.Tensor([[0.]])).to(device))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.norm1 = nn.GroupNorm(32,32)
        self.norm2 = nn.GroupNorm(32,64)
        self.norm3 = nn.GroupNorm(32,128)

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


def train(args, encoder, path_net, grad_x_net, grad_y_net, device, train_loader, optimizer, epoch):
#    encoder = encoder.to(device)
#    path_net = path_net.to(device)
#    grad_x_net = grad_x_net.to(device)
#    grad_y_net = grad_y_net.to(device)
    encoder.train()
    path_net.train()
    grad_x_net.train()
    grad_y_net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #if batch_idx > 100:
        #    break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        ####### neural path integral starts here #######
        #num_eval = 1e3
        #l_bound = 0.
        #u_bound = 1.
        dt = ((args.u_bound-args.l_bound)/args.num_eval)
        g_h_0 = torch.Tensor([[0.,0.]]).to(device)
        p_current = encoder(data).to(device)
        for iter in range(1,int(args.num_eval)+1): # for each random value, integrate from 0 to 1
            t_current = iter*dt*torch.ones((1)).to(device) # calculate the current time
            t_calc = iter*dt*torch.Tensor([0.,1.]).to(device)
            dg_dh_dt_current = path_net(1,torch.Tensor([[t_current]])).to(device) # calculate the current dg/dt
            g_h_current = torch.squeeze(odeint(path_net, g_h_0, t_calc, method='dopri5')[1])
            in_grad = torch.cat((p_current.view(p_current.size()[0], 10), g_h_current.repeat([p_current.size()[0],1]).view(p_current.size()[0],2)), dim=1).to(device)
            #p_current = p_current + dt*(torch.dot(torch.cat((grad_x_net(in_grad), grad_y_net(in_grad)),dim=1),dg_dh_dt_current))
            p_current = p_current + dt*(grad_x_net(in_grad)*dg_dh_dt_current[0][0] + grad_y_net(in_grad)*dg_dh_dt_current[0][1]).to(device)
        soft_max = nn.Softmax(dim=1).to(device)
        p_current = soft_max(p_current).to(device)
        ####### neural path integral ends here #######
        loss = F.nll_loss(p_current, target)
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
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            dt = ((args.u_bound-args.l_bound)/args.num_eval)
            g_h_0 = torch.Tensor([[0.,0.]]).to(device)
            p_current = encoder(data).to(device)
            for iter in range(1,int(args.num_eval)+1): # for each random value, integrate from 0 to 1
                t_current = iter*dt*torch.ones((1)).to(device) # calculate the current time
                t_calc = iter*dt*torch.Tensor([0.,1.]).to(device)
                dg_dh_dt_current = path_net(1,torch.Tensor([[t_current]])).to(device) # calculate the current dg/dt
                g_h_current = torch.squeeze(odeint(path_net, g_h_0, t_calc, method='dopri5')[1])
                in_grad = torch.cat((p_current.view(p_current.size()[0], 10), g_h_current.repeat([p_current.size()[0],1]).view(p_current.size()[0],2)), dim=1).to(device)
                #p_current = p_current + dt*(torch.dot(torch.cat((grad_x_net(in_grad), grad_y_net(in_grad)),dim=1),dg_dh_dt_current))
                p_current = p_current + dt*(grad_x_net(in_grad)*dg_dh_dt_current[0][0] + grad_y_net(in_grad)*dg_dh_dt_current[0][1]).to(device)
            soft_max = nn.Softmax(dim=1).to(device)
            p_current = soft_max(p_current).to(device)
            test_loss += F.nll_loss(p_current, target, reduction='sum').item().to(device)  # sum up batch loss
            pred = p_current.argmax(dim=1, keepdim=True).to(device)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            dt = (args.u_bound-args.l_bound)/args.num_eval
            g_h_0 = torch.Tensor([[0.,0.]]).to(device)
            p_current = encoder(data).to(device)
            for iter in range(1,int(args.num_eval)+1): # for each random value, integrate from 0 to 1
                t_current = iter*dt*torch.ones((1)).to(device) # calculate the current time
                t_calc = iter*dt*torch.Tensor([0.,1.]).to(device)
                dg_dh_dt_current = path_net(1,torch.Tensor([[t_current]])).to(device) # calculate the current dg/dt
                g_h_current = torch.squeeze(odeint(path_net, g_h_0, t_calc, method='dopri5')[1])
                in_grad = torch.cat((p_current.view(p_current.size()[0], 10), g_h_current.repeat([p_current.size()[0],1]).view(p_current.size()[0],2)), dim=1).to(device)
                #p_current = p_current + dt*(torch.dot(torch.cat((grad_x_net(in_grad), grad_y_net(in_grad)),dim=1),dg_dh_dt_current))
                p_current = p_current + dt*(grad_x_net(in_grad)*dg_dh_dt_current[0][0] + grad_y_net(in_grad)*dg_dh_dt_current[0][1]).to(device)
            soft_max = nn.Softmax(dim=1)
            p_current = soft_max(p_current).to(device)
            test_loss += F.nll_loss(p_current, target, reduction='sum').item().to(device)  # sum up batch loss
            pred = p_current.argmax(dim=1, keepdim=True).to(device)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(validation_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
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
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
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
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--l-bound', type=float, default=0., help='Lower bound of line integral t value')
    parser.add_argument('--u-bound', type=float, default=1., help='Upper bound of line integral t value')
    parser.add_argument('--num-eval', type=float, default=1e2, help='Number of evaluations along the line integral')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    validation_kwargs = {'batch_size': args.validation_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        validation_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, download=True,
                       transform=transform)

    dataset3, dataset1 = torch.utils.data.random_split(dataset1, [10000,50000])

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    validation_loader = torch.utils.data.DataLoader(dataset3, **validation_kwargs)

    encoder = Net().to(device)
    input_size_path = 1
    width_path = 32
    output_size_path = 2
    input_size_grad = 12
    width_grad = 32
    output_size_grad = 10
    path_net = ODEFunc(input_size_path, width_path, output_size_path).to(device)
    grad_x_net = Grad_net(input_size_grad, width_grad, output_size_grad).to(device)
    grad_y_net = Grad_net(input_size_grad, width_grad, output_size_grad).to(device)
    optimizer = optim.SGD(list(encoder.parameters())+list(path_net.parameters())+list(grad_x_net.parameters())+list(grad_y_net.parameters()), lr=args.lr)
    
    a=get_n_params(encoder)
    b = get_n_params(path_net)
    c = get_n_params(grad_x_net)
    d = get_n_params(grad_y_net)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    print('setup complete')
    for epoch in range(1, args.epochs + 1):
        train(args, encoder, path_net, grad_x_net, grad_y_net, device, train_loader, optimizer, epoch)
        test(args, encoder, path_net, grad_x_net, grad_y_net, device, test_loader)
        scheduler.step()
    validation(args, encoder, path_net, grad_x_net, grad_y_net, device, validation_loader)

    if args.save_model:
        torch.save(encoder.state_dict(), "mnist_cnn.pt")
        torch.save(path_net.state_dict(), "path_network.pt")
        torch.save(grad_x_net.state_dict(), "grad_x_net.pt")
        torch.save(grad_y_net.state_dict(), "grad_y_net.pt")


if __name__ == '__main__':
    main()
