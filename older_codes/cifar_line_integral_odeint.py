from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchdiffeq import odeint as odeint
from scipy.integrate import odeint as odeint_scipy

class Grad_net(nn.Module):
    def __init__(self, input_size_path : int, width_path : int, output_size_path : int, input_size_grad : int, width_grad : int, output_size_grad : int):
        super().__init__()
        self.path = nn.Sequential(
            nn.Linear(input_size_path, width_path),
            nn.ReLU(),
            nn.Linear(width_path,width_path),
            nn.ReLU(),
            nn.Linear(width_path, output_size_path),
            nn.ReLU()
        )

        self.grad_x = nn.Sequential(
            nn.Linear(input_size_grad,width_grad),
            nn.ReLU(),
            nn.GroupNorm(1,64),
            nn.Linear(width_grad,width_grad),
            nn.ReLU(),
            nn.GroupNorm(1,64),
            nn.Linear(width_grad,output_size_grad),
            nn.Tanh()
        )

        self.grad_y = nn.Sequential(
            nn.Linear(input_size_grad,width_grad),
            nn.ReLU(),
            nn.GroupNorm(1,64),
            nn.Linear(width_grad,width_grad),
            nn.ReLU(),
            nn.GroupNorm(1,64),
            nn.Linear(width_grad,output_size_grad),
            nn.Tanh()
        )

    def forward(self,t,p_current):
        t = t.view(1) # calculate the current time
        g_h_current = self.path(t)
        torch.sum(g_h_current).backward(retain_graph=True)
        dg_dt_current = torch.autograd.grad(g_h_current[0], t, retain_graph = True)[0] # calculate the current dg/dt
        dh_dt_current = torch.autograd.grad(g_h_current[1], t, retain_graph = True)[0]
        in_grad = torch.cat((p_current.view(p_current.size()[0], 10), g_h_current.repeat([p_current.size()[0],1]).view(p_current.size()[0],2)), dim=1)
        dpdt = self.grad_x(in_grad)*dg_dt_current + self.grad_y(in_grad)*dh_dt_current
        return dpdt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(12544, 256)
        self.fc2 = nn.Linear(256, 10)
        self.norm1 = nn.GroupNorm(32,32)
        self.norm2 = nn.GroupNorm(32,64)
        self.norm3 = nn.GroupNorm(64,256)

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



def train(args, encoder, grad_net, device, train_loader, optimizer, epoch):
    encoder.train()
    grad_net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #if batch_idx > 100:
        #    break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        ####### neural path integral starts here #######
        p_current = encoder(data)
        t = torch.Tensor([0.,1.]).to(device)
        t.requires_grad=True
        p_current = torch.squeeze(odeint(grad_net, p_current, t)[1])
        soft_max = nn.Softmax(dim=1)
        ####### neural path integral ends here #######
        p_current = soft_max(p_current)
        loss = F.nll_loss(p_current, target)
        loss.backward()
        optimizer.step()
        constraintLow = 0
        constraintHigh = float('inf')
        clipper = WeightClipper()
        grad_net.path.apply(clipper)
       # grad_net.path[1].weight=torch.nn.Parameter(constraintLow + (constraintHigh-constraintLow)*(grad_net.path.weight - torch.min(grad_net.path.weight))/(torch.max(grad_net.path.weight) - torch.min(grad_net.path.weight)))
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(args, encoder, grad_net, device, test_loader):
#    encoder = encoder.to(device)
#    path_net = path_net.to(device)
#    grad_x_net = grad_x_net.to(device)
#    grad_y_net = grad_y_net.to(device)
    encoder.eval()
    grad_net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        ####### neural path integral starts here #######
        p_current = encoder(data)
        t = torch.Tensor([0.,1.]).to(device)
        t.requires_grad=True
        p_current = torch.squeeze(odeint(grad_net, p_current, t)[0])
        soft_max = nn.Softmax(dim=1)
        ####### neural path integral ends here #######
        p_current = soft_max(p_current)
        test_loss += F.nll_loss(p_current, target, reduction='sum').item()  # sum up batch loss
        pred = p_current.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def validation(args, encoder, grad_net, device, validation_loader):
#    encoder = encoder.to(device)
#    path_net = path_net.to(device)
#    grad_x_net = grad_x_net.to(device)
#    grad_y_net = grad_y_net.to(device)
    encoder.eval()
    grad_net.eval()
    test_loss = 0
    correct = 0
    for data, target in validation_loader:
        data, target = data.to(device), target.to(device)
        ####### neural path integral starts here #######
        p_current = encoder(data)
        t = torch.Tensor([0.,1.]).to(device)
        t.requires_grad=True
        p_current = torch.squeeze(odeint(grad_net, p_current, t)[1])
        soft_max = nn.Softmax(dim=1)
        ####### neural path integral ends here #######
        p_current = soft_max(p_current)
        test_loss += F.nll_loss(p_current, target, reduction='sum').item()  # sum up batch loss
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
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--lr', type=float, default=1e-1, metavar='LR',
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
    dataset1 = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.CIFAR10('../data', train=False, download=True,
                       transform=transform)
 
#    dataset4, dataset2 = torch.utils.data.random_split(dataset2, [9990,10])

    dataset3, dataset1 = torch.utils.data.random_split(dataset1, [10000,40000]) # dataset 1 is training, dataset 2 is testing, dataset 3 is validation

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    validation_loader = torch.utils.data.DataLoader(dataset3, **validation_kwargs)

    encoder = Net().to(device)
    input_size_path = 1
    width_path = 64
    output_size_path = 2
    input_size_grad = 12
    width_grad = 64
    output_size_grad = 10
    grad_net = Grad_net(input_size_path, width_path, output_size_path, input_size_grad, width_grad, output_size_grad).to(device)
    optimizer = optim.SGD(list(encoder.parameters())+list(grad_net.parameters()), lr=args.lr)
    
    a=get_n_params(encoder)
    b = get_n_params(grad_net)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    print('setup complete')
    for epoch in range(1, args.epochs + 1):
        train(args, encoder, grad_net, device, train_loader, optimizer, epoch)
        validation(args, encoder, grad_net, device, test_loader)
        scheduler.step()
    test(args, encoder, grad_net, device, validation_loader)

    if args.save_model:
        torch.save(encoder.state_dict(), "encoder.pt")
        torch.save(grad_net.state_dict(), "grad_net.pt")


if __name__ == '__main__':
    main()