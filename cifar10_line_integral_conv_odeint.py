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
from torch.autograd import Variable


class Grad_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.path = nn.Sequential(
        #nn.Linear(1,16),
        #nn.ReLU(),
        #nn.Linear(16,16),
        #nn.ReLU(),
        #nn.Linear(16,2),
        #nn.GroupNorm(2,2),
        #nn.ReLU(),
        nn.Conv2d(4,16,1,1,0),
        #nn.GroupNorm(2,4),
        nn.ReLU(),
        nn.Conv2d(16,16,3,1,1),
        nn.ReLU(),
        nn.Conv2d(16,3,1,1,0),
        #nn.Conv2d(2,2,3,1,1),
        #nn.GroupNorm(2,4),
        #nn.ReLU(),
        #nn.Conv2d(4,2,1,1,0),
        nn.Flatten(),
        nn.Linear(3072,3)
#        nn.ReLU(),
#        nn.Linear(16,2)
        )
        self.grad_x = nn.Sequential(
            #nn.GroupNorm(3,3),
            #nn.ReLU(),
            nn.Conv2d(6,64,1,1,0),
            #nn.GroupNorm(4,16),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            #nn.GroupNorm(4,16),
            nn.ReLU(),
            nn.Conv2d(64,3,1,1,0)
        )
        self.grad_y = nn.Sequential(
            #nn.GroupNorm(3,3),
            #nn.ReLU(),
            nn.Conv2d(6,64,1,1,0),
            #nn.GroupNorm(4,16),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            #nn.GroupNorm(4,16),
            nn.ReLU(),
            nn.Conv2d(64,3,1,1,0)
        )
        self.grad_z = nn.Sequential(
            nn.Conv2d(6,64,1,1,0),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            nn.Conv2d(64,3,1,1,0)
        )


    def forward(self, t, x):
        device = torch.device("cuda")
        #device = torch.device("cpu")
        t_input = t.expand(x.size(0),1)
        #p_i.size()
        #x_ori = x[:,0,:,:].view(x.size(0),1,x.size(2),x.size(3)).to(device)
        t_channel = ((t_input.view(x.size(0),1,1)).expand(x.size(0),1,x.size(2)*x.size(3))).view(x.size(0),1,x.size(2),x.size(3))
        #t_channel.requires_grad = True
        path_input = torch.cat((t_channel, p_i),dim=1)
        #path_input.requires_grad=True
        g_h_current = self.path(path_input)
        dg_dt_current = torch.autograd.grad(g_h_current[:,0].view(g_h_current.size(0),1), t_input, grad_outputs=torch.ones(x.size(0),1).to(device), create_graph=True)[0]
        dg_dt_current = dg_dt_current.view(dg_dt_current.size(0),1,1)
        dg_dt_current = dg_dt_current.expand(dg_dt_current.size(0),1,x.size(2)*x.size(3))
        dg_dt_current = dg_dt_current.view(dg_dt_current.size(0),1,x.size(2),x.size(3))
        dh_dt_current = torch.autograd.grad(g_h_current[:,1].view(g_h_current.size(0),1), t_input, grad_outputs=torch.ones(x.size(0),1).to(device), create_graph=True)[0]
        dh_dt_current = dh_dt_current.view(dh_dt_current.size(0),1,1)
        dh_dt_current = dh_dt_current.expand(dh_dt_current.size(0),1,x.size(2)*x.size(3))
        dh_dt_current = dh_dt_current.view(dh_dt_current.size(0),1,x.size(2),x.size(3))
        di_dt_current = torch.autograd.grad(g_h_current[:,2].view(g_h_current.size(0),1), t_input, grad_outputs=torch.ones(x.size(0),1).to(device), create_graph=True)[0]
        di_dt_current = di_dt_current.view(di_dt_current.size(0),1,1)
        di_dt_current = di_dt_current.expand(di_dt_current.size(0),1,x.size(2)*x.size(3))
        di_dt_current = di_dt_current.view(di_dt_current.size(0),1,x.size(2),x.size(3))
        #dj_dt_current = torch.autograd.grad(g_h_current[:,3].view(g_h_current.size(0),1), t_input, grad_outputs=torch.ones(x.size(0),1).to(device), create_graph=True)[0]
        #dj_dt_current = dj_dt_current.view(dj_dt_current.size(0),1,1)
        #dj_dt_current = dj_dt_current.expand(dj_dt_current.size(0),1,x.size(2)*x.size(3))
        #dj_dt_current = dj_dt_current.view(dj_dt_current.size(0),1,x.size(2),x.size(3))
        #dk_dt_current = torch.autograd.grad(g_h_current[:,4].view(g_h_current.size(0),1), t_input, grad_outputs=torch.ones(x.size(0),1).to(device), create_graph=True)[0]
        #dk_dt_current = dk_dt_current.view(dk_dt_current.size(0),1,1)
        #dk_dt_current = dk_dt_current.expand(dk_dt_current.size(0),1,x.size(2)*x.size(3))
        #dk_dt_current = dk_dt_current.view(dk_dt_current.size(0),1,x.size(2),x.size(3))
        
        g_h_current_input = g_h_current.view(g_h_current.size(0),g_h_current.size(1),1)
        g_h_current_input = g_h_current_input.expand(g_h_current.size(0),g_h_current.size(1),x.size(2)*x.size(3))
        g_h_current_input = g_h_current_input.view((g_h_current.size(0),g_h_current.size(1),x.size(2),x.size(3)))
        x_aug=torch.cat((x,g_h_current_input),dim=1)
        #in_grad = torch.cat((x, g_h_current_input), dim=1)
        #in_grad = torch.cat((x.view(x.size()[0], 10), g_h_current.repeat([x.size()[0],1]).view(x.size()[0],2)), dim=1)
        dpdt = torch.mul(self.grad_x(x_aug),dg_dt_current) + torch.mul(self.grad_y(x_aug),dh_dt_current) + torch.mul(self.grad_z(x_aug),di_dt_current)# + torch.mul(self.grad_x(x_aug),dj_dt_current) + torch.mul(self.grad_x(x_aug),dk_dt_current)
        #print(t.item())
        return dpdt

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(3072,10)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = self.classifier(x)
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

def train(args, grad_net, classifier_net, device, train_loader, optimizer, epoch):
#    encoder = encoder.to(device)
#    path_net = path_net.to(device)
#    grad_x_net = grad_x_net.to(device)
#    grad_y_net = grad_y_net.to(device)
    grad_net.train()
    classifier_net.train()
    #print(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        #print('we are training')
        #if batch_idx > 100:
        #    break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        ####### neural path integral starts here #######
        p_current = data
        p_current.requires_grad=True
        global p_i
        p_i = p_current
        #aug = torch.zeros(p_current.size(0),5,p_current.size(2),p_current.size(3)).to(device)
        #p_current = torch.cat((p_current,aug),dim=1)
        t = torch.Tensor([0.,1.]).to(device)
        t.requires_grad=True
        #p_current = torch.squeeze(odeint(grad_net, p_current, t,method="euler")[1])
        p_current = torch.squeeze(odeint(grad_net, p_current, t,method="dopri5",rtol=1e-3,atol=1e-3)[1])
        output = classifier_net(p_current)
        soft_max = nn.Softmax(dim=1)
        ####### neural path integral ends here #######
        output = soft_max(output)
        #print('2')
        loss = F.cross_entropy(output, target)
        loss.backward()
        #print('3')
        optimizer.step()
        #print('4')
        #clipper = WeightClipper()
        #grad_net.path.apply(clipper)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(args, grad_net, classifier_net, device, test_loader):
#    encoder = encoder.to(device)
#    path_net = path_net.to(device)
#    grad_x_net = grad_x_net.to(device)
#    grad_y_net = grad_y_net.to(device)
    grad_net.eval()
    classifier_net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        p_current = data
        p_current.requires_grad=True
        global p_i
        p_i = p_current
        #aug = torch.zeros(p_current.size(0),5,p_current.size(2),p_current.size(3)).to(device)
        #p_current = torch.cat((p_current,aug),dim=1)
        t = torch.Tensor([0.,1.]).to(device)
        t.requires_grad=True
        #p_current = torch.squeeze(odeint(grad_net, p_current, t,method="euler")[1])
        p_current = torch.squeeze(odeint(grad_net, p_current, t,method="dopri5",rtol=1e-3,atol=1e-3)[1])
        output = classifier_net(p_current)
        soft_max = nn.Softmax(dim=1)
        ####### neural path integral ends here #######
        output = soft_max(output)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def validation(args, grad_net, classifier_net, device, validation_loader):
#    encoder = encoder.to(device)
#    path_net = path_net.to(device)
#    grad_x_net = grad_x_net.to(device)
#    grad_y_net = grad_y_net.to(device)
    grad_net.eval()
    classifier_net.eval()
    test_loss = 0
    correct = 0
    for data, target in validation_loader:
        data, target = data.to(device), target.to(device)
        p_current = data
        p_current.requires_grad=True
        global p_i
        p_i = p_current
        #aug = torch.zeros(p_current.size(0),5,p_current.size(2),p_current.size(3)).to(device)
        #p_current = torch.cat((p_current,aug),dim=1)
        t = torch.Tensor([0.,1.]).to(device)
        t.requires_grad=True
        #p_current = torch.squeeze(odeint(grad_net, p_current, t,method="euler")[1])
        p_current = torch.squeeze(odeint(grad_net, p_current, t,method="dopri5",rtol=1e-3,atol=1e-3)[1])
        output = classifier_net(p_current)
        soft_max = nn.Softmax(dim=1)
        ####### neural path integral ends here #######
        output = soft_max(output)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
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
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--validation-batch-size', type=int, default=1000, metavar='V',
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--step-size', type=int, default=40, metavar='M',
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


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

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
        transforms.Resize(32),
        transforms.ToTensor()
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

    grad_net = Grad_net().to(device)
    classifier_net = Classifier().to(device)
    optimizer = optim.AdamW(list(grad_net.parameters())+list(classifier_net.parameters()), lr=args.lr)
    
    a = get_n_params(grad_net)
    b = get_n_params(classifier_net)
    print(a+b)

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print('setup complete')
    for epoch in range(1, args.epochs + 1):
        train(args, grad_net, classifier_net, device, train_loader, optimizer, epoch)
        validation(args, grad_net, classifier_net, device, test_loader)
        scheduler.step()
    test(args, grad_net, classifier_net, device, test_loader)

    if args.save_model:
        torch.save(grad_net.state_dict(), "grad_net.pt")
        torch.save(classifier_net.state_dict(), "classifer_net.pt")


if __name__ == '__main__':
    main()
