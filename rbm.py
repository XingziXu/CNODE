import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

batch_size = 128
train_set, test_set, train_loader, test_loader = {},{},{},{}
transform = transforms.Compose(
    [transforms.ToTensor()])

train_set['mnist'] = torchvision.datasets.FashionMNIST('MNIST_data/', download = True, train = True, transform = transform)
test_set['mnist'] = torchvision.datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)
train_loader['mnist'] = torch.utils.data.DataLoader(train_set['mnist'], batch_size=batch_size, shuffle=True, num_workers=0)
test_loader['mnist'] = torch.utils.data.DataLoader(test_set['mnist'], batch_size=batch_size, shuffle=False, num_workers=0)

device = 'cuda'

class RBM(nn.Module):
    """Restricted Boltzmann Machine for generating MNIST images."""
    
    def __init__(self, D: int, F: int, k: int):
        """Creates an instance RBM module.
            
            Args:
                D: Size of the input data.
                F: Size of the hidden variable.
                k: Number of MCMC iterations for negative sampling.
                
            The function initializes the weight (W) and biases (c & b).
        """
        
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(F, D)* 1e-2) # Initialized from Normal(mean=0.0, variance=1e-4)
        self.c = nn.Parameter(torch.zeros(D)) # Initialized as 0.0
        self.b = nn.Parameter(torch.zeros(F)) # Initilaized as 0.0
        self.k = k
    
    def sample(self, p):
        """Sample from a bernoulli distribution defined by a given parameter.
        
           Args:
                p: Parameter of the bernoulli distribution.
           
           Returns:
               bern_sample: Sample from Bernoulli(p)
        """
        
        bern_sample = p.bernoulli()
        return bern_sample
    
    def P_h_x(self, x):
        """Returns the conditional P(h|x). (Slide 9, Lecture 14)
        
        Args:
            x: The parameter of the conditional h|x.
        
        Returns:
            ph_x: probability of hidden vector being element-wise 1.
        """

        ph_x = torch.sigmoid(F.linear(x, self.W, self.b)) # n_batch x F
        return ph_x
    
    def P_x_h(self, h):
        """Returns the conditional P(x|h). (Slide 9, Lecture 14)
        
        Args:
            h: The parameter of the conditional x|h.
        
        Returns:
            px_h: probability of visible vector being element-wise 1.
        """
        
        px_h = torch.sigmoid(F.linear(h, self.W.t(), self.c)) # n_batch x D
        return px_h

    def free_energy(self, x):
        """Returns the Average F(x) free energy. (Slide 11, Lecture 14)."""
        
        vbias_term = x.mv(self.c) # n_batch x 1
        xtx=torch.bmm(x.reshape(x.size(0),1,x.size(1)),torch.transpose(x.reshape(x.size(0),1,x.size(1)),1,2))
        wv_b = F.linear(x, self.W, self.b) # n_batch x F
        hidden_term = F.softplus(wv_b).sum(dim=1) # n_batch x 1
        return (-hidden_term - vbias_term+0.5*xtx).mean() # 1 x 1 
    
    def forward(self, x):
        """Generates x_negative using MCMC Gibbs sampling starting from x."""
        
        x_negative = x
        for _ in range(self.k):
            
            ## Step 1: Sample h from previous iteration.
            # Get the conditional prob h|x
            phx_k = self.P_h_x(x_negative) 
            # Sample from h|x
            h_negative = self.sample(phx_k)
            
            ## Step 2: Sample x using h from step 1.
            # Get the conditional proba x|h
            pxh_k = self.P_x_h(h_negative)
            #mean = torch.bmm(self.W.T.repeat((h_negative.size(0),1,1)),h_negative.reshape((h_negative.size(0),h_negative.size(1),1)))
            #pxh_k = torch.normal(mean=(torch.bmm(self.W.T.repeat((h_negative.size(0),1,1)),h_negative.reshape((h_negative.size(0),h_negative.size(1),1))).squeeze()).to(device),std=torch.ones((h_negative.size(0),self.c.size(0))).to(device))
            loc = torch.bmm(self.W.T.repeat((h_negative.size(0),1,1)),h_negative.reshape((h_negative.size(0),20,1))).squeeze().reshape(h_negative.size(0),1,784)+self.c.repeat(h_negative.size(0),1,1)
            loc = loc.to(device)
            dist = MultivariateNormal(loc, torch.eye(784).to(device))
            #dist = dist.to(device)
            # Sample from x|h
            #x_negative = self.sample(pxh_k)
            x_negative = dist.sample().squeeze()

        return x_negative, pxh_k

def train(model, device, train_loader, optimizer, epoch):
    
    train_loss = 0
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # torchvision provides us with normalized data, s.t. input is in [0,1]
        data = data.view(data.size(0),-1) # flatten the array: Converts n_batchx1x28x28 to n_batchx784
        mean = data.mean()
        
        #data = data.bernoulli() 
        meansq = (data**2).mean()
        std = torch.sqrt(meansq - mean**2)
        data = (data-mean)/std
        data = data.to(device)
        
        optimizer.zero_grad()
        
        x_tilde, _ = model(data)
        x_tilde = x_tilde.detach()
        
        loss = model.free_energy(data) - model.free_energy(x_tilde)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        if (batch_idx+1) % (len(train_loader)//2) == 0:
            print('Train({})[{:.0f}%]: Loss: {:.4f}'.format(
                epoch, 100. * batch_idx / len(train_loader), train_loss/(batch_idx+1)))

def test(model, device, test_loader, epoch):
    
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0),-1)
            mean = data.mean()
        
            #data = data.bernoulli() 
            meansq = (data**2).mean()
            std = torch.sqrt(meansq - mean**2)
            data = (data-mean)/std
            data = data.to(device)
            xh_k,_ = model(data)
            loss = nn.MSELoss()
            loss = loss(xh_k, data)
            #loss = model.free_energy(data) - model.free_energy(xh_k)
            test_loss += loss.item() # sum up batch loss
    
    test_loss = (test_loss*batch_size)/len(test_loader.dataset)
    #print('Test({}): Loss: {:.4f}'.format(epoch, test_loss))
    print('Test({}): Mean squared reconstruction error: {:.4f}'.format(epoch, test_loss))

def make_optimizer(optimizer_name, model, **kwargs):
    if optimizer_name=='Adam':
        optimizer = optim.Adam(model.parameters(),lr=kwargs['lr'])
    elif optimizer_name=='SGD':
        optimizer = optim.SGD(model.parameters(),lr=kwargs['lr'],momentum=kwargs.get('momentum', 0.), 
                              weight_decay=kwargs.get('weight_decay', 0.))
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer
    
def make_scheduler(scheduler_name, optimizer, **kwargs):
    if scheduler_name=='MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=kwargs['milestones'],gamma=kwargs['factor'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler

# General variables

seed = 1
data_name = 'mnist'
optimizer_name = 'Adam'
scheduler_name = 'MultiStepLR'
num_epochs = 25
lr = 0.001

device = torch.device(device)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

rbm = RBM(D=28*28, F=20, k=5).to(device)
optimizer = make_optimizer(optimizer_name, rbm, lr=lr)
scheduler = make_scheduler(scheduler_name, optimizer, milestones=[5], factor=0.1)

for epoch in range(1, num_epochs + 1):
    
    train(rbm, device, train_loader[data_name], optimizer, epoch)
    test(rbm, device, test_loader[data_name], epoch)
    #scheduler.step()
    
    print('Optimizer Learning rate: {0:.4f}\n'.format(optimizer.param_groups[0]['lr']))

def show(img1, img2):
    npimg1 = img1.cpu().numpy()
    npimg2 = img2.cpu().numpy()
    
    fig, axes = plt.subplots(1,2, figsize=(20,10))
    axes[0].imshow(np.transpose(npimg1, (1,2,0)), interpolation='nearest')
    axes[1].imshow(np.transpose(npimg2, (1,2,0)), interpolation='nearest')
    fig.show()

data,_ = next(iter(test_loader[data_name]))
data = data[:32]
data_size = data.size()
data = data.view(data.size(0),-1)
bdata = data.bernoulli().to(device)
vh_k, pvh_k = rbm(bdata)
vh_k, pvh_k = vh_k.detach(), pvh_k.detach()

show(make_grid(data.reshape(data_size), padding=0), make_grid(pvh_k.reshape(data_size), padding=0))