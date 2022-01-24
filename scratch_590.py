import numpy as np
import matplotlib.pyplot as plt
import torchvision
from pylab import rcParams
#rcParams['figure.figsize'] = 10, 8
#rcParams['figure.dpi'] = 300

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms  
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler


class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()
        
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])
        
    def g(self, z): # we have two parts of formula, x1=z1*exp(s1)+t1, and x2 = z2
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i] # calculate z*b=z2
            s = self.s[i](x_)*(1 - self.mask[i]) # calculate s(z*b)*(1-b)=s1
            t = self.t[i](x_)*(1 - self.mask[i]) # calculate t(z*b)*(1-b)=t1
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t) # calculate z*b+(1-b)*(z*exp(s(z*b)*(1-b))+t(z*b)*(1-b)). z*b is our z2=x2, and the rest is z1*(exp(s1))
        return x

    def f(self, x): # we have two parts of formula, z1=(x1-t1)*exp(-s1), and z2 = x2
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z # calculate b*x=x2
            s = self.s[i](z_) * (1-self.mask[i]) # calculate s(b*x)*(1-b)=s1
            t = self.t[i](z_) * (1-self.mask[i]) # calculate t(b*x)*(1-b)=t1
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_ # calculate (1-b)*(x-t(b*x)*(1-b))*exp(-s(b*x)*(1-b))+b*x. b*x is our z2, and the rest is (x1-t1)*exp(-s1)
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self,x):
        z, logp = self.f(x)
        #print(z.device)
        #print(logp.device)
        #print(self.prior.log_prob(z).device)
        return self.prior.log_prob(z) + logp
        
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x

device = torch.device("cuda")

length_width = 16

nets = lambda: nn.Sequential(nn.Linear(length_width*length_width, 256), nn.LeakyReLU(), nn.Linear(length_width*length_width, length_width*length_width), nn.LeakyReLU(), nn.Linear(256, length_width*length_width), nn.Tanh()).to(device)
nett = lambda: nn.Sequential(nn.Linear(length_width*length_width, 256), nn.LeakyReLU(), nn.Linear(length_width*length_width, length_width*length_width), nn.LeakyReLU(), nn.Linear(256, length_width*length_width)).to(device)
masks = torch.arange(0, length_width*length_width) % 2
masks = masks.to(device)
prior = distributions.MultivariateNormal(torch.zeros(length_width*length_width).to(device), torch.eye(length_width*length_width).to(device))
flow = RealNVP(nets, nett, masks, prior).to(device)

train_kwargs = {'batch_size': 60000}
test_kwargs = {'batch_size': 10000}
transform=transforms.Compose([
        transforms.Resize(length_width),
        transforms.ToTensor()
        ])
dataset1 = torchvision.datasets.KMNIST(root = '/scratch/xx84/NeuralPDE', train = True, download = True, transform=transform)
dataset2 = torchvision.datasets.KMNIST(root = '/scratch/xx84/NeuralPDE', train = False, download = True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

for batch_idx, (data, target) in enumerate(train_loader):
    train_data, train_target = data, target # assign data to device

for batch_idx, (data, target) in enumerate(test_loader):
    test_data, test_target = data, target # assign data to device

train_data = train_data.view(60000,length_width*length_width).to(device)
test_data = test_data.view(10000,length_width*length_width).to(device)

optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=1e-4, weight_decay=1e-4)
for t in range(200):    
    noisy_moons = train_data[0:3000,:].to(device)
    loss = -flow.log_prob(noisy_moons).mean().to(device)
    
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    
    if t % 1 == 0:
        print('iter %s:' % t, 'loss = %.3f' % loss)


x = flow.sample(1)
x = x.view(length_width,length_width)
plt.imshow(x.cpu().detach().numpy(),cmap='gray')
plt.savefig('590')

noisy_moons = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
z = flow.f(torch.from_numpy(noisy_moons))[0].detach().numpy()

fig, [[ax0, ax1], [ax2, ax3]] = plt.subplots(2,2, sharex=True,figsize=(5,2))

#plt.subplot(221)
ax0.scatter(z[:, 0], z[:, 1])
ax0.set_title(r'$z = f(X)$')

z = np.random.multivariate_normal(np.zeros(2), np.eye(2), 1000)
#plt.subplot(222)
ax1.scatter(z[:, 0], z[:, 1])
ax1.set_title(r'$z \sim p(z)$')

#plt.subplot(223)
x = datasets.make_moons(n_samples=1000, noise=.05)[0].astype(np.float32)
ax2.scatter(x[:, 0], x[:, 1], c='r')
ax2.set_title(r'$X \sim p(X)$')

#plt.subplot(224)
x = flow.sample(1000).detach().numpy()
ax3.scatter(x[:, 0, 0], x[:, 0, 1], c='r')
ax3.set_title(r'$X = g(z)$')

plt.show()

