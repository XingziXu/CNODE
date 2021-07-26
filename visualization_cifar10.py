import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms  
from torch.optim.lr_scheduler import StepLR
from torchdiffeq import odeint as odeint
from scipy.integrate import odeint as odeint_scipy
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.onnx as onnx
import torchvision.models as models
from torchvision import datasets, transforms  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Grad_net(nn.Module): # the Grad_net defines the networks for the path and for the gradients
    def __init__(self):
        super().__init__()
        self.nfe=0 # initialize the number of function evaluations
        
        self.path = nn.Sequential( # define the network for the integration path
        nn.Conv2d(1,16,1,1,0),
        nn.ReLU(),
        nn.Conv2d(16,16,3,1,1),
        nn.ReLU(),
        nn.Conv2d(16,3,1,1,0),
        nn.Flatten(),
        nn.Linear(3072,3)
        )
        
        self.grad_g = nn.Sequential( # define the network for the gradient on x direction
            nn.GroupNorm(6,6),
            nn.Conv2d(6,64,1,1,0),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            nn.GroupNorm(64,64),
            nn.Conv2d(64,3,1,1,0)
        )
        
        self.grad_h = nn.Sequential( # define the network for the gradient on y direction
            nn.GroupNorm(6,6),
            nn.Conv2d(6,64,1,1,0),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            nn.GroupNorm(64,64),
            nn.Conv2d(64,3,1,1,0)
        )
        
        self.grad_i = nn.Sequential( # define the network for the gradient on z direction
            nn.GroupNorm(6,6),
            nn.Conv2d(6,64,1,1,0),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            nn.GroupNorm(64,64),
            nn.Conv2d(64,3,1,1,0)
        )

#a = torch.load('grad_net_weight_and_bias.pt')
#a = torch.load('grad_net_weight_only.pt')
#a = torch.load('grad_net_no_clamp.pt')

model = Grad_net()
#model.load_state_dict(torch.load('grad_net_weight_and_bias.pt'))
model.load_state_dict(torch.load('grad_net.pt'))
#model.load_state_dict(torch.load('grad_net_no_clamp.pt'))
model.eval()

transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
        ])
dataset2 = datasets.CIFAR10('../data', train=False, download=True,
                       transform=transform)

test_img = dataset2[0][0]

t = torch.linspace(0,1,100)
result = torch.zeros(100,3)
input_current = torch.zeros(100,1,32,32)

for i in range(0,100):
    t_current = t[i] * torch.ones(1,32,32)
    #input_current[i,:,:,:] = torch.cat((t_current, test_img),dim=0)
    input_current[i,:,:,:] = t_current
    # input_current = input_current.resize(1,4,32,32)
    #input_current = torch.ones(1,4,32,32)
    
    
result = model.path(input_current)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(result[:,0].detach().numpy(), result[:,1].detach().numpy(), result[:,2].detach().numpy(), label='parametric curve')
ax.legend()

plt.show()

plt.plot(result[:,0].detach().numpy())

plt.show()