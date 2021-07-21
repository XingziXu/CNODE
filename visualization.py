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

class Grad_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.nfe=0
        self.path = nn.Sequential(
        #nn.Linear(1,16),
        #nn.ReLU(),
        #nn.Linear(16,16),
        #nn.ReLU(),
        #nn.Linear(16,2),
        #nn.GroupNorm(2,2),
        #nn.ReLU(),
        nn.Conv2d(4,8,1,1,0),
        #nn.GroupNorm(2,4),
        nn.Sigmoid(),
        nn.Conv2d(8,8,3,1,1),
        nn.Sigmoid(),
        nn.Conv2d(8,3,1,1,0),
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

a = torch.load('grad_net.pt')

model = Grad_net()
model.load_state_dict(torch.load('grad_net.pt'))
model.eval()

transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
        ])
dataset2 = datasets.SVHN('../data', download=True, split="test",
                       transform=transform)

test_img = dataset2[17][0]

t = torch.linspace(0,1,100)
result = torch.zeros(100,3)

for i in range(0,99):
    t_current = t[i] * torch.ones(1,32,32)
    input_current = torch.cat((test_img,t_current),dim=0)
    input_current = input_current.resize(1,4,32,32)
    result[i,:] = model.path(input_current)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(result[:,0].detach().numpy(), result[:,1].detach().numpy(), result[:,2].detach().numpy(), label='parametric curve')
ax.legend()

plt.show()

plt.plot(result[:,0].detach().numpy())

plt.show()