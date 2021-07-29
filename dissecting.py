import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.functional import accuracy
from utils import get_cifar_dloaders, CIFARLearner

from torchdyn.models import NeuralODE; from torchdyn.nn import Augmenter
#from torchdyn.nn import DataControl, DepthCat, Augmenter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainloader, testloader = get_cifar_dloaders(batch_size=64)

func = nn.Sequential(nn.GroupNorm(42, 42),
                     nn.Conv2d(42, 42, 3, padding=1, bias=False),
                     nn.Softplus(),                   
                     nn.Conv2d(42, 42, 3, padding=1, bias=False),
                     nn.Softplus(), 
                     nn.GroupNorm(42, 42),
                     nn.Conv2d(42, 21, 1)
                     ).to(device)

nde = NeuralODE(func, 
               solver='dopri5',
               sensitivity='adjoint',
               atol=1e-4,
               rtol=1e-4,
               order=2,
               s_span=torch.linspace(0, 1, 2)).to(device)

model = nn.Sequential(nn.Conv2d(3, 21, 3, padding=1, bias=False),
                      Augmenter(1, 21),
                      nde,
                      nn.Conv2d(42, 6, 1),
                      nn.AdaptiveAvgPool2d(4),
                      nn.Flatten(),                     
                      nn.Linear(6*16, 10)).to(device)

learn = CIFARLearner(model, trainloader, testloader)
trainer = pl.Trainer(max_epochs=20, gpus=1, )

trainer.fit(learn)