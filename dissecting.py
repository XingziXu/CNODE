import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.functional import accuracy
from utils import get_cifar_dloaders, CIFARLearner

from torchdyn.torchdyn.models import *; from torchdyn import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainloader, testloader = get_cifar_dloaders(batch_size=64)

class Augmenter(nn.Module):
    """Augmentation class. Can handle several types of augmentation strategies for Neural DEs.
    :param augment_dims: number of augmented dimensions to initialize
    :type augment_dims: int
    :param augment_idx: index of dimension to augment
    :type augment_idx: int
    :param augment_func: nn.Module applied to the input datasets of dimension `d` to determine the augmented initial condition of dimension `d + a`.
                        `a` is defined implicitly in `augment_func` e.g. augment_func=nn.Linear(2, 5) augments a 2 dimensional input with 3 additional dimensions.
    :type augment_func: nn.Module
    :param order: whether to augment before datasets [augmentation, x] or after [x, augmentation] along dimension `augment_idx`. Options: ('first', 'last')
    :type order: str
    """
    def __init__(self, augment_idx:int=1, augment_dims:int=5, augment_func=None, order='first'):
        super().__init__()
        self.augment_dims, self.augment_idx, self.augment_func = augment_dims, augment_idx, augment_func
        self.order = order

    def forward(self, x: torch.Tensor):
        if not self.augment_func:
            new_dims = list(x.shape)
            new_dims[self.augment_idx] = self.augment_dims

            # if-else check for augmentation order
            if self.order == 'first':
                x = torch.cat([torch.zeros(new_dims).to(x), x],
                              self.augment_idx)
            else:
                x = torch.cat([x, torch.zeros(new_dims).to(x)],
                              self.augment_idx)
        else:
            # if-else check for augmentation order
            if self.order == 'first':
                x = torch.cat([self.augment_func(x).to(x), x],
                              self.augment_idx)
            else:
                x = torch.cat([x, self.augment_func(x).to(x)],
                               self.augment_idx)
        return x

func = nn.Sequential(nn.GroupNorm(42, 42),
                     nn.Conv2d(42, 42, 3, padding=1, bias=False),
                     nn.Softplus(),                   
                     nn.Conv2d(42, 42, 3, padding=1, bias=False),
                     nn.Softplus(), 
                     nn.GroupNorm(42, 42),
                     nn.Conv2d(42, 42, 1)
                     ).to(device)

nde = NeuralODE(func, 
               solver='dopri5',
               sensitivity='adjoint',
               atol=1e-4,
               rtol=1e-4).to(device)

# NOTE: the first noop `Augmenter` is used only to keep the `nde` at index `2`. Used to extract NFEs in CIFARLearner.
model = nn.Sequential(Augmenter(1, 0), # does nothing
                      Augmenter(1, 39),
                      nde,
                      nn.Conv2d(42, 6, 1),
                      nn.AdaptiveAvgPool2d(4),
                      nn.Flatten(),                     
                      nn.Linear(6*16, 10)).to(device)

learn = CIFARLearner(model, trainloader, testloader)
trainer = pl.Trainer(max_epochs=20, gpus=1)
                     
trainer.fit(learn)