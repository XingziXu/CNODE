from __future__ import print_function
import argparse
from numpy.lib.function_base import average
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms  
from torch.optim.lr_scheduler import StepLR
from torchdiffeq import odeint_adjoint as odeint
from scipy.integrate import odeint as odeint_scipy
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()+0.2
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size
    )


x = np.linspace(0,1,100)
y1 = x
y2 = 1-x

line1 = plt.plot(x,y1,color='darkblue')[0]
line2 = plt.plot(x,y2,color='purple')[0]

add_arrow(line1)
add_arrow(line2)

plt.scatter(0,0,s=60,color='darkblue')
plt.scatter(1,1,s=60,color='darkblue')
plt.scatter(1,0,s=60,color='purple')
plt.scatter(0,1,s=60,color='purple')
plt.show()