from __future__ import print_function
import argparse
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms  
from torch.optim.lr_scheduler import StepLR
from torchdiffeq import odeint_adjoint as odeint_adjoint
from torchdiffeq import odeint as odeint
from scipy.integrate import odeint as odeint_scipy
from torch.autograd import Variable
from random import random
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from math import pi
from torch.distributions import Normal
import numpy as np
from scipy.interpolate import make_interp_spline
import random
import matplotlib.pylab as pl
from mpl_toolkits import mplot3d
import seaborn as sns
plt.rcParams.update({'font.size': 20})

num_runs = 20
v0 = np.linspace(0,1,num_runs)
num_pts = 100
#v_c = np.linspace(0,1,num_pts)
#g_c = np.linspace(0,1,num_pts)
#h_c = np.linspace(0,1,num_pts)
#v_1 = np.ones(num_pts)
#g_1 = np.ones(num_pts)
#h_1 = np.ones(num_pts)
#v_0 = np.zeros(num_pts)
#g_0 = np.zeros(num_pts)
h_0 = np.zeros(num_pts)
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
#ax.plot3D(v_c,g_0,h_0)
#ax.plot3D(v_c,g_1,h_0)
#ax.plot3D(v_c,g_0,h_1)
#ax.plot3D(v_c,g_1,h_1)
#ax.plot3D(v_0,g_0,h_c)
#ax.plot3D(v_1,g_0,h_c)
#ax.plot3D(v_0,g_1,h_c)
#ax.plot3D(v_1,g_1,h_c)
#ax.plot3D(v_0,g_c,h_0)
#ax.plot3D(v_0,g_c,h_1)
#ax.plot3D(v_1,g_c,h_0)
#ax.plot3D(v_1,g_c,h_1)
colors = sns.color_palette("crest",num_runs)
for i,v0_i in enumerate(v0):
    g = np.linspace(0,1,num_pts)
    v = np.linspace(v0_i,1-v0_i,num_pts)
    h = v0_i * np.linspace(0,1,num_pts)
    ax.plot3D(v,g,h, color=colors[i],alpha=1.0)
    ax.set_xlabel('$u$', fontsize = 35)
    ax.set_ylabel('t', fontsize = 35)
    ax.set_zlabel('x', fontsize = 35)
    #ax.set_title('Arrangement of Integration Path')
g = torch.linspace(0,10,num_pts)
m = torch.nn.LogSigmoid()
v1 = m(g)*0.7
v1 = v1-min(v1)
v2 = 1-v1
#ax.plot3D(v1,g/10,h_0, color = sns.color_palette("rocket")[2],alpha=1)
#ax.plot3D(v2,g/10,h_0, color = sns.color_palette("rocket")[2],alpha=1)
ax.plot3D(0.1*np.ones(num_pts),np.linspace(0,1,num_pts),h_0, '--', color = sns.color_palette("rocket")[2],alpha=1)
ax.plot3D(0.3*np.ones(num_pts),np.linspace(0,1,num_pts),h_0, '--', color = sns.color_palette("rocket")[2],alpha=1)
ax.plot3D(0.5*np.ones(num_pts),np.linspace(0,1,num_pts),h_0, '--', color = sns.color_palette("rocket")[2],alpha=1)
ax.plot3D(0.7*np.ones(num_pts),np.linspace(0,1,num_pts),h_0, '--', color = sns.color_palette("rocket")[2],alpha=1)
ax.plot3D(0.9*np.ones(num_pts),np.linspace(0,1,num_pts),h_0, '--', color = sns.color_palette("rocket")[2],alpha=1)
plt.show()