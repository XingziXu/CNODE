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
import matplotlib.pylab as pl
import seaborn as sns

plt.rcParams.update({'font.size': 15})

mnist_node = 10.24
mnist_line = 10.13
svhn_node = 6.44
svhn_line = 7.76
cifar_node = 8.16
cifar_line = 10.21

mnist_node1 = 2.3
mnist_line1 = 2.3
svhn_node1 = 2.31
svhn_line1 = 2.3
cifar_node1 = 2.31
cifar_line1 = 2.3




length = 100
epochs = np.linspace(0,length,length)

accu1 = np.append(svhn_line,np.load('accuracy_svhn_high_1d0.npy'))
accu2 = np.append(svhn_line,np.load('accuracy_svhn_high_1d1.npy'))
accu3 = np.append(svhn_line,np.load('accuracy_svhn_high_1d2.npy'))
accu4 = np.append(svhn_line,np.load('accuracy_svhn_high_1d3.npy'))

min_accu = np.zeros((length,1))
max_accu = np.zeros((length,1))
ave_accu = np.zeros((length,1))

for i in range(0,length):
    min_accu[i] = min([accu1[i],accu2[i],accu3[i],accu4[i]])
    max_accu[i] = max([accu1[i],accu2[i],accu3[i],accu4[i]])
    ave_accu[i] = average([accu1[i],accu2[i],accu3[i],accu4[i]])

test1 = np.append(svhn_line1,np.load('test_loss_svhn_high_1d0.npy'))
test2 = np.append(svhn_line1,np.load('test_loss_svhn_high_1d1.npy'))
test3 = np.append(svhn_line1,np.load('test_loss_svhn_high_1d2.npy'))
test4 = np.append(svhn_line1,np.load('test_loss_svhn_high_1d3.npy'))

min_test = np.zeros((length,1))
max_test = np.zeros((length,1))
ave_test = np.zeros((length,1))

for i in range(0,length):
    min_test[i] = min([test1[i],test2[i],test3[i],test4[i]])
    max_test[i] = max([test1[i],test2[i],test3[i],test4[i]])
    ave_test[i] = average([test1[i],test2[i],test3[i],test4[i]])

train1 = np.append(svhn_line1,np.load('train_loss_svhn_high_1d0.npy'))
train2 = np.append(svhn_line1,np.load('train_loss_svhn_high_1d1.npy'))
train3 = np.append(svhn_line1,np.load('train_loss_svhn_high_1d2.npy'))
train4 = np.append(svhn_line1,np.load('train_loss_svhn_high_1d3.npy'))

min_train = np.zeros((length,1))
max_train = np.zeros((length,1))
ave_train = np.zeros((length,1))

for i in range(0,length):
    min_train[i] = min([train1[i],train2[i],train3[i],train4[i]])
    max_train[i] = max([train1[i],train2[i],train3[i],train4[i]])
    ave_train[i] = average([train1[i],train2[i],train3[i],train4[i]])


accu1 = np.append(svhn_line,np.load('accuracy_svhn_high_2d0.npy'))
accu2 = np.append(svhn_line,np.load('accuracy_svhn_high_2d1.npy'))
accu3 = np.append(svhn_line,np.load('accuracy_svhn_high_2d2.npy'))
accu4 = np.append(svhn_line,np.load('accuracy_svhn_high_2d3.npy'))

min_accu1 = np.zeros((length,1))
max_accu1 = np.zeros((length,1))
ave_accu1 = np.zeros((length,1))

for i in range(0,length):
    min_accu1[i] = min([accu1[i],accu2[i],accu3[i],accu4[i]])
    max_accu1[i] = max([accu1[i],accu2[i],accu3[i],accu4[i]])
    ave_accu1[i] = average([accu1[i],accu2[i],accu3[i],accu4[i]])

test1 = np.append(svhn_line1,np.load('test_loss_svhn_high_2d0.npy'))
test2 = np.append(svhn_line1,np.load('test_loss_svhn_high_2d1.npy'))
test3 = np.append(svhn_line1,np.load('test_loss_svhn_high_2d2.npy'))
test4 = np.append(svhn_line1,np.load('test_loss_svhn_high_2d3.npy'))

min_test1 = np.zeros((length,1))
max_test1 = np.zeros((length,1))
ave_test1 = np.zeros((length,1))

for i in range(0,length):
    min_test1[i] = min([test1[i],test2[i],test3[i],test4[i]])
    max_test1[i] = max([test1[i],test2[i],test3[i],test4[i]])
    ave_test1[i] = average([test1[i],test2[i],test3[i],test4[i]])

train1 = np.append(svhn_line1,np.load('train_loss_svhn_high_2d0.npy'))
train2 = np.append(svhn_line1,np.load('train_loss_svhn_high_2d1.npy'))
train3 = np.append(svhn_line1,np.load('train_loss_svhn_high_2d2.npy'))
train4 = np.append(svhn_line1,np.load('train_loss_svhn_high_2d3.npy'))

min_train1 = np.zeros((length,1))
max_train1 = np.zeros((length,1))
ave_train1 = np.zeros((length,1))

for i in range(0,length):
    min_train1[i] = min([train1[i],train2[i],train3[i],train4[i]])
    max_train1[i] = max([train1[i],train2[i],train3[i],train4[i]])
    ave_train1[i] = average([train1[i],train2[i],train3[i],train4[i]])

accu1 = np.append(svhn_line,np.load('accuracy_svhn_high_4d0.npy'))
accu2 = np.append(svhn_line,np.load('accuracy_svhn_high_4d1.npy'))
accu3 = np.append(svhn_line,np.load('accuracy_svhn_high_4d2.npy'))
accu4 = np.append(svhn_line,np.load('accuracy_svhn_high_4d3.npy'))

min_accu2 = np.zeros((length,1))
max_accu2 = np.zeros((length,1))
ave_accu2 = np.zeros((length,1))

for i in range(0,length):
    min_accu2[i] = min([accu1[i],accu2[i],accu3[i],accu4[i]])
    max_accu2[i] = max([accu1[i],accu2[i],accu3[i],accu4[i]])
    ave_accu2[i] = average([accu1[i],accu2[i],accu3[i],accu4[i]])

test1 = np.append(svhn_line1,np.load('test_loss_svhn_high_4d0.npy'))
test2 = np.append(svhn_line1,np.load('test_loss_svhn_high_4d1.npy'))
test3 = np.append(svhn_line1,np.load('test_loss_svhn_high_4d2.npy'))
test4 = np.append(svhn_line1,np.load('test_loss_svhn_high_4d3.npy'))

min_test2 = np.zeros((length,1))
max_test2 = np.zeros((length,1))
ave_test2 = np.zeros((length,1))

for i in range(0,length):
    min_test2[i] = min([test1[i],test2[i],test3[i],test4[i]])
    max_test2[i] = max([test1[i],test2[i],test3[i],test4[i]])
    ave_test2[i] = average([test1[i],test2[i],test3[i],test4[i]])

train1 = np.append(svhn_line1,np.load('train_loss_svhn_high_4d0.npy'))
train2 = np.append(svhn_line1,np.load('train_loss_svhn_high_4d1.npy'))
train3 = np.append(svhn_line1,np.load('train_loss_svhn_high_4d2.npy'))
train4 = np.append(svhn_line1,np.load('train_loss_svhn_high_4d3.npy'))

min_train2 = np.zeros((length,1))
max_train2 = np.zeros((length,1))
ave_train2 = np.zeros((length,1))

for i in range(0,length):
    min_train2[i] = min([train1[i],train2[i],train3[i],train4[i]])
    max_train2[i] = max([train1[i],train2[i],train3[i],train4[i]])
    ave_train2[i] = average([train1[i],train2[i],train3[i],train4[i]])

accu1 = np.append(svhn_line,np.load('accuracy_svhn_high_8d0.npy'))
accu2 = np.append(svhn_line,np.load('accuracy_svhn_high_8d1.npy'))
accu3 = np.append(svhn_line,np.load('accuracy_svhn_high_8d2.npy'))
accu4 = np.append(svhn_line,np.load('accuracy_svhn_high_8d3.npy'))

min_accu3 = np.zeros((length,1))
max_accu3 = np.zeros((length,1))
ave_accu3 = np.zeros((length,1))

for i in range(0,length):
    min_accu3[i] = min([accu1[i],accu2[i],accu3[i],accu4[i]])
    max_accu3[i] = max([accu1[i],accu2[i],accu3[i],accu4[i]])
    ave_accu3[i] = average([accu1[i],accu2[i],accu3[i],accu4[i]])

test1 = np.append(svhn_line1,np.load('test_loss_svhn_high_8d0.npy'))
test2 = np.append(svhn_line1,np.load('test_loss_svhn_high_8d1.npy'))
test3 = np.append(svhn_line1,np.load('test_loss_svhn_high_8d2.npy'))
test4 = np.append(svhn_line1,np.load('test_loss_svhn_high_8d3.npy'))

min_test3 = np.zeros((length,1))
max_test3 = np.zeros((length,1))
ave_test3 = np.zeros((length,1))

for i in range(0,length):
    min_test3[i] = min([test1[i],test2[i],test3[i],test4[i]])
    max_test3[i] = max([test1[i],test2[i],test3[i],test4[i]])
    ave_test3[i] = average([test1[i],test2[i],test3[i],test4[i]])

train1 = np.append(svhn_line1,np.load('train_loss_svhn_high_8d0.npy'))
train2 = np.append(svhn_line1,np.load('train_loss_svhn_high_8d1.npy'))
train3 = np.append(svhn_line1,np.load('train_loss_svhn_high_8d2.npy'))
train4 = np.append(svhn_line1,np.load('train_loss_svhn_high_8d3.npy'))

min_train3 = np.zeros((length,1))
max_train3 = np.zeros((length,1))
ave_train3 = np.zeros((length,1))

for i in range(0,length):
    min_train3[i] = min([train1[i],train2[i],train3[i],train4[i]])
    max_train3[i] = max([train1[i],train2[i],train3[i],train4[i]])
    ave_train3[i] = average([train1[i],train2[i],train3[i],train4[i]])

accu1 = np.append(svhn_line,np.load('accuracy_svhn_high_16d0.npy'))
accu2 = np.append(svhn_line,np.load('accuracy_svhn_high_16d1.npy'))
accu3 = np.append(svhn_line,np.load('accuracy_svhn_high_16d2.npy'))
accu4 = np.append(svhn_line,np.load('accuracy_svhn_high_16d3.npy'))

min_accu4 = np.zeros((length,1))
max_accu4 = np.zeros((length,1))
ave_accu4 = np.zeros((length,1))

for i in range(0,length):
    min_accu4[i] = min([accu1[i],accu2[i],accu3[i],accu4[i]])
    max_accu4[i] = max([accu1[i],accu2[i],accu3[i],accu4[i]])
    ave_accu4[i] = average([accu1[i],accu2[i],accu3[i],accu4[i]])

test1 = np.append(svhn_line1,np.load('test_loss_svhn_high_16d0.npy'))
test2 = np.append(svhn_line1,np.load('test_loss_svhn_high_16d1.npy'))
test3 = np.append(svhn_line1,np.load('test_loss_svhn_high_16d2.npy'))
test4 = np.append(svhn_line1,np.load('test_loss_svhn_high_16d3.npy'))

min_test4 = np.zeros((length,1))
max_test4 = np.zeros((length,1))
ave_test4 = np.zeros((length,1))

for i in range(0,length):
    min_test4[i] = min([test1[i],test2[i],test3[i],test4[i]])
    max_test4[i] = max([test1[i],test2[i],test3[i],test4[i]])
    ave_test4[i] = average([test1[i],test2[i],test3[i],test4[i]])

train1 = np.append(svhn_line1,np.load('train_loss_svhn_high_16d0.npy'))
train2 = np.append(svhn_line1,np.load('train_loss_svhn_high_16d1.npy'))
train3 = np.append(svhn_line1,np.load('train_loss_svhn_high_16d2.npy'))
train4 = np.append(svhn_line1,np.load('train_loss_svhn_high_16d3.npy'))

min_train4 = np.zeros((length,1))
max_train4 = np.zeros((length,1))
ave_train4 = np.zeros((length,1))

for i in range(0,length):
    min_train4[i] = min([train1[i],train2[i],train3[i],train4[i]])
    max_train4[i] = max([train1[i],train2[i],train3[i],train4[i]])
    ave_train4[i] = average([train1[i],train2[i],train3[i],train4[i]])

accu1 = np.append(svhn_line,np.load('accuracy_svhn_high_32d0.npy'))
accu2 = np.append(svhn_line,np.load('accuracy_svhn_high_32d1.npy'))
accu3 = np.append(svhn_line,np.load('accuracy_svhn_high_32d2.npy'))
accu4 = np.append(svhn_line,np.load('accuracy_svhn_high_32d3.npy'))

min_accu5 = np.zeros((length,1))
max_accu5 = np.zeros((length,1))
ave_accu5 = np.zeros((length,1))

for i in range(0,length):
    min_accu5[i] = min([accu1[i],accu2[i],accu3[i],accu4[i]])
    max_accu5[i] = max([accu1[i],accu2[i],accu3[i],accu4[i]])
    ave_accu5[i] = average([accu1[i],accu2[i],accu3[i],accu4[i]])

test1 = np.append(svhn_line1,np.load('test_loss_svhn_high_32d0.npy'))
test2 = np.append(svhn_line1,np.load('test_loss_svhn_high_32d1.npy'))
test3 = np.append(svhn_line1,np.load('test_loss_svhn_high_32d2.npy'))
test4 = np.append(svhn_line1,np.load('test_loss_svhn_high_32d3.npy'))

min_test5 = np.zeros((length,1))
max_test5 = np.zeros((length,1))
ave_test5 = np.zeros((length,1))

for i in range(0,length):
    min_test5[i] = min([test1[i],test2[i],test3[i],test4[i]])
    max_test5[i] = max([test1[i],test2[i],test3[i],test4[i]])
    ave_test5[i] = average([test1[i],test2[i],test3[i],test4[i]])

train1 = np.append(svhn_line1,np.load('train_loss_svhn_high_32d0.npy'))
train2 = np.append(svhn_line1,np.load('train_loss_svhn_high_32d1.npy'))
train3 = np.append(svhn_line1,np.load('train_loss_svhn_high_32d2.npy'))
train4 = np.append(svhn_line1,np.load('train_loss_svhn_high_32d3.npy'))

min_train5 = np.zeros((length,1))
max_train5 = np.zeros((length,1))
ave_train5 = np.zeros((length,1))

for i in range(0,length):
    min_train5[i] = min([train1[i],train2[i],train3[i],train4[i]])
    max_train5[i] = max([train1[i],train2[i],train3[i],train4[i]])
    ave_train5[i] = average([train1[i],train2[i],train3[i],train4[i]])

accu1 = np.append(svhn_line,np.load('accuracy_svhn_high_64d0.npy'))
accu2 = np.append(svhn_line,np.load('accuracy_svhn_high_64d1.npy'))
accu3 = np.append(svhn_line,np.load('accuracy_svhn_high_64d2.npy'))
accu4 = np.append(svhn_line,np.load('accuracy_svhn_high_64d3.npy'))

min_accu6 = np.zeros((length,1))
max_accu6 = np.zeros((length,1))
ave_accu6 = np.zeros((length,1))

for i in range(0,length):
    min_accu6[i] = min([accu1[i],accu2[i],accu3[i],accu4[i]])
    max_accu6[i] = max([accu1[i],accu2[i],accu3[i],accu4[i]])
    ave_accu6[i] = average([accu1[i],accu2[i],accu3[i],accu4[i]])

test1 = np.append(svhn_line1,np.load('test_loss_svhn_high_64d0.npy'))
test2 = np.append(svhn_line1,np.load('test_loss_svhn_high_64d1.npy'))
test3 = np.append(svhn_line1,np.load('test_loss_svhn_high_64d2.npy'))
test4 = np.append(svhn_line1,np.load('test_loss_svhn_high_64d3.npy'))

min_test6 = np.zeros((length,1))
max_test6 = np.zeros((length,1))
ave_test6 = np.zeros((length,1))

for i in range(0,length):
    min_test6[i] = min([test1[i],test2[i],test3[i],test4[i]])
    max_test6[i] = max([test1[i],test2[i],test3[i],test4[i]])
    ave_test6[i] = average([test1[i],test2[i],test3[i],test4[i]])

train1 = np.append(svhn_line1,np.load('train_loss_svhn_high_64d0.npy'))
train2 = np.append(svhn_line1,np.load('train_loss_svhn_high_64d1.npy'))
train3 = np.append(svhn_line1,np.load('train_loss_svhn_high_64d2.npy'))
train4 = np.append(svhn_line1,np.load('train_loss_svhn_high_64d3.npy'))

min_train6 = np.zeros((length,1))
max_train6 = np.zeros((length,1))
ave_train6 = np.zeros((length,1))

for i in range(0,length):
    min_train6[i] = min([train1[i],train2[i],train3[i],train4[i]])
    max_train6[i] = max([train1[i],train2[i],train3[i],train4[i]])
    ave_train6[i] = average([train1[i],train2[i],train3[i],train4[i]])

accu1 = np.append(svhn_line,np.load('accuracy_svhn_high_128d0.npy'))
accu2 = np.append(svhn_line,np.load('accuracy_svhn_high_128d1.npy'))
accu3 = np.append(svhn_line,np.load('accuracy_svhn_high_128d2.npy'))
accu4 = np.append(svhn_line,np.load('accuracy_svhn_high_128d3.npy'))

min_accu7 = np.zeros((length,1))
max_accu7 = np.zeros((length,1))
ave_accu7 = np.zeros((length,1))

for i in range(0,length):
    min_accu7[i] = min([accu1[i],accu2[i],accu3[i],accu4[i]])
    max_accu7[i] = max([accu1[i],accu2[i],accu3[i],accu4[i]])
    ave_accu7[i] = average([accu1[i],accu2[i],accu3[i],accu4[i]])

test1 = np.append(svhn_line1,np.load('test_loss_svhn_high_128d0.npy'))
test2 = np.append(svhn_line1,np.load('test_loss_svhn_high_128d1.npy'))
test3 = np.append(svhn_line1,np.load('test_loss_svhn_high_128d2.npy'))
test4 = np.append(svhn_line1,np.load('test_loss_svhn_high_128d3.npy'))

min_test7 = np.zeros((length,1))
max_test7 = np.zeros((length,1))
ave_test7 = np.zeros((length,1))

for i in range(0,length):
    min_test7[i] = min([test1[i],test2[i],test3[i],test4[i]])
    max_test7[i] = max([test1[i],test2[i],test3[i],test4[i]])
    ave_test7[i] = average([test1[i],test2[i],test3[i],test4[i]])

train1 = np.append(svhn_line1,np.load('train_loss_svhn_high_128d0.npy'))
train2 = np.append(svhn_line1,np.load('train_loss_svhn_high_128d1.npy'))
train3 = np.append(svhn_line1,np.load('train_loss_svhn_high_128d2.npy'))
train4 = np.append(svhn_line1,np.load('train_loss_svhn_high_128d3.npy'))

min_train7 = np.zeros((length,1))
max_train7 = np.zeros((length,1))
ave_train7 = np.zeros((length,1))

for i in range(0,length):
    min_train7[i] = min([train1[i],train2[i],train3[i],train4[i]])
    max_train7[i] = max([train1[i],train2[i],train3[i],train4[i]])
    ave_train7[i] = average([train1[i],train2[i],train3[i],train4[i]])

n = 8
colors = sns.color_palette("rocket_r",n)
#colors = pl.cm.jet(np.linspace(0,1,n))

fig, ((ax1, ax2, ax3)) = plt.subplots(3,1, sharex=True)

#ax1.plot(epochs, min_accu.reshape(length),'k--')
#ax1.plot(epochs, max_accu.reshape(length),'k--')
ax1.plot(epochs, ave_accu.reshape(length),color=colors[0],label='1D')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy')
#ax1.fill_between(epochs, min_accu.reshape(length), max_accu.reshape(length), color='#539ecd')
ax1.legend()

#ax2.plot(epochs, min_test.reshape(length),'k--')
#ax2.plot(epochs, max_test.reshape(length),'k--')
ax2.plot(epochs, ave_test.reshape(length),color=colors[0],label='1D')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Testing Loss')
#ax2.fill_between(epochs, min_test.reshape(length), max_test.reshape(length), color='#539ecd')
ax2.legend()

#ax3.plot(epochs, min_train.reshape(length),'k--')
#ax3.plot(epochs, max_train.reshape(length),'k--')
ax3.plot(epochs, ave_train.reshape(length),color=colors[0],label='1D')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Training Loss')
#ax3.fill_between(epochs, min_train.reshape(length), max_train.reshape(length), color='#539ecd')
ax3.legend()

#ax1.plot(epochs, min_accu1.reshape(length),'k--')
#ax1.plot(epochs, max_accu1.reshape(length),'k--')
ax1.plot(epochs, ave_accu1.reshape(length),color=colors[1],label='2D')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy')
#ax1.fill_between(epochs, min_accu1.reshape(length), max_accu1.reshape(length), color='plum')
ax1.legend()

#ax2.plot(epochs, min_test1.reshape(length),'k--')
#ax2.plot(epochs, max_test1.reshape(length),'k--')
ax2.plot(epochs, ave_test1.reshape(length),color=colors[1],label='2D')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Testing Loss')
#ax2.fill_between(epochs, min_test1.reshape(length), max_test1.reshape(length), color='plum')
ax2.legend()

#ax3.plot(epochs, min_train1.reshape(length),'k--')
#ax3.plot(epochs, max_train1.reshape(length),'k--')
ax3.plot(epochs, ave_train1.reshape(length),color=colors[1],label='2D')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Training Loss')
#ax3.fill_between(epochs, min_train1.reshape(length), max_train1.reshape(length), color='plum')
ax3.legend()

#ax1.plot(epochs, min_accu2.reshape(length),'k--')
#ax1.plot(epochs, max_accu2.reshape(length),'k--')
ax1.plot(epochs, ave_accu2.reshape(length),color=colors[2],label='4D')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy')
#ax1.fill_between(epochs, min_accu2.reshape(length), max_accu2.reshape(length), color='plum')
ax1.legend()

#ax2.plot(epochs, min_test2.reshape(length),'k--')
#ax2.plot(epochs, max_test2.reshape(length),'k--')
ax2.plot(epochs, ave_test2.reshape(length),color=colors[2],label='4D')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Testing Loss')
#ax2.fill_between(epochs, min_test2.reshape(length), max_test2.reshape(length), color='plum')
ax2.legend()

#ax3.plot(epochs, min_train2.reshape(length),'k--')
#ax3.plot(epochs, max_train2.reshape(length),'k--')
ax3.plot(epochs, ave_train2.reshape(length),color=colors[2],label='4D')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Training Loss')
#ax3.fill_between(epochs, min_train2.reshape(length), max_train2.reshape(length), color='plum')
ax3.legend()

#ax1.plot(epochs, min_accu3.reshape(length),'k--')
#ax1.plot(epochs, max_accu3.reshape(length),'k--')
ax1.plot(epochs, ave_accu3.reshape(length),color=colors[3],label='8D')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy')
#ax1.fill_between(epochs, min_accu2.reshape(length), max_accu2.reshape(length), color='plum')
ax1.legend()

#ax2.plot(epochs, min_test3.reshape(length),'k--')
#ax2.plot(epochs, max_test3.reshape(length),'k--')
ax2.plot(epochs, ave_test3.reshape(length),color=colors[3],label='8D')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Testing Loss')
#ax2.fill_between(epochs, min_test2.reshape(length), max_test2.reshape(length), color='plum')
ax2.legend()

#ax3.plot(epochs, min_train3.reshape(length),'k--')
#ax3.plot(epochs, max_train3.reshape(length),'k--')
ax3.plot(epochs, ave_train3.reshape(length),color=colors[3],label='8D')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Training Loss')
#ax3.fill_between(epochs, min_train2.reshape(length), max_train2.reshape(length), color='plum')
ax3.legend()

#ax1.plot(epochs, min_accu4.reshape(length),'k--')
#ax1.plot(epochs, max_accu4.reshape(length),'k--')
ax1.plot(epochs, ave_accu4.reshape(length),color=colors[4],label='16D')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy')
#ax1.fill_between(epochs, min_accu2.reshape(length), max_accu2.reshape(length), color='plum')
ax1.legend()

#ax2.plot(epochs, min_test4.reshape(length),'k--')
#ax2.plot(epochs, max_test4.reshape(length),'k--')
ax2.plot(epochs, ave_test4.reshape(length),color=colors[4],label='16D')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Testing Loss')
#ax2.fill_between(epochs, min_test2.reshape(length), max_test2.reshape(length), color='plum')
ax2.legend()

#ax3.plot(epochs, min_train4.reshape(length),'k--')
#ax3.plot(epochs, max_train4.reshape(length),'k--')
ax3.plot(epochs, ave_train4.reshape(length),color=colors[4],label='16D')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Training Loss')
#ax3.fill_between(epochs, min_train2.reshape(length), max_train2.reshape(length), color='plum')
ax3.legend()

#ax1.plot(epochs, min_accu4.reshape(length),'k--')
#ax1.plot(epochs, max_accu4.reshape(length),'k--')
ax1.plot(epochs, ave_accu4.reshape(length),color=colors[5],label='32D')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy')
#ax1.fill_between(epochs, min_accu2.reshape(length), max_accu2.reshape(length), color='plum')
ax1.legend()

#ax2.plot(epochs, min_test4.reshape(length),'k--')
#ax2.plot(epochs, max_test4.reshape(length),'k--')
ax2.plot(epochs, ave_test4.reshape(length),color=colors[5],label='32D')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Testing Loss')
#ax2.fill_between(epochs, min_test2.reshape(length), max_test2.reshape(length), color='plum')
ax2.legend()

#ax3.plot(epochs, min_train4.reshape(length),'k--')
#ax3.plot(epochs, max_train4.reshape(length),'k--')
ax3.plot(epochs, ave_train4.reshape(length),color=colors[5],label='32D')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Training Loss')
#ax3.fill_between(epochs, min_train2.reshape(length), max_train2.reshape(length), color='plum')
ax3.legend()

#ax1.plot(epochs, min_accu4.reshape(length),'k--')
#ax1.plot(epochs, max_accu4.reshape(length),'k--')
ax1.plot(epochs, ave_accu4.reshape(length),color=colors[6],label='64D')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy')
#ax1.fill_between(epochs, min_accu2.reshape(length), max_accu2.reshape(length), color='plum')
ax1.legend()

#ax2.plot(epochs, min_test4.reshape(length),'k--')
#ax2.plot(epochs, max_test4.reshape(length),'k--')
ax2.plot(epochs, ave_test4.reshape(length),color=colors[6],label='64D')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Testing Loss')
#ax2.fill_between(epochs, min_test2.reshape(length), max_test2.reshape(length), color='plum')
ax2.legend()

#ax3.plot(epochs, min_train4.reshape(length),'k--')
#ax3.plot(epochs, max_train4.reshape(length),'k--')
ax3.plot(epochs, ave_train4.reshape(length),color=colors[6],label='64D')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Training Loss')
#ax3.fill_between(epochs, min_train2.reshape(length), max_train2.reshape(length), color='plum')
ax3.legend()

#ax1.plot(epochs, min_accu4.reshape(length),'k--')
#ax1.plot(epochs, max_accu4.reshape(length),'k--')
ax1.plot(epochs, ave_accu4.reshape(length),color=colors[7],label='128D')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy')
#ax1.fill_between(epochs, min_accu2.reshape(length), max_accu2.reshape(length), color='plum')
ax1.legend()

#ax2.plot(epochs, min_test4.reshape(length),'k--')
#ax2.plot(epochs, max_test4.reshape(length),'k--')
ax2.plot(epochs, ave_test4.reshape(length),color=colors[7],label='128D')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Testing Loss')
#ax2.fill_between(epochs, min_test2.reshape(length), max_test2.reshape(length), color='plum')
ax2.legend()

#ax3.plot(epochs, min_train4.reshape(length),'k--')
#ax3.plot(epochs, max_train4.reshape(length),'k--')
ax3.plot(epochs, ave_train4.reshape(length),color=colors[7],label='128D')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Training Loss')
#ax3.fill_between(epochs, min_train2.reshape(length), max_train2.reshape(length), color='plum')
ax3.legend()
plt.show()