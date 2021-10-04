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



accu1 = np.load('accuracy_swiss_line1.npy')
accu2 = np.load('accuracy_swiss_line2.npy')
accu3 = np.load('accuracy_swiss_line3.npy')
accu4 = np.load('accuracy_swiss_line4.npy')
accu5 = np.load('accuracy_swiss_line5.npy')

min_accu = np.zeros((len(accu1),1))
max_accu = np.zeros((len(accu1),1))
ave_accu = np.zeros((len(accu1),1))

for i in range(0,len(accu1)):
    min_accu[i] = min([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    max_accu[i] = max([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    ave_accu[i] = average([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])

test1 = np.load('test_loss_swiss_line1.npy')
test2 = np.load('test_loss_swiss_line2.npy')
test3 = np.load('test_loss_swiss_line3.npy')
test4 = np.load('test_loss_swiss_line4.npy')
test5 = np.load('test_loss_swiss_line5.npy')

min_test = np.zeros((len(test1),1))
max_test = np.zeros((len(test1),1))
ave_test = np.zeros((len(test1),1))

for i in range(0,len(test1)):
    min_test[i] = min([test1[i],test2[i],test3[i],test4[i],test5[i]])
    max_test[i] = max([test1[i],test2[i],test3[i],test4[i],test5[i]])
    ave_test[i] = average([test1[i],test2[i],test3[i],test4[i],test5[i]])

train1 = np.load('train_loss_swiss_line1.npy')
train2 = np.load('train_loss_swiss_line2.npy')
train3 = np.load('train_loss_swiss_line3.npy')
train4 = np.load('train_loss_swiss_line4.npy')
train5 = np.load('train_loss_swiss_line5.npy')

min_train = np.zeros((len(train1),1))
max_train = np.zeros((len(train1),1))
ave_train = np.zeros((len(train1),1))

for i in range(0,len(train1)):
    min_train[i] = min([train1[i],train2[i],train3[i],train4[i],train5[i]])
    max_train[i] = max([train1[i],train2[i],train3[i],train4[i],train5[i]])
    ave_train[i] = average([train1[i],train2[i],train3[i],train4[i],train5[i]])


accu1 = np.load('accuracy_swiss_node1.npy')
accu2 = np.load('accuracy_swiss_node2.npy')
accu3 = np.load('accuracy_swiss_node3.npy')
accu4 = np.load('accuracy_swiss_node4.npy')
accu5 = np.load('accuracy_swiss_node5.npy')

min_accu1 = np.zeros((len(accu1),1))
max_accu1 = np.zeros((len(accu1),1))
ave_accu1 = np.zeros((len(accu1),1))

for i in range(0,len(accu1)):
    min_accu1[i] = min([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    max_accu1[i] = max([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    ave_accu1[i] = average([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])

test1 = np.load('test_loss_swiss_node1.npy')
test2 = np.load('test_loss_swiss_node2.npy')
test3 = np.load('test_loss_swiss_node3.npy')
test4 = np.load('test_loss_swiss_node4.npy')
test5 = np.load('test_loss_swiss_node5.npy')

min_test1 = np.zeros((len(test1),1))
max_test1 = np.zeros((len(test1),1))
ave_test1 = np.zeros((len(test1),1))

for i in range(0,len(test1)):
    min_test1[i] = min([test1[i],test2[i],test3[i],test4[i],test5[i]])
    max_test1[i] = max([test1[i],test2[i],test3[i],test4[i],test5[i]])
    ave_test1[i] = average([test1[i],test2[i],test3[i],test4[i],test5[i]])

train1 = np.load('train_loss_swiss_node1.npy')
train2 = np.load('train_loss_swiss_node2.npy')
train3 = np.load('train_loss_swiss_node3.npy')
train4 = np.load('train_loss_swiss_node4.npy')
train5 = np.load('train_loss_swiss_node5.npy')

min_train1 = np.zeros((len(train1),1))
max_train1 = np.zeros((len(train1),1))
ave_train1 = np.zeros((len(train1),1))

for i in range(0,len(train1)):
    min_train1[i] = min([train1[i],train2[i],train3[i],train4[i],train5[i]])
    max_train1[i] = max([train1[i],train2[i],train3[i],train4[i],train5[i]])
    ave_train1[i] = average([train1[i],train2[i],train3[i],train4[i],train5[i]])

fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)

ax1.plot(np.linspace(1,50,50), min_accu.reshape(50),'k--')
ax1.plot(np.linspace(1,50,50), max_accu.reshape(50),'k--')
ax1.plot(np.linspace(1,50,50), ave_accu.reshape(50),'darkblue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy')
ax1.fill_between(np.linspace(1,50,50), min_accu.reshape(50), max_accu.reshape(50), color='#539ecd')

ax2.plot(np.linspace(1,50,50), min_test.reshape(50),'k--')
ax2.plot(np.linspace(1,50,50), max_test.reshape(50),'k--')
ax2.plot(np.linspace(1,50,50), ave_test.reshape(50),'darkblue')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Testing Loss')
ax2.fill_between(np.linspace(1,50,50), min_test.reshape(50), max_test.reshape(50), color='#539ecd')

ax3.plot(np.linspace(1,50,50), min_train.reshape(50),'k--')
ax3.plot(np.linspace(1,50,50), max_train.reshape(50),'k--')
ax3.plot(np.linspace(1,50,50), ave_train.reshape(50),'darkblue')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Training Loss')
ax3.fill_between(np.linspace(1,50,50), min_train.reshape(50), max_train.reshape(50), color='#539ecd')

ax1.plot(np.linspace(1,50,50), min_accu1.reshape(50),'k--')
ax1.plot(np.linspace(1,50,50), max_accu1.reshape(50),'k--')
ax1.plot(np.linspace(1,50,50), ave_accu1.reshape(50),'purple')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy')
ax1.fill_between(np.linspace(1,50,50), min_accu1.reshape(50), max_accu1.reshape(50), color='plum')

ax2.plot(np.linspace(1,50,50), min_test1.reshape(50),'k--')
ax2.plot(np.linspace(1,50,50), max_test1.reshape(50),'k--')
ax2.plot(np.linspace(1,50,50), ave_test1.reshape(50),'purple')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Testing Loss')
ax2.fill_between(np.linspace(1,50,50), min_test1.reshape(50), max_test1.reshape(50), color='plum')

ax3.plot(np.linspace(1,50,50), min_train1.reshape(50),'k--')
ax3.plot(np.linspace(1,50,50), max_train1.reshape(50),'k--')
ax3.plot(np.linspace(1,50,50), ave_train1.reshape(50),'purple')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Training Loss')
ax3.fill_between(np.linspace(1,50,50), min_train1.reshape(50), max_train1.reshape(50), color='plum')

plt.show()