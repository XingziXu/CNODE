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
plt.rcParams.update({'font.size': 30})

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




length = 40

accu1 = np.append(svhn_line,np.load('accuracy_svhn_line1.npy'))
accu2 = np.append(svhn_line,np.load('accuracy_svhn_line2.npy'))
accu3 = np.append(svhn_line,np.load('accuracy_svhn_line3.npy'))
accu4 = np.append(svhn_line,np.load('accuracy_svhn_line4.npy'))
accu5 = np.append(svhn_line,np.load('accuracy_svhn_line5.npy'))

min_accu = np.zeros((length,1))
max_accu = np.zeros((length,1))
ave_accu = np.zeros((length,1))

for i in range(0,length):
    min_accu[i] = min([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    max_accu[i] = max([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    ave_accu[i] = average([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])

test1 = np.append(svhn_line1,np.load('test_loss_svhn_line1.npy'))
test2 = np.append(svhn_line1,np.load('test_loss_svhn_line2.npy'))
test3 = np.append(svhn_line1,np.load('test_loss_svhn_line3.npy'))
test4 = np.append(svhn_line1,np.load('test_loss_svhn_line4.npy'))
test5 = np.append(svhn_line1,np.load('test_loss_svhn_line5.npy'))

min_test = np.zeros((length,1))
max_test = np.zeros((length,1))
ave_test = np.zeros((length,1))

for i in range(0,length):
    min_test[i] = min([test1[i],test2[i],test3[i],test4[i],test5[i]])
    max_test[i] = max([test1[i],test2[i],test3[i],test4[i],test5[i]])
    ave_test[i] = average([test1[i],test2[i],test3[i],test4[i],test5[i]])

train1 = np.append(svhn_line1,np.load('train_loss_svhn_line1.npy'))
train2 = np.append(svhn_line1,np.load('train_loss_svhn_line2.npy'))
train3 = np.append(svhn_line1,np.load('train_loss_svhn_line3.npy'))
train4 = np.append(svhn_line1,np.load('train_loss_svhn_line4.npy'))
train5 = np.append(svhn_line1,np.load('train_loss_svhn_line5.npy'))

min_train = np.zeros((length,1))
max_train = np.zeros((length,1))
ave_train = np.zeros((length,1))

for i in range(0,length):
    min_train[i] = min([train1[i],train2[i],train3[i],train4[i],train5[i]])
    max_train[i] = max([train1[i],train2[i],train3[i],train4[i],train5[i]])
    ave_train[i] = average([train1[i],train2[i],train3[i],train4[i],train5[i]])


accu1 = np.append(svhn_node,np.load('accuracy_svhn_node1.npy'))
accu2 = np.append(svhn_node,np.load('accuracy_svhn_node2.npy'))
accu3 = np.append(svhn_node,np.load('accuracy_svhn_node3.npy'))
accu4 = np.append(svhn_node,np.load('accuracy_svhn_node4.npy'))
accu5 = np.append(svhn_node,np.load('accuracy_svhn_node5.npy'))

min_accu1 = np.zeros((length,1))
max_accu1 = np.zeros((length,1))
ave_accu1 = np.zeros((length,1))

for i in range(0,length):
    min_accu1[i] = min([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    max_accu1[i] = max([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    ave_accu1[i] = average([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])

test1 = np.append(svhn_node1,np.load('test_loss_svhn_node1.npy'))
test2 = np.append(svhn_node1,np.load('test_loss_svhn_node2.npy'))
test3 = np.append(svhn_node1,np.load('test_loss_svhn_node3.npy'))
test4 = np.append(svhn_node1,np.load('test_loss_svhn_node4.npy'))
test5 = np.append(svhn_node1,np.load('test_loss_svhn_node5.npy'))

min_test1 = np.zeros((length,1))
max_test1 = np.zeros((length,1))
ave_test1 = np.zeros((length,1))

for i in range(0,length):
    min_test1[i] = min([test1[i],test2[i],test3[i],test4[i],test5[i]])
    max_test1[i] = max([test1[i],test2[i],test3[i],test4[i],test5[i]])
    ave_test1[i] = average([test1[i],test2[i],test3[i],test4[i],test5[i]])

train1 = np.append(svhn_node1,np.load('train_loss_svhn_node1.npy'))
train2 = np.append(svhn_node1,np.load('train_loss_svhn_node2.npy'))
train3 = np.append(svhn_node1,np.load('train_loss_svhn_node3.npy'))
train4 = np.append(svhn_node1,np.load('train_loss_svhn_node4.npy'))
train5 = np.append(svhn_node1,np.load('train_loss_svhn_node5.npy'))

min_train1 = np.zeros((length,1))
max_train1 = np.zeros((length,1))
ave_train1 = np.zeros((length,1))

for i in range(0,length):
    min_train1[i] = min([train1[i],train2[i],train3[i],train4[i],train5[i]])
    max_train1[i] = max([train1[i],train2[i],train3[i],train4[i],train5[i]])
    ave_train1[i] = average([train1[i],train2[i],train3[i],train4[i],train5[i]])

epochs = np.linspace(0,length-1,length)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3, sharex=True)

ax1.plot(epochs, min_accu.reshape(length),'k--')
ax1.plot(epochs, max_accu.reshape(length),'k--')
ax1.plot(epochs, ave_accu.reshape(length),'darkblue',label='C-NODE')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy')
ax1.fill_between(epochs, min_accu.reshape(length), max_accu.reshape(length), color='#539ecd')
ax1.legend()

ax2.plot(epochs, min_test.reshape(length),'k--')
ax2.plot(epochs, max_test.reshape(length),'k--')
ax2.plot(epochs, ave_test.reshape(length),'darkblue',label='C-NODE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Testing Loss')
ax2.fill_between(epochs, min_test.reshape(length), max_test.reshape(length), color='#539ecd')
ax2.legend()

ax3.plot(epochs, min_train.reshape(length),'k--')
ax3.plot(epochs, max_train.reshape(length),'k--')
ax3.plot(epochs, ave_train.reshape(length),'darkblue',label='C-NODE')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Training Loss')
ax3.fill_between(epochs, min_train.reshape(length), max_train.reshape(length), color='#539ecd')
ax3.legend()

ax1.plot(epochs, min_accu1.reshape(length),'k--')
ax1.plot(epochs, max_accu1.reshape(length),'k--')
ax1.plot(epochs, ave_accu1.reshape(length),'purple',label='NODE')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test Accuracy')
ax1.fill_between(epochs, min_accu1.reshape(length), max_accu1.reshape(length), color='plum')
ax1.legend()

ax2.plot(epochs, min_test1.reshape(length),'k--')
ax2.plot(epochs, max_test1.reshape(length),'k--')
ax2.plot(epochs, ave_test1.reshape(length),'purple',label='NODE')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Testing Loss')
ax2.fill_between(epochs, min_test1.reshape(length), max_test1.reshape(length), color='plum')
ax2.legend()

ax3.plot(epochs, min_train1.reshape(length),'k--')
ax3.plot(epochs, max_train1.reshape(length),'k--')
ax3.plot(epochs, ave_train1.reshape(length),'purple',label='NODE')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Training Loss')
ax3.fill_between(epochs, min_train1.reshape(length), max_train1.reshape(length), color='plum')
ax3.legend()


accu1 = np.append(cifar_line,np.load('accuracy_cifar_line1.npy'))
accu2 = np.append(cifar_line,np.load('accuracy_cifar_line2.npy'))
accu3 = np.append(cifar_line,np.load('accuracy_cifar_line3.npy'))
accu4 = np.append(cifar_line,np.load('accuracy_cifar_line4.npy'))
accu5 = np.append(cifar_line,np.load('accuracy_cifar_line5.npy'))

min_accu = np.zeros((length,1))
max_accu = np.zeros((length,1))
ave_accu = np.zeros((length,1))

for i in range(0,length):
    min_accu[i] = min([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    max_accu[i] = max([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    ave_accu[i] = average([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])

test1 = np.append(cifar_line1,np.load('test_loss_cifar_line1.npy'))
test2 = np.append(cifar_line1,np.load('test_loss_cifar_line2.npy'))
test3 = np.append(cifar_line1,np.load('test_loss_cifar_line3.npy'))
test4 = np.append(cifar_line1,np.load('test_loss_cifar_line4.npy'))
test5 = np.append(cifar_line1,np.load('test_loss_cifar_line5.npy'))

min_test = np.zeros((length,1))
max_test = np.zeros((length,1))
ave_test = np.zeros((length,1))

for i in range(0,length):
    min_test[i] = min([test1[i],test2[i],test3[i],test4[i],test5[i]])
    max_test[i] = max([test1[i],test2[i],test3[i],test4[i],test5[i]])
    ave_test[i] = average([test1[i],test2[i],test3[i],test4[i],test5[i]])

train1 = np.append(cifar_line1,np.load('train_loss_cifar_line1.npy'))
train2 = np.append(cifar_line1,np.load('train_loss_cifar_line2.npy'))
train3 = np.append(cifar_line1,np.load('train_loss_cifar_line3.npy'))
train4 = np.append(cifar_line1,np.load('train_loss_cifar_line4.npy'))
train5 = np.append(cifar_line1,np.load('train_loss_cifar_line5.npy'))

min_train = np.zeros((length,1))
max_train = np.zeros((length,1))
ave_train = np.zeros((length,1))

for i in range(0,length):
    min_train[i] = min([train1[i],train2[i],train3[i],train4[i],train5[i]])
    max_train[i] = max([train1[i],train2[i],train3[i],train4[i],train5[i]])
    ave_train[i] = average([train1[i],train2[i],train3[i],train4[i],train5[i]])


accu1 = np.append(cifar_node,np.load('accuracy_cifar_node1.npy'))
accu2 = np.append(cifar_node,np.load('accuracy_cifar_node2.npy'))
accu3 = np.append(cifar_node,np.load('accuracy_cifar_node3.npy'))
accu4 = np.append(cifar_node,np.load('accuracy_cifar_node4.npy'))
accu5 = np.append(cifar_node,np.load('accuracy_cifar_node5.npy'))

min_accu1 = np.zeros((length,1))
max_accu1 = np.zeros((length,1))
ave_accu1 = np.zeros((length,1))

for i in range(0,length):
    min_accu1[i] = min([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    max_accu1[i] = max([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    ave_accu1[i] = average([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])

test1 = np.append(cifar_node1,np.load('test_loss_cifar_node1.npy'))
test2 = np.append(cifar_node1,np.load('test_loss_cifar_node2.npy'))
test3 = np.append(cifar_node1,np.load('test_loss_cifar_node3.npy'))
test4 = np.append(cifar_node1,np.load('test_loss_cifar_node4.npy'))
test5 = np.append(cifar_node1,np.load('test_loss_cifar_node5.npy'))

min_test1 = np.zeros((length,1))
max_test1 = np.zeros((length,1))
ave_test1 = np.zeros((length,1))

for i in range(0,length):
    min_test1[i] = min([test1[i],test2[i],test3[i],test4[i],test5[i]])
    max_test1[i] = max([test1[i],test2[i],test3[i],test4[i],test5[i]])
    ave_test1[i] = average([test1[i],test2[i],test3[i],test4[i],test5[i]])

train1 = np.append(cifar_node1,np.load('train_loss_cifar_node1.npy'))
train2 = np.append(cifar_node1,np.load('train_loss_cifar_node2.npy'))
train3 = np.append(cifar_node1,np.load('train_loss_cifar_node3.npy'))
train4 = np.append(cifar_node1,np.load('train_loss_cifar_node4.npy'))
train5 = np.append(cifar_node1,np.load('train_loss_cifar_node5.npy'))

min_train1 = np.zeros((length,1))
max_train1 = np.zeros((length,1))
ave_train1 = np.zeros((length,1))

for i in range(0,length):
    min_train1[i] = min([train1[i],train2[i],train3[i],train4[i],train5[i]])
    max_train1[i] = max([train1[i],train2[i],train3[i],train4[i],train5[i]])
    ave_train1[i] = average([train1[i],train2[i],train3[i],train4[i],train5[i]])


ax4.plot(epochs, min_accu.reshape(length),'k--')
ax4.plot(epochs, max_accu.reshape(length),'k--')
ax4.plot(epochs, ave_accu.reshape(length),'darkblue',label='C-NODE')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Test Accuracy')
ax4.fill_between(epochs, min_accu.reshape(length), max_accu.reshape(length), color='#539ecd')
ax4.legend()

ax5.plot(epochs, min_test.reshape(length),'k--')
ax5.plot(epochs, max_test.reshape(length),'k--')
ax5.plot(epochs, ave_test.reshape(length),'darkblue',label='C-NODE')
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Testing Loss')
ax5.fill_between(epochs, min_test.reshape(length), max_test.reshape(length), color='#539ecd')
ax5.legend()

ax6.plot(epochs, min_train.reshape(length),'k--')
ax6.plot(epochs, max_train.reshape(length),'k--')
ax6.plot(epochs, ave_train.reshape(length),'darkblue',label='C-NODE')
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Training Loss')
ax6.fill_between(epochs, min_train.reshape(length), max_train.reshape(length), color='#539ecd')
ax6.legend()

ax4.plot(epochs, min_accu1.reshape(length),'k--')
ax4.plot(epochs, max_accu1.reshape(length),'k--')
ax4.plot(epochs, ave_accu1.reshape(length),'purple',label='NODE')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Test Accuracy')
ax4.fill_between(epochs, min_accu1.reshape(length), max_accu1.reshape(length), color='plum')
ax4.legend(loc="lower right")

ax5.plot(epochs, min_test1.reshape(length),'k--')
ax5.plot(epochs, max_test1.reshape(length),'k--')
ax5.plot(epochs, ave_test1.reshape(length),'purple',label='NODE')
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Testing Loss')
ax5.fill_between(epochs, min_test1.reshape(length), max_test1.reshape(length), color='plum')
ax5.legend()

ax6.plot(epochs, min_train1.reshape(length),'k--')
ax6.plot(epochs, max_train1.reshape(length),'k--')
ax6.plot(epochs, ave_train1.reshape(length),'purple',label='NODE')
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Training Loss')
ax6.fill_between(epochs, min_train1.reshape(length), max_train1.reshape(length), color='plum')
ax6.legend()

accu1 = np.append(mnist_line,np.load('accuracy_mnist_line1.npy'))
accu2 = np.append(mnist_line,np.load('accuracy_mnist_line2.npy'))
accu3 = np.append(mnist_line,np.load('accuracy_mnist_line3.npy'))
accu4 = np.append(mnist_line,np.load('accuracy_mnist_line5.npy'))
accu5 = np.append(mnist_line,np.load('accuracy_mnist_line5.npy'))

min_accu = np.zeros((length,1))
max_accu = np.zeros((length,1))
ave_accu = np.zeros((length,1))

for i in range(0,length):
    min_accu[i] = min([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    max_accu[i] = max([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    ave_accu[i] = average([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])

test1 = np.append(mnist_line1,np.load('test_loss_mnist_line1.npy'))
test2 = np.append(mnist_line1,np.load('test_loss_mnist_line2.npy'))
test3 = np.append(mnist_line1,np.load('test_loss_mnist_line3.npy'))
test4 = np.append(mnist_line1,np.load('test_loss_mnist_line5.npy'))
test5 = np.append(mnist_line1,np.load('test_loss_mnist_line5.npy'))

min_test = np.zeros((length,1))
max_test = np.zeros((length,1))
ave_test = np.zeros((length,1))

for i in range(0,length):
    min_test[i] = min([test1[i],test2[i],test3[i],test4[i],test5[i]])
    max_test[i] = max([test1[i],test2[i],test3[i],test4[i],test5[i]])
    ave_test[i] = average([test1[i],test2[i],test3[i],test4[i],test5[i]])

train1 = np.append(mnist_line1,np.load('train_loss_mnist_line1.npy'))
train2 = np.append(mnist_line1,np.load('train_loss_mnist_line2.npy'))
train3 = np.append(mnist_line1,np.load('train_loss_mnist_line3.npy'))
train4 = np.append(mnist_line1,np.load('train_loss_mnist_line5.npy'))
train5 = np.append(mnist_line1,np.load('train_loss_mnist_line5.npy'))

min_train = np.zeros((length,1))
max_train = np.zeros((length,1))
ave_train = np.zeros((length,1))

for i in range(0,length):
    min_train[i] = min([train1[i],train2[i],train3[i],train4[i],train5[i]])
    max_train[i] = max([train1[i],train2[i],train3[i],train4[i],train5[i]])
    ave_train[i] = average([train1[i],train2[i],train3[i],train4[i],train5[i]])


accu1 = np.append(mnist_node,np.load('accuracy_mnist_node1.npy'))
accu2 = np.append(mnist_node,np.load('accuracy_mnist_node2.npy'))
accu3 = np.append(mnist_node,np.load('accuracy_mnist_node3.npy'))
accu4 = np.append(mnist_node,np.load('accuracy_mnist_node4.npy'))
accu5 = np.append(mnist_node,np.load('accuracy_mnist_node5.npy'))

min_accu1 = np.zeros((length,1))
max_accu1 = np.zeros((length,1))
ave_accu1 = np.zeros((length,1))

for i in range(0,length):
    min_accu1[i] = min([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    max_accu1[i] = max([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])
    ave_accu1[i] = average([accu1[i],accu2[i],accu3[i],accu4[i],accu5[i]])

test1 = np.append(mnist_node1,np.load('test_loss_mnist_node1.npy'))
test2 = np.append(mnist_node1,np.load('test_loss_mnist_node2.npy'))
test3 = np.append(mnist_node1,np.load('test_loss_mnist_node3.npy'))
test4 = np.append(mnist_node1,np.load('test_loss_mnist_node4.npy'))
test5 = np.append(mnist_node1,np.load('test_loss_mnist_node5.npy'))

min_test1 = np.zeros((length,1))
max_test1 = np.zeros((length,1))
ave_test1 = np.zeros((length,1))

for i in range(0,length):
    min_test1[i] = min([test1[i],test2[i],test3[i],test4[i],test5[i]])
    max_test1[i] = max([test1[i],test2[i],test3[i],test4[i],test5[i]])
    ave_test1[i] = average([test1[i],test2[i],test3[i],test4[i],test5[i]])

train1 = np.append(mnist_node1,np.load('train_loss_mnist_node1.npy'))
train2 = np.append(mnist_node1,np.load('train_loss_mnist_node2.npy'))
train3 = np.append(mnist_node1,np.load('train_loss_mnist_node3.npy'))
train4 = np.append(mnist_node1,np.load('train_loss_mnist_node4.npy'))
train5 = np.append(mnist_node1,np.load('train_loss_mnist_node5.npy'))

min_train1 = np.zeros((length,1))
max_train1 = np.zeros((length,1))
ave_train1 = np.zeros((length,1))

for i in range(0,length):
    min_train1[i] = min([train1[i],train2[i],train3[i],train4[i],train5[i]])
    max_train1[i] = max([train1[i],train2[i],train3[i],train4[i],train5[i]])
    ave_train1[i] = average([train1[i],train2[i],train3[i],train4[i],train5[i]])

ax7.plot(epochs, min_accu.reshape(length),'k--')
ax7.plot(epochs, max_accu.reshape(length),'k--')
ax7.plot(epochs, ave_accu.reshape(length),'darkblue',label='C-NODE')
ax7.set_xlabel('Epoch')
ax7.set_ylabel('Test Accuracy')
ax7.fill_between(epochs, min_accu.reshape(length), max_accu.reshape(length), color='#539ecd')
ax7.legend()

ax8.plot(epochs, min_test.reshape(length),'k--')
ax8.plot(epochs, max_test.reshape(length),'k--')
ax8.plot(epochs, ave_test.reshape(length),'darkblue',label='C-NODE')
ax8.set_xlabel('Epoch')
ax8.set_ylabel('Testing Loss')
ax8.fill_between(epochs, min_test.reshape(length), max_test.reshape(length), color='#539ecd')
ax8.legend()

ax9.plot(epochs, min_train.reshape(length),'k--')
ax9.plot(epochs, max_train.reshape(length),'k--')
ax9.plot(epochs, ave_train.reshape(length),'darkblue',label='C-NODE')
ax9.set_xlabel('Epoch')
ax9.set_ylabel('Training Loss')
ax9.fill_between(epochs, min_train.reshape(length), max_train.reshape(length), color='#539ecd')
ax9.legend()

ax7.plot(epochs, min_accu1.reshape(length),'k--')
ax7.plot(epochs, max_accu1.reshape(length),'k--')
ax7.plot(epochs, ave_accu1.reshape(length),'purple',label='NODE')
ax7.set_xlabel('Epoch')
ax7.set_ylabel('Test Accuracy')
ax7.fill_between(epochs, min_accu1.reshape(length), max_accu1.reshape(length), color='plum')
ax7.legend()

ax8.plot(epochs, min_test1.reshape(length),'k--')
ax8.plot(epochs, max_test1.reshape(length),'k--')
ax8.plot(epochs, ave_test1.reshape(length),'purple',label='NODE')
ax8.set_xlabel('Epoch')
ax8.set_ylabel('Testing Loss')
ax8.fill_between(epochs, min_test1.reshape(length), max_test1.reshape(length), color='plum')
ax8.legend()

ax9.plot(epochs, min_train1.reshape(length),'k--')
ax9.plot(epochs, max_train1.reshape(length),'k--')
ax9.plot(epochs, ave_train1.reshape(length),'purple',label='NODE')
ax9.set_xlabel('Epoch')
ax9.set_ylabel('Training Loss')
ax9.fill_between(epochs, min_train1.reshape(length), max_train1.reshape(length), color='plum')
ax9.legend()
plt.show()