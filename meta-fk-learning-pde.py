#from __future__ import print_function
import argparse
#from tkinter import _XYScrollCommand
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms  
from torch.optim.lr_scheduler import StepLR
#from torchdiffeq import odeint_adjoint as odeint_adjoint
#from torchdiffeq import odeint as odeint
#from scipy.integrate import odeint as odeint_scipy
#from torch.autograd import Variable
import numpy as np
import random
import matplotlib.pyplot as plt
class mu_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu_f = nn.Sequential(
        nn.Linear(2,16),
        #nn.Tanh(),
        nn.Linear(16,16),
        #nn.Tanh(),
        nn.Linear(16,1),
        #nn.Tanh()
        )
    def forward(self,x,t):
        return(self.mu_f(torch.cat((x,t),axis=1)))
class sigma_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma_f = nn.Sequential(
        nn.Linear(2,16),
        #nn.Tanh(),
        nn.Linear(16,16),
        #nn.Tanh(),
        nn.Linear(16,1),
        #nn.Tanh()
        )
    def forward(self,x,t):
        return(self.sigma_f(torch.cat((x,t),axis=1)))
class g_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.g_f = nn.Sequential(
        nn.Linear(1,32),
        nn.Tanh(),
        nn.Linear(32,32),
        nn.Tanh(),
        nn.Linear(32,1),
        #nn.Tanh()
        )
    def forward(self,x):
        return(self.g_f(x))
def rho_parabolic(mu2, mu, s, W, dt, device, dB = None):
    '''
    W is shape N x x-grid x time
    '''
    #f = W.view(500,100,80)
    f = (mu2(W, torch.zeros(W.size()).to(device)) - mu(W, torch.zeros(W.size()).to(device)) )#/ s(W, torch.zeros(W.size()).to(device)))
    f = f.view(500,100,80)
    f_copy = torch.clone(f)
    f_copy[:,:,0] = 0
    f = f_copy
    if dB is None:
        dB = np.sqrt(dt) * np.random.randn(*W.shape)
    norm = ( f ** 2 * dt).cumsum(-1)
    a = -0.5 * norm
    b = (f * dB).cumsum(-1)
    a[:,:,0] = 0
    b[:,:,0] = 0
    interior = a + b
    interior[interior > 6] = 0
    return torch.exp(interior)
def mu(x,t):
        return 0
def train(mu2, s, g, data_train, B, optimizer, device, epoch, dt, dB, x_data_idx, t_data_idx):
    optimizer.zero_grad()
    data_train = data_train.to(device)
    #batch_indices = torch.split(torch.linspace(0,49999,50000,dtype=int),25000)
    #batches = torch.split(data_train,1000)
    #x_indices = torch.split(torch.tensor(x_data_idx),1000)
    #t_indices = torch.split(torch.tensor(t_data_idx),1000)
    #loss = 0
    #for i in range(0,len(batch_indices)):
    #batch_index = batch_indices[i]
    #batch = data_train[batch_index]
    #x_idx = x_data_idx[batch_index]
    #t_idx = t_data_idx[batch_index]
    mu2.train() # set network on training mode
    s.train() # set network on training mode
    g.train()
    #torch.autograd.set_detect_anomaly(True)
    B = torch.tensor(B)
    B = B.to(device)
    B=B.view(500*100*80,1)
    B = B.type(torch.float32)
    #a = g(B.type(torch.float32))
    dB = torch.tensor(dB)
    dB = dB.to(device)
    #biased = ( torch.mul(torch.sin(3*B).view(500,100,80), (rho_parabolic(mu2, mu, s, B, dt, dB)))).mean(0)
    biased = ( torch.mul(g(B).view(500,100,80), (rho_parabolic(mu2, mu, s, B, dt, device, dB)))).mean(0)
    biased_chosen = biased[x_data_idx,t_data_idx]
    loss = torch.norm(biased_chosen-data_train[:,2],p='fro')
    loss.backward(retain_graph=True) # backpropagate through the loss
    optimizer.step() # update the path network's parameters
    torch.cuda.empty_cache()
    print('The loss in epoch {:.0f} is {:.4f}\n'.format(epoch,loss))
    return loss
def test(mu2, s, g, data_train, B, optimizer, device, epoch, dt, dB, x_data_idx, t_data_idx):
    mu2.eval() # set network on training mode
    s.eval() # set network on training mode
    g.eval()
    B = torch.tensor(B)
    B = B.to(device)
    B=B.view(500*100*80,1)
    B = B.type(torch.float32)
    #a = g(B.type(torch.float32))
    dB = torch.tensor(dB)
    dB = dB.to(device)
    biased = ( torch.mul(g(B).view(500,100,80), (rho_parabolic(mu2, mu, s, B, dt, device, dB)))).mean(0)
    return biased
def main():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    mu2 = mu_net()
    s = sigma_net()
    g = g_net()
    mu2 = mu2.to(device)
    s = s.to(device)
    g = g.to(device)
    data = np.load('data.npy')
    T = 0.1
    n_K = 500
    num_data = 50000
    t = np.linspace(0, T, 80)
    dt =  t[1] - t[0]
    x = np.linspace(-np.pi, np.pi, 100)
    x_data_idx = np.asarray([random.randrange(0, 100, 1) for i in range(num_data)])
    t_data_idx = np.asarray([random.randrange(0, 80, 1) for i in range(num_data)])
    x_data = x[x_data_idx]
    t_data = t[t_data_idx]
    u_data = data[x_data_idx,t_data_idx]
    data_train = torch.tensor(np.concatenate((x_data.reshape(len(x_data),1),t_data.reshape(len(t_data),1),u_data.reshape(len(u_data),1)),axis=1))
    dB = np.sqrt(dt) * np.random.randn(n_K, x.shape[0], t.shape[0])
    dB[:,:,0] = 0
    #tart_time = timeit.default_timer()
    B = dB.copy()
    B[:,:,0] = x.copy()
    B = B.cumsum(-1)
    #oue = B
    #exs = g(B).mean(0)
    #biased = ( g(B) * (rho_parabolic(mu2, mu, s, B, dt, dB))).mean(0)
    #optimizer = optim.AdamW(list(mu2.parameters())+list(s.parameters())+list(g.parameters()), lr=5e-5) # define optimizer on the gradients
    optimizer = optim.SGD(list(mu2.parameters())+list(g.parameters()), lr=1e-3,weight_decay = 1e-4) # define optimizer on the gradients
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    loss = torch.zeros((150,1))
    for epoch in range(1, 30 + 1):
        #x_data_idx = np.asarray([random.randrange(0, 100, 1) for i in range(num_data)])
        #t_data_idx = np.asarray([random.randrange(0, 80, 1) for i in range(num_data)])
        #x_data = x[x_data_idx]
        #t_data = t[t_data_idx]
        #u_data = data[x_data_idx,t_data_idx]
        #data_train = torch.tensor(np.concatenate((x_data.reshape(len(x_data),1),t_data.reshape(len(t_data),1),u_data.reshape(len(u_data),1)),axis=1))
        new_loss = train(mu2, s, g, data_train, B, optimizer, device, epoch, dt, dB, x_data_idx, t_data_idx)
        loss[epoch-1] = new_loss
        scheduler.step()
        #print('The best accuracy is {:.4f}%\n'.format(accu))
    result = test(mu2, s, g, data_train, B, optimizer, device, epoch, dt, dB, x_data_idx, t_data_idx)
    #plt.plot(loss)
    plt.figure()
    plt.imshow(result.detach().numpy())
    plt.savefig('pde.png')
    plt.figure()
    plt.imshow(data)
    plt.savefig('pde_true.png')
    a = g(torch.tensor(x.astype(np.float32)).view(100,1))
    plt.figure()
    plt.plot(x,a.detach().numpy(),color='red',label='learnt function')
    plt.plot(x,np.sin(0.5*x),color='blue',label='ground truth')
    plt.legend()
    plt.savefig('g_func.png')
    #plt.figure()
    #plt.plot(x,mu2(a,torch.zeros(100,1)).detach().numpy(),color='red',label='learnt function')
    #plt.plot(x,x,color='blue',label='ground truth')
    #plt.savefig('mup_func.png')
if __name__ == '__main__':
    main()