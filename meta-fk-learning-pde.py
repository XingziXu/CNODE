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
        nn.Tanh(),
        nn.Linear(16,16),
        nn.Tanh(),
        nn.Linear(16,1),
        nn.Tanh()
        )
    def forward(self,x,t):
        x_and_t = torch.cat((x,t),axis=1)
        return(self.mu_f(x_and_t))

class sigma_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma_f = nn.Sequential(
        nn.Linear(2,16),
        nn.Tanh(),
        nn.Linear(16,16),
        nn.Tanh(),
        nn.Linear(16,1),
        nn.Tanh()
        )
    def forward(self,x,t):
        x_and_t = torch.cat((x,t),axis=1)
        return(self.sigma_f(x_and_t))

class g_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.g_f = nn.Sequential(
        nn.Linear(1,32),
        nn.Tanh(),
        nn.Linear(32,32),
        nn.Tanh(),
        nn.Linear(32,1),
        nn.Tanh()
        )
    def forward(self,x):
        return(self.g_f(x))

def rho_parabolic(mu2, mu, s, W, dt, device, dB = None):
    '''
    W is shape N x x-grid x time
    '''
    #f = W.view(500,100,80)
    f = (mu2(W, torch.zeros(W.size()).to(device)) - mu(W, torch.zeros(W.size()).to(device)) / s(W, torch.zeros(W.size()).to(device)))
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
    data_train = data_train.to(device)
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
    
    optimizer = optim.AdamW(list(mu2.parameters())+list(s.parameters())+list(g.parameters()), lr=5e-5) # define optimizer on the gradients
    #optimizer = optim.AdamW(list(mu2.parameters())+list(g.parameters()), lr=1e-2) # define optimizer on the gradients
    scheduler = StepLR(optimizer, step_size=30, gamma=0.2)

    loss = torch.zeros((150,1))
    for epoch in range(1, 150 + 1):
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
    plt.imshow(result.detach().numpy())
    plt.savefig('pde.png')
    plt.imshow(data)
    plt.savefig('pde_true.png')
if __name__ == '__main__':
    main()