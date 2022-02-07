import numpy as np
np.random.seed(42)

from scipy import interpolate

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

import pickle

from timeit import default_timer as timer

import torch
from torchvision.utils import make_grid

from joblib import Parallel, delayed

def get_diffusion_exit(x0, m, s, t, norm=np.inf):

    '''
    Solve a SDE. 
    x0 : initial condition 
    m  : drift function of (x, t) 
    s  : diffusion function of (x, t)
    dt : time step
    '''

    N = t.shape[0]
    dt = t[1]-t[0]
    sdt = np.sqrt(dt)

    xt = np.zeros((x0.shape[0], N, x0.shape[1]))

    xt[:, 0, :] = x0.copy()

    dW = np.sqrt(dt) * np.random.randn(*xt.shape)

    for idx in range(xt.shape[1]-1):

        ti = t[idx]
        xi = xt[:,idx,:]
        xt[:,idx+1,:] = xi + m(xi,ti) * dt + s(xi,ti) * dW[:,idx,:]

    dist  = np.linalg.norm(xt, norm, axis=-1)
    small_dist = dist >= 1 - 2*dt

    num_exit = np.max(dist >= 1, 1).sum()

    if num_exit < x0.shape[0]:
        print('Error {} exit'.format(num_exit/x0.shape[0]))

    inds = np.argmax(small_dist, axis=1)
    final_loc = xt[range(x0.shape[0]), inds, :]

    xt_zeros = xt.copy()
    xt_zeros[small_dist] = 0 

    return final_loc, xt_zeros

def get_exit_location(x0, dt=0.001):
    '''
    function that returns the exit location for brownian motion

    x0 : array of intiial locations
    dt : time step for brownian motion
    '''

    n  = 10*int(1/dt)

    # sample of size locations x num samples x dim
    dW = np.sqrt(dt) * np.random.normal(np.zeros(x0.shape[1]), 
            np.ones(x0.shape[1]), 
            size=(x0.shape[0],n,x0.shape[1]))

    # start the initial condition
    dW[:,0,:] = x0.copy()

    # compute the sample paths
    W = dW.cumsum(1) 

    dist  = np.linalg.norm(W, 2, axis=-1)
    small_dist = dist >= 1 - dt
    num_exit = small_dist.sum() 

    if np.max(dist >= 1, 1).sum() < x0.shape[0]:
        print('Error {} exit'.format(num_exit/x0.shape[0]))

    inds = np.argmax(small_dist, axis=1)

    exit = W[range(x0.shape[0]),inds,:]

    W[small_dist] = 0
    dW[small_dist] = 0 

    return exit, W, dW


# The goal is to: 
# a) solve an equation using Feynman-Kac
# b) compute a change of measure to compute the solution to a new PDE

def rho(mu, mu2, X, sigma, dt, dB):
    '''
    mu : original drift
    mu2: drift we want to change to
    X: shape is n locations x time length x dimension
    sigma : diffusion coefficient
    dt : time step
    dB : brownian motion
    '''
    f  = mu2(X,0) - mu(X,0)
    norm = (f ** 2).sum(-1) * dt 
    a = -0.5*np.sum(norm,-1)
    b = np.sum((f * dB).sum(-1), -1)

    interior = a + b

    #interior[interior > 2] = 0

    rho = np.exp(interior)

    return rho

def plot_circle(Mx, in_circle, x, savename):

    tmp_mx = Mx.mean() * np.ones((x.shape[0] * x.shape[0]))
    tmp_mx[in_circle] = Mx.copy()
    mean_xi_mx = tmp_mx.reshape(-1,x.shape[1])

    plt.imshow(mean_xi_mx, cmap='jet')
    plt.colorbar()
    plt.savefig(savename)
    plt.close('all')

def elliptic():
    print('This one is wrong (i think)')
    N_exp = 100

    theta = np.linspace(0, 2*np.pi, 200)
    x_circle = np.stack((np.cos(theta), np.sin(theta)),1) 

    def a(x,t):
        # drift
        return 0

    def b(x,t):
        # diffusion 
        return 1

    def g(x):
        return np.sin(2*x[:,0])  + np.cos(3*x[:,1])

    def mu2(x,t):
        return -x

    lin = np.linspace(-1, 1, 100)
    x, y = np.meshgrid(lin, lin)
    xy = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),1)
    in_circle = np.nonzero(np.linalg.norm(xy, 2, axis=-1) <= 1)
    xy   = xy[in_circle]

    t = np.linspace(0,4,3000)
    dt = t[1]-t[0]
    print('dt {}'.format(dt))

    norm = 2
    
    paths = np.zeros((N_exp, xy.shape[0], 1))
    paths_b = np.zeros((N_exp, xy.shape[0], 1))

    def solve_exit():

        eloc, W, dW = get_exit_location(xy, dt)

        bias = rho(a, mu2, W, b, dt, dW)
        return (g(eloc), bias)


    exit_coord = Parallel(n_jobs=18)(delayed(solve_exit)() for i in range(N_exp))
    paths = np.stack([ex[0] for ex in exit_coord]) # shape N x xy.shape[0] x d
    paths_b = np.stack([ex[0]*ex[1] for ex in exit_coord]) # shape N x xy.shape[0] x d

    plot_circle(paths.mean(0),   in_circle, x, 'original.png')
    plot_circle(paths_b.mean(0), in_circle, x, 'original_biased.png')

    def solve_exit2():
        exit, _ = get_diffusion_exit(xy,mu2,b,t, norm=2)
        return g(exit)
    exit_coord = Parallel(n_jobs=18)(delayed(solve_exit2)() for i in range(N_exp))

    paths_d = np.stack([ex for ex in exit_coord]) # shape N x xy.shape[0] x d
    plot_circle(paths_d.mean(0),   in_circle, x, 'elliptic_diffusion.png')

def rho_parabolic(mu2, mu, s, W, dt, dB = None):
    '''
    W is shape N x x-grid x time
    '''
    f = (mu2(W, 0) - mu(W, 0) / s(W, 0))
    f[:,:,0] = 0
    if dB is None:
        dB = np.sqrt(dt) * np.random.randn(*W.shape)
    norm = ( f ** 2 * dt).cumsum(-1) 
    a = -0.5 * norm
    b = (f * dB).cumsum(-1)
    a[:,:,0] = 0
    b[:,:,0] = 0

    interior = a + b

    interior[interior > 6] = 0

    return np.exp(interior)

def ou_exact(x,t, A=-1, K=1000):

    x = np.expand_dims(x,0).repeat(K,0)
    t = np.expand_dims(t,0).repeat(K,0)

    dt = t[0,0,1]  - t[0,0,0]

    s = (dt * np.ones_like(t)).cumsum(-1)

    dB = np.sqrt(dt) * np.random.randn(*x.shape)
    dB[:,:,0] = 0 

    z = np.exp(t*A)*x + (np.exp((t - s) * A) * dB).cumsum(-1)

    return z, dB

def euler_parabolic(t, x0, m, s, dW, K=1000):
    '''
    Solves the parabolic probelm using Euler-Maruyama
    t : time points to solve
    x0: starting x positions
    m : drift
    s : diffusion
    dW : brownian motion
    K : number of expectations to take over
    '''
    N = t.shape[0]
    dt = t[1]-t[0]
    sdt = np.sqrt(dt)

    xt = np.zeros((K, x0.shape[0], N))

    xt[:, :, 0] = x0.copy()

    if dW is None:
        dW = np.sqrt(dt) * np.random.randn(*xt.shape)

    for idx in range(xt.shape[2]-1):
      
        ti = t[idx]
        xi = xt[:,:,idx]
        xt[:,:,idx+1] = xi + m(xi,ti) * dt + s(xi,ti) * dW[:,:,idx]

    return xt, dW

def parabolic():

    def g(x):
        return np.sin(0.2*x) #np.sin(x)

    def mu(x,t):
        return 0

    def mu2(x,t):
        #return 1.5 * np.ones(x.shape) #+ np.abs(x)
        return 1*x #2*x #+2*np.sin(x) 

    def mu3(x,t):
        return 1*x  

    def s(x,t):
        return 1

    T = 0.1
    n_K = 3000

    t = np.linspace(0, T, 80)
    dt =  t[1] - t[0]
    x = np.linspace(-np.pi, np.pi, 100)

    import timeit

    dB = np.sqrt(dt) * np.random.randn(n_K, x.shape[0], t.shape[0])
    dB[:,:,0] = 0 

    start_time = timeit.default_timer()
    B = dB.copy()
    B[:,:,0] = x.copy()
    B = B.cumsum(-1)
    oue = B
    exs = g(B).mean(0)
    print('Exact BM sim time: {}'.format(timeit.default_timer() - start_time))

    biased = ( g(B) * (rho_parabolic(mu2, mu, s, B, dt, dB))).mean(0)
    print('Biased sim total time: {}'.format(timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    zt, dW = (euler_parabolic(t, x, mu3, s, dB, K=n_K))
    euler = g(zt)
    print('Euler time: {}'.format(timeit.default_timer() - start_time))

    with open('data.npy', 'wb') as f:
        np.save(f, euler.mean(0))

    fig = plt.figure(figsize=(16,4))

    mapname = 'turbo'
    vmax = biased.max()
    vmin = biased.min()

    plt.subplot(141)
    plt.title('Original')
    plt.grid(False)
    plt.axis('off')
    plt.imshow(g(oue).mean(0), cmap=mapname, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    plt.subplot(142)
    plt.grid(False)
    plt.axis('off')
    plt.title('Girsanov')
    plt.imshow(biased, cmap=mapname, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    plt.subplot(143)
    plt.grid(False)
    plt.axis('off')
    plt.title('Diffusion')
    plt.imshow(euler.mean(0), cmap=mapname, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    plt.subplot(144)
    plt.grid(False)
    plt.axis('off')
    plt.title('Difference')
    plt.imshow(np.abs(biased - euler.mean(0)), cmap=mapname)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('ou-comparison.png')
    plt.close('all')

    print('MSE {}'.format(((euler.mean(0) - biased)**2).mean()))

    plt.plot(g(oue).mean(0)[:,0], label='original')
    plt.plot(biased[:,0], label='biased')
    plt.plot(euler.mean(0)[:,0], label='euler')
    plt.legend()
    plt.savefig('ic.pdf')
    plt.close('all')


parabolic()
#elliptic() this one doesn't work 
