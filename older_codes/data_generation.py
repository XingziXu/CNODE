import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# define the phi functions
def phi_1(t): # define phi_1, function in x direction
  return np.exp(-t/10)*100

def phi_2(t): # define phi_2, funcion in y direction
  return -np.exp(-t/5)*100

def phi_3(t): # define phi_3, function in z direction
  return 1

# initialization
U_0 = 1 # define starting value
x = np.linspace(0,20,num=50) # initialize x values
dx = x[1]-x[0]
y = np.linspace(0,20,num=50) # initialize y values
dy = y[1]-y[0]
#z = np.linspace(0,1,num=50) # initialize z values
t = np.linspace(0,20,num=50) # initialize t values
dt = t[1]-t[0]
data = [[1],[1]]
#data = np.zeros((len(t),len(x),len(y))) # initialize data matrix
#data = np.zeros((len(t),len(x),len(y),len(z))) # initialize data matrix

# data generation
# these are the input
x_g = np.linspace(0,1,num=50) # initialize x values
y_g = np.linspace(0,1,num=50) # initialize y values
t_g = np.linspace(0,1,num=50) # initialize t values
x_temp = [[1,1,1],[1,1,1]]
for i in range(1,50):
    for j in range(1,50):
        for k in range(1,50):
            x_temp = np.append(x_temp,[[t[k],x[i],y[j]]], axis=0)
# these are the output 
for i in range(1,len(x)): # for each time step
  for j in range(1,len(y)): # for each x step
    for k in range(1,len(t)): # for each y step
        phi1 = phi_1(t[k]) # calculate the change in x direction
        phi2 = phi_2(t[k]) # calculate the change in y direction
        phi3 = phi_3(t[k]) # calculate the change in z direction
        data = np.append(data, [[U_0+phi1*dx+phi2*dy]],axis=0)
        #data[m,i,j] = U_0+phi1*x[i]+phi2*y[j] # calculate and store the current data
#      for k in range(1,len(z)): # for each z step
#        data[m,i,j,k] = U_0+phi1*x[i]+phi2*y[j]+phi3*z[k] # calculate and store the current data

"""
# visualization
time1 = data[4,:,:] # extract data

fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(time1)
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()
"""

# save data into a npy file
with open('output.npy', 'wb') as f:
    np.save(f, data[2:])

with open('input.npy', 'wb') as f:
    np.save(f, x_temp[2:])