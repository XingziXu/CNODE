import numpy as np    
import matplotlib.pyplot as plt

class LinearAdvection1D:
    # Matrix for LA1D 
   A=0
   # Initialization of constants 
   def __init__(self, c, x0, xN, N, deltaT,T):
      self.c = c 
      self.x0 = x0   
      self.xN = xN 
      self.N = N   
      self.deltaT = deltaT   
      self.T = T       
   # CFL number funct.   
   def CFL(self):
       deltaX= (self.xN - self.x0)/self.N
       return np.abs(self.c*self.deltaT/deltaX)
   # check CFL number <=1 or not.
   def checkCFL(self):
       if (self.CFL()<=1):
           flag=True 
       else:
           flag=False
       return flag
   # Matrix assembly of LA1D   
   def upwindMatrixAssembly(self):
       alpha_min=min(self.CFL(),0)
       alpha_max=max(self.CFL(),0)
       a1=[alpha_max]*(self.N-1)
       a2=[1+alpha_min-alpha_max]*(self.N)
       a3=[-alpha_min]*(self.N-1)
       self.A=np.diag(a1, -1)+np.diag(a2, 0)+np.diag(a3, 1)
       self.A[0,-1]=alpha_max
       self.A[N-1,0]=-alpha_min
   # Solve u=Au0
   def Solve(self,u0):
       return np.matmul(self.A,u0) 

#############  
# Start of the code
###################

# constants  
u_save = np.ones((100,100))
x_save = np.ones((100,100))
time = 0.1*np.linspace(1,100,100)
for i in range(0,99):


    N,x0,xN,deltaT,c,T=100,0.,10.,0.05,1.,1
    # initialization of constants
    LA1D = LinearAdvection1D(c, x0, xN, N, deltaT,T) 

    # initial value
    x=np.linspace(LA1D.x0,LA1D.xN,LA1D.N)
    #u0=np.exp(-(x-2)*(x-2))
    u0 = np.sin(x)
    #plot of initial value    
    #plt.plot(x,u0,label="Initial value")
    #plt.ylabel('u')
    #plt.xlabel('x')
    #plt.legend()


    # calculating solution if CFL<=1
    if (LA1D.checkCFL() is True):
        #print("CFL number is: ", LA1D.CFL())
        LA1D.upwindMatrixAssembly()
        for t in range(0,int(LA1D.T/LA1D.deltaT)):
            u=LA1D.Solve(u0)
            u0=u
            #print(t)
    else:
        print("CFL number is greater than 1. CFL: ", LA1D.CFL())

    # ploting the last solution
    #plt.plot(x,u,label="Solution at t="+str(LA1D.T))
    #plt.legend()
    #plt.grid(linestyle='dotted')

    #plt.savefig('LA1D.png',dpi=300)
    u_save[i,:]=u
    x_save[i,:]=x
#print(deltaT/((xN - x0)/N))
#print(LA1D.A)

with open('u.npy', 'wb') as f:
    np.save(f, u_save)

with open('x.npy', 'wb') as f:
    np.save(f, x_save)

with open('t.npy', 'wb') as f:
    np.save(f, time)