import numpy as np
import GPy as GPy
import matplotlib.pyplot as plt

np.random.seed(0)
N0 = 30
N1 = 30
N2 = 40
N3 = 30
X0 = np.random.rand(N0)[:,None]
X1 = np.random.rand(N1)[:,None]
X2 = np.random.rand(N2)[:,None]
X3 = np.random.rand(N3)[:,None]
Y0 = 1*X0+np.random.randn(N0,1)*0.15
Y1 = 0.5*X1+np.random.randn(N1,1)*0.1
Y2 = -0.5+0.5*np.sin(X2*10)+np.random.randn(N2,1)*0.1
Y3 = np.sin(X3*10)+np.random.randn(N3,1)*0.05
keep = ((X2<0.4) | (X2>0.7))[:,0]
X2 = X2[keep,:]
Y2 = Y2[keep,:]
keep = ((X0<0.3) | (X0>0.7))[:,0]
X0 = X0[keep,:]
Y0 = Y0[keep,:]
X0widx = np.c_[X0,np.ones(X0.shape[0])*0]
X1widx = np.c_[X1,np.ones(X1.shape[0])*1]
X2widx = np.c_[X2,np.ones(X2.shape[0])*2]
X3widx = np.c_[X3,np.ones(X3.shape[0])*3]
X = np.r_[X0widx,X1widx,X2widx,X3widx]
Y = np.r_[Y0,Y1,Y2,Y3]

cols = ['r','y','b','k']
marks = ['x','.','+','o']
for reg in range(4):
    plt.scatter(X[X[:,1]==reg,0],Y[X[:,1]==reg,0],c=cols[reg],marker=marks[reg],label='data%d'%reg)
plt.legend()
plt.show(block = True)

X[0::5,:] #plotting every fifth row of X, for demonstration

kern = GPy.kern.RBF(1,lengthscale=0.1)**GPy.kern.Coregionalize(input_dim=1,output_dim=4, rank=1)
m = GPy.models.GPRegression(X,Y,kern)
m.optimize()

for region in range(4):
    m.plot(fixed_inputs=[(1,region)],plot_data=False)
    plt.plot(X[X[:,1]==region,0],Y[X[:,1]==region,0],'o')
    plt.show(block = True)

print(m)