import numpy as np
import GPy as GPy
import matplotlib.pyplot as plt

np.random.seed(0)
N0 = 30
N1 = 30
X0 = np.random.rand(N0)[:,None]
X1 = np.random.rand(N1)[:,None]
Y0 = 1*X0+np.random.randn(N0,1)*0.15
Y1 = 0.5*X1+np.random.randn(N1,1)*0.1

X0widx = np.c_[X0,np.ones(X0.shape[0])*0]
X1widx = np.c_[X1,np.ones(X1.shape[0])*1]

X = np.r_[X0widx,X1widx]
Y = np.r_[Y0,Y1]

cols = ['r','y'']
marks = ['x','.']
for reg in range(2):
    plt.scatter(X[X[:,1]==reg,0],Y[X[:,1]==reg,0],c=cols[reg],marker=marks[reg],label='data%d'%reg)
plt.legend()
plt.show(block = True)

kern = GPy.kern.RBF(1,lengthscale=0.1)**GPy.kern.Coregionalize(input_dim=1,output_dim=2, rank=1)
m = GPy.models.GPRegression(X,Y,kern)
m.optimize()

for region in range(2):
    m.plot(fixed_inputs=[(1,region)],plot_data=False)
    plt.plot(X[X[:,1]==region,0],Y[X[:,1]==region,0],'o')
    plt.show(block = True)

print(m)