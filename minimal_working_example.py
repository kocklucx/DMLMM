import numpy as np
from Utilities.b_splines import design_matrix
import torch
import matplotlib.pyplot as plt
from DMLMM.dmlmm import DMLMM

# generate some data
timepoints = []
y = []
for i in range(600):
    ni = np.random.randint(8,13)
    t = np.sort(np.random.choice(np.arange(0,1,0.01)[1:],size=ni,replace=False))
    xi = np.random.normal(scale=np.sqrt([0.1, 0.045, 0.01, 0.001]))
    timepoints.append(t)
    y.append(torch.tensor(np.random.normal(loc=np.random.choice([-1,1])*np.sin(4*np.pi*t)+np.matmul(np.vstack([np.sqrt(2)*np.sin(p*np.pi*t) for p in [1,2,3,4]]).T,xi),scale=0.3)).type(torch.float64))

# generate the design matrices
knots = np.linspace(0,1,10)
knots = np.hstack([knots[0]-np.abs(knots[2]-knots[0]),knots[0]-np.abs(knots[1]-knots[0]),knots,knots[-1]+np.abs(knots[-1]-knots[-2]),knots[-1]+np.abs(knots[-1]-knots[-3])])
x = []
for i in range(len(y)):
    x.append(design_matrix(knots,timepoints[i]))
    
# specify the shape of the DMLMM
K=[10,5] #number of clusters per layer
D=[10,4,1] # dimensions per layer

dmlmm = DMLMM(y,x,K,D) # initalize the DMLMM
dmlmm.train(1000, -1) # fit the DMLMM. Here we use 1000 steps to get results relatively fast

# predict the curve for y_i
i = 0
pos = np.linspace(0,1,250)
design_pred = np.vstack([x[i],design_matrix(knots,pos)]) # concatenate the design matrix for the observed time points with the design matrix for the time points for which y should be predicted
sample_post = dmlmm.predict_y(design_pred,y[i].numpy(),5000) # generate a sample from the predictive distribution

# plot the observed time points and the estimated trajectory. 
fig, ax = plt.subplots(dpi=300)
ax.plot(pos,np.mean(sample_post,axis=0),color='gray')
ax.plot(pos,np.quantile(sample_post,q=0.975,axis=0),color='gray',linestyle='--')
ax.plot(pos,np.quantile(sample_post,q=0.025,axis=0),color='gray',linestyle='--')
ax.scatter(timepoints[i],y[i],color='black')
plt.show()