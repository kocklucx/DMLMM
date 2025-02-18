import numpy as np
import pandas as pd
from Utilities.b_splines import design_matrix

def dgp1():
    posx = np.linspace(0,1,250)
    timepoints = []
    obs = []
    group = []
    true_effect = []
    for i in range(600):
        g = np.random.choice([-1,1])
        xi = np.random.normal(scale=np.sqrt([0.1, 0.045, 0.01, 0.001]))
        group.append(g)
        true_effect.append(g*np.sin(4*np.pi*posx)+np.matmul(np.vstack([np.sqrt(2)*np.sin(p*np.pi*posx) for p in [1,2,3,4]]).T,xi))
        t = np.sort(np.random.choice(np.arange(0,1,0.01)[1:],size=10,replace=False))
        timepoints.append(t)
        obs.append(np.random.normal(loc=g*np.sin(4*np.pi*t)+np.matmul(np.vstack([np.sqrt(2)*np.sin(p*np.pi*t) for p in [1,2,3,4]]).T,xi),scale=0.3))
    group = np.asarray(group)
    knots = np.linspace(0,1,10)
    knots = np.hstack([knots[0]-np.abs(knots[2]-knots[0]),knots[0]-np.abs(knots[1]-knots[0]),knots,knots[-1]+np.abs(knots[-1]-knots[-2]),knots[-1]+np.abs(knots[-1]-knots[-3])])
    x = []
    for i in range(len(obs)):
        x.append(design_matrix(knots,timepoints[i]).numpy()) 
    df = np.zeros((len(np.hstack(obs)),4+len(x[0][0])))
    df[:,1] = np.hstack(obs)
    df[:,0] = np.hstack([(i+1)*np.ones(len(obs[i])) for i in range(len(obs))])
    df[:,2] = np.hstack(timepoints)
    df[:,3] = np.hstack([group[i]*np.ones(len(obs[i])) for i in range(len(obs))])
    df[:,4:] = np.vstack(x)
    df = pd.DataFrame(df,columns=(['id','y','t','cluster']+['x'+str(k+1) for k in range(len(x[0][0]))]))
    df_true = np.zeros((len(np.hstack(true_effect)),4+len(x[0][0])))
    df_true[:,1] = np.hstack(true_effect)
    df_true[:,0] = np.hstack([(i+1)*np.ones(len(true_effect[i])) for i in range(len(true_effect))])
    df_true[:,2] = np.hstack([posx for _ in range(len(obs))])
    df_true[:,3] = np.hstack([group[i]*np.ones(len(true_effect[i])) for i in range(len(obs))])
    df_true[:,4:] = np.vstack([design_matrix(knots,posx).numpy() for _ in range(len(obs))])
    df_true = pd.DataFrame(df_true,columns=(['id','f_t','t','cluster']+['x'+str(k+1) for k in range(len(x[0][0]))]))
    return df, df_true

