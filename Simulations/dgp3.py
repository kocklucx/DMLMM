import numpy as np
import pandas as pd
from Utilities.b_splines import design_matrix

def dgp3():
    posx = np.linspace(0,1,40)
    timepoints = []
    obs = []
    true_effect = []
    for i in range(30*4):
        beta1 = np.random.choice([1,0.1])
        beta2 = np.random.choice([1,0.1])
        w1 = np.random.choice([1,2,3])
        w2 = np.random.choice([7,8,9])
        true_effect.append(np.random.normal(beta1*np.cos(np.pi*posx*w1)+beta2*np.cos(np.pi*posx*w2),np.sqrt(0.1)))
        where = np.sort(np.random.choice(np.arange(40),size=np.random.randint(20,25),replace=False))
        timepoints.append(posx[where])
        obs.append(true_effect[-1][where])
    knots = np.linspace(0,1,10)
    knots = np.hstack([knots[0]-np.abs(knots[2]-knots[0]),knots[0]-np.abs(knots[1]-knots[0]),knots,knots[-1]+np.abs(knots[-1]-knots[-2]),knots[-1]+np.abs(knots[-1]-knots[-3])])
    x = []
    for i in range(len(obs)):
        x.append(design_matrix(knots,timepoints[i]).numpy()) 
    df = np.zeros((len(np.hstack(obs)),3+len(x[0][0])))
    df[:,1] = np.hstack(obs)
    df[:,0] = np.hstack([(i+1)*np.ones(len(obs[i])) for i in range(len(obs))])
    df[:,2] = np.hstack(timepoints)
    df[:,3:] = np.vstack(x)
    df = pd.DataFrame(df,columns=(['id','y','t']+['x'+str(k+1) for k in range(len(x[0][0]))]))
    df_true = np.zeros((len(np.hstack(true_effect)),3+len(x[0][0])))
    df_true[:,1] = np.hstack(true_effect)
    df_true[:,0] = np.hstack([(i+1)*np.ones(len(true_effect[i])) for i in range(len(true_effect))])
    df_true[:,2] = np.hstack([posx for _ in range(len(obs))])
    df_true[:,3:] = np.vstack([design_matrix(knots,posx).numpy() for _ in range(len(obs))])
    df_true = pd.DataFrame(df_true,columns=(['id','f_t','t']+['x'+str(k+1) for k in range(len(x[0][0]))]))
    return df, df_true