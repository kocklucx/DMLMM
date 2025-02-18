import numpy as np
import pandas as pd
from Utilities.b_splines import design_matrix

def dgp2():
     timepoints = []
     obs = []
     true_effect = []
     for i in range(100):
         posx, f_t = vanderpol()
         true_effect.append(f_t)
         where = np.sort(np.unique(np.random.randint(0,len(posx),np.random.randint(15,25))))
         obs.append(f_t[where])
         timepoints.append(posx[where])
     knots = np.linspace(10,20,10)
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

def vanderpol():
    delta=0.1
    T=20
    theta = np.log(np.random.uniform(1,5))
    y = [np.asarray([1,0.1])]
    time = [0]
    for t in range(int(T/delta)):
        d = np.zeros(2)
        d[0]=y[-1][1]+0.5*np.random.normal()
        d[1]=theta*(1-y[-1][0]**2)*y[-1][1]-y[-1][0]+0.5*np.random.normal()
        y.append(y[-1]+delta*d)
        time.append(time[-1]+delta)
    time = np.round(time,2)
    y = np.stack(y)[np.asarray(time)>=10,0]
    time = np.asarray(time)[np.asarray(time)>=10]
    return time, y