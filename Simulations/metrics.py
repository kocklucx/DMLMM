import numpy as np
import pandas as pd
from scipy import stats

def log_rmse(df):
    rmse = []
    for i in np.unique(df['id']):
        rmse.append(np.sqrt(np.mean((df[df['id']==i]['f_t']-df[df['id']==i]['hat_f_t'])**2)))
    return np.log(rmse)

def log_score(df,w_gmm,m_gmm,s_gmm,error_var):
    log_score_local = []
    for i in np.unique(df['id']):
        design=np.asarray(df[df['id']==i][['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']])
        y = np.asarray(df[df['id']==i]['y'])
        log_score_local.append(np.log(np.exp([(np.log(w_gmm[c])+stats.multivariate_normal.logpdf(y,mean=np.dot(design,m_gmm[c]),cov=np.matmul(np.matmul(design,s_gmm[c]),design.T)+np.diag(error_var*np.ones(len(y))),allow_singular=True)) for c in range(len(w_gmm))]).sum(axis=0)))  
    return np.mean(log_score_local)
