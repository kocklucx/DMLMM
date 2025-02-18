import pandas as pd
import numpy as np
import torch
from DMLMM.model_selection import dmlmm_model_selection
import Simulations

df_obs, df_true = Simulations.dgp1.dgp1()

y,x = [],[]
for i in np.unique(df_obs['id']):
    y.append(torch.tensor(np.asarray(df_obs[df_obs['id']==i]['y'])).type(torch.float64))
    x.append(torch.tensor(np.asarray(df_obs[df_obs['id']==i][['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']])).type(torch.float64))

dmlmm = dmlmm_model_selection(y=y,x=x)

df_true['hat_f_t'] = np.zeros(len(df_true))

for i in np.unique(df_true['id']):
    y_obs = np.asarray(df_obs[df_obs['id']==i]['y'])
    design_pred = torch.tensor(np.vstack([np.asarray(df_obs[df_obs['id']==i][['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']]),np.asarray(df_true[df_true['id']==i][['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']])])).type(torch.float64)
    w_post,m_post,s_post = dmlmm.predict_y(design_pred,y_obs,0)
    df_true.loc[df_true['id']==i,'hat_f_t'] = np.matmul(np.vstack(m_post).T,w_post)
    
print(Simulations.metrics.log_rmse(df_true))
print(Simulations.metrics.log_score(df_obs,dmlmm.w_gmm,dmlmm.m_gmm,dmlmm.s_gmm,dmlmm.error_var))