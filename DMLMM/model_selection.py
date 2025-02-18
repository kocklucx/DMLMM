from DMLMM.dmlmm import DMLMM
import numpy as np
import torch

def dmlmm_model_selection(x,y):
    D0=len(x[0][0])
    best_elbo = -np.inf
    possible_D = []
    for D1 in range(3,min(int((D0-1)/2)+1,30)):
        for D2 in range(1,int((D1-1)/2)+1):
            possible_D.append((D1,D2))
    possible_D = [possible_D[0]] + possible_D[1:-1:2] + [possible_D[-1]]
    for D1,D2 in possible_D:
        dmlmm = DMLMM(y,x,K=[10,5],D=[D0,D1,D2])
        dmlmm.train(250,0.001) 
        if dmlmm.elbo>best_elbo:
            best_K=[int(sum(i > 0.01 for i in dmlmm.p_d[0])),int(sum(i > 0.01 for i in dmlmm.p_d[1]))]
            best_D=[D0,D1,D2]
            best_elbo=dmlmm.elbo
    dmlmm = DMLMM(y,x,K=best_K,D=best_D)
    dmlmm.train(20000,0.001) 
    return dmlmm